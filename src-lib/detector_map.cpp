#include "darknet_internal.hpp"

#include <condition_variable>
#include <cstring>
#include <deque>


namespace
{
	static auto & cfg_and_state = Darknet::CfgAndState::get();

	/* ********************* */
	/* PROBABILITY STRUCTURE */
	/* ********************* */
	/// BoxProbability is used to match probabilities to ground truth.  This happens in the "analysis" thread.
	struct BoxProbability
	{
		Darknet::Box bb				= {0.0f, 0.0f, 0.0f, 0.0f}; ///< bounding box
		float probability			= 0.0f;						///< probability (score)
		int class_id				= -1;
		bool matched_ground_truth	= false;					///< @p TRUE if matched to some ground truth (greedy best-IoU of same class), else @p FALSE
		int unique_truth_index		= -1;						///< global index of matched ground truth, to prevent double counting
	};

	/* ******************* */
	/* WORK UNIT STRUCTURE */
	/* ******************* */
	/** A single unit of work ("image") to be performed by the loading thread, prediction thread, and analysis thread.
	 *
	 * Note that @ref filename used to be the @c std::map key in the previous design.  When the per-thread queue
	 * rewrite replaced the maps with FIFOs, the filename had to move into the WorkUnit itself so that the analysis
	 * thread can still locate the matching @c .txt annotation file.
	 */
	struct WorkUnit
	{
		std::string filename;	///< absolute path to the validation image (used to find the matching @c .txt annotation)

		// image resized to the network dimensions
		Darknet::Image img = {0};

		// results of calling Darknet's predict()
		Darknet::Detection * predictions = nullptr;
		int number_of_predictions = 0;

		// results read from .txt annotation file
		box_label * ground_truth_labels = nullptr;
		int number_of_ground_truth_labels = 0;
	};

	/* ************************ */
	/* BOUNDED BLOCKING QUEUE   */
	/* ************************ */
	/** A condition-variable-driven bounded FIFO used to feed work between the loading, prediction, and analysis
	 * threads.  Replaces the previous @c std::map<filename, WorkUnit> + @c std::mutex + sleep-poll pattern.
	 *
	 * Two problems with the previous design motivated this rewrite:
	 *
	 *	1) @c std::map is a tree.  We never needed ordering by filename, only FIFO behaviour, so we paid for log(n)
	 *	   per insert/erase, per-node allocation, and string-comparison on the key for no benefit.
	 *
	 *	2) Consumers checked @c container.empty() outside the lock and slept for 5..150 ms when empty (the "adaptive
	 *	   pause" code).  In multi-threaded prediction mode, every predict thread had to acquire @em the @em same
	 *	   mutex just to take work off the queue, so prediction threads serialised on the input side instead of
	 *	   running in parallel.  At low per-image GPU times (e.g., 224x160 inference on the RTX 4060) this lock
	 *	   contention left the GPU idle between kernels even though there was work waiting.
	 *
	 * The fix is twofold:  (a) this BoundedQueue blocks producers/consumers via condition variables instead of
	 * polling, and (b) we instantiate one queue *per predict thread* (see @ref SharedInfo::predict_queues), so
	 * predict threads no longer contend on the same lock.
	 */
	template <typename T>
	class BoundedQueue
	{
		public:

			explicit BoundedQueue(const size_t capacity)
				: capacity_(capacity)
			{
			}

			BoundedQueue(const BoundedQueue &) = delete;
			BoundedQueue & operator=(const BoundedQueue &) = delete;

			/// Push an item.  Blocks if the queue is full -- this is the natural backpressure that replaces the old
			/// "if map is full, sleep for N ms" loop in the loading thread.  Returns @p false only if the queue has
			/// been closed (i.e., during shutdown).
			bool push(T item)
			{
				std::unique_lock lock(mtx_);
				cv_not_full_.wait(lock, [this]{ return q_.size() < capacity_ or closed_; });
				if (closed_)
				{
					return false;
				}
				q_.push_back(std::move(item));
				lock.unlock();
				cv_not_empty_.notify_one();
				return true;
			}

			/// Pop an item.  Blocks if the queue is empty -- replaces the old "if map is empty, sleep for N ms" loop
			/// in the prediction and analysis threads.  Returns @p false only after the queue has been closed @em and
			/// drained (this is how the consumer threads learn there's no more work coming).
			bool pop(T & out)
			{
				std::unique_lock lock(mtx_);
				cv_not_empty_.wait(lock, [this]{ return not q_.empty() or closed_; });
				if (q_.empty())
				{
					return false;
				}
				out = std::move(q_.front());
				q_.pop_front();
				lock.unlock();
				cv_not_full_.notify_one();
				return true;
			}

			/// Mark the queue as closed.  Producers blocked in @ref push return @p false; consumers blocked in
			/// @ref pop return @p false once the queue drains.  Idempotent and safe to call from any thread.
			void close()
			{
				{
					std::lock_guard lock(mtx_);
					closed_ = true;
				}
				cv_not_empty_.notify_all();
				cv_not_full_.notify_all();
			}

			/// Approximate queue size for diagnostic display only.  Briefly takes the lock.
			size_t size() const
			{
				std::lock_guard lock(mtx_);
				return q_.size();
			}

		private:

			mutable std::mutex mtx_;
			std::condition_variable cv_not_empty_;
			std::condition_variable cv_not_full_;
			std::deque<T> q_;
			const size_t capacity_;
			bool closed_ = false;
	};

	/** Information which needs to be shared between threads.  Group it together in a structure and only share 1 struct
	 * instead of many individual fields passed around between threads.
	 */
	struct SharedInfo
	{
		Darknet::Network net;

		/// A copy of the last YOLO layer in the network.
		Darknet::Layer output_layer;

		float iou_threshold			= 0.5f;		///< user-selected IoU threshold; e.g., see command-line parm "-iou_thresh 0.5"
		float thresh_calc_avg_iou	= 0.25f; 	///< @todo what is this?  e.g., see command-line parm "-thresh 0.25"
		float detection_threshold	= 0.005f;	///< detection threshold
		float nms					= 0.45f;
		float avg_iou				= 0.0f;
		int tp_for_thresh			= 0;		///< diagnostic TP at thresh_calc_avg_iou (across all classes)
		int fp_for_thresh			= 0;		///< diagnostic FP at thresh_calc_avg_iou (across all classes)
		int unique_truth_count		= 0;		///< @todo what is this?

		std::vector<float> avg_iou_per_class;
		std::vector<int> tp_for_thresh_per_class;
		std::vector<int> fp_for_thresh_per_class;

		/** All of the predictions across the entire dataset.  Obviously, this can grow to be quite big.  But this happens in
		 * the analysis thread, which is not the bottleneck, so we can ignore trying to apply performance optimizations.
		 */
		std::vector<BoxProbability> box_probabilities;

		/// The total number of classes in this neural network.
		size_t number_of_classes = 0;

		/** This value is important since the validation images are split into multiple input queues.  This value is the
		 * @em total of all input images, which the threads use to determine when they've finished processing all images.
		 */
		size_t total_number_of_validation_images = 0;

		/// Note the maximum work queue size, where all loading threads fill up a single work queue.  @see @ref max_work_queue_size
		size_t number_of_loading_threads_to_start = 0;

		/** This is the per-queue capacity (in WorkUnits) for both the predict queues and the analyze queue.  The total
		 * RAM headroom is roughly @c max_work_queue_size * (num_predict_threads + 1) * sizeof(image), so we keep this
		 * smaller than the previous design's single-map capacity since we now have multiple queues.
		 */
		size_t max_work_queue_size = 0;

		/** This is in a vector because we often want to start multiple loading threads, and each thread requires a unique
		 * set of validation image filenames.  @see @ref number_of_loading_threads_to_start
		 */
		std::vector<Darknet::SStr> validation_image_filenames;

		/** Remember exactly how many ground truths we have for each class.  The key is the class ID, the value is the
		 * counter for that specific class.
		 */
		std::map<int, size_t> ground_truth_counts;

		/// Similar to @ref ground_truth_counts but for predictions.
		std::map<int, size_t> prediction_counts;

		/** One input queue per prediction thread.  In single-threaded predict mode this vector has size 1; in
		 * multi-threaded mode it has size @c validation_threads.  Loading threads dispatch round-robin across these
		 * queues (see @ref loader_dispatch_counter), so prediction threads never contend on the same lock when
		 * pulling work off their queue -- this is the change that fixes the GPU starvation bottleneck.
		 */
		std::vector<std::unique_ptr<BoundedQueue<WorkUnit>>> predict_queues;

		/** Single output queue feeding the analysis thread.  Push contention here is harmless because (a) only the
		 * predict threads write to it, (b) the operation under the lock is just a @c std::deque::push_back, and
		 * (c) the analysis thread is not on the GPU's critical path.
		 */
		std::unique_ptr<BoundedQueue<WorkUnit>> analyze_queue;

		/** Atomic counter used by loading threads to fan out work round-robin across @ref predict_queues.  We use
		 * @c fetch_add because it's wait-free; the alternative (a mutex around an integer) would re-introduce the
		 * cross-loader contention we just removed.
		 */
		std::atomic<size_t> loader_dispatch_counter = 0;

		/// Set by the control thread to indicate if processing is done, or if threads need to exit early.
		std::atomic<bool> done = false;

		/// The total number of images which have been loaded from disk.
		std::atomic<size_t> count_load_performed	= 0;

		/// The total number of images which have been processed by Darknet predict().
		std::atomic<size_t> count_predict_performed	= 0;

		/// The total number of images which have completed the analysis.
		std::atomic<size_t> count_analyze_performed	= 0;

		/// Per-thread prediction counters for multi-threaded mode (only used when thread_val > 1)
		std::unique_ptr<std::atomic<size_t>[]> per_thread_predict_count;
	};


	/* ************************ */
	/* LOAD LOAD LOAD LOAD LOAD */
	/* ************************ */

	/** Load images.  Try and do the least amount of work possible here since the loading threads are important to
	 * keeping the prediction thread fed.  @note This is called on a secondary thread!
	 *
	 * Compared to the previous version this function is significantly shorter:  the "if queue is full, sleep and
	 * retry" loop is gone, because @ref BoundedQueue::push provides that backpressure naturally by blocking until
	 * a consumer pops.  The only new logic is the round-robin fan-out across the per-predict-thread queues.
	 */
	void detector_map_loading_thread(const size_t loading_thread_id, SharedInfo & shared_info)
	{
		TAT(TATPARMS);

		/* THIS IS CALLED ON A SECONDARY THREAD!
		 *
		 * Multiple instances of this may exist at the same time, since we typically have 2 or more loading threads.
		 *
		 * [[[*** BEWARE! ***]]]  There may be multiple copies of the loading thread running.  Only values in WorkUnit may be
		 * modified.  Values in other places, like SharedInfo, can only be modified with mutex protection or with std::atomic.
		 */

		try
		{
			cfg_and_state.set_thread_name("detector map image loading thread #" + std::to_string(loading_thread_id));

			const int w = shared_info.net.w;
			const int h = shared_info.net.h;
			const int c = shared_info.net.c;
			const size_t num_predict_queues = shared_info.predict_queues.size();

			for (auto iter = shared_info.validation_image_filenames[loading_thread_id].begin(); iter != shared_info.validation_image_filenames[loading_thread_id].end() and cfg_and_state.must_immediately_exit == false and shared_info.done == false; iter ++)
			{
				shared_info.count_load_performed ++;
				const std::string filename = *iter;
				if (cfg_and_state.is_trace)
				{
					*cfg_and_state.output << "-> " << shared_info.count_load_performed << ": loading " << filename << std::endl;
				}

				// load the image and resize it to match the network dimensions
				WorkUnit work;
				work.filename = filename;
				work.img = Darknet::load_image(filename.c_str(), w, h, c);

				// Pick the destination predict queue with a wait-free atomic counter.  Round-robin gives each predict
				// thread roughly equal work; if one thread happens to be slower its queue will fill and push() below
				// will block this loader, which is exactly the desired backpressure.
				const size_t target_queue = shared_info.loader_dispatch_counter.fetch_add(1, std::memory_order_relaxed) % num_predict_queues;

				if (not shared_info.predict_queues[target_queue]->push(std::move(work)))
				{
					// queue was closed while we were trying to push -- shutdown is in progress, free what we still own
					Darknet::free_image(work.img);
					break;
				}
			}

			cfg_and_state.del_thread_name();
		}
		catch (const std::exception & e)
		{
			shared_info.done = true;
			for (auto & q : shared_info.predict_queues) q->close();
			shared_info.analyze_queue->close();
			darknet_fatal_error(DARKNET_LOC, "exception caught while loading images for map: %s", e.what());
		}

		return;
	}


	/* *************************************** */
	/* PREDICT PREDICT PREDICT PREDICT PREDICT */
	/* *************************************** */

	/** Get Darknet predictions for each image (single-threaded prediction path, used when @c thread_val == 1).
	 *
	 * @note This is called on a secondary thread!
	 *
	 * In single-threaded mode there is exactly one predict queue, so this thread pops from @c predict_queues[0].
	 * The old "swap the input map locally and process in batch" trick is no longer necessary -- it existed only to
	 * amortise the cost of holding the input mutex, which we don't have anymore.
	 */
	void detector_map_prediction_thread(SharedInfo & shared_info)
	{
		TAT(TATPARMS);

		// THIS IS CALLED ON A SECONDARY THREAD!

		try
		{
			cfg_and_state.set_thread_name("map prediction thread");

#ifdef DARKNET_GPU
			// Route all CUDA kernels and cuBLAS calls in this thread to the network's own
			// dedicated stream.  Without this, every thread would race through the single
			// global device stream, serialising GPU work and leaving SM capacity idle.
			activate_network_streams(shared_info.net);
#endif

			BoundedQueue<WorkUnit> & my_queue = *shared_info.predict_queues[0];

			WorkUnit work;
			while (my_queue.pop(work))
			{
				if (cfg_and_state.must_immediately_exit or shared_info.done)
				{
					Darknet::free_image(work.img);
					break;
				}

				shared_info.count_predict_performed ++;
				if (cfg_and_state.is_trace)
				{
					*cfg_and_state.output << "-> " << shared_info.count_predict_performed << ": predicting with " << work.filename << std::endl;
				}

				// the image prediction work starts here!

				network_predict(shared_info.net, work.img.data);
				const int image_width	= work.img.w;
				const int image_height	= work.img.h;
				Darknet::free_image(work.img);

				const float hierarchy_threshold = 0.5f;
				if (shared_info.net.letter_box == LETTERBOX_DATA) /// @todo we should eventually get rid of letterbox
				{
					work.predictions = get_network_boxes(&shared_info.net, image_width, image_height, shared_info.detection_threshold, hierarchy_threshold, 0, 1, &work.number_of_predictions, shared_info.net.letter_box);
				}
				else
				{
					work.predictions = get_network_boxes(&shared_info.net, 1, 1, shared_info.detection_threshold, hierarchy_threshold, 0, 0, &work.number_of_predictions, shared_info.net.letter_box);
				}

				if (shared_info.nms)
				{
					if (shared_info.output_layer.nms_kind == DEFAULT_NMS)
					{
						do_nms_sort(work.predictions, work.number_of_predictions, shared_info.output_layer.classes, shared_info.nms);
					}
					else
					{
						// normal codepath is here
						diounms_sort(work.predictions, work.number_of_predictions, shared_info.output_layer.classes, shared_info.nms, shared_info.output_layer.nms_kind, shared_info.output_layer.beta_nms);
					}
				}

				if (not shared_info.analyze_queue->push(std::move(work)))
				{
					// shutdown in progress
					free_detections(work.predictions, work.number_of_predictions);
					break;
				}
			}

			cfg_and_state.del_thread_name();
		}
		catch (const std::exception & e)
		{
			shared_info.done = true;
			for (auto & q : shared_info.predict_queues) q->close();
			shared_info.analyze_queue->close();
			darknet_fatal_error(DARKNET_LOC, "exception caught while obtaining predictions for map: %s", e.what());
		}

		return;
	}


	/* *************************************************************** */
	/* MULTI-THREADED PREDICTION (when thread_val > 1 in .cfg)         */
	/* Uses one network per prediction thread.                         */
	/* *************************************************************** */

	/** Multi-threaded prediction thread - each thread has its own network and its own input queue.
	 *
	 * This is used when @c thread_val > 1 is specified in the .cfg file.  Each instance pops from
	 * @c predict_queues[thread_id], so N predict threads can each take work off their own queue concurrently
	 * without waiting on a shared mutex.  This is the change that allows the GPU to stay busy: previously every
	 * predict thread had to serialise on a single input mutex even when there was plenty of work, which left
	 * gaps between CUDA kernels.  Now each thread's pop path is independent of the others.
	 *
	 * @note This is called on a secondary thread!
	 */
	void detector_map_mt_prediction_thread(
		const int thread_id,
		SharedInfo & shared_info,
		Darknet::Network & my_net,
		const Darknet::Layer & output_layer)
	{
		TAT(TATPARMS);

		try
		{
			cfg_and_state.set_thread_name("map mt predict #" + std::to_string(thread_id));

#ifdef DARKNET_GPU
			// Each prediction thread gets its own network (my_net) which already has a
			// dedicated cudaStream and cublasHandle allocated.  Binding those to this
			// thread's TLS means every kernel and BLAS call in this thread automatically
			// targets the right stream — enabling N threads to saturate N concurrent GPU
			// streams with zero extra synchronisation.
			activate_network_streams(my_net);
#endif

			BoundedQueue<WorkUnit> & my_queue = *shared_info.predict_queues[thread_id];

			WorkUnit work;
			while (my_queue.pop(work))
			{
				if (cfg_and_state.must_immediately_exit or shared_info.done)
				{
					Darknet::free_image(work.img);
					break;
				}

				if (cfg_and_state.is_trace)
				{
					*cfg_and_state.output << "-> thread #" << thread_id << ": predicting " << work.filename << std::endl;
				}

				network_predict(my_net, work.img.data);
				const int image_width = work.img.w;
				const int image_height = work.img.h;
				Darknet::free_image(work.img);

				const float hierarchy_threshold = 0.5f;
				if (my_net.letter_box == LETTERBOX_DATA)
				{
					work.predictions = get_network_boxes(&my_net, image_width, image_height, shared_info.detection_threshold, hierarchy_threshold, 0, 1, &work.number_of_predictions, my_net.letter_box);
				}
				else
				{
					work.predictions = get_network_boxes(&my_net, 1, 1, shared_info.detection_threshold, hierarchy_threshold, 0, 0, &work.number_of_predictions, my_net.letter_box);
				}

				if (shared_info.nms)
				{
					if (output_layer.nms_kind == DEFAULT_NMS)
					{
						do_nms_sort(work.predictions, work.number_of_predictions, output_layer.classes, shared_info.nms);
					}
					else
					{
						diounms_sort(work.predictions, work.number_of_predictions, output_layer.classes, shared_info.nms, output_layer.nms_kind, output_layer.beta_nms);
					}
				}

				// Bump the global counter and the per-thread counter together, so the diagnostic display can report
				// both totals and per-thread progress.
				shared_info.count_predict_performed ++;
				shared_info.per_thread_predict_count[thread_id] ++;

				if (not shared_info.analyze_queue->push(std::move(work)))
				{
					free_detections(work.predictions, work.number_of_predictions);
					break;
				}
			}

			cfg_and_state.del_thread_name();
		}
		catch (const std::exception & e)
		{
			shared_info.done = true;
			for (auto & q : shared_info.predict_queues) q->close();
			shared_info.analyze_queue->close();
			darknet_fatal_error(DARKNET_LOC, "exception caught in multi-threaded prediction for map (thread #%d): %s", thread_id, e.what());
		}

		return;
	}


	/* ******************************************** */
	/* ANALYSIS ANALYSIS ANALYSIS ANALYSIS ANALYSIS */
	/* ******************************************** */

	/** Run mAP calculations for each image.  Ground truth annotations are also loaded here to reduce the tasks done
	 * by the image loading thread.  @note This is called on a secondary thread!
	 *
	 * The analysis thread is the single sequential consumer of prediction results, which is fine because all of the
	 * shared state it mutates (@c box_probabilities, @c ground_truth_counts, etc.) is single-writer here.  Moving
	 * these accesses to multiple threads would require additional locking that would likely cost more than it saves.
	 */
	void detector_map_calculations_thread(SharedInfo & shared_info)
	{
		TAT(TATPARMS);

		// THIS IS CALLED ON A SECONDARY THREAD!

		try
		{
			cfg_and_state.set_thread_name("map calculations thread");

			WorkUnit work;
			while (shared_info.analyze_queue->pop(work))
			{
				if (cfg_and_state.must_immediately_exit or shared_info.done)
				{
					free_detections(work.predictions, work.number_of_predictions);
					break;
				}

				shared_info.count_analyze_performed ++;
				if (cfg_and_state.is_trace)
				{
					*cfg_and_state.output << "-> " << shared_info.count_analyze_performed << ": performing calculations with " << work.filename << std::endl;
				}

				// load the ground truth annotations for this image
				const auto ground_truth_fn = std::filesystem::path(work.filename).replace_extension(".txt");
				work.ground_truth_labels = read_boxes(ground_truth_fn.string().c_str(), &work.number_of_ground_truth_labels);
				if (cfg_and_state.is_trace)
				{
					*cfg_and_state.output << "-> " << shared_info.count_load_performed << ": loading " << ground_truth_fn.string() << " (" << work.number_of_ground_truth_labels << " ground truth labels)" << std::endl;
				}
				for (int j = 0; j < work.number_of_ground_truth_labels and cfg_and_state.must_immediately_exit == false; ++j)
				{
					const auto & ground_truth = work.ground_truth_labels[j];
					if (ground_truth.id < 0 or ground_truth.id >= shared_info.number_of_classes)
					{
						darknet_fatal_error(DARKNET_LOC, "invalid ground truth: class id #%d at line #%d in %s", ground_truth.id, j+1, ground_truth_fn.string().c_str());
					}

					// get an accurate count of ground truths for every class in the neural network
					shared_info.ground_truth_counts[ground_truth.id] ++;
				}

				const size_t checkpoint_box_probabilities = shared_info.box_probabilities.size();

				// go through all the predictions in this image, and try to match each one to a ground truth
				for (size_t idx = 0; idx < work.number_of_predictions and cfg_and_state.must_immediately_exit == false; idx ++)
				{
					const auto & prediction = work.predictions[idx];

					for (int class_id = 0; class_id < shared_info.number_of_classes and cfg_and_state.must_immediately_exit == false; class_id ++)
					{
						const float probability = prediction.prob[class_id];
						if (probability <= 0.0f)
						{
							// this prediction does not include this class
							continue;
						}

						shared_info.prediction_counts[class_id] ++;

						BoxProbability bp;
						bp.bb			= prediction.bbox;
						bp.probability	= probability;
						bp.class_id		= class_id;
						shared_info.box_probabilities.push_back(bp);

						auto & box_probability = *shared_info.box_probabilities.rbegin();

						// see which ground truth best matches this prediction

						int truth_index = -1;
						float best_iou = 0.0f;
						for (int j = 0; j < work.number_of_ground_truth_labels and cfg_and_state.must_immediately_exit == false; j++)
						{
							const auto & ground_truth = work.ground_truth_labels[j];

							if (ground_truth.id != bp.class_id)
							{
								continue;
							}

							// the classes match, so now figure out the IoU between the prediction and the ground truth

							const Darknet::Box box = {ground_truth.x, ground_truth.y, ground_truth.w, ground_truth.h};
							const float current_iou = box_iou(prediction.bbox, box);
							if (current_iou > shared_info.iou_threshold and current_iou > best_iou)
							{
								best_iou = current_iou;
								truth_index = shared_info.unique_truth_count + j;
							}
						}

						// remember the best IoU
						if (truth_index > -1)
						{
							box_probability.matched_ground_truth = true;
							box_probability.unique_truth_index = truth_index;
						}

						// calc avg IoU, true-positives, false-positives for required Threshold
						if (probability > shared_info.thresh_calc_avg_iou)
						{
							bool found = false;

							/* Mirror the old logic: only compare against detections from this image, i.e., those added since
							 * checkpoint_box_probabilities.
							 * shared_info.box_probabilities.size() is the *current* global count.  We stop at size() - 1 to avoid
							 * comparing the detection to itself.
							 */
							const size_t current_count = shared_info.box_probabilities.size();

							if (current_count > checkpoint_box_probabilities)
							{
								for (size_t z = checkpoint_box_probabilities; z < current_count - 1 and cfg_and_state.must_immediately_exit == false; ++z)
								{
									if (shared_info.box_probabilities[z].unique_truth_index == truth_index)
									{
										found = true;
										break;
									}
								}
							}

							if (truth_index > -1 and found == false)
							{
								shared_info.avg_iou += best_iou;
								shared_info.tp_for_thresh ++;
								shared_info.avg_iou_per_class[class_id] += best_iou;
								shared_info.tp_for_thresh_per_class[class_id]++;
							}
							else
							{
								shared_info.fp_for_thresh ++;
								shared_info.fp_for_thresh_per_class[class_id] ++;
							}
						}
					}
				}

				shared_info.unique_truth_count += work.number_of_ground_truth_labels;
				free(work.ground_truth_labels);
				work.ground_truth_labels = nullptr;
				free_detections(work.predictions, work.number_of_predictions);
				work.predictions = nullptr;
			}

			cfg_and_state.del_thread_name();
		}
		catch (const std::exception & e)
		{
			shared_info.done = true;
			for (auto & q : shared_info.predict_queues) q->close();
			shared_info.analyze_queue->close();
			darknet_fatal_error(DARKNET_LOC, "exception caught while calculating results for map: %s", e.what());
		}

		return;
	}

	/* ****************************************************** */
	/* ADDITIONAL MAP PREDICTION NETWORK CACHE FOR thread_val */
	/* ****************************************************** */

	struct DetectorMapPredictionNetworkDeleter
	{
		void operator()(Darknet::Network * net) const
		{
			if (net)
			{
				free_network(*net);
				delete net;
			}
		}
	};

	using DetectorMapPredictionNetworkPtr = std::unique_ptr<Darknet::Network, DetectorMapPredictionNetworkDeleter>;

	struct DetectorMapPredictionNetworkCache
	{
		std::string cfgfile;
		int prediction_threads = 0;
		int net_w = 0;
		int net_h = 0;
		int net_c = 0;
		std::vector<DetectorMapPredictionNetworkPtr> nets;

		bool matches(const char * requested_cfgfile, const Darknet::Network & source_net, const int requested_threads) const
		{
			const size_t additional_network_count = requested_threads > 1 ? static_cast<size_t>(requested_threads - 1) : 0UL;
			return requested_threads > 1 and
				prediction_threads == requested_threads and
				net_w == source_net.w and
				net_h == source_net.h and
				net_c == source_net.c and
				cfgfile == (requested_cfgfile ? requested_cfgfile : "") and
				nets.size() == additional_network_count;
		}

		void clear()
		{
			nets.clear();
			cfgfile.clear();
			prediction_threads = 0;
			net_w = 0;
			net_h = 0;
			net_c = 0;
		}
	};

	static DetectorMapPredictionNetworkCache detector_map_prediction_cache;
	static std::mutex detector_map_prediction_cache_mutex;

	static size_t positive_size(const int value)
	{
		return value > 0 ? static_cast<size_t>(value) : 0UL;
	}

	static size_t layer_bias_count(const Darknet::Layer & l)
	{
		return l.nbiases > 0 ? static_cast<size_t>(l.nbiases) : positive_size(l.n);
	}

	static void copy_host_array_no_alloc(float * dst, const float * src, const size_t count)
	{
		if (dst and src and count > 0)
		{
			std::memcpy(dst, src, count * sizeof(float));
		}
	}

	static void copy_param_no_alloc(float * dst_cpu, const float * src_cpu, const size_t count)
	{
		copy_host_array_no_alloc(dst_cpu, src_cpu, count);
	}

#ifdef DARKNET_GPU
	static void copy_param_no_alloc(float * dst_cpu, float * dst_gpu, const float * src_cpu, const float * src_gpu, const size_t count)
	{
		if (count == 0)
		{
			return;
		}

		const size_t bytes = count * sizeof(float);

		if (cfg_and_state.gpu_index >= 0 and src_gpu)
		{
			if (dst_gpu)
			{
				CHECK_CUDA(cudaMemcpy(dst_gpu, src_gpu, bytes, cudaMemcpyDeviceToDevice));
			}
			if (dst_cpu)
			{
				CHECK_CUDA(cudaMemcpy(dst_cpu, src_gpu, bytes, cudaMemcpyDeviceToHost));
			}
			return;
		}

		copy_host_array_no_alloc(dst_cpu, src_cpu, count);

		if (cfg_and_state.gpu_index >= 0 and dst_gpu)
		{
			const float * cpu_source = dst_cpu ? dst_cpu : src_cpu;
			if (cpu_source)
			{
				cuda_push_array(dst_gpu, const_cast<float*>(cpu_source), count);
			}
		}
	}
#endif

	static void copy_common_layer_params_no_alloc(const Darknet::Layer & src, Darknet::Layer & dst)
	{
		const size_t weights_count = positive_size(src.nweights);
		const size_t biases_count = layer_bias_count(src);
		const size_t filters_count = positive_size(src.n);

#ifdef DARKNET_GPU
		copy_param_no_alloc(dst.weights, dst.weights_gpu, src.weights, src.weights_gpu, weights_count);
		copy_param_no_alloc(dst.biases, dst.biases_gpu, src.biases, src.biases_gpu, biases_count);
		copy_param_no_alloc(dst.scales, dst.scales_gpu, src.scales, src.scales_gpu, filters_count);
		copy_param_no_alloc(dst.rolling_mean, dst.rolling_mean_gpu, src.rolling_mean, src.rolling_mean_gpu, filters_count);
		copy_param_no_alloc(dst.rolling_variance, dst.rolling_variance_gpu, src.rolling_variance, src.rolling_variance_gpu, filters_count);
		copy_param_no_alloc(dst.binary_weights, dst.binary_weights_gpu, src.binary_weights, src.binary_weights_gpu, weights_count);
#else
		copy_param_no_alloc(dst.weights, src.weights, weights_count);
		copy_param_no_alloc(dst.biases, src.biases, biases_count);
		copy_param_no_alloc(dst.scales, src.scales, filters_count);
		copy_param_no_alloc(dst.rolling_mean, src.rolling_mean, filters_count);
		copy_param_no_alloc(dst.rolling_variance, src.rolling_variance, filters_count);
		copy_param_no_alloc(dst.binary_weights, src.binary_weights, weights_count);
#endif
	}

	static void copy_deform_conv_params_no_alloc(const Darknet::Layer & src, Darknet::Layer & dst)
	{
		const int offset_filters = 2 * src.size * src.size;
		const int mask_filters = src.size * src.size;
		const size_t offset_weights_count = positive_size(src.c * offset_filters * src.size * src.size);
		const size_t mask_weights_count = positive_size(src.c * mask_filters * src.size * src.size);

#ifdef DARKNET_GPU
		copy_param_no_alloc(dst.offset_weights, dst.offset_weights_gpu, src.offset_weights, src.offset_weights_gpu, offset_weights_count);
		copy_param_no_alloc(dst.offset_biases, dst.offset_biases_gpu, src.offset_biases, src.offset_biases_gpu, positive_size(offset_filters));
		copy_param_no_alloc(dst.mask_weights, dst.mask_weights_gpu, src.mask_weights, src.mask_weights_gpu, mask_weights_count);
		copy_param_no_alloc(dst.mask_biases, dst.mask_biases_gpu, src.mask_biases, src.mask_biases_gpu, positive_size(mask_filters));
#else
		copy_param_no_alloc(dst.offset_weights, src.offset_weights, offset_weights_count);
		copy_param_no_alloc(dst.offset_biases, src.offset_biases, positive_size(offset_filters));
		copy_param_no_alloc(dst.mask_weights, src.mask_weights, mask_weights_count);
		copy_param_no_alloc(dst.mask_biases, src.mask_biases, positive_size(mask_filters));
#endif
	}

	static void copy_dcnv4_params_no_alloc(const Darknet::Layer & src, Darknet::Layer & dst)
	{
		int k = src.size * src.size;
		if (src.remove_center)
		{
			k -= 1;
		}
		const int offset_filters_raw = src.groups * k * 3;
		const int padded_offset_dim = ((offset_filters_raw + 7) / 8) * 8;
		const size_t offset_weights_count = positive_size(padded_offset_dim * src.c * src.size * src.size);

#ifdef DARKNET_GPU
		copy_param_no_alloc(dst.offset_weights, dst.offset_weights_gpu, src.offset_weights, src.offset_weights_gpu, offset_weights_count);
		copy_param_no_alloc(dst.offset_biases, dst.offset_biases_gpu, src.offset_biases, src.offset_biases_gpu, positive_size(padded_offset_dim));
#else
		copy_param_no_alloc(dst.offset_weights, src.offset_weights, offset_weights_count);
		copy_param_no_alloc(dst.offset_biases, src.offset_biases, positive_size(padded_offset_dim));
#endif
	}

	static void copy_mambavision_params_no_alloc(const Darknet::Layer & src, Darknet::Layer & dst);
	static void copy_clifford_params_no_alloc(const Darknet::Layer & src, Darknet::Layer & dst);
	static void copy_wmhf_params_no_alloc(const Darknet::Layer & src, Darknet::Layer & dst);
	static void copy_layer_learned_params_no_alloc(const Darknet::Layer & src, Darknet::Layer & dst);

	static void copy_mambavision_params_no_alloc(const Darknet::Layer & src, Darknet::Layer & dst)
	{
		const int C = src.c;
		const int N = src.n;
		const int D = N / 2;
		const int R = src.mv_dt_rank;
		const int S = src.mv_d_state;
		const int P = R + 2 * S;
		const int ffn_hidden = N * src.mv_ffn_ratio;

		copy_param_no_alloc(dst.mv_conv_x, src.mv_conv_x, positive_size(D * src.mv_conv_size));
		copy_param_no_alloc(dst.mv_conv_x_bias, src.mv_conv_x_bias, positive_size(D));
		copy_param_no_alloc(dst.mv_conv_z, src.mv_conv_z, positive_size(D * src.mv_conv_size));
		copy_param_no_alloc(dst.mv_conv_z_bias, src.mv_conv_z_bias, positive_size(D));
		copy_param_no_alloc(dst.mv_x_proj, src.mv_x_proj, positive_size(P * D));
		copy_param_no_alloc(dst.mv_dt_proj, src.mv_dt_proj, positive_size(D * R));
		copy_param_no_alloc(dst.mv_dt_bias, src.mv_dt_bias, positive_size(D));
		copy_param_no_alloc(dst.mv_out_proj, src.mv_out_proj, positive_size(N * N));
		copy_param_no_alloc(dst.mv_out_bias, src.mv_out_bias, positive_size(N));
		copy_param_no_alloc(dst.mv_res_proj, src.mv_res_proj, positive_size(N * C));
		copy_param_no_alloc(dst.mv_ffn_w1, src.mv_ffn_w1, positive_size(ffn_hidden * N));
		copy_param_no_alloc(dst.mv_ffn_b1, src.mv_ffn_b1, positive_size(ffn_hidden));
		copy_param_no_alloc(dst.mv_ffn_w2, src.mv_ffn_w2, positive_size(N * ffn_hidden));
		copy_param_no_alloc(dst.mv_ffn_b2, src.mv_ffn_b2, positive_size(N));

#ifdef DARKNET_GPU
		copy_param_no_alloc(dst.mv_ln1_gamma, dst.mv_ln1_gamma_gpu, src.mv_ln1_gamma, src.mv_ln1_gamma_gpu, positive_size(C));
		copy_param_no_alloc(dst.mv_ln1_beta, dst.mv_ln1_beta_gpu, src.mv_ln1_beta, src.mv_ln1_beta_gpu, positive_size(C));
		copy_param_no_alloc(dst.mv_ln2_gamma, dst.mv_ln2_gamma_gpu, src.mv_ln2_gamma, src.mv_ln2_gamma_gpu, positive_size(N));
		copy_param_no_alloc(dst.mv_ln2_beta, dst.mv_ln2_beta_gpu, src.mv_ln2_beta, src.mv_ln2_beta_gpu, positive_size(N));
		copy_param_no_alloc(dst.mv_A_log, dst.mv_A_log_gpu, src.mv_A_log, src.mv_A_log_gpu, positive_size(D * S));
		copy_param_no_alloc(dst.mv_D, dst.mv_D_gpu, src.mv_D, src.mv_D_gpu, positive_size(D));
#else
		copy_param_no_alloc(dst.mv_ln1_gamma, src.mv_ln1_gamma, positive_size(C));
		copy_param_no_alloc(dst.mv_ln1_beta, src.mv_ln1_beta, positive_size(C));
		copy_param_no_alloc(dst.mv_ln2_gamma, src.mv_ln2_gamma, positive_size(N));
		copy_param_no_alloc(dst.mv_ln2_beta, src.mv_ln2_beta, positive_size(N));
		copy_param_no_alloc(dst.mv_A_log, src.mv_A_log, positive_size(D * S));
		copy_param_no_alloc(dst.mv_D, src.mv_D, positive_size(D));
#endif

		// GPU MambaVision uses its sublayers for the projection and depthwise-conv weights.
		// Copy those buffers directly, instead of rebuilding or reallocating them.
		if (src.mv_in_proj_layer and dst.mv_in_proj_layer) copy_layer_learned_params_no_alloc(*src.mv_in_proj_layer, *dst.mv_in_proj_layer);
		if (src.mv_conv_x_layer and dst.mv_conv_x_layer) copy_layer_learned_params_no_alloc(*src.mv_conv_x_layer, *dst.mv_conv_x_layer);
		if (src.mv_conv_z_layer and dst.mv_conv_z_layer) copy_layer_learned_params_no_alloc(*src.mv_conv_z_layer, *dst.mv_conv_z_layer);
		if (src.mv_x_proj_layer and dst.mv_x_proj_layer) copy_layer_learned_params_no_alloc(*src.mv_x_proj_layer, *dst.mv_x_proj_layer);
		if (src.mv_dt_proj_layer and dst.mv_dt_proj_layer) copy_layer_learned_params_no_alloc(*src.mv_dt_proj_layer, *dst.mv_dt_proj_layer);
		if (src.mv_out_proj_layer and dst.mv_out_proj_layer) copy_layer_learned_params_no_alloc(*src.mv_out_proj_layer, *dst.mv_out_proj_layer);
		if (src.mv_res_proj_layer and dst.mv_res_proj_layer) copy_layer_learned_params_no_alloc(*src.mv_res_proj_layer, *dst.mv_res_proj_layer);
		if (src.mv_ffn1_layer and dst.mv_ffn1_layer) copy_layer_learned_params_no_alloc(*src.mv_ffn1_layer, *dst.mv_ffn1_layer);
		if (src.mv_ffn2_layer and dst.mv_ffn2_layer) copy_layer_learned_params_no_alloc(*src.mv_ffn2_layer, *dst.mv_ffn2_layer);
	}

	static void copy_clifford_params_no_alloc(const Darknet::Layer & src, Darknet::Layer & dst)
	{
		if (src.cli_proj_in_dim != dst.cli_proj_in_dim or
			src.cli_num_dwconv != dst.cli_num_dwconv or
			src.cli_gffn_mode != dst.cli_gffn_mode)
		{
			darknet_fatal_error(DARKNET_LOC,
				"mAP prediction cache Clifford mismatch: src proj=%d dwconv=%d mode=%d, dst proj=%d dwconv=%d mode=%d",
				src.cli_proj_in_dim, src.cli_num_dwconv, src.cli_gffn_mode,
				dst.cli_proj_in_dim, dst.cli_num_dwconv, dst.cli_gffn_mode);
		}

		const int C = src.c;
		const size_t det_count = positive_size(C * C);
		const size_t proj_count = positive_size(C * src.cli_proj_in_dim);
		const size_t gate_count = positive_size(C * 2 * C);

#ifdef DARKNET_GPU
		copy_param_no_alloc(dst.cli_ln_gamma, dst.cli_ln_gamma_gpu, src.cli_ln_gamma, src.cli_ln_gamma_gpu, positive_size(C));
		copy_param_no_alloc(dst.cli_ln_beta, dst.cli_ln_beta_gpu, src.cli_ln_beta, src.cli_ln_beta_gpu, positive_size(C));
		copy_param_no_alloc(dst.cli_layer_scale, dst.cli_layer_scale_gpu, src.cli_layer_scale, src.cli_layer_scale_gpu, positive_size(C));

		copy_param_no_alloc(dst.cli_w_det, dst.cli_w_det_gpu, src.cli_w_det, src.cli_w_det_gpu, det_count);
		copy_param_no_alloc(dst.cli_b_det, dst.cli_b_det_gpu, src.cli_b_det, src.cli_b_det_gpu, positive_size(C));
		copy_param_no_alloc(dst.cli_w_proj, dst.cli_w_proj_gpu, src.cli_w_proj, src.cli_w_proj_gpu, proj_count);
		copy_param_no_alloc(dst.cli_b_proj, dst.cli_b_proj_gpu, src.cli_b_proj, src.cli_b_proj_gpu, positive_size(C));
		copy_param_no_alloc(dst.cli_w_gate, dst.cli_w_gate_gpu, src.cli_w_gate, src.cli_w_gate_gpu, gate_count);
		copy_param_no_alloc(dst.cli_b_gate, dst.cli_b_gate_gpu, src.cli_b_gate, src.cli_b_gate_gpu, positive_size(C));

		if (src.cli_gffn_mode != 0)
		{
			copy_param_no_alloc(dst.cli_w_proj_g, dst.cli_w_proj_g_gpu, src.cli_w_proj_g, src.cli_w_proj_g_gpu, proj_count);
			copy_param_no_alloc(dst.cli_b_proj_g, dst.cli_b_proj_g_gpu, src.cli_b_proj_g, src.cli_b_proj_g_gpu, positive_size(C));
			copy_param_no_alloc(dst.cli_w_gate_g, dst.cli_w_gate_g_gpu, src.cli_w_gate_g, src.cli_w_gate_g_gpu, gate_count);
			copy_param_no_alloc(dst.cli_b_gate_g, dst.cli_b_gate_g_gpu, src.cli_b_gate_g, src.cli_b_gate_g_gpu, positive_size(C));
		}
#else
		copy_param_no_alloc(dst.cli_ln_gamma, src.cli_ln_gamma, positive_size(C));
		copy_param_no_alloc(dst.cli_ln_beta, src.cli_ln_beta, positive_size(C));
		copy_param_no_alloc(dst.cli_layer_scale, src.cli_layer_scale, positive_size(C));

		copy_param_no_alloc(dst.cli_w_det, src.cli_w_det, det_count);
		copy_param_no_alloc(dst.cli_b_det, src.cli_b_det, positive_size(C));
		copy_param_no_alloc(dst.cli_w_proj, src.cli_w_proj, proj_count);
		copy_param_no_alloc(dst.cli_b_proj, src.cli_b_proj, positive_size(C));
		copy_param_no_alloc(dst.cli_w_gate, src.cli_w_gate, gate_count);
		copy_param_no_alloc(dst.cli_b_gate, src.cli_b_gate, positive_size(C));

		if (src.cli_gffn_mode != 0)
		{
			copy_param_no_alloc(dst.cli_w_proj_g, src.cli_w_proj_g, proj_count);
			copy_param_no_alloc(dst.cli_b_proj_g, src.cli_b_proj_g, positive_size(C));
			copy_param_no_alloc(dst.cli_w_gate_g, src.cli_w_gate_g, gate_count);
			copy_param_no_alloc(dst.cli_b_gate_g, src.cli_b_gate_g, positive_size(C));
		}
#endif

		for (int i = 0; i < src.cli_num_dwconv; ++i)
		{
			copy_layer_learned_params_no_alloc(src.cli_dwconv[i], dst.cli_dwconv[i]);
		}
	}

	static void copy_wmhf_params_no_alloc(const Darknet::Layer & src, Darknet::Layer & dst)
	{
		static constexpr int wmhf_sub_count = 7;
		if (src.input_layer == nullptr or dst.input_layer == nullptr)
		{
			darknet_fatal_error(DARKNET_LOC, "mAP prediction cache WMHF mismatch while copying weights");
		}

		for (int i = 0; i < wmhf_sub_count; ++i)
		{
			copy_layer_learned_params_no_alloc(src.input_layer[i], dst.input_layer[i]);
		}
	}

	static void copy_layer_learned_params_no_alloc(const Darknet::Layer & src, Darknet::Layer & dst)
	{
		if (src.type != dst.type)
		{
			darknet_fatal_error(DARKNET_LOC, "mAP prediction cache layer type mismatch: source=%d destination=%d", static_cast<int>(src.type), static_cast<int>(dst.type));
		}

		copy_common_layer_params_no_alloc(src, dst);

		switch (src.type)
		{
			case Darknet::ELayerType::DEFORM_CONV:
				copy_deform_conv_params_no_alloc(src, dst);
				break;

			case Darknet::ELayerType::DCNV4:
				copy_dcnv4_params_no_alloc(src, dst);
				break;

			case Darknet::ELayerType::MAMBAVISION:
				copy_mambavision_params_no_alloc(src, dst);
				break;

				case Darknet::ELayerType::CLIFFORD:
					copy_clifford_params_no_alloc(src, dst);
					break;

				case Darknet::ELayerType::WMHF:
					copy_wmhf_params_no_alloc(src, dst);
					break;

				default:
					break;
			}

		dst.batch = 1;
		dst.steps = 1;
		dst.train = 0;
	}

	static void copy_detector_map_network_weights_no_alloc(const Darknet::Network & src, Darknet::Network & dst)
	{
		if (src.n != dst.n)
		{
			darknet_fatal_error(DARKNET_LOC, "mAP prediction cache network mismatch: source layers=%d destination layers=%d", src.n, dst.n);
		}

		dst.letter_box = src.letter_box;
		dst.benchmark_layers = src.benchmark_layers;
		dst.cudnn_half = src.cudnn_half;
		dst.cudnn_bf16 = src.cudnn_bf16;

		for (int i = 0; i < src.n; ++i)
		{
			copy_layer_learned_params_no_alloc(src.layers[i], dst.layers[i]);
		}
	}

	static void warmup_detector_map_prediction_network(Darknet::Network & net)
	{
		const int input_size = get_network_input_size(net);
		if (input_size <= 0)
		{
			return;
		}

		std::vector<float> zero_input(static_cast<size_t>(input_size), 0.0f);
		network_predict(net, zero_input.data());

#ifdef DARKNET_GPU
		if (cfg_and_state.gpu_index >= 0)
		{
			CHECK_CUDA(cudaPeekAtLastError());
		}
#endif
	}

	static bool cached_network_must_be_fused_to_match_source(const Darknet::Network & source_net, const Darknet::Network & candidate_net)
	{
		const int layers_to_check = std::min(source_net.n, candidate_net.n);
		for (int i = 0; i < layers_to_check; ++i)
		{
			const Darknet::Layer & source_layer = source_net.layers[i];
			const Darknet::Layer & candidate_layer = candidate_net.layers[i];
			if (source_layer.type == Darknet::ELayerType::CONVOLUTIONAL and
				candidate_layer.type == Darknet::ELayerType::CONVOLUTIONAL and
				source_layer.batch_normalize == 0 and
				candidate_layer.batch_normalize != 0)
			{
				return true;
			}
		}

		return false;
	}

	static void ensure_detector_map_prediction_cache(const char * cfgfile, const Darknet::Network & source_net, const int requested_threads)
	{
		if (requested_threads <= 1)
		{
			return;
		}

		std::lock_guard lock(detector_map_prediction_cache_mutex);
		if (detector_map_prediction_cache.matches(cfgfile, source_net, requested_threads))
		{
			return;
		}

		detector_map_prediction_cache.clear();
		detector_map_prediction_cache.cfgfile = cfgfile ? cfgfile : "";
		detector_map_prediction_cache.prediction_threads = requested_threads;
		detector_map_prediction_cache.net_w = source_net.w;
		detector_map_prediction_cache.net_h = source_net.h;
		detector_map_prediction_cache.net_c = source_net.c;
		const int additional_network_count = requested_threads - 1;
		detector_map_prediction_cache.nets.reserve(additional_network_count);

#ifdef DARKNET_GPU
		if (cfg_and_state.gpu_index >= 0)
		{
			cuda_set_device(source_net.gpu_index);
		}
#endif

		*cfg_and_state.output << "-> pre-allocating " << additional_network_count << " additional mAP prediction network"
			<< (additional_network_count == 1 ? "" : "s") << " for thread_val=" << requested_threads << std::endl;

		for (int t = 0; t < additional_network_count; ++t)
		{
			auto * new_net = new Darknet::Network();
			*new_net = parse_network_cfg_custom(detector_map_prediction_cache.cfgfile.c_str(), 1, 1);
			if (cached_network_must_be_fused_to_match_source(source_net, *new_net))
			{
				fuse_conv_batchnorm(*new_net);
				calculate_binary_weights(new_net);
			}
			set_batch_network(new_net, 1);
			copy_detector_map_network_weights_no_alloc(source_net, *new_net);
			warmup_detector_map_prediction_network(*new_net);
			detector_map_prediction_cache.nets.emplace_back(new_net);
		}
	}

}


void prepare_detector_map_thread_networks(const char * cfgfile, Darknet::Network * source_net)
{
	TAT(TATPARMS);

	if (source_net == nullptr or source_net->validation_threads <= 1)
	{
		return;
	}

	ensure_detector_map_prediction_cache(cfgfile, *source_net, source_net->validation_threads);
}


void release_detector_map_thread_networks()
{
	TAT(TATPARMS);

	std::lock_guard lock(detector_map_prediction_cache_mutex);
	detector_map_prediction_cache.clear();
}


/* ******************** */
/* DETECTOR MAP COMMAND */
/* ******************** */

float validate_detector_map(const char * datacfg, const char * cfgfile, const char * weightfile, float thresh_calc_avg_iou, const float iou_thresh, const int map_points, int letter_box, Darknet::Network * existing_net)
{
	TAT(TATPARMS);

	/* This function is called in 2 situations:
	 *
	 *		1) During training every once in a while to calculate mAP%
	 *
	 *		2) Manually from the CLI when running a command such as the following:
	 *
	 *				darknet detector map LegoGears.cfg LegoGears_best.weights LegoGears.data
	 *
	 * This re-write of validate_detector_map() was introduced in v5.1.  The previous function was deleted.
	 *
	 * Concurrency note (post-v5.1):  the producer/consumer plumbing originally used two std::map<string, WorkUnit>
	 * containers guarded by std::mutexes, with consumers polling+sleeping when their input was empty.  In multi-
	 * threaded prediction mode (thread_val > 1) every predict thread had to acquire the same input mutex to grab a
	 * work item, which serialised the predict path and left the GPU idle between kernels.  The current design
	 * replaces those containers with a BoundedQueue *per* predict thread, and loaders fan out round-robin across
	 * those queues.  Predict threads now have zero cross-thread contention on the input side, which is what
	 * unblocks GPU saturation when network_predict() is short.
	 */

	*cfg_and_state.output << "Calculating mAP (mean average precision) with threshold " << thresh_calc_avg_iou << " and IoU threshold " << iou_thresh << "." << std::endl;

	SharedInfo shared_info;

	// load the network, or re-use the network already loaded
	list * options = read_data_cfg(datacfg);
	std::string validation_filename = option_find_str(options, "valid", "");
	if (existing_net) // if we're being called in the middle of training a network
	{
		const std::string train_images = option_find_str(options, "train", "");
		validation_filename = option_find_str(options, "valid", train_images);
		shared_info.net = *existing_net;
		free_network_recurrent_state(*existing_net);
	}
	else
	{
		shared_info.net = parse_network_cfg_custom(cfgfile, 1, 1); // set batch=1
		if (weightfile)
		{
			load_weights(&shared_info.net, weightfile);
		}
		fuse_conv_batchnorm(shared_info.net);
		calculate_binary_weights(&shared_info.net);
		Darknet::load_names(&shared_info.net, option_find_str(options, "names", "unknown.names"));
	}
	free_list_contents_kvp(options);
	free_list(options);

	// Keep the prediction queue fed.  thread_val controls prediction threads; image loading
	// needs extra workers so fast GPUs do not starve while mAP is running.
	const size_t hardware_threads = std::max(2U, std::thread::hardware_concurrency());
	const size_t prediction_threads = std::max<size_t>(1UL, static_cast<size_t>(shared_info.net.validation_threads));
	const size_t requested_loading_threads = (shared_info.net.validation_threads > 1)
		? std::max(prediction_threads * 2UL, hardware_threads / 2UL)
		: std::max(4UL, hardware_threads / 2UL);
	shared_info.number_of_loading_threads_to_start = std::clamp(requested_loading_threads, 2UL, 16UL);

	/* Per-queue capacity.  In the previous design this was a single shared map's capacity; now we have one queue
	 * per predict thread plus one analyze queue, so total RAM headroom is roughly
	 * max_work_queue_size * (num_predict_threads + 1) * sizeof(image).  We keep the per-queue capacity smaller than
	 * the old single-map figure to avoid bloating total memory when thread_val is high.
	 */
	shared_info.max_work_queue_size = std::clamp(shared_info.number_of_loading_threads_to_start * 50UL, 100UL, 500UL);

	// split the validation images into multiple sets, where each one will be given to a different thread to load from disk
	shared_info.validation_image_filenames.resize(shared_info.number_of_loading_threads_to_start);
	if (std::filesystem::exists(validation_filename))
	{
		Darknet::SStr check_for_duplicate_filenames;
		std::ifstream ifs(validation_filename);
		std::string line;
		while (std::getline(ifs, line) and cfg_and_state.must_immediately_exit == false)
		{
			std::filesystem::path path = line;
			if (std::filesystem::exists(path) == false)
			{
				darknet_fatal_error(DARKNET_LOC, "%s line #%u: validation image filename is invalid: \"%s\"", validation_filename.c_str(), shared_info.total_number_of_validation_images + 1, path.string().c_str());
			}
			path = std::filesystem::canonical(path);
			if (check_for_duplicate_filenames.count(path.string()) != 0)
			{
				darknet_fatal_error(DARKNET_LOC, "%s line #%u: duplicate validation filename: \"%s\"", validation_filename.c_str(), shared_info.total_number_of_validation_images + 1, path.string().c_str());
			}
			check_for_duplicate_filenames.insert(path.string());

			const auto txt = std::filesystem::path(path).replace_extension(".txt");
			if (std::filesystem::exists(txt) == false)
			{
				darknet_fatal_error(DARKNET_LOC, "%s line #%u: validation image does not have a corresponding .txt annotation file: \"%s\"", validation_filename.c_str(), shared_info.total_number_of_validation_images + 1, path.string().c_str());
			}

			const size_t idx = (shared_info.total_number_of_validation_images % shared_info.number_of_loading_threads_to_start);
			shared_info.validation_image_filenames[idx].insert(path.string());
			shared_info.total_number_of_validation_images ++;
		}
	}

	if (shared_info.total_number_of_validation_images == 0)
	{
		darknet_fatal_error(DARKNET_LOC, "no validation images available (verify %s)", validation_filename.c_str());
	}

	const size_t actual_batch_size = shared_info.net.batch * shared_info.net.subdivisions;
	if (shared_info.total_number_of_validation_images < actual_batch_size)
	{
		Darknet::display_warning_msg("Warning: there seems to be very few validation images (num=" + std::to_string(shared_info.total_number_of_validation_images) + ", batch=" + std::to_string(actual_batch_size) + ")\n");
	}

	shared_info.output_layer = shared_info.net.layers[shared_info.net.n - 1];
	for (int k = 0; k < shared_info.net.n and cfg_and_state.must_immediately_exit == false; ++k)
	{
		Darknet::Layer & lk = shared_info.net.layers[k];
		if (lk.type == Darknet::ELayerType::YOLO			or
			lk.type == Darknet::ELayerType::GAUSSIAN_YOLO	or
			lk.type == Darknet::ELayerType::REGION			or
			lk.type == Darknet::ELayerType::YOLOX			or
			lk.type == Darknet::ELayerType::PPYOLOE			or
			lk.type == Darknet::ELayerType::YOLONAS			or
			lk.type == Darknet::ELayerType::CENTERNET		)
		{
			shared_info.output_layer = lk;
			*cfg_and_state.output << "-> detection layer #" << k << " is type " << static_cast<int>(lk.type) << " (" << Darknet::to_string(lk.type) << ")" << std::endl;
		}
	}

	shared_info.nms					= 0.45f;
	shared_info.iou_threshold		= iou_thresh;
	shared_info.detection_threshold	= 0.005f;
	shared_info.thresh_calc_avg_iou	= thresh_calc_avg_iou;
	shared_info.number_of_classes	= shared_info.output_layer.classes;

	shared_info.avg_iou_per_class		.resize(shared_info.number_of_classes, 0.0f);
	shared_info.tp_for_thresh_per_class	.resize(shared_info.number_of_classes, 0);
	shared_info.fp_for_thresh_per_class	.resize(shared_info.number_of_classes, 0);
	for (int class_idx = 0; class_idx < shared_info.number_of_classes and cfg_and_state.must_immediately_exit == false; class_idx ++)
	{
		shared_info.ground_truth_counts		[class_idx] = 0;
		shared_info.prediction_counts		[class_idx] = 0;
	}

	*cfg_and_state.output
		<< "-> " << shared_info.total_number_of_validation_images << " validation images"
		<< " for " << shared_info.number_of_classes << " class" << (shared_info.number_of_classes == 1 ? "" : "es") << std::endl
		<< "-> " << shared_info.number_of_loading_threads_to_start << " loading thread" << (shared_info.number_of_loading_threads_to_start == 1 ? "" : "s")
		<< " with per-queue capacity of " << shared_info.max_work_queue_size << " images" << std::endl;

	const auto timestamp_start = std::chrono::high_resolution_clock::now();

	// Determine if we should use multi-threaded prediction (thread_val > 1 in .cfg)
	const int num_prediction_threads = shared_info.net.validation_threads;
	const bool use_multi_threaded_prediction = (num_prediction_threads > 1);
	std::vector<Darknet::Network*> pred_nets;

	if (use_multi_threaded_prediction)
	{
		*cfg_and_state.output << "-> using " << num_prediction_threads << " prediction threads with pre-allocated GPU memory" << std::endl;

		pred_nets.reserve(num_prediction_threads);
		pred_nets.push_back(&shared_info.net);
		ensure_detector_map_prediction_cache(cfgfile, shared_info.net, num_prediction_threads);

		// Initialize per-thread prediction counters.
		shared_info.per_thread_predict_count = std::make_unique<std::atomic<size_t>[]>(num_prediction_threads);
		for (int t = 0; t < num_prediction_threads; ++t)
		{
			shared_info.per_thread_predict_count[t] = 0;
		}

		for (int t = 1; t < num_prediction_threads; ++t)
		{
			Darknet::Network * prediction_net = detector_map_prediction_cache.nets[t - 1].get();
			copy_detector_map_network_weights_no_alloc(shared_info.net, *prediction_net);
			pred_nets.push_back(prediction_net);
		}
	}

	/* Allocate the per-predict-thread input queues plus the single analyze queue.  This is the structural change
	 * that fixes the bottleneck:  in the old design all predict threads pulled from one shared mutex-guarded map.
	 * Now each predict thread has its own queue and its own pop lock, so they no longer serialise at the input.
	 */
	shared_info.predict_queues.reserve(std::max(1, num_prediction_threads));
	for (int t = 0; t < std::max(1, num_prediction_threads); ++t)
	{
		shared_info.predict_queues.emplace_back(std::make_unique<BoundedQueue<WorkUnit>>(shared_info.max_work_queue_size));
	}
	shared_info.analyze_queue = std::make_unique<BoundedQueue<WorkUnit>>(shared_info.max_work_queue_size * std::max(1, num_prediction_threads));

	// Reserve to avoid reallocations in the analysis thread's hot loop.  Soft hint -- the actual size depends on
	// per-image detection counts which we don't know yet.
	shared_info.box_probabilities.reserve(shared_info.total_number_of_validation_images * 32);

	/* ************************************************************************* */
	/* START THE THREADS THAT LOAD, PREDICT, AND RUNS THE NECESSARY CALCULATIONS */
	/* ************************************************************************* */

	Darknet::VThreads all_threads;
	for (size_t idx = 0; idx < shared_info.number_of_loading_threads_to_start and cfg_and_state.must_immediately_exit == false; idx ++)
	{
		*cfg_and_state.output << "-> starting loading thread #" << idx << " with " << shared_info.validation_image_filenames[idx].size() << " images" << std::endl;
		all_threads.emplace_back(detector_map_loading_thread, idx, std::ref(shared_info));
	}

	if (use_multi_threaded_prediction)
	{
		// Multi-threaded prediction: thread #0 uses shared_info.net, remaining
		// threads use cached additional networks.
		for (int t = 0; t < num_prediction_threads and cfg_and_state.must_immediately_exit == false; ++t)
		{
			*cfg_and_state.output << "-> starting prediction thread #" << t << " with dedicated network" << std::endl;
			all_threads.emplace_back(
				detector_map_mt_prediction_thread,
				t,
				std::ref(shared_info),
				std::ref(*pred_nets[t]),
				std::cref(shared_info.output_layer)
			);
		}
	}
	else
	{
		// Single-threaded prediction: use shared network (default behavior)
		all_threads.emplace_back(detector_map_prediction_thread, std::ref(shared_info));
	}

	all_threads.emplace_back(detector_map_calculations_thread, std::ref(shared_info));

	/* ******************************* */
	/* WAIT UNTIL ALL THREADS ARE DONE */
	/* ******************************* */

	size_t no_change_detected		= 0;
	size_t previous_count_analyze	= 0;
	auto previous_timestamp			= timestamp_start;

	// For multi-threaded prediction, print blank lines to make room for per-thread progress display
	if (use_multi_threaded_prediction)
	{
		for (int t = 0; t < num_prediction_threads; ++t)
		{
			*cfg_and_state.output << std::endl;
		}
	}

	while (cfg_and_state.must_immediately_exit == false and not shared_info.done)
	{
		std::this_thread::sleep_for(std::chrono::milliseconds(750));

		if (shared_info.count_load_performed	>= shared_info.total_number_of_validation_images and
			shared_info.count_predict_performed	>= shared_info.total_number_of_validation_images and
			shared_info.count_analyze_performed	>= shared_info.total_number_of_validation_images)
		{
			shared_info.done = true;
		}

		if (previous_count_analyze != shared_info.count_analyze_performed)
		{
			no_change_detected = 0;
		}
		else
		{
			no_change_detected ++;

			if (no_change_detected == 8) // @ 750 milliseconds, this means 6 seconds with nothing changed
			{
				Darknet::display_warning_msg("\nLoading or processing images seems to have stalled.\n");
				*cfg_and_state.output << "(Perhaps re-run with --trace to help determine the cause?)" << std::endl;
			}
			if (no_change_detected > 16) // @ 750 milliseconds, this means 12 seconds with nothing changed
			{
				Darknet::display_warning_msg("\nExiting loop early since loading and processing images seems to have stalled.\n");
				shared_info.done = true;
			}
		}

		if (shared_info.done)
		{
			// show the "full" stats since this will be the last time through the loop
			previous_count_analyze	= 0;
			previous_timestamp		= timestamp_start;
		}

		const auto now					= std::chrono::high_resolution_clock::now();
		const float nanoseconds			= std::chrono::duration_cast<std::chrono::nanoseconds>(now - previous_timestamp).count();
		const size_t images_per_second	= std::round((shared_info.count_analyze_performed	- previous_count_analyze)	/ nanoseconds * 1000000000.0f);
		previous_count_analyze			= shared_info.count_analyze_performed;
		previous_timestamp				= now;
		const int loading_percentage	= std::round(100.0f * shared_info.count_load_performed		/ shared_info.total_number_of_validation_images);
		const int predicting_percentage	= std::round(100.0f * shared_info.count_predict_performed	/ shared_info.total_number_of_validation_images);
		const int analyzing_percentage	= std::round(100.0f * shared_info.count_analyze_performed	/ shared_info.total_number_of_validation_images);

		// Sum predict-queue sizes across all predict threads for the diagnostic display.  This replaces the old
		// "work=A+B" field that showed (input map size + predict thread's local batch size).
		size_t total_predict_pending = 0;
		for (auto & q : shared_info.predict_queues)
		{
			total_predict_pending += q->size();
		}
		const size_t analyze_pending = shared_info.analyze_queue->size();

		if (use_multi_threaded_prediction)
		{
			// Multi-threaded display: show full progress line for each prediction thread
			// Move cursor up N lines, then print each thread's progress on its own line
			*cfg_and_state.output << "\033[" << num_prediction_threads << "A";  // Move cursor up N lines

			const size_t nominal_thread_target = std::max<size_t>(1UL, (shared_info.total_number_of_validation_images + num_prediction_threads - 1) / num_prediction_threads);
			for (int t = 0; t < num_prediction_threads; ++t)
			{
				const size_t thread_count = shared_info.per_thread_predict_count[t];
				const int thread_percentage = std::min(100, static_cast<int>(std::round(100.0f * thread_count / nominal_thread_target)));
				const size_t my_q_size = shared_info.predict_queues[t]->size();

				*cfg_and_state.output
					<< "\033[2K"  // Clear line
					<< "-> " << images_per_second << " images/sec: "
					<< "loading #" << Darknet::in_colour(Darknet::EColour::kBrightWhite, int(shared_info.count_load_performed))
					<< " (" << Darknet::format_percentage(loading_percentage) << ")"
					<< ", predicting #" << Darknet::in_colour(Darknet::EColour::kBrightWhite, int(thread_count))
					<< " (" << Darknet::format_percentage(thread_percentage) << ", q=" << my_q_size << ") [thread " << t << "]"
					<< ", analyzing #" << Darknet::in_colour(Darknet::EColour::kBrightWhite, int(shared_info.count_analyze_performed))
					<< " (" << Darknet::format_percentage(analyzing_percentage) << ", q=" << analyze_pending << ")  "
					<< std::endl;
			}
			*cfg_and_state.output << std::flush;
		}
		else
		{
			// Single-threaded display: original format minus the now-removed pause/starve/time fields.
			std::stringstream ss;
			ss	<< "\r"
				<< "-> " << images_per_second << " images/sec: "
				<< "loading #" << Darknet::in_colour(Darknet::EColour::kBrightWhite, int(shared_info.count_load_performed))
				<< " (" << Darknet::format_percentage(loading_percentage) << ")"
				<< ", predicting #" << Darknet::in_colour(Darknet::EColour::kBrightWhite, int(shared_info.count_predict_performed))
				<< " (" << Darknet::format_percentage(predicting_percentage) << ", work=" << total_predict_pending << ")"
				<< ", analyzing #" << Darknet::in_colour(Darknet::EColour::kBrightWhite, int(shared_info.count_analyze_performed))
				<< " (" << Darknet::format_percentage(analyzing_percentage) << ", work=" << analyze_pending << ")  "; // intentional trailing whitespace in case some fields shrink
			if (cfg_and_state.is_verbose)
			{
				ss << std::endl;
			}

			*cfg_and_state.output << ss.str() << std::flush;
		}
	}
	*cfg_and_state.output << std::endl;

	/* Ordered shutdown:  the previous design joined every thread in arbitrary order, which worked because the
	 * sleep-poll consumers always re-checked must_immediately_exit and could exit on their own.  With blocking
	 * pop() we have to actively close() each queue so blocked consumers wake up.  The order matters:
	 *
	 *	1) Wait for all loaders to finish.  After this point, no more items will be pushed to the predict queues.
	 *	2) Close the predict queues.  Predict threads drain whatever is still queued, then their pop() returns
	 *	   false and they exit.  Join them.
	 *	3) Close the analyze queue.  The calc thread drains, its pop() returns false, and it exits.  Join it.
	 *
	 * If we triggered an early exit via must_immediately_exit, also close everything up front so any blocked
	 * thread wakes up immediately.
	 */
	if (cfg_and_state.must_immediately_exit or shared_info.done)
	{
		for (auto & q : shared_info.predict_queues) q->close();
		shared_info.analyze_queue->close();
	}

	for (auto & t : all_threads)
	{
		t.join();
	}

	/* *********************************** */
	/* THREADS ARE DONE, PRINT THE RESULTS */
	/* *********************************** */

	if ((shared_info.tp_for_thresh + shared_info.fp_for_thresh) > 0)
	{
		shared_info.avg_iou /= (shared_info.tp_for_thresh + shared_info.fp_for_thresh);
	}

	for (int class_id = 0; class_id < shared_info.number_of_classes and cfg_and_state.must_immediately_exit == false; class_id ++)
	{
		const int denom = shared_info.tp_for_thresh_per_class[class_id] + shared_info.fp_for_thresh_per_class[class_id];
		if (denom > 0)
		{
			shared_info.avg_iou_per_class[class_id] /= denom;
		}
	}

	// Sort the array from high probability to low probability.
	//
	// With a test of 7125 entries in the array:
	//
	// - qsort() with function took:	576286 nanoseconds
	// - std::sort() with lambda took:	414231 nanoseconds
	//
	std::sort(/** @todo try this again in 2026? std::execution::par_unseq,*/ shared_info.box_probabilities.begin(), shared_info.box_probabilities.end(),
			[](const BoxProbability & lhs, const BoxProbability & rhs)
			{
				return lhs.probability > rhs.probability;
			});

	struct pr_t
	{
		double prob			= 0.0;
		double precision	= 0.0;
		double recall		= 0.0;
		int tp				= 0;
		int tn				= 0;
		int fp				= 0;
		int fn				= 0;
	};

	/* for the precision-recall (PR) curve
	 *
	 * Note this is a pointer-to-a-pointer.  We don't have just 1 of these per class, but these exist for every
	 * prediction...which can be quite big depending on the dataset.
	 */
	pr_t** pr = (pr_t**)xcalloc(shared_info.number_of_classes, sizeof(pr_t*));
	for (int i = 0; i < shared_info.number_of_classes and cfg_and_state.must_immediately_exit == false; ++i)
	{
		pr[i] = (pr_t*)xcalloc(std::max(size_t(1), shared_info.box_probabilities.size()), sizeof(pr_t)); // allocate at least 1 to avoid nullptr deref
	}

	*cfg_and_state.output << "detections_count=" << shared_info.box_probabilities.size() << ", unique_truth_count=" << shared_info.unique_truth_count << std::endl;

	int *truth_flags = (int*)xcalloc(std::max(1, shared_info.unique_truth_count), sizeof(int));

	// Accumulate PR for each rank
	for (int rank = 0; rank < shared_info.box_probabilities.size() and cfg_and_state.must_immediately_exit == false; ++rank)
	{
		if (rank % 100 == 0)
		{
			*cfg_and_state.output << "\rrank=" << rank << " of ranks=" << shared_info.box_probabilities.size() << std::flush;
		}

		if (rank > 0)
		{
			for (int class_id = 0; class_id < shared_info.number_of_classes and cfg_and_state.must_immediately_exit == false; ++class_id)
			{
				pr[class_id][rank].tp = pr[class_id][rank - 1].tp;
				pr[class_id][rank].fp = pr[class_id][rank - 1].fp;
				pr[class_id][rank].tn = pr[class_id][rank - 1].tn;
				pr[class_id][rank].fn = pr[class_id][rank - 1].fn;
			}
		}

		const BoxProbability & d = shared_info.box_probabilities[rank];
		pr[d.class_id][rank].prob = d.probability;

		if (d.matched_ground_truth)
		{
			if (d.unique_truth_index >= 0 and d.unique_truth_index < shared_info.unique_truth_count and truth_flags[d.unique_truth_index] == 0)
			{
				truth_flags[d.unique_truth_index] = 1;
				pr[d.class_id][rank].tp++; // true positive
			}
			else
			{
				pr[d.class_id][rank].fp++; // duplicate hit on same GT
			}
		}
		else
		{
			pr[d.class_id][rank].fp++;    // false-positive
		}

		for (int i = 0; i < shared_info.number_of_classes and cfg_and_state.must_immediately_exit == false; ++i)
		{
			const int tp = pr[i][rank].tp;
			const int fp = pr[i][rank].fp;
//			const int tn = pr[i][rank].tn;
			const int fn = shared_info.ground_truth_counts[i] - tp; // remaining GT are false negatives
			pr[i][rank].fn = fn;
			pr[i][rank].precision	= (tp + fp) > 0 ? (double)tp / (double)(tp + fp) : 0.0;
			pr[i][rank].recall		= (tp + fn) > 0 ? (double)tp / (double)(tp + fn) : 0.0;

			if (rank == (shared_info.box_probabilities.size() - 1) and shared_info.prediction_counts[i] != (tp + fp))
			{
				// check for last rank
				*cfg_and_state.output
					<< "class_id="		<< i
					<< ", detections="	<< shared_info.prediction_counts[i]
					<< ", tp+fp="		<< tp + fp
					<< ", tp="			<< tp
					<< ", fp="			<< fp
					<< std::endl;
			}
		}
	}

	free(truth_flags);

	double mean_average_precision = 0.0;

	// ---- Per-class AP + reporting (no TN/accuracy/specificity) ----
	for (int class_idx = 0; class_idx < shared_info.number_of_classes and cfg_and_state.must_immediately_exit == false; ++class_idx)
	{
		double avg_precision = 0.0;

		// MS COCO - uses 101-Recall-points on PR-chart.
		// PascalVOC2007 - uses 11-Recall-points on PR-chart.
		// PascalVOC2010-2012 - uses Area-Under-Curve on PR-chart.
		// ImageNet - uses Area-Under-Curve on PR-chart.

		// correct mAP calculation: ImageNet, PascalVOC 2010-2012
		const int gt_i = shared_info.ground_truth_counts[class_idx];

		if (shared_info.box_probabilities.empty())
		{
			// No detections at all -> AP remains 0 (unless you prefer to skip classes with gt_i==0)
		}
		else if (map_points == 0) // this is the default functionality, map_points == 0
		{
			// VOC2010 / AUC of the precision envelope
			double last_recall = pr[class_idx][shared_info.box_probabilities.size() - 1].recall;
			double last_precision = pr[class_idx][shared_info.box_probabilities.size() - 1].precision;
			for (int rank = shared_info.box_probabilities.size() - 2; rank >= 0 and cfg_and_state.must_immediately_exit == false; --rank)
			{
				double delta_recall = last_recall - pr[class_idx][rank].recall;
				last_recall = pr[class_idx][rank].recall;

				if (pr[class_idx][rank].precision > last_precision)
				{
					last_precision = pr[class_idx][rank].precision;
				}

				avg_precision += delta_recall * last_precision;
			}
			//add remaining area of PR curve when recall isn't 0 at rank-1
			double delta_recall = last_recall - 0.0;
			avg_precision += delta_recall * last_precision;
		}
		else
		{
			// Sampled AP (VOC2007 11-pt, or COCO-style 101-pt sampling at a SINGLE IoU)
			if (map_points < 2)
			{
				darknet_fatal_error(DARKNET_LOC, "map_points must be >= 2 (e.g., 11 or 101).");
			}

			for (int point = 0; point < map_points and cfg_and_state.must_immediately_exit == false; ++point)
			{
				double cur_recall = (map_points == 1) ? 0.0 : (point * 1.0 / (map_points - 1));
				double cur_precision = 0.0;
				for (int rank = 0; rank < shared_info.box_probabilities.size() and cfg_and_state.must_immediately_exit == false; ++rank)
				{
					if (pr[class_idx][rank].recall		>= cur_recall and
						pr[class_idx][rank].precision	> cur_precision)
					{
						cur_precision = pr[class_idx][rank].precision;
					}
				}
				avg_precision += cur_precision;
			}
			avg_precision = avg_precision / map_points;
		}

		// ---- Per-class stats at the SAME conf_thresh as global F1 ----
		const int tp = shared_info.tp_for_thresh_per_class[class_idx];
		const int fp = shared_info.fp_for_thresh_per_class[class_idx];
		const int fn = std::max(0, gt_i - tp);

		const float diag_avg_iou_at_thresh = (shared_info.tp_for_thresh_per_class[class_idx] + shared_info.fp_for_thresh_per_class[class_idx]) > 0 ? shared_info.avg_iou_per_class[class_idx] : 0.0f;

		float precision	= 0.0f;
		float recall	= 0.0f;
		float f1		= 0.0f;

		if (tp + fp > 0)
		{
			precision = static_cast<float>(tp) / static_cast<float>(tp + fp);
		}
		if (gt_i > 0)
		{
			recall = static_cast<float>(tp) / static_cast<float>(gt_i);
		}
		if (precision + recall > 0.0f)
		{
			f1 = 2.0f * precision * recall / (precision + recall);
		}

		*cfg_and_state.output
			<< Darknet::format_map_ap_row_values(
				class_idx,											// class_id
				shared_info.net.details->class_names[class_idx],	// name
				static_cast<float>(avg_precision),					// AP (threshold-free)
				tp,													// TP @ conf_thresh
				0,													// TN (not shown)
				fp,													// FP @ conf_thresh
				fn,													// FN @ conf_thresh
				gt_i,												// GT
				f1,													// F1 @ conf_thresh
				diag_avg_iou_at_thresh)								// diag IoU @ conf_thresh
			<< std::endl;

		// send the result of this class to the C++ side of things so we can include it the right chart
		Darknet::update_f1_in_new_charts(class_idx, f1);
		Darknet::update_accuracy_in_new_charts(class_idx, (float)avg_precision);

		mean_average_precision += avg_precision;
	}

	// Diagnostic summary (guard divisions)
	float cur_precision	= 0.0f;
	float cur_recall	= 0.0f;
	float f1_score		= 0.0f;

	const int det_denom = shared_info.tp_for_thresh + shared_info.fp_for_thresh;
	if (det_denom > 0)
	{
		cur_precision = (float)shared_info.tp_for_thresh / det_denom;
	}

	if (shared_info.unique_truth_count > 0)
	{
		cur_recall = (float)shared_info.tp_for_thresh / (float)shared_info.unique_truth_count;
	}

	if ((cur_precision + cur_recall) > 0.f)
	{
		f1_score = 2.f * cur_precision * cur_recall / (cur_precision + cur_recall);
	}

	*cfg_and_state.output
		<< ""						<< std::endl
		<< "-> for conf_thresh="	<< shared_info.thresh_calc_avg_iou
		<< ", precision="			<< cur_precision
		<< ", recall="				<< cur_recall
		<< ", F1 score="			<< f1_score
		<< ""						<< std::endl
		<< "-> for conf_thresh="	<< shared_info.thresh_calc_avg_iou
		<< ", TP="					<< shared_info.tp_for_thresh
		<< ", FP="					<< shared_info.fp_for_thresh
		<< ", FN="					<< shared_info.unique_truth_count - shared_info.tp_for_thresh
		<< ", average IoU="			<< shared_info.avg_iou * 100.0f << "%"
		<< ""						<< std::endl
		<< "-> IoU threshold="		<< shared_info.iou_threshold * 100.0f << "%, ";
	if (map_points)
	{
		*cfg_and_state.output << "used " << map_points << " recall points" << std::endl;
	}
	else
	{
		*cfg_and_state.output << "used area-under-curve for each unique recall" << std::endl;
	}

	mean_average_precision = (shared_info.number_of_classes > 0) ? (mean_average_precision / shared_info.number_of_classes) : 0.0;
	*cfg_and_state.output
		<< "-> mean average precision (mAP@" << std::setprecision(2) << iou_thresh << ")="
		<< Darknet::format_map_accuracy(mean_average_precision)
		<< std::endl;

	Darknet::update_f1_in_new_charts(-1, f1_score);

	// free memory
	for (int i = 0; i < shared_info.number_of_classes; ++i)
	{
		free(pr[i]);
	}
	free(pr);

	const auto timestamp_end = std::chrono::high_resolution_clock::now();

	*cfg_and_state.output << "mAP calculations took a total of " << Darknet::format_duration_string(timestamp_end - timestamp_start, 1, Darknet::EFormatDuration::kTrim) << "." << std::endl;

	if (existing_net)
	{
		restore_network_recurrent_state(*existing_net);
	}
	else
	{
		free_network(shared_info.net);
	}

	return mean_average_precision;
}
