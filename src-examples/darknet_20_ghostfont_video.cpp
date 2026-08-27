/* Darknet/YOLO:  https://codeberg.org/CCodeRun/darknet
 * Copyright 2024-2026 Stephane Charette
 */

#define DARKNET_INCLUDE_ORIGINAL_API	// for reset_rnn()
#include "darknet.h"
#include "darknet.hpp"
#include "darknet_image.hpp"	// Darknet::mat_to_image()

#include <fstream>
#include <limits>
#include <numeric>
#include <set>

/** @file
 * Ghost-font ensemble video inference.
 *
 * Ghost-message videos vary:  the hidden-text drift can be inverted (opposite
 * subtraction polarity) and can run at different speeds (shift px/frame), so a
 * single hardcoded decode setting misses messages.  This application:
 *
 *   1. builds a grid of decode hypotheses (axis, shift, polarity),
 *   2. PASS 1 (scan):  runs darknet on the first N frames of each hypothesis
 *      and scores it by the sum of detection confidences,
 *   3. PASS 2 (refine):  re-runs the full video with the top-K hypotheses,
 *   4. stacks the detections across hypotheses (greedy IoU clustering with
 *      support-weighted voting) and returns the most probable boxes.
 *
 * Output:  annotated <video>_output.m4v + <video>_ghost_report.json + stdout
 * score table.  Call it like this:
 *
 *     darknet_20_ghostfont_video ghostfont.names yolov4-tiny-ghostfont-infer.cfg weights.weights video.mp4 \
 *         [--scan-frames 90] [--top 3] [--horizontal] [--shifts 2,3,4,6,8] [--iou 0.5]
 *
 * The network must be 4-channel (channels=4):  RGB + motion plane.  The motion
 * plane math must stay identical to tools/make_ghostfont_dataset.py.
 */


namespace
{
	/// number of initial frames per video used only to warm up the recurrent state
	constexpr size_t WARMUP_FRAMES = 2;

	/// detections below this confidence are ignored when scoring hypotheses
	constexpr float SCORE_THRESHOLD = 0.25f;


	struct Options
	{
		size_t scan_frames			= 90;
		size_t top_k				= 3;
		std::vector<int> axes		= {0};				// 0=vert, 1=horiz, 2=diag \, 3=diag /
		std::vector<int> shifts		= {1, 2, 3, 4, 5, 6, 8};
		float cluster_iou			= 0.5f;
	};


	/// one decode setting to try
	struct DecodeHypothesis
	{
		int axis;		///< 0 = vertical, 1 = horizontal, 2 = diagonal "\", 3 = diagonal "/"
		int shift;		///< drift speed hypothesis in px/frame
		int polarity;	///< +1:  D+ - D-,  -1:  D- - D+

		std::string name() const
		{
			static const char * const axis_names[] = {"vert", "horiz", "diag\\", "diag/"};
			return std::string(axis_names[axis]) + "/shift=" + std::to_string(shift) + "/pol=" + (polarity > 0 ? "+" : "-");
		}
	};


	/// a voted (stacked) detection produced by clustering boxes across hypotheses
	struct VotedBox
	{
		size_t frame;
		cv::Rect2f rect;
		float confidence;
		size_t support;		///< how many distinct hypotheses contributed
		int cls = -1;		///< best class (3-channel letter-detection mode)
	};


	/// shift an image along the given axis, edge rows/columns replicated
	cv::Mat shift_image(const cv::Mat & src, const int axis, const int amount)
	{
		const float dx = (axis == 1 or axis == 2) ? amount : (axis == 3 ? -amount : 0);
		const float dy = (axis != 1) ? amount : 0;
		cv::Mat m = (cv::Mat_<float>(2, 3) << 1, 0, dx, 0, 1, dy);
		cv::Mat dst;
		cv::warpAffine(src, dst, m, src.size(), cv::INTER_LINEAR, cv::BORDER_REPLICATE);
		return dst;
	}


	/** Stateful motion-direction-mismatch plane (mode "dirmatch" in
	 * tools/make_ghostfont_dataset.py):
	 *
	 *     D+  = |shift(gray_i, +SHIFT) - gray_{i-1}|
	 *     D-  = |shift(gray_i, -SHIFT) - gray_{i-1}|
	 *     acc = 0.9*acc + 0.1*polarity*(D+ - D-)
	 *     out = NORM_MINMAX(blur(acc, 5x5))     (128 = neutral before warmup)
	 */
	class MotionChannel
	{
		public:

			explicit MotionChannel(const DecodeHypothesis & h) : hypothesis(h)
			{
			}

			void reset()
			{
				prev_gray	= cv::Mat();
				acc			= cv::Mat();
			}

			cv::Mat update(const cv::Mat & frame_bgr)
			{
				cv::Mat gray;
				cv::cvtColor(frame_bgr, gray, cv::COLOR_BGR2GRAY);
				gray.convertTo(gray, CV_32F);

				cv::Mat out(gray.size(), CV_8UC1, cv::Scalar(128));
				if (not prev_gray.empty())
				{
					cv::Mat d_pos;
					cv::Mat d_neg;
					cv::absdiff(shift_image(gray, hypothesis.axis, +hypothesis.shift), prev_gray, d_pos);
					cv::absdiff(shift_image(gray, hypothesis.axis, -hypothesis.shift), prev_gray, d_neg);

					const cv::Mat raw = static_cast<float>(hypothesis.polarity) * (d_pos - d_neg);
					if (acc.empty())
					{
						acc = raw.clone();
					}
					else
					{
						acc = 0.9 * acc + 0.1 * raw;
					}

					cv::Mat blurred;
					cv::blur(acc, blurred, cv::Size(5, 5));
					cv::normalize(blurred, blurred, 0.0, 255.0, cv::NORM_MINMAX);
					blurred.convertTo(out, CV_8UC1);
				}
				prev_gray = gray;

				return out;
			}

		private:

			DecodeHypothesis hypothesis;
			cv::Mat prev_gray;
			cv::Mat acc;
	};


	float best_confidence(const Darknet::Prediction & pred)
	{
		float conf = 0.0f;
		for (const auto & [k, v] : pred.prob)
		{
			conf = std::max(conf, v);
		}
		return conf;
	}


	/** Run one hypothesis over (part of) a video.  Recurrent [crnn] state and
	 * the motion accumulator are reset first.  Returns the per-frame
	 * predictions (empty vectors for warmup frames) and the hypothesis score.
	 */
	float run_hypothesis(
		Darknet::NetworkPtr net,
		const std::string & video_filename,
		const DecodeHypothesis & hypothesis,
		const cv::Size network_size,
		const size_t max_frames,
		std::vector<Darknet::Predictions> & per_frame)
	{
		cv::VideoCapture cap(video_filename);
		if (not cap.isOpened())
		{
			return -1.0f;
		}

		const cv::Size video_size(cap.get(cv::CAP_PROP_FRAME_WIDTH), cap.get(cv::CAP_PROP_FRAME_HEIGHT));

		reset_rnn(net);
		MotionChannel motion(hypothesis);
		motion.reset();

		per_frame.clear();
		float score = 0.0f;
		size_t frame_counter = 0;

		while (frame_counter < max_frames)
		{
			cv::Mat frame;
			cap >> frame;
			if (frame.empty())
			{
				break;
			}

			const cv::Mat motion_plane = motion.update(frame);

			// build the 4-channel network input:  R, G, B, motion
			cv::Mat rgb;
			cv::cvtColor(frame, rgb, cv::COLOR_BGR2RGB);
			cv::resize(rgb, rgb, network_size, 0, 0, cv::INTER_LINEAR);

			cv::Mat motion_sized;
			cv::resize(motion_plane, motion_sized, network_size, 0, 0, cv::INTER_LINEAR);

			std::vector<cv::Mat> planes;
			cv::split(rgb, planes);
			planes.push_back(motion_sized);

			cv::Mat rgbm;
			cv::merge(planes, rgbm);

			// NOTE:  must use the Darknet::Image overload of predict() -- the
			// cv::Mat overload converts BGRA->RGB and would strip the motion plane
			Darknet::Image img = Darknet::mat_to_image(rgbm);
			const auto results = Darknet::predict(net, img, video_size);	// predict() frees img

			frame_counter ++;
			if (frame_counter <= WARMUP_FRAMES)
			{
				per_frame.emplace_back();	// warmup:  state only, no detections
				continue;
			}

			per_frame.push_back(results);
			for (const auto & pred : results)
			{
				const float conf = best_confidence(pred);
				if (conf >= SCORE_THRESHOLD)
				{
					score += conf;
				}
			}
		}

		return score;
	}


	/** 3-channel mode:  decode the whole video with one hypothesis into a
	 * single "reveal" image (the accumulated motion-direction-mismatch plane,
	 * i.e. what tools/decode does), for letter-detection networks trained on
	 * decoded images rather than 4-channel frames.
	 */
	cv::Mat decode_reveal_image(const std::string & video_filename, const DecodeHypothesis & hypothesis, const size_t max_frames)
	{
		cv::VideoCapture cap(video_filename);
		if (not cap.isOpened())
		{
			return cv::Mat();
		}

		MotionChannel motion(hypothesis);
		motion.reset();

		cv::Mat plane;
		size_t frame_counter = 0;
		while (frame_counter < max_frames)
		{
			cv::Mat frame;
			cap >> frame;
			if (frame.empty())
			{
				break;
			}
			plane = motion.update(frame);
			frame_counter ++;
		}

		if (plane.empty())
		{
			return cv::Mat();
		}

		cv::Mat bgr;
		cv::cvtColor(plane, bgr, cv::COLOR_GRAY2BGR);
		return bgr;
	}


	/** Letter-detection with tiling:  predict on the full image plus 4
	 * overlapping tiles (letters are small relative to a 1280x720 reveal and a
	 * 224x128 network -- tiles give the net a closer look).  All boxes are
	 * returned in full-image coordinates.
	 */
	Darknet::Predictions predict_tiled(Darknet::NetworkPtr net, const cv::Mat & image)
	{
		Darknet::Predictions merged = Darknet::predict(net, image);

		const int tw = image.cols * 3 / 5;	// 60% tiles = 20% overlap in the middle
		const int th = image.rows * 3 / 5;
		for (const int ox : {0, image.cols - tw})
		{
			for (const int oy : {0, image.rows - th})
			{
				cv::Mat tile = image(cv::Rect(ox, oy, tw, th)).clone();
				auto preds = Darknet::predict(net, tile);
				for (auto & pred : preds)
				{
					pred.rect.x += ox;
					pred.rect.y += oy;
					merged.push_back(pred);
				}
			}
		}

		return merged;
	}


	/// group voted letter boxes into text lines (rows by y overlap, sorted by x)
	std::string assemble_message(std::vector<VotedBox> voted, const std::vector<std::string> & class_names, const std::vector<int> & classes)
	{
		if (voted.empty())
		{
			return "";
		}

		std::vector<size_t> order(voted.size());
		std::iota(order.begin(), order.end(), 0);
		std::sort(order.begin(), order.end(), [&](size_t a, size_t b) { return voted[a].rect.y < voted[b].rect.y; });

		std::string message;
		std::vector<size_t> line;
		float line_y = voted[order[0]].rect.y;
		float line_h = voted[order[0]].rect.height;

		auto flush_line = [&]()
		{
			std::sort(line.begin(), line.end(), [&](size_t a, size_t b) { return voted[a].rect.x < voted[b].rect.x; });
			if (not message.empty())
			{
				message += " / ";
			}
			float prev_right = -1.0f;
			float mean_w = 0.0f;
			for (const size_t i : line) { mean_w += voted[i].rect.width; }
			mean_w /= line.size();
			for (const size_t i : line)
			{
				if (prev_right >= 0.0f and voted[i].rect.x - prev_right > 0.8f * mean_w)
				{
					message += ' ';	// gap wider than ~a letter = word break
				}
				const int cls = classes[i];
				message += (cls >= 0 and cls < static_cast<int>(class_names.size())) ? class_names[cls] : "?";
				prev_right = voted[i].rect.x + voted[i].rect.width;
			}
			line.clear();
		};

		for (const size_t i : order)
		{
			if (voted[i].rect.y > line_y + 0.6f * line_h and not line.empty())
			{
				flush_line();
				line_y = voted[i].rect.y;
				line_h = voted[i].rect.height;
			}
			line.push_back(i);
		}
		flush_line();

		return message;
	}


	float iou(const cv::Rect2f & a, const cv::Rect2f & b)
	{
		const float inter = (a & b).area();
		const float uni = a.area() + b.area() - inter;
		return uni > 0.0f ? inter / uni : 0.0f;
	}


	void write_json_report(
		const std::string & filename,
		const std::vector<DecodeHypothesis> & grid,
		const std::vector<float> & scores,
		const std::vector<size_t> & top,
		const std::vector<VotedBox> & voted,
		const std::string & message = "");


	/** 3-channel mode driver:  hypothesis sweep on the accumulated reveal
	 * image, letter detection, cross-hypothesis voting, message assembly.
	 * Returns 0 on success, 2 when nothing was detected.
	 */
	int process_reveal_video(
		Darknet::NetworkPtr net,
		const std::string & video_filename,
		const std::vector<DecodeHypothesis> & grid,
		const Options & opt)
	{
		struct Candidate
		{
			cv::Rect2f rect;
			float conf;
			int cls;
			size_t run;
		};

		// -------- PASS 1:  score every hypothesis on its reveal image --------

		std::vector<float> scores(grid.size(), 0.0f);
		std::vector<Darknet::Predictions> all_preds(grid.size());
		std::vector<cv::Mat> reveals(grid.size());

		for (size_t i = 0; i < grid.size(); ++i)
		{
			reveals[i] = decode_reveal_image(video_filename, grid[i], std::numeric_limits<size_t>::max());
			if (reveals[i].empty())
			{
				scores[i] = -1.0f;
				continue;
			}
			all_preds[i] = Darknet::predict(net, reveals[i]);
			for (const auto & pred : all_preds[i])
			{
				const float conf = best_confidence(pred);
				if (conf >= SCORE_THRESHOLD)
				{
					scores[i] += conf;
				}
			}
			std::cout << "-> scan " << grid[i].name() << " ... score " << scores[i] << " (" << all_preds[i].size() << " letters)" << std::endl;
		}

		std::vector<size_t> order(grid.size());
		std::iota(order.begin(), order.end(), 0);
		std::sort(order.begin(), order.end(), [&](size_t a, size_t b) { return scores[a] > scores[b]; });

		std::vector<size_t> top;
		for (const size_t idx : order)
		{
			if (scores[idx] > 0.0f and top.size() < opt.top_k)
			{
				top.push_back(idx);
			}
		}

		const std::string stem = std::filesystem::path(video_filename).stem().string();
		const std::string report_filename = stem + "_ghost_report.json";

		if (top.empty())
		{
			std::cout << "-> no hypothesis produced detections (all scores 0)" << std::endl;
			write_json_report(report_filename, grid, scores, top, {});
			return 2;
		}

		// -------- PASS 2:  enrich the top hypotheses, then stack --------
		//
		// per top hypothesis:  tiled detection on the full-video reveal plus a
		// mid-video reveal (different accumulation window = independent look)

		std::vector<Candidate> candidates;
		for (size_t r = 0; r < top.size(); ++r)
		{
			Darknet::Predictions preds = predict_tiled(net, reveals[top[r]]);

			cv::VideoCapture probe(video_filename);
			const size_t frame_count = probe.get(cv::CAP_PROP_FRAME_COUNT);
			probe.release();
			if (frame_count > 60)
			{
				const cv::Mat mid = decode_reveal_image(video_filename, grid[top[r]], frame_count / 2);
				if (not mid.empty())
				{
					const auto mid_preds = predict_tiled(net, mid);
					preds.insert(preds.end(), mid_preds.begin(), mid_preds.end());
				}
			}

			for (const auto & pred : preds)
			{
				const float conf = best_confidence(pred);
				if (conf >= SCORE_THRESHOLD)
				{
					candidates.push_back({cv::Rect2f(pred.rect), conf, pred.best_class, r});
				}
			}
		}
		std::sort(candidates.begin(), candidates.end(), [](const Candidate & a, const Candidate & b) { return a.conf > b.conf; });

		std::vector<VotedBox> voted;
		std::vector<bool> used(candidates.size(), false);
		for (size_t i = 0; i < candidates.size(); ++i)
		{
			if (used[i])
			{
				continue;
			}
			std::vector<size_t> members = {i};
			used[i] = true;
			for (size_t j = i + 1; j < candidates.size(); ++j)
			{
				if (not used[j] and iou(candidates[i].rect, candidates[j].rect) >= opt.cluster_iou)
				{
					members.push_back(j);
					used[j] = true;
				}
			}

			std::set<size_t> support_runs;
			float sum_conf = 0.0f;
			for (const size_t m : members)
			{
				support_runs.insert(candidates[m].run);
				sum_conf += candidates[m].conf;
			}
			const float mean_conf = sum_conf / members.size();
			const size_t support = support_runs.size();

			// tiles + mid-video reveals give several looks per hypothesis, so a
			// same-run cluster with 2+ members is also considered confirmed
			if (support >= 2 or members.size() >= 2 or mean_conf >= 0.5f or top.size() == 1)
			{
				// class = the highest-confidence member (candidates are conf-sorted)
				voted.push_back({0, candidates[i].rect, mean_conf * support / top.size(), support, candidates[i].cls});
			}
		}

		// -------- output:  message text + annotated reveal + JSON --------

		const auto & class_names = Darknet::get_class_names(net);
		std::vector<int> classes;
		for (const auto & v : voted)
		{
			classes.push_back(v.cls);
		}
		const std::string message = assemble_message(voted, class_names, classes);

		cv::Mat annotated = reveals[top[0]].clone();
		for (const auto & v : voted)
		{
			const cv::Rect r(v.rect);
			cv::rectangle(annotated, r, cv::Scalar(0, 255, 0), 2);
			const std::string label = (v.cls >= 0 and v.cls < static_cast<int>(class_names.size())) ? class_names[v.cls] : "?";
			cv::putText(annotated, label, cv::Point(r.x, std::max(0, r.y - 6)), cv::FONT_HERSHEY_SIMPLEX, 0.9, cv::Scalar(0, 255, 0), 2);
		}
		const std::string annotated_filename = stem + "_decoded_annotated.png";
		cv::imwrite(annotated_filename, annotated);

		// annotate the voted boxes on the ORIGINAL video (letter positions are in
		// video coordinates -- the reveal image has the same dimensions)
		const std::string output_filename = stem + "_output.m4v";
		{
			cv::VideoCapture cap(video_filename);
			const double fps = cap.get(cv::CAP_PROP_FPS);
			const cv::Size video_size(cap.get(cv::CAP_PROP_FRAME_WIDTH), cap.get(cv::CAP_PROP_FRAME_HEIGHT));
			cv::VideoWriter out(output_filename, cv::VideoWriter::fourcc('m', 'p', '4', 'v'), fps, video_size);

			size_t frame_idx = 0;
			while (out.isOpened())
			{
				cv::Mat frame;
				cap >> frame;
				if (frame.empty())
				{
					break;
				}

				if (frame_idx >= WARMUP_FRAMES)	// detections start at frame 3
				{
					for (const auto & v : voted)
					{
						const cv::Rect r(v.rect);
						cv::rectangle(frame, r, cv::Scalar(0, 255, 0), 2);
						const std::string label = (v.cls >= 0 and v.cls < static_cast<int>(class_names.size())) ? class_names[v.cls] : "?";
						cv::putText(frame, label, cv::Point(r.x, std::max(0, r.y - 6)), cv::FONT_HERSHEY_SIMPLEX, 0.9, cv::Scalar(0, 255, 0), 2);
					}
					cv::putText(frame, message, cv::Point(10, video_size.height - 16), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 255), 2);
				}
				out.write(frame);
				frame_idx ++;
			}
		}

		write_json_report(report_filename, grid, scores, top, voted, message);

		std::cout
			<< "-> winner ................... " << grid[top[0]].name()	<< std::endl
			<< "-> voted letters ............ " << voted.size()			<< std::endl
			<< "-> MESSAGE .................. " << message				<< std::endl
			<< "-> annotated reveal ......... " << annotated_filename	<< std::endl
			<< "-> annotated video .......... " << output_filename		<< std::endl
			<< "-> report ................... " << report_filename		<< std::endl;

		return 0;
	}


	/** Stack the detections of several hypotheses for one frame index into
	 * voted boxes:  greedy IoU clustering, support-weighted confidence.
	 */
	std::vector<VotedBox> vote_frame(
		const size_t frame_idx,
		const std::vector<std::vector<Darknet::Predictions>> & runs,
		const float cluster_iou)
	{
		struct Candidate
		{
			cv::Rect2f rect;
			float conf;
			size_t run;
		};

		std::vector<Candidate> candidates;
		for (size_t r = 0; r < runs.size(); ++r)
		{
			if (frame_idx >= runs[r].size())
			{
				continue;
			}
			for (const auto & pred : runs[r][frame_idx])
			{
				const float conf = best_confidence(pred);
				if (conf >= SCORE_THRESHOLD)
				{
					candidates.push_back({cv::Rect2f(pred.rect), conf, r});
				}
			}
		}

		std::sort(candidates.begin(), candidates.end(), [](const Candidate & a, const Candidate & b) { return a.conf > b.conf; });

		std::vector<VotedBox> voted;
		std::vector<bool> used(candidates.size(), false);
		for (size_t i = 0; i < candidates.size(); ++i)
		{
			if (used[i])
			{
				continue;
			}

			std::vector<size_t> members = {i};
			used[i] = true;
			for (size_t j = i + 1; j < candidates.size(); ++j)
			{
				if (not used[j] and iou(candidates[i].rect, candidates[j].rect) >= cluster_iou)
				{
					members.push_back(j);
					used[j] = true;
				}
			}

			// confidence-weighted average rect, mean confidence, distinct-run support
			float sum_conf = 0.0f;
			cv::Rect2f avg(0, 0, 0, 0);
			std::set<size_t> support_runs;
			for (const size_t m : members)
			{
				const auto & c = candidates[m];
				avg.x += c.rect.x * c.conf;
				avg.y += c.rect.y * c.conf;
				avg.width  += c.rect.width  * c.conf;
				avg.height += c.rect.height * c.conf;
				sum_conf += c.conf;
				support_runs.insert(c.run);
			}
			avg.x /= sum_conf;
			avg.y /= sum_conf;
			avg.width  /= sum_conf;
			avg.height /= sum_conf;

			const float mean_conf = sum_conf / members.size();
			const size_t support = support_runs.size();
			const float voted_conf = mean_conf * static_cast<float>(support) / runs.size();

			// keep clusters confirmed by 2+ hypotheses, or single strong ones
			if (support >= 2 or mean_conf >= 0.6f)
			{
				voted.push_back({frame_idx, avg, voted_conf, support});
			}
		}

		return voted;
	}


	/// strip our own --flags from argv before darknet's argument parser sees them
	Options parse_and_strip_options(int & argc, char ** argv)
	{
		Options opt;
		std::vector<char *> kept;
		kept.push_back(argv[0]);

		for (int i = 1; i < argc; ++i)
		{
			const std::string arg = argv[i];
			auto next = [&]() -> std::string { return (i + 1 < argc) ? argv[++i] : ""; };

			if (arg == "--scan-frames")		{ opt.scan_frames	= std::stoul(next());	}
			else if (arg == "--top")		{ opt.top_k			= std::stoul(next());	}
			else if (arg == "--iou")		{ opt.cluster_iou	= std::stof(next());	}
			else if (arg == "--horizontal")	{ opt.axes.push_back(1);					}
			else if (arg == "--diagonal")	{ opt.axes.push_back(2); opt.axes.push_back(3);	}
			else if (arg == "--all-axes")	{ opt.axes = {0, 1, 2, 3};					}
			else if (arg == "--shifts")
			{
				opt.shifts.clear();
				std::stringstream ss(next());
				std::string token;
				while (std::getline(ss, token, ','))
				{
					opt.shifts.push_back(std::stoi(token));
				}
			}
			else
			{
				kept.push_back(argv[i]);
			}
		}

		for (size_t i = 0; i < kept.size(); ++i)
		{
			argv[i] = kept[i];
		}
		argc = kept.size();

		return opt;
	}


	void write_json_report(
		const std::string & filename,
		const std::vector<DecodeHypothesis> & grid,
		const std::vector<float> & scores,
		const std::vector<size_t> & top,
		const std::vector<VotedBox> & voted,
		const std::string & message)
	{
		std::ofstream ofs(filename);
		ofs << "{\n  \"message\": \"" << message << "\",\n  \"hypotheses\": [\n";
		for (size_t i = 0; i < grid.size(); ++i)
		{
			ofs << "    {\"name\": \"" << grid[i].name() << "\", \"axis\": " << grid[i].axis
				<< ", \"shift\": " << grid[i].shift << ", \"polarity\": " << grid[i].polarity
				<< ", \"score\": " << scores[i] << "}" << (i + 1 < grid.size() ? "," : "") << "\n";
		}
		ofs << "  ],\n  \"top\": [";
		for (size_t i = 0; i < top.size(); ++i)
		{
			ofs << "\"" << grid[top[i]].name() << "\"" << (i + 1 < top.size() ? ", " : "");
		}
		ofs << "],\n  \"winner\": " << (top.empty() ? "null" : "\"" + grid[top[0]].name() + "\"") << ",\n";
		ofs << "  \"detections\": [\n";
		for (size_t i = 0; i < voted.size(); ++i)
		{
			const auto & v = voted[i];
			ofs << "    {\"frame\": " << v.frame << ", \"x\": " << v.rect.x << ", \"y\": " << v.rect.y
				<< ", \"w\": " << v.rect.width << ", \"h\": " << v.rect.height
				<< ", \"conf\": " << v.confidence << ", \"support\": " << v.support << "}"
				<< (i + 1 < voted.size() ? "," : "") << "\n";
		}
		ofs << "  ]\n}\n";
	}
}


int main(int argc, char * argv[])
{
	int rc = 0;

	try
	{
		Options opt = parse_and_strip_options(argc, argv);

		Darknet::Parms parms = Darknet::parse_arguments(argc, argv);
		Darknet::NetworkPtr net = Darknet::load_neural_network(parms);

		int network_width = 0;
		int network_height = 0;
		int network_channels = 0;
		Darknet::network_dimensions(net, network_width, network_height, network_channels);

		if (network_channels != 4 and network_channels != 3)
		{
			std::cout << "ERROR: this application needs a 4-channel (end-to-end) or 3-channel (letter-detection-on-decoded-image) network, but the .cfg is " << network_channels << "-channel" << std::endl;
			return 1;
		}
		const bool reveal_mode = (network_channels == 3);
		const cv::Size network_size(network_width, network_height);

		// the hypothesis grid:  axis x shift x polarity
		std::vector<DecodeHypothesis> grid;
		for (const int axis : opt.axes)
		{
			for (const int shift : opt.shifts)
			{
				for (const int polarity : {+1, -1})
				{
					grid.push_back({axis, shift, polarity});
				}
			}
		}

		for (const auto & parm : parms)
		{
			if (parm.type != Darknet::EParmType::kFilename)
			{
				continue;
			}

			std::cout << "processing " << parm.string << " with " << grid.size() << " decode hypotheses:" << std::endl;

			if (reveal_mode)
			{
				// 3-channel letter-detection network:  sweep hypotheses on the
				// accumulated reveal image and assemble the message text
				const int result = process_reveal_video(net, parm.string, grid, opt);
				rc = std::max(rc, result);
				continue;
			}

			// -------- PASS 1:  scan every hypothesis on the first N frames --------

			std::vector<float> scores(grid.size(), 0.0f);
			for (size_t i = 0; i < grid.size(); ++i)
			{
				std::vector<Darknet::Predictions> ignored;
				scores[i] = run_hypothesis(net, parm.string, grid[i], network_size, opt.scan_frames, ignored);
				std::cout << "-> scan " << grid[i].name() << " ... score " << scores[i] << std::endl;
			}

			// pick the top-K hypotheses with a positive score
			std::vector<size_t> order(grid.size());
			std::iota(order.begin(), order.end(), 0);
			std::sort(order.begin(), order.end(), [&](size_t a, size_t b) { return scores[a] > scores[b]; });

			std::vector<size_t> top;
			for (const size_t idx : order)
			{
				if (scores[idx] > 0.0f and top.size() < opt.top_k)
				{
					top.push_back(idx);
				}
			}

			const std::string stem = std::filesystem::path(parm.string).stem().string();
			const std::string report_filename = stem + "_ghost_report.json";

			if (top.empty())
			{
				std::cout << "-> no hypothesis produced detections (all scores 0)" << std::endl;
				write_json_report(report_filename, grid, scores, top, {});
				rc = 2;
				continue;
			}

			std::cout << "-> winner: " << grid[top[0]].name() << ", refining top " << top.size() << " hypotheses on the full video" << std::endl;

			// -------- PASS 2:  full-video runs with the best hypotheses --------

			std::vector<std::vector<Darknet::Predictions>> runs;
			for (const size_t idx : top)
			{
				std::vector<Darknet::Predictions> per_frame;
				run_hypothesis(net, parm.string, grid[idx], network_size, std::numeric_limits<size_t>::max(), per_frame);
				runs.push_back(std::move(per_frame));
			}

			// -------- stack the results:  per-frame voting --------

			size_t total_frames = 0;
			for (const auto & r : runs)
			{
				total_frames = std::max(total_frames, r.size());
			}

			std::vector<VotedBox> voted;
			for (size_t f = 0; f < total_frames; ++f)
			{
				const auto frame_boxes = vote_frame(f, runs, opt.cluster_iou);
				voted.insert(voted.end(), frame_boxes.begin(), frame_boxes.end());
			}

			// -------- output:  annotated video + JSON report --------

			cv::VideoCapture cap(parm.string);
			const double fps = cap.get(cv::CAP_PROP_FPS);
			const cv::Size video_size(cap.get(cv::CAP_PROP_FRAME_WIDTH), cap.get(cv::CAP_PROP_FRAME_HEIGHT));
			const std::string output_filename = stem + "_output.m4v";
			cv::VideoWriter out(output_filename, cv::VideoWriter::fourcc('m', 'p', '4', 'v'), fps, video_size);

			size_t frame_idx = 0;
			auto it = voted.begin();
			while (true)
			{
				cv::Mat frame;
				cap >> frame;
				if (frame.empty())
				{
					break;
				}

				// per_frame[k] holds the detections of video frame k (warmup slots are empty)
				while (it != voted.end() and it->frame == frame_idx)
				{
					const cv::Rect r(it->rect);
					cv::rectangle(frame, r, cv::Scalar(0, 255, 0), 2);
					char label[64];
					snprintf(label, sizeof(label), "ghost %.2f (%zu)", it->confidence, it->support);
					cv::putText(frame, label, cv::Point(r.x, std::max(0, r.y - 6)), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
					++it;
				}

				out.write(frame);
				frame_idx ++;
			}

			write_json_report(report_filename, grid, scores, top, voted);

			std::cout
				<< "-> total voted detections ... " << voted.size()		<< std::endl
				<< "-> output video ............. " << output_filename	<< std::endl
				<< "-> report ................... " << report_filename	<< std::endl;
		}

		Darknet::free_neural_network(net);
	}
	catch (const std::exception & e)
	{
		std::cout << "Exception: " << e.what() << std::endl;
		rc = 1;
	}

	return rc;
}
