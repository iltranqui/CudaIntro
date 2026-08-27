#include "yolo_layer_cuda.hpp"

#ifdef DARKNET_GPU_CUDA

#include <cfloat>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <new>

namespace
{
	constexpr int kYoloThreads = 256;
	constexpr float kPi = 3.14159265358979323846f;

	struct DeviceBox
	{
		float x;
		float y;
		float w;
		float h;
	};

	struct DeviceBoxAbs
	{
		float left;
		float right;
		float top;
		float bottom;
	};

	struct DeviceDx
	{
		float x;
		float y;
		float w;
		float h;
	};

	struct YoloImageMetrics
	{
		float total_iou;
		float total_iou_loss;
		float total_giou_loss;
		float delta_squared;
		float classification_delta_squared;
		int count;
		int class_count;
		int total_bbox;
		int rewritten_bbox;
	};

	struct YoloMetrics
	{
		float total_iou;
		float total_iou_loss;
		float total_giou_loss;
		float total_loss;
		float classification_loss;
		int count;
		int class_count;
		int total_bbox;
		int rewritten_bbox;
		int invalid_kind;
		int invalid_batch;
		int invalid_truth;
		int invalid_class;
		float invalid_x;
		float invalid_y;
		float invalid_w;
		float invalid_h;
	};

	struct YoloInvalid
	{
		int kind;
		int batch;
		int truth;
		int class_id;
		float x;
		float y;
		float w;
		float h;
	};

	struct YoloTrainingGpuContext
	{
		int * mask_gpu = nullptr;
		float * biases_gpu = nullptr;
		int * truth_counts_gpu = nullptr;
		YoloInvalid * invalid_gpu = nullptr;
		YoloImageMetrics * image_metrics_gpu = nullptr;
		YoloMetrics * metrics_gpu = nullptr;
		YoloMetrics * metrics_host = nullptr;
		int batch = 0;
		int anchors = 0;
		int total_anchors = 0;
		int max_boxes = 0;
		bool pending = false;
	};

	struct YoloKernelArgs
	{
		int batch;
		int width;
		int height;
		int anchors;
		int total_anchors;
		int classes;
		int max_boxes;
		int truth_size;
		int truths;
		int outputs;
		int net_width;
		int net_height;
		float ignore_thresh;
		float iou_normalizer;
		float obj_normalizer;
		float max_delta;
	};

	__device__ __forceinline__ float logistic_activate(const float value)
	{
		return 1.0f / (1.0f + expf(-value));
	}

	__device__ __forceinline__ int entry_index(
		const YoloKernelArgs & args,
		const int batch,
		const int anchor,
		const int location,
		const int entry)
	{
		const int spatial = args.width * args.height;
		return batch * args.outputs + anchor * spatial * (args.classes + 5) + entry * spatial + location;
	}

	__device__ __forceinline__ DeviceBox load_truth(const float * truth, const int base)
	{
		return { truth[base], truth[base + 1], truth[base + 2], truth[base + 3] };
	}

	__device__ __forceinline__ DeviceBox decode_box(
		const float * output,
		const float * biases,
		const int bias_anchor,
		const int box_index,
		const int column,
		const int row,
		const YoloKernelArgs & args)
	{
		const int spatial = args.width * args.height;
		DeviceBox box;
		box.x = (column + output[box_index]) / args.width;
		box.y = (row + output[box_index + spatial]) / args.height;
		box.w = expf(output[box_index + 2 * spatial]) * biases[2 * bias_anchor] / args.net_width;
		box.h = expf(output[box_index + 3 * spatial]) * biases[2 * bias_anchor + 1] / args.net_height;
		return box;
	}

	__device__ __forceinline__ DeviceBoxAbs to_tblr(const DeviceBox box)
	{
		return {
			box.x - box.w * 0.5f,
			box.x + box.w * 0.5f,
			box.y - box.h * 0.5f,
			box.y + box.h * 0.5f,
		};
	}

	__device__ __forceinline__ DeviceBoxAbs enclosing_box(const DeviceBox a, const DeviceBox b)
	{
		return {
			fminf(a.x - a.w * 0.5f, b.x - b.w * 0.5f),
			fmaxf(a.x + a.w * 0.5f, b.x + b.w * 0.5f),
			fminf(a.y - a.h * 0.5f, b.y - b.h * 0.5f),
			fmaxf(a.y + a.h * 0.5f, b.y + b.h * 0.5f),
		};
	}

	__device__ __forceinline__ float intersection(const DeviceBox a, const DeviceBox b)
	{
		const float width = fminf(a.x + a.w * 0.5f, b.x + b.w * 0.5f)
			- fmaxf(a.x - a.w * 0.5f, b.x - b.w * 0.5f);
		if (width <= 0.0f)
		{
			return 0.0f;
		}
		const float height = fminf(a.y + a.h * 0.5f, b.y + b.h * 0.5f)
			- fmaxf(a.y - a.h * 0.5f, b.y - b.h * 0.5f);
		return height > 0.0f ? width * height : 0.0f;
	}

	__device__ __forceinline__ float box_iou_device(const DeviceBox a, const DeviceBox b)
	{
		const float overlap = intersection(a, b);
		if (overlap == 0.0f)
		{
			return 0.0f;
		}
		return overlap / (a.w * a.h + b.w * b.h - overlap);
	}

	__device__ __forceinline__ float box_giou_device(const DeviceBox a, const DeviceBox b)
	{
		const DeviceBoxAbs bounds = enclosing_box(a, b);
		const float enclosing_area = (bounds.right - bounds.left) * (bounds.bottom - bounds.top);
		const float iou = box_iou_device(a, b);
		if (enclosing_area == 0.0f)
		{
			return iou;
		}
		const float overlap = intersection(a, b);
		const float union_area = a.w * a.h + b.w * b.h - overlap;
		return iou - (enclosing_area - union_area) / enclosing_area;
	}

	// Device translation of dx_box_iou(..., CIOU).  Keeping the operation order
	// aligned with box.cpp is important for the verification mode's tight tolerance.
	__device__ __forceinline__ DeviceDx ciou_derivative(DeviceBox pred, const DeviceBox truth)
	{
		const DeviceBoxAbs pred_tblr = to_tblr(pred);
		const float pred_t = fminf(pred_tblr.top, pred_tblr.bottom);
		const float pred_b = fmaxf(pred_tblr.top, pred_tblr.bottom);
		const float pred_l = fminf(pred_tblr.left, pred_tblr.right);
		const float pred_r = fmaxf(pred_tblr.left, pred_tblr.right);
		const DeviceBoxAbs truth_tblr = to_tblr(truth);

		const float pred_area = (pred_b - pred_t) * (pred_r - pred_l);
		const float truth_area = (truth_tblr.bottom - truth_tblr.top) * (truth_tblr.right - truth_tblr.left);
		const float intersection_h = fminf(pred_b, truth_tblr.bottom) - fmaxf(pred_t, truth_tblr.top);
		const float intersection_w = fminf(pred_r, truth_tblr.right) - fmaxf(pred_l, truth_tblr.left);
		const float intersection_area = intersection_w * intersection_h;
		const float union_area = pred_area + truth_area - intersection_area;
		const float center_distance = (pred.x - truth.x) * (pred.x - truth.x)
			+ (pred.y - truth.y) * (pred.y - truth.y);

		const float d_area_t = -(pred_r - pred_l);
		const float d_area_b = pred_r - pred_l;
		const float d_area_l = -(pred_b - pred_t);
		const float d_area_r = pred_b - pred_t;
		const float d_intersection_t = pred_t > truth_tblr.top ? -intersection_w : 0.0f;
		const float d_intersection_b = pred_b < truth_tblr.bottom ? intersection_w : 0.0f;
		const float d_intersection_l = pred_l > truth_tblr.left ? -intersection_h : 0.0f;
		const float d_intersection_r = pred_r < truth_tblr.right ? intersection_h : 0.0f;
		const float d_union_t = d_area_t - d_intersection_t;
		const float d_union_b = d_area_b - d_intersection_b;
		const float d_union_l = d_area_l - d_intersection_l;
		const float d_union_r = d_area_r - d_intersection_r;

		float p_t = 0.0f;
		float p_b = 0.0f;
		float p_l = 0.0f;
		float p_r = 0.0f;
		if (union_area > 0.0f)
		{
			const float union_squared = union_area * union_area;
			p_t = (union_area * d_intersection_t - intersection_area * d_union_t) / union_squared;
			p_b = (union_area * d_intersection_b - intersection_area * d_union_b) / union_squared;
			p_l = (union_area * d_intersection_l - intersection_area * d_union_l) / union_squared;
			p_r = (union_area * d_intersection_r - intersection_area * d_union_r) / union_squared;
		}

		// This deliberately mirrors the original sequential assignments.
		p_t = pred_tblr.top < pred_tblr.bottom ? p_t : p_b;
		p_b = pred_tblr.top < pred_tblr.bottom ? p_b : p_t;
		p_l = pred_tblr.left < pred_tblr.right ? p_l : p_r;
		p_r = pred_tblr.left < pred_tblr.right ? p_r : p_l;

		const float enclosing_top = fminf(pred.y - pred.h * 0.5f, truth.y - truth.h * 0.5f);
		const float enclosing_bottom = fmaxf(pred.y + pred.h * 0.5f, truth.y + truth.h * 0.5f);
		const float enclosing_left = fminf(pred.x - pred.w * 0.5f, truth.x - truth.w * 0.5f);
		const float enclosing_right = fmaxf(pred.x + pred.w * 0.5f, truth.x + truth.w * 0.5f);
		const float enclosing_w = enclosing_right - enclosing_left;
		const float enclosing_h = enclosing_bottom - enclosing_top;
		const float diagonal_squared = enclosing_w * enclosing_w + enclosing_h * enclosing_h;

		const float d_top_y = pred_t < truth_tblr.top ? 1.0f : 0.0f;
		const float d_top_h = pred_t < truth_tblr.top ? -0.5f : 0.0f;
		const float d_bottom_y = pred_b > truth_tblr.bottom ? 1.0f : 0.0f;
		const float d_bottom_h = pred_b > truth_tblr.bottom ? 0.5f : 0.0f;
		const float d_left_x = pred_l < truth_tblr.left ? 1.0f : 0.0f;
		const float d_left_w = pred_l < truth_tblr.left ? -0.5f : 0.0f;
		const float d_right_x = pred_r > truth_tblr.right ? 1.0f : 0.0f;
		const float d_right_w = pred_r > truth_tblr.right ? 0.5f : 0.0f;

		const float d_enclosing_w_x = d_right_x - d_left_x;
		const float d_enclosing_w_w = d_right_w - d_left_w;
		const float d_enclosing_h_y = d_bottom_y - d_top_y;
		const float d_enclosing_h_h = d_bottom_h - d_top_h;

		float dx = p_l + p_r;
		float dy = p_t + p_b;
		float dw = p_r - p_l;
		float dh = p_b - p_t;

		const float aspect_truth = truth.w / truth.h;
		const float aspect_pred = pred.w / pred.h;
		const float aspect_angle = atanf(aspect_truth) - atanf(aspect_pred);
		const float aspect_loss = 4.0f / (kPi * kPi) * aspect_angle * aspect_angle;
		const float alpha = aspect_loss / (1.0f - intersection_area / union_area + aspect_loss + 0.000001f);
		const float aspect_dw = 8.0f / (kPi * kPi) * aspect_angle * pred.h;
		const float aspect_dh = -8.0f / (kPi * kPi) * aspect_angle * pred.w;

		if (diagonal_squared > 0.0f)
		{
			const float diagonal_fourth = diagonal_squared * diagonal_squared;
			dx += (2.0f * (truth.x - pred.x) * diagonal_squared
				- 2.0f * enclosing_w * d_enclosing_w_x * center_distance) / diagonal_fourth;
			dy += (2.0f * (truth.y - pred.y) * diagonal_squared
				- 2.0f * enclosing_h * d_enclosing_h_y * center_distance) / diagonal_fourth;
			dw += (2.0f * enclosing_w * d_enclosing_w_w * center_distance) / diagonal_fourth + alpha * aspect_dw;
			dh += (2.0f * enclosing_h * d_enclosing_h_h * center_distance) / diagonal_fourth + alpha * aspect_dh;
		}

		if (intersection_w <= 0.0f || intersection_h <= 0.0f)
		{
			const float diagonal_fourth = diagonal_squared * diagonal_squared;
			dx = (2.0f * (truth.x - pred.x) * diagonal_squared
				- 2.0f * enclosing_w * d_enclosing_w_x * center_distance) / diagonal_fourth;
			dy = (2.0f * (truth.y - pred.y) * diagonal_squared
				- 2.0f * enclosing_h * d_enclosing_h_y * center_distance) / diagonal_fourth;
			dw = (2.0f * enclosing_w * d_enclosing_w_w * center_distance) / diagonal_fourth + alpha * aspect_dw;
			dh = (2.0f * enclosing_h * d_enclosing_h_h * center_distance) / diagonal_fourth + alpha * aspect_dh;
		}

		return { dx, dy, dw, dh };
	}

	__device__ __forceinline__ float sanitize_and_clip(float value, const float max_delta)
	{
		if (!isfinite(value))
		{
			value = 0.0f;
		}
		if (max_delta != FLT_MAX)
		{
			value = fminf(max_delta, fmaxf(-max_delta, value));
		}
		return value;
	}

	__device__ __forceinline__ int find_mask_anchor(
		const int * mask,
		const int anchors,
		const int bias_anchor)
	{
		for (int anchor = 0; anchor < anchors; ++anchor)
		{
			if (mask[anchor] == bias_anchor)
			{
				return anchor;
			}
		}
		return -1;
	}

	__device__ __forceinline__ void set_invalid(
		YoloInvalid * invalid,
		const int kind,
		const int batch,
		const int truth_index,
		const int class_id,
		const DeviceBox truth)
	{
		// One thread scans each image's truths in ascending truth order.  Keeping
		// one record per image makes the later batch-order reduction deterministic.
		if (invalid->kind != 0) return;
		invalid->kind = kind;
		invalid->batch = batch;
		invalid->truth = truth_index;
		invalid->class_id = class_id;
		invalid->x = truth.x;
		invalid->y = truth.y;
		invalid->w = truth.w;
		invalid->h = truth.h;
	}

	__global__ void activate_yolo_output_kernel(
		const float * input,
		float * output,
		const int elements,
		const int spatial,
		const int entries,
		const float scale_x_y,
		const int new_coords)
	{
		for (int index = blockIdx.x * blockDim.x + threadIdx.x;
			 index < elements;
			 index += blockDim.x * gridDim.x)
		{
			const int entry = (index / spatial) % entries;
			float value = input[index];
			if (!new_coords && (entry < 2 || entry >= 4))
			{
				value = logistic_activate(value);
			}
			if (entry < 2)
			{
				value = value * scale_x_y - 0.5f * (scale_x_y - 1.0f);
			}
			output[index] = value;
		}
	}

	__global__ void yolo_background_kernel(
		float * output,
		const float * truth,
		float * delta,
		const float * biases,
		const int * mask,
		int * truth_counts,
		YoloInvalid * invalid,
		const YoloKernelArgs args)
	{
		extern __shared__ float shared_truth[];
		__shared__ int truth_count;
		const int batch = blockIdx.x;
		YoloInvalid * image_invalid = invalid + batch;
		const int truth_values = args.max_boxes * 5;
		for (int index = threadIdx.x; index < truth_values; index += blockDim.x)
		{
			const int truth_index = index / 5;
			const int component = index % 5;
			shared_truth[index] = truth[batch * args.truths + truth_index * args.truth_size + component];
		}
		__syncthreads();

		if (threadIdx.x == 0)
		{
			truth_count = 0;
			for (int t = 0; t < args.max_boxes; ++t)
			{
				const int base = t * 5;
				const DeviceBox box = {
					shared_truth[base], shared_truth[base + 1], shared_truth[base + 2], shared_truth[base + 3] };
				if (box.x == 0.0f)
				{
					break;
				}
				const int class_id = static_cast<int>(shared_truth[base + 4]);
				if (class_id < 0 || class_id >= args.classes)
				{
					set_invalid(image_invalid, 1, batch, t, class_id, box);
					break;
				}
				if (box.x < 0.0f || box.y < 0.0f || box.x > 1.0f || box.y > 1.0f || box.w < 0.0f || box.h < 0.0f)
				{
					set_invalid(image_invalid, 2, batch, t, class_id, box);
					break;
				}
				++truth_count;
			}
			truth_counts[batch] = truth_count;
		}
		__syncthreads();
		if (image_invalid->kind != 0)
		{
			return;
		}

		const int spatial = args.width * args.height;
		const int predictions = args.anchors * spatial;
		for (int prediction = threadIdx.x; prediction < predictions; prediction += blockDim.x)
		{
			const int anchor = prediction / spatial;
			const int location = prediction % spatial;
			const int row = location / args.width;
			const int column = location % args.width;
			const int box_index = entry_index(args, batch, anchor, location, 0);
			const int obj_index = entry_index(args, batch, anchor, location, 4);
			const int class_index = entry_index(args, batch, anchor, location, 5);
			const DeviceBox pred = decode_box(output, biases, mask[anchor], box_index, column, row, args);
			float best_match_iou = 0.0f;

			for (int t = 0; t < truth_count; ++t)
			{
				const int truth_base = t * 5;
				const DeviceBox target = {
					shared_truth[truth_base], shared_truth[truth_base + 1],
					shared_truth[truth_base + 2], shared_truth[truth_base + 3] };
				bool class_match = false;
				for (int cls = 0; cls < args.classes; ++cls)
				{
					class_match = class_match || output[class_index + cls * spatial] > 0.25f;
				}
				if (class_match)
				{
					best_match_iou = fmaxf(best_match_iou, box_iou_device(pred, target));
				}
			}

			float objectness = output[obj_index];
			// The CPU path writes zero back to the activated output when it sees a
			// non-finite objectness while scanning at least one truth.  Each CUDA
			// prediction has a unique owner, so the same mutation is race-free here.
			if (truth_count > 0 && !isfinite(objectness))
			{
				objectness = 0.0f;
				output[obj_index] = 0.0f;
			}
			delta[obj_index] = best_match_iou > args.ignore_thresh
				? 0.0f
				: -args.obj_normalizer * objectness;
		}
	}

	__device__ __forceinline__ void assign_ciou_box(
		const DeviceBox truth,
		const float * output,
		float * delta,
		const float * biases,
		const int bias_anchor,
		const int box_index,
		const int column,
		const int row,
		const YoloKernelArgs & args,
		YoloImageMetrics & metrics)
	{
		const int spatial = args.width * args.height;
		if (delta[box_index] != 0.0f || delta[box_index + spatial] != 0.0f
			|| delta[box_index + 2 * spatial] != 0.0f || delta[box_index + 3 * spatial] != 0.0f)
		{
			++metrics.rewritten_bbox;
		}

		DeviceBox pred = decode_box(output, biases, bias_anchor, box_index, column, row, args);
		metrics.total_iou += box_iou_device(pred, truth);
		metrics.total_iou_loss += 1.0f - box_iou_device(pred, truth);
		metrics.total_giou_loss += 1.0f - box_giou_device(pred, truth);
		if (pred.w == 0.0f) pred.w = 1.0f;
		if (pred.h == 0.0f) pred.h = 1.0f;
		DeviceDx derivative = ciou_derivative(pred, truth);
		derivative.w *= expf(output[box_index + 2 * spatial]);
		derivative.h *= expf(output[box_index + 3 * spatial]);
		delta[box_index] += sanitize_and_clip(derivative.x * args.iou_normalizer, args.max_delta);
		delta[box_index + spatial] += sanitize_and_clip(derivative.y * args.iou_normalizer, args.max_delta);
		delta[box_index + 2 * spatial] += sanitize_and_clip(derivative.w * args.iou_normalizer, args.max_delta);
		delta[box_index + 3 * spatial] += sanitize_and_clip(derivative.h * args.iou_normalizer, args.max_delta);
		++metrics.total_bbox;
	}

	__device__ __forceinline__ void assign_class(
		const float * output,
		float * delta,
		const int class_index,
		const int class_id,
		const YoloKernelArgs & args)
	{
		const int spatial = args.width * args.height;
		const int target_index = class_index + class_id * spatial;
		if (delta[target_index] != 0.0f)
		{
			const float value = 1.0f - output[target_index];
			if (isfinite(value)) delta[target_index] = value;
			return;
		}
		for (int cls = 0; cls < args.classes; ++cls)
		{
			const int index = class_index + cls * spatial;
			const float value = (cls == class_id ? 1.0f : 0.0f) - output[index];
			if (isfinite(value)) delta[index] = value;
		}
	}

	__global__ void yolo_positive_kernel(
		const float * output,
		const float * truth,
		float * delta,
		const float * biases,
		const int * mask,
		const int * truth_counts,
		const YoloInvalid * invalid,
		YoloImageMetrics * image_metrics,
		const YoloKernelArgs args)
	{
		const int batch = blockIdx.x;
		if (threadIdx.x != 0 || invalid[batch].kind != 0)
		{
			return;
		}

		YoloImageMetrics metrics = {};
		for (int t = 0; t < truth_counts[batch]; ++t)
		{
			const int truth_base = batch * args.truths + t * args.truth_size;
			const DeviceBox target = load_truth(truth, truth_base);
			const int class_id = static_cast<int>(truth[truth_base + 4]);
			DeviceBox shifted_target = target;
			shifted_target.x = 0.0f;
			shifted_target.y = 0.0f;
			float best_iou = 0.0f;
			int best_anchor = 0;
			for (int anchor = 0; anchor < args.total_anchors; ++anchor)
			{
				const DeviceBox candidate = {
					0.0f, 0.0f,
					biases[2 * anchor] / args.net_width,
					biases[2 * anchor + 1] / args.net_height };
				const float iou = box_iou_device(candidate, shifted_target);
				if (iou > best_iou)
				{
					best_iou = iou;
					best_anchor = anchor;
				}
			}

			const int masked_anchor = find_mask_anchor(mask, args.anchors, best_anchor);
			if (masked_anchor < 0)
			{
				continue;
			}
			int column = static_cast<int>(target.x * args.width);
			int row = static_cast<int>(target.y * args.height);
			column = column < 0 ? 0 : (column >= args.width ? args.width - 1 : column);
			row = row < 0 ? 0 : (row >= args.height ? args.height - 1 : row);
			const int location = row * args.width + column;
			const int box_index = entry_index(args, batch, masked_anchor, location, 0);
			assign_ciou_box(target, output, delta, biases, best_anchor, box_index, column, row, args, metrics);

			const int obj_index = entry_index(args, batch, masked_anchor, location, 4);
			delta[obj_index] = args.obj_normalizer * (1.0f - output[obj_index]);
			const int class_index = entry_index(args, batch, masked_anchor, location, 5);
			assign_class(output, delta, class_index, class_id, args);
			++metrics.count;
			++metrics.class_count;
		}
		image_metrics[batch] = metrics;
	}

	__device__ __forceinline__ void block_sum_pair(float & first, float & second)
	{
		__shared__ float first_warp_sums[32];
		__shared__ float second_warp_sums[32];
		for (int offset = 16; offset > 0; offset >>= 1)
		{
			first += __shfl_down_sync(0xffffffffu, first, offset);
			second += __shfl_down_sync(0xffffffffu, second, offset);
		}
		const int lane = threadIdx.x & 31;
		const int warp = threadIdx.x >> 5;
		if (lane == 0)
		{
			first_warp_sums[warp] = first;
			second_warp_sums[warp] = second;
		}
		__syncthreads();
		first = threadIdx.x < (blockDim.x + 31) / 32 ? first_warp_sums[lane] : 0.0f;
		second = threadIdx.x < (blockDim.x + 31) / 32 ? second_warp_sums[lane] : 0.0f;
		if (warp == 0)
		{
			for (int offset = 16; offset > 0; offset >>= 1)
			{
				first += __shfl_down_sync(0xffffffffu, first, offset);
				second += __shfl_down_sync(0xffffffffu, second, offset);
			}
		}
	}

	__global__ void yolo_loss_reduce_kernel(
		const float * delta,
		YoloImageMetrics * image_metrics,
		const YoloKernelArgs args)
	{
		const int batch = blockIdx.x;
		const int spatial = args.width * args.height;
		float total = 0.0f;
		float classification = 0.0f;
		for (int index = threadIdx.x; index < args.outputs; index += blockDim.x)
		{
			const float value = delta[batch * args.outputs + index];
			const float squared = value * value;
			total += squared;
			const int entry = (index / spatial) % (args.classes + 5);
			if (entry >= 4) classification += squared;
		}
		block_sum_pair(total, classification);
		if (threadIdx.x == 0)
		{
			image_metrics[batch].delta_squared = total;
			image_metrics[batch].classification_delta_squared = classification;
		}
	}

	__global__ void yolo_finalize_metrics_kernel(
		const YoloImageMetrics * image_metrics,
		const YoloInvalid * invalid,
		YoloMetrics * metrics,
		const YoloKernelArgs args)
	{
		if (blockIdx.x != 0 || threadIdx.x != 0)
		{
			return;
		}
		YoloMetrics result = {};
		for (int batch = 0; batch < args.batch; ++batch)
		{
			const YoloImageMetrics image = image_metrics[batch];
			result.total_iou += image.total_iou;
			result.total_iou_loss += image.total_iou_loss;
			result.total_giou_loss += image.total_giou_loss;
			result.total_loss += image.delta_squared;
			result.classification_loss += image.classification_delta_squared;
			result.count += image.count;
			result.class_count += image.class_count;
			result.total_bbox += image.total_bbox;
			result.rewritten_bbox += image.rewritten_bbox;
		}
		result.classification_loss *= args.obj_normalizer;
		for (int batch = 0; batch < args.batch; ++batch)
		{
			const YoloInvalid candidate = invalid[batch];
			if (candidate.kind == 0) continue;
			result.invalid_kind = candidate.kind;
			result.invalid_batch = candidate.batch;
			result.invalid_truth = candidate.truth;
			result.invalid_class = candidate.class_id;
			result.invalid_x = candidate.x;
			result.invalid_y = candidate.y;
			result.invalid_w = candidate.w;
			result.invalid_h = candidate.h;
			break;
		}
		*metrics = result;
	}

	void release_context(YoloTrainingGpuContext * context)
	{
		if (!context) return;
		if (context->mask_gpu) cudaFree(context->mask_gpu);
		if (context->biases_gpu) cudaFree(context->biases_gpu);
		if (context->truth_counts_gpu) cudaFree(context->truth_counts_gpu);
		if (context->invalid_gpu) cudaFree(context->invalid_gpu);
		if (context->image_metrics_gpu) cudaFree(context->image_metrics_gpu);
		if (context->metrics_gpu) cudaFree(context->metrics_gpu);
		if (context->metrics_host) cudaFreeHost(context->metrics_host);
		delete context;
	}

	template <typename T>
	bool allocate_device(T ** pointer, const size_t count, const char ** reason)
	{
		const cudaError_t status = cudaMalloc(reinterpret_cast<void **>(pointer), count * sizeof(T));
		if (status == cudaSuccess) return true;
		if (reason) *reason = cudaGetErrorString(status);
		(void)cudaGetLastError();
		return false;
	}

	YoloTrainingGpuContext * make_context(Darknet::Layer & layer, const char ** reason)
	{
		auto * context = new (std::nothrow) YoloTrainingGpuContext;
		if (!context)
		{
			if (reason) *reason = "cannot allocate YOLO CUDA host context";
			return nullptr;
		}
		context->batch = layer.batch;
		context->anchors = layer.n;
		context->total_anchors = layer.total;
		context->max_boxes = layer.max_boxes;
		if (!allocate_device(&context->mask_gpu, static_cast<size_t>(layer.n), reason)
			|| !allocate_device(&context->biases_gpu, static_cast<size_t>(layer.total) * 2, reason)
			|| !allocate_device(&context->truth_counts_gpu, static_cast<size_t>(layer.batch), reason)
			|| !allocate_device(&context->invalid_gpu, static_cast<size_t>(layer.batch), reason)
			|| !allocate_device(&context->image_metrics_gpu, static_cast<size_t>(layer.batch), reason)
			|| !allocate_device(&context->metrics_gpu, static_cast<size_t>(1), reason))
		{
			release_context(context);
			return nullptr;
		}
		const cudaError_t host_status = cudaHostAlloc(
			reinterpret_cast<void **>(&context->metrics_host), sizeof(YoloMetrics), cudaHostAllocPortable);
		if (host_status != cudaSuccess)
		{
			if (reason) *reason = cudaGetErrorString(host_status);
			(void)cudaGetLastError();
			release_context(context);
			return nullptr;
		}
		cudaStream_t stream = get_cuda_stream();
		cudaError_t status = cudaMemcpyAsync(
			context->mask_gpu, layer.mask, static_cast<size_t>(layer.n) * sizeof(int), cudaMemcpyHostToDevice, stream);
		if (status == cudaSuccess)
		{
			status = cudaMemcpyAsync(context->biases_gpu, layer.biases,
				static_cast<size_t>(layer.total) * 2 * sizeof(float), cudaMemcpyHostToDevice, stream);
		}
		if (status != cudaSuccess)
		{
			if (reason) *reason = cudaGetErrorString(status);
			(void)cudaGetLastError();
			release_context(context);
			return nullptr;
		}
		return context;
	}

	YoloKernelArgs make_kernel_args(const Darknet::Layer & layer, const Darknet::NetworkState & state)
	{
		return {
			layer.batch, layer.w, layer.h, layer.n, layer.total, layer.classes,
			layer.max_boxes, layer.truth_size, layer.truths, layer.outputs,
			state.net.w, state.net.h, layer.ignore_thresh, layer.iou_normalizer,
			layer.obj_normalizer, layer.max_delta,
		};
	}
}

namespace Darknet
{
	void yolo_activate_output_gpu(
		const float * input_gpu,
		float * output_gpu,
		const int batch,
		const int anchors,
		const int width,
		const int height,
		const int classes,
		const float scale_x_y,
		const bool new_coords)
	{
		const int spatial = width * height;
		const int entries = classes + 5;
		const int elements = batch * anchors * spatial * entries;
		const int blocks = get_number_of_blocks(elements, kYoloThreads);
		activate_yolo_output_kernel<<<blocks, kYoloThreads, 0, get_cuda_stream()>>>(
			input_gpu, output_gpu, elements, spatial, entries, scale_x_y, new_coords ? 1 : 0);
		CHECK_CUDA(cudaPeekAtLastError());
	}

	YoloCudaLaunchStatus forward_yolo_training_cuda(
		Darknet::Layer & layer,
		Darknet::NetworkState state,
		const char ** reason)
	{
		if (reason) *reason = nullptr;
		if (!state.truth || !layer.output_gpu || !layer.delta_gpu || !layer.mask || !layer.biases)
		{
			if (reason) *reason = "missing YOLO CUDA input, truth, output, or delta buffer";
			return YoloCudaLaunchStatus::unsupported;
		}
		if (layer.batch <= 0 || layer.w <= 0 || layer.h <= 0 || layer.n <= 0
			|| layer.total <= 0 || layer.classes <= 0 || layer.truth_size < 5
			|| state.net.w <= 0 || state.net.h <= 0 || layer.inputs != layer.outputs
			|| layer.iou_loss != CIOU || layer.new_coords || layer.focal_loss
			|| layer.label_smooth_eps != 0.0f || layer.objectness_smooth
			|| layer.classes_multipliers || layer.map || layer.iou_thresh < 1.0f
			|| layer.truth_thresh < 1.0f || state.net.adversarial
			|| state.net.badlabels_rejection_percentage != 0.0f
			|| state.net.num_sigmas_reject_badlabels != 0.0f
			|| state.net.equidistant_point != 0 || state.net.contrastive || state.net.track
			|| layer.embedding_output || layer.max_boxes <= 0 || layer.max_boxes > 1024)
		{
			if (reason) *reason = "YOLO configuration is outside the CUDA classic-training subset";
			return YoloCudaLaunchStatus::unsupported;
		}
		for (int anchor = 0; anchor < layer.n; ++anchor)
		{
			if (layer.mask[anchor] < 0 || layer.mask[anchor] >= layer.total)
			{
				if (reason) *reason = "YOLO mask references an invalid anchor";
				return YoloCudaLaunchStatus::unsupported;
			}
		}
		for (int anchor = 0; anchor < layer.total * 2; ++anchor)
		{
			if (!std::isfinite(layer.biases[anchor]) || layer.biases[anchor] <= 0.0f)
			{
				if (reason) *reason = "YOLO anchor dimensions must be finite and positive";
				return YoloCudaLaunchStatus::unsupported;
			}
		}

		auto * context = static_cast<YoloTrainingGpuContext *>(layer.yolo_training_gpu_context);
		if (!context)
		{
			context = make_context(layer, reason);
			if (!context)
			{
				return YoloCudaLaunchStatus::recoverable_failure;
			}
			layer.yolo_training_gpu_context = context;
		}
		if (context->batch != layer.batch || context->anchors != layer.n
			|| context->total_anchors != layer.total || context->max_boxes != layer.max_boxes)
		{
			if (reason) *reason = "stale YOLO CUDA context after a shape change";
			return YoloCudaLaunchStatus::recoverable_failure;
		}

		const YoloKernelArgs args = make_kernel_args(layer, state);
		cudaStream_t stream = get_cuda_stream();
		CHECK_CUDA(cudaMemsetAsync(layer.delta_gpu, 0,
			static_cast<size_t>(layer.batch) * layer.outputs * sizeof(float), stream));
		CHECK_CUDA(cudaMemsetAsync(context->image_metrics_gpu, 0,
			static_cast<size_t>(layer.batch) * sizeof(YoloImageMetrics), stream));
		CHECK_CUDA(cudaMemsetAsync(context->invalid_gpu, 0,
			static_cast<size_t>(layer.batch) * sizeof(YoloInvalid), stream));

		const size_t shared_bytes = static_cast<size_t>(layer.max_boxes) * 5 * sizeof(float);
		yolo_background_kernel<<<layer.batch, kYoloThreads, shared_bytes, stream>>>(
			layer.output_gpu, state.truth, layer.delta_gpu, context->biases_gpu,
			context->mask_gpu, context->truth_counts_gpu, context->invalid_gpu, args);
		CHECK_CUDA(cudaPeekAtLastError());
		yolo_positive_kernel<<<layer.batch, 1, 0, stream>>>(
			layer.output_gpu, state.truth, layer.delta_gpu, context->biases_gpu,
			context->mask_gpu, context->truth_counts_gpu, context->invalid_gpu,
			context->image_metrics_gpu, args);
		CHECK_CUDA(cudaPeekAtLastError());
		yolo_loss_reduce_kernel<<<layer.batch, kYoloThreads, 0, stream>>>(
			layer.delta_gpu, context->image_metrics_gpu, args);
		CHECK_CUDA(cudaPeekAtLastError());
		yolo_finalize_metrics_kernel<<<1, 1, 0, stream>>>(
			context->image_metrics_gpu, context->invalid_gpu, context->metrics_gpu, args);
		CHECK_CUDA(cudaPeekAtLastError());
		context->pending = true;
		return YoloCudaLaunchStatus::launched;
	}

	void finalize_yolo_training_cuda(Darknet::Network & net)
	{
		bool has_pending = false;
		for (int index = 0; index < net.n; ++index)
		{
			const Darknet::Layer & layer = net.layers[index];
			const auto * context = static_cast<const YoloTrainingGpuContext *>(layer.yolo_training_gpu_context);
			has_pending = has_pending || (context && context->pending);
		}
		if (!has_pending) return;

		cudaStream_t stream = get_cuda_stream();
		for (int index = 0; index < net.n; ++index)
		{
			Darknet::Layer & layer = net.layers[index];
			auto * context = static_cast<YoloTrainingGpuContext *>(layer.yolo_training_gpu_context);
			if (!context || !context->pending) continue;
			CHECK_CUDA(cudaMemcpyAsync(context->metrics_host, context->metrics_gpu,
				sizeof(YoloMetrics), cudaMemcpyDeviceToHost, stream));
		}
		CHECK_CUDA(cudaStreamSynchronize(stream));

		for (int index = 0; index < net.n; ++index)
		{
			Darknet::Layer & layer = net.layers[index];
			auto * context = static_cast<YoloTrainingGpuContext *>(layer.yolo_training_gpu_context);
			if (!context || !context->pending) continue;
			context->pending = false;
			const YoloMetrics & metrics = *context->metrics_host;
			if (metrics.invalid_kind == 1)
			{
				darknet_fatal_error(DARKNET_LOC,
					"invalid class ID #%d in CUDA YOLO truth b=%d t=%d",
					metrics.invalid_class, metrics.invalid_batch, metrics.invalid_truth);
			}
			if (metrics.invalid_kind == 2)
			{
				darknet_fatal_error(DARKNET_LOC,
					"invalid coordinates, width, or height (x=%f, y=%f, w=%f, h=%f)",
					metrics.invalid_x, metrics.invalid_y, metrics.invalid_w, metrics.invalid_h);
			}
			const int count = metrics.count > 0 ? metrics.count : 1;
			*layer.cost = layer.iou_normalizer * (metrics.total_iou_loss / count)
				+ metrics.classification_loss;
			if (net.total_bbox) *net.total_bbox += metrics.total_bbox;
			if (net.rewritten_bbox) *net.rewritten_bbox += metrics.rewritten_bbox;
			if (Darknet::CfgAndState::get().is_verbose)
			{
				std::printf(
					"v3 ciou loss, Normalizer: (iou: %.2f, obj: %.2f, cls: %.2f) "
					"Region %d Avg (IOU: %.6f), count: %d, class_loss: %.6f, "
					"iou_loss: %.6f, total_loss: %.6f\n",
					layer.iou_normalizer, layer.obj_normalizer, layer.cls_normalizer,
					index, metrics.total_iou / count, metrics.count,
					metrics.classification_loss / layer.batch,
					(metrics.total_loss - metrics.classification_loss) / layer.batch,
					metrics.total_loss / layer.batch);
			}
		}
	}

	void resize_yolo_training_cuda(Darknet::Layer & layer)
	{
		release_yolo_training_cuda(layer);
	}

	void release_yolo_training_cuda(Darknet::Layer & layer)
	{
		auto * context = static_cast<YoloTrainingGpuContext *>(layer.yolo_training_gpu_context);
		layer.yolo_training_gpu_context = nullptr;
		release_context(context);
	}
}

#endif // DARKNET_GPU_CUDA
