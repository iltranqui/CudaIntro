#include "modern_yolo_layer.hpp"
#include "yolonas_layer.hpp"
#include "ppyoloe_layer.hpp"
#include "yolox_layer.hpp"

#include <algorithm>
#include <cmath>
#include <cfloat>
#include <cstring>
#include <iomanip>
#include <limits>
#include <numeric>
#include <vector>

namespace
{
	static auto & cfg_and_state = Darknet::CfgAndState::get();

	constexpr float k_eps = 1e-8f;
	constexpr float k_vfl_alpha = 0.75f;

	struct Candidate
	{
		int batch = 0;
		int anchor = 0;
		int loc = 0;
		int truth_index = -1;
		int class_id = -1;
		float iou = 0.0f;
		float score = 0.0f;
		float cost = 0.0f;
		Darknet::Box pred = {};
		Darknet::Box truth = {};
	};

	struct Assignment
	{
		int truth_index = -1;
		int class_id = -1;
		float quality = 0.0f;
		float iou = 0.0f;
		Darknet::Box truth = {};
	};

	struct ModernYoloStats
	{
		float sum_iou = 0.0f;
		float sum_iou_loss = 0.0f;
		float cls_loss = 0.0f;
		float box_loss = 0.0f;
		float dfl_loss = 0.0f;
		int count = 0;
	};

	struct ModernYoloTrainSchedule
	{
		int64_t iteration = 0;
		bool atss_warmup = false;
		bool yolox_final_l1 = false;
	};

	static inline int pred_slot(const Darknet::Layer & l, const int b, const int n, const int loc)
	{
		return (b * l.n + n) * l.w * l.h + loc;
	}

	static inline int modern_entry_index(const Darknet::Layer & l, const int batch, const int location, const int entry)
	{
		const int n = location / (l.w * l.h);
		const int loc = location % (l.w * l.h);
		return batch * l.outputs + n * l.w * l.h * (l.coords + l.classes + 1) + entry * l.w * l.h + loc;
	}

	static inline float finite_or_zero(float value)
	{
		return std::isfinite(value) ? value : 0.0f;
	}

	static inline int64_t modern_yolo_current_iteration(const Darknet::Network & net)
	{
		if (net.cur_iteration)
		{
			return *net.cur_iteration;
		}
		if (net.seen && net.batch > 0 && net.subdivisions > 0)
		{
			return static_cast<int64_t>(*net.seen) / (static_cast<int64_t>(net.batch) * static_cast<int64_t>(net.subdivisions));
		}
		return 0;
	}

	static inline ModernYoloTrainSchedule make_modern_yolo_train_schedule(const Darknet::Layer & l, const Darknet::Network & net)
	{
		ModernYoloTrainSchedule schedule;
		schedule.iteration = modern_yolo_current_iteration(net);
		schedule.atss_warmup = (l.atss_warmup_iters > 0 && schedule.iteration < static_cast<int64_t>(l.atss_warmup_iters));
		schedule.yolox_final_l1 =
			(l.l1_final_iters > 0 &&
			net.max_batches > 0 &&
			schedule.iteration > static_cast<int64_t>(net.max_batches) - static_cast<int64_t>(l.l1_final_iters));
		return schedule;
	}

	static inline float safe_log(float value)
	{
		return std::log(std::max(value, k_eps));
	}

	static inline float safe_sigmoid(float x)
	{
		x = std::clamp(finite_or_zero(x), -30.0f, 30.0f);
		return 1.0f / (1.0f + std::exp(-x));
	}


	static inline void sanitize_box(Darknet::Box & b)
	{
		b.x = finite_or_zero(b.x);
		b.y = finite_or_zero(b.y);
		b.w = std::max(finite_or_zero(b.w), 1e-6f);
		b.h = std::max(finite_or_zero(b.h), 1e-6f);
	}

	static inline float box_left(const Darknet::Box & b) { return b.x - 0.5f * b.w; }
	static inline float box_right(const Darknet::Box & b) { return b.x + 0.5f * b.w; }
	static inline float box_top(const Darknet::Box & b) { return b.y - 0.5f * b.h; }
	static inline float box_bottom(const Darknet::Box & b) { return b.y + 0.5f * b.h; }

	static inline Darknet::Box box_from_corners(float left, float top, float right, float bottom)
	{
		left = finite_or_zero(left);
		top = finite_or_zero(top);
		right = finite_or_zero(right);
		bottom = finite_or_zero(bottom);
		if (right < left) std::swap(left, right);
		if (bottom < top) std::swap(top, bottom);

		Darknet::Box b = {};
		b.x = 0.5f * (left + right);
		b.y = 0.5f * (top + bottom);
		b.w = std::max(right - left, 1e-6f);
		b.h = std::max(bottom - top, 1e-6f);
		sanitize_box(b);
		return b;
	}

	static inline bool valid_truth(const Darknet::Box & truth)
	{
		return std::isfinite(truth.x) && std::isfinite(truth.y) &&
			std::isfinite(truth.w) && std::isfinite(truth.h) &&
			truth.x > 0.0f && truth.x <= 1.0f && truth.y >= 0.0f && truth.y <= 1.0f &&
			truth.w > 0.0f && truth.h > 0.0f;
	}

	static inline bool point_inside_box_grid(const Darknet::Layer & l, const int loc, const Darknet::Box & truth)
	{
		const int col = loc % l.w;
		const int row = loc / l.w;
		const float cx = (static_cast<float>(col) + 0.5f) / static_cast<float>(l.w);
		const float cy = (static_cast<float>(row) + 0.5f) / static_cast<float>(l.h);
		return cx >= box_left(truth) && cx <= box_right(truth) && cy >= box_top(truth) && cy <= box_bottom(truth);
	}

	static inline bool point_inside_center_radius(const Darknet::Layer & l, const int loc, const Darknet::Box & truth, const float radius)
	{
		const int col = loc % l.w;
		const int row = loc / l.w;
		const float gx = (static_cast<float>(col) + 0.5f);
		const float gy = (static_cast<float>(row) + 0.5f);
		const float tx = truth.x * static_cast<float>(l.w);
		const float ty = truth.y * static_cast<float>(l.h);
		return std::fabs(gx - tx) <= radius && std::fabs(gy - ty) <= radius;
	}

	static inline int dfl_bins(const Darknet::Layer & l)
	{
		return std::max(2, l.coords / 4);
	}

	static inline float dfl_expectation(const Darknet::Layer & l, const float * x, const int batch, const int location, const int side)
	{
		const int bins = dfl_bins(l);
		const int stride = l.w * l.h;
		const int start = modern_entry_index(l, batch, location, side * bins);
		float max_logit = -std::numeric_limits<float>::infinity();
		for (int k = 0; k < bins; ++k)
		{
			max_logit = std::max(max_logit, finite_or_zero(x[start + k * stride]));
		}

		float denom = 0.0f;
		float expected = 0.0f;
		for (int k = 0; k < bins; ++k)
		{
			const float e = std::exp(std::clamp(finite_or_zero(x[start + k * stride]) - max_logit, -60.0f, 60.0f));
			denom += e;
			expected += static_cast<float>(k) * e;
		}
		return expected / std::max(denom, k_eps);
	}

	static inline void dfl_probabilities(const Darknet::Layer & l, const float * x, const int batch, const int location, const int side, std::vector<float> & probs)
	{
		const int bins = dfl_bins(l);
		const int stride = l.w * l.h;
		const int start = modern_entry_index(l, batch, location, side * bins);
		probs.assign(bins, 0.0f);

		float max_logit = -std::numeric_limits<float>::infinity();
		for (int k = 0; k < bins; ++k)
		{
			max_logit = std::max(max_logit, finite_or_zero(x[start + k * stride]));
		}

		float denom = 0.0f;
		for (int k = 0; k < bins; ++k)
		{
			probs[k] = std::exp(std::clamp(finite_or_zero(x[start + k * stride]) - max_logit, -60.0f, 60.0f));
			denom += probs[k];
		}
		denom = std::max(denom, k_eps);
		for (float & p : probs)
		{
			p /= denom;
		}
	}

	static inline Darknet::Box decode_yolox_box(const Darknet::Layer & l, const float * x, const int batch, const int n, const int loc)
	{
		const int row = loc / l.w;
		const int col = loc % l.w;
		const int location = n * l.w * l.h + loc;
		const int stride = l.w * l.h;
		const int box_index = modern_entry_index(l, batch, location, 0);

		Darknet::Box b = {};
		b.x = (static_cast<float>(col) + finite_or_zero(x[box_index + 0 * stride])) / static_cast<float>(l.w);
		b.y = (static_cast<float>(row) + finite_or_zero(x[box_index + 1 * stride])) / static_cast<float>(l.h);
		const float tw = std::clamp(finite_or_zero(x[box_index + 2 * stride]), -10.0f, 10.0f);
		const float th = std::clamp(finite_or_zero(x[box_index + 3 * stride]), -10.0f, 10.0f);
		b.w = std::exp(tw) / static_cast<float>(l.w);
		b.h = std::exp(th) / static_cast<float>(l.h);
		sanitize_box(b);
		return b;
	}

	static inline Darknet::Box decode_dfl_box(const Darknet::Layer & l, const float * x, const int batch, const int n, const int loc)
	{
		const int row = loc / l.w;
		const int col = loc % l.w;
		const int location = n * l.w * l.h + loc;

		const float cx = static_cast<float>(col) + 0.5f;
		const float cy = static_cast<float>(row) + 0.5f;
		const float left = dfl_expectation(l, x, batch, location, 0);
		const float top = dfl_expectation(l, x, batch, location, 1);
		const float right = dfl_expectation(l, x, batch, location, 2);
		const float bottom = dfl_expectation(l, x, batch, location, 3);

		return box_from_corners(
			(cx - left) / static_cast<float>(l.w),
			(cy - top) / static_cast<float>(l.h),
			(cx + right) / static_cast<float>(l.w),
			(cy + bottom) / static_cast<float>(l.h));
	}

	static inline Darknet::Box decode_modern_box(const Darknet::Layer & l, const float * x, const int batch, const int n, const int loc, const Darknet::ModernYoloHeadKind kind)
	{
		if (kind == Darknet::ModernYoloHeadKind::YOLOX)
		{
			return decode_yolox_box(l, x, batch, n, loc);
		}
		return decode_dfl_box(l, x, batch, n, loc);
	}

	static inline void activate_modern_output(Darknet::Layer & l)
	{
		const int spatial = l.w * l.h;
		for (int i = 0; i < l.batch * l.outputs; ++i)
		{
			l.output[i] = finite_or_zero(l.output[i]);
		}

		for (int b = 0; b < l.batch; ++b)
		{
			for (int n = 0; n < l.n; ++n)
			{
				const int location = n * spatial;
				const int obj_index = modern_entry_index(l, b, location, l.coords);
				for (int i = 0; i < (1 + l.classes) * spatial; ++i)
				{
					l.output[obj_index + i] = safe_sigmoid(l.output[obj_index + i]);
				}
			}
		}
	}

	static inline int obj_index_for(const Darknet::Layer & l, const int b, const int n, const int loc)
	{
		return modern_entry_index(l, b, n * l.w * l.h + loc, l.coords);
	}

	static inline int class_index_for(const Darknet::Layer & l, const int b, const int n, const int loc, const int class_id)
	{
		return modern_entry_index(l, b, n * l.w * l.h + loc, l.coords + 1 + class_id);
	}

	static inline int box_index_for(const Darknet::Layer & l, const int b, const int n, const int loc)
	{
		return modern_entry_index(l, b, n * l.w * l.h + loc, 0);
	}

	static inline float class_score_for(const Darknet::Layer & l, const int b, const int n, const int loc, const int class_id)
	{
		if (class_id < 0 || class_id >= l.classes)
		{
			return 0.0f;
		}
		return std::clamp(finite_or_zero(l.output[class_index_for(l, b, n, loc, class_id)]), 0.0f, 1.0f);
	}

	static inline float objectness_for(const Darknet::Layer & l, const int b, const int n, const int loc)
	{
		return std::clamp(finite_or_zero(l.output[obj_index_for(l, b, n, loc)]), 0.0f, 1.0f);
	}

	static inline float max_class_score_for(const Darknet::Layer & l, const int b, const int n, const int loc)
	{
		float score = 0.0f;
		for (int c = 0; c < l.classes; ++c)
		{
			score = std::max(score, class_score_for(l, b, n, loc, c));
		}
		return score;
	}

	static inline float assigned_class_loss_value(const Darknet::Layer & l, const int b, const int n, const int loc, const int class_id, const float quality, const bool enable_vfl)
	{
		const bool use_vfl = (enable_vfl && l.vfl_gamma > 0.0f);
		float target = std::clamp(quality, 0.0f, 1.0f);
		if (l.label_smooth_eps && !use_vfl)
		{
			target = target * (1.0f - l.label_smooth_eps) + 0.5f * l.label_smooth_eps;
		}

		const float p = class_score_for(l, b, n, loc, class_id);
		float loss = -target * safe_log(p) - (1.0f - target) * safe_log(1.0f - p);
		if (use_vfl)
		{
			loss *= target;
		}
		return finite_or_zero(loss);
	}

	static inline void validate_class_or_continue(const Darknet::Layer & l, const int class_id)
	{
		if (class_id >= l.classes || class_id < 0)
		{
			darknet_fatal_error(DARKNET_LOC, "invalid class ID #%d", class_id);
		}
	}

	static inline float target_ltrb_distance(const Darknet::Layer & l, const int loc, const Darknet::Box & truth, const int side)
	{
		const int col = loc % l.w;
		const int row = loc / l.w;
		const float cx = static_cast<float>(col) + 0.5f;
		const float cy = static_cast<float>(row) + 0.5f;
		const float left = box_left(truth) * static_cast<float>(l.w);
		const float top = box_top(truth) * static_cast<float>(l.h);
		const float right = box_right(truth) * static_cast<float>(l.w);
		const float bottom = box_bottom(truth) * static_cast<float>(l.h);

		switch (side)
		{
			case 0: return cx - left;
			case 1: return cy - top;
			case 2: return right - cx;
			default: return bottom - cy;
		}
	}

	static inline float dfl_side_loss_value(const Darknet::Layer & l, const int b, const int n, const int loc, const int side, float target_distance)
	{
		const int bins = dfl_bins(l);
		const int max_bin = bins - 1;
		const int stride = l.w * l.h;
		const int location = n * stride + loc;

		target_distance = std::clamp(finite_or_zero(target_distance), 0.0f, static_cast<float>(max_bin) - 1e-4f);
		const int left_bin = std::clamp(static_cast<int>(std::floor(target_distance)), 0, max_bin);
		const int right_bin = std::clamp(left_bin + 1, 0, max_bin);
		const float right_weight = target_distance - static_cast<float>(left_bin);
		const float left_weight = 1.0f - right_weight;

		std::vector<float> probs;
		dfl_probabilities(l, l.output, b, location, side, probs);
		const float loss =
			left_weight * -safe_log(probs[left_bin]) +
			right_weight * -safe_log(probs[right_bin]);
		return finite_or_zero(loss);
	}

	static inline float dfl_box_loss_value(const Darknet::Layer & l, const int b, const int n, const int loc, const Darknet::Box & truth)
	{
		float loss = 0.0f;
		for (int side = 0; side < 4; ++side)
		{
			loss += dfl_side_loss_value(l, b, n, loc, side, target_ltrb_distance(l, loc, truth, side));
		}
		return finite_or_zero(l.dfl_loss_weight * loss);
	}

	static inline void add_dfl_delta(Darknet::Layer & l, const int b, const int n, const int loc, const int side, float target_distance, const float normalizer)
	{
		const int bins = dfl_bins(l);
		const int max_bin = bins - 1;
		const int stride = l.w * l.h;
		const int location = n * stride + loc;
		const int start = modern_entry_index(l, b, location, side * bins);

		target_distance = std::clamp(finite_or_zero(target_distance), 0.0f, static_cast<float>(max_bin) - 1e-4f);
		const int left_bin = std::clamp(static_cast<int>(std::floor(target_distance)), 0, max_bin);
		const int right_bin = std::clamp(left_bin + 1, 0, max_bin);
		const float right_weight = target_distance - static_cast<float>(left_bin);
		const float left_weight = 1.0f - right_weight;

		std::vector<float> probs;
		dfl_probabilities(l, l.output, b, location, side, probs);
		for (int k = 0; k < bins; ++k)
		{
			float target = 0.0f;
			if (k == left_bin) target += left_weight;
			if (k == right_bin) target += right_weight;
			float d = normalizer * (target - probs[k]);
			if (std::isfinite(d))
			{
				l.delta[start + k * stride] += d;
			}
		}
	}

	static inline void add_yolox_l1_box_delta(Darknet::Layer & l, const int b, const int n, const int loc, const Darknet::Box & truth, const float normalizer)
	{
		const int col = loc % l.w;
		const int row = loc / l.w;
		const int stride = l.w * l.h;
		const int index = box_index_for(l, b, n, loc);
		const float tx = truth.x * static_cast<float>(l.w) - static_cast<float>(col);
		const float ty = truth.y * static_cast<float>(l.h) - static_cast<float>(row);
		const float tw = safe_log(truth.w * static_cast<float>(l.w));
		const float th = safe_log(truth.h * static_cast<float>(l.h));

		const float targets[4] = { tx, ty, tw, th };
		for (int c = 0; c < 4; ++c)
		{
			float d = normalizer * (targets[c] - finite_or_zero(l.output[index + c * stride]));
			if (l.max_delta != FLT_MAX)
			{
				d = std::clamp(d, -l.max_delta, l.max_delta);
			}
			if (std::isfinite(d))
			{
				l.delta[index + c * stride] += d;
			}
		}
	}

	static inline void add_yolox_iou_box_delta(Darknet::Layer & l, const int b, const int n, const int loc, const Darknet::Box & truth, const float normalizer)
	{
		const int stride = l.w * l.h;
		const int index = box_index_for(l, b, n, loc);
		const Darknet::Box pred = decode_yolox_box(l, l.output, b, n, loc);
		const float iou = std::clamp(finite_or_zero(box_iou(pred, truth)), 0.0f, 1.0f);
		const dxrep iou_grad = dx_box_iou(pred, truth, l.iou_loss);
		const float factor = (l.iou_loss == IOU)
			? 2.0f * iou * l.box_loss_weight
			: l.box_loss_weight;
		const float total = normalizer * factor;
		const float inv_w = 1.0f / static_cast<float>(l.w);
		const float inv_h = 1.0f / static_cast<float>(l.h);
		float deltas[4] =
		{
			total * iou_grad.dt * inv_w,
			total * iou_grad.db * inv_h,
			total * iou_grad.dl * pred.w,
			total * iou_grad.dr * pred.h
		};

		for (int c = 0; c < 4; ++c)
		{
			float d = deltas[c];
			if (l.max_delta != FLT_MAX)
			{
				d = std::clamp(d, -l.max_delta, l.max_delta);
			}
			if (std::isfinite(d))
			{
				l.delta[index + c * stride] += d;
			}
		}
	}

#ifdef MODERN_YOLO_GRAD_CHECK
	static inline float yolox_box_loss_value(const Darknet::Layer & l, const float * x, const int b, const int n, const int loc, const Darknet::Box & truth, const float normalizer)
	{
		const Darknet::Box pred = decode_yolox_box(l, x, b, n, loc);
		if (l.iou_loss == IOU)
		{
			const float iou = std::clamp(finite_or_zero(box_iou(pred, truth)), 0.0f, 1.0f);
			return normalizer * l.box_loss_weight * (1.0f - iou * iou);
		}
		const float objective = finite_or_zero(box_iou_kind(pred, truth, l.iou_loss));
		return normalizer * l.box_loss_weight * (1.0f - objective);
	}

	static void check_yolox_iou_box_delta_once(Darknet::Layer & l, const int b, const int n, const int loc, const Darknet::Box & truth, const float normalizer)
	{
		static bool checked = false;
		if (checked || l.max_delta != FLT_MAX)
		{
			return;
		}
		checked = true;

		constexpr float eps = 1e-3f;
		const int stride = l.w * l.h;
		const int index = box_index_for(l, b, n, loc);
		std::vector<float> probe(l.output, l.output + l.batch * l.outputs);
		for (int c = 0; c < 4; ++c)
		{
			probe[index + c * stride] += eps;
			const float plus = yolox_box_loss_value(l, probe.data(), b, n, loc, truth, normalizer);
			probe[index + c * stride] -= 2.0f * eps;
			const float minus = yolox_box_loss_value(l, probe.data(), b, n, loc, truth, normalizer);
			probe[index + c * stride] += eps;
			const float numeric = (plus - minus) / (2.0f * eps);
			// dx_box_iou() follows delta_yolo_box() and returns the full edge-pair w/h gradient.
			const float analytical = (c >= 2)
				? -0.5f * l.delta[index + c * stride]
				: -l.delta[index + c * stride];
			const float tolerance = std::max(5e-2f, std::fabs(numeric) * 0.2f);
			if (std::isfinite(numeric) && std::fabs(numeric - analytical) > tolerance)
			{
				darknet_fatal_error(DARKNET_LOC, "YOLOX IoU grad-check failed entry=%d numeric=%f analytical=%f tolerance=%f", c, numeric, analytical, tolerance);
			}
		}
	}
#endif

	static inline void add_dfl_box_delta(Darknet::Layer & l, const int b, const int n, const int loc, const Darknet::Box & truth, const float normalizer)
	{
		for (int side = 0; side < 4; ++side)
		{
			add_dfl_delta(l, b, n, loc, side, target_ltrb_distance(l, loc, truth, side), normalizer);
		}
	}

	static inline void add_dfl_giou_delta(Darknet::Layer & l, const int b, const int n, const int loc, const Darknet::Box & truth, const float weight)
	{
		const int bins = dfl_bins(l);
		const int stride = l.w * l.h;
		const int location = n * stride + loc;
		const Darknet::Box pred = decode_dfl_box(l, l.output, b, n, loc);
		const dxrep g = dx_box_iou(pred, truth, GIOU);
		const float inv_w = 1.0f / static_cast<float>(l.w);
		const float inv_h = 1.0f / static_cast<float>(l.h);
		// dx_box_iou() returns w/h gradients without the 0.5 edge factor
		// (g.dl = dIoU/dx2 - dIoU/dx1), so the true corner gradients are
		// dIoU/dx1 = 0.5*(g.dt - g.dl) and dIoU/dx2 = 0.5*(g.dt + g.dl).
		const float corner_grad[4] =
		{
			finite_or_zero(0.5f * (g.dt - g.dl)),
			finite_or_zero(0.5f * (g.db - g.dr)),
			finite_or_zero(0.5f * (g.dt + g.dl)),
			finite_or_zero(0.5f * (g.db + g.dr))
		};
		const float expectation_scale[4] =
		{
			-inv_w,
			-inv_h,
			inv_w,
			inv_h
		};

		std::vector<float> probs;
		for (int side = 0; side < 4; ++side)
		{
			dfl_probabilities(l, l.output, b, location, side, probs);
			const float expected = finite_or_zero(dfl_expectation(l, l.output, b, location, side));
			const int start = modern_entry_index(l, b, location, side * bins);
			const float chain = finite_or_zero(weight) * corner_grad[side] * expectation_scale[side];
			for (int k = 0; k < bins; ++k)
			{
				float d = chain * finite_or_zero(probs[k]) * (static_cast<float>(k) - expected);
				if (l.max_delta != FLT_MAX)
				{
					d = std::clamp(d, -l.max_delta, l.max_delta);
				}
				if (std::isfinite(d))
				{
					l.delta[start + k * stride] += d;
				}
			}
		}
	}

#ifdef MODERN_YOLO_GRAD_CHECK
	static inline int dfl_grad_check_index(const Darknet::Layer & l, const int b, const int n, const int loc)
	{
		const int bins = dfl_bins(l);
		const int stride = l.w * l.h;
		const int location = n * stride + loc;
		const int k = std::min(bins - 1, std::max(0, bins / 2));
		return modern_entry_index(l, b, location, 0) + k * stride;
	}

	static inline float dfl_giou_loss_value(const Darknet::Layer & l, const float * x, const int b, const int n, const int loc, const Darknet::Box & truth, const float weight)
	{
		const Darknet::Box pred = decode_dfl_box(l, x, b, n, loc);
		return finite_or_zero(weight) * (1.0f - finite_or_zero(box_iou_kind(pred, truth, GIOU)));
	}

	static void check_dfl_giou_delta_once(Darknet::Layer & l, const int b, const int n, const int loc, const Darknet::Box & truth, const float weight, const float delta_before)
	{
		static bool checked = false;
		if (checked || l.max_delta != FLT_MAX)
		{
			return;
		}
		checked = true;

		constexpr float eps = 1e-3f;
		const int index = dfl_grad_check_index(l, b, n, loc);
		std::vector<float> probe(l.output, l.output + l.batch * l.outputs);
		probe[index] += eps;
		const float plus = dfl_giou_loss_value(l, probe.data(), b, n, loc, truth, weight);
		probe[index] -= 2.0f * eps;
		const float minus = dfl_giou_loss_value(l, probe.data(), b, n, loc, truth, weight);
		probe[index] += eps;
		const float numeric = (plus - minus) / (2.0f * eps);
		const float analytical = -(l.delta[index] - delta_before);
		const float tolerance = std::max(5e-2f, std::fabs(numeric) * 0.2f);
		if (std::isfinite(numeric) && std::fabs(numeric - analytical) > tolerance)
		{
			darknet_fatal_error(DARKNET_LOC, "DFL GIoU grad-check failed numeric=%f analytical=%f tolerance=%f", numeric, analytical, tolerance);
		}
	}
#endif

	static inline void add_class_delta(Darknet::Layer & l, const int b, const int n, const int loc, const int class_id, const float quality, const bool enable_vfl)
	{
		const bool use_vfl = (enable_vfl && l.vfl_gamma > 0.0f);
		for (int c = 0; c < l.classes; ++c)
		{
			float target = (c == class_id) ? std::clamp(quality, 0.0f, 1.0f) : 0.0f;
			if (l.label_smooth_eps && !use_vfl)
			{
				target = target * (1.0f - l.label_smooth_eps) + 0.5f * l.label_smooth_eps;
			}

			const int idx = class_index_for(l, b, n, loc, c);
			const float p = std::clamp(finite_or_zero(l.output[idx]), 0.0f, 1.0f);
			float d = l.cls_normalizer * (target - p);
			if (use_vfl)
			{
				const float weight = (c == class_id) ? target : k_vfl_alpha * std::pow(p, l.vfl_gamma);
				d *= weight;
			}
			if (l.classes_multipliers && c == class_id)
			{
				d *= l.classes_multipliers[class_id];
			}
			if (std::isfinite(d))
			{
				l.delta[idx] += d;
			}
		}
	}

	static inline void add_objectness_delta(Darknet::Layer & l, const int b, const int n, const int loc, const float target)
	{
		const int idx = obj_index_for(l, b, n, loc);
		const float d = l.obj_normalizer * (std::clamp(target, 0.0f, 1.0f) - finite_or_zero(l.output[idx]));
		if (std::isfinite(d))
		{
			l.delta[idx] += d;
		}
	}

	static std::vector<Candidate> collect_candidates_for_truth(
		const Darknet::Layer & l,
		const int b,
		const int truth_index,
		const int class_id,
		const Darknet::Box & truth,
		const Darknet::ModernYoloHeadKind kind,
		const bool atss_warmup)
	{
		std::vector<Candidate> candidates;
		const int spatial = l.w * l.h;
		candidates.reserve(spatial);
		for (int n = 0; n < l.n; ++n)
		{
			for (int loc = 0; loc < spatial; ++loc)
			{
				const bool inside_box = point_inside_box_grid(l, loc, truth);
				const bool geometry_ok =
					(kind == Darknet::ModernYoloHeadKind::YOLOX)
					? (inside_box || point_inside_center_radius(l, loc, truth, l.center_radius))
					: (inside_box || (l.center_radius > 0.0f && point_inside_center_radius(l, loc, truth, l.center_radius)));
				if (!geometry_ok)
				{
					continue;
				}

				Candidate c;
				c.batch = b;
				c.anchor = n;
				c.loc = loc;
				c.truth_index = truth_index;
				c.class_id = class_id;
				c.truth = truth;
				c.pred = decode_modern_box(l, l.output, b, n, loc, kind);
				c.iou = std::clamp(finite_or_zero(box_iou(c.pred, truth)), 0.0f, 1.0f);
				const float obj = objectness_for(l, b, n, loc);
				const float cls = class_score_for(l, b, n, loc, class_id);

				if (kind == Darknet::ModernYoloHeadKind::YOLOX)
				{
					const float cls_obj = std::sqrt(std::max(cls * obj, k_eps));
					const float cls_loss = -safe_log(cls_obj);
					const float iou_loss = -safe_log(c.iou + k_eps);
					c.cost = cls_loss + 3.0f * iou_loss;
					c.score = 1.0f / (1.0f + c.cost);
				}
				else
				{
					if (atss_warmup)
					{
						// TODO: replace this ATSS-lite warmup with full ATSS topk-by-center-distance
						// mean+std IoU thresholding when the assignment code is split by head.
						c.score = c.iou;
					}
					else
					{
						const float aligned_cls = std::pow(std::max(cls * obj, k_eps), l.tal_alpha);
						const float aligned_iou = std::pow(std::max(c.iou, k_eps), l.tal_beta);
						c.score = aligned_cls * aligned_iou;
					}
					c.cost = -safe_log(c.score + k_eps);
				}

				if (std::isfinite(c.score) && std::isfinite(c.cost))
				{
					candidates.push_back(c);
				}
			}
		}
		return candidates;
	}

	static inline int simota_dynamic_k(const Darknet::Layer & l, std::vector<Candidate> candidates)
	{
		if (candidates.empty())
		{
			return 0;
		}
		std::sort(candidates.begin(), candidates.end(), [](const Candidate & a, const Candidate & b)
		{
			return a.iou > b.iou;
		});
		const int k = std::min(l.assign_topk, static_cast<int>(candidates.size()));
		float sum_iou = 0.0f;
		for (int i = 0; i < k; ++i)
		{
			sum_iou += candidates[i].iou;
		}
		return std::max(1, static_cast<int>(sum_iou));
	}

	static void assign_yolox(
		const Darknet::Layer & l,
		const Darknet::NetworkState & state,
		const int b,
		std::vector<Assignment> & assignments)
	{
		for (int t = 0; t < l.max_boxes; ++t)
		{
			Darknet::Box truth = float_to_box_stride(state.truth + t * l.truth_size + b * l.truths, 1);
			if (!truth.x)
			{
				break;
			}
			if (!valid_truth(truth))
			{
				darknet_fatal_error(DARKNET_LOC, "invalid YOLOX truth box (x=%f, y=%f, w=%f, h=%f)", truth.x, truth.y, truth.w, truth.h);
			}
			int class_id = static_cast<int>(state.truth[t * l.truth_size + b * l.truths + 4]);
			validate_class_or_continue(l, class_id);
			if (l.map)
			{
				class_id = l.map[class_id];
			}

			std::vector<Candidate> candidates = collect_candidates_for_truth(l, b, t, class_id, truth, Darknet::ModernYoloHeadKind::YOLOX, false);
			if (candidates.empty())
			{
				continue;
			}
			const int dynamic_k = std::min(simota_dynamic_k(l, candidates), static_cast<int>(candidates.size()));
			std::sort(candidates.begin(), candidates.end(), [](const Candidate & lhs, const Candidate & rhs)
			{
				return lhs.cost < rhs.cost;
			});

			for (int i = 0; i < dynamic_k; ++i)
			{
				const Candidate & c = candidates[i];
				const int slot = c.anchor * l.w * l.h + c.loc;
				Assignment & a = assignments[slot];
				const bool replace = (a.truth_index < 0 || c.cost < -a.quality);
				if (replace)
				{
					a.truth_index = c.truth_index;
					a.class_id = c.class_id;
					a.quality = -c.cost; // keep conflict key for YOLOX; overwritten before delta use.
					a.iou = c.iou;
					a.truth = c.truth;
				}
			}
		}
		for (Assignment & a : assignments)
		{
			if (a.truth_index >= 0)
			{
				// YOLOX defaults to hard classification labels when yolox_soft_label=0.
				// Using clamp(iou, 0.05) trains class scores toward 0.05 at cold-start,
				// which permanently suppresses outputs below the detection threshold.
				// Set yolox_soft_label=1 to use the matched IoU as the paper soft label.
				a.quality = (l.yolox_soft_label != 0) ? std::clamp(a.iou, 0.0f, 1.0f) : 1.0f;
			}
		}
	}

	static void assign_tal_like(
		const Darknet::Layer & l,
		const Darknet::NetworkState & state,
		const int b,
		std::vector<Assignment> & assignments,
		const bool nms_aware,
		const ModernYoloTrainSchedule & schedule)
	{
		std::vector<std::vector<int>> per_truth_slots(l.max_boxes);
		std::vector<float> conflict_score(assignments.size(), -1.0f);

		for (int t = 0; t < l.max_boxes; ++t)
		{
			Darknet::Box truth = float_to_box_stride(state.truth + t * l.truth_size + b * l.truths, 1);
			if (!truth.x)
			{
				break;
			}
			if (!valid_truth(truth))
			{
				darknet_fatal_error(DARKNET_LOC, "invalid anchor-free truth box (x=%f, y=%f, w=%f, h=%f)", truth.x, truth.y, truth.w, truth.h);
			}
			int class_id = static_cast<int>(state.truth[t * l.truth_size + b * l.truths + 4]);
			validate_class_or_continue(l, class_id);
			if (l.map)
			{
				class_id = l.map[class_id];
			}

			const auto kind = nms_aware ? Darknet::ModernYoloHeadKind::YOLONAS : Darknet::ModernYoloHeadKind::PPYOLOE;
			std::vector<Candidate> candidates = collect_candidates_for_truth(l, b, t, class_id, truth, kind, schedule.atss_warmup);
			if (candidates.empty())
			{
				continue;
			}
			std::sort(candidates.begin(), candidates.end(), [](const Candidate & lhs, const Candidate & rhs)
			{
				return lhs.score > rhs.score;
			});
			const int keep = std::min(l.assign_topk, static_cast<int>(candidates.size()));
			float max_score = 0.0f;
			float max_iou = 0.0f;
			for (int i = 0; i < keep; ++i)
			{
				max_score = std::max(max_score, candidates[i].score);
				max_iou = std::max(max_iou, candidates[i].iou);
			}
			for (int i = 0; i < keep; ++i)
			{
				const Candidate & c = candidates[i];
				const int slot = c.anchor * l.w * l.h + c.loc;
				if (c.score > conflict_score[slot])
				{
					const float quality = schedule.atss_warmup ?
						std::clamp(c.iou, 0.05f, 1.0f) :
						std::clamp(c.score / std::max(max_score, k_eps) * max_iou, 0.0f, 1.0f);
					conflict_score[slot] = c.score;
					Assignment & a = assignments[slot];
					a.truth_index = c.truth_index;
					a.class_id = c.class_id;
					a.iou = c.iou;
					a.quality = quality;
					a.truth = c.truth;
				}
			}
		}

		for (int slot = 0; slot < static_cast<int>(assignments.size()); ++slot)
		{
			const int t = assignments[slot].truth_index;
			if (t >= 0 && t < static_cast<int>(per_truth_slots.size()))
			{
				per_truth_slots[t].push_back(slot);
			}
		}

		if (nms_aware && l.nas_duplicate_decay != 0.0f)
		{
			for (std::vector<int> & slots : per_truth_slots)
			{
				std::sort(slots.begin(), slots.end(), [&assignments](const int lhs, const int rhs)
				{
					return assignments[lhs].iou > assignments[rhs].iou;
				});
				for (int rank = 0; rank < static_cast<int>(slots.size()); ++rank)
				{
					Assignment & a = assignments[slots[rank]];
					const float duplicate_weight = std::exp(-l.nas_duplicate_decay * static_cast<float>(rank));
					a.quality = std::clamp(a.quality * duplicate_weight, 0.02f, 1.0f);
				}
			}
		}
	}

	static float best_truth_iou_for_prediction(
		const Darknet::Layer & l,
		const Darknet::NetworkState & state,
		const int b,
		const int n,
		const int loc,
		const Darknet::ModernYoloHeadKind kind)
	{
		float best = 0.0f;
		const Darknet::Box pred = decode_modern_box(l, l.output, b, n, loc, kind);
		for (int t = 0; t < l.max_boxes; ++t)
		{
			Darknet::Box truth = float_to_box_stride(state.truth + t * l.truth_size + b * l.truths, 1);
			if (!truth.x)
			{
				break;
			}
			best = std::max(best, std::clamp(finite_or_zero(box_iou(pred, truth)), 0.0f, 1.0f));
		}
		return best;
	}

	static void train_batch_modern(
		Darknet::Layer & l,
		Darknet::NetworkState state,
		const int b,
		const Darknet::ModernYoloHeadKind kind,
		const ModernYoloTrainSchedule & schedule,
		ModernYoloStats & stats)
	{
		const int spatial = l.w * l.h;
		std::vector<Assignment> assignments(l.n * spatial);

		if (kind == Darknet::ModernYoloHeadKind::YOLOX)
		{
			assign_yolox(l, state, b, assignments);
		}
		else
		{
			assign_tal_like(l, state, b, assignments, kind == Darknet::ModernYoloHeadKind::YOLONAS, schedule);
		}

		for (int n = 0; n < l.n; ++n)
		{
			for (int loc = 0; loc < spatial; ++loc)
			{
				const int slot = n * spatial + loc;
				const Assignment & a = assignments[slot];
				if (a.truth_index < 0)
				{
					if (kind != Darknet::ModernYoloHeadKind::YOLOX && l.vfl_gamma > 0.0f)
					{
						add_class_delta(l, b, n, loc, -1, 0.0f, true);
					}
					const bool use_ignore_band = (kind != Darknet::ModernYoloHeadKind::YOLOX || l.yolox_ignore_neg != 0);
					if (use_ignore_band)
					{
						const float best_iou = best_truth_iou_for_prediction(l, state, b, n, loc, kind);
						if (best_iou > l.ignore_thresh)
						{
							if (l.objectness_smooth)
							{
								add_objectness_delta(l, b, n, loc, best_iou);
							}
							continue;
						}
					}
					add_objectness_delta(l, b, n, loc, 0.0f);
					continue;
				}

				const float class_multiplier = (l.classes_multipliers) ? l.classes_multipliers[a.class_id] : 1.0f;
				const float quality = std::clamp(a.quality, 0.0f, 1.0f);

				if (kind == Darknet::ModernYoloHeadKind::YOLOX)
				{
					add_objectness_delta(l, b, n, loc, 1.0f);
					add_class_delta(l, b, n, loc, a.class_id, quality, false);
					add_yolox_iou_box_delta(l, b, n, loc, a.truth, l.iou_normalizer * class_multiplier);
#ifdef MODERN_YOLO_GRAD_CHECK
					check_yolox_iou_box_delta_once(l, b, n, loc, a.truth, l.iou_normalizer * class_multiplier);
#endif
					if (schedule.yolox_final_l1)
					{
						// YOLOX adds L1 during the final no-aug fine-tune phase. Set mosaic=0
						// and augmentation off in cfg for that stage; this layer cannot flip aug per iteration.
						add_yolox_l1_box_delta(l, b, n, loc, a.truth, l.iou_normalizer * class_multiplier);
					}
				}
				else
				{
					add_objectness_delta(l, b, n, loc, quality);
					add_class_delta(l, b, n, loc, a.class_id, quality, true);
					add_dfl_box_delta(l, b, n, loc, a.truth, l.iou_normalizer * class_multiplier * l.dfl_loss_weight);
					const float dfl_giou_weight = l.iou_normalizer * class_multiplier * l.box_loss_weight * quality;
#ifdef MODERN_YOLO_GRAD_CHECK
					const int dfl_giou_check_index = dfl_grad_check_index(l, b, n, loc);
					const float dfl_giou_delta_before = l.delta[dfl_giou_check_index];
#endif
					add_dfl_giou_delta(l, b, n, loc, a.truth, dfl_giou_weight);
#ifdef MODERN_YOLO_GRAD_CHECK
					check_dfl_giou_delta_once(l, b, n, loc, a.truth, dfl_giou_weight, dfl_giou_delta_before);
#endif
				}

				const int out_index = b * l.n * spatial + n * spatial + loc;
				if (l.labels)
				{
					int track_id = 0;
					if (l.truth_size > 5)
					{
						track_id = static_cast<int>(state.truth[a.truth_index * l.truth_size + b * l.truths + 5]);
					}
					l.labels[out_index] = track_id;
				}
				if (l.class_ids)
				{
					l.class_ids[out_index] = a.class_id;
				}

				stats.sum_iou += a.iou;
				stats.sum_iou_loss += 1.0f - a.iou;
				stats.cls_loss += assigned_class_loss_value(l, b, n, loc, a.class_id, quality, kind != Darknet::ModernYoloHeadKind::YOLOX);
				if (kind == Darknet::ModernYoloHeadKind::YOLOX)
				{
					stats.box_loss += l.box_loss_weight * (1.0f - a.iou * a.iou);
				}
				else
				{
					const Darknet::Box pred = decode_modern_box(l, l.output, b, n, loc, kind);
					const float giou = finite_or_zero(box_giou(pred, a.truth));
					stats.box_loss += l.box_loss_weight * quality * (1.0f - giou);
					stats.dfl_loss += dfl_box_loss_value(l, b, n, loc, a.truth);
				}
				++stats.count;
				if (state.net.total_bbox)
				{
					(*state.net.total_bbox)++;
				}
			}
		}
	}

	static inline void set_default_training_params(Darknet::Layer & l, const Darknet::ModernYoloHeadKind kind)
	{
		if (l.ignore_thresh == 0.0f) l.ignore_thresh = 0.7f;
		if (l.truth_thresh == 0.0f) l.truth_thresh = 1.0f;
		if (l.iou_normalizer == 0.0f) l.iou_normalizer = 1.0f;
		if (l.obj_normalizer == 0.0f) l.obj_normalizer = 1.0f;
		if (l.cls_normalizer == 0.0f) l.cls_normalizer = 1.0f;
		if (l.delta_normalizer == 0.0f) l.delta_normalizer = 1.0f;
		if (l.max_delta == 0.0f) l.max_delta = FLT_MAX;
		if (kind == Darknet::ModernYoloHeadKind::YOLOX && l.center_radius == 0.0f) l.center_radius = 2.5f;
		if (l.assign_topk == 0) l.assign_topk = (kind == Darknet::ModernYoloHeadKind::YOLOX) ? 10 : 13;
		if (l.tal_alpha == 0.0f) l.tal_alpha = 1.0f;
		if (l.tal_beta == 0.0f) l.tal_beta = 6.0f;
		if (l.box_loss_weight == 0.0f) l.box_loss_weight = (kind == Darknet::ModernYoloHeadKind::YOLOX) ? 5.0f : 2.5f;
		if (l.dfl_loss_weight == 0.0f)
		{
			l.dfl_loss_weight = (kind == Darknet::ModernYoloHeadKind::YOLONAS) ? 0.25f :
				(kind == Darknet::ModernYoloHeadKind::PPYOLOE) ? 0.5f : 1.0f;
		}
		if (kind == Darknet::ModernYoloHeadKind::YOLONAS && l.nas_duplicate_decay == 0.0f) l.nas_duplicate_decay = 0.35f;
	}

	static void allocate_common_buffers(Darknet::Layer & l)
	{
		l.labels = static_cast<int*>(xcalloc(l.batch * l.w * l.h * l.n, sizeof(int)));
		l.class_ids = static_cast<int*>(xcalloc(l.batch * l.w * l.h * l.n, sizeof(int)));
		for (int i = 0; i < l.batch * l.w * l.h * l.n; ++i)
		{
			l.labels[i] = -1;
			l.class_ids[i] = -1;
		}
		l.delta = static_cast<float*>(xcalloc(l.batch * l.outputs, sizeof(float)));
		l.output = static_cast<float*>(xcalloc(l.batch * l.outputs, sizeof(float)));
		l.cost = static_cast<float*>(xcalloc(1, sizeof(float)));
	}

	static void set_layer_type_for_kind(Darknet::Layer & l, const Darknet::ModernYoloHeadKind kind)
	{
		switch (kind)
		{
			case Darknet::ModernYoloHeadKind::YOLOX:
				l.type = Darknet::ELayerType::YOLOX;
				break;
			case Darknet::ModernYoloHeadKind::PPYOLOE:
				l.type = Darknet::ELayerType::PPYOLOE;
				break;
			case Darknet::ModernYoloHeadKind::YOLONAS:
				l.type = Darknet::ELayerType::YOLONAS;
				break;
		}
	}

	static void set_forward_backward_for_kind(Darknet::Layer & l, const Darknet::ModernYoloHeadKind kind)
	{
		switch (kind)
		{
			case Darknet::ModernYoloHeadKind::YOLOX:
				l.forward = Darknet::forward_yolox_layer;
				l.backward = Darknet::backward_yolox_layer;
#ifdef DARKNET_GPU
				l.forward_gpu = Darknet::forward_yolox_layer_gpu;
				l.backward_gpu = Darknet::backward_yolox_layer_gpu;
#endif
				break;
			case Darknet::ModernYoloHeadKind::PPYOLOE:
				l.forward = Darknet::forward_ppyoloe_layer;
				l.backward = Darknet::backward_ppyoloe_layer;
#ifdef DARKNET_GPU
				l.forward_gpu = Darknet::forward_ppyoloe_layer_gpu;
				l.backward_gpu = Darknet::backward_ppyoloe_layer_gpu;
#endif
				break;
			case Darknet::ModernYoloHeadKind::YOLONAS:
				l.forward = Darknet::forward_yolonas_layer;
				l.backward = Darknet::backward_yolonas_layer;
#ifdef DARKNET_GPU
				l.forward_gpu = Darknet::forward_yolonas_layer_gpu;
				l.backward_gpu = Darknet::backward_yolonas_layer_gpu;
#endif
				break;
		}
	}
}

namespace Darknet
{
	const char * modern_yolo_head_name(ModernYoloHeadKind kind)
	{
		switch (kind)
		{
			case ModernYoloHeadKind::YOLOX: return "YOLOX";
			case ModernYoloHeadKind::PPYOLOE: return "PP-YOLOE";
			case ModernYoloHeadKind::YOLONAS: return "YOLO-NAS-style";
		}
		return "modern-yolo";
	}

	Darknet::Layer make_modern_yolo_layer(
		int batch,
		int w,
		int h,
		int classes,
		int max_boxes,
		ModernYoloHeadKind kind,
		int reg_max)
	{
		TAT(TATPARMS);

		Darknet::Layer l = { (Darknet::ELayerType)0 };
		set_layer_type_for_kind(l, kind);

		l.batch = batch;
		l.w = w;
		l.h = h;
		l.out_w = w;
		l.out_h = h;
		l.n = 1;
		l.total = 1;
		l.classes = classes;
		l.max_boxes = max_boxes;
		l.truth_size = 4 + 2;
		l.truths = l.max_boxes * l.truth_size;

		if (kind == ModernYoloHeadKind::YOLOX)
		{
			l.coords = 4;
		}
		else
		{
			reg_max = std::clamp(reg_max, 1, 64);
			l.coords = 4 * (reg_max + 1);
		}

		l.c = l.n * (l.classes + l.coords + 1);
		l.out_c = l.c;
		l.outputs = l.w * l.h * l.c;
		l.inputs = l.outputs;

		l.mask = static_cast<int*>(xcalloc(l.n, sizeof(int)));
		l.mask[0] = 0;
		l.biases = static_cast<float*>(xcalloc(2, sizeof(float)));
		l.bias_updates = static_cast<float*>(xcalloc(2, sizeof(float)));
		l.nbiases = 2;
		l.biases[0] = 0.5f;
		l.biases[1] = 0.5f;

		set_default_training_params(l, kind);
		allocate_common_buffers(l);
		set_forward_backward_for_kind(l, kind);

#ifdef DARKNET_GPU
		l.output_gpu = cuda_make_array(l.output, l.batch * l.outputs);
		l.output_avg_gpu = cuda_make_array(l.output, l.batch * l.outputs);
		l.delta_gpu = cuda_make_array(l.delta, l.batch * l.outputs);
#endif

		*cfg_and_state.output
			<< modern_yolo_head_name(kind)
			<< " modern anchor-free layer "
			<< l.w << " x " << l.h << " x " << l.c
			<< " -> " << l.outputs
			<< " classes=" << l.classes
			<< " coords=" << l.coords
			<< std::endl;

		return l;
	}

	void resize_modern_yolo_layer(Darknet::Layer * l, int w, int h)
	{
		TAT(TATPARMS);
		if (!l)
		{
			return;
		}
		l->w = w;
		l->h = h;
		l->out_w = w;
		l->out_h = h;
		l->outputs = h * w * l->n * (l->classes + l->coords + 1);
		l->inputs = l->outputs;
		l->out_c = l->n * (l->classes + l->coords + 1);
		l->c = l->out_c;

		if (l->labels) l->labels = static_cast<int*>(xrealloc(l->labels, l->batch * l->n * l->h * l->w * sizeof(int)));
		if (l->class_ids) l->class_ids = static_cast<int*>(xrealloc(l->class_ids, l->batch * l->n * l->h * l->w * sizeof(int)));
		for (int i = 0; i < l->batch * l->n * l->h * l->w; ++i)
		{
			if (l->labels) l->labels[i] = -1;
			if (l->class_ids) l->class_ids[i] = -1;
		}

		if (!l->output_pinned) l->output = static_cast<float*>(xrealloc(l->output, l->batch * l->outputs * sizeof(float)));
		if (!l->delta_pinned) l->delta = static_cast<float*>(xrealloc(l->delta, l->batch * l->outputs * sizeof(float)));

#ifdef DARKNET_GPU
		if (l->output_pinned)
		{
			CHECK_CUDA(cudaFreeHost(l->output));
			if (cudaSuccess != cudaHostAlloc(reinterpret_cast<void**>(&l->output), l->batch * l->outputs * sizeof(float), cudaHostRegisterMapped))
			{
				std::ignore = cudaGetLastError();
				l->output = static_cast<float*>(xcalloc(l->batch * l->outputs, sizeof(float)));
				l->output_pinned = 0;
			}
		}
		if (l->delta_pinned)
		{
			CHECK_CUDA(cudaFreeHost(l->delta));
			if (cudaSuccess != cudaHostAlloc(reinterpret_cast<void**>(&l->delta), l->batch * l->outputs * sizeof(float), cudaHostRegisterMapped))
			{
				std::ignore = cudaGetLastError();
				l->delta = static_cast<float*>(xcalloc(l->batch * l->outputs, sizeof(float)));
				l->delta_pinned = 0;
			}
		}
		cuda_free(l->delta_gpu);
		cuda_free(l->output_gpu);
		cuda_free(l->output_avg_gpu);
		l->delta_gpu = cuda_make_array(l->delta, l->batch * l->outputs);
		l->output_gpu = cuda_make_array(l->output, l->batch * l->outputs);
		l->output_avg_gpu = cuda_make_array(l->output, l->batch * l->outputs);
#endif
	}

	void forward_modern_yolo_layer(Darknet::Layer & l, Darknet::NetworkState state, ModernYoloHeadKind kind)
	{
		TAT(TATPARMS);
		std::memcpy(l.output, state.input, l.outputs * l.batch * sizeof(float));
		activate_modern_output(l);

		if (!state.train || l.onlyforward)
		{
			return;
		}
		const ModernYoloTrainSchedule schedule = make_modern_yolo_train_schedule(l, state.net);

		std::memset(l.delta, 0, l.outputs * l.batch * sizeof(float));
		for (int i = 0; i < l.batch * l.w * l.h * l.n; ++i)
		{
			if (l.labels) l.labels[i] = -1;
			if (l.class_ids) l.class_ids[i] = -1;
		}
		if (l.cost)
		{
			*l.cost = 0.0f;
		}

		if (!state.truth)
		{
			return;
		}

		ModernYoloStats stats;
		for (int b = 0; b < l.batch; ++b)
		{
			train_batch_modern(l, state, b, kind, schedule, stats);
		}

		for (int idx = 0; idx < l.batch * l.outputs; ++idx)
		{
			if (!std::isfinite(l.delta[idx]))
			{
				l.delta[idx] = 0.0f;
			}
		}

		const float loss = std::pow(mag_array(l.delta, l.outputs * l.batch), 2.0f);
		if (l.cost)
		{
			*l.cost = loss;
		}

		if (cfg_and_state.is_verbose)
		{
			const int denom = std::max(stats.count, 1);
			*cfg_and_state.output
				<< modern_yolo_head_name(kind) << " head, "
				<< "Region " << state.index << " "
				<< "Avg IOU: " << std::setprecision(6) << (stats.sum_iou / denom) << ", "
				<< "count: " << stats.count << ", "
				<< "iou_loss: " << std::setprecision(6) << (stats.sum_iou_loss / denom) << ", "
				<< "total_loss: " << std::setprecision(6) << (loss / std::max(l.batch, 1)) << ", "
				<< "cls_loss: " << std::setprecision(6) << (stats.cls_loss / denom) << ", "
				<< "box_loss: " << std::setprecision(6) << (stats.box_loss / denom);
			if (kind != Darknet::ModernYoloHeadKind::YOLOX)
			{
				*cfg_and_state.output
					<< ", dfl_loss: " << std::setprecision(6) << (stats.dfl_loss / denom);
			}
			*cfg_and_state.output
				<< std::setprecision(2) << std::endl;
		}
	}

	void backward_modern_yolo_layer(Darknet::Layer & l, Darknet::NetworkState state)
	{
		TAT(TATPARMS);
		if (!state.delta)
		{
			return;
		}
		axpy_cpu(l.batch * l.inputs, l.delta_normalizer, l.delta, 1, state.delta, 1);
	}

	int modern_yolo_num_detections_batch(const Darknet::Layer & l, float thresh, int batch)
	{
		TAT(TATPARMS);
		int count = 0;
		const bool cls_only = (l.score_mode == 1);
		for (int n = 0; n < l.n; ++n)
		{
			for (int loc = 0; loc < l.w * l.h; ++loc)
			{
				const float score = cls_only ? max_class_score_for(l, batch, n, loc) : objectness_for(l, batch, n, loc);
				if (score > thresh)
				{
					++count;
				}
			}
		}
		return count;
	}

	int modern_yolo_num_detections(const Darknet::Layer & l, float thresh)
	{
		return modern_yolo_num_detections_batch(l, thresh, 0);
	}

	void correct_modern_yolo_boxes(Darknet::Detection * dets, int n, int w, int h, int netw, int neth, int relative, int letter)
	{
		TAT(TATPARMS);
		int new_w = netw;
		int new_h = neth;
		if (letter)
		{
			if ((static_cast<float>(netw) / static_cast<float>(w)) < (static_cast<float>(neth) / static_cast<float>(h)))
			{
				new_h = (h * netw) / w;
			}
			else
			{
				new_w = (w * neth) / h;
			}
		}

		const float deltaw = static_cast<float>(netw - new_w);
		const float deltah = static_cast<float>(neth - new_h);
		const float ratiow = static_cast<float>(new_w) / static_cast<float>(netw);
		const float ratioh = static_cast<float>(new_h) / static_cast<float>(neth);

		for (int i = 0; i < n; ++i)
		{
			Darknet::Box b = dets[i].bbox;
			b.x = (b.x - deltaw * 0.5f / static_cast<float>(netw)) / ratiow;
			b.y = (b.y - deltah * 0.5f / static_cast<float>(neth)) / ratioh;
			b.w *= 1.0f / ratiow;
			b.h *= 1.0f / ratioh;
			if (!relative)
			{
				b.x *= w;
				b.w *= w;
				b.y *= h;
				b.h *= h;
			}
			dets[i].bbox = b;
		}
	}

	int get_modern_yolo_detections_batch(const Darknet::Layer & l, int w, int h, int netw, int neth, float thresh, int * map, int relative, Darknet::Detection * dets, int letter, int batch, ModernYoloHeadKind kind)
	{
		TAT(TATPARMS);
		(void)map;
		int count = 0;
		const bool cls_only = (l.score_mode == 1);
		for (int loc = 0; loc < l.w * l.h; ++loc)
		{
			for (int n = 0; n < l.n; ++n)
			{
				const float objectness = objectness_for(l, batch, n, loc);
				const float detection_score = cls_only ? max_class_score_for(l, batch, n, loc) : objectness;
				if (detection_score <= thresh)
				{
					continue;
				}

				dets[count].bbox = decode_modern_box(l, l.output, batch, n, loc, kind);
				dets[count].objectness = detection_score;
				dets[count].classes = l.classes;

				const int row = loc / l.w;
				const int col = loc % l.w;
				if (l.embedding_output)
				{
					get_embedding(l.embedding_output, l.w, l.h, l.n * l.embedding_size, l.embedding_size, col, row, n, batch, dets[count].embeddings);
				}

				for (int c = 0; c < l.classes; ++c)
				{
					const float class_score = class_score_for(l, batch, n, loc, c);
					const float prob = cls_only ? class_score : objectness * class_score;
					dets[count].prob[c] = (prob > thresh) ? prob : 0.0f;
				}
				++count;
			}
		}

		correct_modern_yolo_boxes(dets, count, w, h, netw, neth, relative, letter);
		return count;
	}

	int get_modern_yolo_detections(const Darknet::Layer & l, int w, int h, int netw, int neth, float thresh, int * map, int relative, Darknet::Detection * dets, int letter, ModernYoloHeadKind kind)
	{
		return get_modern_yolo_detections_batch(l, w, h, netw, neth, thresh, map, relative, dets, letter, 0, kind);
	}

#ifdef DARKNET_GPU
	void forward_modern_yolo_layer_gpu(Darknet::Layer & l, Darknet::NetworkState state, ModernYoloHeadKind kind)
	{
		TAT(TATPARMS);
		float * input_cpu = static_cast<float*>(xcalloc(l.batch * l.inputs, sizeof(float)));
		cuda_pull_array(state.input, input_cpu, l.batch * l.inputs);

		float * truth_cpu = nullptr;
		if (state.train && state.truth)
		{
			truth_cpu = static_cast<float*>(xcalloc(l.batch * l.truths, sizeof(float)));
			cuda_pull_array(state.truth, truth_cpu, l.batch * l.truths);
		}

		Darknet::NetworkState cpu_state = state;
		cpu_state.input = input_cpu;
		cpu_state.truth = truth_cpu;
		forward_modern_yolo_layer(l, cpu_state, kind);
		cuda_push_array(l.output_gpu, l.output, l.batch * l.outputs);
		if (state.train)
		{
			cuda_push_array(l.delta_gpu, l.delta, l.batch * l.outputs);
		}

		free(input_cpu);
		if (truth_cpu) free(truth_cpu);
	}

	void backward_modern_yolo_layer_gpu(Darknet::Layer & l, Darknet::NetworkState state)
	{
		TAT(TATPARMS);
		axpy_ongpu(l.batch * l.inputs, state.net.loss_scale * l.delta_normalizer, l.delta_gpu, 1, state.delta, 1);
	}
#endif
}
