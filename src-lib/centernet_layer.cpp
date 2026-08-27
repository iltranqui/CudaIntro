#include "centernet_layer.hpp"
#include "activations.hpp"
#include "blas.hpp"
#include "box.hpp"
#include "utils.hpp"
#include "dark_cuda.hpp"

#include <algorithm>
#include <cmath>
#include <cfloat>
#include <cstring>
#include <iomanip>
#include <vector>

namespace
{
	static auto & cfg_and_state = Darknet::CfgAndState::get();

	static inline int ct_index(const Darknet::Layer & l, const int batch, const int entry, const int loc)
	{
		return batch * l.outputs + entry * l.w * l.h + loc;
	}

	static inline float finite_or_zero(const float value)
	{
		return std::isfinite(value) ? value : 0.0f;
	}

	static inline float bounded_exp(const float value)
	{
		return std::exp(std::clamp(finite_or_zero(value), -20.0f, 20.0f));
	}

	static inline float clamp_delta(const float value, const float max_delta)
	{
		if (!std::isfinite(value))
		{
			return 0.0f;
		}
		if (max_delta == FLT_MAX)
		{
			return value;
		}
		return std::clamp(value, -max_delta, max_delta);
	}

	static inline float smooth_l1_loss(const float error)
	{
		const float abs_error = std::fabs(error);
		return (abs_error < 1.0f) ? 0.5f * error * error : abs_error - 0.5f;
	}

	static inline float smooth_l1_grad(const float error)
	{
		if (error > 1.0f) return 1.0f;
		if (error < -1.0f) return -1.0f;
		return error;
	}

	static inline float safe_log(const float value)
	{
		return std::log(std::max(value, 1e-9f));
	}

	static int gaussian_radius(const float width, const float height, const float min_overlap)
	{
		if (!std::isfinite(width) || !std::isfinite(height) || width <= 0.0f || height <= 0.0f)
		{
			return 0;
		}

		const float w = std::max(width, 1e-6f);
		const float h = std::max(height, 1e-6f);
		const float m = std::clamp(min_overlap, 0.01f, 0.99f);

		const float a1 = 1.0f;
		const float b1 = h + w;
		const float c1 = w * h * (1.0f - m) / (1.0f + m);
		const float d1 = std::max(0.0f, b1 * b1 - 4.0f * a1 * c1);
		const float r1 = (b1 + std::sqrt(d1)) / 2.0f;

		const float a2 = 4.0f;
		const float b2 = 2.0f * (h + w);
		const float c2 = (1.0f - m) * w * h;
		const float d2 = std::max(0.0f, b2 * b2 - 4.0f * a2 * c2);
		const float r2 = (b2 + std::sqrt(d2)) / 2.0f;

		const float a3 = 4.0f * m;
		const float b3 = -2.0f * m * (h + w);
		const float c3 = (m - 1.0f) * w * h;
		const float d3 = std::max(0.0f, b3 * b3 - 4.0f * a3 * c3);
		const float r3 = (b3 + std::sqrt(d3)) / (2.0f * a3);

		const float radius = std::min(r1, std::min(r2, r3));
		return std::max(0, static_cast<int>(std::floor(radius)));
	}

	static inline float continuous_small_weight(const Darknet::Layer & l, const float min_side_px)
	{
		const float max_weight = std::max(1.0f, l.ct_small_boost);
		const float ref_px = std::max(1.0f, l.ct_small_ref_size);
		if (!std::isfinite(min_side_px) || min_side_px <= 0.0f)
		{
			return max_weight;
		}
		return std::clamp(ref_px / std::max(min_side_px, 1e-3f), 1.0f, max_weight);
	}

	static inline void adaptive_gaussian_radii(
		const Darknet::Layer & l,
		const float w_cells,
		const float h_cells,
		int & radius_x,
		int & radius_y)
	{
		const int base_radius = std::max(l.ct_min_radius, gaussian_radius(w_cells, h_cells, l.ct_gaussian_iou));
		if (!l.ct_anisotropic_gaussian)
		{
			radius_x = base_radius;
			radius_y = base_radius;
			return;
		}

		const float safe_w = std::max(w_cells, 1e-6f);
		const float safe_h = std::max(h_cells, 1e-6f);
		const float aspect_x = std::sqrt(safe_w / safe_h);
		const float aspect_y = 1.0f / aspect_x;

		radius_x = static_cast<int>(std::round(static_cast<float>(base_radius) * aspect_x));
		radius_y = static_cast<int>(std::round(static_cast<float>(base_radius) * aspect_y));

		radius_x = std::max(l.ct_min_radius, radius_x);
		radius_y = std::max(l.ct_min_radius, radius_y);

		// Safety cap: prevent extremely elongated labels from creating huge dense heatmap writes.
		radius_x = std::min(radius_x, std::max(l.ct_min_radius, static_cast<int>(std::ceil(safe_w * 2.0f + 1.0f))));
		radius_y = std::min(radius_y, std::max(l.ct_min_radius, static_cast<int>(std::ceil(safe_h * 2.0f + 1.0f))));
	}

	static void draw_center_gaussian(
		const Darknet::Layer & l,
		std::vector<float> & target,
		std::vector<float> & weight,
		const int batch,
		const int class_id,
		const int cx,
		const int cy,
		const int radius_x,
		const int radius_y,
		const float small_weight)
	{
		const int spatial = l.w * l.h;
		const int base = (batch * l.classes + class_id) * spatial;
		const int rx = std::max(radius_x, 0);
		const int ry = std::max(radius_y, 0);
		const float sigma_x = std::max((2.0f * rx + 1.0f) / 6.0f, 1e-6f);
		const float sigma_y = std::max((2.0f * ry + 1.0f) / 6.0f, 1e-6f);
		const float inv_2sx2 = 1.0f / (2.0f * sigma_x * sigma_x);
		const float inv_2sy2 = 1.0f / (2.0f * sigma_y * sigma_y);

		for (int dy = -ry; dy <= ry; ++dy)
		{
			const int y = cy + dy;
			if (y < 0 || y >= l.h) continue;
			for (int dx = -rx; dx <= rx; ++dx)
			{
				const int x = cx + dx;
				if (x < 0 || x >= l.w) continue;

				const float value = std::exp(-(dx * dx * inv_2sx2 + dy * dy * inv_2sy2));
				const int index = base + y * l.w + x;
				if (value > target[index])
				{
					target[index] = value;
					weight[index] = std::max(weight[index], small_weight);
				}
			}
		}
	}

	static void correct_centernet_boxes(
		Darknet::Detection * dets,
		const int n,
		const int w,
		const int h,
		const int netw,
		const int neth,
		const int relative,
		const int letter)
	{
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

	static bool is_local_peak(const Darknet::Layer & l, const int batch, const int class_id, const int loc)
	{
		if (!l.ct_peak_nms)
		{
			return true;
		}
		const int row = loc / l.w;
		const int col = loc % l.w;
		const float center = l.output[ct_index(l, batch, class_id, loc)];
		for (int dy = -1; dy <= 1; ++dy)
		{
			const int y = row + dy;
			if (y < 0 || y >= l.h) continue;
			for (int dx = -1; dx <= 1; ++dx)
			{
				const int x = col + dx;
				if (x < 0 || x >= l.w) continue;
				if (dx == 0 && dy == 0) continue;
				const float neighbor = l.output[ct_index(l, batch, class_id, y * l.w + x)];
				if (neighbor > center)
				{
					return false;
				}
			}
		}
		return true;
	}

	static Darknet::Box decode_centernet_box(const Darknet::Layer & l, const int batch, const int loc)
	{
		const int row = loc / l.w;
		const int col = loc % l.w;

		const float raw_w = l.output[ct_index(l, batch, l.classes + 0, loc)];
		const float raw_h = l.output[ct_index(l, batch, l.classes + 1, loc)];
		const float raw_ox = l.output[ct_index(l, batch, l.classes + 2, loc)];
		const float raw_oy = l.output[ct_index(l, batch, l.classes + 3, loc)];

		Darknet::Box b = {};
		const float ox = std::clamp(finite_or_zero(raw_ox), -0.75f, 0.75f);
		const float oy = std::clamp(finite_or_zero(raw_oy), -0.75f, 0.75f);
		b.x = (static_cast<float>(col) + 0.5f + ox) / static_cast<float>(l.w);
		b.y = (static_cast<float>(row) + 0.5f + oy) / static_cast<float>(l.h);
		b.w = bounded_exp(raw_w) / static_cast<float>(l.w);
		b.h = bounded_exp(raw_h) / static_cast<float>(l.h);
		b.x = finite_or_zero(b.x);
		b.y = finite_or_zero(b.y);
		b.w = std::max(finite_or_zero(b.w), 1e-6f);
		b.h = std::max(finite_or_zero(b.h), 1e-6f);
		return b;
	}

	static void reset_labels(Darknet::Layer & l)
	{
		for (int i = 0; i < l.batch * l.w * l.h; ++i)
		{
			if (l.labels) l.labels[i] = -1;
			if (l.class_ids) l.class_ids[i] = -1;
		}
	}
}

namespace Darknet
{
	Darknet::Layer make_centernet_layer(const int batch, const int w, const int h, const int classes, const int max_boxes)
	{
		TAT(TATPARMS);

		Darknet::Layer l = { (Darknet::ELayerType)0 };
		l.type = Darknet::ELayerType::CENTERNET;
		l.batch = batch;
		l.w = w;
		l.h = h;
		l.out_w = w;
		l.out_h = h;
		l.n = 1;
		l.total = 1;
		l.classes = classes;
		l.coords = 4;
		l.c = classes + 4;
		l.out_c = l.c;
		l.outputs = w * h * l.c;
		l.inputs = l.outputs;
		l.max_boxes = max_boxes;
		l.truth_size = 4 + 2;
		l.truths = l.max_boxes * l.truth_size;
		l.cost = static_cast<float*>(xcalloc(1, sizeof(float)));
		l.output = static_cast<float*>(xcalloc(batch * l.outputs, sizeof(float)));
		l.delta = static_cast<float*>(xcalloc(batch * l.outputs, sizeof(float)));
		l.labels = static_cast<int*>(xcalloc(batch * w * h, sizeof(int)));
		l.class_ids = static_cast<int*>(xcalloc(batch * w * h, sizeof(int)));
		reset_labels(l);

		// CenterNet/tiny-object defaults.  All are cfg-overridable.
		l.ct_min_radius = 1;
		l.ct_peak_nms = 1;
		l.ct_anisotropic_gaussian = 1; // axis-aligned elliptical Gaussian by default
		l.ct_small_threshold = 8.0f;   // legacy cutoff/diagnostic threshold in input pixels
		l.ct_small_ref_size = 32.0f;   // continuous boost reference size in input pixels
		l.ct_small_boost = 8.0f;       // max total multiplier for tiny-object center/regression loss
		l.ct_scale_min_px = 0.0f;      // optional FPN assignment gate on max-side pixels
		l.ct_scale_max_px = FLT_MAX;
		l.ct_gaussian_iou = 0.7f;
		l.ct_focal_alpha = 2.0f;
		l.ct_focal_beta = 4.0f;
		l.ct_hm_normalizer = 1.0f;
		l.ct_wh_normalizer = 0.1f;
		l.ct_off_normalizer = 1.0f;
		l.max_delta = 10.0f;
		l.delta_normalizer = 1.0f;

		l.forward = Darknet::forward_centernet_layer;
		l.backward = Darknet::backward_centernet_layer;
#ifdef DARKNET_GPU
		l.forward_gpu = Darknet::forward_centernet_layer_gpu;
		l.backward_gpu = Darknet::backward_centernet_layer_gpu;
		l.output_gpu = cuda_make_array(l.output, batch * l.outputs);
		l.output_avg_gpu = cuda_make_array(l.output, batch * l.outputs);
		l.delta_gpu = cuda_make_array(l.delta, batch * l.outputs);
#endif

		*cfg_and_state.output
			<< "CenterNet anchor-free layer "
			<< l.w << " x " << l.h << " x " << l.c
			<< " -> " << l.outputs
			<< " classes=" << l.classes
			<< " min_radius=" << l.ct_min_radius
			<< " aniso=" << l.ct_anisotropic_gaussian
			<< " small_ref=" << l.ct_small_ref_size
			<< " small_boost_max=" << l.ct_small_boost
			<< std::endl;

		return l;
	}

	void resize_centernet_layer(Darknet::Layer * l, const int w, const int h)
	{
		TAT(TATPARMS);
		if (!l) return;

		l->w = w;
		l->h = h;
		l->out_w = w;
		l->out_h = h;
		l->c = l->classes + 4;
		l->out_c = l->c;
		l->outputs = w * h * l->c;
		l->inputs = l->outputs;

		l->labels = static_cast<int*>(xrealloc(l->labels, l->batch * w * h * sizeof(int)));
		l->class_ids = static_cast<int*>(xrealloc(l->class_ids, l->batch * w * h * sizeof(int)));
		l->output = static_cast<float*>(xrealloc(l->output, l->batch * l->outputs * sizeof(float)));
		l->delta = static_cast<float*>(xrealloc(l->delta, l->batch * l->outputs * sizeof(float)));
		reset_labels(*l);

#ifdef DARKNET_GPU
		cuda_free(l->delta_gpu);
		cuda_free(l->output_gpu);
		cuda_free(l->output_avg_gpu);
		l->delta_gpu = cuda_make_array(l->delta, l->batch * l->outputs);
		l->output_gpu = cuda_make_array(l->output, l->batch * l->outputs);
		l->output_avg_gpu = cuda_make_array(l->output, l->batch * l->outputs);
#endif
	}

	void forward_centernet_layer(Darknet::Layer & l, Darknet::NetworkState state)
	{
		TAT(TATPARMS);
		std::memcpy(l.output, state.input, l.outputs * l.batch * sizeof(float));

		const int spatial = l.w * l.h;
		for (int b = 0; b < l.batch; ++b)
		{
			for (int c = 0; c < l.classes; ++c)
			{
				activate_array(l.output + ct_index(l, b, c, 0), spatial, LOGISTIC);
			}
		}

		for (int idx = 0; idx < l.outputs * l.batch; ++idx)
		{
			if (!std::isfinite(l.output[idx]))
			{
				l.output[idx] = 0.0f;
			}
		}

		if (!state.train || l.onlyforward)
		{
			return;
		}

		std::memset(l.delta, 0, l.outputs * l.batch * sizeof(float));
		reset_labels(l);
		if (l.cost)
		{
			*l.cost = 0.0f;
		}
		if (!state.truth)
		{
			return;
		}

		std::vector<float> hm_target(l.batch * l.classes * spatial, 0.0f);
		std::vector<float> hm_weight(l.batch * l.classes * spatial, 1.0f);

		float hm_loss = 0.0f;
		float wh_loss = 0.0f;
		float off_loss = 0.0f;
		float total_iou = 0.0f;
		int positive_count = 0;

		// Build heatmap targets first.  This lets overlapping objects/classes keep the strongest center.
		for (int b = 0; b < l.batch; ++b)
		{
			for (int t = 0; t < l.max_boxes; ++t)
			{
				const float * truth_ptr = state.truth + b * l.truths + t * l.truth_size;
				Darknet::Box truth = float_to_box_stride(truth_ptr, 1);
				if (!truth.x) break;

				int class_id = static_cast<int>(truth_ptr[4]);
				if (class_id < 0 || class_id >= l.classes)
				{
					darknet_fatal_error(DARKNET_LOC, "[centernet] invalid class id %d outside [0,%d)", class_id, l.classes);
				}
				if (l.map)
				{
					class_id = l.map[class_id];
				}
				if (!std::isfinite(truth.x) || !std::isfinite(truth.y) || !std::isfinite(truth.w) || !std::isfinite(truth.h) ||
					truth.x <= 0.0f || truth.x > 1.0f || truth.y <= 0.0f || truth.y > 1.0f || truth.w <= 0.0f || truth.h <= 0.0f)
				{
					darknet_fatal_error(DARKNET_LOC, "[centernet] invalid truth box x=%f y=%f w=%f h=%f", truth.x, truth.y, truth.w, truth.h);
				}

				const float max_side_px = std::max(truth.w * state.net.w, truth.h * state.net.h);
				if (max_side_px < l.ct_scale_min_px || max_side_px >= l.ct_scale_max_px)
				{
					continue;
				}

				const float gx = truth.x * static_cast<float>(l.w);
				const float gy = truth.y * static_cast<float>(l.h);
				const int ix = std::clamp(static_cast<int>(std::floor(gx)), 0, l.w - 1);
				const int iy = std::clamp(static_cast<int>(std::floor(gy)), 0, l.h - 1);

				const float w_cells = std::max(truth.w * static_cast<float>(l.w), 1e-6f);
				const float h_cells = std::max(truth.h * static_cast<float>(l.h), 1e-6f);
				const float min_side_px = std::min(truth.w * state.net.w, truth.h * state.net.h);
				const float small_weight = continuous_small_weight(l, min_side_px);
				int radius_x = 0;
				int radius_y = 0;
				adaptive_gaussian_radii(l, w_cells, h_cells, radius_x, radius_y);

				draw_center_gaussian(l, hm_target, hm_weight, b, class_id, ix, iy, radius_x, radius_y, small_weight);
			}
		}

		// Heatmap focal loss.  Exact centers get positive focal gradient; Gaussian halo suppresses nearby negatives.
		for (int b = 0; b < l.batch; ++b)
		{
			for (int c = 0; c < l.classes; ++c)
			{
				for (int loc = 0; loc < spatial; ++loc)
				{
					const int hti = (b * l.classes + c) * spatial + loc;
					const int outi = ct_index(l, b, c, loc);
					const float y = std::clamp(hm_target[hti], 0.0f, 1.0f);
					const float p = std::clamp(l.output[outi], 1e-6f, 1.0f - 1e-6f);
					const float focal_alpha = std::max(0.0f, l.ct_focal_alpha);
					const float focal_beta = std::max(0.0f, l.ct_focal_beta);
					const float w = std::max(0.0f, hm_weight[hti]);
					if (y >= 0.999f)
					{
						const float focal = std::pow(1.0f - p, focal_alpha);
						const float grad = w * focal * (1.0f - p);
						l.delta[outi] += clamp_delta(l.ct_hm_normalizer * grad, l.max_delta);
						hm_loss += l.ct_hm_normalizer * w * focal * (-safe_log(p));
					}
					else
					{
						const float neg_weight = std::pow(1.0f - y, focal_beta);
						const float focal = std::pow(p, focal_alpha);
						const float grad = -w * neg_weight * focal * p;
						l.delta[outi] += clamp_delta(l.ct_hm_normalizer * grad, l.max_delta);
						hm_loss += l.ct_hm_normalizer * w * neg_weight * focal * (-safe_log(1.0f - p));
					}
				}
			}
		}

		// Sparse regression only at true centers.
		for (int b = 0; b < l.batch; ++b)
		{
			for (int t = 0; t < l.max_boxes; ++t)
			{
				const float * truth_ptr = state.truth + b * l.truths + t * l.truth_size;
				Darknet::Box truth = float_to_box_stride(truth_ptr, 1);
				if (!truth.x) break;

				int class_id = static_cast<int>(truth_ptr[4]);
				if (class_id < 0 || class_id >= l.classes) continue;
				if (l.map)
				{
					class_id = l.map[class_id];
				}

				const float max_side_px = std::max(truth.w * state.net.w, truth.h * state.net.h);
				if (max_side_px < l.ct_scale_min_px || max_side_px >= l.ct_scale_max_px)
				{
					continue;
				}

				const float gx = truth.x * static_cast<float>(l.w);
				const float gy = truth.y * static_cast<float>(l.h);
				const int ix = std::clamp(static_cast<int>(std::floor(gx)), 0, l.w - 1);
				const int iy = std::clamp(static_cast<int>(std::floor(gy)), 0, l.h - 1);
				const int loc = iy * l.w + ix;

				const float min_side_px = std::min(truth.w * state.net.w, truth.h * state.net.h);
				const float small_weight = continuous_small_weight(l, min_side_px);
				const float class_multiplier = l.classes_multipliers ? l.classes_multipliers[class_id] : 1.0f;
				const float weight = small_weight * class_multiplier;

				const float target_log_w = safe_log(std::max(truth.w * static_cast<float>(l.w), 1e-6f));
				const float target_log_h = safe_log(std::max(truth.h * static_cast<float>(l.h), 1e-6f));
				const float target_off_x = gx - static_cast<float>(ix) - 0.5f;
				const float target_off_y = gy - static_cast<float>(iy) - 0.5f;

				const int w_idx = ct_index(l, b, l.classes + 0, loc);
				const int h_idx = ct_index(l, b, l.classes + 1, loc);
				const int ox_idx = ct_index(l, b, l.classes + 2, loc);
				const int oy_idx = ct_index(l, b, l.classes + 3, loc);

				const float ew = finite_or_zero(l.output[w_idx]) - target_log_w;
				const float eh = finite_or_zero(l.output[h_idx]) - target_log_h;
				const float eox = finite_or_zero(l.output[ox_idx]) - target_off_x;
				const float eoy = finite_or_zero(l.output[oy_idx]) - target_off_y;

				l.delta[w_idx] += clamp_delta(-l.ct_wh_normalizer * weight * smooth_l1_grad(ew), l.max_delta);
				l.delta[h_idx] += clamp_delta(-l.ct_wh_normalizer * weight * smooth_l1_grad(eh), l.max_delta);
				l.delta[ox_idx] += clamp_delta(-l.ct_off_normalizer * weight * smooth_l1_grad(eox), l.max_delta);
				l.delta[oy_idx] += clamp_delta(-l.ct_off_normalizer * weight * smooth_l1_grad(eoy), l.max_delta);

				wh_loss += l.ct_wh_normalizer * weight * (smooth_l1_loss(ew) + smooth_l1_loss(eh));
				off_loss += l.ct_off_normalizer * weight * (smooth_l1_loss(eox) + smooth_l1_loss(eoy));

				const int out_label_idx = b * spatial + loc;
				if (l.labels) l.labels[out_label_idx] = static_cast<int>(truth_ptr[5]);
				if (l.class_ids) l.class_ids[out_label_idx] = class_id;

				const Darknet::Box pred = decode_centernet_box(l, b, loc);
				total_iou += box_iou(pred, truth);
				++positive_count;
				if (state.net.total_bbox)
				{
					++(*state.net.total_bbox);
				}
			}
		}

		for (int idx = 0; idx < l.batch * l.outputs; ++idx)
		{
			if (!std::isfinite(l.delta[idx]))
			{
				l.delta[idx] = 0.0f;
			}
		}

		const float total_loss = hm_loss + wh_loss + off_loss;
		if (l.cost)
		{
			*l.cost = total_loss / std::max(l.batch, 1);
		}

		if (cfg_and_state.is_verbose)
		{
			const int denom = std::max(positive_count, 1);
			*cfg_and_state.output
				<< "CenterNet head, Region " << state.index << " "
				<< "Avg IOU: " << std::setprecision(6) << (total_iou / denom) << ", "
				<< "count: " << positive_count << ", "
				<< "hm_loss: " << std::setprecision(6) << hm_loss << ", "
				<< "wh_loss: " << std::setprecision(6) << wh_loss << ", "
				<< "off_loss: " << std::setprecision(6) << off_loss << ", "
				<< "total_loss: " << std::setprecision(6) << (total_loss / std::max(l.batch, 1))
				<< std::setprecision(2) << std::endl;
		}
	}

	void backward_centernet_layer(Darknet::Layer & l, Darknet::NetworkState state)
	{
		TAT(TATPARMS);
		if (!state.delta)
		{
			return;
		}
		axpy_cpu(l.batch * l.inputs, l.delta_normalizer, l.delta, 1, state.delta, 1);
	}

	int centernet_num_detections_batch(const Darknet::Layer & l, const float thresh, const int batch)
	{
		TAT(TATPARMS);
		int count = 0;
		const int spatial = l.w * l.h;
		for (int c = 0; c < l.classes; ++c)
		{
			for (int loc = 0; loc < spatial; ++loc)
			{
				const float score = l.output[ct_index(l, batch, c, loc)];
				if (score > thresh && is_local_peak(l, batch, c, loc))
				{
					++count;
				}
			}
		}
		return count;
	}

	int centernet_num_detections(const Darknet::Layer & l, const float thresh)
	{
		return centernet_num_detections_batch(l, thresh, 0);
	}

	int get_centernet_detections_batch(
		const Darknet::Layer & l,
		const int w,
		const int h,
		const int netw,
		const int neth,
		const float thresh,
		int * map,
		const int relative,
		Darknet::Detection * dets,
		const int letter,
		const int batch)
	{
		TAT(TATPARMS);
		(void)map;
		int count = 0;
		const int spatial = l.w * l.h;
		for (int c = 0; c < l.classes; ++c)
		{
			for (int loc = 0; loc < spatial; ++loc)
			{
				const float score = l.output[ct_index(l, batch, c, loc)];
				if (score <= thresh || !is_local_peak(l, batch, c, loc))
				{
					continue;
				}

				dets[count].bbox = decode_centernet_box(l, batch, loc);
				dets[count].objectness = score;
				dets[count].classes = l.classes;
				dets[count].best_class_idx = c;
				for (int k = 0; k < l.classes; ++k)
				{
					dets[count].prob[k] = 0.0f;
				}
				dets[count].prob[c] = score;
				++count;
			}
		}

		correct_centernet_boxes(dets, count, w, h, netw, neth, relative, letter);
		return count;
	}

	int get_centernet_detections(
		const Darknet::Layer & l,
		const int w,
		const int h,
		const int netw,
		const int neth,
		const float thresh,
		int * map,
		const int relative,
		Darknet::Detection * dets,
		const int letter)
	{
		return get_centernet_detections_batch(l, w, h, netw, neth, thresh, map, relative, dets, letter, 0);
	}

#ifdef DARKNET_GPU
	void forward_centernet_layer_gpu(Darknet::Layer & l, Darknet::NetworkState state)
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
		forward_centernet_layer(l, cpu_state);
		cuda_push_array(l.output_gpu, l.output, l.batch * l.outputs);
		if (state.train)
		{
			cuda_push_array(l.delta_gpu, l.delta, l.batch * l.outputs);
		}

		free(input_cpu);
		if (truth_cpu) free(truth_cpu);
	}

	void backward_centernet_layer_gpu(Darknet::Layer & l, Darknet::NetworkState state)
	{
		TAT(TATPARMS);
		axpy_ongpu(l.batch * l.inputs, state.net.loss_scale * l.delta_normalizer, l.delta_gpu, 1, state.delta, 1);
	}
#endif
}
