/* Darknet/YOLO:  https://codeberg.org/iltranqui/darknet
 *
 * recursive_block_layer — weight-shared recurrent body with three optional extensions.
 *
 * Stage 0 (ouroboros=0): static blend each loop.
 *   next = body_scale*body(h) + residual_scale*h + injection_scale*input
 *
 * Stage 1 (ouroboros=1): FiLM/gate controller.
 *   The controller is a single dense layer over [GAP(h), step/loops].
 *   It emits per-channel gamma (feature-wise modulation) and gate (GRU-like blend):
 *     candidate = body_scale * body(h)
 *     gamma, gate = controller(GAP(h), step)
 *     next = gate * candidate * (1+gamma) + (1-gate) * h   [+ injection_scale*input]
 *
 * Stage 2 (ouroboros=2): Conv-LoRA adapters on body convs.
 *   Wraps each eligible convolutional body layer with a dynamic low-rank branch:
 *     y = Conv_base(x) + (lora_alpha/lora_rank) * Conv_B(diag_t * Conv_A(x))
 *   diag_t is emitted by the same controller that drives Stage 1 gate/gamma;
 *   controller output layout: [gamma * max_c] [gate * max_c] [diag * total_rank].
 *
 * LDT lattice projection (all stages, disabled by default):
 *   Between non-final loops, treat each group of `candidate_group` channels per
 *   spatial cell as a finite candidate set.  Meet the current proposal with the
 *   previous hidden state via min(alive_prev, alive_step); eliminate candidates
 *   below threshold; keep at least one candidate per grouped cell.
 *   Stored in reused generic Layer fields: alpha=mix, beta=threshold,
 *   scale=temperature, groups=candidate_group.
 *   Configured via set_recursive_block_lattice_projection() from the parser.
 *   No gradient is propagated through the projection (stop-gradient, forward-only).
 */

#include <algorithm>
#include <cmath>
#include <vector>

#include "darknet_internal.hpp"
#include "recursive_block_layer.hpp"
#include "convolutional_layer.hpp"
#include "maxpool_layer.hpp"
#include "yolo_layer.hpp"
#include "tucker_attention_layer.hpp"
#include "gemm.hpp"


namespace
{
	static auto & cfg_and_state = Darknet::CfgAndState::get();

	inline int rb_size(const int w, const int h, const int c)
	{
		return w * h * c;
	}

	inline int rb_total_size(const int batch, const int w, const int h, const int c)
	{
		return batch * rb_size(w, h, c);
	}

	inline int rb_nearest_index(const int out_idx, const int in_size, const int out_size)
	{
		if (out_size <= 1 || in_size <= 1)
		{
			return 0;
		}

		int idx = static_cast<int>((static_cast<long long>(out_idx) * in_size) / out_size);
		if (idx < 0) idx = 0;
		if (idx >= in_size) idx = in_size - 1;
		return idx;
	}

	inline float rb_sigmoid(const float x)
	{
		if (x >= 0.0f)
		{
			const float z = std::exp(-x);
			return 1.0f / (1.0f + z);
		}
		const float z = std::exp(x);
		return z / (1.0f + z);
	}

	inline float rb_clip(const float x, const float clip)
	{
		if (clip <= 0.0f || !std::isfinite(clip))
		{
			return x;
		}
		return std::clamp(x, -clip, clip);
	}

	// Controller input is the concatenation of:
	//   [0..c-1] = global average pooled hidden state (one scalar per channel)
	//   [c]      = step / (loops-1), normalised recurrence progress in [0,1]
	// This gives the controller awareness of both what the hidden state looks like
	// and how far along in the recurrence we are.
	inline int rb_controller_input_c(const Darknet::Layer & l)
	{
		return l.c + 1;
	}

	// Controller output width must cover both the hidden-state dimensions (l.c) and
	// the final visible output dimensions (l.out_c) when they differ (body changes size).
	// Using max(c, out_c) guarantees enough slots for both the final and non-final pass.
	inline int rb_controller_max_c(const Darknet::Layer & l)
	{
		return std::max(l.c, l.out_c);
	}

	inline bool rb_stage2_lora_enabled(const Darknet::Layer & l)
	{
		return l.rb_ouroboros == 2 && l.rb_lora_rank > 0;
	}

	inline int rb_lora_diag_offset(const Darknet::Layer & l)
	{
		return 2 * rb_controller_max_c(l);
	}

	// Controller output layout (all values are raw pre-activation unless noted):
	//   [0 .. max_c)           = gamma (feature-wise scale; added to 1 before multiply)
	//   [max_c .. 2*max_c)     = gate raw logits (passed through sigmoid in mix step)
	//   [2*max_c .. total)     = Stage 2 LoRA diagonal values, one per rank slot
	//                            (total = 2*max_c + lora_adapters*lora_rank)
	inline int rb_controller_output_c(const Darknet::Layer & l)
	{
		const int lora_outputs = rb_stage2_lora_enabled(l) ? l.rb_lora_total_rank : 0;
		return rb_lora_diag_offset(l) + lora_outputs;
	}

	inline bool rb_ouroboros_enabled(const Darknet::Layer & l)
	{
		return l.rb_ouroboros > 0;
	}

	inline float rb_lora_scaling(const Darknet::Layer & l)
	{
		return (l.rb_lora_rank > 0) ? (l.rb_lora_alpha / static_cast<float>(l.rb_lora_rank)) : 0.0f;
	}

	// Forward declarations used by the Stage 2 helpers below.
	void rb_controller_pack_input_cpu(Darknet::Layer & l, const float *hidden, const int step, const int loops);
	void rb_controller_forward_cpu(Darknet::Layer & l);

	/**
	 * Add a dimension-adapted NCHW tensor to @p dst.  This is intentionally not learnable:
	 * it is a safe adapter that lets recursive_block run when the body changes width,
	 * height, or channels.  Spatial changes use nearest-neighbour sampling; channels are
	 * copied for the overlapping range and absent channels contribute zero.
	 */
	void rb_adapt_add_cpu(
		const float *src,
		const int src_w,
		const int src_h,
		const int src_c,
		const int batch,
		float alpha,
		float *dst,
		const int dst_w,
		const int dst_h,
		const int dst_c)
	{
		if (src == nullptr || dst == nullptr || alpha == 0.0f || batch <= 0 ||
			src_w <= 0 || src_h <= 0 || src_c <= 0 || dst_w <= 0 || dst_h <= 0 || dst_c <= 0)
		{
			return;
		}

		const int src_area = src_w * src_h;
		const int dst_area = dst_w * dst_h;
		const int src_plane = src_area * src_c;
		const int dst_plane = dst_area * dst_c;
		const int channels = std::min(src_c, dst_c);

		#pragma omp parallel for collapse(2) schedule(static)
		for (int b = 0; b < batch; ++b)
		{
			for (int dc = 0; dc < channels; ++dc)
			{
				for (int dy = 0; dy < dst_h; ++dy)
				{
					const int sy = rb_nearest_index(dy, src_h, dst_h);
					for (int dx = 0; dx < dst_w; ++dx)
					{
						const int sx = rb_nearest_index(dx, src_w, dst_w);
						const int src_index = b * src_plane + dc * src_area + sy * src_w + sx;
						const int dst_index = b * dst_plane + dc * dst_area + dy * dst_w + dx;
						dst[dst_index] += alpha * src[src_index];
					}
				}
			}
		}
	}

	void rb_adapt_set_cpu(
		const float *src,
		const int src_w,
		const int src_h,
		const int src_c,
		const int batch,
		float alpha,
		float *dst,
		const int dst_w,
		const int dst_h,
		const int dst_c)
	{
		fill_cpu(rb_total_size(batch, dst_w, dst_h, dst_c), 0.0f, dst, 1);
		rb_adapt_add_cpu(src, src_w, src_h, src_c, batch, alpha, dst, dst_w, dst_h, dst_c);
	}

	/**
	 * Backward/transpose of rb_adapt_add_cpu(): accumulate output-space gradients into
	 * the source-space gradient buffer.
	 */
	void rb_adapt_backward_add_cpu(
		const float *delta_dst,
		const int dst_w,
		const int dst_h,
		const int dst_c,
		const int batch,
		float alpha,
		float *delta_src,
		const int src_w,
		const int src_h,
		const int src_c)
	{
		if (delta_dst == nullptr || delta_src == nullptr || alpha == 0.0f || batch <= 0 ||
			src_w <= 0 || src_h <= 0 || src_c <= 0 || dst_w <= 0 || dst_h <= 0 || dst_c <= 0)
		{
			return;
		}

		const int src_area = src_w * src_h;
		const int dst_area = dst_w * dst_h;
		const int src_plane = src_area * src_c;
		const int dst_plane = dst_area * dst_c;
		const int channels = std::min(src_c, dst_c);

		for (int b = 0; b < batch; ++b)
		{
			for (int dc = 0; dc < channels; ++dc)
			{
				for (int dy = 0; dy < dst_h; ++dy)
				{
					const int sy = rb_nearest_index(dy, src_h, dst_h);
					for (int dx = 0; dx < dst_w; ++dx)
					{
						const int sx = rb_nearest_index(dx, src_w, dst_w);
						const int dst_index = b * dst_plane + dc * dst_area + dy * dst_w + dx;
						const int src_index = b * src_plane + dc * src_area + sy * src_w + sx;
						delta_src[src_index] += alpha * delta_dst[dst_index];
					}
				}
			}
		}
	}

	// Sigmoid clamped to ±30 to prevent float overflow in expf() before the
	// result would be indistinguishable from 0 or 1 anyway.
	inline float rb_clamped_sigmoid(const float x)
	{
		const float z = std::clamp(x, -30.0f, 30.0f);
		return 1.0f / (1.0f + std::exp(-z));
	}

	// Convert a raw feature map value to an alive probability in [0,1].
	// Three cases: NaN/Inf → 0 (dead); already in [0,1] → pass through;
	// otherwise treat as a logit and apply temperature-scaled sigmoid.
	// The temperature parameter lets the caller control how sharply logits
	// are mapped: temperature→0 makes it a hard threshold, temperature=1
	// is a standard sigmoid.
	inline float rb_alive_probability(const float value, const float temperature)
	{
		if (!std::isfinite(value))
		{
			return 0.0f;
		}
		if (value >= 0.0f && value <= 1.0f)
		{
			return value;
		}
		return rb_clamped_sigmoid(value / std::max(temperature, 1e-6f));
	}

	/**
	 * LDT-style lattice projection applied between non-final recursive passes.
	 *
	 * Algorithm (per spatial position, per channel group of size `candidate_group`):
	 *   1. Map each channel value to an alive probability via rb_alive_probability().
	 *   2. Meet (element-wise min) the current step's alive vector with the previous
	 *      hidden state's alive vector.  The meet is the lattice greatest lower bound:
	 *      a candidate is alive only if both the body and the prior state allow it.
	 *   3. Eliminate candidates whose meet value falls below `elimination_threshold`.
	 *   4. If the whole group would be wiped out (any_alive==false) and group > 1,
	 *      rescue the candidate with the highest alive value.  This prevents a fully
	 *      dead cell and mirrors LDT's "keep-best" fallback when backtracking would
	 *      be needed but is not available.
	 *   5. Blend the projected values back: state = (1-mix)*state + mix*projected.
	 *      mix=0 disables the projection; mix=1 fully replaces the hidden state.
	 *
	 * No gradient flows through this operation.  It acts as a forward-only filter
	 * that biases the recurrence toward monotonically narrowing candidate sets.
	 *
	 * The `previous_state` pointer is l.rb_last_input, which is the stable copy of
	 * the hidden state snapshotted at the start of each loop before the body runs.
	 */
	void rb_lattice_project_cpu(
		const float *previous_state,
		float *state,
		const int batch,
		const int w,
		const int h,
		const int c,
		const int candidate_group,
		const float elimination_threshold,
		const float temperature,
		const float mix)
	{
		if (previous_state == nullptr || state == nullptr || batch <= 0 || w <= 0 || h <= 0 || c <= 0 || mix <= 0.0f)
		{
			return;
		}

		const int area = w * h;
		const int group = (candidate_group > 1 && c % candidate_group == 0) ? candidate_group : 1;
		const int groups_per_cell = c / group;
		const float theta = std::clamp(elimination_threshold, 0.0f, 1.0f);
		const float tau = std::max(temperature, 1e-6f);
		const float blend = std::clamp(mix, 0.0f, 1.0f);

		#pragma omp parallel for collapse(3) schedule(static)
		for (int b = 0; b < batch; ++b)
		{
			for (int sp = 0; sp < area; ++sp)
			{
				for (int g = 0; g < groups_per_cell; ++g)
				{
					float best_alive = -1.0f;
					int best_k = 0;
					bool any_alive = false;
					float projected_stack[64];
					std::vector<float> projected_heap;
					float *projected = projected_stack;
					if (group > static_cast<int>(sizeof(projected_stack) / sizeof(projected_stack[0])))
					{
						projected_heap.assign(group, 0.0f);
						projected = projected_heap.data();
					}

					for (int k = 0; k < group; ++k)
					{
						const int ch = g * group + k;
						const int idx = b * c * area + ch * area + sp;
						const float prev_alive = rb_alive_probability(previous_state[idx], tau);
						const float step_alive = rb_alive_probability(state[idx], tau);
						float alive = std::min(prev_alive, step_alive);
						if (!std::isfinite(alive)) alive = 0.0f;
						if (alive > best_alive)
						{
							best_alive = alive;
							best_k = k;
						}
						if (alive < theta)
						{
							alive = 0.0f;
						}
						else
						{
							any_alive = true;
						}
						projected[k] = alive;
					}

					if (group > 1 && !any_alive && best_alive > 0.0f)
					{
						projected[best_k] = best_alive;
					}

					for (int k = 0; k < group; ++k)
					{
						const int ch = g * group + k;
						const int idx = b * c * area + ch * area + sp;
						state[idx] = (1.0f - blend) * state[idx] + blend * projected[k];
					}
				}
			}
		}
	}

	void rb_resize_body_layer(Darknet::Layer & bl, const int w, const int h)
	{
		switch (bl.type)
		{
			case Darknet::ELayerType::CONVOLUTIONAL:
				resize_convolutional_layer(&bl, w, h);
				break;

			case Darknet::ELayerType::MAXPOOL:
			case Darknet::ELayerType::LOCAL_AVGPOOL:
				resize_maxpool_layer(&bl, w, h);
				break;

			case Darknet::ELayerType::YOLO:
				resize_yolo_layer(&bl, w, h);
				break;

			case Darknet::ELayerType::TUCKER_ATTENTION:
				resize_tucker_attention_layer(&bl, w, h);
				break;

			default:
				// Many layers do not need explicit resize or are not valid inside a recurrent body.
				bl.h = h;
				bl.w = w;
				bl.inputs = w * h * bl.c;
				break;
		}
	}

	void rb_resize_body_stack(Darknet::Layer & l)
	{
		if (l.rb_body == nullptr || l.rb_body_count <= 0)
		{
			return;
		}

		int cur_w = l.w;
		int cur_h = l.h;
		for (int j = 0; j < l.rb_body_count; ++j)
		{
			rb_resize_body_layer(l.rb_body[j], cur_w, cur_h);
			cur_w = l.rb_body[j].out_w;
			cur_h = l.rb_body[j].out_h;
		}
	}


	bool rb_lora_body_layer_supported(const Darknet::Layer & bl)
	{
		if (bl.type != Darknet::ELayerType::CONVOLUTIONAL)
		{
			return false;
		}
		if (bl.groups != 1 || bl.binary || bl.xnor || bl.deform || bl.antialiasing)
		{
			return false;
		}
		if (bl.out_w <= 0 || bl.out_h <= 0 || bl.out_c <= 0 || bl.outputs <= 0)
		{
			return false;
		}
		return true;
	}

	void rb_apply_activation_cpu(Darknet::Layer & bl, const ACTIVATION activation)
	{
		const int total = bl.outputs * bl.batch;
		if (activation == LINEAR || total <= 0)
		{
			return;
		}

		if ((activation == SWISH || activation == MISH || activation == HARD_MISH || activation == EML) && bl.activation_input == nullptr)
		{
			bl.activation_input = (float*)xcalloc(total, sizeof(float));
		}

		if (activation == SWISH) activate_array_swish(bl.output, total, bl.activation_input, bl.output);
		else if (activation == MISH) activate_array_mish(bl.output, total, bl.activation_input, bl.output);
		else if (activation == HARD_MISH) activate_array_hard_mish(bl.output, total, bl.activation_input, bl.output);
		else if (activation == EML) activate_array_eml(bl.output, total, bl.activation_input, bl.output);
		else if (activation == NORM_CHAN) activate_array_normalize_channels(bl.output, total, bl.batch, bl.out_c, bl.out_w * bl.out_h, bl.output);
		else if (activation == NORM_CHAN_SOFTMAX) activate_array_normalize_channels_softmax(bl.output, total, bl.batch, bl.out_c, bl.out_w * bl.out_h, bl.output, 0);
		else if (activation == NORM_CHAN_SOFTMAX_MAXVAL) activate_array_normalize_channels_softmax(bl.output, total, bl.batch, bl.out_c, bl.out_w * bl.out_h, bl.output, 1);
		else activate_array_cpu_custom(bl.output, total, activation);
	}

	void rb_prepare_ouroboros_controller_cpu(Darknet::Layer & l, const float *hidden, const int step, const int loops)
	{
		if (!rb_ouroboros_enabled(l))
		{
			return;
		}
		rb_controller_pack_input_cpu(l, hidden, step, loops);
		rb_controller_forward_cpu(l);
	}

	void rb_free_lora_adapters_cpu(Darknet::Layer & l)
	{
		if (l.rb_lora_A)
		{
			for (int k = 0; k < l.rb_lora_adapters; ++k)
			{
				free_layer_custom(l.rb_lora_A[k], 0);
			}
			free(l.rb_lora_A);
			l.rb_lora_A = nullptr;
		}
		if (l.rb_lora_B)
		{
			for (int k = 0; k < l.rb_lora_adapters; ++k)
			{
				free_layer_custom(l.rb_lora_B[k], 0);
			}
			free(l.rb_lora_B);
			l.rb_lora_B = nullptr;
		}
		if (l.rb_lora_scaled)
		{
			for (int k = 0; k < l.rb_lora_adapters; ++k) free(l.rb_lora_scaled[k]);
			free(l.rb_lora_scaled);
			l.rb_lora_scaled = nullptr;
		}
		if (l.rb_lora_scaled_delta)
		{
			for (int k = 0; k < l.rb_lora_adapters; ++k) free(l.rb_lora_scaled_delta[k]);
			free(l.rb_lora_scaled_delta);
			l.rb_lora_scaled_delta = nullptr;
		}
		free(l.rb_lora_scaled_sizes);
		free(l.rb_lora_body_adapter);
		free(l.rb_lora_body_indices);
		free(l.rb_lora_rank_offsets);
		l.rb_lora_scaled_sizes = nullptr;
		l.rb_lora_body_adapter = nullptr;
		l.rb_lora_body_indices = nullptr;
		l.rb_lora_rank_offsets = nullptr;

#ifdef DARKNET_GPU
		if (l.rb_lora_scaled_gpu)
		{
			for (int k = 0; k < l.rb_lora_adapters; ++k)
			{
				if (l.rb_lora_scaled_gpu[k]) cuda_free(l.rb_lora_scaled_gpu[k]);
			}
			free(l.rb_lora_scaled_gpu);
			l.rb_lora_scaled_gpu = nullptr;
		}
		if (l.rb_lora_scaled_delta_gpu)
		{
			for (int k = 0; k < l.rb_lora_adapters; ++k)
			{
				if (l.rb_lora_scaled_delta_gpu[k]) cuda_free(l.rb_lora_scaled_delta_gpu[k]);
			}
			free(l.rb_lora_scaled_delta_gpu);
			l.rb_lora_scaled_delta_gpu = nullptr;
		}
		free(l.rb_lora_scaled_gpu_sizes);
		l.rb_lora_scaled_gpu_sizes = nullptr;
#endif

		l.rb_lora_adapters = 0;
		l.rb_lora_total_rank = 0;
		l.rb_lora_diag_offset = 0;
		l.rb_lora_configured_body_count = 0;
	}

	void rb_ensure_lora_buffers_cpu(Darknet::Layer & l)
	{
		if (!rb_stage2_lora_enabled(l) || l.rb_lora_adapters <= 0 || l.rb_lora_A == nullptr)
		{
			return;
		}
		for (int k = 0; k < l.rb_lora_adapters; ++k)
		{
			const int needed = l.rb_lora_A[k].batch * l.rb_lora_A[k].outputs;
			if (needed <= 0) continue;
			if (l.rb_lora_scaled_sizes[k] < needed || l.rb_lora_scaled[k] == nullptr || l.rb_lora_scaled_delta[k] == nullptr)
			{
				l.rb_lora_scaled[k] = (float*)xrealloc(l.rb_lora_scaled[k], needed * sizeof(float));
				l.rb_lora_scaled_delta[k] = (float*)xrealloc(l.rb_lora_scaled_delta[k], needed * sizeof(float));
				l.rb_lora_scaled_sizes[k] = needed;
			}
		}
	}

	void rb_resize_lora_adapters_cpu(Darknet::Layer & l)
	{
		if (l.rb_lora_A == nullptr || l.rb_lora_B == nullptr || l.rb_lora_body_indices == nullptr)
		{
			return;
		}
		for (int k = 0; k < l.rb_lora_adapters; ++k)
		{
			const int j = l.rb_lora_body_indices[k];
			if (j < 0 || j >= l.rb_body_count) continue;
			Darknet::Layer & base = l.rb_body[j];
			Darknet::Layer & A = l.rb_lora_A[k];
			Darknet::Layer & B = l.rb_lora_B[k];
			if (A.w != base.w || A.h != base.h)
			{
				resize_convolutional_layer(&A, base.w, base.h);
			}
			if (B.w != A.out_w || B.h != A.out_h)
			{
				resize_convolutional_layer(&B, A.out_w, A.out_h);
			}
			if (B.outputs != base.outputs)
			{
				darknet_fatal_error(DARKNET_LOC, "[recursive_block] Stage 2 LoRA adapter output mismatch: base=%d adapter=%d", base.outputs, B.outputs);
			}
		}
		rb_ensure_lora_buffers_cpu(l);
	}

	void rb_configure_lora_adapters_cpu(Darknet::Layer * l)
	{
		if (l == nullptr || !rb_stage2_lora_enabled(*l) || l->rb_body == nullptr || l->rb_body_count <= 0)
		{
			return;
		}

		if (l->rb_lora_A != nullptr && l->rb_lora_configured_body_count == l->rb_body_count)
		{
			rb_resize_lora_adapters_cpu(*l);
			return;
		}

		rb_free_lora_adapters_cpu(*l);

		int adapter_count = 0;
		for (int j = 0; j < l->rb_body_count; ++j)
		{
			if (rb_lora_body_layer_supported(l->rb_body[j]))
			{
				++adapter_count;
			}
			else if (l->rb_body[j].type == Darknet::ELayerType::CONVOLUTIONAL)
			{
				*cfg_and_state.output << "[recursive_block] Stage 2 Conv-LoRA skips unsupported body conv #" << j
					<< " (groups=" << l->rb_body[j].groups
					<< ", binary=" << l->rb_body[j].binary
					<< ", xnor=" << l->rb_body[j].xnor
					<< ", deform=" << l->rb_body[j].deform
					<< ", antialiasing=" << l->rb_body[j].antialiasing << ")." << std::endl;
			}
		}

		if (adapter_count <= 0)
		{
			*cfg_and_state.output << "[recursive_block] Stage 2 requested, but no eligible convolutional body layers were found. Stage 1 gate/FiLM still runs." << std::endl;
			return;
		}

		l->rb_lora_adapters = adapter_count;
		l->rb_lora_configured_body_count = l->rb_body_count;
		l->rb_lora_diag_offset = rb_lora_diag_offset(*l);
		l->rb_lora_total_rank = adapter_count * l->rb_lora_rank;

		l->rb_lora_A = (Darknet::Layer*)xcalloc(adapter_count, sizeof(Darknet::Layer));
		l->rb_lora_B = (Darknet::Layer*)xcalloc(adapter_count, sizeof(Darknet::Layer));
		l->rb_lora_body_adapter = (int*)xcalloc(l->rb_body_count, sizeof(int));
		l->rb_lora_body_indices = (int*)xcalloc(adapter_count, sizeof(int));
		l->rb_lora_rank_offsets = (int*)xcalloc(adapter_count, sizeof(int));
		l->rb_lora_scaled = (float**)xcalloc(adapter_count, sizeof(float*));
		l->rb_lora_scaled_delta = (float**)xcalloc(adapter_count, sizeof(float*));
		l->rb_lora_scaled_sizes = (int*)xcalloc(adapter_count, sizeof(int));
#ifdef DARKNET_GPU
		l->rb_lora_scaled_gpu = (float**)xcalloc(adapter_count, sizeof(float*));
		l->rb_lora_scaled_delta_gpu = (float**)xcalloc(adapter_count, sizeof(float*));
		l->rb_lora_scaled_gpu_sizes = (int*)xcalloc(adapter_count, sizeof(int));
#endif

		for (int j = 0; j < l->rb_body_count; ++j)
		{
			l->rb_lora_body_adapter[j] = -1;
		}

		int adapter = 0;
		int rank_cursor = 0;
		for (int j = 0; j < l->rb_body_count; ++j)
		{
			Darknet::Layer & base = l->rb_body[j];
			if (!rb_lora_body_layer_supported(base))
			{
				continue;
			}

			l->rb_lora_body_adapter[j] = adapter;
			l->rb_lora_body_indices[adapter] = j;
			l->rb_lora_rank_offsets[adapter] = rank_cursor;

			l->rb_lora_A[adapter] = make_convolutional_layer(
				l->batch, 1, base.h, base.w, base.c,
				l->rb_lora_rank, 1, base.size, base.stride_x, base.stride_y, base.dilation, base.pad,
				LINEAR, 0, 0, 0, base.adam, 0, base.index, 0, nullptr, 0, 0, l->train);

			Darknet::Layer & A = l->rb_lora_A[adapter];
			l->rb_lora_B[adapter] = make_convolutional_layer(
				l->batch, 1, A.out_h, A.out_w, l->rb_lora_rank,
				base.out_c, 1, 1, 1, 1, 1, 0,
				LINEAR, 0, 0, 0, base.adam, 0, base.index, 0, nullptr, 0, 0, l->train);

			Darknet::Layer & B = l->rb_lora_B[adapter];
			fill_cpu(B.nweights, 0.0f, B.weights, 1);
			fill_cpu(B.n, 0.0f, B.biases, 1);
			if (B.weight_updates) fill_cpu(B.nweights, 0.0f, B.weight_updates, 1);
			if (B.bias_updates) fill_cpu(B.n, 0.0f, B.bias_updates, 1);
#ifdef DARKNET_GPU
			if (cfg_and_state.gpu_index >= 0)
			{
				push_convolutional_layer(A);
				push_convolutional_layer(B);
			}
#endif

			if (B.outputs != base.outputs)
			{
				darknet_fatal_error(DARKNET_LOC, "[recursive_block] Stage 2 LoRA adapter output mismatch for body layer %d: base=%d adapter=%d", j, base.outputs, B.outputs);
			}

			rank_cursor += l->rb_lora_rank;
			++adapter;
		}

		l->rb_lora_total_rank = rank_cursor;
		rb_ensure_lora_buffers_cpu(*l);
		*cfg_and_state.output << "[recursive_block] Stage 2 Conv-LoRA: " << l->rb_lora_adapters
			<< " conv adapters, rank=" << l->rb_lora_rank
			<< ", alpha=" << l->rb_lora_alpha
			<< ", total controller diag=" << l->rb_lora_total_rank << std::endl;
	}

	float rb_lora_diag_value_cpu(const Darknet::Layer & l, const int batch_index, const int adapter, const int rank_index)
	{
		if (l.rb_controller_output == nullptr || l.rb_lora_rank_offsets == nullptr)
		{
			return l.rb_lora_diag_init;
		}
		const int controller_out = rb_controller_output_c(l);
		const int idx = l.rb_lora_diag_offset + l.rb_lora_rank_offsets[adapter] + rank_index;
		if (idx < 0 || idx >= controller_out)
		{
			return l.rb_lora_diag_init;
		}
		return rb_clip(l.rb_controller_output[batch_index * controller_out + idx], l.rb_lora_diag_clip);
	}

	// Apply the per-rank diagonal scaling from the controller to Conv_A output.
	// For each rank r and spatial position i:
	//   dst[b, r, i] = src[b, r, i] * diag_t[b, rank_offset + r]
	// diag_t is clipped to ±lora_diag_clip before use.  This is the "dynamic" part
	// of the LoRA adapter — the scalar weight for each rank slot changes every loop
	// based on the current hidden state, whereas Conv_A and Conv_B weights are fixed
	// (shared across loops).
	void rb_lora_scale_forward_cpu(Darknet::Layer & l, const int adapter, const float *src, float *dst)
	{
		if (src == nullptr || dst == nullptr || adapter < 0 || adapter >= l.rb_lora_adapters)
		{
			return;
		}
		const Darknet::Layer & A = l.rb_lora_A[adapter];
		const int rank = l.rb_lora_rank;
		const int area = A.out_w * A.out_h;
		const int plane = rank * area;

		for (int b = 0; b < l.batch; ++b)
		{
			for (int r = 0; r < rank; ++r)
			{
				const float diag = rb_lora_diag_value_cpu(l, b, adapter, r);
				const int base_idx = b * plane + r * area;
				for (int i = 0; i < area; ++i)
				{
					dst[base_idx + i] = src[base_idx + i] * diag;
				}
			}
		}
	}

	void rb_lora_scale_backward_cpu(Darknet::Layer & l, const int adapter, const float *src, const float *scaled_delta, float *src_delta)
	{
		if (src == nullptr || scaled_delta == nullptr || src_delta == nullptr || adapter < 0 || adapter >= l.rb_lora_adapters)
		{
			return;
		}
		const Darknet::Layer & A = l.rb_lora_A[adapter];
		const int rank = l.rb_lora_rank;
		const int area = A.out_w * A.out_h;
		const int plane = rank * area;
		const int controller_out = rb_controller_output_c(l);

		for (int b = 0; b < l.batch; ++b)
		{
			float *ctrl_delta = (l.rb_controller_delta != nullptr) ? (l.rb_controller_delta + b * controller_out) : nullptr;
			const float *ctrl = (l.rb_controller_output != nullptr) ? (l.rb_controller_output + b * controller_out) : nullptr;
			for (int r = 0; r < rank; ++r)
			{
				const int diag_idx = l.rb_lora_diag_offset + l.rb_lora_rank_offsets[adapter] + r;
				const float raw = (ctrl && diag_idx < controller_out) ? ctrl[diag_idx] : l.rb_lora_diag_init;
				const float diag = rb_clip(raw, l.rb_lora_diag_clip);
				const bool diag_pass = (l.rb_lora_diag_clip <= 0.0f || std::fabs(raw) <= l.rb_lora_diag_clip);
				float d_diag = 0.0f;
				const int base_idx = b * plane + r * area;
				for (int i = 0; i < area; ++i)
				{
					const int idx = base_idx + i;
					const float d = scaled_delta[idx];
					src_delta[idx] += d * diag;
					d_diag += d * src[idx];
				}
				if (ctrl_delta && diag_pass && diag_idx < controller_out)
				{
					ctrl_delta[diag_idx] += d_diag;
				}
			}
		}
	}

	// Forward one body layer, optionally wrapping it with a Stage 2 Conv-LoRA adapter.
	// Without LoRA (adapter < 0): normal body forward.
	// With LoRA:
	//   1. Run Conv_base with activation suppressed → raw linear output in bl.output
	//   2. Run Conv_A on the same input → rank-dimensional intermediate in A.output
	//   3. Scale A.output by per-rank diag_t from the controller → scaled
	//   4. Run Conv_B on scaled → low-rank correction in B.output
	//   5. Add (lora_alpha/lora_rank)*B.output to bl.output (residual add)
	//   6. Apply the original activation to bl.output
	// The activation suppression/restore trick avoids a second allocation and keeps
	// the activation gradient in bl.delta correctly aligned with the combined output.
	void rb_forward_body_layer_cpu(Darknet::Layer & l, const int body_index, Darknet::NetworkState & sub)
	{
		Darknet::Layer & bl = l.rb_body[body_index];
		const int adapter = (rb_stage2_lora_enabled(l) && l.rb_lora_body_adapter != nullptr) ? l.rb_lora_body_adapter[body_index] : -1;
		if (adapter < 0)
		{
			bl.forward(bl, sub);
			return;
		}

		Darknet::Layer & A = l.rb_lora_A[adapter];
		Darknet::Layer & B = l.rb_lora_B[adapter];
		float *scaled = l.rb_lora_scaled[adapter];
		const ACTIVATION saved_activation = bl.activation;

		bl.activation = LINEAR;
		bl.forward(bl, sub);
		bl.activation = saved_activation;

		Darknet::NetworkState a_state = sub;
		a_state.input = sub.input;
		A.forward(A, a_state);

		rb_lora_scale_forward_cpu(l, adapter, A.output, scaled);

		Darknet::NetworkState b_state = sub;
		b_state.input = scaled;
		B.forward(B, b_state);

		axpy_cpu(bl.batch * bl.outputs, rb_lora_scaling(l), B.output, 1, bl.output, 1);
		rb_apply_activation_cpu(bl, saved_activation);
	}

	void rb_backward_body_layer_cpu(Darknet::Layer & l, const int body_index, Darknet::NetworkState & sub)
	{
		Darknet::Layer & bl = l.rb_body[body_index];
		const int adapter = (rb_stage2_lora_enabled(l) && l.rb_lora_body_adapter != nullptr) ? l.rb_lora_body_adapter[body_index] : -1;
		if (adapter < 0)
		{
			bl.backward(bl, sub);
			return;
		}

		Darknet::Layer & A = l.rb_lora_A[adapter];
		Darknet::Layer & B = l.rb_lora_B[adapter];
		float *scaled = l.rb_lora_scaled[adapter];
		float *scaled_delta = l.rb_lora_scaled_delta[adapter];
		const int scaled_total = A.batch * A.outputs;
		const int out_total = bl.batch * bl.outputs;

		// Base backward converts bl.delta from dL/d(activated output) to dL/d(preactivation).
		bl.backward(bl, sub);

		if (B.delta == nullptr || A.delta == nullptr || scaled_delta == nullptr)
		{
			return;
		}

		copy_cpu(out_total, bl.delta, 1, B.delta, 1);
		scal_cpu(out_total, rb_lora_scaling(l), B.delta, 1);
		fill_cpu(scaled_total, 0.0f, scaled_delta, 1);

		Darknet::NetworkState b_state = sub;
		b_state.input = scaled;
		b_state.delta = scaled_delta;
		B.backward(B, b_state);

		fill_cpu(scaled_total, 0.0f, A.delta, 1);
		rb_lora_scale_backward_cpu(l, adapter, A.output, scaled_delta, A.delta);

		Darknet::NetworkState a_state = sub;
		a_state.input = sub.input;
		a_state.delta = sub.delta;
		A.backward(A, a_state);
	}

	void rb_update_lora_adapters_cpu(Darknet::Layer & l, int batch, float learning_rate, float momentum, float decay)
	{
		if (!rb_stage2_lora_enabled(l) || l.rb_lora_A == nullptr || l.rb_lora_B == nullptr)
		{
			return;
		}
		for (int k = 0; k < l.rb_lora_adapters; ++k)
		{
			if (l.rb_lora_A[k].update)
			{
				l.rb_lora_A[k].update(l.rb_lora_A[k], batch, learning_rate, momentum, decay);
			}
			if (l.rb_lora_B[k].update)
			{
				l.rb_lora_B[k].update(l.rb_lora_B[k], batch, learning_rate, momentum, decay);
			}
		}
	}

	void rb_free_ouroboros_gpu_arrays(Darknet::Layer * /*l*/)
	{
		// Implemented in recursive_block_layer_gpu.cu for CUDA allocations.
		// CPU-only translation units keep this no-op.
	}

	void rb_allocate_ouroboros_controller_cpu(Darknet::Layer * l)
	{
		if (l == nullptr || !rb_ouroboros_enabled(*l))
		{
			return;
		}

		const int controller_in = rb_controller_input_c(*l);
		const int controller_out = rb_controller_output_c(*l);
		const int desired_weights = controller_in * controller_out;
		const int desired_biases = controller_out;

		l->rb_controller_input_c = controller_in;
		l->rb_controller_output_c = controller_out;

		const bool need_params =
			l->weights == nullptr || l->weight_updates == nullptr || l->biases == nullptr || l->bias_updates == nullptr ||
			l->nweights != desired_weights || l->nbiases != desired_biases;

		if (need_params)
		{
			free(l->weights);
			free(l->weight_updates);
			free(l->biases);
			free(l->bias_updates);

			l->nweights = desired_weights;
			l->nbiases = desired_biases;
			l->weights = (float*)xcalloc(l->nweights, sizeof(float));
			l->weight_updates = (float*)xcalloc(l->nweights, sizeof(float));
			l->biases = (float*)xcalloc(l->nbiases, sizeof(float));
			l->bias_updates = (float*)xcalloc(l->nbiases, sizeof(float));

			const int max_c = rb_controller_max_c(*l);
			for (int c = 0; c < max_c; ++c)
			{
				l->biases[c] = 0.0f;                    // gamma starts at 0 -> multiplier 1.
				l->biases[max_c + c] = l->rb_gate_bias; // gate starts closed-ish, usually sigmoid(-2)=0.119.
			}
			if (rb_stage2_lora_enabled(*l) && l->rb_lora_total_rank > 0)
			{
				l->rb_lora_diag_offset = rb_lora_diag_offset(*l);
				for (int r = 0; r < l->rb_lora_total_rank; ++r)
				{
					l->biases[l->rb_lora_diag_offset + r] = l->rb_lora_diag_init;
				}
			}
		}

		const int max_total = std::max(l->batch * l->inputs, l->batch * l->outputs);
		l->rb_candidate_size = max_total;
		l->rb_controller_input  = (float*)xrealloc(l->rb_controller_input,  l->batch * controller_in  * sizeof(float));
		l->rb_controller_output = (float*)xrealloc(l->rb_controller_output, l->batch * controller_out * sizeof(float));
		l->rb_controller_delta  = (float*)xrealloc(l->rb_controller_delta,  l->batch * controller_out * sizeof(float));
		l->rb_candidate         = (float*)xrealloc(l->rb_candidate,         max_total * sizeof(float));
		l->rb_candidate_delta   = (float*)xrealloc(l->rb_candidate_delta,   max_total * sizeof(float));
	}

	// Pack the controller input from the current hidden state.
	// For each batch item: compute global average pool over every channel of `hidden`
	// and append the normalised step index.  The GAP collapses spatial dimensions so
	// the controller sees a compact per-channel summary without being tied to resolution.
	void rb_controller_pack_input_cpu(Darknet::Layer & l, const float *hidden, const int step, const int loops)
	{
		if (!rb_ouroboros_enabled(l) || hidden == nullptr || l.rb_controller_input == nullptr)
		{
			return;
		}

		const int in_c = rb_controller_input_c(l);
		const int area = l.w * l.h;
		const int plane = area * l.c;
		const float step_value = (loops > 1) ? static_cast<float>(step) / static_cast<float>(loops - 1) : 0.0f;

		for (int b = 0; b < l.batch; ++b)
		{
			float * dst = l.rb_controller_input + b * in_c;
			const float * src = hidden + b * plane;
			for (int c = 0; c < l.c; ++c)
			{
				float sum = 0.0f;
				const float * src_c = src + c * area;
				for (int i = 0; i < area; ++i)
				{
					sum += src_c[i];
				}
				dst[c] = sum / std::max(area, 1);
			}
			dst[l.c] = step_value;
		}
	}

	// One dense (fully-connected) layer with no activation: output = W * input + bias.
	// Weights are stored in l.weights (out_c * in_c), biases in l.biases (out_c).
	// The output feeds directly into rb_ouroboros_mix_forward_cpu which applies
	// the non-linearities (sigmoid for gate, clip for gamma, diag scaling for LoRA).
	void rb_controller_forward_cpu(Darknet::Layer & l)
	{
		if (!rb_ouroboros_enabled(l) || l.weights == nullptr || l.biases == nullptr ||
			l.rb_controller_input == nullptr || l.rb_controller_output == nullptr)
		{
			return;
		}

		const int in_c = rb_controller_input_c(l);
		const int out_c = rb_controller_output_c(l);

		for (int b = 0; b < l.batch; ++b)
		{
			const float * x = l.rb_controller_input + b * in_c;
			float * y = l.rb_controller_output + b * out_c;
			for (int o = 0; o < out_c; ++o)
			{
				float v = l.biases[o];
				const float * w = l.weights + o * in_c;
				for (int i = 0; i < in_c; ++i)
				{
					v += w[i] * x[i];
				}
				y[o] = v;
			}
		}
	}

	inline float rb_old_value_adapted(
		const float *old_hidden,
		const int b,
		const int dc,
		const int dy,
		const int dx,
		const int old_w,
		const int old_h,
		const int old_c,
		const int dst_w,
		const int dst_h)
	{
		if (old_hidden == nullptr || dc >= old_c)
		{
			return 0.0f;
		}
		const int sx = rb_nearest_index(dx, old_w, dst_w);
		const int sy = rb_nearest_index(dy, old_h, dst_h);
		const int old_area = old_w * old_h;
		const int old_plane = old_area * old_c;
		return old_hidden[b * old_plane + dc * old_area + sy * old_w + sx];
	}

	// GRU-like gated mix: combines the body candidate with the previous hidden state
	// using controller-emitted gamma (feature modulation) and gate (blend weight).
	//
	//   gamma = clip(ctrl[dc], gamma_clip)          — feature-wise scale added to 1
	//   gate  = sigmoid(clip(ctrl[max_c+dc], gate_clip))  — blend weight in (0,1)
	//   dst[idx] = gate * candidate[idx] * (1+gamma) + (1-gate) * old_v
	//
	// gate_bias initialises the gate logit to a negative value (typically -2) so that
	// the gate starts near sigmoid(-2)≈0.12 — i.e., the network initially passes through
	// ~88% of the previous hidden state and only 12% of the body output, which stabilises
	// early training.  gamma starts at 0 (bias initialised to 0) so the multiplier is 1.
	//
	// The controller is run once per loop before the body forward, so Stage 2 LoRA
	// diagonal values (also from the controller) are consistent with the gate/gamma.
	void rb_ouroboros_mix_forward_cpu(
		Darknet::Layer & l,
		const float *old_hidden,
		const int step,
		const int loops,
		const float *candidate,
		const int dst_w,
		const int dst_h,
		const int dst_c,
		float *dst)
	{
		(void)step;
		(void)loops;

		const int area = dst_w * dst_h;
		const int dst_plane = area * dst_c;
		const int max_c = rb_controller_max_c(l);
		const int ctrl_out_c = rb_controller_output_c(l);

		#pragma omp parallel for collapse(2) schedule(static)
		for (int b = 0; b < l.batch; ++b)
		{
			for (int dc = 0; dc < dst_c; ++dc)
			{
				const float * ctrl = l.rb_controller_output + b * ctrl_out_c;
				const float gamma_raw = (dc < max_c) ? ctrl[dc] : 0.0f;
				const float gate_raw  = (dc < max_c) ? ctrl[max_c + dc] : l.rb_gate_bias;
				const float gamma = rb_clip(gamma_raw, l.rb_gamma_clip);
				const float gate = rb_sigmoid(rb_clip(gate_raw, l.rb_gate_clip));

				for (int dy = 0; dy < dst_h; ++dy)
				{
					for (int dx = 0; dx < dst_w; ++dx)
					{
						const int idx = b * dst_plane + dc * area + dy * dst_w + dx;
						const float old_v = rb_old_value_adapted(old_hidden, b, dc, dy, dx, l.w, l.h, l.c, dst_w, dst_h);
						const float cand_v = candidate[idx];
						dst[idx] = gate * cand_v * (1.0f + gamma) + (1.0f - gate) * old_v;
					}
				}
			}
		}
	}

	void rb_controller_backward_cpu(Darknet::Layer & l, float *delta_hidden)
	{
		if (!rb_ouroboros_enabled(l) || l.rb_controller_delta == nullptr || l.rb_controller_input == nullptr ||
			l.weights == nullptr || l.weight_updates == nullptr || l.bias_updates == nullptr)
		{
			return;
		}

		const int in_c = rb_controller_input_c(l);
		const int out_c = rb_controller_output_c(l);
		const int area = l.w * l.h;
		const int hidden_plane = area * l.c;

		for (int b = 0; b < l.batch; ++b)
		{
			const float * x = l.rb_controller_input + b * in_c;
			const float * dy = l.rb_controller_delta + b * out_c;

			for (int o = 0; o < out_c; ++o)
			{
				const float d = dy[o];
				l.bias_updates[o] += d;
				float * wu = l.weight_updates + o * in_c;
				for (int i = 0; i < in_c; ++i)
				{
					wu[i] += d * x[i];
				}
			}

			if (delta_hidden != nullptr)
			{
				float * dh = delta_hidden + b * hidden_plane;
				for (int c = 0; c < l.c; ++c)
				{
					float d_gap = 0.0f;
					for (int o = 0; o < out_c; ++o)
					{
						d_gap += dy[o] * l.weights[o * in_c + c];
					}
					const float d_spatial = d_gap / std::max(area, 1);
					float * dh_c = dh + c * area;
					for (int i = 0; i < area; ++i)
					{
						dh_c[i] += d_spatial;
					}
				}
			}
		}
	}

	void rb_ouroboros_mix_backward_cpu(
		Darknet::Layer & l,
		const float *delta_out,
		const int dst_w,
		const int dst_h,
		const int dst_c,
		const float *candidate,
		float *candidate_delta,
		const float *old_hidden,
		float *delta_old)
	{
		if (delta_out == nullptr || candidate == nullptr || candidate_delta == nullptr || old_hidden == nullptr)
		{
			return;
		}

		const int area = dst_w * dst_h;
		const int dst_plane = area * dst_c;
		const int dst_total = l.batch * dst_plane;
		const int old_area = l.w * l.h;
		const int old_plane = old_area * l.c;
		const int max_c = rb_controller_max_c(l);
		const int ctrl_out_c = rb_controller_output_c(l);

		fill_cpu(dst_total, 0.0f, candidate_delta, 1);
		fill_cpu(l.batch * ctrl_out_c, 0.0f, l.rb_controller_delta, 1);

		for (int b = 0; b < l.batch; ++b)
		{
			const float * ctrl = l.rb_controller_output + b * ctrl_out_c;
			float * ctrl_delta = l.rb_controller_delta + b * ctrl_out_c;

			for (int dc = 0; dc < dst_c; ++dc)
			{
				const float gamma_raw = (dc < max_c) ? ctrl[dc] : 0.0f;
				const float gate_raw  = (dc < max_c) ? ctrl[max_c + dc] : l.rb_gate_bias;
				const float gamma = rb_clip(gamma_raw, l.rb_gamma_clip);
				const float gate_pre = rb_clip(gate_raw, l.rb_gate_clip);
				const float gate = rb_sigmoid(gate_pre);
				const float gate_grad = gate * (1.0f - gate);
				const bool gamma_pass = (l.rb_gamma_clip <= 0.0f || std::fabs(gamma_raw) <= l.rb_gamma_clip);
				const bool gate_pass = (l.rb_gate_clip <= 0.0f || std::fabs(gate_raw) <= l.rb_gate_clip);

				for (int dy = 0; dy < dst_h; ++dy)
				{
					const int sy = rb_nearest_index(dy, l.h, dst_h);
					for (int dx = 0; dx < dst_w; ++dx)
					{
						const int idx = b * dst_plane + dc * area + dy * dst_w + dx;
						const float d = delta_out[idx];
						const float cand_v = candidate[idx];
						const float old_v = rb_old_value_adapted(old_hidden, b, dc, dy, dx, l.w, l.h, l.c, dst_w, dst_h);

						candidate_delta[idx] += d * gate * (1.0f + gamma);

						if (delta_old != nullptr && dc < l.c)
						{
							const int sx = rb_nearest_index(dx, l.w, dst_w);
							const int old_idx = b * old_plane + dc * old_area + sy * l.w + sx;
							delta_old[old_idx] += d * (1.0f - gate);
						}

						if (dc < max_c)
						{
							if (gamma_pass)
							{
								ctrl_delta[dc] += d * gate * cand_v;
							}
							if (gate_pass)
							{
								const float modulated = cand_v * (1.0f + gamma);
								ctrl_delta[max_c + dc] += d * (modulated - old_v) * gate_grad;
							}
						}
					}
				}
			}
		}

		// Controller backward is called once after body backward so Stage 2 LoRA
		// diagonal gradients and Stage 1 gate/gamma gradients are accumulated together.
	}

	void rb_update_ouroboros_controller_cpu(Darknet::Layer & l, int batch, float learning_rate, float momentum, float decay)
	{
		if (!rb_ouroboros_enabled(l) || l.weights == nullptr || l.weight_updates == nullptr || l.biases == nullptr || l.bias_updates == nullptr)
		{
			return;
		}

		const float rate = learning_rate * l.learning_rate_scale;
		axpy_cpu(l.nweights, -decay * batch, l.weights, 1, l.weight_updates, 1);
		axpy_cpu(l.nweights, rate / std::max(batch, 1), l.weight_updates, 1, l.weights, 1);
		scal_cpu(l.nweights, momentum, l.weight_updates, 1);

		axpy_cpu(l.nbiases, rate / std::max(batch, 1), l.bias_updates, 1, l.biases, 1);
		scal_cpu(l.nbiases, momentum, l.bias_updates, 1);
	}
}


Darknet::Layer make_recursive_block_layer(int batch, int h, int w, int c, int loops, int shortcut,
	float body_scale, float residual_scale, float injection_scale, int train)
{
	TAT(TATPARMS);

	Darknet::Layer l = {(Darknet::ELayerType)0};
	l.type      = Darknet::ELayerType::RECURSIVE_BLOCK;
	l.batch     = batch;
	l.h         = h;
	l.w         = w;
	l.c         = c;
	l.out_h     = h;
	l.out_w     = w;
	l.out_c     = c;
	l.n         = c;
	l.inputs    = h * w * c;
	l.outputs   = h * w * c;
	l.rb_loops  = std::max(1, loops);
	l.shortcut  = shortcut;
	l.rb_body_scale = body_scale;
	l.rb_residual_scale = residual_scale;
	l.rb_injection_scale = injection_scale;
	// LDT-style lattice projection (disabled by default; configure via set_recursive_block_lattice_projection).
	l.alpha = 0.0f;
	l.beta = 0.1f;
	l.scale = 1.0f;
	l.groups = 0;
	l.train     = train;
	l.learning_rate_scale = 1.0f;

	// Stage 1/2 defaults.  Parser can override these after construction.
	l.rb_ouroboros = 0;
	l.rb_gate_bias = -2.0f;
	l.rb_gamma_clip = 4.0f;
	l.rb_gate_clip = 30.0f;
	l.rb_lora_rank = 0;
	l.rb_lora_alpha = 1.0f;
	l.rb_lora_freeze_base = 0;
	l.rb_lora_diag_clip = 4.0f;
	l.rb_lora_diag_init = 1.0f;
	l.rb_lora_adapters = 0;
	l.rb_lora_total_rank = 0;
	l.rb_lora_diag_offset = 0;
	l.rb_lora_configured_body_count = 0;
	l.rb_lora_body_adapter = nullptr;
	l.rb_lora_body_indices = nullptr;
	l.rb_lora_rank_offsets = nullptr;
	l.rb_lora_A = nullptr;
	l.rb_lora_B = nullptr;
	l.rb_lora_scaled = nullptr;
	l.rb_lora_scaled_delta = nullptr;
	l.rb_lora_scaled_sizes = nullptr;
#ifdef DARKNET_GPU
	l.rb_lora_scaled_gpu = nullptr;
	l.rb_lora_scaled_delta_gpu = nullptr;
	l.rb_lora_scaled_gpu_sizes = nullptr;
#endif
	l.rb_controller_input_c = 0;
	l.rb_controller_output_c = 0;
	l.rb_controller_gpu_input_c = 0;
	l.rb_controller_gpu_output_c = 0;
	l.rb_candidate_size = 0;
	l.rb_candidate_gpu_size = 0;

	l.rb_last_input = (float*)xcalloc(batch * l.inputs, sizeof(float));

	l.output = (float*)xcalloc(batch * l.outputs, sizeof(float));
	if (train)
	{
		l.delta = (float*)xcalloc(batch * l.outputs, sizeof(float));
	}

	l.forward  = forward_recursive_block_layer;
	l.backward = backward_recursive_block_layer;
	l.update   = update_recursive_block_layer;

#ifdef DARKNET_GPU
	l.forward_gpu  = forward_recursive_block_layer_gpu;
	l.backward_gpu = backward_recursive_block_layer_gpu;
	l.update_gpu   = update_recursive_block_layer_gpu;

	if (cfg_and_state.gpu_index >= 0)
	{
		l.rb_last_input_gpu = cuda_make_array(l.rb_last_input, batch * l.inputs);
		l.output_gpu        = cuda_make_array(l.output, batch * l.outputs);
		if (train)
		{
			l.delta_gpu = cuda_make_array(l.delta, batch * l.outputs);
		}
	}
#endif

	return l;
}


void finalize_recursive_block_layer(Darknet::Layer * l)
{
	TAT(TATPARMS);

	if (l == nullptr || l->type != Darknet::ELayerType::RECURSIVE_BLOCK || l->rb_body == nullptr || l->rb_body_count <= 0)
	{
		return;
	}

	const Darknet::Layer & last = l->rb_body[l->rb_body_count - 1];
	const int new_out_w = last.out_w;
	const int new_out_h = last.out_h;
	const int new_out_c = last.out_c;
	const int new_outputs = new_out_w * new_out_h * new_out_c;

	if (new_out_w <= 0 || new_out_h <= 0 || new_out_c <= 0 || new_outputs <= 0)
	{
		darknet_fatal_error(DARKNET_LOC, "[recursive_block] invalid body output dimensions %dx%dx%d", new_out_w, new_out_h, new_out_c);
	}

	const bool dims_changed =
		l->out_w != new_out_w || l->out_h != new_out_h || l->out_c != new_out_c || l->outputs != new_outputs;

	l->out_w = new_out_w;
	l->out_h = new_out_h;
	l->out_c = new_out_c;
	l->n = new_out_c;
	l->outputs = new_outputs;
	l->inputs = l->w * l->h * l->c;

	if (dims_changed)
	{
		l->output = (float*)xrealloc(l->output, l->batch * l->outputs * sizeof(float));
		if (l->delta)
		{
			l->delta = (float*)xrealloc(l->delta, l->batch * l->outputs * sizeof(float));
		}

#ifdef DARKNET_GPU
		if (cfg_and_state.gpu_index >= 0)
		{
			if (l->output_gpu)
			{
				cuda_free(l->output_gpu);
			}
			l->output_gpu = cuda_make_array(l->output, l->batch * l->outputs);

			if (l->delta_gpu)
			{
				cuda_free(l->delta_gpu);
				l->delta_gpu = cuda_make_array(l->delta, l->batch * l->outputs);
			}
		}
#endif
	}

	rb_configure_lora_adapters_cpu(l);
	rb_allocate_ouroboros_controller_cpu(l);
	rb_ensure_lora_buffers_cpu(*l);
}


void set_recursive_block_lattice_projection(Darknet::Layer * l, int candidate_group, float elimination_threshold, float temperature, float mix)
{
	TAT(TATPARMS);

	if (l == nullptr || l->type != Darknet::ELayerType::RECURSIVE_BLOCK)
	{
		darknet_fatal_error(DARKNET_LOC, "set_recursive_block_lattice_projection() called on a non-recursive_block layer");
	}
	if (candidate_group < 0)
	{
		darknet_fatal_error(DARKNET_LOC, "[recursive_block] lattice candidate_group must be >= 0, got %d", candidate_group);
	}
	if (candidate_group > 1 && l->c % candidate_group != 0)
	{
		darknet_fatal_error(DARKNET_LOC, "[recursive_block] lattice candidate_group=%d must divide channels=%d", candidate_group, l->c);
	}

	l->groups = candidate_group;
	l->beta = std::clamp(elimination_threshold, 0.0f, 1.0f);
	l->scale = std::max(temperature, 1e-6f);
	l->alpha = std::clamp(mix, 0.0f, 1.0f);
}


void free_recursive_block_layer(Darknet::Layer & l)
{
	TAT(TATPARMS);

	free(l.rb_last_input);
	l.rb_last_input = nullptr;

	free(l.rb_controller_input);
	free(l.rb_controller_output);
	free(l.rb_controller_delta);
	free(l.rb_candidate);
	free(l.rb_candidate_delta);
	l.rb_controller_input = nullptr;
	l.rb_controller_output = nullptr;
	l.rb_controller_delta = nullptr;
	l.rb_candidate = nullptr;
	l.rb_candidate_delta = nullptr;

	rb_free_lora_adapters_cpu(l);

	// rb_body and rb_body_count are freed by the caller (network cleanup).
	// Standard layer cleanup should free l.weights/l.biases and their updates.

#ifdef DARKNET_GPU
	if (l.rb_last_input_gpu)
	{
		cuda_free(l.rb_last_input_gpu);
		l.rb_last_input_gpu = nullptr;
	}
	if (l.rb_controller_input_gpu)
	{
		cuda_free(l.rb_controller_input_gpu);
		l.rb_controller_input_gpu = nullptr;
	}
	if (l.rb_controller_output_gpu)
	{
		cuda_free(l.rb_controller_output_gpu);
		l.rb_controller_output_gpu = nullptr;
	}
	if (l.rb_controller_delta_gpu)
	{
		cuda_free(l.rb_controller_delta_gpu);
		l.rb_controller_delta_gpu = nullptr;
	}
	if (l.rb_candidate_gpu)
	{
		cuda_free(l.rb_candidate_gpu);
		l.rb_candidate_gpu = nullptr;
	}
	if (l.rb_candidate_delta_gpu)
	{
		cuda_free(l.rb_candidate_delta_gpu);
		l.rb_candidate_delta_gpu = nullptr;
	}
#endif
}


void resize_recursive_block_layer(Darknet::Layer * l, int w, int h)
{
	TAT(TATPARMS);

	l->h       = h;
	l->w       = w;
	l->inputs  = h * w * l->c;
	l->out_h   = h;
	l->out_w   = w;

	rb_resize_body_stack(*l);
	finalize_recursive_block_layer(l);

	l->rb_last_input = (float*)xrealloc(l->rb_last_input, l->batch * l->inputs * sizeof(float));
	l->output        = (float*)xrealloc(l->output,        l->batch * l->outputs * sizeof(float));
	if (l->delta)
	{
		l->delta = (float*)xrealloc(l->delta, l->batch * l->outputs * sizeof(float));
	}

	rb_allocate_ouroboros_controller_cpu(l);

#ifdef DARKNET_GPU
	if (cfg_and_state.gpu_index >= 0)
	{
		cuda_free(l->rb_last_input_gpu);
		l->rb_last_input_gpu = cuda_make_array(l->rb_last_input, l->batch * l->inputs);

		cuda_free(l->output_gpu);
		l->output_gpu = cuda_make_array(l->output, l->batch * l->outputs);

		if (l->delta_gpu)
		{
			cuda_free(l->delta_gpu);
			l->delta_gpu = cuda_make_array(l->delta, l->batch * l->outputs);
		}
	}
#endif
}


// Forward pass.
//
// Loop structure: N = rb_loops iterations.  Iteration 0..N-2 write to `hidden_next`
// (a temporary that becomes h_cur for the next loop).  Iteration N-1 writes to
// `l.output` (the visible output consumed by subsequent layers).
//
// Per-loop sequence:
//   1. Snapshot h_cur → rb_last_input (stable read buffer; prevents aliasing).
//   2. [ouroboros] Run controller: pack GAP(h)+step into controller_input, run
//      the dense layer, store result in controller_output.
//   3. Run each body layer (with optional Stage 2 LoRA wrapping in rb_forward_body_layer_cpu).
//   4. Compute next hidden state:
//        Stage 0: adapt_set(body_out) + residual + injection
//        Stage 1/2: adapt_set(body_out) → candidate; ouroboros_mix(gate+gamma) → next
//   5. [non-final] If l.alpha > 0: run LDT lattice projection on hidden_next.
//        This filters each channel group through a meet with rb_last_input's alive probs,
//        then blends projected with raw hidden_next by lattice_mix (=l.alpha).
//   6. h_cur = hidden_next; repeat.
//
// Backward uses truncated k=1 BPTT: gradients flow only through the FINAL forward
// pass.  The lattice projection has no backward (stop-gradient).
void forward_recursive_block_layer(Darknet::Layer & l, Darknet::NetworkState state)
{
	TAT(TATPARMS);

	if (l.rb_body == nullptr || l.rb_body_count <= 0)
	{
		copy_cpu(l.batch * l.inputs, state.input, 1, l.output, 1);
		return;
	}

	finalize_recursive_block_layer(&l);

	const int N = std::max(1, l.rb_loops);
	const int hidden_size = l.inputs;
	const int hidden_total = l.batch * hidden_size;

	float * h_cur = state.input;
	float * hidden_next = (N > 1) ? (float*)xcalloc(hidden_total, sizeof(float)) : nullptr;

	Darknet::NetworkState sub = state;
	sub.delta = nullptr;

	for (int t = 0; t < N; ++t)
	{
		// Snapshot current hidden state before body runs; rb_last_input is the
		// stable read buffer for both the controller and the residual/gate paths.
		if (h_cur != l.rb_last_input)
		{
			copy_cpu(hidden_total, h_cur, 1, l.rb_last_input, 1);
		}

		if (rb_ouroboros_enabled(l))
		{
			rb_prepare_ouroboros_controller_cpu(l, l.rb_last_input, t, N);
		}

		sub.input = l.rb_last_input;
		for (int j = 0; j < l.rb_body_count; ++j)
		{
			rb_forward_body_layer_cpu(l, j, sub);
			sub.input = l.rb_body[j].output;
		}

		Darknet::Layer & last = l.rb_body[l.rb_body_count - 1];
		float * body_out = last.output;

		if (t == N - 1)
		{
			rb_adapt_set_cpu(body_out, last.out_w, last.out_h, last.out_c, l.batch,
				l.rb_body_scale, rb_ouroboros_enabled(l) ? l.rb_candidate : l.output, l.out_w, l.out_h, l.out_c);

			if (rb_ouroboros_enabled(l))
			{
				rb_ouroboros_mix_forward_cpu(l, l.rb_last_input, t, N, l.rb_candidate, l.out_w, l.out_h, l.out_c, l.output);
			}
			else
			{
				rb_adapt_add_cpu(l.rb_last_input, l.w, l.h, l.c, l.batch,
					l.rb_residual_scale, l.output, l.out_w, l.out_h, l.out_c);
			}

			if (l.shortcut)
			{
				rb_adapt_add_cpu(state.input, l.w, l.h, l.c, l.batch,
					l.rb_injection_scale, l.output, l.out_w, l.out_h, l.out_c);
			}
		}
		else
		{
			rb_adapt_set_cpu(body_out, last.out_w, last.out_h, last.out_c, l.batch,
				l.rb_body_scale, rb_ouroboros_enabled(l) ? l.rb_candidate : hidden_next, l.w, l.h, l.c);

			if (rb_ouroboros_enabled(l))
			{
				rb_ouroboros_mix_forward_cpu(l, l.rb_last_input, t, N, l.rb_candidate, l.w, l.h, l.c, hidden_next);
			}
			else
			{
				rb_adapt_add_cpu(l.rb_last_input, l.w, l.h, l.c, l.batch,
					l.rb_residual_scale, hidden_next, l.w, l.h, l.c);
			}

			if (l.shortcut)
			{
				rb_adapt_add_cpu(state.input, l.w, l.h, l.c, l.batch,
					l.rb_injection_scale, hidden_next, l.w, l.h, l.c);
			}
			// LDT lattice projection: filters hidden_next through a meet with
			// rb_last_input (the pre-body snapshot) interpreted as alive probabilities.
			// Applied only between passes, never to the final visible output, so the
			// YOLO detection head always receives unfiltered activations.
			if (l.alpha > 0.0f)
			{
				rb_lattice_project_cpu(l.rb_last_input, hidden_next, l.batch, l.w, l.h, l.c,
					l.groups, l.beta, l.scale, l.alpha);
			}
			h_cur = hidden_next;
		}
	}

	free(hidden_next);
}


// Backward pass — truncated k=1 BPTT.
//
// Only the FINAL forward pass is differentiated.  Gradients from l.delta are
// propagated through:
//   Stage 0: body_scale * body^T + residual^T + injection^T
//   Stage 1/2: ouroboros_mix_backward (gate, gamma, candidate) + injection^T
//              then body backward in reverse order
//              then controller_backward (weight/bias updates + hidden delta)
//
// The LDT lattice projection has no backward — it is treated as a constant
// forward filter.  This means gradients see the unfiltered recurrence path,
// which is consistent with stop-gradient regularisation.
void backward_recursive_block_layer(Darknet::Layer & l, Darknet::NetworkState state)
{
	TAT(TATPARMS);

	if (l.rb_body == nullptr || l.rb_body_count <= 0)
	{
		if (state.delta && l.delta)
		{
			axpy_cpu(l.batch * l.inputs, 1.0f, l.delta, 1, state.delta, 1);
		}
		return;
	}

	finalize_recursive_block_layer(&l);

	// Truncated k=1: only backprop through the FINAL forward pass.
	// rb_last_input holds h_{T-1} — the input to that final body application.
	Darknet::Layer & last = l.rb_body[l.rb_body_count - 1];

	for (int j = 0; j < l.rb_body_count - 1; ++j)
	{
		if (l.rb_body[j].delta)
		{
			fill_cpu(l.batch * l.rb_body[j].outputs, 0, l.rb_body[j].delta, 1);
		}
	}

	if (last.delta)
	{
		if (rb_ouroboros_enabled(l))
		{
			rb_ouroboros_mix_backward_cpu(l, l.delta, l.out_w, l.out_h, l.out_c,
				l.rb_candidate, l.rb_candidate_delta, l.rb_last_input, state.delta);

			fill_cpu(l.batch * last.outputs, 0.0f, last.delta, 1);
			rb_adapt_backward_add_cpu(l.rb_candidate_delta, l.out_w, l.out_h, l.out_c, l.batch,
				1.0f, last.delta, last.out_w, last.out_h, last.out_c);
		}
		else
		{
			rb_adapt_set_cpu(l.delta, l.out_w, l.out_h, l.out_c, l.batch,
				l.rb_body_scale, last.delta, last.out_w, last.out_h, last.out_c);
		}
	}

	if (state.delta && !rb_ouroboros_enabled(l))
	{
		rb_adapt_backward_add_cpu(l.delta, l.out_w, l.out_h, l.out_c, l.batch,
			l.rb_residual_scale, state.delta, l.w, l.h, l.c);
		if (l.shortcut)
		{
			rb_adapt_backward_add_cpu(l.delta, l.out_w, l.out_h, l.out_c, l.batch,
				l.rb_injection_scale, state.delta, l.w, l.h, l.c);
		}
	}
	else if (state.delta && rb_ouroboros_enabled(l) && l.shortcut)
	{
		rb_adapt_backward_add_cpu(l.delta, l.out_w, l.out_h, l.out_c, l.batch,
			l.rb_injection_scale, state.delta, l.w, l.h, l.c);
	}

	Darknet::NetworkState sub = state;
	for (int j = l.rb_body_count - 1; j >= 0; --j)
	{
		sub.input = (j == 0) ? l.rb_last_input : l.rb_body[j - 1].output;
		sub.delta = (j == 0) ? state.delta      : l.rb_body[j - 1].delta;

		rb_backward_body_layer_cpu(l, j, sub);
	}

	if (rb_ouroboros_enabled(l))
	{
		rb_controller_backward_cpu(l, state.delta);
	}
}


void update_recursive_block_layer(Darknet::Layer & l, int batch, float learning_rate, float momentum, float decay)
{
	TAT(TATPARMS);

	rb_update_ouroboros_controller_cpu(l, batch, learning_rate, momentum, decay);
	rb_update_lora_adapters_cpu(l, batch, learning_rate, momentum, decay);

	for (int j = 0; j < l.rb_body_count; ++j)
	{
		if (l.rb_body[j].update)
		{
			// lora_freeze_base=1: keep base body weights fixed so only the
			// controller, Conv_A, and Conv_B adapter weights update.
			// Useful when loading a Stage 1 checkpoint and fine-tuning adapters.
			if (rb_stage2_lora_enabled(l) && l.rb_lora_freeze_base)
			{
				continue;
			}
			l.rb_body[j].update(l.rb_body[j], batch, learning_rate * l.rb_body[j].learning_rate_scale, momentum, decay);
		}
	}
}
