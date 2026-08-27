/* Darknet/YOLO:  https://codeberg.org/iltranqui/darknet
 *
 * recursive_block_layer_gpu — CUDA counterpart to recursive_block_layer.cpp.
 *
 * GPU kernel strategy summary:
 *
 *   rb_lattice_project_kernel:
 *     One thread per (batch * spatial_position * channel_group).  Each thread
 *     processes all `candidate_group` channels in its group sequentially (two
 *     passes: scan for any_alive + best_k, then write projected values).
 *     Groups up to candidate_group channels without shared memory; works correctly
 *     for any group size but is most efficient when group fits in register file
 *     (tested with group=8, 16).
 *
 *   rb_adapt_add_kernel / rb_adapt_backward_add_kernel:
 *     One thread per output element.  Nearest-neighbour up/down-sample across
 *     spatial dims; channels outside the source range contribute zero.
 *     Backward uses atomicAdd because multiple output positions may map to the
 *     same source position under nearest-neighbour when downsampling.
 *
 *   rb_controller_pack_kernel:
 *     One thread per (batch * controller_in).  Computes global average pool over
 *     the spatial dimensions of the hidden state for each channel, and appends
 *     the normalised step scalar.
 *
 *   rb_controller_forward_kernel:
 *     One thread per (batch * controller_out).  Dense matrix-vector product;
 *     reads the entire controller_input vector per output — fine for small
 *     controller_in (= l.c + 1, typically < 512).
 *
 *   rb_ouroboros_mix_forward_kernel / rb_ouroboros_mix_backward_kernel:
 *     One thread per output element.  Applies per-channel gamma/gate from the
 *     controller output; backward uses atomicAdd into controller_delta because
 *     all spatial positions in the same batch×channel share the same gamma/gate.
 *
 *   rb_lora_scale_forward_kernel / rb_lora_scale_backward_kernel:
 *     One thread per (batch * rank * spatial).  Multiplies each rank slot by
 *     its controller-emitted diagonal value.  Backward accumulates the gradient
 *     into controller_delta via atomicAdd (many spatial positions → one diag slot).
 *
 * The LDT lattice projection has no backward kernel; it is a stop-gradient
 * forward filter.  Gradient flows through rb_last_input_gpu as if the projection
 * were the identity.
 */

#include <algorithm>

#include "darknet_internal.hpp"
#include "recursive_block_layer.hpp"


namespace
{
	static auto & cfg_and_state = Darknet::CfgAndState::get();

	__device__ __forceinline__ int rb_nearest_index_gpu(const int out_idx, const int in_size, const int out_size)
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

	__device__ __forceinline__ float rb_sigmoid_gpu(const float x)
	{
		if (x >= 0.0f)
		{
			const float z = expf(-x);
			return 1.0f / (1.0f + z);
		}
		const float z = expf(x);
		return z / (1.0f + z);
	}

	// Clamped sigmoid: avoids expf overflow at extreme inputs.  expf(30) ≈ 1.07e13
	// is still finite on float32, but the sigmoid result is indistinguishable from 0/1.
	__device__ __forceinline__ float rb_clamped_sigmoid_gpu(const float x)
	{
		const float z = fminf(fmaxf(x, -30.0f), 30.0f);
		return 1.0f / (1.0f + expf(-z));
	}

	// GPU version of rb_alive_probability.  isnan/isinf are CUDA built-ins here.
	// Values already in [0,1] pass through without sigmoid (no temperature distortion).
	__device__ __forceinline__ float rb_alive_probability_gpu(const float value, const float temperature)
	{
		if (isnan(value) || isinf(value))
		{
			return 0.0f;
		}
		if (value >= 0.0f && value <= 1.0f)
		{
			return value;
		}
		return rb_clamped_sigmoid_gpu(value / fmaxf(temperature, 1e-6f));
	}

	// LDT lattice projection kernel.
	// Thread indexing: id = batch_idx * (area * groups_per_cell) + group_idx * area + sp
	//   sp             = spatial position (0 .. w*h-1)
	//   g              = channel group index (0 .. c/candidate_group - 1)
	//   b              = batch index
	// Each thread owns exactly one spatial×group cell and loops over the group's
	// `candidate_group` channels.  Two passes are required:
	//   Pass 1: scan all k in [0, group) to find any_alive and best_k.
	//   Pass 2: write projected values (meet, threshold, rescue, blend).
	// The two-pass design is intentional — the rescue condition (any_alive==false)
	// can only be determined after scanning all candidates.
	__global__ void rb_lattice_project_kernel(
		const int total_groups,
		const float *__restrict__ previous_state,
		float *__restrict__ state,
		const int area,
		const int channels,
		const int candidate_group,
		const float elimination_threshold,
		const float temperature,
		const float mix)
	{
		const int id = blockIdx.x * blockDim.x + threadIdx.x;
		if (id >= total_groups || previous_state == nullptr || state == nullptr || mix <= 0.0f)
		{
			return;
		}

		const int group = (candidate_group > 1 && (channels % candidate_group) == 0) ? candidate_group : 1;
		const int groups_per_cell = channels / group;
		const int sp = id % area;
		const int g = (id / area) % groups_per_cell;
		const int b = id / (area * groups_per_cell);
		const float theta = fminf(fmaxf(elimination_threshold, 0.0f), 1.0f);
		const float tau = fmaxf(temperature, 1e-6f);
		const float blend = fminf(fmaxf(mix, 0.0f), 1.0f);

		float best_alive = -1.0f;
		int best_k = 0;
		bool any_alive = false;

		for (int k = 0; k < group; ++k)
		{
			const int ch = g * group + k;
			const int idx = b * channels * area + ch * area + sp;
			const float prev_alive = rb_alive_probability_gpu(previous_state[idx], tau);
			const float step_alive = rb_alive_probability_gpu(state[idx], tau);
			const float alive = fminf(prev_alive, step_alive);
			if (alive > best_alive)
			{
				best_alive = alive;
				best_k = k;
			}
			if (alive >= theta)
			{
				any_alive = true;
			}
		}

		for (int k = 0; k < group; ++k)
		{
			const int ch = g * group + k;
			const int idx = b * channels * area + ch * area + sp;
			const float prev_alive = rb_alive_probability_gpu(previous_state[idx], tau);
			const float step_alive = rb_alive_probability_gpu(state[idx], tau);
			float projected = fminf(prev_alive, step_alive);
			if (projected < theta)
			{
				projected = 0.0f;
			}
			if (group > 1 && !any_alive && k == best_k && best_alive > 0.0f)
			{
				projected = best_alive;
			}
			state[idx] = (1.0f - blend) * state[idx] + blend * projected;
		}
	}

	// Launch wrapper for rb_lattice_project_kernel.
	// total_groups = batch * (w*h) * (c/group) — one thread per spatial×group cell.
	// When candidate_group does not divide c, or is <= 1, group falls back to 1
	// (each channel is its own group; no rescue fallback, ungrouped cells can die).
	void rb_lattice_project_gpu(
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
		const int total_groups = batch * area * groups_per_cell;
		rb_lattice_project_kernel<<<cuda_gridsize(total_groups), BLOCK, 0, get_cuda_stream()>>>(
			total_groups, previous_state, state, area, c, group,
			elimination_threshold, temperature, mix);
		CHECK_CUDA(cudaPeekAtLastError());
	}

	__device__ __forceinline__ float rb_clip_gpu(const float x, const float clip)
	{
		if (clip <= 0.0f || !isfinite(clip))
		{
			return x;
		}
		return fminf(fmaxf(x, -clip), clip);
	}

	inline int rb_controller_max_c(const Darknet::Layer & l)
	{
		return std::max(l.c, l.out_c);
	}

	inline int rb_controller_input_c(const Darknet::Layer & l)
	{
		return l.c + 1;
	}

	inline bool rb_stage2_lora_enabled(const Darknet::Layer & l)
	{
		return l.rb_ouroboros == 2 && l.rb_lora_rank > 0;
	}

	inline int rb_lora_diag_offset(const Darknet::Layer & l)
	{
		return 2 * rb_controller_max_c(l);
	}

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

	__global__ void rb_adapt_add_kernel(
		const int n,
		const float *src,
		const int src_w,
		const int src_h,
		const int src_c,
		const float alpha,
		float *dst,
		const int dst_w,
		const int dst_h,
		const int dst_c)
	{
		const int id = blockIdx.x * blockDim.x + threadIdx.x;
		if (id >= n || src == nullptr || dst == nullptr || alpha == 0.0f)
		{
			return;
		}

		const int dst_area = dst_w * dst_h;
		const int dst_plane = dst_area * dst_c;
		const int b = id / dst_plane;
		const int r0 = id - b * dst_plane;
		const int dc = r0 / dst_area;
		if (dc >= src_c)
		{
			return;
		}
		const int r1 = r0 - dc * dst_area;
		const int dy = r1 / dst_w;
		const int dx = r1 - dy * dst_w;

		const int sx = rb_nearest_index_gpu(dx, src_w, dst_w);
		const int sy = rb_nearest_index_gpu(dy, src_h, dst_h);
		const int src_area = src_w * src_h;
		const int src_plane = src_area * src_c;
		const int src_index = b * src_plane + dc * src_area + sy * src_w + sx;

		dst[id] += alpha * src[src_index];
	}

	__global__ void rb_adapt_backward_add_kernel(
		const int n,
		const float *delta_dst,
		const int dst_w,
		const int dst_h,
		const int dst_c,
		const float alpha,
		float *delta_src,
		const int src_w,
		const int src_h,
		const int src_c)
	{
		const int id = blockIdx.x * blockDim.x + threadIdx.x;
		if (id >= n || delta_dst == nullptr || delta_src == nullptr || alpha == 0.0f)
		{
			return;
		}

		const int dst_area = dst_w * dst_h;
		const int dst_plane = dst_area * dst_c;
		const int b = id / dst_plane;
		const int r0 = id - b * dst_plane;
		const int dc = r0 / dst_area;
		if (dc >= src_c)
		{
			return;
		}
		const int r1 = r0 - dc * dst_area;
		const int dy = r1 / dst_w;
		const int dx = r1 - dy * dst_w;

		const int sx = rb_nearest_index_gpu(dx, src_w, dst_w);
		const int sy = rb_nearest_index_gpu(dy, src_h, dst_h);
		const int src_area = src_w * src_h;
		const int src_plane = src_area * src_c;
		const int src_index = b * src_plane + dc * src_area + sy * src_w + sx;

		atomicAdd(delta_src + src_index, alpha * delta_dst[id]);
	}

	void rb_adapt_add_gpu(
		const float *src,
		const int src_w,
		const int src_h,
		const int src_c,
		const int batch,
		const float alpha,
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

		const int total = batch * dst_w * dst_h * dst_c;
		rb_adapt_add_kernel<<<cuda_gridsize(total), BLOCK, 0, get_cuda_stream()>>>(
			total, src, src_w, src_h, src_c, alpha, dst, dst_w, dst_h, dst_c);
		CHECK_CUDA(cudaPeekAtLastError());
	}

	void rb_adapt_set_gpu(
		const float *src,
		const int src_w,
		const int src_h,
		const int src_c,
		const int batch,
		const float alpha,
		float *dst,
		const int dst_w,
		const int dst_h,
		const int dst_c)
	{
		const int total = batch * dst_w * dst_h * dst_c;
		fill_ongpu(total, 0.0f, dst, 1);
		rb_adapt_add_gpu(src, src_w, src_h, src_c, batch, alpha, dst, dst_w, dst_h, dst_c);
	}

	void rb_adapt_backward_add_gpu(
		const float *delta_dst,
		const int dst_w,
		const int dst_h,
		const int dst_c,
		const int batch,
		const float alpha,
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

		const int total = batch * dst_w * dst_h * dst_c;
		rb_adapt_backward_add_kernel<<<cuda_gridsize(total), BLOCK, 0, get_cuda_stream()>>>(
			total, delta_dst, dst_w, dst_h, dst_c, alpha, delta_src, src_w, src_h, src_c);
		CHECK_CUDA(cudaPeekAtLastError());
	}


	// Forward declarations used by Stage 2 helpers.
	void rb_controller_forward_gpu(Darknet::Layer & l, const float *hidden, const int step, const int loops);
	void rb_controller_backward_gpu(Darknet::Layer & l, float *delta_hidden);

	void rb_apply_activation_gpu(Darknet::Layer & bl, const ACTIVATION activation)
	{
		const int total = bl.outputs * bl.batch;
		if (activation == LINEAR || total <= 0)
		{
			return;
		}

		if ((activation == SWISH || activation == MISH || activation == HARD_MISH || activation == EML) && bl.activation_input_gpu == nullptr)
		{
			bl.activation_input_gpu = cuda_make_array(nullptr, total);
		}

		if (activation == SWISH) activate_array_swish_ongpu(bl.output_gpu, total, bl.activation_input_gpu, bl.output_gpu);
		else if (activation == MISH) activate_array_mish_ongpu(bl.output_gpu, total, bl.activation_input_gpu, bl.output_gpu);
		else if (activation == HARD_MISH) activate_array_hard_mish_ongpu(bl.output_gpu, total, bl.activation_input_gpu, bl.output_gpu);
		else if (activation == EML) activate_array_eml_ongpu(bl.output_gpu, total, bl.activation_input_gpu, bl.output_gpu);
		else if (activation == NORM_CHAN) activate_array_normalize_channels_ongpu(bl.output_gpu, total, bl.batch, bl.out_c, bl.out_w * bl.out_h, bl.output_gpu);
		else if (activation == NORM_CHAN_SOFTMAX) activate_array_normalize_channels_softmax_ongpu(bl.output_gpu, total, bl.batch, bl.out_c, bl.out_w * bl.out_h, bl.output_gpu, 0);
		else if (activation == NORM_CHAN_SOFTMAX_MAXVAL) activate_array_normalize_channels_softmax_ongpu(bl.output_gpu, total, bl.batch, bl.out_c, bl.out_w * bl.out_h, bl.output_gpu, 1);
		else activate_array_ongpu(bl.output_gpu, total, activation);
	}

	__global__ void rb_lora_scale_forward_kernel(
		const int n,
		const float *src,
		const float *controller_output,
		const int controller_out,
		const int diag_offset,
		const int rank_offset,
		const int rank,
		const int area,
		const float diag_clip,
		const float diag_init,
		float *dst)
	{
		const int id = blockIdx.x * blockDim.x + threadIdx.x;
		if (id >= n || src == nullptr || dst == nullptr)
		{
			return;
		}
		const int plane = rank * area;
		const int b = id / plane;
		const int r0 = id - b * plane;
		const int r = r0 / area;
		float diag = diag_init;
		if (controller_output != nullptr)
		{
			const int ctrl_idx = diag_offset + rank_offset + r;
			if (ctrl_idx >= 0 && ctrl_idx < controller_out)
			{
				diag = rb_clip_gpu(controller_output[b * controller_out + ctrl_idx], diag_clip);
			}
		}
		dst[id] = src[id] * diag;
	}

	__global__ void rb_lora_scale_backward_kernel(
		const int n,
		const float *src,
		const float *scaled_delta,
		float *src_delta,
		const float *controller_output,
		float *controller_delta,
		const int controller_out,
		const int diag_offset,
		const int rank_offset,
		const int rank,
		const int area,
		const float diag_clip,
		const float diag_init)
	{
		const int id = blockIdx.x * blockDim.x + threadIdx.x;
		if (id >= n || src == nullptr || scaled_delta == nullptr || src_delta == nullptr)
		{
			return;
		}
		const int plane = rank * area;
		const int b = id / plane;
		const int r0 = id - b * plane;
		const int r = r0 / area;
		const int ctrl_idx = diag_offset + rank_offset + r;
		float raw = diag_init;
		if (controller_output != nullptr && ctrl_idx >= 0 && ctrl_idx < controller_out)
		{
			raw = controller_output[b * controller_out + ctrl_idx];
		}
		const float diag = rb_clip_gpu(raw, diag_clip);
		src_delta[id] += scaled_delta[id] * diag;

		const bool pass = (diag_clip <= 0.0f || fabsf(raw) <= diag_clip);
		if (controller_delta != nullptr && pass && ctrl_idx >= 0 && ctrl_idx < controller_out)
		{
			atomicAdd(controller_delta + b * controller_out + ctrl_idx, scaled_delta[id] * src[id]);
		}
	}

	void rb_ensure_lora_buffers_gpu(Darknet::Layer & l)
	{
		if (!rb_stage2_lora_enabled(l) || l.rb_lora_adapters <= 0 || l.rb_lora_A == nullptr || cfg_and_state.gpu_index < 0)
		{
			return;
		}
		if (l.rb_lora_scaled_gpu == nullptr)
		{
			l.rb_lora_scaled_gpu = (float**)xcalloc(l.rb_lora_adapters, sizeof(float*));
		}
		if (l.rb_lora_scaled_delta_gpu == nullptr)
		{
			l.rb_lora_scaled_delta_gpu = (float**)xcalloc(l.rb_lora_adapters, sizeof(float*));
		}
		if (l.rb_lora_scaled_gpu_sizes == nullptr)
		{
			l.rb_lora_scaled_gpu_sizes = (int*)xcalloc(l.rb_lora_adapters, sizeof(int));
		}

		for (int k = 0; k < l.rb_lora_adapters; ++k)
		{
			const int needed = l.rb_lora_A[k].batch * l.rb_lora_A[k].outputs;
			if (needed <= 0) continue;
			if (l.rb_lora_scaled_gpu[k] == nullptr || l.rb_lora_scaled_delta_gpu[k] == nullptr || l.rb_lora_scaled_gpu_sizes[k] < needed)
			{
				if (l.rb_lora_scaled_gpu[k]) cuda_free(l.rb_lora_scaled_gpu[k]);
				if (l.rb_lora_scaled_delta_gpu[k]) cuda_free(l.rb_lora_scaled_delta_gpu[k]);
				l.rb_lora_scaled_gpu[k] = cuda_make_array(nullptr, needed);
				l.rb_lora_scaled_delta_gpu[k] = cuda_make_array(nullptr, needed);
				l.rb_lora_scaled_gpu_sizes[k] = needed;
			}
		}
	}

	void rb_lora_scale_forward_gpu(Darknet::Layer & l, const int adapter, const float *src_gpu, float *dst_gpu)
	{
		if (!rb_stage2_lora_enabled(l) || adapter < 0 || adapter >= l.rb_lora_adapters || src_gpu == nullptr || dst_gpu == nullptr)
		{
			return;
		}
		Darknet::Layer & A = l.rb_lora_A[adapter];
		const int rank = l.rb_lora_rank;
		const int area = A.out_w * A.out_h;
		const int total = A.batch * rank * area;
		rb_lora_scale_forward_kernel<<<cuda_gridsize(total), BLOCK, 0, get_cuda_stream()>>>(
			total, src_gpu, l.rb_controller_output_gpu, rb_controller_output_c(l), l.rb_lora_diag_offset,
			l.rb_lora_rank_offsets[adapter], rank, area, l.rb_lora_diag_clip, l.rb_lora_diag_init, dst_gpu);
		CHECK_CUDA(cudaPeekAtLastError());
	}

	void rb_lora_scale_backward_gpu(Darknet::Layer & l, const int adapter, const float *src_gpu, const float *scaled_delta_gpu, float *src_delta_gpu)
	{
		if (!rb_stage2_lora_enabled(l) || adapter < 0 || adapter >= l.rb_lora_adapters || src_gpu == nullptr || scaled_delta_gpu == nullptr || src_delta_gpu == nullptr)
		{
			return;
		}
		Darknet::Layer & A = l.rb_lora_A[adapter];
		const int rank = l.rb_lora_rank;
		const int area = A.out_w * A.out_h;
		const int total = A.batch * rank * area;
		rb_lora_scale_backward_kernel<<<cuda_gridsize(total), BLOCK, 0, get_cuda_stream()>>>(
			total, src_gpu, scaled_delta_gpu, src_delta_gpu,
			l.rb_controller_output_gpu, l.rb_controller_delta_gpu, rb_controller_output_c(l), l.rb_lora_diag_offset,
			l.rb_lora_rank_offsets[adapter], rank, area, l.rb_lora_diag_clip, l.rb_lora_diag_init);
		CHECK_CUDA(cudaPeekAtLastError());
	}

	void rb_prepare_ouroboros_controller_gpu(Darknet::Layer & l, const float *hidden, const int step, const int loops)
	{
		if (!rb_ouroboros_enabled(l))
		{
			return;
		}
		rb_controller_forward_gpu(l, hidden, step, loops);
	}

	void rb_forward_body_layer_gpu(Darknet::Layer & l, const int body_index, Darknet::NetworkState & sub)
	{
		Darknet::Layer & bl = l.rb_body[body_index];
		const int adapter = (rb_stage2_lora_enabled(l) && l.rb_lora_body_adapter != nullptr) ? l.rb_lora_body_adapter[body_index] : -1;
		if (adapter < 0)
		{
			if (bl.forward_gpu) bl.forward_gpu(bl, sub);
			return;
		}

		rb_ensure_lora_buffers_gpu(l);
		Darknet::Layer & A = l.rb_lora_A[adapter];
		Darknet::Layer & B = l.rb_lora_B[adapter];
		float *scaled_gpu = l.rb_lora_scaled_gpu[adapter];
		const ACTIVATION saved_activation = bl.activation;

		bl.activation = LINEAR;
		bl.forward_gpu(bl, sub);
		bl.activation = saved_activation;

		Darknet::NetworkState a_state = sub;
		a_state.input = sub.input;
		A.forward_gpu(A, a_state);

		rb_lora_scale_forward_gpu(l, adapter, A.output_gpu, scaled_gpu);

		Darknet::NetworkState b_state = sub;
		b_state.input = scaled_gpu;
		B.forward_gpu(B, b_state);

		axpy_ongpu(bl.batch * bl.outputs, rb_lora_scaling(l), B.output_gpu, 1, bl.output_gpu, 1);
		rb_apply_activation_gpu(bl, saved_activation);
	}

	void rb_backward_body_layer_gpu(Darknet::Layer & l, const int body_index, Darknet::NetworkState & sub)
	{
		Darknet::Layer & bl = l.rb_body[body_index];
		const int adapter = (rb_stage2_lora_enabled(l) && l.rb_lora_body_adapter != nullptr) ? l.rb_lora_body_adapter[body_index] : -1;
		if (adapter < 0)
		{
			if (bl.backward_gpu) bl.backward_gpu(bl, sub);
			return;
		}

		rb_ensure_lora_buffers_gpu(l);
		Darknet::Layer & A = l.rb_lora_A[adapter];
		Darknet::Layer & B = l.rb_lora_B[adapter];
		float *scaled_gpu = l.rb_lora_scaled_gpu[adapter];
		float *scaled_delta_gpu = l.rb_lora_scaled_delta_gpu[adapter];
		const int scaled_total = A.batch * A.outputs;
		const int out_total = bl.batch * bl.outputs;

		// Base backward converts bl.delta_gpu from dL/d(activated output) to dL/d(preactivation).
		bl.backward_gpu(bl, sub);

		if (B.delta_gpu == nullptr || A.delta_gpu == nullptr || scaled_delta_gpu == nullptr)
		{
			return;
		}

		simple_copy_ongpu(out_total, bl.delta_gpu, B.delta_gpu);
		scal_ongpu(out_total, rb_lora_scaling(l), B.delta_gpu, 1);
		fill_ongpu(scaled_total, 0.0f, scaled_delta_gpu, 1);

		Darknet::NetworkState b_state = sub;
		b_state.input = scaled_gpu;
		b_state.delta = scaled_delta_gpu;
		B.backward_gpu(B, b_state);

		fill_ongpu(scaled_total, 0.0f, A.delta_gpu, 1);
		rb_lora_scale_backward_gpu(l, adapter, A.output_gpu, scaled_delta_gpu, A.delta_gpu);

		Darknet::NetworkState a_state = sub;
		a_state.input = sub.input;
		a_state.delta = sub.delta;
		A.backward_gpu(A, a_state);
	}

	void rb_update_lora_adapters_gpu(Darknet::Layer & l, int batch, float learning_rate, float momentum, float decay, float loss_scale)
	{
		if (!rb_stage2_lora_enabled(l) || l.rb_lora_A == nullptr || l.rb_lora_B == nullptr)
		{
			return;
		}
		for (int k = 0; k < l.rb_lora_adapters; ++k)
		{
			if (l.rb_lora_A[k].update_gpu)
			{
				l.rb_lora_A[k].update_gpu(l.rb_lora_A[k], batch, learning_rate, momentum, decay, loss_scale);
			}
			if (l.rb_lora_B[k].update_gpu)
			{
				l.rb_lora_B[k].update_gpu(l.rb_lora_B[k], batch, learning_rate, momentum, decay, loss_scale);
			}
			if (l.rb_lora_A[k].weights_gpu && l.rb_lora_A[k].nweights > 0) fix_nan_and_inf(l.rb_lora_A[k].weights_gpu, l.rb_lora_A[k].nweights);
			if (l.rb_lora_B[k].weights_gpu && l.rb_lora_B[k].nweights > 0) fix_nan_and_inf(l.rb_lora_B[k].weights_gpu, l.rb_lora_B[k].nweights);
		}
	}

	// Pack controller input: one thread per (batch * controller_in) element.
	// ch < c  → global average pool of channel ch over all w*h spatial positions.
	// ch == c → normalised step scalar (step / (loops-1)).
	// ch > c  → zero (padding guard; should not be reached if controller_in = c+1).
	__global__ void rb_controller_pack_kernel(
		const int n,
		const float *hidden,
		const int batch,
		const int w,
		const int h,
		const int c,
		const int controller_in,
		const float step_value,
		float *controller_input)
	{
		const int id = blockIdx.x * blockDim.x + threadIdx.x;
		if (id >= n || hidden == nullptr || controller_input == nullptr)
		{
			return;
		}

		const int ch = id % controller_in;
		const int b = id / controller_in;
		if (b >= batch)
		{
			return;
		}

		if (ch == c)
		{
			controller_input[id] = step_value;
			return;
		}
		if (ch > c)
		{
			controller_input[id] = 0.0f;
			return;
		}

		const int area = w * h;
		const int plane = area * c;
		const float *src = hidden + b * plane + ch * area;
		float sum = 0.0f;
		for (int i = 0; i < area; ++i)
		{
			sum += src[i];
		}
		controller_input[id] = sum / ((area > 1) ? area : 1);
	}

	// Dense linear layer forward: output[b,o] = bias[o] + sum_i(W[o,i] * input[b,i]).
	// One thread per (batch * controller_out).  Reads the full controller_in vector
	// per thread — efficient for small controller_in (l.c+1, typically < 512).
	__global__ void rb_controller_forward_kernel(
		const int n,
		const float *controller_input,
		const float *weights,
		const float *biases,
		const int controller_in,
		const int controller_out,
		float *controller_output)
	{
		const int id = blockIdx.x * blockDim.x + threadIdx.x;
		if (id >= n || controller_input == nullptr || weights == nullptr || biases == nullptr || controller_output == nullptr)
		{
			return;
		}

		const int o = id % controller_out;
		const int b = id / controller_out;
		const float *x = controller_input + b * controller_in;
		const float *w = weights + o * controller_in;
		float v = biases[o];
		for (int i = 0; i < controller_in; ++i)
		{
			v += w[i] * x[i];
		}
		controller_output[id] = v;
	}

	// Per-element FiLM/gate mix: one thread per output element.
	// Reads gamma/gate from controller_output[b, dc] and [b, max_c+dc].
	// old_hidden is nearest-neighbour sampled when dims differ (body changes size).
	// Formula: dst = sigmoid(gate_clip(gate)) * candidate * (1+clip(gamma)) + (1-gate) * old_v
	__global__ void rb_ouroboros_mix_forward_kernel(
		const int n,
		const float *candidate,
		const float *old_hidden,
		const float *controller_output,
		const int old_w,
		const int old_h,
		const int old_c,
		const int dst_w,
		const int dst_h,
		const int dst_c,
		const int max_c,
		const int controller_out,
		const float gate_bias,
		const float gamma_clip,
		const float gate_clip,
		float *dst)
	{
		const int id = blockIdx.x * blockDim.x + threadIdx.x;
		if (id >= n || candidate == nullptr || old_hidden == nullptr || controller_output == nullptr || dst == nullptr)
		{
			return;
		}

		const int dst_area = dst_w * dst_h;
		const int dst_plane = dst_area * dst_c;
		const int b = id / dst_plane;
		const int r0 = id - b * dst_plane;
		const int dc = r0 / dst_area;
		const int r1 = r0 - dc * dst_area;
		const int dy = r1 / dst_w;
		const int dx = r1 - dy * dst_w;

		const float *ctrl = controller_output + b * controller_out;
		const float gamma_raw = (dc < max_c) ? ctrl[dc] : 0.0f;
		const float gate_raw = (dc < max_c) ? ctrl[max_c + dc] : gate_bias;
		const float gamma = rb_clip_gpu(gamma_raw, gamma_clip);
		const float gate = rb_sigmoid_gpu(rb_clip_gpu(gate_raw, gate_clip));

		float old_v = 0.0f;
		if (dc < old_c)
		{
			const int sx = rb_nearest_index_gpu(dx, old_w, dst_w);
			const int sy = rb_nearest_index_gpu(dy, old_h, dst_h);
			const int old_area = old_w * old_h;
			const int old_plane = old_area * old_c;
			old_v = old_hidden[b * old_plane + dc * old_area + sy * old_w + sx];
		}

		const float cand_v = candidate[id];
		dst[id] = gate * cand_v * (1.0f + gamma) + (1.0f - gate) * old_v;
	}

	// Backward of the FiLM/gate mix: one thread per output element.
	// Gradients from delta_out flow to:
	//   candidate_delta[idx]    += d * gate * (1+gamma)
	//   delta_old[old_idx]      += d * (1-gate)            [atomicAdd, NN sample aliasing]
	//   ctrl_delta[b, dc]       += d * gate * cand_v        [gamma grad, atomicAdd]
	//   ctrl_delta[b, max_c+dc] += d * (modulated-old_v) * gate*(1-gate)  [gate grad]
	// atomicAdd is required for ctrl_delta because many spatial positions map to the
	// same (batch, channel) controller output slot.
	__global__ void rb_ouroboros_mix_backward_kernel(
		const int n,
		const float *delta_out,
		const float *candidate,
		float *candidate_delta,
		const float *old_hidden,
		float *delta_old,
		const float *controller_output,
		float *controller_delta,
		const int old_w,
		const int old_h,
		const int old_c,
		const int dst_w,
		const int dst_h,
		const int dst_c,
		const int max_c,
		const int controller_out,
		const float gate_bias,
		const float gamma_clip,
		const float gate_clip)
	{
		const int id = blockIdx.x * blockDim.x + threadIdx.x;
		if (id >= n || delta_out == nullptr || candidate == nullptr || candidate_delta == nullptr || old_hidden == nullptr || controller_output == nullptr || controller_delta == nullptr)
		{
			return;
		}

		const int dst_area = dst_w * dst_h;
		const int dst_plane = dst_area * dst_c;
		const int b = id / dst_plane;
		const int r0 = id - b * dst_plane;
		const int dc = r0 / dst_area;
		const int r1 = r0 - dc * dst_area;
		const int dy = r1 / dst_w;
		const int dx = r1 - dy * dst_w;

		const float *ctrl = controller_output + b * controller_out;
		float *ctrl_delta = controller_delta + b * controller_out;
		const float gamma_raw = (dc < max_c) ? ctrl[dc] : 0.0f;
		const float gate_raw = (dc < max_c) ? ctrl[max_c + dc] : gate_bias;
		const float gamma = rb_clip_gpu(gamma_raw, gamma_clip);
		const float gate_pre = rb_clip_gpu(gate_raw, gate_clip);
		const float gate = rb_sigmoid_gpu(gate_pre);
		const float gate_grad = gate * (1.0f - gate);
		const bool gamma_pass = (gamma_clip <= 0.0f || fabsf(gamma_raw) <= gamma_clip);
		const bool gate_pass = (gate_clip <= 0.0f || fabsf(gate_raw) <= gate_clip);

		float old_v = 0.0f;
		int old_idx = -1;
		if (dc < old_c)
		{
			const int sx = rb_nearest_index_gpu(dx, old_w, dst_w);
			const int sy = rb_nearest_index_gpu(dy, old_h, dst_h);
			const int old_area = old_w * old_h;
			const int old_plane = old_area * old_c;
			old_idx = b * old_plane + dc * old_area + sy * old_w + sx;
			old_v = old_hidden[old_idx];
		}

		const float d = delta_out[id];
		const float cand_v = candidate[id];
		candidate_delta[id] += d * gate * (1.0f + gamma);

		if (delta_old != nullptr && old_idx >= 0)
		{
			atomicAdd(delta_old + old_idx, d * (1.0f - gate));
		}

		if (dc < max_c)
		{
			if (gamma_pass)
			{
				atomicAdd(ctrl_delta + dc, d * gate * cand_v);
			}
			if (gate_pass)
			{
				const float modulated = cand_v * (1.0f + gamma);
				atomicAdd(ctrl_delta + max_c + dc, d * (modulated - old_v) * gate_grad);
			}
		}
	}

	// Accumulate weight gradients: weight_updates[o,i] += sum_b(ctrl_delta[b,o] * input[b,i]).
	// One thread per weight (controller_in * controller_out).  Loops over batch —
	// acceptable because controller_in is small; a reduction kernel would add complexity.
	__global__ void rb_controller_weight_backward_kernel(
		const int n,
		const float *controller_input,
		const float *controller_delta,
		const int batch,
		const int controller_in,
		const int controller_out,
		float *weight_updates)
	{
		const int id = blockIdx.x * blockDim.x + threadIdx.x;
		if (id >= n || controller_input == nullptr || controller_delta == nullptr || weight_updates == nullptr)
		{
			return;
		}

		const int i = id % controller_in;
		const int o = id / controller_in;
		float sum = 0.0f;
		for (int b = 0; b < batch; ++b)
		{
			sum += controller_delta[b * controller_out + o] * controller_input[b * controller_in + i];
		}
		weight_updates[id] += sum;
	}

	// Accumulate bias gradients: bias_updates[o] += sum_b(ctrl_delta[b,o]).
	// One thread per output unit.
	__global__ void rb_controller_bias_backward_kernel(
		const int n,
		const float *controller_delta,
		const int batch,
		const int controller_out,
		float *bias_updates)
	{
		const int o = blockIdx.x * blockDim.x + threadIdx.x;
		if (o >= n || controller_delta == nullptr || bias_updates == nullptr)
		{
			return;
		}

		float sum = 0.0f;
		for (int b = 0; b < batch; ++b)
		{
			sum += controller_delta[b * controller_out + o];
		}
		bias_updates[o] += sum;
	}

	// Back-propagate controller gradient into the hidden state.
	// For each spatial element in hidden, the gradient of the GAP operation is
	// 1/area (uniform redistribution of the gap gradient back to every spatial position).
	// delta_hidden[b,ch,sp] += (sum_o ctrl_delta[b,o] * W[o,ch]) / area
	__global__ void rb_controller_hidden_backward_kernel(
		const int n,
		const float *controller_delta,
		const float *weights,
		const int w,
		const int h,
		const int c,
		const int controller_in,
		const int controller_out,
		float *delta_hidden)
	{
		const int id = blockIdx.x * blockDim.x + threadIdx.x;
		if (id >= n || controller_delta == nullptr || weights == nullptr || delta_hidden == nullptr)
		{
			return;
		}

		const int area = w * h;
		const int plane = area * c;
		const int b = id / plane;
		const int r0 = id - b * plane;
		const int ch = r0 / area;

		float d_gap = 0.0f;
		const float *dy = controller_delta + b * controller_out;
		for (int o = 0; o < controller_out; ++o)
		{
			d_gap += dy[o] * weights[o * controller_in + ch];
		}
		delta_hidden[id] += d_gap / ((area > 1) ? area : 1);
	}

	void rb_controller_forward_gpu(Darknet::Layer & l, const float *hidden, const int step, const int loops)
	{
		const int controller_in = rb_controller_input_c(l);
		const int controller_out = rb_controller_output_c(l);
		const float step_value = (loops > 1) ? static_cast<float>(step) / static_cast<float>(loops - 1) : 0.0f;

		const int input_total = l.batch * controller_in;
		rb_controller_pack_kernel<<<cuda_gridsize(input_total), BLOCK, 0, get_cuda_stream()>>>(
			input_total, hidden, l.batch, l.w, l.h, l.c, controller_in, step_value, l.rb_controller_input_gpu);
		CHECK_CUDA(cudaPeekAtLastError());

		const int output_total = l.batch * controller_out;
		rb_controller_forward_kernel<<<cuda_gridsize(output_total), BLOCK, 0, get_cuda_stream()>>>(
			output_total, l.rb_controller_input_gpu, l.weights_gpu, l.biases_gpu, controller_in, controller_out, l.rb_controller_output_gpu);
		CHECK_CUDA(cudaPeekAtLastError());
	}

	void rb_ouroboros_mix_forward_gpu(
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
		// Controller output is prepared once at the start of each recurrence step so
		// Stage 2 Conv-LoRA adapters and the Stage 1 gate use the same controller pass.
		const int total = l.batch * dst_w * dst_h * dst_c;
		rb_ouroboros_mix_forward_kernel<<<cuda_gridsize(total), BLOCK, 0, get_cuda_stream()>>>(
			total, candidate, old_hidden, l.rb_controller_output_gpu,
			l.w, l.h, l.c, dst_w, dst_h, dst_c,
			rb_controller_max_c(l), rb_controller_output_c(l),
			l.rb_gate_bias, l.rb_gamma_clip, l.rb_gate_clip, dst);
		CHECK_CUDA(cudaPeekAtLastError());
	}

	void rb_ouroboros_mix_backward_gpu(
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
		const int dst_total = l.batch * dst_w * dst_h * dst_c;
		const int ctrl_total = l.batch * rb_controller_output_c(l);
		fill_ongpu(dst_total, 0.0f, candidate_delta, 1);
		fill_ongpu(ctrl_total, 0.0f, l.rb_controller_delta_gpu, 1);

		rb_ouroboros_mix_backward_kernel<<<cuda_gridsize(dst_total), BLOCK, 0, get_cuda_stream()>>>(
			dst_total, delta_out, candidate, candidate_delta, old_hidden, delta_old,
			l.rb_controller_output_gpu, l.rb_controller_delta_gpu,
			l.w, l.h, l.c, dst_w, dst_h, dst_c,
			rb_controller_max_c(l), rb_controller_output_c(l),
			l.rb_gate_bias, l.rb_gamma_clip, l.rb_gate_clip);
		CHECK_CUDA(cudaPeekAtLastError());

		// Controller backward is called once after body backward so Stage 2 LoRA
		// diagonal gradients and Stage 1 gate/gamma gradients are accumulated together.
	}

	void rb_controller_backward_gpu(Darknet::Layer & l, float *delta_hidden)
	{
		if (!rb_ouroboros_enabled(l) || l.rb_controller_delta_gpu == nullptr || l.weights_gpu == nullptr ||
			l.weight_updates_gpu == nullptr || l.bias_updates_gpu == nullptr)
		{
			return;
		}

		const int controller_in = rb_controller_input_c(l);
		const int controller_out = rb_controller_output_c(l);
		const int nweights = controller_in * controller_out;

		rb_controller_weight_backward_kernel<<<cuda_gridsize(nweights), BLOCK, 0, get_cuda_stream()>>>(
			nweights, l.rb_controller_input_gpu, l.rb_controller_delta_gpu, l.batch, controller_in, controller_out, l.weight_updates_gpu);
		CHECK_CUDA(cudaPeekAtLastError());

		rb_controller_bias_backward_kernel<<<cuda_gridsize(controller_out), BLOCK, 0, get_cuda_stream()>>>(
			controller_out, l.rb_controller_delta_gpu, l.batch, controller_out, l.bias_updates_gpu);
		CHECK_CUDA(cudaPeekAtLastError());

		if (delta_hidden != nullptr)
		{
			const int hidden_total = l.batch * l.inputs;
			rb_controller_hidden_backward_kernel<<<cuda_gridsize(hidden_total), BLOCK, 0, get_cuda_stream()>>>(
				hidden_total, l.rb_controller_delta_gpu, l.weights_gpu,
				l.w, l.h, l.c, controller_in, controller_out, delta_hidden);
			CHECK_CUDA(cudaPeekAtLastError());
		}
	}

	void rb_ensure_ouroboros_gpu(Darknet::Layer & l)
	{
		if (!rb_ouroboros_enabled(l) || cfg_and_state.gpu_index < 0)
		{
			return;
		}

		const int controller_in = rb_controller_input_c(l);
		const int controller_out = rb_controller_output_c(l);
		const int max_total = std::max(l.batch * l.inputs, l.batch * l.outputs);

		const bool controller_changed =
			l.rb_controller_input_gpu == nullptr ||
			l.rb_controller_output_gpu == nullptr ||
			l.rb_controller_delta_gpu == nullptr ||
			l.rb_candidate_gpu == nullptr ||
			l.rb_candidate_delta_gpu == nullptr ||
			l.rb_candidate_gpu_size < max_total ||
			l.weights_gpu == nullptr || l.weight_updates_gpu == nullptr || l.biases_gpu == nullptr || l.bias_updates_gpu == nullptr ||
			l.rb_controller_gpu_input_c != controller_in || l.rb_controller_gpu_output_c != controller_out;

		l.rb_controller_gpu_input_c = controller_in;
		l.rb_controller_gpu_output_c = controller_out;
		l.rb_candidate_gpu_size = max_total;

		rb_ensure_lora_buffers_gpu(l);

		if (controller_changed)
		{
			if (l.weights_gpu) cuda_free(l.weights_gpu);
			if (l.weight_updates_gpu) cuda_free(l.weight_updates_gpu);
			if (l.biases_gpu) cuda_free(l.biases_gpu);
			if (l.bias_updates_gpu) cuda_free(l.bias_updates_gpu);
			if (l.rb_controller_input_gpu) cuda_free(l.rb_controller_input_gpu);
			if (l.rb_controller_output_gpu) cuda_free(l.rb_controller_output_gpu);
			if (l.rb_controller_delta_gpu) cuda_free(l.rb_controller_delta_gpu);
			if (l.rb_candidate_gpu) cuda_free(l.rb_candidate_gpu);
			if (l.rb_candidate_delta_gpu) cuda_free(l.rb_candidate_delta_gpu);

			l.weights_gpu = cuda_make_array(l.weights, l.nweights);
			l.weight_updates_gpu = cuda_make_array(l.weight_updates, l.nweights);
			l.biases_gpu = cuda_make_array(l.biases, l.nbiases);
			l.bias_updates_gpu = cuda_make_array(l.bias_updates, l.nbiases);
			l.rb_controller_input_gpu = cuda_make_array(nullptr, l.batch * controller_in);
			l.rb_controller_output_gpu = cuda_make_array(nullptr, l.batch * controller_out);
			l.rb_controller_delta_gpu = cuda_make_array(nullptr, l.batch * controller_out);
			l.rb_candidate_gpu = cuda_make_array(nullptr, max_total);
			l.rb_candidate_delta_gpu = cuda_make_array(nullptr, max_total);
		}
	}
}


void push_recursive_block_layer(Darknet::Layer & l)
{
	TAT(TATPARMS);

	cuda_push_array(l.rb_last_input_gpu, l.rb_last_input, l.batch * l.inputs);

	if (rb_ouroboros_enabled(l))
	{
		rb_ensure_ouroboros_gpu(l);
		cuda_push_array(l.weights_gpu, l.weights, l.nweights);
		cuda_push_array(l.weight_updates_gpu, l.weight_updates, l.nweights);
		cuda_push_array(l.biases_gpu, l.biases, l.nbiases);
		cuda_push_array(l.bias_updates_gpu, l.bias_updates, l.nbiases);
	}

	for (int j = 0; j < l.rb_body_count; ++j)
	{
		Darknet::Layer & bl = l.rb_body[j];
		if (bl.type == Darknet::ELayerType::CONVOLUTIONAL)
		{
			push_convolutional_layer(bl);
		}
	}

	if (rb_stage2_lora_enabled(l) && l.rb_lora_A && l.rb_lora_B)
	{
		for (int k = 0; k < l.rb_lora_adapters; ++k)
		{
			push_convolutional_layer(l.rb_lora_A[k]);
			push_convolutional_layer(l.rb_lora_B[k]);
		}
	}
}


void pull_recursive_block_layer(Darknet::Layer & l)
{
	TAT(TATPARMS);

	cuda_pull_array(l.rb_last_input_gpu, l.rb_last_input, l.batch * l.inputs);

	if (rb_ouroboros_enabled(l))
	{
		rb_ensure_ouroboros_gpu(l);
		cuda_pull_array(l.weights_gpu, l.weights, l.nweights);
		cuda_pull_array(l.weight_updates_gpu, l.weight_updates, l.nweights);
		cuda_pull_array(l.biases_gpu, l.biases, l.nbiases);
		cuda_pull_array(l.bias_updates_gpu, l.bias_updates, l.nbiases);
	}

	for (int j = 0; j < l.rb_body_count; ++j)
	{
		Darknet::Layer & bl = l.rb_body[j];
		if (bl.type == Darknet::ELayerType::CONVOLUTIONAL)
		{
			pull_convolutional_layer(bl);
		}
	}

	if (rb_stage2_lora_enabled(l) && l.rb_lora_A && l.rb_lora_B)
	{
		for (int k = 0; k < l.rb_lora_adapters; ++k)
		{
			pull_convolutional_layer(l.rb_lora_A[k]);
			pull_convolutional_layer(l.rb_lora_B[k]);
		}
	}
}


void forward_recursive_block_layer_gpu(Darknet::Layer & l, Darknet::NetworkState state)
{
	TAT(TATPARMS);

	if (l.rb_body == nullptr || l.rb_body_count <= 0)
	{
		simple_copy_ongpu(l.batch * l.inputs, state.input, l.output_gpu);
		return;
	}

	finalize_recursive_block_layer(&l);
	rb_ensure_ouroboros_gpu(l);

	const int N = std::max(1, l.rb_loops);
	const int hidden_total = l.batch * l.inputs;
	const int out_total = l.batch * l.outputs;

	float * h_cur_gpu = state.input;
	float * hidden_next_gpu = (N > 1) ? cuda_make_array(NULL, hidden_total) : nullptr;
	if (state.net.try_fix_nan)
	{
		fix_nan_and_inf(h_cur_gpu, hidden_total);
	}

	Darknet::NetworkState sub = state;
	sub.delta = nullptr;

	for (int t = 0; t < N; ++t)
	{
		if (h_cur_gpu != l.rb_last_input_gpu)
		{
			simple_copy_ongpu(hidden_total, h_cur_gpu, l.rb_last_input_gpu);
		}
		if (state.net.try_fix_nan)
		{
			fix_nan_and_inf(l.rb_last_input_gpu, hidden_total);
		}

		if (rb_ouroboros_enabled(l))
		{
			rb_prepare_ouroboros_controller_gpu(l, l.rb_last_input_gpu, t, N);
		}

		sub.input = l.rb_last_input_gpu;
		for (int j = 0; j < l.rb_body_count; ++j)
		{
			rb_forward_body_layer_gpu(l, j, sub);
			if (state.net.try_fix_nan)
			{
				fix_nan_and_inf(l.rb_body[j].output_gpu, l.rb_body[j].outputs * l.rb_body[j].batch);
			}
			sub.input = l.rb_body[j].output_gpu;
		}

		Darknet::Layer & last = l.rb_body[l.rb_body_count - 1];
		float * body_out_gpu = last.output_gpu;

		if (t == N - 1)
		{
			rb_adapt_set_gpu(body_out_gpu, last.out_w, last.out_h, last.out_c, l.batch,
				l.rb_body_scale, rb_ouroboros_enabled(l) ? l.rb_candidate_gpu : l.output_gpu, l.out_w, l.out_h, l.out_c);
			if (rb_ouroboros_enabled(l))
			{
				rb_ouroboros_mix_forward_gpu(l, l.rb_last_input_gpu, t, N, l.rb_candidate_gpu, l.out_w, l.out_h, l.out_c, l.output_gpu);
			}
			else
			{
				rb_adapt_add_gpu(l.rb_last_input_gpu, l.w, l.h, l.c, l.batch,
					l.rb_residual_scale, l.output_gpu, l.out_w, l.out_h, l.out_c);
			}
			if (l.shortcut)
			{
				rb_adapt_add_gpu(state.input, l.w, l.h, l.c, l.batch,
					l.rb_injection_scale, l.output_gpu, l.out_w, l.out_h, l.out_c);
			}
			if (state.net.try_fix_nan)
			{
				fix_nan_and_inf(l.output_gpu, out_total);
			}
		}
		else
		{
			rb_adapt_set_gpu(body_out_gpu, last.out_w, last.out_h, last.out_c, l.batch,
				l.rb_body_scale, rb_ouroboros_enabled(l) ? l.rb_candidate_gpu : hidden_next_gpu, l.w, l.h, l.c);
			if (rb_ouroboros_enabled(l))
			{
				rb_ouroboros_mix_forward_gpu(l, l.rb_last_input_gpu, t, N, l.rb_candidate_gpu, l.w, l.h, l.c, hidden_next_gpu);
			}
			else
			{
				rb_adapt_add_gpu(l.rb_last_input_gpu, l.w, l.h, l.c, l.batch,
					l.rb_residual_scale, hidden_next_gpu, l.w, l.h, l.c);
			}
			if (l.shortcut)
			{
				rb_adapt_add_gpu(state.input, l.w, l.h, l.c, l.batch,
					l.rb_injection_scale, hidden_next_gpu, l.w, l.h, l.c);
			}
			// LDT lattice projection: forward-only, no gradient.
			// Uses rb_last_input_gpu (pre-body snapshot) as previous_state.
			if (l.alpha > 0.0f)
			{
				rb_lattice_project_gpu(l.rb_last_input_gpu, hidden_next_gpu, l.batch, l.w, l.h, l.c,
					l.groups, l.beta, l.scale, l.alpha);
			}
			if (state.net.try_fix_nan)
			{
				fix_nan_and_inf(hidden_next_gpu, hidden_total);
			}
			h_cur_gpu = hidden_next_gpu;
		}
	}

	if (hidden_next_gpu)
	{
		cuda_free(hidden_next_gpu);
	}
}


void backward_recursive_block_layer_gpu(Darknet::Layer & l, Darknet::NetworkState state)
{
	TAT(TATPARMS);

	if (l.rb_body == nullptr || l.rb_body_count <= 0)
	{
		if (state.delta && l.delta_gpu)
		{
			axpy_ongpu(l.batch * l.inputs, 1.0f, l.delta_gpu, 1, state.delta, 1);
		}
		return;
	}

	finalize_recursive_block_layer(&l);
	rb_ensure_ouroboros_gpu(l);

	if (state.net.try_fix_nan)
	{
		fix_nan_and_inf(l.delta_gpu, l.batch * l.outputs);
	}

	for (int j = 0; j < l.rb_body_count - 1; ++j)
	{
		if (l.rb_body[j].delta_gpu)
		{
			fill_ongpu(l.batch * l.rb_body[j].outputs, 0, l.rb_body[j].delta_gpu, 1);
		}
	}

	Darknet::Layer & last = l.rb_body[l.rb_body_count - 1];
	if (last.delta_gpu)
	{
		if (rb_ouroboros_enabled(l))
		{
			rb_ouroboros_mix_backward_gpu(l, l.delta_gpu, l.out_w, l.out_h, l.out_c,
				l.rb_candidate_gpu, l.rb_candidate_delta_gpu, l.rb_last_input_gpu, state.delta);
			fill_ongpu(l.batch * last.outputs, 0.0f, last.delta_gpu, 1);
			rb_adapt_backward_add_gpu(l.rb_candidate_delta_gpu, l.out_w, l.out_h, l.out_c, l.batch,
				1.0f, last.delta_gpu, last.out_w, last.out_h, last.out_c);
		}
		else
		{
			rb_adapt_set_gpu(l.delta_gpu, l.out_w, l.out_h, l.out_c, l.batch,
				l.rb_body_scale, last.delta_gpu, last.out_w, last.out_h, last.out_c);
		}
		if (state.net.try_fix_nan)
		{
			fix_nan_and_inf(last.delta_gpu, l.batch * last.outputs);
		}
	}

	if (state.delta && !rb_ouroboros_enabled(l))
	{
		rb_adapt_backward_add_gpu(l.delta_gpu, l.out_w, l.out_h, l.out_c, l.batch,
			l.rb_residual_scale, state.delta, l.w, l.h, l.c);
		if (l.shortcut)
		{
			rb_adapt_backward_add_gpu(l.delta_gpu, l.out_w, l.out_h, l.out_c, l.batch,
				l.rb_injection_scale, state.delta, l.w, l.h, l.c);
		}
		if (state.net.try_fix_nan)
		{
			fix_nan_and_inf(state.delta, l.batch * l.inputs);
		}
	}
	else if (state.delta && rb_ouroboros_enabled(l) && l.shortcut)
	{
		rb_adapt_backward_add_gpu(l.delta_gpu, l.out_w, l.out_h, l.out_c, l.batch,
			l.rb_injection_scale, state.delta, l.w, l.h, l.c);
		if (state.net.try_fix_nan)
		{
			fix_nan_and_inf(state.delta, l.batch * l.inputs);
		}
	}

	Darknet::NetworkState sub = state;
	for (int j = l.rb_body_count - 1; j >= 0; --j)
	{
		sub.input = (j == 0) ? l.rb_last_input_gpu : l.rb_body[j - 1].output_gpu;
		sub.delta = (j == 0) ? state.delta          : l.rb_body[j - 1].delta_gpu;

		rb_backward_body_layer_gpu(l, j, sub);
		if (state.net.try_fix_nan)
		{
			if (l.rb_body[j].weight_updates_gpu && l.rb_body[j].nweights > 0)
			{
				fix_nan_and_inf(l.rb_body[j].weight_updates_gpu, l.rb_body[j].nweights);
			}
			if (l.rb_body[j].bias_updates_gpu && l.rb_body[j].n > 0)
			{
				fix_nan_and_inf(l.rb_body[j].bias_updates_gpu, l.rb_body[j].n);
			}
			if (sub.delta)
			{
				const int sub_delta_size = (j == 0) ? l.batch * l.inputs : l.batch * l.rb_body[j - 1].outputs;
				fix_nan_and_inf(sub.delta, sub_delta_size);
			}
		}
	}

	if (rb_ouroboros_enabled(l))
	{
		rb_controller_backward_gpu(l, state.delta);
	}
}


void update_recursive_block_layer_gpu(Darknet::Layer & l, int batch, float learning_rate, float momentum, float decay, float loss_scale)
{
	TAT(TATPARMS);

	if (rb_ouroboros_enabled(l))
	{
		rb_ensure_ouroboros_gpu(l);
		const float rate = learning_rate * l.learning_rate_scale;
		axpy_ongpu(l.nweights, -decay * batch * loss_scale, l.weights_gpu, 1, l.weight_updates_gpu, 1);
		axpy_ongpu(l.nweights, rate / std::max(batch, 1), l.weight_updates_gpu, 1, l.weights_gpu, 1);
		scal_ongpu(l.nweights, momentum, l.weight_updates_gpu, 1);

		axpy_ongpu(l.nbiases, rate / std::max(batch, 1), l.bias_updates_gpu, 1, l.biases_gpu, 1);
		scal_ongpu(l.nbiases, momentum, l.bias_updates_gpu, 1);

		if (l.weights_gpu && l.nweights > 0)
		{
			fix_nan_and_inf(l.weights_gpu, l.nweights);
		}
		if (l.biases_gpu && l.nbiases > 0)
		{
			fix_nan_and_inf(l.biases_gpu, l.nbiases);
		}
	}

	rb_update_lora_adapters_gpu(l, batch, learning_rate, momentum, decay, loss_scale);

	for (int j = 0; j < l.rb_body_count; ++j)
	{
		if (l.rb_body[j].update_gpu)
		{
			if (rb_stage2_lora_enabled(l) && l.rb_lora_freeze_base)
			{
				continue;
			}
			l.rb_body[j].update_gpu(l.rb_body[j], batch,
				learning_rate * l.rb_body[j].learning_rate_scale, momentum, decay, loss_scale);
			if (l.rb_body[j].weights_gpu && l.rb_body[j].nweights > 0)
			{
				fix_nan_and_inf(l.rb_body[j].weights_gpu, l.rb_body[j].nweights);
			}
			if (l.rb_body[j].biases_gpu && l.rb_body[j].n > 0)
			{
				fix_nan_and_inf(l.rb_body[j].biases_gpu, l.rb_body[j].n);
			}
		}
	}
}
