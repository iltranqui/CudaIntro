#include "darknet_internal.hpp"

#ifdef DARKNET_GPU

#include "clifford_layer.hpp"
#include "activations.hpp"
#include "blas.hpp"
#include "convolutional_layer.hpp"
#include "dark_cuda.hpp"
#include "gemm.hpp"

namespace
{
	constexpr float k_clifford_ln_eps = 1e-5f;
	constexpr int k_clifford_inner_path = 0;
	constexpr int k_clifford_wedge_path = 1;

	inline bool clifford_has_local(const Darknet::Layer & l)
	{
		return l.cli_gffn_mode != 1;
	}

	inline bool clifford_has_global(const Darknet::Layer & l)
	{
		return l.cli_gffn_mode != 0;
	}

	inline bool clifford_has_higher_local(const Darknet::Layer & l)
	{
		return clifford_has_local(l) && l.cli_higher_mode == 1 && l.cli_interaction_mode != 0;
	}

	inline size_t clifford_tensor_count(const Darknet::Layer & l)
	{
		return static_cast<size_t>(l.batch) * l.c * l.h * l.w;
	}

	inline size_t clifford_token_count(const Darknet::Layer & l)
	{
		return static_cast<size_t>(l.batch) * l.h * l.w;
	}

	inline int clifford_inner_num_shifts(const Darknet::Layer & l)
	{
		return (l.cli_shifts_inner_gpu && l.cli_num_shifts_inner > 0) ? l.cli_num_shifts_inner : l.cli_num_shifts;
	}

	inline int * clifford_inner_shifts_gpu(const Darknet::Layer & l)
	{
		return (l.cli_shifts_inner_gpu && l.cli_num_shifts_inner > 0) ? l.cli_shifts_inner_gpu : l.cli_shifts_gpu;
	}

	inline __host__ __device__ int clifford_wrap_channel(int channel, int shift, int C)
	{
		channel += shift;
		return (channel >= C) ? (channel - C) : channel;
	}

	inline __host__ __device__ int clifford_wedge_base_channel(int shift_idx, int num_inner_shifts, int C)
	{
		const int block = (shift_idx < num_inner_shifts) ? (2 * shift_idx) : (num_inner_shifts + shift_idx);
		return block * C;
	}

	inline __host__ __device__ int clifford_inner_base_channel(int shift_idx, int num_wedge_shifts, int C)
	{
		const int block = (shift_idx < num_wedge_shifts) ? (2 * shift_idx + 1) : (num_wedge_shifts + shift_idx);
		return block * C;
	}

	inline float * make_temp_float(size_t count)
	{
		return (count == 0) ? nullptr : cuda_make_array(nullptr, count);
	}

	inline void free_temp_float(float *& ptr)
	{
		if (ptr)
		{
			cuda_free(ptr);
			ptr = nullptr;
		}
	}

	inline __device__ float clifford_logistic(float x)
	{
		return 1.0f / (1.0f + expf(-x));
	}

	inline __device__ float clifford_silu_grad(float x)
	{
		const float sig = clifford_logistic(x);
		return sig + x * sig * (1.0f - sig);
	}

	static void linear_1x1_forward_gpu(const float *input_gpu, float *output_gpu,
		float *weights_gpu, float *bias_gpu, int B, int C_in, int C_out, int HW)
	{
		for (int b = 0; b < B; ++b)
		{
			gemm_ongpu(0, 0, C_out, HW, C_in, 1.0f,
				weights_gpu, C_in,
				const_cast<float*>(input_gpu) + b * C_in * HW, HW,
				0.0f,
				output_gpu + b * C_out * HW, HW);
		}

		if (bias_gpu)
		{
			add_bias_gpu(output_gpu, bias_gpu, B, C_out, HW);
		}
	}

	static void linear_1x1_backward_gpu(const float *input_gpu, const float *dout_gpu, float *weights_gpu,
		float *weight_updates_gpu, float *bias_updates_gpu, float *dinput_gpu,
		int B, int C_in, int C_out, int HW)
	{
		if (bias_updates_gpu)
		{
			backward_bias_gpu(bias_updates_gpu, const_cast<float*>(dout_gpu), B, C_out, HW);
		}

		for (int b = 0; b < B; ++b)
		{
			gemm_ongpu(0, 1, C_out, C_in, HW, 1.0f,
				const_cast<float*>(dout_gpu) + b * C_out * HW, HW,
				const_cast<float*>(input_gpu) + b * C_in * HW, HW,
				1.0f,
				weight_updates_gpu, C_in);

			if (dinput_gpu)
			{
				gemm_ongpu(1, 0, C_in, HW, C_out, 1.0f,
					weights_gpu, C_in,
					const_cast<float*>(dout_gpu) + b * C_out * HW, HW,
					1.0f,
					dinput_gpu + b * C_in * HW, HW);
			}
		}
	}

	static void gate_linear_forward_gpu(const float *xln_gpu, const float *gfeat_gpu, float *out_gpu,
		float *weights_gpu, float *bias_gpu, int B, int C, int HW)
	{
		for (int b = 0; b < B; ++b)
		{
			float *dst = out_gpu + b * C * HW;
			gemm_ongpu(0, 0, C, HW, C, 1.0f,
				weights_gpu, 2 * C,
				const_cast<float*>(xln_gpu) + b * C * HW, HW,
				0.0f,
				dst, HW);
			gemm_ongpu(0, 0, C, HW, C, 1.0f,
				weights_gpu + C, 2 * C,
				const_cast<float*>(gfeat_gpu) + b * C * HW, HW,
				1.0f,
				dst, HW);
		}

		if (bias_gpu)
		{
			add_bias_gpu(out_gpu, bias_gpu, B, C, HW);
		}
	}

	static void gate_linear_backward_gpu(const float *xln_gpu, const float *gfeat_gpu, const float *dout_gpu,
		float *weights_gpu, float *weight_updates_gpu, float *bias_updates_gpu,
		float *d_xln_gpu, float *d_gfeat_gpu, int B, int C, int HW)
	{
		if (bias_updates_gpu)
		{
			backward_bias_gpu(bias_updates_gpu, const_cast<float*>(dout_gpu), B, C, HW);
		}

		for (int b = 0; b < B; ++b)
		{
			const float *dout_b = dout_gpu + b * C * HW;
			const float *xln_b = xln_gpu + b * C * HW;
			const float *gfeat_b = gfeat_gpu + b * C * HW;

			gemm_ongpu(0, 1, C, C, HW, 1.0f,
				const_cast<float*>(dout_b), HW,
				const_cast<float*>(xln_b), HW,
				1.0f,
				weight_updates_gpu, 2 * C);
			gemm_ongpu(0, 1, C, C, HW, 1.0f,
				const_cast<float*>(dout_b), HW,
				const_cast<float*>(gfeat_b), HW,
				1.0f,
				weight_updates_gpu + C, 2 * C);

			if (d_xln_gpu)
			{
				gemm_ongpu(1, 0, C, HW, C, 1.0f,
					weights_gpu, 2 * C,
					const_cast<float*>(dout_b), HW,
					1.0f,
					d_xln_gpu + b * C * HW, HW);
			}

			if (d_gfeat_gpu)
			{
				gemm_ongpu(1, 0, C, HW, C, 1.0f,
					weights_gpu + C, 2 * C,
					const_cast<float*>(dout_b), HW,
					1.0f,
					d_gfeat_gpu + b * C * HW, HW);
			}
		}
	}

	static void forward_clifford_dwconv_stack_gpu(Darknet::Layer & l, float *input_gpu, float *output_gpu, Darknet::NetworkState state)
	{
		const size_t total = clifford_tensor_count(l);
		float *current_input = input_gpu;

		Darknet::NetworkState sub_state = {0};
		sub_state.workspace = state.workspace;
		sub_state.train = state.train;
		sub_state.index = state.index;
		sub_state.net = state.net;

		for (int i = 0; i < l.cli_num_dwconv; ++i)
		{
			sub_state.input = current_input;
			forward_convolutional_layer_gpu(l.cli_dwconv[i], sub_state);
			current_input = l.cli_dwconv[i].output_gpu;
		}

		simple_copy_ongpu(static_cast<int>(total), current_input, output_gpu);
	}

	static void backward_clifford_dwconv_stack_gpu(Darknet::Layer & l, float *input_gpu, float *d_output_gpu,
		float *d_input_gpu, Darknet::NetworkState state)
	{
		const size_t total = clifford_tensor_count(l);
		simple_copy_ongpu(static_cast<int>(total), d_output_gpu, l.cli_dwconv[l.cli_num_dwconv - 1].delta_gpu);

		Darknet::NetworkState sub_state = {0};
		sub_state.workspace = state.workspace;
		sub_state.train = state.train;
		sub_state.index = state.index;
		sub_state.net = state.net;

		for (int i = l.cli_num_dwconv - 1; i >= 0; --i)
		{
			sub_state.input = (i == 0) ? input_gpu : l.cli_dwconv[i - 1].output_gpu;
			sub_state.delta = (i == 0) ? d_input_gpu : l.cli_dwconv[i - 1].delta_gpu;
			if (sub_state.delta)
			{
				fill_ongpu(static_cast<int>(total), 0.0f, sub_state.delta, 1);
			}
			backward_convolutional_layer_gpu(l.cli_dwconv[i], sub_state);
		}
	}

	static void scale_clifford_updates_for_loss(float *updates_gpu, int count, float loss_scale)
	{
		if (updates_gpu == nullptr || count <= 0 || loss_scale == 0.0f || loss_scale == 1.0f)
		{
			return;
		}

		scal_ongpu(count, 1.0f / loss_scale, updates_gpu, 1);
	}

	static void update_clifford_weights_gpu(float *weights_gpu, float *updates_gpu, int count,
		int batch, float lr, float momentum, float decay)
	{
		if (weights_gpu == nullptr || updates_gpu == nullptr || count <= 0)
		{
			return;
		}

		axpy_ongpu(count, -decay * batch, weights_gpu, 1, updates_gpu, 1);
		axpy_ongpu(count, lr / batch, updates_gpu, 1, weights_gpu, 1);
		scal_ongpu(count, momentum, updates_gpu, 1);
	}

	static void update_clifford_bias_like_gpu(float *values_gpu, float *updates_gpu, int count,
		int batch, float lr, float momentum)
	{
		if (values_gpu == nullptr || updates_gpu == nullptr || count <= 0)
		{
			return;
		}

		axpy_ongpu(count, lr / batch, updates_gpu, 1, values_gpu, 1);
		scal_ongpu(count, momentum, updates_gpu, 1);
	}
}

// ═══════════════════════════════════════════════════════════════════════════════
// Clifford Layer GPU Kernels
// ═══════════════════════════════════════════════════════════════════════════════

// LayerNorm over the channel axis while keeping tensors in NCHW layout.
// One thread handles one spatial token (b, s), loops over C, and writes the
// normalized result back into strided NCHW storage.
__global__ void clifford_layernorm_forward_kernel(
	const float *input, float *output, float *mean, float *var, float *xhat,
	const float *gamma, const float *beta, int total_tokens, int C, int HW)
{
	int token = blockIdx.x * blockDim.x + threadIdx.x;
	if (token >= total_tokens) return;

	int b = token / HW;
	int s = token % HW;

	float token_mean = 0.0f;
	for (int c = 0; c < C; ++c)
	{
		token_mean += input[(b * C + c) * HW + s];
	}
	token_mean /= C;
	mean[token] = token_mean;

	float token_var = 0.0f;
	for (int c = 0; c < C; ++c)
	{
		float diff = input[(b * C + c) * HW + s] - token_mean;
		token_var += diff * diff;
	}
	token_var /= C;
	var[token] = token_var;

	float inv_std = 1.0f / sqrtf(token_var + k_clifford_ln_eps);
	for (int c = 0; c < C; ++c)
	{
		int idx = (b * C + c) * HW + s;
		float normalized = (input[idx] - token_mean) * inv_std;
		xhat[idx] = normalized;
		output[idx] = normalized * gamma[c] + beta[c];
	}
}

// Backward pass for NCHW LayerNorm with per-channel affine parameters.
// Each thread again owns one token, computes its local reduction, and uses
// atomics only for gamma/beta gradients that are shared across tokens.
__global__ void clifford_layernorm_backward_kernel(
	const float *dout, const float *xhat, const float *var, const float *gamma,
	float *dx, float *dgamma, float *dbeta, int total_tokens, int C, int HW)
{
	int token = blockIdx.x * blockDim.x + threadIdx.x;
	if (token >= total_tokens) return;

	int b = token / HW;
	int s = token % HW;
	float inv_std = 1.0f / sqrtf(var[token] + k_clifford_ln_eps);

	float sum_dxhat = 0.0f;
	float dot_dxhat_xhat = 0.0f;
	for (int c = 0; c < C; ++c)
	{
		int idx = (b * C + c) * HW + s;
		float dxhat = dout[idx] * gamma[c];
		sum_dxhat += dxhat;
		dot_dxhat_xhat += dxhat * xhat[idx];
		atomicAdd(&dgamma[c], dout[idx] * xhat[idx]);
		atomicAdd(&dbeta[c], dout[idx]);
	}

	if (dx == nullptr)
	{
		return;
	}

	for (int c = 0; c < C; ++c)
	{
		int idx = (b * C + c) * HW + s;
		float dxhat = dout[idx] * gamma[c];
		dx[idx] = inv_std * (dxhat - (sum_dxhat + xhat[idx] * dot_dxhat_xhat) / C);
	}
}

// Elementwise SiLU used for the contextual branch and the base residual term.
// The pre-activation is preserved elsewhere, so this kernel only writes the
// activated output.
__global__ void clifford_silu_forward_kernel(const float *input, float *output, int total)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= total) return;

	float x = input[idx];
	output[idx] = x * clifford_logistic(x);
}

// In-place SiLU backward when the forward pre-activation is available.
// `delta[idx]` is multiplied by d/dx SiLU(x).
__global__ void clifford_silu_backward_inplace_kernel(const float *pre_activation, float *delta, int total)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= total) return;

	delta[idx] *= clifford_silu_grad(pre_activation[idx]);
}

// Optional differential context mode subtracts the deterministic stream from
// the contextual stream, matching the Laplacian-style `ctx_mode=diff`.
__global__ void clifford_diff_context_kernel(float *z_ctx, const float *z_det, int total)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= total) return;

	z_ctx[idx] -= z_det[idx];
}

// Backward companion for `ctx_mode=diff`: `z_ctx = z_ctx_pre - z_det` means
// the deterministic stream receives `-d_z_ctx`.
__global__ void clifford_diff_context_backward_kernel(float *d_z_det, const float *d_z_ctx, int total)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= total) return;

	d_z_det[idx] -= d_z_ctx[idx];
}

// Rolling local geometric interaction from Algorithm 1.
// One thread owns one `(b, shift, c, s)` tuple and writes the selected
// wedge/inner features into the concatenated raw feature tensor.
__global__ void clifford_rolling_forward_local_kernel(
	const float *z_det, const float *z_ctx, float *g_raw,
	int B, int C, int HW, const int *shifts, int num_shifts,
	int interaction_kind, int num_other_shifts, int raw_C, int full_mode)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int total = B * num_shifts * C * HW;
	if (idx >= total) return;

	int s = idx % HW;
	int rest = idx / HW;
	int c = rest % C;
	rest /= C;
	int shift_idx = rest % num_shifts;
	int b = rest / num_shifts;

	int shift = shifts[shift_idx];
	int c_roll = clifford_wrap_channel(c, shift, C);
	int base_channel = full_mode
		? ((interaction_kind == k_clifford_wedge_path)
			? clifford_wedge_base_channel(shift_idx, num_other_shifts, C)
			: clifford_inner_base_channel(shift_idx, num_other_shifts, C))
		: (shift_idx * C);

	int cur_idx = (b * C + c) * HW + s;
	int roll_idx = (b * C + c_roll) * HW + s;

	float prod = z_det[cur_idx] * z_ctx[roll_idx];
	int out_idx = (b * raw_C + base_channel + c) * HW + s;
	if (interaction_kind == k_clifford_inner_path)
	{
		g_raw[out_idx] = prod * clifford_logistic(prod);
	}
	else
	{
		g_raw[out_idx] = prod - z_ctx[cur_idx] * z_det[roll_idx];
	}
}

// Backward for local rolling interaction.
// Atomics are required because multiple shifted feature paths accumulate into
// the same deterministic/context tensor elements.
__global__ void clifford_rolling_backward_local_kernel(
	const float *z_det, const float *z_ctx, const float *d_g_raw,
	float *d_z_det, float *d_z_ctx,
	int B, int C, int HW, const int *shifts, int num_shifts,
	int interaction_kind, int num_other_shifts, int raw_C, int full_mode)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int total = B * num_shifts * C * HW;
	if (idx >= total) return;

	int s = idx % HW;
	int rest = idx / HW;
	int c = rest % C;
	rest /= C;
	int shift_idx = rest % num_shifts;
	int b = rest / num_shifts;

	int shift = shifts[shift_idx];
	int c_roll = clifford_wrap_channel(c, shift, C);
	int base_channel = full_mode
		? ((interaction_kind == k_clifford_wedge_path)
			? clifford_wedge_base_channel(shift_idx, num_other_shifts, C)
			: clifford_inner_base_channel(shift_idx, num_other_shifts, C))
		: (shift_idx * C);

	int cur_idx = (b * C + c) * HW + s;
	int roll_idx = (b * C + c_roll) * HW + s;

	float prod = z_det[cur_idx] * z_ctx[roll_idx];
	int g_idx = (b * raw_C + base_channel + c) * HW + s;
	if (interaction_kind == k_clifford_inner_path)
	{
		float sig = clifford_logistic(prod);
		float d_inner = d_g_raw[g_idx];
		float d_prod = d_inner * (sig + prod * sig * (1.0f - sig));
		atomicAdd(&d_z_det[cur_idx], d_prod * z_ctx[roll_idx]);
		atomicAdd(&d_z_ctx[roll_idx], d_prod * z_det[cur_idx]);
	}
	else
	{
		float d_wedge = d_g_raw[g_idx];
		atomicAdd(&d_z_det[cur_idx], d_wedge * z_ctx[roll_idx]);
		atomicAdd(&d_z_ctx[roll_idx], d_wedge * z_det[cur_idx]);
		atomicAdd(&d_z_ctx[cur_idx], -d_wedge * z_det[roll_idx]);
		atomicAdd(&d_z_det[roll_idx], -d_wedge * z_ctx[cur_idx]);
	}
}

// Global mean-field variant of the rolling interaction.
// The context vector is `[B, C]` and is broadcast across all spatial tokens.
__global__ void clifford_rolling_forward_global_kernel(
	const float *state_stream, const float *global_ctx, float *g_raw,
	int B, int C, int HW, const int *shifts, int num_shifts,
	int interaction_kind, int num_other_shifts, int raw_C, int full_mode)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int total = B * num_shifts * C * HW;
	if (idx >= total) return;

	int s = idx % HW;
	int rest = idx / HW;
	int c = rest % C;
	rest /= C;
	int shift_idx = rest % num_shifts;
	int b = rest / num_shifts;

	int shift = shifts[shift_idx];
	int c_roll = clifford_wrap_channel(c, shift, C);
	int base_channel = full_mode
		? ((interaction_kind == k_clifford_wedge_path)
			? clifford_wedge_base_channel(shift_idx, num_other_shifts, C)
			: clifford_inner_base_channel(shift_idx, num_other_shifts, C))
		: (shift_idx * C);

	float g_cur = global_ctx[b * C + c];
	float g_roll = global_ctx[b * C + c_roll];
	int cur_idx = (b * C + c) * HW + s;
	int roll_idx = (b * C + c_roll) * HW + s;

	float prod = state_stream[cur_idx] * g_roll;
	int out_idx = (b * raw_C + base_channel + c) * HW + s;
	if (interaction_kind == k_clifford_inner_path)
	{
		g_raw[out_idx] = prod * clifford_logistic(prod);
	}
	else
	{
		g_raw[out_idx] = prod - g_cur * state_stream[roll_idx];
	}
}

// Backward for the global mean-field interaction.
// Atomics accumulate both into the left-hand state stream and the pooled context
// vector because many spatial tokens share the same global channel entry.
__global__ void clifford_rolling_backward_global_kernel(
	const float *state_stream, const float *global_ctx, const float *d_g_raw,
	float *d_state_stream, float *d_global_ctx,
	int B, int C, int HW, const int *shifts, int num_shifts,
	int interaction_kind, int num_other_shifts, int raw_C, int full_mode)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int total = B * num_shifts * C * HW;
	if (idx >= total) return;

	int s = idx % HW;
	int rest = idx / HW;
	int c = rest % C;
	rest /= C;
	int shift_idx = rest % num_shifts;
	int b = rest / num_shifts;

	int shift = shifts[shift_idx];
	int c_roll = clifford_wrap_channel(c, shift, C);
	int base_channel = full_mode
		? ((interaction_kind == k_clifford_wedge_path)
			? clifford_wedge_base_channel(shift_idx, num_other_shifts, C)
			: clifford_inner_base_channel(shift_idx, num_other_shifts, C))
		: (shift_idx * C);

	float g_cur = global_ctx[b * C + c];
	float g_roll = global_ctx[b * C + c_roll];
	int cur_idx = (b * C + c) * HW + s;
	int roll_idx = (b * C + c_roll) * HW + s;

	int g_idx = (b * raw_C + base_channel + c) * HW + s;
	if (interaction_kind == k_clifford_inner_path)
	{
		float prod = state_stream[cur_idx] * g_roll;
		float sig = clifford_logistic(prod);
		float d_inner = d_g_raw[g_idx];
		float d_prod = d_inner * (sig + prod * sig * (1.0f - sig));
		atomicAdd(&d_state_stream[cur_idx], d_prod * g_roll);
		atomicAdd(&d_global_ctx[b * C + c_roll], d_prod * state_stream[cur_idx]);
	}
	else
	{
		float d_wedge = d_g_raw[g_idx];
		atomicAdd(&d_state_stream[cur_idx], d_wedge * g_roll);
		atomicAdd(&d_global_ctx[b * C + c_roll], d_wedge * state_stream[cur_idx]);
		atomicAdd(&d_global_ctx[b * C + c], -d_wedge * state_stream[roll_idx]);
		atomicAdd(&d_state_stream[roll_idx], -d_wedge * g_cur);
	}
}

// Global average pool over the spatial axis of an NCHW tensor.
// One thread owns one `(b, c)` pair and reduces across HW.
__global__ void clifford_global_avg_pool_forward_kernel(const float *input, float *output, int B, int C, int HW)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int total = B * C;
	if (idx >= total) return;

	int b = idx / C;
	int c = idx % C;
	const float *src = input + (b * C + c) * HW;

	float sum = 0.0f;
	for (int s = 0; s < HW; ++s)
	{
		sum += src[s];
	}
	output[idx] = sum / HW;
}

// Scatter the pooled gradient uniformly back over the spatial axis.
__global__ void clifford_global_avg_pool_backward_kernel(const float *d_out, float *d_in, int B, int C, int HW)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int total = B * C;
	if (idx >= total) return;

	int b = idx / C;
	int c = idx % C;
	float grad = d_out[idx] / HW;
	float *dst = d_in + (b * C + c) * HW;

	for (int s = 0; s < HW; ++s)
	{
		dst[s] += grad;
	}
}

__global__ void clifford_vb_forward_local_kernel(
	const float *z_det, const float *bivector_feat, float *vb_feat,
	int B, int C, int HW, const int *shifts, int num_shifts)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int total = B * num_shifts * C * HW;
	if (idx >= total) return;

	int s = idx % HW;
	int rest = idx / HW;
	int c = rest % C;
	rest /= C;
	int shift_idx = rest % num_shifts;
	int b = rest / num_shifts;

	int shift = shifts[shift_idx];
	int c_roll = clifford_wrap_channel(c, shift, C);
	int cur_idx = (b * C + c) * HW + s;
	int roll_idx = (b * C + c_roll) * HW + s;
	float scale = 1.0f / num_shifts;
	float value = scale * (z_det[cur_idx] * bivector_feat[roll_idx] - bivector_feat[cur_idx] * z_det[roll_idx]);
	atomicAdd(&vb_feat[cur_idx], value);
}

__global__ void clifford_vb_backward_local_kernel(
	const float *z_det, const float *bivector_feat, const float *d_vb_feat,
	float *d_z_det, float *d_bivector_feat,
	int B, int C, int HW, const int *shifts, int num_shifts)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int total = B * num_shifts * C * HW;
	if (idx >= total) return;

	int s = idx % HW;
	int rest = idx / HW;
	int c = rest % C;
	rest /= C;
	int shift_idx = rest % num_shifts;
	int b = rest / num_shifts;

	int shift = shifts[shift_idx];
	int c_roll = clifford_wrap_channel(c, shift, C);
	int cur_idx = (b * C + c) * HW + s;
	int roll_idx = (b * C + c_roll) * HW + s;
	float grad = (1.0f / num_shifts) * d_vb_feat[cur_idx];
	if (grad == 0.0f) return;

	atomicAdd(&d_z_det[cur_idx], grad * bivector_feat[roll_idx]);
	atomicAdd(&d_bivector_feat[roll_idx], grad * z_det[cur_idx]);
	atomicAdd(&d_bivector_feat[cur_idx], -grad * z_det[roll_idx]);
	atomicAdd(&d_z_det[roll_idx], -grad * bivector_feat[cur_idx]);
}

// Initialize the gated residual with `SiLU(X_ln)`.
// Later kernels add the gated local/global geometric features on top.
__global__ void clifford_init_hmix_kernel(const float *xln, float *hmix, int total)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= total) return;

	float x = xln[idx];
	hmix[idx] = x * clifford_logistic(x);
}

// Apply the gate sigmoid and add the gated feature contribution into `hmix`.
// The sigmoid output is saved explicitly for the backward pass.
__global__ void clifford_gate_accumulate_kernel(
	const float *pre_sigmoid, const float *gfeat, float *gate_alpha, float *hmix, int total)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= total) return;

	float alpha = clifford_logistic(pre_sigmoid[idx]);
	gate_alpha[idx] = alpha;
	hmix[idx] += alpha * gfeat[idx];
}

// Split `d_hmix` into direct feature flow and gate-logit flow.
// This is the elementwise part of the gated geometric residual backward pass.
__global__ void clifford_gate_backward_kernel(
	const float *d_hmix, const float *gate_alpha, const float *gfeat,
	float *d_gfeat, float *d_gate_pre, int total)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= total) return;

	float alpha = gate_alpha[idx];
	d_gfeat[idx] = d_hmix[idx] * alpha;
	d_gate_pre[idx] = (d_hmix[idx] * gfeat[idx]) * alpha * (1.0f - alpha);
}

// Base residual path gradient from `SiLU(X_ln)`.
// It accumulates into `d_xln` because additional gate/global/local paths also
// contribute to the normalized input.
__global__ void clifford_base_silu_backward_kernel(const float *xln, const float *d_hmix, float *d_xln, int total)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= total) return;

	d_xln[idx] += d_hmix[idx] * clifford_silu_grad(xln[idx]);
}

// Per-sample DropPath mask generation from uniform random values.
// Training uses {0, 1/keep_prob}; inference simply fills ones outside this kernel.
__global__ void clifford_droppath_mask_kernel(float *mask, int batch, float keep_prob)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= batch) return;

	mask[idx] = (mask[idx] < keep_prob) ? (1.0f / keep_prob) : 0.0f;
}

// Final residual update: `output = input + drop_mask[b] * layer_scale[c] * hmix`.
// `hmix` stays untouched so backward can still use the pre-scaled activations.
__global__ void clifford_residual_forward_kernel(
	const float *input, const float *hmix, const float *layer_scale, const float *drop_mask,
	float *output, int total, int C, int HW)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= total) return;

	int c = (idx / HW) % C;
	int b = idx / (C * HW);
	output[idx] = input[idx] + hmix[idx] * layer_scale[c] * drop_mask[b];
}

// First backward step through LayerScale and DropPath.
// `d_hmix` is produced elementwise, while `layer_scale_updates[c]` sums over
// every sample/spatial location in that channel and therefore uses atomics.
__global__ void clifford_residual_backward_kernel(
	const float *dout, const float *hmix, const float *layer_scale, const float *drop_mask,
	float *d_hmix, float *layer_scale_updates, int total, int C, int HW)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= total) return;

	int c = (idx / HW) % C;
	int b = idx / (C * HW);
	float scaled_mask = layer_scale[c] * drop_mask[b];

	atomicAdd(&layer_scale_updates[c], dout[idx] * drop_mask[b] * hmix[idx]);
	d_hmix[idx] = dout[idx] * scaled_mask;
}

void forward_clifford_layer_gpu(Darknet::Layer & l, Darknet::NetworkState state)
{
	TAT(TATPARMS);

	const int B = l.batch;
	const int C = l.c;
	const int HW = l.h * l.w;
	const int total = static_cast<int>(clifford_tensor_count(l));
	const int total_tokens = static_cast<int>(clifford_token_count(l));
	const int inner_shift_count = clifford_inner_num_shifts(l);
	int *inner_shifts_gpu = clifford_inner_shifts_gpu(l);
	const int full_mode = (l.cli_interaction_mode == 2);

	// Step 1: normalize each NCHW token across channels and cache the affine LN state.
	clifford_layernorm_forward_kernel<<<cuda_gridsize(total_tokens), BLOCK, 0, get_cuda_stream()>>>(
		state.input, l.cli_ln_out_gpu, l.cli_ln_mean_gpu, l.cli_ln_var_gpu, l.cli_ln_xhat_gpu,
		l.cli_ln_gamma_gpu, l.cli_ln_beta_gpu, total_tokens, C, HW);
	CHECK_CUDA(cudaPeekAtLastError());

		// Step 2a: deterministic stream is only needed by the local branch.
		if (clifford_has_local(l))
		{
			linear_1x1_forward_gpu(l.cli_ln_out_gpu, l.cli_z_det_gpu, l.cli_w_det_gpu, l.cli_b_det_gpu, B, C, C, HW);
		}

	if (clifford_has_local(l))
	{
		// Step 2b: local context comes from the stacked DWConv path, then SiLU and optional diff mode.
		forward_clifford_dwconv_stack_gpu(l, l.cli_ln_out_gpu, l.cli_z_ctx_pre_diff_gpu, state);
		clifford_silu_forward_kernel<<<cuda_gridsize(total), BLOCK, 0, get_cuda_stream()>>>(
			l.cli_z_ctx_pre_diff_gpu, l.cli_z_ctx_gpu, total);
		CHECK_CUDA(cudaPeekAtLastError());

		if (l.cli_ctx_mode == 1)
		{
			clifford_diff_context_kernel<<<cuda_gridsize(total), BLOCK, 0, get_cuda_stream()>>>(
				l.cli_z_ctx_gpu, l.cli_z_det_gpu, total);
			CHECK_CUDA(cudaPeekAtLastError());
		}

		// Step 3: shifted wedge/inner products are concatenated and projected back to C channels.
		if (l.cli_interaction_mode != 0)
		{
			clifford_rolling_forward_local_kernel<<<cuda_gridsize(static_cast<size_t>(B) * l.cli_num_shifts * C * HW), BLOCK, 0, get_cuda_stream()>>>(
				l.cli_z_det_gpu, l.cli_z_ctx_gpu, l.cli_g_raw_gpu,
				B, C, HW, l.cli_shifts_gpu, l.cli_num_shifts,
				k_clifford_wedge_path, inner_shift_count, l.cli_proj_in_dim, full_mode);
			CHECK_CUDA(cudaPeekAtLastError());
		}
		if (l.cli_interaction_mode != 1)
		{
			clifford_rolling_forward_local_kernel<<<cuda_gridsize(static_cast<size_t>(B) * inner_shift_count * C * HW), BLOCK, 0, get_cuda_stream()>>>(
				l.cli_z_det_gpu, l.cli_z_ctx_gpu, l.cli_g_raw_gpu,
				B, C, HW, inner_shifts_gpu, inner_shift_count,
				k_clifford_inner_path, l.cli_num_shifts, l.cli_proj_in_dim, full_mode);
			CHECK_CUDA(cudaPeekAtLastError());
			}

			linear_1x1_forward_gpu(l.cli_g_raw_gpu, l.cli_g_feat_gpu, l.cli_w_proj_gpu, l.cli_b_proj_gpu, B, l.cli_proj_in_dim, C, HW);
			if (clifford_has_higher_local(l))
			{
				fill_ongpu(total, 0.0f, l.cli_vb_feat_gpu, 1);
				clifford_vb_forward_local_kernel<<<cuda_gridsize(static_cast<size_t>(B) * l.cli_num_shifts * C * HW), BLOCK, 0, get_cuda_stream()>>>(
					l.cli_z_det_gpu, l.cli_g_feat_gpu, l.cli_vb_feat_gpu, B, C, HW, l.cli_shifts_gpu, l.cli_num_shifts);
				CHECK_CUDA(cudaPeekAtLastError());
			}
		}

	if (clifford_has_global(l))
	{
			// Step 3b: the global branch pools once per channel, then interacts directly with the normalized state.
			clifford_global_avg_pool_forward_kernel<<<cuda_gridsize(static_cast<size_t>(B) * C), BLOCK, 0, get_cuda_stream()>>>(
				l.cli_ln_out_gpu, l.cli_global_ctx_gpu, B, C, HW);
			CHECK_CUDA(cudaPeekAtLastError());

			if (l.cli_interaction_mode != 0)
			{
				clifford_rolling_forward_global_kernel<<<cuda_gridsize(static_cast<size_t>(B) * l.cli_num_shifts * C * HW), BLOCK, 0, get_cuda_stream()>>>(
					l.cli_ln_out_gpu, l.cli_global_ctx_gpu, l.cli_g_raw_g_gpu,
					B, C, HW, l.cli_shifts_gpu, l.cli_num_shifts,
					k_clifford_wedge_path, inner_shift_count, l.cli_proj_in_dim, full_mode);
				CHECK_CUDA(cudaPeekAtLastError());
			}
			if (l.cli_interaction_mode != 1)
			{
				clifford_rolling_forward_global_kernel<<<cuda_gridsize(static_cast<size_t>(B) * inner_shift_count * C * HW), BLOCK, 0, get_cuda_stream()>>>(
					l.cli_ln_out_gpu, l.cli_global_ctx_gpu, l.cli_g_raw_g_gpu,
					B, C, HW, inner_shifts_gpu, inner_shift_count,
					k_clifford_inner_path, l.cli_num_shifts, l.cli_proj_in_dim, full_mode);
				CHECK_CUDA(cudaPeekAtLastError());
			}

		linear_1x1_forward_gpu(l.cli_g_raw_g_gpu, l.cli_g_feat_g_gpu, l.cli_w_proj_g_gpu, l.cli_b_proj_g_gpu, B, l.cli_proj_in_dim, C, HW);
	}

	// Step 4: initialize the gated residual with SiLU(X_ln), then add local/global gated features.
	clifford_init_hmix_kernel<<<cuda_gridsize(total), BLOCK, 0, get_cuda_stream()>>>(
		l.cli_ln_out_gpu, l.cli_hmix_gpu, total);
	CHECK_CUDA(cudaPeekAtLastError());

	if (clifford_has_local(l))
	{
		gate_linear_forward_gpu(l.cli_ln_out_gpu, l.cli_g_feat_gpu, l.cli_gate_pre_sigmoid_gpu,
			l.cli_w_gate_gpu, l.cli_b_gate_gpu, B, C, HW);
			clifford_gate_accumulate_kernel<<<cuda_gridsize(total), BLOCK, 0, get_cuda_stream()>>>(
				l.cli_gate_pre_sigmoid_gpu, l.cli_g_feat_gpu, l.cli_gate_alpha_gpu, l.cli_hmix_gpu, total);
			CHECK_CUDA(cudaPeekAtLastError());
			if (clifford_has_higher_local(l))
			{
				axpy_ongpu(total, 1.0f, l.cli_vb_feat_gpu, 1, l.cli_hmix_gpu, 1);
			}
		}

	if (clifford_has_global(l))
	{
		gate_linear_forward_gpu(l.cli_ln_out_gpu, l.cli_g_feat_g_gpu, l.cli_gate_pre_sigmoid_g_gpu,
			l.cli_w_gate_g_gpu, l.cli_b_gate_g_gpu, B, C, HW);
		clifford_gate_accumulate_kernel<<<cuda_gridsize(total), BLOCK, 0, get_cuda_stream()>>>(
			l.cli_gate_pre_sigmoid_g_gpu, l.cli_g_feat_g_gpu, l.cli_gate_alpha_g_gpu, l.cli_hmix_gpu, total);
		CHECK_CUDA(cudaPeekAtLastError());
	}

	if (state.train && l.cli_drop_path > 0.0f)
	{
		const float keep_prob = 1.0f - l.cli_drop_path;
		cuda_random(l.cli_drop_mask_gpu, l.batch);
		clifford_droppath_mask_kernel<<<cuda_gridsize(l.batch), BLOCK, 0, get_cuda_stream()>>>(
			l.cli_drop_mask_gpu, l.batch, keep_prob);
		CHECK_CUDA(cudaPeekAtLastError());
	}
	else
	{
		fill_ongpu(l.batch, 1.0f, l.cli_drop_mask_gpu, 1);
	}

	// Step 5: apply LayerScale and DropPath without mutating `hmix`, then add the residual connection.
	clifford_residual_forward_kernel<<<cuda_gridsize(total), BLOCK, 0, get_cuda_stream()>>>(
		state.input, l.cli_hmix_gpu, l.cli_layer_scale_gpu, l.cli_drop_mask_gpu,
		l.output_gpu, total, C, HW);
	CHECK_CUDA(cudaPeekAtLastError());
}

void backward_clifford_layer_gpu(Darknet::Layer & l, Darknet::NetworkState state)
{
	TAT(TATPARMS);

	const int B = l.batch;
	const int C = l.c;
	const int HW = l.h * l.w;
	const int total = static_cast<int>(clifford_tensor_count(l));
	const int total_tokens = static_cast<int>(clifford_token_count(l));
	const int inner_shift_count = clifford_inner_num_shifts(l);
	int *inner_shifts_gpu = clifford_inner_shifts_gpu(l);
	const int full_mode = (l.cli_interaction_mode == 2);

	float *d_hmix_gpu = make_temp_float(total);
	float *d_xln_gpu = make_temp_float(total);
	float *d_z_det_gpu = make_temp_float(total);
	float *d_input_gpu = state.delta ? make_temp_float(total) : nullptr;

	fill_ongpu(total, 0.0f, d_xln_gpu, 1);
	fill_ongpu(total, 0.0f, d_z_det_gpu, 1);
	if (d_input_gpu)
	{
		fill_ongpu(total, 0.0f, d_input_gpu, 1);
	}

	// Step 5 backward: peel off DropPath and LayerScale while accumulating LayerScale gradients.
	clifford_residual_backward_kernel<<<cuda_gridsize(total), BLOCK, 0, get_cuda_stream()>>>(
		l.delta_gpu, l.cli_hmix_gpu, l.cli_layer_scale_gpu, l.cli_drop_mask_gpu,
		d_hmix_gpu, l.cli_layer_scale_updates_gpu, total, C, HW);
	CHECK_CUDA(cudaPeekAtLastError());

	// Step 4 backward: base SiLU(X_ln) path is always present, even if local/global branches are disabled.
	clifford_base_silu_backward_kernel<<<cuda_gridsize(total), BLOCK, 0, get_cuda_stream()>>>(
		l.cli_ln_out_gpu, d_hmix_gpu, d_xln_gpu, total);
	CHECK_CUDA(cudaPeekAtLastError());

	if (clifford_has_local(l))
	{
		float *d_gfeat_gpu = make_temp_float(total);
		float *d_gate_pre_gpu = make_temp_float(total);
		float *d_graw_gpu = make_temp_float(static_cast<size_t>(B) * l.cli_proj_in_dim * HW);
		float *d_z_ctx_gpu = make_temp_float(total);
		float *d_ctx_input_gpu = make_temp_float(total);

		fill_ongpu(static_cast<int>(static_cast<size_t>(B) * l.cli_proj_in_dim * HW), 0.0f, d_graw_gpu, 1);
		fill_ongpu(total, 0.0f, d_z_ctx_gpu, 1);
		fill_ongpu(total, 0.0f, d_ctx_input_gpu, 1);

		// Local GGR backward: split `d_hmix` into gate-logit and feature gradients.
			clifford_gate_backward_kernel<<<cuda_gridsize(total), BLOCK, 0, get_cuda_stream()>>>(
				d_hmix_gpu, l.cli_gate_alpha_gpu, l.cli_g_feat_gpu, d_gfeat_gpu, d_gate_pre_gpu, total);
			CHECK_CUDA(cudaPeekAtLastError());

			if (clifford_has_higher_local(l))
			{
				clifford_vb_backward_local_kernel<<<cuda_gridsize(static_cast<size_t>(B) * l.cli_num_shifts * C * HW), BLOCK, 0, get_cuda_stream()>>>(
					l.cli_z_det_gpu, l.cli_g_feat_gpu, d_hmix_gpu, d_z_det_gpu, d_gfeat_gpu,
					B, C, HW, l.cli_shifts_gpu, l.cli_num_shifts);
				CHECK_CUDA(cudaPeekAtLastError());
			}

			gate_linear_backward_gpu(l.cli_ln_out_gpu, l.cli_g_feat_gpu, d_gate_pre_gpu,
				l.cli_w_gate_gpu, l.cli_w_gate_updates_gpu, l.cli_b_gate_updates_gpu,
			d_xln_gpu, d_gfeat_gpu, B, C, HW);

		linear_1x1_backward_gpu(l.cli_g_raw_gpu, d_gfeat_gpu, l.cli_w_proj_gpu,
			l.cli_w_proj_updates_gpu, l.cli_b_proj_updates_gpu, d_graw_gpu,
			B, l.cli_proj_in_dim, C, HW);

		// Step 3 backward: unroll the shifted inner/wedge products back into det/context streams.
		if (l.cli_interaction_mode != 0)
		{
			clifford_rolling_backward_local_kernel<<<cuda_gridsize(static_cast<size_t>(B) * l.cli_num_shifts * C * HW), BLOCK, 0, get_cuda_stream()>>>(
				l.cli_z_det_gpu, l.cli_z_ctx_gpu, d_graw_gpu, d_z_det_gpu, d_z_ctx_gpu,
				B, C, HW, l.cli_shifts_gpu, l.cli_num_shifts,
				k_clifford_wedge_path, inner_shift_count, l.cli_proj_in_dim, full_mode);
			CHECK_CUDA(cudaPeekAtLastError());
		}
		if (l.cli_interaction_mode != 1)
		{
			clifford_rolling_backward_local_kernel<<<cuda_gridsize(static_cast<size_t>(B) * inner_shift_count * C * HW), BLOCK, 0, get_cuda_stream()>>>(
				l.cli_z_det_gpu, l.cli_z_ctx_gpu, d_graw_gpu, d_z_det_gpu, d_z_ctx_gpu,
				B, C, HW, inner_shifts_gpu, inner_shift_count,
				k_clifford_inner_path, l.cli_num_shifts, l.cli_proj_in_dim, full_mode);
			CHECK_CUDA(cudaPeekAtLastError());
		}

		if (l.cli_ctx_mode == 1)
		{
			clifford_diff_context_backward_kernel<<<cuda_gridsize(total), BLOCK, 0, get_cuda_stream()>>>(
				d_z_det_gpu, d_z_ctx_gpu, total);
			CHECK_CUDA(cudaPeekAtLastError());
		}

		clifford_silu_backward_inplace_kernel<<<cuda_gridsize(total), BLOCK, 0, get_cuda_stream()>>>(
			l.cli_z_ctx_pre_diff_gpu, d_z_ctx_gpu, total);
		CHECK_CUDA(cudaPeekAtLastError());

		// Step 2 backward: push the contextual gradient through the stacked DWConv path.
		backward_clifford_dwconv_stack_gpu(l, l.cli_ln_out_gpu, d_z_ctx_gpu, d_ctx_input_gpu, state);
		axpy_ongpu(total, 1.0f, d_ctx_input_gpu, 1, d_xln_gpu, 1);

		free_temp_float(d_ctx_input_gpu);
		free_temp_float(d_z_ctx_gpu);
		free_temp_float(d_graw_gpu);
		free_temp_float(d_gate_pre_gpu);
		free_temp_float(d_gfeat_gpu);
	}

	if (clifford_has_global(l))
	{
		float *d_gfeat_g_gpu = make_temp_float(total);
		float *d_gate_pre_g_gpu = make_temp_float(total);
		float *d_graw_g_gpu = make_temp_float(static_cast<size_t>(B) * l.cli_proj_in_dim * HW);
		float *d_global_ctx_gpu = make_temp_float(static_cast<size_t>(B) * C);

		fill_ongpu(static_cast<int>(static_cast<size_t>(B) * l.cli_proj_in_dim * HW), 0.0f, d_graw_g_gpu, 1);
		fill_ongpu(B * C, 0.0f, d_global_ctx_gpu, 1);

		// Global GGR backward mirrors the local branch, then redistributes pooled gradients across HW.
		clifford_gate_backward_kernel<<<cuda_gridsize(total), BLOCK, 0, get_cuda_stream()>>>(
			d_hmix_gpu, l.cli_gate_alpha_g_gpu, l.cli_g_feat_g_gpu, d_gfeat_g_gpu, d_gate_pre_g_gpu, total);
		CHECK_CUDA(cudaPeekAtLastError());

		gate_linear_backward_gpu(l.cli_ln_out_gpu, l.cli_g_feat_g_gpu, d_gate_pre_g_gpu,
			l.cli_w_gate_g_gpu, l.cli_w_gate_g_updates_gpu, l.cli_b_gate_g_updates_gpu,
			d_xln_gpu, d_gfeat_g_gpu, B, C, HW);

		linear_1x1_backward_gpu(l.cli_g_raw_g_gpu, d_gfeat_g_gpu, l.cli_w_proj_g_gpu,
			l.cli_w_proj_g_updates_gpu, l.cli_b_proj_g_updates_gpu, d_graw_g_gpu,
			B, l.cli_proj_in_dim, C, HW);

			if (l.cli_interaction_mode != 0)
			{
				clifford_rolling_backward_global_kernel<<<cuda_gridsize(static_cast<size_t>(B) * l.cli_num_shifts * C * HW), BLOCK, 0, get_cuda_stream()>>>(
					l.cli_ln_out_gpu, l.cli_global_ctx_gpu, d_graw_g_gpu, d_xln_gpu, d_global_ctx_gpu,
					B, C, HW, l.cli_shifts_gpu, l.cli_num_shifts,
					k_clifford_wedge_path, inner_shift_count, l.cli_proj_in_dim, full_mode);
				CHECK_CUDA(cudaPeekAtLastError());
			}
			if (l.cli_interaction_mode != 1)
			{
				clifford_rolling_backward_global_kernel<<<cuda_gridsize(static_cast<size_t>(B) * inner_shift_count * C * HW), BLOCK, 0, get_cuda_stream()>>>(
					l.cli_ln_out_gpu, l.cli_global_ctx_gpu, d_graw_g_gpu, d_xln_gpu, d_global_ctx_gpu,
					B, C, HW, inner_shifts_gpu, inner_shift_count,
					k_clifford_inner_path, l.cli_num_shifts, l.cli_proj_in_dim, full_mode);
				CHECK_CUDA(cudaPeekAtLastError());
			}

		clifford_global_avg_pool_backward_kernel<<<cuda_gridsize(static_cast<size_t>(B) * C), BLOCK, 0, get_cuda_stream()>>>(
			d_global_ctx_gpu, d_xln_gpu, B, C, HW);
		CHECK_CUDA(cudaPeekAtLastError());

		free_temp_float(d_global_ctx_gpu);
		free_temp_float(d_graw_g_gpu);
		free_temp_float(d_gate_pre_g_gpu);
		free_temp_float(d_gfeat_g_gpu);
	}

		if (clifford_has_local(l))
		{
			// Step 2a backward: deterministic 1x1 projection feeds directly into the normalized input.
			linear_1x1_backward_gpu(l.cli_ln_out_gpu, d_z_det_gpu, l.cli_w_det_gpu,
				l.cli_w_det_updates_gpu, l.cli_b_det_updates_gpu, d_xln_gpu, B, C, C, HW);
		}

	// Step 1 backward: collapse all normalized-input contributions through LayerNorm.
	clifford_layernorm_backward_kernel<<<cuda_gridsize(total_tokens), BLOCK, 0, get_cuda_stream()>>>(
		d_xln_gpu, l.cli_ln_xhat_gpu, l.cli_ln_var_gpu, l.cli_ln_gamma_gpu,
		d_input_gpu, l.cli_ln_gamma_updates_gpu, l.cli_ln_beta_updates_gpu, total_tokens, C, HW);
	CHECK_CUDA(cudaPeekAtLastError());

	if (state.delta)
	{
		axpy_ongpu(total, 1.0f, l.delta_gpu, 1, state.delta, 1);
		axpy_ongpu(total, 1.0f, d_input_gpu, 1, state.delta, 1);
	}

	free_temp_float(d_input_gpu);
	free_temp_float(d_z_det_gpu);
	free_temp_float(d_xln_gpu);
	free_temp_float(d_hmix_gpu);
}

void update_clifford_layer_gpu(Darknet::Layer & l, int batch, float lr, float momentum, float decay, float loss_scale)
{
	TAT(TATPARMS);

	const float learning_rate = lr * l.learning_rate_scale;
	const int C = l.c;
	const int det_count = C * C;
	const int proj_count = C * l.cli_proj_in_dim;
	const int gate_count = C * 2 * C;
	const float effective_loss_scale = (loss_scale == 0.0f) ? 1.0f : loss_scale;

	// Scale accumulated gradients back down if mixed-precision loss scaling was active.
	scale_clifford_updates_for_loss(l.cli_w_det_updates_gpu, det_count, effective_loss_scale);
	scale_clifford_updates_for_loss(l.cli_b_det_updates_gpu, C, effective_loss_scale);
	scale_clifford_updates_for_loss(l.cli_w_proj_updates_gpu, proj_count, effective_loss_scale);
	scale_clifford_updates_for_loss(l.cli_b_proj_updates_gpu, C, effective_loss_scale);
	scale_clifford_updates_for_loss(l.cli_w_gate_updates_gpu, gate_count, effective_loss_scale);
	scale_clifford_updates_for_loss(l.cli_b_gate_updates_gpu, C, effective_loss_scale);
	scale_clifford_updates_for_loss(l.cli_ln_gamma_updates_gpu, C, effective_loss_scale);
	scale_clifford_updates_for_loss(l.cli_ln_beta_updates_gpu, C, effective_loss_scale);
	scale_clifford_updates_for_loss(l.cli_layer_scale_updates_gpu, C, effective_loss_scale);

	if (clifford_has_global(l))
	{
		scale_clifford_updates_for_loss(l.cli_w_proj_g_updates_gpu, proj_count, effective_loss_scale);
		scale_clifford_updates_for_loss(l.cli_b_proj_g_updates_gpu, C, effective_loss_scale);
		scale_clifford_updates_for_loss(l.cli_w_gate_g_updates_gpu, gate_count, effective_loss_scale);
		scale_clifford_updates_for_loss(l.cli_b_gate_g_updates_gpu, C, effective_loss_scale);
	}

	update_clifford_weights_gpu(l.cli_w_det_gpu, l.cli_w_det_updates_gpu, det_count, batch, learning_rate, momentum, decay);
	update_clifford_bias_like_gpu(l.cli_b_det_gpu, l.cli_b_det_updates_gpu, C, batch, learning_rate, momentum);

	update_clifford_weights_gpu(l.cli_w_proj_gpu, l.cli_w_proj_updates_gpu, proj_count, batch, learning_rate, momentum, decay);
	update_clifford_bias_like_gpu(l.cli_b_proj_gpu, l.cli_b_proj_updates_gpu, C, batch, learning_rate, momentum);

	update_clifford_weights_gpu(l.cli_w_gate_gpu, l.cli_w_gate_updates_gpu, gate_count, batch, learning_rate, momentum, decay);
	update_clifford_bias_like_gpu(l.cli_b_gate_gpu, l.cli_b_gate_updates_gpu, C, batch, learning_rate, momentum);

	update_clifford_bias_like_gpu(l.cli_ln_gamma_gpu, l.cli_ln_gamma_updates_gpu, C, batch, learning_rate, momentum);
	update_clifford_bias_like_gpu(l.cli_ln_beta_gpu, l.cli_ln_beta_updates_gpu, C, batch, learning_rate, momentum);
	update_clifford_bias_like_gpu(l.cli_layer_scale_gpu, l.cli_layer_scale_updates_gpu, C, batch, learning_rate, momentum);

	if (clifford_has_global(l))
	{
		update_clifford_weights_gpu(l.cli_w_proj_g_gpu, l.cli_w_proj_g_updates_gpu, proj_count, batch, learning_rate, momentum, decay);
		update_clifford_bias_like_gpu(l.cli_b_proj_g_gpu, l.cli_b_proj_g_updates_gpu, C, batch, learning_rate, momentum);

		update_clifford_weights_gpu(l.cli_w_gate_g_gpu, l.cli_w_gate_g_updates_gpu, gate_count, batch, learning_rate, momentum, decay);
		update_clifford_bias_like_gpu(l.cli_b_gate_g_gpu, l.cli_b_gate_g_updates_gpu, C, batch, learning_rate, momentum);
	}

	for (int i = 0; i < l.cli_num_dwconv; ++i)
	{
		update_convolutional_layer_gpu(l.cli_dwconv[i], batch, lr, momentum, decay, effective_loss_scale);
	}
}

void push_clifford_layer(Darknet::Layer & l)
{
	TAT(TATPARMS);

	// Synchronize all persistent Clifford parameters and the shift schedule to device memory.
	if (l.cli_shifts_gpu && l.cli_shifts)
	{
		CHECK_CUDA(cudaMemcpyAsync(l.cli_shifts_gpu, l.cli_shifts, sizeof(int) * l.cli_num_shifts, cudaMemcpyDefault, get_cuda_stream()));
	}
	if (l.cli_shifts_inner_gpu && l.cli_shifts_inner)
	{
		CHECK_CUDA(cudaMemcpyAsync(l.cli_shifts_inner_gpu, l.cli_shifts_inner, sizeof(int) * l.cli_num_shifts_inner, cudaMemcpyDefault, get_cuda_stream()));
	}

	if (l.cli_w_det_gpu) cuda_push_array(l.cli_w_det_gpu, l.cli_w_det, static_cast<size_t>(l.c) * l.c);
	if (l.cli_w_det_updates_gpu) cuda_push_array(l.cli_w_det_updates_gpu, l.cli_w_det_updates, static_cast<size_t>(l.c) * l.c);
	if (l.cli_b_det_gpu) cuda_push_array(l.cli_b_det_gpu, l.cli_b_det, l.c);
	if (l.cli_b_det_updates_gpu) cuda_push_array(l.cli_b_det_updates_gpu, l.cli_b_det_updates, l.c);

	if (l.cli_w_proj_gpu) cuda_push_array(l.cli_w_proj_gpu, l.cli_w_proj, static_cast<size_t>(l.c) * l.cli_proj_in_dim);
	if (l.cli_w_proj_updates_gpu) cuda_push_array(l.cli_w_proj_updates_gpu, l.cli_w_proj_updates, static_cast<size_t>(l.c) * l.cli_proj_in_dim);
	if (l.cli_b_proj_gpu) cuda_push_array(l.cli_b_proj_gpu, l.cli_b_proj, l.c);
	if (l.cli_b_proj_updates_gpu) cuda_push_array(l.cli_b_proj_updates_gpu, l.cli_b_proj_updates, l.c);

	if (l.cli_w_gate_gpu) cuda_push_array(l.cli_w_gate_gpu, l.cli_w_gate, static_cast<size_t>(l.c) * 2 * l.c);
	if (l.cli_w_gate_updates_gpu) cuda_push_array(l.cli_w_gate_updates_gpu, l.cli_w_gate_updates, static_cast<size_t>(l.c) * 2 * l.c);
	if (l.cli_b_gate_gpu) cuda_push_array(l.cli_b_gate_gpu, l.cli_b_gate, l.c);
	if (l.cli_b_gate_updates_gpu) cuda_push_array(l.cli_b_gate_updates_gpu, l.cli_b_gate_updates, l.c);

	if (l.cli_ln_gamma_gpu) cuda_push_array(l.cli_ln_gamma_gpu, l.cli_ln_gamma, l.c);
	if (l.cli_ln_gamma_updates_gpu) cuda_push_array(l.cli_ln_gamma_updates_gpu, l.cli_ln_gamma_updates, l.c);
	if (l.cli_ln_beta_gpu) cuda_push_array(l.cli_ln_beta_gpu, l.cli_ln_beta, l.c);
	if (l.cli_ln_beta_updates_gpu) cuda_push_array(l.cli_ln_beta_updates_gpu, l.cli_ln_beta_updates, l.c);
	if (l.cli_layer_scale_gpu) cuda_push_array(l.cli_layer_scale_gpu, l.cli_layer_scale, l.c);
	if (l.cli_layer_scale_updates_gpu) cuda_push_array(l.cli_layer_scale_updates_gpu, l.cli_layer_scale_updates, l.c);

	if (l.cli_w_proj_g_gpu) cuda_push_array(l.cli_w_proj_g_gpu, l.cli_w_proj_g, static_cast<size_t>(l.c) * l.cli_proj_in_dim);
	if (l.cli_w_proj_g_updates_gpu) cuda_push_array(l.cli_w_proj_g_updates_gpu, l.cli_w_proj_g_updates, static_cast<size_t>(l.c) * l.cli_proj_in_dim);
	if (l.cli_b_proj_g_gpu) cuda_push_array(l.cli_b_proj_g_gpu, l.cli_b_proj_g, l.c);
	if (l.cli_b_proj_g_updates_gpu) cuda_push_array(l.cli_b_proj_g_updates_gpu, l.cli_b_proj_g_updates, l.c);
	if (l.cli_w_gate_g_gpu) cuda_push_array(l.cli_w_gate_g_gpu, l.cli_w_gate_g, static_cast<size_t>(l.c) * 2 * l.c);
	if (l.cli_w_gate_g_updates_gpu) cuda_push_array(l.cli_w_gate_g_updates_gpu, l.cli_w_gate_g_updates, static_cast<size_t>(l.c) * 2 * l.c);
	if (l.cli_b_gate_g_gpu) cuda_push_array(l.cli_b_gate_g_gpu, l.cli_b_gate_g, l.c);
	if (l.cli_b_gate_g_updates_gpu) cuda_push_array(l.cli_b_gate_g_updates_gpu, l.cli_b_gate_g_updates, l.c);

	for (int i = 0; i < l.cli_num_dwconv; ++i)
	{
		push_convolutional_layer(l.cli_dwconv[i]);
	}
}

void pull_clifford_layer(Darknet::Layer & l)
{
	TAT(TATPARMS);

	// Pull parameters back so save/load, inspection, and CPU-side debug stay coherent.
	if (l.cli_w_det_gpu) cuda_pull_array(l.cli_w_det_gpu, l.cli_w_det, static_cast<size_t>(l.c) * l.c);
	if (l.cli_w_det_updates_gpu) cuda_pull_array(l.cli_w_det_updates_gpu, l.cli_w_det_updates, static_cast<size_t>(l.c) * l.c);
	if (l.cli_b_det_gpu) cuda_pull_array(l.cli_b_det_gpu, l.cli_b_det, l.c);
	if (l.cli_b_det_updates_gpu) cuda_pull_array(l.cli_b_det_updates_gpu, l.cli_b_det_updates, l.c);

	if (l.cli_w_proj_gpu) cuda_pull_array(l.cli_w_proj_gpu, l.cli_w_proj, static_cast<size_t>(l.c) * l.cli_proj_in_dim);
	if (l.cli_w_proj_updates_gpu) cuda_pull_array(l.cli_w_proj_updates_gpu, l.cli_w_proj_updates, static_cast<size_t>(l.c) * l.cli_proj_in_dim);
	if (l.cli_b_proj_gpu) cuda_pull_array(l.cli_b_proj_gpu, l.cli_b_proj, l.c);
	if (l.cli_b_proj_updates_gpu) cuda_pull_array(l.cli_b_proj_updates_gpu, l.cli_b_proj_updates, l.c);

	if (l.cli_w_gate_gpu) cuda_pull_array(l.cli_w_gate_gpu, l.cli_w_gate, static_cast<size_t>(l.c) * 2 * l.c);
	if (l.cli_w_gate_updates_gpu) cuda_pull_array(l.cli_w_gate_updates_gpu, l.cli_w_gate_updates, static_cast<size_t>(l.c) * 2 * l.c);
	if (l.cli_b_gate_gpu) cuda_pull_array(l.cli_b_gate_gpu, l.cli_b_gate, l.c);
	if (l.cli_b_gate_updates_gpu) cuda_pull_array(l.cli_b_gate_updates_gpu, l.cli_b_gate_updates, l.c);

	if (l.cli_ln_gamma_gpu) cuda_pull_array(l.cli_ln_gamma_gpu, l.cli_ln_gamma, l.c);
	if (l.cli_ln_gamma_updates_gpu) cuda_pull_array(l.cli_ln_gamma_updates_gpu, l.cli_ln_gamma_updates, l.c);
	if (l.cli_ln_beta_gpu) cuda_pull_array(l.cli_ln_beta_gpu, l.cli_ln_beta, l.c);
	if (l.cli_ln_beta_updates_gpu) cuda_pull_array(l.cli_ln_beta_updates_gpu, l.cli_ln_beta_updates, l.c);
	if (l.cli_layer_scale_gpu) cuda_pull_array(l.cli_layer_scale_gpu, l.cli_layer_scale, l.c);
	if (l.cli_layer_scale_updates_gpu) cuda_pull_array(l.cli_layer_scale_updates_gpu, l.cli_layer_scale_updates, l.c);

	if (l.cli_w_proj_g_gpu) cuda_pull_array(l.cli_w_proj_g_gpu, l.cli_w_proj_g, static_cast<size_t>(l.c) * l.cli_proj_in_dim);
	if (l.cli_w_proj_g_updates_gpu) cuda_pull_array(l.cli_w_proj_g_updates_gpu, l.cli_w_proj_g_updates, static_cast<size_t>(l.c) * l.cli_proj_in_dim);
	if (l.cli_b_proj_g_gpu) cuda_pull_array(l.cli_b_proj_g_gpu, l.cli_b_proj_g, l.c);
	if (l.cli_b_proj_g_updates_gpu) cuda_pull_array(l.cli_b_proj_g_updates_gpu, l.cli_b_proj_g_updates, l.c);
	if (l.cli_w_gate_g_gpu) cuda_pull_array(l.cli_w_gate_g_gpu, l.cli_w_gate_g, static_cast<size_t>(l.c) * 2 * l.c);
	if (l.cli_w_gate_g_updates_gpu) cuda_pull_array(l.cli_w_gate_g_updates_gpu, l.cli_w_gate_g_updates, static_cast<size_t>(l.c) * 2 * l.c);
	if (l.cli_b_gate_g_gpu) cuda_pull_array(l.cli_b_gate_g_gpu, l.cli_b_gate_g, l.c);
	if (l.cli_b_gate_g_updates_gpu) cuda_pull_array(l.cli_b_gate_g_updates_gpu, l.cli_b_gate_g_updates, l.c);

	for (int i = 0; i < l.cli_num_dwconv; ++i)
	{
		pull_convolutional_layer(l.cli_dwconv[i]);
	}
}

#endif
