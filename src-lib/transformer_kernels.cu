#include "darknet_internal.hpp"
#include "transformer_layer.hpp"
#include "blas.hpp"
#include "gemm.hpp"
#include "activations.hpp"
#include "dark_cuda.hpp"

#include <cstdio>

#ifdef DARKNET_GPU

namespace
{
	constexpr int TRANSFORMER_MHC_PARAM_COUNT = 12;
	constexpr float TRANSFORMER_MHC_PARAM_CLAMP = 8.0f;

	static void check_nan_gpu(const char *step_name, float *d_arr, size_t size, int layer_idx)
	{
		if (d_arr == nullptr || size == 0)
		{
			return;
		}

		if (is_nan_or_inf(d_arr, size))
		{
			const std::string layer_label = Darknet::layer_type_diagnostic_label(Darknet::ELayerType::TRANSFORMER);
			std::printf("[%s layer] NaN/Inf detected at layer %d, step: %s\n", layer_label.c_str(), layer_idx, step_name);
		}
	}

	static bool transformer_use_tensor_op(const Darknet::NetworkState &state)
	{
#ifdef DARKNET_GPU_CUDA
		return state.net.cudnn_half || state.net.cudnn_bf16;
#else
		(void)state;
		return false;
#endif
	}

#if defined(CUDNN) && !defined(DARKNET_GPU_ROCM)
	static void transformer_softmax_forward_gpu(float *scores, int rows, int T)
	{
		cudnnTensorDescriptor_t scores_desc;
		CHECK_CUDNN(cudnnCreateTensorDescriptor(&scores_desc));
		CHECK_CUDNN(cudnnSetTensor4dDescriptor(
			scores_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, rows, T, 1, 1));

		const float alpha = 1.0f;
		const float beta = 0.0f;
		CHECK_CUDNN(cudnnSoftmaxForward(
			cudnn_handle(),
			CUDNN_SOFTMAX_ACCURATE,
			CUDNN_SOFTMAX_MODE_INSTANCE,
			&alpha,
			scores_desc,
			scores,
			&beta,
			scores_desc,
			scores));
		CHECK_CUDNN(cudnnDestroyTensorDescriptor(scores_desc));
	}

	static void transformer_softmax_backward_gpu(const float *scores, float *d_scores, int rows, int T)
	{
		cudnnTensorDescriptor_t scores_desc;
		CHECK_CUDNN(cudnnCreateTensorDescriptor(&scores_desc));
		CHECK_CUDNN(cudnnSetTensor4dDescriptor(
			scores_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, rows, T, 1, 1));

		const float alpha = 1.0f;
		const float beta = 0.0f;
		CHECK_CUDNN(cudnnSoftmaxBackward(
			cudnn_handle(),
			CUDNN_SOFTMAX_ACCURATE,
			CUDNN_SOFTMAX_MODE_INSTANCE,
			&alpha,
			scores_desc,
			scores,
			scores_desc,
			d_scores,
			&beta,
			scores_desc,
			d_scores));
		CHECK_CUDNN(cudnnDestroyTensorDescriptor(scores_desc));
	}
#endif
}

// ═══════════════════════════════════════════════════════════════════════════════
// Transformer Layer GPU Implementation
// ═══════════════════════════════════════════════════════════════════════════════
template <typename T>
static inline T *workspace_ptr(T *base, size_t offset)
{
	return base + offset;
}


// ─── CUDA kernels ─────────────────────────────────────────────────────────────

// Warp-based layernorm: one warp (32 threads) per token.
// Each lane accumulates its strided elements then reduces via warp shuffles —
// no shared memory, 5 shuffle rounds replace a serial C-loop for mean/var.
__global__ void transformer_layernorm_forward_kernel(
	const float *x, float *out, float *mean, float *var, float *xhat,
	const float *gamma, const float *beta, int total_tokens, int C)
{
	const int lane     = threadIdx.x % 32;
	const int warp_id  = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
	if (warp_id >= total_tokens) return;

	const float eps    = 1e-5f;
	const float *xi    = x    + warp_id * C;
	float       *oi    = out  + warp_id * C;
	float       *xh    = xhat + warp_id * C;

	// Pass 1: compute mean via warp reduction
	float local_sum = 0.0f;
	for (int j = lane; j < C; j += 32) local_sum += xi[j];
	for (int offset = 16; offset > 0; offset >>= 1)
		local_sum += __shfl_xor_sync(0xFFFFFFFF, local_sum, offset);
	const float m = local_sum / C;
	if (lane == 0) mean[warp_id] = m;

	// Pass 2: compute variance via warp reduction
	float local_var = 0.0f;
	for (int j = lane; j < C; j += 32)
	{
		const float d = xi[j] - m;
		local_var += d * d;
	}
	for (int offset = 16; offset > 0; offset >>= 1)
		local_var += __shfl_xor_sync(0xFFFFFFFF, local_var, offset);
	const float v = local_var / C;
	if (lane == 0) var[warp_id] = v;

	const float inv_std = rsqrtf(v + eps);

	// Pass 3: normalize + affine transform
	for (int j = lane; j < C; j += 32)
	{
		const float xh_val = (xi[j] - m) * inv_std;
		xh[j] = xh_val;
		oi[j]  = xh_val * gamma[j] + beta[j];
	}
}

__global__ void transformer_layernorm_backward_kernel(
	const float *dout, const float *xhat, const float *var,
	const float *gamma, float *dx, float *dgamma, float *dbeta,
	int total_tokens, int C)
{
	const int lane = threadIdx.x & 31;
	const int i = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
	if (i >= total_tokens) return;

	const float eps = 1e-5f;
	const float *doi = dout + i * C;
	const float *xhi = xhat + i * C;
	float *dxi = dx + i * C;
	float inv_std = rsqrtf(var[i] + eps);

	// Note: dgamma/dbeta accumulation needs atomics for GPU correctness
	for (int j = lane; j < C; j += 32)
	{
		atomicAdd(&dgamma[j], doi[j] * xhi[j]);
		atomicAdd(&dbeta[j], doi[j]);
	}

	float sum_dxhat = 0.0f;
	float dot_dxhat_xhat = 0.0f;
	for (int j = lane; j < C; j += 32)
	{
		const float dxhat = doi[j] * gamma[j];
		sum_dxhat += dxhat;
		dot_dxhat_xhat += dxhat * xhi[j];
	}
	for (int offset = 16; offset > 0; offset >>= 1)
	{
		sum_dxhat += __shfl_xor_sync(0xFFFFFFFF, sum_dxhat, offset);
		dot_dxhat_xhat += __shfl_xor_sync(0xFFFFFFFF, dot_dxhat_xhat, offset);
	}
	for (int j = lane; j < C; j += 32)
	{
		const float dxhat = doi[j] * gamma[j];
		dxi[j] = inv_std * (dxhat - (sum_dxhat + xhi[j] * dot_dxhat_xhat) / C);
	}
}

__global__ void transformer_window_partition_kernel(
	const float *input, float *output,
	int B, int C, int Hp, int Wp, int ws)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int nH = Hp / ws;
	int nW = Wp / ws;
	int T = ws * ws;
	int total = B * nH * nW * T * C;
	if (idx >= total) return;

	// Decode index: [win_idx, token, c]
	int c = idx % C;
	int rest = idx / C;
	int token = rest % T;
	int win_idx = rest / T;

	int b = win_idx / (nH * nW);
	int win_rem = win_idx % (nH * nW);
	int wh = win_rem / nW;
	int ww = win_rem % nW;

	int i = token / ws;
	int j = token % ws;
	int y = wh * ws + i;
	int x = ww * ws + j;

	int in_idx = ((b * C + c) * Hp + y) * Wp + x;
	output[idx] = input[in_idx];
}

__global__ void transformer_window_unpartition_kernel(
	const float *input, float *output,
	int B, int C, int Hp, int Wp, int ws)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int nH = Hp / ws;
	int nW = Wp / ws;
	int T = ws * ws;
	int total = B * nH * nW * T * C;
	if (idx >= total) return;

	int c = idx % C;
	int rest = idx / C;
	int token = rest % T;
	int win_idx = rest / T;

	int b = win_idx / (nH * nW);
	int win_rem = win_idx % (nH * nW);
	int wh = win_rem / nW;
	int ww = win_rem % nW;

	int i = token / ws;
	int j = token % ws;
	int y = wh * ws + i;
	int x = ww * ws + j;

	int out_idx = ((b * C + c) * Hp + y) * Wp + x;
	output[out_idx] = input[idx];
}

__global__ void transformer_cyclic_shift_kernel(
	const float *input, float *output,
	int B, int C, int H, int W, int dy, int dx)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int total = B * C * H * W;
	if (idx >= total) return;

	int x = idx % W;
	int rest = idx / W;
	int y = rest % H;
	rest = rest / H;
	int c = rest % C;
	int b = rest / C;

	int sy = ((y - dy) % H + H) % H;
	int sx = ((x - dx) % W + W) % W;

	int in_idx = ((b * C + c) * H + sy) * W + sx;
	output[idx] = input[in_idx];
}

__global__ void transformer_add_bias_kernel(float *x, const float *bias, int M, int N)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= M * N) return;
	int j = idx % N;
	x[idx] += bias[j];
}

__global__ void transformer_affine_from_xhat_kernel(
	const float *xhat, float *out, const float *gamma, const float *beta, int total_tokens, int C)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= total_tokens * C) return;
	int c = idx % C;
	out[idx] = xhat[idx] * gamma[c] + beta[c];
}

__global__ void transformer_attention_softmax_kernel(
	float *scores, const float *mask, const float *rel_pos_bias, const int *rel_pos_index,
	int total_windows, int nW_spatial, int heads, int T, int bias_table_stride)
{
	// One thread per (window, head, query_token) triple
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int total = total_windows * heads * T;
	if (idx >= total) return;

	int t = idx % T;
	int rest = idx / T;
	int h = rest % heads;
	int win = rest / heads;

	int win_in_batch = win % nW_spatial;

	float *row = scores + ((win * heads + h) * T + t) * T;
	const float clip = 20.0f;

	// Add relative position bias and mask
	for (int j = 0; j < T; j++)
	{
		int bias_idx = h * bias_table_stride + rel_pos_index[t * T + j];
		row[j] += rel_pos_bias[bias_idx];
		if (mask) row[j] += mask[(win_in_batch * T + t) * T + j];
		float v = row[j];
		if (isnan(v))
		{
			row[j] = 0.0f;
		}
		else if (isinf(v))
		{
			row[j] = (v > 0.0f) ? clip : -clip;
		}
		else
		{
			row[j] = fminf(clip, fmaxf(-clip, v));
		}
	}

	// Softmax
	float max_val = row[0];
	for (int j = 1; j < T; j++) max_val = fmaxf(max_val, row[j]);
	float sum = 0.0f;
	for (int j = 0; j < T; j++)
	{
		row[j] = expf(row[j] - max_val);
		sum += row[j];
	}
	if (!isfinite(sum) || sum <= 0.0f)
	{
		const float uniform = 1.0f / T;
		for (int j = 0; j < T; j++) row[j] = uniform;
		return;
	}
	float inv_sum = 1.0f / (sum + 1e-9f);
	for (int j = 0; j < T; j++) row[j] *= inv_sum;
}

__global__ void transformer_add_pos_bias_mask_kernel(
	float *scores, const float *mask, const float *rel_pos_bias, const int *rel_pos_index,
	int total_windows, int nW_spatial, int heads, int T, int bias_table_stride)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int total = total_windows * heads * T;
	if (idx >= total) return;

	int t = idx % T;
	int rest = idx / T;
	int h = rest % heads;
	int win = rest / heads;

	int win_in_batch = win % nW_spatial;

	float *row = scores + ((win * heads + h) * T + t) * T;
	const float clip = 20.0f;

	for (int j = 0; j < T; j++)
	{
		int bias_idx = h * bias_table_stride + rel_pos_index[t * T + j];
		row[j] += rel_pos_bias[bias_idx];
		if (mask) row[j] += mask[(win_in_batch * T + t) * T + j];
		float v = row[j];
		if (isnan(v))
		{
			row[j] = 0.0f;
		}
		else if (isinf(v))
		{
			row[j] = (v > 0.0f) ? clip : -clip;
		}
		else
		{
			row[j] = fminf(clip, fmaxf(-clip, v));
		}
	}
}

__global__ void transformer_split_qkv_kernel(
	const float *qkv, float *Q, float *K, float *V,
	int total_windows, int T, int C, int heads, int d)
{
	// Each thread handles one (win, token, head, d_elem)
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int total = total_windows * T * heads * d;
	if (idx >= total) return;

	int dd = idx % d;
	int rest = idx / d;
	int h = rest % heads;
	rest = rest / heads;
	int t = rest % T;
	int win = rest / T;

	int qkv_idx = (win * T + t) * 3 * C + h * d + dd;
	int out_idx = ((win * heads + h) * T + t) * d + dd;

	Q[out_idx] = qkv[qkv_idx];
	K[out_idx] = qkv[qkv_idx + C];
	V[out_idx] = qkv[qkv_idx + 2 * C];
}

__global__ void transformer_merge_heads_kernel(
	const float *attn_result, float *output,
	int total_windows, int T, int C, int heads, int d)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int total = total_windows * T * C;
	if (idx >= total) return;

	int c = idx % C;
	int rest = idx / C;
	int t = rest % T;
	int win = rest / T;

	int h = c / d;
	int dd = c % d;
	int in_idx = ((win * heads + h) * T + t) * d + dd;
	output[idx] = attn_result[in_idx];
}

__global__ void transformer_residual_add_kernel(float *out, const float *residual, int n)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= n) return;
	out[idx] += residual[idx];
}

// ─── mHC residual mixer kernels ─────────────────────────────────────────────
// Two-stream closed-form Sinkhorn projection used at each residual merge.
__device__ __forceinline__ float transformer_mhc_sigmoid_device(float x)
{
	if (!isfinite(x)) return 0.5f;
	if (x >= 0.0f)
	{
		const float z = expf(-x);
		return 1.0f / (1.0f + z);
	}
	const float z = expf(x);
	return z / (1.0f + z);
}

__device__ __forceinline__ void transformer_mhc_coefficients_device(
	const float *params, int site, float &skip_coeff, float &branch_coeff,
	float &p, float &post_skip, float &post_branch)
{
	const int o = site * 6;
	const float z = 0.5f * (params[o + 0] + params[o + 3] - params[o + 1] - params[o + 2]);
	p = transformer_mhc_sigmoid_device(z);
	post_skip = 2.0f * transformer_mhc_sigmoid_device(params[o + 4]);
	post_branch = 2.0f * transformer_mhc_sigmoid_device(params[o + 5]);
	skip_coeff = post_skip * p + post_branch * (1.0f - p);
	branch_coeff = post_skip * (1.0f - p) + post_branch * p;
}

__global__ void transformer_mhc_residual_forward_kernel(
	const float *skip, const float *branch, float *out, const float *params, int site, int n)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= n) return;
	float a, b, p, ps, pb;
	transformer_mhc_coefficients_device(params, site, a, b, p, ps, pb);
	out[idx] = a * skip[idx] + b * branch[idx];
}

__global__ void transformer_mhc_residual_backward_kernel(
	const float *skip, const float *branch, const float *dout,
	float *d_skip, float *d_branch, const float *params, float *updates, int site, int n)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= n) return;

	float a, b, p, post_skip, post_branch;
	transformer_mhc_coefficients_device(params, site, a, b, p, post_skip, post_branch);

	const float g = dout[idx];
	const float s = skip[idx];
	const float br = branch[idx];
	d_skip[idx] = a * g;
	d_branch[idx] = b * g;

	const int o = site * 6;
	const float grad_post_skip = g * (p * s + (1.0f - p) * br);
	const float grad_post_branch = g * ((1.0f - p) * s + p * br);
	const float grad_p = g * (post_skip - post_branch) * (s - br);
	const float dz = 0.5f * grad_p * p * (1.0f - p);
	atomicAdd(&updates[o + 0], dz);
	atomicAdd(&updates[o + 1], -dz);
	atomicAdd(&updates[o + 2], -dz);
	atomicAdd(&updates[o + 3], dz);
	atomicAdd(&updates[o + 4], grad_post_skip * post_skip * (1.0f - 0.5f * post_skip));
	atomicAdd(&updates[o + 5], grad_post_branch * post_branch * (1.0f - 0.5f * post_branch));
}

__global__ void transformer_sum_rows_kernel(const float *x, float *bias_updates, int M, int N)
{
	extern __shared__ float partial[];
	const int j = blockIdx.x;
	if (j >= N) return;

	float sum = 0.0f;
	const int row_stride = gridDim.y * blockDim.x;
	for (int row = blockIdx.y * blockDim.x + threadIdx.x; row < M; row += row_stride)
	{
		sum += x[row * N + j];
	}

	partial[threadIdx.x] = sum;
	__syncthreads();

	for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1)
	{
		if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
		__syncthreads();
	}

	if (threadIdx.x == 0) atomicAdd(&bias_updates[j], partial[0]);
}

static void transformer_sum_rows_gpu(const float *x, float *bias_updates, int M, int N)
{
	const int threads = 256;
	int row_blocks = (M + threads - 1) / threads;
	if (row_blocks < 1) row_blocks = 1;
	if (row_blocks > 1024) row_blocks = 1024;
	dim3 grid(N, row_blocks);
	transformer_sum_rows_kernel<<<grid, threads, threads * sizeof(float)>>>(x, bias_updates, M, N);
	CHECK_CUDA(cudaPeekAtLastError());
}

__global__ void transformer_attention_softmax_backward_kernel(
	const float *scores, float *d_scores, int rows, int T)
{
	int row = blockIdx.x * blockDim.x + threadIdx.x;
	if (row >= rows) return;

	const float *score_row = scores + row * T;
	float *grad_row = d_scores + row * T;

	float dot = 0.0f;
	for (int j = 0; j < T; ++j)
	{
		dot += grad_row[j] * score_row[j];
	}

	for (int j = 0; j < T; ++j)
	{
		grad_row[j] = score_row[j] * (grad_row[j] - dot);
	}
}

__global__ void transformer_rel_pos_bias_backward_kernel(
	const float *d_pre_softmax, float *rel_pos_bias_updates,
	const int *rel_pos_index, int total_windows, int nW_spatial,
	int heads, int T, int bias_table_stride)
{
	// One thread per (window, head, query_token, key_token)
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int total = total_windows * heads * T * T;
	if (idx >= total) return;

	int j = idx % T;
	int rest = idx / T;
	int i = rest % T;
	rest = rest / T;
	int h = rest % heads;
	int win = rest / heads;

	int bias_idx = h * bias_table_stride + rel_pos_index[i * T + j];
	float grad = d_pre_softmax[(win * heads + h) * T * T + i * T + j];
	atomicAdd(&rel_pos_bias_updates[bias_idx], grad);
}

__global__ void transformer_scatter_qkv_grads_kernel(
	const float *dQ, const float *dK, const float *dV, float *d_qkv,
	int total_windows, int T, int C, int heads, int d)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int total = total_windows * T * heads * d;
	if (idx >= total) return;

	int dd = idx % d;
	int rest = idx / d;
	int h = rest % heads;
	rest = rest / heads;
	int t = rest % T;
	int win = rest / T;

	int qkv_idx = (win * T + t) * 3 * C + h * d + dd;
	d_qkv[qkv_idx] = dQ[idx];
	d_qkv[qkv_idx + C] = dK[idx];
	d_qkv[qkv_idx + 2 * C] = dV[idx];
}

__global__ void transformer_extract_heads_kernel(
	const float *tokens, float *heads_out,
	int total_windows, int T, int C, int heads, int d)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int total = total_windows * T * heads * d;
	if (idx >= total) return;

	int dd = idx % d;
	int rest = idx / d;
	int h = rest % heads;
	rest = rest / heads;
	int t = rest % T;
	int win = rest / T;

	heads_out[idx] = tokens[(win * T + t) * C + h * d + dd];
}

__global__ void transformer_pad_kernel(
	const float *input, float *output,
	int B, int C, int H, int W, int Hp, int Wp)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int total = B * C * Hp * Wp;
	if (idx >= total) return;

	int x = idx % Wp;
	int rest = idx / Wp;
	int y = rest % Hp;
	rest = rest / Hp;
	int c = rest % C;
	int b = rest / C;

	if (y < H && x < W)
		output[idx] = input[((b * C + c) * H + y) * W + x];
	else
		output[idx] = 0.0f;
}

__global__ void transformer_crop_kernel(
	const float *input, float *output,
	int B, int C, int H, int W, int Hp, int Wp)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int total = B * C * H * W;
	if (idx >= total) return;

	int x = idx % W;
	int rest = idx / W;
	int y = rest % H;
	rest = rest / H;
	int c = rest % C;
	int b = rest / C;

	output[idx] = input[((b * C + c) * Hp + y) * Wp + x];
}

// ─── GPU forward ──────────────────────────────────────────────────────────────

void forward_transformer_layer_gpu(Darknet::Layer & l, Darknet::NetworkState state)
{
	TAT(TATPARMS);

	const int B = l.batch;
	const int C = l.c;
	const int N = l.n;
	const int H = l.h;
	const int W = l.w;
	const int ws = l.tf_window_size;
	const int Hp = l.tf_pad_h;
	const int Wp = l.tf_pad_w;
	const int T = ws * ws;
	const int nH = Hp / ws;
	const int nW_col = Wp / ws;
	const int nW_spatial = nH * nW_col;
	const int total_windows = B * nW_spatial;
	const int heads = l.tf_heads;
	const int d = l.tf_head_dim;
	const int shift_size = l.tf_shift ? ws / 2 : 0;
	const int ffn_hidden = N * l.tf_ffn_ratio;
	const float scale = 1.0f / sqrtf((float)d);
	const int TRANS_BLOCK = 256;
	const bool use_tensor_op = transformer_use_tensor_op(state);
	const TransformerWorkspaceLayout layout = make_transformer_workspace_layout(B, C, N, Hp, Wp, ws, heads, l.tf_ffn_ratio);

	assert(l.tf_gpu_workspace != nullptr);
	assert(l.tf_gpu_workspace_size >= layout.total);

	float *spatial0 = workspace_ptr(l.tf_gpu_workspace, layout.spatial0);
	float *spatial1 = workspace_ptr(l.tf_gpu_workspace, layout.spatial1);
	float *ln1_out_gpu = workspace_ptr(l.tf_gpu_workspace, layout.token_c0);
	float *proj_out_gpu = workspace_ptr(l.tf_gpu_workspace, layout.token_n0);
	float *res1_out_gpu = workspace_ptr(l.tf_gpu_workspace, layout.token_n1);
	float *Q_gpu = workspace_ptr(l.tf_gpu_workspace, layout.head0);
	float *K_gpu = workspace_ptr(l.tf_gpu_workspace, layout.head1);
	float *V_gpu = workspace_ptr(l.tf_gpu_workspace, layout.head2);
	float *attn_result_gpu = workspace_ptr(l.tf_gpu_workspace, layout.head3);
	const int padded_size = B * C * Hp * Wp;
	const int win_token_n = total_windows * T * N;
	const int qkv_head_size = total_windows * heads * T * d;

	float *padded_gpu = spatial0;
	if (Hp == H && Wp == W)
	{
		simple_copy_ongpu(B * C * H * W, state.input, padded_gpu);
	}
	else
	{
		fill_ongpu(padded_size, 0.0f, padded_gpu, 1);
		int num = B * C * Hp * Wp;
		transformer_pad_kernel<<<(num + TRANS_BLOCK - 1) / TRANS_BLOCK, TRANS_BLOCK>>>(
			state.input, padded_gpu, B, C, H, W, Hp, Wp);
		CHECK_CUDA(cudaPeekAtLastError());
	}

	float *shifted_gpu = padded_gpu;
	if (shift_size > 0)
	{
		shifted_gpu = spatial1;
		int num = B * C * Hp * Wp;
		transformer_cyclic_shift_kernel<<<(num + TRANS_BLOCK - 1) / TRANS_BLOCK, TRANS_BLOCK>>>(
			padded_gpu, shifted_gpu, B, C, Hp, Wp, -shift_size, -shift_size);
		CHECK_CUDA(cudaPeekAtLastError());
	}
	check_nan_gpu("forward: after pad/shift", shifted_gpu, padded_size, l.index);

	{
		int num = total_windows * T * C;
		transformer_window_partition_kernel<<<(num + TRANS_BLOCK - 1) / TRANS_BLOCK, TRANS_BLOCK>>>(
			shifted_gpu, l.tf_windowed_input_gpu, B, C, Hp, Wp, ws);
		CHECK_CUDA(cudaPeekAtLastError());
	}
	check_nan_gpu("forward: after window partition", l.tf_windowed_input_gpu, total_windows * T * C, l.index);

	{
		int num = total_windows * T;
		// Warp kernel: 32 threads per token — launch 32*num threads total
		int warp_threads = num * 32;
		transformer_layernorm_forward_kernel<<<(warp_threads + TRANS_BLOCK - 1) / TRANS_BLOCK, TRANS_BLOCK>>>(
			l.tf_windowed_input_gpu, ln1_out_gpu, l.tf_ln1_mean_gpu, l.tf_ln1_var_gpu,
			l.tf_ln1_xhat_gpu, l.tf_ln1_gamma_gpu, l.tf_ln1_beta_gpu, num, C);
		CHECK_CUDA(cudaPeekAtLastError());
	}
	check_nan_gpu("forward: after layernorm1", ln1_out_gpu, total_windows * T * C, l.index);

	const int M_qkv = total_windows * T;
	gemm_ongpu_tensor_op(0, 1, M_qkv, 3 * C, C, 1.0f,
		ln1_out_gpu, C,
		l.weights_gpu, C,
		0.0f,
		l.tf_qkv_out_gpu, 3 * C,
		use_tensor_op);
	{
		int num = M_qkv * 3 * C;
		transformer_add_bias_kernel<<<(num + TRANS_BLOCK - 1) / TRANS_BLOCK, TRANS_BLOCK>>>(
			l.tf_qkv_out_gpu, l.biases_gpu, M_qkv, 3 * C);
		CHECK_CUDA(cudaPeekAtLastError());
	}
	check_nan_gpu("forward: after qkv projection", l.tf_qkv_out_gpu, total_windows * T * 3 * C, l.index);

	{
		int num = total_windows * T * heads * d;
		transformer_split_qkv_kernel<<<(num + TRANS_BLOCK - 1) / TRANS_BLOCK, TRANS_BLOCK>>>(
			l.tf_qkv_out_gpu, Q_gpu, K_gpu, V_gpu,
			total_windows, T, C, heads, d);
		CHECK_CUDA(cudaPeekAtLastError());
	}
	check_nan_gpu("forward: after q split", Q_gpu, qkv_head_size, l.index);
	check_nan_gpu("forward: after k split", K_gpu, qkv_head_size, l.index);
	constrain_ongpu(qkv_head_size, 256.0f, Q_gpu, 1);
	constrain_ongpu(qkv_head_size, 256.0f, K_gpu, 1);

	const int num_batches = total_windows * heads;
	// Q @ K^T for all windows and heads in one batched GEMM
	gemm_ongpu_strided_batched_tensor_op(0, 1, T, T, d, scale,
		Q_gpu, d, (long long)T * d,
		K_gpu, d, (long long)T * d,
		0.0f,
		l.tf_attn_scores_gpu, T, (long long)T * T,
		num_batches,
		use_tensor_op);
	check_nan_gpu("forward: raw attention scores", l.tf_attn_scores_gpu, total_windows * heads * T * T, l.index);

	{
		const int bias_table_stride = (2 * ws - 1) * (2 * ws - 1);
		const float *mask_ptr = l.tf_attn_mask_gpu;
		int num = total_windows * heads * T;
#if defined(CUDNN) && !defined(DARKNET_GPU_ROCM)
		transformer_add_pos_bias_mask_kernel<<<(num + TRANS_BLOCK - 1) / TRANS_BLOCK, TRANS_BLOCK>>>(
			l.tf_attn_scores_gpu, mask_ptr, l.tf_rel_pos_bias_gpu, l.tf_rel_pos_index_gpu,
			total_windows, nW_spatial, heads, T, bias_table_stride);
		CHECK_CUDA(cudaPeekAtLastError());
		transformer_softmax_forward_gpu(l.tf_attn_scores_gpu, total_windows * heads * T, T);
#else
		transformer_attention_softmax_kernel<<<(num + TRANS_BLOCK - 1) / TRANS_BLOCK, TRANS_BLOCK>>>(
			l.tf_attn_scores_gpu, mask_ptr, l.tf_rel_pos_bias_gpu, l.tf_rel_pos_index_gpu,
			total_windows, nW_spatial, heads, T, bias_table_stride);
		CHECK_CUDA(cudaPeekAtLastError());
#endif
	}
	check_nan_gpu("forward: after attention scores", l.tf_attn_scores_gpu, total_windows * heads * T * T, l.index);

	// attn_scores @ V for all windows and heads in one batched GEMM
	gemm_ongpu_strided_batched_tensor_op(0, 0, T, d, T, 1.0f,
		l.tf_attn_scores_gpu, T, (long long)T * T,
		V_gpu, d, (long long)T * d,
		0.0f,
		attn_result_gpu, d, (long long)T * d,
		num_batches,
		use_tensor_op);
	{
		int num = total_windows * T * C;
		transformer_merge_heads_kernel<<<(num + TRANS_BLOCK - 1) / TRANS_BLOCK, TRANS_BLOCK>>>(
			attn_result_gpu, l.tf_attn_out_gpu, total_windows, T, C, heads, d);
		CHECK_CUDA(cudaPeekAtLastError());
	}
	check_nan_gpu("forward: after attention output", l.tf_attn_out_gpu, total_windows * T * C, l.index);

	gemm_ongpu_tensor_op(0, 1, total_windows * T, N, C, 1.0f,
		l.tf_attn_out_gpu, C,
		l.tf_wo_gpu, C,
		0.0f,
		proj_out_gpu, N,
		use_tensor_op);
	{
		int num = win_token_n;
		transformer_add_bias_kernel<<<(num + TRANS_BLOCK - 1) / TRANS_BLOCK, TRANS_BLOCK>>>(
			proj_out_gpu, l.tf_wo_bias_gpu, total_windows * T, N);
		CHECK_CUDA(cudaPeekAtLastError());
	}
	check_nan_gpu("forward: after output projection", proj_out_gpu, win_token_n, l.index);

	simple_copy_ongpu(win_token_n, proj_out_gpu, l.x_gpu);
	if (C == N)
	{
		int num = win_token_n;
		transformer_mhc_residual_forward_kernel<<<(num + TRANS_BLOCK - 1) / TRANS_BLOCK, TRANS_BLOCK>>>(
			l.tf_windowed_input_gpu, l.x_gpu, res1_out_gpu, l.scales_gpu, 0, num);
		CHECK_CUDA(cudaPeekAtLastError());
	}
	else
	{
		gemm_ongpu_tensor_op(0, 1, total_windows * T, N, C, 1.0f,
			l.tf_windowed_input_gpu, C,
			l.tf_res_proj_gpu, C,
			0.0f,
			res1_out_gpu, N,
			use_tensor_op);
		int num = win_token_n;
		transformer_mhc_residual_forward_kernel<<<(num + TRANS_BLOCK - 1) / TRANS_BLOCK, TRANS_BLOCK>>>(
			res1_out_gpu, l.x_gpu, res1_out_gpu, l.scales_gpu, 0, num);
		CHECK_CUDA(cudaPeekAtLastError());
	}
	check_nan_gpu("forward: after residual1", res1_out_gpu, win_token_n, l.index);
	simple_copy_ongpu(win_token_n, res1_out_gpu, l.tf_pre_res2_gpu);

	float *ln2_out_gpu = proj_out_gpu;
	{
		int num = total_windows * T;
		int warp_threads = num * 32;
		transformer_layernorm_forward_kernel<<<(warp_threads + TRANS_BLOCK - 1) / TRANS_BLOCK, TRANS_BLOCK>>>(
			res1_out_gpu, ln2_out_gpu, l.tf_ln2_mean_gpu, l.tf_ln2_var_gpu,
			l.tf_ln2_xhat_gpu, l.tf_ln2_gamma_gpu, l.tf_ln2_beta_gpu, num, N);
		CHECK_CUDA(cudaPeekAtLastError());
	}
	check_nan_gpu("forward: after layernorm2", ln2_out_gpu, win_token_n, l.index);

	gemm_ongpu_tensor_op(0, 1, total_windows * T, ffn_hidden, N, 1.0f,
		ln2_out_gpu, N,
		l.tf_ffn_w1_gpu, N,
		0.0f,
		l.tf_ffn_hidden_gpu, ffn_hidden,
		use_tensor_op);
	{
		int num = total_windows * T * ffn_hidden;
		transformer_add_bias_kernel<<<(num + TRANS_BLOCK - 1) / TRANS_BLOCK, TRANS_BLOCK>>>(
			l.tf_ffn_hidden_gpu, l.tf_ffn_b1_gpu, total_windows * T, ffn_hidden);
		CHECK_CUDA(cudaPeekAtLastError());
	}
	simple_copy_ongpu(total_windows * T * ffn_hidden, l.tf_ffn_hidden_gpu, l.activation_input_gpu);
	activate_array_ongpu(l.tf_ffn_hidden_gpu, total_windows * T * ffn_hidden, l.activation);
	check_nan_gpu("forward: after ffn hidden", l.tf_ffn_hidden_gpu, total_windows * T * ffn_hidden, l.index);

	float *ffn_out_gpu = proj_out_gpu;
	gemm_ongpu_tensor_op(0, 1, total_windows * T, N, ffn_hidden, 1.0f,
		l.tf_ffn_hidden_gpu, ffn_hidden,
		l.tf_ffn_w2_gpu, ffn_hidden,
		0.0f,
		ffn_out_gpu, N,
		use_tensor_op);
	{
		int num = win_token_n;
		transformer_add_bias_kernel<<<(num + TRANS_BLOCK - 1) / TRANS_BLOCK, TRANS_BLOCK>>>(
			ffn_out_gpu, l.tf_ffn_b2_gpu, total_windows * T, N);
		CHECK_CUDA(cudaPeekAtLastError());
		simple_copy_ongpu(num, ffn_out_gpu, l.x_norm_gpu);
		transformer_mhc_residual_forward_kernel<<<(num + TRANS_BLOCK - 1) / TRANS_BLOCK, TRANS_BLOCK>>>(
			res1_out_gpu, l.x_norm_gpu, ffn_out_gpu, l.scales_gpu, 1, num);
		CHECK_CUDA(cudaPeekAtLastError());
	}
	check_nan_gpu("forward: after ffn output + residual", ffn_out_gpu, win_token_n, l.index);

	if (shift_size == 0 && Hp == H && Wp == W)
	{
		int num = total_windows * T * N;
		transformer_window_unpartition_kernel<<<(num + TRANS_BLOCK - 1) / TRANS_BLOCK, TRANS_BLOCK>>>(
			ffn_out_gpu, l.output_gpu, B, N, Hp, Wp, ws);
		CHECK_CUDA(cudaPeekAtLastError());
	}
	else
	{
		float *unpartitioned_gpu = spatial0;
		{
			int num = total_windows * T * N;
			transformer_window_unpartition_kernel<<<(num + TRANS_BLOCK - 1) / TRANS_BLOCK, TRANS_BLOCK>>>(
				ffn_out_gpu, unpartitioned_gpu, B, N, Hp, Wp, ws);
			CHECK_CUDA(cudaPeekAtLastError());
		}

		float *unshifted_gpu = unpartitioned_gpu;
		if (shift_size > 0)
		{
			unshifted_gpu = spatial1;
			int num = B * N * Hp * Wp;
			transformer_cyclic_shift_kernel<<<(num + TRANS_BLOCK - 1) / TRANS_BLOCK, TRANS_BLOCK>>>(
				unpartitioned_gpu, unshifted_gpu, B, N, Hp, Wp, shift_size, shift_size);
			CHECK_CUDA(cudaPeekAtLastError());
		}

		if (Hp == H && Wp == W)
		{
			simple_copy_ongpu(B * N * H * W, unshifted_gpu, l.output_gpu);
		}
		else
		{
			int num = B * N * H * W;
			transformer_crop_kernel<<<(num + TRANS_BLOCK - 1) / TRANS_BLOCK, TRANS_BLOCK>>>(
				unshifted_gpu, l.output_gpu, B, N, H, W, Hp, Wp);
			CHECK_CUDA(cudaPeekAtLastError());
		}
	}
	constrain_ongpu(B * N * H * W, 100.0f, l.output_gpu, 1);
	check_nan_gpu("forward: final output", l.output_gpu, B * N * H * W, l.index);
}

// ─── GPU backward (native) ───────────────────────────────────────────────────

void backward_transformer_layer_gpu(Darknet::Layer & l, Darknet::NetworkState state)
{
	TAT(TATPARMS);

	const int B = l.batch;
	const int C = l.c;
	const int N = l.n;
	const int H = l.h;
	const int W = l.w;
	const int ws = l.tf_window_size;
	const int Hp = l.tf_pad_h;
	const int Wp = l.tf_pad_w;
	const int T = ws * ws;
	const int nH = Hp / ws;
	const int nW_col = Wp / ws;
	const int nW_spatial = nH * nW_col;
	const int total_windows = B * nW_spatial;
	const int heads = l.tf_heads;
	const int d = l.tf_head_dim;
	const int shift_size = l.tf_shift ? ws / 2 : 0;
	const int ffn_hidden = N * l.tf_ffn_ratio;
	const float scale = 1.0f / sqrtf((float)d);
	const int TRANS_BLOCK = 256;
	const bool use_tensor_op = transformer_use_tensor_op(state);
	const TransformerWorkspaceLayout layout = make_transformer_workspace_layout(B, C, N, Hp, Wp, ws, heads, l.tf_ffn_ratio);

	assert(l.tf_gpu_workspace != nullptr);
	assert(l.tf_gpu_workspace_size >= layout.total);

	const int padded_size = B * C * Hp * Wp;
	const int win_token_n = total_windows * T * N;
	const int win_token_c = total_windows * T * C;
	const int M_ffn = total_windows * T;
	const int num_batches = total_windows * heads;
	const int qkv_head_size = total_windows * heads * T * d;
	const int bias_table_len = (2 * ws - 1) * (2 * ws - 1);

	float *spatial0 = workspace_ptr(l.tf_gpu_workspace, layout.spatial0);
	float *spatial1 = workspace_ptr(l.tf_gpu_workspace, layout.spatial1);
	float *d_windows_gpu = workspace_ptr(l.tf_gpu_workspace, layout.token_n0);
	float *ln2_out_gpu = workspace_ptr(l.tf_gpu_workspace, layout.token_n1);
	float *d_pre_ln2_gpu = workspace_ptr(l.tf_gpu_workspace, layout.token_n2);
	float *d_windowed_input_gpu = workspace_ptr(l.tf_gpu_workspace, layout.token_c0);
	float *d_attn_out_gpu = workspace_ptr(l.tf_gpu_workspace, layout.token_c1);
	float *dK_gpu = workspace_ptr(l.tf_gpu_workspace, layout.token_c2);
	float *d_qkv_gpu = workspace_ptr(l.tf_gpu_workspace, layout.token_3c);
	float *d_ffn_hidden_gpu = workspace_ptr(l.tf_gpu_workspace, layout.ffn);
	float *Q_gpu = workspace_ptr(l.tf_gpu_workspace, layout.head0);
	float *K_gpu = workspace_ptr(l.tf_gpu_workspace, layout.head1);
	float *V_gpu = workspace_ptr(l.tf_gpu_workspace, layout.head2);
	float *head_tmp_gpu = workspace_ptr(l.tf_gpu_workspace, layout.head3);
	float *dV_gpu = workspace_ptr(l.tf_gpu_workspace, layout.head4);
	float *d_scores_gpu = workspace_ptr(l.tf_gpu_workspace, layout.scores);

	reset_nan_and_inf(l.delta_gpu, B * N * H * W);
	constrain_ongpu(B * N * H * W, 1.0f, l.delta_gpu, 1);

	float *dout_padded_gpu = spatial0;
	if (Hp == H && Wp == W)
	{
		simple_copy_ongpu(B * N * H * W, l.delta_gpu, dout_padded_gpu);
	}
	else
	{
		fill_ongpu(B * N * Hp * Wp, 0.0f, dout_padded_gpu, 1);
		int num = B * N * Hp * Wp;
		transformer_pad_kernel<<<(num + TRANS_BLOCK - 1) / TRANS_BLOCK, TRANS_BLOCK>>>(
			l.delta_gpu, dout_padded_gpu, B, N, H, W, Hp, Wp);
		CHECK_CUDA(cudaPeekAtLastError());
	}
	check_nan_gpu("backward: initial delta", dout_padded_gpu, B * N * Hp * Wp, l.index);

	float *dout_shifted_gpu = dout_padded_gpu;
	if (shift_size > 0)
	{
		dout_shifted_gpu = spatial1;
		int num = B * N * Hp * Wp;
		transformer_cyclic_shift_kernel<<<(num + TRANS_BLOCK - 1) / TRANS_BLOCK, TRANS_BLOCK>>>(
			dout_padded_gpu, dout_shifted_gpu, B, N, Hp, Wp, -shift_size, -shift_size);
		CHECK_CUDA(cudaPeekAtLastError());
	}

	{
		int num = win_token_n;
		transformer_window_partition_kernel<<<(num + TRANS_BLOCK - 1) / TRANS_BLOCK, TRANS_BLOCK>>>(
			dout_shifted_gpu, d_windows_gpu, B, N, Hp, Wp, ws);
		CHECK_CUDA(cudaPeekAtLastError());
	}
	check_nan_gpu("backward: after window partition", d_windows_gpu, win_token_n, l.index);

	float *d_ffn_out_gpu = ln2_out_gpu;
	{
		int num = win_token_n;
		transformer_mhc_residual_backward_kernel<<<(num + TRANS_BLOCK - 1) / TRANS_BLOCK, TRANS_BLOCK>>>(
			l.tf_pre_res2_gpu, l.x_norm_gpu, d_windows_gpu,
			d_windows_gpu, d_ffn_out_gpu, l.scales_gpu, l.scale_updates_gpu, 1, num);
		CHECK_CUDA(cudaPeekAtLastError());
	}
	gemm_ongpu_tensor_op(0, 0, M_ffn, ffn_hidden, N, 1.0f,
		d_ffn_out_gpu, N,
		l.tf_ffn_w2_gpu, ffn_hidden,
		0.0f,
		d_ffn_hidden_gpu, ffn_hidden,
		use_tensor_op);
	gemm_ongpu_tensor_op(1, 0, N, ffn_hidden, M_ffn, 1.0f,
		d_ffn_out_gpu, N,
		l.tf_ffn_hidden_gpu, ffn_hidden,
		1.0f,
		l.tf_ffn_w2_updates_gpu, ffn_hidden,
		use_tensor_op);
	{
		transformer_sum_rows_gpu(d_ffn_out_gpu, l.tf_ffn_b2_updates_gpu, M_ffn, N);
	}
	gradient_array_ongpu(l.activation_input_gpu, M_ffn * ffn_hidden, l.activation, d_ffn_hidden_gpu);
	check_nan_gpu("backward: after ffn hidden grad", d_ffn_hidden_gpu, M_ffn * ffn_hidden, l.index);

	{
		int num = win_token_n;
		transformer_affine_from_xhat_kernel<<<(num + TRANS_BLOCK - 1) / TRANS_BLOCK, TRANS_BLOCK>>>(
			l.tf_ln2_xhat_gpu, ln2_out_gpu, l.tf_ln2_gamma_gpu, l.tf_ln2_beta_gpu, total_windows * T, N);
		CHECK_CUDA(cudaPeekAtLastError());
	}
	gemm_ongpu_tensor_op(1, 0, ffn_hidden, N, M_ffn, 1.0f,
		d_ffn_hidden_gpu, ffn_hidden,
		ln2_out_gpu, N,
		1.0f,
		l.tf_ffn_w1_updates_gpu, N,
		use_tensor_op);
	{
		transformer_sum_rows_gpu(d_ffn_hidden_gpu, l.tf_ffn_b1_updates_gpu, M_ffn, ffn_hidden);
	}
	gemm_ongpu_tensor_op(0, 0, M_ffn, N, ffn_hidden, 1.0f,
		d_ffn_hidden_gpu, ffn_hidden,
		l.tf_ffn_w1_gpu, N,
		0.0f,
		ln2_out_gpu, N,
		use_tensor_op);
	check_nan_gpu("backward: before layernorm2 backward", ln2_out_gpu, win_token_n, l.index);

	{
		int num = M_ffn;
		const int warp_threads = num * 32;
		transformer_layernorm_backward_kernel<<<(warp_threads + TRANS_BLOCK - 1) / TRANS_BLOCK, TRANS_BLOCK>>>(
			ln2_out_gpu, l.tf_ln2_xhat_gpu, l.tf_ln2_var_gpu, l.tf_ln2_gamma_gpu,
			d_pre_ln2_gpu, l.tf_ln2_gamma_updates_gpu, l.tf_ln2_beta_updates_gpu, num, N);
		CHECK_CUDA(cudaPeekAtLastError());
	}
	axpy_ongpu(win_token_n, 1.0f, d_windows_gpu, 1, d_pre_ln2_gpu, 1);
	check_nan_gpu("backward: after layernorm2 backward + residual", d_pre_ln2_gpu, win_token_n, l.index);

	float *d_proj_out_gpu = ln2_out_gpu;
	if (C == N)
	{
		int num = win_token_n;
		transformer_mhc_residual_backward_kernel<<<(num + TRANS_BLOCK - 1) / TRANS_BLOCK, TRANS_BLOCK>>>(
			l.tf_windowed_input_gpu, l.x_gpu, d_pre_ln2_gpu,
			d_windowed_input_gpu, d_proj_out_gpu, l.scales_gpu, l.scale_updates_gpu, 0, num);
		CHECK_CUDA(cudaPeekAtLastError());
	}
	else
	{
		gemm_ongpu_tensor_op(0, 1, M_ffn, N, C, 1.0f,
			l.tf_windowed_input_gpu, C,
			l.tf_res_proj_gpu, C,
			0.0f,
			l.x_norm_gpu, N,
			use_tensor_op);
		int num = win_token_n;
		transformer_mhc_residual_backward_kernel<<<(num + TRANS_BLOCK - 1) / TRANS_BLOCK, TRANS_BLOCK>>>(
			l.x_norm_gpu, l.x_gpu, d_pre_ln2_gpu,
			d_pre_ln2_gpu, d_proj_out_gpu, l.scales_gpu, l.scale_updates_gpu, 0, num);
		CHECK_CUDA(cudaPeekAtLastError());
		gemm_ongpu_tensor_op(0, 0, M_ffn, C, N, 1.0f,
			d_pre_ln2_gpu, N,
			l.tf_res_proj_gpu, C,
			0.0f,
			d_windowed_input_gpu, C,
			use_tensor_op);
		gemm_ongpu_tensor_op(1, 0, N, C, M_ffn, 1.0f,
			d_pre_ln2_gpu, N,
			l.tf_windowed_input_gpu, C,
			1.0f,
			l.tf_res_proj_updates_gpu, C,
			use_tensor_op);
	}

	gemm_ongpu_tensor_op(0, 0, M_ffn, C, N, 1.0f,
		d_proj_out_gpu, N,
		l.tf_wo_gpu, C,
		0.0f,
		d_attn_out_gpu, C,
		use_tensor_op);
	check_nan_gpu("backward: after output projection backward", d_attn_out_gpu, win_token_c, l.index);
	gemm_ongpu_tensor_op(1, 0, N, C, M_ffn, 1.0f,
		d_proj_out_gpu, N,
		l.tf_attn_out_gpu, C,
		1.0f,
		l.tf_wo_updates_gpu, C,
		use_tensor_op);
	{
		transformer_sum_rows_gpu(d_proj_out_gpu, l.tf_wo_bias_updates_gpu, M_ffn, N);
	}

	{
		int num = total_windows * T * heads * d;
		transformer_split_qkv_kernel<<<(num + TRANS_BLOCK - 1) / TRANS_BLOCK, TRANS_BLOCK>>>(
			l.tf_qkv_out_gpu, Q_gpu, K_gpu, V_gpu,
			total_windows, T, C, heads, d);
		CHECK_CUDA(cudaPeekAtLastError());
		transformer_extract_heads_kernel<<<(num + TRANS_BLOCK - 1) / TRANS_BLOCK, TRANS_BLOCK>>>(
			d_attn_out_gpu, head_tmp_gpu,
			total_windows, T, C, heads, d);
		CHECK_CUDA(cudaPeekAtLastError());
	}
	constrain_ongpu(qkv_head_size, 256.0f, Q_gpu, 1);
	constrain_ongpu(qkv_head_size, 256.0f, K_gpu, 1);

	// dV = scores^T @ d_head, d_scores = d_head @ V^T — batched
	gemm_ongpu_strided_batched_tensor_op(1, 0, T, d, T, 1.0f,
		l.tf_attn_scores_gpu, T, (long long)T * T,
		head_tmp_gpu, d, (long long)T * d,
		0.0f,
		dV_gpu, d, (long long)T * d,
		num_batches,
		use_tensor_op);
	gemm_ongpu_strided_batched_tensor_op(0, 1, T, T, d, 1.0f,
		head_tmp_gpu, d, (long long)T * d,
		V_gpu, d, (long long)T * d,
		0.0f,
		d_scores_gpu, T, (long long)T * T,
		num_batches,
		use_tensor_op);
#if defined(CUDNN) && !defined(DARKNET_GPU_ROCM)
	transformer_softmax_backward_gpu(l.tf_attn_scores_gpu, d_scores_gpu, num_batches * T, T);
#else
	{
		int num = num_batches * T;
		transformer_attention_softmax_backward_kernel<<<(num + TRANS_BLOCK - 1) / TRANS_BLOCK, TRANS_BLOCK>>>(
			l.tf_attn_scores_gpu, d_scores_gpu, num, T);
		CHECK_CUDA(cudaPeekAtLastError());
	}
#endif
	{
		int num = total_windows * heads * T * T;
		transformer_rel_pos_bias_backward_kernel<<<(num + TRANS_BLOCK - 1) / TRANS_BLOCK, TRANS_BLOCK>>>(
			d_scores_gpu, l.tf_rel_pos_bias_updates_gpu,
			l.tf_rel_pos_index_gpu, total_windows, nW_spatial,
			heads, T, bias_table_len);
		CHECK_CUDA(cudaPeekAtLastError());
	}

	float *dQ_gpu = head_tmp_gpu;
	// dQ = d_scores @ K, dK = d_scores^T @ Q — batched
	gemm_ongpu_strided_batched_tensor_op(0, 0, T, d, T, scale,
		d_scores_gpu, T, (long long)T * T,
		K_gpu, d, (long long)T * d,
		0.0f,
		dQ_gpu, d, (long long)T * d,
		num_batches,
		use_tensor_op);
	gemm_ongpu_strided_batched_tensor_op(1, 0, T, d, T, scale,
		d_scores_gpu, T, (long long)T * T,
		Q_gpu, d, (long long)T * d,
		0.0f,
		dK_gpu, d, (long long)T * d,
		num_batches,
		use_tensor_op);

	{
		int num = total_windows * T * heads * d;
		transformer_scatter_qkv_grads_kernel<<<(num + TRANS_BLOCK - 1) / TRANS_BLOCK, TRANS_BLOCK>>>(
			dQ_gpu, dK_gpu, dV_gpu, d_qkv_gpu,
			total_windows, T, C, heads, d);
		CHECK_CUDA(cudaPeekAtLastError());
	}
	constrain_ongpu(total_windows * T * 3 * C, 100.0f, d_qkv_gpu, 1);
	check_nan_gpu("backward: after attention backward", d_qkv_gpu, total_windows * T * 3 * C, l.index);
	{
		transformer_sum_rows_gpu(d_qkv_gpu, l.bias_updates_gpu, M_ffn, 3 * C);
	}

	float *ln1_out_gpu = d_attn_out_gpu;
	{
		int num = win_token_c;
		transformer_affine_from_xhat_kernel<<<(num + TRANS_BLOCK - 1) / TRANS_BLOCK, TRANS_BLOCK>>>(
			l.tf_ln1_xhat_gpu, ln1_out_gpu, l.tf_ln1_gamma_gpu, l.tf_ln1_beta_gpu, total_windows * T, C);
		CHECK_CUDA(cudaPeekAtLastError());
	}
	gemm_ongpu_tensor_op(1, 0, 3 * C, C, M_ffn, 1.0f,
		d_qkv_gpu, 3 * C,
		ln1_out_gpu, C,
		1.0f,
		l.weight_updates_gpu, C,
		use_tensor_op);
	gemm_ongpu_tensor_op(0, 0, M_ffn, C, 3 * C, 1.0f,
		d_qkv_gpu, 3 * C,
		l.weights_gpu, C,
		0.0f,
		ln1_out_gpu, C,
		use_tensor_op);
	check_nan_gpu("backward: after qkv weight backward", ln1_out_gpu, win_token_c, l.index);

	float *d_pre_ln1_gpu = dK_gpu;
	{
		int num = M_ffn;
		const int warp_threads = num * 32;
		transformer_layernorm_backward_kernel<<<(warp_threads + TRANS_BLOCK - 1) / TRANS_BLOCK, TRANS_BLOCK>>>(
			ln1_out_gpu, l.tf_ln1_xhat_gpu, l.tf_ln1_var_gpu, l.tf_ln1_gamma_gpu,
			d_pre_ln1_gpu, l.tf_ln1_gamma_updates_gpu, l.tf_ln1_beta_updates_gpu, num, C);
		CHECK_CUDA(cudaPeekAtLastError());
	}
	axpy_ongpu(win_token_c, 1.0f, d_windowed_input_gpu, 1, d_pre_ln1_gpu, 1);
	check_nan_gpu("backward: after layernorm1 backward", d_pre_ln1_gpu, win_token_c, l.index);

	if (state.delta)
	{
		float *d_shifted_gpu = spatial0;
		fill_ongpu(padded_size, 0.0f, d_shifted_gpu, 1);
		{
			int num = win_token_c;
			transformer_window_unpartition_kernel<<<(num + TRANS_BLOCK - 1) / TRANS_BLOCK, TRANS_BLOCK>>>(
				d_pre_ln1_gpu, d_shifted_gpu, B, C, Hp, Wp, ws);
			CHECK_CUDA(cudaPeekAtLastError());
		}

		float *d_padded_gpu = d_shifted_gpu;
		if (shift_size > 0)
		{
			d_padded_gpu = spatial1;
			int num = padded_size;
			transformer_cyclic_shift_kernel<<<(num + TRANS_BLOCK - 1) / TRANS_BLOCK, TRANS_BLOCK>>>(
				d_shifted_gpu, d_padded_gpu, B, C, Hp, Wp, shift_size, shift_size);
			CHECK_CUDA(cudaPeekAtLastError());
		}

		if (Hp == H && Wp == W)
		{
			reset_nan_and_inf(d_padded_gpu, B * C * H * W);
			constrain_ongpu(B * C * H * W, 1.0f, d_padded_gpu, 1);
			axpy_ongpu(B * C * H * W, 1.0f, d_padded_gpu, 1, state.delta, 1);
		}
		else
		{
			float *d_cropped_gpu = d_attn_out_gpu;
			int num = B * C * H * W;
			transformer_crop_kernel<<<(num + TRANS_BLOCK - 1) / TRANS_BLOCK, TRANS_BLOCK>>>(
				d_padded_gpu, d_cropped_gpu, B, C, H, W, Hp, Wp);
			CHECK_CUDA(cudaPeekAtLastError());
			reset_nan_and_inf(d_cropped_gpu, num);
			constrain_ongpu(num, 1.0f, d_cropped_gpu, 1);
			axpy_ongpu(B * C * H * W, 1.0f, d_cropped_gpu, 1, state.delta, 1);
		}
		check_nan_gpu("backward: propagated state.delta", state.delta, B * C * H * W, l.index);
	}
}

// ─── GPU update ───────────────────────────────────────────────────────────────

void update_transformer_layer_gpu(Darknet::Layer & l, int batch, float learning_rate_init, float momentum, float decay, float loss_scale)
{
	TAT(TATPARMS);

	const float lr = learning_rate_init * l.learning_rate_scale;
	const int C = l.c;
	const int N = l.n;
	const int ffn_hidden = N * l.tf_ffn_ratio;
	const int heads = l.tf_heads;
	const int ws = l.tf_window_size;
	const int bias_table_len = (2 * ws - 1) * (2 * ws - 1);

	reset_nan_and_inf(l.weight_updates_gpu, l.nweights);
	reset_nan_and_inf(l.bias_updates_gpu, 3 * C);
	reset_nan_and_inf(l.tf_wo_updates_gpu, N * C);
	reset_nan_and_inf(l.tf_wo_bias_updates_gpu, N);
	reset_nan_and_inf(l.tf_ln1_gamma_updates_gpu, C);
	reset_nan_and_inf(l.tf_ln1_beta_updates_gpu, C);
	reset_nan_and_inf(l.tf_ln2_gamma_updates_gpu, N);
	reset_nan_and_inf(l.tf_ln2_beta_updates_gpu, N);
	reset_nan_and_inf(l.tf_ffn_w1_updates_gpu, ffn_hidden * N);
	reset_nan_and_inf(l.tf_ffn_b1_updates_gpu, ffn_hidden);
	reset_nan_and_inf(l.tf_ffn_w2_updates_gpu, N * ffn_hidden);
	reset_nan_and_inf(l.tf_ffn_b2_updates_gpu, N);
	reset_nan_and_inf(l.tf_rel_pos_bias_updates_gpu, heads * bias_table_len);
	reset_nan_and_inf(l.scale_updates_gpu, TRANSFORMER_MHC_PARAM_COUNT);

	if (loss_scale != 1.0f)
	{
		scal_ongpu(l.nweights, 1.0f / loss_scale, l.weight_updates_gpu, 1);
		scal_ongpu(3 * C, 1.0f / loss_scale, l.bias_updates_gpu, 1);
		scal_ongpu(N * C, 1.0f / loss_scale, l.tf_wo_updates_gpu, 1);
		scal_ongpu(N, 1.0f / loss_scale, l.tf_wo_bias_updates_gpu, 1);
		scal_ongpu(C, 1.0f / loss_scale, l.tf_ln1_gamma_updates_gpu, 1);
		scal_ongpu(C, 1.0f / loss_scale, l.tf_ln1_beta_updates_gpu, 1);
		scal_ongpu(N, 1.0f / loss_scale, l.tf_ln2_gamma_updates_gpu, 1);
		scal_ongpu(N, 1.0f / loss_scale, l.tf_ln2_beta_updates_gpu, 1);
		scal_ongpu(ffn_hidden * N, 1.0f / loss_scale, l.tf_ffn_w1_updates_gpu, 1);
		scal_ongpu(ffn_hidden, 1.0f / loss_scale, l.tf_ffn_b1_updates_gpu, 1);
		scal_ongpu(N * ffn_hidden, 1.0f / loss_scale, l.tf_ffn_w2_updates_gpu, 1);
		scal_ongpu(N, 1.0f / loss_scale, l.tf_ffn_b2_updates_gpu, 1);
		scal_ongpu(heads * bias_table_len, 1.0f / loss_scale, l.tf_rel_pos_bias_updates_gpu, 1);
		scal_ongpu(TRANSFORMER_MHC_PARAM_COUNT, 1.0f / loss_scale, l.scale_updates_gpu, 1);
	}

	// ── Gradient norm clipping (L2) ──
	// Compute global L2 norm across all parameter update buffers, then scale
	// if it exceeds max_grad_norm. Preserves gradient direction unlike per-element clamping.
	{
		const float max_grad_norm = 5.0f;
		cublasHandle_t handle = blas_handle();
		float global_norm_sq = 0.0f;
		float partial_norm = 0.0f;

		auto accum_norm = [&](float *buf, int n)
		{
			cublasSnrm2(handle, n, buf, 1, &partial_norm);
			global_norm_sq += partial_norm * partial_norm;
		};

		accum_norm(l.weight_updates_gpu, l.nweights);
		accum_norm(l.bias_updates_gpu, 3 * C);
		accum_norm(l.tf_wo_updates_gpu, N * C);
		accum_norm(l.tf_wo_bias_updates_gpu, N);
		accum_norm(l.tf_ln1_gamma_updates_gpu, C);
		accum_norm(l.tf_ln1_beta_updates_gpu, C);
		accum_norm(l.tf_ln2_gamma_updates_gpu, N);
		accum_norm(l.tf_ln2_beta_updates_gpu, N);
		accum_norm(l.tf_ffn_w1_updates_gpu, ffn_hidden * N);
		accum_norm(l.tf_ffn_b1_updates_gpu, ffn_hidden);
		accum_norm(l.tf_ffn_w2_updates_gpu, N * ffn_hidden);
		accum_norm(l.tf_ffn_b2_updates_gpu, N);
		accum_norm(l.tf_rel_pos_bias_updates_gpu, heads * bias_table_len);
		accum_norm(l.scale_updates_gpu, TRANSFORMER_MHC_PARAM_COUNT);
		if (C != N && l.tf_res_proj_updates_gpu)
			accum_norm(l.tf_res_proj_updates_gpu, N * C);

		const float global_norm = sqrtf(global_norm_sq);
		if (global_norm > max_grad_norm)
		{
			const float clip_coef = max_grad_norm / global_norm;
			scal_ongpu(l.nweights, clip_coef, l.weight_updates_gpu, 1);
			scal_ongpu(3 * C, clip_coef, l.bias_updates_gpu, 1);
			scal_ongpu(N * C, clip_coef, l.tf_wo_updates_gpu, 1);
			scal_ongpu(N, clip_coef, l.tf_wo_bias_updates_gpu, 1);
			scal_ongpu(C, clip_coef, l.tf_ln1_gamma_updates_gpu, 1);
			scal_ongpu(C, clip_coef, l.tf_ln1_beta_updates_gpu, 1);
			scal_ongpu(N, clip_coef, l.tf_ln2_gamma_updates_gpu, 1);
			scal_ongpu(N, clip_coef, l.tf_ln2_beta_updates_gpu, 1);
			scal_ongpu(ffn_hidden * N, clip_coef, l.tf_ffn_w1_updates_gpu, 1);
			scal_ongpu(ffn_hidden, clip_coef, l.tf_ffn_b1_updates_gpu, 1);
			scal_ongpu(N * ffn_hidden, clip_coef, l.tf_ffn_w2_updates_gpu, 1);
			scal_ongpu(N, clip_coef, l.tf_ffn_b2_updates_gpu, 1);
			scal_ongpu(heads * bias_table_len, clip_coef, l.tf_rel_pos_bias_updates_gpu, 1);
			scal_ongpu(TRANSFORMER_MHC_PARAM_COUNT, clip_coef, l.scale_updates_gpu, 1);
			if (C != N && l.tf_res_proj_updates_gpu)
				scal_ongpu(N * C, clip_coef, l.tf_res_proj_updates_gpu, 1);
		}
	}

	// QKV weights
	axpy_ongpu(l.nweights, -decay * batch, l.weights_gpu, 1, l.weight_updates_gpu, 1);
	axpy_ongpu(l.nweights, lr / batch, l.weight_updates_gpu, 1, l.weights_gpu, 1);
	scal_ongpu(l.nweights, momentum, l.weight_updates_gpu, 1);

	// QKV biases
	axpy_ongpu(3 * C, lr / batch, l.bias_updates_gpu, 1, l.biases_gpu, 1);
	scal_ongpu(3 * C, momentum, l.bias_updates_gpu, 1);

	// Output projection
	{
		const int count = N * C;
		axpy_ongpu(count, -decay * batch, l.tf_wo_gpu, 1, l.tf_wo_updates_gpu, 1);
		axpy_ongpu(count, lr / batch, l.tf_wo_updates_gpu, 1, l.tf_wo_gpu, 1);
		scal_ongpu(count, momentum, l.tf_wo_updates_gpu, 1);
	}
	axpy_ongpu(N, lr / batch, l.tf_wo_bias_updates_gpu, 1, l.tf_wo_bias_gpu, 1);
	scal_ongpu(N, momentum, l.tf_wo_bias_updates_gpu, 1);

	// Residual projection (when C != N)
	if (C != N && l.tf_res_proj_gpu)
	{
		const int count = N * C;
		reset_nan_and_inf(l.tf_res_proj_updates_gpu, count);
		if (loss_scale != 1.0f)
			scal_ongpu(count, 1.0f / loss_scale, l.tf_res_proj_updates_gpu, 1);
		axpy_ongpu(count, -decay * batch, l.tf_res_proj_gpu, 1, l.tf_res_proj_updates_gpu, 1);
		axpy_ongpu(count, lr / batch, l.tf_res_proj_updates_gpu, 1, l.tf_res_proj_gpu, 1);
		scal_ongpu(count, momentum, l.tf_res_proj_updates_gpu, 1);
	}

	// mHC residual mixer
	axpy_ongpu(TRANSFORMER_MHC_PARAM_COUNT, lr / batch, l.scale_updates_gpu, 1, l.scales_gpu, 1);
	constrain_ongpu(TRANSFORMER_MHC_PARAM_COUNT, TRANSFORMER_MHC_PARAM_CLAMP, l.scales_gpu, 1);
	scal_ongpu(TRANSFORMER_MHC_PARAM_COUNT, momentum, l.scale_updates_gpu, 1);

	// LayerNorm 1
	axpy_ongpu(C, lr / batch, l.tf_ln1_gamma_updates_gpu, 1, l.tf_ln1_gamma_gpu, 1);
	scal_ongpu(C, momentum, l.tf_ln1_gamma_updates_gpu, 1);
	axpy_ongpu(C, lr / batch, l.tf_ln1_beta_updates_gpu, 1, l.tf_ln1_beta_gpu, 1);
	scal_ongpu(C, momentum, l.tf_ln1_beta_updates_gpu, 1);

	// LayerNorm 2
	axpy_ongpu(N, lr / batch, l.tf_ln2_gamma_updates_gpu, 1, l.tf_ln2_gamma_gpu, 1);
	scal_ongpu(N, momentum, l.tf_ln2_gamma_updates_gpu, 1);
	axpy_ongpu(N, lr / batch, l.tf_ln2_beta_updates_gpu, 1, l.tf_ln2_beta_gpu, 1);
	scal_ongpu(N, momentum, l.tf_ln2_beta_updates_gpu, 1);

	// FFN W1
	{
		const int count = ffn_hidden * N;
		axpy_ongpu(count, -decay * batch, l.tf_ffn_w1_gpu, 1, l.tf_ffn_w1_updates_gpu, 1);
		axpy_ongpu(count, lr / batch, l.tf_ffn_w1_updates_gpu, 1, l.tf_ffn_w1_gpu, 1);
		scal_ongpu(count, momentum, l.tf_ffn_w1_updates_gpu, 1);
	}
	axpy_ongpu(ffn_hidden, lr / batch, l.tf_ffn_b1_updates_gpu, 1, l.tf_ffn_b1_gpu, 1);
	scal_ongpu(ffn_hidden, momentum, l.tf_ffn_b1_updates_gpu, 1);

	// FFN W2
	{
		const int count = N * ffn_hidden;
		axpy_ongpu(count, -decay * batch, l.tf_ffn_w2_gpu, 1, l.tf_ffn_w2_updates_gpu, 1);
		axpy_ongpu(count, lr / batch, l.tf_ffn_w2_updates_gpu, 1, l.tf_ffn_w2_gpu, 1);
		scal_ongpu(count, momentum, l.tf_ffn_w2_updates_gpu, 1);
	}
	axpy_ongpu(N, lr / batch, l.tf_ffn_b2_updates_gpu, 1, l.tf_ffn_b2_gpu, 1);
	scal_ongpu(N, momentum, l.tf_ffn_b2_updates_gpu, 1);

	// Relative position bias
	{
		const int count = heads * bias_table_len;
		axpy_ongpu(count, lr / batch, l.tf_rel_pos_bias_updates_gpu, 1, l.tf_rel_pos_bias_gpu, 1);
		scal_ongpu(count, momentum, l.tf_rel_pos_bias_updates_gpu, 1);
	}

	check_nan_gpu("update: weights", l.weights_gpu, l.nweights, l.index);
	check_nan_gpu("update: qkv bias", l.biases_gpu, 3 * C, l.index);
	check_nan_gpu("update: output projection", l.tf_wo_gpu, N * C, l.index);
	check_nan_gpu("update: mHC scales", l.scales_gpu, TRANSFORMER_MHC_PARAM_COUNT, l.index);
	check_nan_gpu("update: ln1 gamma", l.tf_ln1_gamma_gpu, C, l.index);
	check_nan_gpu("update: ln2 gamma", l.tf_ln2_gamma_gpu, N, l.index);
	check_nan_gpu("update: ffn w1", l.tf_ffn_w1_gpu, ffn_hidden * N, l.index);
	check_nan_gpu("update: ffn w2", l.tf_ffn_w2_gpu, N * ffn_hidden, l.index);
	check_nan_gpu("update: rel_pos_bias", l.tf_rel_pos_bias_gpu, heads * bias_table_len, l.index);
}

// ─── push / pull ──────────────────────────────────────────────────────────────

void push_transformer_layer(Darknet::Layer & l)
{
	const int C = l.c;
	const int N = l.n;
	const int ffn_hidden = N * l.tf_ffn_ratio;
	const int heads = l.tf_heads;
	const int ws = l.tf_window_size;
	const int bias_table_len = (2 * ws - 1) * (2 * ws - 1);

	cuda_push_array(l.weights_gpu, l.weights, l.nweights);
	cuda_push_array(l.biases_gpu, l.biases, 3 * C);
	cuda_push_array(l.tf_wo_gpu, l.tf_wo, N * C);
	cuda_push_array(l.tf_wo_bias_gpu, l.tf_wo_bias, N);
	cuda_push_array(l.tf_ln1_gamma_gpu, l.tf_ln1_gamma, C);
	cuda_push_array(l.tf_ln1_beta_gpu, l.tf_ln1_beta, C);
	cuda_push_array(l.tf_ln2_gamma_gpu, l.tf_ln2_gamma, N);
	cuda_push_array(l.tf_ln2_beta_gpu, l.tf_ln2_beta, N);
	cuda_push_array(l.tf_ffn_w1_gpu, l.tf_ffn_w1, ffn_hidden * N);
	cuda_push_array(l.tf_ffn_b1_gpu, l.tf_ffn_b1, ffn_hidden);
	cuda_push_array(l.tf_ffn_w2_gpu, l.tf_ffn_w2, N * ffn_hidden);
	cuda_push_array(l.tf_ffn_b2_gpu, l.tf_ffn_b2, N);
	cuda_push_array(l.tf_rel_pos_bias_gpu, l.tf_rel_pos_bias, heads * bias_table_len);
	cuda_push_array(l.scales_gpu, l.scales, TRANSFORMER_MHC_PARAM_COUNT);
	if (C != N && l.tf_res_proj_gpu)
		cuda_push_array(l.tf_res_proj_gpu, l.tf_res_proj, N * C);
}

void pull_transformer_layer(Darknet::Layer & l)
{
	const int C = l.c;
	const int N = l.n;
	const int ffn_hidden = N * l.tf_ffn_ratio;
	const int heads = l.tf_heads;
	const int ws = l.tf_window_size;
	const int bias_table_len = (2 * ws - 1) * (2 * ws - 1);

	cuda_pull_array(l.weights_gpu, l.weights, l.nweights);
	cuda_pull_array(l.biases_gpu, l.biases, 3 * C);
	cuda_pull_array(l.tf_wo_gpu, l.tf_wo, N * C);
	cuda_pull_array(l.tf_wo_bias_gpu, l.tf_wo_bias, N);
	cuda_pull_array(l.tf_ln1_gamma_gpu, l.tf_ln1_gamma, C);
	cuda_pull_array(l.tf_ln1_beta_gpu, l.tf_ln1_beta, C);
	cuda_pull_array(l.tf_ln2_gamma_gpu, l.tf_ln2_gamma, N);
	cuda_pull_array(l.tf_ln2_beta_gpu, l.tf_ln2_beta, N);
	cuda_pull_array(l.tf_ffn_w1_gpu, l.tf_ffn_w1, ffn_hidden * N);
	cuda_pull_array(l.tf_ffn_b1_gpu, l.tf_ffn_b1, ffn_hidden);
	cuda_pull_array(l.tf_ffn_w2_gpu, l.tf_ffn_w2, N * ffn_hidden);
	cuda_pull_array(l.tf_ffn_b2_gpu, l.tf_ffn_b2, N);
	cuda_pull_array(l.tf_rel_pos_bias_gpu, l.tf_rel_pos_bias, heads * bias_table_len);
	cuda_pull_array(l.scales_gpu, l.scales, TRANSFORMER_MHC_PARAM_COUNT);
	if (C != N && l.tf_res_proj_gpu)
		cuda_pull_array(l.tf_res_proj_gpu, l.tf_res_proj, N * C);
}

#endif // DARKNET_GPU
