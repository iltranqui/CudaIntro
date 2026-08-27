#include "darknet_internal.hpp"

#ifdef DARKNET_GPU

#include "vit_layer.hpp"
#include "blas.hpp"
#include "gemm.hpp"
#include "activations.hpp"
#include "dark_cuda.hpp"

#include <cmath>
#include <cstdio>
#include <vector>

namespace
{
	constexpr int VIT_BLOCK = 256;
	constexpr float VIT_ATTENTION_QK_CLAMP = 128.0f;
	constexpr float VIT_ATTENTION_SCORE_CLAMP = 10.0f;
	constexpr float VIT_FEATURE_CLAMP = 10.0f;
	constexpr float VIT_GRAD_CLAMP = 20.0f;
	constexpr float VIT_MHC_PARAM_CLAMP = 8.0f;

	static void check_nan_gpu(const char *step_name, float *d_arr, size_t size, int layer_idx)
	{
		if (d_arr == nullptr || size == 0)
		{
			return;
		}

		if (is_nan_or_inf(d_arr, size))
		{
			const std::string layer_label = Darknet::layer_type_diagnostic_label(Darknet::ELayerType::VIT);
			std::printf("[%s layer] NaN/Inf detected at layer %d, step: %s\n", layer_label.c_str(), layer_idx, step_name);
		}
	}

	static bool vit_use_tensor_op(const Darknet::NetworkState &state)
	{
#ifdef DARKNET_GPU_CUDA
		return state.net.cudnn_half || state.net.cudnn_bf16;
#else
		(void)state;
		return false;
#endif
	}

#if defined(CUDNN) && !defined(DARKNET_GPU_ROCM)
	static void vit_softmax_forward_gpu(float *scores, int rows, int T)
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

	static void vit_softmax_backward_gpu(const float *scores, float *d_scores, int rows, int T)
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

// NCHW -> patch-token gather.
// Output layout is [B, T, P*P*C] where T=(H/P)*(W/P).
__global__ void vit_patchify_kernel(
	const float *input, float *patches,
	int B, int C, int H, int W, int P, int S, int pad, int Hp, int Wp)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	const int K = P * P * C;
	const int T = Hp * Wp;
	const int total = B * T * K;
	if (idx >= total) return;

	const int k = idx % K;
	const int t = (idx / K) % T;
	const int b = idx / (K * T);
	const int c = k % C;
	const int patch_site = k / C;
	const int dy = patch_site / P;
	const int dx = patch_site % P;
	const int py = t / Wp;
	const int px = t % Wp;
	const int y = py * S + dy - pad;
	const int x = px * S + dx - pad;

	const bool in_bounds = (y >= 0 && y < H && x >= 0 && x < W);
	patches[idx] = in_bounds ? input[(b * C + c) * H * W + y * W + x] : 0.0f;
}

__global__ void vit_add_pos_embed_kernel(float *tokens, const float *pos_embed, int B, int T, int N)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	const int total = B * T * N;
	if (idx >= total) return;

	const int n = idx % N;
	const int t = (idx / N) % T;
	tokens[idx] += pos_embed[t * N + n];
}

__global__ void vit_patch_delta_to_spatial_kernel(
	const float *patch_delta, float *spatial_delta,
	int B, int C, int H, int W, int P, int S, int pad, int Hp, int Wp)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	const int K = P * P * C;
	const int T = Hp * Wp;
	const int total = B * T * K;
	if (idx >= total) return;

	const int k = idx % K;
	const int t = (idx / K) % T;
	const int b = idx / (K * T);
	const int c = k % C;
	const int patch_site = k / C;
	const int dy = patch_site / P;
	const int dx = patch_site % P;
	const int py = t / Wp;
	const int px = t % Wp;
	const int y = py * S + dy - pad;
	const int x = px * S + dx - pad;

	if (y < 0 || y >= H || x < 0 || x >= W) return;
	atomicAdd(spatial_delta + (b * C + c) * H * W + y * W + x, patch_delta[idx]);
}

// NTC → NCHW scatter: final step of forward pass, puts results back into
// Darknet's expected [B, N, H, W] layout.
// idx encodes NTC (channel-minor), output is NCHW (channel-major).
// Write address: output[ (b*N + n)*T + t ]  — N is the "channel" axis here.
__global__ void vit_tokens_to_spatial_kernel(const float *tokens, float *output, int B, int N, int T)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int total = B * T * N;
	if (idx >= total) return;

	int n = idx % N;
	int rest = idx / N;
	int t = rest % T;
	int b = rest / T;

	output[(b * N + n) * T + t] = tokens[idx];
}

// NCHW → NTC gather: first step of backward pass, converts the incoming
// delta [B, N, H, W] (NCHW) into token layout [B, T, N] (NTC).
// Inverse of vit_tokens_to_spatial_kernel — same index arithmetic, read/write swapped.
__global__ void vit_spatial_to_tokens_kernel(const float *input, float *tokens, int B, int N, int T)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int total = B * T * N;
	if (idx >= total) return;

	int n = idx % N;
	int rest = idx / N;
	int t = rest % T;
	int b = rest / T;

	tokens[idx] = input[(b * N + n) * T + t];
}

__global__ void vit_layernorm_forward_kernel(
	const float *x, float *out, float *mean, float *var, float *xhat,
	const float *gamma, const float *beta, int total_tokens, int C)
{
	const int lane = threadIdx.x & 31;
	const int i = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
	if (i >= total_tokens) return;

	const float eps = 1e-5f;
	const float *xi = x + i * C;
	float *oi = out + i * C;
	float *xh = xhat + i * C;

	float m = 0.0f;
	for (int j = lane; j < C; j += 32) m += xi[j];
	for (int offset = 16; offset > 0; offset >>= 1)
		m += __shfl_xor_sync(0xFFFFFFFF, m, offset);
	m /= C;
	if (lane == 0) mean[i] = m;

	float v = 0.0f;
	for (int j = lane; j < C; j += 32)
	{
		float d = xi[j] - m;
		v += d * d;
	}
	for (int offset = 16; offset > 0; offset >>= 1)
		v += __shfl_xor_sync(0xFFFFFFFF, v, offset);
	v /= C;
	if (lane == 0) var[i] = v;

	float inv_std = rsqrtf(v + eps);
	for (int j = lane; j < C; j += 32)
	{
		xh[j] = (xi[j] - m) * inv_std;
		oi[j] = xh[j] * gamma[j] + beta[j];
	}
}

// Broadcast-add a bias vector of length N to every row of a [M, N] matrix.
// Each thread handles one element; bias index is (idx % N) because the bias
// repeats identically for every row.  No atomics needed: each output element
// has exactly one writer.
__global__ void vit_add_bias_kernel(float *x, const float *bias, int M, int N)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= M * N) return;
	x[idx] += bias[idx % N];
}

// Reduce a [M, N] matrix to a bias-gradient vector of length N by summing
// over all M rows.  This is the backward of vit_add_bias_kernel.
//
// atomicAdd is required here: multiple threads (one per row) all write to the
// same bias_updates[j] for the same column j.  Without atomics, those
// concurrent writes would produce a data race / lost updates.
__global__ void vit_sum_rows_kernel(const float *x, float *bias_updates, int M, int N)
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

static void vit_sum_rows_gpu(const float *x, float *bias_updates, int M, int N)
{
	const int threads = 256;
	int row_blocks = (M + threads - 1) / threads;
	if (row_blocks < 1) row_blocks = 1;
	if (row_blocks > 1024) row_blocks = 1024;
	dim3 grid(N, row_blocks);
	vit_sum_rows_kernel<<<grid, threads, threads * sizeof(float), get_cuda_stream()>>>(x, bias_updates, M, N);
	CHECK_CUDA(cudaPeekAtLastError());
}

// De-interleave the joint QKV buffer into three separate head-major buffers.
//
// The QKV projection produced one contiguous buffer with layout [B, T, 3C]
// where each row is [ Q_0..Q_{C-1} | K_0..K_{C-1} | V_0..V_{C-1} ].
// For efficient batched GEMM in the attention loop we need Q/K/V split into
// separate [B, heads, T, d] buffers (head-major, so each head's T×d slice
// is contiguous).
//
// idx encodes [b, h, t, dd] (head-major, d innermost):
//   dd   = idx % d
//   h    = (idx / d) % heads
//   t    = (idx / d / heads) % T
//   b    = idx / d / heads / T
//
// Source (interleaved NTC layout):
//   Q_h_dd  at qkv[ (b*T + t) * 3C + h*d + dd ]
//   K_h_dd  at qkv[ (b*T + t) * 3C + C + h*d + dd ]
//   V_h_dd  at qkv[ (b*T + t) * 3C + 2C + h*d + dd ]
//
// Destination (head-major):
//   out_idx = ((b * heads + h) * T + t) * d + dd
__global__ void vit_split_qkv_kernel(
	const float *qkv, float *Q, float *K, float *V,
	int B, int T, int C, int heads, int d)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int total = B * T * heads * d;
	if (idx >= total) return;

	int dd = idx % d;
	int rest = idx / d;
	int h = rest % heads;
	rest /= heads;
	int t = rest % T;
	int b = rest / T;

	int qkv_idx = (b * T + t) * 3 * C + h * d + dd;
	int out_idx = ((b * heads + h) * T + t) * d + dd;

	Q[out_idx] = qkv[qkv_idx];
	K[out_idx] = qkv[qkv_idx + C];
	V[out_idx] = qkv[qkv_idx + 2 * C];
}

__global__ void vit_extract_heads_kernel(
	const float *tokens, float *heads_out,
	int B, int T, int C, int heads, int d)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int total = B * T * heads * d;
	if (idx >= total) return;

	int dd = idx % d;
	int rest = idx / d;
	int h = rest % heads;
	rest /= heads;
	int t = rest % T;
	int b = rest / T;

	heads_out[idx] = tokens[(b * T + t) * C + h * d + dd];
}

__global__ void vit_merge_qkv_grads_kernel(
	const float *dQ, const float *dK, const float *dV, float *d_qkv,
	int B, int T, int C, int heads, int d)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int total = B * T * heads * d;
	if (idx >= total) return;

	int dd = idx % d;
	int rest = idx / d;
	int h = rest % heads;
	rest /= heads;
	int t = rest % T;
	int b = rest / T;

	int qkv_idx = (b * T + t) * 3 * C + h * d + dd;
	d_qkv[qkv_idx] = dQ[idx];
	d_qkv[qkv_idx + C] = dK[idx];
	d_qkv[qkv_idx + 2 * C] = dV[idx];
}

// Row-wise softmax over the score matrix.
//
// One thread per row (one query token's attention vector across all T keys).
// A single thread walks the entire row sequentially because:
//   1. Computing the max and sum requires a full scan before any output can
//      be written — parallelising within a row needs shared-memory reductions
//      that would be more complex than this simple single-thread loop, and
//   2. T is typically small (13×13 = 169 to 26×26 = 676) so sequential access
//      is fast enough.
// "batches" here = B * heads — the score matrix has one row-set per (batch, head) pair.
__global__ void vit_attention_softmax_kernel(float *scores, int batches, int T)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int total = batches * T;
	if (idx >= total) return;

	int row = idx;
	float *row_ptr = scores + row * T;
	const float clip = VIT_ATTENTION_SCORE_CLAMP;   // see softmax_row() in vit_layer.cpp for rationale

	for (int j = 0; j < T; ++j)
	{
		float v = row_ptr[j];
		if (isnan(v))
		{
			row_ptr[j] = 0.0f;
		}
		else if (isinf(v))
		{
			row_ptr[j] = (v > 0.0f) ? clip : -clip;
		}
		else
		{
			row_ptr[j] = fminf(clip, fmaxf(-clip, v));
		}
	}

	float max_val = row_ptr[0];
	for (int j = 1; j < T; ++j) max_val = fmaxf(max_val, row_ptr[j]);

	float sum = 0.0f;
	for (int j = 0; j < T; ++j)
	{
		row_ptr[j] = expf(row_ptr[j] - max_val);
		sum += row_ptr[j];
	}
	if (!isfinite(sum) || sum <= 0.0f)
	{
		const float uniform = 1.0f / T;
		for (int j = 0; j < T; ++j) row_ptr[j] = uniform;
		return;
	}

	float inv_sum = 1.0f / (sum + 1e-9f);
	for (int j = 0; j < T; ++j) row_ptr[j] *= inv_sum;
}

// Softmax backward: applies the Jacobian of row-wise softmax in-place on d_scores.
//
// For post-softmax probabilities p and upstream gradient g (both length T):
//   d_pre[j] = p[j] * (g[j] - dot(g, p))
//
// Derivation: softmax Jacobian is  J_ij = p_i * (delta_ij - p_j)
// So  d_pre = J^T g = p ⊙ (g - p·g)  = p ⊙ (g - dot(p,g))
//
// The dot(g, p) term is the "average gradient weighted by attention probability".
// Subtracting it enforces the constraint that changes to pre-softmax scores
// must keep the probabilities summing to 1.
//
// One thread per row; d_scores is overwritten with d_pre_softmax in-place.
__global__ void vit_attention_softmax_backward_kernel(const float *scores, float *d_scores, int rows, int T)
{
	int row = blockIdx.x * blockDim.x + threadIdx.x;
	if (row >= rows) return;

	const float *score_row = scores + row * T;
	float *grad_row = d_scores + row * T;

	// dot(g, p): weighted average of upstream gradients under current attention distribution
	float dot = 0.0f;
	for (int j = 0; j < T; ++j)
	{
		dot += grad_row[j] * score_row[j];
	}

	// Apply Jacobian: p_j * (g_j - dot) — in-place update
	for (int j = 0; j < T; ++j)
	{
		grad_row[j] = score_row[j] * (grad_row[j] - dot);
	}
}

__global__ void vit_clamp_scores_kernel(float *scores, int n)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= n) return;

	float v = scores[idx];
	if (isnan(v))
	{
		scores[idx] = 0.0f;
	}
	else if (isinf(v))
	{
		scores[idx] = (v > 0.0f) ? VIT_ATTENTION_SCORE_CLAMP : -VIT_ATTENTION_SCORE_CLAMP;
	}
	else
	{
		scores[idx] = fminf(VIT_ATTENTION_SCORE_CLAMP, fmaxf(-VIT_ATTENTION_SCORE_CLAMP, v));
	}
}

__device__ __forceinline__ float vit_sanitize_and_constrain_device(float value, float limit)
{
	if (!isfinite(value)) return 0.0f;
	return fminf(limit, fmaxf(-limit, value));
}

__global__ void vit_sanitize_and_constrain_kernel(float *x, int n, float limit)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= n) return;
	x[idx] = vit_sanitize_and_constrain_device(x[idx], limit);
}

static void vit_sanitize_and_constrain_gpu(float *x, int n, float limit)
{
	if (x == nullptr || n <= 0) return;
	vit_sanitize_and_constrain_kernel<<<(n + VIT_BLOCK - 1) / VIT_BLOCK, VIT_BLOCK, 0, get_cuda_stream()>>>(
		x, n, limit);
	CHECK_CUDA(cudaPeekAtLastError());
}

// Re-interleave per-head attention results into a single token-layout output.
// Inverse of vit_split_qkv_kernel's output format.
//
// attn_result is [B, heads, T, d] (head-major).
// output is [B, T, C] (token-major), where C = heads * d.
//
// idx encodes NTC output: c = idx % C tells us both which head (h = c/d)
// and which channel within that head (dd = c%d).
// Read address: attn_result[ ((b*heads + h)*T + t)*d + dd ]
__global__ void vit_merge_heads_kernel(
	const float *attn_result, float *output,
	int B, int T, int C, int heads, int d)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int total = B * T * C;
	if (idx >= total) return;

	int c = idx % C;
	int rest = idx / C;
	int t = rest % T;
	int b = rest / T;

	int h = c / d;    // which head owns this output channel
	int dd = c % d;   // channel index within that head
	int in_idx = ((b * heads + h) * T + t) * d + dd;
	output[idx] = attn_result[in_idx];
}

__global__ void vit_residual_add_kernel(float *out, const float *residual, int n)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= n) return;
	out[idx] += residual[idx];
}

// ─── mHC residual mixer kernels ─────────────────────────────────────────────
// Two-stream closed-form Sinkhorn projection used at each residual merge.
__device__ __forceinline__ float vit_mhc_sigmoid_device(float x)
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

__device__ __forceinline__ void vit_mhc_coefficients_device(
	const float *params, int site, float &skip_coeff, float &branch_coeff,
	float &p, float &post_skip, float &post_branch)
{
	const int o = site * 6;
	const float z = 0.5f * (params[o + 0] + params[o + 3] - params[o + 1] - params[o + 2]);
	p = vit_mhc_sigmoid_device(z);
	post_skip = 2.0f * vit_mhc_sigmoid_device(params[o + 4]);
	post_branch = 2.0f * vit_mhc_sigmoid_device(params[o + 5]);
	skip_coeff = post_skip * p + post_branch * (1.0f - p);
	branch_coeff = post_skip * (1.0f - p) + post_branch * p;
}

__global__ void vit_mhc_residual_forward_kernel(
	const float *skip, const float *branch, float *out, const float *params, int site, int n)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= n) return;
	float a, b, p, ps, pb;
	vit_mhc_coefficients_device(params, site, a, b, p, ps, pb);
	const float s = vit_sanitize_and_constrain_device(skip[idx], VIT_FEATURE_CLAMP);
	const float br = vit_sanitize_and_constrain_device(branch[idx], VIT_FEATURE_CLAMP);
	out[idx] = vit_sanitize_and_constrain_device(a * s + b * br, VIT_FEATURE_CLAMP);
}

__global__ void vit_mhc_residual_backward_kernel(
	const float *skip, const float *branch, const float *dout,
	float *d_skip, float *d_branch, const float *params, float *updates, int site, int n)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= n) return;

	float a, b, p, post_skip, post_branch;
	vit_mhc_coefficients_device(params, site, a, b, p, post_skip, post_branch);

	const float g = vit_sanitize_and_constrain_device(dout[idx], VIT_GRAD_CLAMP);
	const float s = vit_sanitize_and_constrain_device(skip[idx], VIT_FEATURE_CLAMP);
	const float br = vit_sanitize_and_constrain_device(branch[idx], VIT_FEATURE_CLAMP);
	d_skip[idx] = vit_sanitize_and_constrain_device(a * g, VIT_GRAD_CLAMP);
	d_branch[idx] = vit_sanitize_and_constrain_device(b * g, VIT_GRAD_CLAMP);

	const int o = site * 6;
	const float grad_post_skip = g * (p * s + (1.0f - p) * br);
	const float grad_post_branch = g * ((1.0f - p) * s + p * br);
	const float grad_p = g * (post_skip - post_branch) * (s - br);
	const float dz = 0.5f * grad_p * p * (1.0f - p);
	if (updates)
	{
		atomicAdd(&updates[o + 0], dz);
		atomicAdd(&updates[o + 1], -dz);
		atomicAdd(&updates[o + 2], -dz);
		atomicAdd(&updates[o + 3], dz);
		atomicAdd(&updates[o + 4], grad_post_skip * post_skip * (1.0f - 0.5f * post_skip));
		atomicAdd(&updates[o + 5], grad_post_branch * post_branch * (1.0f - 0.5f * post_branch));
	}
}

// GPU LayerNorm backward — mirrors layernorm_backward() in vit_layer.cpp.
// One thread per token; each thread walks all C channels sequentially.
//
// atomicAdd for dgamma/dbeta: every token's thread writes to the same C-length
// parameter arrays.  Without atomics this would be a race condition.
// dx does NOT need atomics because each token writes to its own slice of dx.
__global__ void vit_layernorm_backward_kernel(
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

	// dgamma[j] += dout[j] * xhat[j]  (accumulated across all tokens)
	// dbeta[j]  += dout[j]             (accumulated across all tokens)
	// atomicAdd because all B*T token threads share the same gamma/beta arrays.
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

// Backward through positional embedding addition (forward: token += pos_embed[t]).
// Gradient simply passes through the addition; we accumulate into pos_updates.
//
// atomicAdd because all B batch items share the same T*C positional embedding
// table, so B threads (one per batch element) write to the same pos_updates[t*C+c].
__global__ void vit_pos_embed_backward_kernel(const float *token_delta, float *pos_updates, int B, int T, int C)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int total = B * T * C;
	if (idx >= total) return;

	int c = idx % C;
	int rest = idx / C;
	int t = rest % T;

	atomicAdd(&pos_updates[t * C + c], token_delta[idx]);
}

// Scatter token-layout gradient [B, T, C] (NTC) back to NCHW [B, C, H, W].
// This is the final step of backward — propagates gradients to the previous layer.
//
// atomicAdd is used to accumulate into spatial_delta because multiple tokens
// could in principle map to the same spatial location (though in practice each
// token maps to a unique spatial site, the atomicAdd is the safe pattern here
// when state.delta may already contain residual contributions from earlier passes).
__global__ void vit_add_token_delta_to_spatial_kernel(const float *token_delta, float *spatial_delta, int B, int C, int T)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int total = B * T * C;
	if (idx >= total) return;

	int c = idx % C;
	int rest = idx / C;
	int t = rest % T;
	int b = rest / T;

	// Write to NCHW: spatial_delta[ (b*C + c)*T + t ]
	atomicAdd(&spatial_delta[(b * C + c) * T + t], token_delta[idx]);
}

// Bilinear interpolation of positional embedding table from [old_H, old_W, C] to [new_H, new_W, C].
// Preserves learned positional embeddings across network resize, rather than discarding them.
// Layout: tokens are row-major [H, W, C] → index (h * W + w) * C + c.
// Center-aligned source coordinates avoid half-pixel boundary drift:
//   h_src = (h_dst + 0.5) * old_H / new_H - 0.5
__global__ void vit_pos_embed_bilinear_resize_kernel(
	const float *old_embed, float *new_embed,
	int old_H, int old_W, int new_H, int new_W, int C)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int total = new_H * new_W * C;
	if (idx >= total) return;

	int c   = idx % C;
	int wh  = idx / C;
	int w_d = wh % new_W;
	int h_d = wh / new_W;

	float h_s = (h_d + 0.5f) * (float)old_H / (float)new_H - 0.5f;
	float w_s = (w_d + 0.5f) * (float)old_W / (float)new_W - 0.5f;

	int h0 = (int)floorf(h_s);
	int w0 = (int)floorf(w_s);
	int h1 = h0 + 1;
	int w1 = w0 + 1;

	float dh = h_s - (float)h0;
	float dw = w_s - (float)w0;

	h0 = max(0, min(h0, old_H - 1));
	h1 = max(0, min(h1, old_H - 1));
	w0 = max(0, min(w0, old_W - 1));
	w1 = max(0, min(w1, old_W - 1));

	float v00 = old_embed[(h0 * old_W + w0) * C + c];
	float v01 = old_embed[(h0 * old_W + w1) * C + c];
	float v10 = old_embed[(h1 * old_W + w0) * C + c];
	float v11 = old_embed[(h1 * old_W + w1) * C + c];

	new_embed[(h_d * new_W + w_d) * C + c] =
		v00 * (1.0f - dh) * (1.0f - dw) +
		v01 * (1.0f - dh) * dw +
		v10 * dh * (1.0f - dw) +
		v11 * dh * dw;
}

void resize_vit_pos_embed_gpu(const float *old_embed_gpu, float *new_embed_gpu,
                               int old_H, int old_W, int new_H, int new_W, int C)
{
	int total = new_H * new_W * C;
	int blocks = (total + VIT_BLOCK - 1) / VIT_BLOCK;
	vit_pos_embed_bilinear_resize_kernel<<<blocks, VIT_BLOCK, 0, get_cuda_stream()>>>(
		old_embed_gpu, new_embed_gpu, old_H, old_W, new_H, new_W, C);
	CHECK_CUDA(cudaPeekAtLastError());
}

void forward_vit_layer_gpu(Darknet::Layer & l, Darknet::NetworkState state)
{
	TAT(TATPARMS);

	const int B = l.batch;
	const int C = l.c;
	const int N = l.n;
	const int H = l.h;
	const int W = l.w;
	const int P = l.vit_patch_size;
	const int S = l.vit_patch_stride;
	const int pad = l.vit_patch_pad;
	const int Hp = l.out_h;
	const int Wp = l.out_w;
	const int T = Hp * Wp;
	const int patch_dim = P * P * C;
	const int heads = l.vit_heads;
	const int d = l.vit_head_dim;
	const int ffn_hidden = l.vit_mlp_dim;
	const float scale = 1.0f / sqrtf((float)d);

	const int token_count_n = B * T * N;
	const int patch_count = B * T * patch_dim;
	const int qkv_count = B * T * 3 * N;
	const int attn_scores_count = B * heads * T * T;
	const bool use_tensor_op = vit_use_tensor_op(state);

	// Alias scratch buffers to human-readable names.
	// All vit_tmp_* arrays are pre-allocated in make_vit_layer to avoid per-step
	// malloc overhead.  They are large enough for the worst-case token dimension.
	float *ln1_out_gpu = l.vit_tmp_token_c1_gpu;      // [B, T, N]
	float *Q_gpu = l.vit_tmp_head1_gpu;               // [B, heads, T, d]
	float *K_gpu = l.vit_tmp_head2_gpu;               // [B, heads, T, d]
	float *V_gpu = l.vit_tmp_head3_gpu;               // [B, heads, T, d]
	float *attn_result_gpu = l.vit_tmp_head4_gpu;     // [B, heads, T, d]
	float *proj_out_gpu = l.vit_tmp_token_n1_gpu;     // [B, T, N]
	float *ln2_out_gpu = l.vit_tmp_token_n2_gpu;      // [B, T, N]
	float *ffn_out_gpu = l.vit_tmp_token_n3_gpu;      // [B, T, N]

	{
		int num = patch_count;
		vit_patchify_kernel<<<(num + VIT_BLOCK - 1) / VIT_BLOCK, VIT_BLOCK, 0, get_cuda_stream()>>>(
			state.input, l.vit_patch_tokens_gpu, B, C, H, W, P, S, pad, Hp, Wp);
		CHECK_CUDA(cudaPeekAtLastError());
	}

	gemm_ongpu_tensor_op(0, 1, B * T, N, patch_dim, 1.0f,
		l.vit_patch_tokens_gpu, patch_dim,
		l.vit_patch_embed_gpu, patch_dim,
		0.0f,
		l.vit_pre_res1_gpu, N,
		use_tensor_op);

	{
		int num = token_count_n;
		vit_add_bias_kernel<<<(num + VIT_BLOCK - 1) / VIT_BLOCK, VIT_BLOCK, 0, get_cuda_stream()>>>(
			l.vit_pre_res1_gpu, l.vit_patch_bias_gpu, B * T, N);
		CHECK_CUDA(cudaPeekAtLastError());
		vit_add_pos_embed_kernel<<<(num + VIT_BLOCK - 1) / VIT_BLOCK, VIT_BLOCK, 0, get_cuda_stream()>>>(
			l.vit_pre_res1_gpu, l.vit_pos_embed_gpu, B, T, N);
		CHECK_CUDA(cudaPeekAtLastError());
	}
	check_nan_gpu("forward: patch tokens + pos_embed", l.vit_pre_res1_gpu, token_count_n, l.index);

	{
		int num = B * T;
		const int warp_threads = num * 32;
		vit_layernorm_forward_kernel<<<(warp_threads + VIT_BLOCK - 1) / VIT_BLOCK, VIT_BLOCK, 0, get_cuda_stream()>>>(
			l.vit_pre_res1_gpu, ln1_out_gpu, l.vit_ln1_mean_gpu, l.vit_ln1_var_gpu,
			l.vit_ln1_xhat_gpu, l.vit_ln1_gamma_gpu, l.vit_ln1_beta_gpu, num, N);
		CHECK_CUDA(cudaPeekAtLastError());
	}
	check_nan_gpu("forward: after layernorm1", ln1_out_gpu, token_count_n, l.index);

	gemm_ongpu_tensor_op(0, 1, B * T, 3 * N, N, 1.0f,
		ln1_out_gpu, N,
		l.weights_gpu, N,
		0.0f,
		l.vit_qkv_out_gpu, 3 * N,
		use_tensor_op);

	{
		int num = qkv_count;
		vit_add_bias_kernel<<<(num + VIT_BLOCK - 1) / VIT_BLOCK, VIT_BLOCK, 0, get_cuda_stream()>>>(
			l.vit_qkv_out_gpu, l.biases_gpu, B * T, 3 * N);
		CHECK_CUDA(cudaPeekAtLastError());
	}
	check_nan_gpu("forward: after qkv projection", l.vit_qkv_out_gpu, qkv_count, l.index);

	{
		int num = B * T * heads * d;
		vit_split_qkv_kernel<<<(num + VIT_BLOCK - 1) / VIT_BLOCK, VIT_BLOCK, 0, get_cuda_stream()>>>(
			l.vit_qkv_out_gpu, Q_gpu, K_gpu, V_gpu, B, T, N, heads, d);
		CHECK_CUDA(cudaPeekAtLastError());
		vit_sanitize_and_constrain_gpu(Q_gpu, num, VIT_ATTENTION_QK_CLAMP);
		vit_sanitize_and_constrain_gpu(K_gpu, num, VIT_ATTENTION_QK_CLAMP);
	}

	const int num_batches = B * heads;
	gemm_ongpu_strided_batched_tensor_op(0, 1, T, T, d, scale,
		Q_gpu, d, (long long)T * d,
		K_gpu, d, (long long)T * d,
		0.0f,
		l.vit_attn_scores_gpu, T, (long long)T * T,
		num_batches,
		use_tensor_op);

	{
#if defined(CUDNN) && !defined(DARKNET_GPU_ROCM)
		int num = attn_scores_count;
		vit_clamp_scores_kernel<<<(num + VIT_BLOCK - 1) / VIT_BLOCK, VIT_BLOCK, 0, get_cuda_stream()>>>(
			l.vit_attn_scores_gpu, num);
		CHECK_CUDA(cudaPeekAtLastError());

		vit_softmax_forward_gpu(l.vit_attn_scores_gpu, num_batches * T, T);
#else
		int num = num_batches * T;
		vit_attention_softmax_kernel<<<(num + VIT_BLOCK - 1) / VIT_BLOCK, VIT_BLOCK, 0, get_cuda_stream()>>>(
			l.vit_attn_scores_gpu, num_batches, T);
		CHECK_CUDA(cudaPeekAtLastError());
#endif
	}
	check_nan_gpu("forward: after attention scores", l.vit_attn_scores_gpu, attn_scores_count, l.index);

	gemm_ongpu_strided_batched_tensor_op(0, 0, T, d, T, 1.0f,
		l.vit_attn_scores_gpu, T, (long long)T * T,
		V_gpu, d, (long long)T * d,
		0.0f,
		attn_result_gpu, d, (long long)T * d,
		num_batches,
		use_tensor_op);

	{
		int num = token_count_n;
		vit_merge_heads_kernel<<<(num + VIT_BLOCK - 1) / VIT_BLOCK, VIT_BLOCK, 0, get_cuda_stream()>>>(
			attn_result_gpu, l.vit_attn_out_gpu, B, T, N, heads, d);
		CHECK_CUDA(cudaPeekAtLastError());
	}
	check_nan_gpu("forward: after attention output", l.vit_attn_out_gpu, token_count_n, l.index);

	if (N <= 0 || B <= 0 || T <= 0 || l.vit_attn_out_gpu == nullptr || l.vit_wo_gpu == nullptr || proj_out_gpu == nullptr)
	{
		darknet_fatal_error(DARKNET_LOC,
			"invalid ViT output projection setup: layer=%d B=%d T=%d C=%d N=%d out_c=%d attn_out=%p wo=%p proj_out=%p",
			l.index, B, T, C, N, l.out_c,
			static_cast<void*>(l.vit_attn_out_gpu),
			static_cast<void*>(l.vit_wo_gpu),
			static_cast<void*>(proj_out_gpu));
	}

	gemm_ongpu_tensor_op(0, 1, B * T, N, N, 1.0f,
		l.vit_attn_out_gpu, N,
		l.vit_wo_gpu, N,
		0.0f,
		proj_out_gpu, N,
		use_tensor_op);

	{
		int num = token_count_n;
		vit_add_bias_kernel<<<(num + VIT_BLOCK - 1) / VIT_BLOCK, VIT_BLOCK, 0, get_cuda_stream()>>>(
			proj_out_gpu, l.vit_wo_bias_gpu, B * T, N);
		CHECK_CUDA(cudaPeekAtLastError());
	}
	check_nan_gpu("forward: after output projection", proj_out_gpu, token_count_n, l.index);

	vit_sanitize_and_constrain_gpu(proj_out_gpu, token_count_n, VIT_FEATURE_CLAMP);
	simple_copy_ongpu(token_count_n, proj_out_gpu, l.x_gpu);
	vit_mhc_residual_forward_kernel<<<(token_count_n + VIT_BLOCK - 1) / VIT_BLOCK, VIT_BLOCK, 0, get_cuda_stream()>>>(
		l.vit_pre_res1_gpu, l.x_gpu, proj_out_gpu, l.scales_gpu, 0, token_count_n);
	CHECK_CUDA(cudaPeekAtLastError());
	vit_sanitize_and_constrain_gpu(proj_out_gpu, token_count_n, VIT_FEATURE_CLAMP);
	check_nan_gpu("forward: after residual1", proj_out_gpu, token_count_n, l.index);

	simple_copy_ongpu(token_count_n, proj_out_gpu, l.vit_pre_res2_gpu);

	{
		int num = B * T;
		const int warp_threads = num * 32;
		vit_layernorm_forward_kernel<<<(warp_threads + VIT_BLOCK - 1) / VIT_BLOCK, VIT_BLOCK, 0, get_cuda_stream()>>>(
			l.vit_pre_res2_gpu, ln2_out_gpu, l.vit_ln2_mean_gpu, l.vit_ln2_var_gpu,
			l.vit_ln2_xhat_gpu, l.vit_ln2_gamma_gpu, l.vit_ln2_beta_gpu, num, N);
		CHECK_CUDA(cudaPeekAtLastError());
	}
	check_nan_gpu("forward: after layernorm2", ln2_out_gpu, token_count_n, l.index);

	gemm_ongpu_tensor_op(0, 1, B * T, ffn_hidden, N, 1.0f,
		ln2_out_gpu, N,
		l.vit_ffn_w1_gpu, N,
		0.0f,
		l.vit_ffn_hidden_gpu, ffn_hidden,
		use_tensor_op);

	{
		int num = B * T * ffn_hidden;
		vit_add_bias_kernel<<<(num + VIT_BLOCK - 1) / VIT_BLOCK, VIT_BLOCK, 0, get_cuda_stream()>>>(
			l.vit_ffn_hidden_gpu, l.vit_ffn_b1_gpu, B * T, ffn_hidden);
		CHECK_CUDA(cudaPeekAtLastError());
	}

	vit_sanitize_and_constrain_gpu(l.vit_ffn_hidden_gpu, B * T * ffn_hidden, VIT_FEATURE_CLAMP);
	simple_copy_ongpu(B * T * ffn_hidden, l.vit_ffn_hidden_gpu, l.activation_input_gpu);
	activate_array_ongpu(l.vit_ffn_hidden_gpu, B * T * ffn_hidden, l.activation);
	vit_sanitize_and_constrain_gpu(l.vit_ffn_hidden_gpu, B * T * ffn_hidden, VIT_FEATURE_CLAMP);
	check_nan_gpu("forward: after ffn hidden", l.vit_ffn_hidden_gpu, B * T * ffn_hidden, l.index);

	gemm_ongpu_tensor_op(0, 1, B * T, N, ffn_hidden, 1.0f,
		l.vit_ffn_hidden_gpu, ffn_hidden,
		l.vit_ffn_w2_gpu, ffn_hidden,
		0.0f,
		ffn_out_gpu, N,
		use_tensor_op);

	{
		int num = token_count_n;
		vit_add_bias_kernel<<<(num + VIT_BLOCK - 1) / VIT_BLOCK, VIT_BLOCK, 0, get_cuda_stream()>>>(
			ffn_out_gpu, l.vit_ffn_b2_gpu, B * T, N);
		CHECK_CUDA(cudaPeekAtLastError());
		vit_sanitize_and_constrain_gpu(ffn_out_gpu, num, VIT_FEATURE_CLAMP);
		simple_copy_ongpu(num, ffn_out_gpu, l.x_norm_gpu);
		vit_mhc_residual_forward_kernel<<<(num + VIT_BLOCK - 1) / VIT_BLOCK, VIT_BLOCK, 0, get_cuda_stream()>>>(
			l.vit_pre_res2_gpu, l.x_norm_gpu, ffn_out_gpu, l.scales_gpu, 1, num);
		CHECK_CUDA(cudaPeekAtLastError());
	}
	vit_sanitize_and_constrain_gpu(ffn_out_gpu, token_count_n, VIT_FEATURE_CLAMP);
	check_nan_gpu("forward: after ffn output + residual", ffn_out_gpu, token_count_n, l.index);

	{
		int num = token_count_n;
		vit_tokens_to_spatial_kernel<<<(num + VIT_BLOCK - 1) / VIT_BLOCK, VIT_BLOCK, 0, get_cuda_stream()>>>(
			ffn_out_gpu, l.output_gpu, B, N, T);
		CHECK_CUDA(cudaPeekAtLastError());
	}
	vit_sanitize_and_constrain_gpu(l.output_gpu, B * N * T, VIT_FEATURE_CLAMP);
	check_nan_gpu("forward: final output", l.output_gpu, B * N * T, l.index);

}

void backward_vit_layer_gpu(Darknet::Layer & l, Darknet::NetworkState state)
{
	TAT(TATPARMS);

	const int B = l.batch;
	const int C = l.c;
	const int N = l.n;
	const int H = l.h;
	const int W = l.w;
	const int P = l.vit_patch_size;
	const int S = l.vit_patch_stride;
	const int pad = l.vit_patch_pad;
	const int Hp = l.out_h;
	const int Wp = l.out_w;
	const int T = Hp * Wp;
	const int patch_dim = P * P * C;
	const int heads = l.vit_heads;
	const int d = l.vit_head_dim;
	const int ffn_hidden = l.vit_mlp_dim;
	const float scale = 1.0f / sqrtf((float)d);
	const int token_count_n = B * T * N;
	const int patch_count = B * T * patch_dim;
	const int qkv_count = B * T * 3 * N;
	const int attn_head_count = B * heads * T * d;
	const bool use_tensor_op = vit_use_tensor_op(state);

	// Buffer aliases for the backward pass.
	// Note: several buffers are reused at different stages (e.g. dQ_gpu reuses
	// V_gpu's slot after V is no longer needed).  This is safe because the backward
	// proceeds strictly sequentially — there is no parallel read/write conflict.
	// d_pre_ln2_gpu reuses output_gpu: at this point the forward output has been
	// consumed by the loss layer and its GPU buffer can be repurposed.
	float *dout_tokens_gpu = l.vit_tmp_token_n1_gpu;   // dL/d(output tokens) [B,T,N]
	float *d_ffn_hidden_gpu = l.vit_tmp_ffn_hidden_gpu;// dL/d(FFN hidden)    [B,T,ffn_hidden]
	float *ln2_out_gpu = l.vit_tmp_token_n2_gpu;       // recomputed LN2 out  [B,T,N]
	float *d_ln2_out_gpu = l.vit_tmp_token_n3_gpu;     // dL/d(LN2 output)    [B,T,N]
	float *d_pre_ln2_gpu = l.output_gpu;               // dL/d(pre-LN2)       [B,T,N] — reuses output buffer
	float *d_attn_out_gpu = l.vit_tmp_token_c2_gpu;    // dL/d(attn output)   [B,T,N]
	float *ln1_out_gpu = l.vit_tmp_token_c1_gpu;       // recomputed LN1 out  [B,T,N]
	float *Q_gpu = l.vit_tmp_head1_gpu;                // Q [B,heads,T,d]
	float *K_gpu = l.vit_tmp_head2_gpu;                // K [B,heads,T,d]
	float *V_gpu = l.vit_tmp_head3_gpu;                // V [B,heads,T,d]
	float *d_head_gpu = l.vit_tmp_head5_gpu;           // per-head upstream grad
	float *dV_gpu = l.vit_tmp_head4_gpu;               // dV [B,heads,T,d]
	float *dQ_gpu = l.vit_tmp_head3_gpu;               // dQ — reuses V slot (V no longer needed)
	float *dK_gpu = l.vit_tmp_head2_gpu;               // dK — reuses K slot (K no longer needed)
	float *d_scores_gpu = l.vit_tmp_scores_gpu;        // dL/d(attention scores) [B,heads,T,T]
	float *d_qkv_gpu = l.vit_qkv_out_gpu;             // dL/d(QKV) — reuses forward buffer
	float *d_ln1_out_gpu = l.vit_tmp_token_c2_gpu;     // dL/d(LN1 output)    [B,T,N]
	float *d_pre_ln1_gpu = l.vit_tmp_token_c1_gpu;     // dL/d(pre-LN1)       [B,T,N]

	reset_nan_and_inf(l.delta_gpu, token_count_n);
	constrain_ongpu(token_count_n, 1.0f, l.delta_gpu, 1);

	{
		int num = token_count_n;
		vit_spatial_to_tokens_kernel<<<(num + VIT_BLOCK - 1) / VIT_BLOCK, VIT_BLOCK, 0, get_cuda_stream()>>>(
			l.delta_gpu, dout_tokens_gpu, B, N, T);
		CHECK_CUDA(cudaPeekAtLastError());
	}
	check_nan_gpu("backward: initial delta", dout_tokens_gpu, token_count_n, l.index);

	float *d_ffn_out_gpu = ln2_out_gpu;
	{
		int num = token_count_n;
		vit_mhc_residual_backward_kernel<<<(num + VIT_BLOCK - 1) / VIT_BLOCK, VIT_BLOCK, 0, get_cuda_stream()>>>(
			l.vit_pre_res2_gpu, l.x_norm_gpu, dout_tokens_gpu,
			dout_tokens_gpu, d_ffn_out_gpu, l.scales_gpu, l.scale_updates_gpu, 1, num);
		CHECK_CUDA(cudaPeekAtLastError());
		vit_sanitize_and_constrain_gpu(dout_tokens_gpu, num, VIT_GRAD_CLAMP);
		vit_sanitize_and_constrain_gpu(d_ffn_out_gpu, num, VIT_GRAD_CLAMP);
	}

	gemm_ongpu_tensor_op(0, 0, B * T, ffn_hidden, N, 1.0f,
		d_ffn_out_gpu, N,
		l.vit_ffn_w2_gpu, ffn_hidden,
		0.0f,
		d_ffn_hidden_gpu, ffn_hidden,
		use_tensor_op);

	gemm_ongpu_tensor_op(1, 0, N, ffn_hidden, B * T, 1.0f,
		d_ffn_out_gpu, N,
		l.vit_ffn_hidden_gpu, ffn_hidden,
		1.0f,
		l.vit_ffn_w2_updates_gpu, ffn_hidden,
		use_tensor_op);

	{
		vit_sum_rows_gpu(d_ffn_out_gpu, l.vit_ffn_b2_updates_gpu, B * T, N);
	}

	vit_sanitize_and_constrain_gpu(d_ffn_hidden_gpu, B * T * ffn_hidden, VIT_GRAD_CLAMP);
	gradient_array_ongpu(l.activation_input_gpu, B * T * ffn_hidden, l.activation, d_ffn_hidden_gpu);
	vit_sanitize_and_constrain_gpu(d_ffn_hidden_gpu, B * T * ffn_hidden, VIT_GRAD_CLAMP);
	check_nan_gpu("backward: after ffn hidden grad", d_ffn_hidden_gpu, B * T * ffn_hidden, l.index);

	{
		int num = B * T;
		const int warp_threads = num * 32;
		vit_layernorm_forward_kernel<<<(warp_threads + VIT_BLOCK - 1) / VIT_BLOCK, VIT_BLOCK, 0, get_cuda_stream()>>>(
			l.vit_pre_res2_gpu, ln2_out_gpu, l.vit_ln2_mean_gpu, l.vit_ln2_var_gpu,
			l.vit_ln2_xhat_gpu, l.vit_ln2_gamma_gpu, l.vit_ln2_beta_gpu, num, N);
		CHECK_CUDA(cudaPeekAtLastError());
	}

	gemm_ongpu_tensor_op(1, 0, ffn_hidden, N, B * T, 1.0f,
		d_ffn_hidden_gpu, ffn_hidden,
		ln2_out_gpu, N,
		1.0f,
		l.vit_ffn_w1_updates_gpu, N,
		use_tensor_op);

	{
		vit_sum_rows_gpu(d_ffn_hidden_gpu, l.vit_ffn_b1_updates_gpu, B * T, ffn_hidden);
	}

	gemm_ongpu_tensor_op(0, 0, B * T, N, ffn_hidden, 1.0f,
		d_ffn_hidden_gpu, ffn_hidden,
		l.vit_ffn_w1_gpu, N,
		0.0f,
		d_ln2_out_gpu, N,
		use_tensor_op);
	vit_sanitize_and_constrain_gpu(d_ln2_out_gpu, token_count_n, VIT_GRAD_CLAMP);
	check_nan_gpu("backward: before layernorm2 backward", d_ln2_out_gpu, token_count_n, l.index);

	{
		int num = B * T;
		const int warp_threads = num * 32;
		vit_layernorm_backward_kernel<<<(warp_threads + VIT_BLOCK - 1) / VIT_BLOCK, VIT_BLOCK, 0, get_cuda_stream()>>>(
			d_ln2_out_gpu, l.vit_ln2_xhat_gpu, l.vit_ln2_var_gpu, l.vit_ln2_gamma_gpu,
			d_pre_ln2_gpu, l.vit_ln2_gamma_updates_gpu, l.vit_ln2_beta_updates_gpu, num, N);
		CHECK_CUDA(cudaPeekAtLastError());
	}

	axpy_ongpu(token_count_n, 1.0f, dout_tokens_gpu, 1, d_pre_ln2_gpu, 1);
	vit_sanitize_and_constrain_gpu(d_pre_ln2_gpu, token_count_n, VIT_GRAD_CLAMP);
	check_nan_gpu("backward: after layernorm2 backward + residual", d_pre_ln2_gpu, token_count_n, l.index);

	float *d_proj_out_gpu = d_ln2_out_gpu;
	vit_mhc_residual_backward_kernel<<<(token_count_n + VIT_BLOCK - 1) / VIT_BLOCK, VIT_BLOCK, 0, get_cuda_stream()>>>(
		l.vit_pre_res1_gpu, l.x_gpu, d_pre_ln2_gpu,
		dout_tokens_gpu, d_proj_out_gpu, l.scales_gpu, l.scale_updates_gpu, 0, token_count_n);
	CHECK_CUDA(cudaPeekAtLastError());
	vit_sanitize_and_constrain_gpu(dout_tokens_gpu, token_count_n, VIT_GRAD_CLAMP);
	vit_sanitize_and_constrain_gpu(d_proj_out_gpu, token_count_n, VIT_GRAD_CLAMP);

	gemm_ongpu_tensor_op(0, 0, B * T, N, N, 1.0f,
		d_proj_out_gpu, N,
		l.vit_wo_gpu, N,
		0.0f,
		d_attn_out_gpu, N,
		use_tensor_op);
	vit_sanitize_and_constrain_gpu(d_attn_out_gpu, token_count_n, VIT_GRAD_CLAMP);
	check_nan_gpu("backward: after output projection backward", d_attn_out_gpu, token_count_n, l.index);

	gemm_ongpu_tensor_op(1, 0, N, N, B * T, 1.0f,
		d_proj_out_gpu, N,
		l.vit_attn_out_gpu, N,
		1.0f,
		l.vit_wo_updates_gpu, N,
		use_tensor_op);

	{
		vit_sum_rows_gpu(d_proj_out_gpu, l.vit_wo_bias_updates_gpu, B * T, N);
	}

	{
		int num = B * T;
		const int warp_threads = num * 32;
		vit_layernorm_forward_kernel<<<(warp_threads + VIT_BLOCK - 1) / VIT_BLOCK, VIT_BLOCK, 0, get_cuda_stream()>>>(
			l.vit_pre_res1_gpu, ln1_out_gpu, l.vit_ln1_mean_gpu, l.vit_ln1_var_gpu,
			l.vit_ln1_xhat_gpu, l.vit_ln1_gamma_gpu, l.vit_ln1_beta_gpu, num, N);
		CHECK_CUDA(cudaPeekAtLastError());
	}

	{
		int num = attn_head_count;
		vit_split_qkv_kernel<<<(num + VIT_BLOCK - 1) / VIT_BLOCK, VIT_BLOCK, 0, get_cuda_stream()>>>(
			l.vit_qkv_out_gpu, Q_gpu, K_gpu, V_gpu, B, T, N, heads, d);
		CHECK_CUDA(cudaPeekAtLastError());
		vit_sanitize_and_constrain_gpu(Q_gpu, num, VIT_ATTENTION_QK_CLAMP);
		vit_sanitize_and_constrain_gpu(K_gpu, num, VIT_ATTENTION_QK_CLAMP);
		vit_extract_heads_kernel<<<(num + VIT_BLOCK - 1) / VIT_BLOCK, VIT_BLOCK, 0, get_cuda_stream()>>>(
			d_attn_out_gpu, d_head_gpu, B, T, N, heads, d);
		CHECK_CUDA(cudaPeekAtLastError());
	}

	const int num_batches = B * heads;
	gemm_ongpu_strided_batched_tensor_op(1, 0, T, d, T, 1.0f,
		l.vit_attn_scores_gpu, T, (long long)T * T,
		d_head_gpu, d, (long long)T * d,
		0.0f,
		dV_gpu, d, (long long)T * d,
		num_batches,
		use_tensor_op);
	gemm_ongpu_strided_batched_tensor_op(0, 1, T, T, d, 1.0f,
		d_head_gpu, d, (long long)T * d,
		V_gpu, d, (long long)T * d,
		0.0f,
		d_scores_gpu, T, (long long)T * T,
		num_batches,
		use_tensor_op);

	{
#if defined(CUDNN) && !defined(DARKNET_GPU_ROCM)
		vit_softmax_backward_gpu(l.vit_attn_scores_gpu, d_scores_gpu, num_batches * T, T);
#else
		int num = num_batches * T;
		vit_attention_softmax_backward_kernel<<<(num + VIT_BLOCK - 1) / VIT_BLOCK, VIT_BLOCK, 0, get_cuda_stream()>>>(
			l.vit_attn_scores_gpu, d_scores_gpu, num, T);
		CHECK_CUDA(cudaPeekAtLastError());
#endif
	}

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
		int num = attn_head_count;
		vit_merge_qkv_grads_kernel<<<(num + VIT_BLOCK - 1) / VIT_BLOCK, VIT_BLOCK, 0, get_cuda_stream()>>>(
			dQ_gpu, dK_gpu, dV_gpu, d_qkv_gpu, B, T, N, heads, d);
		CHECK_CUDA(cudaPeekAtLastError());
	}
	vit_sanitize_and_constrain_gpu(d_qkv_gpu, qkv_count, VIT_GRAD_CLAMP);
	check_nan_gpu("backward: after attention backward", d_qkv_gpu, qkv_count, l.index);

	{
		vit_sum_rows_gpu(d_qkv_gpu, l.bias_updates_gpu, B * T, 3 * N);
	}

	gemm_ongpu_tensor_op(1, 0, 3 * N, N, B * T, 1.0f,
		d_qkv_gpu, 3 * N,
		ln1_out_gpu, N,
		1.0f,
		l.weight_updates_gpu, N,
		use_tensor_op);

	gemm_ongpu_tensor_op(0, 0, B * T, N, 3 * N, 1.0f,
		d_qkv_gpu, 3 * N,
		l.weights_gpu, N,
		0.0f,
		d_ln1_out_gpu, N,
		use_tensor_op);
	vit_sanitize_and_constrain_gpu(d_ln1_out_gpu, token_count_n, VIT_GRAD_CLAMP);

	{
		int num = B * T;
		const int warp_threads = num * 32;
		vit_layernorm_backward_kernel<<<(warp_threads + VIT_BLOCK - 1) / VIT_BLOCK, VIT_BLOCK, 0, get_cuda_stream()>>>(
			d_ln1_out_gpu, l.vit_ln1_xhat_gpu, l.vit_ln1_var_gpu, l.vit_ln1_gamma_gpu,
			d_pre_ln1_gpu, l.vit_ln1_gamma_updates_gpu, l.vit_ln1_beta_updates_gpu, num, N);
		CHECK_CUDA(cudaPeekAtLastError());
	}

	axpy_ongpu(token_count_n, 1.0f, dout_tokens_gpu, 1, d_pre_ln1_gpu, 1);
	vit_sanitize_and_constrain_gpu(d_pre_ln1_gpu, token_count_n, VIT_GRAD_CLAMP);
	check_nan_gpu("backward: after layernorm1 backward", d_pre_ln1_gpu, token_count_n, l.index);

	if (l.vit_pos_embed_type == 0)
	{
		int num = token_count_n;
		vit_pos_embed_backward_kernel<<<(num + VIT_BLOCK - 1) / VIT_BLOCK, VIT_BLOCK, 0, get_cuda_stream()>>>(
			d_pre_ln1_gpu, l.vit_pos_embed_updates_gpu, B, T, N);
		CHECK_CUDA(cudaPeekAtLastError());
	}

	{
		int num = patch_count;
		vit_patchify_kernel<<<(num + VIT_BLOCK - 1) / VIT_BLOCK, VIT_BLOCK, 0, get_cuda_stream()>>>(
			state.input, l.vit_patch_tokens_gpu, B, C, H, W, P, S, pad, Hp, Wp);
		CHECK_CUDA(cudaPeekAtLastError());
		vit_sum_rows_gpu(d_pre_ln1_gpu, l.vit_patch_bias_updates_gpu, B * T, N);
		gemm_ongpu_tensor_op(1, 0, N, patch_dim, B * T, 1.0f,
			d_pre_ln1_gpu, N,
			l.vit_patch_tokens_gpu, patch_dim,
			1.0f,
			l.vit_patch_embed_updates_gpu, patch_dim,
			use_tensor_op);
		gemm_ongpu_tensor_op(0, 0, B * T, patch_dim, N, 1.0f,
			d_pre_ln1_gpu, N,
			l.vit_patch_embed_gpu, patch_dim,
			0.0f,
			l.vit_patch_delta_gpu, patch_dim,
			use_tensor_op);
	}

	if (state.delta)
	{
		reset_nan_and_inf(state.delta, B * C * H * W);
		int num = patch_count;
		vit_patch_delta_to_spatial_kernel<<<(num + VIT_BLOCK - 1) / VIT_BLOCK, VIT_BLOCK, 0, get_cuda_stream()>>>(
			l.vit_patch_delta_gpu, state.delta, B, C, H, W, P, S, pad, Hp, Wp);
		CHECK_CUDA(cudaPeekAtLastError());
		reset_nan_and_inf(state.delta, B * C * H * W);
		constrain_ongpu(B * C * H * W, 1.0f, state.delta, 1);
	}
	check_nan_gpu("backward: propagated state.delta", state.delta, state.delta ? B * C * H * W : 0, l.index);

}

void update_vit_layer_gpu(Darknet::Layer & l, int batch, float learning_rate_init, float momentum, float decay, float loss_scale)
{
	TAT(TATPARMS);

	const float lr = learning_rate_init * l.learning_rate_scale;
	const int C = l.c;
	const int N = l.n;
	const int T = l.out_h * l.out_w;
	const int patch_dim = l.vit_patch_size * l.vit_patch_size * C;
	const int ffn_hidden = l.vit_mlp_dim;

	reset_nan_and_inf(l.vit_patch_embed_updates_gpu, N * patch_dim);
	reset_nan_and_inf(l.vit_patch_bias_updates_gpu, N);
	reset_nan_and_inf(l.weight_updates_gpu, l.nweights);
	reset_nan_and_inf(l.bias_updates_gpu, 3 * N);
	reset_nan_and_inf(l.vit_wo_updates_gpu, N * N);
	reset_nan_and_inf(l.vit_wo_bias_updates_gpu, N);
	reset_nan_and_inf(l.vit_ln1_gamma_updates_gpu, N);
	reset_nan_and_inf(l.vit_ln1_beta_updates_gpu, N);
	reset_nan_and_inf(l.vit_ln2_gamma_updates_gpu, N);
	reset_nan_and_inf(l.vit_ln2_beta_updates_gpu, N);
	reset_nan_and_inf(l.vit_ffn_w1_updates_gpu, ffn_hidden * N);
	reset_nan_and_inf(l.vit_ffn_b1_updates_gpu, ffn_hidden);
	reset_nan_and_inf(l.vit_ffn_w2_updates_gpu, N * ffn_hidden);
	reset_nan_and_inf(l.vit_ffn_b2_updates_gpu, N);
	if (l.vit_pos_embed_type == 0) reset_nan_and_inf(l.vit_pos_embed_updates_gpu, T * N);
	reset_nan_and_inf(l.scale_updates_gpu, 12);

	if (loss_scale != 1.0f)
	{
		scal_ongpu(N * patch_dim, 1.0f / loss_scale, l.vit_patch_embed_updates_gpu, 1);
		scal_ongpu(N, 1.0f / loss_scale, l.vit_patch_bias_updates_gpu, 1);
		scal_ongpu(l.nweights, 1.0f / loss_scale, l.weight_updates_gpu, 1);
		scal_ongpu(3 * N, 1.0f / loss_scale, l.bias_updates_gpu, 1);
		scal_ongpu(N * N, 1.0f / loss_scale, l.vit_wo_updates_gpu, 1);
		scal_ongpu(N, 1.0f / loss_scale, l.vit_wo_bias_updates_gpu, 1);
		scal_ongpu(N, 1.0f / loss_scale, l.vit_ln1_gamma_updates_gpu, 1);
		scal_ongpu(N, 1.0f / loss_scale, l.vit_ln1_beta_updates_gpu, 1);
		scal_ongpu(N, 1.0f / loss_scale, l.vit_ln2_gamma_updates_gpu, 1);
		scal_ongpu(N, 1.0f / loss_scale, l.vit_ln2_beta_updates_gpu, 1);
		scal_ongpu(ffn_hidden * N, 1.0f / loss_scale, l.vit_ffn_w1_updates_gpu, 1);
		scal_ongpu(ffn_hidden, 1.0f / loss_scale, l.vit_ffn_b1_updates_gpu, 1);
		scal_ongpu(N * ffn_hidden, 1.0f / loss_scale, l.vit_ffn_w2_updates_gpu, 1);
		scal_ongpu(N, 1.0f / loss_scale, l.vit_ffn_b2_updates_gpu, 1);
		if (l.vit_pos_embed_type == 0) scal_ongpu(T * N, 1.0f / loss_scale, l.vit_pos_embed_updates_gpu, 1);
		scal_ongpu(12, 1.0f / loss_scale, l.scale_updates_gpu, 1);
	}

	// Match the transformer layer's global gradient clipping. The mHC residual
	// mixer adds trainable residual parameters, so ViT needs the same guard.
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

		accum_norm(l.vit_patch_embed_updates_gpu, N * patch_dim);
		accum_norm(l.vit_patch_bias_updates_gpu, N);
		accum_norm(l.weight_updates_gpu, l.nweights);
		accum_norm(l.bias_updates_gpu, 3 * N);
		accum_norm(l.vit_wo_updates_gpu, N * N);
		accum_norm(l.vit_wo_bias_updates_gpu, N);
		accum_norm(l.vit_ln1_gamma_updates_gpu, N);
		accum_norm(l.vit_ln1_beta_updates_gpu, N);
		accum_norm(l.vit_ln2_gamma_updates_gpu, N);
		accum_norm(l.vit_ln2_beta_updates_gpu, N);
		accum_norm(l.vit_ffn_w1_updates_gpu, ffn_hidden * N);
		accum_norm(l.vit_ffn_b1_updates_gpu, ffn_hidden);
		accum_norm(l.vit_ffn_w2_updates_gpu, N * ffn_hidden);
		accum_norm(l.vit_ffn_b2_updates_gpu, N);
		if (l.vit_pos_embed_type == 0) accum_norm(l.vit_pos_embed_updates_gpu, T * N);
		accum_norm(l.scale_updates_gpu, 12);

		const float global_norm = sqrtf(global_norm_sq);
		if (!std::isfinite(global_norm) || global_norm > max_grad_norm)
		{
			const float clip_coef = std::isfinite(global_norm) ? max_grad_norm / global_norm : 0.0f;
			scal_ongpu(N * patch_dim, clip_coef, l.vit_patch_embed_updates_gpu, 1);
			scal_ongpu(N, clip_coef, l.vit_patch_bias_updates_gpu, 1);
			scal_ongpu(l.nweights, clip_coef, l.weight_updates_gpu, 1);
			scal_ongpu(3 * N, clip_coef, l.bias_updates_gpu, 1);
			scal_ongpu(N * N, clip_coef, l.vit_wo_updates_gpu, 1);
			scal_ongpu(N, clip_coef, l.vit_wo_bias_updates_gpu, 1);
			scal_ongpu(N, clip_coef, l.vit_ln1_gamma_updates_gpu, 1);
			scal_ongpu(N, clip_coef, l.vit_ln1_beta_updates_gpu, 1);
			scal_ongpu(N, clip_coef, l.vit_ln2_gamma_updates_gpu, 1);
			scal_ongpu(N, clip_coef, l.vit_ln2_beta_updates_gpu, 1);
			scal_ongpu(ffn_hidden * N, clip_coef, l.vit_ffn_w1_updates_gpu, 1);
			scal_ongpu(ffn_hidden, clip_coef, l.vit_ffn_b1_updates_gpu, 1);
			scal_ongpu(N * ffn_hidden, clip_coef, l.vit_ffn_w2_updates_gpu, 1);
			scal_ongpu(N, clip_coef, l.vit_ffn_b2_updates_gpu, 1);
			if (l.vit_pos_embed_type == 0) scal_ongpu(T * N, clip_coef, l.vit_pos_embed_updates_gpu, 1);
			scal_ongpu(12, clip_coef, l.scale_updates_gpu, 1);
		}
	}

	{
		const int count = N * patch_dim;
		axpy_ongpu(count, -decay * batch, l.vit_patch_embed_gpu, 1, l.vit_patch_embed_updates_gpu, 1);
		axpy_ongpu(count, lr / batch, l.vit_patch_embed_updates_gpu, 1, l.vit_patch_embed_gpu, 1);
		scal_ongpu(count, momentum, l.vit_patch_embed_updates_gpu, 1);
	}
	axpy_ongpu(N, lr / batch, l.vit_patch_bias_updates_gpu, 1, l.vit_patch_bias_gpu, 1);
	scal_ongpu(N, momentum, l.vit_patch_bias_updates_gpu, 1);

	axpy_ongpu(l.nweights, -decay * batch, l.weights_gpu, 1, l.weight_updates_gpu, 1);
	axpy_ongpu(l.nweights, lr / batch, l.weight_updates_gpu, 1, l.weights_gpu, 1);
	scal_ongpu(l.nweights, momentum, l.weight_updates_gpu, 1);

	axpy_ongpu(3 * N, lr / batch, l.bias_updates_gpu, 1, l.biases_gpu, 1);
	scal_ongpu(3 * N, momentum, l.bias_updates_gpu, 1);

	{
		const int count = N * N;
		axpy_ongpu(count, -decay * batch, l.vit_wo_gpu, 1, l.vit_wo_updates_gpu, 1);
		axpy_ongpu(count, lr / batch, l.vit_wo_updates_gpu, 1, l.vit_wo_gpu, 1);
		scal_ongpu(count, momentum, l.vit_wo_updates_gpu, 1);
	}
	axpy_ongpu(N, lr / batch, l.vit_wo_bias_updates_gpu, 1, l.vit_wo_bias_gpu, 1);
	scal_ongpu(N, momentum, l.vit_wo_bias_updates_gpu, 1);

	// mHC residual mixer
	// mHC scales = slow parameter (0.1x LR). Matches CPU update_vit_layer to keep
	// branch coefficient from jumping at LR-schedule steps.
	axpy_ongpu(12, 0.1f * lr / batch, l.scale_updates_gpu, 1, l.scales_gpu, 1);
	vit_sanitize_and_constrain_gpu(l.scales_gpu, 12, VIT_MHC_PARAM_CLAMP);
	scal_ongpu(12, momentum, l.scale_updates_gpu, 1);

	axpy_ongpu(N, lr / batch, l.vit_ln1_gamma_updates_gpu, 1, l.vit_ln1_gamma_gpu, 1);
	scal_ongpu(N, momentum, l.vit_ln1_gamma_updates_gpu, 1);
	axpy_ongpu(N, lr / batch, l.vit_ln1_beta_updates_gpu, 1, l.vit_ln1_beta_gpu, 1);
	scal_ongpu(N, momentum, l.vit_ln1_beta_updates_gpu, 1);

	axpy_ongpu(N, lr / batch, l.vit_ln2_gamma_updates_gpu, 1, l.vit_ln2_gamma_gpu, 1);
	scal_ongpu(N, momentum, l.vit_ln2_gamma_updates_gpu, 1);
	axpy_ongpu(N, lr / batch, l.vit_ln2_beta_updates_gpu, 1, l.vit_ln2_beta_gpu, 1);
	scal_ongpu(N, momentum, l.vit_ln2_beta_updates_gpu, 1);

	{
		const int count = ffn_hidden * N;
		axpy_ongpu(count, -decay * batch, l.vit_ffn_w1_gpu, 1, l.vit_ffn_w1_updates_gpu, 1);
		axpy_ongpu(count, lr / batch, l.vit_ffn_w1_updates_gpu, 1, l.vit_ffn_w1_gpu, 1);
		scal_ongpu(count, momentum, l.vit_ffn_w1_updates_gpu, 1);
	}
	axpy_ongpu(ffn_hidden, lr / batch, l.vit_ffn_b1_updates_gpu, 1, l.vit_ffn_b1_gpu, 1);
	scal_ongpu(ffn_hidden, momentum, l.vit_ffn_b1_updates_gpu, 1);

	{
		const int count = N * ffn_hidden;
		axpy_ongpu(count, -decay * batch, l.vit_ffn_w2_gpu, 1, l.vit_ffn_w2_updates_gpu, 1);
		axpy_ongpu(count, lr / batch, l.vit_ffn_w2_updates_gpu, 1, l.vit_ffn_w2_gpu, 1);
		scal_ongpu(count, momentum, l.vit_ffn_w2_updates_gpu, 1);
	}
	axpy_ongpu(N, lr / batch, l.vit_ffn_b2_updates_gpu, 1, l.vit_ffn_b2_gpu, 1);
	scal_ongpu(N, momentum, l.vit_ffn_b2_updates_gpu, 1);

	if (l.vit_pos_embed_type == 0)
	{
		const int count = T * N;
		axpy_ongpu(count, -decay * batch, l.vit_pos_embed_gpu, 1, l.vit_pos_embed_updates_gpu, 1);
		axpy_ongpu(count, lr / batch, l.vit_pos_embed_updates_gpu, 1, l.vit_pos_embed_gpu, 1);
		scal_ongpu(count, momentum, l.vit_pos_embed_updates_gpu, 1);
	}

	check_nan_gpu("update: patch embed", l.vit_patch_embed_gpu, N * patch_dim, l.index);
	check_nan_gpu("update: weights", l.weights_gpu, l.nweights, l.index);
	check_nan_gpu("update: qkv bias", l.biases_gpu, 3 * N, l.index);
	check_nan_gpu("update: output projection", l.vit_wo_gpu, N * N, l.index);
	check_nan_gpu("update: mHC scales", l.scales_gpu, 12, l.index);
	check_nan_gpu("update: ln1 gamma", l.vit_ln1_gamma_gpu, N, l.index);
	check_nan_gpu("update: ln2 gamma", l.vit_ln2_gamma_gpu, N, l.index);
	check_nan_gpu("update: ffn w1", l.vit_ffn_w1_gpu, ffn_hidden * N, l.index);
	check_nan_gpu("update: ffn w2", l.vit_ffn_w2_gpu, N * ffn_hidden, l.index);
	check_nan_gpu("update: pos_embed", l.vit_pos_embed_gpu, T * N, l.index);
}

void push_vit_layer(Darknet::Layer & l)
{
	const int patch_dim = l.vit_patch_size * l.vit_patch_size * l.c;
	cuda_push_array(l.vit_patch_embed_gpu, l.vit_patch_embed, l.out_c * patch_dim);
	cuda_push_array(l.vit_patch_bias_gpu, l.vit_patch_bias, l.out_c);
	cuda_push_array(l.weights_gpu, l.weights, l.nweights);
	cuda_push_array(l.biases_gpu, l.biases, l.nbiases);
	cuda_push_array(l.vit_wo_gpu, l.vit_wo, l.out_c * l.out_c);
	cuda_push_array(l.vit_wo_bias_gpu, l.vit_wo_bias, l.out_c);
	cuda_push_array(l.vit_ln1_gamma_gpu, l.vit_ln1_gamma, l.out_c);
	cuda_push_array(l.vit_ln1_beta_gpu, l.vit_ln1_beta, l.out_c);
	cuda_push_array(l.vit_ln2_gamma_gpu, l.vit_ln2_gamma, l.out_c);
	cuda_push_array(l.vit_ln2_beta_gpu, l.vit_ln2_beta, l.out_c);
	cuda_push_array(l.vit_ffn_w1_gpu, l.vit_ffn_w1, l.out_c * l.vit_mlp_dim);
	cuda_push_array(l.vit_ffn_b1_gpu, l.vit_ffn_b1, l.vit_mlp_dim);
	cuda_push_array(l.vit_ffn_w2_gpu, l.vit_ffn_w2, l.out_c * l.vit_mlp_dim);
	cuda_push_array(l.vit_ffn_b2_gpu, l.vit_ffn_b2, l.out_c);
	cuda_push_array(l.vit_pos_embed_gpu, l.vit_pos_embed, l.out_h * l.out_w * l.out_c);
	cuda_push_array(l.scales_gpu, l.scales, 12);
}

void pull_vit_layer(Darknet::Layer & l)
{
	const int patch_dim = l.vit_patch_size * l.vit_patch_size * l.c;
	cuda_pull_array(l.vit_patch_embed_gpu, l.vit_patch_embed, l.out_c * patch_dim);
	cuda_pull_array(l.vit_patch_bias_gpu, l.vit_patch_bias, l.out_c);
	cuda_pull_array(l.weights_gpu, l.weights, l.nweights);
	cuda_pull_array(l.biases_gpu, l.biases, l.nbiases);
	cuda_pull_array(l.vit_wo_gpu, l.vit_wo, l.out_c * l.out_c);
	cuda_pull_array(l.vit_wo_bias_gpu, l.vit_wo_bias, l.out_c);
	cuda_pull_array(l.vit_ln1_gamma_gpu, l.vit_ln1_gamma, l.out_c);
	cuda_pull_array(l.vit_ln1_beta_gpu, l.vit_ln1_beta, l.out_c);
	cuda_pull_array(l.vit_ln2_gamma_gpu, l.vit_ln2_gamma, l.out_c);
	cuda_pull_array(l.vit_ln2_beta_gpu, l.vit_ln2_beta, l.out_c);
	cuda_pull_array(l.vit_ffn_w1_gpu, l.vit_ffn_w1, l.out_c * l.vit_mlp_dim);
	cuda_pull_array(l.vit_ffn_b1_gpu, l.vit_ffn_b1, l.vit_mlp_dim);
	cuda_pull_array(l.vit_ffn_w2_gpu, l.vit_ffn_w2, l.out_c * l.vit_mlp_dim);
	cuda_pull_array(l.vit_ffn_b2_gpu, l.vit_ffn_b2, l.out_c);
	cuda_pull_array(l.vit_pos_embed_gpu, l.vit_pos_embed, l.out_h * l.out_w * l.out_c);
	cuda_pull_array(l.scales_gpu, l.scales, 12);
}

#endif // DARKNET_GPU
