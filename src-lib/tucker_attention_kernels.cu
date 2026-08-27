#include "tucker_attention_layer.hpp"
#include "activations.hpp"
#ifdef DARKNET_HAS_FP8
#include "convolutional_layer.hpp"
#include "fp8_gemm.hpp"
#include "fp8_attention_kernels.hpp"
#endif
#ifdef DARKNET_HAS_FP4
#include "convolutional_layer.hpp"
#include "fp4_gemm.hpp"
#include "fp4_attention_kernels.hpp"
#endif

#include <cmath>

#if defined(CUDNN) && defined(CUDNN_HALF) && !defined(DARKNET_GPU_ROCM)
#define DARKNET_TUCKER_USE_CUDNN_HALF 1
#else
#define DARKNET_TUCKER_USE_CUDNN_HALF 0
#endif

#if DARKNET_TUCKER_USE_CUDNN_HALF && defined(DARKNET_GPU_CUDA) && defined(CUBLAS_VERSION) && CUBLAS_VERSION >= 11000
#define DARKNET_TUCKER_USE_CUBLAS_HALF 1
#else
#define DARKNET_TUCKER_USE_CUBLAS_HALF 0
#endif

#if DARKNET_TUCKER_USE_CUDNN_HALF
#include <cuda_fp16.h>
#endif

#ifdef DARKNET_GPU

namespace
{
	struct TuckerOffsetsGpu
	{
		size_t q_basis;
		size_t k_basis;
		size_t v_basis;
		size_t q_core;
		size_t k_core;
		size_t v_core;
		size_t o_core;
		size_t o_basis;
	};

	static TuckerOffsetsGpu gpu_offsets(const Darknet::Layer &l)
	{
		TuckerOffsetsGpu o = {};
		const size_t C = l.c;
		const size_t H = l.tucker_heads;
		const size_t D = l.tucker_head_dim;
		const size_t Rq = l.tucker_rank_q;
		const size_t Rk = l.tucker_rank_k;
		const size_t Rv = l.tucker_rank_v;
		const size_t Ro = l.tucker_rank_o;
		o.q_basis = 0;
		o.k_basis = o.q_basis + C * Rq;
		o.v_basis = o.k_basis + C * Rk;
		o.q_core = o.v_basis + C * Rv;
		o.k_core = o.q_core + H * Rq * D;
		o.v_core = o.k_core + H * Rk * D;
		o.o_core = o.v_core + H * Rv * D;
		o.o_basis = o.o_core + H * D * Ro;
		return o;
	}

#if !DARKNET_TUCKER_USE_CUDNN_HALF || !DARKNET_TUCKER_USE_CUBLAS_HALF
	static int tucker_pow2_threads(int n)
	{
		int threads = 32;
		while (threads < n && threads < 1024) threads <<= 1;
		return threads;
	}
#endif

#if !DARKNET_TUCKER_USE_CUDNN_HALF
	__device__ __forceinline__ float tucker_qk_score_from_projected(
		const float *q, const float *k,
		int gw, int tq, int tk, int hidx,
		int T, int heads, int D)
	{
		const float *qv = q + ((gw * T + tq) * heads + hidx) * D;
		const float *kv = k + ((gw * T + tk) * heads + hidx) * D;
		float dot = 0.0f;
		for (int d = 0; d < D; ++d)
		{
			dot += qv[d] * kv[d];
		}
		return dot * rsqrtf((float)D);
	}
#endif

#if !DARKNET_TUCKER_USE_CUDNN_HALF || !DARKNET_TUCKER_USE_CUBLAS_HALF
	__global__ void tucker_pack_windows_kernel(
		const int total,
		const float *input,
		float *tokens,
		int B, int H, int W, int C,
		int window, int win_h, int win_w)
	{
		const int idx = blockIdx.x * blockDim.x + threadIdx.x;
		if (idx >= total) return;

		const int c = idx % C;
		const int t = (idx / C) % (window * window);
		const int gw = idx / (C * window * window);
		const int b = gw / (win_h * win_w);
		const int local_w = gw % (win_h * win_w);
		const int wy = local_w / win_w;
		const int wx = local_w % win_w;
		const int yy = t / window;
		const int xx = t % window;
		const int y = wy * window + yy;
		const int x = wx * window + xx;

		float value = 0.0f;
		if (b < B && y < H && x < W)
		{
			value = input[((b * C + c) * H + y) * W + x];
		}
		tokens[idx] = value;
	}
#endif

#if !DARKNET_TUCKER_USE_CUDNN_HALF
	__global__ void tucker_project_qkv_latent_kernel(
		const int total,
		const float *__restrict__ tokens,
		const float *__restrict__ q_basis,
		const float *__restrict__ k_basis,
		const float *__restrict__ v_basis,
		float *__restrict__ q_latent,
		float *__restrict__ k_latent,
		float *__restrict__ v_latent,
		int windows, int T, int C, int Rq, int Rk, int Rv)
	{
		int idx = blockIdx.x * blockDim.x + threadIdx.x;
		if (idx >= total) return;

		const int q_count = windows * T * Rq;
		const int k_count = windows * T * Rk;
		const float *basis = q_basis;
		float *latent = q_latent;
		int R = Rq;
		if (idx >= q_count)
		{
			idx -= q_count;
			if (idx < k_count)
			{
				basis = k_basis;
				latent = k_latent;
				R = Rk;
			}
			else
			{
				idx -= k_count;
				basis = v_basis;
				latent = v_latent;
				R = Rv;
			}
		}

		const int r = idx % R;
		const int t = (idx / R) % T;
		const int gw = idx / (R * T);
		const float *token = tokens + (gw * T + t) * C;

		float sum = 0.0f;
		for (int c = 0; c < C; ++c)
		{
			sum += token[c] * basis[c * R + r];
		}
		latent[idx] = sum;
	}

	__global__ void tucker_expand_qkv_heads_kernel(
		const int total,
		const float *__restrict__ q_latent,
		const float *__restrict__ k_latent,
		const float *__restrict__ v_latent,
		const float *__restrict__ q_core,
		const float *__restrict__ k_core,
		const float *__restrict__ v_core,
		float *__restrict__ q,
		float *__restrict__ k,
		float *__restrict__ v,
		int head_count, int T, int heads, int D, int Rq, int Rk, int Rv)
	{
		int idx = blockIdx.x * blockDim.x + threadIdx.x;
		if (idx >= total) return;

		const float *latent = q_latent;
		const float *core = q_core;
		float *out = q;
		int R = Rq;
		if (idx >= head_count)
		{
			idx -= head_count;
			if (idx < head_count)
			{
				latent = k_latent;
				core = k_core;
				out = k;
				R = Rk;
			}
			else
			{
				idx -= head_count;
				latent = v_latent;
				core = v_core;
				out = v;
				R = Rv;
			}
		}

		const int d = idx % D;
		const int hidx = (idx / D) % heads;
		const int t = (idx / (D * heads)) % T;
		const int gw = idx / (D * heads * T);

		float sum = 0.0f;
		for (int r = 0; r < R; ++r)
		{
			sum += latent[(gw * T + t) * R + r] * core[(hidx * R + r) * D + d];
		}
		out[idx] = sum;
	}

	__global__ void tucker_scores_kernel(
		const int total,
		const float *q,
		const float *k,
		float *scores,
		int T, int heads, int D)
	{
		const int idx = blockIdx.x * blockDim.x + threadIdx.x;
		if (idx >= total) return;

		const int tk = idx % T;
		const int tq = (idx / T) % T;
		const int hidx = (idx / (T * T)) % heads;
		const int gw = idx / (T * T * heads);

		float dot = 0.0f;
		for (int d = 0; d < D; ++d)
		{
			dot += q[((gw * T + tq) * heads + hidx) * D + d] *
				k[((gw * T + tk) * heads + hidx) * D + d];
		}
		scores[idx] = dot * rsqrtf((float)D);
	}

	__global__ void tucker_scores_softmax_kernel(
		const int rows,
		const float *__restrict__ q,
		const float *__restrict__ k,
		float *__restrict__ attn,
		int T, int heads, int D)
	{
		const int row_idx = blockIdx.x;
		if (row_idx >= rows) return;

		extern __shared__ float shared[];
		const int tid = threadIdx.x;
		const int tq = row_idx % T;
		const int hidx = (row_idx / T) % heads;
		const int gw = row_idx / (T * heads);
		float *row = attn + ((gw * heads + hidx) * T + tq) * T;

		if (T <= blockDim.x)
		{
			float score = -3.402823466e+38F;
			if (tid < T)
			{
				score = tucker_qk_score_from_projected(q, k, gw, tq, tid, hidx, T, heads, D);
			}
			shared[tid] = score;
			__syncthreads();

			for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1)
			{
				if (tid < stride) shared[tid] = fmaxf(shared[tid], shared[tid + stride]);
				__syncthreads();
			}
			const float max_score = shared[0];

			float e = 0.0f;
			if (tid < T)
			{
				e = expf(score - max_score);
				row[tid] = e;
			}
			shared[tid] = e;
			__syncthreads();

			for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1)
			{
				if (tid < stride) shared[tid] += shared[tid + stride];
				__syncthreads();
			}
			const float inv_denom = shared[0] > 0.0f ? 1.0f / shared[0] : 0.0f;
			if (tid < T) row[tid] *= inv_denom;
			return;
		}

		float local_max = -3.402823466e+38F;
		for (int tk = tid; tk < T; tk += blockDim.x)
		{
			local_max = fmaxf(local_max, tucker_qk_score_from_projected(q, k, gw, tq, tk, hidx, T, heads, D));
		}
		shared[tid] = local_max;
		__syncthreads();
		for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1)
		{
			if (tid < stride) shared[tid] = fmaxf(shared[tid], shared[tid + stride]);
			__syncthreads();
		}
		const float max_score = shared[0];

		float local_sum = 0.0f;
		for (int tk = tid; tk < T; tk += blockDim.x)
		{
			local_sum += expf(tucker_qk_score_from_projected(q, k, gw, tq, tk, hidx, T, heads, D) - max_score);
		}
		shared[tid] = local_sum;
		__syncthreads();
		for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1)
		{
			if (tid < stride) shared[tid] += shared[tid + stride];
			__syncthreads();
		}
		const float inv_denom = shared[0] > 0.0f ? 1.0f / shared[0] : 0.0f;

		for (int tk = tid; tk < T; tk += blockDim.x)
		{
			row[tk] = expf(tucker_qk_score_from_projected(q, k, gw, tq, tk, hidx, T, heads, D) - max_score) * inv_denom;
		}
	}

	__global__ void tucker_context_from_attn_kernel(
		const int rows,
		const float *__restrict__ attn,
		const float *__restrict__ v,
		float *__restrict__ context,
		int T, int heads, int D)
	{
		const int row_idx = blockIdx.x;
		if (row_idx >= rows) return;

		const int tq = row_idx % T;
		const int hidx = (row_idx / T) % heads;
		const int gw = row_idx / (T * heads);
		const float *row = attn + ((gw * heads + hidx) * T + tq) * T;

		for (int d = threadIdx.x; d < D; d += blockDim.x)
		{
			float sum = 0.0f;
			for (int tk = 0; tk < T; ++tk)
			{
				sum += row[tk] * v[((gw * T + tk) * heads + hidx) * D + d];
			}
			context[((gw * T + tq) * heads + hidx) * D + d] = sum;
		}
	}



#ifdef CUDNN
	__global__ void tucker_make_dattn_float_kernel(
		const int rows,
		const float *__restrict__ v,
		const float *__restrict__ d_context,
		float *__restrict__ d_attn,
		int T, int heads, int D)
	{
		const int row_idx = blockIdx.x;
		if (row_idx >= rows) return;

		const int tq = row_idx % T;
		const int hidx = (row_idx / T) % heads;
		const int gw = row_idx / (T * heads);
		const int qbase = ((gw * T + tq) * heads + hidx) * D;
		float *row = d_attn + ((gw * heads + hidx) * T + tq) * T;

		for (int tk = threadIdx.x; tk < T; tk += blockDim.x)
		{
			const int kbase = ((gw * T + tk) * heads + hidx) * D;
			float sum = 0.0f;
			for (int d = 0; d < D; ++d)
			{
				sum += d_context[qbase + d] * v[kbase + d];
			}
			row[tk] = sum;
		}
	}

	__global__ void tucker_qkv_backward_from_softmax_float_kernel(
		const int total,
		const float *__restrict__ attn,
		const float *__restrict__ d_scores,
		const float *__restrict__ q,
		const float *__restrict__ k,
		const float *__restrict__ d_context,
		float *__restrict__ d_q,
		float *__restrict__ d_k,
		float *__restrict__ d_v,
		int T, int heads, int D)
	{
		const int idx = blockIdx.x * blockDim.x + threadIdx.x;
		if (idx >= total) return;

		const int d = idx % D;
		const int hidx = (idx / D) % heads;
		const int t = (idx / (D * heads)) % T;
		const int gw = idx / (D * heads * T);
		const float scale = rsqrtf((float)D);
		const int base = ((gw * T + t) * heads + hidx) * D;

		float dq = 0.0f;
		const float *score_q_row = d_scores + ((gw * heads + hidx) * T + t) * T;
		for (int tk = 0; tk < T; ++tk)
		{
			const int kbase = ((gw * T + tk) * heads + hidx) * D;
			dq += score_q_row[tk] * k[kbase + d] * scale;
		}
		d_q[base + d] = dq;

		float dk = 0.0f;
		for (int tq = 0; tq < T; ++tq)
		{
			const int qidx = ((gw * T + tq) * heads + hidx) * D + d;
			const float *score_row = d_scores + ((gw * heads + hidx) * T + tq) * T;
			dk += score_row[t] * q[qidx] * scale;
		}
		d_k[base + d] = dk;

		float dv = 0.0f;
		for (int tq = 0; tq < T; ++tq)
		{
			const float *attn_row = attn + ((gw * heads + hidx) * T + tq) * T;
			const int dcidx = ((gw * T + tq) * heads + hidx) * D + d;
			dv += attn_row[t] * d_context[dcidx];
		}
		d_v[base + d] = dv;
	}
#endif
#endif

#if DARKNET_TUCKER_USE_CUDNN_HALF
	__device__ __forceinline__ float tucker_h2f(const __half v)
	{
		return __half2float(v);
	}

	__device__ __forceinline__ __half tucker_f2h(const float v)
	{
		return __float2half_rn(v);
	}

#if DARKNET_TUCKER_USE_CUBLAS_HALF
	static void tucker_gemm_half(
		int TA, int TB, int M, int N, int K,
		float ALPHA,
		const __half *A, int lda,
		const __half *B, int ldb,
		float BETA,
		void *C, int ldc,
		cudaDataType_t c_type)
	{
		if (M == 0 || N == 0 || K == 0) return;

		cublasHandle_t handle = blas_handle();
		CHECK_CUBLAS(cublasSetStream(handle, get_cuda_stream()));
		cublasStatus_t status = cublasGemmEx(
			handle,
			(TB ? CUBLAS_OP_T : CUBLAS_OP_N),
			(TA ? CUBLAS_OP_T : CUBLAS_OP_N),
			N, M, K,
			&ALPHA,
			B, CUDA_R_16F, ldb,
			A, CUDA_R_16F, lda,
			&BETA,
			C, c_type, ldc,
			CUBLAS_COMPUTE_32F,
			CUBLAS_GEMM_DEFAULT_TENSOR_OP);
		if (status != CUBLAS_STATUS_SUCCESS)
		{
			status = cublasGemmEx(
				handle,
				(TB ? CUBLAS_OP_T : CUBLAS_OP_N),
				(TA ? CUBLAS_OP_T : CUBLAS_OP_N),
				N, M, K,
				&ALPHA,
				B, CUDA_R_16F, ldb,
				A, CUDA_R_16F, lda,
				&BETA,
				C, c_type, ldc,
				CUBLAS_COMPUTE_32F,
				CUBLAS_GEMM_DEFAULT);
		}
		CHECK_CUBLAS(status);
	}

	static void tucker_gemm_half_strided_batched(
		int TA, int TB, int M, int N, int K,
		float ALPHA,
		const __half *A, int lda, long long strideA,
		const __half *B, int ldb, long long strideB,
		float BETA,
		void *C, int ldc, long long strideC,
		int batch_count,
		cudaDataType_t c_type)
	{
		if (M == 0 || N == 0 || K == 0 || batch_count == 0) return;

		cublasHandle_t handle = blas_handle();
		CHECK_CUBLAS(cublasSetStream(handle, get_cuda_stream()));
		cublasStatus_t status = cublasGemmStridedBatchedEx(
			handle,
			(TB ? CUBLAS_OP_T : CUBLAS_OP_N),
			(TA ? CUBLAS_OP_T : CUBLAS_OP_N),
			N, M, K,
			&ALPHA,
			B, CUDA_R_16F, ldb, strideB,
			A, CUDA_R_16F, lda, strideA,
			&BETA,
			C, c_type, ldc, strideC,
			batch_count,
			CUBLAS_COMPUTE_32F,
			CUBLAS_GEMM_DEFAULT_TENSOR_OP);
		if (status != CUBLAS_STATUS_SUCCESS)
		{
			status = cublasGemmStridedBatchedEx(
				handle,
				(TB ? CUBLAS_OP_T : CUBLAS_OP_N),
				(TA ? CUBLAS_OP_T : CUBLAS_OP_N),
				N, M, K,
				&ALPHA,
				B, CUDA_R_16F, ldb, strideB,
				A, CUDA_R_16F, lda, strideA,
				&BETA,
				C, c_type, ldc, strideC,
				batch_count,
				CUBLAS_COMPUTE_32F,
				CUBLAS_GEMM_DEFAULT);
		}
		CHECK_CUBLAS(status);
	}
#endif

	__global__ void tucker_pack_windows_half_kernel(
		const int total,
		const float *__restrict__ input,
		__half *__restrict__ tokens,
		int B, int H, int W, int C,
		int window, int win_h, int win_w)
	{
		const int idx = blockIdx.x * blockDim.x + threadIdx.x;
		if (idx >= total) return;

		const int c = idx % C;
		const int t = (idx / C) % (window * window);
		const int gw = idx / (C * window * window);
		const int b = gw / (win_h * win_w);
		const int local_w = gw % (win_h * win_w);
		const int wy = local_w / win_w;
		const int wx = local_w % win_w;
		const int yy = t / window;
		const int xx = t % window;
		const int y = wy * window + yy;
		const int x = wx * window + xx;

		float value = 0.0f;
		if (b < B && y < H && x < W)
		{
			value = input[((b * C + c) * H + y) * W + x];
		}
		tokens[idx] = tucker_f2h(value);
	}

	__global__ void tucker_output_from_rank_half_kernel(
		const int total,
		const float *__restrict__ input,
		const __half *__restrict__ rank_output,
		const float *__restrict__ biases,
		float *__restrict__ output,
		int B, int H, int W, int C, int N,
		int window, int win_h, int win_w)
	{
		const int idx = blockIdx.x * blockDim.x + threadIdx.x;
		if (idx >= total) return;

		const int x = idx % W;
		const int y = (idx / W) % H;
		const int n = (idx / (W * H)) % N;
		const int b = idx / (W * H * N);
		const int wy = y / window;
		const int wx = x / window;
		const int t = (y % window) * window + (x % window);
		const int gw = (b * win_h + wy) * win_w + wx;
		const int T = window * window;

		float out = tucker_h2f(rank_output[(gw * T + t) * N + n]);
		if (biases) out += biases[n];
		if (n < C) out += input[((b * C + n) * H + y) * W + x];
		output[idx] = out;
	}

	__global__ void tucker_pack_output_delta_half_kernel(
		const int total,
		const float *__restrict__ output_delta,
		__half *__restrict__ d_out,
		float *__restrict__ bias_updates,
		float *__restrict__ input_delta,
		int B, int H, int W, int C, int N,
		int window, int win_h, int win_w)
	{
		const int idx = blockIdx.x * blockDim.x + threadIdx.x;
		if (idx >= total) return;

		const int x = idx % W;
		const int y = (idx / W) % H;
		const int n = (idx / (W * H)) % N;
		const int b = idx / (W * H * N);
		const float dy = output_delta[idx];
		const int wy = y / window;
		const int wx = x / window;
		const int t = (y % window) * window + (x % window);
		const int gw = (b * win_h + wy) * win_w + wx;
		const int T = window * window;

		d_out[(gw * T + t) * N + n] = tucker_f2h(dy);
		if (dy != 0.0f)
		{
			if (bias_updates) atomicAdd(bias_updates + n, dy);
			if (input_delta && n < C)
			{
				atomicAdd(input_delta + ((b * C + n) * H + y) * W + x, dy);
			}
		}
	}

	__global__ void tucker_scatter_token_delta_half_kernel(
		const int total,
		const __half *__restrict__ d_tokens,
		float *__restrict__ input_delta,
		int B, int H, int W, int C,
		int window, int win_h, int win_w)
	{
		const int idx = blockIdx.x * blockDim.x + threadIdx.x;
		if (idx >= total || input_delta == nullptr) return;

		const int c = idx % C;
		const int t = (idx / C) % (window * window);
		const int gw = idx / (C * window * window);
		const int b = gw / (win_h * win_w);
		const int local_w = gw % (win_h * win_w);
		const int wy = local_w / win_w;
		const int wx = local_w % win_w;
		const int yy = t / window;
		const int xx = t % window;
		const int y = wy * window + yy;
		const int x = wx * window + xx;

		if (b < B && y < H && x < W)
		{
			atomicAdd(input_delta + ((b * C + c) * H + y) * W + x, tucker_h2f(d_tokens[idx]));
		}
	}

#if !DARKNET_TUCKER_USE_CUBLAS_HALF
	__global__ void tucker_project_qkv_latent_half_weight_kernel(
		const int total,
		const float *__restrict__ tokens,
		const __half *__restrict__ q_basis,
		const __half *__restrict__ k_basis,
		const __half *__restrict__ v_basis,
		float *__restrict__ q_latent,
		float *__restrict__ k_latent,
		float *__restrict__ v_latent,
		int windows, int T, int C, int Rq, int Rk, int Rv)
	{
		int idx = blockIdx.x * blockDim.x + threadIdx.x;
		if (idx >= total) return;

		const int q_count = windows * T * Rq;
		const int k_count = windows * T * Rk;
		const __half *basis = q_basis;
		float *latent = q_latent;
		int R = Rq;
		if (idx >= q_count)
		{
			idx -= q_count;
			if (idx < k_count)
			{
				basis = k_basis;
				latent = k_latent;
				R = Rk;
			}
			else
			{
				idx -= k_count;
				basis = v_basis;
				latent = v_latent;
				R = Rv;
			}
		}

		const int r = idx % R;
		const int t = (idx / R) % T;
		const int gw = idx / (R * T);
		const float *token = tokens + (gw * T + t) * C;

		float sum = 0.0f;
		for (int c = 0; c < C; ++c)
		{
			sum += token[c] * tucker_h2f(basis[c * R + r]);
		}
		latent[idx] = sum;
	}

	__global__ void tucker_expand_qkv_heads_half_kernel(
		const int total,
		const float *__restrict__ q_latent,
		const float *__restrict__ k_latent,
		const float *__restrict__ v_latent,
		const __half *__restrict__ q_core,
		const __half *__restrict__ k_core,
		const __half *__restrict__ v_core,
		__half *__restrict__ q,
		__half *__restrict__ k,
		__half *__restrict__ v,
		int head_count, int T, int heads, int D, int Rq, int Rk, int Rv)
	{
		int idx = blockIdx.x * blockDim.x + threadIdx.x;
		if (idx >= total) return;

		const float *latent = q_latent;
		const __half *core = q_core;
		__half *out = q;
		int R = Rq;
		if (idx >= head_count)
		{
			idx -= head_count;
			if (idx < head_count)
			{
				latent = k_latent;
				core = k_core;
				out = k;
				R = Rk;
			}
			else
			{
				idx -= head_count;
				latent = v_latent;
				core = v_core;
				out = v;
				R = Rv;
			}
		}

		const int d = idx % D;
		const int hidx = (idx / D) % heads;
		const int t = (idx / (D * heads)) % T;
		const int gw = idx / (D * heads * T);

		float sum = 0.0f;
		for (int r = 0; r < R; ++r)
		{
			sum += latent[(gw * T + t) * R + r] * tucker_h2f(core[(hidx * R + r) * D + d]);
		}
		out[idx] = tucker_f2h(sum);
	}

	__global__ void tucker_scores_half_kernel(
		const int total,
		const __half *__restrict__ q,
		const __half *__restrict__ k,
		__half *__restrict__ scores,
		int T, int heads, int D)
	{
		const int idx = blockIdx.x * blockDim.x + threadIdx.x;
		if (idx >= total) return;

		const int tk = idx % T;
		const int tq = (idx / T) % T;
		const int hidx = (idx / (T * T)) % heads;
		const int gw = idx / (T * T * heads);
		const int qbase = ((gw * T + tq) * heads + hidx) * D;
		const int kbase = ((gw * T + tk) * heads + hidx) * D;

		float dot = 0.0f;
		for (int d = 0; d < D; ++d)
		{
			dot += tucker_h2f(q[qbase + d]) * tucker_h2f(k[kbase + d]);
		}
		scores[idx] = tucker_f2h(dot * rsqrtf((float)D));
	}

	__global__ void tucker_context_from_attn_half_kernel(
		const int rows,
		const __half *__restrict__ attn,
		const __half *__restrict__ v,
		__half *__restrict__ context,
		int T, int heads, int D)
	{
		const int row_idx = blockIdx.x;
		if (row_idx >= rows) return;

		const int tq = row_idx % T;
		const int hidx = (row_idx / T) % heads;
		const int gw = row_idx / (T * heads);
		const __half *row = attn + ((gw * heads + hidx) * T + tq) * T;

		for (int d = threadIdx.x; d < D; d += blockDim.x)
		{
			float sum = 0.0f;
			for (int tk = 0; tk < T; ++tk)
			{
				sum += tucker_h2f(row[tk]) * tucker_h2f(v[((gw * T + tk) * heads + hidx) * D + d]);
			}
			context[((gw * T + tq) * heads + hidx) * D + d] = tucker_f2h(sum);
		}
	}

	__global__ void tucker_context_to_rank_half_kernel(
		const int total,
		const __half *__restrict__ context,
		const __half *__restrict__ o_core,
		float *__restrict__ o_mix,
		int T, int heads, int D, int Ro)
	{
		const int idx = blockIdx.x * blockDim.x + threadIdx.x;
		if (idx >= total) return;

		const int r = idx % Ro;
		const int t = (idx / Ro) % T;
		const int gw = idx / (Ro * T);

		float sum = 0.0f;
		for (int hidx = 0; hidx < heads; ++hidx)
		{
			for (int d = 0; d < D; ++d)
			{
				const int hd = hidx * D + d;
				sum += tucker_h2f(context[((gw * T + t) * heads + hidx) * D + d]) * tucker_h2f(o_core[hd * Ro + r]);
			}
		}
		o_mix[idx] = sum;
	}

	__global__ void tucker_output_half_basis_kernel(
		const int total,
		const float *__restrict__ input,
		const float *__restrict__ o_mix,
		const __half *__restrict__ o_basis,
		const float *__restrict__ biases,
		float *__restrict__ output,
		int B, int H, int W, int C, int N,
		int window, int win_h, int win_w,
		int Ro)
	{
		const int idx = blockIdx.x * blockDim.x + threadIdx.x;
		if (idx >= total) return;

		const int x = idx % W;
		const int y = (idx / W) % H;
		const int n = (idx / (W * H)) % N;
		const int b = idx / (W * H * N);
		const int wy = y / window;
		const int wx = x / window;
		const int t = (y % window) * window + (x % window);
		const int gw = (b * win_h + wy) * win_w + wx;
		const int T = window * window;

		float out = biases ? biases[n] : 0.0f;
		for (int r = 0; r < Ro; ++r)
		{
			out += o_mix[(gw * T + t) * Ro + r] * tucker_h2f(o_basis[r * N + n]);
		}
		if (n < C) out += input[((b * C + n) * H + y) * W + x];
		output[idx] = out;
	}

	__global__ void tucker_output_to_rank_backward_half_basis_kernel(
		const int total,
		const float *__restrict__ output_delta,
		const float *__restrict__ o_mix,
		const __half *__restrict__ o_basis,
		float *__restrict__ do_basis,
		float *__restrict__ bias_updates,
		float *__restrict__ input_delta,
		float *__restrict__ d_o_mix,
		int B, int H, int W, int C, int N,
		int window, int win_h, int win_w,
		int Ro)
	{
		const int idx = blockIdx.x * blockDim.x + threadIdx.x;
		if (idx >= total) return;

		const int x = idx % W;
		const int y = (idx / W) % H;
		const int n = (idx / (W * H)) % N;
		const int b = idx / (W * H * N);
		const float dy = output_delta[idx];
		if (dy == 0.0f) return;

		if (bias_updates) atomicAdd(bias_updates + n, dy);
		if (input_delta && n < C)
		{
			atomicAdd(input_delta + ((b * C + n) * H + y) * W + x, dy);
		}

		const int wy = y / window;
		const int wx = x / window;
		const int t = (y % window) * window + (x % window);
		const int T = window * window;
		const int gw = (b * win_h + wy) * win_w + wx;

		for (int r = 0; r < Ro; ++r)
		{
			atomicAdd(do_basis + r * N + n, o_mix[(gw * T + t) * Ro + r] * dy);
			atomicAdd(d_o_mix + (gw * T + t) * Ro + r, tucker_h2f(o_basis[r * N + n]) * dy);
		}
	}

	__global__ void tucker_rank_to_context_backward_half_core_kernel(
		const int total,
		const __half *__restrict__ context,
		const float *__restrict__ d_o_mix,
		const __half *__restrict__ o_core,
		float *__restrict__ do_core,
		__half *__restrict__ d_context,
		int T, int heads, int D, int Ro)
	{
		const int idx = blockIdx.x * blockDim.x + threadIdx.x;
		if (idx >= total) return;

		const int d = idx % D;
		const int hidx = (idx / D) % heads;
		const int t = (idx / (D * heads)) % T;
		const int gw = idx / (D * heads * T);
		const int hd = hidx * D + d;

		float dctx = 0.0f;
		const float ctx = tucker_h2f(context[idx]);
		for (int r = 0; r < Ro; ++r)
		{
			const float grad = d_o_mix[(gw * T + t) * Ro + r];
			atomicAdd(do_core + hd * Ro + r, ctx * grad);
			dctx += tucker_h2f(o_core[hd * Ro + r]) * grad;
		}
		d_context[idx] = tucker_f2h(dctx);
	}

	__global__ void tucker_make_dattn_half_kernel(
		const int rows,
		const __half *__restrict__ v,
		const __half *__restrict__ d_context,
		__half *__restrict__ d_attn,
		int T, int heads, int D)
	{
		const int row_idx = blockIdx.x;
		if (row_idx >= rows) return;

		const int tq = row_idx % T;
		const int hidx = (row_idx / T) % heads;
		const int gw = row_idx / (T * heads);
		const int qbase = ((gw * T + tq) * heads + hidx) * D;
		__half *row = d_attn + ((gw * heads + hidx) * T + tq) * T;

		for (int tk = threadIdx.x; tk < T; tk += blockDim.x)
		{
			const int kbase = ((gw * T + tk) * heads + hidx) * D;
			float sum = 0.0f;
			for (int d = 0; d < D; ++d)
			{
				sum += tucker_h2f(d_context[qbase + d]) * tucker_h2f(v[kbase + d]);
			}
			row[tk] = tucker_f2h(sum);
		}
	}

	__global__ void tucker_qkv_backward_from_softmax_half_kernel(
		const int total,
		const __half *__restrict__ attn,
		const __half *__restrict__ d_scores,
		const __half *__restrict__ q,
		const __half *__restrict__ k,
		const __half *__restrict__ d_context,
		__half *__restrict__ d_q,
		__half *__restrict__ d_k,
		__half *__restrict__ d_v,
		int T, int heads, int D)
	{
		const int idx = blockIdx.x * blockDim.x + threadIdx.x;
		if (idx >= total) return;

		const int d = idx % D;
		const int hidx = (idx / D) % heads;
		const int t = (idx / (D * heads)) % T;
		const int gw = idx / (D * heads * T);
		const float scale = rsqrtf((float)D);

		float dq = 0.0f;
		const int qbase = ((gw * T + t) * heads + hidx) * D;
		const __half *score_q_row = d_scores + ((gw * heads + hidx) * T + t) * T;
		for (int tk = 0; tk < T; ++tk)
		{
			const int kbase = ((gw * T + tk) * heads + hidx) * D;
			dq += tucker_h2f(score_q_row[tk]) * tucker_h2f(k[kbase + d]) * scale;
		}
		d_q[qbase + d] = tucker_f2h(dq);

		float dk = 0.0f;
		for (int tq = 0; tq < T; ++tq)
		{
			const int qidx = ((gw * T + tq) * heads + hidx) * D + d;
			const __half *score_row = d_scores + ((gw * heads + hidx) * T + tq) * T;
			dk += tucker_h2f(score_row[t]) * tucker_h2f(q[qidx]) * scale;
		}
		d_k[qbase + d] = tucker_f2h(dk);

		float dv = 0.0f;
		for (int tq = 0; tq < T; ++tq)
		{
			const __half *attn_row = attn + ((gw * heads + hidx) * T + tq) * T;
			const int dcidx = ((gw * T + tq) * heads + hidx) * D + d;
			dv += tucker_h2f(attn_row[t]) * tucker_h2f(d_context[dcidx]);
		}
		d_v[qbase + d] = tucker_f2h(dv);
	}

	__global__ void tucker_expand_backward_half_kernel(
		const int total,
		const float *__restrict__ latent,
		const __half *__restrict__ core,
		const __half *__restrict__ d_head,
		float *__restrict__ d_latent,
		float *__restrict__ d_core,
		int T, int heads, int R, int D)
	{
		const int idx = blockIdx.x * blockDim.x + threadIdx.x;
		if (idx >= total) return;

		const int d = idx % D;
		const int hidx = (idx / D) % heads;
		const int t = (idx / (D * heads)) % T;
		const int gw = idx / (D * heads * T);
		const float grad = tucker_h2f(d_head[idx]);
		if (grad == 0.0f) return;

		for (int r = 0; r < R; ++r)
		{
			atomicAdd(d_core + (hidx * R + r) * D + d, latent[(gw * T + t) * R + r] * grad);
			atomicAdd(d_latent + (gw * T + t) * R + r, tucker_h2f(core[(hidx * R + r) * D + d]) * grad);
		}
	}

	__global__ void tucker_basis_backward_half_weight_kernel(
		const int total,
		const float *__restrict__ tokens,
		const __half *__restrict__ q_basis, const __half *__restrict__ k_basis, const __half *__restrict__ v_basis,
		const float *__restrict__ d_q_latent, const float *__restrict__ d_k_latent, const float *__restrict__ d_v_latent,
		float *__restrict__ dq_basis, float *__restrict__ dk_basis, float *__restrict__ dv_basis,
		float *__restrict__ input_delta,
		int B, int H, int W, int C,
		int window, int win_h, int win_w,
		int Rq, int Rk, int Rv)
	{
		const int idx = blockIdx.x * blockDim.x + threadIdx.x;
		if (idx >= total) return;

		const int c = idx % C;
		const int t = (idx / C) % (window * window);
		const int gw = idx / (C * window * window);
		const int b = gw / (win_h * win_w);
		const int local_w = gw % (win_h * win_w);
		const int wy = local_w / win_w;
		const int wx = local_w % win_w;
		const int yy = t / window;
		const int xx = t % window;
		const int y = wy * window + yy;
		const int x = wx * window + xx;
		const int T = window * window;
		const float token = tokens[idx];

		float dx = 0.0f;
		for (int r = 0; r < Rq; ++r)
		{
			const float grad = d_q_latent[(gw * T + t) * Rq + r];
			atomicAdd(dq_basis + c * Rq + r, token * grad);
			dx += tucker_h2f(q_basis[c * Rq + r]) * grad;
		}
		for (int r = 0; r < Rk; ++r)
		{
			const float grad = d_k_latent[(gw * T + t) * Rk + r];
			atomicAdd(dk_basis + c * Rk + r, token * grad);
			dx += tucker_h2f(k_basis[c * Rk + r]) * grad;
		}
		for (int r = 0; r < Rv; ++r)
		{
			const float grad = d_v_latent[(gw * T + t) * Rv + r];
			atomicAdd(dv_basis + c * Rv + r, token * grad);
			dx += tucker_h2f(v_basis[c * Rv + r]) * grad;
		}
		if (input_delta && b < B && y < H && x < W)
		{
			atomicAdd(input_delta + ((b * C + c) * H + y) * W + x, dx);
		}
		}
#endif
#endif

#if !DARKNET_TUCKER_USE_CUDNN_HALF
	__global__ void tucker_attention_backward_row_kernel(
		const int rows,
		const float *__restrict__ attn,
		const float *__restrict__ q,
		const float *__restrict__ k,
		const float *__restrict__ v,
		const float *__restrict__ d_context,
		float *__restrict__ d_q,
		float *__restrict__ d_k,
		float *__restrict__ d_v,
		int T, int heads, int D)
	{
		const int row_idx = blockIdx.x;
		if (row_idx >= rows) return;

		extern __shared__ float shared[];
		const int tid = threadIdx.x;
		const int tq = row_idx % T;
		const int hidx = (row_idx / T) % heads;
		const int gw = row_idx / (T * heads);
		const float *row = attn + ((gw * heads + hidx) * T + tq) * T;
		const int qbase = ((gw * T + tq) * heads + hidx) * D;
		const float scale = rsqrtf((float)D);

		float local_row_dot = 0.0f;
		for (int tk = tid; tk < T; tk += blockDim.x)
		{
			const int kbase = ((gw * T + tk) * heads + hidx) * D;
			float d_attn = 0.0f;
			for (int d = 0; d < D; ++d)
			{
				d_attn += d_context[qbase + d] * v[kbase + d];
			}
			local_row_dot += d_attn * row[tk];
		}
		shared[tid] = local_row_dot;
		__syncthreads();
		for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1)
		{
			if (tid < stride) shared[tid] += shared[tid + stride];
			__syncthreads();
		}
		const float row_dot = shared[0];

		for (int tk = tid; tk < T; tk += blockDim.x)
		{
			const float a = row[tk];
			const int kbase = ((gw * T + tk) * heads + hidx) * D;
			float d_attn = 0.0f;
			for (int d = 0; d < D; ++d)
			{
				d_attn += d_context[qbase + d] * v[kbase + d];
			}
			const float d_dot = a * (d_attn - row_dot) * scale;
			for (int d = 0; d < D; ++d)
			{
				atomicAdd(d_v + kbase + d, a * d_context[qbase + d]);
				atomicAdd(d_q + qbase + d, d_dot * k[kbase + d]);
				atomicAdd(d_k + kbase + d, d_dot * q[qbase + d]);
			}
		}
	}

	__global__ void tucker_context_to_rank_kernel(
		const int total,
		const float *context,
		const float *o_core,
		float *o_mix,
		int T, int heads, int D, int Ro)
	{
		const int idx = blockIdx.x * blockDim.x + threadIdx.x;
		if (idx >= total) return;

		const int r = idx % Ro;
		const int t = (idx / Ro) % T;
		const int gw = idx / (Ro * T);

		float sum = 0.0f;
		for (int hidx = 0; hidx < heads; ++hidx)
		{
			for (int d = 0; d < D; ++d)
			{
				const int hd = hidx * D + d;
				sum += context[((gw * T + t) * heads + hidx) * D + d] * o_core[hd * Ro + r];
			}
		}
		o_mix[idx] = sum;
	}

	__global__ void tucker_output_kernel(
		const int total,
		const float *input,
		const float *o_mix,
		const float *o_basis,
		const float *biases,
		float *output,
		int B, int H, int W, int C, int N,
		int window, int win_h, int win_w,
		int Ro)
	{
		const int idx = blockIdx.x * blockDim.x + threadIdx.x;
		if (idx >= total) return;

		const int x = idx % W;
		const int y = (idx / W) % H;
		const int n = (idx / (W * H)) % N;
		const int b = idx / (W * H * N);
		const int wy = y / window;
		const int wx = x / window;
		const int t = (y % window) * window + (x % window);
		const int gw = (b * win_h + wy) * win_w + wx;
		const int T = window * window;

		float out = biases ? biases[n] : 0.0f;
		for (int r = 0; r < Ro; ++r)
		{
			out += o_mix[(gw * T + t) * Ro + r] * o_basis[r * N + n];
		}
		if (n < C) out += input[((b * C + n) * H + y) * W + x];
		output[idx] = out;
	}

	__global__ void tucker_output_to_rank_backward_kernel(
		const int total,
		const float *output_delta,
		const float *o_mix,
		const float *o_basis,
		float *do_basis,
		float *bias_updates,
		float *input_delta,
		float *d_o_mix,
		int B, int H, int W, int C, int N,
		int window, int win_h, int win_w,
		int Ro)
	{
		const int idx = blockIdx.x * blockDim.x + threadIdx.x;
		if (idx >= total) return;

		const int x = idx % W;
		const int y = (idx / W) % H;
		const int n = (idx / (W * H)) % N;
		const int b = idx / (W * H * N);
		const float dy = output_delta[idx];
		if (dy == 0.0f) return;

		if (bias_updates) atomicAdd(bias_updates + n, dy);
		if (input_delta && n < C)
		{
			atomicAdd(input_delta + ((b * C + n) * H + y) * W + x, dy);
		}

		const int wy = y / window;
		const int wx = x / window;
		const int t = (y % window) * window + (x % window);
		const int T = window * window;
		const int gw = (b * win_h + wy) * win_w + wx;

		for (int r = 0; r < Ro; ++r)
		{
			atomicAdd(do_basis + r * N + n, o_mix[(gw * T + t) * Ro + r] * dy);
			atomicAdd(d_o_mix + (gw * T + t) * Ro + r, o_basis[r * N + n] * dy);
		}
	}

	__global__ void tucker_rank_to_context_backward_kernel(
		const int total,
		const float *context,
		const float *d_o_mix,
		const float *o_core,
		float *do_core,
		float *d_context,
		int T, int heads, int D, int Ro)
	{
		const int idx = blockIdx.x * blockDim.x + threadIdx.x;
		if (idx >= total) return;

		const int d = idx % D;
		const int hidx = (idx / D) % heads;
		const int t = (idx / (D * heads)) % T;
		const int gw = idx / (D * heads * T);
		const int hd = hidx * D + d;

		float dctx = 0.0f;
		const float ctx = context[idx];
		for (int r = 0; r < Ro; ++r)
		{
			const float grad = d_o_mix[(gw * T + t) * Ro + r];
			atomicAdd(do_core + hd * Ro + r, ctx * grad);
			dctx += o_core[hd * Ro + r] * grad;
		}
		d_context[idx] = dctx;
	}

	__global__ void tucker_expand_backward_kernel(
		const int total,
		const float *latent,
		const float *core,
		const float *d_head,
		float *d_latent,
		float *d_core,
		int T, int heads, int R, int D)
	{
		const int idx = blockIdx.x * blockDim.x + threadIdx.x;
		if (idx >= total) return;

		const int d = idx % D;
		const int hidx = (idx / D) % heads;
		const int t = (idx / (D * heads)) % T;
		const int gw = idx / (D * heads * T);
		const float grad = d_head[idx];
		if (grad == 0.0f) return;

		for (int r = 0; r < R; ++r)
		{
			atomicAdd(d_core + (hidx * R + r) * D + d, latent[(gw * T + t) * R + r] * grad);
			atomicAdd(d_latent + (gw * T + t) * R + r, core[(hidx * R + r) * D + d] * grad);
		}
	}

	__global__ void tucker_basis_backward_kernel(
		const int total,
		const float *tokens,
		const float *q_basis, const float *k_basis, const float *v_basis,
		const float *d_q_latent, const float *d_k_latent, const float *d_v_latent,
		float *dq_basis, float *dk_basis, float *dv_basis,
		float *input_delta,
		int B, int H, int W, int C,
		int window, int win_h, int win_w,
		int Rq, int Rk, int Rv)
	{
		const int idx = blockIdx.x * blockDim.x + threadIdx.x;
		if (idx >= total) return;

		const int c = idx % C;
		const int t = (idx / C) % (window * window);
		const int gw = idx / (C * window * window);
		const int b = gw / (win_h * win_w);
		const int local_w = gw % (win_h * win_w);
		const int wy = local_w / win_w;
		const int wx = local_w % win_w;
		const int yy = t / window;
		const int xx = t % window;
		const int y = wy * window + yy;
		const int x = wx * window + xx;
		const int T = window * window;
		const float token = tokens[idx];

		float dx = 0.0f;
		for (int r = 0; r < Rq; ++r)
		{
			const float grad = d_q_latent[(gw * T + t) * Rq + r];
			atomicAdd(dq_basis + c * Rq + r, token * grad);
			dx += q_basis[c * Rq + r] * grad;
		}
		for (int r = 0; r < Rk; ++r)
		{
			const float grad = d_k_latent[(gw * T + t) * Rk + r];
			atomicAdd(dk_basis + c * Rk + r, token * grad);
			dx += k_basis[c * Rk + r] * grad;
		}
		for (int r = 0; r < Rv; ++r)
		{
			const float grad = d_v_latent[(gw * T + t) * Rv + r];
			atomicAdd(dv_basis + c * Rv + r, token * grad);
			dx += v_basis[c * Rv + r] * grad;
		}
		if (input_delta && b < B && y < H && x < W)
		{
			atomicAdd(input_delta + ((b * C + c) * H + y) * W + x, dx);
		}
	}
#endif

	#if 0
	__global__ void tucker_attention_forward_kernel(
		const int total,
		const float *input,
		const float *weights,
		const float *biases,
		float *output,
		int B, int H, int W, int C, int N,
		int window, int heads, int D,
		int Rq, int Rk, int Rv, int Ro,
		size_t q_basis_off, size_t k_basis_off, size_t v_basis_off,
		size_t q_core_off, size_t k_core_off, size_t v_core_off,
		size_t o_core_off, size_t o_basis_off)
	{
		const int idx = blockIdx.x * blockDim.x + threadIdx.x;
		if (idx >= total) return;

		const int x = idx % W;
		const int y = (idx / W) % H;
		const int n = (idx / (W * H)) % N;
		const int b = idx / (W * H * N);

		const float *q_basis = weights + q_basis_off;
		const float *k_basis = weights + k_basis_off;
		const float *v_basis = weights + v_basis_off;
		const float *q_core = weights + q_core_off;
		const float *k_core = weights + k_core_off;
		const float *v_core = weights + v_core_off;
		const float *o_core = weights + o_core_off;
		const float *o_basis = weights + o_basis_off;

		const int wy = (y / window) * window;
		const int wx = (x / window) * window;
		const float inv_sqrt_d = rsqrtf((float)D);

		float out = biases ? biases[n] : 0.0f;

		for (int hidx = 0; hidx < heads; ++hidx)
		{
			for (int d = 0; d < D; ++d)
			{
				const float q = tucker_project_head_dim(input, q_basis, q_core, b, y, x, hidx, d, Rq, C, H, W, D);

				float max_score = -3.402823466e+38F;
				for (int yy = 0; yy < window; ++yy)
				{
					const int ky = wy + yy;
					if (ky >= H) continue;
					for (int xx = 0; xx < window; ++xx)
					{
						const int kx = wx + xx;
						if (kx >= W) continue;
						float dot = 0.0f;
						for (int dd = 0; dd < D; ++dd)
						{
							const float qd = (dd == d) ? q : tucker_project_head_dim(input, q_basis, q_core, b, y, x, hidx, dd, Rq, C, H, W, D);
							const float kd = tucker_project_head_dim(input, k_basis, k_core, b, ky, kx, hidx, dd, Rk, C, H, W, D);
							dot += qd * kd;
						}
						max_score = fmaxf(max_score, dot * inv_sqrt_d);
					}
				}

				float denom = 0.0f;
				float context = 0.0f;
				for (int yy = 0; yy < window; ++yy)
				{
					const int ky = wy + yy;
					if (ky >= H) continue;
					for (int xx = 0; xx < window; ++xx)
					{
						const int kx = wx + xx;
						if (kx >= W) continue;
						float dot = 0.0f;
						for (int dd = 0; dd < D; ++dd)
						{
							const float qd = (dd == d) ? q : tucker_project_head_dim(input, q_basis, q_core, b, y, x, hidx, dd, Rq, C, H, W, D);
							const float kd = tucker_project_head_dim(input, k_basis, k_core, b, ky, kx, hidx, dd, Rk, C, H, W, D);
							dot += qd * kd;
						}
						const float e = expf(dot * inv_sqrt_d - max_score);
						const float vv = tucker_project_head_dim(input, v_basis, v_core, b, ky, kx, hidx, d, Rv, C, H, W, D);
						denom += e;
						context += e * vv;
					}
				}
				context = denom > 0.0f ? context / denom : 0.0f;

				float ocoef = 0.0f;
				const int hd = hidx * D + d;
				for (int r = 0; r < Ro; ++r)
				{
					ocoef += o_core[hd * Ro + r] * o_basis[r * N + n];
				}
				out += context * ocoef;
			}
		}

		if (n < C)
		{
			out += tucker_input_at(input, b, n, y, x, B, C, H, W);
		}
		output[idx] = out;
	}

	__global__ void tucker_attention_backward_kernel(
		const int total,
		const float *input,
		const float *output_delta,
		const float *weights,
		float *weight_updates,
		float *bias_updates,
		float *input_delta,
		int B, int H, int W, int C, int N,
		int window, int heads, int D,
		int Rq, int Rk, int Rv, int Ro,
		size_t q_basis_off, size_t k_basis_off, size_t v_basis_off,
		size_t q_core_off, size_t k_core_off, size_t v_core_off,
		size_t o_core_off, size_t o_basis_off)
	{
		const int idx = blockIdx.x * blockDim.x + threadIdx.x;
		if (idx >= total) return;

		const int x = idx % W;
		const int y = (idx / W) % H;
		const int n = (idx / (W * H)) % N;
		const int b = idx / (W * H * N);
		const float dy = output_delta[idx];
		if (dy == 0.0f) return;

		const float *q_basis = weights + q_basis_off;
		const float *k_basis = weights + k_basis_off;
		const float *v_basis = weights + v_basis_off;
		const float *q_core = weights + q_core_off;
		const float *k_core = weights + k_core_off;
		const float *v_core = weights + v_core_off;
		const float *o_core = weights + o_core_off;
		const float *o_basis = weights + o_basis_off;
		float *dq_basis = weight_updates + q_basis_off;
		float *dk_basis = weight_updates + k_basis_off;
		float *dv_basis = weight_updates + v_basis_off;
		float *dq_core = weight_updates + q_core_off;
		float *dk_core = weight_updates + k_core_off;
		float *dv_core = weight_updates + v_core_off;
		float *do_core = weight_updates + o_core_off;
		float *do_basis = weight_updates + o_basis_off;

		if (bias_updates) atomicAdd(bias_updates + n, dy);
		if (input_delta && n < C)
		{
			atomicAdd(input_delta + ((b * C + n) * H + y) * W + x, dy);
		}

		const int wy = (y / window) * window;
		const int wx = (x / window) * window;
		const float inv_sqrt_d = rsqrtf((float)D);

		for (int r = 0; r < Ro; ++r)
		{
			float head_mix = 0.0f;
			for (int hidx = 0; hidx < heads; ++hidx)
			{
				float max_score = -3.402823466e+38F;
				for (int yy = 0; yy < window; ++yy)
				{
					const int ky = wy + yy;
					if (ky >= H) continue;
					for (int xx = 0; xx < window; ++xx)
					{
						const int kx = wx + xx;
						if (kx >= W) continue;
						max_score = fmaxf(max_score, tucker_attention_score(input, q_basis, k_basis, q_core, k_core, b, y, x, ky, kx, hidx, Rq, Rk, C, H, W, D));
					}
				}
				float denom = 0.0f;
				for (int yy = 0; yy < window; ++yy)
				{
					const int ky = wy + yy;
					if (ky >= H) continue;
					for (int xx = 0; xx < window; ++xx)
					{
						const int kx = wx + xx;
						if (kx >= W) continue;
						denom += expf(tucker_attention_score(input, q_basis, k_basis, q_core, k_core, b, y, x, ky, kx, hidx, Rq, Rk, C, H, W, D) - max_score);
					}
				}
				for (int d = 0; d < D; ++d)
				{
					float context = 0.0f;
					for (int yy = 0; yy < window; ++yy)
					{
						const int ky = wy + yy;
						if (ky >= H) continue;
						for (int xx = 0; xx < window; ++xx)
						{
							const int kx = wx + xx;
							if (kx >= W) continue;
							const float a = expf(tucker_attention_score(input, q_basis, k_basis, q_core, k_core, b, y, x, ky, kx, hidx, Rq, Rk, C, H, W, D) - max_score) / denom;
							const float v = tucker_project_head_dim(input, v_basis, v_core, b, ky, kx, hidx, d, Rv, C, H, W, D);
							context += a * v;
						}
					}
					head_mix += context * o_core[(hidx * D + d) * Ro + r];
				}
			}
			atomicAdd(do_basis + r * N + n, head_mix * dy);
		}

		for (int hidx = 0; hidx < heads; ++hidx)
		{
			float max_score = -3.402823466e+38F;
			for (int yy = 0; yy < window; ++yy)
			{
				const int ky = wy + yy;
				if (ky >= H) continue;
				for (int xx = 0; xx < window; ++xx)
				{
					const int kx = wx + xx;
					if (kx >= W) continue;
					max_score = fmaxf(max_score, tucker_attention_score(input, q_basis, k_basis, q_core, k_core, b, y, x, ky, kx, hidx, Rq, Rk, C, H, W, D));
				}
			}
			float denom = 0.0f;
			for (int yy = 0; yy < window; ++yy)
			{
				const int ky = wy + yy;
				if (ky >= H) continue;
				for (int xx = 0; xx < window; ++xx)
				{
					const int kx = wx + xx;
					if (kx >= W) continue;
					denom += expf(tucker_attention_score(input, q_basis, k_basis, q_core, k_core, b, y, x, ky, kx, hidx, Rq, Rk, C, H, W, D) - max_score);
				}
			}

			float row_dot = 0.0f;
			for (int yy = 0; yy < window; ++yy)
			{
				const int ky = wy + yy;
				if (ky >= H) continue;
				for (int xx = 0; xx < window; ++xx)
				{
					const int kx = wx + xx;
					if (kx >= W) continue;
					const float attn = expf(tucker_attention_score(input, q_basis, k_basis, q_core, k_core, b, y, x, ky, kx, hidx, Rq, Rk, C, H, W, D) - max_score) / denom;
					float d_attn = 0.0f;
					for (int d = 0; d < D; ++d)
					{
						float d_context = 0.0f;
						for (int r = 0; r < Ro; ++r)
						{
							d_context += o_core[(hidx * D + d) * Ro + r] * o_basis[r * N + n] * dy;
						}
						const float v = tucker_project_head_dim(input, v_basis, v_core, b, ky, kx, hidx, d, Rv, C, H, W, D);
						d_attn += d_context * v;
					}
					row_dot += d_attn * attn;
				}
			}

			for (int d = 0; d < D; ++d)
			{
				float d_context = 0.0f;
				for (int r = 0; r < Ro; ++r)
				{
					d_context += o_core[(hidx * D + d) * Ro + r] * o_basis[r * N + n] * dy;
				}

				float context = 0.0f;
				float d_q = 0.0f;
				for (int yy = 0; yy < window; ++yy)
				{
					const int ky = wy + yy;
					if (ky >= H) continue;
					for (int xx = 0; xx < window; ++xx)
					{
						const int kx = wx + xx;
						if (kx >= W) continue;
						const float score = tucker_attention_score(input, q_basis, k_basis, q_core, k_core, b, y, x, ky, kx, hidx, Rq, Rk, C, H, W, D);
						const float attn = expf(score - max_score) / denom;
						const float v = tucker_project_head_dim(input, v_basis, v_core, b, ky, kx, hidx, d, Rv, C, H, W, D);
						context += attn * v;
					}
				}

				for (int r = 0; r < Ro; ++r)
				{
					atomicAdd(do_core + (hidx * D + d) * Ro + r, context * o_basis[r * N + n] * dy);
				}

				for (int yy = 0; yy < window; ++yy)
				{
					const int ky = wy + yy;
					if (ky >= H) continue;
					for (int xx = 0; xx < window; ++xx)
					{
						const int kx = wx + xx;
						if (kx >= W) continue;
						const float score = tucker_attention_score(input, q_basis, k_basis, q_core, k_core, b, y, x, ky, kx, hidx, Rq, Rk, C, H, W, D);
						const float attn = expf(score - max_score) / denom;

						float d_attn = 0.0f;
						for (int dd = 0; dd < D; ++dd)
						{
							float d_context_dd = 0.0f;
							for (int r = 0; r < Ro; ++r)
							{
								d_context_dd += o_core[(hidx * D + dd) * Ro + r] * o_basis[r * N + n] * dy;
							}
							const float vdd = tucker_project_head_dim(input, v_basis, v_core, b, ky, kx, hidx, dd, Rv, C, H, W, D);
							d_attn += d_context_dd * vdd;
						}
						const float d_score = attn * (d_attn - row_dot) * inv_sqrt_d;
						const float qd = tucker_project_head_dim(input, q_basis, q_core, b, y, x, hidx, d, Rq, C, H, W, D);
						const float kd = tucker_project_head_dim(input, k_basis, k_core, b, ky, kx, hidx, d, Rk, C, H, W, D);
						d_q += d_score * kd;
						const float d_k = d_score * qd;
						const float d_v = attn * d_context;
						tucker_accumulate_projection_backward(input, k_basis, k_core, dk_basis, dk_core, input_delta, d_k, b, ky, kx, hidx, d, Rk, C, H, W, D);
						tucker_accumulate_projection_backward(input, v_basis, v_core, dv_basis, dv_core, input_delta, d_v, b, ky, kx, hidx, d, Rv, C, H, W, D);
					}
				}
				tucker_accumulate_projection_backward(input, q_basis, q_core, dq_basis, dq_core, input_delta, d_q, b, y, x, hidx, d, Rq, C, H, W, D);
			}
		}
	}
	#endif
}

void forward_tucker_attention_layer_gpu(Darknet::Layer & l, Darknet::NetworkState state)
{
	TAT(TATPARMS);

	const TuckerOffsetsGpu off = gpu_offsets(l);
	const int total = l.batch * l.outputs;
	const int M = l.tucker_window_size;
	const int T = M * M;
	const int win_h = (l.h + M - 1) / M;
	const int win_w = (l.w + M - 1) / M;
	const int windows = l.batch * win_h * win_w;
	const int head_count = windows * T * l.tucker_heads * l.tucker_head_dim;
	const int score_rows = windows * l.tucker_heads * T;
	const int score_count = score_rows * T;
	const int token_count = windows * T * l.c;
	const int o_mix_count = windows * T * l.tucker_rank_o;
#if !DARKNET_TUCKER_USE_CUDNN_HALF || !DARKNET_TUCKER_USE_CUBLAS_HALF
	float *o_mix = l.tucker_windowed_input_gpu + token_count;
#endif
#if DARKNET_TUCKER_USE_CUDNN_HALF && DARKNET_TUCKER_USE_CUBLAS_HALF
	(void)head_count;
#endif

#if DARKNET_TUCKER_USE_CUDNN_HALF && DARKNET_TUCKER_USE_CUBLAS_HALF
	__half *tokens16 = reinterpret_cast<__half *>(l.tucker_windowed_input_gpu);
	tucker_pack_windows_half_kernel<<<cuda_gridsize(token_count), BLOCK, 0, get_cuda_stream()>>>(
		token_count, state.input, tokens16,
		l.batch, l.h, l.w, l.c, M, win_h, win_w);
#else
	tucker_pack_windows_kernel<<<cuda_gridsize(token_count), BLOCK, 0, get_cuda_stream()>>>(
		token_count, state.input, l.tucker_windowed_input_gpu,
		l.batch, l.h, l.w, l.c, M, win_h, win_w);
#endif
	CHECK_CUDA(cudaPeekAtLastError());

#if DARKNET_TUCKER_USE_CUDNN_HALF
	cuda_convert_f32_to_cudnn_16bit(l.weights_gpu, l.nweights, l.weights_gpu16, DARKNET_CUDNN_16BIT_HALF);
	const __half *w16 = reinterpret_cast<const __half *>(l.weights_gpu16);
	const __half *q_basis16 = w16 + off.q_basis;
	const __half *k_basis16 = w16 + off.k_basis;
	const __half *v_basis16 = w16 + off.v_basis;
	const __half *q_core16 = w16 + off.q_core;
	const __half *k_core16 = w16 + off.k_core;
	const __half *v_core16 = w16 + off.v_core;
	const __half *o_core16 = w16 + off.o_core;
	const __half *o_basis16 = w16 + off.o_basis;
	__half *q16 = reinterpret_cast<__half *>(l.tucker_q_gpu);
	__half *k16 = reinterpret_cast<__half *>(l.tucker_k_gpu);
	__half *v16 = reinterpret_cast<__half *>(l.tucker_v_gpu);
	__half *scores16 = reinterpret_cast<__half *>(l.tucker_scores_gpu);
	__half *attn16 = scores16 + score_count;
	__half *context16 = reinterpret_cast<__half *>(l.tucker_context_gpu);

#if DARKNET_TUCKER_USE_CUBLAS_HALF
	__half *o_mix16 = tokens16 + token_count;
	__half *rank_output16 = o_mix16 + o_mix_count;
	__half *q_latent16 = reinterpret_cast<__half *>(l.tucker_q_latent_gpu);
	__half *k_latent16 = reinterpret_cast<__half *>(l.tucker_k_latent_gpu);
	__half *v_latent16 = reinterpret_cast<__half *>(l.tucker_v_latent_gpu);
	const int WT = windows * T;
	const int D = l.tucker_head_dim;
	const float score_scale = 1.0f / sqrtf((float)D);

	tucker_gemm_half(0, 0, WT, l.tucker_rank_q, l.c, 1.0f,
		tokens16, l.c, q_basis16, l.tucker_rank_q, 0.0f, q_latent16, l.tucker_rank_q, CUDA_R_16F);
	tucker_gemm_half(0, 0, WT, l.tucker_rank_k, l.c, 1.0f,
		tokens16, l.c, k_basis16, l.tucker_rank_k, 0.0f, k_latent16, l.tucker_rank_k, CUDA_R_16F);
	tucker_gemm_half(0, 0, WT, l.tucker_rank_v, l.c, 1.0f,
		tokens16, l.c, v_basis16, l.tucker_rank_v, 0.0f, v_latent16, l.tucker_rank_v, CUDA_R_16F);

	for (int hidx = 0; hidx < l.tucker_heads; ++hidx)
	{
		tucker_gemm_half(0, 0, WT, D, l.tucker_rank_q, 1.0f,
			q_latent16, l.tucker_rank_q,
			q_core16 + hidx * l.tucker_rank_q * D, D,
			0.0f,
			q16 + hidx * WT * D, D,
			CUDA_R_16F);
		tucker_gemm_half(0, 0, WT, D, l.tucker_rank_k, 1.0f,
			k_latent16, l.tucker_rank_k,
			k_core16 + hidx * l.tucker_rank_k * D, D,
			0.0f,
			k16 + hidx * WT * D, D,
			CUDA_R_16F);
		tucker_gemm_half(0, 0, WT, D, l.tucker_rank_v, 1.0f,
			v_latent16, l.tucker_rank_v,
			v_core16 + hidx * l.tucker_rank_v * D, D,
			0.0f,
			v16 + hidx * WT * D, D,
			CUDA_R_16F);
	}

#ifdef DARKNET_HAS_FP8
	const bool use_fp8_attention = l.fp8_tucker_attention != 0 && state.net.fp8_inference != 0
		&& l.fp8_tucker_scores_gemm_plan != nullptr && l.fp8_tucker_context_gemm_plan != nullptr;
#else
	const bool use_fp8_attention = false;
#endif
#ifdef DARKNET_HAS_FP4
	const bool use_fp4_attention = !use_fp8_attention && l.fp4_tucker_attention != 0 && state.net.fp4_inference != 0
		&& l.fp4_tucker_scores_gemm_plan != nullptr && l.fp4_tucker_context_gemm_plan != nullptr;
#else
	const bool use_fp4_attention = false;
#endif

	if (use_fp8_attention)
	{
#ifdef DARKNET_HAS_FP8
		const int batch = l.tucker_heads * windows;
		const int key_pad = Darknet::fp8_round_up_to_16(T);
		const size_t lt_workspace_bytes = Darknet::fp8_gemm_workspace_bytes();

		// Scores GEMM: A=K (key_pad, D), B=Q (T, D) -- see darknet_layers.hpp field comments.
		Darknet::fp8_quantize_half_pad_rows_cols_gpu(k16, T, D, key_pad, D, l.fp8_tucker_k_scale_gpu,
			l.fp8_tucker_k_gpu, batch, (size_t)T * D, (size_t)key_pad * D);
		Darknet::fp8_quantize_half_rowmajor_pad_cols_gpu(q16, T, D, D, l.fp8_tucker_q_scale_gpu,
			l.fp8_tucker_q_gpu, 0, batch, (size_t)T * D, (size_t)T * D);

		if (!Darknet::fp8_gemm(
			static_cast<Darknet::Fp8GemmPlan *>(l.fp8_tucker_scores_gemm_plan),
			l.fp8_tucker_k_gpu, l.fp8_tucker_q_gpu, l.fp8_tucker_scores_out_gpu,
			l.fp8_tucker_lt_workspace_gpu, lt_workspace_bytes))
		{
			darknet_fatal_error(DARKNET_LOC, "tucker_attention: FP8 scores GEMM failed");
		}

		Darknet::fp8_dequant_compact_scores_half_gpu(
			l.fp8_tucker_scores_out_gpu, T, T, key_pad, score_scale, scores16,
			batch, (size_t)key_pad * T, (size_t)T * T);

		{
			const float alpha = 1.0f;
			const float beta = 0.0f;
			CHECK_CUDNN(cudnnSoftmaxForward(
				cudnn_handle(),
				CUDNN_SOFTMAX_ACCURATE,
				CUDNN_SOFTMAX_MODE_CHANNEL,
				&alpha,
				l.srcTensorDesc16,
				scores16,
				&beta,
				l.dstTensorDesc16,
				attn16));
		}

		// Context GEMM: A=V^T (D, key_pad), B=attn (T, key_pad).
		Darknet::fp8_quantize_half_transpose_rowmajor_pad_cols_gpu(v16, T, D, key_pad, l.fp8_tucker_v_scale_gpu,
			l.fp8_tucker_v_t_gpu, batch, (size_t)T * D, (size_t)D * key_pad);
		Darknet::fp8_quantize_half_rowmajor_pad_cols_gpu(attn16, T, T, key_pad, l.fp8_tucker_attn_scale_gpu,
			l.fp8_tucker_attn_gpu, 0, batch, (size_t)T * T, (size_t)T * key_pad);

		if (!Darknet::fp8_gemm(
			static_cast<Darknet::Fp8GemmPlan *>(l.fp8_tucker_context_gemm_plan),
			l.fp8_tucker_v_t_gpu, l.fp8_tucker_attn_gpu, l.fp8_tucker_context_out_gpu,
			l.fp8_tucker_lt_workspace_gpu, lt_workspace_bytes))
		{
			darknet_fatal_error(DARKNET_LOC, "tucker_attention: FP8 context GEMM failed");
		}

		// Raw column-major (D, T) GEMM2 output is byte-identical to row-major (T, D) --
		// just cast fp32 -> fp16 straight into context16, no transpose needed.
		cuda_convert_f32_to_cudnn_16bit(l.fp8_tucker_context_out_gpu, (size_t)batch * D * T,
			reinterpret_cast<float *>(context16), DARKNET_CUDNN_16BIT_HALF);
#endif
	}
	else if (use_fp4_attention)
	{
#ifdef DARKNET_HAS_FP4
		const int batch = l.tucker_heads * windows;
		const int key_pad = ((T + 15) / 16) * 16;
		auto * const scores_plan = static_cast<Darknet::Fp4GemmPlan *>(l.fp4_tucker_scores_gemm_plan);
		auto * const context_plan = static_cast<Darknet::Fp4GemmPlan *>(l.fp4_tucker_context_gemm_plan);
		const size_t scores_ws = Darknet::fp4_gemm_workspace_bytes(scores_plan);
		const size_t context_ws = Darknet::fp4_gemm_workspace_bytes(context_plan);

		// Scores GEMM: A=Q (T, D), B=K (T, D) -- K's natural layout is already the
		// required B-transposed (T, D) shape, no transform needed.
		Darknet::fp4_half_to_float_gpu(q16, (size_t)T * D, l.fp4_tucker_a_gpu, batch, (size_t)T * D, (size_t)T * D);
		Darknet::fp4_half_to_float_gpu(k16, (size_t)T * D, l.fp4_tucker_b_gpu, batch, (size_t)T * D, (size_t)T * D);
		for (int b = 0; b < batch; ++b)
		{
			if (!Darknet::fp4_gemm_execute(scores_plan,
				l.fp4_tucker_a_gpu + (size_t)b * T * D, l.fp4_tucker_b_gpu + (size_t)b * T * D,
				l.fp4_tucker_out_gpu + (size_t)b * T * T, l.fp4_tucker_scores_lt_workspace, scores_ws))
			{
				darknet_fatal_error(DARKNET_LOC, "tucker_attention: FP4 scores GEMM failed");
			}
		}
		Darknet::fp4_scale_cast_float_to_half_gpu(l.fp4_tucker_out_gpu, (size_t)T * T, score_scale, scores16,
			batch, (size_t)T * T, (size_t)T * T);

		{
			const float alpha = 1.0f;
			const float beta = 0.0f;
			CHECK_CUDNN(cudnnSoftmaxForward(
				cudnn_handle(),
				CUDNN_SOFTMAX_ACCURATE,
				CUDNN_SOFTMAX_MODE_CHANNEL,
				&alpha,
				l.srcTensorDesc16,
				scores16,
				&beta,
				l.dstTensorDesc16,
				attn16));
		}

		// Context GEMM: A=attn (T, key_pad, col-padded), B=V^T (D, key_pad, transposed+padded).
		Darknet::fp4_pad_cols_half_to_float_gpu(attn16, T, T, key_pad, l.fp4_tucker_a_gpu,
			batch, (size_t)T * T, (size_t)T * key_pad);
		Darknet::fp4_transpose_pad_cols_half_to_float_gpu(v16, T, D, key_pad, l.fp4_tucker_b_gpu,
			batch, (size_t)T * D, (size_t)D * key_pad);
		for (int b = 0; b < batch; ++b)
		{
			if (!Darknet::fp4_gemm_execute(context_plan,
				l.fp4_tucker_a_gpu + (size_t)b * T * key_pad, l.fp4_tucker_b_gpu + (size_t)b * D * key_pad,
				l.fp4_tucker_out_gpu + (size_t)b * T * D, l.fp4_tucker_context_lt_workspace, context_ws))
			{
				darknet_fatal_error(DARKNET_LOC, "tucker_attention: FP4 context GEMM failed");
			}
		}
		Darknet::fp4_scale_cast_float_to_half_gpu(l.fp4_tucker_out_gpu, (size_t)T * D, 1.0f, context16,
			batch, (size_t)T * D, (size_t)T * D);
#endif
	}
	else
	{
		tucker_gemm_half_strided_batched(0, 1, T, T, D, score_scale,
			q16, D, (long long)T * D,
			k16, D, (long long)T * D,
			0.0f,
			scores16, T, (long long)T * T,
			l.tucker_heads * windows,
			CUDA_R_16F);

		{
			const float alpha = 1.0f;
			const float beta = 0.0f;
			CHECK_CUDNN(cudnnSoftmaxForward(
				cudnn_handle(),
				CUDNN_SOFTMAX_ACCURATE,
				CUDNN_SOFTMAX_MODE_CHANNEL,
				&alpha,
				l.srcTensorDesc16,
				scores16,
				&beta,
				l.dstTensorDesc16,
				attn16));
		}

		tucker_gemm_half_strided_batched(0, 0, T, D, T, 1.0f,
			attn16, T, (long long)T * T,
			v16, D, (long long)T * D,
			0.0f,
			context16, D, (long long)T * D,
			l.tucker_heads * windows,
			CUDA_R_16F);
	}

	for (int hidx = 0; hidx < l.tucker_heads; ++hidx)
	{
		tucker_gemm_half(0, 0, WT, l.tucker_rank_o, D, 1.0f,
			context16 + hidx * WT * D, D,
			o_core16 + hidx * D * l.tucker_rank_o, l.tucker_rank_o,
			hidx == 0 ? 0.0f : 1.0f,
			o_mix16, l.tucker_rank_o,
			CUDA_R_16F);
	}

	tucker_gemm_half(0, 0, WT, l.n, l.tucker_rank_o, 1.0f,
		o_mix16, l.tucker_rank_o,
		o_basis16, l.n,
		0.0f,
		rank_output16, l.n,
		CUDA_R_16F);

	tucker_output_from_rank_half_kernel<<<cuda_gridsize(total), BLOCK, 0, get_cuda_stream()>>>(
		total, state.input, rank_output16, l.biases_gpu, l.output_gpu,
		l.batch, l.h, l.w, l.c, l.n, M, win_h, win_w);
	CHECK_CUDA(cudaPeekAtLastError());
#else
	const int qkv_latent_count = windows * T * (l.tucker_rank_q + l.tucker_rank_k + l.tucker_rank_v);
	tucker_project_qkv_latent_half_weight_kernel<<<cuda_gridsize(qkv_latent_count), BLOCK, 0, get_cuda_stream()>>>(
		qkv_latent_count, l.tucker_windowed_input_gpu, q_basis16, k_basis16, v_basis16,
		l.tucker_q_latent_gpu, l.tucker_k_latent_gpu, l.tucker_v_latent_gpu,
		windows, T, l.c, l.tucker_rank_q, l.tucker_rank_k, l.tucker_rank_v);
	CHECK_CUDA(cudaPeekAtLastError());

	tucker_expand_qkv_heads_half_kernel<<<cuda_gridsize(3 * head_count), BLOCK, 0, get_cuda_stream()>>>(
		3 * head_count, l.tucker_q_latent_gpu, l.tucker_k_latent_gpu, l.tucker_v_latent_gpu,
		q_core16, k_core16, v_core16, q16, k16, v16,
		head_count, T, l.tucker_heads, l.tucker_head_dim, l.tucker_rank_q, l.tucker_rank_k, l.tucker_rank_v);
	CHECK_CUDA(cudaPeekAtLastError());

	tucker_scores_half_kernel<<<cuda_gridsize(score_count), BLOCK, 0, get_cuda_stream()>>>(
		score_count, q16, k16, scores16, T, l.tucker_heads, l.tucker_head_dim);
	CHECK_CUDA(cudaPeekAtLastError());

	{
		const float alpha = 1.0f;
		const float beta = 0.0f;
		CHECK_CUDNN(cudnnSoftmaxForward(
			cudnn_handle(),
			CUDNN_SOFTMAX_ACCURATE,
			CUDNN_SOFTMAX_MODE_CHANNEL,
			&alpha,
			l.srcTensorDesc16,
			scores16,
			&beta,
			l.dstTensorDesc16,
			attn16));
	}

	tucker_context_from_attn_half_kernel<<<score_rows, tucker_pow2_threads(l.tucker_head_dim), 0, get_cuda_stream()>>>(
		score_rows, attn16, v16, context16, T, l.tucker_heads, l.tucker_head_dim);
	CHECK_CUDA(cudaPeekAtLastError());

	tucker_context_to_rank_half_kernel<<<cuda_gridsize(o_mix_count), BLOCK, 0, get_cuda_stream()>>>(
		o_mix_count, context16, o_core16, o_mix, T, l.tucker_heads, l.tucker_head_dim, l.tucker_rank_o);
	CHECK_CUDA(cudaPeekAtLastError());

	tucker_output_half_basis_kernel<<<cuda_gridsize(total), BLOCK, 0, get_cuda_stream()>>>(
		total, state.input, o_mix, o_basis16, l.biases_gpu, l.output_gpu,
		l.batch, l.h, l.w, l.c, l.n, M, win_h, win_w, l.tucker_rank_o);
	CHECK_CUDA(cudaPeekAtLastError());
#endif
#else
	const float *q_basis = l.weights_gpu + off.q_basis;
	const float *k_basis = l.weights_gpu + off.k_basis;
	const float *v_basis = l.weights_gpu + off.v_basis;
	const float *q_core = l.weights_gpu + off.q_core;
	const float *k_core = l.weights_gpu + off.k_core;
	const float *v_core = l.weights_gpu + off.v_core;
	const float *o_core = l.weights_gpu + off.o_core;
	const float *o_basis = l.weights_gpu + off.o_basis;

	const int qkv_latent_count = windows * T * (l.tucker_rank_q + l.tucker_rank_k + l.tucker_rank_v);
	tucker_project_qkv_latent_kernel<<<cuda_gridsize(qkv_latent_count), BLOCK, 0, get_cuda_stream()>>>(
		qkv_latent_count, l.tucker_windowed_input_gpu, q_basis, k_basis, v_basis,
		l.tucker_q_latent_gpu, l.tucker_k_latent_gpu, l.tucker_v_latent_gpu,
		windows, T, l.c, l.tucker_rank_q, l.tucker_rank_k, l.tucker_rank_v);
	CHECK_CUDA(cudaPeekAtLastError());

	tucker_expand_qkv_heads_kernel<<<cuda_gridsize(3 * head_count), BLOCK, 0, get_cuda_stream()>>>(
		3 * head_count, l.tucker_q_latent_gpu, l.tucker_k_latent_gpu, l.tucker_v_latent_gpu,
		q_core, k_core, v_core, l.tucker_q_gpu, l.tucker_k_gpu, l.tucker_v_gpu,
		head_count, T, l.tucker_heads, l.tucker_head_dim, l.tucker_rank_q, l.tucker_rank_k, l.tucker_rank_v);
	CHECK_CUDA(cudaPeekAtLastError());

#ifdef CUDNN
	float *attn = l.tucker_scores_gpu + score_count;
	tucker_scores_kernel<<<cuda_gridsize(score_count), BLOCK, 0, get_cuda_stream()>>>(
		score_count, l.tucker_q_gpu, l.tucker_k_gpu, l.tucker_scores_gpu, T, l.tucker_heads, l.tucker_head_dim);
	CHECK_CUDA(cudaPeekAtLastError());
	{
		const float alpha = 1.0f;
		const float beta = 0.0f;
		CHECK_CUDNN(cudnnSoftmaxForward(
			cudnn_handle(), CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_CHANNEL,
			&alpha, l.srcTensorDesc, l.tucker_scores_gpu, &beta, l.dstTensorDesc, attn));
		tucker_context_from_attn_kernel<<<score_rows, tucker_pow2_threads(l.tucker_head_dim), 0, get_cuda_stream()>>>(
			score_rows, attn, l.tucker_v_gpu, l.tucker_context_gpu, T, l.tucker_heads, l.tucker_head_dim);
	}
#else
	{
		const int score_threads = tucker_pow2_threads(T);
		tucker_scores_softmax_kernel<<<score_rows, score_threads, score_threads * sizeof(float), get_cuda_stream()>>>(
			score_rows, l.tucker_q_gpu, l.tucker_k_gpu, l.tucker_scores_gpu, T, l.tucker_heads, l.tucker_head_dim);
		CHECK_CUDA(cudaPeekAtLastError());
	}

	tucker_context_from_attn_kernel<<<score_rows, tucker_pow2_threads(l.tucker_head_dim), 0, get_cuda_stream()>>>(
		score_rows, l.tucker_scores_gpu, l.tucker_v_gpu, l.tucker_context_gpu, T, l.tucker_heads, l.tucker_head_dim);
#endif
	CHECK_CUDA(cudaPeekAtLastError());

	tucker_context_to_rank_kernel<<<cuda_gridsize(o_mix_count), BLOCK, 0, get_cuda_stream()>>>(
		o_mix_count, l.tucker_context_gpu, o_core, o_mix, T, l.tucker_heads, l.tucker_head_dim, l.tucker_rank_o);
	CHECK_CUDA(cudaPeekAtLastError());

	tucker_output_kernel<<<cuda_gridsize(total), BLOCK, 0, get_cuda_stream()>>>(
		total, state.input, o_mix, o_basis, l.biases_gpu, l.output_gpu,
		l.batch, l.h, l.w, l.c, l.n, M, win_h, win_w, l.tucker_rank_o);
	CHECK_CUDA(cudaPeekAtLastError());
#endif
	activate_array_ongpu(l.output_gpu, total, l.activation);
}

void backward_tucker_attention_layer_gpu(Darknet::Layer & l, Darknet::NetworkState state)
{
	TAT(TATPARMS);

	const TuckerOffsetsGpu off = gpu_offsets(l);
	const int total = l.batch * l.outputs;
	const int M = l.tucker_window_size;
	const int T = M * M;
	const int win_h = (l.h + M - 1) / M;
	const int win_w = (l.w + M - 1) / M;
	const int windows = l.batch * win_h * win_w;
	const int head_count = windows * T * l.tucker_heads * l.tucker_head_dim;
	const int token_count = windows * T * l.c;
	const int score_rows = windows * l.tucker_heads * T;
	const int score_count = score_rows * T;
	const int o_mix_count = windows * T * l.tucker_rank_o;
#if !DARKNET_TUCKER_USE_CUDNN_HALF || !DARKNET_TUCKER_USE_CUBLAS_HALF
	float *o_mix = l.tucker_windowed_input_gpu + token_count;
	float *d_o_mix = o_mix + o_mix_count;
#endif

	float *dq_basis = l.weight_updates_gpu + off.q_basis;
	float *dk_basis = l.weight_updates_gpu + off.k_basis;
	float *dv_basis = l.weight_updates_gpu + off.v_basis;
	float *dq_core = l.weight_updates_gpu + off.q_core;
	float *dk_core = l.weight_updates_gpu + off.k_core;
	float *dv_core = l.weight_updates_gpu + off.v_core;
	float *do_core = l.weight_updates_gpu + off.o_core;
	float *do_basis = l.weight_updates_gpu + off.o_basis;

	gradient_array_ongpu(l.output_gpu, total, l.activation, l.delta_gpu);

#if DARKNET_TUCKER_USE_CUDNN_HALF
	cuda_convert_f32_to_cudnn_16bit(l.weights_gpu, l.nweights, l.weights_gpu16, DARKNET_CUDNN_16BIT_HALF);
	const __half *w16 = reinterpret_cast<const __half *>(l.weights_gpu16);
	const __half *q_basis16 = w16 + off.q_basis;
	const __half *k_basis16 = w16 + off.k_basis;
	const __half *v_basis16 = w16 + off.v_basis;
	const __half *q_core16 = w16 + off.q_core;
	const __half *k_core16 = w16 + off.k_core;
	const __half *v_core16 = w16 + off.v_core;
	const __half *o_core16 = w16 + off.o_core;
	const __half *o_basis16 = w16 + off.o_basis;

	__half *q16 = reinterpret_cast<__half *>(l.tucker_q_gpu);
	__half *k16 = reinterpret_cast<__half *>(l.tucker_k_gpu);
	__half *v16 = reinterpret_cast<__half *>(l.tucker_v_gpu);
	__half *d_q16 = q16 + head_count;
	__half *d_k16 = k16 + head_count;
	__half *d_v16 = v16 + head_count;
	__half *scores16 = reinterpret_cast<__half *>(l.tucker_scores_gpu);
	__half *attn16 = scores16 + score_count;
	__half *d_attn16 = attn16 + score_count;
	__half *d_scores16 = scores16;
	__half *context16 = reinterpret_cast<__half *>(l.tucker_context_gpu);
	__half *d_context16 = context16 + head_count;

#if DARKNET_TUCKER_USE_CUBLAS_HALF
	__half *tokens16 = reinterpret_cast<__half *>(l.tucker_windowed_input_gpu);
	__half *o_mix16 = tokens16 + token_count;
	__half *d_out16 = o_mix16 + o_mix_count;
	__half *d_o_mix16 = d_out16 + token_count;
	__half *d_tokens16 = d_out16;
	__half *q_latent16 = reinterpret_cast<__half *>(l.tucker_q_latent_gpu);
	__half *k_latent16 = reinterpret_cast<__half *>(l.tucker_k_latent_gpu);
	__half *v_latent16 = reinterpret_cast<__half *>(l.tucker_v_latent_gpu);
	__half *d_q_latent16 = q_latent16 + windows * T * l.tucker_rank_q;
	__half *d_k_latent16 = k_latent16 + windows * T * l.tucker_rank_k;
	__half *d_v_latent16 = v_latent16 + windows * T * l.tucker_rank_v;
	const int WT = windows * T;
	const int D = l.tucker_head_dim;
	const float score_scale = 1.0f / sqrtf((float)D);

	CHECK_CUDA(cudaMemsetAsync(d_out16, 0, static_cast<size_t>(token_count) * sizeof(__half), get_cuda_stream()));
	CHECK_CUDA(cudaMemsetAsync(d_o_mix16, 0, static_cast<size_t>(o_mix_count) * sizeof(__half), get_cuda_stream()));

	tucker_pack_output_delta_half_kernel<<<cuda_gridsize(total), BLOCK, 0, get_cuda_stream()>>>(
		total, l.delta_gpu, d_out16, l.bias_updates_gpu, state.delta,
		l.batch, l.h, l.w, l.c, l.n, M, win_h, win_w);
	CHECK_CUDA(cudaPeekAtLastError());

	tucker_gemm_half(1, 0, l.tucker_rank_o, l.n, WT, 1.0f,
		o_mix16, l.tucker_rank_o,
		d_out16, l.n,
		1.0f,
		do_basis, l.n,
		CUDA_R_32F);
	tucker_gemm_half(0, 1, WT, l.tucker_rank_o, l.n, 1.0f,
		d_out16, l.n,
		o_basis16, l.n,
		0.0f,
		d_o_mix16, l.tucker_rank_o,
		CUDA_R_16F);

	for (int hidx = 0; hidx < l.tucker_heads; ++hidx)
	{
		tucker_gemm_half(1, 0, D, l.tucker_rank_o, WT, 1.0f,
			context16 + hidx * WT * D, D,
			d_o_mix16, l.tucker_rank_o,
			1.0f,
			do_core + hidx * D * l.tucker_rank_o, l.tucker_rank_o,
			CUDA_R_32F);
		tucker_gemm_half(0, 1, WT, D, l.tucker_rank_o, 1.0f,
			d_o_mix16, l.tucker_rank_o,
			o_core16 + hidx * D * l.tucker_rank_o, l.tucker_rank_o,
			0.0f,
			d_context16 + hidx * WT * D, D,
			CUDA_R_16F);
	}

#ifdef DARKNET_HAS_FP4
	const bool use_fp4_attention_bwd = l.fp4_tucker_attention != 0 && state.net.fp4_training != 0
		&& l.fp4_tucker_scores_gemm_plan != nullptr && l.fp4_tucker_context_gemm_plan != nullptr;
#else
	const bool use_fp4_attention_bwd = false;
#endif

	if (use_fp4_attention_bwd)
	{
#ifdef DARKNET_HAS_FP4
		const int batch = l.tucker_heads * windows;
		const int key_pad = ((T + 15) / 16) * 16;
		auto * const scores_plan = static_cast<Darknet::Fp4GemmPlan *>(l.fp4_tucker_scores_gemm_plan);
		auto * const context_plan = static_cast<Darknet::Fp4GemmPlan *>(l.fp4_tucker_context_gemm_plan);
		const size_t scores_ws = Darknet::fp4_gemm_workspace_bytes(scores_plan);
		const size_t context_ws = Darknet::fp4_gemm_workspace_bytes(context_plan);

		// dAttn = dContext @ V^T -- same (T, T, D) shape/orientation as the forward
		// scores GEMM: A=dContext plain, B=V plain (already the required B-transposed layout).
		Darknet::fp4_half_to_float_gpu(d_context16, (size_t)T * D, l.fp4_tucker_a_gpu, batch, (size_t)T * D, (size_t)T * D);
		Darknet::fp4_half_to_float_gpu(v16, (size_t)T * D, l.fp4_tucker_b_gpu, batch, (size_t)T * D, (size_t)T * D);
		for (int b = 0; b < batch; ++b)
		{
			if (!Darknet::fp4_gemm_execute(scores_plan,
				l.fp4_tucker_a_gpu + (size_t)b * T * D, l.fp4_tucker_b_gpu + (size_t)b * T * D,
				l.fp4_tucker_out_gpu + (size_t)b * T * T, l.fp4_tucker_scores_lt_workspace, scores_ws))
			{
				darknet_fatal_error(DARKNET_LOC, "tucker_attention: FP4 dAttn GEMM failed");
			}
		}
		Darknet::fp4_scale_cast_float_to_half_gpu(l.fp4_tucker_out_gpu, (size_t)T * T, 1.0f, d_attn16,
			batch, (size_t)T * T, (size_t)T * T);

		{
			const float alpha = 1.0f;
			const float beta = 0.0f;
			CHECK_CUDNN(cudnnSoftmaxBackward(
				cudnn_handle(),
				CUDNN_SOFTMAX_ACCURATE,
				CUDNN_SOFTMAX_MODE_CHANNEL,
				&alpha,
				l.dstTensorDesc16,
				attn16,
				l.ddstTensorDesc16,
				d_attn16,
				&beta,
				l.dsrcTensorDesc16,
				d_scores16));
		}

		// dV = attn @ dContext: A=attn plain+padded, B=dContext transposed+padded.
		Darknet::fp4_pad_cols_half_to_float_gpu(attn16, T, T, key_pad, l.fp4_tucker_a_gpu,
			batch, (size_t)T * T, (size_t)T * key_pad);
		Darknet::fp4_transpose_pad_cols_half_to_float_gpu(d_context16, T, D, key_pad, l.fp4_tucker_b_gpu,
			batch, (size_t)T * D, (size_t)D * key_pad);
		for (int b = 0; b < batch; ++b)
		{
			if (!Darknet::fp4_gemm_execute(context_plan,
				l.fp4_tucker_a_gpu + (size_t)b * T * key_pad, l.fp4_tucker_b_gpu + (size_t)b * D * key_pad,
				l.fp4_tucker_out_gpu + (size_t)b * T * D, l.fp4_tucker_context_lt_workspace, context_ws))
			{
				darknet_fatal_error(DARKNET_LOC, "tucker_attention: FP4 dV GEMM failed");
			}
		}
		Darknet::fp4_scale_cast_float_to_half_gpu(l.fp4_tucker_out_gpu, (size_t)T * D, 1.0f, d_v16,
			batch, (size_t)T * D, (size_t)T * D);

		// dQ = dScores @ K: A=dScores plain+padded, B=K transposed+padded.
		Darknet::fp4_pad_cols_half_to_float_gpu(d_scores16, T, T, key_pad, l.fp4_tucker_a_gpu,
			batch, (size_t)T * T, (size_t)T * key_pad);
		Darknet::fp4_transpose_pad_cols_half_to_float_gpu(k16, T, D, key_pad, l.fp4_tucker_b_gpu,
			batch, (size_t)T * D, (size_t)D * key_pad);
		for (int b = 0; b < batch; ++b)
		{
			if (!Darknet::fp4_gemm_execute(context_plan,
				l.fp4_tucker_a_gpu + (size_t)b * T * key_pad, l.fp4_tucker_b_gpu + (size_t)b * D * key_pad,
				l.fp4_tucker_out_gpu + (size_t)b * T * D, l.fp4_tucker_context_lt_workspace, context_ws))
			{
				darknet_fatal_error(DARKNET_LOC, "tucker_attention: FP4 dQ GEMM failed");
			}
		}
		Darknet::fp4_scale_cast_float_to_half_gpu(l.fp4_tucker_out_gpu, (size_t)T * D, score_scale, d_q16,
			batch, (size_t)T * D, (size_t)T * D);

		// dK = dScores^T @ Q: A=dScores transposed+padded, B=Q transposed+padded.
		Darknet::fp4_transpose_pad_cols_half_to_float_gpu(d_scores16, T, T, key_pad, l.fp4_tucker_a_gpu,
			batch, (size_t)T * T, (size_t)T * key_pad);
		Darknet::fp4_transpose_pad_cols_half_to_float_gpu(q16, T, D, key_pad, l.fp4_tucker_b_gpu,
			batch, (size_t)T * D, (size_t)D * key_pad);
		for (int b = 0; b < batch; ++b)
		{
			if (!Darknet::fp4_gemm_execute(context_plan,
				l.fp4_tucker_a_gpu + (size_t)b * T * key_pad, l.fp4_tucker_b_gpu + (size_t)b * D * key_pad,
				l.fp4_tucker_out_gpu + (size_t)b * T * D, l.fp4_tucker_context_lt_workspace, context_ws))
			{
				darknet_fatal_error(DARKNET_LOC, "tucker_attention: FP4 dK GEMM failed");
			}
		}
		Darknet::fp4_scale_cast_float_to_half_gpu(l.fp4_tucker_out_gpu, (size_t)T * D, score_scale, d_k16,
			batch, (size_t)T * D, (size_t)T * D);
#endif
	}
	else
	{
		tucker_gemm_half_strided_batched(0, 1, T, T, D, 1.0f,
			d_context16, D, (long long)T * D,
			v16, D, (long long)T * D,
			0.0f,
			d_attn16, T, (long long)T * T,
			l.tucker_heads * windows,
			CUDA_R_16F);

		{
			const float alpha = 1.0f;
			const float beta = 0.0f;
			CHECK_CUDNN(cudnnSoftmaxBackward(
				cudnn_handle(),
				CUDNN_SOFTMAX_ACCURATE,
				CUDNN_SOFTMAX_MODE_CHANNEL,
				&alpha,
				l.dstTensorDesc16,
				attn16,
				l.ddstTensorDesc16,
				d_attn16,
				&beta,
				l.dsrcTensorDesc16,
				d_scores16));
		}

		tucker_gemm_half_strided_batched(0, 0, T, D, T, 1.0f,
			attn16, T, (long long)T * T,
			d_context16, D, (long long)T * D,
			0.0f,
			d_v16, D, (long long)T * D,
			l.tucker_heads * windows,
			CUDA_R_16F);
		tucker_gemm_half_strided_batched(0, 0, T, D, T, score_scale,
			d_scores16, T, (long long)T * T,
			k16, D, (long long)T * D,
			0.0f,
			d_q16, D, (long long)T * D,
			l.tucker_heads * windows,
			CUDA_R_16F);
		tucker_gemm_half_strided_batched(1, 0, T, D, T, score_scale,
			d_scores16, T, (long long)T * T,
			q16, D, (long long)T * D,
			0.0f,
			d_k16, D, (long long)T * D,
			l.tucker_heads * windows,
			CUDA_R_16F);
	}

	for (int hidx = 0; hidx < l.tucker_heads; ++hidx)
	{
		tucker_gemm_half(1, 0, l.tucker_rank_q, D, WT, 1.0f,
			q_latent16, l.tucker_rank_q,
			d_q16 + hidx * WT * D, D,
			1.0f,
			dq_core + hidx * l.tucker_rank_q * D, D,
			CUDA_R_32F);
		tucker_gemm_half(0, 1, WT, l.tucker_rank_q, D, 1.0f,
			d_q16 + hidx * WT * D, D,
			q_core16 + hidx * l.tucker_rank_q * D, D,
			hidx == 0 ? 0.0f : 1.0f,
			d_q_latent16, l.tucker_rank_q,
			CUDA_R_16F);

		tucker_gemm_half(1, 0, l.tucker_rank_k, D, WT, 1.0f,
			k_latent16, l.tucker_rank_k,
			d_k16 + hidx * WT * D, D,
			1.0f,
			dk_core + hidx * l.tucker_rank_k * D, D,
			CUDA_R_32F);
		tucker_gemm_half(0, 1, WT, l.tucker_rank_k, D, 1.0f,
			d_k16 + hidx * WT * D, D,
			k_core16 + hidx * l.tucker_rank_k * D, D,
			hidx == 0 ? 0.0f : 1.0f,
			d_k_latent16, l.tucker_rank_k,
			CUDA_R_16F);

		tucker_gemm_half(1, 0, l.tucker_rank_v, D, WT, 1.0f,
			v_latent16, l.tucker_rank_v,
			d_v16 + hidx * WT * D, D,
			1.0f,
			dv_core + hidx * l.tucker_rank_v * D, D,
			CUDA_R_32F);
		tucker_gemm_half(0, 1, WT, l.tucker_rank_v, D, 1.0f,
			d_v16 + hidx * WT * D, D,
			v_core16 + hidx * l.tucker_rank_v * D, D,
			hidx == 0 ? 0.0f : 1.0f,
			d_v_latent16, l.tucker_rank_v,
			CUDA_R_16F);
	}

	tucker_gemm_half(1, 0, l.c, l.tucker_rank_q, WT, 1.0f,
		tokens16, l.c,
		d_q_latent16, l.tucker_rank_q,
		1.0f,
		dq_basis, l.tucker_rank_q,
		CUDA_R_32F);
	tucker_gemm_half(1, 0, l.c, l.tucker_rank_k, WT, 1.0f,
		tokens16, l.c,
		d_k_latent16, l.tucker_rank_k,
		1.0f,
		dk_basis, l.tucker_rank_k,
		CUDA_R_32F);
	tucker_gemm_half(1, 0, l.c, l.tucker_rank_v, WT, 1.0f,
		tokens16, l.c,
		d_v_latent16, l.tucker_rank_v,
		1.0f,
		dv_basis, l.tucker_rank_v,
		CUDA_R_32F);

	tucker_gemm_half(0, 1, WT, l.c, l.tucker_rank_q, 1.0f,
		d_q_latent16, l.tucker_rank_q,
		q_basis16, l.tucker_rank_q,
		0.0f,
		d_tokens16, l.c,
		CUDA_R_16F);
	tucker_gemm_half(0, 1, WT, l.c, l.tucker_rank_k, 1.0f,
		d_k_latent16, l.tucker_rank_k,
		k_basis16, l.tucker_rank_k,
		1.0f,
		d_tokens16, l.c,
		CUDA_R_16F);
	tucker_gemm_half(0, 1, WT, l.c, l.tucker_rank_v, 1.0f,
		d_v_latent16, l.tucker_rank_v,
		v_basis16, l.tucker_rank_v,
		1.0f,
		d_tokens16, l.c,
		CUDA_R_16F);

	tucker_scatter_token_delta_half_kernel<<<cuda_gridsize(token_count), BLOCK, 0, get_cuda_stream()>>>(
		token_count, d_tokens16, state.delta,
		l.batch, l.h, l.w, l.c, M, win_h, win_w);
	CHECK_CUDA(cudaPeekAtLastError());
#else
	float *d_q_latent = l.tucker_q_latent_gpu + windows * T * l.tucker_rank_q;
	float *d_k_latent = l.tucker_k_latent_gpu + windows * T * l.tucker_rank_k;
	float *d_v_latent = l.tucker_v_latent_gpu + windows * T * l.tucker_rank_v;

	CHECK_CUDA(cudaMemsetAsync(d_q16, 0, static_cast<size_t>(head_count) * sizeof(__half), get_cuda_stream()));
	CHECK_CUDA(cudaMemsetAsync(d_k16, 0, static_cast<size_t>(head_count) * sizeof(__half), get_cuda_stream()));
	CHECK_CUDA(cudaMemsetAsync(d_v16, 0, static_cast<size_t>(head_count) * sizeof(__half), get_cuda_stream()));
	CHECK_CUDA(cudaMemsetAsync(d_q_latent, 0, static_cast<size_t>(windows) * T * l.tucker_rank_q * sizeof(float), get_cuda_stream()));
	CHECK_CUDA(cudaMemsetAsync(d_k_latent, 0, static_cast<size_t>(windows) * T * l.tucker_rank_k * sizeof(float), get_cuda_stream()));
	CHECK_CUDA(cudaMemsetAsync(d_v_latent, 0, static_cast<size_t>(windows) * T * l.tucker_rank_v * sizeof(float), get_cuda_stream()));
	CHECK_CUDA(cudaMemsetAsync(d_o_mix, 0, static_cast<size_t>(o_mix_count) * sizeof(float), get_cuda_stream()));

	tucker_output_to_rank_backward_half_basis_kernel<<<cuda_gridsize(total), BLOCK, 0, get_cuda_stream()>>>(
		total, l.delta_gpu, o_mix, o_basis16,
		do_basis, l.bias_updates_gpu, state.delta, d_o_mix,
		l.batch, l.h, l.w, l.c, l.n, M, win_h, win_w, l.tucker_rank_o);
	CHECK_CUDA(cudaPeekAtLastError());

	tucker_rank_to_context_backward_half_core_kernel<<<cuda_gridsize(head_count), BLOCK, 0, get_cuda_stream()>>>(
		head_count, context16, d_o_mix, o_core16, do_core, d_context16,
		T, l.tucker_heads, l.tucker_head_dim, l.tucker_rank_o);
	CHECK_CUDA(cudaPeekAtLastError());

	tucker_make_dattn_half_kernel<<<score_rows, tucker_pow2_threads(T), 0, get_cuda_stream()>>>(
		score_rows, v16, d_context16, d_attn16, T, l.tucker_heads, l.tucker_head_dim);
	CHECK_CUDA(cudaPeekAtLastError());

	{
		const float alpha = 1.0f;
		const float beta = 0.0f;
		CHECK_CUDNN(cudnnSoftmaxBackward(
			cudnn_handle(),
			CUDNN_SOFTMAX_ACCURATE,
			CUDNN_SOFTMAX_MODE_CHANNEL,
			&alpha,
			l.dstTensorDesc16,
			attn16,
			l.ddstTensorDesc16,
			d_attn16,
			&beta,
			l.dsrcTensorDesc16,
			d_scores16));
	}

	tucker_qkv_backward_from_softmax_half_kernel<<<cuda_gridsize(head_count), BLOCK, 0, get_cuda_stream()>>>(
		head_count, attn16, d_scores16, q16, k16, d_context16, d_q16, d_k16, d_v16,
		T, l.tucker_heads, l.tucker_head_dim);
	CHECK_CUDA(cudaPeekAtLastError());

	tucker_expand_backward_half_kernel<<<cuda_gridsize(head_count), BLOCK, 0, get_cuda_stream()>>>(
		head_count, l.tucker_q_latent_gpu, q_core16, d_q16, d_q_latent, dq_core,
		T, l.tucker_heads, l.tucker_rank_q, l.tucker_head_dim);
	tucker_expand_backward_half_kernel<<<cuda_gridsize(head_count), BLOCK, 0, get_cuda_stream()>>>(
		head_count, l.tucker_k_latent_gpu, k_core16, d_k16, d_k_latent, dk_core,
		T, l.tucker_heads, l.tucker_rank_k, l.tucker_head_dim);
	tucker_expand_backward_half_kernel<<<cuda_gridsize(head_count), BLOCK, 0, get_cuda_stream()>>>(
		head_count, l.tucker_v_latent_gpu, v_core16, d_v16, d_v_latent, dv_core,
		T, l.tucker_heads, l.tucker_rank_v, l.tucker_head_dim);
	CHECK_CUDA(cudaPeekAtLastError());

	tucker_basis_backward_half_weight_kernel<<<cuda_gridsize(token_count), BLOCK, 0, get_cuda_stream()>>>(
		token_count, l.tucker_windowed_input_gpu, q_basis16, k_basis16, v_basis16,
		d_q_latent, d_k_latent, d_v_latent, dq_basis, dk_basis, dv_basis, state.delta,
		l.batch, l.h, l.w, l.c, M, win_h, win_w, l.tucker_rank_q, l.tucker_rank_k, l.tucker_rank_v);
	CHECK_CUDA(cudaPeekAtLastError());
#endif
#else
	const float *q_basis = l.weights_gpu + off.q_basis;
	const float *k_basis = l.weights_gpu + off.k_basis;
	const float *v_basis = l.weights_gpu + off.v_basis;
	const float *q_core = l.weights_gpu + off.q_core;
	const float *k_core = l.weights_gpu + off.k_core;
	const float *v_core = l.weights_gpu + off.v_core;
	const float *o_core = l.weights_gpu + off.o_core;
	const float *o_basis = l.weights_gpu + off.o_basis;

	float *d_context = l.tucker_context_gpu + head_count;
	float *d_q = l.tucker_q_gpu + head_count;
	float *d_k = l.tucker_k_gpu + head_count;
	float *d_v = l.tucker_v_gpu + head_count;
	float *d_q_latent = l.tucker_q_latent_gpu + windows * T * l.tucker_rank_q;
	float *d_k_latent = l.tucker_k_latent_gpu + windows * T * l.tucker_rank_k;
	float *d_v_latent = l.tucker_v_latent_gpu + windows * T * l.tucker_rank_v;

	CHECK_CUDA(cudaMemsetAsync(d_q, 0, static_cast<size_t>(head_count) * sizeof(float), get_cuda_stream()));
	CHECK_CUDA(cudaMemsetAsync(d_k, 0, static_cast<size_t>(head_count) * sizeof(float), get_cuda_stream()));
	CHECK_CUDA(cudaMemsetAsync(d_v, 0, static_cast<size_t>(head_count) * sizeof(float), get_cuda_stream()));
	CHECK_CUDA(cudaMemsetAsync(d_q_latent, 0, static_cast<size_t>(windows) * T * l.tucker_rank_q * sizeof(float), get_cuda_stream()));
	CHECK_CUDA(cudaMemsetAsync(d_k_latent, 0, static_cast<size_t>(windows) * T * l.tucker_rank_k * sizeof(float), get_cuda_stream()));
	CHECK_CUDA(cudaMemsetAsync(d_v_latent, 0, static_cast<size_t>(windows) * T * l.tucker_rank_v * sizeof(float), get_cuda_stream()));
	CHECK_CUDA(cudaMemsetAsync(d_o_mix, 0, static_cast<size_t>(o_mix_count) * sizeof(float), get_cuda_stream()));

	tucker_output_to_rank_backward_kernel<<<cuda_gridsize(total), BLOCK, 0, get_cuda_stream()>>>(
		total, l.delta_gpu, o_mix, o_basis,
		do_basis, l.bias_updates_gpu, state.delta, d_o_mix,
		l.batch, l.h, l.w, l.c, l.n, M, win_h, win_w, l.tucker_rank_o);
	CHECK_CUDA(cudaPeekAtLastError());

	tucker_rank_to_context_backward_kernel<<<cuda_gridsize(head_count), BLOCK, 0, get_cuda_stream()>>>(
		head_count, l.tucker_context_gpu, d_o_mix, o_core, do_core, d_context,
		T, l.tucker_heads, l.tucker_head_dim, l.tucker_rank_o);
	CHECK_CUDA(cudaPeekAtLastError());

#ifdef CUDNN
	float *attn = l.tucker_scores_gpu + score_count;
	float *d_attn = l.tucker_scores_gpu + 2 * score_count;
	float *d_scores_for_qk = l.tucker_scores_gpu;
	tucker_make_dattn_float_kernel<<<score_rows, tucker_pow2_threads(T), 0, get_cuda_stream()>>>(
		score_rows, l.tucker_v_gpu, d_context, d_attn, T, l.tucker_heads, l.tucker_head_dim);
	CHECK_CUDA(cudaPeekAtLastError());
	{
		const float alpha = 1.0f;
		const float beta = 0.0f;
		CHECK_CUDNN(cudnnSoftmaxBackward(
			cudnn_handle(), CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_CHANNEL,
			&alpha, l.dstTensorDesc, attn, l.dstTensorDesc, d_attn, &beta, l.srcTensorDesc, l.tucker_scores_gpu));
	}
	tucker_qkv_backward_from_softmax_float_kernel<<<cuda_gridsize(head_count), BLOCK, 0, get_cuda_stream()>>>(
		head_count, attn, d_scores_for_qk, l.tucker_q_gpu, l.tucker_k_gpu, d_context, d_q, d_k, d_v,
		T, l.tucker_heads, l.tucker_head_dim);
#else
	const int score_threads = tucker_pow2_threads(T);
	tucker_attention_backward_row_kernel<<<score_rows, score_threads, score_threads * sizeof(float), get_cuda_stream()>>>(
		score_rows, l.tucker_scores_gpu, l.tucker_q_gpu, l.tucker_k_gpu, l.tucker_v_gpu,
		d_context, d_q, d_k, d_v, T, l.tucker_heads, l.tucker_head_dim);
#endif
	CHECK_CUDA(cudaPeekAtLastError());

	tucker_expand_backward_kernel<<<cuda_gridsize(head_count), BLOCK, 0, get_cuda_stream()>>>(
		head_count, l.tucker_q_latent_gpu, q_core, d_q, d_q_latent, dq_core,
		T, l.tucker_heads, l.tucker_rank_q, l.tucker_head_dim);
	tucker_expand_backward_kernel<<<cuda_gridsize(head_count), BLOCK, 0, get_cuda_stream()>>>(
		head_count, l.tucker_k_latent_gpu, k_core, d_k, d_k_latent, dk_core,
		T, l.tucker_heads, l.tucker_rank_k, l.tucker_head_dim);
	tucker_expand_backward_kernel<<<cuda_gridsize(head_count), BLOCK, 0, get_cuda_stream()>>>(
		head_count, l.tucker_v_latent_gpu, v_core, d_v, d_v_latent, dv_core,
		T, l.tucker_heads, l.tucker_rank_v, l.tucker_head_dim);
	CHECK_CUDA(cudaPeekAtLastError());

	tucker_basis_backward_kernel<<<cuda_gridsize(token_count), BLOCK, 0, get_cuda_stream()>>>(
		token_count, l.tucker_windowed_input_gpu, q_basis, k_basis, v_basis,
		d_q_latent, d_k_latent, d_v_latent, dq_basis, dk_basis, dv_basis, state.delta,
		l.batch, l.h, l.w, l.c, M, win_h, win_w, l.tucker_rank_q, l.tucker_rank_k, l.tucker_rank_v);
	CHECK_CUDA(cudaPeekAtLastError());
#endif

}

void update_tucker_attention_layer_gpu(Darknet::Layer & l, int batch, float learning_rate, float momentum, float decay, float loss_scale)
{
	TAT(TATPARMS);

	if (loss_scale != 1.0f)
	{
		scal_ongpu(l.nweights, 1.0f / loss_scale, l.weight_updates_gpu, 1);
		scal_ongpu(l.nbiases, 1.0f / loss_scale, l.bias_updates_gpu, 1);
	}

	axpy_ongpu(l.nweights, -decay * batch, l.weights_gpu, 1, l.weight_updates_gpu, 1);
	axpy_ongpu(l.nweights, learning_rate / batch, l.weight_updates_gpu, 1, l.weights_gpu, 1);
	scal_ongpu(l.nweights, momentum, l.weight_updates_gpu, 1);

	axpy_ongpu(l.nbiases, learning_rate / batch, l.bias_updates_gpu, 1, l.biases_gpu, 1);
	scal_ongpu(l.nbiases, momentum, l.bias_updates_gpu, 1);
}

#endif
