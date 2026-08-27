#include "darknet_internal.hpp"
#include "fp8_attention_kernels.hpp"

#include <cmath>

#if defined(DARKNET_GPU_CUDA) && defined(CUDART_VERSION) && CUDART_VERSION >= 12010
#include <cuda_fp16.h>
#include <cuda_fp8.h>

namespace
{
	dim3 fp8_attn_gridsize_256(const size_t n)
	{
		constexpr size_t threads = 256;
		size_t k = (n - 1) / threads + 1;
		size_t x = k;
		size_t y = 1;

		if (x > 65535)
		{
			x = static_cast<size_t>(std::ceil(std::sqrt(static_cast<double>(k))));
			y = (n - 1) / (x * threads) + 1;
		}

		return dim3(static_cast<unsigned int>(x), static_cast<unsigned int>(y), 1);
	}

	__device__ float fp8_attn_safe_scale(const float * scale_gpu)
	{
		const float scale = scale_gpu ? *scale_gpu : 1.0f;
		return (isfinite(scale) && scale > 0.0f) ? scale : 1.0f;
	}

	// dst_ld (0 = cols_pad) is the row stride of the destination in elements; batch/src_stride/
	// dst_stride let grid.z iterate multiple (window, head) slices in one launch, same convention
	// as fp8_quantize_pad_cols_kernel in fp8_kernels.cu.
	__global__ void fp8_quantize_half_pad_cols_kernel(const __half * src, int rows, int cols, int cols_pad, size_t dst_ld, const float * scale_gpu, __nv_fp8_e4m3 * dst, size_t src_stride, size_t dst_stride)
	{
		src += blockIdx.z * src_stride;
		dst += blockIdx.z * dst_stride;
		const size_t total = static_cast<size_t>(rows) * static_cast<size_t>(cols_pad);
		const size_t index = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
		if (index >= total)
		{
			return;
		}

		const int row = static_cast<int>(index / cols_pad);
		const int col = static_cast<int>(index - static_cast<size_t>(row) * cols_pad);
		float value = 0.0f;
		if (col < cols)
		{
			value = __half2float(src[static_cast<size_t>(row) * cols + col]);
			if (!isfinite(value))
			{
				value = 0.0f;
			}
			value /= fp8_attn_safe_scale(scale_gpu);
		}
		dst[static_cast<size_t>(row) * dst_ld + col] = __nv_fp8_e4m3(value);
	}

	__global__ void fp8_quantize_half_transpose_pad_cols_kernel(const __half * src, int rows, int cols, int rows_pad, const float * scale_gpu, __nv_fp8_e4m3 * dst, size_t src_stride, size_t dst_stride)
	{
		src += blockIdx.z * src_stride;
		dst += blockIdx.z * dst_stride;
		const size_t total = static_cast<size_t>(cols) * static_cast<size_t>(rows_pad);
		const size_t index = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
		if (index >= total)
		{
			return;
		}

		const int transposed_row = static_cast<int>(index / rows_pad);
		const int transposed_col = static_cast<int>(index - static_cast<size_t>(transposed_row) * rows_pad);
		float value = 0.0f;
		if (transposed_col < rows)
		{
			value = __half2float(src[static_cast<size_t>(transposed_col) * cols + transposed_row]);
			if (!isfinite(value))
			{
				value = 0.0f;
			}
			value /= fp8_attn_safe_scale(scale_gpu);
		}
		dst[index] = __nv_fp8_e4m3(value);
	}

	// Zero-pads BOTH rows and cols -- needed for cuBLASLt FP8's "A operand" (output_rows and
	// reduction_pad must both be multiples of 16).  Mirrors fp8_quantize_e5m2_pad_rows_cols_amax_kernel
	// in fp8_kernels.cu (minus amax recording, plus batch/stride support), for a __half source.
	__global__ void fp8_quantize_half_pad_rows_cols_kernel(const __half * src, int rows, int cols, int rows_pad, int cols_pad, const float * scale_gpu, __nv_fp8_e4m3 * dst, size_t src_stride, size_t dst_stride)
	{
		src += blockIdx.z * src_stride;
		dst += blockIdx.z * dst_stride;
		const size_t total = static_cast<size_t>(rows_pad) * static_cast<size_t>(cols_pad);
		const size_t index = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
		if (index >= total)
		{
			return;
		}

		const int row = static_cast<int>(index / cols_pad);
		const int col = static_cast<int>(index - static_cast<size_t>(row) * cols_pad);
		float value = 0.0f;
		if (row < rows && col < cols)
		{
			value = __half2float(src[static_cast<size_t>(row) * cols + col]);
			if (!isfinite(value))
			{
				value = 0.0f;
			}
			value /= fp8_attn_safe_scale(scale_gpu);
		}
		dst[index] = __nv_fp8_e4m3(value);
	}

	// Same as above, float32 source -- used for the post-softmax attention weights.
	__global__ void fp8_quantize_pad_rows_cols_kernel(const float * src, int rows, int cols, int rows_pad, int cols_pad, const float * scale_gpu, __nv_fp8_e4m3 * dst, size_t src_stride, size_t dst_stride)
	{
		src += blockIdx.z * src_stride;
		dst += blockIdx.z * dst_stride;
		const size_t total = static_cast<size_t>(rows_pad) * static_cast<size_t>(cols_pad);
		const size_t index = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
		if (index >= total)
		{
			return;
		}

		const int row = static_cast<int>(index / cols_pad);
		const int col = static_cast<int>(index - static_cast<size_t>(row) * cols_pad);
		float value = 0.0f;
		if (row < rows && col < cols)
		{
			value = src[static_cast<size_t>(row) * cols + col];
			if (!isfinite(value))
			{
				value = 0.0f;
			}
			value /= fp8_attn_safe_scale(scale_gpu);
		}
		dst[index] = __nv_fp8_e4m3(value);
	}

	// Reads the raw FP32 cuBLASLt D output of the scores GEMM (A=K padded to key_pad rows,
	// B=Q with T unconstrained columns).  Column-major (key_pad, T_query) is byte-identical
	// to row-major (T_query, key_pad) -- see the Fp8GemmSpec doc comment -- so this kernel
	// just drops the key_pad-T garbage columns per row (from K's zero-padded rows) while
	// casting into the tightly packed FP16 buffer cudnnSoftmaxForward already expects.
	// `score_scale` folds in the attention 1/sqrt(D) scaling the FP16 path applies via the
	// GEMM alpha (cuBLASLt's FP8 matmul has no alpha, only per-tensor dequant scale pointers).
	__global__ void fp8_dequant_compact_scores_half_kernel(const float * src, int t_query, int t_key, int key_pad, float score_scale, __half * dst, size_t src_stride, size_t dst_stride)
	{
		src += blockIdx.z * src_stride;
		dst += blockIdx.z * dst_stride;
		const size_t total = static_cast<size_t>(t_query) * static_cast<size_t>(t_key);
		const size_t index = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
		if (index >= total)
		{
			return;
		}

		const int q = static_cast<int>(index / t_key);
		const int k = static_cast<int>(index - static_cast<size_t>(q) * t_key);
		dst[index] = __float2half(src[static_cast<size_t>(k) + static_cast<size_t>(q) * key_pad] * score_scale);
	}

	__global__ void fp8_amax_half_kernel(const __half * src, size_t count, float * amax_gpu)
	{
		__shared__ float shared[256];
		const unsigned int tid = threadIdx.x;
		const size_t index = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + tid;
		float value = 0.0f;
		if (index < count)
		{
			value = fabsf(__half2float(src[index]));
			if (!isfinite(value))
			{
				value = 0.0f;
			}
		}
		shared[tid] = value;
		__syncthreads();

		for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1)
		{
			if (tid < stride)
			{
				shared[tid] = fmaxf(shared[tid], shared[tid + stride]);
			}
			__syncthreads();
		}

		if (tid == 0)
		{
			atomicMax(reinterpret_cast<int *>(amax_gpu), __float_as_int(shared[0]));
		}
	}
}

namespace Darknet
{
	void fp8_quantize_half_rowmajor_pad_cols_gpu(const void * src_half, const int rows, const int cols, const int cols_pad, const float * scale_gpu, void * dst_fp8, const size_t dst_ld, const int batch, const size_t src_stride, const size_t dst_stride)
	{
		TAT(TATPARMS);

		if (src_half == nullptr || dst_fp8 == nullptr || rows <= 0 || cols <= 0 || cols_pad < cols || batch <= 0)
		{
			return;
		}

		const size_t ld = dst_ld > 0 ? dst_ld : static_cast<size_t>(cols_pad);
		const size_t total = static_cast<size_t>(rows) * static_cast<size_t>(cols_pad);
		dim3 grid = cuda_gridsize(total);
		grid.z = static_cast<unsigned int>(batch);
		fp8_quantize_half_pad_cols_kernel<<<grid, BLOCK, 0, get_cuda_stream()>>>(
			static_cast<const __half *>(src_half), rows, cols, cols_pad, ld, scale_gpu, static_cast<__nv_fp8_e4m3 *>(dst_fp8), src_stride, dst_stride);
		CHECK_CUDA(cudaPeekAtLastError());
	}

	void fp8_quantize_half_transpose_rowmajor_pad_cols_gpu(const void * src_half, const int rows, const int cols, const int rows_pad, const float * scale_gpu, void * dst_fp8, const int batch, const size_t src_stride, const size_t dst_stride)
	{
		TAT(TATPARMS);

		if (src_half == nullptr || dst_fp8 == nullptr || rows <= 0 || cols <= 0 || rows_pad < rows || batch <= 0)
		{
			return;
		}

		const size_t total = static_cast<size_t>(cols) * static_cast<size_t>(rows_pad);
		dim3 grid = cuda_gridsize(total);
		grid.z = static_cast<unsigned int>(batch);
		fp8_quantize_half_transpose_pad_cols_kernel<<<grid, BLOCK, 0, get_cuda_stream()>>>(
			static_cast<const __half *>(src_half), rows, cols, rows_pad, scale_gpu, static_cast<__nv_fp8_e4m3 *>(dst_fp8), src_stride, dst_stride);
		CHECK_CUDA(cudaPeekAtLastError());
	}

	void fp8_dequant_compact_scores_half_gpu(const float * src, const int t_query, const int t_key, const int key_pad, const float score_scale, void * dst_half, const int batch, const size_t src_stride, const size_t dst_stride)
	{
		TAT(TATPARMS);

		if (src == nullptr || dst_half == nullptr || t_query <= 0 || t_key <= 0 || key_pad < t_key || batch <= 0)
		{
			return;
		}

		const size_t total = static_cast<size_t>(t_query) * static_cast<size_t>(t_key);
		dim3 grid = cuda_gridsize(total);
		grid.z = static_cast<unsigned int>(batch);
		fp8_dequant_compact_scores_half_kernel<<<grid, BLOCK, 0, get_cuda_stream()>>>(
			src, t_query, t_key, key_pad, score_scale, static_cast<__half *>(dst_half), src_stride, dst_stride);
		CHECK_CUDA(cudaPeekAtLastError());
	}

	void fp8_accumulate_amax_half_gpu(const void * src_half, const size_t count, float * amax_gpu)
	{
		TAT(TATPARMS);

		if (src_half == nullptr || amax_gpu == nullptr || count == 0)
		{
			return;
		}

		fp8_amax_half_kernel<<<fp8_attn_gridsize_256(count), 256, 0, get_cuda_stream()>>>(
			static_cast<const __half *>(src_half), count, amax_gpu);
		CHECK_CUDA(cudaPeekAtLastError());
	}

	void fp8_quantize_half_pad_rows_cols_gpu(const void * src_half, const int rows, const int cols, const int rows_pad, const int cols_pad, const float * scale_gpu, void * dst_fp8, const int batch, const size_t src_stride, const size_t dst_stride)
	{
		TAT(TATPARMS);

		if (src_half == nullptr || dst_fp8 == nullptr || rows <= 0 || cols <= 0 || rows_pad < rows || cols_pad < cols || batch <= 0)
		{
			return;
		}

		const size_t total = static_cast<size_t>(rows_pad) * static_cast<size_t>(cols_pad);
		dim3 grid = cuda_gridsize(total);
		grid.z = static_cast<unsigned int>(batch);
		fp8_quantize_half_pad_rows_cols_kernel<<<grid, BLOCK, 0, get_cuda_stream()>>>(
			static_cast<const __half *>(src_half), rows, cols, rows_pad, cols_pad, scale_gpu, static_cast<__nv_fp8_e4m3 *>(dst_fp8), src_stride, dst_stride);
		CHECK_CUDA(cudaPeekAtLastError());
	}

	void fp8_quantize_pad_rows_cols_gpu(const float * src, const int rows, const int cols, const int rows_pad, const int cols_pad, const float * scale_gpu, void * dst_fp8, const int batch, const size_t src_stride, const size_t dst_stride)
	{
		TAT(TATPARMS);

		if (src == nullptr || dst_fp8 == nullptr || rows <= 0 || cols <= 0 || rows_pad < rows || cols_pad < cols || batch <= 0)
		{
			return;
		}

		const size_t total = static_cast<size_t>(rows_pad) * static_cast<size_t>(cols_pad);
		dim3 grid = cuda_gridsize(total);
		grid.z = static_cast<unsigned int>(batch);
		fp8_quantize_pad_rows_cols_kernel<<<grid, BLOCK, 0, get_cuda_stream()>>>(
			src, rows, cols, rows_pad, cols_pad, scale_gpu, static_cast<__nv_fp8_e4m3 *>(dst_fp8), src_stride, dst_stride);
		CHECK_CUDA(cudaPeekAtLastError());
	}
}

#else

namespace Darknet
{
	void fp8_quantize_half_rowmajor_pad_cols_gpu(const void *, int, int, int, const float *, void *, size_t, int, size_t, size_t) {}
	void fp8_quantize_half_transpose_rowmajor_pad_cols_gpu(const void *, int, int, int, const float *, void *, int, size_t, size_t) {}
	void fp8_dequant_compact_scores_half_gpu(const float *, int, int, int, float, void *, int, size_t, size_t) {}
	void fp8_accumulate_amax_half_gpu(const void *, size_t, float *) {}
	void fp8_quantize_half_pad_rows_cols_gpu(const void *, int, int, int, int, const float *, void *, int, size_t, size_t) {}
	void fp8_quantize_pad_rows_cols_gpu(const float *, int, int, int, int, const float *, void *, int, size_t, size_t) {}
}

#endif
