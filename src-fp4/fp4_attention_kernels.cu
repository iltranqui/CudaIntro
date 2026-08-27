#include "darknet_internal.hpp"
#include "fp4_attention_kernels.hpp"

#include <cuda_fp16.h>

namespace
{
	__global__ void fp4_half_to_float_kernel(const __half * src, size_t count, float * dst, size_t src_stride, size_t dst_stride)
	{
		src += blockIdx.z * src_stride;
		dst += blockIdx.z * dst_stride;
		const size_t index = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
		if (index < count)
		{
			dst[index] = __half2float(src[index]);
		}
	}

	__global__ void fp4_pad_cols_half_to_float_kernel(const __half * src, int rows, int cols, int cols_pad, float * dst, size_t src_stride, size_t dst_stride)
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
		dst[index] = (col < cols) ? __half2float(src[static_cast<size_t>(row) * cols + col]) : 0.0f;
	}

	__global__ void fp4_transpose_pad_cols_half_to_float_kernel(const __half * src, int rows, int cols, int rows_pad, float * dst, size_t src_stride, size_t dst_stride)
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
		dst[index] = (transposed_col < rows) ? __half2float(src[static_cast<size_t>(transposed_col) * cols + transposed_row]) : 0.0f;
	}

	__global__ void fp4_scale_cast_float_to_half_kernel(const float * src, size_t count, float scale, __half * dst, size_t src_stride, size_t dst_stride)
	{
		src += blockIdx.z * src_stride;
		dst += blockIdx.z * dst_stride;
		const size_t index = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
		if (index < count)
		{
			dst[index] = __float2half(src[index] * scale);
		}
	}
}

namespace Darknet
{
	void fp4_half_to_float_gpu(const void * src_half, const size_t count, float * dst, const int batch, const size_t src_stride, const size_t dst_stride)
	{
		TAT(TATPARMS);

		if (src_half == nullptr || dst == nullptr || count == 0 || batch <= 0)
		{
			return;
		}

		dim3 grid = cuda_gridsize(count);
		grid.z = static_cast<unsigned int>(batch);
		fp4_half_to_float_kernel<<<grid, BLOCK, 0, get_cuda_stream()>>>(
			static_cast<const __half *>(src_half), count, dst, src_stride, dst_stride);
		CHECK_CUDA(cudaPeekAtLastError());
	}

	void fp4_pad_cols_half_to_float_gpu(const void * src_half, const int rows, const int cols, const int cols_pad, float * dst, const int batch, const size_t src_stride, const size_t dst_stride)
	{
		TAT(TATPARMS);

		if (src_half == nullptr || dst == nullptr || rows <= 0 || cols <= 0 || cols_pad < cols || batch <= 0)
		{
			return;
		}

		const size_t total = static_cast<size_t>(rows) * static_cast<size_t>(cols_pad);
		dim3 grid = cuda_gridsize(total);
		grid.z = static_cast<unsigned int>(batch);
		fp4_pad_cols_half_to_float_kernel<<<grid, BLOCK, 0, get_cuda_stream()>>>(
			static_cast<const __half *>(src_half), rows, cols, cols_pad, dst, src_stride, dst_stride);
		CHECK_CUDA(cudaPeekAtLastError());
	}

	void fp4_transpose_pad_cols_half_to_float_gpu(const void * src_half, const int rows, const int cols, const int rows_pad, float * dst, const int batch, const size_t src_stride, const size_t dst_stride)
	{
		TAT(TATPARMS);

		if (src_half == nullptr || dst == nullptr || rows <= 0 || cols <= 0 || rows_pad < rows || batch <= 0)
		{
			return;
		}

		const size_t total = static_cast<size_t>(cols) * static_cast<size_t>(rows_pad);
		dim3 grid = cuda_gridsize(total);
		grid.z = static_cast<unsigned int>(batch);
		fp4_transpose_pad_cols_half_to_float_kernel<<<grid, BLOCK, 0, get_cuda_stream()>>>(
			static_cast<const __half *>(src_half), rows, cols, rows_pad, dst, src_stride, dst_stride);
		CHECK_CUDA(cudaPeekAtLastError());
	}

	void fp4_scale_cast_float_to_half_gpu(const float * src, const size_t count, const float scale, void * dst_half, const int batch, const size_t src_stride, const size_t dst_stride)
	{
		TAT(TATPARMS);

		if (src == nullptr || dst_half == nullptr || count == 0 || batch <= 0)
		{
			return;
		}

		dim3 grid = cuda_gridsize(count);
		grid.z = static_cast<unsigned int>(batch);
		fp4_scale_cast_float_to_half_kernel<<<grid, BLOCK, 0, get_cuda_stream()>>>(
			src, count, scale, static_cast<__half *>(dst_half), src_stride, dst_stride);
		CHECK_CUDA(cudaPeekAtLastError());
	}
}
