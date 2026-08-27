#include "darknet_internal.hpp"
#include "fp8_kernels.hpp"
#include "fp8_scaling.hpp"

#include <cmath>

#if defined(DARKNET_GPU_CUDA) && defined(CUDART_VERSION) && CUDART_VERSION >= 12010
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#endif

namespace Darknet
{
	size_t fp8_tensor_bytes(const size_t elements)
	{
		return elements;
	}

	size_t fp8_rowmajor_pad_cols_bytes(const int rows, const int cols_pad)
	{
		return rows > 0 && cols_pad > 0 ? static_cast<size_t>(rows) * static_cast<size_t>(cols_pad) : 0;
	}

	size_t fp8_rowmajor_pad_rows_bytes(const int rows_pad, const int cols)
	{
		return rows_pad > 0 && cols > 0 ? static_cast<size_t>(rows_pad) * static_cast<size_t>(cols) : 0;
	}

	Fp8Im2colQuantizeKind fp8_im2col_quantize_kind(
		const int kernel_h, const int kernel_w,
		const int pad_h, const int pad_w,
		const int stride_h, const int stride_w,
		const int dilation_h, const int dilation_w)
	{
		if (kernel_h == 3 && kernel_w == 3 &&
			pad_h == 1 && pad_w == 1 &&
			dilation_h == 1 && dilation_w == 1 &&
			stride_h == stride_w)
		{
			if (stride_h == 1)
			{
				return Fp8Im2colQuantizeKind::Conv3x3Pad1Stride1;
			}
			if (stride_h == 2)
			{
				return Fp8Im2colQuantizeKind::Conv3x3Pad1Stride2;
			}
		}
		return Fp8Im2colQuantizeKind::Generic;
	}

	Fp8DgradEpilogueKind fp8_dgrad_epilogue_kind(
		const int kernel_h, const int kernel_w,
		const int pad_h, const int pad_w,
		const int stride_h, const int stride_w,
		const int dilation_h, const int dilation_w,
		const int height, const int width,
		const int height_col, const int width_col)
	{
		if (stride_h != 1 || stride_w != 1 ||
			dilation_h != 1 || dilation_w != 1 ||
			height_col != height || width_col != width)
		{
			return Fp8DgradEpilogueKind::Generic;
		}
		if (kernel_h == 1 && kernel_w == 1 && pad_h == 0 && pad_w == 0)
		{
			return Fp8DgradEpilogueKind::Direct1x1;
		}
		if (kernel_h == 3 && kernel_w == 3 && pad_h == 1 && pad_w == 1)
		{
			return Fp8DgradEpilogueKind::Conv3x3Stride1Pad1;
		}
		return Fp8DgradEpilogueKind::Generic;
	}
}

#if defined(DARKNET_GPU_CUDA) && defined(CUDART_VERSION) && CUDART_VERSION >= 12010

namespace
{
	dim3 fp8_gridsize_256(const size_t n)
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

	__device__ float fp8_safe_scale(const float * scale_gpu)
	{
		const float scale = scale_gpu ? *scale_gpu : 1.0f;
		return (isfinite(scale) && scale > 0.0f) ? scale : 1.0f;
	}

	__device__ float fp8_finite_value_and_amax(float value, float & amax)
	{
		amax = fabsf(value);
		if (!isfinite(value) || !isfinite(amax))
		{
			amax = 0.0f;
			value = 0.0f;
		}
		return value;
	}

	__device__ void fp8_record_block_amax(float amax, float * amax_gpu, float * shared, const unsigned int tid)
	{
		shared[tid] = amax;
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

	__device__ float fp8_im2col_ext_value(
		const float * data_im,
		const int row,
		const int col,
		const int height,
		const int width,
		const int kernel_h,
		const int kernel_w,
		const int pad_h,
		const int pad_w,
		const int stride_h,
		const int stride_w,
		const int dilation_h,
		const int dilation_w,
		const int height_col,
		const int width_col)
	{
		const int kernel_index = row % (kernel_h * kernel_w);
		const int c_im = row / (kernel_h * kernel_w);
		const int kernel_row = kernel_index / kernel_w;
		const int kernel_col = kernel_index - kernel_row * kernel_w;
		const int h_col = col / width_col;
		const int w_col = col - h_col * width_col;
		const int h_im = h_col * stride_h - pad_h + kernel_row * dilation_h;
		const int w_im = w_col * stride_w - pad_w + kernel_col * dilation_w;

		if (h_im < 0 || w_im < 0 || h_im >= height || w_im >= width)
		{
			return 0.0f;
		}
		return data_im[(c_im * height + h_im) * width + w_im];
	}

	// dst_ld lets several launches tile stripes of one wide row-major matrix (batch folded into k)
	__global__ void fp8_quantize_pad_cols_kernel(const float * src, int rows, int cols, int cols_pad, size_t dst_ld, const float * scale_gpu, __nv_fp8_e4m3 * dst, size_t src_stride, size_t dst_stride)
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
			value = src[static_cast<size_t>(row) * cols + col];
			if (!isfinite(value))
			{
				value = 0.0f;
			}
			value /= fp8_safe_scale(scale_gpu);
		}
		dst[static_cast<size_t>(row) * dst_ld + col] = __nv_fp8_e4m3(value);
	}

	__global__ void fp8_quantize_pad_cols_amax_kernel(const float * src, int rows, int cols, int cols_pad, const float * scale_gpu, __nv_fp8_e4m3 * dst, float * amax_gpu)
	{
		__shared__ float shared[256];
		const unsigned int tid = threadIdx.x;
		const size_t total = static_cast<size_t>(rows) * static_cast<size_t>(cols_pad);
		const size_t index = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + tid;

		float amax = 0.0f;
		float value = 0.0f;
		if (index < total)
		{
			const int row = static_cast<int>(index / cols_pad);
			const int col = static_cast<int>(index - static_cast<size_t>(row) * cols_pad);
			if (col < cols)
			{
				value = fp8_finite_value_and_amax(src[static_cast<size_t>(row) * cols + col], amax);
			}
			dst[index] = __nv_fp8_e4m3(value / fp8_safe_scale(scale_gpu));
		}

		fp8_record_block_amax(amax, amax_gpu, shared, tid);
	}

	__global__ void fp8_quantize_pad_rows_amax_kernel(const float * src, int rows, int cols, int rows_pad, const float * scale_gpu, __nv_fp8_e4m3 * dst, float * amax_gpu)
	{
		__shared__ float shared[256];
		const unsigned int tid = threadIdx.x;
		const size_t total = static_cast<size_t>(rows_pad) * static_cast<size_t>(cols);
		const size_t index = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + tid;

		float amax = 0.0f;
		float value = 0.0f;
		if (index < total)
		{
			const int row = static_cast<int>(index / cols);
			const int col = static_cast<int>(index - static_cast<size_t>(row) * cols);
			if (row < rows)
			{
				value = fp8_finite_value_and_amax(src[static_cast<size_t>(row) * cols + col], amax);
			}
			dst[index] = __nv_fp8_e4m3(value / fp8_safe_scale(scale_gpu));
		}

		fp8_record_block_amax(amax, amax_gpu, shared, tid);
	}

	__global__ void fp8_quantize_transpose_pad_cols_kernel(const float * src, int rows, int cols, int rows_pad, const float * scale_gpu, __nv_fp8_e4m3 * dst, size_t src_stride, size_t dst_stride)
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
			value = src[static_cast<size_t>(transposed_col) * cols + transposed_row];
			if (!isfinite(value))
			{
				value = 0.0f;
			}
			value /= fp8_safe_scale(scale_gpu);
		}
		dst[index] = __nv_fp8_e4m3(value);
	}

	__global__ void fp8_quantize_transpose_pad_cols_amax_kernel(const float * src, int rows, int cols, int rows_pad, const float * scale_gpu, __nv_fp8_e4m3 * dst, float * amax_gpu, size_t src_stride, size_t dst_stride)
	{
		src += blockIdx.z * src_stride;
		dst += blockIdx.z * dst_stride;
		__shared__ float shared[256];
		const unsigned int tid = threadIdx.x;
		const size_t total = static_cast<size_t>(cols) * static_cast<size_t>(rows_pad);
		const size_t index = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + tid;

		float amax = 0.0f;
		float value = 0.0f;
		if (index < total)
		{
			const int transposed_row = static_cast<int>(index / rows_pad);
			const int transposed_col = static_cast<int>(index - static_cast<size_t>(transposed_row) * rows_pad);
			if (transposed_col < rows)
			{
				value = fp8_finite_value_and_amax(src[static_cast<size_t>(transposed_col) * cols + transposed_row], amax);
			}
			dst[index] = __nv_fp8_e4m3(value / fp8_safe_scale(scale_gpu));
		}

		fp8_record_block_amax(amax, amax_gpu, shared, tid);
	}

	__global__ void fp8_quantize_transpose_pad_rows_kernel(const float * src, int rows, int cols, int cols_pad, const float * scale_gpu, __nv_fp8_e4m3 * dst)
	{
		const size_t total = static_cast<size_t>(cols_pad) * static_cast<size_t>(rows);
		const size_t index = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
		if (index >= total)
		{
			return;
		}

		const int transposed_row = static_cast<int>(index / rows);
		const int transposed_col = static_cast<int>(index - static_cast<size_t>(transposed_row) * rows);
		float value = 0.0f;
		if (transposed_row < cols)
		{
			value = src[static_cast<size_t>(transposed_col) * cols + transposed_row];
			if (!isfinite(value))
			{
				value = 0.0f;
			}
			value /= fp8_safe_scale(scale_gpu);
		}
		dst[index] = __nv_fp8_e4m3(value);
	}

	__global__ void fp8_quantize_transpose_pad_rows_amax_kernel(const float * src, int rows, int cols, int cols_pad, const float * scale_gpu, __nv_fp8_e4m3 * dst, float * amax_gpu)
	{
		__shared__ float shared[256];
		const unsigned int tid = threadIdx.x;
		const size_t total = static_cast<size_t>(cols_pad) * static_cast<size_t>(rows);
		const size_t index = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + tid;

		float amax = 0.0f;
		float value = 0.0f;
		if (index < total)
		{
			const int transposed_row = static_cast<int>(index / rows);
			const int transposed_col = static_cast<int>(index - static_cast<size_t>(transposed_row) * rows);
			if (transposed_row < cols)
			{
				value = fp8_finite_value_and_amax(src[static_cast<size_t>(transposed_col) * cols + transposed_row], amax);
			}
			dst[index] = __nv_fp8_e4m3(value / fp8_safe_scale(scale_gpu));
		}

		fp8_record_block_amax(amax, amax_gpu, shared, tid);
	}

	__global__ void fp8_im2col_quantize_pad_rows_amax_kernel(
		const float * data_im,
		const int rows,
		const int cols,
		const int rows_pad,
		const int height,
		const int width,
		const int kernel_h,
		const int kernel_w,
		const int pad_h,
		const int pad_w,
		const int stride_h,
		const int stride_w,
		const int dilation_h,
		const int dilation_w,
		const int height_col,
		const int width_col,
		const float * scale_gpu,
		__nv_fp8_e4m3 * dst,
		float * amax_gpu)
	{
		__shared__ float shared[256];
		const unsigned int tid = threadIdx.x;
		const size_t total = static_cast<size_t>(rows_pad) * static_cast<size_t>(cols);
		const size_t index = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + tid;

		float amax = 0.0f;
		float value = 0.0f;
		if (index < total)
		{
			const int row = static_cast<int>(index / cols);
			const int col = static_cast<int>(index - static_cast<size_t>(row) * cols);
			if (row < rows)
			{
				value = fp8_finite_value_and_amax(
					fp8_im2col_ext_value(
						data_im,
						row,
						col,
						height,
						width,
						kernel_h,
						kernel_w,
						pad_h,
						pad_w,
						stride_h,
						stride_w,
						dilation_h,
						dilation_w,
						height_col,
						width_col),
					amax);
			}
			dst[index] = __nv_fp8_e4m3(value / fp8_safe_scale(scale_gpu));
		}

		fp8_record_block_amax(amax, amax_gpu, shared, tid);
	}

	__global__ void fp8_im2col_quantize_transpose_pad_rows_amax_kernel(
		const float * data_im,
		const int rows,
		const int cols,
		const int cols_pad,
		const int height,
		const int width,
		const int kernel_h,
		const int kernel_w,
		const int pad_h,
		const int pad_w,
		const int stride_h,
		const int stride_w,
		const int dilation_h,
		const int dilation_w,
		const int height_col,
		const int width_col,
		const float * scale_gpu,
		__nv_fp8_e4m3 * dst,
		float * amax_gpu)
	{
		__shared__ float shared[256];
		const unsigned int tid = threadIdx.x;
		const size_t total = static_cast<size_t>(cols_pad) * static_cast<size_t>(rows);
		const size_t index = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + tid;

		float amax = 0.0f;
		float value = 0.0f;
		if (index < total)
		{
			const int transposed_row = static_cast<int>(index / rows);
			const int transposed_col = static_cast<int>(index - static_cast<size_t>(transposed_row) * rows);
			if (transposed_row < cols)
			{
				value = fp8_finite_value_and_amax(
					fp8_im2col_ext_value(
						data_im,
						transposed_col,
						transposed_row,
						height,
						width,
						kernel_h,
						kernel_w,
						pad_h,
						pad_w,
						stride_h,
						stride_w,
						dilation_h,
						dilation_w,
						height_col,
						width_col),
					amax);
			}
			dst[index] = __nv_fp8_e4m3(value / fp8_safe_scale(scale_gpu));
		}

		fp8_record_block_amax(amax, amax_gpu, shared, tid);
	}

	__global__ void fp8_quantize_e5m2_amax_kernel(const float * src, size_t count, const float * scale_gpu, __nv_fp8_e5m2 * dst, float * amax_gpu)
	{
		__shared__ float shared[256];
		const unsigned int tid = threadIdx.x;
		const size_t index = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + tid;

		float amax = 0.0f;
		float value = 0.0f;
		if (index < count)
		{
			value = fp8_finite_value_and_amax(src[index], amax);
			dst[index] = __nv_fp8_e5m2(value / fp8_safe_scale(scale_gpu));
		}

		fp8_record_block_amax(amax, amax_gpu, shared, tid);
	}

	__global__ void fp8_quantize_e5m2_pad_cols_amax_kernel(const float * src, int rows, int cols, int cols_pad, size_t dst_ld, const float * scale_gpu, __nv_fp8_e5m2 * dst, float * amax_gpu)
	{
		__shared__ float shared[256];
		const unsigned int tid = threadIdx.x;
		const size_t total = static_cast<size_t>(rows) * static_cast<size_t>(cols_pad);
		const size_t index = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + tid;

		float amax = 0.0f;
		float value = 0.0f;
		if (index < total)
		{
			const int row = static_cast<int>(index / cols_pad);
			const int col = static_cast<int>(index - static_cast<size_t>(row) * cols_pad);
			if (col < cols)
			{
				value = fp8_finite_value_and_amax(src[static_cast<size_t>(row) * cols + col], amax);
			}
			dst[static_cast<size_t>(row) * dst_ld + col] = __nv_fp8_e5m2(value / fp8_safe_scale(scale_gpu));
		}

		fp8_record_block_amax(amax, amax_gpu, shared, tid);
	}

	__global__ void fp8_quantize_e5m2_pad_rows_amax_kernel(const float * src, int rows, int cols, int rows_pad, const float * scale_gpu, __nv_fp8_e5m2 * dst, float * amax_gpu)
	{
		__shared__ float shared[256];
		const unsigned int tid = threadIdx.x;
		const size_t total = static_cast<size_t>(rows_pad) * static_cast<size_t>(cols);
		const size_t index = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + tid;

		float amax = 0.0f;
		float value = 0.0f;
		if (index < total)
		{
			const int row = static_cast<int>(index / cols);
			const int col = static_cast<int>(index - static_cast<size_t>(row) * cols);
			if (row < rows)
			{
				value = fp8_finite_value_and_amax(src[static_cast<size_t>(row) * cols + col], amax);
			}
			dst[index] = __nv_fp8_e5m2(value / fp8_safe_scale(scale_gpu));
		}

		fp8_record_block_amax(amax, amax_gpu, shared, tid);
	}

	__global__ void fp8_quantize_e5m2_pad_rows_cols_amax_kernel(const float * src, int rows, int cols, int rows_pad, int cols_pad, const float * scale_gpu, __nv_fp8_e5m2 * dst, float * amax_gpu)
	{
		__shared__ float shared[256];
		const unsigned int tid = threadIdx.x;
		const size_t total = static_cast<size_t>(rows_pad) * static_cast<size_t>(cols_pad);
		const size_t index = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + tid;

		float amax = 0.0f;
		float value = 0.0f;
		if (index < total)
		{
			const int row = static_cast<int>(index / cols_pad);
			const int col = static_cast<int>(index - static_cast<size_t>(row) * cols_pad);
			if (row < rows && col < cols)
			{
				value = fp8_finite_value_and_amax(src[static_cast<size_t>(row) * cols + col], amax);
			}
			dst[index] = __nv_fp8_e5m2(value / fp8_safe_scale(scale_gpu));
		}

		fp8_record_block_amax(amax, amax_gpu, shared, tid);
	}

		// dst = im2col^T row-major (spatial x rows_pad); pads the kernel dimension (TN GEMM B operand, forward)
		__global__ void fp8_im2col_quantize_transpose_pad_cols_kernel(
			const float * data_im,
			const int rows,
			const int cols,
			const int rows_pad,
			const int height,
			const int width,
			const int kernel_h,
			const int kernel_w,
			const int pad_h,
			const int pad_w,
			const int stride_h,
			const int stride_w,
			const int dilation_h,
			const int dilation_w,
			const int height_col,
			const int width_col,
			const float * scale_gpu,
			__nv_fp8_e4m3 * dst,
			size_t src_stride,
			size_t dst_stride)
		{
			data_im += blockIdx.z * src_stride;
			dst += blockIdx.z * dst_stride;
			const size_t total = static_cast<size_t>(cols) * static_cast<size_t>(rows_pad);
			const size_t index = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
			if (index >= total)
			{
				return;
			}

			const int out_row = static_cast<int>(index / rows_pad);
			const int out_col = static_cast<int>(index - static_cast<size_t>(out_row) * rows_pad);
			float value = 0.0f;
			if (out_col < rows)
			{
				value = fp8_im2col_ext_value(
					data_im,
					out_col,
					out_row,
					height,
					width,
					kernel_h,
					kernel_w,
					pad_h,
					pad_w,
					stride_h,
					stride_w,
					dilation_h,
					dilation_w,
					height_col,
					width_col);
				if (!isfinite(value))
				{
					value = 0.0f;
				}
			}
			dst[index] = __nv_fp8_e4m3(value / fp8_safe_scale(scale_gpu));
		}

		__global__ void fp8_im2col_quantize_transpose_pad_cols_amax_kernel(
			const float * data_im,
			const int rows,
			const int cols,
			const int rows_pad,
			const int height,
			const int width,
			const int kernel_h,
			const int kernel_w,
			const int pad_h,
			const int pad_w,
			const int stride_h,
			const int stride_w,
			const int dilation_h,
			const int dilation_w,
			const int height_col,
			const int width_col,
			const float * scale_gpu,
			__nv_fp8_e4m3 * dst,
			float * amax_gpu,
			size_t src_stride,
			size_t dst_stride)
		{
			data_im += blockIdx.z * src_stride;
			dst += blockIdx.z * dst_stride;
			__shared__ float shared[256];
			const unsigned int tid = threadIdx.x;
			const size_t total = static_cast<size_t>(cols) * static_cast<size_t>(rows_pad);
			const size_t index = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + tid;

			float amax = 0.0f;
			float value = 0.0f;
			if (index < total)
			{
				const int out_row = static_cast<int>(index / rows_pad);
				const int out_col = static_cast<int>(index - static_cast<size_t>(out_row) * rows_pad);
				if (out_col < rows)
				{
					value = fp8_finite_value_and_amax(
						fp8_im2col_ext_value(
							data_im,
							out_col,
							out_row,
							height,
							width,
							kernel_h,
							kernel_w,
							pad_h,
							pad_w,
							stride_h,
							stride_w,
							dilation_h,
							dilation_w,
							height_col,
							width_col),
						amax);
				}
				dst[index] = __nv_fp8_e4m3(value / fp8_safe_scale(scale_gpu));
			}

			fp8_record_block_amax(amax, amax_gpu, shared, tid);
		}

		// dst = im2col row-major (rows x cols_pad); pads the spatial dimension (TN GEMM B operand, wgrad)
		__global__ void fp8_im2col_quantize_pad_cols_kernel(
			const float * data_im,
			const int rows,
			const int cols,
			const int cols_pad,
			const size_t dst_ld,
			const int height,
			const int width,
			const int kernel_h,
			const int kernel_w,
			const int pad_h,
			const int pad_w,
			const int stride_h,
			const int stride_w,
			const int dilation_h,
			const int dilation_w,
			const int height_col,
			const int width_col,
			const float * scale_gpu,
			__nv_fp8_e4m3 * dst,
			size_t src_stride,
			size_t dst_stride)
		{
			data_im += blockIdx.z * src_stride;
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
				value = fp8_im2col_ext_value(
					data_im,
					row,
					col,
					height,
					width,
					kernel_h,
					kernel_w,
					pad_h,
					pad_w,
					stride_h,
					stride_w,
					dilation_h,
					dilation_w,
					height_col,
					width_col);
				if (!isfinite(value))
				{
					value = 0.0f;
				}
			}
			dst[static_cast<size_t>(row) * dst_ld + col] = __nv_fp8_e4m3(value / fp8_safe_scale(scale_gpu));
		}

		__global__ void fp8_im2col_quantize_3x3_pad1_stride_kernel(
			const float * data_im,
			const int channels,
			const int height,
			const int width,
			const int cols,
			const int cols_pad,
			const size_t dst_ld,
			const int height_col,
			const int width_col,
			const int stride,
			const float * scale_gpu,
			__nv_fp8_e4m3 * dst,
			size_t src_stride,
			size_t dst_stride)
		{
			data_im += blockIdx.z * src_stride;
			dst += blockIdx.z * dst_stride;
			const int rows = channels * 9;
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
				const int c = row / 9;
				const int kernel_index = row - c * 9;
				const int kernel_y = kernel_index / 3;
				const int kernel_x = kernel_index - kernel_y * 3;
				const int out_y = col / width_col;
				const int out_x = col - out_y * width_col;
				const int in_y = out_y * stride - 1 + kernel_y;
				const int in_x = out_x * stride - 1 + kernel_x;
				if (out_y < height_col && in_y >= 0 && in_y < height && in_x >= 0 && in_x < width)
				{
					value = data_im[(c * height + in_y) * width + in_x];
					if (!isfinite(value))
					{
						value = 0.0f;
					}
				}
			}
			dst[static_cast<size_t>(row) * dst_ld + col] = __nv_fp8_e4m3(value / fp8_safe_scale(scale_gpu));
		}

		// 4-wide variant: each thread quantizes 4 consecutive output columns and stores them as one
		// packed 32-bit write, so the store path issues 128-byte transactions per warp instead of 32.
		// Requires cols_pad, dst_ld and dst_stride to be multiples of 4 (they are multiples of 16).
		__global__ void fp8_im2col_quantize_pad_cols_x4_kernel(
			const float * data_im,
			const int rows,
			const int cols,
			const int cols_pad,
			const size_t dst_ld,
			const int height,
			const int width,
			const int kernel_h,
			const int kernel_w,
			const int pad_h,
			const int pad_w,
			const int stride_h,
			const int stride_w,
			const int dilation_h,
			const int dilation_w,
			const int height_col,
			const int width_col,
			const float * scale_gpu,
			__nv_fp8_e4m3 * dst,
			size_t src_stride,
			size_t dst_stride)
		{
			data_im += blockIdx.z * src_stride;
			dst += blockIdx.z * dst_stride;
			const int quads_per_row = cols_pad >> 2;
			const size_t total = static_cast<size_t>(rows) * static_cast<size_t>(quads_per_row);
			const size_t index = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
			if (index >= total)
			{
				return;
			}

			const int row = static_cast<int>(index / quads_per_row);
			const int col0 = static_cast<int>(index - static_cast<size_t>(row) * quads_per_row) << 2;
			// the (channel, kernel-offset) decomposition is shared by all 4 columns
			const int kernel_index = row % (kernel_h * kernel_w);
			const int c_im = row / (kernel_h * kernel_w);
			const int kernel_row = kernel_index / kernel_w;
			const int kernel_col = kernel_index - kernel_row * kernel_w;
			const float scale = fp8_safe_scale(scale_gpu);

			unsigned int packed = 0;
			#pragma unroll
			for (int j = 0; j < 4; ++j)
			{
				const int col = col0 + j;
				float value = 0.0f;
				if (col < cols)
				{
					const int h_col = col / width_col;
					const int w_col = col - h_col * width_col;
					const int h_im = h_col * stride_h - pad_h + kernel_row * dilation_h;
					const int w_im = w_col * stride_w - pad_w + kernel_col * dilation_w;
					if (h_im >= 0 && w_im >= 0 && h_im < height && w_im < width)
					{
						value = data_im[(c_im * height + h_im) * width + w_im];
						if (!isfinite(value))
						{
							value = 0.0f;
						}
					}
				}
				const __nv_fp8_e4m3 q(value / scale);
				packed |= static_cast<unsigned int>(*reinterpret_cast<const unsigned char *>(&q)) << (8 * j);
			}
			*reinterpret_cast<unsigned int *>(dst + static_cast<size_t>(row) * dst_ld + col0) = packed;
		}

		// 4-wide variant of the 3x3/pad1 fast path (same packed-store trick)
		__global__ void fp8_im2col_quantize_3x3_pad1_stride_x4_kernel(
			const float * data_im,
			const int channels,
			const int height,
			const int width,
			const int cols,
			const int cols_pad,
			const size_t dst_ld,
			const int height_col,
			const int width_col,
			const int stride,
			const float * scale_gpu,
			__nv_fp8_e4m3 * dst,
			size_t src_stride,
			size_t dst_stride)
		{
			data_im += blockIdx.z * src_stride;
			dst += blockIdx.z * dst_stride;
			const int rows = channels * 9;
			const int quads_per_row = cols_pad >> 2;
			const size_t total = static_cast<size_t>(rows) * static_cast<size_t>(quads_per_row);
			const size_t index = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
			if (index >= total)
			{
				return;
			}

			const int row = static_cast<int>(index / quads_per_row);
			const int col0 = static_cast<int>(index - static_cast<size_t>(row) * quads_per_row) << 2;
			const int c = row / 9;
			const int kernel_index = row - c * 9;
			const int kernel_y = kernel_index / 3;
			const int kernel_x = kernel_index - kernel_y * 3;
			const float scale = fp8_safe_scale(scale_gpu);

			unsigned int packed = 0;
			#pragma unroll
			for (int j = 0; j < 4; ++j)
			{
				const int col = col0 + j;
				float value = 0.0f;
				if (col < cols)
				{
					const int out_y = col / width_col;
					const int out_x = col - out_y * width_col;
					const int in_y = out_y * stride - 1 + kernel_y;
					const int in_x = out_x * stride - 1 + kernel_x;
					if (out_y < height_col && in_y >= 0 && in_y < height && in_x >= 0 && in_x < width)
					{
						value = data_im[(c * height + in_y) * width + in_x];
						if (!isfinite(value))
						{
							value = 0.0f;
						}
					}
				}
				const __nv_fp8_e4m3 q(value / scale);
				packed |= static_cast<unsigned int>(*reinterpret_cast<const unsigned char *>(&q)) << (8 * j);
			}
			*reinterpret_cast<unsigned int *>(dst + static_cast<size_t>(row) * dst_ld + col0) = packed;
		}

		// dst = src^T row-major (cols x rows_pad) in E5M2; pads the transposed inner dimension (TN GEMM B operand, dgrad)
		__global__ void fp8_quantize_e5m2_transpose_pad_cols_amax_kernel(const float * src, int rows, int cols, int rows_pad, const float * scale_gpu, __nv_fp8_e5m2 * dst, float * amax_gpu, size_t src_stride, size_t dst_stride)
		{
			src += blockIdx.z * src_stride;
			dst += blockIdx.z * dst_stride;
			__shared__ float shared[256];
			const unsigned int tid = threadIdx.x;
			const size_t total = static_cast<size_t>(cols) * static_cast<size_t>(rows_pad);
			const size_t index = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + tid;

			float amax = 0.0f;
			float value = 0.0f;
			if (index < total)
			{
				const int transposed_row = static_cast<int>(index / rows_pad);
				const int transposed_col = static_cast<int>(index - static_cast<size_t>(transposed_row) * rows_pad);
				if (transposed_col < rows)
				{
					value = fp8_finite_value_and_amax(src[static_cast<size_t>(transposed_col) * cols + transposed_row], amax);
				}
				dst[index] = __nv_fp8_e5m2(value / fp8_safe_scale(scale_gpu));
			}

			fp8_record_block_amax(amax, amax_gpu, shared, tid);
		}

		// one pass over dy produces both backward operand layouts through a shared-memory tile so
		// the transposed writes stay coalesced: dst_a = row-major (rows x cols) stripe padded to
		// cols_pad with row stride ld_a, dst_b = transposed (cols x rows_pad). blockIdx.z = image.
		__global__ void fp8_quantize_e5m2_dual_layout_amax_kernel(
			const float * src, const int rows, const int cols, const int cols_pad, const int rows_pad,
			const float * scale_gpu,
			__nv_fp8_e5m2 * dst_a, const size_t ld_a,
			__nv_fp8_e5m2 * dst_b,
			float * amax_gpu,
			const size_t src_stride, const size_t stride_a, const size_t stride_b)
		{
			__shared__ float tile[32][33];
			__shared__ float shared[256];
			src += blockIdx.z * src_stride;
			if (dst_a) dst_a += blockIdx.z * stride_a;
			if (dst_b) dst_b += blockIdx.z * stride_b;

			const int row0 = blockIdx.y * 32;
			const int col0 = blockIdx.x * 32;
			const float scale = fp8_safe_scale(scale_gpu);
			const unsigned int tx = threadIdx.x;			// 0..31, fast dimension
			const unsigned int ty = threadIdx.y;			// 0..7
			float amax = 0.0f;

			// load + row-major store: threads walk 4 tile rows, columns coalesced
			for (int dy = 0; dy < 32; dy += 8)
			{
				const int r = row0 + ty + dy;
				const int c = col0 + tx;
				float value = 0.0f;
				if (r < rows && c < cols)
				{
					float a;
					value = fp8_finite_value_and_amax(src[static_cast<size_t>(r) * cols + c], a);
					amax = fmaxf(amax, a);
				}
				tile[ty + dy][tx] = value;
				if (dst_a && r < rows && c < cols_pad)
				{
					dst_a[static_cast<size_t>(r) * ld_a + c] = __nv_fp8_e5m2(value / scale);
				}
			}

			__syncthreads();

			// transposed store: dst_b rows are source columns, coalesced along the padded row dim
			if (dst_b)
			{
				for (int dy = 0; dy < 32; dy += 8)
				{
					const int out_row = col0 + ty + dy;		// source column
					const int out_col = row0 + tx;			// source row (padded to rows_pad)
					if (out_row < cols && out_col < rows_pad)
					{
						const float value = out_col < rows ? tile[tx][ty + dy] : 0.0f;
						dst_b[static_cast<size_t>(out_row) * rows_pad + out_col] = __nv_fp8_e5m2(value / scale);
					}
				}
			}

			// fp8_record_block_amax reduces over blockDim.x; this block is 32x8, so reduce the
			// full 256 entries explicitly
			const unsigned int tid = ty * 32 + tx;
			shared[tid] = amax;
			__syncthreads();
			for (unsigned int stride = 128; stride > 0; stride >>= 1)
			{
				if (tid < stride)
				{
					shared[tid] = fmaxf(shared[tid], shared[tid + stride]);
				}
				__syncthreads();
			}
			if (tid == 0 && amax_gpu)
			{
				atomicMax(reinterpret_cast<int *>(amax_gpu), __float_as_int(shared[0]));
			}
		}

		// one pass over the FP32 weights writes both GEMM operand layouts:
		// dst_a = row-major (filters x kernel_pad), dst_b = transposed (kernel x filters_pad)
		__global__ void fp8_quantize_dual_layout_weights_kernel(
			const float * src, const int rows, const int cols, const int cols_pad, const int rows_pad,
			const float * scale_gpu,
			__nv_fp8_e4m3 * dst_a,
			__nv_fp8_e4m3 * dst_b)
		{
			__shared__ float tile[32][33];
			const int row0 = blockIdx.y * 32;
			const int col0 = blockIdx.x * 32;
			const float scale = fp8_safe_scale(scale_gpu);
			const unsigned int tx = threadIdx.x;
			const unsigned int ty = threadIdx.y;

			for (int dy = 0; dy < 32; dy += 8)
			{
				const int r = row0 + ty + dy;
				const int c = col0 + tx;
				float value = 0.0f;
				if (r < rows && c < cols)
				{
					value = src[static_cast<size_t>(r) * cols + c];
					if (!isfinite(value))
					{
						value = 0.0f;
					}
				}
				tile[ty + dy][tx] = value;
				if (r < rows && c < cols_pad)
				{
					dst_a[static_cast<size_t>(r) * cols_pad + c] = __nv_fp8_e4m3(value / scale);
				}
			}

			__syncthreads();

			for (int dy = 0; dy < 32; dy += 8)
			{
				const int out_row = col0 + ty + dy;
				const int out_col = row0 + tx;
				if (out_row < cols && out_col < rows_pad)
				{
					const float value = out_col < rows ? tile[tx][ty + dy] : 0.0f;
					dst_b[static_cast<size_t>(out_row) * rows_pad + out_col] = __nv_fp8_e4m3(value / scale);
				}
			}
		}

		// Same tiled pass as the dual GEMM layouts, with an optional unpadded KRSC store for cuDNN NHWC fprop.
		__global__ void fp8_quantize_triple_layout_weights_kernel(
			const float * src,
			const int filters,
			const int channels,
			const int kernel_h,
			const int kernel_w,
			const int kernel,
			const int kernel_pad,
			const int filters_pad,
			const float * scale_gpu,
			__nv_fp8_e4m3 * dst_rowmajor,
			__nv_fp8_e4m3 * dst_transposed,
			__nv_fp8_e4m3 * dst_krsc)
		{
			__shared__ float tile[32][33];
			const int row0 = blockIdx.y * 32;
			const int col0 = blockIdx.x * 32;
			const float scale = fp8_safe_scale(scale_gpu);
			const unsigned int tx = threadIdx.x;
			const unsigned int ty = threadIdx.y;

			for (int dy = 0; dy < 32; dy += 8)
			{
				const int k = row0 + ty + dy;
				const int col = col0 + tx;
				float value = 0.0f;
				if (k < filters && col < kernel)
				{
					value = src[static_cast<size_t>(k) * kernel + col];
					if (!isfinite(value))
					{
						value = 0.0f;
					}
				}
				tile[ty + dy][tx] = value;
				const __nv_fp8_e4m3 quantized(value / scale);
				if (dst_rowmajor && k < filters && col < kernel_pad)
				{
					dst_rowmajor[static_cast<size_t>(k) * kernel_pad + col] = quantized;
				}
				if (dst_krsc && k < filters && col < kernel)
				{
					const int c = col / (kernel_h * kernel_w);
					const int kernel_index = col - c * kernel_h * kernel_w;
					const int r = kernel_index / kernel_w;
					const int s = kernel_index - r * kernel_w;
					dst_krsc[((static_cast<size_t>(k) * kernel_h + r) * kernel_w + s) * channels + c] = quantized;
				}
			}

			__syncthreads();

			if (dst_transposed)
			{
				for (int dy = 0; dy < 32; dy += 8)
				{
					const int out_row = col0 + ty + dy;
					const int out_col = row0 + tx;
					if (out_row < kernel && out_col < filters_pad)
					{
						const float value = out_col < filters ? tile[tx][ty + dy] : 0.0f;
						dst_transposed[static_cast<size_t>(out_row) * filters_pad + out_col] = __nv_fp8_e4m3(value / scale);
					}
				}
			}
		}

		// NCHW->NHWC is a (C x HW) -> (HW x C) transpose per image; a 32x33 shared tile keeps both
		// the HW-contiguous loads and the C-contiguous stores coalesced (the old one-thread-per-
		// element form read the source with an H*W stride). blockIdx.z = image.
		__global__ void fp8_quantize_nchw_to_nhwc_kernel(
			const float * src,
			const int batch,
			const int channels,
			const int height,
			const int width,
			const float * scale_gpu,
			__nv_fp8_e4m3 * dst)
		{
			__shared__ float tile[32][33];
			const size_t spatial = static_cast<size_t>(height) * width;
			const int n = blockIdx.z;
			src += static_cast<size_t>(n) * channels * spatial;
			dst += static_cast<size_t>(n) * channels * spatial;

			const size_t p0 = static_cast<size_t>(blockIdx.x) * 32;	// spatial tile origin
			const int c0 = blockIdx.y * 32;							// channel tile origin
			const float scale = fp8_safe_scale(scale_gpu);
			const unsigned int tx = threadIdx.x;
			const unsigned int ty = threadIdx.y;

			for (int dy = 0; dy < 32; dy += 8)
			{
				const int c = c0 + ty + dy;
				const size_t p = p0 + tx;
				float value = 0.0f;
				if (c < channels && p < spatial)
				{
					value = src[static_cast<size_t>(c) * spatial + p];
					if (!isfinite(value))
					{
						value = 0.0f;
					}
				}
				tile[ty + dy][tx] = value;
			}

			__syncthreads();

			for (int dy = 0; dy < 32; dy += 8)
			{
				const size_t p = p0 + ty + dy;
				const int c = c0 + tx;
				if (p < spatial && c < channels)
				{
					dst[p * channels + c] = __nv_fp8_e4m3(tile[tx][ty + dy] / scale);
				}
			}
		}

		__global__ void fp8_quantize_nchw_to_nhwc_amax_kernel(
			const float * src,
			const int batch,
			const int channels,
			const int height,
			const int width,
			const float * scale_gpu,
			__nv_fp8_e4m3 * dst,
			float * amax_gpu)
		{
			__shared__ float tile[32][33];
			__shared__ float shared[256];
			const size_t spatial = static_cast<size_t>(height) * width;
			const int n = blockIdx.z;
			src += static_cast<size_t>(n) * channels * spatial;
			dst += static_cast<size_t>(n) * channels * spatial;

			const size_t p0 = static_cast<size_t>(blockIdx.x) * 32;
			const int c0 = blockIdx.y * 32;
			const float scale = fp8_safe_scale(scale_gpu);
			const unsigned int tx = threadIdx.x;
			const unsigned int ty = threadIdx.y;
			float amax = 0.0f;

			for (int dy = 0; dy < 32; dy += 8)
			{
				const int c = c0 + ty + dy;
				const size_t p = p0 + tx;
				float value = 0.0f;
				if (c < channels && p < spatial)
				{
					float a;
					value = fp8_finite_value_and_amax(src[static_cast<size_t>(c) * spatial + p], a);
					amax = fmaxf(amax, a);
				}
				tile[ty + dy][tx] = value;
			}

			__syncthreads();

			for (int dy = 0; dy < 32; dy += 8)
			{
				const size_t p = p0 + ty + dy;
				const int c = c0 + tx;
				if (p < spatial && c < channels)
				{
					dst[p * channels + c] = __nv_fp8_e4m3(tile[tx][ty + dy] / scale);
				}
			}

			// 32x8 block: reduce all 256 lanes explicitly (fp8_record_block_amax assumes 1-D blocks)
			const unsigned int tid = ty * 32 + tx;
			shared[tid] = amax;
			__syncthreads();
			for (unsigned int stride = 128; stride > 0; stride >>= 1)
			{
				if (tid < stride)
				{
					shared[tid] = fmaxf(shared[tid], shared[tid + stride]);
				}
				__syncthreads();
			}
			if (tid == 0 && amax_gpu)
			{
				atomicMax(reinterpret_cast<int *>(amax_gpu), __float_as_int(shared[0]));
			}
		}

		// Keep the normal FP32 NCHW output valid while directly creating the
		// following convolution's E4M3/NHWC operand.  There is no intermediate
		// dequantization: cuDNN consumes this allocation as FP8 on the next call.
		__global__ void fp8_relu_quantize_nchw_to_nhwc_kernel(
			float * src_dst,
			const int channels,
			const int spatial,
			const float * scale_gpu,
			__nv_fp8_e4m3 * dst,
			float * amax_gpu,
			const size_t total)
		{
			__shared__ float shared[256];
			const unsigned int tid = threadIdx.x;
			const size_t index = (blockIdx.x + static_cast<size_t>(blockIdx.y) * gridDim.x) * blockDim.x + tid;
			float amax = 0.0f;
			if (index < total)
			{
				float value = src_dst[index];
				value = isfinite(value) && value > 0.0f ? value : 0.0f;
				src_dst[index] = value;
				amax = value;
				const size_t image_size = static_cast<size_t>(channels) * spatial;
				const int image = static_cast<int>(index / image_size);
				const size_t remainder = index - static_cast<size_t>(image) * image_size;
				const int channel = static_cast<int>(remainder / spatial);
				const int pixel = static_cast<int>(remainder - static_cast<size_t>(channel) * spatial);
				dst[(static_cast<size_t>(image) * spatial + pixel) * channels + channel] =
					__nv_fp8_e4m3(value / fp8_safe_scale(scale_gpu));
			}
			if (amax_gpu)
			{
				fp8_record_block_amax(amax, amax_gpu, shared, tid);
			}
		}

		__global__ void fp8_quantize_weights_krsc_kernel(
			const float * src,
			const int filters,
			const int channels,
			const int kernel_h,
			const int kernel_w,
			const float * scale_gpu,
			__nv_fp8_e4m3 * dst)
		{
			const size_t total = static_cast<size_t>(filters) * channels * kernel_h * kernel_w;
			const size_t index = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
			if (index >= total)
			{
				return;
			}

			const int c = static_cast<int>(index % channels);
			const size_t krs_index = index / channels;
			const int s = static_cast<int>(krs_index % kernel_w);
			const size_t kr_index = krs_index / kernel_w;
			const int r = static_cast<int>(kr_index % kernel_h);
			const int k = static_cast<int>(kr_index / kernel_h);
			float value = src[((static_cast<size_t>(k) * channels + c) * kernel_h + r) * kernel_w + s];
			if (!isfinite(value))
			{
				value = 0.0f;
			}
			dst[index] = __nv_fp8_e4m3(value / fp8_safe_scale(scale_gpu));
		}

		__device__ __forceinline__ float fp8_nhwc_load_as_float(const float value)
		{
			return value;
		}

		__device__ __forceinline__ float fp8_nhwc_load_as_float(const __nv_bfloat16 value)
		{
			return __bfloat162float(value);
		}

		// NHWC->NCHW is the (HW x C) -> (C x HW) transpose per image, i.e. the exact mirror of
		// fp8_quantize_nchw_to_nhwc_kernel above. Reuse the same 32x33 shared tile so both the
		// channel-contiguous NHWC load and the spatial-contiguous NCHW store stay coalesced --
		// the earlier one-thread-per-element form gathered src with a `channels`-element stride,
		// which is fine at 512 channels x 35 pixels but wastes bandwidth at larger resolutions.
		template <typename SrcT>
		__global__ void fp8_nhwc_to_nchw_tiled_kernel(
			const SrcT * src,
			const int channels,
			const int height,
			const int width,
			const float * bias,
			float * dst)
		{
			__shared__ float tile[32][33];
			const size_t spatial = static_cast<size_t>(height) * width;
			const int n = blockIdx.z;
			src += static_cast<size_t>(n) * channels * spatial;
			dst += static_cast<size_t>(n) * channels * spatial;

			const size_t p0 = static_cast<size_t>(blockIdx.x) * 32;	// spatial tile origin
			const int c0 = blockIdx.y * 32;							// channel tile origin
			const unsigned int tx = threadIdx.x;
			const unsigned int ty = threadIdx.y;

			// load NHWC tile: tx walks channels (contiguous in src), coalesced read
			for (int dy = 0; dy < 32; dy += 8)
			{
				const size_t p = p0 + ty + dy;
				const int c = c0 + tx;
				float value = 0.0f;
				if (p < spatial && c < channels)
				{
					value = fp8_nhwc_load_as_float(src[p * channels + c]);
				}
				tile[ty + dy][tx] = value;
			}

			__syncthreads();

			// store NCHW tile: tx walks spatial (contiguous in dst), coalesced write
			for (int dy = 0; dy < 32; dy += 8)
			{
				const int c = c0 + ty + dy;
				const size_t p = p0 + tx;
				if (c < channels && p < spatial)
				{
					float value = tile[tx][ty + dy];
					if (bias)
					{
						value += bias[c];
					}
					dst[static_cast<size_t>(c) * spatial + p] = value;
				}
			}
		}

		__device__ float fp8_gemm_output_value(const void * src, const bool src_bf16, const size_t index)
		{
			return src_bf16 ?
				__bfloat162float(static_cast<const __nv_bfloat16 *>(src)[index]) :
				static_cast<const float *>(src)[index];
		}

		__global__ void fp8_colmajor_output_accumulate_rowmajor_kernel(
			const void * src,
			const int rows,
			const int cols,
			const bool src_bf16,
			const float alpha,
			float * dst)
		{
			const size_t total = static_cast<size_t>(rows) * cols;
			const size_t index = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
			if (index >= total)
			{
				return;
			}

			const int row = static_cast<int>(index / cols);
			const int col = static_cast<int>(index - static_cast<size_t>(row) * cols);
			const float value = fp8_gemm_output_value(src, src_bf16, static_cast<size_t>(col) * rows + row);
			dst[index] += alpha * value;
		}

		__global__ void fp8_colmajor_output_to_nchw_delta_kernel(
			const void * src,
			const int batch,
			const int channels,
			const int height,
			const int width,
			const int kernel_h,
			const int kernel_w,
			const int pad_h,
			const int pad_w,
			const int stride_h,
			const int stride_w,
			const int dilation_h,
			const int dilation_w,
			const int height_col,
			const int width_col,
			const bool src_bf16,
			float * delta)
		{
			const size_t total = static_cast<size_t>(batch) * channels * height * width;
			const size_t index = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
			if (index >= total)
			{
				return;
			}

			const int w = static_cast<int>(index % width);
			const size_t nch_index = index / width;
			const int h = static_cast<int>(nch_index % height);
			const size_t nc_index = nch_index / height;
			const int c = static_cast<int>(nc_index % channels);
			const int n = static_cast<int>(nc_index / channels);
			const int kernel = channels * kernel_h * kernel_w;
			const int spatial = height_col * width_col;
			const size_t src_batch_offset = static_cast<size_t>(n) * kernel * spatial;
			const int h_im = h + pad_h;
			const int w_im = w + pad_w;
			const int kernel_extent_h = (kernel_h - 1) * dilation_h + 1;
			const int kernel_extent_w = (kernel_w - 1) * dilation_w + 1;
			const int h_col_start = (h_im < kernel_extent_h) ? 0 : (h_im - kernel_extent_h) / stride_h + 1;
			const int h_col_end = min(h_im / stride_h + 1, height_col);
			const int w_col_start = (w_im < kernel_extent_w) ? 0 : (w_im - kernel_extent_w) / stride_w + 1;
			const int w_col_end = min(w_im / stride_w + 1, width_col);

			float value = 0.0f;
			for (int h_col = h_col_start; h_col < h_col_end; ++h_col)
			{
				const int h_k = h_im - h_col * stride_h;
				if (h_k % dilation_h != 0)
				{
					continue;
				}
				for (int w_col = w_col_start; w_col < w_col_end; ++w_col)
				{
					const int w_k = w_im - w_col * stride_w;
					if (w_k % dilation_w != 0)
					{
						continue;
					}
					const int kernel_index = (c * kernel_h + h_k / dilation_h) * kernel_w + w_k / dilation_w;
					const int spatial_index = h_col * width_col + w_col;
					value += fp8_gemm_output_value(
						src,
						src_bf16,
						src_batch_offset + static_cast<size_t>(spatial_index) * kernel + kernel_index);
				}
			}
			delta[index] += value;
		}

		__global__ void fp8_colmajor_output_to_nchw_delta_1x1_kernel(
			const void * src,
			const int batch,
			const int channels,
			const int height,
			const int width,
			const bool src_bf16,
			float * delta)
		{
			const size_t total = static_cast<size_t>(batch) * channels * height * width;
			const size_t index = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
			if (index >= total)
			{
				return;
			}

			const int w = static_cast<int>(index % width);
			const size_t nch_index = index / width;
			const int h = static_cast<int>(nch_index % height);
			const size_t nc_index = nch_index / height;
			const int c = static_cast<int>(nc_index % channels);
			const int n = static_cast<int>(nc_index / channels);
			const int spatial = height * width;
			const int spatial_index = h * width + w;
			const size_t src_index =
				static_cast<size_t>(n) * channels * spatial +
				static_cast<size_t>(spatial_index) * channels + c;
			delta[index] += fp8_gemm_output_value(src, src_bf16, src_index);
		}

		__global__ void fp8_colmajor_output_to_nchw_delta_3x3_s1_p1_kernel(
			const void * src,
			const int batch,
			const int channels,
			const int height,
			const int width,
			const bool src_bf16,
			float * delta)
		{
			const size_t total = static_cast<size_t>(batch) * channels * height * width;
			const size_t index = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
			if (index >= total)
			{
				return;
			}

			const int w = static_cast<int>(index % width);
			const size_t nch_index = index / width;
			const int h = static_cast<int>(nch_index % height);
			const size_t nc_index = nch_index / height;
			const int c = static_cast<int>(nc_index % channels);
			const int n = static_cast<int>(nc_index / channels);
			const int kernel = channels * 9;
			const int spatial = height * width;
			const size_t src_batch_offset = static_cast<size_t>(n) * kernel * spatial;

			float value = 0.0f;
			for (int kernel_y = 0; kernel_y < 3; ++kernel_y)
			{
				const int out_y = h + 1 - kernel_y;
				if (out_y < 0 || out_y >= height)
				{
					continue;
				}
				for (int kernel_x = 0; kernel_x < 3; ++kernel_x)
				{
					const int out_x = w + 1 - kernel_x;
					if (out_x < 0 || out_x >= width)
					{
						continue;
					}
					const int kernel_index = (c * 3 + kernel_y) * 3 + kernel_x;
					const int spatial_index = out_y * width + out_x;
					value += fp8_gemm_output_value(
						src,
						src_bf16,
						src_batch_offset + static_cast<size_t>(spatial_index) * kernel + kernel_index);
				}
			}
			delta[index] += value;
		}

		// single-thread delayed-scaling update: mirror of fp8_delayed_scaling_record_amax(), kept on
		// the device so training never has to synchronize amax values back to the host.
		// state layout: [0..15] amax history ring, [16] next write index (stored as float)
		__device__ void fp8_delayed_scale_update_device(float * amax_gpu, float * state, const int history_length, const float format_max, const int margin, float * scale_gpu)
		{
			const float amax = *amax_gpu;
			int next = static_cast<int>(state[history_length]);
			if (next < 0 || next >= history_length)
			{
				next = 0;
			}
			state[next] = (isfinite(amax) && amax > 0.0f) ? amax : 0.0f;
			state[history_length] = static_cast<float>((next + 1) % history_length);

			float history_max = 0.0f;
			for (int i = 0; i < history_length; ++i)
			{
				history_max = fmaxf(history_max, state[i]);
			}
			*scale_gpu = (history_max > 0.0f && format_max > 0.0f) ? ldexpf(history_max, margin) / format_max : 1.0f;
			*amax_gpu = 0.0f;
		}

		// three tensors' delayed-scaling updates in one launch: block b handles tensor b
		__global__ void fp8_delayed_scale_update3_kernel(
			const int history_length,
			float * amax0, float * state0, float format_max0, int margin0, float * scale0,
			float * amax1, float * state1, float format_max1, int margin1, float * scale1,
			float * amax2, float * state2, float format_max2, int margin2, float * scale2)
		{
			if (threadIdx.x != 0)
			{
				return;
			}
			switch (blockIdx.x)
			{
				case 0: if (amax0) fp8_delayed_scale_update_device(amax0, state0, history_length, format_max0, margin0, scale0); break;
				case 1: if (amax1) fp8_delayed_scale_update_device(amax1, state1, history_length, format_max1, margin1, scale1); break;
				case 2: if (amax2) fp8_delayed_scale_update_device(amax2, state2, history_length, format_max2, margin2, scale2); break;
			}
		}

		__global__ void fp8_clear_amax_kernel(float * amax_gpu)
		{
			if (threadIdx.x == 0 && blockIdx.x == 0)
			{
				*amax_gpu = 0.0f;
		}
	}

	__global__ void fp8_amax_kernel(const float * src, size_t count, float * amax_gpu)
	{
		__shared__ float shared[256];
		const unsigned int tid = threadIdx.x;
		const size_t index = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + tid;
		float value = 0.0f;
		if (index < count)
		{
			value = fabsf(src[index]);
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
	void fp8_quantize_rowmajor_pad_cols_gpu(const float * src, const int rows, const int cols, const int cols_pad, const float * scale_gpu, void * dst_fp8, const size_t dst_ld, const int batch, const size_t src_stride, const size_t dst_stride)
	{
		TAT(TATPARMS);

		if (src == nullptr || dst_fp8 == nullptr || rows <= 0 || cols <= 0 || cols_pad < cols || batch <= 0)
		{
			return;
		}

		const size_t ld = dst_ld > 0 ? dst_ld : static_cast<size_t>(cols_pad);
		const size_t total = static_cast<size_t>(rows) * static_cast<size_t>(cols_pad);
		dim3 grid = cuda_gridsize(total);
		grid.z = static_cast<unsigned int>(batch);
		fp8_quantize_pad_cols_kernel<<<grid, BLOCK, 0, get_cuda_stream()>>>(
			src, rows, cols, cols_pad, ld, scale_gpu, static_cast<__nv_fp8_e4m3 *>(dst_fp8), src_stride, dst_stride);
		CHECK_CUDA(cudaPeekAtLastError());
	}

	void fp8_quantize_transpose_rowmajor_pad_cols_gpu(const float * src, const int rows, const int cols, const int rows_pad, const float * scale_gpu, void * dst_fp8, const int batch, const size_t src_stride, const size_t dst_stride)
	{
		TAT(TATPARMS);

		if (src == nullptr || dst_fp8 == nullptr || rows <= 0 || cols <= 0 || rows_pad < rows || batch <= 0)
		{
			return;
		}

		const size_t total = static_cast<size_t>(cols) * static_cast<size_t>(rows_pad);
		dim3 grid = cuda_gridsize(total);
		grid.z = static_cast<unsigned int>(batch);
		fp8_quantize_transpose_pad_cols_kernel<<<grid, BLOCK, 0, get_cuda_stream()>>>(
			src, rows, cols, rows_pad, scale_gpu, static_cast<__nv_fp8_e4m3 *>(dst_fp8), src_stride, dst_stride);
		CHECK_CUDA(cudaPeekAtLastError());
	}

	void fp8_quantize_transpose_rowmajor_pad_rows_gpu(const float * src, const int rows, const int cols, const int rows_pad, const float * scale_gpu, void * dst_fp8)
	{
		TAT(TATPARMS);

		if (src == nullptr || dst_fp8 == nullptr || rows <= 0 || cols <= 0 || rows_pad < cols)
		{
			return;
		}

		const size_t total = static_cast<size_t>(rows_pad) * static_cast<size_t>(rows);
		fp8_quantize_transpose_pad_rows_kernel<<<cuda_gridsize(total), BLOCK, 0, get_cuda_stream()>>>(
			src, rows, cols, rows_pad, scale_gpu, static_cast<__nv_fp8_e4m3 *>(dst_fp8));
		CHECK_CUDA(cudaPeekAtLastError());
	}

	void fp8_quantize_rowmajor_pad_cols_record_amax_gpu(const float * src, const int rows, const int cols, const int cols_pad, const float * scale_gpu, void * dst_fp8, float * amax_gpu)
	{
		TAT(TATPARMS);

		if (src == nullptr || dst_fp8 == nullptr || amax_gpu == nullptr || rows <= 0 || cols <= 0 || cols_pad < cols)
		{
			return;
		}

		const size_t total = static_cast<size_t>(rows) * static_cast<size_t>(cols_pad);
		fp8_quantize_pad_cols_amax_kernel<<<fp8_gridsize_256(total), 256, 0, get_cuda_stream()>>>(
			src, rows, cols, cols_pad, scale_gpu, static_cast<__nv_fp8_e4m3 *>(dst_fp8), amax_gpu);
		CHECK_CUDA(cudaPeekAtLastError());
	}

	void fp8_quantize_rowmajor_pad_rows_record_amax_gpu(const float * src, const int rows, const int cols, const int rows_pad, const float * scale_gpu, void * dst_fp8, float * amax_gpu)
	{
		TAT(TATPARMS);

		if (src == nullptr || dst_fp8 == nullptr || amax_gpu == nullptr || rows <= 0 || cols <= 0 || rows_pad < rows)
		{
			return;
		}

		const size_t total = static_cast<size_t>(rows_pad) * static_cast<size_t>(cols);
		fp8_quantize_pad_rows_amax_kernel<<<fp8_gridsize_256(total), 256, 0, get_cuda_stream()>>>(
			src, rows, cols, rows_pad, scale_gpu, static_cast<__nv_fp8_e4m3 *>(dst_fp8), amax_gpu);
		CHECK_CUDA(cudaPeekAtLastError());
	}

	void fp8_quantize_transpose_rowmajor_pad_cols_record_amax_gpu(const float * src, const int rows, const int cols, const int rows_pad, const float * scale_gpu, void * dst_fp8, float * amax_gpu, const int batch, const size_t src_stride, const size_t dst_stride)
	{
		TAT(TATPARMS);

		if (src == nullptr || dst_fp8 == nullptr || amax_gpu == nullptr || rows <= 0 || cols <= 0 || rows_pad < rows || batch <= 0)
		{
			return;
		}

		const size_t total = static_cast<size_t>(cols) * static_cast<size_t>(rows_pad);
		dim3 grid = fp8_gridsize_256(total);
		grid.z = static_cast<unsigned int>(batch);
		fp8_quantize_transpose_pad_cols_amax_kernel<<<grid, 256, 0, get_cuda_stream()>>>(
			src, rows, cols, rows_pad, scale_gpu, static_cast<__nv_fp8_e4m3 *>(dst_fp8), amax_gpu, src_stride, dst_stride);
		CHECK_CUDA(cudaPeekAtLastError());
	}

	void fp8_quantize_transpose_rowmajor_pad_rows_record_amax_gpu(const float * src, const int rows, const int cols, const int rows_pad, const float * scale_gpu, void * dst_fp8, float * amax_gpu)
	{
		TAT(TATPARMS);

		if (src == nullptr || dst_fp8 == nullptr || amax_gpu == nullptr || rows <= 0 || cols <= 0 || rows_pad < cols)
		{
			return;
		}

		const size_t total = static_cast<size_t>(rows_pad) * static_cast<size_t>(rows);
		fp8_quantize_transpose_pad_rows_amax_kernel<<<fp8_gridsize_256(total), 256, 0, get_cuda_stream()>>>(
			src, rows, cols, rows_pad, scale_gpu, static_cast<__nv_fp8_e4m3 *>(dst_fp8), amax_gpu);
		CHECK_CUDA(cudaPeekAtLastError());
	}

	void fp8_im2col_quantize_rowmajor_pad_rows_record_amax_gpu(
		const float * data_im,
		const int channels, const int height, const int width,
		const int kernel_h, const int kernel_w,
		const int pad_h, const int pad_w,
		const int stride_h, const int stride_w,
		const int dilation_h, const int dilation_w,
		const int rows_pad,
		const float * scale_gpu,
		void * dst_fp8,
		float * amax_gpu)
	{
		TAT(TATPARMS);

		if (data_im == nullptr || dst_fp8 == nullptr || amax_gpu == nullptr ||
			channels <= 0 || height <= 0 || width <= 0 ||
			kernel_h <= 0 || kernel_w <= 0 ||
			stride_h <= 0 || stride_w <= 0 ||
			dilation_h <= 0 || dilation_w <= 0)
		{
			return;
		}

		const int height_col = (height + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
		const int width_col = (width + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
		const int rows = channels * kernel_h * kernel_w;
		if (height_col <= 0 || width_col <= 0 || rows_pad < rows)
		{
			return;
		}

		const int cols = height_col * width_col;
		const size_t total = static_cast<size_t>(rows_pad) * static_cast<size_t>(cols);
		fp8_im2col_quantize_pad_rows_amax_kernel<<<fp8_gridsize_256(total), 256, 0, get_cuda_stream()>>>(
			data_im,
			rows,
			cols,
			rows_pad,
			height,
			width,
			kernel_h,
			kernel_w,
			pad_h,
			pad_w,
			stride_h,
			stride_w,
			dilation_h,
			dilation_w,
			height_col,
			width_col,
			scale_gpu,
			static_cast<__nv_fp8_e4m3 *>(dst_fp8),
			amax_gpu);
		CHECK_CUDA(cudaPeekAtLastError());
	}

	void fp8_im2col_quantize_transpose_rowmajor_pad_rows_record_amax_gpu(
		const float * data_im,
		const int channels, const int height, const int width,
		const int kernel_h, const int kernel_w,
		const int pad_h, const int pad_w,
		const int stride_h, const int stride_w,
		const int dilation_h, const int dilation_w,
		const int cols_pad,
		const float * scale_gpu,
		void * dst_fp8,
		float * amax_gpu)
	{
		TAT(TATPARMS);

		if (data_im == nullptr || dst_fp8 == nullptr || amax_gpu == nullptr ||
			channels <= 0 || height <= 0 || width <= 0 ||
			kernel_h <= 0 || kernel_w <= 0 ||
			stride_h <= 0 || stride_w <= 0 ||
			dilation_h <= 0 || dilation_w <= 0)
		{
			return;
		}

		const int height_col = (height + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
		const int width_col = (width + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
		const int rows = channels * kernel_h * kernel_w;
		const int cols = height_col * width_col;
		if (height_col <= 0 || width_col <= 0 || cols_pad < cols)
		{
			return;
		}

		const size_t total = static_cast<size_t>(cols_pad) * static_cast<size_t>(rows);
		fp8_im2col_quantize_transpose_pad_rows_amax_kernel<<<fp8_gridsize_256(total), 256, 0, get_cuda_stream()>>>(
			data_im,
			rows,
			cols,
			cols_pad,
			height,
			width,
			kernel_h,
			kernel_w,
			pad_h,
			pad_w,
			stride_h,
			stride_w,
			dilation_h,
			dilation_w,
			height_col,
			width_col,
			scale_gpu,
			static_cast<__nv_fp8_e4m3 *>(dst_fp8),
			amax_gpu);
		CHECK_CUDA(cudaPeekAtLastError());
	}

	void fp8_quantize_e5m2_record_amax_gpu(const float * src, const size_t count, const float * scale_gpu, void * dst_fp8, float * amax_gpu)
	{
		TAT(TATPARMS);

		if (src == nullptr || dst_fp8 == nullptr || amax_gpu == nullptr || count == 0)
		{
			return;
		}

		fp8_quantize_e5m2_amax_kernel<<<fp8_gridsize_256(count), 256, 0, get_cuda_stream()>>>(
			src, count, scale_gpu, static_cast<__nv_fp8_e5m2 *>(dst_fp8), amax_gpu);
		CHECK_CUDA(cudaPeekAtLastError());
	}

	void fp8_im2col_quantize_transpose_rowmajor_pad_cols_gpu(
		const float * data_im,
		const int channels, const int height, const int width,
		const int kernel_h, const int kernel_w,
		const int pad_h, const int pad_w,
		const int stride_h, const int stride_w,
		const int dilation_h, const int dilation_w,
		const int rows_pad,
		const float * scale_gpu,
		void * dst_fp8,
		const int batch, const size_t src_stride, const size_t dst_stride)
	{
		TAT(TATPARMS);

		if (data_im == nullptr || dst_fp8 == nullptr ||
			channels <= 0 || height <= 0 || width <= 0 ||
			kernel_h <= 0 || kernel_w <= 0 ||
			stride_h <= 0 || stride_w <= 0 ||
			dilation_h <= 0 || dilation_w <= 0 || batch <= 0)
		{
			return;
		}

		const int height_col = (height + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
		const int width_col = (width + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
		const int rows = channels * kernel_h * kernel_w;
		if (height_col <= 0 || width_col <= 0 || rows_pad < rows)
		{
			return;
		}

		const int cols = height_col * width_col;
		const size_t total = static_cast<size_t>(cols) * static_cast<size_t>(rows_pad);
		dim3 grid = fp8_gridsize_256(total);
		grid.z = static_cast<unsigned int>(batch);
		fp8_im2col_quantize_transpose_pad_cols_kernel<<<grid, 256, 0, get_cuda_stream()>>>(
			data_im, rows, cols, rows_pad,
			height, width, kernel_h, kernel_w, pad_h, pad_w,
			stride_h, stride_w, dilation_h, dilation_w, height_col, width_col,
			scale_gpu, static_cast<__nv_fp8_e4m3 *>(dst_fp8), src_stride, dst_stride);
		CHECK_CUDA(cudaPeekAtLastError());
	}

	void fp8_im2col_quantize_transpose_rowmajor_pad_cols_record_amax_gpu(
		const float * data_im,
		const int channels, const int height, const int width,
		const int kernel_h, const int kernel_w,
		const int pad_h, const int pad_w,
		const int stride_h, const int stride_w,
		const int dilation_h, const int dilation_w,
		const int rows_pad,
		const float * scale_gpu,
		void * dst_fp8,
		float * amax_gpu,
		const int batch, const size_t src_stride, const size_t dst_stride)
	{
		TAT(TATPARMS);

		if (data_im == nullptr || dst_fp8 == nullptr || amax_gpu == nullptr ||
			channels <= 0 || height <= 0 || width <= 0 ||
			kernel_h <= 0 || kernel_w <= 0 ||
			stride_h <= 0 || stride_w <= 0 ||
			dilation_h <= 0 || dilation_w <= 0 || batch <= 0)
		{
			return;
		}

		const int height_col = (height + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
		const int width_col = (width + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
		const int rows = channels * kernel_h * kernel_w;
		if (height_col <= 0 || width_col <= 0 || rows_pad < rows)
		{
			return;
		}

		const int cols = height_col * width_col;
		const size_t total = static_cast<size_t>(cols) * static_cast<size_t>(rows_pad);
		dim3 grid = fp8_gridsize_256(total);
		grid.z = static_cast<unsigned int>(batch);
		fp8_im2col_quantize_transpose_pad_cols_amax_kernel<<<grid, 256, 0, get_cuda_stream()>>>(
			data_im, rows, cols, rows_pad,
			height, width, kernel_h, kernel_w, pad_h, pad_w,
			stride_h, stride_w, dilation_h, dilation_w, height_col, width_col,
			scale_gpu, static_cast<__nv_fp8_e4m3 *>(dst_fp8), amax_gpu, src_stride, dst_stride);
		CHECK_CUDA(cudaPeekAtLastError());
	}

	void fp8_im2col_quantize_rowmajor_pad_cols_gpu(
		const float * data_im,
		const int channels, const int height, const int width,
		const int kernel_h, const int kernel_w,
		const int pad_h, const int pad_w,
		const int stride_h, const int stride_w,
		const int dilation_h, const int dilation_w,
		const int cols_pad,
		const float * scale_gpu,
		void * dst_fp8,
		const size_t dst_ld,
		const int batch, const size_t src_stride, const size_t dst_stride)
	{
		TAT(TATPARMS);

		if (data_im == nullptr || dst_fp8 == nullptr ||
			channels <= 0 || height <= 0 || width <= 0 ||
			kernel_h <= 0 || kernel_w <= 0 ||
			stride_h <= 0 || stride_w <= 0 ||
			dilation_h <= 0 || dilation_w <= 0 || batch <= 0)
		{
			return;
		}

		const int height_col = (height + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
		const int width_col = (width + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
		const int rows = channels * kernel_h * kernel_w;
		const int cols = height_col * width_col;
		if (height_col <= 0 || width_col <= 0 || cols_pad < cols)
		{
			return;
		}

		const size_t ld = dst_ld > 0 ? dst_ld : static_cast<size_t>(cols_pad);
		const size_t total = static_cast<size_t>(rows) * static_cast<size_t>(cols_pad);
		// packed 32-bit stores need 4-element alignment of every per-row/per-image dst offset;
		// all are multiples of 16 in practice, so the scalar kernels are a rarely-taken fallback
		const bool can_x4 = (cols_pad % 4 == 0) && (ld % 4 == 0) && (dst_stride % 4 == 0);
		dim3 grid = can_x4 ? fp8_gridsize_256(total / 4) : fp8_gridsize_256(total);
		grid.z = static_cast<unsigned int>(batch);
		const Fp8Im2colQuantizeKind kind = fp8_im2col_quantize_kind(
			kernel_h, kernel_w, pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w);
		if (kind != Fp8Im2colQuantizeKind::Generic)
		{
			const int fast_stride = kind == Fp8Im2colQuantizeKind::Conv3x3Pad1Stride2 ? 2 : 1;
			if (can_x4)
			{
				fp8_im2col_quantize_3x3_pad1_stride_x4_kernel<<<grid, 256, 0, get_cuda_stream()>>>(
					data_im, channels, height, width, cols, cols_pad, ld,
					height_col, width_col, fast_stride,
					scale_gpu, static_cast<__nv_fp8_e4m3 *>(dst_fp8), src_stride, dst_stride);
			}
			else
			{
				fp8_im2col_quantize_3x3_pad1_stride_kernel<<<grid, 256, 0, get_cuda_stream()>>>(
					data_im, channels, height, width, cols, cols_pad, ld,
					height_col, width_col, fast_stride,
					scale_gpu, static_cast<__nv_fp8_e4m3 *>(dst_fp8), src_stride, dst_stride);
			}
			CHECK_CUDA(cudaPeekAtLastError());
			return;
		}
		if (can_x4)
		{
			fp8_im2col_quantize_pad_cols_x4_kernel<<<grid, 256, 0, get_cuda_stream()>>>(
				data_im, rows, cols, cols_pad, ld,
				height, width, kernel_h, kernel_w, pad_h, pad_w,
				stride_h, stride_w, dilation_h, dilation_w, height_col, width_col,
				scale_gpu, static_cast<__nv_fp8_e4m3 *>(dst_fp8), src_stride, dst_stride);
		}
		else
		{
			fp8_im2col_quantize_pad_cols_kernel<<<grid, 256, 0, get_cuda_stream()>>>(
				data_im, rows, cols, cols_pad, ld,
				height, width, kernel_h, kernel_w, pad_h, pad_w,
				stride_h, stride_w, dilation_h, dilation_w, height_col, width_col,
				scale_gpu, static_cast<__nv_fp8_e4m3 *>(dst_fp8), src_stride, dst_stride);
		}
		CHECK_CUDA(cudaPeekAtLastError());
	}

	void fp8_quantize_e5m2_transpose_rowmajor_pad_cols_record_amax_gpu(const float * src, const int rows, const int cols, const int rows_pad, const float * scale_gpu, void * dst_fp8, float * amax_gpu, const int batch, const size_t src_stride, const size_t dst_stride)
	{
		TAT(TATPARMS);

		if (src == nullptr || dst_fp8 == nullptr || amax_gpu == nullptr || rows <= 0 || cols <= 0 || rows_pad < rows || batch <= 0)
		{
			return;
		}

		const size_t total = static_cast<size_t>(cols) * static_cast<size_t>(rows_pad);
		dim3 grid = fp8_gridsize_256(total);
		grid.z = static_cast<unsigned int>(batch);
		fp8_quantize_e5m2_transpose_pad_cols_amax_kernel<<<grid, 256, 0, get_cuda_stream()>>>(
			src, rows, cols, rows_pad, scale_gpu, static_cast<__nv_fp8_e5m2 *>(dst_fp8), amax_gpu, src_stride, dst_stride);
		CHECK_CUDA(cudaPeekAtLastError());
	}

	void fp8_quantize_e5m2_dual_layout_record_amax_gpu(
		const float * src, const int rows, const int cols, const int cols_pad, const int rows_pad,
		const float * scale_gpu,
		void * dst_wgrad_fp8, const size_t wgrad_ld,
		void * dst_dgrad_fp8,
		float * amax_gpu,
		const int batch, const size_t src_stride, const size_t wgrad_stride, const size_t dgrad_stride)
	{
		TAT(TATPARMS);

		if (src == nullptr || amax_gpu == nullptr || (dst_wgrad_fp8 == nullptr && dst_dgrad_fp8 == nullptr) ||
			rows <= 0 || cols <= 0 || cols_pad < cols || rows_pad < rows || batch <= 0)
		{
			return;
		}

		const size_t ld_a = wgrad_ld > 0 ? wgrad_ld : static_cast<size_t>(cols_pad);
		const dim3 block(32, 8, 1);
		const dim3 grid(
			static_cast<unsigned int>((cols_pad + 31) / 32),
			static_cast<unsigned int>((rows_pad + 31) / 32),
			static_cast<unsigned int>(batch));
		fp8_quantize_e5m2_dual_layout_amax_kernel<<<grid, block, 0, get_cuda_stream()>>>(
			src, rows, cols, cols_pad, rows_pad, scale_gpu,
			static_cast<__nv_fp8_e5m2 *>(dst_wgrad_fp8), ld_a,
			static_cast<__nv_fp8_e5m2 *>(dst_dgrad_fp8),
			amax_gpu, src_stride, wgrad_stride, dgrad_stride);
		CHECK_CUDA(cudaPeekAtLastError());
	}

	void fp8_quantize_dual_layout_weights_gpu(
		const float * src, const int filters, const int kernel, const int kernel_pad, const int filters_pad,
		const float * scale_gpu, void * dst_rowmajor_fp8, void * dst_transposed_fp8)
	{
		TAT(TATPARMS);

		if (src == nullptr || dst_rowmajor_fp8 == nullptr || dst_transposed_fp8 == nullptr ||
			filters <= 0 || kernel <= 0 || kernel_pad < kernel || filters_pad < filters)
		{
			return;
		}

		const dim3 block(32, 8, 1);
		const dim3 grid(
			static_cast<unsigned int>((kernel_pad + 31) / 32),
			static_cast<unsigned int>((filters_pad + 31) / 32),
			1);
		fp8_quantize_dual_layout_weights_kernel<<<grid, block, 0, get_cuda_stream()>>>(
			src, filters, kernel, kernel_pad, filters_pad, scale_gpu,
			static_cast<__nv_fp8_e4m3 *>(dst_rowmajor_fp8),
			static_cast<__nv_fp8_e4m3 *>(dst_transposed_fp8));
		CHECK_CUDA(cudaPeekAtLastError());
	}

	void fp8_quantize_triple_layout_weights_gpu(
		const float * src_kcrs,
		const int filters,
		const int channels,
		const int kernel_h,
		const int kernel_w,
		const int kernel_pad,
		const int filters_pad,
		const float * scale_gpu,
		void * dst_rowmajor_fp8,
		void * dst_transposed_fp8,
		void * dst_krsc_fp8)
	{
		TAT(TATPARMS);

		const int kernel = channels * kernel_h * kernel_w;
		if (src_kcrs == nullptr || (dst_rowmajor_fp8 == nullptr && dst_transposed_fp8 == nullptr && dst_krsc_fp8 == nullptr) ||
			filters <= 0 || channels <= 0 || kernel_h <= 0 || kernel_w <= 0 || kernel_pad < kernel || filters_pad < filters)
		{
			return;
		}

		const dim3 block(32, 8, 1);
		const dim3 grid(
			static_cast<unsigned int>((kernel_pad + 31) / 32),
			static_cast<unsigned int>((filters_pad + 31) / 32),
			1);
		fp8_quantize_triple_layout_weights_kernel<<<grid, block, 0, get_cuda_stream()>>>(
			src_kcrs,
			filters,
			channels,
			kernel_h,
			kernel_w,
			kernel,
			kernel_pad,
			filters_pad,
			scale_gpu,
			static_cast<__nv_fp8_e4m3 *>(dst_rowmajor_fp8),
			static_cast<__nv_fp8_e4m3 *>(dst_transposed_fp8),
			static_cast<__nv_fp8_e4m3 *>(dst_krsc_fp8));
		CHECK_CUDA(cudaPeekAtLastError());
	}

	void fp8_quantize_nchw_to_nhwc_gpu(
		const float * src,
		const int batch,
		const int channels,
		const int height,
		const int width,
		const float * scale_gpu,
		void * dst_fp8)
	{
		TAT(TATPARMS);

		if (src == nullptr || dst_fp8 == nullptr || batch <= 0 || channels <= 0 || height <= 0 || width <= 0)
		{
			return;
		}

		const size_t spatial = static_cast<size_t>(height) * width;
		const dim3 block(32, 8, 1);
		const dim3 grid(
			static_cast<unsigned int>((spatial + 31) / 32),
			static_cast<unsigned int>((channels + 31) / 32),
			static_cast<unsigned int>(batch));
		fp8_quantize_nchw_to_nhwc_kernel<<<grid, block, 0, get_cuda_stream()>>>(
			src, batch, channels, height, width, scale_gpu, static_cast<__nv_fp8_e4m3 *>(dst_fp8));
		CHECK_CUDA(cudaPeekAtLastError());
	}

	void fp8_quantize_nchw_to_nhwc_record_amax_gpu(
		const float * src,
		const int batch,
		const int channels,
		const int height,
		const int width,
		const float * scale_gpu,
		void * dst_fp8,
		float * amax_gpu)
	{
		TAT(TATPARMS);

		if (src == nullptr || dst_fp8 == nullptr || amax_gpu == nullptr || batch <= 0 || channels <= 0 || height <= 0 || width <= 0)
		{
			return;
		}

		const size_t spatial = static_cast<size_t>(height) * width;
		const dim3 block(32, 8, 1);
		const dim3 grid(
			static_cast<unsigned int>((spatial + 31) / 32),
			static_cast<unsigned int>((channels + 31) / 32),
			static_cast<unsigned int>(batch));
		fp8_quantize_nchw_to_nhwc_amax_kernel<<<grid, block, 0, get_cuda_stream()>>>(
			src, batch, channels, height, width, scale_gpu, static_cast<__nv_fp8_e4m3 *>(dst_fp8), amax_gpu);
		CHECK_CUDA(cudaPeekAtLastError());
	}

	bool fp8_relu_quantize_nchw_to_nhwc_gpu(
		float * src_dst,
		const int batch,
		const int channels,
		const int height,
		const int width,
		const float * scale_gpu,
		void * dst_fp8,
		float * amax_gpu)
	{
		TAT(TATPARMS);

		if (src_dst == nullptr || dst_fp8 == nullptr || batch <= 0 || channels <= 0 || height <= 0 || width <= 0)
		{
			return false;
		}
		const size_t total = static_cast<size_t>(batch) * channels * height * width;
		fp8_relu_quantize_nchw_to_nhwc_kernel<<<fp8_gridsize_256(total), 256, 0, get_cuda_stream()>>>(
			src_dst,
			channels,
			height * width,
			scale_gpu,
			static_cast<__nv_fp8_e4m3 *>(dst_fp8),
			amax_gpu,
			total);
		return cudaPeekAtLastError() == cudaSuccess;
	}

	void fp8_quantize_weights_krsc_gpu(
		const float * src_kcrs,
		const int filters,
		const int channels,
		const int kernel_h,
		const int kernel_w,
		const float * scale_gpu,
		void * dst_fp8_krsc)
	{
		TAT(TATPARMS);

		if (src_kcrs == nullptr || dst_fp8_krsc == nullptr || filters <= 0 || channels <= 0 || kernel_h <= 0 || kernel_w <= 0)
		{
			return;
		}

		const size_t total = static_cast<size_t>(filters) * channels * kernel_h * kernel_w;
		fp8_quantize_weights_krsc_kernel<<<fp8_gridsize_256(total), 256, 0, get_cuda_stream()>>>(
			src_kcrs, filters, channels, kernel_h, kernel_w, scale_gpu, static_cast<__nv_fp8_e4m3 *>(dst_fp8_krsc));
		CHECK_CUDA(cudaPeekAtLastError());
	}

	void fp8_nhwc_output_to_nchw_gpu(
		const void * src,
		const int batch,
		const int channels,
		const int height,
		const int width,
		const bool src_bf16,
		const float * bias,
		float * dst)
	{
		TAT(TATPARMS);

		if (src == nullptr || dst == nullptr || batch <= 0 || channels <= 0 || height <= 0 || width <= 0)
		{
			return;
		}

		const size_t spatial = static_cast<size_t>(height) * width;
		const dim3 block(32, 8, 1);
		const dim3 grid(
			static_cast<unsigned int>((spatial + 31) / 32),
			static_cast<unsigned int>((channels + 31) / 32),
			static_cast<unsigned int>(batch));

		if (src_bf16)
		{
			fp8_nhwc_to_nchw_tiled_kernel<__nv_bfloat16><<<grid, block, 0, get_cuda_stream()>>>(
				static_cast<const __nv_bfloat16 *>(src), channels, height, width, bias, dst);
		}
		else
		{
			fp8_nhwc_to_nchw_tiled_kernel<float><<<grid, block, 0, get_cuda_stream()>>>(
				static_cast<const float *>(src), channels, height, width, bias, dst);
		}
		CHECK_CUDA(cudaPeekAtLastError());
	}

	void fp8_colmajor_output_accumulate_rowmajor_gpu(
		const void * src_colmajor,
		const int rows,
		const int cols,
		const bool src_bf16,
		const float alpha,
		float * dst_rowmajor)
	{
		TAT(TATPARMS);

		if (src_colmajor == nullptr || dst_rowmajor == nullptr || rows <= 0 || cols <= 0)
		{
			return;
		}

		const size_t total = static_cast<size_t>(rows) * cols;
		fp8_colmajor_output_accumulate_rowmajor_kernel<<<fp8_gridsize_256(total), 256, 0, get_cuda_stream()>>>(
			src_colmajor, rows, cols, src_bf16, alpha, dst_rowmajor);
		CHECK_CUDA(cudaPeekAtLastError());
	}

	void fp8_colmajor_output_to_nchw_delta_gpu(
		const void * src_colmajor,
		const int batch,
		const int channels,
		const int height,
		const int width,
		const int kernel_h,
		const int kernel_w,
		const int pad_h,
		const int pad_w,
		const int stride_h,
		const int stride_w,
		const int dilation_h,
		const int dilation_w,
		const bool src_bf16,
		float * delta_nchw)
	{
		TAT(TATPARMS);

		if (src_colmajor == nullptr || delta_nchw == nullptr ||
			batch <= 0 || channels <= 0 || height <= 0 || width <= 0 ||
			kernel_h <= 0 || kernel_w <= 0 ||
			stride_h <= 0 || stride_w <= 0 ||
			dilation_h <= 0 || dilation_w <= 0)
		{
			return;
		}

		const int height_col = (height + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
		const int width_col = (width + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
		if (height_col <= 0 || width_col <= 0)
		{
			return;
		}

		const size_t total = static_cast<size_t>(batch) * channels * height * width;
		const Fp8DgradEpilogueKind kind = fp8_dgrad_epilogue_kind(
			kernel_h, kernel_w, pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w,
			height, width, height_col, width_col);
		if (kind == Fp8DgradEpilogueKind::Direct1x1)
		{
			fp8_colmajor_output_to_nchw_delta_1x1_kernel<<<fp8_gridsize_256(total), 256, 0, get_cuda_stream()>>>(
				src_colmajor,
				batch,
				channels,
				height,
				width,
				src_bf16,
				delta_nchw);
			CHECK_CUDA(cudaPeekAtLastError());
			return;
		}
		if (kind == Fp8DgradEpilogueKind::Conv3x3Stride1Pad1)
		{
			fp8_colmajor_output_to_nchw_delta_3x3_s1_p1_kernel<<<fp8_gridsize_256(total), 256, 0, get_cuda_stream()>>>(
				src_colmajor,
				batch,
				channels,
				height,
				width,
				src_bf16,
				delta_nchw);
			CHECK_CUDA(cudaPeekAtLastError());
			return;
		}
		fp8_colmajor_output_to_nchw_delta_kernel<<<fp8_gridsize_256(total), 256, 0, get_cuda_stream()>>>(
			src_colmajor,
			batch,
			channels,
			height,
			width,
			kernel_h,
			kernel_w,
			pad_h,
			pad_w,
			stride_h,
			stride_w,
			dilation_h,
			dilation_w,
			height_col,
			width_col,
			src_bf16,
			delta_nchw);
		CHECK_CUDA(cudaPeekAtLastError());
	}

	void fp8_quantize_e5m2_rowmajor_pad_cols_record_amax_gpu(const float * src, const int rows, const int cols, const int cols_pad, const float * scale_gpu, void * dst_fp8, float * amax_gpu, const size_t dst_ld)
	{
		TAT(TATPARMS);

		if (src == nullptr || dst_fp8 == nullptr || amax_gpu == nullptr || rows <= 0 || cols <= 0 || cols_pad < cols)
		{
			return;
		}

		const size_t ld = dst_ld > 0 ? dst_ld : static_cast<size_t>(cols_pad);
		const size_t total = static_cast<size_t>(rows) * static_cast<size_t>(cols_pad);
		fp8_quantize_e5m2_pad_cols_amax_kernel<<<fp8_gridsize_256(total), 256, 0, get_cuda_stream()>>>(
			src, rows, cols, cols_pad, ld, scale_gpu, static_cast<__nv_fp8_e5m2 *>(dst_fp8), amax_gpu);
		CHECK_CUDA(cudaPeekAtLastError());
	}

		void fp8_quantize_e5m2_rowmajor_pad_rows_record_amax_gpu(const float * src, const int rows, const int cols, const int rows_pad, const float * scale_gpu, void * dst_fp8, float * amax_gpu)
		{
			TAT(TATPARMS);

			if (src == nullptr || dst_fp8 == nullptr || amax_gpu == nullptr || rows <= 0 || cols <= 0 || rows_pad < rows)
			{
				return;
		}

		const size_t total = static_cast<size_t>(rows_pad) * static_cast<size_t>(cols);
		fp8_quantize_e5m2_pad_rows_amax_kernel<<<fp8_gridsize_256(total), 256, 0, get_cuda_stream()>>>(
				src, rows, cols, rows_pad, scale_gpu, static_cast<__nv_fp8_e5m2 *>(dst_fp8), amax_gpu);
			CHECK_CUDA(cudaPeekAtLastError());
		}

		void fp8_quantize_e5m2_rowmajor_pad_rows_cols_record_amax_gpu(const float * src, const int rows, const int cols, const int rows_pad, const int cols_pad, const float * scale_gpu, void * dst_fp8, float * amax_gpu)
		{
			TAT(TATPARMS);

			if (src == nullptr || dst_fp8 == nullptr || amax_gpu == nullptr || rows <= 0 || cols <= 0 || rows_pad < rows || cols_pad < cols)
			{
				return;
			}

			const size_t total = static_cast<size_t>(rows_pad) * static_cast<size_t>(cols_pad);
			fp8_quantize_e5m2_pad_rows_cols_amax_kernel<<<fp8_gridsize_256(total), 256, 0, get_cuda_stream()>>>(
				src, rows, cols, rows_pad, cols_pad, scale_gpu, static_cast<__nv_fp8_e5m2 *>(dst_fp8), amax_gpu);
			CHECK_CUDA(cudaPeekAtLastError());
		}

		size_t fp8_scale_state_floats()
		{
			return static_cast<size_t>(kFp8AmaxHistoryLength) + 1;
		}

		void fp8_delayed_scale_update3_gpu(const Fp8ScaleUpdate & a, const Fp8ScaleUpdate & b, const Fp8ScaleUpdate & c)
		{
			TAT(TATPARMS);

			const bool ok_a = a.amax_gpu && a.state_gpu && a.scale_gpu;
			const bool ok_b = b.amax_gpu && b.state_gpu && b.scale_gpu;
			const bool ok_c = c.amax_gpu && c.state_gpu && c.scale_gpu;
			if (!ok_a && !ok_b && !ok_c)
			{
				return;
			}
			fp8_delayed_scale_update3_kernel<<<3, 1, 0, get_cuda_stream()>>>(
				kFp8AmaxHistoryLength,
				ok_a ? a.amax_gpu : nullptr, a.state_gpu, a.format_max, a.margin < 0 ? 0 : a.margin, a.scale_gpu,
				ok_b ? b.amax_gpu : nullptr, b.state_gpu, b.format_max, b.margin < 0 ? 0 : b.margin, b.scale_gpu,
				ok_c ? c.amax_gpu : nullptr, c.state_gpu, c.format_max, c.margin < 0 ? 0 : c.margin, c.scale_gpu);
			CHECK_CUDA(cudaPeekAtLastError());
		}

		void fp8_clear_amax_gpu(float * amax_gpu)
		{
			TAT(TATPARMS);

			if (amax_gpu == nullptr)
			{
				return;
		}
		fp8_clear_amax_kernel<<<1, 1, 0, get_cuda_stream()>>>(amax_gpu);
		CHECK_CUDA(cudaPeekAtLastError());
	}

	void fp8_accumulate_amax_gpu(const float * src, const size_t count, float * amax_gpu)
	{
		TAT(TATPARMS);

		if (src == nullptr || amax_gpu == nullptr || count == 0)
		{
			return;
		}
		fp8_amax_kernel<<<fp8_gridsize_256(count), 256, 0, get_cuda_stream()>>>(src, count, amax_gpu);
		CHECK_CUDA(cudaPeekAtLastError());
	}

	float fp8_pull_amax_gpu(float * amax_gpu)
	{
		float amax = 0.0f;
		if (amax_gpu)
		{
			CHECK_CUDA(cudaMemcpyAsync(&amax, amax_gpu, sizeof(float), cudaMemcpyDeviceToHost, get_cuda_stream()));
			CHECK_CUDA(cudaStreamSynchronize(get_cuda_stream()));
		}
		return amax;
	}
}

#else

namespace Darknet
{
	void fp8_quantize_rowmajor_pad_cols_gpu(const float *, int, int, int, const float *, void *, size_t, int, size_t, size_t) {}
	void fp8_quantize_transpose_rowmajor_pad_cols_gpu(const float *, int, int, int, const float *, void *, int, size_t, size_t) {}
	void fp8_quantize_transpose_rowmajor_pad_rows_gpu(const float *, int, int, int, const float *, void *) {}
	void fp8_quantize_rowmajor_pad_cols_record_amax_gpu(const float *, int, int, int, const float *, void *, float *) {}
	void fp8_quantize_rowmajor_pad_rows_record_amax_gpu(const float *, int, int, int, const float *, void *, float *) {}
	void fp8_quantize_transpose_rowmajor_pad_cols_record_amax_gpu(const float *, int, int, int, const float *, void *, float *, int, size_t, size_t) {}
	void fp8_quantize_transpose_rowmajor_pad_rows_record_amax_gpu(const float *, int, int, int, const float *, void *, float *) {}
	void fp8_im2col_quantize_rowmajor_pad_rows_record_amax_gpu(const float *, int, int, int, int, int, int, int, int, int, int, int, int, const float *, void *, float *) {}
	void fp8_im2col_quantize_transpose_rowmajor_pad_rows_record_amax_gpu(const float *, int, int, int, int, int, int, int, int, int, int, int, int, const float *, void *, float *) {}
	void fp8_im2col_quantize_transpose_rowmajor_pad_cols_gpu(const float *, int, int, int, int, int, int, int, int, int, int, int, int, const float *, void *, int, size_t, size_t) {}
	void fp8_im2col_quantize_transpose_rowmajor_pad_cols_record_amax_gpu(const float *, int, int, int, int, int, int, int, int, int, int, int, int, const float *, void *, float *, int, size_t, size_t) {}
	void fp8_im2col_quantize_rowmajor_pad_cols_gpu(const float *, int, int, int, int, int, int, int, int, int, int, int, int, const float *, void *, size_t, int, size_t, size_t) {}
	void fp8_quantize_e5m2_transpose_rowmajor_pad_cols_record_amax_gpu(const float *, int, int, int, const float *, void *, float *, int, size_t, size_t) {}
	void fp8_quantize_e5m2_dual_layout_record_amax_gpu(const float *, int, int, int, int, const float *, void *, size_t, void *, float *, int, size_t, size_t, size_t) {}
	void fp8_quantize_dual_layout_weights_gpu(const float *, int, int, int, int, const float *, void *, void *) {}
	void fp8_quantize_triple_layout_weights_gpu(const float *, int, int, int, int, int, int, const float *, void *, void *, void *) {}
	void fp8_colmajor_output_accumulate_rowmajor_gpu(const void *, int, int, bool, float, float *) {}
	void fp8_colmajor_output_to_nchw_delta_gpu(const void *, int, int, int, int, int, int, int, int, int, int, int, int, bool, float *) {}
	void fp8_quantize_nchw_to_nhwc_gpu(const float *, int, int, int, int, const float *, void *) {}
	void fp8_quantize_nchw_to_nhwc_record_amax_gpu(const float *, int, int, int, int, const float *, void *, float *) {}
	bool fp8_relu_quantize_nchw_to_nhwc_gpu(float *, int, int, int, int, const float *, void *, float *) { return false; }
	void fp8_quantize_weights_krsc_gpu(const float *, int, int, int, int, const float *, void *) {}
	void fp8_nhwc_output_to_nchw_gpu(const void *, int, int, int, int, bool, const float *, float *) {}
	void fp8_delayed_scale_update3_gpu(const Fp8ScaleUpdate &, const Fp8ScaleUpdate &, const Fp8ScaleUpdate &) {}
		void fp8_quantize_e5m2_record_amax_gpu(const float *, size_t, const float *, void *, float *) {}
		void fp8_quantize_e5m2_rowmajor_pad_cols_record_amax_gpu(const float *, int, int, int, const float *, void *, float *, size_t) {}
		void fp8_quantize_e5m2_rowmajor_pad_rows_record_amax_gpu(const float *, int, int, int, const float *, void *, float *) {}
		void fp8_quantize_e5m2_rowmajor_pad_rows_cols_record_amax_gpu(const float *, int, int, int, int, const float *, void *, float *) {}
		size_t fp8_scale_state_floats() { return 17; }
		void fp8_clear_amax_gpu(float *) {}
		void fp8_accumulate_amax_gpu(const float *, size_t, float *) {}
		float fp8_pull_amax_gpu(float *) { return 0.0f; }
	}

#endif
