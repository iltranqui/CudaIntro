#include "fp4_kernels.hpp"

#include "darknet_internal.hpp"

#include <cuda_runtime.h>
#include <cuda_fp4.h>
#include <cuda_fp8.h>

#include <cfloat>
#include <cmath>
#include <limits>

namespace Darknet
{
	namespace
	{
		__device__ __constant__ float fp4_positive_values[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};

		__device__ uint64_t fp4_splitmix64(uint64_t value)
		{
			value += 0x9e3779b97f4a7c15ULL;
			value = (value ^ (value >> 30U)) * 0xbf58476d1ce4e5b9ULL;
			value = (value ^ (value >> 27U)) * 0x94d049bb133111ebULL;
			return value ^ (value >> 31U);
		}

		__device__ uint8_t fp4_encode_nearest(const float value)
		{
			if (isnan(value)) return 0x07U;
			const uint8_t sign = (__float_as_uint(value) >> 31U) != 0U ? 0x08U : 0U;
			const float magnitude = isinf(value) ? 6.0f : fminf(fabsf(value), 6.0f);
			unsigned best = 0U;
			float best_distance = FLT_MAX;
			#pragma unroll
			for (unsigned idx = 0; idx < 8U; ++idx)
			{
				const float distance = fabsf(magnitude - fp4_positive_values[idx]);
				if (distance < best_distance || (distance == best_distance && (idx & 1U) == 0U))
				{
					best = idx;
					best_distance = distance;
				}
			}
			return static_cast<uint8_t>(sign | best);
		}

		__device__ uint8_t fp4_encode_stochastic(const float value, const uint64_t seed, const size_t index)
		{
			if (isnan(value)) return 0x07U;
			const uint8_t sign = (__float_as_uint(value) >> 31U) != 0U ? 0x08U : 0U;
			const float magnitude = isinf(value) ? 6.0f : fminf(fabsf(value), 6.0f);
			if (magnitude >= 6.0f) return static_cast<uint8_t>(sign | 0x07U);
			unsigned upper = 1U;
			while (fp4_positive_values[upper] < magnitude) ++upper;
			const unsigned lower = upper - 1U;
			const float probability_upper = (magnitude - fp4_positive_values[lower]) /
				(fp4_positive_values[upper] - fp4_positive_values[lower]);
			const uint64_t bits = fp4_splitmix64(seed ^ (static_cast<uint64_t>(index) * 0x9e3779b97f4a7c15ULL));
			const float uniform = static_cast<float>((bits >> 40U) & 0xffffffU) * (1.0f / 16777216.0f);
			return static_cast<uint8_t>(sign | (uniform < probability_upper ? upper : lower));
		}

		template<bool stochastic>
		__global__ void fp4_pack_kernel(const float * input, const size_t count, const uint64_t seed, uint8_t * packed)
		{
			const size_t byte_index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
			const size_t first = byte_index * 2U;
			if (first >= count) return;
			const uint8_t low = stochastic ? fp4_encode_stochastic(input[first], seed, first) : fp4_encode_nearest(input[first]);
			uint8_t high = 0U;
			if (first + 1U < count)
				high = stochastic ? fp4_encode_stochastic(input[first + 1U], seed, first + 1U) : fp4_encode_nearest(input[first + 1U]);
			packed[byte_index] = static_cast<uint8_t>(low | (high << 4U));
		}

		template<bool stochastic>
		void launch_pack(const float * input, const size_t count, const uint64_t seed, uint8_t * packed)
		{
			if (count == 0U) return;
			if (input == nullptr || packed == nullptr) darknet_fatal_error(DARKNET_LOC, "FP4 pack received a null GPU pointer");
			constexpr unsigned threads = 256U;
			const size_t bytes = (count + 1U) / 2U;
			const unsigned blocks = static_cast<unsigned>((bytes + threads - 1U) / threads);
			fp4_pack_kernel<stochastic><<<blocks, threads, 0, get_cuda_stream()>>>(input, count, seed, packed);
			CHECK_CUDA(cudaPeekAtLastError());
		}

		__global__ void transpose_rowmajor_kernel(const float * input, const int rows, const int columns, float * output)
		{
			const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
			const size_t count = static_cast<size_t>(rows) * columns;
			if (index >= count) return;
			const int row = static_cast<int>(index / columns);
			const int column = static_cast<int>(index % columns);
			output[static_cast<size_t>(column) * rows + row] = input[index];
		}

		__global__ void copy_matrix_columns_kernel(const float * input, const int rows, const int columns,
			const int output_columns, const int column_offset, float * output)
		{
			const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
			const size_t count = static_cast<size_t>(rows) * columns;
			if (index >= count) return;
			const int row = static_cast<int>(index / columns);
			const int column = static_cast<int>(index % columns);
			output[static_cast<size_t>(row) * output_columns + column_offset + column] = input[index];
		}

		__global__ void pack_batch_rows_kernel(const float * input, const int batch, const int rows,
			const int columns, float * output)
		{
			const size_t matrix = static_cast<size_t>(rows) * columns;
			const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
			const size_t count = matrix * batch;
			if (index >= count) return;
			const int b = static_cast<int>(index / matrix);
			const size_t rem = index % matrix;
			const int row = static_cast<int>(rem / columns);
			const int column = static_cast<int>(rem % columns);
			output[static_cast<size_t>(row) * batch * columns + static_cast<size_t>(b) * columns + column] = input[index];
		}

		constexpr size_t round_up(const size_t value, const size_t multiple)
		{
			return (value + multiple - 1U) / multiple * multiple;
		}

		__device__ size_t fp4_scale_offset(const int outer, const int reduction_block,
			const int scale_inner_tiles)
		{
			// cuBLASLt stores 128 outer rows x 4 K-block scales in a 512-byte
			// tile.  Padding is mandatory even when M/N or K has a partial tile.
			const int tile_outer = outer / 128;
			const int tile_inner = reduction_block / 4;
			const int local_outer = outer % 128;
			const int local_inner = reduction_block % 4;
			const size_t tile = static_cast<size_t>(tile_outer * scale_inner_tiles + tile_inner) * 512U;
			return tile + static_cast<size_t>(local_outer % 32) * 16U +
				static_cast<size_t>(local_outer / 32) * 4U + local_inner;
		}

		// One warp quantizes one 16-element NVFP4 block; kWarpsPerBlock warps are
		// batched per CTA so blocks aren't launched for a single 64B transaction.
		// Used by both the host launch config and the kernel body below -- keep
		// this the single source of truth for that factor, never duplicate it.
		constexpr int kQuantizeWarpsPerCta = 4;

		__global__ void quantize_cublaslt_kernel(const float * __restrict__ input, const int reduction,
			uint8_t * packed, uint8_t * scales, const int reduction_blocks,
			const int scale_inner_tiles, const int reduction_block_groups)
		{
			// Flatten row and K-block-group into x: convolution spatial dimensions
			// can exceed CUDA's legacy 65,535 y-grid limit (for example 416x416).
			const int warp_id = static_cast<int>(threadIdx.x) >> 5;
			const int lane = static_cast<int>(threadIdx.x) & 31;
			const int flattened_block = static_cast<int>(blockIdx.x);
			const int group = flattened_block % reduction_block_groups;
			const int row = flattened_block / reduction_block_groups;
			const int reduction_block = group * kQuantizeWarpsPerCta + warp_id;
			if (reduction_block >= reduction_blocks) return;

			const int k = reduction_block * 16 + lane;
			float value = lane < 16 && k < reduction ? input[static_cast<size_t>(row) * reduction + k] : 0.0f;
			float amax = lane < 16 && isfinite(value) ? fabsf(value) : 0.0f;
			for (int offset = 16; offset > 0; offset /= 2) amax = fmaxf(amax, __shfl_down_sync(0xffffffffU, amax, offset));
			amax = __shfl_sync(0xffffffffU, amax, 0);

			// NVFP4 stores a multiplicative dequantization scale.  Quantized
			// E2M1 values therefore divide by the CUDA-RNE UE4M3 scale here, and
			// cuBLASLt multiplies the same scale back during tensor-core GEMM.
			// 2^-9 is the smallest positive E4M3 subnormal.  Clamping avoids a
			// tiny nonzero block scale rounding to zero before division.
			const float requested_scale = amax > 0.0f ? fmaxf(amax / 6.0f, 0.001953125f) : 1.0f;
			// CUDA_R_8F_UE4M3 uses the positive E4M3 encoding (the sign bit is
			// ignored by cuBLASLt), so CUDA's saturating E4M3 conversion is the
			// required producer for a non-negative scale byte.
			const __nv_fp8_storage_t encoded_scale = __nv_cvt_float_to_fp8(
				requested_scale, __NV_SATFINITE, __NV_E4M3);
			__nv_fp8_e4m3 decoded_scale_fp8(0.0f);
			decoded_scale_fp8.__x = static_cast<uint8_t>(encoded_scale);
			const float decoded_scale = static_cast<float>(decoded_scale_fp8);
			if (lane == 0)
				scales[fp4_scale_offset(row, reduction_block, scale_inner_tiles)] = static_cast<uint8_t>(encoded_scale);

			// Reuse the phase-1 register value instead of re-reading global
			// memory: even lane L<16 already holds k=block*16+L ("first"); its
			// neighbor odd lane holds "second". Must run unconditionally across
			// the full warp -- never nested inside a lane guard.
			const float neighbor = __shfl_down_sync(0xffffffffU, value, 1);
			if (lane < 16 && (lane & 1) == 0)
			{
				const int first_k = reduction_block * 16 + lane;
				const size_t byte = (static_cast<size_t>(row) * reduction + first_k) / 2U;
				const float first = value / decoded_scale;
				const float second = neighbor / decoded_scale;
				// Use CUDA's NVFP4 conversion rather than duplicating the E2M1
				// rounding/packing rules locally.  Its packed-byte lane order is the
				// one defined by cuda_fp4.h and consumed by cuBLASLt.
				const __nv_fp4x2_storage_t pair = __nv_cvt_float2_to_fp4x2(
					make_float2(first, second), __NV_E2M1, cudaRoundNearest);
				packed[byte] = static_cast<uint8_t>(pair);
			}
		}

		__global__ void quantize_nchw_to_cublaslt_kernel(const float * __restrict__ input, const int channels,
			const int spatial, const int reduction_blocks, uint8_t * packed, uint8_t * scales,
			const size_t packed_bytes_per_image, const size_t scale_bytes_per_image,
			const int scale_inner_tiles, const bool fuse_relu, const int reduction_block_groups)
		{
			const int warp_id = static_cast<int>(threadIdx.x) >> 5;
			const int lane = static_cast<int>(threadIdx.x) & 31;
			const int image = static_cast<int>(blockIdx.y);
			const int flattened_block = static_cast<int>(blockIdx.x);
			const int group = flattened_block % reduction_block_groups;
			const int row = flattened_block / reduction_block_groups;
			const int reduction_block = group * kQuantizeWarpsPerCta + warp_id;
			if (reduction_block >= reduction_blocks) return;

			const int channel = reduction_block * 16 + lane;
			const size_t image_input = static_cast<size_t>(image) * channels * spatial;
			float value = lane < 16 && channel < channels
				? input[image_input + static_cast<size_t>(channel) * spatial + row]
				: 0.0f;
			if (fuse_relu && lane < 16 && channel < channels)
			{
				value = isfinite(value) && value > 0.0f ? value : 0.0f;
				const_cast<float *>(input)[image_input + static_cast<size_t>(channel) * spatial + row] = value;
			}
			if (fuse_relu)
			{
				__syncwarp();
			}
			float amax = lane < 16 && isfinite(value) ? fabsf(value) : 0.0f;
			for (int offset = 16; offset > 0; offset /= 2) amax = fmaxf(amax, __shfl_down_sync(0xffffffffU, amax, offset));
			amax = __shfl_sync(0xffffffffU, amax, 0);

			const float requested_scale = amax > 0.0f ? fmaxf(amax / 6.0f, 0.001953125f) : 1.0f;
			const __nv_fp8_storage_t encoded_scale = __nv_cvt_float_to_fp8(
				requested_scale, __NV_SATFINITE, __NV_E4M3);
			__nv_fp8_e4m3 decoded_scale_fp8(0.0f);
			decoded_scale_fp8.__x = static_cast<uint8_t>(encoded_scale);
			const float decoded_scale = static_cast<float>(decoded_scale_fp8);
			uint8_t * const image_scales = scales + static_cast<size_t>(image) * scale_bytes_per_image;
			uint8_t * const image_packed = packed + static_cast<size_t>(image) * packed_bytes_per_image;
			if (lane == 0)
				image_scales[fp4_scale_offset(row, reduction_block, scale_inner_tiles)] = static_cast<uint8_t>(encoded_scale);

			const float neighbor = __shfl_down_sync(0xffffffffU, value, 1);
			if (lane < 16 && (lane & 1) == 0)
			{
				const int first_channel = reduction_block * 16 + lane;
				const float first = value / decoded_scale;
				const float second = neighbor / decoded_scale;
				const __nv_fp4x2_storage_t pair = __nv_cvt_float2_to_fp4x2(
					make_float2(first, second), __NV_E2M1, cudaRoundNearest);
				image_packed[(static_cast<size_t>(row) * channels + first_channel) / 2U] = static_cast<uint8_t>(pair);
			}
		}

		__global__ void fp4_clear_amax_kernel(float * amax_gpu)
		{
			if (threadIdx.x == 0 && blockIdx.x == 0)
			{
				*amax_gpu = 0.0f;
			}
		}

		__global__ void fp4_amax_kernel(const float * src, size_t count, float * amax_gpu)
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
			for (unsigned int stride = blockDim.x / 2U; stride > 0U; stride >>= 1U)
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

		dim3 fp4_gridsize_256(const size_t n)
		{
			constexpr size_t threads = 256;
			size_t k = (n - 1) / threads + 1;
			size_t x = k;
			size_t y = 1;
			if (x > 65535)
			{
				x = static_cast<size_t>(std::ceil(std::sqrt(static_cast<double>(k))));
				y = (k - 1) / x + 1;
			}
			return dim3(static_cast<unsigned>(x), static_cast<unsigned>(y), 1);
		}
	}

	void fp4_clear_amax_gpu(float * amax_gpu)
	{
		TAT(TATPARMS);

		if (amax_gpu == nullptr)
		{
			return;
		}
		fp4_clear_amax_kernel<<<1, 1, 0, get_cuda_stream()>>>(amax_gpu);
		CHECK_CUDA(cudaPeekAtLastError());
	}

	void fp4_accumulate_amax_gpu(const float * src, const size_t count, float * amax_gpu)
	{
		TAT(TATPARMS);

		if (src == nullptr || amax_gpu == nullptr || count == 0)
		{
			return;
		}
		fp4_amax_kernel<<<fp4_gridsize_256(count), 256, 0, get_cuda_stream()>>>(src, count, amax_gpu);
		CHECK_CUDA(cudaPeekAtLastError());
	}

	float fp4_pull_amax_gpu(float * amax_gpu)
	{
		float amax = 0.0f;
		if (amax_gpu)
		{
			CHECK_CUDA(cudaMemcpyAsync(&amax, amax_gpu, sizeof(float), cudaMemcpyDeviceToHost, get_cuda_stream()));
			CHECK_CUDA(cudaStreamSynchronize(get_cuda_stream()));
		}
		return amax;
	}

	void fp4_pack_e2m1_gpu(const float * input, const size_t count, uint8_t * packed)
	{
		TAT(TATPARMS);

		launch_pack<false>(input, count, 0U, packed);
	}

	void fp4_pack_e2m1_stochastic_gpu(const float * input, const size_t count, const uint64_t seed, uint8_t * packed)
	{
		TAT(TATPARMS);

		launch_pack<true>(input, count, seed, packed);
	}

	void fp4_transpose_rowmajor_gpu(const float * input, const int rows, const int columns, float * output)
	{
		TAT(TATPARMS);

		if (!input || !output || rows <= 0 || columns <= 0) return;
		const size_t count = static_cast<size_t>(rows) * columns;
		transpose_rowmajor_kernel<<<static_cast<unsigned>((count + 255U) / 256U), 256, 0, get_cuda_stream()>>>(input, rows, columns, output);
		CHECK_CUDA(cudaPeekAtLastError());
	}

	void fp4_copy_matrix_columns_gpu(const float * input, const int rows, const int columns,
		const int output_columns, const int column_offset, float * output)
	{
		TAT(TATPARMS);

		if (!input || !output || rows <= 0 || columns <= 0 || output_columns < columns ||
			column_offset < 0 || column_offset + columns > output_columns) return;
		const size_t count = static_cast<size_t>(rows) * columns;
		copy_matrix_columns_kernel<<<static_cast<unsigned>((count + 255U) / 256U), 256, 0, get_cuda_stream()>>>(
			input, rows, columns, output_columns, column_offset, output);
		CHECK_CUDA(cudaPeekAtLastError());
	}

	void fp4_pack_batch_rows_gpu(const float * input, const int batch, const int rows, const int columns, float * output)
	{
		TAT(TATPARMS);

		if (!input || !output || batch <= 0 || rows <= 0 || columns <= 0) return;
		const size_t count = static_cast<size_t>(batch) * rows * columns;
		pack_batch_rows_kernel<<<static_cast<unsigned>((count + 255U) / 256U), 256, 0, get_cuda_stream()>>>(input, batch, rows, columns, output);
		CHECK_CUDA(cudaPeekAtLastError());
	}

	size_t fp4_cublaslt_packed_bytes(const int outer, const int reduction)
	{
		if (outer <= 0 || reduction <= 0) return 0U;
		return (static_cast<size_t>(outer) * reduction + 1U) / 2U;
	}

	size_t fp4_cublaslt_scale_bytes(const int outer, const int reduction)
	{
		if (outer <= 0 || reduction <= 0) return 0U;
		const size_t outer_tiles = round_up(static_cast<size_t>(outer), 128U) / 128U;
		const size_t inner_tiles = round_up((static_cast<size_t>(reduction) + 15U) / 16U, 4U) / 4U;
		return outer_tiles * inner_tiles * 512U;
	}

	bool fp4_quantize_cublaslt_gpu(const float * input, const int outer, const int reduction,
		uint8_t * packed, uint8_t * scales)
	{
		TAT(TATPARMS);

		// Each CUDA block packs exactly one 16-element NVFP4 scale block.  The
		// direct cuBLASLt plan also requires this granularity, so reject ragged K
		// rather than writing padded nibbles past a tightly packed row.
		if (!input || !packed || !scales || outer <= 0 || reduction <= 0 || (reduction % 16) != 0) return false;
		const size_t scale_bytes = fp4_cublaslt_scale_bytes(outer, reduction);
		CHECK_CUDA(cudaMemsetAsync(scales, 0, scale_bytes, get_cuda_stream()));
		const int reduction_blocks = (reduction + 15) / 16;
		const int scale_inner_tiles = (reduction_blocks + 3) / 4;
		const int reduction_block_groups = (reduction_blocks + kQuantizeWarpsPerCta - 1) / kQuantizeWarpsPerCta;
		const size_t launch_blocks = static_cast<size_t>(reduction_block_groups) * static_cast<size_t>(outer);
		if (launch_blocks > static_cast<size_t>(std::numeric_limits<int>::max())) return false;
		quantize_cublaslt_kernel<<<static_cast<unsigned>(launch_blocks), 32 * kQuantizeWarpsPerCta, 0,
			get_cuda_stream()>>>(input, reduction, packed, scales, reduction_blocks, scale_inner_tiles,
			reduction_block_groups);
		return cudaPeekAtLastError() == cudaSuccess;
	}

	bool fp4_quantize_nchw_to_cublaslt_gpu(const float * input, const int batch, const int channels,
		const int height, const int width, uint8_t * packed, uint8_t * scales)
	{
		TAT(TATPARMS);

		if (!input || !packed || !scales || batch <= 0 || channels <= 0 || height <= 0 || width <= 0 ||
			(channels % 16) != 0)
		{
			return false;
		}
		const int spatial = height * width;
		const int reduction_blocks = (channels + 15) / 16;
		const int scale_inner_tiles = (reduction_blocks + 3) / 4;
		const int reduction_block_groups = (reduction_blocks + kQuantizeWarpsPerCta - 1) / kQuantizeWarpsPerCta;
		const size_t packed_bytes_per_image = fp4_cublaslt_packed_bytes(spatial, channels);
		const size_t scale_bytes_per_image = fp4_cublaslt_scale_bytes(spatial, channels);
		CHECK_CUDA(cudaMemsetAsync(scales, 0, scale_bytes_per_image * static_cast<size_t>(batch), get_cuda_stream()));
		quantize_nchw_to_cublaslt_kernel<<<dim3(static_cast<unsigned>(reduction_block_groups * spatial), static_cast<unsigned>(batch)),
			32 * kQuantizeWarpsPerCta, 0, get_cuda_stream()>>>(input, channels, spatial, reduction_blocks, packed, scales,
			packed_bytes_per_image, scale_bytes_per_image, scale_inner_tiles, false, reduction_block_groups);
		return cudaPeekAtLastError() == cudaSuccess;
	}

	bool fp4_relu_quantize_nchw_to_cublaslt_gpu(float * input, const int batch, const int channels,
		const int height, const int width, uint8_t * packed, uint8_t * scales)
	{
		TAT(TATPARMS);

		if (!input || !packed || !scales || batch <= 0 || channels <= 0 || height <= 0 || width <= 0 ||
			(channels % 16) != 0)
		{
			return false;
		}
		const int spatial = height * width;
		const int reduction_blocks = (channels + 15) / 16;
		const int scale_inner_tiles = (reduction_blocks + 3) / 4;
		const int reduction_block_groups = (reduction_blocks + kQuantizeWarpsPerCta - 1) / kQuantizeWarpsPerCta;
		const size_t packed_bytes_per_image = fp4_cublaslt_packed_bytes(spatial, channels);
		const size_t scale_bytes_per_image = fp4_cublaslt_scale_bytes(spatial, channels);
		CHECK_CUDA(cudaMemsetAsync(scales, 0, scale_bytes_per_image * static_cast<size_t>(batch), get_cuda_stream()));
		quantize_nchw_to_cublaslt_kernel<<<dim3(static_cast<unsigned>(reduction_block_groups * spatial), static_cast<unsigned>(batch)),
			32 * kQuantizeWarpsPerCta, 0, get_cuda_stream()>>>(input, channels, spatial, reduction_blocks, packed, scales,
			packed_bytes_per_image, scale_bytes_per_image, scale_inner_tiles, true, reduction_block_groups);
		return cudaPeekAtLastError() == cudaSuccess;
	}
}
