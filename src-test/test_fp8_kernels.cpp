#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <vector>

#include "darknet_internal.hpp"
#include "fp8_kernels.hpp"
#include "im2col.hpp"

namespace
{
	unsigned short bf16_storage_bits(const float value)
	{
		static_assert(sizeof(float) == sizeof(uint32_t), "unexpected float storage");
		uint32_t bits = 0;
		std::memcpy(&bits, &value, sizeof(bits));
		return static_cast<unsigned short>(bits >> 16);
	}

	float bf16_storage_to_float(const unsigned short value)
	{
		uint32_t bits = static_cast<uint32_t>(value) << 16;
		float result = 0.0f;
		std::memcpy(&result, &bits, sizeof(result));
		return result;
	}

	std::vector<float> reference_col2im_accumulate(
		const std::vector<float> & initial_delta,
		const std::vector<float> & col_rowmajor,
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
		const int dilation_w)
	{
		std::vector<float> expected = initial_delta;
		const int height_col = (height + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
		const int width_col = (width + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
		const int kernel = channels * kernel_h * kernel_w;
		const int spatial = height_col * width_col;
		for (int b = 0; b < batch; ++b)
		{
			for (int c = 0; c < channels; ++c)
			{
				for (int h = 0; h < height; ++h)
				{
					for (int w = 0; w < width; ++w)
					{
						float value = expected[static_cast<size_t>(((b * channels + c) * height + h) * width + w)];
						const int h_im = h + pad_h;
						const int w_im = w + pad_w;
						for (int h_col = 0; h_col < height_col; ++h_col)
						{
							const int h_k = h_im - h_col * stride_h;
							if (h_k < 0 || h_k >= kernel_h * dilation_h || h_k % dilation_h != 0)
							{
								continue;
							}
							for (int w_col = 0; w_col < width_col; ++w_col)
							{
								const int w_k = w_im - w_col * stride_w;
								if (w_k < 0 || w_k >= kernel_w * dilation_w || w_k % dilation_w != 0)
								{
									continue;
								}
								const int kernel_index = (c * kernel_h + h_k / dilation_h) * kernel_w + w_k / dilation_w;
								const int spatial_index = h_col * width_col + w_col;
								value += col_rowmajor[static_cast<size_t>(b) * kernel * spatial +
									static_cast<size_t>(kernel_index) * spatial + spatial_index];
							}
						}
						expected[static_cast<size_t>(((b * channels + c) * height + h) * width + w)] = value;
					}
				}
			}
		}
		return expected;
	}
}

TEST(Fp8Kernels, TensorBytesAreOneBytePerElement)
{
	EXPECT_EQ(Darknet::fp8_tensor_bytes(0), 0);
	EXPECT_EQ(Darknet::fp8_tensor_bytes(1), 1);
	EXPECT_EQ(Darknet::fp8_tensor_bytes(17), 17);
}

TEST(Fp8Kernels, PaddedRowMajorBytesMatchOneByteElements)
{
	EXPECT_EQ(Darknet::fp8_rowmajor_pad_cols_bytes(3, 16), 48);
	EXPECT_EQ(Darknet::fp8_rowmajor_pad_rows_bytes(16, 3), 48);
	EXPECT_EQ(Darknet::fp8_rowmajor_pad_cols_bytes(0, 16), 0);
	EXPECT_EQ(Darknet::fp8_rowmajor_pad_rows_bytes(16, 0), 0);
}

TEST(Fp8Kernels, TransposedPadColsQuantizeApiIsLinked)
{
	const auto fn = &Darknet::fp8_quantize_transpose_rowmajor_pad_cols_gpu;
	EXPECT_NE(fn, nullptr);
}

TEST(Fp8Kernels, WgradQuantizeApisAreLinked)
{
	const auto dy_fn = &Darknet::fp8_quantize_e5m2_rowmajor_pad_cols_record_amax_gpu;
	const auto im2col_fn = &Darknet::fp8_quantize_transpose_rowmajor_pad_rows_gpu;

	EXPECT_NE(dy_fn, nullptr);
	EXPECT_NE(im2col_fn, nullptr);
}

TEST(Fp8Kernels, DgradQuantizeApisAreLinked)
{
	const auto dy_fn = &Darknet::fp8_quantize_e5m2_rowmajor_pad_rows_record_amax_gpu;
	const auto weights_t_fn = &Darknet::fp8_quantize_transpose_rowmajor_pad_cols_gpu;

	EXPECT_NE(dy_fn, nullptr);
	EXPECT_NE(weights_t_fn, nullptr);
}

TEST(Fp8Kernels, Im2colQuantizeKindSelectsCommonFastPaths)
{
	EXPECT_EQ(
		Darknet::fp8_im2col_quantize_kind(3, 3, 1, 1, 1, 1, 1, 1),
		Darknet::Fp8Im2colQuantizeKind::Conv3x3Pad1Stride1);
	EXPECT_EQ(
		Darknet::fp8_im2col_quantize_kind(3, 3, 1, 1, 2, 2, 1, 1),
		Darknet::Fp8Im2colQuantizeKind::Conv3x3Pad1Stride2);
	EXPECT_EQ(
		Darknet::fp8_im2col_quantize_kind(3, 3, 1, 1, 1, 2, 1, 1),
		Darknet::Fp8Im2colQuantizeKind::Generic);
	EXPECT_EQ(
		Darknet::fp8_im2col_quantize_kind(3, 3, 1, 1, 1, 1, 2, 2),
		Darknet::Fp8Im2colQuantizeKind::Generic);
	EXPECT_EQ(
		Darknet::fp8_im2col_quantize_kind(1, 1, 0, 0, 1, 1, 1, 1),
		Darknet::Fp8Im2colQuantizeKind::Generic);
}

TEST(Fp8Kernels, DgradEpilogueKindSelectsCommonFastPaths)
{
	EXPECT_EQ(
		Darknet::fp8_dgrad_epilogue_kind(1, 1, 0, 0, 1, 1, 1, 1, 7, 5, 7, 5),
		Darknet::Fp8DgradEpilogueKind::Direct1x1);
	EXPECT_EQ(
		Darknet::fp8_dgrad_epilogue_kind(3, 3, 1, 1, 1, 1, 1, 1, 7, 5, 7, 5),
		Darknet::Fp8DgradEpilogueKind::Conv3x3Stride1Pad1);
	EXPECT_EQ(
		Darknet::fp8_dgrad_epilogue_kind(3, 3, 1, 1, 2, 2, 1, 1, 7, 5, 4, 3),
		Darknet::Fp8DgradEpilogueKind::Generic);
	EXPECT_EQ(
		Darknet::fp8_dgrad_epilogue_kind(3, 3, 1, 1, 1, 1, 2, 2, 7, 5, 7, 5),
		Darknet::Fp8DgradEpilogueKind::Generic);
	EXPECT_EQ(
		Darknet::fp8_dgrad_epilogue_kind(1, 1, 0, 0, 1, 1, 1, 1, 7, 5, 6, 5),
		Darknet::Fp8DgradEpilogueKind::Generic);
}

TEST(Fp8Kernels, E4m3RecordAmaxApisAreLinked)
{
	const auto pad_cols_fn = &Darknet::fp8_quantize_rowmajor_pad_cols_record_amax_gpu;
	const auto pad_rows_fn = &Darknet::fp8_quantize_rowmajor_pad_rows_record_amax_gpu;
	const auto transpose_pad_cols_fn = &Darknet::fp8_quantize_transpose_rowmajor_pad_cols_record_amax_gpu;
	const auto transpose_pad_rows_fn = &Darknet::fp8_quantize_transpose_rowmajor_pad_rows_record_amax_gpu;

	EXPECT_NE(pad_cols_fn, nullptr);
	EXPECT_NE(pad_rows_fn, nullptr);
	EXPECT_NE(transpose_pad_cols_fn, nullptr);
	EXPECT_NE(transpose_pad_rows_fn, nullptr);
}

TEST(Fp8Kernels, E5m2QuantizeAndAmaxCoverTailPastOneBlock)
{
#if defined(DARKNET_GPU_CUDA) && defined(CUDART_VERSION) && CUDART_VERSION >= 12010
	constexpr size_t count = 200003;
	std::vector<float> input(count, 1.0f);
	input.back() = 128.0f;
	float scale = 1.0f;
	float * input_gpu = cuda_make_array(input.data(), input.size());
	float * scale_gpu = cuda_make_array(&scale, 1);
	float * amax_gpu = cuda_make_array(nullptr, 1);
	void * fp8_gpu = nullptr;
	CHECK_CUDA(cudaMalloc(&fp8_gpu, Darknet::fp8_tensor_bytes(input.size())));
	CHECK_CUDA(cudaMemsetAsync(fp8_gpu, 0, Darknet::fp8_tensor_bytes(input.size()), get_cuda_stream()));

	Darknet::fp8_clear_amax_gpu(amax_gpu);
	Darknet::fp8_quantize_e5m2_record_amax_gpu(input_gpu, input.size(), scale_gpu, fp8_gpu, amax_gpu);
	const float amax = Darknet::fp8_pull_amax_gpu(amax_gpu);
	std::vector<unsigned char> fp8_bytes(input.size());
	CHECK_CUDA(cudaMemcpyAsync(fp8_bytes.data(), fp8_gpu, fp8_bytes.size(), cudaMemcpyDeviceToHost, get_cuda_stream()));
	CHECK_CUDA(cudaStreamSynchronize(get_cuda_stream()));

	EXPECT_FLOAT_EQ(amax, 128.0f);
	EXPECT_NE(fp8_bytes[256], 0);
	EXPECT_NE(fp8_bytes.back(), 0);

	CHECK_CUDA(cudaFree(fp8_gpu));
	cuda_free(amax_gpu);
	cuda_free(scale_gpu);
	cuda_free(input_gpu);
#else
	GTEST_SKIP() << "CUDA FP8 kernel test requires CUDA 12.1+";
#endif
}

TEST(Fp8Kernels, E4m3PadRowsQuantizeAndAmaxCoverLargeTail)
{
#if defined(DARKNET_GPU_CUDA) && defined(CUDART_VERSION) && CUDART_VERSION >= 12010
	constexpr int rows = 1001;
	constexpr int cols = 203;
	constexpr int rows_pad = 1008;
	constexpr size_t count = static_cast<size_t>(rows) * cols;
	constexpr size_t bytes = static_cast<size_t>(rows_pad) * cols;
	static_assert(count >= 200000, "large-count FP8 test must cover the reviewed launch-grid class");
	std::vector<float> input(count, 1.0f);
	input.back() = 240.0f;
	float scale = 1.0f;
	float * input_gpu = cuda_make_array(input.data(), input.size());
	float * scale_gpu = cuda_make_array(&scale, 1);
	float * amax_gpu = cuda_make_array(nullptr, 1);
	void * fp8_gpu = nullptr;
	CHECK_CUDA(cudaMalloc(&fp8_gpu, bytes));
	CHECK_CUDA(cudaMemsetAsync(fp8_gpu, 0, bytes, get_cuda_stream()));

	Darknet::fp8_clear_amax_gpu(amax_gpu);
	Darknet::fp8_quantize_rowmajor_pad_rows_record_amax_gpu(
		input_gpu, rows, cols, rows_pad, scale_gpu, fp8_gpu, amax_gpu);
	const float amax = Darknet::fp8_pull_amax_gpu(amax_gpu);
	std::vector<unsigned char> fp8_bytes(bytes);
	CHECK_CUDA(cudaMemcpyAsync(fp8_bytes.data(), fp8_gpu, fp8_bytes.size(), cudaMemcpyDeviceToHost, get_cuda_stream()));
	CHECK_CUDA(cudaStreamSynchronize(get_cuda_stream()));

	EXPECT_FLOAT_EQ(amax, 240.0f);
	EXPECT_NE(fp8_bytes[count - 1], 0);
	EXPECT_EQ(fp8_bytes[count], 0);

	CHECK_CUDA(cudaFree(fp8_gpu));
	cuda_free(amax_gpu);
	cuda_free(scale_gpu);
	cuda_free(input_gpu);
#else
	GTEST_SKIP() << "CUDA FP8 kernel test requires CUDA 12.1+";
#endif
}

TEST(Fp8Kernels, E5m2PadRowsColsQuantizeCoversBothPaddingAxes)
{
#if defined(DARKNET_GPU_CUDA) && defined(CUDART_VERSION) && CUDART_VERSION >= 12010
	constexpr int rows = 17;
	constexpr int cols = 19;
	constexpr int rows_pad = 32;
	constexpr int cols_pad = 32;
	constexpr size_t count = static_cast<size_t>(rows) * cols;
	constexpr size_t bytes = static_cast<size_t>(rows_pad) * cols_pad;
	std::vector<float> input(count, 1.0f);
	input.back() = 512.0f;
	float scale = 4.0f;
	float * input_gpu = cuda_make_array(input.data(), input.size());
	float * scale_gpu = cuda_make_array(&scale, 1);
	float * amax_gpu = cuda_make_array(nullptr, 1);
	void * fp8_gpu = nullptr;
	CHECK_CUDA(cudaMalloc(&fp8_gpu, bytes));
	CHECK_CUDA(cudaMemsetAsync(fp8_gpu, 0x7f, bytes, get_cuda_stream()));

	Darknet::fp8_clear_amax_gpu(amax_gpu);
	Darknet::fp8_quantize_e5m2_rowmajor_pad_rows_cols_record_amax_gpu(
		input_gpu, rows, cols, rows_pad, cols_pad, scale_gpu, fp8_gpu, amax_gpu);
	const float amax = Darknet::fp8_pull_amax_gpu(amax_gpu);
	std::vector<unsigned char> fp8_bytes(bytes);
	CHECK_CUDA(cudaMemcpyAsync(fp8_bytes.data(), fp8_gpu, fp8_bytes.size(), cudaMemcpyDeviceToHost, get_cuda_stream()));
	CHECK_CUDA(cudaStreamSynchronize(get_cuda_stream()));

	EXPECT_FLOAT_EQ(amax, 512.0f);
	EXPECT_NE(fp8_bytes[static_cast<size_t>(rows - 1) * cols_pad + cols - 1], 0);
	EXPECT_EQ(fp8_bytes[static_cast<size_t>(rows - 1) * cols_pad + cols], 0);
	EXPECT_EQ(fp8_bytes[static_cast<size_t>(rows) * cols_pad], 0);

	CHECK_CUDA(cudaFree(fp8_gpu));
	cuda_free(amax_gpu);
	cuda_free(scale_gpu);
	cuda_free(input_gpu);
#else
	GTEST_SKIP() << "CUDA FP8 kernel test requires CUDA 12.1+";
#endif
}

TEST(Fp8Kernels, FusedIm2colQuantizeMatchesReferenceBytes)
{
#if defined(DARKNET_GPU_CUDA) && defined(CUDART_VERSION) && CUDART_VERSION >= 12010
	constexpr int channels = 2;
	constexpr int height = 5;
	constexpr int width = 4;
	constexpr int kernel_h = 3;
	constexpr int kernel_w = 3;
	constexpr int pad_h = 1;
	constexpr int pad_w = 1;
	constexpr int stride_h = 1;
	constexpr int stride_w = 1;
	constexpr int dilation_h = 1;
	constexpr int dilation_w = 1;
	constexpr int height_col = (height + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
	constexpr int width_col = (width + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
	constexpr int rows = channels * kernel_h * kernel_w;
	constexpr int cols = height_col * width_col;
	constexpr int rows_pad = 32;
	constexpr size_t input_count = static_cast<size_t>(channels) * height * width;
	constexpr size_t im2col_count = static_cast<size_t>(rows) * cols;
	constexpr size_t fp8_bytes = static_cast<size_t>(rows_pad) * cols;

	std::vector<float> input(input_count);
	for (size_t idx = 0; idx < input.size(); ++idx)
	{
		input[idx] = static_cast<float>(static_cast<int>(idx % 17) - 8) * 0.25f;
	}
	input.back() = 6.5f;
	float scale = 0.5f;

	float * input_gpu = cuda_make_array(input.data(), input.size());
	float * im2col_gpu = cuda_make_array(nullptr, im2col_count);
	float * scale_gpu = cuda_make_array(&scale, 1);
	float * reference_amax_gpu = cuda_make_array(nullptr, 1);
	float * fused_amax_gpu = cuda_make_array(nullptr, 1);
	void * reference_fp8_gpu = nullptr;
	void * fused_fp8_gpu = nullptr;
	CHECK_CUDA(cudaMalloc(&reference_fp8_gpu, fp8_bytes));
	CHECK_CUDA(cudaMalloc(&fused_fp8_gpu, fp8_bytes));
	CHECK_CUDA(cudaMemsetAsync(reference_fp8_gpu, 0, fp8_bytes, get_cuda_stream()));
	CHECK_CUDA(cudaMemsetAsync(fused_fp8_gpu, 0, fp8_bytes, get_cuda_stream()));

	im2col_gpu_ext(
		input_gpu,
		channels,
		height, width,
		kernel_h, kernel_w,
		pad_h, pad_w,
		stride_h, stride_w,
		dilation_h, dilation_w,
		im2col_gpu);
	Darknet::fp8_clear_amax_gpu(reference_amax_gpu);
	Darknet::fp8_quantize_rowmajor_pad_rows_record_amax_gpu(
		im2col_gpu, rows, cols, rows_pad, scale_gpu, reference_fp8_gpu, reference_amax_gpu);
	Darknet::fp8_clear_amax_gpu(fused_amax_gpu);
	Darknet::fp8_im2col_quantize_rowmajor_pad_rows_record_amax_gpu(
		input_gpu,
		channels,
		height, width,
		kernel_h, kernel_w,
		pad_h, pad_w,
		stride_h, stride_w,
		dilation_h, dilation_w,
		rows_pad,
		scale_gpu,
		fused_fp8_gpu,
		fused_amax_gpu);

	const float reference_amax = Darknet::fp8_pull_amax_gpu(reference_amax_gpu);
	const float fused_amax = Darknet::fp8_pull_amax_gpu(fused_amax_gpu);
	std::vector<unsigned char> reference_bytes(fp8_bytes);
	std::vector<unsigned char> fused_bytes(fp8_bytes);
	CHECK_CUDA(cudaMemcpyAsync(reference_bytes.data(), reference_fp8_gpu, fp8_bytes, cudaMemcpyDeviceToHost, get_cuda_stream()));
	CHECK_CUDA(cudaMemcpyAsync(fused_bytes.data(), fused_fp8_gpu, fp8_bytes, cudaMemcpyDeviceToHost, get_cuda_stream()));
	CHECK_CUDA(cudaStreamSynchronize(get_cuda_stream()));

	EXPECT_FLOAT_EQ(fused_amax, reference_amax);
	EXPECT_EQ(fused_bytes, reference_bytes);

	CHECK_CUDA(cudaFree(fused_fp8_gpu));
	CHECK_CUDA(cudaFree(reference_fp8_gpu));
	cuda_free(fused_amax_gpu);
	cuda_free(reference_amax_gpu);
	cuda_free(scale_gpu);
	cuda_free(im2col_gpu);
	cuda_free(input_gpu);
#else
	GTEST_SKIP() << "CUDA FP8 kernel test requires CUDA 12.1+";
#endif
}

TEST(Fp8Kernels, FusedIm2colTransposeQuantizeMatchesReferenceBytes)
{
#if defined(DARKNET_GPU_CUDA) && defined(CUDART_VERSION) && CUDART_VERSION >= 12010
	constexpr int channels = 2;
	constexpr int height = 5;
	constexpr int width = 4;
	constexpr int kernel_h = 3;
	constexpr int kernel_w = 3;
	constexpr int pad_h = 1;
	constexpr int pad_w = 1;
	constexpr int stride_h = 1;
	constexpr int stride_w = 1;
	constexpr int dilation_h = 1;
	constexpr int dilation_w = 1;
	constexpr int height_col = (height + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
	constexpr int width_col = (width + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
	constexpr int rows = channels * kernel_h * kernel_w;
	constexpr int cols = height_col * width_col;
	constexpr int cols_pad = 32;
	constexpr size_t input_count = static_cast<size_t>(channels) * height * width;
	constexpr size_t im2col_count = static_cast<size_t>(rows) * cols;
	constexpr size_t fp8_bytes = static_cast<size_t>(cols_pad) * rows;

	std::vector<float> input(input_count);
	for (size_t idx = 0; idx < input.size(); ++idx)
	{
		input[idx] = static_cast<float>(static_cast<int>((idx * 3) % 19) - 9) * 0.125f;
	}
	input[input.size() / 2] = 7.0f;
	float scale = 0.25f;

	float * input_gpu = cuda_make_array(input.data(), input.size());
	float * im2col_gpu = cuda_make_array(nullptr, im2col_count);
	float * scale_gpu = cuda_make_array(&scale, 1);
	float * reference_amax_gpu = cuda_make_array(nullptr, 1);
	float * fused_amax_gpu = cuda_make_array(nullptr, 1);
	void * reference_fp8_gpu = nullptr;
	void * fused_fp8_gpu = nullptr;
	CHECK_CUDA(cudaMalloc(&reference_fp8_gpu, fp8_bytes));
	CHECK_CUDA(cudaMalloc(&fused_fp8_gpu, fp8_bytes));
	CHECK_CUDA(cudaMemsetAsync(reference_fp8_gpu, 0, fp8_bytes, get_cuda_stream()));
	CHECK_CUDA(cudaMemsetAsync(fused_fp8_gpu, 0, fp8_bytes, get_cuda_stream()));

	im2col_gpu_ext(
		input_gpu,
		channels,
		height, width,
		kernel_h, kernel_w,
		pad_h, pad_w,
		stride_h, stride_w,
		dilation_h, dilation_w,
		im2col_gpu);
	Darknet::fp8_clear_amax_gpu(reference_amax_gpu);
	Darknet::fp8_quantize_transpose_rowmajor_pad_rows_record_amax_gpu(
		im2col_gpu, rows, cols, cols_pad, scale_gpu, reference_fp8_gpu, reference_amax_gpu);
	Darknet::fp8_clear_amax_gpu(fused_amax_gpu);
	Darknet::fp8_im2col_quantize_transpose_rowmajor_pad_rows_record_amax_gpu(
		input_gpu,
		channels,
		height, width,
		kernel_h, kernel_w,
		pad_h, pad_w,
		stride_h, stride_w,
		dilation_h, dilation_w,
		cols_pad,
		scale_gpu,
		fused_fp8_gpu,
		fused_amax_gpu);

	const float reference_amax = Darknet::fp8_pull_amax_gpu(reference_amax_gpu);
	const float fused_amax = Darknet::fp8_pull_amax_gpu(fused_amax_gpu);
	std::vector<unsigned char> reference_bytes(fp8_bytes);
	std::vector<unsigned char> fused_bytes(fp8_bytes);
	CHECK_CUDA(cudaMemcpyAsync(reference_bytes.data(), reference_fp8_gpu, fp8_bytes, cudaMemcpyDeviceToHost, get_cuda_stream()));
	CHECK_CUDA(cudaMemcpyAsync(fused_bytes.data(), fused_fp8_gpu, fp8_bytes, cudaMemcpyDeviceToHost, get_cuda_stream()));
	CHECK_CUDA(cudaStreamSynchronize(get_cuda_stream()));

	EXPECT_FLOAT_EQ(fused_amax, reference_amax);
	EXPECT_EQ(fused_bytes, reference_bytes);

	CHECK_CUDA(cudaFree(fused_fp8_gpu));
	CHECK_CUDA(cudaFree(reference_fp8_gpu));
	cuda_free(fused_amax_gpu);
	cuda_free(reference_amax_gpu);
	cuda_free(scale_gpu);
	cuda_free(im2col_gpu);
	cuda_free(input_gpu);
#else
	GTEST_SKIP() << "CUDA FP8 kernel test requires CUDA 12.1+";
#endif
}

TEST(Fp8Kernels, FusedIm2colPadColsStride2MatchesReferenceBytes)
{
#if defined(DARKNET_GPU_CUDA) && defined(CUDART_VERSION) && CUDART_VERSION >= 12010
	constexpr int channels = 2;
	constexpr int height = 7;
	constexpr int width = 6;
	constexpr int kernel_h = 3;
	constexpr int kernel_w = 3;
	constexpr int pad_h = 1;
	constexpr int pad_w = 1;
	constexpr int stride_h = 2;
	constexpr int stride_w = 2;
	constexpr int dilation_h = 1;
	constexpr int dilation_w = 1;
	constexpr int height_col = (height + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
	constexpr int width_col = (width + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
	constexpr int rows = channels * kernel_h * kernel_w;
	constexpr int cols = height_col * width_col;
	constexpr int cols_pad = 16;
	constexpr size_t input_count = static_cast<size_t>(channels) * height * width;
	constexpr size_t im2col_count = static_cast<size_t>(rows) * cols;
	constexpr size_t fp8_bytes = static_cast<size_t>(rows) * cols_pad;

	std::vector<float> input(input_count);
	for (size_t idx = 0; idx < input.size(); ++idx)
	{
		input[idx] = static_cast<float>(static_cast<int>((idx * 5) % 23) - 11) * 0.125f;
	}
	input[input.size() - 3] = 5.75f;
	float scale = 0.5f;

	float * input_gpu = cuda_make_array(input.data(), input.size());
	float * im2col_gpu = cuda_make_array(nullptr, im2col_count);
	float * scale_gpu = cuda_make_array(&scale, 1);
	void * reference_fp8_gpu = nullptr;
	void * fused_fp8_gpu = nullptr;
	CHECK_CUDA(cudaMalloc(&reference_fp8_gpu, fp8_bytes));
	CHECK_CUDA(cudaMalloc(&fused_fp8_gpu, fp8_bytes));
	CHECK_CUDA(cudaMemsetAsync(reference_fp8_gpu, 0, fp8_bytes, get_cuda_stream()));
	CHECK_CUDA(cudaMemsetAsync(fused_fp8_gpu, 0, fp8_bytes, get_cuda_stream()));

	im2col_gpu_ext(
		input_gpu,
		channels,
		height, width,
		kernel_h, kernel_w,
		pad_h, pad_w,
		stride_h, stride_w,
		dilation_h, dilation_w,
		im2col_gpu);
	Darknet::fp8_quantize_rowmajor_pad_cols_gpu(
		im2col_gpu, rows, cols, cols_pad, scale_gpu, reference_fp8_gpu);
	Darknet::fp8_im2col_quantize_rowmajor_pad_cols_gpu(
		input_gpu,
		channels,
		height, width,
		kernel_h, kernel_w,
		pad_h, pad_w,
		stride_h, stride_w,
		dilation_h, dilation_w,
		cols_pad,
		scale_gpu,
		fused_fp8_gpu);

	std::vector<unsigned char> reference_bytes(fp8_bytes);
	std::vector<unsigned char> fused_bytes(fp8_bytes);
	CHECK_CUDA(cudaMemcpyAsync(reference_bytes.data(), reference_fp8_gpu, fp8_bytes, cudaMemcpyDeviceToHost, get_cuda_stream()));
	CHECK_CUDA(cudaMemcpyAsync(fused_bytes.data(), fused_fp8_gpu, fp8_bytes, cudaMemcpyDeviceToHost, get_cuda_stream()));
	CHECK_CUDA(cudaStreamSynchronize(get_cuda_stream()));

	EXPECT_EQ(fused_bytes, reference_bytes);

	CHECK_CUDA(cudaFree(fused_fp8_gpu));
	CHECK_CUDA(cudaFree(reference_fp8_gpu));
	cuda_free(scale_gpu);
	cuda_free(im2col_gpu);
	cuda_free(input_gpu);
#else
	GTEST_SKIP() << "CUDA FP8 kernel test requires CUDA 12.1+";
#endif
}

TEST(Fp8Kernels, ColmajorBf16OutputAccumulatesDgradDirectly)
{
#if defined(DARKNET_GPU_CUDA) && defined(CUDART_VERSION) && CUDART_VERSION >= 12010
	constexpr int batch = 1;
	constexpr int channels = 1;
	constexpr int height = 3;
	constexpr int width = 3;
	constexpr int kernel_h = 3;
	constexpr int kernel_w = 3;
	constexpr int pad_h = 1;
	constexpr int pad_w = 1;
	constexpr int stride_h = 1;
	constexpr int stride_w = 1;
	constexpr int dilation_h = 1;
	constexpr int dilation_w = 1;
	constexpr int height_col = (height + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
	constexpr int width_col = (width + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
	constexpr int kernel = channels * kernel_h * kernel_w;
	constexpr int spatial = height_col * width_col;
	constexpr size_t col_count = static_cast<size_t>(batch) * kernel * spatial;
	constexpr size_t delta_count = static_cast<size_t>(batch) * channels * height * width;

	std::vector<float> col_rowmajor(col_count);
	for (size_t idx = 0; idx < col_rowmajor.size(); ++idx)
	{
		col_rowmajor[idx] = static_cast<float>((idx % 9) + 1);
	}
	std::vector<unsigned short> col_colmajor_bf16(col_count);
	for (int b = 0; b < batch; ++b)
	{
		for (int row = 0; row < kernel; ++row)
		{
			for (int col = 0; col < spatial; ++col)
			{
				const float value = col_rowmajor[static_cast<size_t>(b) * kernel * spatial + row * spatial + col];
				col_colmajor_bf16[static_cast<size_t>(b) * kernel * spatial + col * kernel + row] = bf16_storage_bits(value);
			}
		}
	}
	std::vector<float> initial_delta(delta_count);
	for (size_t idx = 0; idx < initial_delta.size(); ++idx)
	{
		initial_delta[idx] = 100.0f + static_cast<float>(idx);
	}
	const std::vector<float> expected = reference_col2im_accumulate(
		initial_delta,
		col_rowmajor,
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
		dilation_w);

	void * col_gpu = nullptr;
	CHECK_CUDA(cudaMalloc(&col_gpu, col_colmajor_bf16.size() * sizeof(col_colmajor_bf16[0])));
	CHECK_CUDA(cudaMemcpyAsync(
		col_gpu,
		col_colmajor_bf16.data(),
		col_colmajor_bf16.size() * sizeof(col_colmajor_bf16[0]),
		cudaMemcpyHostToDevice,
		get_cuda_stream()));
	float * delta_gpu = cuda_make_array(initial_delta.data(), initial_delta.size());

	Darknet::fp8_colmajor_output_to_nchw_delta_gpu(
		col_gpu,
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
		true,
		delta_gpu);

	std::vector<float> actual(delta_count);
	cuda_pull_array(delta_gpu, actual.data(), actual.size());
	for (size_t idx = 0; idx < actual.size(); ++idx)
	{
		EXPECT_NEAR(actual[idx], expected[idx], 1.0e-4f) << "idx=" << idx;
	}

	cuda_free(delta_gpu);
	CHECK_CUDA(cudaFree(col_gpu));
#else
	GTEST_SKIP() << "CUDA dgrad accumulation kernel test requires CUDA 12.1+";
#endif
}

TEST(Fp8Kernels, ColmajorBf16OutputAccumulatesDirect1x1Batched)
{
#if defined(DARKNET_GPU_CUDA) && defined(CUDART_VERSION) && CUDART_VERSION >= 12010
	constexpr int batch = 2;
	constexpr int channels = 3;
	constexpr int height = 2;
	constexpr int width = 3;
	constexpr int kernel_h = 1;
	constexpr int kernel_w = 1;
	constexpr int pad_h = 0;
	constexpr int pad_w = 0;
	constexpr int stride_h = 1;
	constexpr int stride_w = 1;
	constexpr int dilation_h = 1;
	constexpr int dilation_w = 1;
	constexpr int height_col = height;
	constexpr int width_col = width;
	constexpr int kernel = channels * kernel_h * kernel_w;
	constexpr int spatial = height_col * width_col;
	constexpr size_t col_count = static_cast<size_t>(batch) * kernel * spatial;
	constexpr size_t delta_count = static_cast<size_t>(batch) * channels * height * width;

	std::vector<float> col_rowmajor(col_count);
	for (size_t idx = 0; idx < col_rowmajor.size(); ++idx)
	{
		col_rowmajor[idx] = static_cast<float>(static_cast<int>(idx % 13) - 6) * 0.25f;
	}
	std::vector<unsigned short> col_colmajor_bf16(col_count);
	for (int b = 0; b < batch; ++b)
	{
		for (int row = 0; row < kernel; ++row)
		{
			for (int col = 0; col < spatial; ++col)
			{
				const float value = col_rowmajor[static_cast<size_t>(b) * kernel * spatial + row * spatial + col];
				col_colmajor_bf16[static_cast<size_t>(b) * kernel * spatial + col * kernel + row] = bf16_storage_bits(value);
			}
		}
	}
	std::vector<float> initial_delta(delta_count);
	for (size_t idx = 0; idx < initial_delta.size(); ++idx)
	{
		initial_delta[idx] = 50.0f + static_cast<float>(idx) * 0.5f;
	}
	const std::vector<float> expected = reference_col2im_accumulate(
		initial_delta,
		col_rowmajor,
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
		dilation_w);

	void * col_gpu = nullptr;
	CHECK_CUDA(cudaMalloc(&col_gpu, col_colmajor_bf16.size() * sizeof(col_colmajor_bf16[0])));
	CHECK_CUDA(cudaMemcpyAsync(
		col_gpu,
		col_colmajor_bf16.data(),
		col_colmajor_bf16.size() * sizeof(col_colmajor_bf16[0]),
		cudaMemcpyHostToDevice,
		get_cuda_stream()));
	float * delta_gpu = cuda_make_array(initial_delta.data(), initial_delta.size());

	Darknet::fp8_colmajor_output_to_nchw_delta_gpu(
		col_gpu,
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
		true,
		delta_gpu);

	std::vector<float> actual(delta_count);
	cuda_pull_array(delta_gpu, actual.data(), actual.size());
	for (size_t idx = 0; idx < actual.size(); ++idx)
	{
		EXPECT_NEAR(actual[idx], expected[idx], 1.0e-4f) << "idx=" << idx;
	}

	cuda_free(delta_gpu);
	CHECK_CUDA(cudaFree(col_gpu));
#else
	GTEST_SKIP() << "CUDA dgrad accumulation kernel test requires CUDA 12.1+";
#endif
}

TEST(Fp8Kernels, ColmajorBf16OutputAccumulatesRowmajorWeights)
{
#if defined(DARKNET_GPU_CUDA) && defined(CUDART_VERSION) && CUDART_VERSION >= 12010
	constexpr int rows = 3;
	constexpr int cols = 4;
	constexpr float alpha = 0.5f;
	std::vector<float> initial(static_cast<size_t>(rows) * cols);
	std::vector<float> rowmajor_values(static_cast<size_t>(rows) * cols);
	for (size_t idx = 0; idx < initial.size(); ++idx)
	{
		initial[idx] = 10.0f + static_cast<float>(idx);
		rowmajor_values[idx] = static_cast<float>((idx % 7) + 1);
	}
	std::vector<unsigned short> colmajor_bf16(rowmajor_values.size());
	for (int row = 0; row < rows; ++row)
	{
		for (int col = 0; col < cols; ++col)
		{
			const float value = rowmajor_values[static_cast<size_t>(row) * cols + col];
			colmajor_bf16[static_cast<size_t>(col) * rows + row] = bf16_storage_bits(value);
		}
	}
	std::vector<float> expected(initial.size());
	for (size_t idx = 0; idx < expected.size(); ++idx)
	{
		expected[idx] = initial[idx] + alpha * bf16_storage_to_float(bf16_storage_bits(rowmajor_values[idx]));
	}

	void * src_gpu = nullptr;
	CHECK_CUDA(cudaMalloc(&src_gpu, colmajor_bf16.size() * sizeof(colmajor_bf16[0])));
	CHECK_CUDA(cudaMemcpyAsync(
		src_gpu,
		colmajor_bf16.data(),
		colmajor_bf16.size() * sizeof(colmajor_bf16[0]),
		cudaMemcpyHostToDevice,
		get_cuda_stream()));
	float * dst_gpu = cuda_make_array(initial.data(), initial.size());

	Darknet::fp8_colmajor_output_accumulate_rowmajor_gpu(src_gpu, rows, cols, true, alpha, dst_gpu);

	std::vector<float> actual(initial.size());
	cuda_pull_array(dst_gpu, actual.data(), actual.size());
	for (size_t idx = 0; idx < actual.size(); ++idx)
	{
		EXPECT_NEAR(actual[idx], expected[idx], 1.0e-4f) << "idx=" << idx;
	}

	cuda_free(dst_gpu);
	CHECK_CUDA(cudaFree(src_gpu));
#else
	GTEST_SKIP() << "CUDA wgrad accumulation kernel test requires CUDA 12.1+";
#endif
}

TEST(Fp8Kernels, TripleLayoutWeightQuantizeMatchesSeparateQuantizers)
{
#if defined(DARKNET_GPU_CUDA) && defined(CUDART_VERSION) && CUDART_VERSION >= 12010
	constexpr int filters = 3;
	constexpr int channels = 2;
	constexpr int kernel_h = 3;
	constexpr int kernel_w = 2;
	constexpr int kernel = channels * kernel_h * kernel_w;
	constexpr int kernel_pad = 16;
	constexpr int filters_pad = 16;
	constexpr size_t rowmajor_bytes = static_cast<size_t>(filters) * kernel_pad;
	constexpr size_t transposed_bytes = static_cast<size_t>(kernel) * filters_pad;
	constexpr size_t krsc_bytes = static_cast<size_t>(filters) * kernel_h * kernel_w * channels;

	std::vector<float> weights(static_cast<size_t>(filters) * kernel);
	for (size_t idx = 0; idx < weights.size(); ++idx)
	{
		weights[idx] = static_cast<float>(static_cast<int>(idx % 11) - 5) * 0.25f;
	}
	float scale = 0.5f;
	float * weights_gpu = cuda_make_array(weights.data(), weights.size());
	float * scale_gpu = cuda_make_array(&scale, 1);
	void * ref_rowmajor_gpu = nullptr;
	void * ref_transposed_gpu = nullptr;
	void * ref_krsc_gpu = nullptr;
	void * fused_rowmajor_gpu = nullptr;
	void * fused_transposed_gpu = nullptr;
	void * fused_krsc_gpu = nullptr;
	CHECK_CUDA(cudaMalloc(&ref_rowmajor_gpu, rowmajor_bytes));
	CHECK_CUDA(cudaMalloc(&ref_transposed_gpu, transposed_bytes));
	CHECK_CUDA(cudaMalloc(&ref_krsc_gpu, krsc_bytes));
	CHECK_CUDA(cudaMalloc(&fused_rowmajor_gpu, rowmajor_bytes));
	CHECK_CUDA(cudaMalloc(&fused_transposed_gpu, transposed_bytes));
	CHECK_CUDA(cudaMalloc(&fused_krsc_gpu, krsc_bytes));
	CHECK_CUDA(cudaMemsetAsync(ref_rowmajor_gpu, 0, rowmajor_bytes, get_cuda_stream()));
	CHECK_CUDA(cudaMemsetAsync(ref_transposed_gpu, 0, transposed_bytes, get_cuda_stream()));
	CHECK_CUDA(cudaMemsetAsync(ref_krsc_gpu, 0, krsc_bytes, get_cuda_stream()));
	CHECK_CUDA(cudaMemsetAsync(fused_rowmajor_gpu, 0x7f, rowmajor_bytes, get_cuda_stream()));
	CHECK_CUDA(cudaMemsetAsync(fused_transposed_gpu, 0x7f, transposed_bytes, get_cuda_stream()));
	CHECK_CUDA(cudaMemsetAsync(fused_krsc_gpu, 0x7f, krsc_bytes, get_cuda_stream()));

	Darknet::fp8_quantize_dual_layout_weights_gpu(
		weights_gpu, filters, kernel, kernel_pad, filters_pad, scale_gpu, ref_rowmajor_gpu, ref_transposed_gpu);
	Darknet::fp8_quantize_weights_krsc_gpu(
		weights_gpu, filters, channels, kernel_h, kernel_w, scale_gpu, ref_krsc_gpu);
	Darknet::fp8_quantize_triple_layout_weights_gpu(
		weights_gpu,
		filters,
		channels,
		kernel_h,
		kernel_w,
		kernel_pad,
		filters_pad,
		scale_gpu,
		fused_rowmajor_gpu,
		fused_transposed_gpu,
		fused_krsc_gpu);

	std::vector<unsigned char> ref_rowmajor(rowmajor_bytes);
	std::vector<unsigned char> ref_transposed(transposed_bytes);
	std::vector<unsigned char> ref_krsc(krsc_bytes);
	std::vector<unsigned char> fused_rowmajor(rowmajor_bytes);
	std::vector<unsigned char> fused_transposed(transposed_bytes);
	std::vector<unsigned char> fused_krsc(krsc_bytes);
	CHECK_CUDA(cudaMemcpyAsync(ref_rowmajor.data(), ref_rowmajor_gpu, rowmajor_bytes, cudaMemcpyDeviceToHost, get_cuda_stream()));
	CHECK_CUDA(cudaMemcpyAsync(ref_transposed.data(), ref_transposed_gpu, transposed_bytes, cudaMemcpyDeviceToHost, get_cuda_stream()));
	CHECK_CUDA(cudaMemcpyAsync(ref_krsc.data(), ref_krsc_gpu, krsc_bytes, cudaMemcpyDeviceToHost, get_cuda_stream()));
	CHECK_CUDA(cudaMemcpyAsync(fused_rowmajor.data(), fused_rowmajor_gpu, rowmajor_bytes, cudaMemcpyDeviceToHost, get_cuda_stream()));
	CHECK_CUDA(cudaMemcpyAsync(fused_transposed.data(), fused_transposed_gpu, transposed_bytes, cudaMemcpyDeviceToHost, get_cuda_stream()));
	CHECK_CUDA(cudaMemcpyAsync(fused_krsc.data(), fused_krsc_gpu, krsc_bytes, cudaMemcpyDeviceToHost, get_cuda_stream()));
	CHECK_CUDA(cudaStreamSynchronize(get_cuda_stream()));

	EXPECT_EQ(fused_rowmajor, ref_rowmajor);
	EXPECT_EQ(fused_transposed, ref_transposed);
	EXPECT_EQ(fused_krsc, ref_krsc);

	CHECK_CUDA(cudaFree(fused_krsc_gpu));
	CHECK_CUDA(cudaFree(fused_transposed_gpu));
	CHECK_CUDA(cudaFree(fused_rowmajor_gpu));
	CHECK_CUDA(cudaFree(ref_krsc_gpu));
	CHECK_CUDA(cudaFree(ref_transposed_gpu));
	CHECK_CUDA(cudaFree(ref_rowmajor_gpu));
	cuda_free(scale_gpu);
	cuda_free(weights_gpu);
#else
	GTEST_SKIP() << "CUDA triple-layout weight quantize test requires CUDA 12.1+";
#endif
}

TEST(Fp8Kernels, TripleLayoutWeightQuantizeAcceptsNullTransposed)
{
#if defined(DARKNET_GPU_CUDA) && defined(CUDART_VERSION) && CUDART_VERSION >= 12010
	constexpr int filters = 3;
	constexpr int channels = 2;
	constexpr int kernel_h = 3;
	constexpr int kernel_w = 2;
	constexpr int kernel = channels * kernel_h * kernel_w;
	constexpr int kernel_pad = 16;
	constexpr int filters_pad = 16;
	constexpr size_t rowmajor_bytes = static_cast<size_t>(filters) * kernel_pad;
	constexpr size_t krsc_bytes = static_cast<size_t>(filters) * kernel_h * kernel_w * channels;

	std::vector<float> weights(static_cast<size_t>(filters) * kernel);
	for (size_t idx = 0; idx < weights.size(); ++idx)
	{
		weights[idx] = static_cast<float>(static_cast<int>(idx % 13) - 6) * 0.125f;
	}
	float scale = 0.25f;
	float * weights_gpu = cuda_make_array(weights.data(), weights.size());
	float * scale_gpu = cuda_make_array(&scale, 1);
	void * ref_rowmajor_gpu = nullptr;
	void * ref_krsc_gpu = nullptr;
	void * fused_rowmajor_gpu = nullptr;
	void * fused_krsc_gpu = nullptr;
	CHECK_CUDA(cudaMalloc(&ref_rowmajor_gpu, rowmajor_bytes));
	CHECK_CUDA(cudaMalloc(&ref_krsc_gpu, krsc_bytes));
	CHECK_CUDA(cudaMalloc(&fused_rowmajor_gpu, rowmajor_bytes));
	CHECK_CUDA(cudaMalloc(&fused_krsc_gpu, krsc_bytes));
	CHECK_CUDA(cudaMemsetAsync(ref_rowmajor_gpu, 0, rowmajor_bytes, get_cuda_stream()));
	CHECK_CUDA(cudaMemsetAsync(ref_krsc_gpu, 0, krsc_bytes, get_cuda_stream()));
	CHECK_CUDA(cudaMemsetAsync(fused_rowmajor_gpu, 0x7f, rowmajor_bytes, get_cuda_stream()));
	CHECK_CUDA(cudaMemsetAsync(fused_krsc_gpu, 0x7f, krsc_bytes, get_cuda_stream()));

	Darknet::fp8_quantize_rowmajor_pad_cols_gpu(
		weights_gpu, filters, kernel, kernel_pad, scale_gpu, ref_rowmajor_gpu);
	Darknet::fp8_quantize_weights_krsc_gpu(
		weights_gpu, filters, channels, kernel_h, kernel_w, scale_gpu, ref_krsc_gpu);
	Darknet::fp8_quantize_triple_layout_weights_gpu(
		weights_gpu,
		filters,
		channels,
		kernel_h,
		kernel_w,
		kernel_pad,
		filters_pad,
		scale_gpu,
		fused_rowmajor_gpu,
		nullptr,
		fused_krsc_gpu);

	std::vector<unsigned char> ref_rowmajor(rowmajor_bytes);
	std::vector<unsigned char> ref_krsc(krsc_bytes);
	std::vector<unsigned char> fused_rowmajor(rowmajor_bytes);
	std::vector<unsigned char> fused_krsc(krsc_bytes);
	CHECK_CUDA(cudaMemcpyAsync(ref_rowmajor.data(), ref_rowmajor_gpu, rowmajor_bytes, cudaMemcpyDeviceToHost, get_cuda_stream()));
	CHECK_CUDA(cudaMemcpyAsync(ref_krsc.data(), ref_krsc_gpu, krsc_bytes, cudaMemcpyDeviceToHost, get_cuda_stream()));
	CHECK_CUDA(cudaMemcpyAsync(fused_rowmajor.data(), fused_rowmajor_gpu, rowmajor_bytes, cudaMemcpyDeviceToHost, get_cuda_stream()));
	CHECK_CUDA(cudaMemcpyAsync(fused_krsc.data(), fused_krsc_gpu, krsc_bytes, cudaMemcpyDeviceToHost, get_cuda_stream()));
	CHECK_CUDA(cudaStreamSynchronize(get_cuda_stream()));

	EXPECT_EQ(fused_rowmajor, ref_rowmajor);
	EXPECT_EQ(fused_krsc, ref_krsc);

	CHECK_CUDA(cudaFree(fused_krsc_gpu));
	CHECK_CUDA(cudaFree(fused_rowmajor_gpu));
	CHECK_CUDA(cudaFree(ref_krsc_gpu));
	CHECK_CUDA(cudaFree(ref_rowmajor_gpu));
	cuda_free(scale_gpu);
	cuda_free(weights_gpu);
#else
	GTEST_SKIP() << "CUDA triple-layout weight quantize test requires CUDA 12.1+";
#endif
}

TEST(Fp8Kernels, TripleLayoutWeightQuantizeAcceptsNullRowmajor)
{
#if defined(DARKNET_GPU_CUDA) && defined(CUDART_VERSION) && CUDART_VERSION >= 12010
	constexpr int filters = 3;
	constexpr int channels = 2;
	constexpr int kernel_h = 3;
	constexpr int kernel_w = 2;
	constexpr int kernel = channels * kernel_h * kernel_w;
	constexpr int kernel_pad = 16;
	constexpr int filters_pad = 16;
	constexpr size_t transposed_bytes = static_cast<size_t>(kernel) * filters_pad;
	constexpr size_t krsc_bytes = static_cast<size_t>(filters) * kernel_h * kernel_w * channels;
	std::vector<float> weights(static_cast<size_t>(filters) * kernel);
	for (size_t idx = 0; idx < weights.size(); ++idx)
	{
		weights[idx] = static_cast<float>(static_cast<int>(idx % 13) - 6) * 0.125f;
	}
	float scale = 0.25f;
	float * weights_gpu = cuda_make_array(weights.data(), weights.size());
	float * scale_gpu = cuda_make_array(&scale, 1);
	void * ref_transposed_gpu = nullptr;
	void * ref_krsc_gpu = nullptr;
	void * fused_transposed_gpu = nullptr;
	void * fused_krsc_gpu = nullptr;
	CHECK_CUDA(cudaMalloc(&ref_transposed_gpu, transposed_bytes));
	CHECK_CUDA(cudaMalloc(&ref_krsc_gpu, krsc_bytes));
	CHECK_CUDA(cudaMalloc(&fused_transposed_gpu, transposed_bytes));
	CHECK_CUDA(cudaMalloc(&fused_krsc_gpu, krsc_bytes));
	Darknet::fp8_quantize_transpose_rowmajor_pad_cols_gpu(
		weights_gpu, filters, kernel, filters_pad, scale_gpu, ref_transposed_gpu);
	Darknet::fp8_quantize_weights_krsc_gpu(
		weights_gpu, filters, channels, kernel_h, kernel_w, scale_gpu, ref_krsc_gpu);
	Darknet::fp8_quantize_triple_layout_weights_gpu(
		weights_gpu, filters, channels, kernel_h, kernel_w, kernel_pad, filters_pad,
		scale_gpu, nullptr, fused_transposed_gpu, fused_krsc_gpu);
	std::vector<unsigned char> ref_transposed(transposed_bytes);
	std::vector<unsigned char> ref_krsc(krsc_bytes);
	std::vector<unsigned char> fused_transposed(transposed_bytes);
	std::vector<unsigned char> fused_krsc(krsc_bytes);
	CHECK_CUDA(cudaMemcpyAsync(ref_transposed.data(), ref_transposed_gpu, transposed_bytes, cudaMemcpyDeviceToHost, get_cuda_stream()));
	CHECK_CUDA(cudaMemcpyAsync(ref_krsc.data(), ref_krsc_gpu, krsc_bytes, cudaMemcpyDeviceToHost, get_cuda_stream()));
	CHECK_CUDA(cudaMemcpyAsync(fused_transposed.data(), fused_transposed_gpu, transposed_bytes, cudaMemcpyDeviceToHost, get_cuda_stream()));
	CHECK_CUDA(cudaMemcpyAsync(fused_krsc.data(), fused_krsc_gpu, krsc_bytes, cudaMemcpyDeviceToHost, get_cuda_stream()));
	CHECK_CUDA(cudaStreamSynchronize(get_cuda_stream()));
	EXPECT_EQ(fused_transposed, ref_transposed);
	EXPECT_EQ(fused_krsc, ref_krsc);
	CHECK_CUDA(cudaFree(fused_krsc_gpu));
	CHECK_CUDA(cudaFree(fused_transposed_gpu));
	CHECK_CUDA(cudaFree(ref_krsc_gpu));
	CHECK_CUDA(cudaFree(ref_transposed_gpu));
	cuda_free(scale_gpu);
	cuda_free(weights_gpu);
#else
	GTEST_SKIP() << "CUDA triple-layout weight quantize test requires CUDA 12.1+";
#endif
}
