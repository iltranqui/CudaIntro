#include <gtest/gtest.h>

#include <vector>

#include "darknet_internal.hpp"
#include "fp8_conv.hpp"
#include "fp8_kernels.hpp"

TEST(Fp8Conv, FpropPlanExecutesProbeShape)
{
#if defined(DARKNET_GPU_CUDA) && defined(DARKNET_FP8_CUDNN_CONV) && defined(CUDART_VERSION) && CUDART_VERSION >= 12010
	if (!Darknet::fp8_conv_supported())
	{
		GTEST_SKIP() << "FP8 cuDNN graph convolution is not supported on this device/runtime";
	}

	Darknet::Fp8ConvSpec spec;
	spec.batch = 64;
	spec.channels = 64;
	spec.height = 40;
	spec.width = 56;
	spec.filters = 64;
	spec.kernel_h = 3;
	spec.kernel_w = 3;
	spec.pad_h = 1;
	spec.pad_w = 1;
	spec.output = Darknet::Fp8ConvOutput::Fp32;

	float scale = 1.0f;
	float * input_scale_gpu = cuda_make_array(&scale, 1);
	float * weight_scale_gpu = cuda_make_array(&scale, 1);
	auto * plan = Darknet::fp8_conv_plan_create_fprop(spec, input_scale_gpu, weight_scale_gpu);
	ASSERT_NE(plan, nullptr);

	void * input_fp8 = nullptr;
	void * weights_fp8 = nullptr;
	void * output = nullptr;
	void * workspace = nullptr;
	const size_t input_bytes = static_cast<size_t>(spec.batch) * spec.channels * spec.height * spec.width;
	const size_t weight_bytes = static_cast<size_t>(spec.filters) * spec.channels * spec.kernel_h * spec.kernel_w;
	const int out_h = Darknet::fp8_conv_out_dim(spec.height, spec.pad_h, spec.dilation_h, spec.kernel_h, spec.stride_h);
	const int out_w = Darknet::fp8_conv_out_dim(spec.width, spec.pad_w, spec.dilation_w, spec.kernel_w, spec.stride_w);
	const size_t output_bytes = static_cast<size_t>(spec.batch) * spec.filters * out_h * out_w * sizeof(float);
	const size_t workspace_bytes = Darknet::fp8_conv_workspace_bytes(plan);
	CHECK_CUDA(cudaMalloc(&input_fp8, input_bytes));
	CHECK_CUDA(cudaMalloc(&weights_fp8, weight_bytes));
	CHECK_CUDA(cudaMalloc(&output, output_bytes));
	if (workspace_bytes > 0)
	{
		CHECK_CUDA(cudaMalloc(&workspace, workspace_bytes));
	}
	CHECK_CUDA(cudaMemsetAsync(input_fp8, 0, input_bytes, get_cuda_stream()));
	CHECK_CUDA(cudaMemsetAsync(weights_fp8, 0, weight_bytes, get_cuda_stream()));
	CHECK_CUDA(cudaMemsetAsync(output, 0x7f, output_bytes, get_cuda_stream()));

	EXPECT_TRUE(Darknet::fp8_conv_fprop(plan, input_fp8, weights_fp8, nullptr, output, workspace, workspace_bytes));
	CHECK_CUDA(cudaStreamSynchronize(get_cuda_stream()));

	if (workspace)
	{
		CHECK_CUDA(cudaFree(workspace));
	}
	CHECK_CUDA(cudaFree(output));
	CHECK_CUDA(cudaFree(weights_fp8));
	CHECK_CUDA(cudaFree(input_fp8));
	Darknet::fp8_conv_plan_destroy(plan);
	cuda_free(weight_scale_gpu);
	cuda_free(input_scale_gpu);
#else
	GTEST_SKIP() << "FP8 cuDNN graph convolution test requires CUDA 12.1+ and DARKNET_FP8_CUDNN_CONV";
#endif
}

TEST(Fp8Conv, NhwcOutputConvertsToNchwWithBias)
{
#if defined(DARKNET_GPU_CUDA) && defined(CUDART_VERSION) && CUDART_VERSION >= 12010
	constexpr int batch = 2;
	constexpr int channels = 3;
	constexpr int height = 2;
	constexpr int width = 2;
	std::vector<float> nhwc(static_cast<size_t>(batch) * height * width * channels);
	for (int n = 0; n < batch; ++n)
	{
		for (int h = 0; h < height; ++h)
		{
			for (int w = 0; w < width; ++w)
			{
				for (int c = 0; c < channels; ++c)
				{
					nhwc[((static_cast<size_t>(n) * height + h) * width + w) * channels + c] =
						static_cast<float>(100 * n + 10 * c + 2 * h + w);
				}
			}
		}
	}
	std::vector<float> bias = {0.5f, 1.5f, 2.5f};
	float * nhwc_gpu = cuda_make_array(nhwc.data(), nhwc.size());
	float * bias_gpu = cuda_make_array(bias.data(), bias.size());
	float * nchw_gpu = cuda_make_array(nullptr, nhwc.size());

	Darknet::fp8_nhwc_output_to_nchw_gpu(nhwc_gpu, batch, channels, height, width, false, bias_gpu, nchw_gpu);
	std::vector<float> nchw(nhwc.size(), 0.0f);
	CHECK_CUDA(cudaMemcpyAsync(nchw.data(), nchw_gpu, nchw.size() * sizeof(float), cudaMemcpyDeviceToHost, get_cuda_stream()));
	CHECK_CUDA(cudaStreamSynchronize(get_cuda_stream()));

	for (int n = 0; n < batch; ++n)
	{
		for (int c = 0; c < channels; ++c)
		{
			for (int h = 0; h < height; ++h)
			{
				for (int w = 0; w < width; ++w)
				{
					const size_t dst = ((static_cast<size_t>(n) * channels + c) * height + h) * width + w;
					const float expected = static_cast<float>(100 * n + 10 * c + 2 * h + w) + bias[c];
					EXPECT_FLOAT_EQ(nchw[dst], expected);
				}
			}
		}
	}

	cuda_free(nchw_gpu);
	cuda_free(bias_gpu);
	cuda_free(nhwc_gpu);
#else
	GTEST_SKIP() << "CUDA FP8 layout test requires CUDA 12.1+";
#endif
}
