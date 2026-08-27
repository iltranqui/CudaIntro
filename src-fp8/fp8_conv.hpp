#pragma once

#include <cstddef>

namespace Darknet
{
	struct Fp8ConvPlan;

	enum class Fp8ConvOutput
	{
		Fp32,
		Bf16
	};

	struct Fp8ConvSpec
	{
		int batch = 0;
		int channels = 0;
		int height = 0;
		int width = 0;
		int filters = 0;
		int kernel_h = 0;
		int kernel_w = 0;
		int pad_h = 0;
		int pad_w = 0;
		int stride_h = 1;
		int stride_w = 1;
		int dilation_h = 1;
		int dilation_w = 1;
		Fp8ConvOutput output = Fp8ConvOutput::Fp32;
		bool fuse_bias = false;
		bool fuse_relu = false;
	};

	bool fp8_conv_supported();
	int fp8_conv_out_dim(int input, int pad, int dilation, int kernel, int stride);
	Fp8ConvPlan * fp8_conv_plan_create_fprop(
		const Fp8ConvSpec & spec,
		const float * input_scale_gpu,
		const float * weight_scale_gpu);
	void fp8_conv_plan_destroy(Fp8ConvPlan * plan);
	size_t fp8_conv_workspace_bytes(const Fp8ConvPlan * plan);
	bool fp8_conv_output_is_bf16(const Fp8ConvPlan * plan);
	bool fp8_conv_fuses_bias(const Fp8ConvPlan * plan);
	bool fp8_conv_fuses_relu(const Fp8ConvPlan * plan);

	bool fp8_conv_fprop(
		Fp8ConvPlan * plan,
		const void * input_fp8_nhwc_gpu,
		const void * weights_fp8_krsc_gpu,
		const float * bias_gpu,
		void * output_nhwc_gpu,
		void * workspace,
		size_t workspace_bytes);
}
