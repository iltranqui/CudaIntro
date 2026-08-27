#pragma once

namespace Darknet
{
	enum class ConvolutionPrecision
	{
		None,
		Cudnn,
		Fp8,
		Fp4
	};

	struct ConvolutionPrecisionAvailability
	{
		bool request_fp4 = false;
		bool request_fp8 = false;
		bool fp4_ready = false;
		bool fp8_ready = false;
		bool cudnn_ready = false;
	};

	constexpr ConvolutionPrecision select_convolution_precision(const ConvolutionPrecisionAvailability availability)
	{
		if (availability.request_fp4 && availability.fp4_ready)
		{
			return ConvolutionPrecision::Fp4;
		}
		if ((availability.request_fp4 || availability.request_fp8) && availability.fp8_ready)
		{
			return ConvolutionPrecision::Fp8;
		}
		if (availability.cudnn_ready)
		{
			return ConvolutionPrecision::Cudnn;
		}
		return ConvolutionPrecision::None;
	}

	struct ConvolutionBackwardProgress
	{
		bool wgrad_done = false;
		bool dgrad_done = false;
	};

	constexpr ConvolutionBackwardProgress remaining_convolution_gradients(const ConvolutionBackwardProgress completed)
	{
		return {!completed.wgrad_done, !completed.dgrad_done};
	}
}
