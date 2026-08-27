#pragma once

#include <cstdint>
#include <limits>

namespace Darknet
{
	enum class Fp4ConvolutionDirection
	{
		Forward,
		WeightGradient,
		DataGradient
	};

	struct Fp4ConvolutionShape
	{
		int batch;
		int channels;
		int output_height;
		int output_width;
		int filters;
		int kernel_height;
		int kernel_width;
	};

	struct Fp4ConvolutionGemm
	{
		int batch = 0;
		int rows = 0;
		int columns = 0;
		int reduction = 0;
	};

	constexpr bool fp4_convolution_shape_valid(const Fp4ConvolutionShape & shape)
	{
		if (shape.batch <= 0 || shape.channels <= 0 || shape.output_height <= 0 || shape.output_width <= 0 ||
			shape.filters <= 0 || shape.kernel_height <= 0 || shape.kernel_width <= 0)
		{
			return false;
		}
		const int64_t spatial = static_cast<int64_t>(shape.output_height) * shape.output_width;
		const int64_t kernel = static_cast<int64_t>(shape.channels) * shape.kernel_height * shape.kernel_width;
		const int64_t batch_spatial = static_cast<int64_t>(shape.batch) * spatial;
		return spatial <= std::numeric_limits<int>::max() && kernel <= std::numeric_limits<int>::max() &&
			batch_spatial <= std::numeric_limits<int>::max();
	}

	constexpr Fp4ConvolutionGemm fp4_convolution_gemm(
		const Fp4ConvolutionShape & shape, const Fp4ConvolutionDirection direction)
	{
		if (!fp4_convolution_shape_valid(shape)) return {};
		const int spatial = shape.output_height * shape.output_width;
		const int kernel = shape.channels * shape.kernel_height * shape.kernel_width;
		switch (direction)
		{
			case Fp4ConvolutionDirection::Forward:
				return {shape.batch, shape.filters, spatial, kernel};
			case Fp4ConvolutionDirection::WeightGradient:
				return {1, shape.filters, kernel, shape.batch * spatial};
			case Fp4ConvolutionDirection::DataGradient:
				return {shape.batch, kernel, spatial, shape.filters};
		}
		return {};
	}
}
