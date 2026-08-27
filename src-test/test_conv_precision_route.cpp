#include "convolution_precision_route.hpp"
#include "fp4_gemm.hpp"

#include <gtest/gtest.h>

TEST(ConvolutionPrecisionRoute, ChoosesHighestReadyRequestedBackend)
{
	using Darknet::ConvolutionPrecision;
	using Darknet::ConvolutionPrecisionAvailability;

	EXPECT_EQ(Darknet::select_convolution_precision({true, true, true, true, true}), ConvolutionPrecision::Fp4);
	EXPECT_EQ(Darknet::select_convolution_precision({true, true, false, true, true}), ConvolutionPrecision::Fp8);
	EXPECT_EQ(Darknet::select_convolution_precision({true, true, false, false, true}), ConvolutionPrecision::Cudnn);
}

TEST(ConvolutionPrecisionRoute, NeverSelectsAnUnrequestedOrUnavailableBackend)
{
	using Darknet::ConvolutionPrecision;

	EXPECT_EQ(Darknet::select_convolution_precision({false, true, true, true, true}), ConvolutionPrecision::Fp8);
	EXPECT_EQ(Darknet::select_convolution_precision({false, false, true, true, true}), ConvolutionPrecision::Cudnn);
	EXPECT_EQ(Darknet::select_convolution_precision({true, true, false, false, false}), ConvolutionPrecision::None);
}

TEST(ConvolutionPrecisionRoute, BackwardFallbackRequestsOnlyUnfinishedDirections)
{
	EXPECT_EQ(Darknet::remaining_convolution_gradients({true, false}).wgrad_done, false);
	EXPECT_EQ(Darknet::remaining_convolution_gradients({true, false}).dgrad_done, true);
	EXPECT_EQ(Darknet::remaining_convolution_gradients({false, true}).wgrad_done, true);
	EXPECT_EQ(Darknet::remaining_convolution_gradients({false, true}).dgrad_done, false);
}

TEST(ConvolutionPrecisionRoute, Fp4GemmPrefersFrontendThenCublasLt)
{
	using Darknet::Fp4GemmBackend;
	EXPECT_EQ(Darknet::select_fp4_gemm_backend(true, true), Fp4GemmBackend::CublasLt);
	EXPECT_EQ(Darknet::select_fp4_gemm_backend(false, true), Fp4GemmBackend::CublasLt);
	EXPECT_EQ(Darknet::select_fp4_gemm_backend(false, false), Fp4GemmBackend::None);
}
