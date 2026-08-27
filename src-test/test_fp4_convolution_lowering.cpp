#include "fp4_convolution_lowering.hpp"

#include <gtest/gtest.h>

TEST(Fp4ConvolutionLowering, ProducesForwardWeightAndDataGradientGemmShapes)
{
	const Darknet::Fp4ConvolutionShape conv{2, 32, 14, 14, 64, 3, 3};

	const auto fprop = Darknet::fp4_convolution_gemm(conv, Darknet::Fp4ConvolutionDirection::Forward);
	EXPECT_EQ(fprop.batch, 2);
	EXPECT_EQ(fprop.rows, 64);
	EXPECT_EQ(fprop.columns, 196);
	EXPECT_EQ(fprop.reduction, 288);

	const auto wgrad = Darknet::fp4_convolution_gemm(conv, Darknet::Fp4ConvolutionDirection::WeightGradient);
	EXPECT_EQ(wgrad.batch, 1);
	EXPECT_EQ(wgrad.rows, 64);
	EXPECT_EQ(wgrad.columns, 288);
	EXPECT_EQ(wgrad.reduction, 392);

	const auto dgrad = Darknet::fp4_convolution_gemm(conv, Darknet::Fp4ConvolutionDirection::DataGradient);
	EXPECT_EQ(dgrad.batch, 2);
	EXPECT_EQ(dgrad.rows, 288);
	EXPECT_EQ(dgrad.columns, 196);
	EXPECT_EQ(dgrad.reduction, 64);
}

TEST(Fp4ConvolutionLowering, RejectsInvalidAndOverflowingShapes)
{
	EXPECT_FALSE(Darknet::fp4_convolution_shape_valid({0, 32, 14, 14, 64, 3, 3}));
	EXPECT_FALSE(Darknet::fp4_convolution_shape_valid({1, 32, 0, 14, 64, 3, 3}));
	EXPECT_FALSE(Darknet::fp4_convolution_shape_valid({1, 32, 14, 14, 64, 0, 3}));
}
