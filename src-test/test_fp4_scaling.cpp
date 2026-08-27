#include <gtest/gtest.h>

#include "fp4_scaling.hpp"
#include "darknet_internal.hpp"

#include <array>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <vector>

void save_convolutional_weights(Darknet::Layer & l, FILE * fp);

TEST(Fp4Scaling, StorageAndScaleCountsRoundUpTails)
{
	EXPECT_EQ(Darknet::fp4_packed_byte_count(0), 0U);
	EXPECT_EQ(Darknet::fp4_packed_byte_count(1), 1U);
	EXPECT_EQ(Darknet::fp4_packed_byte_count(17), 9U);
	EXPECT_EQ(Darknet::fp4_block_count(17), 2U);
	EXPECT_EQ(Darknet::fp4_weight_tile_count(17, 33), 6U);
}

TEST(Fp4Scaling, E2m1ReferenceHandlesZeroSignRoundingAndSaturation)
{
	EXPECT_EQ(Darknet::fp4_encode_e2m1(0.0f), 0x0);
	EXPECT_EQ(Darknet::fp4_encode_e2m1(-0.0f), 0x8);
	EXPECT_EQ(Darknet::fp4_encode_e2m1(100.0f), 0x7);
	EXPECT_EQ(Darknet::fp4_encode_e2m1(-100.0f), 0xf);
	EXPECT_EQ(Darknet::fp4_encode_e2m1(1.6f), 0x3);
	EXPECT_FLOAT_EQ(Darknet::fp4_decode_e2m1(0x1), 0.5f);
	EXPECT_FLOAT_EQ(Darknet::fp4_decode_e2m1(0xf), -6.0f);
}

TEST(Fp4Scaling, ExactMidpointsRoundToNearestEven)
{
	EXPECT_EQ(Darknet::fp4_encode_e2m1(0.75f), 0x2);
	EXPECT_EQ(Darknet::fp4_encode_e2m1(1.75f), 0x4);
	EXPECT_EQ(Darknet::fp4_encode_e2m1(3.5f), 0x6);
	EXPECT_EQ(Darknet::fp4_encode_e4m3(1.0625f), 0x38);
	EXPECT_EQ(Darknet::fp4_encode_e4m3(1.1875f), 0x3a);
}

TEST(Fp4Scaling, ExceptionalValuesUseSatfinitePolicy)
{
	const float infinity = std::numeric_limits<float>::infinity();
	// cuda_fp4.h specifies that NaN converts to positive MAXNORM.
	EXPECT_EQ(Darknet::fp4_encode_e2m1(std::numeric_limits<float>::quiet_NaN()), 0x7);
	EXPECT_EQ(Darknet::fp4_encode_e2m1(infinity), 0x7);
	EXPECT_EQ(Darknet::fp4_encode_e2m1(-infinity), 0xf);
	EXPECT_EQ(Darknet::fp4_encode_e4m3(std::numeric_limits<float>::quiet_NaN()), 0x7e);
	EXPECT_EQ(Darknet::fp4_encode_e4m3(infinity), 0x7e);
	EXPECT_EQ(Darknet::fp4_encode_e4m3(-infinity), 0xfe);

	const std::array<float, 3> input = {6.0f, infinity, -infinity};
	const auto scaled = Darknet::fp4_quantize_1d_reference(input.data(), input.size());
	EXPECT_FLOAT_EQ(scaled.global_scale, 1.0f / 448.0f);
	EXPECT_EQ(scaled.values[0] & 0x0fU, 0x7U);
	EXPECT_EQ(static_cast<unsigned>(scaled.values[0]) >> 4U, 0x7U);
	EXPECT_EQ(scaled.values[1] & 0x0fU, 0xfU);
}

TEST(Fp4Scaling, PackingKeepsOddTailAndClearsUnusedNibble)
{
	const std::array<float, 3> input = {0.5f, -1.0f, 6.0f};
	const auto packed = Darknet::fp4_pack_e2m1(input.data(), input.size());
	ASSERT_EQ(packed.size(), 2U);
	EXPECT_EQ(packed[0], 0xa1);
	EXPECT_EQ(packed[1], 0x07);
}

TEST(Fp4Scaling, OneDimensionalMetadataUsesGlobalAndLocalScales)
{
	std::array<float, 17> input = {};
	input[0] = 3.0f;
	input[16] = 6.0f;
	const auto scaled = Darknet::fp4_quantize_1d_reference(input.data(), input.size());
	ASSERT_EQ(scaled.local_scales_e4m3.size(), 2U);
	EXPECT_FLOAT_EQ(scaled.global_scale, 1.0f / 448.0f);
	EXPECT_EQ(scaled.local_scales_e4m3[0], 0x76);
	EXPECT_EQ(scaled.local_scales_e4m3[1], 0x7e);
	EXPECT_EQ(scaled.values[0], 0x07);
	EXPECT_EQ(scaled.values[8], 0x07);
	EXPECT_EQ(scaled.values.size(), Darknet::fp4_packed_byte_count(input.size()));
}

TEST(Fp4Scaling, WeightMetadataUsesPaddedSixteenBySixteenTiles)
{
	std::vector<float> weights(17U * 17U, 0.0f);
	weights[0] = 3.0f;
	weights[16] = 6.0f;
	weights[16U * 17U] = 1.5f;
	const auto scaled = Darknet::fp4_quantize_weights_2d_reference(weights.data(), 17, 17);
	EXPECT_EQ(scaled.tile_rows, 2U);
	EXPECT_EQ(scaled.tile_columns, 2U);
	ASSERT_EQ(scaled.local_scales_e4m3.size(), 4U);
	EXPECT_FLOAT_EQ(scaled.global_scale, 1.0f / 448.0f);
	EXPECT_EQ(scaled.local_scales_e4m3[0], 0x76);
	EXPECT_EQ(scaled.local_scales_e4m3[1], 0x7e);
	EXPECT_EQ(scaled.local_scales_e4m3[2], 0x6e);
	EXPECT_EQ(scaled.local_scales_e4m3[3], 0x38);
	EXPECT_EQ(scaled.values[0], 0x07);
	EXPECT_EQ(scaled.values[8], 0x07);
	EXPECT_EQ(scaled.values[136], 0x07);
	EXPECT_EQ(scaled.values.size(), Darknet::fp4_packed_byte_count(weights.size()));
}

TEST(Fp4StateIsolation, DerivingFp4WeightsDoesNotModifyFp32MasterOrBatchNormState)
{
	std::vector<float> weights = {0.125f, -0.25f, 0.5f, -1.0f};
	std::vector<float> biases = {1.25f, -2.5f};
	std::vector<float> scales = {0.75f, 1.5f};
	std::vector<float> rolling_mean = {-0.125f, 0.375f};
	std::vector<float> rolling_variance = {0.625f, 1.875f};
	const auto original_weights = weights;
	const auto original_biases = biases;
	const auto original_scales = scales;
	const auto original_mean = rolling_mean;
	const auto original_variance = rolling_variance;

	auto derived = Darknet::fp4_quantize_weights_2d_reference(weights.data(), 2, 2);
	ASSERT_FALSE(derived.values.empty());
	std::fill(derived.values.begin(), derived.values.end(), 0xffU);
	std::fill(derived.local_scales_e4m3.begin(), derived.local_scales_e4m3.end(), 0U);
	derived.global_scale = -999.0f;

	EXPECT_EQ(weights, original_weights);
	EXPECT_EQ(biases, original_biases);
	EXPECT_EQ(scales, original_scales);
	EXPECT_EQ(rolling_mean, original_mean);
	EXPECT_EQ(rolling_variance, original_variance);
}

TEST(Fp4StateIsolation, ConvolutionCheckpointSerializesOnlyFp32MasterAndBatchNormState)
{
	std::vector<float> weights = {0.125f, -0.25f, 0.5f, -1.0f};
	std::vector<float> biases = {1.25f, -2.5f};
	std::vector<float> scales = {0.75f, 1.5f};
	std::vector<float> rolling_mean = {-0.125f, 0.375f};
	std::vector<float> rolling_variance = {0.625f, 1.875f};
	auto derived = Darknet::fp4_quantize_weights_2d_reference(weights.data(), 2, 2);
	std::fill(derived.values.begin(), derived.values.end(), 0xffU);
	std::fill(derived.local_scales_e4m3.begin(), derived.local_scales_e4m3.end(), 0U);
	derived.global_scale = -999.0f;

	Darknet::Layer layer = {};
	layer.n = 2;
	layer.nweights = static_cast<int>(weights.size());
	layer.batch_normalize = 1;
	layer.weights = weights.data();
	layer.biases = biases.data();
	layer.scales = scales.data();
	layer.rolling_mean = rolling_mean.data();
	layer.rolling_variance = rolling_variance.data();
	layer.fp4_gemm_plan = reinterpret_cast<void *>(0x1);
	layer.fp4_workspace_size = derived.values.size();
	layer.fp4_eligible = 1;

	const int previous_gpu = Darknet::CfgAndState::get().gpu_index;
	Darknet::CfgAndState::get().gpu_index = -1;
	FILE * file = std::tmpfile();
	ASSERT_NE(file, nullptr);
	save_convolutional_weights(layer, file);
	Darknet::CfgAndState::get().gpu_index = previous_gpu;
	std::rewind(file);
	std::vector<float> serialized(biases.size() + scales.size() + rolling_mean.size() + rolling_variance.size() + weights.size());
	ASSERT_EQ(std::fread(serialized.data(), sizeof(float), serialized.size(), file), serialized.size());
	EXPECT_EQ(std::fgetc(file), EOF);
	std::fclose(file);

	std::vector<float> expected;
	expected.insert(expected.end(), biases.begin(), biases.end());
	expected.insert(expected.end(), scales.begin(), scales.end());
	expected.insert(expected.end(), rolling_mean.begin(), rolling_mean.end());
	expected.insert(expected.end(), rolling_variance.begin(), rolling_variance.end());
	expected.insert(expected.end(), weights.begin(), weights.end());
	EXPECT_EQ(serialized, expected);
}

TEST(Fp4Scaling, RejectsInvalidPointersAndOverflowingDimensions)
{
	EXPECT_THROW(Darknet::fp4_pack_e2m1(nullptr, 1), std::invalid_argument);
	EXPECT_THROW(Darknet::fp4_quantize_1d_reference(nullptr, 1), std::invalid_argument);
	EXPECT_THROW(Darknet::fp4_quantize_weights_2d_reference(nullptr, 1, 1), std::invalid_argument);
	EXPECT_NO_THROW(Darknet::fp4_pack_e2m1(nullptr, 0));
	EXPECT_THROW(Darknet::fp4_packed_byte_count(std::numeric_limits<size_t>::max()), std::overflow_error);
	EXPECT_THROW(Darknet::fp4_block_count(std::numeric_limits<size_t>::max()), std::overflow_error);
	EXPECT_THROW(Darknet::fp4_weight_tile_count(std::numeric_limits<size_t>::max(),
		std::numeric_limits<size_t>::max()), std::overflow_error);
	EXPECT_THROW(Darknet::fp4_quantize_weights_2d_reference(reinterpret_cast<const float *>(1),
		std::numeric_limits<size_t>::max(), 2), std::overflow_error);
}

TEST(Fp4Scaling, DeterministicRht16IsSelfInverseAndCancelsInDotProduct)
{
	std::array<float, Darknet::kFp4BlockSize> lhs = {};
	std::array<float, Darknet::kFp4BlockSize> rhs = {};
	for (size_t idx = 0; idx < lhs.size(); ++idx)
	{
		lhs[idx] = static_cast<float>(idx + 1);
		rhs[idx] = static_cast<float>(2 * static_cast<int>(idx) - 7);
	}
	const auto lhs_transformed = Darknet::fp4_rht16_reference(lhs, 1234);
	const auto rhs_transformed = Darknet::fp4_rht16_reference(rhs, 1234);
	const auto round_trip = Darknet::fp4_rht16_reference(lhs_transformed, 1234);
	float original_dot = 0.0f;
	float transformed_dot = 0.0f;
	for (size_t idx = 0; idx < lhs.size(); ++idx)
	{
		EXPECT_NEAR(round_trip[idx], lhs[idx], 1.0e-5f);
		original_dot += lhs[idx] * rhs[idx];
		transformed_dot += lhs_transformed[idx] * rhs_transformed[idx];
	}
	EXPECT_NEAR(transformed_dot, original_dot, std::fabs(original_dot) * 1.0e-5f);
}
