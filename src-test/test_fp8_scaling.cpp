#include <gtest/gtest.h>

#include <array>

#include "darknet_internal.hpp"
#include "fp8_scaling.hpp"

TEST(Fp8Scaling, FormatMaxValuesMatchTrainingPlan)
{
	EXPECT_FLOAT_EQ(Darknet::fp8_format_max(Darknet::Fp8Format::E4M3), 448.0f);
	EXPECT_FLOAT_EQ(Darknet::fp8_format_max(Darknet::Fp8Format::E5M2), 57344.0f);
}

TEST(Fp8Scaling, DelayedScaleUsesMaxFiniteAmaxHistory)
{
	std::array<float, Darknet::kFp8AmaxHistoryLength> history = {};
	history[0] = 224.0f;
	history[1] = 448.0f;
	history[2] = -100.0f;
	history[3] = std::numeric_limits<float>::infinity();

	EXPECT_FLOAT_EQ(Darknet::fp8_delayed_scale_from_history(history, Darknet::Fp8Format::E4M3), 1.0f);
}

TEST(Fp8Scaling, MarginDoublesScaleForGradientHeadroom)
{
	std::array<float, Darknet::kFp8AmaxHistoryLength> history = {};
	history[0] = 448.0f;

	EXPECT_FLOAT_EQ(Darknet::fp8_delayed_scale_from_history(history, Darknet::Fp8Format::E4M3, 0), 1.0f);
	EXPECT_FLOAT_EQ(Darknet::fp8_delayed_scale_from_history(history, Darknet::Fp8Format::E4M3, 1), 2.0f);
	EXPECT_FLOAT_EQ(Darknet::fp8_delayed_scale_from_history(history, Darknet::Fp8Format::E4M3, 2), 4.0f);
	// negative margins are clamped to zero rather than shrinking the scale
	EXPECT_FLOAT_EQ(Darknet::fp8_delayed_scale_from_history(history, Darknet::Fp8Format::E4M3, -1), 1.0f);
}

TEST(Fp8Scaling, RecordsAmaxWithMarginThroughRing)
{
	Darknet::Fp8DelayedScalingState state;

	Darknet::fp8_delayed_scaling_record_amax(state, 57344.0f, Darknet::Fp8Format::E5M2, Darknet::kFp8DyScaleMargin);
	EXPECT_FLOAT_EQ(state.scale, static_cast<float>(1 << Darknet::kFp8DyScaleMargin));
}

TEST(Fp8Scaling, RecordsAmaxIntoRingAndUpdatesScale)
{
	Darknet::Fp8DelayedScalingState state;

	Darknet::fp8_delayed_scaling_record_amax(state, 28672.0f, Darknet::Fp8Format::E5M2);
	EXPECT_EQ(state.valid_count, 1);
	EXPECT_EQ(state.next_index, 1);
	EXPECT_FLOAT_EQ(state.scale, 0.5f);

	Darknet::fp8_delayed_scaling_record_amax(state, 57344.0f, Darknet::Fp8Format::E5M2);
	EXPECT_EQ(state.valid_count, 2);
	EXPECT_EQ(state.next_index, 2);
	EXPECT_FLOAT_EQ(state.scale, 1.0f);
}

TEST(Fp8Scaling, RecordsAmaxIntoRawHistoryForLayerStorage)
{
	float history[Darknet::kFp8AmaxHistoryLength] = {};
	int next_index = 0;
	int valid_count = 0;
	float scale = 1.0f;

	Darknet::fp8_delayed_scaling_record_amax(
		history,
		Darknet::kFp8AmaxHistoryLength,
		next_index,
		valid_count,
		scale,
		224.0f,
		Darknet::Fp8Format::E4M3);

	EXPECT_EQ(next_index, 1);
	EXPECT_EQ(valid_count, 1);
	EXPECT_FLOAT_EQ(scale, 0.5f);

	Darknet::fp8_delayed_scaling_record_amax(
		history,
		Darknet::kFp8AmaxHistoryLength,
		next_index,
		valid_count,
		scale,
		std::numeric_limits<float>::infinity(),
		Darknet::Fp8Format::E4M3);

	EXPECT_EQ(next_index, 2);
	EXPECT_EQ(valid_count, 2);
	EXPECT_FLOAT_EQ(scale, 0.5f);
}

TEST(Fp8Scaling, ResolvesNegativeLayerSkipIndexesFromNetworkEnd)
{
	EXPECT_EQ(Darknet::fp8_resolve_layer_index(0, 5), 0);
	EXPECT_EQ(Darknet::fp8_resolve_layer_index(3, 5), 3);
	EXPECT_EQ(Darknet::fp8_resolve_layer_index(-1, 5), 4);
	EXPECT_EQ(Darknet::fp8_resolve_layer_index(-5, 5), 0);
}

TEST(Fp8Scaling, RejectsOutOfRangeLayerSkipIndexes)
{
	EXPECT_EQ(Darknet::fp8_resolve_layer_index(5, 5), -1);
	EXPECT_EQ(Darknet::fp8_resolve_layer_index(-6, 5), -1);
	EXPECT_EQ(Darknet::fp8_resolve_layer_index(0, 0), -1);
}

TEST(Fp8Scaling, FindsSkippedLayerFromPositiveAndNegativeIndexes)
{
	const int skip_layers[] = {0, -1, 3};

	EXPECT_TRUE(Darknet::fp8_layer_is_skipped(0, 6, skip_layers, 3));
	EXPECT_TRUE(Darknet::fp8_layer_is_skipped(5, 6, skip_layers, 3));
	EXPECT_TRUE(Darknet::fp8_layer_is_skipped(3, 6, skip_layers, 3));
	EXPECT_FALSE(Darknet::fp8_layer_is_skipped(2, 6, skip_layers, 3));
	EXPECT_FALSE(Darknet::fp8_layer_is_skipped(0, 6, nullptr, 3));
}

TEST(Fp8Scaling, BackwardModeParserDefaultsToFp8)
{
	EXPECT_EQ(Darknet::fp8_backward_mode_from_string(nullptr), Darknet::Fp8BackwardMode::Fp8);
	EXPECT_EQ(Darknet::fp8_backward_mode_from_string(""), Darknet::Fp8BackwardMode::Fp8);
	EXPECT_EQ(Darknet::fp8_backward_mode_from_string("fp8"), Darknet::Fp8BackwardMode::Fp8);
	EXPECT_EQ(Darknet::fp8_backward_mode_from_string("unknown"), Darknet::Fp8BackwardMode::Fp8);
}

TEST(Fp8Scaling, BackwardModeParserAcceptsCudnn)
{
	EXPECT_EQ(Darknet::fp8_backward_mode_from_string("cudnn"), Darknet::Fp8BackwardMode::Cudnn);
	EXPECT_EQ(Darknet::fp8_backward_mode_from_string("CUDNN"), Darknet::Fp8BackwardMode::Cudnn);
}

TEST(Fp8Scaling, NetworkDefaultsKeepFp8TrainingDisabledWithWarmupReady)
{
	Darknet::Network net = make_network(0);

	EXPECT_EQ(net.fp8_training, 0);
	EXPECT_EQ(net.fp8_warmup_iters, 4);
	EXPECT_EQ(net.fp8_skip_layer_count, 0);
	EXPECT_EQ(net.fp8_skip_layers, nullptr);

	free_network(net);
}
