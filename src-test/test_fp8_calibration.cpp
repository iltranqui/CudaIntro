#include <gtest/gtest.h>

#include <chrono>
#include <cmath>
#include <filesystem>
#include <vector>

#include "fp8_calibration.hpp"

namespace
{
	std::filesystem::path temp_scales_path()
	{
		const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
		return std::filesystem::temp_directory_path() / ("darknet_fp8_" + std::to_string(stamp) + ".weights.fp8scales");
	}
}

TEST(Fp8Calibration, SidecarPathAppendsFp8Scales)
{
	const std::filesystem::path weights = "/tmp/model.weights";
	EXPECT_EQ(Darknet::fp8_scales_sidecar_path(weights), "/tmp/model.weights.fp8scales");
}

TEST(Fp8Calibration, ScaleFromAmaxUsesE4m3Range)
{
	EXPECT_FLOAT_EQ(Darknet::fp8_scale_from_amax(448.0f), 1.0f);
	EXPECT_FLOAT_EQ(Darknet::fp8_scale_from_amax(224.0f), 0.5f);
	EXPECT_FLOAT_EQ(Darknet::fp8_scale_from_amax(0.0f), 1.0f);
}

TEST(Fp8Calibration, WritesAndReadsLayerScaleSidecar)
{
	const std::filesystem::path path = temp_scales_path();
	const std::vector<Darknet::Fp8CalibrationEntry> entries =
	{
		{3, 448.0f, 1.0f},
		{7, 112.0f, 0.25f},
	};

	ASSERT_TRUE(Darknet::fp8_write_calibration_scales(path, entries));

	Darknet::Fp8CalibrationTable table;
	ASSERT_TRUE(Darknet::fp8_read_calibration_scales(path, table));
	ASSERT_EQ(table.size(), 2);
	ASSERT_TRUE(table.count(3));
	ASSERT_TRUE(table.count(7));
	EXPECT_FLOAT_EQ(table.at(3).amax, 448.0f);
	EXPECT_FLOAT_EQ(table.at(3).scale, 1.0f);
	EXPECT_FLOAT_EQ(table.at(7).amax, 112.0f);
	EXPECT_FLOAT_EQ(table.at(7).scale, 0.25f);

	std::filesystem::remove(path);
}
