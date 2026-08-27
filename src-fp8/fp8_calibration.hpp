#pragma once

#include <filesystem>
#include <map>
#include <vector>

namespace Darknet
{
	constexpr float kFp8E4m3Max = 448.0f;

	struct Fp8CalibrationEntry
	{
		int layer_index = -1;
		float amax = 0.0f;
		float scale = 1.0f;
	};

	using Fp8CalibrationTable = std::map<int, Fp8CalibrationEntry>;

	std::filesystem::path fp8_scales_sidecar_path(const std::filesystem::path & weights_path);
	float fp8_scale_from_amax(float amax);
	bool fp8_read_calibration_scales(const std::filesystem::path & path, Fp8CalibrationTable & table);
	bool fp8_write_calibration_scales(const std::filesystem::path & path, const std::vector<Fp8CalibrationEntry> & entries);
}
