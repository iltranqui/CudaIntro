#pragma once

#include <filesystem>
#include <map>
#include <vector>

namespace Darknet
{
	// Matches choose_global_scale()'s denominator in fp4_scaling.cpp: the largest
	// representable NVFP4 magnitude is kFp4E2m1Max (E2M1 payload) scaled by a
	// kFp4E4m3Max (E4M3 block scale) factor.
	constexpr float kFp4GlobalScaleDenominator = 6.0f * 448.0f;

	struct Fp4CalibrationEntry
	{
		int layer_index = -1;
		float amax = 0.0f;
		float scale = 1.0f;
	};

	using Fp4CalibrationTable = std::map<int, Fp4CalibrationEntry>;

	std::filesystem::path fp4_scales_sidecar_path(const std::filesystem::path & weights_path);
	float fp4_scale_from_amax(float amax);
	bool fp4_read_calibration_scales(const std::filesystem::path & path, Fp4CalibrationTable & table);
	bool fp4_write_calibration_scales(const std::filesystem::path & path, const std::vector<Fp4CalibrationEntry> & entries);
}
