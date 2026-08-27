#include "fp8_calibration.hpp"
#include "fp_calibration_io.hpp"

#include <cmath>

namespace Darknet
{
	std::filesystem::path fp8_scales_sidecar_path(const std::filesystem::path & weights_path)
	{
		return std::filesystem::path(weights_path.string() + ".fp8scales");
	}

	float fp8_scale_from_amax(const float amax)
	{
		if (!std::isfinite(amax) || amax <= 0.0f)
		{
			return 1.0f;
		}
		return amax / kFp8E4m3Max;
	}

	bool fp8_read_calibration_scales(const std::filesystem::path & path, Fp8CalibrationTable & table)
	{
		return read_calibration_scales_impl(path, table);
	}

	bool fp8_write_calibration_scales(const std::filesystem::path & path, const std::vector<Fp8CalibrationEntry> & entries)
	{
		return write_calibration_scales_impl(path, entries);
	}
}
