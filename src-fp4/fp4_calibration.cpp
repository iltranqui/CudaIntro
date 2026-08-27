#include "fp4_calibration.hpp"
#include "fp_calibration_io.hpp"

#include <cmath>

namespace Darknet
{
	std::filesystem::path fp4_scales_sidecar_path(const std::filesystem::path & weights_path)
	{
		return std::filesystem::path(weights_path.string() + ".fp4scales");
	}

	float fp4_scale_from_amax(const float amax)
	{
		if (!std::isfinite(amax) || amax <= 0.0f)
		{
			return 1.0f;
		}
		return amax / kFp4GlobalScaleDenominator;
	}

	bool fp4_read_calibration_scales(const std::filesystem::path & path, Fp4CalibrationTable & table)
	{
		return read_calibration_scales_impl(path, table);
	}

	bool fp4_write_calibration_scales(const std::filesystem::path & path, const std::vector<Fp4CalibrationEntry> & entries)
	{
		return write_calibration_scales_impl(path, entries);
	}
}
