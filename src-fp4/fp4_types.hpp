#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace Darknet
{
	constexpr size_t kFp4BlockSize = 16;
	constexpr size_t kFp4WeightTileRows = 16;
	constexpr size_t kFp4WeightTileColumns = 16;

	/// Host-side representation used as the numerical contract for CUDA kernels.
	/// FP32 remains authoritative; values and scales are transient execution data.
	struct Fp4ScaledTensorReference
	{
		std::vector<uint8_t> values;
		std::vector<uint8_t> local_scales_e4m3;
		float global_scale = 1.0f;
	};

	struct Fp4ScaledWeightsReference : Fp4ScaledTensorReference
	{
		size_t rows = 0;
		size_t columns = 0;
		size_t tile_rows = 0;
		size_t tile_columns = 0;
	};
}
