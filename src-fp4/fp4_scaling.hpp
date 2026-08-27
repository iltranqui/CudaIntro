#pragma once

#include "fp4_types.hpp"

#include <array>

namespace Darknet
{
	size_t fp4_packed_byte_count(size_t element_count);
	size_t fp4_block_count(size_t element_count);
	size_t fp4_weight_tile_count(size_t rows, size_t columns);

	uint8_t fp4_encode_e2m1(float value);
	uint8_t fp4_encode_e2m1_stochastic(float value, uint64_t seed, size_t element_index);
	float fp4_decode_e2m1(uint8_t value);
	uint8_t fp4_encode_e4m3(float value);
	float fp4_decode_e4m3(uint8_t value);
	std::vector<uint8_t> fp4_pack_e2m1(const float * values, size_t element_count);

	Fp4ScaledTensorReference fp4_quantize_1d_reference(const float * values, size_t element_count);
	Fp4ScaledWeightsReference fp4_quantize_weights_2d_reference(const float * values, size_t rows, size_t columns);

	std::array<float, kFp4BlockSize> fp4_rht16_reference(
		const std::array<float, kFp4BlockSize> & values, uint64_t seed);
}
