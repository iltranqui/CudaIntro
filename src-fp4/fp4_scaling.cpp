#include "fp4_scaling.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace Darknet
{
	namespace
	{
		constexpr float kFp4E2m1Max = 6.0f;
		constexpr float kFp4E4m3Max = 448.0f;
		constexpr std::array<float, 8> kFp4PositiveValues = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};

		float scalar_satfinite_magnitude(const float value, const float maximum)
		{
			if (std::isnan(value)) return maximum;
			if (std::isinf(value)) return maximum;
			return std::fabs(value);
		}

		float finite_amax_value(const float value)
		{
			return std::isfinite(value) ? std::fabs(value) : 0.0f;
		}

		size_t checked_ceil_div(const size_t value, const size_t divisor)
		{
			if (value > std::numeric_limits<size_t>::max() - (divisor - 1U))
			{
				throw std::overflow_error("FP4 rounded size overflows size_t");
			}
			return (value + divisor - 1U) / divisor;
		}

		size_t checked_multiply(const size_t lhs, const size_t rhs)
		{
			if (lhs != 0U && rhs > std::numeric_limits<size_t>::max() / lhs)
			{
				throw std::overflow_error("FP4 tensor size overflows size_t");
			}
			return lhs * rhs;
		}

		float choose_global_scale(const float amax)
		{
			return amax > 0.0f && std::isfinite(amax) ? amax / (kFp4E2m1Max * kFp4E4m3Max) : 1.0f;
		}

		uint8_t quantized_local_scale(const float amax, const float global_scale)
		{
			if (!(amax > 0.0f) || !(global_scale > 0.0f))
			{
				return fp4_encode_e4m3(1.0f);
			}
			return fp4_encode_e4m3(std::min(kFp4E4m3Max, amax / (global_scale * kFp4E2m1Max)));
		}

		void pack_nibble(std::vector<uint8_t> & packed, const size_t index, const uint8_t nibble)
		{
			const unsigned shift = static_cast<unsigned>((index & 1U) * 4U);
			packed[index / 2U] |= static_cast<uint8_t>((nibble & 0x0fU) << shift);
		}

		uint64_t splitmix64(uint64_t value)
		{
			value += 0x9e3779b97f4a7c15ULL;
			value = (value ^ (value >> 30U)) * 0xbf58476d1ce4e5b9ULL;
			value = (value ^ (value >> 27U)) * 0x94d049bb133111ebULL;
			return value ^ (value >> 31U);
		}
	}

	size_t fp4_packed_byte_count(const size_t element_count) { return checked_ceil_div(element_count, 2U); }
	size_t fp4_block_count(const size_t element_count) { return checked_ceil_div(element_count, kFp4BlockSize); }

	size_t fp4_weight_tile_count(const size_t rows, const size_t columns)
	{
		return checked_multiply(checked_ceil_div(rows, kFp4WeightTileRows),
			checked_ceil_div(columns, kFp4WeightTileColumns));
	}

	uint8_t fp4_encode_e2m1(const float value)
	{
		if (std::isnan(value)) return 0x07U;
		const uint8_t sign = (std::bit_cast<uint32_t>(value) >> 31U) != 0U ? 0x08U : 0U;
		const float magnitude = scalar_satfinite_magnitude(value, kFp4E2m1Max);
		size_t best = 0;
		float best_distance = std::numeric_limits<float>::max();
		for (size_t idx = 0; idx < kFp4PositiveValues.size(); ++idx)
		{
			const float distance = std::fabs(magnitude - kFp4PositiveValues[idx]);
			if (distance < best_distance || (distance == best_distance && (idx & 1U) == 0U))
			{
				best = idx;
				best_distance = distance;
			}
		}
		return static_cast<uint8_t>(sign | best);
	}

	uint8_t fp4_encode_e2m1_stochastic(const float value, const uint64_t seed, const size_t element_index)
	{
		if (std::isnan(value)) return 0x07U;
		const uint8_t sign = (std::bit_cast<uint32_t>(value) >> 31U) != 0U ? 0x08U : 0U;
		const float magnitude = scalar_satfinite_magnitude(value, kFp4E2m1Max);
		if (magnitude >= kFp4E2m1Max) return static_cast<uint8_t>(sign | 0x07U);
		size_t upper = 1U;
		while (upper < kFp4PositiveValues.size() && kFp4PositiveValues[upper] < magnitude) ++upper;
		const size_t lower = upper - 1U;
		const float span = kFp4PositiveValues[upper] - kFp4PositiveValues[lower];
		const float probability_upper = span > 0.0f ? (magnitude - kFp4PositiveValues[lower]) / span : 0.0f;
		const uint64_t random_bits = splitmix64(seed ^ (static_cast<uint64_t>(element_index) * 0x9e3779b97f4a7c15ULL));
		const float uniform = static_cast<float>((random_bits >> 40U) & 0xffffffU) * (1.0f / 16777216.0f);
		return static_cast<uint8_t>(sign | (uniform < probability_upper ? upper : lower));
	}

	float fp4_decode_e2m1(const uint8_t value)
	{
		const float decoded = kFp4PositiveValues[value & 0x07U];
		return (value & 0x08U) != 0U ? -decoded : decoded;
	}

	float fp4_decode_e4m3(const uint8_t value)
	{
		const unsigned magnitude = value & 0x7fU;
		const unsigned exponent = magnitude >> 3U;
		const unsigned mantissa = magnitude & 0x07U;
		if (magnitude == 0x7fU) return std::numeric_limits<float>::quiet_NaN();
		float decoded = exponent == 0U
			? std::ldexp(static_cast<float>(mantissa), -9)
			: std::ldexp(1.0f + static_cast<float>(mantissa) / 8.0f, static_cast<int>(exponent) - 7);
		return (value & 0x80U) != 0U ? -decoded : decoded;
	}

	uint8_t fp4_encode_e4m3(const float value)
	{
		if (std::isnan(value)) return 0x7eU;
		const uint8_t sign = std::signbit(value) ? 0x80U : 0U;
		const float magnitude = scalar_satfinite_magnitude(value, kFp4E4m3Max);
		uint8_t best = 0;
		float best_distance = std::numeric_limits<float>::max();
		for (uint16_t code = 0; code <= 0x7eU; ++code)
		{
			const float distance = std::fabs(magnitude - fp4_decode_e4m3(static_cast<uint8_t>(code)));
			if (distance < best_distance || (distance == best_distance && (code & 1U) == 0U))
			{
				best = static_cast<uint8_t>(code);
				best_distance = distance;
			}
		}
		return static_cast<uint8_t>(sign | best);
	}

	std::vector<uint8_t> fp4_pack_e2m1(const float * values, const size_t element_count)
	{
		if (values == nullptr && element_count != 0U) throw std::invalid_argument("FP4 input is null with nonzero size");
		std::vector<uint8_t> result(fp4_packed_byte_count(element_count), 0U);
		if (values == nullptr) return result;
		for (size_t idx = 0; idx < element_count; ++idx) pack_nibble(result, idx, fp4_encode_e2m1(values[idx]));
		return result;
	}

	Fp4ScaledTensorReference fp4_quantize_1d_reference(const float * values, const size_t element_count)
	{
		if (values == nullptr && element_count != 0U) throw std::invalid_argument("FP4 input is null with nonzero size");
		Fp4ScaledTensorReference result;
		result.values.assign(fp4_packed_byte_count(element_count), 0U);
		result.local_scales_e4m3.assign(fp4_block_count(element_count), fp4_encode_e4m3(1.0f));
		if (values == nullptr || element_count == 0U) return result;
		float global_amax = 0.0f;
		for (size_t idx = 0; idx < element_count; ++idx) global_amax = std::max(global_amax, finite_amax_value(values[idx]));
		result.global_scale = choose_global_scale(global_amax);
		for (size_t block = 0; block < result.local_scales_e4m3.size(); ++block)
		{
			const size_t begin = block * kFp4BlockSize;
			const size_t end = std::min(element_count, begin + kFp4BlockSize);
			float amax = 0.0f;
			for (size_t idx = begin; idx < end; ++idx) amax = std::max(amax, finite_amax_value(values[idx]));
			result.local_scales_e4m3[block] = quantized_local_scale(amax, result.global_scale);
			const float divisor = result.global_scale * fp4_decode_e4m3(result.local_scales_e4m3[block]);
			for (size_t idx = begin; idx < end; ++idx) pack_nibble(result.values, idx, fp4_encode_e2m1(values[idx] / divisor));
		}
		return result;
	}

	Fp4ScaledWeightsReference fp4_quantize_weights_2d_reference(const float * values, const size_t rows, const size_t columns)
	{
		const size_t element_count = checked_multiply(rows, columns);
		if (values == nullptr && element_count != 0U) throw std::invalid_argument("FP4 weight input is null with nonzero dimensions");
		Fp4ScaledWeightsReference result;
		result.rows = rows;
		result.columns = columns;
		result.tile_rows = checked_ceil_div(rows, kFp4WeightTileRows);
		result.tile_columns = checked_ceil_div(columns, kFp4WeightTileColumns);
		result.values.assign(fp4_packed_byte_count(element_count), 0U);
		result.local_scales_e4m3.assign(checked_multiply(result.tile_rows, result.tile_columns), fp4_encode_e4m3(1.0f));
		if (values == nullptr || rows == 0U || columns == 0U) return result;
		float global_amax = 0.0f;
		for (size_t idx = 0; idx < element_count; ++idx) global_amax = std::max(global_amax, finite_amax_value(values[idx]));
		result.global_scale = choose_global_scale(global_amax);
		for (size_t tile_row = 0; tile_row < result.tile_rows; ++tile_row)
		{
			for (size_t tile_column = 0; tile_column < result.tile_columns; ++tile_column)
			{
				const size_t tile = tile_row * result.tile_columns + tile_column;
				const size_t row_end = std::min(rows, (tile_row + 1U) * kFp4WeightTileRows);
				const size_t column_end = std::min(columns, (tile_column + 1U) * kFp4WeightTileColumns);
				float amax = 0.0f;
				for (size_t row = tile_row * kFp4WeightTileRows; row < row_end; ++row)
					for (size_t column = tile_column * kFp4WeightTileColumns; column < column_end; ++column)
						amax = std::max(amax, finite_amax_value(values[row * columns + column]));
				result.local_scales_e4m3[tile] = quantized_local_scale(amax, result.global_scale);
				const float divisor = result.global_scale * fp4_decode_e4m3(result.local_scales_e4m3[tile]);
				for (size_t row = tile_row * kFp4WeightTileRows; row < row_end; ++row)
					for (size_t column = tile_column * kFp4WeightTileColumns; column < column_end; ++column)
					{
						const size_t idx = row * columns + column;
						pack_nibble(result.values, idx, fp4_encode_e2m1(values[idx] / divisor));
					}
			}
		}
		return result;
	}

	std::array<float, kFp4BlockSize> fp4_rht16_reference(const std::array<float, kFp4BlockSize> & values, const uint64_t seed)
	{
		std::array<float, kFp4BlockSize> result = values;
		std::array<float, kFp4BlockSize> signs = {};
		for (size_t idx = 0; idx < result.size(); ++idx)
		{
			signs[idx] = (splitmix64(seed + idx) & 1U) != 0U ? -1.0f : 1.0f;
			result[idx] *= signs[idx];
		}
		for (size_t width = 1; width < result.size(); width *= 2U)
			for (size_t begin = 0; begin < result.size(); begin += 2U * width)
				for (size_t offset = 0; offset < width; ++offset)
				{
					const float lhs = result[begin + offset];
					const float rhs = result[begin + offset + width];
					result[begin + offset] = lhs + rhs;
					result[begin + offset + width] = lhs - rhs;
				}
		for (size_t idx = 0; idx < result.size(); ++idx) result[idx] *= signs[idx] * 0.25f;
		return result;
	}
}
