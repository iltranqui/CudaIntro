#include "fp8_scaling.hpp"

#include <algorithm>
#include <cstdlib>
#include <cmath>
#include <limits>

namespace Darknet
{
	namespace
	{
		bool fp8_valid_positive_amax(const float value)
		{
			return value > 0.0f && value <= std::numeric_limits<float>::max();
		}

		bool fp8_ascii_equal_ignore_case(const char * value, const char * expected)
		{
			if (value == nullptr || expected == nullptr)
			{
				return false;
			}

			for (int idx = 0; value[idx] != '\0' || expected[idx] != '\0'; ++idx)
			{
				char lhs = value[idx];
				char rhs = expected[idx];
				if (lhs >= 'A' && lhs <= 'Z')
				{
					lhs = static_cast<char>(lhs - 'A' + 'a');
				}
				if (rhs >= 'A' && rhs <= 'Z')
				{
					rhs = static_cast<char>(rhs - 'A' + 'a');
				}
				if (lhs != rhs)
				{
					return false;
				}
			}
			return true;
		}
	}

	float fp8_format_max(const Fp8Format format)
	{
		switch (format)
		{
			case Fp8Format::E4M3: return 448.0f;
			case Fp8Format::E5M2: return 57344.0f;
		}
		return 448.0f;
	}

	float fp8_delayed_scale_from_history(const std::array<float, kFp8AmaxHistoryLength> & history, const Fp8Format format, const int margin)
	{
		return fp8_delayed_scale_from_history(history.data(), static_cast<int>(history.size()), format, margin);
	}

	float fp8_delayed_scale_from_history(const float * history, const int history_length, const Fp8Format format, const int margin)
	{
		float amax = 0.0f;
		if (history == nullptr || history_length <= 0)
		{
			return 1.0f;
		}

		for (int idx = 0; idx < history_length; ++idx)
		{
			const float value = history[idx];
			if (fp8_valid_positive_amax(value))
			{
				amax = std::max(amax, value);
			}
		}

		const float max_value = fp8_format_max(format);
		if (amax <= 0.0f || max_value <= 0.0f)
		{
			return 1.0f;
		}
		// map amax to max_value / 2^margin so spikes beyond the history window keep headroom
		return std::ldexp(amax, std::max(0, margin)) / max_value;
	}

	void fp8_delayed_scaling_record_amax(Fp8DelayedScalingState & state, const float amax, const Fp8Format format, const int margin)
	{
		fp8_delayed_scaling_record_amax(
			state.amax_history.data(),
			static_cast<int>(state.amax_history.size()),
			state.next_index,
			state.valid_count,
			state.scale,
			amax,
			format,
			margin);
	}

	void fp8_delayed_scaling_record_amax(float * history, const int history_length, int & next_index, int & valid_count, float & scale, const float amax, const Fp8Format format, const int margin)
	{
		if (history == nullptr || history_length <= 0)
		{
			scale = 1.0f;
			next_index = 0;
			valid_count = 0;
			return;
		}

		if (next_index < 0 || next_index >= history_length)
		{
			next_index = 0;
		}
		history[next_index] = fp8_valid_positive_amax(amax) ? amax : 0.0f;
		next_index = (next_index + 1) % history_length;
		if (valid_count < history_length)
		{
			valid_count += 1;
		}
		scale = fp8_delayed_scale_from_history(history, history_length, format, margin);
	}

	Fp8BackwardMode fp8_backward_mode_from_string(const char * value)
	{
		if (value == nullptr || value[0] == '\0')
		{
			return Fp8BackwardMode::Fp8;
		}
		if (fp8_ascii_equal_ignore_case(value, "cudnn"))
		{
			return Fp8BackwardMode::Cudnn;
		}
		return Fp8BackwardMode::Fp8;
	}

	Fp8BackwardMode fp8_backward_mode_from_env()
	{
		return fp8_backward_mode_from_string(std::getenv("DARKNET_FP8_BACKWARD_MODE"));
	}

	int fp8_resolve_layer_index(const int requested_layer_index, const int layer_count)
	{
		if (layer_count <= 0)
		{
			return -1;
		}

		const int resolved = requested_layer_index < 0 ? layer_count + requested_layer_index : requested_layer_index;
		if (resolved < 0 || resolved >= layer_count)
		{
			return -1;
		}
		return resolved;
	}

	bool fp8_layer_is_skipped(const int layer_index, const int layer_count, const int * skip_layers, const int skip_layer_count)
	{
		if (skip_layers == nullptr || skip_layer_count <= 0)
		{
			return false;
		}

		const int resolved_layer_index = fp8_resolve_layer_index(layer_index, layer_count);
		if (resolved_layer_index < 0)
		{
			return false;
		}

		for (int i = 0; i < skip_layer_count; ++i)
		{
			if (fp8_resolve_layer_index(skip_layers[i], layer_count) == resolved_layer_index)
			{
				return true;
			}
		}
		return false;
	}
}
