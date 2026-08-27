#pragma once

#include <array>

namespace Darknet
{
	constexpr int kFp8AmaxHistoryLength = 16;

	/// Extra headroom applied to the E5M2 gradient scale:  the observed amax is mapped to
	/// fp8_max / 2^margin instead of fp8_max, so an activation spike beyond the recorded
	/// history window still lands inside the representable range instead of saturating.
	constexpr int kFp8DyScaleMargin = 1;

	enum class Fp8Format
	{
		E4M3,
		E5M2
	};

	enum class Fp8BackwardMode
	{
		Fp8,
		Cudnn
	};

	struct Fp8DelayedScalingState
	{
		std::array<float, kFp8AmaxHistoryLength> amax_history = {};
		int next_index = 0;
		int valid_count = 0;
		float scale = 1.0f;
	};

	float fp8_format_max(Fp8Format format);
	float fp8_delayed_scale_from_history(const std::array<float, kFp8AmaxHistoryLength> & history, Fp8Format format, int margin = 0);
	float fp8_delayed_scale_from_history(const float * history, int history_length, Fp8Format format, int margin = 0);
	void fp8_delayed_scaling_record_amax(Fp8DelayedScalingState & state, float amax, Fp8Format format, int margin = 0);
	void fp8_delayed_scaling_record_amax(float * history, int history_length, int & next_index, int & valid_count, float & scale, float amax, Fp8Format format, int margin = 0);
	Fp8BackwardMode fp8_backward_mode_from_string(const char * value);
	Fp8BackwardMode fp8_backward_mode_from_env();
	int fp8_resolve_layer_index(int requested_layer_index, int layer_count);
	bool fp8_layer_is_skipped(int layer_index, int layer_count, const int * skip_layers, int skip_layer_count);
}
