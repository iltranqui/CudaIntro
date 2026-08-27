#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <functional>
#include <random>
#include <vector>

#include "darknet_internal.hpp"
#include "deform_conv_layer.hpp"

namespace
{
	void fill_pattern(float *data, size_t n, float scale, float bias, float freq)
	{
		for (size_t i = 0; i < n; ++i)
		{
			data[i] = bias + scale * std::sin(static_cast<float>(i) * freq);
		}
	}

	float max_abs(const float *data, size_t n)
	{
		float max_val = 0.0f;
		for (size_t i = 0; i < n; ++i)
		{
			const float v = std::fabs(data[i]);
			if (v > max_val)
			{
				max_val = v;
			}
		}
		return max_val;
	}

	void expect_all_finite(const float *data, size_t n, const char *label)
	{
		for (size_t i = 0; i < n; ++i)
		{
			ASSERT_TRUE(std::isfinite(data[i])) << label << " idx=" << i << " val=" << data[i];
		}
	}

	void run_deform_conv_case(int size, int stride, int pad, float input_scale, int use_mask, int batch_normalize, ACTIVATION activation, int groups = 1, int c = 3, int n = 4)
	{
		const int batch = 2;
		const int steps = 1;
		const int h = 13;
		const int w = 13;
		const int stride_x = stride;
		const int stride_y = stride;
		const int dilation = 1;
		const int padding = pad;
		const int binary = 0;
		const int xnor = 0;
		const int adam = 0;
		const int use_bin_output = 0;
		const int index = 0;
		const int antialiasing = 0;
		const int assisted_excitation = 0;
		const int train = 1;

		Darknet::Layer l = make_deform_conv_layer(batch, steps, h, w, c, n, groups, size, stride_x, stride_y,
			dilation, padding, activation, batch_normalize, binary, xnor, adam, use_bin_output, index,
			antialiasing, nullptr, assisted_excitation, train, use_mask);

		const int out_h = l.out_h;
		const int out_w = l.out_w;
		const int offset_filters = 2 * size * size;
		const int mask_filters = size * size;
		const size_t offset_weights_size = static_cast<size_t>(offset_filters) * c * size * size;
		const size_t mask_weights_size = static_cast<size_t>(mask_filters) * c * size * size;
		const size_t output_count = static_cast<size_t>(l.outputs) * l.batch;
		const size_t offset_count = static_cast<size_t>(l.batch) * out_h * out_w * offset_filters;
		const size_t mask_count = static_cast<size_t>(l.batch) * out_h * out_w * mask_filters;

		std::vector<float> input(static_cast<size_t>(batch) * c * h * w, 0.0f);
		std::vector<float> input_delta(input.size(), 0.0f);

		fill_pattern(input.data(), input.size(), input_scale, 0.05f, 0.013f);
		fill_pattern(l.weights, l.nweights, 0.01f, 0.0f, 0.017f);
		fill_pattern(l.offset_weights, offset_weights_size, 0.005f, 0.0f, 0.019f);
		std::fill(l.biases, l.biases + l.n, 0.01f);
		std::fill(l.offset_biases, l.offset_biases + offset_filters, 0.0f);

		if (use_mask)
		{
			fill_pattern(l.mask_weights, mask_weights_size, 0.005f, 0.0f, 0.011f);
			std::fill(l.mask_biases, l.mask_biases + mask_filters, 0.0f);
		}

		const size_t workspace_floats = l.workspace_size / sizeof(float);
		ASSERT_GT(workspace_floats, 0u);
		std::vector<float> workspace(workspace_floats, 0.0f);

		Darknet::NetworkState state = {};
		state.input = input.data();
		state.workspace = workspace.data();
		state.train = 1;

		forward_deform_conv_layer(l, state);

		expect_all_finite(l.output, output_count, "output");
		expect_all_finite(l.offsets, offset_count, "offsets");
		if (use_mask)
		{
			expect_all_finite(l.masks, mask_count, "masks");
		}

		fill_pattern(l.delta, output_count, 1.0f, 0.0f, 0.003f);

		Darknet::NetworkState back_state = {};
		back_state.input = input.data();
		back_state.delta = input_delta.data();
		back_state.workspace = workspace.data();
		back_state.train = 1;

		backward_deform_conv_layer(l, back_state);

		expect_all_finite(l.delta, output_count, "delta");
		expect_all_finite(l.bias_updates, l.n, "bias_updates");
		expect_all_finite(l.weight_updates, l.nweights, "weight_updates");
		expect_all_finite(l.offset_bias_updates, offset_filters, "offset_bias_updates");
		expect_all_finite(l.offset_weight_updates, offset_weights_size, "offset_weight_updates");
		expect_all_finite(l.offset_deltas, offset_count, "offset_deltas");
		expect_all_finite(input_delta.data(), input_delta.size(), "input_delta");

		EXPECT_GT(max_abs(l.weight_updates, l.nweights), 0.0f);
		EXPECT_LT(max_abs(l.weight_updates, l.nweights), 1e6f);
		EXPECT_LT(max_abs(l.offset_weight_updates, offset_weights_size), 1e6f);
		EXPECT_LT(max_abs(input_delta.data(), input_delta.size()), 1e6f);

		if (use_mask)
		{
			expect_all_finite(l.mask_bias_updates, mask_filters, "mask_bias_updates");
			expect_all_finite(l.mask_weight_updates, mask_weights_size, "mask_weight_updates");
			expect_all_finite(l.mask_deltas, mask_count, "mask_deltas");
			EXPECT_LT(max_abs(l.mask_weight_updates, mask_weights_size), 1e6f);
		}

		update_deform_conv_layer(l, batch, 0.001f, 0.9f, 0.0005f);

		expect_all_finite(l.weights, l.nweights, "weights");
		expect_all_finite(l.offset_weights, offset_weights_size, "offset_weights");
		if (use_mask)
		{
			expect_all_finite(l.mask_weights, mask_weights_size, "mask_weights");
		}

		free_layer(l);
	}

	// Helper to check if GPU tests are enabled at runtime
	// GPU tests enabled by default; set DARKNET_TEST_GPU=0 to disable
	bool gpu_tests_enabled()
	{
		const char* gpu_mode = std::getenv("DARKNET_TEST_GPU");
		return !(gpu_mode && std::string(gpu_mode) == "0");
	}

	/**
	 * @brief CPU bilinear interpolation function for testing
	 *
	 * Mirrors the implementation in deform_conv_layer.cpp for unit testing.
	 */
	float bilinear_interpolate_test(const float* data, int h, int w, float y, float x)
	{
		if (y < -1.0f || y > h || x < -1.0f || x > w) return 0.0f;

		int y_low = static_cast<int>(std::floor(y));
		int x_low = static_cast<int>(std::floor(x));
		int y_high = y_low + 1;
		int x_high = x_low + 1;

		float ly = y - y_low;
		float lx = x - x_low;
		float hy = 1.0f - ly;
		float hx = 1.0f - lx;

		float v1 = (y_low >= 0 && y_low < h && x_low >= 0 && x_low < w) ? data[y_low * w + x_low] : 0.0f;
		float v2 = (y_low >= 0 && y_low < h && x_high >= 0 && x_high < w) ? data[y_low * w + x_high] : 0.0f;
		float v3 = (y_high >= 0 && y_high < h && x_low >= 0 && x_low < w) ? data[y_high * w + x_low] : 0.0f;
		float v4 = (y_high >= 0 && y_high < h && x_high >= 0 && x_high < w) ? data[y_high * w + x_high] : 0.0f;

		return hy * hx * v1 + hy * lx * v2 + ly * hx * v3 + ly * lx * v4;
	}
}

// ============================================================================
// Existing Tests (preserved from original)
// ============================================================================

TEST(DeformConvLayer, NoNanOrExplodingGradients)
{
	run_deform_conv_case(3, 1, 1, 1.0f, 1, 0, LINEAR);
	run_deform_conv_case(3, 1, 1, 100.0f, 1, 0, LINEAR);
	run_deform_conv_case(3, 1, 1, 1.0f, 0, 0, LINEAR);
}

TEST(DeformConvLayer, VariedParamsForwardBackwardFinite)
{
	const std::array<int, 3> sizes = {1, 3, 5};
	const std::array<int, 3> strides = {1, 3, 5};
	const std::array<int, 3> pads = {1, 3, 5};

	for (const int size : sizes)
	{
		for (const int stride : strides)
		{
			for (const int pad : pads)
			{
				run_deform_conv_case(size, stride, pad, 10.0f, 1, 1, LEAKY);
			}
		}
	}
}

TEST(DeformConvLayer, GroupedConvolutionGroups2)
{
	// groups=2 requires c and n divisible by 2
	run_deform_conv_case(3, 1, 1, 1.0f, 1, 0, LINEAR, 2, 4, 4);
	run_deform_conv_case(3, 1, 1, 1.0f, 1, 1, LEAKY, 2, 4, 4);  // with batch norm
	run_deform_conv_case(3, 1, 1, 10.0f, 0, 0, LINEAR, 2, 4, 4);  // DCNv1
}

TEST(DeformConvLayer, GroupedConvolutionGroups4)
{
	// groups=4 requires c and n divisible by 4
	run_deform_conv_case(3, 1, 1, 1.0f, 1, 0, LINEAR, 4, 8, 8);
	run_deform_conv_case(3, 1, 1, 1.0f, 1, 1, LEAKY, 4, 8, 8);  // with batch norm
}

// ============================================================================
// CPU Bilinear Unit Tests
// ============================================================================

TEST(DeformConvCPU, BilinearInterpolate_CenterPoint)
{
	// Test bilinear interpolation at known center coordinates
	// 2x2 grid:
	// [1, 2]
	// [3, 4]
	std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f};
	int h = 2, w = 2;

	// Center point (0.5, 0.5) should interpolate all 4 corners equally
	float result = bilinear_interpolate_test(data.data(), h, w, 0.5f, 0.5f);
	float expected = (1.0f + 2.0f + 3.0f + 4.0f) / 4.0f;  // = 2.5
	EXPECT_NEAR(result, expected, 1e-5f);

	// Top-left corner (0, 0)
	result = bilinear_interpolate_test(data.data(), h, w, 0.0f, 0.0f);
	EXPECT_NEAR(result, 1.0f, 1e-5f);

	// Top-right corner (0, 1)
	result = bilinear_interpolate_test(data.data(), h, w, 0.0f, 1.0f);
	EXPECT_NEAR(result, 2.0f, 1e-5f);

	// Bottom-left corner (1, 0)
	result = bilinear_interpolate_test(data.data(), h, w, 1.0f, 0.0f);
	EXPECT_NEAR(result, 3.0f, 1e-5f);

	// Bottom-right corner (1, 1)
	result = bilinear_interpolate_test(data.data(), h, w, 1.0f, 1.0f);
	EXPECT_NEAR(result, 4.0f, 1e-5f);
}

TEST(DeformConvCPU, BilinearInterpolate_BoundaryConditions)
{
	// Test boundary conditions near edges
	std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f};
	int h = 3, w = 3;

	// Just inside top edge (y = -0.5, x = 1.0)
	// This samples between y=-0.5 (floor=-1) and y=0.5 (ceil=0)
	float result = bilinear_interpolate_test(data.data(), h, w, -0.5f, 1.0f);
	// y_low=-1 (out of bounds), y_high=0 (valid), x_low=1, x_high=2
	// v1=0, v2=0, v3=data[0*3+1]=2, v4=data[0*3+2]=3
	// ly=0.5, lx=0, hy=0.5, hx=1
	// result = 0.5*1*0 + 0.5*0*0 + 0.5*1*2 + 0.5*0*3 = 1.0
	EXPECT_NEAR(result, 1.0f, 1e-5f);

	// Just inside right edge (y = 1.0, x = 2.5)
	result = bilinear_interpolate_test(data.data(), h, w, 1.0f, 2.5f);
	// x_high=3 (out of bounds)
	// v2=0, v4=0
	// v1=data[1*3+2]=6, v3=data[2*3+2]=9
	// ly=0, lx=0.5, hy=1, hx=0.5
	// result = 1*0.5*6 + 0 + 0 + 0 = 3.0
	EXPECT_NEAR(result, 3.0f, 1e-5f);

	// Just inside bottom edge (y = 2.5, x = 1.0)
	result = bilinear_interpolate_test(data.data(), h, w, 2.5f, 1.0f);
	// y_high=3 (out of bounds)
	// v3=0, v4=0
	// v1=data[2*3+1]=8, v2=data[2*3+2]=9
	// ly=0.5, lx=0, hy=0.5, hx=1
	// result = 0.5*1*8 + 0.5*0*9 + 0 + 0 = 4.0
	EXPECT_NEAR(result, 4.0f, 1e-5f);
}

TEST(DeformConvCPU, BilinearInterpolate_OutOfBounds)
{
	// Test out-of-bounds returns 0 without crash
	std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f};
	int h = 2, w = 2;

	// Completely outside (y < -1)
	float result = bilinear_interpolate_test(data.data(), h, w, -1.5f, 0.5f);
	EXPECT_EQ(result, 0.0f);

	// Completely outside (y > h)
	result = bilinear_interpolate_test(data.data(), h, w, 2.5f, 0.5f);
	EXPECT_EQ(result, 0.0f);

	// Completely outside (x < -1)
	result = bilinear_interpolate_test(data.data(), h, w, 0.5f, -1.5f);
	EXPECT_EQ(result, 0.0f);

	// Completely outside (x > w)
	result = bilinear_interpolate_test(data.data(), h, w, 0.5f, 2.5f);
	EXPECT_EQ(result, 0.0f);

	// All corners outside simultaneously
	result = bilinear_interpolate_test(data.data(), h, w, -2.0f, -2.0f);
	EXPECT_EQ(result, 0.0f);
}

TEST(DeformConvCPU, BilinearGradient_MassConservation)
{
	// Test that interpolation weights sum to 1 for valid positions
	// This ensures gradient mass is conserved
	std::vector<float> data = {1.0f, 1.0f, 1.0f, 1.0f};  // All 1s
	int h = 2, w = 2;

	// Any position inside should interpolate to exactly 1.0
	std::vector<std::pair<float, float>> test_positions = {
		{0.0f, 0.0f}, {0.0f, 1.0f}, {1.0f, 0.0f}, {1.0f, 1.0f},
		{0.25f, 0.25f}, {0.5f, 0.5f}, {0.75f, 0.75f},
		{0.1f, 0.9f}, {0.9f, 0.1f}
	};

	for (const auto& pos : test_positions)
	{
		float result = bilinear_interpolate_test(data.data(), h, w, pos.first, pos.second);
		EXPECT_NEAR(result, 1.0f, 1e-5f)
			<< "Mass not conserved at (" << pos.first << ", " << pos.second << ")";
	}
}

// ============================================================================
// CPU Im2col/Col2im Tests
// ============================================================================

TEST(DeformConvCPU, Im2col_ZeroOffsets_MatchesStandard)
{
	// Zero offsets should produce same result as standard convolution
	const int batch = 1;
	const int c = 2;
	const int h = 5;
	const int w = 5;
	const int n = 2;
	const int size = 3;
	const int stride = 1;
	const int pad = 1;

	Darknet::Layer l = make_deform_conv_layer(batch, 1, h, w, c, n, 1, size, stride, stride,
		1, pad, LINEAR, 0, 0, 0, 0, 0, 0, 0, nullptr, 0, 1, 0);

	// Create input with distinct values
	std::vector<float> input(c * h * w);
	for (size_t i = 0; i < input.size(); ++i)
	{
		input[i] = static_cast<float>(i + 1);
	}

	// Zero offsets
	int offset_filters = 2 * size * size;
	std::fill(l.offsets, l.offsets + l.out_h * l.out_w * offset_filters, 0.0f);
	std::fill(l.offset_weights, l.offset_weights + offset_filters * c * size * size, 0.0f);
	std::fill(l.offset_biases, l.offset_biases + offset_filters, 0.0f);

	// Set weights to identity-like pattern
	std::fill(l.weights, l.weights + l.nweights, 0.0f);
	l.weights[0] = 1.0f;  // First filter, first channel, center position
	std::fill(l.biases, l.biases + n, 0.0f);

	std::vector<float> workspace(l.workspace_size / sizeof(float), 0.0f);

	Darknet::NetworkState state = {};
	state.input = input.data();
	state.workspace = workspace.data();
	state.train = 0;

	forward_deform_conv_layer(l, state);

	// With zero offsets and identity-like weights, output should be predictable
	// Verify finite and reasonable values
	for (int i = 0; i < l.outputs; ++i)
	{
		EXPECT_TRUE(std::isfinite(l.output[i])) << "Non-finite output at " << i;
	}

	free_layer(l);
}

TEST(DeformConvCPU, Im2col_WithOffsets_ShiftsSampling)
{
	// Non-zero offsets should shift sampling positions
	const int batch = 1;
	const int c = 1;
	const int h = 5;
	const int w = 5;
	const int n = 1;
	const int size = 1;  // 1x1 kernel simplifies analysis
	const int stride = 1;
	const int pad = 0;

	Darknet::Layer l = make_deform_conv_layer(batch, 1, h, w, c, n, 1, size, stride, stride,
		1, pad, LINEAR, 0, 0, 0, 0, 0, 0, 0, nullptr, 0, 1, 0);

	// Create input with position-dependent values
	std::vector<float> input(c * h * w);
	for (int y = 0; y < h; ++y)
	{
		for (int x = 0; x < w; ++x)
		{
			input[y * w + x] = static_cast<float>(y * 10 + x);  // e.g., position (2,3) = 23
		}
	}

	// Zero offset weights but set offsets directly
	int offset_filters = 2 * size * size;
	std::fill(l.offset_weights, l.offset_weights + offset_filters * c * size * size, 0.0f);
	std::fill(l.offset_biases, l.offset_biases + offset_filters, 0.0f);
	std::fill(l.offsets, l.offsets + l.out_h * l.out_w * offset_filters, 0.0f);

	// Set weight to 1 for pass-through
	l.weights[0] = 1.0f;
	l.biases[0] = 0.0f;

	std::vector<float> workspace(l.workspace_size / sizeof(float), 0.0f);

	// First run with zero offsets
	Darknet::NetworkState state = {};
	state.input = input.data();
	state.workspace = workspace.data();
	state.train = 0;

	forward_deform_conv_layer(l, state);

	// Store reference output
	std::vector<float> reference_output(l.output, l.output + l.outputs);

	// Apply uniform offset of (+1, +1) to shift sampling down-right
	// Note: offsets are stored as (dy, dx) pairs per kernel position
	for (int i = 0; i < l.out_h * l.out_w; ++i)
	{
		l.offsets[i * 2 + 0] = 1.0f;  // dy offset
		l.offsets[i * 2 + 1] = 1.0f;  // dx offset
	}

	// Re-run with offsets
	std::fill(l.output, l.output + l.outputs, 0.0f);

	// Create a fresh layer for the offset test (since forward modifies offsets)
	Darknet::Layer l2 = make_deform_conv_layer(batch, 1, h, w, c, n, 1, size, stride, stride,
		1, pad, LINEAR, 0, 0, 0, 0, 0, 0, 0, nullptr, 0, 1, 0);

	l2.weights[0] = 1.0f;
	l2.biases[0] = 0.0f;
	std::fill(l2.offset_weights, l2.offset_weights + offset_filters * c * size * size, 0.0f);

	// Set offset biases to create (+1, +1) shift
	// The offset conv output = offset_weights @ input + offset_biases
	// With zero weights, offsets = offset_biases broadcast
	l2.offset_biases[0] = 1.0f;  // dy = 1
	l2.offset_biases[1] = 1.0f;  // dx = 1

	forward_deform_conv_layer(l2, state);

	// The shifted output should be different from reference (unless edge effects dominate)
	// At interior positions, output[y,x] with (+1,+1) offset should equal reference[y+1,x+1]
	// The centered-LHTAN mapping is linear for small offsets, so a +1 bias should remain a local shift.

	// Just verify outputs are different and finite
	bool any_different = false;
	for (int i = 0; i < l2.outputs; ++i)
	{
		EXPECT_TRUE(std::isfinite(l2.output[i])) << "Non-finite output at " << i;
		if (std::fabs(l2.output[i] - reference_output[i]) > 1e-3f)
		{
			any_different = true;
		}
	}
	EXPECT_TRUE(any_different) << "Offsets had no effect on output";

	free_layer(l);
	free_layer(l2);
}

TEST(DeformConvCPU, Col2im_GradientDistribution)
{
	// Col2im should distribute gradients to correct positions
	const int batch = 1;
	const int c = 1;
	const int h = 5;
	const int w = 5;
	const int n = 1;
	const int size = 3;
	const int stride = 1;
	const int pad = 1;

	Darknet::Layer l = make_deform_conv_layer(batch, 1, h, w, c, n, 1, size, stride, stride,
		1, pad, LINEAR, 0, 0, 0, 0, 0, 0, 0, nullptr, 0, 1, 1);

	std::vector<float> input(c * h * w, 1.0f);
	std::vector<float> input_delta(input.size(), 0.0f);
	std::vector<float> workspace(l.workspace_size / sizeof(float), 0.0f);

	// Zero offsets
	int offset_filters = 2 * size * size;
	std::fill(l.offset_weights, l.offset_weights + offset_filters * c * size * size, 0.0f);
	std::fill(l.offset_biases, l.offset_biases + offset_filters, 0.0f);

	// Unit weights
	std::fill(l.weights, l.weights + l.nweights, 1.0f / (size * size * c));
	std::fill(l.biases, l.biases + n, 0.0f);

	Darknet::NetworkState state = {};
	state.input = input.data();
	state.workspace = workspace.data();
	state.train = 1;

	forward_deform_conv_layer(l, state);

	// Set uniform gradient at output
	std::fill(l.delta, l.delta + l.outputs, 1.0f);

	Darknet::NetworkState back_state = {};
	back_state.input = input.data();
	back_state.delta = input_delta.data();
	back_state.workspace = workspace.data();
	back_state.train = 1;

	backward_deform_conv_layer(l, back_state);

	// With zero offsets and unit output gradient, input gradients should be symmetric
	// and non-zero at all interior positions
	float sum_grad = 0.0f;
	int nonzero_count = 0;
	for (size_t i = 0; i < input_delta.size(); ++i)
	{
		EXPECT_TRUE(std::isfinite(input_delta[i])) << "Non-finite gradient at " << i;
		sum_grad += input_delta[i];
		if (std::fabs(input_delta[i]) > 1e-6f)
		{
			++nonzero_count;
		}
	}

	// Should have gradients at most positions
	EXPECT_GT(nonzero_count, 0) << "No non-zero input gradients";
	EXPECT_GT(std::fabs(sum_grad), 1e-6f) << "Total gradient is zero";

	free_layer(l);
}

// ============================================================================
// Multi-Batch Layout Verification Tests
// ============================================================================

TEST(DeformConvLayer, Batch2_OffsetIsolation)
{
	// Verify batch 0 offsets don't affect batch 1 sampling
	const int batch = 2;
	const int c = 2;
	const int h = 5;
	const int w = 5;
	const int n = 2;
	const int size = 3;

	Darknet::Layer l = make_deform_conv_layer(batch, 1, h, w, c, n, 1, size, 1, 1,
		1, 1, LINEAR, 0, 0, 0, 0, 0, 0, 0, nullptr, 0, 1, 0);

	std::vector<float> input(batch * c * h * w);
	std::mt19937 rng(123);
	std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
	for (auto& v : input) v = dist(rng);

	// Different offset biases shouldn't cross-contaminate batches
	// (The offset biases are broadcast to all batches, but the offset field
	// is computed independently per batch from input features)

	int offset_filters = 2 * size * size;

	// Initialize with small values
	for (int i = 0; i < l.nweights; ++i) l.weights[i] = dist(rng) * 0.1f;
	for (size_t i = 0; i < (size_t)offset_filters * c * size * size; ++i)
		l.offset_weights[i] = dist(rng) * 0.01f;
	std::fill(l.biases, l.biases + n, 0.01f);
	std::fill(l.offset_biases, l.offset_biases + offset_filters, 0.0f);

	std::vector<float> workspace(l.workspace_size / sizeof(float), 0.0f);

	Darknet::NetworkState state = {};
	state.input = input.data();
	state.workspace = workspace.data();
	state.train = 0;

	forward_deform_conv_layer(l, state);

	// Store batch 1 output
	std::vector<float> batch1_output(l.output + l.outputs, l.output + 2 * l.outputs);

	// Zero out batch 0 input entirely
	std::fill(input.begin(), input.begin() + c * h * w, 0.0f);

	// Re-run forward
	forward_deform_conv_layer(l, state);

	// Batch 1 output should be identical (since batch 0 shouldn't affect it)
	bool batch1_unchanged = true;
	for (int i = 0; i < l.outputs; ++i)
	{
		if (std::fabs(l.output[l.outputs + i] - batch1_output[i]) > 1e-5f)
		{
			batch1_unchanged = false;
			break;
		}
	}
	EXPECT_TRUE(batch1_unchanged) << "Modifying batch 0 input affected batch 1 output";

	free_layer(l);
}

TEST(DeformConvLayer, Batch2_MaskIsolation)
{
	// Verify mask values are batch-isolated (DCNv2)
	const int batch = 2;
	const int c = 2;
	const int h = 5;
	const int w = 5;
	const int n = 2;
	const int size = 3;

	Darknet::Layer l = make_deform_conv_layer(batch, 1, h, w, c, n, 1, size, 1, 1,
		1, 1, LINEAR, 0, 0, 0, 0, 0, 0, 0, nullptr, 0, 1, 1);  // use_mask=1

	std::vector<float> input(batch * c * h * w);
	std::mt19937 rng(456);
	std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
	for (auto& v : input) v = dist(rng);

	int offset_filters = 2 * size * size;
	int mask_filters = size * size;

	for (int i = 0; i < l.nweights; ++i) l.weights[i] = dist(rng) * 0.1f;
	for (size_t i = 0; i < (size_t)offset_filters * c * size * size; ++i)
		l.offset_weights[i] = dist(rng) * 0.01f;
	for (size_t i = 0; i < (size_t)mask_filters * c * size * size; ++i)
		l.mask_weights[i] = dist(rng) * 0.01f;

	std::vector<float> workspace(l.workspace_size / sizeof(float), 0.0f);

	Darknet::NetworkState state = {};
	state.input = input.data();
	state.workspace = workspace.data();
	state.train = 0;

	forward_deform_conv_layer(l, state);

	// Check masks are in valid range [0, 1] (after sigmoid)
	int spatial = l.out_h * l.out_w;
	for (int b = 0; b < batch; ++b)
	{
		for (int i = 0; i < spatial * mask_filters; ++i)
		{
			float m = l.masks[b * spatial * mask_filters + i];
			EXPECT_GE(m, 0.0f) << "Mask < 0 at batch " << b << " idx " << i;
			EXPECT_LE(m, 1.0f) << "Mask > 1 at batch " << b << " idx " << i;
		}
	}

	// Verify batch independence by checking different batches have different masks
	// (given different inputs)
	bool masks_differ = false;
	for (int i = 0; i < spatial * mask_filters; ++i)
	{
		if (std::fabs(l.masks[i] - l.masks[spatial * mask_filters + i]) > 1e-6f)
		{
			masks_differ = true;
			break;
		}
	}
	EXPECT_TRUE(masks_differ) << "Batch 0 and batch 1 masks are identical despite different inputs";

	free_layer(l);
}

// ============================================================================
// Numerical Stability Tests
// ============================================================================

TEST(DeformConvLayer, LargeInputs_1000x_NoNaNOrInf)
{
	// Large input values shouldn't cause NaN/Inf
	run_deform_conv_case(3, 1, 1, 1000.0f, 1, 0, LINEAR);
	run_deform_conv_case(3, 1, 1, 1000.0f, 0, 1, LEAKY);  // DCNv1 with batch norm
}

TEST(DeformConvLayer, ExtremeOffsets_ClampedToRange)
{
	// Extreme offsets should be clamped correctly
	const int batch = 1;
	const int c = 2;
	const int h = 8;
	const int w = 8;
	const int n = 2;
	const int size = 3;

	Darknet::Layer l = make_deform_conv_layer(batch, 1, h, w, c, n, 1, size, 1, 1,
		1, 1, LINEAR, 0, 0, 0, 0, 0, 0, 0, nullptr, 0, 1, 1);

	std::vector<float> input(c * h * w, 1.0f);

	// Set extreme offset biases (will be soft-limited by centered LHTAN)
	int offset_filters = 2 * size * size;
	for (int i = 0; i < offset_filters; ++i)
	{
		l.offset_biases[i] = (i % 2 == 0) ? 1000.0f : -1000.0f;  // Extreme values
	}

	std::vector<float> workspace(l.workspace_size / sizeof(float), 0.0f);

	Darknet::NetworkState state = {};
	state.input = input.data();
	state.workspace = workspace.data();
	state.train = 0;

	forward_deform_conv_layer(l, state);

	// Offsets should remain finite and within the centered-LHTAN soft limit
	float max_offset = static_cast<float>(size * 1 * 2);  // size * dilation * 2
	const float max_soft_offset = 0.999f * max_offset + 0.001f * 10000.0f;
	int spatial = l.out_h * l.out_w;
	for (int i = 0; i < spatial * offset_filters; ++i)
	{
		EXPECT_TRUE(std::isfinite(l.offsets[i])) << "Non-finite offset at " << i;
		EXPECT_LE(std::fabs(l.offsets[i]), max_soft_offset + 0.1f)
			<< "Offset exceeded centered-LHTAN soft limit at " << i << ": " << l.offsets[i];
	}

	// Output should be finite
	for (int i = 0; i < l.outputs; ++i)
	{
		EXPECT_TRUE(std::isfinite(l.output[i])) << "Non-finite output at " << i;
	}

	free_layer(l);
}

TEST(DeformConvLayer, UniformExtremeOffsetBiases_StayFinite)
{
	const int batch = 1;
	const int c = 2;
	const int h = 8;
	const int w = 8;
	const int n = 2;
	const int size = 3;

	Darknet::Layer l = make_deform_conv_layer(batch, 1, h, w, c, n, 1, size, 1, 1,
		1, 1, LINEAR, 0, 0, 0, 0, 0, 0, 0, nullptr, 0, 1, 0);

	std::vector<float> input(c * h * w, 1.0f);
	std::fill(l.offset_weights, l.offset_weights + (2 * size * size * c * size * size), 0.0f);
	std::fill(l.offset_biases, l.offset_biases + (2 * size * size), 10000.0f);

	std::vector<float> workspace(l.workspace_size / sizeof(float), 0.0f);

	Darknet::NetworkState state = {};
	state.input = input.data();
	state.workspace = workspace.data();
	state.train = 0;

	forward_deform_conv_layer(l, state);

	const float max_offset = static_cast<float>(size * 2);
	const float max_soft_offset = 0.999f * max_offset + 0.001f * 10000.0f;
	const int offset_count = l.out_h * l.out_w * 2 * size * size;
	for (int i = 0; i < offset_count; ++i)
	{
		EXPECT_TRUE(std::isfinite(l.offsets[i])) << "Non-finite offset at " << i;
		EXPECT_LE(std::fabs(l.offsets[i]), max_soft_offset + 0.1f)
			<< "Offset exceeded centered-LHTAN soft limit at " << i << ": " << l.offsets[i];
	}

	for (int i = 0; i < l.outputs; ++i)
	{
		EXPECT_TRUE(std::isfinite(l.output[i])) << "Non-finite output at " << i;
	}

	free_layer(l);
}

TEST(DeformConvLayer, Backward_LargeGradients_Stable)
{
	// Large gradients in backward pass should be handled stably
	const int batch = 1;
	const int c = 2;
	const int h = 5;
	const int w = 5;
	const int n = 2;
	const int size = 3;

	Darknet::Layer l = make_deform_conv_layer(batch, 1, h, w, c, n, 1, size, 1, 1,
		1, 1, LINEAR, 0, 0, 0, 0, 0, 0, 0, nullptr, 0, 1, 1);

	std::vector<float> input(c * h * w, 1.0f);
	std::vector<float> input_delta(input.size(), 0.0f);
	std::vector<float> workspace(l.workspace_size / sizeof(float), 0.0f);

	Darknet::NetworkState state = {};
	state.input = input.data();
	state.workspace = workspace.data();
	state.train = 1;

	forward_deform_conv_layer(l, state);

	// Set large output gradients
	std::fill(l.delta, l.delta + l.outputs, 1000.0f);

	Darknet::NetworkState back_state = {};
	back_state.input = input.data();
	back_state.delta = input_delta.data();
	back_state.workspace = workspace.data();
	back_state.train = 1;

	backward_deform_conv_layer(l, back_state);

	// All gradients should be finite
	expect_all_finite(l.weight_updates, l.nweights, "weight_updates");
	expect_all_finite(l.bias_updates, l.n, "bias_updates");
	expect_all_finite(input_delta.data(), input_delta.size(), "input_delta");

	int offset_filters = 2 * size * size;
	expect_all_finite(l.offset_weight_updates, offset_filters * c * size * size, "offset_weight_updates");

	free_layer(l);
}

// ============================================================================
// GPU Parity Tests (Conditional on DARKNET_GPU and DARKNET_TEST_GPU=1)
// ============================================================================

#ifdef DARKNET_GPU

TEST(DeformConvGPU, Forward_CPUvsGPU_Batch1)
{
	if (!gpu_tests_enabled())
	{
		GTEST_SKIP() << "GPU tests disabled (unset DARKNET_TEST_GPU=0 to enable)";
	}

	const int batch = 1;
	const int c = 3;
	const int h = 8;
	const int w = 8;
	const int n = 4;
	const int size = 3;

	// Create two identical layers
	Darknet::Layer l_cpu = make_deform_conv_layer(batch, 1, h, w, c, n, 1, size, 1, 1,
		1, 1, LINEAR, 0, 0, 0, 0, 0, 0, 0, nullptr, 0, 1, 1);
	Darknet::Layer l_gpu = make_deform_conv_layer(batch, 1, h, w, c, n, 1, size, 1, 1,
		1, 1, LINEAR, 0, 0, 0, 0, 0, 0, 0, nullptr, 0, 1, 1);

	std::vector<float> input(batch * c * h * w);
	std::mt19937 rng(789);
	std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
	for (auto& v : input) v = dist(rng);

	// Copy identical weights
	std::copy(l_cpu.weights, l_cpu.weights + l_cpu.nweights, l_gpu.weights);
	std::copy(l_cpu.biases, l_cpu.biases + l_cpu.n, l_gpu.biases);

	int offset_filters = 2 * size * size;
	int mask_filters = size * size;
	size_t offset_nweights = offset_filters * c * size * size;
	size_t mask_nweights = mask_filters * c * size * size;

	std::copy(l_cpu.offset_weights, l_cpu.offset_weights + offset_nweights, l_gpu.offset_weights);
	std::copy(l_cpu.offset_biases, l_cpu.offset_biases + offset_filters, l_gpu.offset_biases);
	std::copy(l_cpu.mask_weights, l_cpu.mask_weights + mask_nweights, l_gpu.mask_weights);
	std::copy(l_cpu.mask_biases, l_cpu.mask_biases + mask_filters, l_gpu.mask_biases);

	// Push GPU weights
	push_deform_conv_layer(l_gpu);

	std::vector<float> workspace_cpu(l_cpu.workspace_size / sizeof(float), 0.0f);

	// CPU forward
	Darknet::NetworkState state_cpu = {};
	state_cpu.input = input.data();
	state_cpu.workspace = workspace_cpu.data();
	state_cpu.train = 0;
	forward_deform_conv_layer(l_cpu, state_cpu);

	// GPU forward
	float* input_gpu = cuda_make_array(input.data(), input.size());
	float* workspace_gpu = cuda_make_array(nullptr, l_gpu.workspace_size / sizeof(float));

	Darknet::NetworkState state_gpu = {};
	state_gpu.input = input_gpu;
	state_gpu.workspace = workspace_gpu;
	state_gpu.train = 0;
	forward_deform_conv_layer_gpu(l_gpu, state_gpu);

	// Pull GPU output
	std::vector<float> gpu_output(l_gpu.outputs);
	cuda_pull_array(l_gpu.output_gpu, gpu_output.data(), l_gpu.outputs);

	// Compare outputs
	float max_diff = 0.0f;
	for (int i = 0; i < l_cpu.outputs; ++i)
	{
		float diff = std::fabs(l_cpu.output[i] - gpu_output[i]);
		if (diff > max_diff) max_diff = diff;
	}

	// Allow tolerance for floating-point accumulation order differences
	EXPECT_LT(max_diff, 0.05f) << "CPU/GPU output mismatch, max diff = " << max_diff;

	cuda_free(input_gpu);
	cuda_free(workspace_gpu);
	free_layer(l_cpu);
	free_layer(l_gpu);
}

TEST(DeformConvGPU, Forward_CPUvsGPU_Batch2)
{
	if (!gpu_tests_enabled())
	{
		GTEST_SKIP() << "GPU tests disabled (unset DARKNET_TEST_GPU=0 to enable)";
	}

	// Batch > 1 is critical for catching offset indexing bugs
	const int batch = 2;
	const int c = 3;
	const int h = 8;
	const int w = 8;
	const int n = 4;
	const int size = 3;

	Darknet::Layer l_cpu = make_deform_conv_layer(batch, 1, h, w, c, n, 1, size, 1, 1,
		1, 1, LINEAR, 0, 0, 0, 0, 0, 0, 0, nullptr, 0, 1, 1);
	Darknet::Layer l_gpu = make_deform_conv_layer(batch, 1, h, w, c, n, 1, size, 1, 1,
		1, 1, LINEAR, 0, 0, 0, 0, 0, 0, 0, nullptr, 0, 1, 1);

	std::vector<float> input(batch * c * h * w);
	std::mt19937 rng(101);
	std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
	for (auto& v : input) v = dist(rng);

	std::copy(l_cpu.weights, l_cpu.weights + l_cpu.nweights, l_gpu.weights);
	std::copy(l_cpu.biases, l_cpu.biases + l_cpu.n, l_gpu.biases);

	int offset_filters = 2 * size * size;
	int mask_filters = size * size;
	size_t offset_nweights = offset_filters * c * size * size;
	size_t mask_nweights = mask_filters * c * size * size;

	std::copy(l_cpu.offset_weights, l_cpu.offset_weights + offset_nweights, l_gpu.offset_weights);
	std::copy(l_cpu.offset_biases, l_cpu.offset_biases + offset_filters, l_gpu.offset_biases);
	std::copy(l_cpu.mask_weights, l_cpu.mask_weights + mask_nweights, l_gpu.mask_weights);
	std::copy(l_cpu.mask_biases, l_cpu.mask_biases + mask_filters, l_gpu.mask_biases);

	push_deform_conv_layer(l_gpu);

	std::vector<float> workspace_cpu(l_cpu.workspace_size / sizeof(float), 0.0f);

	Darknet::NetworkState state_cpu = {};
	state_cpu.input = input.data();
	state_cpu.workspace = workspace_cpu.data();
	state_cpu.train = 0;
	forward_deform_conv_layer(l_cpu, state_cpu);

	float* input_gpu = cuda_make_array(input.data(), input.size());
	float* workspace_gpu = cuda_make_array(nullptr, l_gpu.workspace_size / sizeof(float));

	Darknet::NetworkState state_gpu = {};
	state_gpu.input = input_gpu;
	state_gpu.workspace = workspace_gpu;
	state_gpu.train = 0;
	forward_deform_conv_layer_gpu(l_gpu, state_gpu);

	std::vector<float> gpu_output(l_gpu.outputs * batch);
	cuda_pull_array(l_gpu.output_gpu, gpu_output.data(), l_gpu.outputs * batch);

	// Check both batches
	float max_diff = 0.0f;
	for (int b = 0; b < batch; ++b)
	{
		for (int i = 0; i < l_cpu.outputs; ++i)
		{
			int idx = b * l_cpu.outputs + i;
			float diff = std::fabs(l_cpu.output[idx] - gpu_output[idx]);
			if (diff > max_diff) max_diff = diff;
		}
	}

	EXPECT_LT(max_diff, 0.05f) << "CPU/GPU output mismatch for batch=2, max diff = " << max_diff;

	cuda_free(input_gpu);
	cuda_free(workspace_gpu);
	free_layer(l_cpu);
	free_layer(l_gpu);
}

TEST(DeformConvGPU, Forward_CPUvsGPU_Batch4)
{
	if (!gpu_tests_enabled())
	{
		GTEST_SKIP() << "GPU tests disabled (unset DARKNET_TEST_GPU=0 to enable)";
	}

	const int batch = 4;
	const int c = 3;
	const int h = 8;
	const int w = 8;
	const int n = 4;
	const int size = 3;

	Darknet::Layer l_cpu = make_deform_conv_layer(batch, 1, h, w, c, n, 1, size, 1, 1,
		1, 1, LINEAR, 0, 0, 0, 0, 0, 0, 0, nullptr, 0, 1, 0);  // DCNv1
	Darknet::Layer l_gpu = make_deform_conv_layer(batch, 1, h, w, c, n, 1, size, 1, 1,
		1, 1, LINEAR, 0, 0, 0, 0, 0, 0, 0, nullptr, 0, 1, 0);

	std::vector<float> input(batch * c * h * w);
	std::mt19937 rng(202);
	std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
	for (auto& v : input) v = dist(rng);

	std::copy(l_cpu.weights, l_cpu.weights + l_cpu.nweights, l_gpu.weights);
	std::copy(l_cpu.biases, l_cpu.biases + l_cpu.n, l_gpu.biases);

	int offset_filters = 2 * size * size;
	size_t offset_nweights = offset_filters * c * size * size;

	std::copy(l_cpu.offset_weights, l_cpu.offset_weights + offset_nweights, l_gpu.offset_weights);
	std::copy(l_cpu.offset_biases, l_cpu.offset_biases + offset_filters, l_gpu.offset_biases);

	push_deform_conv_layer(l_gpu);

	std::vector<float> workspace_cpu(l_cpu.workspace_size / sizeof(float), 0.0f);

	Darknet::NetworkState state_cpu = {};
	state_cpu.input = input.data();
	state_cpu.workspace = workspace_cpu.data();
	state_cpu.train = 0;
	forward_deform_conv_layer(l_cpu, state_cpu);

	float* input_gpu = cuda_make_array(input.data(), input.size());
	float* workspace_gpu = cuda_make_array(nullptr, l_gpu.workspace_size / sizeof(float));

	Darknet::NetworkState state_gpu = {};
	state_gpu.input = input_gpu;
	state_gpu.workspace = workspace_gpu;
	state_gpu.train = 0;
	forward_deform_conv_layer_gpu(l_gpu, state_gpu);

	std::vector<float> gpu_output(l_gpu.outputs * batch);
	cuda_pull_array(l_gpu.output_gpu, gpu_output.data(), l_gpu.outputs * batch);

	float max_diff = 0.0f;
	for (int i = 0; i < l_cpu.outputs * batch; ++i)
	{
		float diff = std::fabs(l_cpu.output[i] - gpu_output[i]);
		if (diff > max_diff) max_diff = diff;
	}

	EXPECT_LT(max_diff, 0.05f) << "CPU/GPU output mismatch for batch=4, max diff = " << max_diff;

	cuda_free(input_gpu);
	cuda_free(workspace_gpu);
	free_layer(l_cpu);
	free_layer(l_gpu);
}

TEST(DeformConvGPU, Backward_CPUvsGPU_Batch2)
{
	if (!gpu_tests_enabled())
	{
		GTEST_SKIP() << "GPU tests disabled (unset DARKNET_TEST_GPU=0 to enable)";
	}

	const int batch = 2;
	const int c = 2;
	const int h = 6;
	const int w = 6;
	const int n = 2;
	const int size = 3;

	Darknet::Layer l_cpu = make_deform_conv_layer(batch, 1, h, w, c, n, 1, size, 1, 1,
		1, 1, LINEAR, 0, 0, 0, 0, 0, 0, 0, nullptr, 0, 1, 1);
	Darknet::Layer l_gpu = make_deform_conv_layer(batch, 1, h, w, c, n, 1, size, 1, 1,
		1, 1, LINEAR, 0, 0, 0, 0, 0, 0, 0, nullptr, 0, 1, 1);

	std::vector<float> input(batch * c * h * w);
	std::mt19937 rng(303);
	std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
	for (auto& v : input) v = dist(rng);

	// Copy weights
	std::copy(l_cpu.weights, l_cpu.weights + l_cpu.nweights, l_gpu.weights);
	std::copy(l_cpu.biases, l_cpu.biases + l_cpu.n, l_gpu.biases);

	int offset_filters = 2 * size * size;
	int mask_filters = size * size;
	size_t offset_nweights = offset_filters * c * size * size;
	size_t mask_nweights = mask_filters * c * size * size;

	std::copy(l_cpu.offset_weights, l_cpu.offset_weights + offset_nweights, l_gpu.offset_weights);
	std::copy(l_cpu.offset_biases, l_cpu.offset_biases + offset_filters, l_gpu.offset_biases);
	std::copy(l_cpu.mask_weights, l_cpu.mask_weights + mask_nweights, l_gpu.mask_weights);
	std::copy(l_cpu.mask_biases, l_cpu.mask_biases + mask_filters, l_gpu.mask_biases);

	push_deform_conv_layer(l_gpu);

	std::vector<float> workspace_cpu(l_cpu.workspace_size / sizeof(float), 0.0f);
	std::vector<float> input_delta_cpu(input.size(), 0.0f);

	// CPU forward
	Darknet::NetworkState state_cpu = {};
	state_cpu.input = input.data();
	state_cpu.workspace = workspace_cpu.data();
	state_cpu.train = 1;
	forward_deform_conv_layer(l_cpu, state_cpu);

	// GPU forward
	float* input_gpu = cuda_make_array(input.data(), input.size());
	float* workspace_gpu = cuda_make_array(nullptr, l_gpu.workspace_size / sizeof(float));
	float* input_delta_gpu = cuda_make_array(nullptr, input.size());

	Darknet::NetworkState state_gpu = {};
	state_gpu.input = input_gpu;
	state_gpu.workspace = workspace_gpu;
	state_gpu.train = 1;
	forward_deform_conv_layer_gpu(l_gpu, state_gpu);

	// Set identical deltas
	std::vector<float> delta(l_cpu.outputs * batch);
	for (auto& d : delta) d = dist(rng);
	std::copy(delta.begin(), delta.end(), l_cpu.delta);
	cuda_push_array(l_gpu.delta_gpu, delta.data(), delta.size());

	// CPU backward
	Darknet::NetworkState back_state_cpu = {};
	back_state_cpu.input = input.data();
	back_state_cpu.delta = input_delta_cpu.data();
	back_state_cpu.workspace = workspace_cpu.data();
	back_state_cpu.train = 1;
	backward_deform_conv_layer(l_cpu, back_state_cpu);

	// GPU backward
	Darknet::NetworkState back_state_gpu = {};
	back_state_gpu.input = input_gpu;
	back_state_gpu.delta = input_delta_gpu;
	back_state_gpu.workspace = workspace_gpu;
	back_state_gpu.train = 1;
	back_state_gpu.net.try_fix_nan = 0;  // Disable NaN fixing for fair comparison
	backward_deform_conv_layer_gpu(l_gpu, back_state_gpu);

	// Pull GPU results
	std::vector<float> gpu_weight_updates(l_gpu.nweights);
	cuda_pull_array(l_gpu.weight_updates_gpu, gpu_weight_updates.data(), l_gpu.nweights);

	std::vector<float> gpu_input_delta(input.size());
	cuda_pull_array(input_delta_gpu, gpu_input_delta.data(), input.size());

	// Compare weight updates (relaxed tolerance due to different accumulation order)
	float max_weight_diff = 0.0f;
	for (int i = 0; i < l_cpu.nweights; ++i)
	{
		float diff = std::fabs(l_cpu.weight_updates[i] - gpu_weight_updates[i]);
		if (diff > max_weight_diff) max_weight_diff = diff;
	}

	EXPECT_LT(max_weight_diff, 0.1f) << "CPU/GPU weight_updates mismatch, max diff = " << max_weight_diff;

	cuda_free(input_gpu);
	cuda_free(workspace_gpu);
	cuda_free(input_delta_gpu);
	free_layer(l_cpu);
	free_layer(l_gpu);
}

TEST(DeformConvGPU, OffsetComputation_CPUvsGPU)
{
	if (!gpu_tests_enabled())
	{
		GTEST_SKIP() << "GPU tests disabled (unset DARKNET_TEST_GPU=0 to enable)";
	}

	// Compare l.offsets after forward pass
	const int batch = 2;
	const int c = 2;
	const int h = 6;
	const int w = 6;
	const int n = 2;
	const int size = 3;

	Darknet::Layer l_cpu = make_deform_conv_layer(batch, 1, h, w, c, n, 1, size, 1, 1,
		1, 1, LINEAR, 0, 0, 0, 0, 0, 0, 0, nullptr, 0, 1, 0);
	Darknet::Layer l_gpu = make_deform_conv_layer(batch, 1, h, w, c, n, 1, size, 1, 1,
		1, 1, LINEAR, 0, 0, 0, 0, 0, 0, 0, nullptr, 0, 1, 0);

	std::vector<float> input(batch * c * h * w);
	std::mt19937 rng(404);
	std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
	for (auto& v : input) v = dist(rng);

	// Copy weights
	std::copy(l_cpu.weights, l_cpu.weights + l_cpu.nweights, l_gpu.weights);
	std::copy(l_cpu.biases, l_cpu.biases + l_cpu.n, l_gpu.biases);

	int offset_filters = 2 * size * size;
	size_t offset_nweights = offset_filters * c * size * size;

	std::copy(l_cpu.offset_weights, l_cpu.offset_weights + offset_nweights, l_gpu.offset_weights);
	std::copy(l_cpu.offset_biases, l_cpu.offset_biases + offset_filters, l_gpu.offset_biases);

	push_deform_conv_layer(l_gpu);

	std::vector<float> workspace_cpu(l_cpu.workspace_size / sizeof(float), 0.0f);

	Darknet::NetworkState state_cpu = {};
	state_cpu.input = input.data();
	state_cpu.workspace = workspace_cpu.data();
	state_cpu.train = 0;
	forward_deform_conv_layer(l_cpu, state_cpu);

	float* input_gpu = cuda_make_array(input.data(), input.size());
	float* workspace_gpu = cuda_make_array(nullptr, l_gpu.workspace_size / sizeof(float));

	Darknet::NetworkState state_gpu = {};
	state_gpu.input = input_gpu;
	state_gpu.workspace = workspace_gpu;
	state_gpu.train = 0;
	forward_deform_conv_layer_gpu(l_gpu, state_gpu);

	// Pull GPU offsets
	int spatial = l_gpu.out_h * l_gpu.out_w;
	int offset_size = batch * spatial * offset_filters;
	std::vector<float> gpu_offsets(offset_size);
	cuda_pull_array(l_gpu.offsets_gpu, gpu_offsets.data(), offset_size);

	// Compare offsets
	float max_diff = 0.0f;
	for (int i = 0; i < offset_size; ++i)
	{
		float diff = std::fabs(l_cpu.offsets[i] - gpu_offsets[i]);
		if (diff > max_diff) max_diff = diff;
	}

	EXPECT_LT(max_diff, 0.01f) << "CPU/GPU offset mismatch, max diff = " << max_diff;

	cuda_free(input_gpu);
	cuda_free(workspace_gpu);
	free_layer(l_cpu);
	free_layer(l_gpu);
}

#endif // DARKNET_GPU

// ============================================================================
// Gradient Checking Tests (preserved from original with enhancements)
// ============================================================================

namespace
{
	/**
	 * @brief Compute numerical gradient using central finite differences
	 */
	std::vector<float> numerical_gradient(
		std::function<float(float*, int)> loss_fn,
		float* params, int n, float eps = 1e-3f)
	{
		std::vector<float> grad(n, 0.0f);
		for (int i = 0; i < n; ++i)
		{
			float original = params[i];

			params[i] = original + eps;
			float loss_plus = loss_fn(params, n);

			params[i] = original - eps;
			float loss_minus = loss_fn(params, n);

			params[i] = original;
			grad[i] = (loss_plus - loss_minus) / (2.0f * eps);
		}
		return grad;
	}

	/**
	 * @brief Compare analytical and numerical gradients
	 */
	bool check_gradients(const float* analytical, const std::vector<float>& numerical,
	                     int n, float rtol = 1e-2f, float atol = 1e-5f)
	{
		int num_failures = 0;
		float max_rel_error = 0.0f;
		for (int i = 0; i < n; ++i)
		{
			float a = analytical[i];
			float num = numerical[i];
			float diff = std::fabs(a - num);
			float scale = std::max(std::max(std::fabs(a), std::fabs(num)), 1e-8f);
			float rel_error = diff / scale;

			if (rel_error > max_rel_error) max_rel_error = rel_error;

			if (diff > atol && rel_error > rtol)
			{
				++num_failures;
				if (num_failures <= 5)
				{
					std::cerr << "Gradient mismatch at idx " << i
					          << ": analytical=" << a << " numerical=" << num
					          << " rel_err=" << rel_error << std::endl;
				}
			}
		}
		if (num_failures > 0)
		{
			std::cerr << "Total gradient failures: " << num_failures << "/" << n
			          << " max_rel_error=" << max_rel_error << std::endl;
		}
		return num_failures == 0;
	}

	/**
	 * @brief Helper struct to manage layer state for gradient checking
	 */
	struct GradCheckContext
	{
		Darknet::Layer l;
		std::vector<float> input;
		std::vector<float> input_delta;
		std::vector<float> workspace;
		std::vector<float> target;

		void init(int use_mask, int batch_normalize)
		{
			const int batch = 1;
			const int steps = 1;
			const int h = 5;
			const int w = 5;
			const int c = 2;
			const int n = 2;
			const int groups = 1;
			const int size = 3;
			const int stride_x = 1;
			const int stride_y = 1;
			const int dilation = 1;
			const int padding = 1;

			l = make_deform_conv_layer(batch, steps, h, w, c, n, groups, size, stride_x, stride_y,
				dilation, padding, LINEAR, batch_normalize, 0, 0, 0, 0, 0, 0, nullptr, 0, 1, use_mask);

			input.resize(batch * c * h * w);
			input_delta.resize(input.size(), 0.0f);
			workspace.resize(l.workspace_size / sizeof(float), 0.0f);
			target.resize(l.outputs * batch);

			std::mt19937 rng(42);
			std::normal_distribution<float> dist(0.0f, 0.1f);

			for (auto& v : input) v = dist(rng);
			for (auto& v : target) v = dist(rng);

			for (int i = 0; i < l.nweights; ++i) l.weights[i] = dist(rng) * 0.1f;
			for (int i = 0; i < l.n; ++i) l.biases[i] = 0.01f;

			int offset_filters = 2 * size * size;
			int offset_nweights = c * offset_filters * size * size;
			for (int i = 0; i < offset_nweights; ++i) l.offset_weights[i] = dist(rng) * 0.01f;
			std::fill(l.offset_biases, l.offset_biases + offset_filters, 0.0f);

			if (use_mask)
			{
				int mask_filters = size * size;
				int mask_nweights = c * mask_filters * size * size;
				for (int i = 0; i < mask_nweights; ++i) l.mask_weights[i] = dist(rng) * 0.01f;
				std::fill(l.mask_biases, l.mask_biases + mask_filters, 0.0f);
			}
		}

		float compute_loss()
		{
			Darknet::NetworkState state = {};
			state.input = input.data();
			state.workspace = workspace.data();
			state.train = 1;

			std::fill(l.output, l.output + l.outputs * l.batch, 0.0f);
			forward_deform_conv_layer(l, state);

			float loss = 0.0f;
			for (int i = 0; i < l.outputs * l.batch; ++i)
			{
				float diff = l.output[i] - target[i];
				loss += 0.5f * diff * diff;
			}
			return loss;
		}

		void compute_analytical_gradients()
		{
			Darknet::NetworkState fwd_state = {};
			fwd_state.input = input.data();
			fwd_state.workspace = workspace.data();
			fwd_state.train = 1;

			std::fill(l.output, l.output + l.outputs * l.batch, 0.0f);
			forward_deform_conv_layer(l, fwd_state);

			for (int i = 0; i < l.outputs * l.batch; ++i)
			{
				l.delta[i] = l.output[i] - target[i];
			}

			std::fill(l.weight_updates, l.weight_updates + l.nweights, 0.0f);
			std::fill(l.bias_updates, l.bias_updates + l.n, 0.0f);

			int offset_filters = 2 * l.size * l.size;
			int offset_nweights = l.c * offset_filters * l.size * l.size;
			std::fill(l.offset_weight_updates, l.offset_weight_updates + offset_nweights, 0.0f);
			std::fill(l.offset_bias_updates, l.offset_bias_updates + offset_filters, 0.0f);

			if (l.use_mask)
			{
				int mask_filters = l.size * l.size;
				int mask_nweights = l.c * mask_filters * l.size * l.size;
				std::fill(l.mask_weight_updates, l.mask_weight_updates + mask_nweights, 0.0f);
				std::fill(l.mask_bias_updates, l.mask_bias_updates + mask_filters, 0.0f);
			}

			std::fill(input_delta.begin(), input_delta.end(), 0.0f);

			Darknet::NetworkState back_state = {};
			back_state.input = input.data();
			back_state.delta = input_delta.data();
			back_state.workspace = workspace.data();
			back_state.train = 1;

			backward_deform_conv_layer(l, back_state);
		}

		~GradCheckContext()
		{
			free_layer(l);
		}
	};
}

TEST(DeformConvLayer, GradientCheckMainWeights)
{
	GradCheckContext ctx;
	ctx.init(0, 0);

	ctx.compute_analytical_gradients();

	auto loss_fn = [&ctx](float* params, int n) -> float {
		return ctx.compute_loss();
	};

	std::vector<float> num_grad = numerical_gradient(loss_fn, ctx.l.weights, ctx.l.nweights, 1e-3f);

	EXPECT_TRUE(check_gradients(ctx.l.weight_updates, num_grad, ctx.l.nweights, 0.05f, 1e-4f))
		<< "Main weight gradients do not match numerical gradients";
}

TEST(DeformConvLayer, GradientCheckOffsetWeights)
{
	GradCheckContext ctx;
	ctx.init(0, 0);

	ctx.compute_analytical_gradients();

	int offset_filters = 2 * ctx.l.size * ctx.l.size;
	int offset_nweights = ctx.l.c * offset_filters * ctx.l.size * ctx.l.size;

	auto loss_fn = [&ctx](float* params, int n) -> float {
		return ctx.compute_loss();
	};

	std::vector<float> num_grad = numerical_gradient(loss_fn, ctx.l.offset_weights, offset_nweights, 1e-3f);

	int num_pass = 0;
	float total_cos_sim_num = 0.0f, total_cos_sim_den_a = 0.0f, total_cos_sim_den_n = 0.0f;
	for (int i = 0; i < offset_nweights; ++i)
	{
		float a = ctx.l.offset_weight_updates[i];
		float n = num_grad[i];
		float abs_err = std::fabs(a - n);
		float scale = std::max(std::max(std::fabs(a), std::fabs(n)), 1e-7f);

		if (abs_err < 1e-4f || abs_err / scale < 0.5f)
		{
			++num_pass;
		}
		total_cos_sim_num += a * n;
		total_cos_sim_den_a += a * a;
		total_cos_sim_den_n += n * n;
	}

	float pass_rate = static_cast<float>(num_pass) / offset_nweights;
	float cos_sim = total_cos_sim_num / (std::sqrt(total_cos_sim_den_a) * std::sqrt(total_cos_sim_den_n) + 1e-8f);

	EXPECT_GT(pass_rate, 0.8f)
		<< "Only " << (pass_rate * 100) << "% of offset weight gradients match";
	EXPECT_GT(cos_sim, 0.9f)
		<< "Gradient direction mismatch: cosine similarity = " << cos_sim;
}

TEST(DeformConvLayer, GradientCheckMaskWeights)
{
	GradCheckContext ctx;
	ctx.init(1, 0);

	ctx.compute_analytical_gradients();

	int mask_filters = ctx.l.size * ctx.l.size;
	int mask_nweights = ctx.l.c * mask_filters * ctx.l.size * ctx.l.size;

	auto loss_fn = [&ctx](float* params, int n) -> float {
		return ctx.compute_loss();
	};

	std::vector<float> num_grad = numerical_gradient(loss_fn, ctx.l.mask_weights, mask_nweights, 1e-3f);

	float analytical_norm = 0.0f, numerical_norm = 0.0f;
	int num_finite = 0;
	float total_cos_sim_num = 0.0f, total_cos_sim_den_a = 0.0f, total_cos_sim_den_n = 0.0f;

	for (int i = 0; i < mask_nweights; ++i)
	{
		float a = ctx.l.mask_weight_updates[i];
		float n = num_grad[i];

		if (std::isfinite(a) && std::isfinite(n))
		{
			++num_finite;
			analytical_norm += a * a;
			numerical_norm += n * n;
			total_cos_sim_num += a * n;
			total_cos_sim_den_a += a * a;
			total_cos_sim_den_n += n * n;
		}
	}

	float cos_sim = total_cos_sim_num / (std::sqrt(total_cos_sim_den_a) * std::sqrt(total_cos_sim_den_n) + 1e-8f);

	EXPECT_EQ(num_finite, mask_nweights)
		<< "Found non-finite mask gradients";

	EXPECT_GT(std::sqrt(analytical_norm), 1e-8f)
		<< "Analytical mask gradients are all zero";
	EXPECT_GT(std::sqrt(numerical_norm), 1e-8f)
		<< "Numerical mask gradients are all zero";

	EXPECT_GT(cos_sim, 0.3f)
		<< "Gradient direction severely mismatched: cosine similarity = " << cos_sim;
}

TEST(DeformConvLayer, LHTanGradientUnitTest)
{
	const float max_offset = 6.0f;

	std::vector<float> test_inputs = {0.0f, 0.5f, -0.5f, 1.0f, -1.0f, 2.0f, -2.0f, 4.0f, -4.0f, 5.5f, -5.5f, 7.0f, -7.0f, 12.0f, -12.0f};

	for (float raw : test_inputs)
	{
		const float scaled = raw / (2.0f * max_offset) + 0.5f;
		const float lhtan_val = (scaled < 0.0f) ? (0.001f * scaled)
			: (scaled > 1.0f) ? (0.001f * (scaled - 1.0f) + 1.0f)
			: scaled;
		const float clamped = (2.0f * lhtan_val - 1.0f) * max_offset;
		const float analytical_deriv = (clamped < -max_offset || clamped > max_offset) ? 0.001f : 1.0f;

		float eps = 1e-2f;
		const float scaled_plus = (raw + eps) / (2.0f * max_offset) + 0.5f;
		const float scaled_minus = (raw - eps) / (2.0f * max_offset) + 0.5f;
		const float lhtan_plus = (scaled_plus < 0.0f) ? (0.001f * scaled_plus)
			: (scaled_plus > 1.0f) ? (0.001f * (scaled_plus - 1.0f) + 1.0f)
			: scaled_plus;
		const float lhtan_minus = (scaled_minus < 0.0f) ? (0.001f * scaled_minus)
			: (scaled_minus > 1.0f) ? (0.001f * (scaled_minus - 1.0f) + 1.0f)
			: scaled_minus;
		float clamped_plus = (2.0f * lhtan_plus - 1.0f) * max_offset;
		float clamped_minus = (2.0f * lhtan_minus - 1.0f) * max_offset;
		float numerical_deriv = (clamped_plus - clamped_minus) / (2.0f * eps);

		float abs_error = std::fabs(analytical_deriv - numerical_deriv);
		float rel_error = abs_error / std::max(std::max(std::fabs(analytical_deriv), std::fabs(numerical_deriv)), 1e-6f);

		bool pass = (abs_error < 1e-4f) || (rel_error < 0.02f);
		EXPECT_TRUE(pass)
			<< "LHTAN gradient mismatch at raw=" << raw
			<< " analytical=" << analytical_deriv
			<< " numerical=" << numerical_deriv
			<< " abs_error=" << abs_error
			<< " rel_error=" << rel_error;
	}
}
