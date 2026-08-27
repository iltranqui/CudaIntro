#include <gtest/gtest.h>

#include <vector>

#include "darknet_internal.hpp"

TEST(BlasKernels, SimpleCopyCopiesContiguousDeviceFloats)
{
	std::vector<float> input{1.0f, -2.0f, 3.5f, 4.25f, -5.75f};
	std::vector<float> output(input.size(), 0.0f);
	float *input_gpu = cuda_make_array(input.data(), input.size());
	float *output_gpu = cuda_make_array(output.data(), output.size());

	simple_copy_ongpu(static_cast<int>(input.size()), input_gpu, output_gpu);
	cuda_pull_array(output_gpu, output.data(), output.size());

	EXPECT_EQ(output, input);

	cuda_free(output_gpu);
	cuda_free(input_gpu);
}

TEST(BlasKernels, FillZeroClearsContiguousDeviceFloats)
{
	std::vector<float> input{1.0f, -2.0f, 3.5f, 4.25f, -5.75f};
	std::vector<float> output(input.size(), 1.0f);
	float *data_gpu = cuda_make_array(input.data(), input.size());

	fill_ongpu(static_cast<int>(input.size()), 0.0f, data_gpu, 1);
	cuda_pull_array(data_gpu, output.data(), output.size());

	for (const float value : output)
	{
		EXPECT_FLOAT_EQ(value, 0.0f);
	}

	cuda_free(data_gpu);
}

TEST(BlasKernels, FillKeepsKernelFallbackForStridedNonzeroFill)
{
	std::vector<float> input(8, 0.0f);
	std::vector<float> output(input.size(), -1.0f);
	float *data_gpu = cuda_make_array(input.data(), input.size());

	fill_ongpu(4, 2.5f, data_gpu, 2);
	cuda_pull_array(data_gpu, output.data(), output.size());

	for (size_t idx = 0; idx < output.size(); ++idx)
	{
		const float expected = (idx % 2 == 0) ? 2.5f : 0.0f;
		EXPECT_FLOAT_EQ(output[idx], expected);
	}

	cuda_free(data_gpu);
}

TEST(BlasKernels, SgdUpdateFusesDecayWeightUpdateAndMomentum)
{
	std::vector<float> values{1.0f, -2.0f, 0.5f, 4.0f};
	std::vector<float> updates{0.2f, -0.4f, 0.0f, 1.5f};
	const std::vector<float> initial_values = values;
	const std::vector<float> initial_updates = updates;
	const float rate = 0.25f;
	const float momentum = 0.9f;
	const float decay = -0.1f;

	float *values_gpu = cuda_make_array(values.data(), values.size());
	float *updates_gpu = cuda_make_array(updates.data(), updates.size());

	sgd_update_ongpu(static_cast<int>(values.size()), values_gpu, updates_gpu, rate, momentum, decay);
	cuda_pull_array(values_gpu, values.data(), values.size());
	cuda_pull_array(updates_gpu, updates.data(), updates.size());

	for (size_t idx = 0; idx < values.size(); ++idx)
	{
		const float update_after_decay = initial_updates[idx] + decay * initial_values[idx];
		EXPECT_FLOAT_EQ(values[idx], initial_values[idx] + rate * update_after_decay);
		EXPECT_FLOAT_EQ(updates[idx], momentum * update_after_decay);
	}

	cuda_free(updates_gpu);
	cuda_free(values_gpu);
}

TEST(BlasKernels, LeakyActivationAndGradientMatchExpectedValues)
{
	std::vector<float> input{-2.0f, -0.5f, 0.0f, 3.0f, -0.25f, 7.0f, -10.0f, 0.125f};
	std::vector<float> output(input.size(), 0.0f);
	float *input_gpu = cuda_make_array(input.data(), input.size());

	activate_array_ongpu(input_gpu, static_cast<int>(input.size()), LEAKY);
	cuda_pull_array(input_gpu, output.data(), output.size());

	for (size_t idx = 0; idx < input.size(); ++idx)
	{
		const float expected = (input[idx] > 0.0f) ? input[idx] : 0.1f * input[idx];
		EXPECT_FLOAT_EQ(output[idx], expected);
	}

	std::vector<float> delta{1.0f, -2.0f, 0.5f, 4.0f, -1.5f, 3.0f, 2.0f, -8.0f};
	const std::vector<float> initial_delta = delta;
	float *grad_input_gpu = cuda_make_array(input.data(), input.size());
	float *delta_gpu = cuda_make_array(delta.data(), delta.size());

	gradient_array_ongpu(grad_input_gpu, static_cast<int>(input.size()), LEAKY, delta_gpu);
	cuda_pull_array(delta_gpu, delta.data(), delta.size());

	for (size_t idx = 0; idx < input.size(); ++idx)
	{
		const float expected_scale = (input[idx] > 0.0f) ? 1.0f : 0.1f;
		EXPECT_FLOAT_EQ(delta[idx], initial_delta[idx] * expected_scale);
	}

	cuda_free(delta_gpu);
	cuda_free(grad_input_gpu);
	cuda_free(input_gpu);
}

TEST(BlasKernels, LeakyActivationFallbackHandlesMisalignedOffsets)
{
	std::vector<float> input{-99.0f, -2.0f, -0.5f, 0.0f, 3.0f, -0.25f, 7.0f, -10.0f, 0.125f};
	std::vector<float> output(input.size(), 0.0f);
	float *input_gpu = cuda_make_array(input.data(), input.size());

	activate_array_ongpu(input_gpu + 1, 8, LEAKY);
	cuda_pull_array(input_gpu, output.data(), output.size());

	EXPECT_FLOAT_EQ(output[0], input[0]);
	for (size_t idx = 1; idx < input.size(); ++idx)
	{
		const float expected = (input[idx] > 0.0f) ? input[idx] : 0.1f * input[idx];
		EXPECT_FLOAT_EQ(output[idx], expected);
	}

	cuda_free(input_gpu);
}
