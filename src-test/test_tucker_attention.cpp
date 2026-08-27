#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <string>
#include <vector>

#include "darknet_internal.hpp"
#include "tucker_attention_layer.hpp"

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
		float value = 0.0f;
		for (size_t i = 0; i < n; ++i)
		{
			value = std::max(value, std::fabs(data[i]));
		}
		return value;
	}

	float max_abs_diff(const float *a, const float *b, size_t n)
	{
		float value = 0.0f;
		for (size_t i = 0; i < n; ++i)
		{
			value = std::max(value, std::fabs(a[i] - b[i]));
		}
		return value;
	}

	void expect_all_finite(const float *data, size_t n, const char *label)
	{
		for (size_t i = 0; i < n; ++i)
		{
			ASSERT_TRUE(std::isfinite(data[i])) << label << " idx=" << i << " val=" << data[i];
		}
	}

#ifdef DARKNET_GPU
	bool gpu_tests_enabled()
	{
		const char *gpu_mode = std::getenv("DARKNET_TEST_GPU");
		return !(gpu_mode && std::string(gpu_mode) == "0");
	}

	void copy_tucker_trainable_params(const Darknet::Layer &src, Darknet::Layer &dst)
	{
		std::copy(src.weights, src.weights + src.nweights, dst.weights);
		std::copy(src.biases, src.biases + src.nbiases, dst.biases);
		cuda_push_array(dst.weights_gpu, dst.weights, dst.nweights);
		cuda_push_array(dst.biases_gpu, dst.biases, dst.nbiases);
	}
#endif
}

#ifdef DARKNET_GPU
TEST(TuckerAttentionGPU, ForwardBackwardMatchesCPUWithPaddedWindows)
{
	if (!gpu_tests_enabled())
	{
		GTEST_SKIP() << "GPU tests disabled (set DARKNET_TEST_GPU=1 or leave it unset to enable)";
	}

	const int batch = 2;
	const int h = 5;
	const int w = 6;
	const int c = 8;
	const int filters = 8;
	const int window = 3;
	const int heads = 2;

	Darknet::Layer l_cpu = make_tucker_attention_layer(batch, h, w, c, filters,
		window, heads, 3, 2, 4, 3, LINEAR, 0, 1);
	Darknet::Layer l_gpu = make_tucker_attention_layer(batch, h, w, c, filters,
		window, heads, 3, 2, 4, 3, LINEAR, 0, 1);

	fill_pattern(l_cpu.weights, l_cpu.nweights, 0.025f, -0.004f, 0.013f);
	fill_pattern(l_cpu.biases, l_cpu.nbiases, 0.006f, 0.001f, 0.071f);
	copy_tucker_trainable_params(l_cpu, l_gpu);

	std::vector<float> input(static_cast<size_t>(batch) * c * h * w);
	std::vector<float> input_delta_cpu(input.size(), 0.0f);
	std::vector<float> delta(static_cast<size_t>(batch) * l_cpu.outputs);
	fill_pattern(input.data(), input.size(), 0.08f, -0.02f, 0.017f);
	fill_pattern(delta.data(), delta.size(), 0.05f, 0.0f, 0.029f);

	Darknet::NetworkState fwd_cpu = {};
	fwd_cpu.input = input.data();
	fwd_cpu.train = 1;
	forward_tucker_attention_layer(l_cpu, fwd_cpu);

	float *input_gpu = cuda_make_array(input.data(), input.size());
	float *input_delta_gpu = cuda_make_array(input_delta_cpu.data(), input_delta_cpu.size());

	Darknet::NetworkState fwd_gpu = {};
	fwd_gpu.input = input_gpu;
	fwd_gpu.train = 1;
	forward_tucker_attention_layer_gpu(l_gpu, fwd_gpu);

	std::vector<float> output_gpu(static_cast<size_t>(batch) * l_gpu.outputs, 0.0f);
	cuda_pull_array(l_gpu.output_gpu, output_gpu.data(), output_gpu.size());
	expect_all_finite(output_gpu.data(), output_gpu.size(), "tucker gpu output");
	EXPECT_LT(max_abs_diff(l_cpu.output, output_gpu.data(), output_gpu.size()), 3e-2f);

	std::copy(delta.begin(), delta.end(), l_cpu.delta);
	cuda_push_array(l_gpu.delta_gpu, delta.data(), delta.size());

	Darknet::NetworkState back_cpu = {};
	back_cpu.input = input.data();
	back_cpu.delta = input_delta_cpu.data();
	back_cpu.train = 1;
	backward_tucker_attention_layer(l_cpu, back_cpu);

	Darknet::NetworkState back_gpu = {};
	back_gpu.input = input_gpu;
	back_gpu.delta = input_delta_gpu;
	back_gpu.train = 1;
	backward_tucker_attention_layer_gpu(l_gpu, back_gpu);

	std::vector<float> input_delta_gpu_host(input.size(), 0.0f);
	std::vector<float> weight_updates_gpu(l_gpu.nweights, 0.0f);
	std::vector<float> bias_updates_gpu(l_gpu.nbiases, 0.0f);
	cuda_pull_array(input_delta_gpu, input_delta_gpu_host.data(), input_delta_gpu_host.size());
	cuda_pull_array(l_gpu.weight_updates_gpu, weight_updates_gpu.data(), weight_updates_gpu.size());
	cuda_pull_array(l_gpu.bias_updates_gpu, bias_updates_gpu.data(), bias_updates_gpu.size());

	expect_all_finite(input_delta_gpu_host.data(), input_delta_gpu_host.size(), "tucker gpu input delta");
	expect_all_finite(weight_updates_gpu.data(), weight_updates_gpu.size(), "tucker gpu weight updates");
	expect_all_finite(bias_updates_gpu.data(), bias_updates_gpu.size(), "tucker gpu bias updates");
	EXPECT_GT(max_abs(weight_updates_gpu.data(), weight_updates_gpu.size()), 0.0f);
	EXPECT_LT(max_abs_diff(input_delta_cpu.data(), input_delta_gpu_host.data(), input_delta_cpu.size()), 4e-2f);
	EXPECT_LT(max_abs_diff(l_cpu.weight_updates, weight_updates_gpu.data(), weight_updates_gpu.size()), 5e-2f);
	EXPECT_LT(max_abs_diff(l_cpu.bias_updates, bias_updates_gpu.data(), bias_updates_gpu.size()), 2e-3f);

	cuda_free(input_gpu);
	cuda_free(input_delta_gpu);
	free_layer(l_cpu);
	free_layer(l_gpu);
}
#endif
