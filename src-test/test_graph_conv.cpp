#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>

#include "darknet_internal.hpp"
#include "graph_conv_layer.hpp"

namespace
{
	void expect_all_finite(const float *data, size_t n, const char *label)
	{
		for (size_t i = 0; i < n; ++i)
		{
			ASSERT_TRUE(std::isfinite(data[i])) << label << " idx=" << i << " val=" << data[i];
		}
	}

	int output_index(const Darknet::Layer & l, int b, int c, int y, int x)
	{
		return ((b * l.out_c + c) * l.out_h + y) * l.out_w + x;
	}

	int input_index(const Darknet::Layer & l, int b, int c, int y, int x)
	{
		return ((b * l.c + c) * l.h + y) * l.w + x;
	}

	int alpha_index(const Darknet::Layer & l, int b, int g, int y, int x, int k)
	{
		return ((((b * l.groups + g) * l.out_h + y) * l.out_w + x) * l.graph_k + k);
	}

	float manual_valid_mean(const std::vector<float> &input, const Darknet::Layer &l, int channel, int oy, int ox)
	{
		float sum = 0.0f;
		int count = 0;
		for (int ky = 0; ky < l.size; ++ky)
		{
			for (int kx = 0; kx < l.size; ++kx)
			{
				const int iy = oy * l.stride_y - l.pad + ky * l.dilation;
				const int ix = ox * l.stride_x - l.pad + kx * l.dilation;
				if (iy >= 0 && iy < l.h && ix >= 0 && ix < l.w)
				{
					sum += input[input_index(l, 0, channel, iy, ix)];
					++count;
				}
			}
		}
		return (count > 0) ? (sum / count) : 0.0f;
	}

	bool gpu_tests_enabled()
	{
		const char *gpu_mode = std::getenv("DARKNET_TEST_GPU");
		return !(gpu_mode && std::string(gpu_mode) == "0");
	}

	std::vector<float> numerical_gradient(const std::function<float(float *, int)> &loss_fn, float *params, int n, float eps = 1e-4f)
	{
		std::vector<float> grad(n, 0.0f);
		for (int i = 0; i < n; ++i)
		{
			const float original = params[i];
			params[i] = original + eps;
			const float loss_plus = loss_fn(params, n);
			params[i] = original - eps;
			const float loss_minus = loss_fn(params, n);
			params[i] = original;
			grad[i] = (loss_plus - loss_minus) / (2.0f * eps);
		}
		return grad;
	}

	bool check_gradients(const float *analytical, const std::vector<float> &numerical, int n, float rtol = 1e-2f, float atol = 1e-5f)
	{
		int failures = 0;
		float max_rel_error = 0.0f;
		for (int i = 0; i < n; ++i)
		{
			const float a = analytical[i];
			const float num = numerical[i];
			const float diff = std::fabs(a - num);
			const float scale = std::max(std::max(std::fabs(a), std::fabs(num)), 1e-8f);
			const float rel_error = diff / scale;
			max_rel_error = std::max(max_rel_error, rel_error);

			if (diff > atol && rel_error > rtol)
			{
				++failures;
				if (failures <= 5)
				{
					std::cerr << "Gradient mismatch at idx " << i
					          << ": analytical=" << a
					          << " numerical=" << num
					          << " rel_err=" << rel_error
					          << std::endl;
				}
			}
		}

		if (failures > 0)
		{
			std::cerr << "Total gradient failures: " << failures << "/" << n
			          << " max_rel_error=" << max_rel_error << std::endl;
		}
		return failures == 0;
	}

	void fill_small_random(float *data, size_t n, std::mt19937 &rng, float scale)
	{
		std::normal_distribution<float> dist(0.0f, scale);
		for (size_t i = 0; i < n; ++i)
		{
			data[i] = dist(rng);
		}
	}

	struct GraphGradCheckContext
	{
		Darknet::Layer l = { static_cast<Darknet::ELayerType>(0) };
		std::vector<float> input;
		std::vector<float> input_delta;
		std::vector<float> target;
		bool initialized = false;

		void init(int graph_use_self = 1, int graph_edge_mode = 1)
		{
			l = make_graph_conv_layer(1, 1, 3, 3, 2, 2, 1, 3, 1, 1,
				1, 1, LINEAR, 0, 0, 0, 0, 0, 0, 0, nullptr, 0, 1,
				graph_edge_mode, graph_use_self, 1);

			input.resize(static_cast<size_t>(l.batch) * l.c * l.h * l.w);
			input_delta.assign(input.size(), 0.0f);
			target.resize(static_cast<size_t>(l.batch) * l.outputs);

			std::mt19937 rng(42);
			fill_small_random(input.data(), input.size(), rng, 0.1f);
			fill_small_random(target.data(), target.size(), rng, 0.1f);
			fill_small_random(l.weights, l.nweights, rng, 0.05f);
			std::fill(l.biases, l.biases + l.n, 0.0f);

			if (l.graph_use_self)
			{
				fill_small_random(l.graph_self_weights, static_cast<size_t>(l.n) * l.graph_cpg, rng, 0.05f);
			}

			if (l.graph_edge_mode == 1)
			{
				fill_small_random(l.graph_edge_kernel, static_cast<size_t>(l.groups) * l.graph_k * 2 * l.graph_cpg, rng, 0.03f);
				fill_small_random(l.graph_edge_biases, static_cast<size_t>(l.groups) * l.graph_k, rng, 0.02f);
			}

			initialized = true;
		}

		float compute_loss()
		{
			Darknet::NetworkState state = {};
			state.input = input.data();
			state.train = 1;

			forward_graph_conv_layer(l, state);

			float loss = 0.0f;
			for (int i = 0; i < l.outputs * l.batch; ++i)
			{
				const float diff = l.output[i] - target[i];
				loss += 0.5f * diff * diff;
			}
			return loss;
		}

		void compute_analytical_gradients()
		{
			Darknet::NetworkState fwd_state = {};
			fwd_state.input = input.data();
			fwd_state.train = 1;
			forward_graph_conv_layer(l, fwd_state);

			for (int i = 0; i < l.outputs * l.batch; ++i)
			{
				l.delta[i] = l.output[i] - target[i];
			}

			std::fill(l.weight_updates, l.weight_updates + l.nweights, 0.0f);
			std::fill(l.bias_updates, l.bias_updates + l.n, 0.0f);
			if (l.graph_use_self)
			{
				std::fill(l.graph_self_weight_updates, l.graph_self_weight_updates + l.n * l.graph_cpg, 0.0f);
			}
			if (l.graph_edge_mode == 1)
			{
				std::fill(l.graph_edge_kernel_updates, l.graph_edge_kernel_updates + l.groups * l.graph_k * 2 * l.graph_cpg, 0.0f);
				std::fill(l.graph_edge_bias_updates, l.graph_edge_bias_updates + l.groups * l.graph_k, 0.0f);
			}
			std::fill(input_delta.begin(), input_delta.end(), 0.0f);

			Darknet::NetworkState back_state = {};
			back_state.input = input.data();
			back_state.delta = input_delta.data();
			back_state.train = 1;
			backward_graph_conv_layer(l, back_state);
		}

		~GraphGradCheckContext()
		{
			if (initialized)
			{
				free_layer(l);
			}
		}
	};

	void copy_graph_conv_params(const Darknet::Layer &src, Darknet::Layer &dst)
	{
		std::copy(src.weights, src.weights + src.nweights, dst.weights);
		std::copy(src.biases, src.biases + src.n, dst.biases);

		if (src.graph_use_self)
		{
			std::copy(src.graph_self_weights, src.graph_self_weights + src.n * src.graph_cpg, dst.graph_self_weights);
		}

		if (src.graph_edge_mode == 1)
		{
			std::copy(src.graph_edge_kernel, src.graph_edge_kernel + src.groups * src.graph_k * 2 * src.graph_cpg, dst.graph_edge_kernel);
			std::copy(src.graph_edge_biases, src.graph_edge_biases + src.groups * src.graph_k, dst.graph_edge_biases);
		}

		if (src.batch_normalize)
		{
			std::copy(src.scales, src.scales + src.n, dst.scales);
			std::copy(src.rolling_mean, src.rolling_mean + src.n, dst.rolling_mean);
			std::copy(src.rolling_variance, src.rolling_variance + src.n, dst.rolling_variance);
		}
	}

	float max_abs_diff(const float *lhs, const float *rhs, size_t n)
	{
		float max_diff = 0.0f;
		for (size_t i = 0; i < n; ++i)
		{
			max_diff = std::max(max_diff, std::fabs(lhs[i] - rhs[i]));
		}
		return max_diff;
	}
}

TEST(GraphConvLayer, ConstructionBasicShapes)
{
	const struct
	{
		int size;
		int stride;
		int pad;
		int expected_h;
		int expected_w;
	} cases[] =
	{
		{1, 1, 0, 7, 9},
		{3, 1, 1, 7, 9},
		{3, 2, 1, 4, 5},
	};

	for (const auto &tc : cases)
	{
		Darknet::Layer l = make_graph_conv_layer(2, 1, 7, 9, 4, 6, 1, tc.size, tc.stride, tc.stride,
			1, tc.pad, LINEAR, 0, 0, 0, 0, 0, 0, 0, nullptr, 0, 1, 0, 1, 1);

		EXPECT_EQ(l.out_h, tc.expected_h);
		EXPECT_EQ(l.out_w, tc.expected_w);
		EXPECT_EQ(l.out_c, 6);
		EXPECT_EQ(l.outputs, tc.expected_h * tc.expected_w * 6);
		EXPECT_EQ(l.inputs, 7 * 9 * 4);
		EXPECT_EQ(l.graph_k, tc.size * tc.size);
		EXPECT_NE(l.graph_ref, nullptr);
		EXPECT_NE(l.graph_agg, nullptr);
		EXPECT_NE(l.graph_alpha, nullptr);
		EXPECT_NE(l.graph_valid, nullptr);

		free_layer(l);
	}
}

TEST(GraphConvForward, ZeroInputGivesZeroOutput)
{
	Darknet::Layer l = make_graph_conv_layer(1, 1, 5, 5, 3, 4, 1, 3, 1, 1,
		1, 1, LINEAR, 0, 0, 0, 0, 0, 0, 0, nullptr, 0, 1, 1, 1, 1);

	std::vector<float> input(5 * 5 * 3, 0.0f);
	std::fill(l.biases, l.biases + l.n, 0.0f);

	Darknet::NetworkState state = {};
	state.input = input.data();
	state.train = 1;

	forward_graph_conv_layer(l, state);

	expect_all_finite(l.output, l.outputs, "output");
	for (int i = 0; i < l.outputs; ++i)
	{
		EXPECT_FLOAT_EQ(l.output[i], 0.0f);
	}

	free_layer(l);
}

TEST(GraphConvForward, SoftmaxNormalizesPerNode)
{
	Darknet::Layer l = make_graph_conv_layer(1, 1, 5, 5, 2, 2, 1, 3, 1, 1,
		1, 1, LINEAR, 0, 0, 0, 0, 0, 0, 0, nullptr, 0, 1, 1, 1, 1);

	std::vector<float> input(5 * 5 * 2, 0.0f);
	for (size_t i = 0; i < input.size(); ++i)
	{
		input[i] = 0.1f + static_cast<float>(i) * 0.01f;
	}

	Darknet::NetworkState state = {};
	state.input = input.data();
	state.train = 1;

	forward_graph_conv_layer(l, state);

	for (int oy = 0; oy < l.out_h; ++oy)
	{
		for (int ox = 0; ox < l.out_w; ++ox)
		{
			float sum = 0.0f;
			for (int k = 0; k < l.graph_k; ++k)
			{
				sum += l.graph_alpha[alpha_index(l, 0, 0, oy, ox, k)];
			}
			EXPECT_NEAR(sum, 1.0f, 1e-5f) << "node=(" << oy << "," << ox << ")";
		}
	}

	free_layer(l);
}

TEST(GraphConvForward, LearnedEdgesRespondToNeighborFeatures)
{
	Darknet::Layer l = make_graph_conv_layer(1, 1, 3, 3, 1, 1, 1, 3, 1, 1,
		1, 1, LINEAR, 0, 0, 0, 0, 0, 0, 0, nullptr, 0, 1, 1, 0, 1);

	std::fill(l.weights, l.weights + l.nweights, 0.0f);
	std::fill(l.biases, l.biases + l.n, 0.0f);
	std::fill(l.graph_edge_kernel, l.graph_edge_kernel + (l.groups * l.graph_k * 2 * l.graph_cpg), 0.0f);
	std::fill(l.graph_edge_biases, l.graph_edge_biases + (l.groups * l.graph_k), 0.0f);

	l.graph_edge_kernel[l.graph_cpg] = 1.0f;

	std::vector<float> input_a(3 * 3, 0.0f);
	std::vector<float> input_b(3 * 3, 0.0f);
	input_b[input_index(l, 0, 0, 0, 0)] = 5.0f;

	Darknet::NetworkState state = {};
	state.train = 1;

	state.input = input_a.data();
	forward_graph_conv_layer(l, state);
	const float alpha_a = l.graph_alpha[alpha_index(l, 0, 0, 1, 1, 0)];

	state.input = input_b.data();
	forward_graph_conv_layer(l, state);
	const float alpha_b = l.graph_alpha[alpha_index(l, 0, 0, 1, 1, 0)];

	EXPECT_NEAR(alpha_a, 1.0f / 9.0f, 1e-5f);
	EXPECT_GT(alpha_b, alpha_a);

	free_layer(l);
}

TEST(GraphConvForward, UniformEdgesReduceToValidMean)
{
	Darknet::Layer l = make_graph_conv_layer(1, 1, 4, 4, 2, 2, 1, 3, 1, 1,
		1, 1, LINEAR, 0, 0, 0, 0, 0, 0, 0, nullptr, 0, 1, 0, 0, 1);

	std::vector<float> input(4 * 4 * 2, 0.0f);
	for (int y = 0; y < 4; ++y)
	{
		for (int x = 0; x < 4; ++x)
		{
			input[input_index(l, 0, 0, y, x)] = static_cast<float>(1 + y * 4 + x);
			input[input_index(l, 0, 1, y, x)] = static_cast<float>(101 + y * 4 + x);
		}
	}

	std::fill(l.weights, l.weights + l.nweights, 0.0f);
	std::fill(l.biases, l.biases + l.n, 0.0f);
	l.weights[0 * l.graph_cpg + 0] = 1.0f;
	l.weights[1 * l.graph_cpg + 1] = 1.0f;

	Darknet::NetworkState state = {};
	state.input = input.data();
	state.train = 1;

	forward_graph_conv_layer(l, state);

	for (int y = 0; y < l.out_h; ++y)
	{
		for (int x = 0; x < l.out_w; ++x)
		{
			EXPECT_NEAR(l.output[output_index(l, 0, 0, y, x)], manual_valid_mean(input, l, 0, y, x), 1e-5f);
			EXPECT_NEAR(l.output[output_index(l, 0, 1, y, x)], manual_valid_mean(input, l, 1, y, x), 1e-5f);
		}
	}

	free_layer(l);
}

TEST(GraphConvForward, Size1ActsLikePointwiseLinear)
{
	Darknet::Layer l = make_graph_conv_layer(1, 1, 3, 3, 2, 2, 1, 1, 1, 1,
		1, 0, LINEAR, 0, 0, 0, 0, 0, 0, 0, nullptr, 0, 1, 0, 0, 1);

	std::vector<float> input(3 * 3 * 2, 0.0f);
	for (size_t i = 0; i < input.size(); ++i)
	{
		input[i] = static_cast<float>(i + 1);
	}

	std::fill(l.weights, l.weights + l.nweights, 0.0f);
	std::fill(l.biases, l.biases + l.n, 0.0f);
	l.weights[0 * l.graph_cpg + 0] = 2.0f;
	l.weights[0 * l.graph_cpg + 1] = -1.0f;
	l.weights[1 * l.graph_cpg + 0] = 0.5f;
	l.weights[1 * l.graph_cpg + 1] = 3.0f;

	Darknet::NetworkState state = {};
	state.input = input.data();
	state.train = 1;

	forward_graph_conv_layer(l, state);

	for (int y = 0; y < l.out_h; ++y)
	{
		for (int x = 0; x < l.out_w; ++x)
		{
			const float in0 = input[input_index(l, 0, 0, y, x)];
			const float in1 = input[input_index(l, 0, 1, y, x)];
			EXPECT_NEAR(l.output[output_index(l, 0, 0, y, x)], 2.0f * in0 - in1, 1e-5f);
			EXPECT_NEAR(l.output[output_index(l, 0, 1, y, x)], 0.5f * in0 + 3.0f * in1, 1e-5f);
		}
	}

	free_layer(l);
}

TEST(GraphConvBackward, ZeroUpstreamGradientStaysZero)
{
	Darknet::Layer l = make_graph_conv_layer(1, 1, 4, 4, 2, 2, 1, 3, 1, 1,
		1, 1, LINEAR, 0, 0, 0, 0, 0, 0, 0, nullptr, 0, 1, 1, 1, 1);

	std::vector<float> input(4 * 4 * 2, 1.0f);
	std::vector<float> input_delta(input.size(), 0.0f);

	Darknet::NetworkState fwd_state = {};
	fwd_state.input = input.data();
	fwd_state.train = 1;
	forward_graph_conv_layer(l, fwd_state);

	std::fill(l.delta, l.delta + l.outputs, 0.0f);

	Darknet::NetworkState back_state = {};
	back_state.input = input.data();
	back_state.delta = input_delta.data();
	back_state.train = 1;
	backward_graph_conv_layer(l, back_state);

	for (int i = 0; i < l.nweights; ++i)
	{
		EXPECT_FLOAT_EQ(l.weight_updates[i], 0.0f);
	}
	for (int i = 0; i < l.n; ++i)
	{
		EXPECT_FLOAT_EQ(l.bias_updates[i], 0.0f);
	}
	for (float v : input_delta)
	{
		EXPECT_FLOAT_EQ(v, 0.0f);
	}
	for (int i = 0; i < l.n * l.graph_cpg; ++i)
	{
		EXPECT_FLOAT_EQ(l.graph_self_weight_updates[i], 0.0f);
	}
	for (int i = 0; i < l.groups * l.graph_k * 2 * l.graph_cpg; ++i)
	{
		EXPECT_FLOAT_EQ(l.graph_edge_kernel_updates[i], 0.0f);
	}
	for (int i = 0; i < l.groups * l.graph_k; ++i)
	{
		EXPECT_FLOAT_EQ(l.graph_edge_bias_updates[i], 0.0f);
	}

	free_layer(l);
}

TEST(GraphConvBackward, GradientCheckMainWeights)
{
	GraphGradCheckContext ctx;
	ctx.init(1, 1);
	ctx.compute_analytical_gradients();

	auto loss_fn = [&ctx](float *, int) -> float { return ctx.compute_loss(); };
	const std::vector<float> num_grad = numerical_gradient(loss_fn, ctx.l.weights, ctx.l.nweights, 1e-4f);

	EXPECT_TRUE(check_gradients(ctx.l.weight_updates, num_grad, ctx.l.nweights, 3e-2f, 1e-4f));
}

TEST(GraphConvBackward, GradientCheckSelfWeights)
{
	GraphGradCheckContext ctx;
	ctx.init(1, 1);
	ctx.compute_analytical_gradients();

	auto loss_fn = [&ctx](float *, int) -> float { return ctx.compute_loss(); };
	const int self_count = ctx.l.n * ctx.l.graph_cpg;
	const std::vector<float> num_grad = numerical_gradient(loss_fn, ctx.l.graph_self_weights, self_count, 1e-4f);

	EXPECT_TRUE(check_gradients(ctx.l.graph_self_weight_updates, num_grad, self_count, 3e-2f, 1e-4f));
}

TEST(GraphConvBackward, GradientCheckEdgeKernel)
{
	GraphGradCheckContext ctx;
	ctx.init(1, 1);
	ctx.compute_analytical_gradients();

	auto loss_fn = [&ctx](float *, int) -> float { return ctx.compute_loss(); };
	const int kernel_count = ctx.l.groups * ctx.l.graph_k * 2 * ctx.l.graph_cpg;
	const std::vector<float> num_grad = numerical_gradient(loss_fn, ctx.l.graph_edge_kernel, kernel_count, 1e-4f);

	EXPECT_TRUE(check_gradients(ctx.l.graph_edge_kernel_updates, num_grad, kernel_count, 7e-2f, 2e-4f));
}

TEST(GraphConvBackward, GradientCheckEdgeBiases)
{
	GraphGradCheckContext ctx;
	ctx.init(1, 1);
	ctx.compute_analytical_gradients();

	auto loss_fn = [&ctx](float *, int) -> float { return ctx.compute_loss(); };
	const int bias_count = ctx.l.groups * ctx.l.graph_k;
	const std::vector<float> num_grad = numerical_gradient(loss_fn, ctx.l.graph_edge_biases, bias_count, 1e-4f);

	EXPECT_TRUE(check_gradients(ctx.l.graph_edge_bias_updates, num_grad, bias_count, 7e-2f, 2e-4f));
}

TEST(GraphConvBackward, GradientCheckInput)
{
	GraphGradCheckContext ctx;
	ctx.init(1, 1);
	ctx.compute_analytical_gradients();

	auto loss_fn = [&ctx](float *, int) -> float { return ctx.compute_loss(); };
	const std::vector<float> num_grad = numerical_gradient(loss_fn, ctx.input.data(), static_cast<int>(ctx.input.size()), 1e-4f);

	EXPECT_TRUE(check_gradients(ctx.input_delta.data(), num_grad, static_cast<int>(ctx.input.size()), 7e-2f, 2e-4f));
}

TEST(GraphConvBackward, GradientCheckBiases)
{
	GraphGradCheckContext ctx;
	ctx.init(1, 1);
	ctx.compute_analytical_gradients();

	auto loss_fn = [&ctx](float *, int) -> float { return ctx.compute_loss(); };
	const std::vector<float> num_grad = numerical_gradient(loss_fn, ctx.l.biases, ctx.l.n, 1e-4f);

	EXPECT_TRUE(check_gradients(ctx.l.bias_updates, num_grad, ctx.l.n, 2e-2f, 1e-5f));
}

#ifdef DARKNET_GPU

TEST(GraphConvGPU, Forward_CPUvsGPU_Batch1)
{
	if (!gpu_tests_enabled())
	{
		GTEST_SKIP() << "GPU tests disabled (unset DARKNET_TEST_GPU=0 to enable)";
	}

	Darknet::Layer l_cpu = make_graph_conv_layer(1, 1, 5, 5, 2, 2, 1, 3, 1, 1,
		1, 1, LINEAR, 0, 0, 0, 0, 0, 0, 0, nullptr, 0, 1, 1, 1, 1);
	Darknet::Layer l_gpu = make_graph_conv_layer(1, 1, 5, 5, 2, 2, 1, 3, 1, 1,
		1, 1, LINEAR, 0, 0, 0, 0, 0, 0, 0, nullptr, 0, 1, 1, 1, 1);

	copy_graph_conv_params(l_cpu, l_gpu);
	push_graph_conv_layer(l_gpu);

	std::vector<float> input(1 * 2 * 5 * 5, 0.0f);
	std::mt19937 rng(77);
	fill_small_random(input.data(), input.size(), rng, 0.1f);

	Darknet::NetworkState state_cpu = {};
	state_cpu.input = input.data();
	state_cpu.train = 0;
	forward_graph_conv_layer(l_cpu, state_cpu);

	float *input_gpu = cuda_make_array(input.data(), input.size());
	Darknet::NetworkState state_gpu = {};
	state_gpu.input = input_gpu;
	state_gpu.train = 0;
	forward_graph_conv_layer_gpu(l_gpu, state_gpu);

	std::vector<float> gpu_output(l_gpu.outputs * l_gpu.batch, 0.0f);
	std::vector<float> gpu_alpha(l_gpu.batch * l_gpu.groups * l_gpu.out_h * l_gpu.out_w * l_gpu.graph_k, 0.0f);
	cuda_pull_array(l_gpu.output_gpu, gpu_output.data(), gpu_output.size());
	cuda_pull_array(l_gpu.graph_alpha_gpu, gpu_alpha.data(), gpu_alpha.size());

	EXPECT_LT(max_abs_diff(l_cpu.output, gpu_output.data(), gpu_output.size()), 1e-5f);
	EXPECT_LT(max_abs_diff(l_cpu.graph_alpha, gpu_alpha.data(), gpu_alpha.size()), 1e-5f);

	cuda_free(input_gpu);
	free_layer(l_cpu);
	free_layer(l_gpu);
}

TEST(GraphConvGPU, Forward_CPUvsGPU_GroupedBatch2)
{
	if (!gpu_tests_enabled())
	{
		GTEST_SKIP() << "GPU tests disabled (unset DARKNET_TEST_GPU=0 to enable)";
	}

	Darknet::Layer l_cpu = make_graph_conv_layer(2, 1, 4, 4, 4, 4, 2, 3, 1, 1,
		1, 1, LINEAR, 0, 0, 0, 0, 0, 0, 0, nullptr, 0, 1, 1, 1, 1);
	Darknet::Layer l_gpu = make_graph_conv_layer(2, 1, 4, 4, 4, 4, 2, 3, 1, 1,
		1, 1, LINEAR, 0, 0, 0, 0, 0, 0, 0, nullptr, 0, 1, 1, 1, 1);

	copy_graph_conv_params(l_cpu, l_gpu);
	push_graph_conv_layer(l_gpu);

	std::vector<float> input(2 * 4 * 4 * 4, 0.0f);
	std::mt19937 rng(99);
	fill_small_random(input.data(), input.size(), rng, 0.1f);

	Darknet::NetworkState state_cpu = {};
	state_cpu.input = input.data();
	state_cpu.train = 0;
	forward_graph_conv_layer(l_cpu, state_cpu);

	float *input_gpu = cuda_make_array(input.data(), input.size());
	Darknet::NetworkState state_gpu = {};
	state_gpu.input = input_gpu;
	state_gpu.train = 0;
	forward_graph_conv_layer_gpu(l_gpu, state_gpu);

	std::vector<float> gpu_output(l_gpu.outputs * l_gpu.batch, 0.0f);
	cuda_pull_array(l_gpu.output_gpu, gpu_output.data(), gpu_output.size());

	EXPECT_LT(max_abs_diff(l_cpu.output, gpu_output.data(), gpu_output.size()), 1e-5f);

	cuda_free(input_gpu);
	free_layer(l_cpu);
	free_layer(l_gpu);
}

TEST(GraphConvGPU, Forward_CPUvsGPU_PointwiseFastPath)
{
	if (!gpu_tests_enabled())
	{
		GTEST_SKIP() << "GPU tests disabled (unset DARKNET_TEST_GPU=0 to enable)";
	}

	Darknet::Layer l_cpu = make_graph_conv_layer(2, 1, 4, 4, 8, 8, 1, 1, 1, 1,
		1, 0, LINEAR, 0, 0, 0, 0, 0, 0, 0, nullptr, 0, 1, 0, 0, 1);
	Darknet::Layer l_gpu = make_graph_conv_layer(2, 1, 4, 4, 8, 8, 1, 1, 1, 1,
		1, 0, LINEAR, 0, 0, 0, 0, 0, 0, 0, nullptr, 0, 1, 0, 0, 1);

	copy_graph_conv_params(l_cpu, l_gpu);
	push_graph_conv_layer(l_gpu);

	std::vector<float> input(2 * 8 * 4 * 4, 0.0f);
	std::mt19937 rng(1001);
	fill_small_random(input.data(), input.size(), rng, 0.1f);

	Darknet::NetworkState state_cpu = {};
	state_cpu.input = input.data();
	state_cpu.train = 1;
	forward_graph_conv_layer(l_cpu, state_cpu);

	float *input_gpu = cuda_make_array(input.data(), input.size());
	Darknet::NetworkState state_gpu = {};
	state_gpu.input = input_gpu;
	state_gpu.train = 1;
	forward_graph_conv_layer_gpu(l_gpu, state_gpu);

	std::vector<float> gpu_output(l_gpu.outputs * l_gpu.batch, 0.0f);
	std::vector<float> gpu_alpha(l_gpu.batch * l_gpu.groups * l_gpu.out_h * l_gpu.out_w * l_gpu.graph_k, 0.0f);
	std::vector<float> gpu_valid(gpu_alpha.size(), 0.0f);
	cuda_pull_array(l_gpu.output_gpu, gpu_output.data(), gpu_output.size());
	cuda_pull_array(l_gpu.graph_alpha_gpu, gpu_alpha.data(), gpu_alpha.size());
	cuda_pull_array(l_gpu.graph_valid_gpu, gpu_valid.data(), gpu_valid.size());

	EXPECT_LT(max_abs_diff(l_cpu.output, gpu_output.data(), gpu_output.size()), 1e-5f);
	EXPECT_LT(max_abs_diff(l_cpu.graph_alpha, gpu_alpha.data(), gpu_alpha.size()), 1e-6f);
	EXPECT_LT(max_abs_diff(l_cpu.graph_valid, gpu_valid.data(), gpu_valid.size()), 1e-6f);

	cuda_free(input_gpu);
	free_layer(l_cpu);
	free_layer(l_gpu);
}

#if defined(CUDNN) && defined(CUDNN_HALF)
TEST(GraphConvGPU, Forward_CPUvsGPU_CudnnHalfProjection)
{
	if (!gpu_tests_enabled())
	{
		GTEST_SKIP() << "GPU tests disabled (unset DARKNET_TEST_GPU=0 to enable)";
	}

	char device_name[1024] = {};
	if (get_gpu_compute_capability(cuda_get_device(), device_name) < 700)
	{
		GTEST_SKIP() << "cuDNN FP16 projection test requires Tensor Core-capable GPU";
	}

	Darknet::Layer l_cpu = make_graph_conv_layer(1, 1, 5, 5, 8, 8, 1, 3, 1, 1,
		1, 1, LINEAR, 0, 0, 0, 0, 0, 0, 0, nullptr, 0, 1, 1, 1, 1);
	Darknet::Layer l_gpu = make_graph_conv_layer(1, 1, 5, 5, 8, 8, 1, 3, 1, 1,
		1, 1, LINEAR, 0, 0, 0, 0, 0, 0, 0, nullptr, 0, 1, 1, 1, 1);

	copy_graph_conv_params(l_cpu, l_gpu);
	push_graph_conv_layer(l_gpu);

	std::vector<float> input(1 * 8 * 5 * 5, 0.0f);
	std::mt19937 rng(1002);
	fill_small_random(input.data(), input.size(), rng, 0.05f);

	Darknet::NetworkState state_cpu = {};
	state_cpu.input = input.data();
	state_cpu.train = 1;
	forward_graph_conv_layer(l_cpu, state_cpu);

	float *input_gpu = cuda_make_array(input.data(), input.size());
	float *input16_gpu = nullptr;
	float *output16_gpu = nullptr;
	size_t max_input16_size = 0;
	size_t max_output16_size = 0;

	Darknet::NetworkState state_gpu = {};
	state_gpu.input = input_gpu;
	state_gpu.train = 1;
	state_gpu.net.cudnn_half = 1;
	state_gpu.net.input16_gpu = &input16_gpu;
	state_gpu.net.output16_gpu = &output16_gpu;
	state_gpu.net.max_input16_size = &max_input16_size;
	state_gpu.net.max_output16_size = &max_output16_size;
	state_gpu.workspace = cuda_make_array(nullptr, l_gpu.workspace_size / sizeof(float) + 1);
	forward_graph_conv_layer_gpu(l_gpu, state_gpu);

	std::vector<float> gpu_output(l_gpu.outputs * l_gpu.batch, 0.0f);
	cuda_pull_array(l_gpu.output_gpu, gpu_output.data(), gpu_output.size());

	EXPECT_LT(max_abs_diff(l_cpu.output, gpu_output.data(), gpu_output.size()), 3e-2f);

	cuda_free(input_gpu);
	if (input16_gpu) cuda_free(input16_gpu);
	if (output16_gpu) cuda_free(output16_gpu);
	cuda_free(state_gpu.workspace);
	free_layer(l_cpu);
	free_layer(l_gpu);
}

TEST(GraphConvGPU, Backward_CPUvsGPU_CudnnHalfMixedGraph)
{
	if (!gpu_tests_enabled())
	{
		GTEST_SKIP() << "GPU tests disabled (unset DARKNET_TEST_GPU=0 to enable)";
	}

	char device_name[1024] = {};
	if (get_gpu_compute_capability(cuda_get_device(), device_name) < 700)
	{
		GTEST_SKIP() << "cuDNN FP16 graph test requires Tensor Core-capable GPU";
	}

	Darknet::Layer l_cpu = make_graph_conv_layer(1, 1, 5, 5, 8, 8, 1, 3, 1, 1,
		1, 1, LINEAR, 0, 0, 0, 0, 0, 0, 0, nullptr, 0, 1, 1, 1, 1);
	Darknet::Layer l_gpu = make_graph_conv_layer(1, 1, 5, 5, 8, 8, 1, 3, 1, 1,
		1, 1, LINEAR, 0, 0, 0, 0, 0, 0, 0, nullptr, 0, 1, 1, 1, 1);

	copy_graph_conv_params(l_cpu, l_gpu);
	push_graph_conv_layer(l_gpu);

	std::vector<float> input(1 * 8 * 5 * 5, 0.0f);
	std::vector<float> input_delta_cpu(input.size(), 0.0f);
	std::mt19937 rng(1003);
	fill_small_random(input.data(), input.size(), rng, 0.05f);

	Darknet::NetworkState fwd_cpu = {};
	fwd_cpu.input = input.data();
	fwd_cpu.train = 1;
	forward_graph_conv_layer(l_cpu, fwd_cpu);

	float *input_gpu = cuda_make_array(input.data(), input.size());
	float *input_delta_gpu = cuda_make_array(nullptr, input.size());
	float *input16_gpu = nullptr;
	float *output16_gpu = nullptr;
	size_t max_input16_size = 0;
	size_t max_output16_size = 0;
	fill_ongpu(input.size(), 0.0f, input_delta_gpu, 1);

	Darknet::NetworkState fwd_gpu = {};
	fwd_gpu.input = input_gpu;
	fwd_gpu.train = 1;
	fwd_gpu.net.cudnn_half = 1;
	fwd_gpu.net.input16_gpu = &input16_gpu;
	fwd_gpu.net.output16_gpu = &output16_gpu;
	fwd_gpu.net.max_input16_size = &max_input16_size;
	fwd_gpu.net.max_output16_size = &max_output16_size;
	fwd_gpu.workspace = cuda_make_array(nullptr, l_gpu.workspace_size / sizeof(float) + 1);
	forward_graph_conv_layer_gpu(l_gpu, fwd_gpu);

	std::vector<float> delta(l_cpu.outputs * l_cpu.batch, 0.0f);
	fill_small_random(delta.data(), delta.size(), rng, 0.05f);
	std::copy(delta.begin(), delta.end(), l_cpu.delta);
	cuda_push_array(l_gpu.delta_gpu, delta.data(), delta.size());

	std::fill(l_cpu.weight_updates, l_cpu.weight_updates + l_cpu.nweights, 0.0f);
	std::fill(l_cpu.bias_updates, l_cpu.bias_updates + l_cpu.n, 0.0f);
	std::fill(l_cpu.graph_self_weight_updates, l_cpu.graph_self_weight_updates + l_cpu.n * l_cpu.graph_cpg, 0.0f);
	std::fill(l_cpu.graph_edge_kernel_updates, l_cpu.graph_edge_kernel_updates + l_cpu.groups * l_cpu.graph_k * 2 * l_cpu.graph_cpg, 0.0f);
	std::fill(l_cpu.graph_edge_bias_updates, l_cpu.graph_edge_bias_updates + l_cpu.groups * l_cpu.graph_k, 0.0f);

	fill_ongpu(l_gpu.nweights, 0.0f, l_gpu.weight_updates_gpu, 1);
	fill_ongpu(l_gpu.n, 0.0f, l_gpu.bias_updates_gpu, 1);
	fill_ongpu(l_gpu.n * l_gpu.graph_cpg, 0.0f, l_gpu.graph_self_weight_updates_gpu, 1);
	fill_ongpu(l_gpu.groups * l_gpu.graph_k * 2 * l_gpu.graph_cpg, 0.0f, l_gpu.graph_edge_kernel_updates_gpu, 1);
	fill_ongpu(l_gpu.groups * l_gpu.graph_k, 0.0f, l_gpu.graph_edge_bias_updates_gpu, 1);

	Darknet::NetworkState back_cpu = {};
	back_cpu.input = input.data();
	back_cpu.delta = input_delta_cpu.data();
	back_cpu.train = 1;
	backward_graph_conv_layer(l_cpu, back_cpu);

	Darknet::NetworkState back_gpu = fwd_gpu;
	back_gpu.delta = input_delta_gpu;
	back_gpu.net.try_fix_nan = 0;
	backward_graph_conv_layer_gpu(l_gpu, back_gpu);

	std::vector<float> gpu_weight_updates(l_gpu.nweights, 0.0f);
	std::vector<float> gpu_bias_updates(l_gpu.n, 0.0f);
	std::vector<float> gpu_self_updates(l_gpu.n * l_gpu.graph_cpg, 0.0f);
	std::vector<float> gpu_edge_kernel_updates(l_gpu.groups * l_gpu.graph_k * 2 * l_gpu.graph_cpg, 0.0f);
	std::vector<float> gpu_edge_bias_updates(l_gpu.groups * l_gpu.graph_k, 0.0f);
	std::vector<float> gpu_input_delta(input.size(), 0.0f);

	cuda_pull_array(l_gpu.weight_updates_gpu, gpu_weight_updates.data(), gpu_weight_updates.size());
	cuda_pull_array(l_gpu.bias_updates_gpu, gpu_bias_updates.data(), gpu_bias_updates.size());
	cuda_pull_array(l_gpu.graph_self_weight_updates_gpu, gpu_self_updates.data(), gpu_self_updates.size());
	cuda_pull_array(l_gpu.graph_edge_kernel_updates_gpu, gpu_edge_kernel_updates.data(), gpu_edge_kernel_updates.size());
	cuda_pull_array(l_gpu.graph_edge_bias_updates_gpu, gpu_edge_bias_updates.data(), gpu_edge_bias_updates.size());
	cuda_pull_array(input_delta_gpu, gpu_input_delta.data(), gpu_input_delta.size());

	EXPECT_LT(max_abs_diff(l_cpu.weight_updates, gpu_weight_updates.data(), gpu_weight_updates.size()), 5e-2f);
	EXPECT_LT(max_abs_diff(l_cpu.bias_updates, gpu_bias_updates.data(), gpu_bias_updates.size()), 5e-2f);
	EXPECT_LT(max_abs_diff(l_cpu.graph_self_weight_updates, gpu_self_updates.data(), gpu_self_updates.size()), 5e-2f);
	EXPECT_LT(max_abs_diff(l_cpu.graph_edge_kernel_updates, gpu_edge_kernel_updates.data(), gpu_edge_kernel_updates.size()), 5e-2f);
	EXPECT_LT(max_abs_diff(l_cpu.graph_edge_bias_updates, gpu_edge_bias_updates.data(), gpu_edge_bias_updates.size()), 5e-2f);
	EXPECT_LT(max_abs_diff(input_delta_cpu.data(), gpu_input_delta.data(), gpu_input_delta.size()), 5e-2f);

	cuda_free(input_gpu);
	cuda_free(input_delta_gpu);
	if (input16_gpu) cuda_free(input16_gpu);
	if (output16_gpu) cuda_free(output16_gpu);
	cuda_free(fwd_gpu.workspace);
	free_layer(l_cpu);
	free_layer(l_gpu);
}
#endif

TEST(GraphConvGPU, Backward_CPUvsGPU)
{
	if (!gpu_tests_enabled())
	{
		GTEST_SKIP() << "GPU tests disabled (unset DARKNET_TEST_GPU=0 to enable)";
	}

	Darknet::Layer l_cpu = make_graph_conv_layer(2, 1, 4, 4, 2, 2, 1, 3, 1, 1,
		1, 1, LINEAR, 0, 0, 0, 0, 0, 0, 0, nullptr, 0, 1, 1, 1, 1);
	Darknet::Layer l_gpu = make_graph_conv_layer(2, 1, 4, 4, 2, 2, 1, 3, 1, 1,
		1, 1, LINEAR, 0, 0, 0, 0, 0, 0, 0, nullptr, 0, 1, 1, 1, 1);

	copy_graph_conv_params(l_cpu, l_gpu);
	push_graph_conv_layer(l_gpu);

	std::vector<float> input(2 * 2 * 4 * 4, 0.0f);
	std::vector<float> input_delta_cpu(input.size(), 0.0f);
	std::mt19937 rng(123);
	fill_small_random(input.data(), input.size(), rng, 0.1f);

	Darknet::NetworkState fwd_cpu = {};
	fwd_cpu.input = input.data();
	fwd_cpu.train = 1;
	forward_graph_conv_layer(l_cpu, fwd_cpu);

	float *input_gpu = cuda_make_array(input.data(), input.size());
	float *input_delta_gpu = cuda_make_array(nullptr, input.size());
	fill_ongpu(input.size(), 0.0f, input_delta_gpu, 1);

	Darknet::NetworkState fwd_gpu = {};
	fwd_gpu.input = input_gpu;
	fwd_gpu.train = 1;
	forward_graph_conv_layer_gpu(l_gpu, fwd_gpu);

	std::vector<float> delta(l_cpu.outputs * l_cpu.batch, 0.0f);
	fill_small_random(delta.data(), delta.size(), rng, 0.1f);
	std::copy(delta.begin(), delta.end(), l_cpu.delta);
	cuda_push_array(l_gpu.delta_gpu, delta.data(), delta.size());

	std::fill(l_cpu.weight_updates, l_cpu.weight_updates + l_cpu.nweights, 0.0f);
	std::fill(l_cpu.bias_updates, l_cpu.bias_updates + l_cpu.n, 0.0f);
	std::fill(l_cpu.graph_self_weight_updates, l_cpu.graph_self_weight_updates + l_cpu.n * l_cpu.graph_cpg, 0.0f);
	std::fill(l_cpu.graph_edge_kernel_updates, l_cpu.graph_edge_kernel_updates + l_cpu.groups * l_cpu.graph_k * 2 * l_cpu.graph_cpg, 0.0f);
	std::fill(l_cpu.graph_edge_bias_updates, l_cpu.graph_edge_bias_updates + l_cpu.groups * l_cpu.graph_k, 0.0f);

	fill_ongpu(l_gpu.nweights, 0.0f, l_gpu.weight_updates_gpu, 1);
	fill_ongpu(l_gpu.n, 0.0f, l_gpu.bias_updates_gpu, 1);
	fill_ongpu(l_gpu.n * l_gpu.graph_cpg, 0.0f, l_gpu.graph_self_weight_updates_gpu, 1);
	fill_ongpu(l_gpu.groups * l_gpu.graph_k * 2 * l_gpu.graph_cpg, 0.0f, l_gpu.graph_edge_kernel_updates_gpu, 1);
	fill_ongpu(l_gpu.groups * l_gpu.graph_k, 0.0f, l_gpu.graph_edge_bias_updates_gpu, 1);

	Darknet::NetworkState back_cpu = {};
	back_cpu.input = input.data();
	back_cpu.delta = input_delta_cpu.data();
	back_cpu.train = 1;
	backward_graph_conv_layer(l_cpu, back_cpu);

	Darknet::NetworkState back_gpu = {};
	back_gpu.input = input_gpu;
	back_gpu.delta = input_delta_gpu;
	back_gpu.train = 1;
	back_gpu.net.try_fix_nan = 0;
	backward_graph_conv_layer_gpu(l_gpu, back_gpu);

	std::vector<float> gpu_weight_updates(l_gpu.nweights, 0.0f);
	std::vector<float> gpu_bias_updates(l_gpu.n, 0.0f);
	std::vector<float> gpu_self_updates(l_gpu.n * l_gpu.graph_cpg, 0.0f);
	std::vector<float> gpu_edge_kernel_updates(l_gpu.groups * l_gpu.graph_k * 2 * l_gpu.graph_cpg, 0.0f);
	std::vector<float> gpu_edge_bias_updates(l_gpu.groups * l_gpu.graph_k, 0.0f);
	std::vector<float> gpu_input_delta(input.size(), 0.0f);

	cuda_pull_array(l_gpu.weight_updates_gpu, gpu_weight_updates.data(), gpu_weight_updates.size());
	cuda_pull_array(l_gpu.bias_updates_gpu, gpu_bias_updates.data(), gpu_bias_updates.size());
	cuda_pull_array(l_gpu.graph_self_weight_updates_gpu, gpu_self_updates.data(), gpu_self_updates.size());
	cuda_pull_array(l_gpu.graph_edge_kernel_updates_gpu, gpu_edge_kernel_updates.data(), gpu_edge_kernel_updates.size());
	cuda_pull_array(l_gpu.graph_edge_bias_updates_gpu, gpu_edge_bias_updates.data(), gpu_edge_bias_updates.size());
	cuda_pull_array(input_delta_gpu, gpu_input_delta.data(), gpu_input_delta.size());

	EXPECT_LT(max_abs_diff(l_cpu.weight_updates, gpu_weight_updates.data(), gpu_weight_updates.size()), 1e-4f);
	EXPECT_LT(max_abs_diff(l_cpu.bias_updates, gpu_bias_updates.data(), gpu_bias_updates.size()), 1e-4f);
	EXPECT_LT(max_abs_diff(l_cpu.graph_self_weight_updates, gpu_self_updates.data(), gpu_self_updates.size()), 1e-4f);
	EXPECT_LT(max_abs_diff(l_cpu.graph_edge_kernel_updates, gpu_edge_kernel_updates.data(), gpu_edge_kernel_updates.size()), 1e-4f);
	EXPECT_LT(max_abs_diff(l_cpu.graph_edge_bias_updates, gpu_edge_bias_updates.data(), gpu_edge_bias_updates.size()), 1e-4f);
	EXPECT_LT(max_abs_diff(input_delta_cpu.data(), gpu_input_delta.data(), gpu_input_delta.size()), 1e-4f);

	cuda_free(input_gpu);
	cuda_free(input_delta_gpu);
	free_layer(l_cpu);
	free_layer(l_gpu);
}

TEST(GraphConvGPU, Backward_CPUvsGPU_PointwiseFastPath)
{
	if (!gpu_tests_enabled())
	{
		GTEST_SKIP() << "GPU tests disabled (unset DARKNET_TEST_GPU=0 to enable)";
	}

	Darknet::Layer l_cpu = make_graph_conv_layer(2, 1, 4, 4, 8, 8, 1, 1, 1, 1,
		1, 0, LINEAR, 0, 0, 0, 0, 0, 0, 0, nullptr, 0, 1, 0, 0, 1);
	Darknet::Layer l_gpu = make_graph_conv_layer(2, 1, 4, 4, 8, 8, 1, 1, 1, 1,
		1, 0, LINEAR, 0, 0, 0, 0, 0, 0, 0, nullptr, 0, 1, 0, 0, 1);

	copy_graph_conv_params(l_cpu, l_gpu);
	push_graph_conv_layer(l_gpu);

	std::vector<float> input(2 * 8 * 4 * 4, 0.0f);
	std::vector<float> input_delta_cpu(input.size(), 0.0f);
	std::mt19937 rng(1003);
	fill_small_random(input.data(), input.size(), rng, 0.1f);

	Darknet::NetworkState fwd_cpu = {};
	fwd_cpu.input = input.data();
	fwd_cpu.train = 1;
	forward_graph_conv_layer(l_cpu, fwd_cpu);

	float *input_gpu = cuda_make_array(input.data(), input.size());
	float *input_delta_gpu = cuda_make_array(nullptr, input.size());
	fill_ongpu(input.size(), 0.0f, input_delta_gpu, 1);

	Darknet::NetworkState fwd_gpu = {};
	fwd_gpu.input = input_gpu;
	fwd_gpu.train = 1;
	forward_graph_conv_layer_gpu(l_gpu, fwd_gpu);

	std::vector<float> delta(l_cpu.outputs * l_cpu.batch, 0.0f);
	fill_small_random(delta.data(), delta.size(), rng, 0.1f);
	std::copy(delta.begin(), delta.end(), l_cpu.delta);
	cuda_push_array(l_gpu.delta_gpu, delta.data(), delta.size());

	std::fill(l_cpu.weight_updates, l_cpu.weight_updates + l_cpu.nweights, 0.0f);
	std::fill(l_cpu.bias_updates, l_cpu.bias_updates + l_cpu.n, 0.0f);
	fill_ongpu(l_gpu.nweights, 0.0f, l_gpu.weight_updates_gpu, 1);
	fill_ongpu(l_gpu.n, 0.0f, l_gpu.bias_updates_gpu, 1);

	Darknet::NetworkState back_cpu = {};
	back_cpu.input = input.data();
	back_cpu.delta = input_delta_cpu.data();
	back_cpu.train = 1;
	backward_graph_conv_layer(l_cpu, back_cpu);

	Darknet::NetworkState back_gpu = {};
	back_gpu.input = input_gpu;
	back_gpu.delta = input_delta_gpu;
	back_gpu.train = 1;
	back_gpu.net.try_fix_nan = 0;
	backward_graph_conv_layer_gpu(l_gpu, back_gpu);

	std::vector<float> gpu_weight_updates(l_gpu.nweights, 0.0f);
	std::vector<float> gpu_bias_updates(l_gpu.n, 0.0f);
	std::vector<float> gpu_input_delta(input.size(), 0.0f);
	cuda_pull_array(l_gpu.weight_updates_gpu, gpu_weight_updates.data(), gpu_weight_updates.size());
	cuda_pull_array(l_gpu.bias_updates_gpu, gpu_bias_updates.data(), gpu_bias_updates.size());
	cuda_pull_array(input_delta_gpu, gpu_input_delta.data(), gpu_input_delta.size());

	EXPECT_LT(max_abs_diff(l_cpu.weight_updates, gpu_weight_updates.data(), gpu_weight_updates.size()), 1e-4f);
	EXPECT_LT(max_abs_diff(l_cpu.bias_updates, gpu_bias_updates.data(), gpu_bias_updates.size()), 1e-4f);
	EXPECT_LT(max_abs_diff(input_delta_cpu.data(), gpu_input_delta.data(), gpu_input_delta.size()), 1e-4f);

	cuda_free(input_gpu);
	cuda_free(input_delta_gpu);
	free_layer(l_cpu);
	free_layer(l_gpu);
}

#endif
