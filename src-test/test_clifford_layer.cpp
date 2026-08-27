#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <vector>

#include "clifford_layer.hpp"
#include "darknet_internal.hpp"

namespace
{
	size_t workspace_float_count(const Darknet::Layer & l)
	{
		return (l.workspace_size + sizeof(float) - 1) / sizeof(float);
	}

	void fill_pattern(float *data, size_t n, float scale)
	{
		for (size_t i = 0; i < n; ++i)
		{
			data[i] = scale * std::sin(static_cast<float>(i + 1) * 0.173f);
		}
	}

	void expect_all_finite(const float *data, size_t n, const char *label)
	{
		for (size_t i = 0; i < n; ++i)
		{
			ASSERT_TRUE(std::isfinite(data[i])) << label << " idx=" << i << " val=" << data[i];
		}
	}

	bool has_signal(const float *data, size_t n, float threshold = 1e-7f)
	{
		for (size_t i = 0; i < n; ++i)
		{
			if (std::fabs(data[i]) > threshold)
			{
				return true;
			}
		}
		return false;
	}

	bool gpu_tests_enabled()
	{
		const char *gpu_mode = std::getenv("DARKNET_TEST_GPU");
		return !(gpu_mode && std::string(gpu_mode) == "0");
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

	struct CliffordContext
	{
		Darknet::Layer l = { static_cast<Darknet::ELayerType>(0) };
		std::vector<int> shifts;
		std::vector<float> input;
		std::vector<float> input_delta;
		std::vector<float> workspace;
		bool initialized = false;

			void init(int gffn_mode, int num_dwconv, float layer_scale, int higher_mode = 0)
			{
				shifts = {1, 2};
				l = make_clifford_layer(
					1, 3, 3, 4, 4,
					shifts.data(), static_cast<int>(shifts.size()),
					nullptr, 0,
					1, 2, gffn_mode, higher_mode,
					3, num_dwconv,
					SWISH, 0.0f, layer_scale,
					0, 1);

			input.resize(static_cast<size_t>(l.batch) * l.inputs);
			input_delta.assign(input.size(), 0.0f);
			workspace.assign(workspace_float_count(l), 0.0f);
			fill_pattern(input.data(), input.size(), 0.25f);

			std::fill(l.cli_layer_scale, l.cli_layer_scale + l.c, layer_scale);
			initialized = true;
		}

		Darknet::NetworkState make_state()
		{
			Darknet::NetworkState state = {};
			state.input = input.data();
			state.delta = input_delta.data();
			state.workspace = workspace.empty() ? nullptr : workspace.data();
			state.train = 1;
			return state;
		}

		~CliffordContext()
		{
			if (initialized)
			{
				free_layer(l);
			}
		}
	};
}

TEST(CliffordLayer, ZeroLayerScalePreservesResidualIdentity)
{
	CliffordContext ctx;
	ctx.init(/*gffn_mode=*/2, /*num_dwconv=*/1, /*layer_scale=*/0.0f);

	auto state = ctx.make_state();
	forward_clifford_layer(ctx.l, state);

	expect_all_finite(ctx.l.output, static_cast<size_t>(ctx.l.batch) * ctx.l.outputs, "output");
	for (size_t i = 0; i < ctx.input.size(); ++i)
	{
		EXPECT_FLOAT_EQ(ctx.l.output[i], ctx.input[i]) << "idx=" << i;
	}
}

#ifdef DARKNET_GPU

TEST(CliffordLayerGPU, ForwardBackwardCPUvsGPU_VectorBivectorMode)
{
	if (!gpu_tests_enabled())
	{
		GTEST_SKIP() << "GPU tests disabled (unset DARKNET_TEST_GPU=0 to enable)";
	}

	const int old_gpu_index = Darknet::CfgAndState::get().gpu_index;
	Darknet::set_gpu_index(-1);

	std::vector<int> shifts = {1, 2};
	Darknet::Layer l_cpu = make_clifford_layer(
		1, 3, 3, 4, 4,
		shifts.data(), static_cast<int>(shifts.size()),
		nullptr, 0,
		1, 2, 0, 1,
		3, 1,
		SWISH, 0.0f, 1.0f,
		0, 1);

	Darknet::set_gpu_index(0);
	Darknet::Layer l_gpu = make_clifford_layer(
		1, 3, 3, 4, 4,
		shifts.data(), static_cast<int>(shifts.size()),
		nullptr, 0,
		1, 2, 0, 1,
		3, 1,
		SWISH, 0.0f, 1.0f,
		0, 1);

	std::copy(l_cpu.cli_w_det, l_cpu.cli_w_det + l_cpu.c * l_cpu.c, l_gpu.cli_w_det);
	std::copy(l_cpu.cli_b_det, l_cpu.cli_b_det + l_cpu.c, l_gpu.cli_b_det);
	std::copy(l_cpu.cli_w_proj, l_cpu.cli_w_proj + l_cpu.c * l_cpu.cli_proj_in_dim, l_gpu.cli_w_proj);
	std::copy(l_cpu.cli_b_proj, l_cpu.cli_b_proj + l_cpu.c, l_gpu.cli_b_proj);
	std::copy(l_cpu.cli_w_gate, l_cpu.cli_w_gate + l_cpu.c * 2 * l_cpu.c, l_gpu.cli_w_gate);
	std::copy(l_cpu.cli_b_gate, l_cpu.cli_b_gate + l_cpu.c, l_gpu.cli_b_gate);
	std::copy(l_cpu.cli_ln_gamma, l_cpu.cli_ln_gamma + l_cpu.c, l_gpu.cli_ln_gamma);
	std::copy(l_cpu.cli_ln_beta, l_cpu.cli_ln_beta + l_cpu.c, l_gpu.cli_ln_beta);
	std::copy(l_cpu.cli_layer_scale, l_cpu.cli_layer_scale + l_cpu.c, l_gpu.cli_layer_scale);
	for (int i = 0; i < l_cpu.cli_num_dwconv; ++i)
	{
		// Keep this parity test focused on Clifford kernels; CPU BN and CUDNN BN
		// use different variance conventions on tiny maps.
		l_cpu.cli_dwconv[i].batch_normalize = 0;
		l_gpu.cli_dwconv[i].batch_normalize = 0;
		std::copy(l_cpu.cli_dwconv[i].weights, l_cpu.cli_dwconv[i].weights + l_cpu.cli_dwconv[i].nweights, l_gpu.cli_dwconv[i].weights);
		std::copy(l_cpu.cli_dwconv[i].biases, l_cpu.cli_dwconv[i].biases + l_cpu.cli_dwconv[i].n, l_gpu.cli_dwconv[i].biases);
	}
	push_clifford_layer(l_gpu);

	std::vector<float> input(static_cast<size_t>(l_cpu.batch) * l_cpu.inputs, 0.0f);
	std::vector<float> input_delta_cpu(input.size(), 0.0f);
	fill_pattern(input.data(), input.size(), 0.25f);

	std::vector<float> workspace_cpu(workspace_float_count(l_cpu), 0.0f);
	Darknet::set_gpu_index(-1);
	Darknet::NetworkState state_cpu = {};
	state_cpu.input = input.data();
	state_cpu.workspace = workspace_cpu.data();
	state_cpu.train = 1;
	forward_clifford_layer(l_cpu, state_cpu);

	Darknet::set_gpu_index(0);
	float *input_gpu = cuda_make_array(input.data(), input.size());
	float *input_delta_gpu = cuda_make_array(nullptr, input.size());
	fill_ongpu(static_cast<int>(input.size()), 0.0f, input_delta_gpu, 1);
	Darknet::Network net_gpu = make_network(0);
	net_gpu.batch = l_gpu.batch;
	net_gpu.subdivisions = 1;
	net_gpu.max_batches = 1;
	net_gpu.loss_scale = 1.0f;
	net_gpu.workspace = cuda_make_array(nullptr, workspace_float_count(l_gpu));
	Darknet::NetworkState state_gpu = {};
	state_gpu.input = input_gpu;
	state_gpu.delta = input_delta_gpu;
	state_gpu.workspace = net_gpu.workspace;
	state_gpu.train = 1;
	state_gpu.net = net_gpu;
	forward_clifford_layer_gpu(l_gpu, state_gpu);

	std::vector<float> gpu_output(static_cast<size_t>(l_gpu.batch) * l_gpu.outputs, 0.0f);
	std::vector<float> gpu_vb(static_cast<size_t>(l_gpu.batch) * l_gpu.outputs, 0.0f);
	cuda_pull_array(l_gpu.output_gpu, gpu_output.data(), gpu_output.size());
	cuda_pull_array(l_gpu.cli_vb_feat_gpu, gpu_vb.data(), gpu_vb.size());

	EXPECT_LT(max_abs_diff(l_cpu.output, gpu_output.data(), gpu_output.size()), 1e-4f);
	EXPECT_LT(max_abs_diff(l_cpu.cli_vb_feat, gpu_vb.data(), gpu_vb.size()), 1e-4f);

	std::vector<float> delta(static_cast<size_t>(l_cpu.batch) * l_cpu.outputs, 0.0f);
	for (size_t i = 0; i < delta.size(); ++i)
	{
		delta[i] = 0.125f * l_cpu.output[i] - 0.03f * std::sin(static_cast<float>(i + 1));
		l_cpu.delta[i] = delta[i];
	}
	cuda_push_array(l_gpu.delta_gpu, delta.data(), delta.size());

	std::fill(l_cpu.cli_w_det_updates, l_cpu.cli_w_det_updates + l_cpu.c * l_cpu.c, 0.0f);
	std::fill(l_cpu.cli_w_proj_updates, l_cpu.cli_w_proj_updates + l_cpu.c * l_cpu.cli_proj_in_dim, 0.0f);
	Darknet::set_gpu_index(-1);
	Darknet::NetworkState back_cpu = {};
	back_cpu.input = input.data();
	back_cpu.delta = input_delta_cpu.data();
	back_cpu.workspace = workspace_cpu.data();
	back_cpu.train = 1;
	backward_clifford_layer(l_cpu, back_cpu);

	Darknet::set_gpu_index(0);
	fill_ongpu(l_gpu.c * l_gpu.c, 0.0f, l_gpu.cli_w_det_updates_gpu, 1);
	fill_ongpu(l_gpu.c * l_gpu.cli_proj_in_dim, 0.0f, l_gpu.cli_w_proj_updates_gpu, 1);
	Darknet::NetworkState back_gpu = {};
	back_gpu.input = input_gpu;
	back_gpu.delta = input_delta_gpu;
	back_gpu.workspace = net_gpu.workspace;
	back_gpu.train = 1;
	back_gpu.net = net_gpu;
	backward_clifford_layer_gpu(l_gpu, back_gpu);

	std::vector<float> gpu_w_det_updates(static_cast<size_t>(l_gpu.c) * l_gpu.c, 0.0f);
	std::vector<float> gpu_w_proj_updates(static_cast<size_t>(l_gpu.c) * l_gpu.cli_proj_in_dim, 0.0f);
	std::vector<float> gpu_input_delta(input.size(), 0.0f);
	cuda_pull_array(l_gpu.cli_w_det_updates_gpu, gpu_w_det_updates.data(), gpu_w_det_updates.size());
	cuda_pull_array(l_gpu.cli_w_proj_updates_gpu, gpu_w_proj_updates.data(), gpu_w_proj_updates.size());
	cuda_pull_array(input_delta_gpu, gpu_input_delta.data(), gpu_input_delta.size());

	EXPECT_LT(max_abs_diff(l_cpu.cli_w_det_updates, gpu_w_det_updates.data(), gpu_w_det_updates.size()), 1e-3f);
	EXPECT_LT(max_abs_diff(l_cpu.cli_w_proj_updates, gpu_w_proj_updates.data(), gpu_w_proj_updates.size()), 1e-3f);
	EXPECT_LT(max_abs_diff(input_delta_cpu.data(), gpu_input_delta.data(), gpu_input_delta.size()), 1e-3f);

	cuda_free(input_gpu);
	cuda_free(input_delta_gpu);
	free_layer(l_cpu);
	free_layer(l_gpu);
	free_network(net_gpu);
	Darknet::set_gpu_index(old_gpu_index);
}

#endif

TEST(CliffordLayer, TwoDwconvStackBackwardProducesFiniteGradients)
{
	CliffordContext ctx;
	ctx.init(/*gffn_mode=*/0, /*num_dwconv=*/2, /*layer_scale=*/1.0f);

	auto state = ctx.make_state();
	forward_clifford_layer(ctx.l, state);
	expect_all_finite(ctx.l.output, static_cast<size_t>(ctx.l.batch) * ctx.l.outputs, "forward output");

	for (int i = 0; i < ctx.l.outputs * ctx.l.batch; ++i)
	{
		ctx.l.delta[i] = ctx.l.output[i] - 0.05f * std::cos(static_cast<float>(i + 1));
	}

	backward_clifford_layer(ctx.l, state);

	expect_all_finite(ctx.input_delta.data(), ctx.input_delta.size(), "input_delta");
	expect_all_finite(ctx.l.cli_w_det_updates, static_cast<size_t>(ctx.l.c) * ctx.l.c, "cli_w_det_updates");
	expect_all_finite(ctx.l.cli_w_proj_updates, static_cast<size_t>(ctx.l.c) * ctx.l.cli_proj_in_dim, "cli_w_proj_updates");
	expect_all_finite(ctx.l.cli_dwconv[0].weight_updates, ctx.l.cli_dwconv[0].nweights, "dwconv0 weight_updates");
	expect_all_finite(ctx.l.cli_dwconv[1].weight_updates, ctx.l.cli_dwconv[1].nweights, "dwconv1 weight_updates");

	EXPECT_TRUE(has_signal(ctx.input_delta.data(), ctx.input_delta.size()));
	EXPECT_TRUE(has_signal(ctx.l.cli_w_det_updates, static_cast<size_t>(ctx.l.c) * ctx.l.c));
	EXPECT_TRUE(has_signal(ctx.l.cli_w_proj_updates, static_cast<size_t>(ctx.l.c) * ctx.l.cli_proj_in_dim));
	EXPECT_TRUE(has_signal(ctx.l.cli_dwconv[0].weight_updates, ctx.l.cli_dwconv[0].nweights));
	EXPECT_TRUE(has_signal(ctx.l.cli_dwconv[1].weight_updates, ctx.l.cli_dwconv[1].nweights));
}

TEST(CliffordLayer, GlobalOnlyBranchProducesGlobalGradients)
{
	CliffordContext ctx;
	ctx.init(/*gffn_mode=*/1, /*num_dwconv=*/1, /*layer_scale=*/1.0f);

	auto state = ctx.make_state();
	forward_clifford_layer(ctx.l, state);
	expect_all_finite(ctx.l.output, static_cast<size_t>(ctx.l.batch) * ctx.l.outputs, "forward output");

	for (int i = 0; i < ctx.l.outputs * ctx.l.batch; ++i)
	{
		ctx.l.delta[i] = 0.1f * ctx.l.output[i] + 0.02f;
	}

	backward_clifford_layer(ctx.l, state);

	ASSERT_NE(ctx.l.cli_w_proj_g_updates, nullptr);
	ASSERT_NE(ctx.l.cli_w_gate_g_updates, nullptr);

	expect_all_finite(ctx.l.cli_w_proj_g_updates, static_cast<size_t>(ctx.l.c) * ctx.l.cli_proj_in_dim, "cli_w_proj_g_updates");
	expect_all_finite(ctx.l.cli_w_gate_g_updates, static_cast<size_t>(ctx.l.c) * 2 * ctx.l.c, "cli_w_gate_g_updates");
	expect_all_finite(ctx.input_delta.data(), ctx.input_delta.size(), "input_delta");

	EXPECT_TRUE(has_signal(ctx.l.cli_w_proj_g_updates, static_cast<size_t>(ctx.l.c) * ctx.l.cli_proj_in_dim));
	EXPECT_TRUE(has_signal(ctx.l.cli_w_gate_g_updates, static_cast<size_t>(ctx.l.c) * 2 * ctx.l.c));
	EXPECT_TRUE(has_signal(ctx.input_delta.data(), ctx.input_delta.size()));
}

TEST(CliffordLayer, GlobalOnlyForwardDoesNotDependOnLinearDet)
{
	CliffordContext ctx;
	ctx.init(/*gffn_mode=*/1, /*num_dwconv=*/1, /*layer_scale=*/1.0f);

	auto state = ctx.make_state();
	forward_clifford_layer(ctx.l, state);
	std::vector<float> baseline(ctx.l.output, ctx.l.output + static_cast<size_t>(ctx.l.batch) * ctx.l.outputs);

	for (int i = 0; i < ctx.l.c * ctx.l.c; ++i)
	{
		ctx.l.cli_w_det[i] = (i % 5 == 0) ? 7.0f : -3.5f;
	}
	for (int i = 0; i < ctx.l.c; ++i)
	{
		ctx.l.cli_b_det[i] = 2.0f - 0.25f * i;
	}

	forward_clifford_layer(ctx.l, state);

	for (size_t i = 0; i < baseline.size(); ++i)
	{
		EXPECT_FLOAT_EQ(ctx.l.output[i], baseline[i]) << "idx=" << i;
	}
}

TEST(CliffordLayer, VectorBivectorModeProducesFiniteGradients)
{
	CliffordContext ctx;
	ctx.init(/*gffn_mode=*/0, /*num_dwconv=*/1, /*layer_scale=*/1.0f, /*higher_mode=*/1);

	auto state = ctx.make_state();
	forward_clifford_layer(ctx.l, state);
	expect_all_finite(ctx.l.output, static_cast<size_t>(ctx.l.batch) * ctx.l.outputs, "forward output");
	ASSERT_NE(ctx.l.cli_vb_feat, nullptr);
	expect_all_finite(ctx.l.cli_vb_feat, static_cast<size_t>(ctx.l.batch) * ctx.l.outputs, "cli_vb_feat");

	for (int i = 0; i < ctx.l.outputs * ctx.l.batch; ++i)
	{
		ctx.l.delta[i] = 0.125f * ctx.l.output[i] - 0.03f * std::sin(static_cast<float>(i + 1));
	}

	backward_clifford_layer(ctx.l, state);

	expect_all_finite(ctx.input_delta.data(), ctx.input_delta.size(), "input_delta");
	expect_all_finite(ctx.l.cli_w_det_updates, static_cast<size_t>(ctx.l.c) * ctx.l.c, "cli_w_det_updates");
	expect_all_finite(ctx.l.cli_w_proj_updates, static_cast<size_t>(ctx.l.c) * ctx.l.cli_proj_in_dim, "cli_w_proj_updates");
	EXPECT_TRUE(has_signal(ctx.l.cli_vb_feat, static_cast<size_t>(ctx.l.batch) * ctx.l.outputs));
	EXPECT_TRUE(has_signal(ctx.input_delta.data(), ctx.input_delta.size()));
}
