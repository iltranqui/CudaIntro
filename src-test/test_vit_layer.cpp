#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <string>
#include <vector>

#include "darknet_internal.hpp"
#include "vit_layer.hpp"

void save_vit_weights(Darknet::Layer & l, FILE *fp);
size_t load_vit_weights(Darknet::Layer & l, FILE *fp, bool has_mhc_scales);

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

	void copy_vit_trainable_params(const Darknet::Layer & src, Darknet::Layer & dst)
	{
		const int patch_dim = src.vit_patch_size * src.vit_patch_size * src.c;
		const int token_dim = src.out_h * src.out_w * src.out_c;

		std::copy(src.vit_patch_embed, src.vit_patch_embed + src.out_c * patch_dim, dst.vit_patch_embed);
		std::copy(src.vit_patch_bias, src.vit_patch_bias + src.out_c, dst.vit_patch_bias);
		std::copy(src.weights, src.weights + src.nweights, dst.weights);
		std::copy(src.biases, src.biases + src.nbiases, dst.biases);
		std::copy(src.vit_wo, src.vit_wo + src.out_c * src.out_c, dst.vit_wo);
		std::copy(src.vit_wo_bias, src.vit_wo_bias + src.out_c, dst.vit_wo_bias);
		std::copy(src.vit_ln1_gamma, src.vit_ln1_gamma + src.out_c, dst.vit_ln1_gamma);
		std::copy(src.vit_ln1_beta, src.vit_ln1_beta + src.out_c, dst.vit_ln1_beta);
		std::copy(src.vit_ln2_gamma, src.vit_ln2_gamma + src.out_c, dst.vit_ln2_gamma);
		std::copy(src.vit_ln2_beta, src.vit_ln2_beta + src.out_c, dst.vit_ln2_beta);
		std::copy(src.vit_ffn_w1, src.vit_ffn_w1 + src.out_c * src.vit_mlp_dim, dst.vit_ffn_w1);
		std::copy(src.vit_ffn_b1, src.vit_ffn_b1 + src.vit_mlp_dim, dst.vit_ffn_b1);
		std::copy(src.vit_ffn_w2, src.vit_ffn_w2 + src.out_c * src.vit_mlp_dim, dst.vit_ffn_w2);
		std::copy(src.vit_ffn_b2, src.vit_ffn_b2 + src.out_c, dst.vit_ffn_b2);
		std::copy(src.vit_pos_embed, src.vit_pos_embed + token_dim, dst.vit_pos_embed);
		std::copy(src.scales, src.scales + 12, dst.scales);
	}
#endif

	void run_vit_case(int h, int w, int c, int filters, int patch_size, int heads, int mlp_dim,
		int pos_embed_type = 0, int pos_init_type = 0)
	{
		const int batch = 2;
		Darknet::Layer l = make_vit_layer(batch, h, w, c, filters,
			patch_size, patch_size, 0, heads, 2, mlp_dim, pos_embed_type, pos_init_type, GELU, 0, 1);

		const int out_h = h / patch_size;
		const int out_w = w / patch_size;
		const int patch_dim = patch_size * patch_size * c;
		ASSERT_EQ(l.out_h, out_h);
		ASSERT_EQ(l.out_w, out_w);
		ASSERT_EQ(l.out_c, filters);
		ASSERT_EQ(l.outputs, out_h * out_w * filters);
		ASSERT_EQ(l.vit_mlp_dim, mlp_dim);

		std::vector<float> input(static_cast<size_t>(batch) * c * h * w);
		std::vector<float> input_delta(input.size(), 0.0f);
		fill_pattern(input.data(), input.size(), 0.1f, 0.01f, 0.017f);

		Darknet::NetworkState state = {};
		state.input = input.data();
		state.train = 1;
		forward_vit_layer(l, state);

		const size_t output_count = static_cast<size_t>(batch) * l.outputs;
		expect_all_finite(l.output, output_count, "vit output");
		EXPECT_GT(max_abs(l.output, output_count), 0.0f);

		fill_pattern(l.delta, output_count, 0.2f, 0.0f, 0.011f);

		Darknet::NetworkState back_state = {};
		back_state.input = input.data();
		back_state.delta = input_delta.data();
		back_state.train = 1;
		backward_vit_layer(l, back_state);

		expect_all_finite(l.vit_patch_embed_updates, static_cast<size_t>(filters) * patch_dim, "patch updates");
		expect_all_finite(l.weight_updates, l.nweights, "qkv updates");
		expect_all_finite(l.vit_pos_embed_updates, static_cast<size_t>(out_h) * out_w * filters, "pos updates");
		expect_all_finite(input_delta.data(), input_delta.size(), "input delta");

		EXPECT_GT(max_abs(l.vit_patch_embed_updates, static_cast<size_t>(filters) * patch_dim), 0.0f);
		EXPECT_GT(max_abs(l.weight_updates, l.nweights), 0.0f);
		if (pos_embed_type == 0)
		{
			EXPECT_GT(max_abs(l.vit_pos_embed_updates, static_cast<size_t>(out_h) * out_w * filters), 0.0f);
		}
		else
		{
			EXPECT_EQ(max_abs(l.vit_pos_embed_updates, static_cast<size_t>(out_h) * out_w * filters), 0.0f);
		}
		EXPECT_GT(max_abs(input_delta.data(), input_delta.size()), 0.0f);

		free_layer(l);
	}

	void expect_dropin_patch_embed(const Darknet::Layer & l)
	{
		const int P = l.vit_patch_size;
		const int C = l.c;
		const int K = P * P * C;
		ASSERT_EQ(l.n, C);
		const float expected = 1.0f / static_cast<float>(P * P);

		for (int oc = 0; oc < C; ++oc)
		{
			for (int k = 0; k < K; ++k)
			{
				const int ic = k % C;
				const float want = (ic == oc) ? expected : 0.0f;
				EXPECT_FLOAT_EQ(l.vit_patch_embed[oc * K + k], want)
					<< "oc=" << oc << " k=" << k;
			}
		}
	}
}

TEST(ViTLayer, PatchSizeOneKeepsSpatialShape)
{
	run_vit_case(3, 5, 3, 6, 1, 3, 12);
}

TEST(ViTLayer, PatchSizeTwoUsesPatchGridAndBackpropagates)
{
	run_vit_case(4, 6, 3, 8, 2, 4, 10);
}

TEST(ViTLayer, SinusoidalPosEmbedIsFixed)
{
	run_vit_case(4, 4, 3, 8, 2, 4, 16, 1);
}

TEST(ViTLayer, PatchSizeOneDropinInitIsIdentity)
{
	Darknet::Layer l = make_vit_layer(1, 3, 5, 4, 4,
		1, 1, 0, 2, 2, 8, 0, 1, GELU, 0, 1);

	expect_dropin_patch_embed(l);
	EXPECT_EQ(max_abs(l.vit_pos_embed, static_cast<size_t>(l.out_h) * l.out_w * l.n), 0.0f);

	free_layer(l);
}

TEST(ViTLayer, PatchSizeTwoDropinInitIsChannelAverage)
{
	Darknet::Layer l = make_vit_layer(1, 4, 6, 3, 3,
		2, 2, 0, 3, 2, 9, 0, 1, GELU, 0, 1);

	expect_dropin_patch_embed(l);
	EXPECT_EQ(max_abs(l.vit_pos_embed, static_cast<size_t>(l.out_h) * l.out_w * l.n), 0.0f);

	free_layer(l);
}

TEST(ViTLayer, ZeroLearnedPosEmbedStillBackpropagates)
{
	run_vit_case(4, 4, 3, 6, 2, 3, 12, 0, 1);
}

TEST(ViTLayer, SaveLoadPreservesResidualMixerScales)
{
	Darknet::Layer src = make_vit_layer(1, 4, 4, 4, 4,
		2, 2, 0, 2, 2, 8, 0, 1, GELU, 0, 1);
	Darknet::Layer dst = make_vit_layer(1, 4, 4, 4, 4,
		2, 2, 0, 2, 2, 8, 0, 1, GELU, 0, 1);

	for (int i = 0; i < 12; ++i)
	{
		src.scales[i] = -1.25f + 0.17f * static_cast<float>(i);
	}

#ifdef DARKNET_GPU
	push_vit_layer(src);
#endif

	FILE *fp = std::tmpfile();
	ASSERT_NE(fp, nullptr);

	save_vit_weights(src, fp);
	std::rewind(fp);
	load_vit_weights(dst, fp, true);

	for (int i = 0; i < 12; ++i)
	{
		EXPECT_FLOAT_EQ(dst.scales[i], src.scales[i]) << "idx=" << i;
	}

	std::fclose(fp);
	free_layer(src);
	free_layer(dst);
}

#ifdef DARKNET_GPU
TEST(ViTLayerGPU, PatchTwoForwardBackwardMatchesCPU)
{
	if (!gpu_tests_enabled())
	{
		GTEST_SKIP() << "GPU tests disabled (set DARKNET_TEST_GPU=1 or leave it unset to enable)";
	}

	const int batch = 2;
	const int h = 4;
	const int w = 4;
	const int c = 4;
	const int filters = 4;
	const int patch_size = 2;
	const int heads = 2;
	const int mlp_dim = 8;

	Darknet::Layer l_cpu = make_vit_layer(batch, h, w, c, filters,
		patch_size, patch_size, 0, heads, 2, mlp_dim, 0, 1, GELU, 0, 1);
	Darknet::Layer l_gpu = make_vit_layer(batch, h, w, c, filters,
		patch_size, patch_size, 0, heads, 2, mlp_dim, 0, 1, GELU, 0, 1);

	copy_vit_trainable_params(l_cpu, l_gpu);
	push_vit_layer(l_gpu);

	std::vector<float> input(static_cast<size_t>(batch) * c * h * w);
	std::vector<float> input_delta_cpu(input.size(), 0.0f);
	std::vector<float> delta(static_cast<size_t>(batch) * l_cpu.outputs);
	fill_pattern(input.data(), input.size(), 0.13f, -0.02f, 0.019f);
	fill_pattern(delta.data(), delta.size(), 0.07f, 0.0f, 0.031f);

	Darknet::NetworkState fwd_cpu = {};
	fwd_cpu.input = input.data();
	fwd_cpu.train = 1;
	forward_vit_layer(l_cpu, fwd_cpu);

	float *input_gpu = cuda_make_array(input.data(), input.size());
	float *input_delta_gpu = cuda_make_array(nullptr, input.size());
	fill_ongpu(input.size(), 0.0f, input_delta_gpu, 1);

	Darknet::NetworkState fwd_gpu = {};
	fwd_gpu.input = input_gpu;
	fwd_gpu.train = 1;
	forward_vit_layer_gpu(l_gpu, fwd_gpu);

	std::vector<float> output_gpu(static_cast<size_t>(batch) * l_gpu.outputs, 0.0f);
	cuda_pull_array(l_gpu.output_gpu, output_gpu.data(), output_gpu.size());
	EXPECT_LT(max_abs_diff(l_cpu.output, output_gpu.data(), output_gpu.size()), 5e-4f);

	std::copy(delta.begin(), delta.end(), l_cpu.delta);
	cuda_push_array(l_gpu.delta_gpu, delta.data(), delta.size());

	Darknet::NetworkState back_cpu = {};
	back_cpu.input = input.data();
	back_cpu.delta = input_delta_cpu.data();
	back_cpu.train = 1;
	backward_vit_layer(l_cpu, back_cpu);

	Darknet::NetworkState back_gpu = {};
	back_gpu.input = input_gpu;
	back_gpu.delta = input_delta_gpu;
	back_gpu.train = 1;
	backward_vit_layer_gpu(l_gpu, back_gpu);

	std::vector<float> input_delta_gpu_host(input.size(), 0.0f);
	std::vector<float> patch_updates_gpu(static_cast<size_t>(filters) * patch_size * patch_size * c, 0.0f);
	std::vector<float> qkv_updates_gpu(l_gpu.nweights, 0.0f);
	std::vector<float> wo_updates_gpu(static_cast<size_t>(filters) * filters, 0.0f);
	std::vector<float> ffn_w2_updates_gpu(static_cast<size_t>(filters) * mlp_dim, 0.0f);
	std::vector<float> scale_updates_gpu(12, 0.0f);

	cuda_pull_array(input_delta_gpu, input_delta_gpu_host.data(), input_delta_gpu_host.size());
	cuda_pull_array(l_gpu.vit_patch_embed_updates_gpu, patch_updates_gpu.data(), patch_updates_gpu.size());
	cuda_pull_array(l_gpu.weight_updates_gpu, qkv_updates_gpu.data(), qkv_updates_gpu.size());
	cuda_pull_array(l_gpu.vit_wo_updates_gpu, wo_updates_gpu.data(), wo_updates_gpu.size());
	cuda_pull_array(l_gpu.vit_ffn_w2_updates_gpu, ffn_w2_updates_gpu.data(), ffn_w2_updates_gpu.size());
	cuda_pull_array(l_gpu.scale_updates_gpu, scale_updates_gpu.data(), scale_updates_gpu.size());

	EXPECT_LT(max_abs_diff(input_delta_cpu.data(), input_delta_gpu_host.data(), input_delta_cpu.size()), 1e-3f);
	EXPECT_LT(max_abs_diff(l_cpu.vit_patch_embed_updates, patch_updates_gpu.data(), patch_updates_gpu.size()), 1e-3f);
	EXPECT_LT(max_abs_diff(l_cpu.weight_updates, qkv_updates_gpu.data(), qkv_updates_gpu.size()), 1e-3f);
	EXPECT_LT(max_abs_diff(l_cpu.vit_wo_updates, wo_updates_gpu.data(), wo_updates_gpu.size()), 1e-3f);
	EXPECT_LT(max_abs_diff(l_cpu.vit_ffn_w2_updates, ffn_w2_updates_gpu.data(), ffn_w2_updates_gpu.size()), 1e-3f);
	EXPECT_LT(max_abs_diff(l_cpu.scale_updates, scale_updates_gpu.data(), scale_updates_gpu.size()), 1e-3f);

	cuda_free(input_gpu);
	cuda_free(input_delta_gpu);
	free_layer(l_cpu);
	free_layer(l_gpu);
}
#endif
