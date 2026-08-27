#include <gtest/gtest.h>

#include "darknet_onnx.hpp"
#include "fp4_scaling.hpp"

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <string>
#include <vector>

namespace
{
	const onnx::TensorProto & find_initializer(const onnx::GraphProto & graph, const std::string & name)
	{
		for (const auto & initializer : graph.initializer())
		{
			if (initializer.name() == name) return initializer;
		}
		throw std::runtime_error("missing ONNX initializer: " + name);
	}
}

TEST(Fp4StateIsolation, OnnxInitializersUseFp32MasterWeightsAndBatchNormState)
{
	std::vector<float> weights = {0.125f, -0.25f, 0.5f, -1.0f};
	std::vector<float> biases = {1.25f, -2.5f};
	std::vector<float> scales = {0.75f, 1.5f};
	std::vector<float> rolling_mean = {-0.125f, 0.375f};
	std::vector<float> rolling_variance = {0.625f, 1.875f};
	auto derived = Darknet::fp4_quantize_weights_2d_reference(weights.data(), 2, 2);
	std::fill(derived.values.begin(), derived.values.end(), 0xffU);
	std::fill(derived.local_scales_e4m3.begin(), derived.local_scales_e4m3.end(), 0U);
	derived.global_scale = -999.0f;

	Darknet::Layer layer = {};
	layer.type = Darknet::ELayerType::CONVOLUTIONAL;
	layer.n = 2;
	layer.nweights = static_cast<int>(weights.size());
	layer.size = 1;
	layer.fp4_gemm_plan = reinterpret_cast<void *>(0x1);
	layer.fp4_workspace_size = derived.values.size();
	layer.fp4_eligible = 1;

	const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
	const auto output = std::filesystem::temp_directory_path() / ("darknet_fp4_onnx_" + std::to_string(stamp) + ".onnx");
	Darknet::ONNXExport exporter({}, {}, output);
	exporter.graph = exporter.model.mutable_graph();
	exporter.bit_size = 32;
	exporter.populate_graph_initializer(weights.data(), weights.size(), layer, "conv_weights");
	exporter.populate_graph_initializer(biases.data(), biases.size(), layer, "bn_bias");
	exporter.populate_graph_initializer(scales.data(), scales.size(), layer, "bn_scale");
	exporter.populate_graph_initializer(rolling_mean.data(), rolling_mean.size(), layer, "bn_mean");
	exporter.populate_graph_initializer(rolling_variance.data(), rolling_variance.size(), layer, "bn_variance");

	const auto values = [&] (const std::string & name)
	{
		const auto & data = find_initializer(*exporter.graph, name).float_data();
		return std::vector<float>(data.begin(), data.end());
	};
	EXPECT_EQ(values("conv_weights"), weights);
	EXPECT_EQ(values("bn_bias"), biases);
	EXPECT_EQ(values("bn_scale"), scales);
	EXPECT_EQ(values("bn_mean"), rolling_mean);
	EXPECT_EQ(values("bn_variance"), rolling_variance);
}
