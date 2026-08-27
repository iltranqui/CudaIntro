#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <filesystem>
#include <sstream>
#include <string>
#include <vector>

#include "box.hpp"
#include "yolonas_layer.hpp"
#include "ppyoloe_layer.hpp"
#include "yolox_layer.hpp"

namespace
{
	std::filesystem::path find_repo_root()
	{
		std::filesystem::path path = std::filesystem::current_path();
		for (int depth = 0; depth < 8; ++depth)
		{
			if (std::filesystem::exists(path / "src-lib" / "modern_yolo_layer.cpp") &&
				std::filesystem::exists(path / "src-test" / "CMakeLists.txt"))
			{
				return path;
			}
			path = path.parent_path();
		}
		return {};
	}

	std::string read_text_file(const std::filesystem::path & path)
	{
		std::ifstream input(path);
		std::ostringstream text;
		text << input.rdbuf();
		return text.str();
	}

	Darknet::Box decode_yolox_1x1_box(const std::array<float, 4> & raw)
	{
		Darknet::Box box = {};
		box.x = raw[0];
		box.y = raw[1];
		box.w = std::exp(std::clamp(raw[2], -10.0f, 10.0f));
		box.h = std::exp(std::clamp(raw[3], -10.0f, 10.0f));
		return box;
	}

	float expected_yolox_iou_delta(const std::array<float, 4> & raw, const Darknet::Box & truth, const float box_loss_weight, const int entry)
	{
		const Darknet::Box pred = decode_yolox_1x1_box(raw);
		const float iou = std::clamp(box_iou(pred, truth), 0.0f, 1.0f);
		const dxrep grad = dx_box_iou(pred, truth, IOU);
		const float factor = 2.0f * iou * box_loss_weight;
		switch (entry)
		{
			case 0: return factor * grad.dt;
			case 1: return factor * grad.db;
			case 2: return factor * grad.dl * pred.w;
			default: return factor * grad.dr * pred.h;
		}
	}
}

TEST(ModernYoloHeads, SplitContractHasHeadOwnedSources)
{
	const std::filesystem::path root = find_repo_root();
	ASSERT_FALSE(root.empty());

	for (const char * filename : {
		"yolox_layer.hpp",
		"yolox_layer.cpp",
		"ppyoloe_layer.hpp",
		"ppyoloe_layer.cpp",
		"yolonas_layer.hpp",
		"yolonas_layer.cpp"})
	{
		EXPECT_TRUE(std::filesystem::exists(root / "src-lib" / filename)) << filename;
	}

	const std::string common_cpp = read_text_file(root / "src-lib" / "modern_yolo_layer.cpp");
	EXPECT_EQ(common_cpp.find("Darknet::Layer make_yolox_layer("), std::string::npos);
	EXPECT_EQ(common_cpp.find("Darknet::Layer make_ppyoloe_layer("), std::string::npos);
	EXPECT_EQ(common_cpp.find("Darknet::Layer make_yolonas_layer("), std::string::npos);
	EXPECT_EQ(common_cpp.find("void forward_yolox_layer("), std::string::npos);
	EXPECT_EQ(common_cpp.find("void forward_ppyoloe_layer("), std::string::npos);
	EXPECT_EQ(common_cpp.find("void forward_yolonas_layer("), std::string::npos);
}

TEST(ModernYoloHeads, YoloxBoxDeltaMatchesIouSquaredDxBoxIouGradient)
{
	constexpr int batch = 1;
	constexpr int w = 1;
	constexpr int h = 1;
	constexpr int classes = 1;
	constexpr int max_boxes = 2;
	constexpr float box_loss_weight = 5.0f;

	Darknet::Layer l = Darknet::make_yolox_layer(batch, w, h, classes, max_boxes);
	l.assign_topk = 1;
	l.box_loss_weight = box_loss_weight;
	l.iou_loss = IOU;
	l.l1_final_iters = 0;

	const std::array<float, 4> raw_box = {
		0.45f,
		0.55f,
		std::log(0.35f),
		std::log(0.25f)
	};
	Darknet::Box truth = {};
	truth.x = 0.52f;
	truth.y = 0.48f;
	truth.w = 0.40f;
	truth.h = 0.30f;

	std::vector<float> input(l.outputs * l.batch, 0.0f);
	for (int entry = 0; entry < 4; ++entry)
	{
		input[entry] = raw_box[entry];
	}
	input[4] = 5.0f; // objectness logit
	input[5] = 5.0f; // class logit

	std::vector<float> truth_data(l.truths * l.batch, 0.0f);
	truth_data[0] = truth.x;
	truth_data[1] = truth.y;
	truth_data[2] = truth.w;
	truth_data[3] = truth.h;
	truth_data[4] = 0.0f;

	int cur_iteration = 0;
	Darknet::NetworkState state = {};
	state.train = 1;
	state.input = input.data();
	state.truth = truth_data.data();
	state.net.cur_iteration = &cur_iteration;
	state.net.max_batches = 100;

	Darknet::forward_yolox_layer(l, state);

	for (int entry = 0; entry < 4; ++entry)
	{
		const float expected = expected_yolox_iou_delta(raw_box, truth, box_loss_weight, entry);
		const float tolerance = std::max(0.05f, std::fabs(expected) * 0.15f);
		EXPECT_NEAR(l.delta[entry], expected, tolerance)
			<< "entry=" << entry << " actual=" << l.delta[entry] << " expected=" << expected;
	}

	free_layer(l);
}

namespace
{
	// Replicates decode_dfl_box for a 1x1 grid layer: 4 sides, `bins` logits per
	// side stored channel-major, anchor point at (0.5, 0.5).
	Darknet::Box decode_dfl_1x1_box(const std::vector<float> & logits, const int bins)
	{
		std::array<float, 4> expected = {};
		for (int side = 0; side < 4; ++side)
		{
			float max_logit = logits[side * bins];
			for (int k = 1; k < bins; ++k)
			{
				max_logit = std::max(max_logit, logits[side * bins + k]);
			}
			float denom = 0.0f;
			float e = 0.0f;
			for (int k = 0; k < bins; ++k)
			{
				const float p = std::exp(logits[side * bins + k] - max_logit);
				denom += p;
				e += static_cast<float>(k) * p;
			}
			expected[side] = e / denom;
		}

		const float left   = 0.5f - expected[0];
		const float top    = 0.5f - expected[1];
		const float right  = 0.5f + expected[2];
		const float bottom = 0.5f + expected[3];

		Darknet::Box box = {};
		box.x = 0.5f * (std::min(left, right) + std::max(left, right));
		box.y = 0.5f * (std::min(top, bottom) + std::max(top, bottom));
		box.w = std::max(std::fabs(right - left), 1e-6f);
		box.h = std::max(std::fabs(bottom - top), 1e-6f);
		return box;
	}

	float dfl_giou_loss_1x1(const std::vector<float> & logits, const int bins, const Darknet::Box & truth, const float weight)
	{
		const Darknet::Box pred = decode_dfl_1x1_box(logits, bins);
		return weight * (1.0f - box_iou_kind(pred, truth, GIOU));
	}
}

TEST(ModernYoloHeads, PpyoloeDflGiouDeltaMatchesFiniteDifference)
{
	constexpr int batch = 1;
	constexpr int w = 1;
	constexpr int h = 1;
	constexpr int classes = 1;
	constexpr int max_boxes = 2;
	constexpr int reg_max = 3; // bins = 4 per side, coords = 16
	constexpr int bins = reg_max + 1;
	constexpr float box_loss_weight = 2.5f;

	Darknet::Layer l = Darknet::make_ppyoloe_layer(batch, w, h, classes, max_boxes, reg_max);
	l.assign_topk = 1;
	l.box_loss_weight = box_loss_weight;
	l.dfl_loss_weight = 0.0f; // isolate the GIoU-through-DFL term on the box bins

	std::vector<float> input(l.outputs * l.batch, 0.0f);
	// mildly asymmetric logits so the softmax expectations differ per side
	for (int side = 0; side < 4; ++side)
	{
		for (int k = 0; k < bins; ++k)
		{
			input[side * bins + k] = 0.15f * static_cast<float>((side + 1) * k);
		}
	}
	input[l.coords + 0] = 5.0f; // quality/objectness logit
	input[l.coords + 1] = 5.0f; // class logit

	Darknet::Box truth = {};
	truth.x = 0.5f;
	truth.y = 0.5f;
	truth.w = 0.6f;
	truth.h = 0.5f;

	std::vector<float> truth_data(l.truths * l.batch, 0.0f);
	truth_data[0] = truth.x;
	truth_data[1] = truth.y;
	truth_data[2] = truth.w;
	truth_data[3] = truth.h;
	truth_data[4] = 0.0f;

	int cur_iteration = 0;
	Darknet::NetworkState state = {};
	state.train = 1;
	state.input = input.data();
	state.truth = truth_data.data();
	state.net.cur_iteration = &cur_iteration;
	state.net.max_batches = 100;

	Darknet::forward_ppyoloe_layer(l, state);

	// single candidate: t_hat = (score/max_score)*max_iou = iou; replicate the applied weight
	std::vector<float> logits(input.begin(), input.begin() + 4 * bins);
	const Darknet::Box pred = decode_dfl_1x1_box(logits, bins);
	const float quality = std::clamp(box_iou(pred, truth), 0.0f, 1.0f);
	const float weight = box_loss_weight * quality;

	constexpr float eps = 1e-3f;
	for (int idx = 0; idx < 4 * bins; ++idx)
	{
		std::vector<float> probe = logits;
		probe[idx] += eps;
		const float plus = dfl_giou_loss_1x1(probe, bins, truth, weight);
		probe[idx] -= 2.0f * eps;
		const float minus = dfl_giou_loss_1x1(probe, bins, truth, weight);
		const float numeric = (plus - minus) / (2.0f * eps);
		const float analytical = -l.delta[idx]; // delta accumulates the negative loss gradient
		const float tolerance = std::max(0.01f, std::fabs(numeric) * 0.10f);
		EXPECT_NEAR(analytical, numeric, tolerance)
			<< "bin index=" << idx << " delta=" << l.delta[idx] << " numeric=" << numeric;
	}

	free_layer(l);
}
