// Tests for [yolo_2stage] — Filter 1 matcher, Filter 2 IoU gate, gradient sign.
//
// These tests build a 4x4 single-anchor [yolo] + [yolo_2stage] pair entirely
// in-memory, inject a known GT, and run the real forward path on CPU.  No
// dataloader, no network init, no image data — just verifying that:
//
//   Test 1: Filter 1 picks the correct cell for a known GT.
//   Test 2: Filter 2 admits when Stage-1's box IoU > thresh, skips when below.
//   Test 3: Loss = 0 and delta ≈ 0 when predicted heading matches truth.
//   Test 4: Loss ≈ 2 and delta has correct sign when prediction is 180° off.

#include <cmath>
#include <cstring>
#include <vector>
#include <gtest/gtest.h>

#include "darknet_internal.hpp"
#include "yolo_layer.hpp"
#include "yolo_2stage_layer.hpp"

namespace
{
	constexpr int    kW          = 4;
	constexpr int    kH          = 4;
	constexpr int    kBatch      = 1;
	constexpr int    kClasses    = 10;
	constexpr int    kMaxBoxes   = 5;
	constexpr int    kTruthSize  = 7;
	constexpr int    kAnchorWPx  = 49;
	constexpr int    kAnchorHPx  = 77;
	constexpr int    kNetW       = 128;
	constexpr int    kNetH       = 128;

	float inv_tanh(float y)
	{
		// Clamp to keep finite outside [-1, 1].
		const float yc = std::max(-0.999f, std::min(0.999f, y));
		return 0.5f * std::log((1.0f + yc) / (1.0f - yc));
	}

	// Build a minimal classical [yolo] head with one anchor.  yolo.output is
	// laid out so decode_yolo_box at (i*, j*) returns the box specified by
	// (pred_cx, pred_cy, pred_w, pred_h) in normalized coords.
	struct FakeYolo
	{
		Darknet::Layer l{};
		std::vector<float>  output;
		std::vector<float>  biases;
		std::vector<int>    mask;

		void build(int i_star, int j_star,
				   float pred_cx, float pred_cy, float pred_w, float pred_h)
		{
			l.type        = Darknet::ELayerType::YOLO;
			l.batch       = kBatch;
			l.w           = kW;
			l.h           = kH;
			l.n           = 1;
			l.total       = 1;
			l.classes     = kClasses;
			l.coords      = 4;
			l.max_boxes   = kMaxBoxes;
			l.truth_size  = kTruthSize;
			l.truths      = kMaxBoxes * kTruthSize;
			l.new_coords  = 0;

			biases = { float(kAnchorWPx), float(kAnchorHPx) };
			l.biases = biases.data();

			mask = { 0 };
			l.mask = mask.data();

			const int channels = (4 + 1 + kClasses);  // 15
			l.outputs = channels * kW * kH;
			output.assign(l.outputs * kBatch, 0.0f);
			l.output = output.data();

			// get_yolo_box (new_coords=0) does NOT re-apply sigmoid; box channels in
			// l.output are stored already-activated.  So we write the fractional
			// in-cell offset directly:
			//   l.output[box_x] = pred_cx * lw - i_star   ∈ [0, 1]
			//   l.output[box_y] = pred_cy * lh - j_star
			//   l.output[box_w] = log(pred_w * netw / biases_w)   (no exp pre-applied)
			//   l.output[box_h] = log(pred_h * neth / biases_h)
			const float frac_x = pred_cx * kW - i_star;
			const float frac_y = pred_cy * kH - j_star;
			const float raw_w  = std::log(pred_w * kNetW / kAnchorWPx);
			const float raw_h  = std::log(pred_h * kNetH / kAnchorHPx);

			const int stride = kW * kH;
			const int cell   = j_star * kW + i_star;
			output[0 * stride + cell] = frac_x;
			output[1 * stride + cell] = frac_y;
			output[2 * stride + cell] = raw_w;
			output[3 * stride + cell] = raw_h;
			// objectness + class logits left at 0 (sigmoid(0) = 0.5 — irrelevant
			// for Filter 2 which only reads box channels).
		}
	};

	// Build a [yolo_2stage] with input pre-set so that after forward's tanh
	// activation, l.output[c_idx], l.output[s_idx] hold (c_raw, s_raw).
	struct FakeStage2
	{
		Darknet::Layer    l{};
		std::vector<float> input;  // state.input lives here
		FakeStage2() {}

		void build(const FakeYolo & yolo_pair, int i_star, int j_star, float c_raw_at_pos, float s_raw_at_pos)
		{
			l = Darknet::make_yolo_2stage_layer(kBatch, kW, kH, kClasses, /*stage1_idx*/ 0);
			l.share_layer         = const_cast<Darknet::Layer*>(&yolo_pair.l);
			l.stage2_match_thresh = 0.5f;
			l.iou_normalizer      = 1.0f;
			l.max_boxes           = kMaxBoxes;
			l.truth_size          = kTruthSize;
			l.truths              = kMaxBoxes * kTruthSize;

			input.assign(l.batch * l.inputs, 0.0f);
			const int stride = kW * kH;
			const int cell   = j_star * kW + i_star;
			input[0 * stride + cell] = inv_tanh(c_raw_at_pos);
			input[1 * stride + cell] = inv_tanh(s_raw_at_pos);
		}
	};

	// Run forward with a single GT and return l for inspection.
	void run_forward(FakeStage2 & s2, const FakeYolo & yolo, float gt_cx, float gt_cy,
					 float gt_w, float gt_h, float fx, float fy)
	{
		std::vector<float> truth(kBatch * kMaxBoxes * kTruthSize, 0.0f);
		truth[0] = gt_cx;
		truth[1] = gt_cy;
		truth[2] = gt_w;
		truth[3] = gt_h;
		truth[4] = 0.0f;  // class_id
		truth[5] = fx;
		truth[6] = fy;

		Darknet::NetworkState state{};
		state.train = 1;
		state.input = s2.input.data();
		state.truth = truth.data();
		state.delta = nullptr;
		state.net.w = kNetW;
		state.net.h = kNetH;
		state.net.cur_iteration = nullptr;

		Darknet::forward_yolo_2stage_layer(s2.l, state);
		(void)yolo;
	}
}


// =====================================================================
// Test 1 — Filter 1: positive cell == floor(gt.x * lw, gt.y * lh)
// =====================================================================
TEST(YoloTwoStage, Filter1PicksCorrectCell)
{
	// GT at (0.55, 0.30) on a 4x4 grid → cell (2, 1).
	const float gt_cx = 0.55f, gt_cy = 0.30f;
	const float gt_w  = 0.30f, gt_h  = 0.30f;
	const float fx    = gt_cx + 0.10f;  // heading +x
	const float fy    = gt_cy;
	const int   exp_i = int(gt_cx * kW);  // 2
	const int   exp_j = int(gt_cy * kH);  // 1

	FakeYolo yolo;
	yolo.build(exp_i, exp_j, gt_cx, gt_cy, gt_w, gt_h);  // pred == GT → high IoU → admit

	FakeStage2 s2;
	// Misalign Stage-2 prediction with GT heading so loss is non-trivial.
	// GT heading is +x = (1, 0); load pred = (0, 1) so dot = 0, loss = 1.
	s2.build(yolo, exp_i, exp_j, /*c_raw=*/0.0f, /*s_raw=*/0.5f);

	run_forward(s2, yolo, gt_cx, gt_cy, gt_w, gt_h, fx, fy);

	// Delta must be non-zero at the expected cell, zero elsewhere.
	const int stride = kW * kH;
	const int cell   = exp_j * kW + exp_i;

	bool found_positive = (std::fabs(s2.l.delta[0 * stride + cell]) > 0.0f) ||
						  (std::fabs(s2.l.delta[1 * stride + cell]) > 0.0f);
	EXPECT_TRUE(found_positive) << "expected non-zero delta at cell (" << exp_i << "," << exp_j << ")";

	int other_nonzero = 0;
	for (int loc = 0; loc < stride; ++loc)
	{
		if (loc == cell) continue;
		if (std::fabs(s2.l.delta[0 * stride + loc]) > 1e-12f) ++other_nonzero;
		if (std::fabs(s2.l.delta[1 * stride + loc]) > 1e-12f) ++other_nonzero;
	}
	EXPECT_EQ(other_nonzero, 0) << "Stage-2 should write delta ONLY at the matched cell";
}


// =====================================================================
// Test 2 — Filter 2: low IoU skips, high IoU admits.
// =====================================================================
TEST(YoloTwoStage, Filter2GatesByMatchIoU)
{
	const float gt_cx = 0.50f, gt_cy = 0.50f;
	const float gt_w  = 0.30f, gt_h  = 0.30f;
	const int   exp_i = int(gt_cx * kW);  // 2
	const int   exp_j = int(gt_cy * kH);  // 2

	// --- Subcase A: pred far from GT  →  IoU ≈ 0  →  Filter 2 closes.
	{
		FakeYolo yolo;
		yolo.build(exp_i, exp_j, /*pred_cx*/ 0.10f, /*pred_cy*/ 0.10f, /*pred_w*/ 0.05f, /*pred_h*/ 0.05f);

		FakeStage2 s2;
		s2.build(yolo, exp_i, exp_j, 0.5f, 0.0f);

		run_forward(s2, yolo, gt_cx, gt_cy, gt_w, gt_h, gt_cx + 0.1f, gt_cy);

		const int stride = kW * kH;
		const int cell   = exp_j * kW + exp_i;
		EXPECT_NEAR(s2.l.delta[0 * stride + cell], 0.0f, 1e-12f);
		EXPECT_NEAR(s2.l.delta[1 * stride + cell], 0.0f, 1e-12f);
		EXPECT_NEAR(*s2.l.cost, 0.0f, 1e-12f) << "no admitted positives → cost must be 0";
	}

	// --- Subcase B: pred == GT  →  IoU = 1.0  →  Filter 2 admits.
	{
		FakeYolo yolo;
		yolo.build(exp_i, exp_j, gt_cx, gt_cy, gt_w, gt_h);

		FakeStage2 s2;
		// Misalign heading prediction so admittance produces non-zero loss.
		s2.build(yolo, exp_i, exp_j, /*c_raw=*/0.0f, /*s_raw=*/0.5f);

		run_forward(s2, yolo, gt_cx, gt_cy, gt_w, gt_h, gt_cx + 0.1f, gt_cy);

		const int stride = kW * kH;
		const int cell   = exp_j * kW + exp_i;
		const bool delta_nonzero =
			std::fabs(s2.l.delta[0 * stride + cell]) > 1e-9f ||
			std::fabs(s2.l.delta[1 * stride + cell]) > 1e-9f;
		EXPECT_TRUE(delta_nonzero) << "admitted positive → expected non-zero delta";
		EXPECT_GT(*s2.l.cost, 0.0f)  << "admitted positive → cost must accumulate";
	}
}


// =====================================================================
// Test 3 — Loss = 0 when prediction matches truth heading.
// =====================================================================
TEST(YoloTwoStage, LossZeroOnPerfectAngle)
{
	const float gt_cx = 0.50f, gt_cy = 0.50f;
	const float gt_w  = 0.30f, gt_h  = 0.30f;
	const float fx    = gt_cx + 0.10f;  // heading exactly +x → target (c_t, s_t) = (1, 0)
	const float fy    = gt_cy;
	const int   exp_i = int(gt_cx * kW);
	const int   exp_j = int(gt_cy * kH);

	FakeYolo yolo;
	yolo.build(exp_i, exp_j, gt_cx, gt_cy, gt_w, gt_h);

	FakeStage2 s2;
	// Pre-load (c_raw, s_raw) = (0.99, 0) → unit-norm ≈ (1, 0). Matches GT.
	s2.build(yolo, exp_i, exp_j, 0.99f, 0.0f);

	run_forward(s2, yolo, gt_cx, gt_cy, gt_w, gt_h, fx, fy);

	EXPECT_LT(*s2.l.cost, 0.01f) << "perfect alignment → 1 - cos(0) ≈ 0";

	const int stride = kW * kH;
	const int cell   = exp_j * kW + exp_i;
	EXPECT_NEAR(s2.l.delta[0 * stride + cell], 0.0f, 0.05f);
	EXPECT_NEAR(s2.l.delta[1 * stride + cell], 0.0f, 0.05f);
}


// =====================================================================
// Test 4 — Loss ≈ 2 and delta has correct sign when prediction is opposite.
// =====================================================================
TEST(YoloTwoStage, LossLargeOnOppositeAngle)
{
	const float gt_cx = 0.50f, gt_cy = 0.50f;
	const float gt_w  = 0.30f, gt_h  = 0.30f;
	const float fx    = gt_cx + 0.10f;  // target (c_t, s_t) = (1, 0)
	const float fy    = gt_cy;
	const int   exp_i = int(gt_cx * kW);
	const int   exp_j = int(gt_cy * kH);

	FakeYolo yolo;
	yolo.build(exp_i, exp_j, gt_cx, gt_cy, gt_w, gt_h);

	FakeStage2 s2;
	// (c_raw, s_raw) = (-0.99, 0) → unit-norm ≈ (-1, 0). Opposite of GT.
	s2.build(yolo, exp_i, exp_j, -0.99f, 0.0f);

	run_forward(s2, yolo, gt_cx, gt_cy, gt_w, gt_h, fx, fy);

	EXPECT_GT(*s2.l.cost, 1.95f) << "opposite alignment → 1 - cos(180°) ≈ 2";

	// Delta sign: c_raw is -0.99, we want to push c_raw UP toward +1, so
	// delta_c (which gets added in gradient descent: x += lr * delta) must be > 0.
	const int stride = kW * kH;
	const int cell   = exp_j * kW + exp_i;
	EXPECT_GT(s2.l.delta[0 * stride + cell], 0.0f)
		<< "expected positive delta_c to push c_raw from -0.99 toward +1";
}
