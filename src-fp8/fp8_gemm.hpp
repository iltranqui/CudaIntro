#pragma once

#include <cstddef>

#include "fp8_scaling.hpp"

namespace Darknet
{
	struct Fp8GemmPlan;

	enum class Fp8TrainingGemm
	{
		Forward,
		WeightGradient,
		WeightGradientDirectUpdate,
		DataGradient,
		DataGradientDirectUpdate
	};

	enum class Fp8GemmOutput
	{
		Fp32,
		Bf16
	};

	/* Buffer contract (cuBLASLt FP8 only supports the TN matmul case):
	 *   A buffer: row-major (output_rows x reduction_pad)
	 *   B buffer: row-major (output_cols x reduction_pad)   <- note: transposed operand
	 *   D buffer: column-major (output_rows x output_cols)  <- convert to row-major after
	 * reduction_pad must be a multiple of 16 -- every caller enforces this via
	 * fp8_round_up_to_16(). output_rows and output_cols are NOT checked or rounded
	 * anywhere in this codebase (confirmed: fp8_gemm_plan_create_ex's gate only
	 * checks positivity/batch/scale-pointer validity, never output_rows%16) --
	 * this comment previously claimed output_rows must also be a multiple of 16,
	 * but no vendored cuBLASLt header or sample documents that requirement for an
	 * FP8 (E4M3/E5M2) TN matmul, unlike FP4's genuine K%16 block-scale-axis
	 * constraint. Odd output_rows (e.g. a YOLO detection head's filters=255) is
	 * passed through as-is; whether cublasLtMatmulAlgoGetHeuristic() accepts or
	 * rejects it is decided by cuBLASLt at plan-creation time, not by this code.
	 * b_leading_dim (optional) is the row stride of the B buffer in elements.
	 * batch > 1 creates a strided-batched matmul: by default the A buffer is shared across
	 * the batch (stride 0) -- the conv/connected-layer shape, one weight matrix applied to
	 * many inputs -- while B and D consist of `batch` contiguous per-matrix blocks. Set
	 * batch_a to give A its own per-batch-item stride too (output_rows * reduction_pad
	 * elements apart). Set batch_b=false to broadcast B across the batch, as required by
	 * direct convolution dgrad where dY varies per image and the weight matrix is shared. */
	struct Fp8GemmSpec
	{
		int output_rows = 0;
		int output_cols = 0;
		int reduction = 0;
		int reduction_pad = 0;
		int b_leading_dim = 0;
		int batch = 1;
		bool batch_a = false;
		bool batch_b = true;
		Fp8Format a_format = Fp8Format::E4M3;
		Fp8Format b_format = Fp8Format::E4M3;
		Fp8GemmOutput output = Fp8GemmOutput::Fp32;
	};

	struct Fp8SharedDyLayout
	{
		int filters_pad = 0;
		int spatial_pad = 0;
		size_t bytes = 0;
	};

	int fp8_round_up_to_16(int value);
	Fp8SharedDyLayout fp8_shared_dy_layout(int filters, int spatial);
	size_t fp8_gemm_workspace_bytes();
	bool fp8_gemm_supported();
	bool fp8_sm89_optimization_supported();
	Fp8GemmSpec fp8_gemm_training_spec(Fp8TrainingGemm kind, int filters, int kernel, int spatial);

	Fp8GemmPlan * fp8_gemm_plan_create(
		int m,
		int n,
		int k,
		int k_pad,
		const float * weight_scale_gpu,
		const float * input_scale_gpu);
	Fp8GemmPlan * fp8_gemm_plan_create_ex(
		const Fp8GemmSpec & spec,
		const float * a_scale_gpu,
		const float * b_scale_gpu);
		void fp8_gemm_plan_destroy(Fp8GemmPlan * plan);
		bool fp8_gemm_output_is_column_major(const Fp8GemmPlan * plan);
		bool fp8_gemm_output_is_fp32(const Fp8GemmPlan * plan);
		size_t fp8_gemm_output_element_bytes(const Fp8GemmPlan * plan);

		bool fp8_gemm(
			Fp8GemmPlan * plan,
			const void * weights_fp8_gpu,
			const void * input_fp8_gpu,
			void * output_gpu,
			void * workspace,
			size_t workspace_bytes,
			float beta = 0.0f);
	}
