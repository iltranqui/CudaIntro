#pragma once

#include <cstddef>
#include <cstdint>

namespace Darknet
{
	enum class Fp4GemmBackend
	{
		None,
		CudnnFrontend,
		CublasLt
	};

	constexpr Fp4GemmBackend select_fp4_gemm_backend(const bool frontend_ready, const bool cublaslt_ready)
	{
		// The cuBLASLt path consumes CUDA_R_4F_E2M1 operands directly, which is
		// the path whose execution proves that the NVFP4 Tensor Core GEMM was
		// selected.  The cuDNN Frontend graph remains a supported fallback.
		return cublaslt_ready ? Fp4GemmBackend::CublasLt :
			(frontend_ready ? Fp4GemmBackend::CudnnFrontend : Fp4GemmBackend::None);
	}

	struct Fp4GemmReport
	{
		// The backend selected when the plan was built.  Direct cuBLASLt NVFP4 is
		// preferred; cuDNN Frontend is used only when the direct plan cannot run.
		Fp4GemmBackend preferred_backend = Fp4GemmBackend::None;
		Fp4GemmBackend last_backend = Fp4GemmBackend::None;
		std::uint64_t cudnn_frontend_executions = 0;
		std::uint64_t cublaslt_executions = 0;
		std::uint64_t prequantized_right_executions = 0;
		std::uint64_t failed_executions = 0;
	};

	struct Fp4GemmPlan;

	struct Fp4GemmSpec
	{
		int batch = 1;
		int rows = 0;
		int columns = 0;
		int reduction = 0;
		// Inference weights are immutable after loading/fusion, so retain their
		// packed NVFP4 representation after the first forward pass.  Training
		// plans keep this false because their weights can change every update.
		bool cache_left_operand = false;
	};

	bool fp4_runtime_supported();
	Fp4GemmPlan * fp4_gemm_plan_create(const Fp4GemmSpec & spec);
	void fp4_gemm_plan_destroy(Fp4GemmPlan * plan);
	size_t fp4_gemm_workspace_bytes(const Fp4GemmPlan * plan);
	/// Prepack a static inference left operand (normally convolution weights)
	/// into the plan's cuBLASLt NVFP4 storage.  The work is enqueued on Darknet's
	/// CUDA stream and is therefore ordered before the first inference request.
	bool fp4_gemm_prepare_cached_left_operand(Fp4GemmPlan * plan, const float * left_operand);
	/// True when the plan can consume a retained packed [N,K] right operand
	/// directly through the CUDA_R_4F_E2M1 cuBLASLt interface.
	bool fp4_gemm_supports_prequantized_right(const Fp4GemmPlan * plan);
	Fp4GemmReport fp4_gemm_report(const Fp4GemmPlan * plan);
	const char * fp4_gemm_backend_name(Fp4GemmBackend backend);
	bool fp4_gemm_execute(Fp4GemmPlan * plan,
		const float * a_rowmajor, const float * b_transposed_rowmajor,
		float * output, void * workspace, size_t workspace_bytes);
	/// Run a direct NVFP4 GEMM using a prequantized right operand.  The packed
	/// data and scales must use fp4_quantize_nchw_to_cublaslt_gpu()'s [N,K]
	/// layout for this plan.  This is used only for compatible 1x1 conv chains.
	bool fp4_gemm_execute_prequantized_right(Fp4GemmPlan * plan,
		const uint8_t * packed_right, const uint8_t * right_scales,
		float * output, void * workspace, size_t workspace_bytes);
}
