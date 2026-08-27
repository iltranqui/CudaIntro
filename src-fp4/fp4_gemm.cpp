#include "fp4_gemm.hpp"
#include "fp4_kernels.hpp"

#include "darknet_internal.hpp"

#include <algorithm>
#include <memory>
#include <unordered_map>

#include <cuda_fp4.h>
#include <cuda_runtime_api.h>
#include <cudnn.h>
#include <cudnn_frontend.h>
#include <cublasLt.h>

#if !defined(CUDART_VERSION) || CUDART_VERSION < 13020
#error "Darknet FP4 requires CUDA runtime headers from CUDA 13.2 or newer"
#endif

// No hardcoded CUDNN_VERSION floor here: this TU only builds once
// CM_dependencies.cmake's configure-time feature-detection test confirms the
// real API (fe::graph::Block_scale_quantize_attributes /
// Block_scale_dequantize_attributes) actually compiles against whatever
// cuDNN Frontend headers are found, regardless of version number.

namespace fe = cudnn_frontend;

namespace Darknet
{
	struct Fp4GemmPlan
	{
		Fp4GemmSpec spec;
		size_t workspace_bytes = 0;
		size_t frontend_workspace_bytes = 0;
		size_t cublaslt_workspace_bytes = 0;
		std::shared_ptr<fe::graph::Graph> graph;
		std::shared_ptr<fe::graph::Tensor_attributes> a, b, output;
		cublasLtMatmulDesc_t operation = nullptr;
		cublasLtMatrixLayout_t lt_a = nullptr, lt_b = nullptr, lt_c = nullptr, lt_d = nullptr;
		cublasLtMatmulAlgo_t algorithm = {};
		bool cublaslt_ready = false;
		// The public GEMM signature is (left_operand, right_operand).  cuBLASLt
		// swaps them internally to produce Darknet's row-major output, so the
		// immutable left operand is stored in its B layout.
		uint8_t * cublaslt_cached_b = nullptr;
		uint8_t * cublaslt_cached_b_scales = nullptr;
		const float * cublaslt_cached_left_operand = nullptr;
		// Prepacking can run while a model is being loaded on Darknet's setup
		// stream, whereas prediction can later activate a private network stream.
		// This event bridges those streams without putting weight quantization in
		// the timed/request path.
		cudaEvent_t cublaslt_cached_b_ready = nullptr;
		Fp4GemmReport report;

		~Fp4GemmPlan()
		{
			if (cublaslt_cached_b_ready) cudaEventDestroy(cublaslt_cached_b_ready);
			if (cublaslt_cached_b_scales) cudaFree(cublaslt_cached_b_scales);
			if (cublaslt_cached_b) cudaFree(cublaslt_cached_b);
			if (lt_d) cublasLtMatrixLayoutDestroy(lt_d);
			if (lt_c) cublasLtMatrixLayoutDestroy(lt_c);
			if (lt_b) cublasLtMatrixLayoutDestroy(lt_b);
			if (lt_a) cublasLtMatrixLayoutDestroy(lt_a);
			if (operation) cublasLtMatmulDescDestroy(operation);
		}
	};

	namespace
	{
		constexpr size_t align_256(const size_t bytes) { return (bytes + 255U) & ~size_t{255U}; }

		bool setup_cublaslt_nvfp4(Fp4GemmPlan & plan)
		{
			const int m = plan.spec.rows, n = plan.spec.columns, k = plan.spec.reduction;
			// cuBLASLt NVFP4 requires TN input and 16-byte-aligned packed K rows.
			// Swapping operands computes C^T as column-major [N,M], whose bytes are
			// exactly the row-major [M,N] output expected by Darknet.
			if (plan.spec.batch != 1 || k % 32 != 0 || m % 8 != 0 || n % 8 != 0) return false;
			if (cublasLtMatmulDescCreate(&plan.operation, CUBLAS_COMPUTE_32F, CUDA_R_32F) != CUBLAS_STATUS_SUCCESS) return false;
			const cublasOperation_t transpose = CUBLAS_OP_T, no_transpose = CUBLAS_OP_N;
			const cublasLtMatmulMatrixScale_t block_scale = CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3;
			if (cublasLtMatmulDescSetAttribute(plan.operation, CUBLASLT_MATMUL_DESC_TRANSA, &transpose, sizeof(transpose)) != CUBLAS_STATUS_SUCCESS ||
				cublasLtMatmulDescSetAttribute(plan.operation, CUBLASLT_MATMUL_DESC_TRANSB, &no_transpose, sizeof(no_transpose)) != CUBLAS_STATUS_SUCCESS ||
				cublasLtMatmulDescSetAttribute(plan.operation, CUBLASLT_MATMUL_DESC_A_SCALE_MODE, &block_scale, sizeof(block_scale)) != CUBLAS_STATUS_SUCCESS ||
				cublasLtMatmulDescSetAttribute(plan.operation, CUBLASLT_MATMUL_DESC_B_SCALE_MODE, &block_scale, sizeof(block_scale)) != CUBLAS_STATUS_SUCCESS) return false;
			if (cublasLtMatrixLayoutCreate(&plan.lt_a, CUDA_R_4F_E2M1, k, n, k) != CUBLAS_STATUS_SUCCESS ||
				cublasLtMatrixLayoutCreate(&plan.lt_b, CUDA_R_4F_E2M1, k, m, k) != CUBLAS_STATUS_SUCCESS ||
				cublasLtMatrixLayoutCreate(&plan.lt_c, CUDA_R_32F, n, m, n) != CUBLAS_STATUS_SUCCESS ||
				cublasLtMatrixLayoutCreate(&plan.lt_d, CUDA_R_32F, n, m, n) != CUBLAS_STATUS_SUCCESS) return false;

			cublasLtMatmulPreference_t preference = nullptr;
			if (cublasLtMatmulPreferenceCreate(&preference) != CUBLAS_STATUS_SUCCESS) return false;
			// Blackwell's NVFP4 kernels frequently benefit from a larger heuristic
			// search/workspace budget.  The caller's existing convolution workspace
			// accounting reserves the selected amount.
			constexpr size_t workspace_limit = 32U * 1024U * 1024U;
			const cublasStatus_t preference_status = cublasLtMatmulPreferenceSetAttribute(preference,
				CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspace_limit, sizeof(workspace_limit));
			cublasLtMatmulHeuristicResult_t heuristic = {};
			int count = 0;
			const cublasStatus_t heuristic_status = preference_status == CUBLAS_STATUS_SUCCESS
				? cublasLtMatmulAlgoGetHeuristic(cublaslt_handle(), plan.operation, plan.lt_a, plan.lt_b,
					plan.lt_c, plan.lt_d, preference, 1, &heuristic, &count)
				: preference_status;
			cublasLtMatmulPreferenceDestroy(preference);
			if (heuristic_status != CUBLAS_STATUS_SUCCESS || count == 0 || heuristic.state != CUBLAS_STATUS_SUCCESS) return false;
			plan.algorithm = heuristic.algo;
			plan.cublaslt_workspace_bytes = heuristic.workspaceSize;
			plan.cublaslt_ready = true;
			return true;
		}

		size_t cublaslt_storage_bytes(const Fp4GemmPlan & plan)
		{
			const int m = plan.spec.rows, n = plan.spec.columns, k = plan.spec.reduction;
			size_t bytes = align_256(fp4_cublaslt_packed_bytes(n, k)) +
				align_256(fp4_cublaslt_scale_bytes(n, k));
			if (!plan.spec.cache_left_operand)
			{
				bytes += align_256(fp4_cublaslt_packed_bytes(m, k)) +
					align_256(fp4_cublaslt_scale_bytes(m, k));
			}
			return bytes + plan.cublaslt_workspace_bytes;
		}

		void release_cached_left_operand(Fp4GemmPlan & plan)
		{
			if (plan.cublaslt_cached_b_ready)
			{
				cudaEventDestroy(plan.cublaslt_cached_b_ready);
				plan.cublaslt_cached_b_ready = nullptr;
			}
			if (plan.cublaslt_cached_b_scales)
			{
				cudaFree(plan.cublaslt_cached_b_scales);
				plan.cublaslt_cached_b_scales = nullptr;
			}
			if (plan.cublaslt_cached_b)
			{
				cudaFree(plan.cublaslt_cached_b);
				plan.cublaslt_cached_b = nullptr;
			}
			plan.cublaslt_cached_left_operand = nullptr;
		}

		bool ensure_cached_left_operand(Fp4GemmPlan & plan, const float * left_operand)
		{
			if (!plan.spec.cache_left_operand) return false;
			if (plan.cublaslt_cached_left_operand == left_operand && plan.cublaslt_cached_b && plan.cublaslt_cached_b_scales)
			{
				return true;
			}

			release_cached_left_operand(plan);
			const size_t packed_bytes = fp4_cublaslt_packed_bytes(plan.spec.rows, plan.spec.reduction);
			const size_t scale_bytes = fp4_cublaslt_scale_bytes(plan.spec.rows, plan.spec.reduction);
			if (cudaMalloc(reinterpret_cast<void **>(&plan.cublaslt_cached_b), packed_bytes) != cudaSuccess ||
				cudaMalloc(reinterpret_cast<void **>(&plan.cublaslt_cached_b_scales), scale_bytes) != cudaSuccess ||
				!fp4_quantize_cublaslt_gpu(left_operand, plan.spec.rows, plan.spec.reduction,
					plan.cublaslt_cached_b, plan.cublaslt_cached_b_scales))
			{
				release_cached_left_operand(plan);
				return false;
			}
			plan.cublaslt_cached_left_operand = left_operand;
			if (cudaEventCreateWithFlags(&plan.cublaslt_cached_b_ready, cudaEventDisableTiming) != cudaSuccess ||
				cudaEventRecord(plan.cublaslt_cached_b_ready, get_cuda_stream()) != cudaSuccess)
			{
				release_cached_left_operand(plan);
				return false;
			}
			return true;
		}

		bool wait_for_cached_left_operand(const Fp4GemmPlan & plan, const cudaStream_t stream)
		{
			if (plan.cublaslt_cached_b_ready == nullptr) return true;

			// CUDA Graph capture cannot import a wait for an event that was recorded
			// before capture began.  Darknet finishes model preparation before its
			// graph capture (network_predict_gpu() synchronizes immediately before
			// cudaStreamBeginCapture()), so the immutable packed weights are already
			// visible at this point.  Normal non-captured execution still uses the
			// event to bridge the setup stream and a private network stream.
			cudaStreamCaptureStatus capture_status = cudaStreamCaptureStatusNone;
			if (cudaStreamIsCapturing(stream, &capture_status) == cudaSuccess &&
				capture_status != cudaStreamCaptureStatusNone)
			{
				return true;
			}
			return cudaStreamWaitEvent(stream, plan.cublaslt_cached_b_ready, 0) == cudaSuccess;
		}

		bool execute_cublaslt_nvfp4(Fp4GemmPlan & plan, const float * left_operand,
			const float * right_operand, float * output, void * workspace)
		{
			if (!plan.cublaslt_ready || !workspace) return false;

			char * cursor = static_cast<char *>(workspace);
			uint8_t * const packed_a = reinterpret_cast<uint8_t *>(cursor);
			cursor += align_256(fp4_cublaslt_packed_bytes(plan.spec.columns, plan.spec.reduction));
			uint8_t * const scales_a = reinterpret_cast<uint8_t *>(cursor);
			cursor += align_256(fp4_cublaslt_scale_bytes(plan.spec.columns, plan.spec.reduction));

			uint8_t * packed_b = nullptr;
			uint8_t * scales_b = nullptr;
			if (plan.spec.cache_left_operand)
			{
				if (!ensure_cached_left_operand(plan, left_operand)) return false;
				packed_b = plan.cublaslt_cached_b;
				scales_b = plan.cublaslt_cached_b_scales;
			}
			else
			{
				packed_b = reinterpret_cast<uint8_t *>(cursor);
				cursor += align_256(fp4_cublaslt_packed_bytes(plan.spec.rows, plan.spec.reduction));
				scales_b = reinterpret_cast<uint8_t *>(cursor);
				cursor += align_256(fp4_cublaslt_scale_bytes(plan.spec.rows, plan.spec.reduction));
			}

			if (!fp4_quantize_cublaslt_gpu(right_operand, plan.spec.columns, plan.spec.reduction, packed_a, scales_a) ||
				(!plan.spec.cache_left_operand && !fp4_quantize_cublaslt_gpu(left_operand,
					plan.spec.rows, plan.spec.reduction, packed_b, scales_b)))
			{
				return false;
			}
			const void * a_scale = scales_a;
			const void * b_scale = scales_b;
			if (cublasLtMatmulDescSetAttribute(plan.operation, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &a_scale, sizeof(a_scale)) != CUBLAS_STATUS_SUCCESS ||
				cublasLtMatmulDescSetAttribute(plan.operation, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &b_scale, sizeof(b_scale)) != CUBLAS_STATUS_SUCCESS)
			{
				return false;
			}
			const cudaStream_t stream = get_cuda_stream();
			if (plan.spec.cache_left_operand && !wait_for_cached_left_operand(plan, stream)) return false;
			constexpr float alpha = 1.0f, beta = 0.0f;
			return cublasLtMatmul(cublaslt_handle(), plan.operation, &alpha, packed_a, plan.lt_a,
				packed_b, plan.lt_b, &beta, output, plan.lt_c, output, plan.lt_d,
				&plan.algorithm, cursor, plan.cublaslt_workspace_bytes, stream) == CUBLAS_STATUS_SUCCESS;
		}

		bool execute_cublaslt_nvfp4_prequantized_right(Fp4GemmPlan & plan,
			const uint8_t * packed_right, const uint8_t * right_scales,
			float * output, void * workspace)
		{
			if (!plan.cublaslt_ready || !plan.spec.cache_left_operand || !packed_right || !right_scales || !workspace ||
				!plan.cublaslt_cached_left_operand ||
				!ensure_cached_left_operand(plan, plan.cublaslt_cached_left_operand))
			{
				return false;
			}

			// Reserve the normal right-operand staging range so the heuristic's
			// workspace remains aligned and never overlaps a caller-visible relay.
			char * cursor = static_cast<char *>(workspace);
			cursor += align_256(fp4_cublaslt_packed_bytes(plan.spec.columns, plan.spec.reduction));
			cursor += align_256(fp4_cublaslt_scale_bytes(plan.spec.columns, plan.spec.reduction));
			const void * a_scale = right_scales;
			const void * b_scale = plan.cublaslt_cached_b_scales;
			if (cublasLtMatmulDescSetAttribute(plan.operation, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &a_scale, sizeof(a_scale)) != CUBLAS_STATUS_SUCCESS ||
				cublasLtMatmulDescSetAttribute(plan.operation, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &b_scale, sizeof(b_scale)) != CUBLAS_STATUS_SUCCESS)
			{
				return false;
			}
			const cudaStream_t stream = get_cuda_stream();
			if (!wait_for_cached_left_operand(plan, stream)) return false;
			constexpr float alpha = 1.0f, beta = 0.0f;
			return cublasLtMatmul(cublaslt_handle(), plan.operation, &alpha, packed_right, plan.lt_a,
				plan.cublaslt_cached_b, plan.lt_b, &beta, output, plan.lt_c, output, plan.lt_d,
				&plan.algorithm, cursor, plan.cublaslt_workspace_bytes, stream) == CUBLAS_STATUS_SUCCESS;
		}
	}

	bool fp4_runtime_supported()
	{
		int runtime_version = 0;
		int device = 0;
		cudaDeviceProp prop = {};
		if (cudaRuntimeGetVersion(&runtime_version) != cudaSuccess || runtime_version < 13020 ||
			cudaGetDevice(&device) != cudaSuccess || cudaGetDeviceProperties(&prop, device) != cudaSuccess)
		{
			return false;
		}
		// The cuDNN Frontend block-scale API this file uses was already confirmed to
		// exist at configure time (this TU only builds when DARKNET_BUILD_FP4 passed
		// the feature-detection compile test in CM_dependencies.cmake) -- do not
		// re-gate on a hardcoded cuDNN version number here too, or this silently
		// disables FP4 on any machine whose installed cuDNN differs from whatever
		// number was last hand-picked.
		const int sm = prop.major * 10 + prop.minor;
		return sm == 100 || sm == 103 || sm == 110 || sm == 120 || sm == 121;
	}

	Fp4GemmPlan * fp4_gemm_plan_create(const Fp4GemmSpec & spec)
	{
		// NVFP4 uses 16-element reduction blocks.  The Frontend's inferred scale
		// dimension truncates this division, so accepting a ragged K would silently
		// describe fewer scales than the input requires.
		if (!fp4_runtime_supported() || spec.batch <= 0 || spec.rows <= 0 || spec.columns <= 0 ||
			spec.reduction <= 0 || spec.reduction % 16 != 0)
		{
			return nullptr;
		}
		auto plan = std::make_unique<Fp4GemmPlan>();
		plan->spec = spec;
		plan->graph = std::make_shared<fe::graph::Graph>();
		plan->graph->set_intermediate_data_type(fe::DataType_t::FLOAT).set_compute_data_type(fe::DataType_t::FLOAT);
		const int64_t b = spec.batch, m = spec.rows, n = spec.columns, k = spec.reduction;
		plan->a = plan->graph->tensor(fe::graph::Tensor_attributes().set_name("fp4_a_source").set_data_type(fe::DataType_t::FLOAT).set_dim({b,m,k}).set_stride({m*k,k,1}));
		plan->b = plan->graph->tensor(fe::graph::Tensor_attributes().set_name("fp4_b_source").set_data_type(fe::DataType_t::FLOAT).set_dim({b,k,n}).set_stride({k*n,1,k}));
		auto [qa, sa] = plan->graph->block_scale_quantize(plan->a,
			fe::graph::Block_scale_quantize_attributes().set_block_size(16).set_axis(2).set_transpose(false));
		auto [qb, sb] = plan->graph->block_scale_quantize(plan->b,
			fe::graph::Block_scale_quantize_attributes().set_block_size(16).set_axis(1).set_transpose(false));
		qa->set_data_type(fe::DataType_t::FP4_E2M1);
		qb->set_data_type(fe::DataType_t::FP4_E2M1);
		sa->set_data_type(fe::DataType_t::FP8_E4M3).set_reordering_type(fe::TensorReordering_t::F8_128x4);
		sb->set_data_type(fe::DataType_t::FP8_E4M3).set_reordering_type(fe::TensorReordering_t::F8_128x4);
		// The graph fallback consumes positive UE4M3 scales.  Set this explicitly:
		// cuDNN Frontend exposes the attribute without a default initializer.
		auto da = plan->graph->block_scale_dequantize(qa, sa,
			fe::graph::Block_scale_dequantize_attributes().set_block_size({1,16}).set_is_negative_scale(false));
		auto db = plan->graph->block_scale_dequantize(qb, sb,
			fe::graph::Block_scale_dequantize_attributes().set_block_size({16,1}).set_is_negative_scale(false));
		auto output = plan->graph->matmul(da, db, fe::graph::Matmul_attributes().set_name("fp4_gemm").set_compute_data_type(fe::DataType_t::FLOAT));
		plan->output = output;
		plan->output->set_output(true).set_data_type(fe::DataType_t::FLOAT);
		auto & graph = *plan->graph;

		// Run each build stage individually (instead of one short-circuited ||
		// chain) so that if every backend ultimately fails, we can report which
		// specific stage rejected the graph and cuDNN Frontend's own message --
		// "no plan supported" alone doesn't say whether it was an unsupported
		// shape, a missing kernel for this GPU/cuDNN combo, or something else.
		std::string frontend_failure_reason;
		const auto run_stage = [&](fe::error_t status, const char * stage_name) -> bool
		{
			if (status.is_good())
			{
				return true;
			}
			frontend_failure_reason = std::string(stage_name) + ": " + status.get_message();
			return false;
		};

		const bool graph_built =
			run_stage(graph.validate(), "validate") &&
			run_stage(graph.build_operation_graph(cudnn_handle()), "build_operation_graph") &&
			run_stage(graph.create_execution_plans({fe::HeurMode_t::A}), "create_execution_plans") &&
			run_stage(graph.check_support(cudnn_handle()), "check_support") &&
			run_stage(graph.build_plans(cudnn_handle(), fe::BuildPlanPolicy_t::HEURISTICS_CHOICE), "build_plans");

		if (!graph_built)
		{
			plan->graph.reset();
		}
		if (plan->graph)
		{
			int64_t bytes = 0;
			if (!graph.get_workspace_size(bytes).is_good()) plan->graph.reset();
			else plan->frontend_workspace_bytes = static_cast<size_t>(std::max<int64_t>(bytes, 0));
		}
		setup_cublaslt_nvfp4(*plan);
		if (!plan->graph && !plan->cublaslt_ready)
		{
			if (Darknet::CfgAndState::get().is_verbose)
			{
				Darknet::display_warning_msg("FP4 GEMM plan [" + std::to_string(m) + "x" + std::to_string(n) +
					"x" + std::to_string(k) + "] rejected -- cuDNN Frontend " +
					(frontend_failure_reason.empty() ? "graph unavailable" : frontend_failure_reason) +
					"; cuBLASLt NVFP4 fallback also unavailable for this shape (needs batch=1, k%32==0, m%8==0, n%8==0).\n");
			}
			return nullptr;
		}
		plan->report.preferred_backend = select_fp4_gemm_backend(plan->graph != nullptr, plan->cublaslt_ready);
		plan->workspace_bytes = std::max(plan->frontend_workspace_bytes,
			plan->cublaslt_ready ? cublaslt_storage_bytes(*plan) : 0U);
		return plan.release();
	}

	void fp4_gemm_plan_destroy(Fp4GemmPlan * plan) { delete plan; }
	size_t fp4_gemm_workspace_bytes(const Fp4GemmPlan * plan) { return plan ? plan->workspace_bytes : 0; }
	bool fp4_gemm_prepare_cached_left_operand(Fp4GemmPlan * plan, const float * left_operand)
	{
		return plan && left_operand && plan->cublaslt_ready && plan->spec.cache_left_operand &&
			ensure_cached_left_operand(*plan, left_operand);
	}
	bool fp4_gemm_supports_prequantized_right(const Fp4GemmPlan * plan)
	{
		return plan && plan->cublaslt_ready && plan->spec.cache_left_operand;
	}
	Fp4GemmReport fp4_gemm_report(const Fp4GemmPlan * plan) { return plan ? plan->report : Fp4GemmReport{}; }

	const char * fp4_gemm_backend_name(const Fp4GemmBackend backend)
	{
		switch (backend)
		{
			case Fp4GemmBackend::CudnnFrontend: return "cudnn-frontend";
			case Fp4GemmBackend::CublasLt:      return "cublaslt";
			case Fp4GemmBackend::None:          return "none";
		}
		return "none";
	}

	bool fp4_gemm_execute(Fp4GemmPlan * plan, const float * a, const float * b,
		float * output, void * workspace, const size_t workspace_bytes)
	{
		TAT(TATPARMS);

		if (!plan || !a || !b || !output || workspace_bytes < plan->workspace_bytes ||
			(plan->workspace_bytes != 0 && !workspace)) return false;
		// Prefer the explicit cuBLASLt CUDA_R_4F_E2M1 plan.  Unlike a graph whose
		// internal engine is opaque to the caller, this dispatch is an unambiguous
		// NVFP4 Tensor Core GEMM and is therefore the benchmark's proof point.
		if (plan->cublaslt_ready && execute_cublaslt_nvfp4(*plan, a, b, output, workspace))
		{
			plan->report.last_backend = Fp4GemmBackend::CublasLt;
			++plan->report.cublaslt_executions;
			return true;
		}

		// cuDNN Frontend remains useful for shapes whose direct cuBLASLt plan is
		// unavailable.  Its graph keeps block-scale quantization/dequantization
		// and processing entirely on the GPU.
		if (plan->graph)
		{
			std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void *> pack = {
				{plan->a, const_cast<float *>(a)}, {plan->b, const_cast<float *>(b)},
				{plan->output, output}};
			if (plan->graph->execute(cudnn_handle(), pack, workspace).is_good())
			{
				plan->report.last_backend = Fp4GemmBackend::CudnnFrontend;
				++plan->report.cudnn_frontend_executions;
				return true;
			}
		}
		plan->report.last_backend = Fp4GemmBackend::None;
		++plan->report.failed_executions;
		return false;
	}

	bool fp4_gemm_execute_prequantized_right(Fp4GemmPlan * plan,
		const uint8_t * packed_right, const uint8_t * right_scales,
		float * output, void * workspace, const size_t workspace_bytes)
	{
		TAT(TATPARMS);

		if (!plan || !packed_right || !right_scales || !output || workspace_bytes < plan->workspace_bytes ||
			(plan->workspace_bytes != 0 && !workspace))
		{
			return false;
		}
		if (execute_cublaslt_nvfp4_prequantized_right(*plan, packed_right, right_scales, output, workspace))
		{
			plan->report.last_backend = Fp4GemmBackend::CublasLt;
			++plan->report.cublaslt_executions;
			++plan->report.prequantized_right_executions;
			return true;
		}
		plan->report.last_backend = Fp4GemmBackend::None;
		++plan->report.failed_executions;
		return false;
	}
}
