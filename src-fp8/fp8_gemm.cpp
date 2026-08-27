#include "fp8_gemm.hpp"
#include "darknet_internal.hpp"

#include <cstdio>
#include <cstdlib>

namespace Darknet
{
	int fp8_round_up_to_16(const int value)
	{
		return (value + 15) & ~15;
	}

	Fp8SharedDyLayout fp8_shared_dy_layout(const int filters, const int spatial)
	{
		Fp8SharedDyLayout layout;
		layout.filters_pad = fp8_round_up_to_16(filters);
		layout.spatial_pad = fp8_round_up_to_16(spatial);
		layout.bytes = static_cast<size_t>(layout.filters_pad) * layout.spatial_pad;
		return layout;
	}

	size_t fp8_gemm_workspace_bytes()
	{
		return 32ULL * 1024ULL * 1024ULL;
	}

	Fp8GemmSpec fp8_gemm_training_spec(const Fp8TrainingGemm kind, const int filters, const int kernel, const int spatial)
	{
		Fp8GemmSpec spec;
		switch (kind)
		{
			case Fp8TrainingGemm::Forward:
				spec.output_rows = filters;
				spec.output_cols = spatial;
				spec.reduction = kernel;
				spec.a_format = Fp8Format::E4M3;
				spec.b_format = Fp8Format::E4M3;
				break;
			case Fp8TrainingGemm::WeightGradient:
				spec.output_rows = filters;
				spec.output_cols = kernel;
				spec.reduction = spatial;
				spec.a_format = Fp8Format::E5M2;
				spec.b_format = Fp8Format::E4M3;
				break;
			case Fp8TrainingGemm::WeightGradientDirectUpdate:
				spec.output_rows = kernel;
				spec.output_cols = filters;
				spec.reduction = spatial;
				spec.a_format = Fp8Format::E4M3;
				spec.b_format = Fp8Format::E5M2;
				break;
			case Fp8TrainingGemm::DataGradient:
				spec.output_rows = kernel;
				spec.output_cols = spatial;
				spec.reduction = filters;
				spec.a_format = Fp8Format::E4M3;
				spec.b_format = Fp8Format::E5M2;
				break;
			case Fp8TrainingGemm::DataGradientDirectUpdate:
				spec.output_rows = spatial;
				spec.output_cols = kernel;
				spec.reduction = filters;
				spec.a_format = Fp8Format::E5M2;
				spec.b_format = Fp8Format::E4M3;
				spec.output = Fp8GemmOutput::Fp32;
				spec.batch_a = true;
				spec.batch_b = false;
				break;
		}
		spec.reduction_pad = fp8_round_up_to_16(spec.reduction);
		return spec;
	}
}

#if defined(DARKNET_GPU_CUDA) && defined(CUDART_VERSION) && CUDART_VERSION >= 12010

namespace Darknet
{
	struct Fp8GemmPlan
	{
		int m = 0;
		int n = 0;
		int k = 0;
		int k_pad = 0;
		size_t workspace_bytes = 0;
		cublasLtMatmulDesc_t op_desc = nullptr;
		cublasLtMatrixLayout_t a_desc = nullptr;
		cublasLtMatrixLayout_t b_desc = nullptr;
		cublasLtMatrixLayout_t c_desc = nullptr;
		cublasLtMatrixLayout_t d_desc = nullptr;
			cublasLtMatmulAlgo_t algo = {};
			bool reverse_operands = false;
			bool output_column_major = false;
			Fp8GemmOutput output = Fp8GemmOutput::Fp32;
		};

	namespace
	{
		void destroy_layout(cublasLtMatrixLayout_t & desc)
		{
			if (desc)
			{
				CHECK_CUBLAS(cublasLtMatrixLayoutDestroy(desc));
				desc = nullptr;
			}
		}

		void destroy_op(cublasLtMatmulDesc_t & desc)
		{
			if (desc)
			{
				CHECK_CUBLAS(cublasLtMatmulDescDestroy(desc));
				desc = nullptr;
			}
		}

		bool create_col_major_layout(cublasLtMatrixLayout_t * desc, cudaDataType_t type, int rows, int cols, int ld)
		{
			return cublasLtMatrixLayoutCreate(desc, type, rows, cols, ld) == CUBLAS_STATUS_SUCCESS;
		}

		bool set_layout_batch(cublasLtMatrixLayout_t desc, const int batch, const int64_t stride)
		{
			cublasStatus_t status = cublasLtMatrixLayoutSetAttribute(desc, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch, sizeof(batch));
			if (status == CUBLAS_STATUS_SUCCESS)
			{
				status = cublasLtMatrixLayoutSetAttribute(desc, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &stride, sizeof(stride));
			}
			return status == CUBLAS_STATUS_SUCCESS;
		}

		cudaDataType_t fp8_cuda_data_type(const Fp8Format format)
		{
			switch (format)
			{
				case Fp8Format::E4M3: return CUDA_R_8F_E4M3;
				case Fp8Format::E5M2: return CUDA_R_8F_E5M2;
			}
			return CUDA_R_8F_E4M3;
		}

		cudaDataType_t fp8_output_cuda_data_type(const Fp8GemmOutput output)
		{
			switch (output)
			{
				case Fp8GemmOutput::Fp32: return CUDA_R_32F;
				case Fp8GemmOutput::Bf16: return CUDA_R_16BF;
			}
			return CUDA_R_32F;
		}

		bool set_scale_pointers(cublasLtMatmulDesc_t op_desc, const float * a_scale_gpu, const float * b_scale_gpu)
		{
			cublasStatus_t status = cublasLtMatmulDescSetAttribute(
				op_desc,
				CUBLASLT_MATMUL_DESC_A_SCALE_POINTER,
				&a_scale_gpu,
				sizeof(a_scale_gpu));
			if (status == CUBLAS_STATUS_SUCCESS)
			{
				status = cublasLtMatmulDescSetAttribute(
					op_desc,
					CUBLASLT_MATMUL_DESC_B_SCALE_POINTER,
					&b_scale_gpu,
					sizeof(b_scale_gpu));
			}
			return status == CUBLAS_STATUS_SUCCESS;
		}

		void reset_plan_descriptors(Fp8GemmPlan * plan)
		{
			if (plan == nullptr)
			{
				return;
			}
			destroy_layout(plan->d_desc);
			destroy_layout(plan->c_desc);
			destroy_layout(plan->b_desc);
			destroy_layout(plan->a_desc);
			destroy_op(plan->op_desc);
			plan->workspace_bytes = 0;
		}

		/* cuBLASLt only supports the TN case for FP8 matmuls (A transposed, B not, both
		 * column-major).  We exploit the fact that a row-major (rows x cols) buffer is
		 * bit-identical to a column-major (cols x rows) buffer:
		 *
		 *   A buffer: row-major (output_rows x reduction_pad)  == col-major (k_pad x m), OP_T
		 *   B buffer: row-major (output_cols x reduction_pad)  == col-major (k_pad x n), OP_N
		 *   D buffer: column-major (output_rows x output_cols) -> converted to row-major after
		 */
		bool initialize_plan_descriptors(
			Fp8GemmPlan * plan,
			const Fp8GemmSpec & spec,
			const float * a_scale_gpu,
			const float * b_scale_gpu)
		{
			reset_plan_descriptors(plan);
			plan->reverse_operands = false;
			plan->output_column_major = true;
			plan->output = spec.output;

			cublasStatus_t status = cublasLtMatmulDescCreate(&plan->op_desc, CUBLAS_COMPUTE_32F, CUDA_R_32F);
			if (status != CUBLAS_STATUS_SUCCESS)
			{
				reset_plan_descriptors(plan);
				return false;
			}

			const cublasOperation_t op_t = CUBLAS_OP_T;
			const cublasOperation_t op_n = CUBLAS_OP_N;
			status = cublasLtMatmulDescSetAttribute(plan->op_desc, CUBLASLT_MATMUL_DESC_TRANSA, &op_t, sizeof(op_t));
			if (status == CUBLAS_STATUS_SUCCESS)
			{
				status = cublasLtMatmulDescSetAttribute(plan->op_desc, CUBLASLT_MATMUL_DESC_TRANSB, &op_n, sizeof(op_n));
			}
			if (status == CUBLAS_STATUS_SUCCESS && !set_scale_pointers(plan->op_desc, a_scale_gpu, b_scale_gpu))
			{
				status = CUBLAS_STATUS_INVALID_VALUE;
			}
			if (status != CUBLAS_STATUS_SUCCESS)
			{
				reset_plan_descriptors(plan);
				return false;
			}

			const cudaDataType_t output_type = fp8_output_cuda_data_type(spec.output);
			const int b_ld = spec.b_leading_dim > 0 ? spec.b_leading_dim : spec.reduction_pad;
			bool layouts_created =
				create_col_major_layout(&plan->a_desc, fp8_cuda_data_type(spec.a_format), spec.reduction_pad, spec.output_rows, spec.reduction_pad) &&
				create_col_major_layout(&plan->b_desc, fp8_cuda_data_type(spec.b_format), spec.reduction_pad, spec.output_cols, b_ld) &&
				create_col_major_layout(&plan->c_desc, output_type, spec.output_rows, spec.output_cols, spec.output_rows) &&
				create_col_major_layout(&plan->d_desc, output_type, spec.output_rows, spec.output_cols, spec.output_rows);
			if (layouts_created && spec.batch > 1)
			{
				// A (weights) is shared by default: stride 0.  B and D always advance by one
				// matrix per batch entry.  The batch flags select whether each operand advances
				// or broadcasts one matrix across all batch entries.
				const int64_t a_stride = spec.batch_a ? static_cast<int64_t>(spec.reduction_pad) * spec.output_rows : 0;
				const int64_t b_stride = spec.batch_b ? static_cast<int64_t>(b_ld) * spec.output_cols : 0;
				const int64_t d_stride = static_cast<int64_t>(spec.output_rows) * spec.output_cols;
				layouts_created =
					set_layout_batch(plan->a_desc, spec.batch, a_stride) &&
					set_layout_batch(plan->b_desc, spec.batch, b_stride) &&
					set_layout_batch(plan->c_desc, spec.batch, d_stride) &&
					set_layout_batch(plan->d_desc, spec.batch, d_stride);
			}
			if (!layouts_created)
			{
				reset_plan_descriptors(plan);
				return false;
			}

			cublasLtMatmulPreference_t preference = nullptr;
			status = cublasLtMatmulPreferenceCreate(&preference);
			if (status != CUBLAS_STATUS_SUCCESS)
			{
				reset_plan_descriptors(plan);
				return false;
			}

			const uint64_t max_workspace = fp8_gemm_workspace_bytes();
			status = cublasLtMatmulPreferenceSetAttribute(
				preference,
				CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
				&max_workspace,
				sizeof(max_workspace));
			if (status != CUBLAS_STATUS_SUCCESS)
			{
				CHECK_CUBLAS(cublasLtMatmulPreferenceDestroy(preference));
				reset_plan_descriptors(plan);
				return false;
			}

			cublasLtMatmulHeuristicResult_t heuristic = {};
			int returned = 0;
			status = cublasLtMatmulAlgoGetHeuristic(
				cublaslt_handle(),
				plan->op_desc,
				plan->a_desc,
				plan->b_desc,
				plan->c_desc,
				plan->d_desc,
				preference,
				1,
				&heuristic,
				&returned);
			CHECK_CUBLAS(cublasLtMatmulPreferenceDestroy(preference));
			if (status != CUBLAS_STATUS_SUCCESS || returned <= 0 || heuristic.state != CUBLAS_STATUS_SUCCESS)
			{
				if (std::getenv("DARKNET_FP8_DEBUG") != nullptr)
				{
					std::fprintf(
						stderr,
						"fp8_gemm heuristic failed layout=TN status=%d returned=%d state=%d m=%d n=%d k=%d kpad=%d a_format=%d b_format=%d output=%d\n",
						static_cast<int>(status),
						returned,
						static_cast<int>(heuristic.state),
						spec.output_rows,
						spec.output_cols,
						spec.reduction,
						spec.reduction_pad,
						static_cast<int>(spec.a_format),
						static_cast<int>(spec.b_format),
						static_cast<int>(spec.output));
				}
				reset_plan_descriptors(plan);
				return false;
			}

			plan->algo = heuristic.algo;
			plan->workspace_bytes = heuristic.workspaceSize;
			return true;
		}
	}

	bool fp8_gemm_supported()
	{
		int device = 0;
		cudaError_t status = cudaGetDevice(&device);
		if (status != cudaSuccess)
		{
			return false;
		}

		cudaDeviceProp prop = {};
		status = cudaGetDeviceProperties(&prop, device);
		if (status != cudaSuccess)
		{
			return false;
		}

		const int capability = prop.major * 100 + prop.minor * 10;
		return capability >= 890;
	}

	bool fp8_sm89_optimization_supported()
	{
#ifdef DARKNET_FP8_TARGET_SM89
		int device = 0;
		cudaDeviceProp prop = {};
		return cudaGetDevice(&device) == cudaSuccess &&
			cudaGetDeviceProperties(&prop, device) == cudaSuccess &&
			prop.major == 8 && prop.minor == 9;
#else
		return false;
#endif
	}

	Fp8GemmPlan * fp8_gemm_plan_create(
		const int m,
		const int n,
		const int k,
		const int k_pad,
		const float * weight_scale_gpu,
		const float * input_scale_gpu)
	{
		Fp8GemmSpec spec;
		spec.output_rows = m;
		spec.output_cols = n;
		spec.reduction = k;
		spec.reduction_pad = k_pad;
		spec.a_format = Fp8Format::E4M3;
		spec.b_format = Fp8Format::E4M3;
		return fp8_gemm_plan_create_ex(spec, weight_scale_gpu, input_scale_gpu);
	}

	Fp8GemmPlan * fp8_gemm_plan_create_ex(const Fp8GemmSpec & spec, const float * a_scale_gpu, const float * b_scale_gpu)
	{
		// Note: output_rows (M) is intentionally not checked or rounded here -- no
		// vendored cuBLASLt documentation requires it to be a multiple of 16 for an
		// FP8 TN matmul (see the contract comment in fp8_gemm.hpp). An odd
		// output_rows (e.g. filters=255 on a YOLO detection head) falls through to
		// cublasLtMatmulAlgoGetHeuristic() below, which is the sole arbiter of
		// whether the shape is actually supported.
		if (!fp8_gemm_supported() ||
			spec.output_rows <= 0 ||
			spec.output_cols <= 0 ||
			spec.reduction <= 0 ||
			spec.reduction_pad < spec.reduction ||
			(spec.b_leading_dim > 0 && spec.b_leading_dim < spec.reduction_pad) ||
			spec.batch < 1 ||
			a_scale_gpu == nullptr ||
			b_scale_gpu == nullptr)
		{
			return nullptr;
		}

		Fp8GemmPlan * plan = new Fp8GemmPlan;
		plan->m = spec.output_rows;
		plan->n = spec.output_cols;
		plan->k = spec.reduction;
		plan->k_pad = spec.reduction_pad;

		if (!initialize_plan_descriptors(plan, spec, a_scale_gpu, b_scale_gpu))
		{
			fp8_gemm_plan_destroy(plan);
			return nullptr;
		}

		return plan;
	}

	void fp8_gemm_plan_destroy(Fp8GemmPlan * plan)
	{
		if (plan == nullptr)
		{
			return;
		}

		reset_plan_descriptors(plan);
		delete plan;
	}

		bool fp8_gemm_output_is_column_major(const Fp8GemmPlan * plan)
		{
			return plan != nullptr && plan->output_column_major;
		}

		bool fp8_gemm_output_is_fp32(const Fp8GemmPlan * plan)
		{
			return plan != nullptr && plan->output == Fp8GemmOutput::Fp32;
		}

		size_t fp8_gemm_output_element_bytes(const Fp8GemmPlan * plan)
		{
			if (plan == nullptr)
			{
				return 0;
			}
			return plan->output == Fp8GemmOutput::Fp32 ? sizeof(float) : sizeof(unsigned short);
		}

		bool fp8_gemm(
			Fp8GemmPlan * plan,
			const void * weights_fp8_gpu,
			const void * input_fp8_gpu,
			void * output_gpu,
			void * workspace,
			const size_t workspace_bytes,
			const float beta)
		{
			TAT(TATPARMS);

			if (plan == nullptr || weights_fp8_gpu == nullptr || input_fp8_gpu == nullptr || output_gpu == nullptr)
			{
				return false;
		}
		if (workspace_bytes < plan->workspace_bytes)
		{
			return false;
		}

			const float alpha = 1.0f;
			const void * a_gpu = plan->reverse_operands ? input_fp8_gpu : weights_fp8_gpu;
			const void * b_gpu = plan->reverse_operands ? weights_fp8_gpu : input_fp8_gpu;
			const cublasStatus_t status = cublasLtMatmul(
			cublaslt_handle(),
			plan->op_desc,
			&alpha,
			a_gpu,
			plan->a_desc,
			b_gpu,
				plan->b_desc,
				&beta,
				output_gpu,
				plan->c_desc,
				output_gpu,
			plan->d_desc,
			&plan->algo,
			workspace,
			plan->workspace_bytes,
			get_cuda_stream());

		return status == CUBLAS_STATUS_SUCCESS;
	}
}

#else

namespace Darknet
{
	struct Fp8GemmPlan {};

	bool fp8_gemm_supported()
	{
		return false;
	}

	bool fp8_sm89_optimization_supported()
	{
		return false;
	}

	Fp8GemmPlan * fp8_gemm_plan_create(int, int, int, int, const float *, const float *)
	{
		return nullptr;
	}

	Fp8GemmPlan * fp8_gemm_plan_create_ex(const Fp8GemmSpec &, const float *, const float *)
	{
		return nullptr;
	}

	void fp8_gemm_plan_destroy(Fp8GemmPlan *)
	{
	}

		bool fp8_gemm_output_is_column_major(const Fp8GemmPlan *)
		{
			return false;
		}

		bool fp8_gemm_output_is_fp32(const Fp8GemmPlan *)
		{
			return false;
		}

		size_t fp8_gemm_output_element_bytes(const Fp8GemmPlan *)
		{
			return 0;
		}

		bool fp8_gemm(Fp8GemmPlan *, const void *, const void *, void *, void *, size_t, float)
		{
			return false;
		}
}

#endif
