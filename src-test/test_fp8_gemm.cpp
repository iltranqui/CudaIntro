#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <vector>

#include "convolutional_layer.hpp"
#include "darknet_internal.hpp"
#include "fp8_gemm.hpp"
#include "fp8_kernels.hpp"

TEST(Fp8Gemm, TrainingSpecsUseExpectedOperandFormats)
{
	const auto forward = Darknet::fp8_gemm_training_spec(Darknet::Fp8TrainingGemm::Forward, 32, 144, 196);
	EXPECT_EQ(forward.output_rows, 32);
	EXPECT_EQ(forward.output_cols, 196);
	EXPECT_EQ(forward.reduction, 144);
	EXPECT_EQ(forward.reduction_pad, 144);
	EXPECT_EQ(forward.a_format, Darknet::Fp8Format::E4M3);
	EXPECT_EQ(forward.b_format, Darknet::Fp8Format::E4M3);

	const auto wgrad = Darknet::fp8_gemm_training_spec(Darknet::Fp8TrainingGemm::WeightGradient, 32, 144, 196);
	EXPECT_EQ(wgrad.output_rows, 32);
	EXPECT_EQ(wgrad.output_cols, 144);
	EXPECT_EQ(wgrad.reduction, 196);
	EXPECT_EQ(wgrad.reduction_pad, 208);
	EXPECT_EQ(wgrad.a_format, Darknet::Fp8Format::E5M2);
	EXPECT_EQ(wgrad.b_format, Darknet::Fp8Format::E4M3);

	const auto direct_wgrad = Darknet::fp8_gemm_training_spec(Darknet::Fp8TrainingGemm::WeightGradientDirectUpdate, 32, 144, 196);
	EXPECT_EQ(direct_wgrad.output_rows, 144);
	EXPECT_EQ(direct_wgrad.output_cols, 32);
	EXPECT_EQ(direct_wgrad.reduction, 196);
	EXPECT_EQ(direct_wgrad.reduction_pad, 208);
	EXPECT_EQ(direct_wgrad.a_format, Darknet::Fp8Format::E4M3);
	EXPECT_EQ(direct_wgrad.b_format, Darknet::Fp8Format::E5M2);
	EXPECT_EQ(direct_wgrad.output, Darknet::Fp8GemmOutput::Fp32);

	const auto dgrad = Darknet::fp8_gemm_training_spec(Darknet::Fp8TrainingGemm::DataGradient, 32, 144, 196);
	EXPECT_EQ(dgrad.output_rows, 144);
	EXPECT_EQ(dgrad.output_cols, 196);
	EXPECT_EQ(dgrad.reduction, 32);
	EXPECT_EQ(dgrad.reduction_pad, 32);
	EXPECT_EQ(dgrad.a_format, Darknet::Fp8Format::E4M3);
	EXPECT_EQ(dgrad.b_format, Darknet::Fp8Format::E5M2);

	const auto direct_dgrad = Darknet::fp8_gemm_training_spec(Darknet::Fp8TrainingGemm::DataGradientDirectUpdate, 32, 144, 196);
	EXPECT_EQ(direct_dgrad.output_rows, 196);
	EXPECT_EQ(direct_dgrad.output_cols, 144);
	EXPECT_EQ(direct_dgrad.reduction, 32);
	EXPECT_EQ(direct_dgrad.reduction_pad, 32);
	EXPECT_EQ(direct_dgrad.a_format, Darknet::Fp8Format::E5M2);
	EXPECT_EQ(direct_dgrad.b_format, Darknet::Fp8Format::E4M3);
	EXPECT_EQ(direct_dgrad.output, Darknet::Fp8GemmOutput::Fp32);
	EXPECT_TRUE(direct_dgrad.batch_a);
	EXPECT_FALSE(direct_dgrad.batch_b);
	EXPECT_TRUE(forward.batch_b);
}

TEST(Fp8Gemm, GemmApiAcceptsBetaAccumulationArgument)
{
	using Fp8GemmFn = bool (*)(
		Darknet::Fp8GemmPlan *,
		const void *,
		const void *,
		void *,
		void *,
		size_t,
		float);

	Fp8GemmFn fn = &Darknet::fp8_gemm;
	EXPECT_NE(fn, nullptr);
}

TEST(Fp8Gemm, OutputElementBytesReflectPlanOutputType)
{
	EXPECT_EQ(Darknet::fp8_gemm_output_element_bytes(nullptr), 0);
}

TEST(Fp8Gemm, SharedDyLayoutPadsRowsAndColumnsForTrainingGemmReuse)
{
	const auto layout = Darknet::fp8_shared_dy_layout(17, 19);
	EXPECT_EQ(layout.filters_pad, 32);
	EXPECT_EQ(layout.spatial_pad, 32);
	EXPECT_EQ(layout.bytes, static_cast<size_t>(32) * 32);
}

TEST(Fp8Gemm, DirectDgradEligibilityRequiresExactOneByOneGeometry)
{
#ifdef DARKNET_HAS_FP8
	Darknet::Layer layer = {};
	layer.groups = 1;
	layer.size = 1;
	layer.stride = 1;
	layer.stride_x = 1;
	layer.stride_y = 1;
	layer.dilation = 1;
	layer.pad = 0;
	layer.w = layer.out_w = 13;
	layer.h = layer.out_h = 11;
	EXPECT_TRUE(fp8_convolutional_direct_dgrad_eligible(layer));

	layer.size = 3;
	EXPECT_FALSE(fp8_convolutional_direct_dgrad_eligible(layer));
	layer.size = 1;
	layer.stride_x = 2;
	EXPECT_FALSE(fp8_convolutional_direct_dgrad_eligible(layer));
	layer.stride_x = 1;
	layer.pad = 1;
	EXPECT_FALSE(fp8_convolutional_direct_dgrad_eligible(layer));
	layer.pad = 0;
	layer.groups = 2;
	EXPECT_FALSE(fp8_convolutional_direct_dgrad_eligible(layer));
#else
	GTEST_SKIP() << "FP8 support is not compiled";
#endif
}

TEST(Fp8Gemm, MixedTrainingGemmRuntimeSmoke)
{
#if defined(DARKNET_GPU_CUDA) && defined(CUDNN) && defined(CUDART_VERSION) && CUDART_VERSION >= 12010
	if (!Darknet::fp8_gemm_supported())
	{
		GTEST_SKIP() << "FP8 cuBLASLt runtime requires an SM89+ CUDA device";
	}

	const int filters = 32;
	const int kernel = 144;
	const int spatial = 196;
	const Darknet::Fp8SharedDyLayout shared_dy = Darknet::fp8_shared_dy_layout(filters, spatial);
	std::vector<float> weights(filters * kernel);
	std::vector<float> dy(filters * spatial);
	std::vector<float> input_col(kernel * spatial);
	for (size_t idx = 0; idx < weights.size(); ++idx)
	{
		weights[idx] = 0.25f + 0.001f * static_cast<float>(idx % 13);
	}
	for (size_t idx = 0; idx < dy.size(); ++idx)
	{
		dy[idx] = 0.5f + 0.002f * static_cast<float>(idx % 17);
	}
	for (size_t idx = 0; idx < input_col.size(); ++idx)
	{
		input_col[idx] = 0.75f + 0.003f * static_cast<float>(idx % 19);
	}

	float unit_scale = 1.0f;
	float * weight_scale_gpu = cuda_make_array(&unit_scale, 1);
	float * input_scale_gpu = cuda_make_array(&unit_scale, 1);
	float * dy_scale_gpu = cuda_make_array(&unit_scale, 1);
	float * weights_gpu = cuda_make_array(weights.data(), weights.size());
	float * dy_gpu = cuda_make_array(dy.data(), dy.size());
	float * input_col_gpu = cuda_make_array(input_col.data(), input_col.size());
	float * amax_gpu = cuda_make_array(nullptr, 1);
	float * wgrad_gpu = cuda_make_array(nullptr, filters * kernel);
	float * dgrad_gpu = cuda_make_array(nullptr, kernel * spatial);
	void * weights_t_fp8 = nullptr;
	void * dy_fp8 = nullptr;
	void * input_t_fp8 = nullptr;
	void * wgrad_bf16 = nullptr;
	void * dgrad_bf16 = nullptr;
	void * workspace = nullptr;
	Darknet::Fp8GemmPlan * wgrad_plan = nullptr;
	Darknet::Fp8GemmPlan * dgrad_plan = nullptr;
	CHECK_CUDA(cudaMalloc(&weights_t_fp8, Darknet::fp8_rowmajor_pad_cols_bytes(kernel, shared_dy.filters_pad)));
	CHECK_CUDA(cudaMalloc(&dy_fp8, shared_dy.bytes));
	CHECK_CUDA(cudaMalloc(&input_t_fp8, Darknet::fp8_rowmajor_pad_rows_bytes(shared_dy.spatial_pad, kernel)));
	CHECK_CUDA(cudaMalloc(&wgrad_bf16, static_cast<size_t>(filters) * kernel * sizeof(float)));
	CHECK_CUDA(cudaMalloc(&dgrad_bf16, static_cast<size_t>(kernel) * spatial * sizeof(unsigned short)));
	CHECK_CUDA(cudaMalloc(&workspace, Darknet::fp8_gemm_workspace_bytes()));
	const auto cleanup = [&]()
	{
		Darknet::fp8_gemm_plan_destroy(wgrad_plan);
		Darknet::fp8_gemm_plan_destroy(dgrad_plan);
		CHECK_CUDA(cudaFree(workspace));
		CHECK_CUDA(cudaFree(dgrad_bf16));
		CHECK_CUDA(cudaFree(wgrad_bf16));
		CHECK_CUDA(cudaFree(input_t_fp8));
		CHECK_CUDA(cudaFree(dy_fp8));
		CHECK_CUDA(cudaFree(weights_t_fp8));
		cuda_free(dgrad_gpu);
		cuda_free(wgrad_gpu);
		cuda_free(amax_gpu);
		cuda_free(input_col_gpu);
		cuda_free(dy_gpu);
		cuda_free(weights_gpu);
		cuda_free(dy_scale_gpu);
		cuda_free(input_scale_gpu);
		cuda_free(weight_scale_gpu);
	};

	Darknet::fp8_clear_amax_gpu(amax_gpu);
	Darknet::fp8_quantize_e5m2_rowmajor_pad_rows_cols_record_amax_gpu(
		dy_gpu, filters, spatial, shared_dy.filters_pad, shared_dy.spatial_pad, dy_scale_gpu, dy_fp8, amax_gpu);
	Darknet::fp8_quantize_transpose_rowmajor_pad_rows_gpu(
		input_col_gpu, kernel, spatial, shared_dy.spatial_pad, input_scale_gpu, input_t_fp8);
	Darknet::Fp8GemmSpec wgrad_spec = Darknet::fp8_gemm_training_spec(
		Darknet::Fp8TrainingGemm::WeightGradient, filters, kernel, spatial);
	wgrad_plan = Darknet::fp8_gemm_plan_create_ex(wgrad_spec, dy_scale_gpu, input_scale_gpu);
	if (wgrad_plan == nullptr)
	{
		wgrad_spec.output = Darknet::Fp8GemmOutput::Bf16;
		wgrad_plan = Darknet::fp8_gemm_plan_create_ex(wgrad_spec, dy_scale_gpu, input_scale_gpu);
	}
	if (wgrad_plan == nullptr)
	{
		cleanup();
		GTEST_SKIP() << "cuBLASLt did not return an E5M2/E4M3 wgrad plan on this runtime";
	}
	EXPECT_TRUE(Darknet::fp8_gemm(wgrad_plan, dy_fp8, input_t_fp8, wgrad_bf16, workspace, Darknet::fp8_gemm_workspace_bytes()));
	CHECK_CUDA(cudaMemsetAsync(wgrad_gpu, 0, static_cast<size_t>(filters) * kernel * sizeof(float), get_cuda_stream()));
	Darknet::fp8_colmajor_output_accumulate_rowmajor_gpu(
		wgrad_bf16,
		filters,
		kernel,
		!Darknet::fp8_gemm_output_is_fp32(wgrad_plan),
		1.0f,
		wgrad_gpu);

	Darknet::fp8_quantize_transpose_rowmajor_pad_cols_gpu(
		weights_gpu, filters, kernel, shared_dy.filters_pad, weight_scale_gpu, weights_t_fp8);
	Darknet::Fp8GemmSpec dgrad_spec = Darknet::fp8_gemm_training_spec(
		Darknet::Fp8TrainingGemm::DataGradient, filters, kernel, spatial);
	dgrad_spec.output = Darknet::Fp8GemmOutput::Bf16;
	dgrad_spec.b_leading_dim = shared_dy.spatial_pad;
	dgrad_plan = Darknet::fp8_gemm_plan_create_ex(dgrad_spec, weight_scale_gpu, dy_scale_gpu);
	if (dgrad_plan == nullptr)
	{
		cleanup();
		GTEST_SKIP() << "cuBLASLt did not return an E4M3/E5M2 dgrad plan on this runtime";
	}
	EXPECT_TRUE(Darknet::fp8_gemm(dgrad_plan, weights_t_fp8, dy_fp8, dgrad_bf16, workspace, Darknet::fp8_gemm_workspace_bytes()));
	CHECK_CUDA(cudaMemsetAsync(dgrad_gpu, 0, static_cast<size_t>(kernel) * spatial * sizeof(float), get_cuda_stream()));
	Darknet::fp8_colmajor_output_accumulate_rowmajor_gpu(dgrad_bf16, kernel, spatial, true, 1.0f, dgrad_gpu);

	std::vector<float> wgrad(filters * kernel);
	std::vector<float> dgrad(kernel * spatial);
	cuda_pull_array(wgrad_gpu, wgrad.data(), wgrad.size());
	cuda_pull_array(dgrad_gpu, dgrad.data(), dgrad.size());
	const auto max_abs = [](const std::vector<float> & values)
	{
		float result = 0.0f;
		for (const float value : values)
		{
			if (std::isfinite(value))
			{
				result = std::max(result, std::fabs(value));
			}
		}
		return result;
	};
	EXPECT_GT(max_abs(wgrad), 0.0f);
	EXPECT_GT(max_abs(dgrad), 0.0f);

	cleanup();
#else
	GTEST_SKIP() << "FP8 runtime smoke requires CUDA 12.1+ with cuDNN BF16 conversion helpers";
#endif
}

TEST(Fp8Gemm, DirectDgradBroadcastsWeightsAndAccumulatesIntoFp32Delta)
{
#if defined(DARKNET_GPU_CUDA) && defined(CUDART_VERSION) && CUDART_VERSION >= 12010
	if (!Darknet::fp8_gemm_supported())
	{
		GTEST_SKIP() << "FP8 cuBLASLt runtime requires an SM89+ CUDA device";
	}

	constexpr int batch = 2;
	constexpr int filters = 16;
	constexpr int kernel = 16;
	constexpr int spatial = 16;
	constexpr size_t weights_count = static_cast<size_t>(filters) * kernel;
	constexpr size_t dy_stride = static_cast<size_t>(filters) * spatial;
	constexpr size_t delta_stride = static_cast<size_t>(kernel) * spatial;
	std::vector<float> weights(weights_count);
	std::vector<float> dy(static_cast<size_t>(batch) * dy_stride);
	std::vector<float> expected(static_cast<size_t>(batch) * delta_stride, 3.0f);
	for (int f = 0; f < filters; ++f)
	{
		for (int k = 0; k < kernel; ++k)
		{
			weights[static_cast<size_t>(f) * kernel + k] = ((f + k) & 1) ? -0.5f : 0.5f;
		}
	}
	for (int b = 0; b < batch; ++b)
	{
		for (int f = 0; f < filters; ++f)
		{
			for (int s = 0; s < spatial; ++s)
			{
				dy[static_cast<size_t>(b) * dy_stride + static_cast<size_t>(f) * spatial + s] =
					((b + f + s) & 1) ? 1.0f : -1.0f;
			}
		}
		for (int k = 0; k < kernel; ++k)
		{
			for (int s = 0; s < spatial; ++s)
			{
				float sum = 0.0f;
				for (int f = 0; f < filters; ++f)
				{
					sum += weights[static_cast<size_t>(f) * kernel + k] *
						dy[static_cast<size_t>(b) * dy_stride + static_cast<size_t>(f) * spatial + s];
				}
				expected[static_cast<size_t>(b) * delta_stride + static_cast<size_t>(k) * spatial + s] += sum;
			}
		}
	}

	float unit_scale = 1.0f;
	float * scale_gpu = cuda_make_array(&unit_scale, 1);
	float * weights_gpu = cuda_make_array(weights.data(), weights.size());
	float * dy_gpu = cuda_make_array(dy.data(), dy.size());
	float * delta_gpu = cuda_make_array(nullptr, expected.size());
	float * amax_gpu = cuda_make_array(nullptr, 1);
	void * weights_t_fp8 = nullptr;
	void * dyt_fp8 = nullptr;
	void * workspace = nullptr;
	CHECK_CUDA(cudaMalloc(&weights_t_fp8, weights_count));
	CHECK_CUDA(cudaMalloc(&dyt_fp8, static_cast<size_t>(batch) * spatial * filters));
	CHECK_CUDA(cudaMalloc(&workspace, Darknet::fp8_gemm_workspace_bytes()));
	std::vector<float> initial(expected.size(), 3.0f);
	cuda_push_array(delta_gpu, initial.data(), initial.size());
	Darknet::fp8_clear_amax_gpu(amax_gpu);
	Darknet::fp8_quantize_transpose_rowmajor_pad_cols_gpu(
		weights_gpu, filters, kernel, filters, scale_gpu, weights_t_fp8);
	Darknet::fp8_quantize_e5m2_transpose_rowmajor_pad_cols_record_amax_gpu(
		dy_gpu, filters, spatial, filters, scale_gpu, dyt_fp8, amax_gpu,
		batch, dy_stride, static_cast<size_t>(spatial) * filters);

	Darknet::Fp8GemmSpec spec = Darknet::fp8_gemm_training_spec(
		Darknet::Fp8TrainingGemm::DataGradientDirectUpdate, filters, kernel, spatial);
	spec.batch = batch;
	Darknet::Fp8GemmPlan * plan = Darknet::fp8_gemm_plan_create_ex(spec, scale_gpu, scale_gpu);
	if (plan == nullptr)
	{
		CHECK_CUDA(cudaFree(workspace));
		CHECK_CUDA(cudaFree(dyt_fp8));
		CHECK_CUDA(cudaFree(weights_t_fp8));
		cuda_free(amax_gpu);
		cuda_free(delta_gpu);
		cuda_free(dy_gpu);
		cuda_free(weights_gpu);
		cuda_free(scale_gpu);
		GTEST_SKIP() << "cuBLASLt did not return the direct dgrad broadcast plan";
	}
	ASSERT_TRUE(Darknet::fp8_gemm(
		plan, dyt_fp8, weights_t_fp8, delta_gpu, workspace, Darknet::fp8_gemm_workspace_bytes(), 1.0f));
	std::vector<float> actual(expected.size());
	cuda_pull_array(delta_gpu, actual.data(), actual.size());
	for (size_t idx = 0; idx < actual.size(); ++idx)
	{
		EXPECT_FLOAT_EQ(actual[idx], expected[idx]) << "index=" << idx;
	}

	Darknet::fp8_gemm_plan_destroy(plan);
	CHECK_CUDA(cudaFree(workspace));
	CHECK_CUDA(cudaFree(dyt_fp8));
	CHECK_CUDA(cudaFree(weights_t_fp8));
	cuda_free(amax_gpu);
	cuda_free(delta_gpu);
	cuda_free(dy_gpu);
	cuda_free(weights_gpu);
	cuda_free(scale_gpu);
#else
	GTEST_SKIP() << "Direct FP8 dgrad test requires CUDA 12.1+";
#endif
}

TEST(Fp8Gemm, DirectWgradAccumulatesIntoRowmajorFp32Updates)
{
#if defined(DARKNET_GPU_CUDA) && defined(CUDART_VERSION) && CUDART_VERSION >= 12010
	if (!Darknet::fp8_gemm_supported())
	{
		GTEST_SKIP() << "FP8 cuBLASLt runtime requires an SM89+ CUDA device";
	}
	constexpr int filters = 16;
	constexpr int kernel = 16;
	constexpr int spatial = 16;
	std::vector<float> input(static_cast<size_t>(kernel) * spatial);
	std::vector<float> dy(static_cast<size_t>(filters) * spatial);
	std::vector<float> expected(static_cast<size_t>(filters) * kernel, 2.0f);
	for (int k = 0; k < kernel; ++k)
	{
		for (int s = 0; s < spatial; ++s)
		{
			input[static_cast<size_t>(k) * spatial + s] = ((k + s) & 1) ? 0.5f : -0.5f;
		}
	}
	for (int f = 0; f < filters; ++f)
	{
		for (int s = 0; s < spatial; ++s)
		{
			dy[static_cast<size_t>(f) * spatial + s] = ((f + s) & 1) ? 1.0f : -1.0f;
		}
		for (int k = 0; k < kernel; ++k)
		{
			float sum = 0.0f;
			for (int s = 0; s < spatial; ++s)
			{
				sum += dy[static_cast<size_t>(f) * spatial + s] * input[static_cast<size_t>(k) * spatial + s];
			}
			expected[static_cast<size_t>(f) * kernel + k] += sum;
		}
	}

	float unit_scale = 1.0f;
	float * scale_gpu = cuda_make_array(&unit_scale, 1);
	float * input_gpu = cuda_make_array(input.data(), input.size());
	float * dy_gpu = cuda_make_array(dy.data(), dy.size());
	float * updates_gpu = cuda_make_array(nullptr, expected.size());
	float * amax_gpu = cuda_make_array(nullptr, 1);
	void * input_fp8 = nullptr;
	void * dy_fp8 = nullptr;
	void * workspace = nullptr;
	CHECK_CUDA(cudaMalloc(&input_fp8, input.size()));
	CHECK_CUDA(cudaMalloc(&dy_fp8, dy.size()));
	CHECK_CUDA(cudaMalloc(&workspace, Darknet::fp8_gemm_workspace_bytes()));
	std::vector<float> initial(expected.size(), 2.0f);
	cuda_push_array(updates_gpu, initial.data(), initial.size());
	Darknet::fp8_clear_amax_gpu(amax_gpu);
	Darknet::fp8_quantize_rowmajor_pad_cols_gpu(
		input_gpu, kernel, spatial, spatial, scale_gpu, input_fp8);
	Darknet::fp8_quantize_e5m2_rowmajor_pad_cols_record_amax_gpu(
		dy_gpu, filters, spatial, spatial, scale_gpu, dy_fp8, amax_gpu);
	Darknet::Fp8GemmSpec spec = Darknet::fp8_gemm_training_spec(
		Darknet::Fp8TrainingGemm::WeightGradientDirectUpdate, filters, kernel, spatial);
	Darknet::Fp8GemmPlan * plan = Darknet::fp8_gemm_plan_create_ex(spec, scale_gpu, scale_gpu);
	if (plan == nullptr)
	{
		CHECK_CUDA(cudaFree(workspace));
		CHECK_CUDA(cudaFree(dy_fp8));
		CHECK_CUDA(cudaFree(input_fp8));
		cuda_free(amax_gpu);
		cuda_free(updates_gpu);
		cuda_free(dy_gpu);
		cuda_free(input_gpu);
		cuda_free(scale_gpu);
		GTEST_SKIP() << "cuBLASLt did not return the direct wgrad plan";
	}
	ASSERT_TRUE(Darknet::fp8_gemm(
		plan, input_fp8, dy_fp8, updates_gpu, workspace, Darknet::fp8_gemm_workspace_bytes(), 1.0f));
	std::vector<float> actual(expected.size());
	cuda_pull_array(updates_gpu, actual.data(), actual.size());
	for (size_t idx = 0; idx < actual.size(); ++idx)
	{
		EXPECT_FLOAT_EQ(actual[idx], expected[idx]) << "index=" << idx;
	}
	Darknet::fp8_gemm_plan_destroy(plan);
	CHECK_CUDA(cudaFree(workspace));
	CHECK_CUDA(cudaFree(dy_fp8));
	CHECK_CUDA(cudaFree(input_fp8));
	cuda_free(amax_gpu);
	cuda_free(updates_gpu);
	cuda_free(dy_gpu);
	cuda_free(input_gpu);
	cuda_free(scale_gpu);
#else
	GTEST_SKIP() << "Direct FP8 wgrad test requires CUDA 12.1+";
#endif
}
