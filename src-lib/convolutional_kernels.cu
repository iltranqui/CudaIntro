#include "darknet_internal.hpp"
#include "gemm.hpp"
#include "col2im.hpp"
#include "im2col.hpp"
#include "convolution_precision_route.hpp"
#ifdef DARKNET_HAS_FP4
#include "fp4_gemm.hpp"
#include "fp4_kernels.hpp"
#endif
#ifdef DARKNET_HAS_FP8
#include "fp8_conv.hpp"
#include "fp8_gemm.hpp"
#include "fp8_kernels.hpp"
#include "fp8_scaling.hpp"
#endif

#ifdef DARKNET_GPU_CUDA
#include <cuda_bf16.h>
#endif

#include <algorithm>
#include <cstdlib>


namespace
{
	static auto & cfg_and_state = Darknet::CfgAndState::get();

#ifdef DARKNET_HAS_FP8
	static size_t fp8_align_workspace_offset(const size_t value)
	{
		constexpr size_t alignment = 256;
		return (value + alignment - 1) & ~(alignment - 1);
	}

	static bool fp8_env_is_set(const char * name)
	{
		const char * const value = std::getenv(name);
		return value != nullptr && value[0] != '\0' && !(value[0] == '0' && value[1] == '\0');
	}

	// src/dst hold `batch` contiguous (rows x cols) matrices; each is transposed col-major -> row-major.
	// bias (optional, length rows) is fused in so the separate add_bias pass over the output disappears.
	__global__ void fp8_f32_colmajor_to_rowmajor_kernel(const float * src, int rows, int cols, int batch, const float * bias, float * dst)
	{
		const size_t matrix = static_cast<size_t>(rows) * cols;
		const size_t total = matrix * batch;
		const size_t index = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
		if (index >= total)
		{
			return;
		}

		const size_t b = index / matrix;
		const size_t rem = index - b * matrix;
		const int row = static_cast<int>(rem / cols);
		const int col = static_cast<int>(rem - static_cast<size_t>(row) * cols);
		float value = src[b * matrix + row + static_cast<size_t>(col) * rows];
		if (bias)
		{
			value += bias[row];
		}
		dst[index] = value;
	}

#ifdef DARKNET_GPU_CUDA
	__global__ void fp8_bf16_colmajor_to_f32_rowmajor_kernel(const __nv_bfloat16 * src, int rows, int cols, int batch, const float * bias, float * dst)
	{
		const size_t matrix = static_cast<size_t>(rows) * cols;
		const size_t total = matrix * batch;
		const size_t index = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
		if (index >= total)
		{
			return;
		}

		const size_t b = index / matrix;
		const size_t rem = index - b * matrix;
		const int row = static_cast<int>(rem / cols);
		const int col = static_cast<int>(rem - static_cast<size_t>(row) * cols);
		float value = __bfloat162float(src[b * matrix + row + static_cast<size_t>(col) * rows]);
		if (bias)
		{
			value += bias[row];
		}
		dst[index] = value;
	}

	__global__ void fp8_bf16_to_f32_kernel(const __nv_bfloat16 * src, size_t total, float * dst)
	{
		const size_t index = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
		if (index < total)
		{
			dst[index] = __bfloat162float(src[index]);
		}
	}
#endif

	static void fp8_f32_colmajor_to_rowmajor(const float * src, const int rows, const int cols, float * dst, const int batch = 1, const float * bias = nullptr)
	{
		const size_t total = static_cast<size_t>(rows) * cols * batch;
		fp8_f32_colmajor_to_rowmajor_kernel<<<cuda_gridsize(total), BLOCK, 0, get_cuda_stream()>>>(src, rows, cols, batch, bias, dst);
		CHECK_CUDA(cudaPeekAtLastError());
	}

	static void fp8_bf16_gemm_output_to_f32(
		const Darknet::Fp8GemmPlan * plan,
		void * src,
		const int rows,
		const int cols,
		float * dst,
		const int batch = 1,
		const float * bias = nullptr)
	{
#ifdef DARKNET_GPU_CUDA
		const size_t total = static_cast<size_t>(rows) * cols * batch;
		if (Darknet::fp8_gemm_output_is_column_major(plan))
		{
			fp8_bf16_colmajor_to_f32_rowmajor_kernel<<<cuda_gridsize(total), BLOCK, 0, get_cuda_stream()>>>(
				static_cast<const __nv_bfloat16 *>(src),
				rows,
				cols,
				batch,
				bias,
				dst);
		}
		else
		{
			fp8_bf16_to_f32_kernel<<<cuda_gridsize(total), BLOCK, 0, get_cuda_stream()>>>(
				static_cast<const __nv_bfloat16 *>(src),
				total,
				dst);
		}
		CHECK_CUDA(cudaPeekAtLastError());
#else
		(void)plan;
		(void)src;
		(void)rows;
		(void)cols;
		(void)dst;
		(void)batch;
#endif
	}

#endif

	static void check_nan_gpu(const char *step_name, float *d_arr, size_t size, int layer_idx)
	{
		if (d_arr == nullptr || size == 0)
		{
			return;
		}

		if (is_nan_or_inf(d_arr, size))
		{
			const std::string layer_label = Darknet::layer_type_diagnostic_label(Darknet::ELayerType::CONVOLUTIONAL);
			std::printf("[%s layer] NaN/Inf detected at layer %d, step: %s\n", layer_label.c_str(), layer_idx, step_name);
		}
	}

	struct ConvBackwardResult
	{
		bool wgrad_requested = false;
		bool wgrad_done = false;
		bool dgrad_requested = false;
		bool dgrad_done = false;
	};

#ifdef DARKNET_HAS_FP4
	static size_t fp4_align_workspace(const size_t value)
	{
		return (value + 255U) & ~size_t{255U};
	}

	static void fp4_make_im2col(const Darknet::Layer & l, const float * input, float * column)
	{
		im2col_gpu_ext(input, l.c, l.h, l.w, l.size, l.size,
			l.pad * l.dilation, l.pad * l.dilation, l.stride_y, l.stride_x,
			l.dilation, l.dilation, column);
	}

	static void fp4_update_convolutional_relay(Darknet::Layer & l, Darknet::NetworkState state)
	{
		if (state.train || !state.net.fp4_inference || l.fp4_relay_next_layer < 0 ||
			!l.fp4_relay_gpu || !l.fp4_relay_scales_gpu)
		{
			l.fp4_relay_valid = 0;
			return;
		}
		const bool fuse_relu = l.activation == RELU;
		l.fp4_relay_valid = (fuse_relu
			? Darknet::fp4_relu_quantize_nchw_to_cublaslt_gpu(
				l.output_gpu, l.batch, l.out_c, l.out_h, l.out_w,
				reinterpret_cast<uint8_t *>(l.fp4_relay_gpu),
				reinterpret_cast<uint8_t *>(l.fp4_relay_scales_gpu))
			: Darknet::fp4_quantize_nchw_to_cublaslt_gpu(
				l.output_gpu, l.batch, l.out_c, l.out_h, l.out_w,
				reinterpret_cast<uint8_t *>(l.fp4_relay_gpu),
				reinterpret_cast<uint8_t *>(l.fp4_relay_scales_gpu))) ? 1 : 0;
	}

	static bool forward_convolutional_layer_gpu_fp4(Darknet::Layer & l, Darknet::NetworkState state)
	{
		if (state.net.fp4_calibrating && l.fp4_amax_gpu)
		{
			Darknet::fp4_accumulate_amax_gpu(state.input, static_cast<size_t>(l.inputs) * l.batch, l.fp4_amax_gpu);
		}

		// FP32 remains authoritative.  im2col converts the convolution input to
		// GEMM shape, then fp4_gemm_execute derives FP4 values/scales internally.
		// Results stay staged until the complete batch succeeds, so FP8/cuDNN can
		// safely retry without observing a partially committed FP4 output.
		const bool requested = state.train ? state.net.fp4_training : state.net.fp4_inference;
		if (!requested || !l.fp4_eligible || !l.fp4_gemm_plan || !state.workspace) return false;
		const int filters = l.n;
		const int kernel = l.c * l.size * l.size;
		const int spatial = l.out_h * l.out_w;
		const size_t matrix = static_cast<size_t>(kernel) * spatial;
		const size_t output_matrix = static_cast<size_t>(filters) * spatial;
		char * const base = reinterpret_cast<char *>(state.workspace);
		float * const column = reinterpret_cast<float *>(base);
		float * const input_t = column + matrix;
		float * const staged_output = input_t + matrix;
		const size_t backend_offset = fp4_align_workspace(
			reinterpret_cast<char *>(staged_output + output_matrix * l.batch) - base);
		void * const backend_workspace = base + backend_offset;
		const size_t backend_bytes = l.fp4_workspace_size > backend_offset ? l.fp4_workspace_size - backend_offset : 0;
		auto * const plan = static_cast<Darknet::Fp4GemmPlan *>(l.fp4_gemm_plan);

		bool used_relay = false;
		if (!state.train && state.index > 0 && l.fp4_relay_source_layer == state.index - 1)
		{
			Darknet::Layer & producer = state.net.layers[state.index - 1];
			if (producer.fp4_relay_next_layer == state.index && producer.fp4_relay_valid &&
				producer.fp4_relay_gpu && producer.fp4_relay_scales_gpu &&
				producer.fp4_relay_packed_bytes > 0 && producer.fp4_relay_scale_bytes > 0)
			{
				used_relay = true;
				for (int b = 0; b < l.batch; ++b)
				{
					const auto * const packed = reinterpret_cast<const uint8_t *>(producer.fp4_relay_gpu) +
						static_cast<size_t>(b) * producer.fp4_relay_packed_bytes;
					const auto * const scales = reinterpret_cast<const uint8_t *>(producer.fp4_relay_scales_gpu) +
						static_cast<size_t>(b) * producer.fp4_relay_scale_bytes;
					if (!Darknet::fp4_gemm_execute_prequantized_right(plan, packed, scales,
						staged_output + static_cast<size_t>(b) * output_matrix, backend_workspace, backend_bytes))
					{
						used_relay = false;
						break;
					}
				}
			}
		}

		if (!used_relay) for (int b = 0; b < l.batch; ++b)
		{
			const float * const input = state.input + static_cast<size_t>(b) * l.c * l.h * l.w;
			fp4_make_im2col(l, input, column);
			Darknet::fp4_transpose_rowmajor_gpu(column, kernel, spatial, input_t);
			if (!Darknet::fp4_gemm_execute(plan, l.weights_gpu, input_t,
				staged_output + static_cast<size_t>(b) * output_matrix, backend_workspace, backend_bytes))
			{
				return false;
			}
		}
		simple_copy_ongpu(output_matrix * l.batch, staged_output, l.output_gpu);
		if (!l.batch_normalize) add_bias_gpu(l.output_gpu, l.biases_gpu, l.batch, filters, spatial);
		else forward_batchnorm_layer_gpu(l, state);
		return true;
	}

	static ConvBackwardResult backward_convolutional_layer_gpu_fp4(Darknet::Layer & l, Darknet::NetworkState state)
	{
		// Weight and data gradients are separate GEMMs and therefore separate
		// transactions.  A successful direction is accumulated exactly once;
		// only unfinished directions continue through FP8 and regular cuDNN.
		ConvBackwardResult result;
		result.wgrad_requested = !state.net.adversarial && !l.train_only_bn;
		result.dgrad_requested = state.delta != nullptr;
		if (!state.net.fp4_training || !l.fp4_train_eligible || !state.workspace) return result;
		const int filters = l.n;
		const int kernel = l.c * l.size * l.size;
		const int spatial = l.out_h * l.out_w;
		const int reduction = l.batch * spatial;
		const size_t col_matrix = static_cast<size_t>(kernel) * spatial;
		char * const base = reinterpret_cast<char *>(state.workspace);

		if (result.wgrad_requested && l.fp4_wgrad_gemm_plan)
		{
			float * const dy = reinterpret_cast<float *>(base);
			float * const input = dy + static_cast<size_t>(filters) * reduction;
			float * const column = input + static_cast<size_t>(kernel) * reduction;
			float * const output = column + col_matrix;
			const size_t backend_offset = fp4_align_workspace(reinterpret_cast<char *>(output + static_cast<size_t>(filters) * kernel) - base);
			Darknet::fp4_pack_batch_rows_gpu(l.delta_gpu, l.batch, filters, spatial, dy);
			for (int b = 0; b < l.batch; ++b)
			{
				fp4_make_im2col(l, state.input + static_cast<size_t>(b) * l.c * l.h * l.w, column);
				Darknet::fp4_copy_matrix_columns_gpu(column, kernel, spatial, reduction, b * spatial, input);
			}
			if (Darknet::fp4_gemm_execute(static_cast<Darknet::Fp4GemmPlan *>(l.fp4_wgrad_gemm_plan),
				dy, input, output, base + backend_offset,
				l.fp4_workspace_size > backend_offset ? l.fp4_workspace_size - backend_offset : 0))
			{
				axpy_ongpu(l.nweights, 1.0f, output, 1, l.weight_updates_gpu, 1);
				result.wgrad_done = true;
			}
		}

		if (result.dgrad_requested && l.fp4_dgrad_gemm_plan)
		{
			float * const weights_t = reinterpret_cast<float *>(base);
			float * const dy_t = weights_t + static_cast<size_t>(kernel) * filters;
			float * const staged = dy_t + static_cast<size_t>(spatial) * filters;
			float * const input_tmp = staged + col_matrix * l.batch;
			const size_t backend_offset = fp4_align_workspace(reinterpret_cast<char *>(input_tmp + static_cast<size_t>(l.c) * l.h * l.w) - base);
			Darknet::fp4_transpose_rowmajor_gpu(l.weights_gpu, filters, kernel, weights_t);
			bool success = true;
			for (int b = 0; b < l.batch && success; ++b)
			{
				Darknet::fp4_transpose_rowmajor_gpu(l.delta_gpu + static_cast<size_t>(b) * filters * spatial, filters, spatial, dy_t);
				success = Darknet::fp4_gemm_execute(static_cast<Darknet::Fp4GemmPlan *>(l.fp4_dgrad_gemm_plan),
					weights_t, dy_t, staged + static_cast<size_t>(b) * col_matrix, base + backend_offset,
					l.fp4_workspace_size > backend_offset ? l.fp4_workspace_size - backend_offset : 0);
			}
			if (success)
			{
				const bool direct = l.size == 1 && l.stride_x == 1 && l.stride_y == 1 && l.dilation == 1 && l.pad == 0;
				for (int b = 0; b < l.batch; ++b)
				{
					float * const delta = state.delta + static_cast<size_t>(b) * l.c * l.h * l.w;
					const float * const col = staged + static_cast<size_t>(b) * col_matrix;
					if (direct) axpy_ongpu(kernel * spatial, 1.0f, const_cast<float *>(col), 1, delta, 1);
					else
					{
						col2im_gpu_ext(col, l.c, l.h, l.w, l.size, l.size,
							l.pad * l.dilation, l.pad * l.dilation, l.stride_y, l.stride_x,
							l.dilation, l.dilation, input_tmp);
						axpy_ongpu(l.c * l.h * l.w, 1.0f, input_tmp, 1, delta, 1);
					}
				}
				result.dgrad_done = true;
			}
		}
		return result;
	}
#else
	static bool forward_convolutional_layer_gpu_fp4(Darknet::Layer &, Darknet::NetworkState) { return false; }
	static ConvBackwardResult backward_convolutional_layer_gpu_fp4(Darknet::Layer & l, Darknet::NetworkState state)
	{
		return {!state.net.adversarial && !l.train_only_bn, false, state.delta != nullptr, false};
	}
#endif

	static void fp8_update_convolutional_relay(Darknet::Layer & l, Darknet::NetworkState state)
	{
#ifndef DARKNET_HAS_FP8
		(void)l;
		(void)state;
#else
		if (state.train || !state.net.fp8_inference || l.fp8_relay_next_layer < 0 ||
			!l.fp8_relay_gpu || !l.fp8_relay_amax_gpu)
		{
			l.fp8_relay_valid = 0;
			return;
		}
		Darknet::Layer & consumer = state.net.layers[l.fp8_relay_next_layer];
		if (!consumer.fp8_input_scale_gpu)
		{
			l.fp8_relay_valid = 0;
			return;
		}

		const bool record_amax = state.net.fp8_activation_calibration_pending != 0;
		bool ok = false;
		// Fold the last ReLU and FP4-style persistent pack when the cuDNN graph
		// could not fuse that ReLU itself.  Other activations keep their exact
		// existing Darknet epilogue, followed by a single E4M3 layout conversion.
		if (l.activation == RELU && !l.fp8_graph_activation_fused)
		{
			ok = Darknet::fp8_relu_quantize_nchw_to_nhwc_gpu(
				l.output_gpu, l.batch, l.out_c, l.out_h, l.out_w,
				consumer.fp8_input_scale_gpu, l.fp8_relay_gpu,
				record_amax ? l.fp8_relay_amax_gpu : nullptr);
		}
		else if (record_amax)
		{
			Darknet::fp8_quantize_nchw_to_nhwc_record_amax_gpu(
				l.output_gpu, l.batch, l.out_c, l.out_h, l.out_w,
				consumer.fp8_input_scale_gpu, l.fp8_relay_gpu, l.fp8_relay_amax_gpu);
			ok = true;
		}
		else
		{
			Darknet::fp8_quantize_nchw_to_nhwc_gpu(
				l.output_gpu, l.batch, l.out_c, l.out_h, l.out_w,
				consumer.fp8_input_scale_gpu, l.fp8_relay_gpu);
			ok = true;
		}
		l.fp8_relay_valid = ok ? 1 : 0;
		if (ok)
		{
			state.net.fp8_activation_relay_executions += 1;
			if (l.fp8_graph_activation_fused)
			{
				state.net.fp8_graph_fused_executions += 1;
			}
		}
#endif
	}

	static void forward_convolutional_layer_gpu_epilogue(Darknet::Layer & l, Darknet::NetworkState state)
	{
		bool fp4_relu_relay = false;
#ifdef DARKNET_HAS_FP4
		fp4_relu_relay = !state.train && state.net.fp4_inference && l.fp4_relay_next_layer >= 0 &&
			l.fp4_relay_gpu != nullptr && l.fp4_relay_scales_gpu != nullptr;
#endif
		if (l.activation == SWISH) activate_array_swish_ongpu(l.output_gpu, l.outputs*l.batch, l.activation_input_gpu, l.output_gpu);
		else if (l.activation == MISH) activate_array_mish_ongpu(l.output_gpu, l.outputs*l.batch, l.activation_input_gpu, l.output_gpu);
		else if (l.activation == HARD_MISH) activate_array_hard_mish_ongpu(l.output_gpu, l.outputs*l.batch, l.activation_input_gpu, l.output_gpu);
		else if (l.activation == EML) activate_array_eml_ongpu(l.output_gpu, l.outputs*l.batch, l.activation_input_gpu, l.output_gpu);
		else if (l.activation == NORM_CHAN) activate_array_normalize_channels_ongpu(l.output_gpu, l.outputs*l.batch, l.batch, l.out_c, l.out_w*l.out_h, l.output_gpu);
		else if (l.activation == NORM_CHAN_SOFTMAX) activate_array_normalize_channels_softmax_ongpu(l.output_gpu, l.outputs*l.batch, l.batch, l.out_c, l.out_w*l.out_h, l.output_gpu, 0);
		else if (l.activation == NORM_CHAN_SOFTMAX_MAXVAL) activate_array_normalize_channels_softmax_ongpu(l.output_gpu, l.outputs*l.batch, l.batch, l.out_c, l.out_w*l.out_h, l.output_gpu, 1);
		else if (l.activation != LINEAR && !(l.activation == RELU && (l.fp8_graph_activation_fused || fp4_relu_relay))) activate_array_ongpu(l.output_gpu, l.outputs*l.batch, l.activation);
		//if(l.dot > 0) dot_error_gpu(l);
		if(l.binary || l.xnor) swap_binary(&l);
		//cudaDeviceSynchronize();    // for correct profiling of performance

		if (state.net.try_fix_nan)
		{
			fix_nan_and_inf(l.output_gpu, l.outputs*l.batch);
		}

		if (l.assisted_excitation && state.train)
		{
			assisted_excitation_forward_gpu(l, state);
		}

		if (l.antialiasing)
		{
			Darknet::NetworkState s = { 0 };
			s.train = state.train;
			s.workspace = state.workspace;
			s.net = state.net;
			if (!state.train) s.index = state.index;  // don't use TC for training (especially without cuda_convert_f32_to_f16() )
			s.input = l.output_gpu;
			forward_convolutional_layer_gpu(*(l.input_layer), s);
			simple_copy_ongpu(l.outputs*l.batch, l.output_gpu, l.input_antialiasing_gpu);
			simple_copy_ongpu(l.input_layer->outputs*l.input_layer->batch, l.input_layer->output_gpu, l.output_gpu);
		}

		if (l.coordconv)
		{
			coord_conv_gpu(l.output_gpu, l.outputs*l.batch, l.out_w, l.out_h, l.out_c, l.batch, 0);
		}

#ifdef DARKNET_HAS_FP4
		fp4_update_convolutional_relay(l, state);
#endif
		fp8_update_convolutional_relay(l, state);
	}

	static bool forward_convolutional_layer_gpu_fp8(Darknet::Layer & l, Darknet::NetworkState state)
	{
#ifndef DARKNET_HAS_FP8
		(void)l;
		(void)state;
		return false;
#else
		if (state.net.fp8_calibrating && l.fp8_amax_gpu)
		{
			Darknet::fp8_accumulate_amax_gpu(state.input, static_cast<size_t>(l.inputs) * l.batch, l.fp8_amax_gpu);
		}

		// The first-frame relay calibration intentionally evaluates the model on
		// the established BF16/FP32 path.  It measures real post-epilogue
		// activations without risking an unverified sidecar scale in an FP8
		// consumer, then the next frame uses only approved relays/plans.
		const bool use_fp8_inference = !state.train && state.net.fp8_inference && l.fp8_eligible &&
			!state.net.fp8_activation_calibration_pending;
		const bool use_fp8_training =
			state.train &&
			state.net.fp8_training &&
			get_current_iteration(state.net) >= state.net.fp8_warmup_iters &&
			l.fp8_train_eligible;
#ifndef CUDNN
		if (use_fp8_training)
		{
			return false;
		}
#endif
		if ((!use_fp8_inference && !use_fp8_training) ||
			l.fp8_input_scale_gpu == nullptr)
		{
			return false;
		}

		if (l.fp8_conv_fwd_plan && l.weights_fp8_nhwc_gpu)
		{
			auto * const conv_plan = static_cast<Darknet::Fp8ConvPlan *>(l.fp8_conv_fwd_plan);
			bool use_relay = false;
			const void * relay_input_fp8 = nullptr;
			if (!state.train && state.index > 0 && l.fp8_relay_source_layer == state.index - 1)
			{
				Darknet::Layer & producer = state.net.layers[state.index - 1];
				if (producer.fp8_relay_next_layer == state.index && producer.fp8_relay_enabled &&
					producer.fp8_relay_valid && producer.fp8_relay_gpu)
				{
					use_relay = true;
					relay_input_fp8 = producer.fp8_relay_gpu;
				}
			}
			const bool output_bf16 = Darknet::fp8_conv_output_is_bf16(conv_plan);
			const int out_spatial = l.out_w * l.out_h;
			const size_t input_fp8_bytes = static_cast<size_t>(l.batch) * l.c * l.h * l.w;
			const size_t output_tmp_bytes = static_cast<size_t>(l.batch) * l.n * out_spatial * (output_bf16 ? sizeof(unsigned short) : sizeof(float));
			const size_t output_tmp_offset = fp8_align_workspace_offset(input_fp8_bytes);
			const size_t conv_workspace_offset = fp8_align_workspace_offset(output_tmp_offset + output_tmp_bytes);
			const size_t conv_workspace_bytes = l.fp8_workspace_size > conv_workspace_offset ? l.fp8_workspace_size - conv_workspace_offset : 0;
			char * const workspace = reinterpret_cast<char *>(state.workspace);
			void * const input_fp8 = use_relay ? const_cast<void *>(relay_input_fp8) : workspace;
			void * const output_tmp = workspace + output_tmp_offset;
			void * const conv_workspace = workspace + conv_workspace_offset;

			if (!use_relay && use_fp8_training && l.fp8_input_amax_gpu)
			{
				Darknet::fp8_quantize_nchw_to_nhwc_record_amax_gpu(
					state.input,
					l.batch,
					l.c,
					l.h,
					l.w,
					l.fp8_input_scale_gpu,
					input_fp8,
					l.fp8_input_amax_gpu);
			}
			else if (!use_relay)
			{
				Darknet::fp8_quantize_nchw_to_nhwc_gpu(
					state.input,
					l.batch,
					l.c,
					l.h,
					l.w,
					l.fp8_input_scale_gpu,
					input_fp8);
			}

			const float * const graph_bias = Darknet::fp8_conv_fuses_bias(conv_plan) ? l.biases_gpu : nullptr;
			if (!Darknet::fp8_conv_fprop(
					conv_plan,
					input_fp8,
					l.weights_fp8_nhwc_gpu,
					graph_bias,
					output_tmp,
					conv_workspace,
					conv_workspace_bytes))
			{
				return false;
			}

			const float * const conversion_bias = (!l.batch_normalize && !Darknet::fp8_conv_fuses_bias(conv_plan)) ? l.biases_gpu : nullptr;
			Darknet::fp8_nhwc_output_to_nchw_gpu(
				output_tmp,
				l.batch,
				l.n,
				l.out_h,
				l.out_w,
				output_bf16,
				conversion_bias,
				l.output_gpu);
			if (l.batch_normalize)
			{
				forward_batchnorm_layer_gpu(l, state);
			}
			l.fp8_graph_activation_fused = Darknet::fp8_conv_fuses_relu(conv_plan) ? 1 : 0;
			return true;
		}

		if (l.fp8_gemm_plan == nullptr ||
			l.weights_fp8_gpu == nullptr)
		{
			return false;
		}

		auto * const gemm_plan = static_cast<Darknet::Fp8GemmPlan *>(l.fp8_gemm_plan);
		const int m = l.n;
		const int k = l.size * l.size * l.c;
		const int n = l.out_w * l.out_h;
		const int k_pad = l.fp8_k_pad;
		const int chunk = std::max(1, l.fp8_forward_batch);
		if (l.batch % chunk != 0)
		{
			return false;
		}
		const bool needs_im2col = !(l.size == 1 && l.stride == 1 && l.stride_x == 1 && l.stride_y == 1 && l.dilation == 1);
		// TN GEMM B operand: `chunk` contiguous blocks of input, each transposed row-major (n x k_pad)
		const size_t input_fp8_bytes = Darknet::fp8_rowmajor_pad_cols_bytes(n, k_pad);
		const size_t output_elem = use_fp8_training ? sizeof(unsigned short) : sizeof(float);
		const size_t output_tmp_bytes = static_cast<size_t>(m) * n * output_elem;
		const size_t output_tmp_offset = fp8_align_workspace_offset(input_fp8_bytes * chunk);
		const size_t lt_workspace_offset = fp8_align_workspace_offset(output_tmp_offset + output_tmp_bytes * chunk);
		const size_t lt_workspace_bytes = l.fp8_workspace_size > lt_workspace_offset ? l.fp8_workspace_size - lt_workspace_offset : 0;
		char * const workspace = reinterpret_cast<char *>(state.workspace);
		void * const lt_workspace = workspace + lt_workspace_offset;

		const size_t input_stride = static_cast<size_t>(l.c) * l.h * l.w;
		for (int chunk_start = 0; chunk_start < l.batch; chunk_start += chunk)
		{
			// one batched launch quantizes the whole chunk (grid z = image index)
			float * const input = state.input + static_cast<size_t>(chunk_start) * input_stride;
			if (needs_im2col)
			{
				if (use_fp8_training && l.fp8_input_amax_gpu)
				{
					Darknet::fp8_im2col_quantize_transpose_rowmajor_pad_cols_record_amax_gpu(
						input,
						l.c,
						l.h, l.w,
						l.size, l.size,
						l.pad * l.dilation, l.pad * l.dilation,
						l.stride_y, l.stride_x,
						l.dilation, l.dilation,
						k_pad,
						l.fp8_input_scale_gpu,
						workspace,
						l.fp8_input_amax_gpu,
						chunk, input_stride, input_fp8_bytes);
				}
				else
				{
					Darknet::fp8_im2col_quantize_transpose_rowmajor_pad_cols_gpu(
						input,
						l.c,
						l.h, l.w,
						l.size, l.size,
						l.pad * l.dilation, l.pad * l.dilation,
						l.stride_y, l.stride_x,
						l.dilation, l.dilation,
						k_pad,
						l.fp8_input_scale_gpu,
						workspace,
						chunk, input_stride, input_fp8_bytes);
				}
			}
			else if (use_fp8_training && l.fp8_input_amax_gpu)
			{
				Darknet::fp8_quantize_transpose_rowmajor_pad_cols_record_amax_gpu(
					input, k, n, k_pad, l.fp8_input_scale_gpu, workspace, l.fp8_input_amax_gpu,
					chunk, input_stride, input_fp8_bytes);
			}
			else
			{
				Darknet::fp8_quantize_transpose_rowmajor_pad_cols_gpu(input, k, n, k_pad, l.fp8_input_scale_gpu, workspace,
					chunk, input_stride, input_fp8_bytes);
			}

			void * const gemm_output = workspace + output_tmp_offset;
			if (!Darknet::fp8_gemm(gemm_plan, l.weights_fp8_gpu, workspace, gemm_output, lt_workspace, lt_workspace_bytes))
			{
				return false;
			}

			// bias is fused into the layout conversion; batchnorm layers apply it inside batchnorm instead
			const float * const fused_bias = l.batch_normalize ? nullptr : l.biases_gpu;
			float * const output = l.output_gpu + static_cast<size_t>(chunk_start) * m * n;
			if (use_fp8_training)
			{
				fp8_bf16_gemm_output_to_f32(gemm_plan, gemm_output, m, n, output, chunk, fused_bias);
			}
			else
			{
				fp8_f32_colmajor_to_rowmajor(static_cast<const float *>(gemm_output), m, n, output, chunk, fused_bias);
			}
		}

		if (l.batch_normalize)
		{
			forward_batchnorm_layer_gpu(l, state);
		}
		return true;
#endif
	}

	static ConvBackwardResult backward_convolutional_layer_gpu_fp8(
		Darknet::Layer & l, Darknet::NetworkState state, const bool request_wgrad = true, const bool request_dgrad = true)
	{
		ConvBackwardResult result;
		result.wgrad_requested = request_wgrad && !state.net.adversarial && !l.train_only_bn;
		result.dgrad_requested = request_dgrad && state.delta != nullptr;
	#if !defined(DARKNET_HAS_FP8) || !defined(CUDNN)
		return result;
#else
		l.fp8_dy_amax_valid = 0;
		if (Darknet::fp8_backward_mode_from_env() == Darknet::Fp8BackwardMode::Cudnn)
		{
			return result;
		}

		const bool common_ready =
			state.train &&
			state.net.fp8_training &&
			get_current_iteration(state.net) >= state.net.fp8_warmup_iters &&
			l.fp8_train_eligible &&
			l.fp8_dy_scale_gpu != nullptr &&
			l.fp8_dy_amax_gpu != nullptr &&
			state.workspace != nullptr &&
			l.groups == 1;
		if (!common_ready || (!result.wgrad_requested && !result.dgrad_requested))
		{
			return result;
		}

		const bool can_wgrad =
			result.wgrad_requested &&
			l.fp8_wgrad_gemm_plan != nullptr &&
			l.fp8_input_scale_gpu != nullptr &&
			l.weight_updates_gpu != nullptr;
		const bool can_dgrad =
			result.dgrad_requested &&
			l.fp8_dgrad_gemm_plan != nullptr &&
			l.weights_fp8_t_gpu != nullptr;
		if (!can_wgrad && !can_dgrad)
		{
			return result;
		}

		const int filters = l.n;
		const int kernel = l.size * l.size * l.c;
		const int spatial = l.out_w * l.out_h;
		const int spatial_pad = Darknet::fp8_round_up_to_16(spatial);
		const int filters_pad = Darknet::fp8_round_up_to_16(filters);
		const bool needs_im2col = !(l.size == 1 && l.stride == 1 && l.stride_x == 1 && l.stride_y == 1 && l.dilation == 1);
		const int dchunk = std::max(1, l.fp8_dgrad_batch);
		const bool dgrad_batch_ok = can_dgrad && (l.batch % dchunk == 0);
		const int wchunk = std::max(1, l.fp8_wgrad_batch);
		const bool wgrad_batch_ok = can_wgrad && (l.batch % wchunk == 0);

		/* workspace layout: dy^T for the WHOLE batch persists at offset 0 so one quantize pass of
		 * l.delta_gpu (fused into the wgrad loop) can feed both phases; the phase scratch regions
		 * follow it and are reused between the sequential phases:
		 *   dyt_all: batch x (spatial x filters_pad) E5M2 (only when dgrad runs)
		 *   wgrad:   dy stripes (filters x wchunk*spatial_pad) | im2col stripes | optional output/tmp
		 *   dgrad:   dchunk x bf16 out | dchunk x col | delta tmp */
		const size_t dyt_fp8_bytes = Darknet::fp8_rowmajor_pad_cols_bytes(spatial, filters_pad);
		const size_t dyt_all_end = dgrad_batch_ok ? fp8_align_workspace_offset(dyt_fp8_bytes * l.batch) : 0;

		const size_t wgrad_k_total = static_cast<size_t>(wchunk) * spatial_pad;
		const size_t dy_fp8_offset = dyt_all_end;
		const size_t dy_fp8_bytes = static_cast<size_t>(filters) * wgrad_k_total;
		const size_t im2col_t_fp8_offset = fp8_align_workspace_offset(dy_fp8_offset + dy_fp8_bytes);
		const size_t im2col_t_fp8_bytes = static_cast<size_t>(kernel) * wgrad_k_total;
		auto * const wgrad_plan = static_cast<Darknet::Fp8GemmPlan *>(l.fp8_wgrad_gemm_plan);
		auto * const dgrad_plan = static_cast<Darknet::Fp8GemmPlan *>(l.fp8_dgrad_gemm_plan);
		const bool disable_fused_wgrad_accum = fp8_env_is_set("DARKNET_FP8_DISABLE_FUSED_WGRAD_ACCUM");
		const bool disable_fused_dgrad_accum = fp8_env_is_set("DARKNET_FP8_DISABLE_FUSED_DGRAD_ACCUM");
		const bool direct_wgrad = can_wgrad && l.fp8_wgrad_direct_update && !disable_fused_wgrad_accum;
		const bool direct_dgrad = can_dgrad && l.fp8_dgrad_direct_update;
		const bool wgrad_output_fp32 = can_wgrad && Darknet::fp8_gemm_output_is_fp32(wgrad_plan);
		const bool dgrad_output_fp32 = can_dgrad && Darknet::fp8_gemm_output_is_fp32(dgrad_plan);
		const size_t wgrad_elements = static_cast<size_t>(filters) * kernel;
		const size_t wgrad_staging_end = fp8_align_workspace_offset(im2col_t_fp8_offset + im2col_t_fp8_bytes);
		const size_t wgrad_output_offset = wgrad_staging_end;
		const size_t wgrad_output_bytes = direct_wgrad ? 0 : wgrad_elements * Darknet::fp8_gemm_output_element_bytes(wgrad_plan);
		const size_t wgrad_tmp_offset = fp8_align_workspace_offset(wgrad_output_offset + wgrad_output_bytes);
		const size_t wgrad_tmp_bytes = direct_wgrad ? 0 : wgrad_elements * sizeof(float);
		const size_t wgrad_scratch_end = can_wgrad ? (direct_wgrad ? wgrad_staging_end : wgrad_tmp_offset + wgrad_tmp_bytes) : 0;

		const size_t dgrad_matrix = static_cast<size_t>(kernel) * spatial;
		const size_t dgrad_output_offset = dyt_all_end;
		const size_t dgrad_output_bytes = direct_dgrad ? 0 : dgrad_matrix * dchunk * Darknet::fp8_gemm_output_element_bytes(dgrad_plan);
		const size_t dgrad_col_offset = fp8_align_workspace_offset(dgrad_output_offset + dgrad_output_bytes);
		const size_t input_delta_offset = dgrad_col_offset + (direct_dgrad ? 0 : dgrad_matrix * sizeof(float) * dchunk);
		const size_t input_delta_bytes = (!direct_dgrad && needs_im2col) ? static_cast<size_t>(l.c) * l.h * l.w * sizeof(float) : 0;
		const size_t dgrad_scratch_end = dgrad_batch_ok ? (direct_dgrad ? dyt_all_end : input_delta_offset + input_delta_bytes) : 0;

		const size_t lt_workspace_offset = fp8_align_workspace_offset(std::max(wgrad_scratch_end, dgrad_scratch_end));
		const size_t lt_workspace_bytes = l.fp8_workspace_size > lt_workspace_offset ? l.fp8_workspace_size - lt_workspace_offset : 0;
		char * const workspace = reinterpret_cast<char *>(state.workspace);
		void * const lt_workspace = workspace + lt_workspace_offset;

		bool wgrad_failed = !wgrad_batch_ok;
		bool dgrad_failed = !dgrad_batch_ok;
		Darknet::fp8_clear_amax_gpu(l.fp8_dy_amax_gpu);

		const size_t dy_src_stride = static_cast<size_t>(filters) * spatial;
		const size_t input_src_stride = static_cast<size_t>(l.c) * l.h * l.w;
		void * const dyt_all = dgrad_batch_ok ? workspace : nullptr;
		// dy^T for dgrad is produced inside the wgrad quantize (one read of delta, two layouts);
		// valid for phase 2 only if the wgrad loop covered the full batch
		bool dyt_ready = false;

		// phase 1: weight gradients - all images of a chunk share one GEMM (batch folded into
		// the reduction dimension; the per-stripe zero padding contributes nothing to the dot
		// products), and chunks accumulate into the same output with beta=1
		for (int chunk_start = 0; can_wgrad && !wgrad_failed && chunk_start < l.batch; chunk_start += wchunk)
		{
			void * const dy_fp8 = workspace + dy_fp8_offset;
			void * const im2col_t_fp8 = workspace + im2col_t_fp8_offset;
			void * const wgrad_output = direct_wgrad ? static_cast<void *>(l.weight_updates_gpu) : static_cast<void *>(workspace + wgrad_output_offset);
			const float * const dy = l.delta_gpu + static_cast<size_t>(chunk_start) * dy_src_stride;
			float * const input = state.input + static_cast<size_t>(chunk_start) * input_src_stride;

			// whole chunk in one launch: row-major stripes for wgrad + transposed dy^T for dgrad
			Darknet::fp8_quantize_e5m2_dual_layout_record_amax_gpu(
				dy, filters, spatial, spatial_pad, filters_pad, l.fp8_dy_scale_gpu,
				dy_fp8, wgrad_k_total,
				dyt_all ? static_cast<char *>(dyt_all) + static_cast<size_t>(chunk_start) * dyt_fp8_bytes : nullptr,
				l.fp8_dy_amax_gpu,
				wchunk, dy_src_stride, spatial_pad, dyt_fp8_bytes);

			if (needs_im2col)
			{
				Darknet::fp8_im2col_quantize_rowmajor_pad_cols_gpu(
					input,
					l.c,
					l.h, l.w,
					l.size, l.size,
					l.pad * l.dilation, l.pad * l.dilation,
					l.stride_y, l.stride_x,
					l.dilation, l.dilation,
					spatial_pad,
					l.fp8_input_scale_gpu,
					im2col_t_fp8,
					wgrad_k_total,
					wchunk, input_src_stride, spatial_pad);
			}
			else
			{
				Darknet::fp8_quantize_rowmajor_pad_cols_gpu(
					input, kernel, spatial, spatial_pad, l.fp8_input_scale_gpu,
					im2col_t_fp8, wgrad_k_total,
					wchunk, input_src_stride, spatial_pad);
			}

			if (!Darknet::fp8_gemm(
					wgrad_plan,
					direct_wgrad ? im2col_t_fp8 : dy_fp8,
					direct_wgrad ? dy_fp8 : im2col_t_fp8,
					wgrad_output,
					lt_workspace,
					lt_workspace_bytes,
					direct_wgrad ? 1.0f : (chunk_start > 0 ? 1.0f : 0.0f)))
			{
				// falling back to cuDNN is only safe before any FP8 gradients were
				// accumulated; later chunks would double-count
				if (direct_wgrad || chunk_start > 0)
				{
					darknet_fatal_error(DARKNET_LOC, "FP8 wgrad GEMM failed at batch %d of %d for layer %d after partial accumulation", chunk_start, l.batch, l.index);
				}
				wgrad_failed = true;
			}
		}
		if (can_wgrad && !wgrad_failed)
		{
			void * const wgrad_output = workspace + wgrad_output_offset;
			if (!direct_wgrad)
			{
				if (disable_fused_wgrad_accum)
				{
					float * const wgrad_tmp = reinterpret_cast<float *>(workspace + wgrad_tmp_offset);
					if (wgrad_output_fp32)
					{
						fp8_f32_colmajor_to_rowmajor(static_cast<const float *>(wgrad_output), filters, kernel, wgrad_tmp);
					}
					else
					{
						fp8_bf16_gemm_output_to_f32(wgrad_plan, wgrad_output, filters, kernel, wgrad_tmp);
					}
					axpy_ongpu(l.nweights, 1.0f, wgrad_tmp, 1, l.weight_updates_gpu, 1);
				}
				else
				{
					Darknet::fp8_colmajor_output_accumulate_rowmajor_gpu(
						wgrad_output,
						filters,
						kernel,
						!wgrad_output_fp32,
						1.0f,
						l.weight_updates_gpu);
				}
			}
			dyt_ready = dgrad_batch_ok;
			l.fp8_dy_amax_valid = 1;
		}

		// phase 2: data gradients, one strided-batched GEMM per chunk of images
		if (dgrad_batch_ok && !dyt_ready)
		{
			// wgrad didn't run (or fell back), so quantize dy^T for the whole batch in one launch
			Darknet::fp8_quantize_e5m2_transpose_rowmajor_pad_cols_record_amax_gpu(
				l.delta_gpu, filters, spatial, filters_pad, l.fp8_dy_scale_gpu, dyt_all, l.fp8_dy_amax_gpu,
				l.batch, dy_src_stride, dyt_fp8_bytes);
			l.fp8_dy_amax_valid = 1;
		}
		for (int chunk_start = 0; can_dgrad && !dgrad_failed && chunk_start < l.batch; chunk_start += dchunk)
		{
			void * const dgrad_output = direct_dgrad ?
				static_cast<void *>(state.delta + static_cast<size_t>(chunk_start) * l.c * l.h * l.w) :
				static_cast<void *>(workspace + dgrad_output_offset);
			float * const dgrad_col = reinterpret_cast<float *>(workspace + dgrad_col_offset);
			float * const input_delta_tmp = reinterpret_cast<float *>(workspace + input_delta_offset);
			if (!Darknet::fp8_gemm(
					dgrad_plan,
					direct_dgrad ? static_cast<char *>(dyt_all) + static_cast<size_t>(chunk_start) * dyt_fp8_bytes : l.weights_fp8_t_gpu,
					direct_dgrad ? static_cast<void *>(l.weights_fp8_t_gpu) : static_cast<char *>(dyt_all) + static_cast<size_t>(chunk_start) * dyt_fp8_bytes,
					dgrad_output,
					lt_workspace,
					lt_workspace_bytes,
					direct_dgrad ? 1.0f : 0.0f))
			{
				// same double-count hazard as wgrad, but against state.delta
				if (direct_dgrad || chunk_start > 0)
				{
					darknet_fatal_error(DARKNET_LOC, "FP8 dgrad GEMM failed at batch %d of %d for layer %d after partial accumulation", chunk_start, l.batch, l.index);
				}
				dgrad_failed = true;
				break;
			}

			if (direct_dgrad)
			{
				continue;
			}
			if (disable_fused_dgrad_accum)
			{
				if (dgrad_output_fp32)
				{
					fp8_f32_colmajor_to_rowmajor(static_cast<const float *>(dgrad_output), kernel, spatial, dgrad_col, dchunk);
				}
				else
				{
					fp8_bf16_gemm_output_to_f32(dgrad_plan, dgrad_output, kernel, spatial, dgrad_col, dchunk);
				}
				for (int i = 0; i < dchunk; ++i)
				{
					float * const delta = state.delta + static_cast<size_t>(chunk_start + i) * l.c * l.h * l.w;
					float * const col = dgrad_col + static_cast<size_t>(i) * dgrad_matrix;
					if (needs_im2col)
					{
						col2im_gpu_ext(
							col,
							l.c,
							l.h, l.w,
							l.size, l.size,
							l.pad * l.dilation, l.pad * l.dilation,
							l.stride_y, l.stride_x,
							l.dilation, l.dilation,
							input_delta_tmp);
						axpy_ongpu(l.c * l.h * l.w, 1.0f, input_delta_tmp, 1, delta, 1);
					}
					else
					{
						axpy_ongpu(kernel * spatial, 1.0f, col, 1, delta, 1);
					}
				}
			}
			else
			{
				Darknet::fp8_colmajor_output_to_nchw_delta_gpu(
					dgrad_output,
					dchunk,
					l.c,
					l.h,
					l.w,
					l.size,
					l.size,
					l.pad * l.dilation,
					l.pad * l.dilation,
					l.stride_y,
					l.stride_x,
					l.dilation,
					l.dilation,
					!dgrad_output_fp32,
					state.delta + static_cast<size_t>(chunk_start) * l.c * l.h * l.w);
			}
		}

		result.wgrad_done = can_wgrad && !wgrad_failed;
		result.dgrad_done = can_dgrad && !dgrad_failed;
		return result;
#endif
	}
}


__global__ void binarize_kernel(float *x, int n, float *binary)
{
	int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
	if (i >= n) return;
	binary[i] = (x[i] >= 0) ? 1 : -1;
}

void binarize_gpu(float *x, int n, float *binary)
{
	TAT(TATPARMS);

	binarize_kernel<<<cuda_gridsize(n), BLOCK, 0, get_cuda_stream() >>>(x, n, binary);
	CHECK_CUDA(cudaPeekAtLastError());
}

__device__ float eml_softplus_kernel(float x)
{
	if (x > 20.0f) return x;
	if (x < -20.0f) return expf(x);
	return log1pf(expf(x));
}

__device__ float eml_sigmoid_kernel(float x)
{
	if (x >= 0.0f)
	{
		const float z = expf(-x);
		return 1.0f / (1.0f + z);
	}
	const float z = expf(x);
	return z / (1.0f + z);
}

__global__ void eml_convolutional_forward_kernel(const float *x_branch, const float *y_branch, const float *shortcut, int n, float clamp, float eps, float scale, int residual, float *output)
{
	const int i = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
	if (i >= n) return;

	const float x = fminf(fmaxf(x_branch[i], -clamp), clamp);
	const float y = y_branch[i];
	float out = scale * (expf(x) - logf(eml_softplus_kernel(y) + eps));
	if (residual && shortcut)
	{
		out += shortcut[i];
	}
	output[i] = out;
}

__global__ void eml_convolutional_backward_kernel(const float *x_branch, const float *y_branch, const float *delta, int n, float clamp, float eps, float scale, float *x_delta, float *y_delta)
{
	const int i = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
	if (i >= n) return;

	const float d = delta[i] * scale;
	const float x = x_branch[i];
	const float y = y_branch[i];
	const float dx = (x > -clamp && x < clamp) ? expf(fminf(fmaxf(x, -clamp), clamp)) : 0.0f;
	const float dy = -eml_sigmoid_kernel(y) / (eml_softplus_kernel(y) + eps);
	x_delta[i] = d * dx;
	y_delta[i] = d * dy;
}

__global__ void binarize_input_kernel(float *input, int n, int size, float *binary)
{
	int s = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
	if (s >= size) return;
	int i = 0;
	float mean = 0;
	for(i = 0; i < n; ++i){
		mean += fabs(input[i*size + s]);
	}
	mean = mean / n;
	for(i = 0; i < n; ++i){
		binary[i*size + s] = (input[i*size + s] > 0) ? mean : -mean;
	}
}

static void backward_convolutional_layer_gpu_1x1(Darknet::Layer & l, Darknet::NetworkState state, float *original_input)
{
	TAT(TATPARMS);

	if (l.batch_normalize)
	{
		backward_batchnorm_layer_gpu(l, state);
	}

	const int m = l.n / l.groups;
	const int n = l.c / l.groups;
	const int k = l.out_w * l.out_h;

	// Debug: print GEMM parameters for 1x1 backward
	if (cfg_and_state.is_trace)
	{
		*cfg_and_state.output << "1x1 bwd GEMM: layer=" << l.index
			<< " m=" << m << " n=" << n << " k=" << k
			<< " l.h=" << l.h << " l.w=" << l.w
			<< " l.c=" << l.c << " l.n=" << l.n
			<< " nweights=" << l.nweights
			<< " delta_gpu=" << (void*)l.delta_gpu
			<< " state.input=" << (void*)state.input
			<< " weight_updates_gpu=" << (void*)l.weight_updates_gpu
			<< std::endl;
	}

	for (int i = 0; i < l.batch; ++i)
	{
		for (int j = 0; j < l.groups; ++j)
		{
			float * a = l.delta_gpu + (i*l.groups + j)*m*k;
			float * b = state.input + (i*l.groups + j)*n*l.h*l.w;
			float * c = l.weight_updates_gpu + j*l.nweights / l.groups;

			if (!state.net.adversarial && !l.train_only_bn)
			{
				gemm_ongpu(0, 1, m, n, k, 1, a, k, b, k, 1, c, n);
			}

			if (state.delta)
			{
				if (l.binary || l.xnor) swap_binary(&l);

				float * aw = l.weights_gpu + j*l.nweights / l.groups;
				float * bw = l.delta_gpu + (i*l.groups + j)*m*k;
				float * dw = state.delta + (i*l.groups + j)*n*l.h*l.w;

				gemm_ongpu(1, 0, n, k, m, 1, aw, n, bw, k, 0, dw, k);

				if (l.binary || l.xnor) swap_binary(&l);
				if (l.xnor)
				{
					gradient_array_ongpu(original_input + i*l.c*l.h*l.w, l.c*l.h*l.w, HARDTAN, state.delta + i*l.c*l.h*l.w);
				}
			}
		}
	}
}

void binarize_input_gpu(float *input, int n, int size, float *binary)
{
	TAT(TATPARMS);

	binarize_input_kernel<<<cuda_gridsize(size), BLOCK, 0, get_cuda_stream() >>>(input, n, size, binary);
	CHECK_CUDA(cudaPeekAtLastError());
}

__global__ void binarize_weights_kernel(float *weights, int n, int size, float *binary)
{
	int f = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
	if (f >= n) return;
	int i = 0;
	float mean = 0;
	for (i = 0; i < size; ++i)
	{
		mean += fabs(weights[f*size + i]);
	}
	mean = mean / size;
	for (i = 0; i < size; ++i)
	{
		binary[f*size + i] = (weights[f*size + i] > 0) ? mean : -mean;
	}
}

void binarize_weights_gpu(float *weights, int n, int size, float *binary)
{
	TAT(TATPARMS);

	binarize_weights_kernel <<<cuda_gridsize(n), BLOCK, 0, get_cuda_stream() >>>(weights, n, size, binary);
	CHECK_CUDA(cudaPeekAtLastError());
}


__global__ void set_zero_kernel(float *src, int size)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < size) src[i] = 0;
}

__inline__ __device__ float warpAllReduceSum(float val)
{
	for (int mask = WARP_SIZE / 2; mask > 0; mask /= 2)
#if CUDART_VERSION >= 9000
		val += __shfl_xor_sync(0xffffffff, val, mask);
#else
		val += __shfl_xor(val, mask);
#endif
	return val;
}

// only if (size % 32 == 0)
__global__ void reduce_kernel(float *weights, int n, int size, float *mean_arr_gpu)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int f = i / size;
	if (f >= n) return;
	float warp_mean = warpAllReduceSum(fabs(weights[i]));
	if (i % 32 == 0)
	{
		atomicAdd(&mean_arr_gpu[f], warp_mean / size);
	}
}

__global__ void binarize_weights_mean_kernel(float *weights, int n, int size, float *binary, float *mean_arr_gpu)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int f = i / size;
	if (f >= n) return;
	float mean = mean_arr_gpu[f];
	binary[i] = (weights[i] > 0) ? mean : -mean;
}

void fast_binarize_weights_gpu(float *weights, int n, int size, float *binary, float *mean_arr_gpu)
{
	TAT(TATPARMS);

	if (size % 32 == 0) {
		size_t gridsize = n * size;
		const int num_blocks = get_number_of_blocks(gridsize, BLOCK);// gridsize / BLOCK + 1;

		set_zero_kernel <<<(n/BLOCK + 1), BLOCK, 0, get_cuda_stream() >>> (mean_arr_gpu, n);
		reduce_kernel <<<num_blocks, BLOCK, 0, get_cuda_stream() >>> (weights, n, size, mean_arr_gpu);
		binarize_weights_mean_kernel <<<num_blocks, BLOCK, 0, get_cuda_stream() >>> (weights, n, size, binary, mean_arr_gpu);
		CHECK_CUDA(cudaPeekAtLastError());
	}
	else {
		binarize_weights_gpu(weights, n, size, binary);
	}
}


__global__ void cuda_f32_to_f16(float* input_f32, size_t size, half *output_f16)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < size) output_f16[idx] = __float2half(input_f32[idx]);
}

void cuda_convert_f32_to_f16(float* input_f32, size_t size, float *output_f16)
{
	TAT(TATPARMS);

	cuda_f32_to_f16 <<< get_number_of_blocks(size, BLOCK), BLOCK, 0, get_cuda_stream() >>> (input_f32, size, (half *)output_f16);
	CHECK_CUDA(cudaPeekAtLastError());
}

__global__ void cuda_f16_to_f32(half* input_f16, size_t size, float *output_f32)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < size) output_f32[idx] = __half2float(input_f16[idx]);
}

void cuda_convert_f16_to_f32(float* input_f16, size_t size, float *output_f32)
{
	TAT(TATPARMS);

	cuda_f16_to_f32 <<< get_number_of_blocks(size, BLOCK), BLOCK, 0, get_cuda_stream() >>> ((half *)input_f16, size, output_f32);
	CHECK_CUDA(cudaPeekAtLastError());
}

#if defined(DARKNET_GPU_CUDA) && defined(CUDNN_DATA_BFLOAT16)
__global__ void cuda_f32_to_bf16(float *input_f32, size_t size, __nv_bfloat16 *output_bf16)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < size) output_bf16[idx] = __float2bfloat16(input_f32[idx]);
}

void cuda_convert_f32_to_bf16(float *input_f32, size_t size, float *output_bf16)
{
	TAT(TATPARMS);

	cuda_f32_to_bf16 <<< get_number_of_blocks(size, BLOCK), BLOCK, 0, get_cuda_stream() >>> (input_f32, size, (__nv_bfloat16 *)output_bf16);
	CHECK_CUDA(cudaPeekAtLastError());
}

__global__ void cuda_bf16_to_f32(__nv_bfloat16 *input_bf16, size_t size, float *output_f32)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < size) output_f32[idx] = __bfloat162float(input_bf16[idx]);
}

void cuda_convert_bf16_to_f32(float *input_bf16, size_t size, float *output_f32)
{
	TAT(TATPARMS);

	cuda_bf16_to_f32 <<< get_number_of_blocks(size, BLOCK), BLOCK, 0, get_cuda_stream() >>> ((__nv_bfloat16 *)input_bf16, size, output_f32);
	CHECK_CUDA(cudaPeekAtLastError());
}
#endif

#ifdef CUDNN
void cuda_convert_f32_to_cudnn_16bit(float *input_f32, size_t size, float *output_16, int mode)
{
	TAT(TATPARMS);

	if (mode == DARKNET_CUDNN_16BIT_HALF)
	{
		cuda_convert_f32_to_f16(input_f32, size, output_16);
		return;
	}

#if defined(DARKNET_GPU_CUDA) && defined(CUDNN_DATA_BFLOAT16)
	if (mode == DARKNET_CUDNN_16BIT_BF16)
	{
		cuda_convert_f32_to_bf16(input_f32, size, output_16);
		return;
	}
#endif

	darknet_fatal_error(DARKNET_LOC, "unsupported cuDNN 16-bit mode %d", mode);
}

void cuda_convert_cudnn_16bit_to_f32(float *input_16, size_t size, float *output_f32, int mode)
{
	TAT(TATPARMS);

	if (mode == DARKNET_CUDNN_16BIT_HALF)
	{
		cuda_convert_f16_to_f32(input_16, size, output_f32);
		return;
	}

#if defined(DARKNET_GPU_CUDA) && defined(CUDNN_DATA_BFLOAT16)
	if (mode == DARKNET_CUDNN_16BIT_BF16)
	{
		cuda_convert_bf16_to_f32(input_16, size, output_f32);
		return;
	}
#endif

	darknet_fatal_error(DARKNET_LOC, "unsupported cuDNN 16-bit mode %d", mode);
}
#endif

half *cuda_make_f16_from_f32_array(float *src, size_t n)
{
	TAT(TATPARMS);

	half *dst16;
	size_t size = sizeof(half)*n;
	CHECK_CUDA(cudaMalloc((void **)&dst16, size));
	if (src) {
		assert(n > 0);
		cuda_convert_f32_to_f16(src, n, (float *)dst16);
	}
	if (!dst16)
	{
		darknet_fatal_error(DARKNET_LOC, "CUDA malloc failed (n=%d)", n);
	}
	return dst16;
}

void forward_convolutional_layer_gpu(Darknet::Layer & l, Darknet::NetworkState state)
{
	TAT(TATPARMS);
	// Set only by a successful cuDNN Frontend Conv+Bias+ReLU graph below.
	// Clearing it here prevents a fallback path from accidentally skipping its
	// ordinary activation after a previous FP8 graph execution.
	l.fp8_graph_activation_fused = 0;

	if (l.train == 0) state.train = 0;

	if (l.stream >= 0)
	{
		switch_stream(l.stream);
	}

	if (l.wait_stream_id >= 0)
	{
		wait_stream(l.wait_stream_id);
	}

	if (forward_convolutional_layer_gpu_fp4(l, state))
	{
		forward_convolutional_layer_gpu_epilogue(l, state);
		return;
	}

	if (forward_convolutional_layer_gpu_fp8(l, state))
	{
		forward_convolutional_layer_gpu_epilogue(l, state);
		return;
	}

	//fill_ongpu(l.outputs*l.batch, 0, l.output_gpu, 1);
	if (l.binary)
	{
		binarize_weights_gpu(l.weights_gpu, l.n, (l.c / l.groups)*l.size*l.size, l.binary_weights_gpu);
		swap_binary(&l);
	}

	if (l.xnor)
	{
		if (!l.align_bit_weights_gpu || state.train)
		{
			fast_binarize_weights_gpu(l.weights_gpu, l.n, (l.c / l.groups)*l.size*l.size, l.binary_weights_gpu, l.mean_arr_gpu);
		}

		if (l.align_bit_weights_gpu && !state.train && l.c >= 32 && l.stride_x == l.stride_y)
		{
			int m = l.n / l.groups;
			int k = l.size*l.size*l.c / l.groups;
			int n = l.out_w*l.out_h;

			const int ldb_align = l.lda_align;
			const size_t new_ldb = k + (ldb_align - k%ldb_align); // (k / 8 + 1) * 8;

			if (l.c % 32 == 0)
			{
				const int new_c = l.c / 32;

				repack_input_gpu_bin(state.input, (uint32_t *)l.align_workspace_gpu, l.w, l.h, l.c);

				im2col_ongpu(l.align_workspace_gpu, new_c, l.h, l.w, l.size, l.stride, l.pad, state.workspace);

				int new_k = l.size*l.size*l.c / 32;

				transpose_uint32_gpu((uint32_t *)state.workspace, (uint32_t *)l.transposed_align_workspace_gpu, new_k, n, n, new_ldb);
				gemm_nn_custom_bin_mean_transposed_gpu(m, n, k,
					(unsigned char *)l.align_bit_weights_gpu, new_ldb, (unsigned char *)l.transposed_align_workspace_gpu,
					new_ldb, l.output_gpu, n, l.mean_arr_gpu, l.biases_gpu, l.activation == LEAKY,
					l.bin_conv_shortcut_in_gpu, l.bin_conv_shortcut_out_gpu);
			}
			else
			{
				int i = 0;
				{
					im2col_align_ongpu(state.input + i*l.c*l.h*l.w, l.c, l.h, l.w, l.size, l.stride, l.pad, l.align_workspace_gpu, l.bit_align);

					// should be optimized
					float_to_bit_gpu(l.align_workspace_gpu, (unsigned char *)state.workspace, l.align_workspace_size);
				}
				transpose_bin_gpu((unsigned char *)state.workspace, (unsigned char *)l.transposed_align_workspace_gpu, k, n, l.bit_align, new_ldb, 8);

				gemm_nn_custom_bin_mean_transposed_gpu(m, n, k,
						(unsigned char *)l.align_bit_weights_gpu, new_ldb, (unsigned char *)l.transposed_align_workspace_gpu,
						new_ldb, l.output_gpu, n, l.mean_arr_gpu, l.biases_gpu, l.activation == LEAKY,
						l.bin_conv_shortcut_in_gpu, l.bin_conv_shortcut_out_gpu);
			}

			if (l.activation == SWISH) activate_array_swish_ongpu(l.output_gpu, l.outputs*l.batch, l.activation_input_gpu, l.output_gpu);
			else if (l.activation == MISH) activate_array_mish_ongpu(l.output_gpu, l.outputs*l.batch, l.activation_input_gpu, l.output_gpu);
			else if (l.activation == HARD_MISH) activate_array_hard_mish_ongpu(l.output_gpu, l.outputs*l.batch, l.activation_input_gpu, l.output_gpu);
			else if (l.activation == EML) activate_array_eml_ongpu(l.output_gpu, l.outputs*l.batch, l.activation_input_gpu, l.output_gpu);
			else if (l.activation == NORM_CHAN) activate_array_normalize_channels_ongpu(l.output_gpu, l.outputs*l.batch, l.batch, l.out_c, l.out_w*l.out_h, l.output_gpu);
			else if (l.activation == NORM_CHAN_SOFTMAX) activate_array_normalize_channels_softmax_ongpu(l.output_gpu, l.outputs*l.batch, l.batch, l.out_c, l.out_w*l.out_h, l.output_gpu, 0);
			else if (l.activation == NORM_CHAN_SOFTMAX_MAXVAL) activate_array_normalize_channels_softmax_ongpu(l.output_gpu, l.outputs*l.batch, l.batch, l.out_c, l.out_w*l.out_h, l.output_gpu, 1);
			else if (l.activation != LINEAR && l.activation != LEAKY) activate_array_ongpu(l.output_gpu, l.outputs*l.batch, l.activation);
			return;
		}
	}

	if (l.xnor)
	{
		swap_binary(&l);
		binarize_gpu(state.input, l.c*l.h*l.w*l.batch, l.binary_input_gpu);
		state.input = l.binary_input_gpu;
	}

	//fill_ongpu(l.outputs*l.batch, 0, l.output_gpu, 1);

#ifdef CUDNN
	//float one = 1;    // alpha[0], beta[0] is float for HALF and FLOAT
	float alpha = 1, beta = 0;

//#ifdef CUDNN_HALF
	//if (state.use_mixed_precision) {
	int iteration_num = get_current_iteration(state.net); // (*state.net.seen) / (state.net.batch*state.net.subdivisions);
	const bool use_cudnn_16bit = (state.net.cudnn_half || state.net.cudnn_bf16);
	const bool training_ready = (!state.train || state.net.cudnn_bf16 ||
		((iteration_num > 3 * state.net.burn_in) && state.net.loss_scale != 1));
	const int cudnn_16bit_mode = state.net.cudnn_bf16 ? DARKNET_CUDNN_16BIT_BF16 : DARKNET_CUDNN_16BIT_HALF;
	if (state.index != 0 && use_cudnn_16bit && training_ready && !l.xnor &&
		(l.c / l.groups) % 8 == 0 && l.n % 8 == 0 && l.groups <= 1 && l.size > 1)
	{
		if (l.cudnn_16bit_mode != cudnn_16bit_mode)
		{
			set_convolutional_cudnn_16bit_mode(&l, cudnn_16bit_mode);
		}

		// Note: For improved performance it is advised to use beta[0] = 0.0.
		// For Tensor Core: cudnnSetConvolutionMathType() where cudnnMathType_t mathType = CUDNN_TENSOR_OP_MATH;
		// 1. or CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM and use CUDNN_DATA_HALF
		// 2. or CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED
		// More: http://docs.nvidia.com/deeplearning/sdk/cudnn-developer-guide/index.html#tensor_ops

		const size_t input16_size = l.batch*l.c*l.w*l.h;
		const size_t output16_size = l.batch*l.out_c*l.out_h*l.out_w;

		if (*state.net.max_input16_size < input16_size)
		{
			*state.net.max_input16_size = input16_size;
			if (*state.net.input16_gpu) cuda_free(*state.net.input16_gpu);
			assert(*state.net.max_input16_size > 0);
			CHECK_CUDA(cudaMalloc((void **)state.net.input16_gpu, *state.net.max_input16_size * sizeof(short)));
		}
		float *input16 = *state.net.input16_gpu;

		if (*state.net.max_output16_size < output16_size) {
			*state.net.max_output16_size = output16_size;
			if (*state.net.output16_gpu) cuda_free(*state.net.output16_gpu);
			assert(*state.net.max_output16_size > 0);
			CHECK_CUDA(cudaMalloc((void **)state.net.output16_gpu, *state.net.max_output16_size * sizeof(short)));
		}
		float *output16 = *state.net.output16_gpu;

		assert(input16_size > 0);
		if (!state.train && l.weights_gpu && l.weights_gpu16)
		{
			cuda_convert_f32_to_cudnn_16bit(l.weights_gpu, l.nweights, l.weights_gpu16, cudnn_16bit_mode);
		}
		cuda_convert_f32_to_cudnn_16bit(state.input, input16_size, input16, cudnn_16bit_mode);

		CHECK_CUDNN(cudnnConvolutionForward(cudnn_handle(),
			&alpha,
			l.srcTensorDesc16,
			input16,
			l.weightDesc16,
			l.weights_gpu16,
			l.convDesc,
			l.fw_algo16,
			state.workspace,
			l.workspace_size,
			&beta,
			l.dstTensorDesc16,
			output16));


		if (l.batch_normalize)
		{
			if (state.train && !state.net.adversarial) // Training
			{
				simple_copy_ongpu(l.outputs*l.batch / 2, output16, l.x_gpu);
				float one = 1.0f;
				float zero = 0.0f;
				// Batch-normalization can still take FP16 inputs and outputs, saving half the bandwidth
				// compared to FP32, it's just that the statistics and value adjustment should be done in FP32.
				CHECK_CUDNN(cudnnBatchNormalizationForwardTraining(cudnn_handle(),
					CUDNN_BATCHNORM_SPATIAL,
					&one,
					&zero,
					l.normDstTensorDescF16,
					l.x_gpu,            // input
					l.normDstTensorDescF16,
					output16,            // output
					l.normTensorDesc,
					l.scales_gpu,       // input
					l.biases_gpu,       // input
					.01,
					l.rolling_mean_gpu,        // input/output (should be FP32)
					l.rolling_variance_gpu,    // input/output (should be FP32)
					.00001,
					l.mean_gpu,            // output (should be FP32) - optional cache to speedup cudnnBatchNormalizationBackward()
					l.variance_gpu));    // output (should be FP32) - optional cache to speedup cudnnBatchNormalizationBackward()

				cuda_convert_cudnn_16bit_to_f32(output16, output16_size, l.output_gpu, cudnn_16bit_mode);
				//forward_batchnorm_layer_gpu(l, state);
			}
			else // Detection
			{
				cuda_convert_cudnn_16bit_to_f32(output16, output16_size, l.output_gpu, cudnn_16bit_mode);
				normalize_gpu(l.output_gpu, l.rolling_mean_gpu, l.rolling_variance_gpu, l.batch, l.out_c, l.out_h*l.out_w);
				scale_bias_gpu(l.output_gpu, l.scales_gpu, l.batch, l.out_c, l.out_h*l.out_w);
				add_bias_gpu(l.output_gpu, l.biases_gpu, l.batch, l.out_c, l.out_w*l.out_h);
			}
		}
		else // BIAS only
		{
			cuda_convert_cudnn_16bit_to_f32(output16, output16_size, l.output_gpu, cudnn_16bit_mode);
			add_bias_gpu(l.output_gpu, l.biases_gpu, l.batch, l.n, l.out_w*l.out_h);
		}
	}
	else
	{
		CHECK_CUDNN(cudnnConvolutionForward(cudnn_handle(),
			&alpha, //&one,
			l.srcTensorDesc,
			state.input,
			l.weightDesc,
			l.weights_gpu,
			l.convDesc,
			l.fw_algo,
			state.workspace,
			l.workspace_size,
			&beta,  //&one,
			l.dstTensorDesc,
			l.output_gpu));

		//cudaDeviceSynchronize();
		if (l.batch_normalize) {
			forward_batchnorm_layer_gpu(l, state);
		}
		else {
			add_bias_gpu(l.output_gpu, l.biases_gpu, l.batch, l.n, l.out_w*l.out_h);
		}
	//#endif    // CUDNN_HALF
	}


#else
	fill_ongpu(l.outputs*l.batch, 0, l.output_gpu, 1);

	int i, j;
	int m = l.n / l.groups;
	int k = l.size*l.size*l.c / l.groups;
	int n = l.out_w*l.out_h;

	for(i = 0; i < l.batch; ++i)
	{
		for (j = 0; j < l.groups; ++j)
		{
			//float *im = state.input + i*l.c*l.h*l.w;
			float *im = state.input + (i*l.groups + j)*l.c / l.groups*l.h*l.w;
			float *a = l.weights_gpu + j*l.nweights / l.groups;
			float *b = state.workspace;
			float *c = l.output_gpu + (i*l.groups + j)*n*m;
			if (l.size == 1 && l.stride == 1 && l.dilation == 1)
			{
				b = im;
			}
			else
			{
				//im2col_ongpu(im, l.c / l.groups, l.h, l.w, l.size, l.stride, l.pad, state.workspace);

				im2col_gpu_ext(im,          // input
					l.c / l.groups,         // input channels
					l.h, l.w,               // input size (h, w)
					l.size, l.size,         // kernel size (h, w)
					l.pad * l.dilation, l.pad * l.dilation,   // padding (h, w)
					l.stride_y, l.stride_x,     // stride (h, w)
					l.dilation, l.dilation, // dilation (h, w)
					state.workspace);       // output

			}
			//gemm_ongpu(0, 0, m, n, k, 1., a, k, b, n, 1., c + i*m*n, n);
			gemm_ongpu(0, 0, m, n, k, 1, a, k, b, n, 1, c, n);
		}
	}

	if (l.batch_normalize)
	{
		forward_batchnorm_layer_gpu(l, state);
	}
	else
	{
		add_bias_gpu(l.output_gpu, l.biases_gpu, l.batch, l.n, l.out_w*l.out_h);
	}
#endif

//#ifndef CUDNN_HALF
//#endif // no CUDNN_HALF

	forward_convolutional_layer_gpu_epilogue(l, state);
}

void forward_eml_convolutional_layer_gpu(Darknet::Layer & l, Darknet::NetworkState state)
{
	TAT(TATPARMS);

	Darknet::NetworkState branch_state = state;
	forward_convolutional_layer_gpu(*(l.input_layer), branch_state);
	forward_convolutional_layer_gpu(*(l.self_layer), branch_state);

	const int total = l.outputs * l.batch;
	const float clamp = l.alpha > 0.0f ? l.alpha : 4.0f;
	const float eps = l.beta > 0.0f ? l.beta : 0.000001f;
	const int residual = (l.shortcut && l.inputs == l.outputs) ? 1 : 0;

	eml_convolutional_forward_kernel<<<cuda_gridsize(total), BLOCK, 0, get_cuda_stream()>>>(
		l.input_layer->output_gpu,
		l.self_layer->output_gpu,
		state.input,
		total,
		clamp,
		eps,
		l.scale,
		residual,
		l.output_gpu);
	CHECK_CUDA(cudaPeekAtLastError());
}


void backward_convolutional_layer_gpu(Darknet::Layer & l, Darknet::NetworkState state)
{
	TAT(TATPARMS);

	if (l.coordconv) {
		coord_conv_gpu(l.delta_gpu, l.outputs*l.batch, l.out_w, l.out_h, l.out_c, l.batch, 1);
	}

	if (l.antialiasing) {
		Darknet::NetworkState s = { 0 };
		s.train = state.train;
		s.workspace = state.workspace;
		s.net = state.net;
		s.delta = l.delta_gpu;  // s.delta will be returned to l.delta_gpu
		s.input = l.input_antialiasing_gpu;
		//if (!state.train) s.index = state.index;  // don't use TC for training (especially without cuda_convert_f32_to_f16() )
		simple_copy_ongpu(l.input_layer->outputs*l.input_layer->batch, l.delta_gpu, l.input_layer->delta_gpu);
		backward_convolutional_layer_gpu(*(l.input_layer), s);

		simple_copy_ongpu(l.outputs*l.batch, l.input_antialiasing_gpu, l.output_gpu);
	}

	if(state.net.try_fix_nan) constrain_ongpu(l.outputs*l.batch, 1, l.delta_gpu, 1);

	if (l.activation == SWISH) gradient_array_swish_ongpu(l.output_gpu, l.outputs*l.batch, l.activation_input_gpu, l.delta_gpu);
	else if (l.activation == MISH) gradient_array_mish_ongpu(l.outputs*l.batch, l.activation_input_gpu, l.delta_gpu);
	else if (l.activation == HARD_MISH) gradient_array_hard_mish_ongpu(l.outputs*l.batch, l.activation_input_gpu, l.delta_gpu);
	else if (l.activation == EML) gradient_array_eml_ongpu(l.outputs*l.batch, l.activation_input_gpu, l.delta_gpu);
	else if (l.activation == NORM_CHAN_SOFTMAX || l.activation == NORM_CHAN_SOFTMAX_MAXVAL) gradient_array_normalize_channels_softmax_ongpu(l.output_gpu, l.outputs*l.batch, l.batch, l.out_c, l.out_w*l.out_h, l.delta_gpu);
	else if (l.activation == NORM_CHAN) gradient_array_normalize_channels_ongpu(l.output_gpu, l.outputs*l.batch, l.batch, l.out_c, l.out_w*l.out_h, l.delta_gpu);
	else gradient_array_ongpu(l.output_gpu, l.outputs*l.batch, l.activation, l.delta_gpu);

	if (!l.batch_normalize)
		backward_bias_gpu(l.bias_updates_gpu, l.delta_gpu, l.batch, l.n, l.out_w*l.out_h);

//#ifndef CUDNN_HALF
	//if(l.batch_normalize){
	//    backward_batchnorm_layer_gpu(l, state);
	//} else {
	//    //backward_bias_gpu(l.bias_updates_gpu, l.delta_gpu, l.batch, l.n, l.out_w*l.out_h);
	//}
//#endif // no CUDNN_HALF
	float *original_input = state.input;

	if(l.xnor) state.input = l.binary_input_gpu;
#ifdef CUDNN
	const bool use_1x1_fallback = (l.size == 1 &&
		l.stride == 1 && l.stride_x == 1 && l.stride_y == 1 &&
		l.pad == 0 && l.dilation == 1 &&
		l.out_w == l.w && l.out_h == l.h &&
		l.n == 1);
	if (use_1x1_fallback)
	{
		if (cfg_and_state.is_trace)
		{
			*cfg_and_state.output << "conv bwd filter: using 1x1 fallback for layer " << l.index << std::endl;
		}
		backward_convolutional_layer_gpu_1x1(l, state, original_input);
	}
	else
	{
	float alpha = 1.0f;
	float beta = 0.0f;

//#ifdef CUDNN_HALF
	int iteration_num = get_current_iteration(state.net); //(*state.net.seen) / (state.net.batch*state.net.subdivisions);
	const bool use_cudnn_16bit = (state.net.cudnn_half || state.net.cudnn_bf16);
	const bool training_ready = (!state.train || state.net.cudnn_bf16 ||
		((iteration_num > 3 * state.net.burn_in) && state.net.loss_scale != 1));
	const int cudnn_16bit_mode = state.net.cudnn_bf16 ? DARKNET_CUDNN_16BIT_BF16 : DARKNET_CUDNN_16BIT_HALF;
	if (state.index != 0 && use_cudnn_16bit && training_ready && !l.xnor &&
		(l.c / l.groups) % 8 == 0 && l.n % 8 == 0  && l.groups <= 1 && l.size > 1)
	{
		if (l.cudnn_16bit_mode != cudnn_16bit_mode)
		{
			set_convolutional_cudnn_16bit_mode(&l, cudnn_16bit_mode);
		}

		const size_t input16_size = l.batch*l.c*l.w*l.h;
		const size_t delta16_size = l.batch*l.n*l.out_w*l.out_h;

		if (*state.net.max_input16_size < input16_size)
		{
			*state.net.max_input16_size = input16_size;
			if (*state.net.input16_gpu) cuda_free(*state.net.input16_gpu);
			assert(*state.net.max_input16_size > 0);
			CHECK_CUDA(cudaMalloc((void **)state.net.input16_gpu, *state.net.max_input16_size * sizeof(short)));
		}
		float *input16 = *state.net.input16_gpu;

		if (*state.net.max_output16_size < delta16_size)
		{
			*state.net.max_output16_size = delta16_size;
			if (*state.net.output16_gpu) cuda_free(*state.net.output16_gpu);
			assert(*state.net.max_output16_size > 0);
			CHECK_CUDA(cudaMalloc((void **)state.net.output16_gpu, *state.net.max_output16_size * sizeof(short)));
		}
		float *delta16 = *state.net.output16_gpu;

		assert(input16_size > 0);
		assert(delta16_size > 0);
		cuda_convert_f32_to_cudnn_16bit(state.input, input16_size, input16, cudnn_16bit_mode);
		cuda_convert_f32_to_cudnn_16bit(l.delta_gpu, delta16_size, delta16, cudnn_16bit_mode);

		if (l.batch_normalize)
		{
			float one = 1.0f;
			float zero = 0.0f;
			CHECK_CUDNN(cudnnBatchNormalizationBackward(cudnn_handle(),
				CUDNN_BATCHNORM_SPATIAL,
				&one,
				&zero,
				&one,
				&one,
				l.normDstTensorDescF16,
				l.x_gpu,                // input (input in BN-forward-inference)
				l.normDstTensorDescF16,
				delta16,                // input
				l.normDstTensorDescF16,
				l.output_gpu, //l.x_norm_gpu,            // output (new delta)
				l.normTensorDesc,
				l.scales_gpu,            // input (should be FP32)
				l.scale_updates_gpu,    // output (should be FP32)
				l.bias_updates_gpu,        // output (should be FP32)
				.00001,
				l.mean_gpu,                // input (should be FP32)
				l.variance_gpu));        // input (should be FP32)

			simple_copy_ongpu(l.outputs*l.batch / 2, l.output_gpu, delta16);
		}

		// convert input: state.input (x), l.delta_gpu (y) from fp32 to fp16
		// get output: l.weight_updates_gpu (dw) and convert it to fp32 (ONLY if it is fp16)

		// calculate conv weight updates
		// Already: l.weight_updates_gpu = (l.weight_updates_gpu - l.weight*decay*batch*subdivision)*momentum
		//   so we should copy f32 to f16, or compute: f16=(w_up - w*d*b*s)*m
		assert((l.nweights) > 0);
		cuda_convert_f32_to_cudnn_16bit(l.weight_updates_gpu, l.nweights, l.weight_updates_gpu16, cudnn_16bit_mode);

		float one = 1.0f;
		if (!state.net.adversarial && !l.train_only_bn)
		{
			CHECK_CUDNN(cudnnConvolutionBackwardFilter(cudnn_handle(),
				&one,
				l.srcTensorDesc16,
				input16, //state.input,
				l.ddstTensorDesc16,
				delta16, //l.delta_gpu,
				l.convDesc,
				l.bf_algo16,
				state.workspace,
				l.workspace_size,
				&one,
				l.dweightDesc16,
				l.weight_updates_gpu16));    // l.weight_updates_gpu);

			cuda_convert_cudnn_16bit_to_f32(l.weight_updates_gpu16, l.nweights, l.weight_updates_gpu, cudnn_16bit_mode);
		}

		if (state.delta)
		{
			if (l.binary || l.xnor) swap_binary(&l);

			// http://docs.nvidia.com/deeplearning/sdk/cudnn-developer-guide/index.html#cudnnConvolutionBackwardData
			// calculate delta for the next layer
			// convert input: l.weights_gpu (w), l.delta_gpu (dy) from fp32 to fp16
			// get output: state.delta (dx) and convert it to fp32 (ONLY if it is fp16)
			CHECK_CUDNN(cudnnConvolutionBackwardData(cudnn_handle(),
				&alpha,
				l.weightDesc16,
				l.weights_gpu16, //l.weights_gpu,
				l.ddstTensorDesc16,
				delta16, //l.delta_gpu,
				l.convDesc,
				l.bd_algo16,
				state.workspace,
				l.workspace_size,
				&beta,
				l.dsrcTensorDesc16,
				input16));    // state.delta);

			cuda_convert_cudnn_16bit_to_f32(input16, input16_size, state.delta, cudnn_16bit_mode);

			if (l.binary || l.xnor) swap_binary(&l);
			if (l.xnor) gradient_array_ongpu(original_input, l.batch*l.c*l.h*l.w, HARDTAN, state.delta);
		}
	}
	else
	{
		//#else    // CUDNN_HALF

		if(l.batch_normalize){
			backward_batchnorm_layer_gpu(l, state);
		}

		const ConvBackwardResult fp4_backward = backward_convolutional_layer_gpu_fp4(l, state);
		const auto fp8_remaining = Darknet::remaining_convolution_gradients(
			{fp4_backward.wgrad_done, fp4_backward.dgrad_done});
		const ConvBackwardResult fp8_backward = backward_convolutional_layer_gpu_fp8(
			l, state, fp8_remaining.wgrad_done, fp8_remaining.dgrad_done);
		const bool low_precision_wgrad_done = fp4_backward.wgrad_done || fp8_backward.wgrad_done;
		const bool low_precision_dgrad_done = fp4_backward.dgrad_done || fp8_backward.dgrad_done;

		if (!state.net.adversarial && !l.train_only_bn)
		{
			float *old_input = state.input;

			// calculate conv weight updates
			// if used: beta=1 then loss decreases faster
			float one = 1.0f;
			if (!low_precision_wgrad_done)
			{
				if (cfg_and_state.is_trace)
				{
					*cfg_and_state.output
						<< "conv bwd filter: layer=" << l.index
						<< " batch=" << l.batch
						<< " c=" << l.c
						<< " h=" << l.h
						<< " w=" << l.w
						<< " n=" << l.n
						<< " size=" << l.size
						<< " stride=" << l.stride
						<< " pad=" << l.pad
						<< " out=" << l.out_w << "x" << l.out_h
						<< " groups=" << l.groups
						<< " workspace=" << l.workspace_size
						<< " algo=" << l.bf_algo
						<< std::endl;
				}
				CHECK_CUDNN(cudnnConvolutionBackwardFilter(cudnn_handle(),
					&one,
					l.srcTensorDesc,
					state.input,
					l.ddstTensorDesc,
					l.delta_gpu,
					l.convDesc,
					l.bf_algo,
					state.workspace,
					l.workspace_size,
					&one,
					l.dweightDesc,
					l.weight_updates_gpu));
			}

			state.input = old_input;
		}

		if (state.delta)
		{
			if (l.binary || l.xnor) swap_binary(&l);

			float *old_weights = l.weights_gpu;

			// http://docs.nvidia.com/deeplearning/sdk/cudnn-developer-guide/index.html#cudnnConvolutionBackwardData
			// calculate delta for the next layer
			float one = 1.0f;
			if (!low_precision_dgrad_done)
			{
				CHECK_CUDNN(cudnnConvolutionBackwardData(cudnn_handle(),
					&one,
					l.weightDesc,
					l.weights_gpu,
					l.ddstTensorDesc,
					l.delta_gpu,
					l.convDesc,
					l.bd_algo,
					state.workspace,
					l.workspace_size,
					&one,
					l.dsrcTensorDesc,
					state.delta));
			}

			l.weights_gpu = old_weights;

			if (l.binary || l.xnor) swap_binary(&l);
			if (l.xnor) gradient_array_ongpu(original_input, l.batch*l.c*l.h*l.w, HARDTAN, state.delta);
		}
	}

//#endif    // CUDNN_HALF

	}
#else    // CUDNN
	if (l.batch_normalize)
	{
		backward_batchnorm_layer_gpu(l, state);
	}

	int m = l.n / l.groups;
	int n = l.size*l.size*l.c / l.groups;
	int k = l.out_w*l.out_h;

	int i, j;
	for(i = 0; i < l.batch; ++i)
	{
		for (j = 0; j < l.groups; ++j)
		{
			float * a = l.delta_gpu + (i*l.groups + j)*m*k;
			float * b = state.workspace;
			float * c = l.weight_updates_gpu + j*l.nweights / l.groups;

			float *im = state.input + (i*l.groups + j)*l.c / l.groups*l.h*l.w;

			if (!state.net.adversarial && !l.train_only_bn)
			{
				//im2col_ongpu(im, l.c / l.groups, l.h, l.w, l.size, l.stride, l.pad, state.workspace);
				im2col_gpu_ext(im,          // input
					l.c / l.groups,         // input channels
					l.h, l.w,               // input size (h, w)
					l.size, l.size,         // kernel size (h, w)
					l.pad * l.dilation, l.pad * l.dilation,   // padding (h, w)
					l.stride_y, l.stride_x,     // stride (h, w)
					l.dilation, l.dilation, // dilation (h, w)
					state.workspace);       // output
				//gemm_ongpu(0, 1, m, n, k, 1, a + i*m*k, k, b, k, 1, c, n);
				gemm_ongpu(0, 1, m, n, k, 1, a, k, b, k, 1, c, n);
			}

			if (state.delta)
			{
				if (l.binary || l.xnor) swap_binary(&l);
				float * a = l.weights_gpu + j*l.nweights / l.groups;
				float * b = l.delta_gpu + (i*l.groups + j)*m*k;
				float * c = state.workspace;

				//gemm_ongpu(1, 0, n, k, m, 1, a, n, b + i*k*m, k, 0, c, k);
				gemm_ongpu(1, 0, n, k, m, 1, a, n, b, k, 0, c, k);

				float *delta = state.delta + (i*l.groups + j)*l.c / l.groups*l.h*l.w;

				//col2im_ongpu(state.workspace, l.c / l.groups, l.h, l.w, l.size, l.stride, l.pad, delta);
				col2im_gpu_ext(
					state.workspace,        // input
					l.c / l.groups,         // input channels
					l.h, l.w,               // input size (h, w)
					l.size, l.size,         // kernel size (h, w)
					l.pad * l.dilation, l.pad * l.dilation,   // padding size (h, w)
					l.stride_y, l.stride_x,     // stride size (h, w)
					l.dilation, l.dilation, // dilation size (h, w)
					delta);                 // output (delta)

				if (l.binary || l.xnor)
				{
					swap_binary(&l);
				}
				if (l.xnor)
				{
					gradient_array_ongpu(original_input + i*l.c*l.h*l.w, l.c*l.h*l.w, HARDTAN, state.delta + i*l.c*l.h*l.w);
				}
			}
		}
	}
#endif
	if (state.net.try_fix_nan)
	{
		if (state.delta)
		{
			reset_nan_and_inf(state.delta, l.inputs * l.batch);
		}
		int size = l.nweights;
		reset_nan_and_inf(l.weight_updates_gpu, size);
		fix_nan_and_inf(l.weights_gpu, size);
	}


}

void backward_eml_convolutional_layer_gpu(Darknet::Layer & l, Darknet::NetworkState state)
{
	TAT(TATPARMS);

	const int total = l.outputs * l.batch;
	const float clamp = l.alpha > 0.0f ? l.alpha : 4.0f;
	const float eps = l.beta > 0.0f ? l.beta : 0.000001f;

	eml_convolutional_backward_kernel<<<cuda_gridsize(total), BLOCK, 0, get_cuda_stream()>>>(
		l.input_layer->output_gpu,
		l.self_layer->output_gpu,
		l.delta_gpu,
		total,
		clamp,
		eps,
		l.scale,
		l.input_layer->delta_gpu,
		l.self_layer->delta_gpu);
	CHECK_CUDA(cudaPeekAtLastError());

	if (l.shortcut && l.inputs == l.outputs && state.delta)
	{
		axpy_ongpu(total, 1.0f, l.delta_gpu, 1, state.delta, 1);
	}

	Darknet::NetworkState branch_state = state;
	backward_convolutional_layer_gpu(*(l.input_layer), branch_state);
	backward_convolutional_layer_gpu(*(l.self_layer), branch_state);
}

__global__ void calc_avg_activation_kernel(float *src, float *dst, int size, int channels, int batches)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int xy = i % size;
	int b = i / size;

	if (i < size*batches) {
		dst[i] = 0;
		for (int c = 0; c < channels; ++c) {
			dst[i] += src[xy + size*(c + channels*b)];
		}
		dst[i] = dst[i] / channels;
	}
}

void calc_avg_activation_gpu(float *src, float *dst, int size, int channels, int batches)
{
	TAT(TATPARMS);

	const int num_blocks = get_number_of_blocks(size*batches, BLOCK);

	calc_avg_activation_kernel <<<num_blocks, BLOCK, 0, get_cuda_stream() >>> (src, dst, size, channels, batches);
}


__global__ void assisted_activation_kernel(float alpha, float *output, float *gt_gpu, float *a_avg_gpu, int size, int channels, int batches)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int xy = i % size;
	int b = i / size;

	if (b < batches)
	{
		for (int c = 0; c < channels; ++c)
		{
			output[xy + size*(c + channels*b)] += alpha * gt_gpu[i] * a_avg_gpu[i];
		}
	}
}

void assisted_activation_gpu(float alpha, float *output, float *gt_gpu, float *a_avg_gpu, int size, int channels, int batches)
{
	TAT(TATPARMS);

	const int num_blocks = get_number_of_blocks(size*batches, BLOCK);

	assisted_activation_kernel <<<num_blocks, BLOCK, 0, get_cuda_stream() >>> (alpha, output, gt_gpu, a_avg_gpu, size, channels, batches);
}


__global__ void assisted_activation2_kernel(float alpha, float *output, float *gt_gpu, float *a_avg_gpu, int size, int channels, int batches)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int xy = i % size;
	int b = i / size;
	float beta = 1 - alpha;

	if (b < batches) {
		for (int c = 0; c < channels; ++c) {
			if(gt_gpu[i] == 0)
				output[xy + size*(c + channels*b)] *= beta;

		}
	}
}

void assisted_activation2_gpu(float alpha, float *output, float *gt_gpu, float *a_avg_gpu, int size, int channels, int batches)
{
	TAT(TATPARMS);

	const int num_blocks = get_number_of_blocks(size*batches, BLOCK);

	assisted_activation2_kernel <<<num_blocks, BLOCK, 0, get_cuda_stream() >>> (alpha, output, gt_gpu, a_avg_gpu, size, channels, batches);
}

void assisted_excitation_forward_gpu(Darknet::Layer & l, Darknet::NetworkState state)
{
	TAT(TATPARMS);

	const int iteration_num = get_current_iteration(state.net); //(*state.net.seen) / (state.net.batch*state.net.subdivisions);

	float alpha = (1 + cos(3.141592 * iteration_num / state.net.max_batches)) / 2;

	if (l.assisted_excitation == 1)
	{
		if (iteration_num > state.net.max_batches / 2) return;
	}
	else
	{
		if (iteration_num < state.net.burn_in) return;
		else
			if (iteration_num > l.assisted_excitation) return;
		else
			alpha = (1 + cos(3.141592 * iteration_num / (state.net.burn_in + l.assisted_excitation))) / 2; // from 1 to 0
	}

	float *a_avg = (float *)calloc(l.out_w * l.out_h * l.batch, sizeof(float));
	float *gt = (float *)calloc(l.out_w * l.out_h * l.batch, sizeof(float));

	int b;
	int w, h;

	l.max_boxes = state.net.num_boxes;
	l.truths = l.max_boxes*(4 + 1);

	int num_truth = l.batch*l.truths;
	float *truth_cpu = (float *)calloc(num_truth, sizeof(float));
	cuda_pull_array(state.truth, truth_cpu, num_truth);

	for (b = 0; b < l.batch; ++b)
	{
		// calculate G
		int t;
		for (t = 0; t < state.net.num_boxes; ++t) {
			Darknet::Box truth = float_to_box_stride(truth_cpu + t*(4 + 1) + b*l.truths, 1);
			if (!truth.x) break;  // continue;
			float beta = 0;
			//float beta = 1 - alpha; // from 0 to 1
			float dw = (1 - truth.w) * beta;
			float dh = (1 - truth.h) * beta;

			int left = floorf((truth.x - (dw + truth.w) / 2) * l.out_w);
			int right = ceilf((truth.x + (dw + truth.w) / 2) * l.out_w);
			int top = floorf((truth.y - (dh + truth.h) / 2) * l.out_h);
			int bottom = ceilf((truth.y + (dh + truth.h) / 2) * l.out_h);
			if (left < 0) left = 0;
			if (top < 0) top = 0;
			if (right > l.out_w) right = l.out_w;
			if (bottom > l.out_h) bottom = l.out_h;

			for (w = left; w <= right; w++) {
				for (h = top; h < bottom; h++) {
					gt[w + l.out_w * h + l.out_w*l.out_h*b] = 1;
				}
			}
		}
	}

	cuda_push_array(l.gt_gpu, gt, l.out_w * l.out_h * l.batch);

	// calc avg_output on GPU - for whole batch
	calc_avg_activation_gpu(l.output_gpu, l.a_avg_gpu, l.out_w * l.out_h, l.out_c, l.batch);

	// calc new output
	assisted_activation_gpu(alpha, l.output_gpu, l.gt_gpu, l.a_avg_gpu, l.out_w * l.out_h, l.out_c, l.batch);

	if (0)   // visualize ground truth
	{
		cuda_pull_array(l.output_gpu, l.output, l.outputs * l.batch);
		CHECK_CUDA(cudaStreamSynchronize(get_cuda_stream()));

		for (b = 0; b < l.batch; ++b)
		{
			*cfg_and_state.output << "Assisted Excitation alpha = " << alpha << std::endl;
			Darknet::Image img = Darknet::float_to_image(l.out_w, l.out_h, 1, &gt[l.out_w*l.out_h*b]);
			char buff[100];
			sprintf(buff, "a_excitation_gt_%d", b);
			show_image_cv(img, buff);

			//image img2 = float_to_image(l.out_w, l.out_h, 1, &l.output[l.out_w*l.out_h*l.out_c*b]);
			Darknet::Image img2 = Darknet::float_to_image_scaled(l.out_w, l.out_h, 1, &l.output[l.out_w*l.out_h*l.out_c*b]);
			char buff2[100];
			sprintf(buff2, "a_excitation_output_%d", b);
			show_image_cv(img2, buff2);

			cv::waitKey(5);
		}
		cv::waitKey(0);
	}

	free(truth_cpu);
	free(gt);
	free(a_avg);
}

void pull_convolutional_layer(Darknet::Layer & l)
{
	TAT(TATPARMS);

	cuda_pull_array_async(l.weights_gpu, l.weights, l.nweights);
	cuda_pull_array_async(l.biases_gpu, l.biases, l.n);
	if (l.weight_updates_gpu) cuda_pull_array_async(l.weight_updates_gpu, l.weight_updates, l.nweights);
	if (l.bias_updates_gpu) cuda_pull_array_async(l.bias_updates_gpu, l.bias_updates, l.n);
	if (l.batch_normalize){
		cuda_pull_array_async(l.scales_gpu, l.scales, l.n);
		cuda_pull_array_async(l.rolling_mean_gpu, l.rolling_mean, l.n);
		cuda_pull_array_async(l.rolling_variance_gpu, l.rolling_variance, l.n);
	}
	if (l.adam){
		cuda_pull_array_async(l.m_gpu, l.m, l.nweights);
		cuda_pull_array_async(l.v_gpu, l.v, l.nweights);
	}
	CHECK_CUDA(cudaPeekAtLastError());
	CHECK_CUDA(cudaStreamSynchronize(get_cuda_stream()));
}

void push_convolutional_layer(Darknet::Layer & l)
{
	TAT(TATPARMS);

	cuda_push_array(l.weights_gpu, l.weights, l.nweights);
#ifdef CUDNN_HALF
	assert(l.nweights > 0);
	cuda_convert_f32_to_cudnn_16bit(l.weights_gpu, l.nweights, l.weights_gpu16, l.cudnn_16bit_mode);
#endif
	cuda_push_array(l.biases_gpu, l.biases, l.n);
	if (l.train) {
		cuda_push_array(l.weight_updates_gpu, l.weight_updates, l.nweights);
		cuda_push_array(l.bias_updates_gpu, l.bias_updates, l.n);
	}
	if (l.batch_normalize){
		cuda_push_array(l.scales_gpu, l.scales, l.n);
		cuda_push_array(l.rolling_mean_gpu, l.rolling_mean, l.n);
		cuda_push_array(l.rolling_variance_gpu, l.rolling_variance, l.n);
	}
	if (l.adam){
		cuda_push_array(l.m_gpu, l.m, l.nweights);
		cuda_push_array(l.v_gpu, l.v, l.nweights);
	}
	CHECK_CUDA(cudaPeekAtLastError());
}

namespace
{
	void fp8_requantize_convolutional_training_weights(Darknet::Layer & l)
	{
#ifndef DARKNET_HAS_FP8
		(void)l;
#else
		if (!l.fp8_train_eligible ||
			l.weights_gpu == nullptr ||
			(l.weights_fp8_gpu == nullptr && l.weights_fp8_t_gpu == nullptr && l.weights_fp8_nhwc_gpu == nullptr) ||
			l.fp8_weight_scale_gpu == nullptr)
		{
			return;
		}

		const int filters = l.n;
		const int kernel = l.size * l.size * l.c;
		const int kernel_pad = Darknet::fp8_round_up_to_16(kernel);
		const int filters_pad = Darknet::fp8_round_up_to_16(filters);

		// all scale updates run on-device: no GPU->CPU amax pulls, no pipeline stalls
		if (l.fp8_amax_gpu && l.fp8_weight_scale_state_gpu)
		{
			Darknet::fp8_clear_amax_gpu(l.fp8_amax_gpu);
			Darknet::fp8_accumulate_amax_gpu(l.weights_gpu, l.nweights, l.fp8_amax_gpu);
		}

		// weights/input/dy delayed-scaling updates share one launch instead of three <<<1,1>>> ones
		Darknet::Fp8ScaleUpdate weight_update;
		if (l.fp8_amax_gpu && l.fp8_weight_scale_state_gpu)
		{
			weight_update = { l.fp8_amax_gpu, l.fp8_weight_scale_state_gpu, Darknet::fp8_format_max(Darknet::Fp8Format::E4M3), 0, l.fp8_weight_scale_gpu };
		}
		Darknet::Fp8ScaleUpdate input_update;
		if (l.fp8_input_amax_gpu && l.fp8_input_scale_state_gpu)
		{
			input_update = { l.fp8_input_amax_gpu, l.fp8_input_scale_state_gpu, Darknet::fp8_format_max(Darknet::Fp8Format::E4M3), 0, l.fp8_input_scale_gpu };
		}
		Darknet::Fp8ScaleUpdate dy_update;
		if (l.fp8_dy_amax_valid && l.fp8_dy_amax_gpu && l.fp8_dy_scale_state_gpu)
		{
			dy_update = { l.fp8_dy_amax_gpu, l.fp8_dy_scale_state_gpu, Darknet::fp8_format_max(Darknet::Fp8Format::E5M2), Darknet::kFp8DyScaleMargin, l.fp8_dy_scale_gpu };
		}
		Darknet::fp8_delayed_scale_update3_gpu(weight_update, input_update, dy_update);
		l.fp8_dy_amax_valid = 0;

		if (fp8_env_is_set("DARKNET_FP8_DISABLE_TRIPLE_WEIGHT_QUANT"))
		{
			if (l.weights_fp8_gpu)
			{
				Darknet::fp8_quantize_rowmajor_pad_cols_gpu(
					l.weights_gpu, filters, kernel, kernel_pad, l.fp8_weight_scale_gpu, l.weights_fp8_gpu);
			}
			if (l.weights_fp8_t_gpu)
			{
				Darknet::fp8_quantize_transpose_rowmajor_pad_cols_gpu(
					l.weights_gpu, filters, kernel, filters_pad, l.fp8_weight_scale_gpu, l.weights_fp8_t_gpu);
			}
			if (l.weights_fp8_nhwc_gpu)
			{
				Darknet::fp8_quantize_weights_krsc_gpu(
					l.weights_gpu,
					filters,
					l.c,
					l.size,
					l.size,
					l.fp8_weight_scale_gpu,
					l.weights_fp8_nhwc_gpu);
			}
		}
		else
		{
			// one read of the weights writes both GEMM operand layouts plus optional cuDNN KRSC
			Darknet::fp8_quantize_triple_layout_weights_gpu(
				l.weights_gpu,
				filters,
				l.c,
				l.size,
				l.size,
				kernel_pad,
				filters_pad,
				l.fp8_weight_scale_gpu,
				l.weights_fp8_gpu,
				l.weights_fp8_t_gpu,
				l.weights_fp8_nhwc_gpu);
		}
#endif
	}
}

void update_convolutional_layer_gpu(Darknet::Layer & l, int batch, float learning_rate_init, float momentum, float decay, float loss_scale)
{
	TAT(TATPARMS);

	if (l.deform)
	{
		if (l.rotate) rotate_weights_gpu(l.weight_updates_gpu, l.weight_deform_gpu, l.nweights, l.n, l.size, 1);
		else if (l.sway) sway_and_flip_weights_gpu(l.weight_updates_gpu, l.weight_deform_gpu, l.nweights, l.n, l.size, l.angle, 1);
		else if (l.stretch) stretch_weights_gpu(l.weight_updates_gpu, l.weight_deform_gpu, l.nweights, l.n, l.size, 0, 1);
		else if (l.stretch_sway) stretch_sway_flip_weights_gpu(l.weight_updates_gpu, l.weight_deform_gpu, l.nweights, l.n, l.size, l.angle, 1);

		reduce_and_expand_array_gpu(l.weight_deform_gpu, l.weight_updates_gpu, l.nweights, 4);
	}

	// Loss scale for Mixed-Precision on Tensor-Cores
	float learning_rate = learning_rate_init*l.learning_rate_scale / loss_scale;

	reset_nan_and_inf(l.weight_updates_gpu, l.nweights);
	fix_nan_and_inf(l.weights_gpu, l.nweights);

	// Gradient Centralization
	if (l.grad_centr && l.batch_normalize)
	{
		gradient_centralization_gpu(l.size, l.size, l.c / l.groups, l.n, l.weight_updates_gpu);
	}

	if (l.adam)
	{
		adam_update_gpu(l.weights_gpu, l.weight_updates_gpu, l.m_gpu, l.v_gpu, l.B1, l.B2, l.eps, decay, learning_rate, l.nweights, batch, l.t);

		adam_update_gpu(l.biases_gpu, l.bias_updates_gpu, l.bias_m_gpu, l.bias_v_gpu, l.B1, l.B2, l.eps, decay, learning_rate, l.n, batch, l.t);
		if (l.scales_gpu)
		{
			adam_update_gpu(l.scales_gpu, l.scale_updates_gpu, l.scale_m_gpu, l.scale_v_gpu, l.B1, l.B2, l.eps, decay, learning_rate, l.n, batch, l.t);
		}
	}
	else
	{
		float *old_weight_updates_gpu = l.weight_updates_gpu;
		const float update_rate = learning_rate / batch;

		if (l.reverse)
		{
			float clip = 0.0;
			float divider = 1.0;
			float abs_add = 1.0;
			mult_inverse_array_gpu(l.weight_updates_gpu, l.output_gpu, l.inputs*l.batch, l.reverse, divider, clip, abs_add);
			l.weight_updates_gpu = l.output_gpu;

			axpy_ongpu(l.nweights, -decay*batch*loss_scale, l.weights_gpu, 1, l.weight_updates_gpu, 1);
			axpy_ongpu(l.nweights, update_rate, l.weight_updates_gpu, 1, l.weights_gpu, 1);

			l.weight_updates_gpu = old_weight_updates_gpu;
			scal_ongpu(l.nweights, momentum, l.weight_updates_gpu, 1);
		}
		else
		{
			sgd_update_ongpu(l.nweights, l.weights_gpu, l.weight_updates_gpu, update_rate, momentum, -decay*batch*loss_scale);
		}

		sgd_update_ongpu(l.n, l.biases_gpu, l.bias_updates_gpu, update_rate, momentum, 0.0f);

		if (l.scales_gpu) {
			sgd_update_ongpu(l.n, l.scales_gpu, l.scale_updates_gpu, update_rate, momentum, 0.0f);
		}
	}

	if (l.deform)
	{
		expand_array_gpu(l.weights_gpu, l.weight_deform_gpu, l.nweights, 4);

		if (l.rotate) rotate_weights_gpu(l.weight_deform_gpu, l.weights_gpu, l.nweights, l.n, l.size, 0);
		else if (l.sway) sway_and_flip_weights_gpu(l.weight_deform_gpu, l.weights_gpu, l.nweights, l.n, l.size, l.angle, 0);
		else if (l.stretch) stretch_weights_gpu(l.weight_deform_gpu, l.weights_gpu, l.nweights, l.n, l.size, 0, 0);
		else if (l.stretch_sway) stretch_sway_flip_weights_gpu(l.weight_deform_gpu, l.weights_gpu, l.nweights, l.n, l.size, l.angle, 0);
	}

	if (l.clip)
	{
		constrain_ongpu(l.nweights, l.clip, l.weights_gpu, 1);
	}

	fp8_requantize_convolutional_training_weights(l);
}

void update_eml_convolutional_layer_gpu(Darknet::Layer & l, int batch, float learning_rate, float momentum, float decay, float loss_scale)
{
	TAT(TATPARMS);

	update_convolutional_layer_gpu(*(l.input_layer), batch, learning_rate, momentum, decay, loss_scale);
	update_convolutional_layer_gpu(*(l.self_layer), batch, learning_rate, momentum, decay, loss_scale);
}
