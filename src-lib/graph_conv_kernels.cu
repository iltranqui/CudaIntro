#include "darknet_internal.hpp"
#include "graph_conv_layer.hpp"
#include "batchnorm_layer.hpp"
#include "activations.hpp"
#include "blas.hpp"
#include "convolutional_layer.hpp"
#include "gemm.hpp"
#include "dark_cuda.hpp"

#include <cfloat>
#include <cstdio>

#if defined(CUDNN) && defined(CUDNN_HALF) && defined(DARKNET_GPU_CUDA) && !defined(DARKNET_GPU_ROCM)
#define DARKNET_GRAPH_USE_CUDNN_HALF 1
#else
#define DARKNET_GRAPH_USE_CUDNN_HALF 0
#endif

#if DARKNET_GRAPH_USE_CUDNN_HALF
#include <cuda_fp16.h>
#endif

#ifdef DARKNET_GPU

namespace
{
	constexpr int GRAPH_LOCAL_REF_CACHE_MAX = 32;
	constexpr int GRAPH_LOCAL_LOGITS_MAX = 49;

#ifdef DARKNET_GPU_CUDA
	struct ScopedTensorOpMath
	{
		cublasHandle_t handle = nullptr;
		cublasMath_t previous_math = CUBLAS_DEFAULT_MATH;
		bool active = false;

		explicit ScopedTensorOpMath(bool enable)
		{
			if (!enable)
			{
				return;
			}

			handle = blas_handle();
			CHECK_CUBLAS(cublasGetMathMode(handle, &previous_math));
			CHECK_CUBLAS(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH));
			active = true;
		}

		~ScopedTensorOpMath()
		{
			if (active)
			{
				CHECK_CUBLAS(cublasSetMathMode(handle, previous_math));
			}
		}
	};
#else
	struct ScopedTensorOpMath
	{
		explicit ScopedTensorOpMath(bool) {}
	};
#endif

	__host__ __device__ __forceinline__ size_t graph_temp_workspace_bytes(int batch, int c, int out_h, int out_w, int graph_use_self)
	{
		size_t size = static_cast<size_t>(batch) * c * out_h * out_w * sizeof(float);
		if (graph_use_self)
		{
			size *= 2;
		}
		return size;
	}

	__host__ __device__ __forceinline__ size_t graph_align_workspace_bytes(size_t size)
	{
		constexpr size_t alignment = 256;
		return (size + alignment - 1) & ~(alignment - 1);
	}

	__host__ __forceinline__ size_t graph_cudnn_workspace_offset(const Darknet::Layer & l)
	{
		return graph_align_workspace_bytes(graph_temp_workspace_bytes(l.batch, l.c, l.out_h, l.out_w, l.graph_use_self));
	}

	__host__ __forceinline__ size_t graph_cudnn_workspace_size(const Darknet::Layer & l)
	{
		const size_t offset = graph_cudnn_workspace_offset(l);
		return (l.workspace_size > offset) ? (l.workspace_size - offset) : 0;
	}

	__host__ __forceinline__ void *graph_cudnn_workspace_ptr(const Darknet::Layer & l, float *workspace)
	{
		const size_t workspace_size = graph_cudnn_workspace_size(l);
		if (workspace == nullptr || workspace_size == 0)
		{
			return nullptr;
		}
		return reinterpret_cast<void*>(reinterpret_cast<char*>(workspace) + graph_cudnn_workspace_offset(l));
	}

	__host__ __forceinline__ bool graph_pointwise_fast_path(const Darknet::Layer & l)
	{
		return l.size == 1 &&
			l.graph_edge_mode == 0 &&
			l.graph_use_self == 0 &&
			l.stride_x == 1 &&
			l.stride_y == 1 &&
			l.pad == 0 &&
			l.dilation == 1 &&
			l.out_h == l.h &&
			l.out_w == l.w;
	}

#if defined(CUDNN) && defined(CUDNN_HALF)
	__host__ bool ensure_graph_16bit_buffer(float **buffer, size_t *capacity, size_t required)
	{
		if (buffer == nullptr || capacity == nullptr || required == 0)
		{
			return false;
		}

		if (*capacity < required)
		{
			*capacity = required;
			if (*buffer) cuda_free(*buffer);
			CHECK_CUDA(cudaMalloc(reinterpret_cast<void **>(buffer), *capacity * sizeof(short)));
		}
		return *buffer != nullptr;
	}

	__host__ bool graph_projection_16bit_ready(const Darknet::Layer & l, const Darknet::NetworkState & state, bool need_backward_buffers)
	{
		if (!state.net.cudnn_half || state.net.cudnn_bf16)
		{
			return false;
		}
		if (state.net.input16_gpu == nullptr || state.net.output16_gpu == nullptr ||
			state.net.max_input16_size == nullptr || state.net.max_output16_size == nullptr)
		{
			return false;
		}
		if (l.groups != 1 || (l.graph_cpg % 8) != 0 || (l.graph_npg % 8) != 0)
		{
			return false;
		}
		if (l.weights_gpu16 == nullptr ||
			l.convDesc == nullptr || l.srcTensorDesc16 == nullptr || l.dstTensorDesc16 == nullptr ||
			l.dsrcTensorDesc16 == nullptr || l.ddstTensorDesc16 == nullptr ||
			l.weightDesc16 == nullptr || l.dweightDesc16 == nullptr)
		{
			return false;
		}
		if (need_backward_buffers && l.weight_updates_gpu16 == nullptr)
		{
			return false;
		}
		if (l.graph_use_self && l.graph_self_weights_gpu16 == nullptr)
		{
			return false;
		}
		if (need_backward_buffers && l.graph_use_self && l.graph_self_weight_updates_gpu16 == nullptr)
		{
			return false;
		}
		return true;
	}

	__host__ bool prepare_graph_projection_16bit_buffers(const Darknet::Layer & l, Darknet::NetworkState & state,
		float **input16, float **output16)
	{
		const size_t input16_size = static_cast<size_t>(l.batch) * l.c * l.out_h * l.out_w;
		const size_t output16_size = static_cast<size_t>(l.batch) * l.out_c * l.out_h * l.out_w;
		if (input16_size == 0 || output16_size == 0)
		{
			return false;
		}

		if (!ensure_graph_16bit_buffer(state.net.input16_gpu, state.net.max_input16_size, input16_size))
		{
			return false;
		}
		if (!ensure_graph_16bit_buffer(state.net.output16_gpu, state.net.max_output16_size, output16_size))
		{
			return false;
		}

		*input16 = *state.net.input16_gpu;
		*output16 = *state.net.output16_gpu;
		return *input16 != nullptr && *output16 != nullptr;
	}

	__host__ bool forward_graph_projection_cudnn_16bit(Darknet::Layer & l, Darknet::NetworkState state,
		float *projection_input)
	{
		if (!graph_projection_16bit_ready(l, state, false))
		{
			return false;
		}

		float *input16 = nullptr;
		float *output16 = nullptr;
		if (!prepare_graph_projection_16bit_buffers(l, state, &input16, &output16))
		{
			return false;
		}

		const size_t input_count = static_cast<size_t>(l.batch) * l.c * l.out_h * l.out_w;
		const size_t output_count = static_cast<size_t>(l.batch) * l.out_c * l.out_h * l.out_w;
		const float one = 1.0f;
		const float zero = 0.0f;
		const int mode = DARKNET_CUDNN_16BIT_HALF;
		void *workspace = graph_cudnn_workspace_ptr(l, state.workspace);
		const size_t workspace_size = graph_cudnn_workspace_size(l);

		cuda_convert_f32_to_cudnn_16bit(projection_input, input_count, input16, mode);
		cuda_convert_f32_to_cudnn_16bit(l.weights_gpu, l.nweights, l.weights_gpu16, mode);
		CHECK_CUDNN(cudnnConvolutionForward(cudnn_handle(),
			&one, l.srcTensorDesc16, input16, l.weightDesc16, l.weights_gpu16,
			l.convDesc, l.fw_algo16, workspace, workspace_size,
			&zero, l.dstTensorDesc16, output16));

		if (l.graph_use_self)
		{
			cuda_convert_f32_to_cudnn_16bit(l.graph_ref_gpu, input_count, input16, mode);
			cuda_convert_f32_to_cudnn_16bit(l.graph_self_weights_gpu, l.n * l.graph_cpg, l.graph_self_weights_gpu16, mode);
			CHECK_CUDNN(cudnnConvolutionForward(cudnn_handle(),
				&one, l.srcTensorDesc16, input16, l.weightDesc16, l.graph_self_weights_gpu16,
				l.convDesc, l.fw_algo16, workspace, workspace_size,
				&one, l.dstTensorDesc16, output16));
		}

		cuda_convert_cudnn_16bit_to_f32(output16, output_count, l.output_gpu, mode);
		return true;
	}

	__host__ bool backward_graph_projection_cudnn_16bit(Darknet::Layer & l, Darknet::NetworkState state,
		float *projection_input, float *d_agg_workspace, float *d_ref_workspace)
	{
		if (!graph_projection_16bit_ready(l, state, true))
		{
			return false;
		}

		float *input16 = nullptr;
		float *delta16 = nullptr;
		if (!prepare_graph_projection_16bit_buffers(l, state, &input16, &delta16))
		{
			return false;
		}

		const size_t input_count = static_cast<size_t>(l.batch) * l.c * l.out_h * l.out_w;
		const size_t output_count = static_cast<size_t>(l.batch) * l.out_c * l.out_h * l.out_w;
		const float one = 1.0f;
		const float zero = 0.0f;
		const int mode = DARKNET_CUDNN_16BIT_HALF;
		void *workspace = graph_cudnn_workspace_ptr(l, state.workspace);
		const size_t workspace_size = graph_cudnn_workspace_size(l);

		cuda_convert_f32_to_cudnn_16bit(l.delta_gpu, output_count, delta16, mode);

		cuda_convert_f32_to_cudnn_16bit(projection_input, input_count, input16, mode);
		cuda_convert_f32_to_cudnn_16bit(l.weight_updates_gpu, l.nweights, l.weight_updates_gpu16, mode);
		CHECK_CUDNN(cudnnConvolutionBackwardFilter(cudnn_handle(),
			&one, l.srcTensorDesc16, input16, l.ddstTensorDesc16, delta16,
			l.convDesc, l.bf_algo16, workspace, workspace_size,
			&one, l.dweightDesc16, l.weight_updates_gpu16));
		cuda_convert_cudnn_16bit_to_f32(l.weight_updates_gpu16, l.nweights, l.weight_updates_gpu, mode);

		cuda_convert_f32_to_cudnn_16bit(l.weights_gpu, l.nweights, l.weights_gpu16, mode);
		CHECK_CUDNN(cudnnConvolutionBackwardData(cudnn_handle(),
			&one, l.weightDesc16, l.weights_gpu16, l.ddstTensorDesc16, delta16,
			l.convDesc, l.bd_algo16, workspace, workspace_size,
			&zero, l.dsrcTensorDesc16, input16));
		cuda_convert_cudnn_16bit_to_f32(input16, input_count, d_agg_workspace, mode);

		if (l.graph_use_self)
		{
			cuda_convert_f32_to_cudnn_16bit(l.graph_ref_gpu, input_count, input16, mode);
			cuda_convert_f32_to_cudnn_16bit(l.graph_self_weight_updates_gpu, l.n * l.graph_cpg, l.graph_self_weight_updates_gpu16, mode);
			CHECK_CUDNN(cudnnConvolutionBackwardFilter(cudnn_handle(),
				&one, l.srcTensorDesc16, input16, l.ddstTensorDesc16, delta16,
				l.convDesc, l.bf_algo16, workspace, workspace_size,
				&one, l.dweightDesc16, l.graph_self_weight_updates_gpu16));
			cuda_convert_cudnn_16bit_to_f32(l.graph_self_weight_updates_gpu16, l.n * l.graph_cpg,
				l.graph_self_weight_updates_gpu, mode);

			cuda_convert_f32_to_cudnn_16bit(l.graph_self_weights_gpu, l.n * l.graph_cpg, l.graph_self_weights_gpu16, mode);
			CHECK_CUDNN(cudnnConvolutionBackwardData(cudnn_handle(),
				&one, l.weightDesc16, l.graph_self_weights_gpu16, l.ddstTensorDesc16, delta16,
				l.convDesc, l.bd_algo16, workspace, workspace_size,
				&zero, l.dsrcTensorDesc16, input16));
			cuda_convert_cudnn_16bit_to_f32(input16, input_count, d_ref_workspace, mode);
		}

		return true;
	}
#else
	__host__ bool forward_graph_projection_cudnn_16bit(Darknet::Layer &, Darknet::NetworkState, float *)
	{
		return false;
	}

	__host__ bool backward_graph_projection_cudnn_16bit(Darknet::Layer &, Darknet::NetworkState, float *, float *, float *)
	{
		return false;
	}
#endif

#if DARKNET_GRAPH_USE_CUDNN_HALF
	__host__ bool graph_mixed_16bit_ready(const Darknet::Layer & l, const Darknet::NetworkState & state)
	{
		if (!state.net.cudnn_half || state.net.cudnn_bf16)
		{
			return false;
		}
		if (l.graph_edge_mode != 1 || l.graph_edge_kernel_gpu16 == nullptr || state.input == nullptr)
		{
			return false;
		}
		if (state.net.input16_gpu == nullptr || state.net.output16_gpu == nullptr ||
			state.net.max_input16_size == nullptr || state.net.max_output16_size == nullptr)
		{
			return false;
		}
		// The mixed kernels use paired half math for the dot-product heavy loops.
		return l.graph_cpg > 0 && (l.graph_cpg % 2) == 0;
	}

	__host__ bool prepare_graph_forward_mixed_16bit(Darknet::Layer & l, Darknet::NetworkState & state, float **input16)
	{
		if (!graph_mixed_16bit_ready(l, state) || input16 == nullptr)
		{
			return false;
		}

		const size_t input_count = static_cast<size_t>(l.batch) * l.c * l.h * l.w;
		const size_t kernel_count = static_cast<size_t>(l.groups) * l.graph_k * (2 * l.graph_cpg);
		if (!ensure_graph_16bit_buffer(state.net.input16_gpu, state.net.max_input16_size, input_count))
		{
			return false;
		}

		const int mode = DARKNET_CUDNN_16BIT_HALF;
		cuda_convert_f32_to_cudnn_16bit(state.input, input_count, *state.net.input16_gpu, mode);
		cuda_convert_f32_to_cudnn_16bit(l.graph_edge_kernel_gpu, kernel_count, l.graph_edge_kernel_gpu16, mode);
		*input16 = *state.net.input16_gpu;
		return *input16 != nullptr;
	}

	__host__ bool prepare_graph_backward_mixed_16bit(Darknet::Layer & l, Darknet::NetworkState & state,
		float *d_agg_workspace, float **input16, float **d_agg16)
	{
		if (!graph_mixed_16bit_ready(l, state) || d_agg_workspace == nullptr || input16 == nullptr || d_agg16 == nullptr)
		{
			return false;
		}

		const size_t input_count = static_cast<size_t>(l.batch) * l.c * l.h * l.w;
		const size_t feature_count = static_cast<size_t>(l.batch) * l.c * l.out_h * l.out_w;
		const size_t kernel_count = static_cast<size_t>(l.groups) * l.graph_k * (2 * l.graph_cpg);
		if (!ensure_graph_16bit_buffer(state.net.input16_gpu, state.net.max_input16_size, input_count))
		{
			return false;
		}
		if (!ensure_graph_16bit_buffer(state.net.output16_gpu, state.net.max_output16_size, feature_count))
		{
			return false;
		}

		const int mode = DARKNET_CUDNN_16BIT_HALF;
		cuda_convert_f32_to_cudnn_16bit(state.input, input_count, *state.net.input16_gpu, mode);
		cuda_convert_f32_to_cudnn_16bit(d_agg_workspace, feature_count, *state.net.output16_gpu, mode);
		cuda_convert_f32_to_cudnn_16bit(l.graph_edge_kernel_gpu, kernel_count, l.graph_edge_kernel_gpu16, mode);
		*input16 = *state.net.input16_gpu;
		*d_agg16 = *state.net.output16_gpu;
		return *input16 != nullptr && *d_agg16 != nullptr;
	}
#else
	__host__ bool prepare_graph_forward_mixed_16bit(Darknet::Layer &, Darknet::NetworkState &, float **)
	{
		return false;
	}

	__host__ bool prepare_graph_backward_mixed_16bit(Darknet::Layer &, Darknet::NetworkState &, float *, float **, float **)
	{
		return false;
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
			const std::string layer_label = Darknet::layer_type_diagnostic_label(Darknet::ELayerType::GRAPH_CONV);
			std::printf("[%s layer] NaN/Inf detected at layer %d, step: %s\n", layer_label.c_str(), layer_idx, step_name);
		}
	}

	// ── GPU index helper functions ────────────────────────────────────────────

	// These mirror the CPU helpers in graph_conv_layer.cpp but take explicit
	// dimension parameters instead of a Layer& object.  CUDA kernels run on
	// the device and cannot dereference host-side C++ objects, so all needed
	// metadata must be passed as plain scalar arguments.
	// The index formulas are identical to the CPU versions — see that file for
	// a full explanation of each tensor shape.
	// ──────────────────────────────────────────────────────────────────────────

	// Input tensor: [B, C, H, W]
	__device__ __forceinline__ int input_index_gpu(int c, int h, int w, int b, int ch, int y, int x)
	{
		return ((b * c + ch) * h + y) * w + x;
	}

	// Graph intermediate buffers: [B, C, out_H, out_W]  (graph_ref / graph_agg)
	__device__ __forceinline__ int graph_feature_index_gpu(int c, int out_h, int out_w, int b, int ch, int y, int x)
	{
		return ((b * c + ch) * out_h + y) * out_w + x;
	}

	// Edge buffers: [B, groups, out_H, out_W, K²]  (graph_alpha / graph_valid)
	__device__ __forceinline__ int graph_edge_index_gpu(int groups, int out_h, int out_w, int graph_k, int b, int g, int y, int x, int k)
	{
		return ((((b * groups + g) * out_h + y) * out_w + x) * graph_k + k);
	}

	// Weight matrix: [groups, npg, cpg] — returns flat offset of row start.
	// Marked __host__ __device__ so it can also be called from the GEMM launch loops on the host.
	__host__ __device__ __forceinline__ int weight_row_index_gpu(int graph_npg, int graph_cpg, int g, int oc_local)
	{
		return (g * graph_npg + oc_local) * graph_cpg;
	}

	// Bounds check: returns true iff (y, x) is inside the [0,h) × [0,w) input grid.
	__device__ __forceinline__ bool valid_input_coord_gpu(int h, int w, int y, int x)
	{
		return y >= 0 && y < h && x >= 0 && x < w;
	}

	// Map output position (oy, ox) to the center input pixel of its receptive field.
	// For a 3×3 kernel: center=1, so this is the pixel at k=4 in the K² neighbor loop.
	__device__ __forceinline__ void graph_reference_coord_gpu(int size, int stride_x, int stride_y, int dilation, int pad, int oy, int ox, int & ref_y, int & ref_x)
	{
		const int center = size / 2;
		ref_y = oy * stride_y - pad + center * dilation;
		ref_x = ox * stride_x - pad + center * dilation;
	}

	// Map flat neighbor index k ∈ [0, size²) to its input pixel coordinate.
	// k is row-major: ky = k / size (row), kx = k % size (column).
	__device__ __forceinline__ void graph_neighbor_coord_gpu(int size, int stride_x, int stride_y, int dilation, int pad, int oy, int ox, int k, int & iy, int & ix)
	{
		const int ky = k / size;   // row in the KxK grid
		const int kx = k % size;   // column in the KxK grid
		iy = oy * stride_y - pad + ky * dilation;
		ix = ox * stride_x - pad + kx * dilation;
	}

	__device__ __forceinline__ float cached_ref_value_gpu(const float *input, int c, int h, int w,
		int b, int input_channel_base, int ci, int ref_y, int ref_x, bool ref_valid,
		bool use_ref_cache, const float *ref_cache)
	{
		if (!ref_valid)
		{
			return 0.0f;
		}

		if (use_ref_cache)
		{
			return ref_cache[ci];
		}

		return input[input_index_gpu(c, h, w, b, input_channel_base + ci, ref_y, ref_x)];
	}

	__device__ __forceinline__ float graph_attention_logit_gpu(const float *input, const float *edge_kernel,
		const float *edge_biases, int c, int h, int w, int b, int g, int input_channel_base,
		int graph_cpg, int graph_k, int k, int ref_y, int ref_x, bool ref_valid,
		bool use_ref_cache, const float *ref_cache, int iy, int ix)
	{
		const int edge_kernel_width = 2 * graph_cpg;
		const int kernel_base = (g * graph_k + k) * edge_kernel_width;
		float logit = edge_biases[g * graph_k + k];
		for (int ci = 0; ci < graph_cpg; ++ci)
		{
			const float ref_value = cached_ref_value_gpu(input, c, h, w, b, input_channel_base, ci,
				ref_y, ref_x, ref_valid, use_ref_cache, ref_cache);
			const float neighbor_value = input[input_index_gpu(c, h, w, b, input_channel_base + ci, iy, ix)];
			logit += edge_kernel[kernel_base + ci] * ref_value;
			logit += edge_kernel[kernel_base + graph_cpg + ci] * neighbor_value;
		}
		return logit;
	}

#if DARKNET_GRAPH_USE_CUDNN_HALF
	__device__ __forceinline__ __half graph_zero_half_gpu()
	{
		return __float2half_rn(0.0f);
	}

	__device__ __forceinline__ float graph_h2f_gpu(const __half value)
	{
		return __half2float(value);
	}

	__device__ __forceinline__ float graph_hmul2_sum_gpu(__half a0, __half a1, const __half *weights)
	{
		const __half2 a = __halves2half2(a0, a1);
		const __half2 w = *reinterpret_cast<const __half2 *>(weights);
		const float2 product = __half22float2(__hmul2(a, w));
		return product.x + product.y;
	}

	__device__ __forceinline__ __half cached_ref_value_half_gpu(const __half *input, int c, int h, int w,
		int b, int input_channel_base, int ci, int ref_y, int ref_x, bool ref_valid,
		bool use_ref_cache, const __half *ref_cache)
	{
		if (!ref_valid)
		{
			return graph_zero_half_gpu();
		}

		if (use_ref_cache)
		{
			return ref_cache[ci];
		}

		return input[input_index_gpu(c, h, w, b, input_channel_base + ci, ref_y, ref_x)];
	}

	__device__ __forceinline__ float graph_attention_logit_half_gpu(const __half *input, const __half *edge_kernel,
		const float *edge_biases, int c, int h, int w, int b, int g, int input_channel_base,
		int graph_cpg, int graph_k, int k, int ref_y, int ref_x, bool ref_valid,
		bool use_ref_cache, const __half *ref_cache, int iy, int ix)
	{
		const int edge_kernel_width = 2 * graph_cpg;
		const int kernel_base = (g * graph_k + k) * edge_kernel_width;
		float logit = edge_biases[g * graph_k + k];
		for (int ci = 0; ci < graph_cpg; ci += 2)
		{
			const __half ref0 = cached_ref_value_half_gpu(input, c, h, w, b, input_channel_base, ci,
				ref_y, ref_x, ref_valid, use_ref_cache, ref_cache);
			const __half ref1 = cached_ref_value_half_gpu(input, c, h, w, b, input_channel_base, ci + 1,
				ref_y, ref_x, ref_valid, use_ref_cache, ref_cache);
			const __half neighbor0 = input[input_index_gpu(c, h, w, b, input_channel_base + ci, iy, ix)];
			const __half neighbor1 = input[input_index_gpu(c, h, w, b, input_channel_base + ci + 1, iy, ix)];
			logit += graph_hmul2_sum_gpu(ref0, ref1, edge_kernel + kernel_base + ci);
			logit += graph_hmul2_sum_gpu(neighbor0, neighbor1, edge_kernel + kernel_base + graph_cpg + ci);
		}
		return logit;
	}

	__device__ __forceinline__ float graph_d_alpha_half_gpu(const __half *input, const __half *d_agg,
		int c, int h, int w, int out_h, int out_w, int b, int input_channel_base,
		int graph_cpg, int oy, int ox, int iy, int ix)
	{
		float d_alpha = 0.0f;
		for (int ci = 0; ci < graph_cpg; ci += 2)
		{
			const int agg_idx0 = graph_feature_index_gpu(c, out_h, out_w, b, input_channel_base + ci, oy, ox);
			const int agg_idx1 = graph_feature_index_gpu(c, out_h, out_w, b, input_channel_base + ci + 1, oy, ox);
			const __half d_agg0 = d_agg[agg_idx0];
			const __half d_agg1 = d_agg[agg_idx1];
			const __half neighbor0 = input[input_index_gpu(c, h, w, b, input_channel_base + ci, iy, ix)];
			const __half neighbor1 = input[input_index_gpu(c, h, w, b, input_channel_base + ci + 1, iy, ix)];
			const __half2 upstream = __halves2half2(d_agg0, d_agg1);
			const __half2 neighbor = __halves2half2(neighbor0, neighbor1);
			const float2 product = __half22float2(__hmul2(upstream, neighbor));
			d_alpha += product.x + product.y;
		}
		return d_alpha;
	}

	__global__ void graph_conv_forward_kernel_mixed_16bit(const __half *input, const __half *edge_kernel,
		const float *edge_biases, float *graph_ref, float *graph_agg, float *graph_alpha,
		float *graph_valid, int batch, int c, int h, int w, int out_h,
		int out_w, int groups, int graph_cpg, int size, int stride_x, int stride_y,
		int dilation, int pad, int graph_k, int store_ref, int store_training_edges)
	{
		const int index = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
		const int total = batch * groups * out_h * out_w;
		if (index >= total)
		{
			return;
		}

		const int ox = index % out_w;
		const int oy = (index / out_w) % out_h;
		const int g = (index / (out_w * out_h)) % groups;
		const int b = index / (groups * out_h * out_w);
		const int input_channel_base = g * graph_cpg;

		int ref_y = 0;
		int ref_x = 0;
		graph_reference_coord_gpu(size, stride_x, stride_y, dilation, pad, oy, ox, ref_y, ref_x);
		const bool ref_valid = valid_input_coord_gpu(h, w, ref_y, ref_x);
		const bool use_ref_cache = graph_cpg <= GRAPH_LOCAL_REF_CACHE_MAX;
		__half ref_cache[GRAPH_LOCAL_REF_CACHE_MAX];

		for (int ci = 0; ci < graph_cpg; ++ci)
		{
			__half ref_value = graph_zero_half_gpu();
			if (ref_valid)
			{
				ref_value = input[input_index_gpu(c, h, w, b, input_channel_base + ci, ref_y, ref_x)];
			}
			if (use_ref_cache)
			{
				ref_cache[ci] = ref_value;
			}
			if (store_ref)
			{
				graph_ref[graph_feature_index_gpu(c, out_h, out_w, b, input_channel_base + ci, oy, ox)] =
					graph_h2f_gpu(ref_value);
			}
			graph_agg[graph_feature_index_gpu(c, out_h, out_w, b, input_channel_base + ci, oy, ox)] = 0.0f;
		}

		float max_logit = -FLT_MAX;
		int valid_count = 0;
		float logits_cache[GRAPH_LOCAL_LOGITS_MAX];
		const bool use_local_logits = graph_k <= GRAPH_LOCAL_LOGITS_MAX;
		for (int k = 0; k < graph_k; ++k)
		{
			int iy = 0;
			int ix = 0;
			graph_neighbor_coord_gpu(size, stride_x, stride_y, dilation, pad, oy, ox, k, iy, ix);

			const bool valid = valid_input_coord_gpu(h, w, iy, ix);
			if (store_training_edges)
			{
				const int edge_idx = graph_edge_index_gpu(groups, out_h, out_w, graph_k, b, g, oy, ox, k);
				graph_valid[edge_idx] = valid ? 1.0f : 0.0f;
				graph_alpha[edge_idx] = 0.0f;
			}
			if (!valid)
			{
				continue;
			}

			++valid_count;
			const float logit = graph_attention_logit_half_gpu(input, edge_kernel, edge_biases, c, h, w,
				b, g, input_channel_base, graph_cpg, graph_k, k, ref_y, ref_x, ref_valid,
				use_ref_cache, ref_cache, iy, ix);
			if (use_local_logits)
			{
				logits_cache[k] = logit;
			}
			if (logit > max_logit)
			{
				max_logit = logit;
			}
		}

		if (valid_count == 0)
		{
			return;
		}

		float denom = 0.0f;
		for (int k = 0; k < graph_k; ++k)
		{
			int iy = 0;
			int ix = 0;
			graph_neighbor_coord_gpu(size, stride_x, stride_y, dilation, pad, oy, ox, k, iy, ix);
			if (!valid_input_coord_gpu(h, w, iy, ix))
			{
				continue;
			}

			const float logit = use_local_logits
				? logits_cache[k]
				: graph_attention_logit_half_gpu(input, edge_kernel, edge_biases, c, h, w, b, g,
					input_channel_base, graph_cpg, graph_k, k, ref_y, ref_x, ref_valid,
					use_ref_cache, ref_cache, iy, ix);
			denom += expf(logit - max_logit);
		}

		for (int k = 0; k < graph_k; ++k)
		{
			int iy = 0;
			int ix = 0;
			graph_neighbor_coord_gpu(size, stride_x, stride_y, dilation, pad, oy, ox, k, iy, ix);
			if (!valid_input_coord_gpu(h, w, iy, ix))
			{
				continue;
			}

			const float logit = use_local_logits
				? logits_cache[k]
				: graph_attention_logit_half_gpu(input, edge_kernel, edge_biases, c, h, w, b, g,
					input_channel_base, graph_cpg, graph_k, k, ref_y, ref_x, ref_valid,
					use_ref_cache, ref_cache, iy, ix);
			const float alpha = expf(logit - max_logit) / denom;
			if (store_training_edges)
			{
				graph_alpha[graph_edge_index_gpu(groups, out_h, out_w, graph_k, b, g, oy, ox, k)] = alpha;
			}

			for (int ci = 0; ci < graph_cpg; ++ci)
			{
				const int agg_idx = graph_feature_index_gpu(c, out_h, out_w, b, input_channel_base + ci, oy, ox);
				const __half neighbor_value = input[input_index_gpu(c, h, w, b, input_channel_base + ci, iy, ix)];
				graph_agg[agg_idx] += alpha * graph_h2f_gpu(neighbor_value);
			}
		}
	}

	__global__ void graph_conv_backward_kernel_mixed_16bit(const __half *input, const __half *edge_kernel,
		const float *graph_alpha, const float *graph_valid, const __half *d_agg,
		const float *d_ref_self, float *edge_kernel_updates, float *edge_bias_updates,
		float *input_delta, int batch, int c, int h, int w, int out_h, int out_w,
		int groups, int graph_cpg, int size, int stride_x, int stride_y, int dilation,
		int pad, int graph_k)
	{
		const int index = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
		const int total = batch * groups * out_h * out_w;
		if (index >= total)
		{
			return;
		}

		const int ox = index % out_w;
		const int oy = (index / out_w) % out_h;
		const int g = (index / (out_w * out_h)) % groups;
		const int b = index / (groups * out_h * out_w);
		const int input_channel_base = g * graph_cpg;
		const int edge_kernel_width = 2 * graph_cpg;

		int ref_y = 0;
		int ref_x = 0;
		graph_reference_coord_gpu(size, stride_x, stride_y, dilation, pad, oy, ox, ref_y, ref_x);
		const bool ref_valid = valid_input_coord_gpu(h, w, ref_y, ref_x);

		if (ref_valid && input_delta != nullptr && d_ref_self != nullptr)
		{
			for (int ci = 0; ci < graph_cpg; ++ci)
			{
				const int ref_feature_idx = graph_feature_index_gpu(c, out_h, out_w, b, input_channel_base + ci, oy, ox);
				const float d_ref = d_ref_self[ref_feature_idx];
				input_delta[input_index_gpu(c, h, w, b, input_channel_base + ci, ref_y, ref_x)] += d_ref;
			}
		}

		float sum_term = 0.0f;
		for (int k = 0; k < graph_k; ++k)
		{
			const int edge_idx = graph_edge_index_gpu(groups, out_h, out_w, graph_k, b, g, oy, ox, k);
			if (graph_valid[edge_idx] <= 0.5f)
			{
				continue;
			}

			int iy = 0;
			int ix = 0;
			graph_neighbor_coord_gpu(size, stride_x, stride_y, dilation, pad, oy, ox, k, iy, ix);
			const float d_alpha = graph_d_alpha_half_gpu(input, d_agg, c, h, w, out_h, out_w,
				b, input_channel_base, graph_cpg, oy, ox, iy, ix);
			sum_term += graph_alpha[edge_idx] * d_alpha;
		}

		for (int k = 0; k < graph_k; ++k)
		{
			const int edge_idx = graph_edge_index_gpu(groups, out_h, out_w, graph_k, b, g, oy, ox, k);
			if (graph_valid[edge_idx] <= 0.5f)
			{
				continue;
			}

			int iy = 0;
			int ix = 0;
			graph_neighbor_coord_gpu(size, stride_x, stride_y, dilation, pad, oy, ox, k, iy, ix);

			const float d_alpha = graph_d_alpha_half_gpu(input, d_agg, c, h, w, out_h, out_w,
				b, input_channel_base, graph_cpg, oy, ox, iy, ix);
			const float d_logit = graph_alpha[edge_idx] * (d_alpha - sum_term);
			const int kernel_base = (g * graph_k + k) * edge_kernel_width;
			atomicAdd(edge_bias_updates + g * graph_k + k, d_logit);

			for (int ci = 0; ci < graph_cpg; ++ci)
			{
				const int agg_idx = graph_feature_index_gpu(c, out_h, out_w, b, input_channel_base + ci, oy, ox);
				const int nbr_idx = input_index_gpu(c, h, w, b, input_channel_base + ci, iy, ix);
				const __half d_agg_value = d_agg[agg_idx];
				const __half ref_value = ref_valid
					? input[input_index_gpu(c, h, w, b, input_channel_base + ci, ref_y, ref_x)]
					: graph_zero_half_gpu();
				const __half neighbor_value = input[nbr_idx];

				if (input_delta != nullptr)
				{
					atomicAdd(input_delta + nbr_idx, graph_alpha[edge_idx] * graph_h2f_gpu(d_agg_value));
				}

				atomicAdd(edge_kernel_updates + kernel_base + ci, d_logit * graph_h2f_gpu(ref_value));
				atomicAdd(edge_kernel_updates + kernel_base + graph_cpg + ci, d_logit * graph_h2f_gpu(neighbor_value));

				if (input_delta != nullptr)
				{
					if (ref_valid)
					{
						input_delta[input_index_gpu(c, h, w, b, input_channel_base + ci, ref_y, ref_x)] +=
							d_logit * graph_h2f_gpu(edge_kernel[kernel_base + ci]);
					}
					atomicAdd(input_delta + nbr_idx, d_logit * graph_h2f_gpu(edge_kernel[kernel_base + graph_cpg + ci]));
				}
			}
		}
	}
#endif

/**
 * @brief CUDA kernel for graph convolution forward pass
 *
 * TEACHING MOMENT: The Graph Aggregation Kernel
 * This kernel treats pixels as nodes in a graph. For every pixel, it:
 * 1.  Finds its spatial neighbors.
 * 2.  Computes dynamic edge weights (how important is this neighbor?).
 * 3.  Gathers features from neighbors.
 * 4.  Combines them into a new feature for the center pixel.
 */
	__global__ void graph_conv_forward_kernel(const float *input, const float *edge_kernel,
		const float *edge_biases, float *graph_ref, float *graph_agg, float *graph_alpha,
		float *graph_valid, int batch, int c, int h, int w, int out_h,
		int out_w, int groups, int graph_cpg, int size, int stride_x, int stride_y,
		int dilation, int pad, int graph_k, int graph_edge_mode, int store_ref,
		int store_training_edges)
	{
		// One thread handles one output node for one group.
		// It does only the graph-specific work:
		//   1. fetch center feature
		//   2. score neighbors
		//   3. normalize scores
		//   4. accumulate weighted neighbor features
		//
		// The dense channel projection used to happen here too, but that part has
		// been moved to GEMM after this kernel finishes.
		const int index = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
		const int total = batch * groups * out_h * out_w;
		if (index >= total)
		{
			return;
		}

		const int ox = index % out_w;
		const int oy = (index / out_w) % out_h;
		const int g = (index / (out_w * out_h)) % groups;
		const int b = index / (groups * out_h * out_w);
		const int input_channel_base = g * graph_cpg;

		int ref_y = 0;
		int ref_x = 0;
		graph_reference_coord_gpu(size, stride_x, stride_y, dilation, pad, oy, ox, ref_y, ref_x);
		const bool ref_valid = valid_input_coord_gpu(h, w, ref_y, ref_x);
		const bool need_ref_value = (store_ref != 0) || (graph_edge_mode == 1);
		const bool use_ref_cache = need_ref_value && graph_cpg <= GRAPH_LOCAL_REF_CACHE_MAX;
		float ref_cache[GRAPH_LOCAL_REF_CACHE_MAX];

		// Save the center feature vector for this output position.
		// `graph_ref` is later reused by:
		// - the self branch GEMM in forward
		// - the self branch GEMM in backward
		// - the edge-kernel gradient path
		for (int ci = 0; ci < graph_cpg; ++ci)
		{
			float ref_value = 0.0f;
			if (need_ref_value && ref_valid)
			{
				ref_value = input[input_index_gpu(c, h, w, b, input_channel_base + ci, ref_y, ref_x)];
			}
			if (use_ref_cache)
			{
				// Cache the center feature locally for common low-channel cases so the
				// attention loop does not keep rereading it from device memory.
				ref_cache[ci] = ref_value;
			}
			if (store_ref)
			{
				graph_ref[graph_feature_index_gpu(c, out_h, out_w, b, input_channel_base + ci, oy, ox)] = ref_value;
			}
			graph_agg[graph_feature_index_gpu(c, out_h, out_w, b, input_channel_base + ci, oy, ox)] = 0.0f;
		}

		if (graph_edge_mode == 0)
		{
			// Mean mode is the graph layer's cheap path, so keep it to a true
			// valid-neighbor average instead of paying the attention bookkeeping cost.
			int valid_count = 0;
			for (int k = 0; k < graph_k; ++k)
			{
				int iy = 0;
				int ix = 0;
				graph_neighbor_coord_gpu(size, stride_x, stride_y, dilation, pad, oy, ox, k, iy, ix);
				const bool valid = valid_input_coord_gpu(h, w, iy, ix);
				if (store_training_edges)
				{
					const int edge_idx = graph_edge_index_gpu(groups, out_h, out_w, graph_k, b, g, oy, ox, k);
					graph_valid[edge_idx] = valid ? 1.0f : 0.0f;
					graph_alpha[edge_idx] = 0.0f;
				}
				if (!valid)
				{
					continue;
				}

				++valid_count;
				for (int ci = 0; ci < graph_cpg; ++ci)
				{
					const int agg_idx = graph_feature_index_gpu(c, out_h, out_w, b, input_channel_base + ci, oy, ox);
					graph_agg[agg_idx] += input[input_index_gpu(c, h, w, b, input_channel_base + ci, iy, ix)];
				}
			}

			if (valid_count > 0)
			{
				const float inv_count = 1.0f / static_cast<float>(valid_count);
				for (int ci = 0; ci < graph_cpg; ++ci)
				{
					const int agg_idx = graph_feature_index_gpu(c, out_h, out_w, b, input_channel_base + ci, oy, ox);
					graph_agg[agg_idx] *= inv_count;
				}

				if (store_training_edges)
				{
					for (int k = 0; k < graph_k; ++k)
					{
						const int edge_idx = graph_edge_index_gpu(groups, out_h, out_w, graph_k, b, g, oy, ox, k);
						if (graph_valid[edge_idx] > 0.5f)
						{
							graph_alpha[edge_idx] = inv_count;
						}
					}
				}
			}
			return;
		}

		float max_logit = -FLT_MAX;
		int valid_count = 0;
		float logits_cache[GRAPH_LOCAL_LOGITS_MAX];
		const bool use_local_logits = graph_k <= GRAPH_LOCAL_LOGITS_MAX;
		for (int k = 0; k < graph_k; ++k)
		{
			int iy = 0;
			int ix = 0;
			graph_neighbor_coord_gpu(size, stride_x, stride_y, dilation, pad, oy, ox, k, iy, ix);

			const bool valid = valid_input_coord_gpu(h, w, iy, ix);
			if (store_training_edges)
			{
				const int edge_idx = graph_edge_index_gpu(groups, out_h, out_w, graph_k, b, g, oy, ox, k);
				graph_valid[edge_idx] = valid ? 1.0f : 0.0f;
				graph_alpha[edge_idx] = 0.0f;
			}
			if (!valid)
			{
				continue;
			}

			++valid_count;
			const float logit = graph_attention_logit_gpu(input, edge_kernel, edge_biases, c, h, w,
				b, g, input_channel_base, graph_cpg, graph_k, k, ref_y, ref_x, ref_valid,
				use_ref_cache, ref_cache, iy, ix);
			if (use_local_logits)
			{
				// Keep common 3x3/5x5/7x7 logits in thread-local storage and fall back
				// to recomputation for larger kernels instead of a global-memory tensor.
				logits_cache[k] = logit;
			}
			if (logit > max_logit)
			{
				max_logit = logit;
			}
		}

		if (valid_count == 0)
		{
			return;
		}

		float denom = 0.0f;
		for (int k = 0; k < graph_k; ++k)
		{
			int iy = 0;
			int ix = 0;
			graph_neighbor_coord_gpu(size, stride_x, stride_y, dilation, pad, oy, ox, k, iy, ix);
			if (!valid_input_coord_gpu(h, w, iy, ix))
			{
				continue;
			}

			const float logit = use_local_logits
				? logits_cache[k]
				: graph_attention_logit_gpu(input, edge_kernel, edge_biases, c, h, w, b, g,
					input_channel_base, graph_cpg, graph_k, k, ref_y, ref_x, ref_valid,
					use_ref_cache, ref_cache, iy, ix);
			denom += expf(logit - max_logit);
		}

		for (int k = 0; k < graph_k; ++k)
		{
			int iy = 0;
			int ix = 0;
			graph_neighbor_coord_gpu(size, stride_x, stride_y, dilation, pad, oy, ox, k, iy, ix);
			if (!valid_input_coord_gpu(h, w, iy, ix))
			{
				continue;
			}

			const float logit = use_local_logits
				? logits_cache[k]
				: graph_attention_logit_gpu(input, edge_kernel, edge_biases, c, h, w, b, g,
					input_channel_base, graph_cpg, graph_k, k, ref_y, ref_x, ref_valid,
					use_ref_cache, ref_cache, iy, ix);
			const float alpha = expf(logit - max_logit) / denom;
			if (store_training_edges)
			{
				graph_alpha[graph_edge_index_gpu(groups, out_h, out_w, graph_k, b, g, oy, ox, k)] = alpha;
			}

			for (int ci = 0; ci < graph_cpg; ++ci)
			{
				const int agg_idx = graph_feature_index_gpu(c, out_h, out_w, b, input_channel_base + ci, oy, ox);
				const float neighbor_value = input[input_index_gpu(c, h, w, b, input_channel_base + ci, iy, ix)];
				graph_agg[agg_idx] += alpha * neighbor_value;
			}
		}
	}

	__global__ void graph_conv_pointwise_edges_kernel(float *graph_alpha, float *graph_valid, int total)
	{
		const int index = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
		if (index >= total)
		{
			return;
		}
		graph_alpha[index] = 1.0f;
		graph_valid[index] = 1.0f;
	}

	__global__ void graph_conv_backward_kernel(const float *input, const float *edge_kernel,
		const float *graph_ref, const float *graph_alpha, const float *graph_valid,
		const float *d_agg, const float *d_ref_self, float *edge_kernel_updates,
		float *edge_bias_updates, float *input_delta, int batch, int c, int h, int w,
		int out_h, int out_w, int groups, int graph_cpg, int size, int stride_x,
		int stride_y, int dilation, int pad, int graph_k, int graph_edge_mode)
	{
		// This kernel handles only the irregular graph part of backward.
		// The dense matrix algebra has already been done with GEMM on the host side:
		//   dW_neighbor, dW_self, d_agg, d_ref_self
		//
		// Here we use those precomputed tensors to push gradients through:
		//   neighbor aggregation
		//   softmax edge weighting
		//   edge-kernel parameters
		const int index = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
		const int total = batch * groups * out_h * out_w;
		if (index >= total)
		{
			return;
		}

		const int ox = index % out_w;
		const int oy = (index / out_w) % out_h;
		const int g = (index / (out_w * out_h)) % groups;
		const int b = index / (groups * out_h * out_w);
		const int input_channel_base = g * graph_cpg;
		const int edge_kernel_width = 2 * graph_cpg;

		int ref_y = 0;
		int ref_x = 0;
		graph_reference_coord_gpu(size, stride_x, stride_y, dilation, pad, oy, ox, ref_y, ref_x);
		const bool ref_valid = valid_input_coord_gpu(h, w, ref_y, ref_x);

		if (ref_valid && input_delta != nullptr && d_ref_self != nullptr)
		{
			// The self branch gradient was already projected by GEMM:
			//   d_ref_self = W_self^T * delta_out
			// Now we scatter that gradient back to the true center pixel location.
			// This write is unique per thread, so an atomic is unnecessary overhead.
			for (int ci = 0; ci < graph_cpg; ++ci)
			{
				const int ref_feature_idx = graph_feature_index_gpu(c, out_h, out_w, b, input_channel_base + ci, oy, ox);
				const float d_ref = d_ref_self[ref_feature_idx];
				input_delta[input_index_gpu(c, h, w, b, input_channel_base + ci, ref_y, ref_x)] += d_ref;
			}
		}

		// Softmax backward for learned edge weights (edge_mode == 1):
		//   d_logit_k = alpha_k * (d_alpha_k - sum_term)
		// where d_alpha_k = sum_ci d_agg[ci] * x_k[ci]  (dot of upstream gradient with neighbor feature)
		// and   sum_term   = sum_j alpha_j * d_alpha_j   (scalar common to all k in this output pixel)
		//
		// We compute sum_term in a dedicated pre-pass so it's available when we loop over k below.
		// If we tried to compute it inline we'd need an extra synchronisation point (not possible in
		// this kernel's data-parallel design) or an O(K^4) nested loop — both are worse.
		float sum_term = 0.0f;
		if (graph_edge_mode == 1)
		{
			// Pre-pass: accumulate sum_term = sum_k alpha_k * d_alpha_k
			// d_alpha_k is not stored explicitly; it is re-derived here as a dot product.
			for (int k = 0; k < graph_k; ++k)
			{
				const int edge_idx = graph_edge_index_gpu(groups, out_h, out_w, graph_k, b, g, oy, ox, k);
				if (graph_valid[edge_idx] <= 0.5f)
				{
					continue;
				}

				int iy = 0;
				int ix = 0;
				graph_neighbor_coord_gpu(size, stride_x, stride_y, dilation, pad, oy, ox, k, iy, ix);

				float d_alpha = 0.0f;
				for (int ci = 0; ci < graph_cpg; ++ci)
				{
					const int agg_idx = graph_feature_index_gpu(c, out_h, out_w, b, input_channel_base + ci, oy, ox);
					// d_alpha_k = dL/d_alpha_k = sum_ci d_agg[ci] * x_k[ci]
					d_alpha += d_agg[agg_idx] * input[input_index_gpu(c, h, w, b, input_channel_base + ci, iy, ix)];
				}
				sum_term += graph_alpha[edge_idx] * d_alpha;
			}
		}

		for (int k = 0; k < graph_k; ++k)
		{
			const int edge_idx = graph_edge_index_gpu(groups, out_h, out_w, graph_k, b, g, oy, ox, k);
			if (graph_valid[edge_idx] <= 0.5f)
			{
				continue;
			}

			int iy = 0;
			int ix = 0;
			graph_neighbor_coord_gpu(size, stride_x, stride_y, dilation, pad, oy, ox, k, iy, ix);

			float d_alpha = 0.0f;
			for (int ci = 0; ci < graph_cpg; ++ci)
			{
				const int agg_idx = graph_feature_index_gpu(c, out_h, out_w, b, input_channel_base + ci, oy, ox);
				const float d_agg_value = d_agg[agg_idx];

				if (input_delta != nullptr)
				{
					// Neighbor feature path from:
					//   agg = sum_k alpha_k * x_k
					atomicAdd(input_delta + input_index_gpu(c, h, w, b, input_channel_base + ci, iy, ix),
						graph_alpha[edge_idx] * d_agg_value);
				}
				// `d_alpha` tells us how sensitive the loss is to changing this one
				// edge weight before we differentiate through the softmax itself.
				d_alpha += d_agg_value * input[input_index_gpu(c, h, w, b, input_channel_base + ci, iy, ix)];
			}

			if (graph_edge_mode == 1)
			{
				// Softmax Jacobian contraction:
				//   d_logit_k = alpha_k * (d_alpha_k - sum_j alpha_j * d_alpha_j)
				const float d_logit = graph_alpha[edge_idx] * (d_alpha - sum_term);
				const int kernel_base = (g * graph_k + k) * edge_kernel_width;
				atomicAdd(edge_bias_updates + g * graph_k + k, d_logit);
				for (int ci = 0; ci < graph_cpg; ++ci)
				{
					const int ref_idx = graph_feature_index_gpu(c, out_h, out_w, b, input_channel_base + ci, oy, ox);
					const int nbr_idx = input_index_gpu(c, h, w, b, input_channel_base + ci, iy, ix);
					const float ref_value = graph_ref[ref_idx];
					const float neighbor_value = input[nbr_idx];
					// Edge-kernel parameters see both the reference feature and the
					// current neighbor feature because the forward logit used both.
					atomicAdd(edge_kernel_updates + kernel_base + ci, d_logit * ref_value);
					atomicAdd(edge_kernel_updates + kernel_base + graph_cpg + ci, d_logit * neighbor_value);

					if (input_delta != nullptr)
					{
						if (ref_valid)
						{
							// The reference feature also participates in the edge score, but
							// each thread owns a unique center pixel so this add does not need
							// to contend with other threads.
							input_delta[input_index_gpu(c, h, w, b, input_channel_base + ci, ref_y, ref_x)] +=
								d_logit * edge_kernel[kernel_base + ci];
						}
						atomicAdd(input_delta + nbr_idx, d_logit * edge_kernel[kernel_base + graph_cpg + ci]);
					}
				}
			}
		}
	}
}

void forward_graph_conv_layer_gpu(Darknet::Layer & l, Darknet::NetworkState state)
{
	TAT(TATPARMS);
	// Same matrix view as on CPU:
	//   graph_agg[group] : [graph_cpg, spatial]
	//   weights[group]   : [graph_npg, graph_cpg]
	//   output[group]    : [graph_npg, spatial]
	const int spatial = l.out_h * l.out_w;
	const bool store_training_edges = state.train != 0;
	const bool store_ref = store_training_edges || l.graph_use_self;
	const bool pointwise_fast_path = graph_pointwise_fast_path(l);
	float *projection_input = pointwise_fast_path ? state.input : l.graph_agg_gpu;

	// STEP 1: NEIGHBORHOOD AGGREGATION
	// This is where the "graph" magic happens. We launch a kernel that:
	// - Finds neighbors for every pixel (node).
	// - Calculates "attention" edge weights (how much to trust each neighbor).
	// - Sums up neighbor features weighted by those scores.
	const int total_nodes = l.batch * l.groups * l.out_h * l.out_w;
	if (pointwise_fast_path)
	{
		if (store_training_edges)
		{
			graph_conv_pointwise_edges_kernel<<<cuda_gridsize(total_nodes), BLOCK, 0, get_cuda_stream()>>>(
				l.graph_alpha_gpu, l.graph_valid_gpu, total_nodes);
			CHECK_CUDA(cudaPeekAtLastError());
		}
	}
	else
	{
		float *input16 = nullptr;
		if (prepare_graph_forward_mixed_16bit(l, state, &input16))
		{
#if DARKNET_GRAPH_USE_CUDNN_HALF
			graph_conv_forward_kernel_mixed_16bit<<<cuda_gridsize(total_nodes), BLOCK, 0, get_cuda_stream()>>>(
				reinterpret_cast<const __half *>(input16), reinterpret_cast<const __half *>(l.graph_edge_kernel_gpu16),
				l.graph_edge_biases_gpu, l.graph_ref_gpu, l.graph_agg_gpu, l.graph_alpha_gpu, l.graph_valid_gpu,
				l.batch, l.c, l.h, l.w, l.out_h, l.out_w, l.groups, l.graph_cpg, l.size, l.stride_x,
				l.stride_y, l.dilation, l.pad, l.graph_k, store_ref ? 1 : 0, store_training_edges ? 1 : 0);
#endif
		}
		else
		{
			graph_conv_forward_kernel<<<cuda_gridsize(total_nodes), BLOCK, 0, get_cuda_stream()>>>(
				state.input, l.graph_edge_kernel_gpu, l.graph_edge_biases_gpu, l.graph_ref_gpu, l.graph_agg_gpu,
				l.graph_alpha_gpu, l.graph_valid_gpu, l.batch, l.c, l.h, l.w,
				l.out_h, l.out_w, l.groups, l.graph_cpg, l.size, l.stride_x, l.stride_y, l.dilation,
				l.pad, l.graph_k, l.graph_edge_mode, store_ref ? 1 : 0, store_training_edges ? 1 : 0);
		}
		CHECK_CUDA(cudaPeekAtLastError());
	}

	// Once the graph kernel has produced `graph_agg` and `graph_ref`, let cuBLAS
	// do the channel mixing.  This removes the expensive per-thread output-channel
	// loops from the custom CUDA kernel.
	//
	// Launch one strided-batched GEMM per group instead of one GEMM per `(batch, group)`.
	// This cuts host launch overhead without changing the underlying math.
#ifdef DARKNET_GPU_CUDA
	const bool use_tensor_op_projection =
		(!state.train) &&
		state.net.cudnn_half &&
		(l.graph_cpg % 8 == 0) &&
		(l.graph_npg % 8 == 0);
#else
	const bool use_tensor_op_projection = false;
#endif
	// The irregular graph gather stays in FP32, but the dense projection is regular
	// enough to benefit from Tensor Core-friendly math on supported CUDA inference runs.
	ScopedTensorOpMath tensor_op_projection(use_tensor_op_projection);
	if (forward_graph_projection_cudnn_16bit(l, state, projection_input))
	{
		// FP16 cuDNN handled the dense projection.  The graph-specific gather may
		// already have used the mixed FP16 path above; normalization/activation stay unchanged.
	}
	else
	{
		const long long batch_input_stride = static_cast<long long>(l.c) * spatial;
		const long long batch_output_stride = static_cast<long long>(l.out_c) * spatial;
		for (int g = 0; g < l.groups; ++g)
		{
			const int input_channel_base = g * l.graph_cpg;
			const int output_channel_base = g * l.graph_npg;
			float *weights = l.weights_gpu + weight_row_index_gpu(l.graph_npg, l.graph_cpg, g, 0);
			float *agg = projection_input + (input_channel_base * spatial);
			float *out = l.output_gpu + (output_channel_base * spatial);

			// Neighbor contribution: out = W_neighbor * graph_agg
			gemm_ongpu_strided_batched(0, 0, l.graph_npg, spatial, l.graph_cpg, 1,
				weights, l.graph_cpg, 0,
				agg, spatial, batch_input_stride,
				0, out, spatial, batch_output_stride, l.batch);

			if (l.graph_use_self)
			{
				float *self_weights = l.graph_self_weights_gpu + weight_row_index_gpu(l.graph_npg, l.graph_cpg, g, 0);
				float *ref = l.graph_ref_gpu + (input_channel_base * spatial);
				// Self contribution: out += W_self * graph_ref
				gemm_ongpu_strided_batched(0, 0, l.graph_npg, spatial, l.graph_cpg, 1,
					self_weights, l.graph_cpg, 0,
					ref, spatial, batch_input_stride,
					1, out, spatial, batch_output_stride, l.batch);
			}
		}
	}

	// STEP 2: NORMALIZATION
	// Just like standard convolution, we can apply Batch Normalization or a simple Bias.
	if (l.batch_normalize)
	{
		forward_batchnorm_layer_gpu(l, state);
	}
	else
	{
		add_bias_gpu(l.output_gpu, l.biases_gpu, l.batch, l.n, l.out_h * l.out_w);
	}

	// STEP 3: ACTIVATION
	// Apply the activation function (Leaky ReLU, Mish, etc.) to squash the outputs.
	if (l.activation == SWISH) activate_array_swish_ongpu(l.output_gpu, l.outputs * l.batch, l.activation_input_gpu, l.output_gpu);
	else if (l.activation == MISH) activate_array_mish_ongpu(l.output_gpu, l.outputs * l.batch, l.activation_input_gpu, l.output_gpu);
	else if (l.activation == HARD_MISH) activate_array_hard_mish_ongpu(l.output_gpu, l.outputs * l.batch, l.activation_input_gpu, l.output_gpu);
	else if (l.activation == EML) activate_array_eml_ongpu(l.output_gpu, l.outputs * l.batch, l.activation_input_gpu, l.output_gpu);
	else if (l.activation == NORM_CHAN) activate_array_normalize_channels_ongpu(l.output_gpu, l.outputs * l.batch, l.batch, l.out_c, l.out_w * l.out_h, l.output_gpu);
	else if (l.activation == NORM_CHAN_SOFTMAX) activate_array_normalize_channels_softmax_ongpu(l.output_gpu, l.outputs * l.batch, l.batch, l.out_c, l.out_w * l.out_h, l.output_gpu, 0);
	else if (l.activation == NORM_CHAN_SOFTMAX_MAXVAL) activate_array_normalize_channels_softmax_ongpu(l.output_gpu, l.outputs * l.batch, l.batch, l.out_c, l.out_w * l.out_h, l.output_gpu, 1);
	else if (l.activation != LINEAR) activate_array_ongpu(l.output_gpu, l.outputs * l.batch, l.activation);

	check_nan_gpu("forward: after activation (graph)", l.output_gpu, l.outputs * l.batch, l.index);

	// STEP 4: CLEANUP
	// Catch any numerical instability (NaN/Inf) before it poisons the next layer.
	if (state.net.try_fix_nan)
	{
		fix_nan_and_inf(l.output_gpu, l.outputs * l.batch);
	}
}

/**
 * @brief Backward pass for graph convolutional layer (GPU version)
 *
 * TEACHING MOMENT: Reversing the graph flow!
 * 1. PREPARE DELTAS: Apply activation and batch norm gradients.
 * 2. GRAPH BACKPROP: This is where we calculate gradients for:
 *    - Node features (the pixels themselves).
 *    - Transformation weights (the W matrices).
 *    - Edge kernels (the attention mechanism).
 *    - Self-connection weights.
 */
void backward_graph_conv_layer_gpu(Darknet::Layer & l, Darknet::NetworkState state)
{
	TAT(TATPARMS);
	const int spatial = l.out_h * l.out_w;
	const size_t feature_count = static_cast<size_t>(l.batch) * l.c * spatial;
	const bool pointwise_fast_path = graph_pointwise_fast_path(l);
	float *projection_input = pointwise_fast_path ? state.input : l.graph_agg_gpu;
	float *workspace = state.workspace;
	float *workspace_fallback = nullptr;
	if (workspace == nullptr)
	{
		// Standalone callers may not provide `state.workspace`.
		// Network execution does, but keep a defensive fallback here.
		workspace_fallback = cuda_make_array(nullptr, l.workspace_size / sizeof(float) + 1);
		workspace = workspace_fallback;
	}
	// Workspace layout mirrors the CPU path:
	//   [0, feature_count)                -> d_agg
	//   [feature_count, 2*feature_count) -> d_ref_self (only if self branch is enabled)
	float *d_agg_workspace = workspace;
	float *d_ref_workspace = l.graph_use_self ? (workspace + feature_count) : nullptr;

	// STEP 1: INITIAL CLEANUP
	if (state.net.try_fix_nan)
	{
		constrain_ongpu(l.outputs * l.batch, 1.0f, l.delta_gpu, 1);
	}

	// STEP 2: ACTIVATION GRADIENT
	if (l.activation == SWISH) gradient_array_swish_ongpu(l.output_gpu, l.outputs * l.batch, l.activation_input_gpu, l.delta_gpu);
	else if (l.activation == MISH) gradient_array_mish_ongpu(l.outputs * l.batch, l.activation_input_gpu, l.delta_gpu);
	else if (l.activation == HARD_MISH) gradient_array_hard_mish_ongpu(l.outputs * l.batch, l.activation_input_gpu, l.delta_gpu);
	else if (l.activation == EML) gradient_array_eml_ongpu(l.outputs * l.batch, l.activation_input_gpu, l.delta_gpu);
	else if (l.activation == NORM_CHAN_SOFTMAX || l.activation == NORM_CHAN_SOFTMAX_MAXVAL) gradient_array_normalize_channels_softmax_ongpu(l.output_gpu, l.outputs * l.batch, l.batch, l.out_c, l.out_w * l.out_h, l.delta_gpu);
	else if (l.activation == NORM_CHAN) gradient_array_normalize_channels_ongpu(l.output_gpu, l.outputs * l.batch, l.batch, l.out_c, l.out_w * l.out_h, l.delta_gpu);
	else gradient_array_ongpu(l.output_gpu, l.outputs * l.batch, l.activation, l.delta_gpu);

	// STEP 3: BATCH NORM OR BIAS GRADIENT
	if (l.batch_normalize)
	{
		backward_batchnorm_layer_gpu(l, state);
	}
	else
	{
		backward_bias_gpu(l.bias_updates_gpu, l.delta_gpu, l.batch, l.n, l.out_h * l.out_w);
	}

	// Dense backward phase:
	//   dW_neighbor = delta_out * graph_agg^T
	//   d_agg       = W_neighbor^T * delta_out
	//   dW_self     = delta_out * graph_ref^T
	//   d_ref_self  = W_self^T * delta_out
	//
	// All of these are regular matrix multiplies, so cuBLAS is the right tool.
	if (!backward_graph_projection_cudnn_16bit(l, state, projection_input, d_agg_workspace, d_ref_workspace))
	{
		const long long batch_input_stride = static_cast<long long>(l.c) * spatial;
		const long long batch_output_stride = static_cast<long long>(l.out_c) * spatial;
		for (int g = 0; g < l.groups; ++g)
		{
			const int input_channel_base = g * l.graph_cpg;
			const int output_channel_base = g * l.graph_npg;
			float *weights = l.weights_gpu + weight_row_index_gpu(l.graph_npg, l.graph_cpg, g, 0);
			float *delta_out_base = l.delta_gpu + (output_channel_base * spatial);
			float *d_agg_base = d_agg_workspace + (input_channel_base * spatial);

			// Batched activation-gradient GEMMs cut the same `(batch, group)` launch overhead
			// as forward, while the weight-gradient reductions still need per-batch accumulation.
			gemm_ongpu_strided_batched(1, 0, l.graph_cpg, spatial, l.graph_npg, 1,
				weights, l.graph_cpg, 0,
				delta_out_base, spatial, batch_output_stride,
				0, d_agg_base, spatial, batch_input_stride, l.batch);

			if (l.graph_use_self)
			{
				float *self_weights = l.graph_self_weights_gpu + weight_row_index_gpu(l.graph_npg, l.graph_cpg, g, 0);
				float *d_ref_base = d_ref_workspace + (input_channel_base * spatial);
				gemm_ongpu_strided_batched(1, 0, l.graph_cpg, spatial, l.graph_npg, 1,
					self_weights, l.graph_cpg, 0,
					delta_out_base, spatial, batch_output_stride,
					0, d_ref_base, spatial, batch_input_stride, l.batch);
			}
		}

		for (int b = 0; b < l.batch; ++b)
		{
			for (int g = 0; g < l.groups; ++g)
			{
				const int input_channel_base = g * l.graph_cpg;
				const int output_channel_base = g * l.graph_npg;
				float *weight_updates = l.weight_updates_gpu + weight_row_index_gpu(l.graph_npg, l.graph_cpg, g, 0);
				float *agg = projection_input + ((b * l.c + input_channel_base) * spatial);
				float *ref = l.graph_ref_gpu + ((b * l.c + input_channel_base) * spatial);
				float *delta_out = l.delta_gpu + ((b * l.out_c + output_channel_base) * spatial);

				gemm_ongpu(0, 1, l.graph_npg, l.graph_cpg, spatial, 1, delta_out, spatial, agg, spatial, 1, weight_updates, l.graph_cpg);

				if (l.graph_use_self)
				{
					float *self_weight_updates = l.graph_self_weight_updates_gpu + weight_row_index_gpu(l.graph_npg, l.graph_cpg, g, 0);
					gemm_ongpu(0, 1, l.graph_npg, l.graph_cpg, spatial, 1, delta_out, spatial, ref, spatial, 1, self_weight_updates, l.graph_cpg);
				}
			}
		}
	}

	if (pointwise_fast_path)
	{
		if (state.delta)
		{
			axpy_ongpu(static_cast<int>(feature_count), 1.0f, d_agg_workspace, 1, state.delta, 1);
		}

		if (state.net.try_fix_nan)
		{
			reset_nan_and_inf(l.weight_updates_gpu, l.nweights);
		}

		if (workspace_fallback)
		{
			cuda_free(workspace_fallback);
		}
		return;
	}

	// STEP 4: THE CORE GRAPH BACKWARD KERNEL
	// This kernel performs the complex task of calculating how changing
	// neighbors and their attention scores would have affected the result.
	const int total_nodes = l.batch * l.groups * l.out_h * l.out_w;
	float *input16 = nullptr;
	float *d_agg16 = nullptr;
	if (prepare_graph_backward_mixed_16bit(l, state, d_agg_workspace, &input16, &d_agg16))
	{
#if DARKNET_GRAPH_USE_CUDNN_HALF
		graph_conv_backward_kernel_mixed_16bit<<<cuda_gridsize(total_nodes), BLOCK, 0, get_cuda_stream()>>>(
			reinterpret_cast<const __half *>(input16), reinterpret_cast<const __half *>(l.graph_edge_kernel_gpu16),
			l.graph_alpha_gpu, l.graph_valid_gpu, reinterpret_cast<const __half *>(d_agg16),
			d_ref_workspace, l.graph_edge_kernel_updates_gpu, l.graph_edge_bias_updates_gpu,
			state.delta, l.batch, l.c, l.h, l.w, l.out_h, l.out_w, l.groups, l.graph_cpg, l.size,
			l.stride_x, l.stride_y, l.dilation, l.pad, l.graph_k);
#endif
	}
	else
	{
		graph_conv_backward_kernel<<<cuda_gridsize(total_nodes), BLOCK, 0, get_cuda_stream()>>>(
			state.input, l.graph_edge_kernel_gpu, l.graph_ref_gpu, l.graph_alpha_gpu, l.graph_valid_gpu,
			d_agg_workspace, d_ref_workspace, l.graph_edge_kernel_updates_gpu, l.graph_edge_bias_updates_gpu,
			state.delta, l.batch, l.c, l.h, l.w, l.out_h, l.out_w, l.groups, l.graph_cpg, l.size,
			l.stride_x, l.stride_y, l.dilation, l.pad, l.graph_k, l.graph_edge_mode);
	}
	CHECK_CUDA(cudaPeekAtLastError());

	// STEP 5: CLEANUP AND STABILIZE
	// Fix any NaNs produced during the complex graph backprop calculations.
	if (state.net.try_fix_nan)
	{
		reset_nan_and_inf(l.weight_updates_gpu, l.nweights);
		if (l.graph_use_self)
		{
			reset_nan_and_inf(l.graph_self_weight_updates_gpu, l.n * l.graph_cpg);
		}
		if (l.graph_edge_mode == 1)
		{
			reset_nan_and_inf(l.graph_edge_kernel_updates_gpu, l.groups * l.graph_k * (2 * l.graph_cpg));
			reset_nan_and_inf(l.graph_edge_bias_updates_gpu, l.groups * l.graph_k);
		}
	}

	if (workspace_fallback)
	{
		cuda_free(workspace_fallback);
	}
}

void update_graph_conv_layer_gpu(Darknet::Layer & l, int batch, float learning_rate_init, float momentum, float decay, float loss_scale)
{
	TAT(TATPARMS);

	const float learning_rate = (learning_rate_init * l.learning_rate_scale) / loss_scale;

	reset_nan_and_inf(l.weight_updates_gpu, l.nweights);
	fix_nan_and_inf(l.weights_gpu, l.nweights);

	axpy_ongpu(l.nweights, -decay * batch, l.weights_gpu, 1, l.weight_updates_gpu, 1);
	axpy_ongpu(l.nweights, learning_rate / batch, l.weight_updates_gpu, 1, l.weights_gpu, 1);
	scal_ongpu(l.nweights, momentum, l.weight_updates_gpu, 1);

	axpy_ongpu(l.n, learning_rate / batch, l.bias_updates_gpu, 1, l.biases_gpu, 1);
	scal_ongpu(l.n, momentum, l.bias_updates_gpu, 1);

	if (l.batch_normalize)
	{
		axpy_ongpu(l.n, learning_rate / batch, l.scale_updates_gpu, 1, l.scales_gpu, 1);
		scal_ongpu(l.n, momentum, l.scale_updates_gpu, 1);
	}

	if (l.graph_use_self)
	{
		const int self_count = l.n * l.graph_cpg;
		reset_nan_and_inf(l.graph_self_weight_updates_gpu, self_count);
		fix_nan_and_inf(l.graph_self_weights_gpu, self_count);
		axpy_ongpu(self_count, -decay * batch, l.graph_self_weights_gpu, 1, l.graph_self_weight_updates_gpu, 1);
		axpy_ongpu(self_count, learning_rate / batch, l.graph_self_weight_updates_gpu, 1, l.graph_self_weights_gpu, 1);
		scal_ongpu(self_count, momentum, l.graph_self_weight_updates_gpu, 1);
	}

	if (l.graph_edge_mode == 1)
	{
		const int kernel_count = l.groups * l.graph_k * (2 * l.graph_cpg);
		const int bias_count = l.groups * l.graph_k;
		reset_nan_and_inf(l.graph_edge_kernel_updates_gpu, kernel_count);
		fix_nan_and_inf(l.graph_edge_kernel_gpu, kernel_count);
		axpy_ongpu(kernel_count, -decay * batch, l.graph_edge_kernel_gpu, 1, l.graph_edge_kernel_updates_gpu, 1);
		axpy_ongpu(kernel_count, learning_rate / batch, l.graph_edge_kernel_updates_gpu, 1, l.graph_edge_kernel_gpu, 1);
		scal_ongpu(kernel_count, momentum, l.graph_edge_kernel_updates_gpu, 1);

		axpy_ongpu(bias_count, learning_rate / batch, l.graph_edge_bias_updates_gpu, 1, l.graph_edge_biases_gpu, 1);
		scal_ongpu(bias_count, momentum, l.graph_edge_bias_updates_gpu, 1);
	}
}

void push_graph_conv_layer(Darknet::Layer & l)
{
	if (l.weights_gpu) cuda_push_array(l.weights_gpu, l.weights, l.nweights);
	if (l.biases_gpu) cuda_push_array(l.biases_gpu, l.biases, l.n);

	if (l.graph_use_self && l.graph_self_weights_gpu)
	{
		cuda_push_array(l.graph_self_weights_gpu, l.graph_self_weights, l.n * l.graph_cpg);
	}

	if (l.graph_edge_mode == 1)
	{
		if (l.graph_edge_kernel_gpu) cuda_push_array(l.graph_edge_kernel_gpu, l.graph_edge_kernel, l.groups * l.graph_k * (2 * l.graph_cpg));
		if (l.graph_edge_biases_gpu) cuda_push_array(l.graph_edge_biases_gpu, l.graph_edge_biases, l.groups * l.graph_k);
	}

	if (l.batch_normalize)
	{
		if (l.scales_gpu) cuda_push_array(l.scales_gpu, l.scales, l.n);
		if (l.rolling_mean_gpu) cuda_push_array(l.rolling_mean_gpu, l.rolling_mean, l.n);
		if (l.rolling_variance_gpu) cuda_push_array(l.rolling_variance_gpu, l.rolling_variance, l.n);
	}
}

void pull_graph_conv_layer(Darknet::Layer & l)
{
	if (l.weights_gpu) cuda_pull_array(l.weights_gpu, l.weights, l.nweights);
	if (l.biases_gpu) cuda_pull_array(l.biases_gpu, l.biases, l.n);

	if (l.graph_use_self && l.graph_self_weights_gpu)
	{
		cuda_pull_array(l.graph_self_weights_gpu, l.graph_self_weights, l.n * l.graph_cpg);
	}

	if (l.graph_edge_mode == 1)
	{
		if (l.graph_edge_kernel_gpu) cuda_pull_array(l.graph_edge_kernel_gpu, l.graph_edge_kernel, l.groups * l.graph_k * (2 * l.graph_cpg));
		if (l.graph_edge_biases_gpu) cuda_pull_array(l.graph_edge_biases_gpu, l.graph_edge_biases, l.groups * l.graph_k);
	}

	if (l.batch_normalize)
	{
		if (l.scales_gpu) cuda_pull_array(l.scales_gpu, l.scales, l.n);
		if (l.rolling_mean_gpu) cuda_pull_array(l.rolling_mean_gpu, l.rolling_mean, l.n);
		if (l.rolling_variance_gpu) cuda_pull_array(l.rolling_variance_gpu, l.rolling_variance, l.n);
	}
}

#endif
