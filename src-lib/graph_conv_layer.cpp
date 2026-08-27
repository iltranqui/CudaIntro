#include "darknet_internal.hpp"
#include "graph_conv_layer.hpp"
#include "batchnorm_layer.hpp"
#include "convolutional_layer.hpp"
#include "gemm.hpp"

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <vector>

namespace
{
	static auto & cfg_and_state = Darknet::CfgAndState::get();

	static void check_nan(const char *step_name, float *arr, size_t size, int layer_idx)
	{
		if (arr == nullptr || size == 0) return;
		for (size_t i = 0; i < size; ++i)
		{
			if (std::isnan(arr[i]) || std::isinf(arr[i]))
			{
				const std::string layer_label = Darknet::layer_type_diagnostic_label(Darknet::ELayerType::GRAPH_CONV);
				std::printf("[%s layer] NaN/Inf detected at layer %d, step: %s\n", layer_label.c_str(), layer_idx, step_name);
				break;
			}
		}
	}

	// ── Index helper functions ────────────────────────────────────────────────
	// All tensors use NCHW flat storage.  An element at position (b, c, y, x)
	// maps to flat offset:  b*C*H*W + c*H*W + y*W + x
	// Different tensors have different spatial dimensions (H,W vs out_H,out_W),
	// so separate helpers exist for each shape to avoid confusion at call sites.
	// ──────────────────────────────────────────────────────────────────────────

	// Input tensor layout: [B, C, H, W]  (original spatial resolution)
	inline int input_index(const Darknet::Layer & l, int b, int c, int y, int x)
	{
		return ((b * l.c + c) * l.h + y) * l.w + x;
	}

	// Output tensor layout: [B, N, out_H, out_W]  (N = number of output filters)
	inline int output_index(const Darknet::Layer & l, int b, int c, int y, int x)
	{
		return ((b * l.out_c + c) * l.out_h + y) * l.out_w + x;
	}

	// Graph intermediate buffers: [B, C, out_H, out_W]
	// Same channel count C as the input, but output (strided/padded) spatial size.
	// Used for `graph_ref` (center features) and `graph_agg` (aggregated features).
	// Note: uses l.c, not l.out_c — channel count matches input, not output.
	inline int graph_feature_index(const Darknet::Layer & l, int b, int c, int y, int x)
	{
		return ((b * l.c + c) * l.out_h + y) * l.out_w + x;
	}

	// Edge buffers: [B, groups, out_H, out_W, K²]
	// The innermost dimension K² (graph_k = size*size) indexes over the flat
	// KxK spatial neighborhood in row-major order (k=0 is top-left corner).
	// Used for graph_alpha and graph_valid.
	inline int graph_edge_index(const Darknet::Layer & l, int b, int g, int y, int x, int k)
	{
		return ((((b * l.groups + g) * l.out_h + y) * l.out_w + x) * l.graph_k + k);
	}

	// Weight matrix: [groups, npg, cpg]
	// Returns the flat offset of the first element of row (g*npg + oc_local).
	// The full row spans `graph_cpg` consecutive floats (one per input channel/group).
	inline int weight_row_index(const Darknet::Layer & l, int g, int oc_local)
	{
		return (g * l.graph_npg + oc_local) * l.graph_cpg;
	}

	// The learned edge kernel vector has 2*cpg elements per (group, neighbor) pair:
	//   [0 .. cpg-1]     : W_ref  coefficients (dot with center reference feature)
	//   [cpg .. 2*cpg-1] : W_nbr  coefficients (dot with neighbor feature)
	// The full logit formula is: logit_k = bias_k + W_ref · ref + W_nbr · neighbor_k
	inline int edge_kernel_width(const Darknet::Layer & l)
	{
		return 2 * l.graph_cpg;
	}

	inline size_t graph_conv_temp_workspace_size(const Darknet::Layer & l)
	{
		size_t size = static_cast<size_t>(l.batch) * l.c * l.out_h * l.out_w * sizeof(float);
		if (l.graph_use_self)
		{
			size *= 2;
		}
		return size;
	}

	inline size_t align_graph_workspace(size_t size)
	{
		constexpr size_t alignment = 256;
		return (size + alignment - 1) & ~(alignment - 1);
	}

#if defined(DARKNET_GPU) && defined(CUDNN) && defined(CUDNN_HALF)
	cudnnDataType_t graph_cudnn_16bit_data_type(const Darknet::Layer & l)
	{
#if defined(DARKNET_GPU_CUDA) && defined(CUDNN_DATA_BFLOAT16)
		if (l.cudnn_16bit_mode == DARKNET_CUDNN_16BIT_BF16)
		{
			return CUDNN_DATA_BFLOAT16;
		}
#endif
		return CUDNN_DATA_HALF;
	}

	template <typename Descriptor, typename CreateFn>
	void create_graph_cudnn_descriptor_if_needed(Descriptor *desc, CreateFn create_fn)
	{
		if (*desc == nullptr)
		{
			CHECK_CUDNN(create_fn(desc));
		}
	}

	void configure_graph_projection_cudnn_descriptors(Darknet::Layer & l, int cudnn_16bit_mode)
	{
		l.cudnn_16bit_mode = cudnn_16bit_mode;
		const cudnnDataType_t data_type = graph_cudnn_16bit_data_type(l);
		constexpr cudnnTensorFormat_t tensor_format = CUDNN_TENSOR_NCHW;

		create_graph_cudnn_descriptor_if_needed(&l.srcTensorDesc16, cudnnCreateTensorDescriptor);
		create_graph_cudnn_descriptor_if_needed(&l.dstTensorDesc16, cudnnCreateTensorDescriptor);
		create_graph_cudnn_descriptor_if_needed(&l.dsrcTensorDesc16, cudnnCreateTensorDescriptor);
		create_graph_cudnn_descriptor_if_needed(&l.ddstTensorDesc16, cudnnCreateTensorDescriptor);
		create_graph_cudnn_descriptor_if_needed(&l.weightDesc16, cudnnCreateFilterDescriptor);
		create_graph_cudnn_descriptor_if_needed(&l.dweightDesc16, cudnnCreateFilterDescriptor);
		create_graph_cudnn_descriptor_if_needed(&l.convDesc, cudnnCreateConvolutionDescriptor);

		CHECK_CUDNN(cudnnSetTensor4dDescriptor(l.srcTensorDesc16, tensor_format, data_type, l.batch, l.c, l.out_h, l.out_w));
		CHECK_CUDNN(cudnnSetTensor4dDescriptor(l.dsrcTensorDesc16, tensor_format, data_type, l.batch, l.c, l.out_h, l.out_w));
		CHECK_CUDNN(cudnnSetTensor4dDescriptor(l.dstTensorDesc16, tensor_format, data_type, l.batch, l.out_c, l.out_h, l.out_w));
		CHECK_CUDNN(cudnnSetTensor4dDescriptor(l.ddstTensorDesc16, tensor_format, data_type, l.batch, l.out_c, l.out_h, l.out_w));
		CHECK_CUDNN(cudnnSetFilter4dDescriptor(l.weightDesc16, data_type, tensor_format, l.n, l.graph_cpg, 1, 1));
		CHECK_CUDNN(cudnnSetFilter4dDescriptor(l.dweightDesc16, data_type, tensor_format, l.n, l.graph_cpg, 1, 1));
#if CUDNN_MAJOR >= 6
		CHECK_CUDNN(cudnnSetConvolution2dDescriptor(l.convDesc, 0, 0, 1, 1, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));
#else
		CHECK_CUDNN(cudnnSetConvolution2dDescriptor(l.convDesc, 0, 0, 1, 1, 1, 1, CUDNN_CROSS_CORRELATION));
#endif
		CHECK_CUDNN(cudnnSetConvolutionGroupCount(l.convDesc, l.groups));
		CHECK_CUDNN(cudnnSetConvolutionMathType(l.convDesc, CUDNN_TENSOR_OP_MATH));

		l.fw_algo16 = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;
		l.bd_algo16 = CUDNN_CONVOLUTION_BWD_DATA_ALGO_1;
		l.bf_algo16 = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1;

		if (l.batch_normalize)
		{
			create_graph_cudnn_descriptor_if_needed(&l.normTensorDesc, cudnnCreateTensorDescriptor);
			create_graph_cudnn_descriptor_if_needed(&l.normDstTensorDesc, cudnnCreateTensorDescriptor);
			create_graph_cudnn_descriptor_if_needed(&l.normDstTensorDescF16, cudnnCreateTensorDescriptor);
			CHECK_CUDNN(cudnnSetTensor4dDescriptor(l.normTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, l.out_c, 1, 1));
			CHECK_CUDNN(cudnnSetTensor4dDescriptor(l.normDstTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, l.batch, l.out_c, l.out_h, l.out_w));
			CHECK_CUDNN(cudnnSetTensor4dDescriptor(l.normDstTensorDescF16, CUDNN_TENSOR_NCHW, data_type, l.batch, l.out_c, l.out_h, l.out_w));
		}
	}

	size_t get_graph_projection_cudnn_workspace_size(const Darknet::Layer & l)
	{
		if (cfg_and_state.gpu_index < 0 || l.convDesc == nullptr || l.srcTensorDesc16 == nullptr)
		{
			return 0;
		}

		size_t most = 0;
		size_t current = 0;
		CHECK_CUDNN(cudnnGetConvolutionForwardWorkspaceSize(cudnn_handle(),
			l.srcTensorDesc16, l.weightDesc16, l.convDesc, l.dstTensorDesc16, l.fw_algo16, &current));
		most = std::max(most, current);

		if (l.train)
		{
			CHECK_CUDNN(cudnnGetConvolutionBackwardFilterWorkspaceSize(cudnn_handle(),
				l.srcTensorDesc16, l.ddstTensorDesc16, l.convDesc, l.dweightDesc16, l.bf_algo16, &current));
			most = std::max(most, current);
			CHECK_CUDNN(cudnnGetConvolutionBackwardDataWorkspaceSize(cudnn_handle(),
				l.weightDesc16, l.ddstTensorDesc16, l.convDesc, l.dsrcTensorDesc16, l.bd_algo16, &current));
			most = std::max(most, current);
		}

		return most;
	}
#else
	void configure_graph_projection_cudnn_descriptors(Darknet::Layer &, int) {}

	size_t get_graph_projection_cudnn_workspace_size(const Darknet::Layer &)
	{
		return 0;
	}
#endif

	inline bool valid_input_coord(const Darknet::Layer & l, int y, int x)
	{
		return y >= 0 && y < l.h && x >= 0 && x < l.w;
	}

	// Map output position (oy, ox) to the center input pixel of its receptive field.
	// For a 3×3 kernel: center=1, so this is the pixel at k=4 in the K² neighbor loop.
	// The center pixel is the "self" node; its feature is stored in graph_ref.
	inline void graph_reference_coord(const Darknet::Layer & l, int oy, int ox, int & ref_y, int & ref_x)
	{
		const int center = l.size / 2;    // integer center of the KxK grid
		ref_y = oy * l.stride_y - l.pad + center * l.dilation;
		ref_x = ox * l.stride_x - l.pad + center * l.dilation;
	}

	// Map flat neighbor index k ∈ [0, size²) to its input pixel coordinate.
	// k is in row-major order: ky = k / size (row), kx = k % size (column).
	// Coordinates may be negative or ≥ input dimensions — caller must check validity.
	inline void graph_neighbor_coord(const Darknet::Layer & l, int oy, int ox, int k, int & iy, int & ix)
	{
		const int ky = k / l.size;   // row in the KxK grid
		const int kx = k % l.size;   // column in the KxK grid
		iy = oy * l.stride_y - l.pad + ky * l.dilation;
		ix = ox * l.stride_x - l.pad + kx * l.dilation;
	}

	void allocate_graph_runtime_buffers(Darknet::Layer & l, int total_batch)
	{
		// feature_count: one float per (batch, channel, out_y, out_x) position.
		// Both graph_ref and graph_agg share this shape [B, C, out_H, out_W].
		const size_t feature_count = static_cast<size_t>(total_batch) * l.c * l.out_h * l.out_w;
		// edge_count: one float per (batch, group, out_y, out_x, neighbor_k) — i.e., one entry per graph edge.
		const size_t edge_count = static_cast<size_t>(total_batch) * l.groups * l.out_h * l.out_w * l.graph_k;

		// graph_ref: the center (reference) feature vector at each output position.
		//   Written during GATHER; read by the self-branch GEMM and the edge-kernel gradient in backward.
		l.graph_ref = (float*)xcalloc(feature_count, sizeof(float));
		// graph_agg: the aggregated neighbor feature at each output position (GATHER result).
		//   Written during GATHER; fed directly into the PROJECT GEMM.
		l.graph_agg = (float*)xcalloc(feature_count, sizeof(float));
		// graph_alpha: normalised edge weights — softmax output (edge_mode=1) or uniform 1/K (edge_mode=0).
		//   Saved in forward; needed during backward to compute d_agg → d_input and d_logit.
		l.graph_alpha = (float*)xcalloc(edge_count, sizeof(float));
		// graph_valid: 1.0 if the neighbor lies within the image, 0.0 if out-of-bounds (padding region).
		//   Avoids repeating the bounds check in every backward iteration.
		l.graph_valid = (float*)xcalloc(edge_count, sizeof(float));
	}

	void resize_graph_runtime_buffers(Darknet::Layer * l, int total_batch)
	{
		const size_t feature_count = static_cast<size_t>(total_batch) * l->c * l->out_h * l->out_w;
		const size_t edge_count = static_cast<size_t>(total_batch) * l->groups * l->out_h * l->out_w * l->graph_k;

		l->graph_ref = (float*)xrealloc(l->graph_ref, feature_count * sizeof(float));
		l->graph_agg = (float*)xrealloc(l->graph_agg, feature_count * sizeof(float));
		l->graph_alpha = (float*)xrealloc(l->graph_alpha, edge_count * sizeof(float));
		l->graph_valid = (float*)xrealloc(l->graph_valid, edge_count * sizeof(float));
	}

	void activate_graph_output(Darknet::Layer & l)
	{
		if (l.activation == SWISH) activate_array_swish(l.output, l.outputs * l.batch, l.activation_input, l.output);
		else if (l.activation == MISH) activate_array_mish(l.output, l.outputs * l.batch, l.activation_input, l.output);
		else if (l.activation == HARD_MISH) activate_array_hard_mish(l.output, l.outputs * l.batch, l.activation_input, l.output);
		else if (l.activation == EML) activate_array_eml(l.output, l.outputs * l.batch, l.activation_input, l.output);
		else if (l.activation == NORM_CHAN) activate_array_normalize_channels(l.output, l.outputs * l.batch, l.batch, l.out_c, l.out_w * l.out_h, l.output);
		else if (l.activation == NORM_CHAN_SOFTMAX) activate_array_normalize_channels_softmax(l.output, l.outputs * l.batch, l.batch, l.out_c, l.out_w * l.out_h, l.output, 0);
		else if (l.activation == NORM_CHAN_SOFTMAX_MAXVAL) activate_array_normalize_channels_softmax(l.output, l.outputs * l.batch, l.batch, l.out_c, l.out_w * l.out_h, l.output, 1);
		else activate_array(l.output, l.outputs * l.batch, l.activation);
	}

	void gradient_graph_output(Darknet::Layer & l)
	{
		if (l.activation == SWISH) gradient_array_swish(l.output, l.outputs * l.batch, l.activation_input, l.delta);
		else if (l.activation == MISH) gradient_array_mish(l.outputs * l.batch, l.activation_input, l.delta);
		else if (l.activation == HARD_MISH) gradient_array_hard_mish(l.outputs * l.batch, l.activation_input, l.delta);
		else if (l.activation == EML) gradient_array_eml(l.outputs * l.batch, l.activation_input, l.delta);
		else if (l.activation == NORM_CHAN_SOFTMAX || l.activation == NORM_CHAN_SOFTMAX_MAXVAL) gradient_array_normalize_channels_softmax(l.output, l.outputs * l.batch, l.batch, l.out_c, l.out_w * l.out_h, l.delta);
		else if (l.activation == NORM_CHAN) gradient_array_normalize_channels(l.output, l.outputs * l.batch, l.batch, l.out_c, l.out_w * l.out_h, l.delta);
		else gradient_array(l.output, l.outputs * l.batch, l.activation, l.delta);
	}
}

size_t get_graph_conv_workspace_size(const Darknet::Layer & l)
{
	const size_t temp_workspace = graph_conv_temp_workspace_size(l);
	const size_t projection_workspace = get_graph_projection_cudnn_workspace_size(l);
	return align_graph_workspace(temp_workspace) + projection_workspace;
}

int graph_conv_out_height(const Darknet::Layer & l)
{
	return (l.h + 2 * l.pad - l.dilation * (l.size - 1) - 1) / l.stride_y + 1;
}

int graph_conv_out_width(const Darknet::Layer & l)
{
	return (l.w + 2 * l.pad - l.dilation * (l.size - 1) - 1) / l.stride_x + 1;
}

Darknet::Layer make_graph_conv_layer(int batch, int steps, int h, int w, int c, int n, int groups,
	int size, int stride_x, int stride_y, int dilation, int padding, ACTIVATION activation,
	int batch_normalize, int binary, int xnor, int adam, int use_bin_output, int index,
	int antialiasing, Darknet::Layer * share_layer, int assisted_excitation, int train,
	int graph_edge_mode, int graph_use_self, int graph_valid_mask_zero)
{
	TAT(TATPARMS);

	if (binary || xnor)
	{
		darknet_fatal_error(DARKNET_LOC, "graph_conv does not support binary/xnor modes");
	}
	if (antialiasing)
	{
		darknet_fatal_error(DARKNET_LOC, "graph_conv does not support antialiasing");
	}
	if (share_layer != nullptr)
	{
		darknet_fatal_error(DARKNET_LOC, "graph_conv does not support share_layer");
	}
	if (assisted_excitation)
	{
		darknet_fatal_error(DARKNET_LOC, "graph_conv does not support assisted_excitation");
	}
	if (adam)
	{
		darknet_fatal_error(DARKNET_LOC, "graph_conv does not support adam yet");
	}
	if (size < 1 || (size % 2) == 0)
	{
		darknet_fatal_error(DARKNET_LOC, "graph_conv requires an odd kernel size >= 1");
	}
	if (groups < 1)
	{
		groups = 1;
	}
	if ((c % groups) != 0 || (n % groups) != 0)
	{
		darknet_fatal_error(DARKNET_LOC, "graph_conv requires both c=%d and n=%d to be divisible by groups=%d", c, n, groups);
	}
	if (graph_edge_mode != 0 && graph_edge_mode != 1)
	{
		darknet_fatal_error(DARKNET_LOC, "graph_conv only supports graph_edge_mode 0 or 1 for now");
	}
	if (graph_valid_mask_zero != 1)
	{
		darknet_fatal_error(DARKNET_LOC, "graph_conv currently requires graph_valid_mask_zero=1");
	}

	const int total_batch = batch * steps;
	Darknet::Layer l = { (Darknet::ELayerType)0 };
	l.type = Darknet::ELayerType::GRAPH_CONV;
	l.train = train;
	l.batch = batch;
	l.steps = steps;
	l.h = h;
	l.w = w;
	l.c = c;
	l.n = n;
	l.groups = groups;
	l.stride = stride_x;
	l.stride_x = stride_x;
	l.stride_y = stride_y;
	l.dilation = dilation;
	l.size = size;
	l.pad = padding;
	l.activation = activation;
	l.batch_normalize = batch_normalize;
	l.use_bin_output = use_bin_output;
	l.index = index;
	l.share_layer = share_layer;
	l.learning_rate_scale = 1.0f;
	l.graph_k = size * size;
	l.graph_edge_mode = graph_edge_mode;
	l.graph_use_self = graph_use_self ? 1 : 0;
	l.graph_valid_mask_zero = graph_valid_mask_zero;
	l.graph_cpg = c / groups;
	l.graph_npg = n / groups;

	l.out_h = graph_conv_out_height(l);
	l.out_w = graph_conv_out_width(l);
	l.out_c = n;
	l.outputs = l.out_h * l.out_w * l.out_c;
	l.inputs = l.w * l.h * l.c;
	l.nweights = l.n * l.graph_cpg;
	l.workspace_size = get_graph_conv_workspace_size(l);

	l.weights = (float*)xcalloc(l.nweights, sizeof(float));
	l.weight_updates = (float*)xcalloc(l.nweights, sizeof(float));
	l.biases = (float*)xcalloc(l.n, sizeof(float));
	l.bias_updates = (float*)xcalloc(l.n, sizeof(float));

	// He-style initialisation: scale = sqrt(2 / fan_in).
	// The fan-in for the channel projection is graph_cpg (input channels per group),
	// since each output neuron aggregates cpg inputs from the neighborhood.
	// `rand_uniform_many_weight_init` draws from U(-1, 1) and then multiplies by weight_scale.
	const float weight_scale = std::sqrt(2.0f / static_cast<float>(l.graph_cpg));
	rand_uniform_many_weight_init(l.weights, l.nweights, -1.0f, 1.0f, weight_scale);

	if (l.graph_use_self)
	{
		const size_t self_count = static_cast<size_t>(l.n) * l.graph_cpg;
		l.graph_self_weights = (float*)xcalloc(self_count, sizeof(float));
		l.graph_self_weight_updates = (float*)xcalloc(self_count, sizeof(float));
		rand_uniform_many_weight_init(l.graph_self_weights, self_count, -1.0f, 1.0f, weight_scale);
	}

	if (l.graph_edge_mode == 1)
	{
		const size_t kernel_count = static_cast<size_t>(l.groups) * l.graph_k * edge_kernel_width(l);
		const size_t bias_count = static_cast<size_t>(l.groups) * l.graph_k;
		l.graph_edge_kernel = (float*)xcalloc(kernel_count, sizeof(float));
		l.graph_edge_kernel_updates = (float*)xcalloc(kernel_count, sizeof(float));
		l.graph_edge_biases = (float*)xcalloc(bias_count, sizeof(float));
		l.graph_edge_bias_updates = (float*)xcalloc(bias_count, sizeof(float));
		// Edge kernel is initialised with a very small scale (0.001) so that all neighbors
		// are weighted roughly equally at the start of training.  A larger init would cause
		// the softmax to be sharply peaked, which kills gradients for non-winning neighbors
		// and prevents the network from discovering useful neighbourhood weighting patterns.
		rand_uniform_many_weight_init(l.graph_edge_kernel, kernel_count, -1.0f, 1.0f, 0.001f);
	}

	l.output = (float*)xcalloc(static_cast<size_t>(total_batch) * l.outputs, sizeof(float));
	l.delta = (float*)xcalloc(static_cast<size_t>(total_batch) * l.outputs, sizeof(float));
	allocate_graph_runtime_buffers(l, total_batch);

	if (batch_normalize)
	{
		l.scales = (float*)xcalloc(l.n, sizeof(float));
		l.scale_updates = (float*)xcalloc(l.n, sizeof(float));
		l.mean = (float*)xcalloc(l.n, sizeof(float));
		l.variance = (float*)xcalloc(l.n, sizeof(float));
		l.mean_delta = (float*)xcalloc(l.n, sizeof(float));
		l.variance_delta = (float*)xcalloc(l.n, sizeof(float));
		l.rolling_mean = (float*)xcalloc(l.n, sizeof(float));
		l.rolling_variance = (float*)xcalloc(l.n, sizeof(float));
		l.x = (float*)xcalloc(static_cast<size_t>(total_batch) * l.outputs, sizeof(float));
		l.x_norm = (float*)xcalloc(static_cast<size_t>(total_batch) * l.outputs, sizeof(float));
		for (int i = 0; i < l.n; ++i)
		{
			l.scales[i] = 1.0f;
		}
	}

#ifndef DARKNET_GPU
	if (l.activation == SWISH || l.activation == MISH || l.activation == HARD_MISH || l.activation == EML)
	{
		l.activation_input = (float*)xcalloc(static_cast<size_t>(total_batch) * l.outputs, sizeof(float));
	}
#endif

	l.forward = forward_graph_conv_layer;
	l.backward = backward_graph_conv_layer;
	l.update = update_graph_conv_layer;

#ifdef DARKNET_GPU
	l.forward_gpu = forward_graph_conv_layer_gpu;
	l.backward_gpu = backward_graph_conv_layer_gpu;
	l.update_gpu = update_graph_conv_layer_gpu;

	if (cfg_and_state.gpu_index >= 0)
	{
		l.weights_gpu = cuda_make_array(l.weights, l.nweights);
		l.weight_updates_gpu = cuda_make_array(l.weight_updates, l.nweights);
#ifdef CUDNN_HALF
		l.weights_gpu16 = cuda_make_array(nullptr, l.nweights / 2 + 1);
		if (train)
		{
			l.weight_updates_gpu16 = cuda_make_array(nullptr, l.nweights / 2 + 1);
		}
#endif
		l.biases_gpu = cuda_make_array(l.biases, l.n);
		l.bias_updates_gpu = cuda_make_array(l.bias_updates, l.n);
		l.output_gpu = cuda_make_array(l.output, static_cast<size_t>(total_batch) * l.outputs);
		l.delta_gpu = cuda_make_array(l.delta, static_cast<size_t>(total_batch) * l.outputs);
		l.graph_ref_gpu = cuda_make_array(l.graph_ref, static_cast<size_t>(total_batch) * l.c * l.out_h * l.out_w);
		l.graph_agg_gpu = cuda_make_array(l.graph_agg, static_cast<size_t>(total_batch) * l.c * l.out_h * l.out_w);
		l.graph_alpha_gpu = cuda_make_array(l.graph_alpha, static_cast<size_t>(total_batch) * l.groups * l.out_h * l.out_w * l.graph_k);
		l.graph_valid_gpu = cuda_make_array(l.graph_valid, static_cast<size_t>(total_batch) * l.groups * l.out_h * l.out_w * l.graph_k);

		if (l.graph_use_self)
		{
			const size_t self_count = static_cast<size_t>(l.n) * l.graph_cpg;
			l.graph_self_weights_gpu = cuda_make_array(l.graph_self_weights, self_count);
			l.graph_self_weight_updates_gpu = cuda_make_array(l.graph_self_weight_updates, self_count);
#ifdef CUDNN_HALF
			l.graph_self_weights_gpu16 = cuda_make_array(nullptr, self_count / 2 + 1);
			if (train)
			{
				l.graph_self_weight_updates_gpu16 = cuda_make_array(nullptr, self_count / 2 + 1);
			}
#endif
		}

		if (l.graph_edge_mode == 1)
		{
			const size_t kernel_count = static_cast<size_t>(l.groups) * l.graph_k * edge_kernel_width(l);
			const size_t bias_count = static_cast<size_t>(l.groups) * l.graph_k;
			l.graph_edge_kernel_gpu = cuda_make_array(l.graph_edge_kernel, kernel_count);
#ifdef CUDNN_HALF
			l.graph_edge_kernel_gpu16 = cuda_make_array(nullptr, kernel_count / 2 + 1);
#endif
			l.graph_edge_kernel_updates_gpu = cuda_make_array(l.graph_edge_kernel_updates, kernel_count);
			l.graph_edge_biases_gpu = cuda_make_array(l.graph_edge_biases, bias_count);
			l.graph_edge_bias_updates_gpu = cuda_make_array(l.graph_edge_bias_updates, bias_count);
		}

		if (l.batch_normalize)
		{
			l.scales_gpu = cuda_make_array(l.scales, l.n);
			l.scale_updates_gpu = cuda_make_array(l.scale_updates, l.n);
			l.mean_gpu = cuda_make_array(l.mean, l.n);
			l.variance_gpu = cuda_make_array(l.variance, l.n);
			l.rolling_mean_gpu = cuda_make_array(l.rolling_mean, l.n);
			l.rolling_variance_gpu = cuda_make_array(l.rolling_variance, l.n);
			l.mean_delta_gpu = cuda_make_array(l.mean_delta, l.n);
			l.variance_delta_gpu = cuda_make_array(l.variance_delta, l.n);
			l.x_gpu = cuda_make_array(l.output, static_cast<size_t>(total_batch) * l.outputs);
			l.x_norm_gpu = cuda_make_array(l.output, static_cast<size_t>(total_batch) * l.outputs);
#ifdef CUDNN
			if (l.normTensorDesc == nullptr) CHECK_CUDNN(cudnnCreateTensorDescriptor(&l.normTensorDesc));
			if (l.normDstTensorDesc == nullptr) CHECK_CUDNN(cudnnCreateTensorDescriptor(&l.normDstTensorDesc));
			CHECK_CUDNN(cudnnSetTensor4dDescriptor(l.normTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, l.out_c, 1, 1));
			CHECK_CUDNN(cudnnSetTensor4dDescriptor(l.normDstTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, l.batch, l.out_c, l.out_h, l.out_w));
#endif
		}

		if (l.activation == SWISH || l.activation == MISH || l.activation == HARD_MISH || l.activation == EML)
		{
			l.activation_input = (float*)xcalloc(static_cast<size_t>(total_batch) * l.outputs, sizeof(float));
			l.activation_input_gpu = cuda_make_array(l.activation_input, static_cast<size_t>(total_batch) * l.outputs);
		}

#if defined(CUDNN) && defined(CUDNN_HALF)
		configure_graph_projection_cudnn_descriptors(l, DARKNET_CUDNN_16BIT_HALF);
		l.workspace_size = get_graph_conv_workspace_size(l);
#endif
	}
#endif

	if (cfg_and_state.is_verbose)
	{
		*cfg_and_state.output << "graph_conv " << size << " x " << size
			<< " / " << stride_x << " x " << stride_y
			<< ", " << n << " filters, " << l.inputs << " inputs, " << l.outputs
			<< " outputs, edge_mode=" << l.graph_edge_mode
			<< ", self=" << l.graph_use_self << std::endl;
	}

	return l;
}

void resize_graph_conv_layer(Darknet::Layer * l, int w, int h)
{
	TAT(TATPARMS);

	l->w = w;
	l->h = h;
	l->out_h = graph_conv_out_height(*l);
	l->out_w = graph_conv_out_width(*l);
	l->outputs = l->out_h * l->out_w * l->out_c;
	l->inputs = l->w * l->h * l->c;
	l->workspace_size = get_graph_conv_workspace_size(*l);

	const int total_batch = l->batch * l->steps;
	l->output = (float*)xrealloc(l->output, static_cast<size_t>(total_batch) * l->outputs * sizeof(float));
	l->delta = (float*)xrealloc(l->delta, static_cast<size_t>(total_batch) * l->outputs * sizeof(float));
	resize_graph_runtime_buffers(l, total_batch);

	if (l->batch_normalize)
	{
		l->x = (float*)xrealloc(l->x, static_cast<size_t>(total_batch) * l->outputs * sizeof(float));
		l->x_norm = (float*)xrealloc(l->x_norm, static_cast<size_t>(total_batch) * l->outputs * sizeof(float));
	}

	if (l->activation == SWISH || l->activation == MISH || l->activation == HARD_MISH || l->activation == EML)
	{
		l->activation_input = (float*)xrealloc(l->activation_input, static_cast<size_t>(total_batch) * l->outputs * sizeof(float));
	}

#ifdef DARKNET_GPU
	if (cfg_and_state.gpu_index >= 0)
	{
		cuda_free(l->output_gpu);
		cuda_free(l->delta_gpu);
		l->output_gpu = cuda_make_array(l->output, static_cast<size_t>(total_batch) * l->outputs);
		l->delta_gpu = cuda_make_array(l->delta, static_cast<size_t>(total_batch) * l->outputs);
		cuda_free(l->graph_ref_gpu);
		cuda_free(l->graph_agg_gpu);
		cuda_free(l->graph_alpha_gpu);
		cuda_free(l->graph_valid_gpu);
		l->graph_ref_gpu = cuda_make_array(l->graph_ref, static_cast<size_t>(total_batch) * l->c * l->out_h * l->out_w);
		l->graph_agg_gpu = cuda_make_array(l->graph_agg, static_cast<size_t>(total_batch) * l->c * l->out_h * l->out_w);
		l->graph_alpha_gpu = cuda_make_array(l->graph_alpha, static_cast<size_t>(total_batch) * l->groups * l->out_h * l->out_w * l->graph_k);
		l->graph_valid_gpu = cuda_make_array(l->graph_valid, static_cast<size_t>(total_batch) * l->groups * l->out_h * l->out_w * l->graph_k);

		if (l->batch_normalize)
		{
			cuda_free(l->x_gpu);
			cuda_free(l->x_norm_gpu);
			l->x_gpu = cuda_make_array(l->output, static_cast<size_t>(total_batch) * l->outputs);
			l->x_norm_gpu = cuda_make_array(l->output, static_cast<size_t>(total_batch) * l->outputs);
#ifdef CUDNN
			CHECK_CUDNN(cudnnSetTensor4dDescriptor(l->normDstTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, l->batch, l->out_c, l->out_h, l->out_w));
#endif
		}

		if (l->activation == SWISH || l->activation == MISH || l->activation == HARD_MISH || l->activation == EML)
		{
			cuda_free(l->activation_input_gpu);
			l->activation_input_gpu = cuda_make_array(l->activation_input, static_cast<size_t>(total_batch) * l->outputs);
		}

#if defined(CUDNN) && defined(CUDNN_HALF)
		configure_graph_projection_cudnn_descriptors(*l, l->cudnn_16bit_mode);
		l->workspace_size = get_graph_conv_workspace_size(*l);
#endif
	}
#endif
}

void forward_graph_conv_layer(Darknet::Layer & l, Darknet::NetworkState state)
{
	TAT(TATPARMS);

	// `spatial` is the number of output positions per channel.
	// We repeatedly treat tensors as 2D matrices with shape:
	//   [channels, spatial]
	// so that the dense channel projection can be handled by GEMM.
	const int spatial = l.out_h * l.out_w;

	// These intermediate buffers are the "graph" side of the layer:
	// - `graph_ref`   : the center feature at each output location when training or using the self branch
	// - `graph_agg`   : the neighbor-weighted aggregate at each output location
	// - `graph_alpha` : normalized per-neighbor weights saved only when backward will need them
	// - `graph_valid` : whether a given neighbor lies inside the input image
	//
	// We deliberately do not clear the full buffers up front anymore.
	// Every live element is written explicitly below, which avoids a large memory-bandwidth tax.
	const bool store_training_edges = state.train != 0;
	const bool store_ref = store_training_edges || l.graph_use_self;

#ifdef DARKNET_OPENMP
	// Each `(batch, group)` slice is independent here, so CPU parallelism is cheap and safe.
	#pragma omp parallel for collapse(2) schedule(static)
#endif
	for (int b = 0; b < l.batch; ++b)
	{
		for (int g = 0; g < l.groups; ++g)
		{
			const int input_channel_base = g * l.graph_cpg;
			std::vector<float> ref(l.graph_cpg, 0.0f);
			std::vector<float> agg(l.graph_cpg, 0.0f);
			std::vector<float> logits((l.graph_edge_mode == 1) ? l.graph_k : 0, -FLT_MAX);

			for (int oy = 0; oy < l.out_h; ++oy)
			{
				for (int ox = 0; ox < l.out_w; ++ox)
				{
					// `ref` is the feature vector for the center node.
					// `agg` is the weighted sum of neighbor feature vectors.
					// `logits[k]` stores the edge score for the k-th spatial neighbor.
					std::fill(ref.begin(), ref.end(), 0.0f);
					std::fill(agg.begin(), agg.end(), 0.0f);
					if (l.graph_edge_mode == 1)
					{
						std::fill(logits.begin(), logits.end(), -FLT_MAX);
					}

					int ref_y = 0;
					int ref_x = 0;
					// Map this output location back to the center input pixel for the receptive field.
					graph_reference_coord(l, oy, ox, ref_y, ref_x);
					const bool ref_valid = valid_input_coord(l, ref_y, ref_x);
					for (int ci = 0; ci < l.graph_cpg; ++ci)
					{
						if (ref_valid)
						{
							ref[ci] = state.input[input_index(l, b, input_channel_base + ci, ref_y, ref_x)];
						}
						if (store_ref)
						{
							// Training and the self branch both need the center feature later, so
							// only materialize it when one of those consumers actually exists.
							l.graph_ref[graph_feature_index(l, b, input_channel_base + ci, oy, ox)] = ref[ci];
						}
					}

					if (l.graph_edge_mode == 0)
					{
						// Mean mode is the cheapest graph variant.
						// Avoid the attention-style score/softmax machinery and just average valid neighbors.
						int valid_count = 0;
						for (int k = 0; k < l.graph_k; ++k)
						{
							int iy = 0;
							int ix = 0;
							graph_neighbor_coord(l, oy, ox, k, iy, ix);
							const bool valid = valid_input_coord(l, iy, ix);
							if (store_training_edges)
							{
								const int edge_idx = graph_edge_index(l, b, g, oy, ox, k);
								l.graph_valid[edge_idx] = valid ? 1.0f : 0.0f;
								l.graph_alpha[edge_idx] = 0.0f;
							}
							if (!valid)
							{
								continue;
							}

							++valid_count;
							for (int ci = 0; ci < l.graph_cpg; ++ci)
							{
								agg[ci] += state.input[input_index(l, b, input_channel_base + ci, iy, ix)];
							}
						}
						if (valid_count > 0)
						{
							const float inv_count = 1.0f / static_cast<float>(valid_count);
							for (int ci = 0; ci < l.graph_cpg; ++ci)
							{
								agg[ci] *= inv_count;
							}
							if (store_training_edges)
							{
								// Backward expects the normalized neighbor weights, so fill them
								// only when this forward pass is feeding a backward pass.
								for (int k = 0; k < l.graph_k; ++k)
								{
									const int edge_idx = graph_edge_index(l, b, g, oy, ox, k);
									if (l.graph_valid[edge_idx] > 0.5f)
									{
										l.graph_alpha[edge_idx] = inv_count;
									}
								}
							}
						}
					}
					else
					{
						float max_logit = -FLT_MAX;
						int valid_count = 0;
						for (int k = 0; k < l.graph_k; ++k)
						{
							int iy = 0;
							int ix = 0;
							graph_neighbor_coord(l, oy, ox, k, iy, ix);
							const bool valid = valid_input_coord(l, iy, ix);
							if (store_training_edges)
							{
								const int edge_idx = graph_edge_index(l, b, g, oy, ox, k);
								l.graph_valid[edge_idx] = valid ? 1.0f : 0.0f;
								l.graph_alpha[edge_idx] = 0.0f;
							}
							if (!valid)
							{
								continue;
							}

							++valid_count;
							// Attention mode is more expressive but more expensive, so keep the
							// logits in a local vector and never spill them to global memory.
							const int kernel_base = (g * l.graph_k + k) * edge_kernel_width(l);
							float logit = l.graph_edge_biases[g * l.graph_k + k];
							for (int ci = 0; ci < l.graph_cpg; ++ci)
							{
								logit += l.graph_edge_kernel[kernel_base + ci] * ref[ci];
								logit += l.graph_edge_kernel[kernel_base + l.graph_cpg + ci] *
									state.input[input_index(l, b, input_channel_base + ci, iy, ix)];
							}
							logits[k] = logit;
							if (logit > max_logit)
							{
								max_logit = logit;
							}
						}

						if (valid_count > 0)
						{
							float denom = 0.0f;
							for (int k = 0; k < l.graph_k; ++k)
							{
								if (logits[k] != -FLT_MAX)
								{
									denom += expf(logits[k] - max_logit);
								}
							}

							for (int k = 0; k < l.graph_k; ++k)
							{
								if (logits[k] == -FLT_MAX)
								{
									continue;
								}

								const float alpha = expf(logits[k] - max_logit) / denom;
								if (store_training_edges)
								{
									l.graph_alpha[graph_edge_index(l, b, g, oy, ox, k)] = alpha;
								}

								int iy = 0;
								int ix = 0;
								graph_neighbor_coord(l, oy, ox, k, iy, ix);
								for (int ci = 0; ci < l.graph_cpg; ++ci)
								{
									agg[ci] += alpha * state.input[input_index(l, b, input_channel_base + ci, iy, ix)];
								}
							}
						}
					}

					for (int ci = 0; ci < l.graph_cpg; ++ci)
					{
						// Store the final aggregated node feature map in `[c_per_group, spatial]` layout.
						// This exact layout is what lets us hand the next phase to GEMM unchanged.
						l.graph_agg[graph_feature_index(l, b, input_channel_base + ci, oy, ox)] = agg[ci];
					}
				}
			}
		}
	}

	// Dense projection stage:
	// We now reinterpret the graph buffers as matrices and let GEMM do the
	// expensive channel mixing.  For one `(batch, group)` slice:
	//
	//   weights   : [graph_npg, graph_cpg]
	//   graph_agg : [graph_cpg, spatial]
	//   output    : [graph_npg, spatial]
	//
	// This is exactly the same math as the old nested loops, but mapped onto
	// the optimized matrix multiplication path.
	for (int b = 0; b < l.batch; ++b)
	{
		for (int g = 0; g < l.groups; ++g)
		{
			const int input_channel_base = g * l.graph_cpg;
			const int output_channel_base = g * l.graph_npg;
			float *weights = l.weights + weight_row_index(l, g, 0);
			float *agg_ptr = l.graph_agg + ((b * l.c + input_channel_base) * spatial);
			float *out = l.output + ((b * l.out_c + output_channel_base) * spatial);

			// Neighbor branch:
			// output_group = W_neighbor * aggregated_features
			gemm_cpu(0, 0, l.graph_npg, spatial, l.graph_cpg, 1, weights, l.graph_cpg, agg_ptr, spatial, 0, out, spatial);

			if (l.graph_use_self)
			{
				float *self_weights = l.graph_self_weights + weight_row_index(l, g, 0);
				float *ref_ptr = l.graph_ref + ((b * l.c + input_channel_base) * spatial);
				// Self branch:
				// output_group += W_self * center_features
				gemm_cpu(0, 0, l.graph_npg, spatial, l.graph_cpg, 1, self_weights, l.graph_cpg, ref_ptr, spatial, 1, out, spatial);
			}
		}
	}

	if (l.batch_normalize)
	{
		forward_batchnorm_layer(l, state);
	}
	else
	{
		add_bias(l.output, l.biases, l.batch, l.n, l.out_h * l.out_w);
	}

	activate_graph_output(l);
}

void backward_graph_conv_layer(Darknet::Layer & l, Darknet::NetworkState state)
{
	TAT(TATPARMS);

	// Backward uses the same `[channels, spatial]` matrix view as forward.
	const int spatial = l.out_h * l.out_w;
	const size_t feature_count = static_cast<size_t>(l.batch) * l.c * spatial;
	std::vector<float> workspace_fallback;
	if (state.workspace == nullptr)
	{
		// The full network path provides workspace.  Some unit tests call the layer
		// directly, so keep a local fallback to avoid crashing in standalone use.
		workspace_fallback.resize(feature_count * (l.graph_use_self ? 2u : 1u), 0.0f);
	}
	float *workspace = state.workspace ? state.workspace : workspace_fallback.data();
	// Workspace layout:
	//   first  feature_count floats -> d_agg  (gradient wrt aggregated neighbor features)
	//   second feature_count floats -> d_ref  (gradient wrt self/reference features), optional
	float *d_agg_workspace = workspace;
	float *d_ref_workspace = l.graph_use_self ? (workspace + feature_count) : nullptr;

	gradient_graph_output(l);

	if (l.batch_normalize)
	{
		backward_batchnorm_layer(l, state);
	}
	else
	{
		backward_bias(l.bias_updates, l.delta, l.batch, l.n, l.out_h * l.out_w);
	}

	std::vector<float> d_alpha(l.graph_k, 0.0f);
	std::vector<float> alpha(l.graph_k, 0.0f);

	// First backward stage: dense linear algebra only.
	//
	// We recover the same quantities that the old nested loops accumulated:
	// - weight gradients
	// - self-weight gradients
	// - d_agg : gradient wrt aggregated neighbor features
	// - d_ref : gradient wrt self/reference features
	//
	// GEMM handles all of these because they are just matrix products.
	for (int b = 0; b < l.batch; ++b)
	{
		for (int g = 0; g < l.groups; ++g)
		{
			const int input_channel_base = g * l.graph_cpg;
			const int output_channel_base = g * l.graph_npg;
			float *weights = l.weights + weight_row_index(l, g, 0);
			float *weight_updates = l.weight_updates + weight_row_index(l, g, 0);
			float *agg = l.graph_agg + ((b * l.c + input_channel_base) * spatial);
			float *ref = l.graph_ref + ((b * l.c + input_channel_base) * spatial);
			float *delta_out = l.delta + ((b * l.out_c + output_channel_base) * spatial);
			float *d_agg = d_agg_workspace + ((b * l.c + input_channel_base) * spatial);

			// dW_neighbor += delta_out * graph_agg^T
			gemm_cpu(0, 1, l.graph_npg, l.graph_cpg, spatial, 1, delta_out, spatial, agg, spatial, 1, weight_updates, l.graph_cpg);
			// d_agg = W_neighbor^T * delta_out
			gemm_cpu(1, 0, l.graph_cpg, spatial, l.graph_npg, 1, weights, l.graph_cpg, delta_out, spatial, 0, d_agg, spatial);

			if (l.graph_use_self)
			{
				float *self_weights = l.graph_self_weights + weight_row_index(l, g, 0);
				float *self_weight_updates = l.graph_self_weight_updates + weight_row_index(l, g, 0);
				float *d_ref = d_ref_workspace + ((b * l.c + input_channel_base) * spatial);
				// dW_self += delta_out * graph_ref^T
				gemm_cpu(0, 1, l.graph_npg, l.graph_cpg, spatial, 1, delta_out, spatial, ref, spatial, 1, self_weight_updates, l.graph_cpg);
				// d_ref = W_self^T * delta_out
				gemm_cpu(1, 0, l.graph_cpg, spatial, l.graph_npg, 1, self_weights, l.graph_cpg, delta_out, spatial, 0, d_ref, spatial);
			}
		}
	}

	// Second backward stage: graph-specific logic only.
	//
	// GEMM cannot do this part for us because the neighborhood topology and the
	// edge weights vary per output pixel.  Here we:
	// - distribute d_agg back to contributing neighbors
	// - differentiate through the softmax edge weights when edge_mode=1
	// - push d_ref back into the original center pixel location
	for (int b = 0; b < l.batch; ++b)
	{
		for (int g = 0; g < l.groups; ++g)
		{
			const int input_channel_base = g * l.graph_cpg;
			const float *ref = l.graph_ref + ((b * l.c + input_channel_base) * spatial);
			const float *d_agg = d_agg_workspace + ((b * l.c + input_channel_base) * spatial);
			const float *d_ref = l.graph_use_self ? (d_ref_workspace + ((b * l.c + input_channel_base) * spatial)) : nullptr;

			for (int oy = 0; oy < l.out_h; ++oy)
			{
				for (int ox = 0; ox < l.out_w; ++ox)
				{
					std::fill(d_alpha.begin(), d_alpha.end(), 0.0f);
					std::fill(alpha.begin(), alpha.end(), 0.0f);

					int valid_count = 0;
					for (int k = 0; k < l.graph_k; ++k)
					{
						const int edge_idx = graph_edge_index(l, b, g, oy, ox, k);
						alpha[k] = l.graph_alpha[edge_idx];
						if (l.graph_valid[edge_idx] > 0.5f)
						{
							++valid_count;
						}
					}

					if (valid_count == 0)
					{
						continue;
					}

					for (int k = 0; k < l.graph_k; ++k)
					{
						const int edge_idx = graph_edge_index(l, b, g, oy, ox, k);
						if (l.graph_valid[edge_idx] <= 0.5f)
						{
							continue;
						}

						int iy = 0;
						int ix = 0;
						graph_neighbor_coord(l, oy, ox, k, iy, ix);

						float dot = 0.0f;
						for (int ci = 0; ci < l.graph_cpg; ++ci)
						{
							// `d_agg` is stored as a `[channels, spatial]` matrix.
							// Convert the current `(ci, oy, ox)` back into the flat offset.
							const float d_agg_value = d_agg[ci * spatial + oy * l.out_w + ox];
							const float input_value = state.input[input_index(l, b, input_channel_base + ci, iy, ix)];
							dot += d_agg_value * input_value;
							if (state.delta)
							{
								// Neighbor feature path:
								// agg = sum_k alpha_k * x_k
								// so each neighbor receives alpha_k * d_agg.
								state.delta[input_index(l, b, input_channel_base + ci, iy, ix)] += alpha[k] * d_agg_value;
							}
						}
						// d_alpha[k] = dL/d_alpha_k
						//   = sum_ci  d_agg[ci] * x_k[ci]
						// Derivation: agg[ci] = sum_k alpha_k * x_k[ci]
						//   → d_agg[ci]/d_alpha_k = x_k[ci]
						//   → chain rule gives the dot product above.
						d_alpha[k] = dot;
					}

					if (l.graph_edge_mode == 1)
					{
						// Softmax backward via the Jacobian contraction:
						//   d_logit_k = alpha_k * (d_alpha_k - sum_term)
						// where sum_term = sum_j alpha_j * d_alpha_j.
						// We compute sum_term in one pass first, then reuse it for every k.
						// This avoids an O(K²) inner product inside the per-k loop.
						float sum_term = 0.0f;
						for (int k = 0; k < l.graph_k; ++k)
						{
							sum_term += alpha[k] * d_alpha[k];
						}

						int ref_y = 0;
						int ref_x = 0;
						graph_reference_coord(l, oy, ox, ref_y, ref_x);
						const bool ref_valid = valid_input_coord(l, ref_y, ref_x);

						for (int k = 0; k < l.graph_k; ++k)
						{
							const int edge_idx = graph_edge_index(l, b, g, oy, ox, k);
							if (l.graph_valid[edge_idx] <= 0.5f)
							{
								continue;
							}

							const float d_logit = alpha[k] * (d_alpha[k] - sum_term);
							const int kernel_base = (g * l.graph_k + k) * edge_kernel_width(l);
							int iy = 0;
							int ix = 0;
							graph_neighbor_coord(l, oy, ox, k, iy, ix);
							l.graph_edge_bias_updates[g * l.graph_k + k] += d_logit;
							for (int ci = 0; ci < l.graph_cpg; ++ci)
							{
								const int nbr_idx = input_index(l, b, input_channel_base + ci, iy, ix);
								// The reference feature was saved in graph buffers using the same
								// `[channels, spatial]` layout used by the GEMM stage.
								const float ref_value = ref[ci * spatial + oy * l.out_w + ox];
								const float neighbor_value = state.input[nbr_idx];
								// Edge MLP/kernel gradients come from the softmax path only.
								l.graph_edge_kernel_updates[kernel_base + ci] += d_logit * ref_value;
								l.graph_edge_kernel_updates[kernel_base + l.graph_cpg + ci] += d_logit * neighbor_value;
								if (state.delta)
								{
									if (ref_valid)
									{
										// Reference feature also participates in the learned edge score.
										state.delta[input_index(l, b, input_channel_base + ci, ref_y, ref_x)] += d_logit * l.graph_edge_kernel[kernel_base + ci];
									}
									state.delta[nbr_idx] += d_logit * l.graph_edge_kernel[kernel_base + l.graph_cpg + ci];
								}
							}
						}
					}

					if (state.delta && d_ref)
					{
						int ref_y = 0;
						int ref_x = 0;
						graph_reference_coord(l, oy, ox, ref_y, ref_x);
						if (valid_input_coord(l, ref_y, ref_x))
						{
							for (int ci = 0; ci < l.graph_cpg; ++ci)
							{
								// Self branch contribution computed earlier by GEMM:
								// d_ref = W_self^T * delta_out
								state.delta[input_index(l, b, input_channel_base + ci, ref_y, ref_x)] += d_ref[ci * spatial + oy * l.out_w + ox];
							}
						}
					}
				}
			}
		}
	}
}

void update_graph_conv_layer(Darknet::Layer & l, int batch, float learning_rate_init, float momentum, float decay)
{
	TAT(TATPARMS);

	const float learning_rate = learning_rate_init * l.learning_rate_scale;

	axpy_cpu(l.nweights, -decay * batch, l.weights, 1, l.weight_updates, 1);
	axpy_cpu(l.nweights, learning_rate / batch, l.weight_updates, 1, l.weights, 1);
	scal_cpu(l.nweights, momentum, l.weight_updates, 1);

	axpy_cpu(l.n, learning_rate / batch, l.bias_updates, 1, l.biases, 1);
	scal_cpu(l.n, momentum, l.bias_updates, 1);

	if (l.scales)
	{
		axpy_cpu(l.n, learning_rate / batch, l.scale_updates, 1, l.scales, 1);
		scal_cpu(l.n, momentum, l.scale_updates, 1);
	}

	if (l.graph_use_self)
	{
		const int self_count = l.n * l.graph_cpg;
		axpy_cpu(self_count, -decay * batch, l.graph_self_weights, 1, l.graph_self_weight_updates, 1);
		axpy_cpu(self_count, learning_rate / batch, l.graph_self_weight_updates, 1, l.graph_self_weights, 1);
		scal_cpu(self_count, momentum, l.graph_self_weight_updates, 1);
	}

	if (l.graph_edge_mode == 1)
	{
		const int kernel_count = l.groups * l.graph_k * edge_kernel_width(l);
		const int bias_count = l.groups * l.graph_k;
		axpy_cpu(kernel_count, -decay * batch, l.graph_edge_kernel, 1, l.graph_edge_kernel_updates, 1);
		axpy_cpu(kernel_count, learning_rate / batch, l.graph_edge_kernel_updates, 1, l.graph_edge_kernel, 1);
		scal_cpu(kernel_count, momentum, l.graph_edge_kernel_updates, 1);

		axpy_cpu(bias_count, learning_rate / batch, l.graph_edge_bias_updates, 1, l.graph_edge_biases, 1);
		scal_cpu(bias_count, momentum, l.graph_edge_bias_updates, 1);
	}
}
