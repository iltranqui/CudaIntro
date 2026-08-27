#include "im2col.hpp"
#include "col2im.hpp"
#include "gemm.hpp"
#ifdef DARKNET_HAS_FP4
#include "fp4_gemm.hpp"
#include "fp4_kernels.hpp"
#endif
#ifdef DARKNET_HAS_FP8
#include "fp8_calibration.hpp"
#include "fp8_conv.hpp"
#include "fp8_gemm.hpp"
#include "fp8_kernels.hpp"
#include "fp8_layer_release.hpp"
#endif
#include "darknet_internal.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>

namespace
{
	static auto & cfg_and_state = Darknet::CfgAndState::get();

	inline float eml_softplus(float x)
	{
		if (x > 20.0f) return x;
		if (x < -20.0f) return expf(x);
		return log1pf(expf(x));
	}

	inline float eml_sigmoid(float x)
	{
		if (x >= 0.0f)
		{
			const float z = expf(-x);
			return 1.0f / (1.0f + z);
		}
		const float z = expf(x);
		return z / (1.0f + z);
	}

	inline float eml_clamp_value(float x, float clamp)
	{
		return fminf(fmaxf(x, -clamp), clamp);
	}

#ifdef CUDNN
	inline cudnnDataType_t get_cudnn_16bit_data_type(const Darknet::Layer & l)
	{
#if defined(DARKNET_GPU_CUDA) && defined(CUDNN_DATA_BFLOAT16)
		if (l.cudnn_16bit_mode == DARKNET_CUDNN_16BIT_BF16)
		{
			return CUDNN_DATA_BFLOAT16;
		}
#endif
		return CUDNN_DATA_HALF;
	}

#ifdef DARKNET_GPU_ROCM
	inline int get_convolution_cudnn_tensor_format(const Darknet::Layer & /*l*/)
#else
	inline cudnnTensorFormat_t get_convolution_cudnn_tensor_format(const Darknet::Layer & /*l*/)
#endif
	{
		return CUDNN_TENSOR_NCHW;
	}

	inline void configure_convolution_cudnn_descriptors(
		Darknet::Layer *l,
		cudnnDataType_t data_type,
		cudnnTensorDescriptor_t src_desc,
		cudnnTensorDescriptor_t dst_desc,
		cudnnTensorDescriptor_t dsrc_desc,
		cudnnTensorDescriptor_t ddst_desc,
		cudnnFilterDescriptor_t weight_desc,
		cudnnFilterDescriptor_t dweight_desc,
		cudnnTensorDescriptor_t norm_dst_desc)
	{
		const auto tensor_format = get_convolution_cudnn_tensor_format(*l);
		const int input_channels = l->c;
		const int output_channels = l->out_c;
		const int grouped_input_channels = l->c / l->groups;

		CHECK_CUDNN(cudnnSetTensor4dDescriptor(dsrc_desc, tensor_format, data_type, l->batch, input_channels, l->h, l->w));
		CHECK_CUDNN(cudnnSetTensor4dDescriptor(ddst_desc, tensor_format, data_type, l->batch, output_channels, l->out_h, l->out_w));
		CHECK_CUDNN(cudnnSetFilter4dDescriptor(dweight_desc, data_type, tensor_format, l->n, grouped_input_channels, l->size, l->size));

		CHECK_CUDNN(cudnnSetTensor4dDescriptor(src_desc, tensor_format, data_type, l->batch, input_channels, l->h, l->w));
		CHECK_CUDNN(cudnnSetTensor4dDescriptor(dst_desc, tensor_format, data_type, l->batch, output_channels, l->out_h, l->out_w));
		CHECK_CUDNN(cudnnSetFilter4dDescriptor(weight_desc, data_type, tensor_format, l->n, grouped_input_channels, l->size, l->size));

		CHECK_CUDNN(cudnnSetTensor4dDescriptor(norm_dst_desc, tensor_format, data_type, l->batch, output_channels, l->out_h, l->out_w));
	}
#endif

	inline void binarize_cpu(float *input, int n, float *binary)
	{
		TAT_COMMENT(TATPARMS, "2024-05-14 inlined");

		for(int i = 0; i < n; ++i)
		{
			binary[i] = (input[i] > 0) ? 1 : -1;
		}
	}

	inline size_t get_workspace_size32(const Darknet::Layer & l)
	{
		TAT(TATPARMS);

		#ifdef CUDNN
		if (cfg_and_state.gpu_index >= 0)
		{
			size_t most = 0;
			size_t s = 0;
			CHECK_CUDNN(cudnnGetConvolutionForwardWorkspaceSize(cudnn_handle(),
					l.srcTensorDesc,
					l.weightDesc,
					l.convDesc,
					l.dstTensorDesc,
					l.fw_algo,
					&s));
			if (s > most)
			{
				most = s;
			}
			CHECK_CUDNN(cudnnGetConvolutionBackwardFilterWorkspaceSize(cudnn_handle(),
					l.srcTensorDesc,
					l.ddstTensorDesc,
					l.convDesc,
					l.dweightDesc,
					l.bf_algo,
					&s));
			if (s > most && l.train)
			{
				most = s;
			}
			CHECK_CUDNN(cudnnGetConvolutionBackwardDataWorkspaceSize(cudnn_handle(),
					l.weightDesc,
					l.ddstTensorDesc,
					l.convDesc,
					l.dsrcTensorDesc,
					l.bd_algo,
					&s));
			if (s > most && l.train)
			{
				most = s;
			}
			return most;
		}
		#endif
		if (l.xnor)
		{
			size_t re_packed_input_size = l.c * l.w * l.h * sizeof(float);
			size_t workspace_size = (size_t)l.bit_align*l.size*l.size*l.c * sizeof(float);
			if (workspace_size < re_packed_input_size)
			{
				workspace_size = re_packed_input_size;
			}

			return workspace_size;
		}

		return (size_t)l.out_h*l.out_w*l.size*l.size*(l.c / l.groups)*sizeof(float);
	}


	inline size_t get_workspace_size16(const Darknet::Layer & l)
	{
		TAT(TATPARMS);

		#if defined(CUDNN) && defined(CUDNN_HALF)
		if (cfg_and_state.gpu_index >= 0)
		{
			size_t most = 0;
			size_t s = 0;
			CHECK_CUDNN(cudnnGetConvolutionForwardWorkspaceSize(cudnn_handle(),
					l.srcTensorDesc16,
					l.weightDesc16,
					l.convDesc,
					l.dstTensorDesc16,
					l.fw_algo16,
					&s));
			if (s > most)
			{
				most = s;
			}
			CHECK_CUDNN(cudnnGetConvolutionBackwardFilterWorkspaceSize(cudnn_handle(),
					l.srcTensorDesc16,
					l.ddstTensorDesc16,
					l.convDesc,
					l.dweightDesc16,
					l.bf_algo16,
					&s));
			if (s > most && l.train)
			{
				most = s;
			}
			CHECK_CUDNN(cudnnGetConvolutionBackwardDataWorkspaceSize(cudnn_handle(),
					l.weightDesc16,
					l.ddstTensorDesc16,
					l.convDesc,
					l.dsrcTensorDesc16,
					l.bd_algo16,
					&s));
			if (s > most && l.train)
			{
				most = s;
			}
			return most;
		}
		#endif
		return 0;
	}


	inline size_t get_workspace_size_fp8(const Darknet::Layer & l)
	{
		TAT(TATPARMS);

		return l.fp8_workspace_size;
	}

#ifdef DARKNET_HAS_FP8
	inline size_t fp8_align_workspace_offset(const size_t value)
	{
		constexpr size_t alignment = 256;
		return (value + alignment - 1) & ~(alignment - 1);
	}


	/// Pick how many images to fold into one strided-batched FP8 GEMM: the largest divisor of the
	/// layer batch whose staging buffers stay within a fixed workspace budget.  Small nets get full
	/// batching (fewer kernel launches); high-resolution nets fall back to smaller chunks.
	inline int fp8_pick_gemm_batch(const int layer_batch, const size_t per_image_bytes)
	{
		constexpr size_t budget = 128ULL * 1024ULL * 1024ULL;
		for (int chunk = layer_batch; chunk > 1; --chunk)
		{
			if (layer_batch % chunk == 0 && per_image_bytes * static_cast<size_t>(chunk) <= budget)
			{
				return chunk;
			}
		}
		return 1;
	}


	inline bool fp8_layer_env_is_set(const char * name)
	{
		const char * const value = std::getenv(name);
		return value != nullptr && value[0] != '\0' && !(value[0] == '0' && value[1] == '\0');
	}
#endif


	inline void get_mean_array(const float * src, const size_t size, const size_t filters, float * mean_arr)
	{
		TAT(TATPARMS);

		size_t counter = 0;
		for (size_t i = 0; i < size; i += size / filters)
		{
			mean_arr[counter++] = fabs(src[i]);
		}
	}


	// binary transpose
	inline size_t binary_transpose_align_input(const int k, const int n, const float * b, char **t_bit_input, const size_t ldb_align, const int bit_align)
	{
		TAT(TATPARMS);

		size_t new_ldb = k + (ldb_align - k%ldb_align); // (k / 8 + 1) * 8;
		size_t t_intput_size = new_ldb * bit_align;// n;
		size_t t_bit_input_size = t_intput_size / 8;// +1;

		memset(*t_bit_input, 0, t_bit_input_size * sizeof(char));

#ifdef DARKNET_GPU
		transpose_bin_gpu((uint8_t*)b, (uint8_t*)*t_bit_input, k, n, bit_align, new_ldb, 8);
#else
		transpose_bin((uint32_t*)b, (uint32_t*)*t_bit_input, k, n, bit_align, new_ldb, 8);
#endif

		return t_intput_size;
	}


	inline Darknet::Image *get_weights(const Darknet::Layer & l)
	{
		TAT(TATPARMS);

		Darknet::Image * weights = (Darknet::Image *)xcalloc(l.n, sizeof(Darknet::Image));
		for (int i = 0; i < l.n; ++i)
		{
			weights[i] = Darknet::copy_image(get_convolutional_weight(l, i));
			Darknet::normalize_image(weights[i]);
		}

		return weights;
	}
}


void swap_binary(Darknet::Layer * l)
{
	TAT(TATPARMS);

	float *swap = l->weights;
	l->weights = l->binary_weights;
	l->binary_weights = swap;

	#ifdef DARKNET_GPU
	swap = l->weights_gpu;
	l->weights_gpu = l->binary_weights_gpu;
	l->binary_weights_gpu = swap;
	#endif
}


void binarize_weights(float * weights, int n, int size, float * binary)
{
	TAT(TATPARMS);

	for (int f = 0; f < n; ++f)
	{
		float mean = 0;
		for (int i = 0; i < size; ++i)
		{
			mean += fabs(weights[f*size + i]);
		}
		mean = mean / size;
		for(int i = 0; i < size; ++i)
		{
			binary[f*size + i] = (weights[f*size + i] > 0) ? mean: -mean;
		}
	}
}


void binarize_input(float *input, int n, int size, float *binary)
{
	TAT(TATPARMS);

	for(int s = 0; s < size; ++s)
	{
		float mean = 0;
		for (int i = 0; i < n; ++i)
		{
			mean += fabs(input[i*size + s]);
		}

		mean = mean / n;

		for (int i = 0; i < n; ++i)
		{
			binary[i*size + s] = (input[i*size + s] > 0) ? mean : -mean;
		}
	}
}


int convolutional_out_height(const Darknet::Layer & l)
{
	TAT(TATPARMS);

	return (l.h + 2 * l.pad - l.size) / l.stride_y + 1;
}


int convolutional_out_width(const Darknet::Layer & l)
{
	TAT(TATPARMS);

	return (l.w + 2 * l.pad - l.size) / l.stride_x + 1;
}


Darknet::Image get_convolutional_image(const Darknet::Layer & l)
{
	TAT(TATPARMS);

	const int h = convolutional_out_height(l);
	const int w = convolutional_out_width(l);
	const int c = l.n;

	return Darknet::float_to_image(w, h, c, l.output);
}


Darknet::Image get_convolutional_delta(const Darknet::Layer & l)
{
	TAT(TATPARMS);

	const int h = convolutional_out_height(l);
	const int w = convolutional_out_width(l);
	const int c = l.n;

	return Darknet::float_to_image(w, h, c, l.delta);
}


size_t get_convolutional_workspace_size(const Darknet::Layer & l)
{
	TAT(TATPARMS);

	size_t workspace_size = get_workspace_size32(l);
	size_t workspace_size16 = get_workspace_size16(l);
	if (workspace_size16 > workspace_size)
	{
		workspace_size = workspace_size16;
	}
	const size_t workspace_size_fp8 = get_workspace_size_fp8(l);
	if (workspace_size_fp8 > workspace_size)
	{
		workspace_size = workspace_size_fp8;
	}
#ifdef DARKNET_HAS_FP4
	workspace_size = std::max(workspace_size, l.fp4_workspace_size);
#endif

	return workspace_size;
}


// **********************************************


#ifdef DARKNET_GPU
#ifdef CUDNN


void create_convolutional_cudnn_tensors(Darknet::Layer *l)
{
	TAT(TATPARMS);

	cudnnTensorDescriptor_t * const tensor_descs[] =
	{
		&l->normTensorDesc,
		&l->normDstTensorDesc,
		&l->srcTensorDesc,
		&l->dstTensorDesc,
		&l->dsrcTensorDesc,
		&l->ddstTensorDesc,
		&l->normDstTensorDescF16,
		&l->srcTensorDesc16,
		&l->dstTensorDesc16,
		&l->dsrcTensorDesc16,
		&l->ddstTensorDesc16
	};
	for (auto desc : tensor_descs)
	{
		CHECK_CUDNN(cudnnCreateTensorDescriptor(desc));
	}

	cudnnFilterDescriptor_t * const filter_descs[] =
	{
		&l->weightDesc,
		&l->dweightDesc,
		&l->weightDesc16,
		&l->dweightDesc16
	};
	for (auto desc : filter_descs)
	{
		CHECK_CUDNN(cudnnCreateFilterDescriptor(desc));
	}

	CHECK_CUDNN(cudnnCreateConvolutionDescriptor(&l->convDesc));
}


void cudnn_convolutional_setup(Darknet::Layer *l, int cudnn_preference, size_t workspace_size_specify)
{
	TAT(TATPARMS);

// CUDNN_HALF
	// TRUE_HALF_CONFIG is only supported on architectures with true fp16 support (compute capability 5.3 and 6.0):
	//   Tegra X1, Jetson TX1, DRIVE CX, DRIVE PX, Quadro GP100, Tesla P100
	// PSEUDO_HALF_CONFIG is required for Tensor Cores - our case!

	cudnnDataType_t data_type = CUDNN_DATA_FLOAT;

#if (CUDNN_MAJOR >= 7)
	// Tensor Core uses CUDNN_TENSOR_OP_MATH instead of CUDNN_DEFAULT_MATH
	// For *_ALGO_WINOGRAD_NONFUSED can be used CUDNN_DATA_FLOAT
	// otherwise Input, Filter and Output descriptors (xDesc, yDesc, wDesc, dxDesc, dyDesc and dwDesc as applicable) have dataType = CUDNN_DATA_HALF
	// Three techniques for training using Mixed-precision: https://devblogs.nvidia.com/mixed-precision-training-deep-neural-networks/
	// 1. Accumulation into FP32
	// 2. Loss Scaling - required only for: activation gradients. We do not use.
	// 3. FP32 Master Copy of Weights
	// More: http://docs.nvidia.com/deeplearning/sdk/cudnn-developer-guide/index.html#tensor_ops
	if (l->groups < 1) l->groups = 1;
	if (l->stride_x < 1) l->stride_x = 1;
	if (l->stride_y < 1) l->stride_y = 1;
	CHECK_CUDNN(cudnnSetConvolutionGroupCount(l->convDesc, l->groups));
	CHECK_CUDNN(cudnnSetConvolutionMathType(l->convDesc, CUDNN_TENSOR_OP_MATH));
#if ((CUDNN_MAJOR*10 + CUDNN_MINOR) >= 72)   // cuDNN >= 7.2
	//CHECK_CUDNN(cudnnSetConvolutionMathType(l->convDesc, CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION)); // reduces the speed of regular and group convolution
#endif
#else   //if (CUDNN_MAJOR >= 7)
	if (l->groups > 1)
	{
		darknet_fatal_error(DARKNET_LOC, "CUDNN < 7 doesn't support groups, please upgrade!");
	}
#endif

	// INT8_CONFIG, INT8_EXT_CONFIG, INT8x4_CONFIG and INT8x4_EXT_CONFIG are only supported
	//   on architectures with DP4A support (compute capability 6.1 and later).
	//cudnnDataType_t data_type = CUDNN_DATA_INT8;

	configure_convolution_cudnn_descriptors(
		l,
		data_type,
		l->srcTensorDesc,
		l->dstTensorDesc,
		l->dsrcTensorDesc,
		l->ddstTensorDesc,
		l->weightDesc,
		l->dweightDesc,
		l->normDstTensorDesc);

	const cudnnDataType_t data_type_16 = get_cudnn_16bit_data_type(*l);

	configure_convolution_cudnn_descriptors(
		l,
		data_type_16,
		l->srcTensorDesc16,
		l->dstTensorDesc16,
		l->dsrcTensorDesc16,
		l->ddstTensorDesc16,
		l->weightDesc16,
		l->dweightDesc16,
		l->normDstTensorDescF16);

	// batch norm
	CHECK_CUDNN(cudnnSetTensor4dDescriptor(l->normTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, l->out_c, 1, 1));

#if (CUDNN_MAJOR >= 6)
	CHECK_CUDNN(cudnnSetConvolution2dDescriptor(l->convDesc, l->pad * l->dilation, l->pad * l->dilation, l->stride_y, l->stride_x, l->dilation, l->dilation, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));    // cudnn >= 6.0
#else
	CHECK_CUDNN(cudnnSetConvolution2dDescriptor(l->convDesc, l->pad * l->dilation, l->pad * l->dilation, l->stride_y, l->stride_x, l->dilation, l->dilation, CUDNN_CROSS_CORRELATION));    // cudnn 5.1
#endif


#if CUDNN_MAJOR >= 8

	if (cudnn_preference == cudnn_smallest)
	{
		workspace_size_specify = 0;
	}

	size_t free_memory, total_memory;
	int requested_algo_count = 0, returned_algo_count = 0;
	int found_conv_algorithm = 0;
	float min_time = 1000000;   // 1000 sec

	// FWD
	cudnnConvolutionFwdAlgoPerf_t conv_fwd_results[100];
	CHECK_CUDNN(cudnnGetConvolutionForwardAlgorithmMaxCount(cudnn_handle(), &requested_algo_count));

	CHECK_CUDNN(cudnnGetConvolutionForwardAlgorithm_v7(cudnn_handle(),
		l->srcTensorDesc,
		l->weightDesc,
		l->convDesc,
		l->dstTensorDesc,
		requested_algo_count, // (cudnnConvolutionFwdPreference_t)forward_algo,
		&returned_algo_count, // workspace_size_specify,
		conv_fwd_results));

	CHECK_CUDA(cudaMemGetInfo(&free_memory, &total_memory));
//	*cfg_and_state.output << "CUDA memory: free=" << size_to_IEC_string(free_memory) << " total=" << size_to_IEC_string(total_memory) << std::endl;

#if 0
	cudaDeviceProp prop;
	CHECK_CUDA(cudaGetDeviceProperties(&prop, std::max(0, cfg_and_state.gpu_index)));
	const auto compu_capability_ver = prop.major * 10 + prop.minor; // e.g., "86" for RTX30xx, or "89" for RTX40xx
#endif

	const bool is_training	= (cfg_and_state.command == "detector" and cfg_and_state.function == "train");
	const bool is_map		= (cfg_and_state.command == "detector" and cfg_and_state.function == "map");

	found_conv_algorithm = 0;
	min_time = 1000000;   // 1000 sec

	for (int i = 0; i < returned_algo_count; i++)
	{
		/* Summary of a 2015 blog post on cuDNN:  https://developer.nvidia.com/blog/cudnn-v2-higher-performance-deep-learning-gpus/
		 *
		 * There are 4 algorithms for forward convolution:
		 *
		 * - IMPLICIT_GEMM
		 * - IMPLICIT_PRECOMP_GEMM
		 * - GEMM
		 * - DIRECT
		 *
		 * IMPLICIT_GEMM supports all input sizes and requires no extra working space.  When there isn't much memory, or the
		 * network is large, this is the algorithm to use.
		 *
		 * IMPLICIT_PRECOMP_GEMM is a modification of IMPLICIT_GEMM which uses a small amount of working space to achieve
		 * higher performance than IMPLICIT_GEMM.
		 *
		 * GEMM is an "im2col" approach that requires significant working space but in some cases is the fastest approach.
		 *
		 * DIRECT is not implemented but is a placeholder for a future feature.
		 */

#if 0
		*cfg_and_state.output
			<< "FWD ALGO:"
			<< " i="			<< i
//			<< " name="			<< std::left << std::setw(22) << to_string(conv_fwd_results[i].algo)
			<< " algo="			<< conv_fwd_results[i].algo
			<< " status="		<< conv_fwd_results[i].status
			<< " time="			<< conv_fwd_results[i].time
			<< " memory="		<< size_to_IEC_string(conv_fwd_results[i].memory)
			<< " determinism="	<< conv_fwd_results[i].determinism
			<< " math="			<< conv_fwd_results[i].mathType
			<< std::endl;
#endif

		if (conv_fwd_results[i].status != CUDNN_STATUS_SUCCESS)
		{
			// algorithm is not supported, so skip to the next one
			continue;
		}

		if (conv_fwd_results[i].algo == CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED)
		{
			/// @todo V3 why are we skipping this algorithm?
			continue;
		}

		if (conv_fwd_results[i].algo == CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM)
		{
			/* The IMPLICIT_PRECOMP_GEMM algorithm causes problems for some people.  Maybe due to low memory?
			 *
			 * For example, see:  https://github.com/hank-ai/darknet/pull/36
			 *
			 * If you get a cuDNN status of BAD_PARAM during the mAP calculations, this algorithms may need to be skipped.
			 * For now, because we don't understand the exact cause of the error, we'll only skip it on older GPUs.
			 *
			 *		- major=6, minor=x, "Pascal":  GTX 10xx, Quadro Pxxxx, Tesla P4
			 *		- major=7, minor=5, "Turing":  RTX 20xx, GTX 16xx, Quadro RTX, Tesla T4
			 *		- major=8, minor=6, "Ampere":  RTX 30xx, A6xxx, A5xxx
			 *		- major=8, minor=7, "Ampere":  Jetson Orin -- reported on Discord on 2024-09-09, training on an Orin device
			 *		- major=8, minor=9, "Lovelace":  RTX 40xx
			 *		- major=9, minor=x, "Hopper":  RTX 50xx
			 *
			 * If you think you've run into this error and you'd like to skip this algorithm, change the version number
			 * we verify against on the next line from "86" to a very large value such as "999".
			 *
			 * ------------------------
			 *
			 * 2024-10-16 update:  Just ran into this error on my RTX 3090 when training YOLOv4-tiny-3L.  Network was large
			 * (1440x800x3, subdiv=4) but only used 15 GiB out of the 24 GiB available.  For now, disabling this algo until
			 * a proper fix can be found.
			 *
			 * 2024-12-02 update:  This doesn't work while training, but no reason why we cannot use it during inference.
			 */
			if (is_training or is_map)
			{
				continue;
			}
		}

		if (conv_fwd_results[i].time >= min_time)
		{
			// this algorithm is slower, or the same as a previous algo we already selected
			continue;
		}

		if (conv_fwd_results[i].memory < free_memory &&
			(conv_fwd_results[i].memory <= workspace_size_specify || cudnn_preference == cudnn_fastest))
		{
			found_conv_algorithm = 1;
			l->fw_algo = conv_fwd_results[i].algo;

			// use the algo with the lowest time; if there are multiple algos with the exact same time,
			// then we end up using the first one in the list returned by cudnn
			min_time = conv_fwd_results[i].time;
		}
	}

	if (!found_conv_algorithm)
	{
		darknet_fatal_error(DARKNET_LOC, "cuDNN did not find a usable algorithm to use for forward convolution");
	}

	// Bwd-Data
	cudnnConvolutionBwdDataAlgoPerf_t conv_bwd_data_results[100];
	CHECK_CUDNN(cudnnGetConvolutionBackwardDataAlgorithmMaxCount(cudnn_handle(), &requested_algo_count));

	CHECK_CUDNN(cudnnGetConvolutionBackwardDataAlgorithm_v7(cudnn_handle(),
		l->weightDesc,
		l->ddstTensorDesc,
		l->convDesc,
		l->dsrcTensorDesc,
		requested_algo_count, // (cudnnConvolutionFwdPreference_t)forward_algo,
		&returned_algo_count, // workspace_size_specify,
		&conv_bwd_data_results[0]));

	CHECK_CUDA(cudaMemGetInfo(&free_memory, &total_memory));

	found_conv_algorithm = 0;
	min_time = 1000000;   // 1000 sec
	for (int i = 0; i < returned_algo_count; i++)
	{
		if (conv_bwd_data_results[i].status == CUDNN_STATUS_SUCCESS &&
			conv_bwd_data_results[i].memory < free_memory &&
			(conv_bwd_data_results[i].memory <= workspace_size_specify || cudnn_preference == cudnn_fastest) &&
			conv_bwd_data_results[i].time < min_time)
		{
			found_conv_algorithm = 1;
			l->bd_algo = conv_bwd_data_results[i].algo;
			min_time = conv_bwd_data_results[i].time;
		}
	}

	if (!found_conv_algorithm)
	{
		darknet_fatal_error(DARKNET_LOC, "cuDNN did not find a usable algorithm to use for backward convolution");
	}

	// Bwd-Filters
	cudnnConvolutionBwdFilterAlgoPerf_t conv_bwd_filter_results[100];
	CHECK_CUDNN(cudnnGetConvolutionBackwardFilterAlgorithmMaxCount(cudnn_handle(), &requested_algo_count));

	CHECK_CUDNN(cudnnGetConvolutionBackwardFilterAlgorithm_v7(cudnn_handle(),
		l->srcTensorDesc,
		l->ddstTensorDesc,
		l->convDesc,
		l->dweightDesc,
		requested_algo_count, // (cudnnConvolutionFwdPreference_t)forward_algo,
		&returned_algo_count, // workspace_size_specify,
		&conv_bwd_filter_results[0]));

	CHECK_CUDA(cudaMemGetInfo(&free_memory, &total_memory));

	found_conv_algorithm = 0;
	min_time = 1000000;   // 1000 sec
	for (int i = 0; i < returned_algo_count; i++)
	{
		if (conv_bwd_filter_results[i].status == CUDNN_STATUS_SUCCESS &&
			conv_bwd_filter_results[i].memory < free_memory &&
			(conv_bwd_filter_results[i].memory <= workspace_size_specify || cudnn_preference == cudnn_fastest) &&
			conv_bwd_filter_results[i].time < min_time)
		{
			found_conv_algorithm = 1;
			l->bf_algo = conv_bwd_filter_results[i].algo;
			min_time = conv_bwd_filter_results[i].time;
		}
	}

	if (!found_conv_algorithm)
	{
		darknet_fatal_error(DARKNET_LOC, "cuDNN did not find BWD-filter algo for convolution");
	}

#else   // CUDNN_MAJOR >= 8

	int forward_algo = CUDNN_CONVOLUTION_FWD_PREFER_FASTEST;
	int backward_algo = CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST;
	int backward_filter = CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST;
	if (cudnn_preference == cudnn_smallest)
	{
		forward_algo = CUDNN_CONVOLUTION_FWD_NO_WORKSPACE;
		backward_algo = CUDNN_CONVOLUTION_BWD_DATA_NO_WORKSPACE;
		backward_filter = CUDNN_CONVOLUTION_BWD_FILTER_NO_WORKSPACE;
		*cfg_and_state.output << " CUDNN-slow ";
	}
	if (cudnn_preference == cudnn_specify)
	{
		forward_algo = CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT;
		backward_algo = CUDNN_CONVOLUTION_BWD_DATA_SPECIFY_WORKSPACE_LIMIT;
		backward_filter = CUDNN_CONVOLUTION_BWD_FILTER_SPECIFY_WORKSPACE_LIMIT;
	}

	CHECK_CUDNN(cudnnGetConvolutionForwardAlgorithm(cudnn_handle(),
			l->srcTensorDesc,
			l->weightDesc,
			l->convDesc,
			l->dstTensorDesc,
			(cudnnConvolutionFwdPreference_t)forward_algo,
			workspace_size_specify,
			&l->fw_algo));

	CHECK_CUDNN(cudnnGetConvolutionBackwardDataAlgorithm(cudnn_handle(),
		l->weightDesc,
		l->ddstTensorDesc,
		l->convDesc,
		l->dsrcTensorDesc,
		(cudnnConvolutionBwdDataPreference_t)backward_algo,
		workspace_size_specify,
		&l->bd_algo));

	CHECK_CUDNN(cudnnGetConvolutionBackwardFilterAlgorithm(cudnn_handle(),
		l->srcTensorDesc,
		l->ddstTensorDesc,
		l->convDesc,
		l->dweightDesc,
		(cudnnConvolutionBwdFilterPreference_t)backward_filter,
		workspace_size_specify,
		&l->bf_algo));
#endif  // CUDNN_MAJOR >= 8


	//if (data_type == CUDNN_DATA_HALF)
	{
		// HALF-16 if (data_type == CUDNN_DATA_HALF)
		l->fw_algo16 = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;
		l->bd_algo16 = CUDNN_CONVOLUTION_BWD_DATA_ALGO_1;
		l->bf_algo16 = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1;

		// FLOAT-32 if (data_type == CUDNN_DATA_FLOAT)
		//l->fw_algo16 = CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED;
		//l->bd_algo16 = CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD_NONFUSED;
		//l->bf_algo16 = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD_NONFUSED;
	}
}

void set_convolutional_cudnn_16bit_mode(Darknet::Layer *l, int mode)
{
	TAT(TATPARMS);

	if (mode != DARKNET_CUDNN_16BIT_HALF && mode != DARKNET_CUDNN_16BIT_BF16)
	{
		darknet_fatal_error(DARKNET_LOC, "invalid cuDNN 16-bit mode %d", mode);
	}
	if (l->cudnn_16bit_mode == mode)
	{
		return;
	}

	l->cudnn_16bit_mode = mode;
	cudnn_convolutional_setup(l, cudnn_fastest, 0);
	l->workspace_size = get_convolutional_workspace_size(*l);
	if (l->weights_gpu && l->weights_gpu16 && l->nweights > 0)
	{
		cuda_convert_f32_to_cudnn_16bit(l->weights_gpu, l->nweights, l->weights_gpu16, l->cudnn_16bit_mode);
	}
}


#endif
#endif

// **********************************************

#ifdef DARKNET_GPU

#ifdef DARKNET_HAS_FP4
void fp4_clear_convolutional_relay(Darknet::Layer & l)
{
	if (l.fp4_relay_scales_gpu)
	{
		CHECK_CUDA(cudaFree(l.fp4_relay_scales_gpu));
		l.fp4_relay_scales_gpu = nullptr;
	}
	if (l.fp4_relay_gpu)
	{
		CHECK_CUDA(cudaFree(l.fp4_relay_gpu));
		l.fp4_relay_gpu = nullptr;
	}
	l.fp4_relay_packed_bytes = 0;
	l.fp4_relay_scale_bytes = 0;
	l.fp4_relay_next_layer = -1;
	l.fp4_relay_source_layer = -1;
	l.fp4_relay_valid = 0;
}

void fp4_release_convolutional_layer(Darknet::Layer & l)
{
	fp4_clear_convolutional_relay(l);
	if (l.fp4_gemm_plan)
	{
		Darknet::fp4_gemm_plan_destroy(static_cast<Darknet::Fp4GemmPlan *>(l.fp4_gemm_plan));
		l.fp4_gemm_plan = nullptr;
	}
	if (l.fp4_wgrad_gemm_plan)
	{
		Darknet::fp4_gemm_plan_destroy(static_cast<Darknet::Fp4GemmPlan *>(l.fp4_wgrad_gemm_plan));
		l.fp4_wgrad_gemm_plan = nullptr;
	}
	if (l.fp4_dgrad_gemm_plan)
	{
		Darknet::fp4_gemm_plan_destroy(static_cast<Darknet::Fp4GemmPlan *>(l.fp4_dgrad_gemm_plan));
		l.fp4_dgrad_gemm_plan = nullptr;
	}
	l.fp4_workspace_size = 0;
	l.fp4_eligible = 0;
	l.fp4_train_eligible = 0;
	l.fp4_weights_prepacked = 0;
	if (l.fp4_amax_gpu)
	{
		cuda_free(l.fp4_amax_gpu);
		l.fp4_amax_gpu = nullptr;
	}
}

bool fp4_prepare_convolutional_calibration_layer(Darknet::Layer & l)
{
	TAT(TATPARMS);

	fp4_release_convolutional_layer(l);
	if (l.type != Darknet::ELayerType::CONVOLUTIONAL || l.share_layer != nullptr || l.binary || l.xnor ||
		l.groups != 1 || l.c <= 0 || l.size <= 0 || (static_cast<long long>(l.c) * l.size * l.size) % 16 != 0)
	{
		return false;
	}

	l.fp4_amax_gpu = cuda_make_array(nullptr, 1);
	Darknet::fp4_clear_amax_gpu(l.fp4_amax_gpu);
	return true;
}

bool fp4_setup_convolutional_layer(Darknet::Layer & l, const bool training)
{
	fp4_release_convolutional_layer(l);
	if (!Darknet::fp4_runtime_supported() || l.type != Darknet::ELayerType::CONVOLUTIONAL ||
		l.share_layer || l.binary || l.xnor || l.groups != 1 || l.batch <= 0 || l.n <= 0 ||
		l.c <= 0 || l.size <= 0 || l.out_h <= 0 || l.out_w <= 0 || !l.weights_gpu || !l.output_gpu)
	{
		return false;
	}

	const int filters = l.n;
	const int kernel = l.c * l.size * l.size;
	const int spatial = l.out_h * l.out_w;
	const auto align_workspace = [](const size_t value) { return (value + 255U) & ~size_t{255U}; };
	// The inference path quantizes static weights once during model preparation.
	// Warm-up only verifies the ready-to-run path.  Training
	// must quantize live weights each iteration, so it deliberately does not
	// retain the packed representation.
	l.fp4_gemm_plan = Darknet::fp4_gemm_plan_create({1, filters, spatial, kernel, !training});
	if (!l.fp4_gemm_plan)
	{
		fp4_release_convolutional_layer(l);
		return false;
	}

	const size_t matrix = static_cast<size_t>(kernel) * spatial;
	const size_t forward_bytes = align_workspace((2U * matrix + static_cast<size_t>(filters) * spatial * l.batch) * sizeof(float)) +
		Darknet::fp4_gemm_workspace_bytes(static_cast<Darknet::Fp4GemmPlan *>(l.fp4_gemm_plan));
	l.fp4_workspace_size = forward_bytes;
	l.fp4_eligible = 1;
	if (!training)
	{
		// calculate_binary_weights() runs after loading and batch-norm fusion in
		// every normal inference loader.  Prepack static FP32 weights now, before
		// the first request, instead of charging the first image (or every image)
		// for NVFP4 weight quantization.
		l.fp4_weights_prepacked = Darknet::fp4_gemm_prepare_cached_left_operand(
			static_cast<Darknet::Fp4GemmPlan *>(l.fp4_gemm_plan), l.weights_gpu) ? 1 : 0;
	}

	if (!training)
	{
		return true;
	}
	l.fp4_wgrad_gemm_plan = Darknet::fp4_gemm_plan_create({1, filters, kernel, l.batch * spatial});
	l.fp4_dgrad_gemm_plan = Darknet::fp4_gemm_plan_create({1, kernel, spatial, filters});
	const size_t wgrad_bytes = l.fp4_wgrad_gemm_plan ? align_workspace((static_cast<size_t>(filters + kernel) * l.batch * spatial +
		static_cast<size_t>(kernel) * spatial + static_cast<size_t>(filters) * kernel) * sizeof(float)) +
		Darknet::fp4_gemm_workspace_bytes(static_cast<Darknet::Fp4GemmPlan *>(l.fp4_wgrad_gemm_plan)) : 0U;
	const size_t dgrad_bytes = (static_cast<size_t>(kernel) * filters + static_cast<size_t>(spatial) * filters +
		matrix * l.batch + static_cast<size_t>(l.c) * l.h * l.w) * sizeof(float);
	const size_t dgrad_total = l.fp4_dgrad_gemm_plan ? align_workspace(dgrad_bytes) +
		Darknet::fp4_gemm_workspace_bytes(static_cast<Darknet::Fp4GemmPlan *>(l.fp4_dgrad_gemm_plan)) : 0U;
	l.fp4_workspace_size = std::max(std::max(l.fp4_workspace_size, wgrad_bytes), dgrad_total);
	l.fp4_train_eligible = 1;
	return true;
}

bool fp4_setup_convolutional_relay(Darknet::Layer & producer, Darknet::Layer & consumer,
	const int producer_index, const int consumer_index)
{
	// A middle layer can be both a consumer of the previous relay and the
	// producer of the next one.  Clearing its outgoing allocation must not drop
	// the incoming source recorded by the preceding link, otherwise a run of
	// three or more compatible convolutions would only retain the final link.
	const int incoming_source = producer.fp4_relay_source_layer;
	fp4_clear_convolutional_relay(producer);
	producer.fp4_relay_source_layer = incoming_source;
	if (!producer.fp4_eligible || !consumer.fp4_eligible ||
		!producer.fp4_gemm_plan || !consumer.fp4_gemm_plan ||
		producer.batch != consumer.batch || producer.out_c != consumer.c ||
		producer.out_h != consumer.h || producer.out_w != consumer.w ||
		producer.antialiasing || producer.coordconv || consumer.share_layer ||
		consumer.groups != 1 || consumer.size != 1 || consumer.stride != 1 ||
		consumer.stride_x != 1 || consumer.stride_y != 1 || consumer.pad != 0 ||
		consumer.dilation != 1 ||
		!Darknet::fp4_gemm_supports_prequantized_right(
			static_cast<const Darknet::Fp4GemmPlan *>(consumer.fp4_gemm_plan)))
	{
		return false;
	}

	const int spatial = consumer.h * consumer.w;
	const size_t packed_per_image = Darknet::fp4_cublaslt_packed_bytes(spatial, consumer.c);
	const size_t scale_per_image = Darknet::fp4_cublaslt_scale_bytes(spatial, consumer.c);
	if (packed_per_image == 0 || scale_per_image == 0)
	{
		return false;
	}
	const size_t packed_bytes = packed_per_image * static_cast<size_t>(consumer.batch);
	const size_t scale_bytes = scale_per_image * static_cast<size_t>(consumer.batch);
	if (cudaMalloc(reinterpret_cast<void **>(&producer.fp4_relay_gpu), packed_bytes) != cudaSuccess ||
		cudaMalloc(reinterpret_cast<void **>(&producer.fp4_relay_scales_gpu), scale_bytes) != cudaSuccess)
	{
		fp4_clear_convolutional_relay(producer);
		producer.fp4_relay_source_layer = incoming_source;
		return false;
	}

	producer.fp4_relay_packed_bytes = packed_per_image;
	producer.fp4_relay_scale_bytes = scale_per_image;
	producer.fp4_relay_next_layer = consumer_index;
	consumer.fp4_relay_source_layer = producer_index;
	return true;
}
#endif

#ifdef DARKNET_HAS_FP8
void fp8_clear_convolutional_relay(Darknet::Layer & l)
{
	if (l.fp8_relay_gpu)
	{
		CHECK_CUDA(cudaFree(l.fp8_relay_gpu));
		l.fp8_relay_gpu = nullptr;
	}
	if (l.fp8_relay_amax_gpu)
	{
		cuda_free(l.fp8_relay_amax_gpu);
		l.fp8_relay_amax_gpu = nullptr;
	}
	l.fp8_relay_bytes = 0;
	l.fp8_relay_next_layer = -1;
	l.fp8_relay_source_layer = -1;
	l.fp8_relay_valid = 0;
	l.fp8_relay_enabled = 0;
	l.fp8_relay_saturation_fallback = 0;
}

void fp8_release_convolutional_layer(Darknet::Layer & l)
{
	TAT(TATPARMS);
	fp8_clear_convolutional_relay(l);

	Darknet::fp8_release_device_ptr(l.weights_fp8_gpu);
	Darknet::fp8_release_device_ptr(l.weights_fp8_t_gpu);
	Darknet::fp8_release_device_ptr(l.weights_fp8_nhwc_gpu);
	Darknet::fp8_release_cuda_alloc(l.fp8_weight_scale_gpu);
	Darknet::fp8_release_cuda_alloc(l.fp8_input_scale_gpu);
	Darknet::fp8_release_cuda_alloc(l.fp8_dy_scale_gpu);
	Darknet::fp8_release_cuda_alloc(l.fp8_input_amax_gpu);
	Darknet::fp8_release_cuda_alloc(l.fp8_dy_amax_gpu);
	Darknet::fp8_release_cuda_alloc(l.fp8_amax_gpu);
	Darknet::fp8_release_cuda_alloc(l.fp8_weight_scale_state_gpu);
	Darknet::fp8_release_cuda_alloc(l.fp8_input_scale_state_gpu);
	Darknet::fp8_release_cuda_alloc(l.fp8_dy_scale_state_gpu);
	Darknet::fp8_release_plan(l.fp8_gemm_plan, Darknet::fp8_gemm_plan_destroy);
	Darknet::fp8_release_plan(l.fp8_wgrad_gemm_plan, Darknet::fp8_gemm_plan_destroy);
	Darknet::fp8_release_plan(l.fp8_dgrad_gemm_plan, Darknet::fp8_gemm_plan_destroy);
	Darknet::fp8_release_plan(l.fp8_conv_fwd_plan, Darknet::fp8_conv_plan_destroy);
	Darknet::fp8_release_plan(l.fp8_conv_wgrad_plan, Darknet::fp8_conv_plan_destroy);
	Darknet::fp8_release_plan(l.fp8_conv_dgrad_plan, Darknet::fp8_conv_plan_destroy);

	l.fp8_workspace_size = 0;
	l.fp8_eligible = 0;
	l.fp8_train_eligible = 0;
	l.fp8_k_pad = 0;
	l.fp8_forward_batch = 0;
	l.fp8_dgrad_batch = 0;
	l.fp8_wgrad_batch = 0;
	l.fp8_wgrad_direct_update = 0;
	l.fp8_dgrad_direct_update = 0;
	l.fp8_dy_amax_valid = 0;
	l.fp8_weight_scale_host = 1.0f;
	l.fp8_input_scale_host = 1.0f;
	l.fp8_dy_scale_host = 1.0f;
	l.fp8_weight_amax_next = 0;
	l.fp8_input_amax_next = 0;
	l.fp8_dy_amax_next = 0;
	l.fp8_weight_amax_count = 0;
	l.fp8_input_amax_count = 0;
	l.fp8_dy_amax_count = 0;
	l.fp8_graph_activation_fused = 0;
	for (int idx = 0; idx < Darknet::kFp8AmaxHistoryLength; ++idx)
	{
		l.fp8_weight_amax_history[idx] = 0.0f;
		l.fp8_input_amax_history[idx] = 0.0f;
		l.fp8_dy_amax_history[idx] = 0.0f;
	}
}


bool fp8_setup_convolutional_relay(Darknet::Layer & producer, Darknet::Layer & consumer,
	const int producer_index, const int consumer_index)
{
	// A middle convolution can be both the previous consumer and the following
	// producer, so only its outgoing allocation is replaced here.
	const int incoming_source = producer.fp8_relay_source_layer;
	fp8_clear_convolutional_relay(producer);
	producer.fp8_relay_source_layer = incoming_source;

	// The relay layout is FP8 NHWC and is consumed directly by the cuDNN
	// Frontend convolution graph.  GEMM plans use another layout and fall back
	// rather than pretending that a reinterpret cast is valid.
	if (!producer.fp8_eligible || !consumer.fp8_eligible ||
		consumer.fp8_conv_fwd_plan == nullptr || consumer.weights_fp8_nhwc_gpu == nullptr ||
		producer.batch != consumer.batch || producer.out_c != consumer.c ||
		producer.out_h != consumer.h || producer.out_w != consumer.w ||
		producer.antialiasing || producer.coordconv || consumer.share_layer ||
		consumer.groups != 1 || consumer.batch_normalize)
	{
		return false;
	}

	const size_t relay_bytes = static_cast<size_t>(producer.batch) * producer.out_c * producer.out_h * producer.out_w;
	if (relay_bytes == 0 || cudaMalloc(reinterpret_cast<void **>(&producer.fp8_relay_gpu), relay_bytes) != cudaSuccess)
	{
		fp8_clear_convolutional_relay(producer);
		producer.fp8_relay_source_layer = incoming_source;
		return false;
	}
	producer.fp8_relay_amax_gpu = cuda_make_array(nullptr, 1);
	Darknet::fp8_clear_amax_gpu(producer.fp8_relay_amax_gpu);
	producer.fp8_relay_bytes = relay_bytes;
	producer.fp8_relay_next_layer = consumer_index;
	consumer.fp8_relay_source_layer = producer_index;
	return true;
}


bool fp8_convolutional_direct_dgrad_eligible(const Darknet::Layer & l)
{
	return l.groups == 1 &&
		l.size == 1 &&
		l.stride == 1 &&
		l.stride_x == 1 &&
		l.stride_y == 1 &&
		l.dilation == 1 &&
		l.pad == 0 &&
		l.out_w == l.w &&
		l.out_h == l.h;
}


namespace
{
	bool fp8_convolutional_candidate(const Darknet::Layer & l)
	{
		TAT(TATPARMS);

		return l.type == Darknet::ELayerType::CONVOLUTIONAL &&
			l.share_layer == nullptr &&
			l.binary == 0 &&
			l.xnor == 0 &&
			l.groups == 1 &&
			l.nweights > 0 &&
			l.weights != nullptr &&
			l.weights_gpu != nullptr &&
			l.output_gpu != nullptr;
	}


	float fp8_weight_amax_from_layer(const Darknet::Layer & l)
	{
		TAT(TATPARMS);

		float amax = 0.0f;
		for (int idx = 0; idx < l.nweights; ++idx)
		{
			const float value = std::fabs(l.weights[idx]);
			if (std::isfinite(value))
			{
				amax = std::max(amax, value);
			}
		}
		return amax;
	}

	float fp8_weight_scale_from_layer(const Darknet::Layer & l)
	{
		TAT(TATPARMS);

		return Darknet::fp8_scale_from_amax(fp8_weight_amax_from_layer(l));
	}

	bool fp8_try_setup_convolutional_fprop_plan(Darknet::Layer & l, const Darknet::Fp8ConvOutput output)
	{
		if (!Darknet::fp8_conv_supported() ||
			l.fp8_input_scale_gpu == nullptr ||
			l.fp8_weight_scale_gpu == nullptr ||
			l.batch <= 0 ||
			l.c <= 0 ||
			l.h <= 0 ||
			l.w <= 0 ||
			l.n <= 0 ||
			l.size <= 0)
		{
			return false;
		}

		Darknet::Fp8ConvSpec spec;
		spec.batch = l.batch;
		spec.channels = l.c;
		spec.height = l.h;
		spec.width = l.w;
		spec.filters = l.n;
		spec.kernel_h = l.size;
		spec.kernel_w = l.size;
		spec.pad_h = l.pad * l.dilation;
		spec.pad_w = l.pad * l.dilation;
		spec.stride_h = l.stride_y;
		spec.stride_w = l.stride_x;
		spec.dilation_h = l.dilation;
		spec.dilation_w = l.dilation;
		spec.output = output;
		spec.fuse_bias = !l.batch_normalize && l.biases_gpu != nullptr;
		spec.fuse_relu = !l.batch_normalize && l.activation == RELU;

		auto * plan = Darknet::fp8_conv_plan_create_fprop(spec, l.fp8_input_scale_gpu, l.fp8_weight_scale_gpu);
		if (plan == nullptr)
		{
			return false;
		}

		if (l.weights_fp8_nhwc_gpu)
		{
			CHECK_CUDA(cudaFree(l.weights_fp8_nhwc_gpu));
			l.weights_fp8_nhwc_gpu = nullptr;
		}
		const size_t weight_bytes = static_cast<size_t>(l.n) * l.c * l.size * l.size;
		CHECK_CUDA(cudaMalloc(reinterpret_cast<void **>(&l.weights_fp8_nhwc_gpu), weight_bytes));
		Darknet::fp8_quantize_weights_krsc_gpu(
			l.weights_gpu,
			l.n,
			l.c,
			l.size,
			l.size,
			l.fp8_weight_scale_gpu,
			l.weights_fp8_nhwc_gpu);

		l.fp8_conv_fwd_plan = plan;
		const size_t input_fp8_bytes = static_cast<size_t>(l.batch) * l.c * l.h * l.w;
		const size_t output_element_bytes = output == Darknet::Fp8ConvOutput::Bf16 ? sizeof(unsigned short) : sizeof(float);
		const size_t output_tmp_bytes = static_cast<size_t>(l.batch) * l.n * l.out_h * l.out_w * output_element_bytes;
		const size_t conv_workspace =
			fp8_align_workspace_offset(input_fp8_bytes) +
			fp8_align_workspace_offset(output_tmp_bytes) +
			Darknet::fp8_conv_workspace_bytes(plan);
		l.fp8_workspace_size = std::max(l.fp8_workspace_size, conv_workspace);
		return true;
	}
}


bool fp8_prepare_convolutional_calibration_layer(Darknet::Layer & l)
{
	TAT(TATPARMS);

	fp8_release_convolutional_layer(l);
	if (!fp8_convolutional_candidate(l))
	{
		return false;
	}

	l.fp8_amax_gpu = cuda_make_array(nullptr, 1);
	Darknet::fp8_clear_amax_gpu(l.fp8_amax_gpu);
	return true;
}


bool fp8_setup_convolutional_layer(Darknet::Layer & l)
{
	TAT(TATPARMS);

	fp8_release_convolutional_layer(l);
	if (!Darknet::fp8_gemm_supported() || !fp8_convolutional_candidate(l) || !l.fp8_scales_loaded)
	{
		return false;
	}

	const int m = l.n;
	const int k = l.size * l.size * l.c;
	const int n = l.out_w * l.out_h;
	const int k_pad = Darknet::fp8_round_up_to_16(k);
	const float weight_scale = fp8_weight_scale_from_layer(l);
	const float input_scale = (std::isfinite(l.fp8_input_scale_host) && l.fp8_input_scale_host > 0.0f) ? l.fp8_input_scale_host : 1.0f;

	l.fp8_weight_scale_host = weight_scale;
	l.fp8_input_scale_host = input_scale;
	l.fp8_weight_scale_gpu = cuda_make_array(const_cast<float *>(&l.fp8_weight_scale_host), 1);
	l.fp8_input_scale_gpu = cuda_make_array(const_cast<float *>(&l.fp8_input_scale_host), 1);

	if (fp8_try_setup_convolutional_fprop_plan(l, Darknet::Fp8ConvOutput::Fp32))
	{
		l.fp8_eligible = 1;
		return true;
	}

	const size_t weight_bytes = Darknet::fp8_rowmajor_pad_cols_bytes(m, k_pad);
	CHECK_CUDA(cudaMalloc(reinterpret_cast<void **>(&l.weights_fp8_gpu), weight_bytes));
	Darknet::fp8_quantize_rowmajor_pad_cols_gpu(l.weights_gpu, m, k, k_pad, l.fp8_weight_scale_gpu, l.weights_fp8_gpu);

	const size_t input_fp8_bytes = Darknet::fp8_rowmajor_pad_cols_bytes(n, k_pad);
	const size_t output_tmp_bytes = static_cast<size_t>(m) * n * sizeof(float);

	Darknet::Fp8GemmSpec forward_spec;
	forward_spec.output_rows = m;
	forward_spec.output_cols = n;
	forward_spec.reduction = k;
	forward_spec.reduction_pad = k_pad;
	forward_spec.batch = fp8_pick_gemm_batch(l.batch, input_fp8_bytes + output_tmp_bytes);
	l.fp8_gemm_plan = Darknet::fp8_gemm_plan_create_ex(forward_spec, l.fp8_weight_scale_gpu, l.fp8_input_scale_gpu);
	if (l.fp8_gemm_plan == nullptr && forward_spec.batch > 1)
	{
		forward_spec.batch = 1;
		l.fp8_gemm_plan = Darknet::fp8_gemm_plan_create_ex(forward_spec, l.fp8_weight_scale_gpu, l.fp8_input_scale_gpu);
	}
	if (l.fp8_gemm_plan == nullptr)
	{
		fp8_release_convolutional_layer(l);
		return false;
	}

	l.fp8_forward_batch = forward_spec.batch;
	l.fp8_workspace_size =
		fp8_align_workspace_offset(input_fp8_bytes * forward_spec.batch) +
		fp8_align_workspace_offset(output_tmp_bytes * forward_spec.batch) +
		Darknet::fp8_gemm_workspace_bytes();
	l.fp8_k_pad = k_pad;
	l.fp8_eligible = 1;
	return true;
}


bool fp8_setup_convolutional_training_layer(Darknet::Layer & l)
{
	TAT(TATPARMS);

	fp8_release_convolutional_layer(l);
	if (!Darknet::fp8_gemm_supported() || !fp8_convolutional_candidate(l))
	{
		return false;
	}

	const int filters = l.n;
	const int kernel = l.size * l.size * l.c;
	const int spatial = l.out_w * l.out_h;
	const int kernel_pad = Darknet::fp8_round_up_to_16(kernel);
	const int filters_pad = Darknet::fp8_round_up_to_16(filters);
	const int spatial_pad = Darknet::fp8_round_up_to_16(spatial);
	const float weight_amax = fp8_weight_amax_from_layer(l);
	const float weight_scale = Darknet::fp8_scale_from_amax(weight_amax);
	const float input_scale = 1.0f;
	const float dy_scale = 1.0f;
	const bool use_cudnn_backward = Darknet::fp8_backward_mode_from_env() == Darknet::Fp8BackwardMode::Cudnn;
	const bool sm89_fast_paths = Darknet::fp8_sm89_optimization_supported();

	l.fp8_weight_scale_host = weight_scale;
	l.fp8_input_scale_host = input_scale;
	l.fp8_dy_scale_host = dy_scale;
	Darknet::fp8_delayed_scaling_record_amax(
		l.fp8_weight_amax_history,
		Darknet::kFp8AmaxHistoryLength,
		l.fp8_weight_amax_next,
		l.fp8_weight_amax_count,
		l.fp8_weight_scale_host,
		weight_amax,
		Darknet::Fp8Format::E4M3);
	l.fp8_weight_scale_gpu = cuda_make_array(const_cast<float *>(&l.fp8_weight_scale_host), 1);
	l.fp8_input_scale_gpu = cuda_make_array(const_cast<float *>(&l.fp8_input_scale_host), 1);
	l.fp8_input_amax_gpu = cuda_make_array(nullptr, 1);
	l.fp8_amax_gpu = cuda_make_array(nullptr, 1);
	Darknet::fp8_clear_amax_gpu(l.fp8_input_amax_gpu);
	Darknet::fp8_clear_amax_gpu(l.fp8_amax_gpu);
	if (!use_cudnn_backward)
	{
		l.fp8_dy_scale_gpu = cuda_make_array(const_cast<float *>(&l.fp8_dy_scale_host), 1);
		l.fp8_dy_amax_gpu = cuda_make_array(nullptr, 1);
		Darknet::fp8_clear_amax_gpu(l.fp8_dy_amax_gpu);
	}

	// device-side delayed-scaling state so per-iteration scale updates never sync to the host
	const size_t state_floats = Darknet::fp8_scale_state_floats();
	l.fp8_weight_scale_state_gpu = cuda_make_array(nullptr, state_floats);
	l.fp8_input_scale_state_gpu = cuda_make_array(nullptr, state_floats);
	CHECK_CUDA(cudaMemset(l.fp8_weight_scale_state_gpu, 0, state_floats * sizeof(float)));
	CHECK_CUDA(cudaMemset(l.fp8_input_scale_state_gpu, 0, state_floats * sizeof(float)));
	if (!use_cudnn_backward)
	{
		l.fp8_dy_scale_state_gpu = cuda_make_array(nullptr, state_floats);
		CHECK_CUDA(cudaMemset(l.fp8_dy_scale_state_gpu, 0, state_floats * sizeof(float)));
	}

	const bool conv_fprop_ready = fp8_try_setup_convolutional_fprop_plan(l, Darknet::Fp8ConvOutput::Bf16);
	const bool prune_forward_layout =
		sm89_fast_paths &&
		conv_fprop_ready &&
		!fp8_layer_env_is_set("DARKNET_FP8_DISABLE_SM89_LAYOUT_PRUNING");
	const size_t weight_bytes = Darknet::fp8_rowmajor_pad_cols_bytes(filters, kernel_pad);
	const size_t weight_t_bytes = Darknet::fp8_rowmajor_pad_cols_bytes(kernel, filters_pad);
	if (!prune_forward_layout)
	{
		CHECK_CUDA(cudaMalloc(reinterpret_cast<void **>(&l.weights_fp8_gpu), weight_bytes));
	}
	if (!use_cudnn_backward)
	{
		CHECK_CUDA(cudaMalloc(reinterpret_cast<void **>(&l.weights_fp8_t_gpu), weight_t_bytes));
	}
	if (l.weights_fp8_gpu)
	{
		Darknet::fp8_quantize_rowmajor_pad_cols_gpu(l.weights_gpu, filters, kernel, kernel_pad, l.fp8_weight_scale_gpu, l.weights_fp8_gpu);
	}
	if (l.weights_fp8_t_gpu)
	{
		Darknet::fp8_quantize_transpose_rowmajor_pad_cols_gpu(l.weights_gpu, filters, kernel, filters_pad, l.fp8_weight_scale_gpu, l.weights_fp8_t_gpu);
	}

	Darknet::Fp8GemmSpec forward_spec = Darknet::fp8_gemm_training_spec(Darknet::Fp8TrainingGemm::Forward, filters, kernel, spatial);
	Darknet::Fp8GemmSpec wgrad_spec = Darknet::fp8_gemm_training_spec(Darknet::Fp8TrainingGemm::WeightGradient, filters, kernel, spatial);
	Darknet::Fp8GemmSpec direct_wgrad_spec = Darknet::fp8_gemm_training_spec(Darknet::Fp8TrainingGemm::WeightGradientDirectUpdate, filters, kernel, spatial);
	Darknet::Fp8GemmSpec dgrad_spec = Darknet::fp8_gemm_training_spec(Darknet::Fp8TrainingGemm::DataGradient, filters, kernel, spatial);
	Darknet::Fp8GemmSpec direct_dgrad_spec = Darknet::fp8_gemm_training_spec(Darknet::Fp8TrainingGemm::DataGradientDirectUpdate, filters, kernel, spatial);
	forward_spec.output = Darknet::Fp8GemmOutput::Bf16;
	dgrad_spec.output = Darknet::Fp8GemmOutput::Bf16;

	// strided-batched GEMMs: one call per chunk of images instead of one per image
	const size_t forward_input_bytes = Darknet::fp8_rowmajor_pad_cols_bytes(spatial, kernel_pad);
	const size_t forward_output_bf16_bytes = static_cast<size_t>(filters) * spatial * sizeof(unsigned short);
	const size_t dyt_fp8_bytes = Darknet::fp8_rowmajor_pad_cols_bytes(spatial, filters_pad);
	const size_t dgrad_matrix = static_cast<size_t>(kernel) * spatial;
	forward_spec.batch = fp8_pick_gemm_batch(l.batch, forward_input_bytes + forward_output_bf16_bytes);
	dgrad_spec.batch = fp8_pick_gemm_batch(l.batch, dyt_fp8_bytes + dgrad_matrix * (sizeof(unsigned short) + sizeof(float)));
	direct_dgrad_spec.batch = dgrad_spec.batch;

	// wgrad folds the image batch into the reduction dimension (one wide GEMM per chunk)
	int wgrad_chunk = fp8_pick_gemm_batch(l.batch, static_cast<size_t>(filters + kernel) * spatial_pad);
	const int preferred_wgrad_chunk = wgrad_chunk;
	const auto configure_wgrad_chunk = [spatial, spatial_pad](Darknet::Fp8GemmSpec & spec, const int chunk)
	{
		spec.reduction = spatial * chunk;
		spec.reduction_pad = spatial_pad * chunk;
	};
	configure_wgrad_chunk(wgrad_spec, wgrad_chunk);
	configure_wgrad_chunk(direct_wgrad_spec, wgrad_chunk);

	if (!prune_forward_layout)
	{
		l.fp8_gemm_plan = Darknet::fp8_gemm_plan_create_ex(forward_spec, l.fp8_weight_scale_gpu, l.fp8_input_scale_gpu);
		if (l.fp8_gemm_plan == nullptr && forward_spec.batch > 1)
		{
			forward_spec.batch = 1;
			l.fp8_gemm_plan = Darknet::fp8_gemm_plan_create_ex(forward_spec, l.fp8_weight_scale_gpu, l.fp8_input_scale_gpu);
		}
	}
	if (!use_cudnn_backward)
	{
		const bool enable_direct_wgrad =
			(sm89_fast_paths || fp8_layer_env_is_set("DARKNET_FP8_ENABLE_DIRECT_WGRAD_UPDATE")) &&
			!fp8_layer_env_is_set("DARKNET_FP8_DISABLE_DIRECT_WGRAD_UPDATE") &&
			!fp8_layer_env_is_set("DARKNET_FP8_DISABLE_FUSED_WGRAD_ACCUM");
		if (enable_direct_wgrad)
		{
			int direct_wgrad_chunk = preferred_wgrad_chunk;
			configure_wgrad_chunk(direct_wgrad_spec, direct_wgrad_chunk);
			l.fp8_wgrad_gemm_plan = Darknet::fp8_gemm_plan_create_ex(direct_wgrad_spec, l.fp8_input_scale_gpu, l.fp8_dy_scale_gpu);
			if (l.fp8_wgrad_gemm_plan == nullptr && direct_wgrad_chunk > 1)
			{
				direct_wgrad_chunk = 1;
				configure_wgrad_chunk(direct_wgrad_spec, direct_wgrad_chunk);
				l.fp8_wgrad_gemm_plan = Darknet::fp8_gemm_plan_create_ex(direct_wgrad_spec, l.fp8_input_scale_gpu, l.fp8_dy_scale_gpu);
			}
			if (l.fp8_wgrad_gemm_plan)
			{
				wgrad_chunk = direct_wgrad_chunk;
				l.fp8_wgrad_direct_update = 1;
			}
		}
		if (l.fp8_wgrad_gemm_plan == nullptr)
		{
			l.fp8_wgrad_direct_update = 0;
			wgrad_chunk = preferred_wgrad_chunk;
			configure_wgrad_chunk(wgrad_spec, wgrad_chunk);
			l.fp8_wgrad_gemm_plan = Darknet::fp8_gemm_plan_create_ex(wgrad_spec, l.fp8_dy_scale_gpu, l.fp8_input_scale_gpu);
			if (l.fp8_wgrad_gemm_plan == nullptr && wgrad_chunk > 1)
			{
				wgrad_chunk = 1;
				configure_wgrad_chunk(wgrad_spec, wgrad_chunk);
				l.fp8_wgrad_gemm_plan = Darknet::fp8_gemm_plan_create_ex(wgrad_spec, l.fp8_dy_scale_gpu, l.fp8_input_scale_gpu);
			}
		}
		if (l.fp8_wgrad_gemm_plan == nullptr)
		{
			wgrad_spec.output = Darknet::Fp8GemmOutput::Bf16;
			l.fp8_wgrad_gemm_plan = Darknet::fp8_gemm_plan_create_ex(wgrad_spec, l.fp8_dy_scale_gpu, l.fp8_input_scale_gpu);
		}
		const bool enable_direct_dgrad =
			sm89_fast_paths &&
			fp8_convolutional_direct_dgrad_eligible(l) &&
			!fp8_layer_env_is_set("DARKNET_FP8_DISABLE_DIRECT_DGRAD_UPDATE");
		if (enable_direct_dgrad)
		{
			l.fp8_dgrad_gemm_plan = Darknet::fp8_gemm_plan_create_ex(direct_dgrad_spec, l.fp8_dy_scale_gpu, l.fp8_weight_scale_gpu);
			if (l.fp8_dgrad_gemm_plan == nullptr && direct_dgrad_spec.batch > 1)
			{
				direct_dgrad_spec.batch = 1;
				l.fp8_dgrad_gemm_plan = Darknet::fp8_gemm_plan_create_ex(direct_dgrad_spec, l.fp8_dy_scale_gpu, l.fp8_weight_scale_gpu);
			}
			l.fp8_dgrad_direct_update = l.fp8_dgrad_gemm_plan != nullptr;
		}
		if (l.fp8_dgrad_gemm_plan == nullptr)
		{
			l.fp8_dgrad_direct_update = 0;
			l.fp8_dgrad_gemm_plan = Darknet::fp8_gemm_plan_create_ex(dgrad_spec, l.fp8_weight_scale_gpu, l.fp8_dy_scale_gpu);
			if (l.fp8_dgrad_gemm_plan == nullptr && dgrad_spec.batch > 1)
			{
				dgrad_spec.batch = 1;
				l.fp8_dgrad_gemm_plan = Darknet::fp8_gemm_plan_create_ex(dgrad_spec, l.fp8_weight_scale_gpu, l.fp8_dy_scale_gpu);
			}
		}
	}
	if ((!prune_forward_layout && l.fp8_gemm_plan == nullptr) ||
		(!use_cudnn_backward && (l.fp8_wgrad_gemm_plan == nullptr || l.fp8_dgrad_gemm_plan == nullptr)))
	{
		fp8_release_convolutional_layer(l);
		return false;
	}
	l.fp8_forward_batch = l.fp8_gemm_plan ? forward_spec.batch : 0;
	l.fp8_dgrad_batch = l.fp8_dgrad_direct_update ? direct_dgrad_spec.batch : dgrad_spec.batch;
	l.fp8_wgrad_batch = wgrad_chunk;

	// workspace sizing must stay in sync with the runtime offsets computed in
	// forward/backward_convolutional_layer_gpu_fp8 (convolutional_kernels.cu)
	const bool needs_im2col = !(l.size == 1 && l.stride == 1 && l.stride_x == 1 && l.stride_y == 1 && l.dilation == 1);
	const size_t forward_end = l.fp8_gemm_plan ?
		fp8_align_workspace_offset(forward_input_bytes * forward_spec.batch) +
			forward_output_bf16_bytes * forward_spec.batch : 0;
	// backward layout: dy^T for the WHOLE batch persists at offset 0 (shared between wgrad and
	// dgrad after the single dual-layout dy quantize), then the per-phase scratch regions
	const size_t dyt_all_end = fp8_align_workspace_offset(dyt_fp8_bytes * l.batch);
	const size_t wgrad_k_total = static_cast<size_t>(wgrad_chunk) * spatial_pad;
	const size_t dy_fp8_bytes = static_cast<size_t>(filters) * wgrad_k_total;
	const size_t wgrad_input_bytes = static_cast<size_t>(kernel) * wgrad_k_total;
	const auto * const wgrad_plan = static_cast<Darknet::Fp8GemmPlan *>(l.fp8_wgrad_gemm_plan);
	const auto * const dgrad_plan = static_cast<Darknet::Fp8GemmPlan *>(l.fp8_dgrad_gemm_plan);
	const size_t wgrad_elements = static_cast<size_t>(filters) * kernel;
	const bool direct_wgrad = !use_cudnn_backward && l.fp8_wgrad_direct_update;
	const size_t wgrad_staging_end = fp8_align_workspace_offset(fp8_align_workspace_offset(dyt_all_end + dy_fp8_bytes) + wgrad_input_bytes);
	const size_t wgrad_output_bytes = (use_cudnn_backward || direct_wgrad) ? 0 : wgrad_elements * Darknet::fp8_gemm_output_element_bytes(wgrad_plan);
	const size_t wgrad_tmp_bytes = (use_cudnn_backward || direct_wgrad) ? 0 : wgrad_elements * sizeof(float);
	const size_t wgrad_output_offset = wgrad_staging_end;
	const size_t wgrad_tmp_offset = fp8_align_workspace_offset(wgrad_output_offset + wgrad_output_bytes);
	const size_t wgrad_end = use_cudnn_backward ? 0 : (direct_wgrad ? wgrad_staging_end : wgrad_tmp_offset + wgrad_tmp_bytes);
	const size_t dgrad_output_offset = dyt_all_end;
	const size_t dgrad_output_bytes = (use_cudnn_backward || l.fp8_dgrad_direct_update) ? 0 :
		dgrad_matrix * dgrad_spec.batch * Darknet::fp8_gemm_output_element_bytes(dgrad_plan);
	const size_t dgrad_col_offset = fp8_align_workspace_offset(dgrad_output_offset + dgrad_output_bytes);
	const size_t dgrad_input_bytes = (!l.fp8_dgrad_direct_update && needs_im2col) ? static_cast<size_t>(l.c) * l.h * l.w * sizeof(float) : 0;
	const size_t dgrad_end = use_cudnn_backward ? 0 : (l.fp8_dgrad_direct_update ? dyt_all_end :
		dgrad_col_offset + dgrad_matrix * sizeof(float) * dgrad_spec.batch + dgrad_input_bytes);
	const size_t matrix_workspace = std::max(std::max(forward_end, wgrad_end), dgrad_end);
	const size_t gemm_workspace = fp8_align_workspace_offset(matrix_workspace) + Darknet::fp8_gemm_workspace_bytes();
	l.fp8_workspace_size = conv_fprop_ready ? std::max(l.fp8_workspace_size, gemm_workspace) : gemm_workspace;
	l.fp8_k_pad = kernel_pad;
	l.fp8_train_eligible = 1;
	if (fp8_layer_env_is_set("DARKNET_FP8_DEBUG"))
	{
		std::fprintf(stderr,
			"fp8 layer=%d sm89=%d fprop=%s rowmajor_weights=%s wgrad=%s dgrad=%s workspace=%zu\n",
			l.index,
			sm89_fast_paths ? 1 : 0,
			conv_fprop_ready ? "cudnn" : "cublaslt",
			l.weights_fp8_gpu ? "yes" : "no",
			l.fp8_wgrad_direct_update ? "direct-fp32" : "staged",
			l.fp8_dgrad_direct_update ? "direct-fp32" : "staged",
			l.fp8_workspace_size);
	}
	return true;
}
#endif

#endif


void free_convolutional_batchnorm(Darknet::Layer *l)
{
	TAT(TATPARMS);

	if (!l->share_layer)
	{
		if (l->scales)					{free(l->scales);						l->scales = nullptr;				}
		if (l->scale_updates)			{free(l->scale_updates);				l->scale_updates = nullptr;			}
		if (l->mean)					{free(l->mean);							l->mean = nullptr;					}
		if (l->variance)				{free(l->variance);						l->variance = nullptr;				}
		if (l->mean_delta)				{free(l->mean_delta);					l->mean_delta = nullptr;			}
		if (l->variance_delta)			{free(l->variance_delta);				l->variance_delta = nullptr;		}
		if (l->rolling_mean)			{free(l->rolling_mean);					l->rolling_mean = nullptr;			}
		if (l->rolling_variance)		{free(l->rolling_variance);				l->rolling_variance = nullptr;		}
		if (l->x)						{free(l->x);							l->x = nullptr;						}
		if (l->x_norm)					{free(l->x_norm);						l->x_norm = nullptr;				}

#ifdef DARKNET_GPU
		if (l->scales_gpu)				{cuda_free(l->scales_gpu);				l->scales_gpu = nullptr;			}
		if (l->scale_updates_gpu)		{cuda_free(l->scale_updates_gpu);		l->scale_updates_gpu = nullptr;		}
		if (l->mean_gpu)				{cuda_free(l->mean_gpu);				l->mean_gpu = nullptr;				}
		if (l->variance_gpu)			{cuda_free(l->variance_gpu);			l->variance_gpu = nullptr;			}
		if (l->mean_delta_gpu)			{cuda_free(l->mean_delta_gpu);			l->mean_delta_gpu = nullptr;		}
		if (l->variance_delta_gpu)		{cuda_free(l->variance_delta_gpu);		l->variance_delta_gpu = nullptr;	}
		if (l->rolling_mean_gpu)		{cuda_free(l->rolling_mean_gpu);		l->rolling_mean_gpu = nullptr;		}
		if (l->rolling_variance_gpu)	{cuda_free(l->rolling_variance_gpu);	l->rolling_variance_gpu = nullptr;	}
		if (l->x_gpu)					{cuda_free(l->x_gpu);					l->x_gpu = nullptr;					}
		if (l->x_norm_gpu)				{cuda_free(l->x_norm_gpu);				l->x_norm_gpu = nullptr;			}
#endif
	}
}


Darknet::Layer make_convolutional_layer(int batch, int steps, int h, int w, int c, int n, int groups, int size, int stride_x, int stride_y, int dilation, int padding, ACTIVATION activation, int batch_normalize, int binary, int xnor, int adam, int use_bin_output, int index, int antialiasing, Darknet::Layer *share_layer, int assisted_excitation, int deform, int train)
{
	TAT(TATPARMS);

	int total_batch = batch * steps;
	Darknet::Layer l = { (Darknet::ELayerType)0 };
	l.type = Darknet::ELayerType::CONVOLUTIONAL;
	l.train = train;
	// Zero-initialization is convenient for most layer metadata, but a missing
	// precision-chain link must be distinguishable from layer zero.
	l.fp4_relay_next_layer = -1;
	l.fp4_relay_source_layer = -1;
	l.fp8_relay_next_layer = -1;
	l.fp8_relay_source_layer = -1;

	if (xnor)
	{
		groups = 1;   // disable groups for XNOR-net
	}
	if (groups < 1)
	{
		groups = 1;
	}

	const int blur_stride_x = stride_x;
	const int blur_stride_y = stride_y;
	l.antialiasing = antialiasing;
	if (antialiasing)
	{
		stride_x = stride_y = l.stride = l.stride_x = l.stride_y = 1; // use stride=1 in host-layer
	}

	l.wait_stream_id = -1;
	l.deform = deform;
	l.assisted_excitation = assisted_excitation;
	l.share_layer = share_layer;
	l.index = index;
	l.h = h;
	l.w = w;
	l.c = c;
	l.groups = groups;
	l.n = n;
	l.binary = binary;
	l.xnor = xnor;
	l.use_bin_output = use_bin_output;
	l.batch = batch;
	l.steps = steps;
	l.stride = stride_x;
	l.stride_x = stride_x;
	l.stride_y = stride_y;
	l.dilation = dilation;
	l.size = size;
	l.pad = padding;
	l.batch_normalize = batch_normalize;
	l.learning_rate_scale = 1;
	l.nweights = (c / groups) * n * size * size;

	if (l.share_layer)
	{
		if (l.size != l.share_layer->size || l.nweights != l.share_layer->nweights || l.c != l.share_layer->c || l.n != l.share_layer->n)
		{
			darknet_fatal_error(DARKNET_LOC, "Layer size, nweights, channels or filters don't match for the share_layer");
		}

		l.weights = l.share_layer->weights;
		l.weight_updates = l.share_layer->weight_updates;

		l.biases = l.share_layer->biases;
		l.bias_updates = l.share_layer->bias_updates;
	}
	else
	{
		l.weights = (float*)xcalloc(l.nweights, sizeof(float));
		l.biases = (float*)xcalloc(n, sizeof(float));

		if (train)
		{
			l.weight_updates = (float*)xcalloc(l.nweights, sizeof(float));
			l.bias_updates = (float*)xcalloc(n, sizeof(float));

			l.weights_ema = (float*)xcalloc(l.nweights, sizeof(float));
			l.biases_ema = (float*)xcalloc(n, sizeof(float));
		}
	}

	float scale = sqrt(2./(size*size*c/groups));
	if (l.activation == NORM_CHAN || l.activation == NORM_CHAN_SOFTMAX || l.activation == NORM_CHAN_SOFTMAX_MAXVAL)
	{
		for (int i = 0; i < l.nweights; ++i)
		{
			l.weights[i] = 1.0f;
		}
	}
	else
	{
		rand_uniform_many_weight_init(l.weights, l.nweights, -1.0f, 1.0f, scale);
	}
	int out_h = convolutional_out_height(l);
	int out_w = convolutional_out_width(l);
	l.out_h = out_h;
	l.out_w = out_w;
	l.out_c = n;
	l.outputs = l.out_h * l.out_w * l.out_c;
	l.inputs = l.w * l.h * l.c;
	l.activation = activation;

	l.output = (float*)xcalloc(total_batch*l.outputs, sizeof(float));
#ifndef DARKNET_GPU
	if (train)
	{
		l.delta = (float*)xcalloc(total_batch*l.outputs, sizeof(float));
	}
#endif  // not DARKNET_GPU

	l.forward = forward_convolutional_layer;
	l.backward = backward_convolutional_layer;
	l.update = update_convolutional_layer;

	if (binary)
	{
		l.binary_weights = (float*)xcalloc(l.nweights, sizeof(float));
		l.cweights = (char*)xcalloc(l.nweights, sizeof(char));
		l.scales = (float*)xcalloc(n, sizeof(float));
	}
	if (xnor)
	{
		l.binary_weights = (float*)xcalloc(l.nweights, sizeof(float));
		l.binary_input = (float*)xcalloc(l.inputs * l.batch, sizeof(float));

		int align = 32;// 8;
		int src_align = l.out_h*l.out_w;
		l.bit_align = src_align + (align - src_align % align);

		l.mean_arr = (float*)xcalloc(l.n, sizeof(float));

		const size_t new_c = l.c / 32;
		size_t in_re_packed_input_size = new_c * l.w * l.h + 1;
		l.bin_re_packed_input = (uint32_t*)xcalloc(in_re_packed_input_size, sizeof(uint32_t));

		l.lda_align = 256;  // AVX2
		int k = l.size*l.size*l.c;
		size_t k_aligned = k + (l.lda_align - k%l.lda_align);
		size_t t_bit_input_size = k_aligned * l.bit_align / 8;
		l.t_bit_input = (char*)xcalloc(t_bit_input_size, sizeof(char));
	}

	if (batch_normalize)
	{
		if (l.share_layer)
		{
			l.scales = l.share_layer->scales;
			l.scale_updates = l.share_layer->scale_updates;
			l.mean = l.share_layer->mean;
			l.variance = l.share_layer->variance;
			l.mean_delta = l.share_layer->mean_delta;
			l.variance_delta = l.share_layer->variance_delta;
			l.rolling_mean = l.share_layer->rolling_mean;
			l.rolling_variance = l.share_layer->rolling_variance;
		}
		else
		{
			l.scales = (float*)xcalloc(n, sizeof(float));
			for (int i = 0; i < n; ++i)
			{
				l.scales[i] = 1.0f;
			}
			if (train)
			{
				l.scales_ema = (float*)xcalloc(n, sizeof(float));
				l.scale_updates = (float*)xcalloc(n, sizeof(float));

				l.mean = (float*)xcalloc(n, sizeof(float));
				l.variance = (float*)xcalloc(n, sizeof(float));

				l.mean_delta = (float*)xcalloc(n, sizeof(float));
				l.variance_delta = (float*)xcalloc(n, sizeof(float));
			}
			l.rolling_mean = (float*)xcalloc(n, sizeof(float));
			l.rolling_variance = (float*)xcalloc(n, sizeof(float));
		}

#ifndef DARKNET_GPU
		if (train)
		{
			l.x = (float*)xcalloc(total_batch * l.outputs, sizeof(float));
			l.x_norm = (float*)xcalloc(total_batch * l.outputs, sizeof(float));
		}
#endif  // not DARKNET_GPU
	}

#ifndef DARKNET_GPU
	if (l.activation == SWISH || l.activation == MISH || l.activation == HARD_MISH || l.activation == EML) l.activation_input = (float*)calloc(total_batch*l.outputs, sizeof(float));
#endif  // not DARKNET_GPU

	if (adam)
	{
		l.adam = 1;
		l.m = (float*)xcalloc(l.nweights, sizeof(float));
		l.v = (float*)xcalloc(l.nweights, sizeof(float));
		l.bias_m = (float*)xcalloc(n, sizeof(float));
		l.scale_m = (float*)xcalloc(n, sizeof(float));
		l.bias_v = (float*)xcalloc(n, sizeof(float));
		l.scale_v = (float*)xcalloc(n, sizeof(float));
	}

#ifdef DARKNET_GPU

	l.forward_gpu = forward_convolutional_layer_gpu;
	l.backward_gpu = backward_convolutional_layer_gpu;
	l.update_gpu = update_convolutional_layer_gpu;

	if (cfg_and_state.gpu_index >= 0)
	{
		if (train && (l.activation == SWISH || l.activation == MISH || l.activation == HARD_MISH || l.activation == EML))
		{
			l.activation_input_gpu = cuda_make_array(l.activation_input, total_batch*l.outputs);
		}

		if (l.deform) l.weight_deform_gpu = cuda_make_array(NULL, l.nweights);

		if (adam)
		{
			l.m_gpu = cuda_make_array(l.m, l.nweights);
			l.v_gpu = cuda_make_array(l.v, l.nweights);
			l.bias_m_gpu = cuda_make_array(l.bias_m, n);
			l.bias_v_gpu = cuda_make_array(l.bias_v, n);
			l.scale_m_gpu = cuda_make_array(l.scale_m, n);
			l.scale_v_gpu = cuda_make_array(l.scale_v, n);
		}
		if (l.share_layer)
		{
			l.weights_gpu = l.share_layer->weights_gpu;
			l.weight_updates_gpu = l.share_layer->weight_updates_gpu;
			l.weights_gpu16 = l.share_layer->weights_gpu16;
			l.weight_updates_gpu16 = l.share_layer->weight_updates_gpu16;
			l.biases_gpu = l.share_layer->biases_gpu;
			l.bias_updates_gpu = l.share_layer->bias_updates_gpu;
		}
		else
		{
			l.weights_gpu = cuda_make_array(l.weights, l.nweights);
			if (train)
			{
				l.weight_updates_gpu = cuda_make_array(l.weight_updates, l.nweights);
			}
#ifdef CUDNN_HALF
			l.weights_gpu16 = cuda_make_array(NULL, l.nweights / 2 + 1);
			if (train)
			{
				l.weight_updates_gpu16 = cuda_make_array(NULL, l.nweights / 2 + 1);
			}
#endif  // CUDNN_HALF
			l.biases_gpu = cuda_make_array(l.biases, n);
			if (train)
			{
				l.bias_updates_gpu = cuda_make_array(l.bias_updates, n);
			}
		}

		l.output_gpu = cuda_make_array(l.output, total_batch*out_h*out_w*n);
		if (train)
		{
			l.delta_gpu = cuda_make_array(l.delta, total_batch*out_h*out_w*n);
		}

		if (binary)
		{
			l.binary_weights_gpu = cuda_make_array(l.weights, l.nweights);
		}
		if (xnor)
		{
			l.binary_weights_gpu = cuda_make_array(l.weights, l.nweights);
			l.mean_arr_gpu = cuda_make_array(0, l.n);
			l.binary_input_gpu = cuda_make_array(0, l.inputs*l.batch);
		}

		if (batch_normalize)
		{
			if (l.share_layer)
			{
				l.scales_gpu = l.share_layer->scales_gpu;
				l.scale_updates_gpu = l.share_layer->scale_updates_gpu;
				l.mean_gpu = l.share_layer->mean_gpu;
				l.variance_gpu = l.share_layer->variance_gpu;
				l.rolling_mean_gpu = l.share_layer->rolling_mean_gpu;
				l.rolling_variance_gpu = l.share_layer->rolling_variance_gpu;
				l.mean_delta_gpu = l.share_layer->mean_delta_gpu;
				l.variance_delta_gpu = l.share_layer->variance_delta_gpu;
			}
			else
			{
				l.scales_gpu = cuda_make_array(l.scales, n);

				if (train)
				{
					l.scale_updates_gpu = cuda_make_array(l.scale_updates, n);

					l.mean_gpu = cuda_make_array(l.mean, n);
					l.variance_gpu = cuda_make_array(l.variance, n);
					l.m_cbn_avg_gpu = cuda_make_array(l.mean, n);
					l.v_cbn_avg_gpu = cuda_make_array(l.variance, n);
#ifndef CUDNN
					l.mean_delta_gpu = cuda_make_array(l.mean, n);
					l.variance_delta_gpu = cuda_make_array(l.variance, n);
#endif  // CUDNN
				}

				l.rolling_mean_gpu = cuda_make_array(l.mean, n);
				l.rolling_variance_gpu = cuda_make_array(l.variance, n);
			}

			if (train)
			{
				l.x_gpu = cuda_make_array(l.output, total_batch*out_h*out_w*n);
#ifndef CUDNN
				l.x_norm_gpu = cuda_make_array(l.output, total_batch*out_h*out_w*n);
#endif  // CUDNN
			}
		}

		if (l.assisted_excitation)
		{
			const int size2 = l.out_w * l.out_h * l.batch;
			l.gt_gpu = cuda_make_array(NULL, size2);
			l.a_avg_gpu = cuda_make_array(NULL, size2);
		}
#ifdef CUDNN
		create_convolutional_cudnn_tensors(&l);
		cudnn_convolutional_setup(&l, cudnn_fastest, 0);
#endif  // CUDNN
	}
#endif  // DARKNET_GPU
	l.workspace_size = get_convolutional_workspace_size(l);

	l.bflops = (2.0 * l.nweights * l.out_h*l.out_w) / 1000000000.;
	if (l.xnor)
	{
		l.bflops = l.bflops / 32;
	}

	if (l.antialiasing)
	{
		l.input_layer = (Darknet::Layer*)xcalloc(1, sizeof(Darknet::Layer));
		int blur_size = 3;
		int blur_pad = blur_size / 2;
		if (l.antialiasing == 2)
		{
			blur_size = 2;
			blur_pad = 0;
		}
		*(l.input_layer) = make_convolutional_layer(batch, steps, out_h, out_w, n, n, n, blur_size, blur_stride_x, blur_stride_y, 1, blur_pad, LINEAR, 0, 0, 0, 0, 0, index, 0, NULL, 0, 0, train);
		const int blur_nweights = n * blur_size * blur_size;  // (n / n) * n * blur_size * blur_size;
		if (blur_size == 2)
		{
			for (int i = 0; i < blur_nweights; i += (blur_size*blur_size))
			{
				l.input_layer->weights[i + 0] = 1 / 4.0f;
				l.input_layer->weights[i + 1] = 1 / 4.0f;
				l.input_layer->weights[i + 2] = 1 / 4.0f;
				l.input_layer->weights[i + 3] = 1 / 4.0f;
			}
		}
		else
		{
			for (int i = 0; i < blur_nweights; i += (blur_size*blur_size))
			{
				l.input_layer->weights[i + 0] = 1 / 16.0f;
				l.input_layer->weights[i + 1] = 2 / 16.0f;
				l.input_layer->weights[i + 2] = 1 / 16.0f;

				l.input_layer->weights[i + 3] = 2 / 16.0f;
				l.input_layer->weights[i + 4] = 4 / 16.0f;
				l.input_layer->weights[i + 5] = 2 / 16.0f;

				l.input_layer->weights[i + 6] = 1 / 16.0f;
				l.input_layer->weights[i + 7] = 2 / 16.0f;
				l.input_layer->weights[i + 8] = 1 / 16.0f;
			}
		}
		for (int i = 0; i < n; ++i)
		{
			l.input_layer->biases[i] = 0.0f;
		}
#ifdef DARKNET_GPU
		if (cfg_and_state.gpu_index >= 0)
		{
			l.input_antialiasing_gpu = cuda_make_array(NULL, l.batch*l.outputs);
			push_convolutional_layer(*(l.input_layer));
		}
#endif  // DARKNET_GPU
	}

	return l;
}

Darknet::Layer make_eml_convolutional_layer(int batch, int h, int w, int c, int n, int groups, int size, int stride_x, int stride_y, int dilation, int padding, int batch_normalize, int adam, int index, int train, float eml_clamp, float eml_eps, float eml_scale, int residual)
{
	TAT(TATPARMS);

	Darknet::Layer l = { (Darknet::ELayerType)0 };
	l.type = Darknet::ELayerType::EML_CONV;
	l.train = train;
	l.index = index;
	l.h = h;
	l.w = w;
	l.c = c;
	l.n = n;
	l.groups = groups < 1 ? 1 : groups;
	l.size = size;
	l.stride = stride_x;
	l.stride_x = stride_x;
	l.stride_y = stride_y;
	l.dilation = dilation;
	l.pad = padding;
	l.batch = batch;
	l.batch_normalize = batch_normalize;
	l.learning_rate_scale = 1;
	l.alpha = eml_clamp;
	l.beta = eml_eps;
	l.scale = eml_scale;
	l.shortcut = residual;

	l.input_layer = (Darknet::Layer*)xcalloc(1, sizeof(Darknet::Layer));
	l.self_layer = (Darknet::Layer*)xcalloc(1, sizeof(Darknet::Layer));

	*(l.input_layer) = make_convolutional_layer(batch, 1, h, w, c, n, l.groups, size, stride_x, stride_y, dilation, padding, LINEAR, batch_normalize, 0, 0, adam, 0, index, 0, NULL, 0, 0, train);
	*(l.self_layer) = make_convolutional_layer(batch, 1, h, w, c, n, l.groups, size, stride_x, stride_y, dilation, padding, LINEAR, batch_normalize, 0, 0, adam, 0, index, 0, NULL, 0, 0, train);

	l.out_h = l.input_layer->out_h;
	l.out_w = l.input_layer->out_w;
	l.out_c = n;
	l.outputs = l.out_h * l.out_w * l.out_c;
	l.inputs = h * w * c;
	l.nweights = l.input_layer->nweights + l.self_layer->nweights;
	l.workspace_size = std::max(l.input_layer->workspace_size, l.self_layer->workspace_size);
	l.bflops = l.input_layer->bflops + l.self_layer->bflops;

	l.output = (float*)xcalloc(batch * l.outputs, sizeof(float));
#ifndef DARKNET_GPU
	if (train)
	{
		l.delta = (float*)xcalloc(batch * l.outputs, sizeof(float));
	}
#endif

	l.forward = forward_eml_convolutional_layer;
	l.backward = backward_eml_convolutional_layer;
	l.update = update_eml_convolutional_layer;

#ifdef DARKNET_GPU
	l.forward_gpu = forward_eml_convolutional_layer_gpu;
	l.backward_gpu = backward_eml_convolutional_layer_gpu;
	l.update_gpu = update_eml_convolutional_layer_gpu;

	if (cfg_and_state.gpu_index >= 0)
	{
		l.output_gpu = cuda_make_array(l.output, batch * l.outputs);
		if (train)
		{
			l.delta_gpu = cuda_make_array(l.delta, batch * l.outputs);
		}
	}
#endif

	return l;
}

void denormalize_convolutional_layer(Darknet::Layer & l)
{
	TAT(TATPARMS);

	for(int i = 0; i < l.n; ++i)
	{
		const float scale = l.scales[i] / sqrt(l.rolling_variance[i] + 0.00001f);
		for(int j = 0; j < l.nweights; ++j)
		{
			l.weights[i*l.nweights + j] *= scale;
		}
		l.biases[i] -= l.rolling_mean[i] * scale;
		l.scales[i] = 1;
		l.rolling_mean[i] = 0;
		l.rolling_variance[i] = 1;
	}
}

void test_convolutional_layer()
{
	TAT(TATPARMS);

	Darknet::Layer l = make_convolutional_layer(1, 1, 5, 5, 3, 2, 1, 5, 2, 2, 1, 1, LEAKY, 1, 0, 0, 0, 0, 0, 0, NULL, 0, 0, 0);
	l.batch_normalize = 1;
	float data[] = {1,1,1,1,1,
		1,1,1,1,1,
		1,1,1,1,1,
		1,1,1,1,1,
		1,1,1,1,1,
		2,2,2,2,2,
		2,2,2,2,2,
		2,2,2,2,2,
		2,2,2,2,2,
		2,2,2,2,2,
		3,3,3,3,3,
		3,3,3,3,3,
		3,3,3,3,3,
		3,3,3,3,3,
		3,3,3,3,3};
	Darknet::NetworkState state = {0};
	state.input = data;
	forward_convolutional_layer(l, state);
}

void resize_convolutional_layer(Darknet::Layer *l, int w, int h)
{
	TAT(TATPARMS);

	int total_batch = l->batch*l->steps;

#ifdef DARKNET_GPU
	int old_w = l->w;
	int old_h = l->h;
#endif

	l->w = w;
	l->h = h;
	int out_w = convolutional_out_width(*l);
	int out_h = convolutional_out_height(*l);

	l->out_w = out_w;
	l->out_h = out_h;

	l->outputs = l->out_h * l->out_w * l->out_c;
	l->inputs = l->w * l->h * l->c;


	l->output = (float*)xrealloc(l->output, total_batch * l->outputs * sizeof(float));
	if (l->train)
	{
		l->delta = (float*)xrealloc(l->delta, total_batch * l->outputs * sizeof(float));

		if (l->batch_normalize)
		{
			l->x = (float*)xrealloc(l->x, total_batch * l->outputs * sizeof(float));
			l->x_norm = (float*)xrealloc(l->x_norm, total_batch * l->outputs * sizeof(float));
		}
	}

#if 0
	if (l->xnor)
	{
		//l->binary_input = realloc(l->inputs*l->batch, sizeof(float));
	}
#endif

	if (l->activation == SWISH || l->activation == MISH || l->activation == HARD_MISH || l->activation == EML) l->activation_input = (float*)realloc(l->activation_input, total_batch*l->outputs * sizeof(float));
#ifdef DARKNET_GPU
	if (old_w < w || old_h < h || l->dynamic_minibatch)
	{
		if (l->train)
		{
			cuda_free(l->delta_gpu);
			l->delta_gpu = cuda_make_array(l->delta, total_batch*l->outputs);
		}

		cuda_free(l->output_gpu);
		l->output_gpu = cuda_make_array(l->output, total_batch*l->outputs);

		if (l->batch_normalize)
		{
			cuda_free(l->x_gpu);
			l->x_gpu = cuda_make_array(l->output, total_batch*l->outputs);

#ifndef CUDNN
			cuda_free(l->x_norm_gpu);
			l->x_norm_gpu = cuda_make_array(l->output, total_batch*l->outputs);
#endif  // CUDNN
		}

		if (l->xnor)
		{
			cuda_free(l->binary_input_gpu);
			l->binary_input_gpu = cuda_make_array(0, l->inputs*l->batch);
		}

		if (l->activation == SWISH || l->activation == MISH || l->activation == HARD_MISH || l->activation == EML)
		{
			cuda_free(l->activation_input_gpu);
			l->activation_input_gpu = cuda_make_array(l->activation_input, total_batch*l->outputs);
		}

		if (l->assisted_excitation)
		{
			cuda_free(l->gt_gpu);
			cuda_free(l->a_avg_gpu);

			const int size = l->out_w * l->out_h * l->batch;
			l->gt_gpu = cuda_make_array(NULL, size);
			l->a_avg_gpu = cuda_make_array(NULL, size);
		}
	}
#ifdef DARKNET_HAS_FP4
	fp4_release_convolutional_layer(*l);
#endif
#ifdef DARKNET_HAS_FP8
	fp8_release_convolutional_layer(*l);
#endif
#ifdef CUDNN
	cudnn_convolutional_setup(l, cudnn_fastest, 0);
#endif
#endif
	l->workspace_size = get_convolutional_workspace_size(*l);

#ifdef CUDNN
	// check for excessive memory consumption
	size_t free_byte;
	size_t total_byte;
	CHECK_CUDA(cudaMemGetInfo(&free_byte, &total_byte));
	if (l->workspace_size > free_byte || l->workspace_size >= total_byte / 2)
	{
		*cfg_and_state.output << " used slow CUDNN algo without Workspace! Need memory: " << l->workspace_size << ", available: " << ((free_byte < total_byte/2) ? free_byte : total_byte/2) << std::endl;
		cudnn_convolutional_setup(l, cudnn_smallest, 0);
		l->workspace_size = get_convolutional_workspace_size(*l);
	}
#endif
}

void set_specified_workspace_limit(Darknet::Layer * l, size_t workspace_size_limit)
{
	TAT(TATPARMS);

#ifdef CUDNN
	size_t free_byte;
	size_t total_byte;
	CHECK_CUDA(cudaMemGetInfo(&free_byte, &total_byte));
	cudnn_convolutional_setup(l, cudnn_specify, workspace_size_limit);
	l->workspace_size = get_convolutional_workspace_size(*l);
#endif  // CUDNN
}

void add_bias(float *output, float *biases, int batch, int n, int size)
{
	TAT(TATPARMS);

	#pragma omp parallel for collapse(2) schedule(static)
	for (int b = 0; b < batch; ++b)
	{
		for (int i = 0; i < n; ++i)
		{
			const float & bias = biases[i];
			float * __restrict out_ptr = output + (b * n + i) * size;

			#pragma omp simd
			for (int j = 0; j < size; ++j)
			{
				out_ptr[j] += bias;
//				output[(b * n + i) * size + j] += biases[i];
			}
		}
	}
}

void scale_bias(float *output, float *scales, int batch, int n, int size)
{
	TAT(TATPARMS);

	for (int b = 0; b < batch; ++b)
	{
		for (int i = 0; i < n; ++i)
		{
			for (int j = 0; j < size; ++j)
			{
				output[(b * n + i) * size + j] *= scales[i];
			}
		}
	}
}

void backward_bias(float *bias_updates, float *delta, int batch, int n, int size)
{
	TAT(TATPARMS);

	for (int b = 0; b < batch; ++b)
	{
		for (int i = 0; i < n; ++i)
		{
			bias_updates[i] += sum_array(delta + size * (i + b * n), size);
		}
	}
}

void gemm_nn_custom(int M, int N, int K, float ALPHA, float *A, int lda, float *B, int ldb, float *C, int ldc)
{
	TAT(TATPARMS);

	for (int i = 0; i < M; ++i)
	{
		for (int k = 0; k < K; ++k)
		{
			const float A_PART = ALPHA * A[i * lda + k];

			for (int j = 0; j < N; ++j)
			{
				C[i*ldc + j] += A_PART*B[k*ldb + j];
			}
		}
	}
}


void binary_align_weights(Darknet::Layer *l)
{
	TAT(TATPARMS);

	int m = l->n;   // (l->n / l->groups)
	int k = l->size*l->size*l->c;   // ->size*l->size*(l->c / l->groups)
	size_t new_lda = k + (l->lda_align - k % l->lda_align); // (k / 8 + 1) * 8;
	l->new_lda = new_lda;

	binarize_weights(l->weights, m, k, l->binary_weights);

	size_t align_weights_size = new_lda * m;
	l->align_bit_weights_size = align_weights_size / 8 + 1;
	float* align_weights = (float*)xcalloc(align_weights_size, sizeof(float));
	l->align_bit_weights = (char*)xcalloc(l->align_bit_weights_size, sizeof(char));

	// align A without transpose
	for (size_t i = 0; i < m; ++i)
	{
		for (size_t j = 0; j < k; ++j)
		{
			align_weights[i*new_lda + j] = l->binary_weights[i*k + j];
		}
	}

	if (l->c % 32 == 0)
	//if (gpu_old_index < 0 && l->stride == 1 && l->pad == 1 && l->c % 32 == 0)
	//if (l->stride == 1 && l->pad == 1 && l->c % 32 == 0)
	{
		int fil, chan;
		const int items_per_filter = l->c * l->size * l->size;
		//const int dst_items_per_filter = new_lda;
		for (fil = 0; fil < l->n; ++fil)
		{
			for (chan = 0; chan < l->c; chan += 32)
			{
				const int items_per_channel = l->size*l->size;
				for (size_t i = 0; i < items_per_channel; ++i)
				{
					//uint32_t val = 0;
					int c_pack;
					for (c_pack = 0; c_pack < 32; ++c_pack) {
						float src = l->binary_weights[fil*items_per_filter + (chan + c_pack)*items_per_channel + i];

						//align_weights[fil*items_per_filter + chan*items_per_channel + i * 32 + c_pack] = src;

						align_weights[fil*new_lda + chan*items_per_channel + i*32 + c_pack] = src;
						//val |= (src << c);
					}

				}
			}
		}

		float_to_bit(align_weights, (unsigned char*)l->align_bit_weights, align_weights_size);

		if (cfg_and_state.gpu_index >= 0)
		{
			for (size_t i = 0; i < align_weights_size / 8; ++i)
			{
				l->align_bit_weights[i] = ~(l->align_bit_weights[i]);
			}
		}

		get_mean_array(l->binary_weights, m*k, l->n, l->mean_arr);
		//get_mean_array(l->binary_weights, m*new_lda, l->n, l->mean_arr);
	}
	else
	{
		float_to_bit(align_weights, (unsigned char*)l->align_bit_weights, align_weights_size);

		get_mean_array(l->binary_weights, m*k, l->n, l->mean_arr);
	}

#ifdef DARKNET_GPU
	cudaError_t status;
	l->align_workspace_size = l->bit_align * l->size * l->size * l->c;
	status = cudaMalloc((void **)&l->align_workspace_gpu, l->align_workspace_size * sizeof(float));
	status = cudaMalloc((void **)&l->transposed_align_workspace_gpu, l->align_workspace_size * sizeof(float));
	CHECK_CUDA(status);

	//l->align_bit_weights_gpu = cuda_make_array(l->align_bit_weights, l->align_bit_weights_size * sizeof(char)/sizeof(float));
	status = cudaMalloc((void **)&l->align_bit_weights_gpu, l->align_bit_weights_size);
	CHECK_CUDA(status);
	status = cudaMemcpy(l->align_bit_weights_gpu, l->align_bit_weights, l->align_bit_weights_size, cudaMemcpyHostToDevice);
	CHECK_CUDA(status);
	status = cudaMemcpy(l->binary_weights_gpu, l->binary_weights, m*k * sizeof(float), cudaMemcpyHostToDevice);
	CHECK_CUDA(status);

	//l->mean_arr_gpu = cuda_make_array(l->mean_arr, l->n);
	cuda_push_array(l->mean_arr_gpu, l->mean_arr, l->n);
	CHECK_CUDA(cudaDeviceSynchronize());
#endif // DARKNET_GPU

	free(align_weights);
}


void forward_convolutional_layer(Darknet::Layer & l, Darknet::NetworkState state)
{
	TAT(TATPARMS);

	int out_h = convolutional_out_height(l);
	int out_w = convolutional_out_width(l);

#ifdef DARKNET_USE_MPS
	if (not state.train and not l.binary and not l.xnor)
	{
		const Darknet::Layer *prev = mps_prev_layer(state);
		bool defer_readback = mps_should_defer_readback(state);
		bool activation_applied = false;
		if (mps_convolution_forward(l, prev, state.input, l.output, defer_readback, &activation_applied, nullptr))
		{
			if (not activation_applied)
			{
				if (l.activation == SWISH) activate_array_swish(l.output, l.outputs*l.batch, l.activation_input, l.output);
				else if (l.activation == MISH) activate_array_mish(l.output, l.outputs*l.batch, l.activation_input, l.output);
				else if (l.activation == HARD_MISH) activate_array_hard_mish(l.output, l.outputs*l.batch, l.activation_input, l.output);
				else if (l.activation == EML) activate_array_eml(l.output, l.outputs*l.batch, l.activation_input, l.output);
				else if (l.activation == NORM_CHAN) activate_array_normalize_channels(l.output, l.outputs*l.batch, l.batch, l.out_c, l.out_w*l.out_h, l.output);
				else if (l.activation == NORM_CHAN_SOFTMAX) activate_array_normalize_channels_softmax(l.output, l.outputs*l.batch, l.batch, l.out_c, l.out_w*l.out_h, l.output, 0);
				else if (l.activation == NORM_CHAN_SOFTMAX_MAXVAL) activate_array_normalize_channels_softmax(l.output, l.outputs*l.batch, l.batch, l.out_c, l.out_w*l.out_h, l.output, 1);
				else activate_array_cpu_custom(l.output, l.outputs*l.batch, l.activation);
			}

			if (l.assisted_excitation && state.train)
			{
				assisted_excitation_forward(l, state);
			}

			if (l.antialiasing)
			{
				Darknet::NetworkState s = { 0 };
				s.train = state.train;
				s.workspace = state.workspace;
				s.net = state.net;
				s.input = l.output;
				forward_convolutional_layer(*(l.input_layer), s);
				memcpy(l.output, l.input_layer->output, l.input_layer->outputs * l.input_layer->batch * sizeof(float));
			}

			return;
		}
		mps_flush_deferred_output(prev);
	}
#endif

	fill_cpu(l.outputs*l.batch, 0, l.output, 1);

	if (l.xnor && (!l.align_bit_weights || state.train))
	{
		if (!l.align_bit_weights || state.train)
		{
			binarize_weights(l.weights, l.n, l.nweights, l.binary_weights);
		}
		swap_binary(&l);
		binarize_cpu(state.input, l.c*l.h*l.w*l.batch, l.binary_input);
		state.input = l.binary_input;
	}

	int m = l.n / l.groups;
	int k = l.size*l.size*l.c / l.groups;
	int n = out_h*out_w;

	for (int i = 0; i < l.batch; ++i)
	{
		for (int j = 0; j < l.groups; ++j)
		{
			float *a = l.weights +j*l.nweights / l.groups;
			float *b = state.workspace;
			float *c = l.output +(i*l.groups + j)*n*m;

			//gemm(0,0,m,n,k,1,a,k,b,n,1,c,n);
			//gemm_nn_custom(m, n, k, 1, a, k, b, n, c, n);
			if (l.xnor && l.align_bit_weights && !state.train && l.stride_x == l.stride_y)
			{
				memset(b, 0, l.bit_align*l.size*l.size*l.c * sizeof(float));

				if (l.c % 32 == 0)
				{
					int ldb_align = l.lda_align;
					size_t new_ldb = k + (ldb_align - k%ldb_align); // (k / 8 + 1) * 8;
					//size_t t_intput_size = new_ldb * l.bit_align;// n;
					//size_t t_bit_input_size = t_intput_size / 8;// +1;

					int re_packed_input_size = l.c * l.w * l.h;
					memset(state.workspace, 0, re_packed_input_size * sizeof(float));

					const size_t new_c = l.c / 32;
					size_t in_re_packed_input_size = new_c * l.w * l.h + 1;
					memset(l.bin_re_packed_input, 0, in_re_packed_input_size * sizeof(uint32_t));

					//float *re_packed_input = calloc(l.c * l.w * l.h, sizeof(float));
					//uint32_t *bin_re_packed_input = calloc(new_c * l.w * l.h + 1, sizeof(uint32_t));

					// float32x4 by channel (as in cuDNN)
					repack_input(state.input, state.workspace, l.w, l.h, l.c);

					// 32 x floats -> 1 x uint32_t
					float_to_bit(state.workspace, (unsigned char *)l.bin_re_packed_input, l.c * l.w * l.h);

					//free(re_packed_input);

					// slow - convolution the packed inputs and weights: float x 32 by channel (as in cuDNN)
					//convolution_repacked((uint32_t *)bin_re_packed_input, (uint32_t *)l.align_bit_weights, l.output,
					//    l.w, l.h, l.c, l.n, l.size, l.pad, l.new_lda, l.mean_arr);

					// // then exit from if ()

					im2col_cpu_custom((float *)l.bin_re_packed_input, new_c, l.h, l.w, l.size, l.stride, l.pad, state.workspace);
					//im2col_cpu((float *)bin_re_packed_input, new_c, l.h, l.w, l.size, l.stride, l.pad, b);

					//free(bin_re_packed_input);

					int new_k = l.size*l.size*l.c / 32;

					// good for (l.c == 64)
					//gemm_nn_bin_32bit_packed(m, n, new_k, 1,
					//    l.align_bit_weights, l.new_lda/32,
					//    b, n,
					//    c, n, l.mean_arr);

					// // then exit from if ()

					transpose_uint32((uint32_t *)state.workspace, (uint32_t*)l.t_bit_input, new_k, n, n, new_ldb);

					// the main GEMM function
					gemm_nn_custom_bin_mean_transposed(m, n, k, 1, (unsigned char*)l.align_bit_weights, new_ldb, (unsigned char*)l.t_bit_input, new_ldb, c, n, l.mean_arr);

					// // alternative GEMM
					//gemm_nn_bin_transposed_32bit_packed(m, n, new_k, 1,
					//    l.align_bit_weights, l.new_lda/32,
					//    t_bit_input, new_ldb / 32,
					//    c, n, l.mean_arr);

					//free(t_bit_input);
				}
				else
				{
					im2col_cpu_custom_bin(state.input, l.c, l.h, l.w, l.size, l.stride, l.pad, state.workspace, l.bit_align);

					//size_t ldb_align = 256; // 256 bit for AVX2
					int ldb_align = l.lda_align;
					size_t new_ldb = k + (ldb_align - k%ldb_align);
					/*size_t t_intput_size = */ binary_transpose_align_input(k, n, state.workspace, &l.t_bit_input, ldb_align, l.bit_align);

					// 5x times faster than gemm()-float32
					gemm_nn_custom_bin_mean_transposed(m, n, k, 1, (unsigned char*)l.align_bit_weights, new_ldb, (unsigned char*)l.t_bit_input, new_ldb, c, n, l.mean_arr);
				}

				add_bias(l.output, l.biases, l.batch, l.n, out_h*out_w);

				//activate_array(l.output, m*n*l.batch, l.activation);
				if (l.activation == SWISH) activate_array_swish(l.output, l.outputs*l.batch, l.activation_input, l.output);
				else if (l.activation == MISH) activate_array_mish(l.output, l.outputs*l.batch, l.activation_input, l.output);
				else if (l.activation == HARD_MISH) activate_array_hard_mish(l.output, l.outputs*l.batch, l.activation_input, l.output);
				else if (l.activation == EML) activate_array_eml(l.output, l.outputs*l.batch, l.activation_input, l.output);
				else if (l.activation == NORM_CHAN) activate_array_normalize_channels(l.output, l.outputs*l.batch, l.batch, l.out_c, l.out_w*l.out_h, l.output);
				else if (l.activation == NORM_CHAN_SOFTMAX) activate_array_normalize_channels_softmax(l.output, l.outputs*l.batch, l.batch, l.out_c, l.out_w*l.out_h, l.output, 0);
				else if (l.activation == NORM_CHAN_SOFTMAX_MAXVAL) activate_array_normalize_channels_softmax(l.output, l.outputs*l.batch, l.batch, l.out_c, l.out_w*l.out_h, l.output, 1);
				else activate_array_cpu_custom(l.output, m*n*l.batch, l.activation);
				return;
			}
			else
			{
				float *im = state.input + (i*l.groups + j)*(l.c / l.groups)*l.h*l.w;
				if (l.size == 1 && l.stride == 1 && l.dilation == 1)
				{
					b = im;
				}
				else
				{
					im2col_cpu_ext(im,   // input
						l.c / l.groups,     // input channels
						l.h, l.w,           // input size (h, w)
						l.size, l.size,     // kernel size (h, w)
						l.pad * l.dilation, l.pad * l.dilation,       // padding (h, w)
						l.stride_y, l.stride_x, // stride (h, w)
						l.dilation, l.dilation, // dilation (h, w)
						b);                 // output
				}

				gemm_cpu(0, 0, m, n, k, 1, a, k, b, n, 1, c, n);
				// bit-count to float
			}
		}
	}

	if (l.batch_normalize)
	{
		forward_batchnorm_layer(l, state);
	}
	else
	{
		add_bias(l.output, l.biases, l.batch, l.n, out_h*out_w);
	}

	//activate_array(l.output, m*n*l.batch, l.activation);
	if (l.activation == SWISH) activate_array_swish(l.output, l.outputs*l.batch, l.activation_input, l.output);
	else if (l.activation == MISH) activate_array_mish(l.output, l.outputs*l.batch, l.activation_input, l.output);
	else if (l.activation == HARD_MISH) activate_array_hard_mish(l.output, l.outputs*l.batch, l.activation_input, l.output);
	else if (l.activation == EML) activate_array_eml(l.output, l.outputs*l.batch, l.activation_input, l.output);
	else if (l.activation == NORM_CHAN) activate_array_normalize_channels(l.output, l.outputs*l.batch, l.batch, l.out_c, l.out_w*l.out_h, l.output);
	else if (l.activation == NORM_CHAN_SOFTMAX) activate_array_normalize_channels_softmax(l.output, l.outputs*l.batch, l.batch, l.out_c, l.out_w*l.out_h, l.output, 0);
	else if (l.activation == NORM_CHAN_SOFTMAX_MAXVAL) activate_array_normalize_channels_softmax(l.output, l.outputs*l.batch, l.batch, l.out_c, l.out_w*l.out_h, l.output, 1);
	else activate_array_cpu_custom(l.output, l.outputs*l.batch, l.activation);

	if (l.binary || l.xnor)
	{
		swap_binary(&l);
	}

	//visualize_convolutional_layer(l, "conv_visual", NULL);
	//cv::waitKey(0);

	if (l.assisted_excitation && state.train)
	{
		assisted_excitation_forward(l, state);
	}

	if (l.antialiasing)
	{
		Darknet::NetworkState s = { 0 };
		s.train = state.train;
		s.workspace = state.workspace;
		s.net = state.net;
		s.input = l.output;
		forward_convolutional_layer(*(l.input_layer), s);
		//simple_copy_ongpu(l.outputs*l.batch, l.output, l.input_antialiasing);
		memcpy(l.output, l.input_layer->output, l.input_layer->outputs * l.input_layer->batch * sizeof(float));
	}
}

void forward_eml_convolutional_layer(Darknet::Layer & l, Darknet::NetworkState state)
{
	TAT(TATPARMS);

	Darknet::NetworkState branch_state = state;
	forward_convolutional_layer(*(l.input_layer), branch_state);
	forward_convolutional_layer(*(l.self_layer), branch_state);

	const int total = l.outputs * l.batch;
	const float clamp = l.alpha > 0.0f ? l.alpha : 4.0f;
	const float eps = l.beta > 0.0f ? l.beta : 0.000001f;
	const float scale = l.scale;

	#pragma omp parallel for
	for (int i = 0; i < total; ++i)
	{
		const float x = l.input_layer->output[i];
		const float y = l.self_layer->output[i];
		const float eml = expf(eml_clamp_value(x, clamp)) - logf(eml_softplus(y) + eps);
		float out = scale * eml;
		if (l.shortcut && l.inputs == l.outputs)
		{
			out += state.input[i];
		}
		l.output[i] = out;
	}
}


void assisted_excitation_forward(Darknet::Layer & l, Darknet::NetworkState state)
{
	TAT(TATPARMS);

	const int iteration_num = (*state.net.seen) / (state.net.batch*state.net.subdivisions);

	// calculate alpha
	float alpha = (1 + cos(M_PI * iteration_num / state.net.max_batches));

	if (l.assisted_excitation > 1)
	{
		if (iteration_num > l.assisted_excitation)
		{
			alpha = 0;
		}
		else
		{
			alpha = (1 + cos(M_PI * iteration_num / l.assisted_excitation));
		}
	}

	float *a_avg = (float *)xcalloc(l.out_w * l.out_h * l.batch, sizeof(float));
	float *g = (float *)xcalloc(l.out_w * l.out_h * l.batch, sizeof(float));

	l.max_boxes = state.net.num_boxes;
	l.truths = l.max_boxes*(4 + 1);

	for (int b = 0; b < l.batch; ++b)
	{
		// calculate G
		for (int t = 0; t < state.net.num_boxes; ++t)
		{
			Darknet::Box truth = float_to_box_stride(state.truth + t*(4 + 1) + b*l.truths, 1);
			if (!truth.x)
			{
				break;  // continue;
			}

			int left = floor((truth.x - truth.w / 2) * l.out_w);
			int right = ceil((truth.x + truth.w / 2) * l.out_w);
			int top = floor((truth.y - truth.h / 2) * l.out_h);
			int bottom = ceil((truth.y + truth.h / 2) * l.out_h);

			for (int w = left; w <= right; w++)
			{
				for (int h = top; h < bottom; h++)
				{
					g[w + l.out_w * h + l.out_w*l.out_h*b] = 1;
				}
			}
		}
	}

	for (int b = 0; b < l.batch; ++b)
	{
		// calculate average A
		for (int w = 0; w < l.out_w; w++)
		{
			for (int h = 0; h < l.out_h; h++)
			{
				for (int c = 0; c < l.out_c; c++)
				{
					a_avg[w + l.out_w*(h + l.out_h*b)] += l.output[w + l.out_w*(h + l.out_h*(c + l.out_c*b))];
				}
				a_avg[w + l.out_w*(h + l.out_h*b)] /= l.out_c;  // a_avg / d
			}
		}
	}

	// change activation
	for (int b = 0; b < l.batch; ++b)
	{
		for (int w = 0; w < l.out_w; w++)
		{
			for (int h = 0; h < l.out_h; h++)
			{
				for (int c = 0; c < l.out_c; c++)
				{
					// a = a + alpha(t) + e(c,i,j) = a + alpha(t) + g(i,j) * avg_a(i,j) / channels
					l.output[w + l.out_w*(h + l.out_h*(c + l.out_c*b))] +=
						alpha *
						g[w + l.out_w*(h + l.out_h*b)] *
						a_avg[w + l.out_w*(h + l.out_h*b)];

					//l.output[w + l.out_w*(h + l.out_h*(c + l.out_c*b))] =
					//    alpha * g[w + l.out_w*(h + l.out_h*b)] * a_avg[w + l.out_w*(h + l.out_h*b)];
				}
			}
		}
	}

	free(g);
	free(a_avg);
}


void backward_convolutional_layer(Darknet::Layer & l, Darknet::NetworkState state)
{
	TAT(TATPARMS);

	int m = l.n / l.groups;
	int n = l.size*l.size*l.c / l.groups;
	int k = l.out_w*l.out_h;

	if (l.activation == SWISH) gradient_array_swish(l.output, l.outputs*l.batch, l.activation_input, l.delta);
	else if (l.activation == MISH) gradient_array_mish(l.outputs*l.batch, l.activation_input, l.delta);
	else if (l.activation == HARD_MISH) gradient_array_hard_mish(l.outputs*l.batch, l.activation_input, l.delta);
	else if (l.activation == EML) gradient_array_eml(l.outputs*l.batch, l.activation_input, l.delta);
	else if (l.activation == NORM_CHAN_SOFTMAX || l.activation == NORM_CHAN_SOFTMAX_MAXVAL) gradient_array_normalize_channels_softmax(l.output, l.outputs*l.batch, l.batch, l.out_c, l.out_w*l.out_h, l.delta);
	else if (l.activation == NORM_CHAN) gradient_array_normalize_channels(l.output, l.outputs*l.batch, l.batch, l.out_c, l.out_w*l.out_h, l.delta);
	else gradient_array(l.output, l.outputs*l.batch, l.activation, l.delta);

	if (l.batch_normalize)
	{
		backward_batchnorm_layer(l, state);
	}
	else
	{
		backward_bias(l.bias_updates, l.delta, l.batch, l.n, k);
	}

	for (int i = 0; i < l.batch; ++i)
	{
		for (int j = 0; j < l.groups; ++j)
		{
			float *a = l.delta + (i*l.groups + j)*m*k;
			float *b = state.workspace;
			float *c = l.weight_updates + j*l.nweights / l.groups;

			float *im = state.input + (i*l.groups + j)* (l.c / l.groups)*l.h*l.w;

			//im2col_cpu(im, l.c / l.groups, l.h, l.w, l.size, l.stride, l.pad, b);
			im2col_cpu_ext(
				im,                 // input
				l.c / l.groups,     // input channels
				l.h, l.w,           // input size (h, w)
				l.size, l.size,     // kernel size (h, w)
				l.pad * l.dilation, l.pad * l.dilation,       // padding (h, w)
				l.stride_y, l.stride_x, // stride (h, w)
				l.dilation, l.dilation, // dilation (h, w)
				b);                 // output

			gemm_cpu(0, 1, m, n, k, 1, a, k, b, k, 1, c, n);

			if (state.delta)
			{
				a = l.weights + j*l.nweights / l.groups;
				b = l.delta + (i*l.groups + j)*m*k;
				c = state.workspace;

				gemm_cpu(1, 0, n, k, m, 1, a, n, b, k, 0, c, k);

				//col2im_cpu(state.workspace, l.c / l.groups, l.h, l.w, l.size, l.stride,
				//     l.pad, state.delta + (i*l.groups + j)*l.c / l.groups*l.h*l.w);

				col2im_cpu_ext(
					state.workspace,        // input
					l.c / l.groups,         // input channels (h, w)
					l.h, l.w,               // input size (h, w)
					l.size, l.size,         // kernel size (h, w)
					l.pad * l.dilation, l.pad * l.dilation,           // padding (h, w)
					l.stride_y, l.stride_x,     // stride (h, w)
					l.dilation, l.dilation, // dilation (h, w)
					state.delta + (i*l.groups + j)* (l.c / l.groups)*l.h*l.w); // output (delta)
			}
		}
	}
}

void backward_eml_convolutional_layer(Darknet::Layer & l, Darknet::NetworkState state)
{
	TAT(TATPARMS);

	const int total = l.outputs * l.batch;
	const float clamp = l.alpha > 0.0f ? l.alpha : 4.0f;
	const float eps = l.beta > 0.0f ? l.beta : 0.000001f;
	const float scale = l.scale;

	fill_cpu(total, 0, l.input_layer->delta, 1);
	fill_cpu(total, 0, l.self_layer->delta, 1);

	#pragma omp parallel for
	for (int i = 0; i < total; ++i)
	{
		const float d = l.delta[i] * scale;
		const float x = l.input_layer->output[i];
		const float y = l.self_layer->output[i];
		const float dx = (x > -clamp && x < clamp) ? expf(eml_clamp_value(x, clamp)) : 0.0f;
		const float sp = eml_softplus(y);
		const float dy = -eml_sigmoid(y) / (sp + eps);
		l.input_layer->delta[i] = d * dx;
		l.self_layer->delta[i] = d * dy;
	}

	if (l.shortcut && l.inputs == l.outputs && state.delta)
	{
		axpy_cpu(total, 1.0f, l.delta, 1, state.delta, 1);
	}

	Darknet::NetworkState branch_state = state;
	backward_convolutional_layer(*(l.input_layer), branch_state);
	backward_convolutional_layer(*(l.self_layer), branch_state);
}


void update_convolutional_layer(Darknet::Layer & l, int batch, float learning_rate_init, float momentum, float decay)
{
	TAT(TATPARMS);

	const float learning_rate = learning_rate_init * l.learning_rate_scale;

	axpy_cpu(l.nweights, -decay*batch, l.weights, 1, l.weight_updates, 1);
	axpy_cpu(l.nweights, learning_rate / batch, l.weight_updates, 1, l.weights, 1);
	scal_cpu(l.nweights, momentum, l.weight_updates, 1);

	axpy_cpu(l.n, learning_rate / batch, l.bias_updates, 1, l.biases, 1);
	scal_cpu(l.n, momentum, l.bias_updates, 1);

	if (l.scales)
	{
		axpy_cpu(l.n, learning_rate / batch, l.scale_updates, 1, l.scales, 1);
		scal_cpu(l.n, momentum, l.scale_updates, 1);
	}
}

void update_eml_convolutional_layer(Darknet::Layer & l, int batch, float learning_rate, float momentum, float decay)
{
	TAT(TATPARMS);

	update_convolutional_layer(*(l.input_layer), batch, learning_rate, momentum, decay);
	update_convolutional_layer(*(l.self_layer), batch, learning_rate, momentum, decay);
}


Darknet::Image get_convolutional_weight(const Darknet::Layer & l, int i)
{
	TAT(TATPARMS);

	const int h = l.size;
	const int w = l.size;
	const int c = l.c / l.groups;

	return Darknet::float_to_image(w, h, c, l.weights + i * h * w * c);
}


void rgbgr_weights(const Darknet::Layer & l)
{
	TAT(TATPARMS);

	for (int i = 0; i < l.n; ++i)
	{
		Darknet::Image im = get_convolutional_weight(l, i);
		if (im.c == 3)
		{
			Darknet::rgbgr_image(im);
		}
	}
}

void rescale_weights(Darknet::Layer & l, float scale, float trans)
{
	TAT(TATPARMS);

	for (int i = 0; i < l.n; ++i)
	{
		Darknet::Image im = get_convolutional_weight(l, i);
		if (im.c == 3)
		{
			Darknet::scale_image(im, scale);
			float sum = sum_array(im.data, im.w*im.h*im.c);
			l.biases[i] += sum*trans;
		}
	}
}


Darknet::Image *visualize_convolutional_layer(const Darknet::Layer & l, const char * window, Darknet::Image * prev_weights)
{
	TAT(TATPARMS);

	Darknet::Image *single_weights = get_weights(l);

	std::string title = window;
	title += " " + std::to_string(single_weights->w) + "x" + std::to_string(single_weights->h) + "x" + std::to_string(single_weights->c);
	Darknet::show_images(single_weights, l.n, title.c_str());

	Darknet::Image delta = get_convolutional_image(l);
	Darknet::Image dc = Darknet::collapse_image_layers(delta, 1);

	title += " [Output]";

	Darknet::show_image(dc, title.c_str());
	//save_image(dc, buff);
	Darknet::free_image(dc);
	return single_weights;
}
