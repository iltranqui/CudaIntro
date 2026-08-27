#include "darknet_internal.hpp"
#include "dcnv4_layer.hpp"
#include "activations.hpp"
#include "gemm.hpp"
#include "blas.hpp"
#include "batchnorm_layer.hpp"
#include "convolutional_layer.hpp"
#include "im2col.hpp"

#include <algorithm>
#include <cmath>
#include <cfloat>
#include <cstring>
#include <vector>

namespace
{
    static auto & cfg_and_state = Darknet::CfgAndState::get();

    inline bool dcnv4_needs_activation_input(const ACTIVATION activation)
    {
        return activation == SWISH || activation == MISH || activation == HARD_MISH || activation == EML;
    }

    inline int dcnv4_kernel_points(const Darknet::Layer & l)
    {
        return l.size * l.size - (l.remove_center ? 1 : 0);
    }

    inline int dcnv4_padded_offset_dim(const Darknet::Layer & l)
    {
        const int raw = l.groups * dcnv4_kernel_points(l) * 3;
        return ((raw + 7) / 8) * 8;
    }

#if defined(DARKNET_GPU) && defined(CUDNN)
    inline cudnnDataType_t dcnv4_cudnn_16bit_data_type(const Darknet::Layer & l)
    {
#if defined(DARKNET_GPU_CUDA) && defined(CUDNN_DATA_BFLOAT16)
        if (l.cudnn_16bit_mode == DARKNET_CUDNN_16BIT_BF16)
        {
            return CUDNN_DATA_BFLOAT16;
        }
#else
        (void) l;
#endif
        return CUDNN_DATA_HALF;
    }

    inline void dcnv4_create_cudnn_descriptors_if_needed(Darknet::Layer & l)
    {
        if (l.srcTensorDesc == nullptr) CHECK_CUDNN(cudnnCreateTensorDescriptor(&l.srcTensorDesc));
        if (l.dstTensorDesc == nullptr) CHECK_CUDNN(cudnnCreateTensorDescriptor(&l.dstTensorDesc));
        if (l.dsrcTensorDesc == nullptr) CHECK_CUDNN(cudnnCreateTensorDescriptor(&l.dsrcTensorDesc));
        if (l.ddstTensorDesc == nullptr) CHECK_CUDNN(cudnnCreateTensorDescriptor(&l.ddstTensorDesc));
        if (l.srcTensorDesc16 == nullptr) CHECK_CUDNN(cudnnCreateTensorDescriptor(&l.srcTensorDesc16));
        if (l.dstTensorDesc16 == nullptr) CHECK_CUDNN(cudnnCreateTensorDescriptor(&l.dstTensorDesc16));
        if (l.dsrcTensorDesc16 == nullptr) CHECK_CUDNN(cudnnCreateTensorDescriptor(&l.dsrcTensorDesc16));
        if (l.ddstTensorDesc16 == nullptr) CHECK_CUDNN(cudnnCreateTensorDescriptor(&l.ddstTensorDesc16));
        if (l.weightDesc == nullptr) CHECK_CUDNN(cudnnCreateFilterDescriptor(&l.weightDesc));
        if (l.dweightDesc == nullptr) CHECK_CUDNN(cudnnCreateFilterDescriptor(&l.dweightDesc));
        if (l.weightDesc16 == nullptr) CHECK_CUDNN(cudnnCreateFilterDescriptor(&l.weightDesc16));
        if (l.dweightDesc16 == nullptr) CHECK_CUDNN(cudnnCreateFilterDescriptor(&l.dweightDesc16));
        if (l.convDesc == nullptr) CHECK_CUDNN(cudnnCreateConvolutionDescriptor(&l.convDesc));
    }

    void dcnv4_setup_cudnn_offset_descriptors(Darknet::Layer & l)
    {
        if (cfg_and_state.gpu_index < 0)
        {
            return;
        }

        dcnv4_create_cudnn_descriptors_if_needed(l);

        const int padded_offset_dim = dcnv4_padded_offset_dim(l);
        const int d_stride = std::max(1, l.d_stride);
        const int H_c = (l.out_h + d_stride - 1) / d_stride;
        const int W_c = (l.out_w + d_stride - 1) / d_stride;
        const int stride_h = l.stride_y * d_stride;
        const int stride_w = l.stride_x * d_stride;

        CHECK_CUDNN(cudnnSetTensor4dDescriptor(l.srcTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, l.batch, l.c, l.h, l.w));
        CHECK_CUDNN(cudnnSetTensor4dDescriptor(l.dstTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, l.batch, padded_offset_dim, H_c, W_c));
        CHECK_CUDNN(cudnnSetTensor4dDescriptor(l.dsrcTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, l.batch, l.c, l.h, l.w));
        CHECK_CUDNN(cudnnSetTensor4dDescriptor(l.ddstTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, l.batch, padded_offset_dim, H_c, W_c));
        CHECK_CUDNN(cudnnSetFilter4dDescriptor(l.weightDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, padded_offset_dim, l.c, l.size, l.size));
        CHECK_CUDNN(cudnnSetFilter4dDescriptor(l.dweightDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, padded_offset_dim, l.c, l.size, l.size));

        const cudnnDataType_t data_type_16 = dcnv4_cudnn_16bit_data_type(l);
        CHECK_CUDNN(cudnnSetTensor4dDescriptor(l.srcTensorDesc16, CUDNN_TENSOR_NCHW, data_type_16, l.batch, l.c, l.h, l.w));
        CHECK_CUDNN(cudnnSetTensor4dDescriptor(l.dstTensorDesc16, CUDNN_TENSOR_NCHW, data_type_16, l.batch, padded_offset_dim, H_c, W_c));
        CHECK_CUDNN(cudnnSetTensor4dDescriptor(l.dsrcTensorDesc16, CUDNN_TENSOR_NCHW, data_type_16, l.batch, l.c, l.h, l.w));
        CHECK_CUDNN(cudnnSetTensor4dDescriptor(l.ddstTensorDesc16, CUDNN_TENSOR_NCHW, data_type_16, l.batch, padded_offset_dim, H_c, W_c));
        CHECK_CUDNN(cudnnSetFilter4dDescriptor(l.weightDesc16, data_type_16, CUDNN_TENSOR_NCHW, padded_offset_dim, l.c, l.size, l.size));
        CHECK_CUDNN(cudnnSetFilter4dDescriptor(l.dweightDesc16, data_type_16, CUDNN_TENSOR_NCHW, padded_offset_dim, l.c, l.size, l.size));

#if (CUDNN_MAJOR >= 6)
        CHECK_CUDNN(cudnnSetConvolution2dDescriptor(l.convDesc, l.pad, l.pad, stride_h, stride_w, l.dilation, l.dilation, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));
#else
        CHECK_CUDNN(cudnnSetConvolution2dDescriptor(l.convDesc, l.pad, l.pad, stride_h, stride_w, l.dilation, l.dilation, CUDNN_CROSS_CORRELATION));
#endif
#if (CUDNN_MAJOR >= 7)
        CHECK_CUDNN(cudnnSetConvolutionGroupCount(l.convDesc, 1));
        CHECK_CUDNN(cudnnSetConvolutionMathType(l.convDesc, CUDNN_TENSOR_OP_MATH));
#endif

        l.fw_algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
        l.bd_algo = CUDNN_CONVOLUTION_BWD_DATA_ALGO_0;
        l.bf_algo = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0;
        l.fw_algo16 = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
        l.bd_algo16 = CUDNN_CONVOLUTION_BWD_DATA_ALGO_0;
        l.bf_algo16 = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0;
    }
#endif

    inline void dcnv4_activate_cpu(Darknet::Layer & l)
    {
        const int total = l.outputs * l.batch;
        if (l.activation == SWISH) activate_array_swish(l.output, total, l.activation_input, l.output);
        else if (l.activation == MISH) activate_array_mish(l.output, total, l.activation_input, l.output);
        else if (l.activation == HARD_MISH) activate_array_hard_mish(l.output, total, l.activation_input, l.output);
        else if (l.activation == EML) activate_array_eml(l.output, total, l.activation_input, l.output);
        else if (l.activation == NORM_CHAN) activate_array_normalize_channels(l.output, total, l.batch, l.out_c, l.out_w * l.out_h, l.output);
        else if (l.activation == NORM_CHAN_SOFTMAX) activate_array_normalize_channels_softmax(l.output, total, l.batch, l.out_c, l.out_w * l.out_h, l.output, 0);
        else if (l.activation == NORM_CHAN_SOFTMAX_MAXVAL) activate_array_normalize_channels_softmax(l.output, total, l.batch, l.out_c, l.out_w * l.out_h, l.output, 1);
        else if (l.activation != LINEAR) activate_array_cpu_custom(l.output, total, l.activation);
    }

    inline void dcnv4_add_bias_nhwc_cpu(float * dst, const float * bias, const int batch, const int spatial, const int channels)
    {
        const size_t total = static_cast<size_t>(batch) * spatial * channels;
        for (size_t idx = 0; idx < total; ++idx)
        {
            dst[idx] += bias[idx % channels];
        }
    }

    inline float dcnv4_bilinear_sample_zero_cpu(
        const float * input,
        const int height,
        const int width,
        const int channels,
        const int channel,
        const float h,
        const float w)
    {
        const int h0 = static_cast<int>(std::floor(h));
        const int w0 = static_cast<int>(std::floor(w));
        const int h1 = h0 + 1;
        const int w1 = w0 + 1;
        const float lh = h - static_cast<float>(h0);
        const float lw = w - static_cast<float>(w0);
        const float hh = 1.0f - lh;
        const float hw = 1.0f - lw;

        float value = 0.0f;
        const int plane = height * width;
        const float * channel_base = input + channel * plane;

        if (h0 >= 0 && h0 < height && w0 >= 0 && w0 < width) value += hh * hw * channel_base[h0 * width + w0];
        if (h0 >= 0 && h0 < height && w1 >= 0 && w1 < width) value += hh * lw * channel_base[h0 * width + w1];
        if (h1 >= 0 && h1 < height && w0 >= 0 && w0 < width) value += lh * hw * channel_base[h1 * width + w0];
        if (h1 >= 0 && h1 < height && w1 >= 0 && w1 < width) value += lh * lw * channel_base[h1 * width + w1];

        return value;
    }

    inline float dcnv4_bilinear_sample_border_nhwc_cpu(
        const float * src,
        const int channels,
        const int height,
        const int width,
        float h,
        float w,
        const int c)
    {
        if (height <= 0 || width <= 0) return 0.0f;

        const int h0_raw = static_cast<int>(std::floor(h));
        const int w0_raw = static_cast<int>(std::floor(w));
        const int h1_raw = h0_raw + 1;
        const int w1_raw = w0_raw + 1;
        const float lh = h - static_cast<float>(h0_raw);
        const float lw = w - static_cast<float>(w0_raw);
        const float hh = 1.0f - lh;
        const float hw = 1.0f - lw;

        const int h0 = std::clamp(h0_raw, 0, height - 1);
        const int h1 = std::clamp(h1_raw, 0, height - 1);
        const int w0 = std::clamp(w0_raw, 0, width - 1);
        const int w1 = std::clamp(w1_raw, 0, width - 1);

        return
            hh * hw * src[((h0 * width + w0) * channels) + c] +
            hh * lw * src[((h0 * width + w1) * channels) + c] +
            lh * hw * src[((h1 * width + w0) * channels) + c] +
            lh * lw * src[((h1 * width + w1) * channels) + c];
    }

    // Upsample offset tensors from the coarse output lattice to the full output lattice.
    // Coarse offsets are produced by im2col with stride=(layer_stride * d_stride), so
    // destination output location q maps to source coordinate q / d_stride.  This is
    // deliberately not image-resize center alignment.
    void dcnv4_upsample_offsets_lattice_cpu(
        const float * src,
        float * dst,
        const int batch,
        const int channels,
        const int src_h,
        const int src_w,
        const int dst_h,
        const int dst_w,
        const int d_stride)
    {
        const int stride = std::max(1, d_stride);
        const int src_spatial = src_h * src_w;
        const int dst_spatial = dst_h * dst_w;

        for (int b = 0; b < batch; ++b)
        {
            const float * src_b = src + static_cast<size_t>(b) * src_spatial * channels;
            float * dst_b = dst + static_cast<size_t>(b) * dst_spatial * channels;
            for (int y = 0; y < dst_h; ++y)
            {
                const float src_y = static_cast<float>(y) / static_cast<float>(stride);
                for (int x = 0; x < dst_w; ++x)
                {
                    const float src_x = static_cast<float>(x) / static_cast<float>(stride);
                    float * out = dst_b + static_cast<size_t>(y * dst_w + x) * channels;
                    for (int c = 0; c < channels; ++c)
                    {
                        out[c] = dcnv4_bilinear_sample_border_nhwc_cpu(src_b, channels, src_h, src_w, src_y, src_x, c);
                    }
                }
            }
        }
    }

    void dcnv4_predict_offsets_cpu(Darknet::Layer & l, Darknet::NetworkState state, float * workspace)
    {
        const int K = dcnv4_kernel_points(l);
        const int padded_offset_dim = dcnv4_padded_offset_dim(l);
        const int spatial_out = l.out_h * l.out_w;
        const int k_gemm = l.c * l.size * l.size;
        const int d_stride = std::max(1, l.d_stride);

        if (d_stride > 1)
        {
            const int h_coarse = (l.out_h + d_stride - 1) / d_stride;
            const int w_coarse = (l.out_w + d_stride - 1) / d_stride;
            const int spatial_coarse = h_coarse * w_coarse;
            const size_t coarse_im2col_floats = static_cast<size_t>(k_gemm) * spatial_coarse;
            float * coarse_im2col = workspace;
            float * coarse_offsets = workspace + coarse_im2col_floats;

            for (int b = 0; b < l.batch; ++b)
            {
                const float * im = state.input + static_cast<size_t>(b) * l.inputs;
                float * coarse_b = coarse_offsets + static_cast<size_t>(b) * spatial_coarse * padded_offset_dim;
                im2col_cpu_ext(
                    const_cast<float *>(im),
                    l.c, l.h, l.w,
                    l.size, l.size,
                    l.pad, l.pad,
                    l.stride_y * d_stride, l.stride_x * d_stride,
                    l.dilation, l.dilation,
                    coarse_im2col);
                gemm_cpu(1, 1, spatial_coarse, padded_offset_dim, k_gemm, 1.0f,
                    coarse_im2col, spatial_coarse,
                    l.offset_weights, k_gemm,
                    0.0f, coarse_b, padded_offset_dim);
            }

            dcnv4_add_bias_nhwc_cpu(coarse_offsets, l.offset_biases, l.batch, spatial_coarse, padded_offset_dim);
            dcnv4_upsample_offsets_lattice_cpu(coarse_offsets, l.offsets, l.batch, padded_offset_dim, h_coarse, w_coarse, l.out_h, l.out_w, d_stride);
        }
        else
        {
            for (int b = 0; b < l.batch; ++b)
            {
                const float * im = state.input + static_cast<size_t>(b) * l.inputs;
                float * offsets_b = l.offsets + static_cast<size_t>(b) * spatial_out * padded_offset_dim;
                im2col_cpu_ext(
                    const_cast<float *>(im),
                    l.c, l.h, l.w,
                    l.size, l.size,
                    l.pad, l.pad,
                    l.stride_y, l.stride_x,
                    l.dilation, l.dilation,
                    workspace);
                gemm_cpu(1, 1, spatial_out, padded_offset_dim, k_gemm, 1.0f,
                    workspace, spatial_out,
                    l.offset_weights, k_gemm,
                    0.0f, offsets_b, padded_offset_dim);
            }

            dcnv4_add_bias_nhwc_cpu(l.offsets, l.offset_biases, l.batch, spatial_out, padded_offset_dim);
        }

        (void)K;
    }

    void dcnv4_deformable_forward_cpu(Darknet::Layer & l, Darknet::NetworkState state)
    {
        const int K = dcnv4_kernel_points(l);
        const int padded_offset_dim = dcnv4_padded_offset_dim(l);
        const int spatial_out = l.out_h * l.out_w;
        const int G = l.groups;
        const int D = l.n / G;
        const int center_h = l.size / 2;
        const int center_w = l.size / 2;
        const int kernel_center_w = (l.dilation * (l.size - 1)) >> 1;
        const int kernel_center_h = (l.dilation * (l.size - 1)) >> 1;

        std::vector<float> masks(static_cast<size_t>(K));

        for (int b = 0; b < l.batch; ++b)
        {
            const float * input_b = state.input + static_cast<size_t>(b) * l.inputs;
            float * output_b = l.output + static_cast<size_t>(b) * l.outputs;
            const float * offsets_b = l.offsets + static_cast<size_t>(b) * spatial_out * padded_offset_dim;

            for (int q = 0; q < spatial_out; ++q)
            {
                const int out_y = q / l.out_w;
                const int out_x = q % l.out_w;
                const int p0_w = kernel_center_w - l.pad + out_x * l.stride_x;
                const int p0_h = kernel_center_h - l.pad + out_y * l.stride_y;
                const float p0_w_scaled = static_cast<float>(p0_w) - static_cast<float>(kernel_center_w) * l.offset_scale;
                const float p0_h_scaled = static_cast<float>(p0_h) - static_cast<float>(kernel_center_h) * l.offset_scale;

                for (int g = 0; g < G; ++g)
                {
                    const float * offset_g = offsets_b + static_cast<size_t>(q) * padded_offset_dim + g * K * 3;
                    for (int k = 0; k < K; ++k)
                    {
                        masks[k] = offset_g[K * 2 + k];
                    }

                    if (l.softmax)
                    {
                        float max_mask = -FLT_MAX;
                        for (int k = 0; k < K; ++k) max_mask = std::max(max_mask, masks[k]);
                        float sum = 0.0f;
                        for (int k = 0; k < K; ++k)
                        {
                            masks[k] = std::exp(masks[k] - max_mask);
                            sum += masks[k];
                        }
                        const float inv_sum = 1.0f / (sum + 1e-6f);
                        for (int k = 0; k < K; ++k) masks[k] *= inv_sum;
                    }

                    int offset_idx = 0;
                    int mask_idx = 0;
                    for (int kw = 0; kw < l.size; ++kw)
                    {
                        for (int kh = 0; kh < l.size; ++kh)
                        {
                            if (kw != center_w || kh != center_h || !l.remove_center)
                            {
                                const float sample_w = p0_w_scaled + (static_cast<float>(kw * l.dilation) + offset_g[offset_idx]) * l.offset_scale;
                                const float sample_h = p0_h_scaled + (static_cast<float>(kh * l.dilation) + offset_g[offset_idx + 1]) * l.offset_scale;
                                const float attn = masks[mask_idx];
                                const int channel_base = g * D;

                                for (int d = 0; d < D; ++d)
                                {
                                    const int c = channel_base + d;
                                    output_b[static_cast<size_t>(c) * spatial_out + q] +=
                                        attn * dcnv4_bilinear_sample_zero_cpu(input_b, l.h, l.w, l.c, c, sample_h, sample_w);
                                }

                                offset_idx += 2;
                                ++mask_idx;
                            }
                        }
                    }
                }
            }
        }
    }
}

size_t get_dcnv4_workspace_size(const Darknet::Layer & l)
{
    const int K = dcnv4_kernel_points(l);
    const int padded_offset_dim = dcnv4_padded_offset_dim(l);

    const int d_stride = std::max(1, l.d_stride);
    const int H_c = (l.out_h + d_stride - 1) / d_stride;
    const int W_c = (l.out_w + d_stride - 1) / d_stride;
    const size_t input_nhwc_size = static_cast<size_t>(l.batch) * l.c * l.h * l.w * sizeof(float);
    const size_t output_nhwc_size = static_cast<size_t>(l.batch) * l.n * l.out_h * l.out_w * sizeof(float);
    const size_t offsets_nhwc_size = static_cast<size_t>(l.batch) * l.out_h * l.out_w * padded_offset_dim * sizeof(float);
    const size_t coarse_offsets_size = static_cast<size_t>(l.batch) * H_c * W_c * padded_offset_dim * sizeof(float);
    const size_t im2col_size = static_cast<size_t>(l.c) * l.size * l.size * l.out_h * l.out_w * sizeof(float);
    const size_t coarse_im2col_size = static_cast<size_t>(l.c) * l.size * l.size * H_c * W_c * sizeof(float);
    const size_t offset_source_nchw_size = (d_stride > 1) ? coarse_offsets_size : offsets_nhwc_size;
    const size_t cudnn_forward_tail = offset_source_nchw_size + ((d_stride > 1) ? coarse_offsets_size : 0);
    const size_t cudnn_backward_tail = offset_source_nchw_size + ((d_stride > 1) ? coarse_offsets_size : 0);

    const size_t fallback_tail = coarse_im2col_size + coarse_offsets_size;
    const size_t forward_tail = std::max(im2col_size, std::max(fallback_tail, cudnn_forward_tail));
    const size_t backward_tail = 2 * input_nhwc_size + std::max(im2col_size, std::max(fallback_tail, cudnn_backward_tail));
    (void)K;
    return input_nhwc_size + output_nhwc_size + offsets_nhwc_size + std::max(forward_tail, backward_tail);
}

#if defined(DARKNET_GPU) && defined(CUDNN)
void set_dcnv4_cudnn_16bit_mode(Darknet::Layer * l, int mode)
{
    TAT(TATPARMS);

    if (l == nullptr)
    {
        return;
    }
    if (mode != DARKNET_CUDNN_16BIT_HALF && mode != DARKNET_CUDNN_16BIT_BF16)
    {
        darknet_fatal_error(DARKNET_LOC, "invalid DCNv4 cuDNN 16-bit mode %d", mode);
    }
    if (l->cudnn_16bit_mode == mode && l->srcTensorDesc16 != nullptr)
    {
        return;
    }

    l->cudnn_16bit_mode = mode;
    dcnv4_setup_cudnn_offset_descriptors(*l);
    l->workspace_size = get_dcnv4_workspace_size(*l);
    if (l->offset_weights_gpu && l->offset_weights_gpu16)
    {
        const size_t offset_weights_size = static_cast<size_t>(dcnv4_padded_offset_dim(*l)) * l->c * l->size * l->size;
        cuda_convert_f32_to_cudnn_16bit(l->offset_weights_gpu, offset_weights_size, l->offset_weights_gpu16, l->cudnn_16bit_mode);
    }
}
#endif

Darknet::Layer make_dcnv4_layer(int batch, int steps, int h, int w, int c, int n, int groups,
                               int size, int stride_x, int stride_y, int dilation,
                               int padding, ACTIVATION activation, int batch_normalize,
                               float offset_scale, int remove_center, int d_stride, int block_thread, int softmax,
                               int index, int train)
{
    TAT(TATPARMS);

    if (groups < 1) groups = 1;
    if (stride_x < 1) stride_x = 1;
    if (stride_y < 1) stride_y = 1;
    if (dilation < 1) dilation = 1;
    if (d_stride < 1) d_stride = 1;
    if (size < 1) size = 1;
    if (size * size - (remove_center ? 1 : 0) <= 0) {
        darknet_fatal_error(DARKNET_LOC, "DCNv4 layer %d: remove_center=1 is invalid with size=%d", index, size);
    }

    const int total_batch = batch * std::max(1, steps);
    Darknet::Layer l = { (Darknet::ELayerType)0 };
    l.type = Darknet::ELayerType::DCNV4;
    l.train = train;
    l.steps = steps;

    l.groups = groups;
    l.batch = batch;
    l.h = h;
    l.w = w;
    l.c = c;
    l.n = n;
    l.batch_normalize = batch_normalize;
    l.size = size;
    l.stride = stride_x;
    l.stride_x = stride_x;
    l.stride_y = stride_y;
    l.dilation = dilation;
    l.pad = padding;
    l.index = index;
    l.offset_scale = offset_scale;
    l.remove_center = remove_center;
    l.d_stride = d_stride;
    l.softmax = softmax;

    // block_thread must be a power-of-2 in [1, 1024] — round up silently
    {
        int bt = 1;
        while (bt < block_thread) bt <<= 1;
        if (bt > 1024) bt = 1024;
        if (bt != block_thread) {
            Darknet::display_warning_msg("DCNv4 layer " + std::to_string(index) +
                ": block_thread=" + std::to_string(block_thread) +
                " is not a power-of-2 <= 1024, rounding to " + std::to_string(bt) + "\n");
            block_thread = bt;
        }
    }
    l.block_thread = block_thread;

    // groups must divide C — silent wrong behavior otherwise
    if (c % groups != 0) {
        darknet_fatal_error(DARKNET_LOC, "DCNv4 layer %d: groups=%d must evenly divide input channels=%d", index, groups, c);
    }

    // DCNv4 is a spatial aggregation operator (like depthwise conv). It cannot
    // change the channel dimension — use a separate 1x1 conv for that.
    if (n != c) {
        Darknet::display_warning_msg("DCNv4 layer " + std::to_string(index) +
            ": filters=" + std::to_string(n) + " != input channels=" + std::to_string(c) +
            ". Forcing filters=" + std::to_string(c) + ". Use a 1x1 conv to change channels.\n");
        n = c;
        l.n = n;
    }

    l.out_w = (w + 2 * padding - dilation * (size - 1) - 1) / stride_x + 1;
    l.out_h = (h + 2 * padding - dilation * (size - 1) - 1) / stride_y + 1;
    if (l.out_w <= 0 || l.out_h <= 0) {
        darknet_fatal_error(DARKNET_LOC,
            "DCNv4 layer %d: invalid output size %dx%d from input=%dx%d, size=%d, stride=%dx%d, dilation=%d, padding=%d",
            index, l.out_w, l.out_h, w, h, size, stride_x, stride_y, dilation, padding);
    }
    l.out_c = n;
    l.outputs = l.out_h * l.out_w * l.out_c;
    l.inputs = l.w * l.h * l.c;
    l.activation = activation;

    l.workspace_size = get_dcnv4_workspace_size(l);

    // NOTE: DCNv4 per the paper has no separate conv weights (spatial aggregation
    // uses attention masks). These weights are allocated for weight file compatibility
    // but are not used in forward/backward/update.
    l.nweights = (c / groups) * n * size * size;
    l.weights = (float*)xcalloc(l.nweights, sizeof(float));
    l.weight_updates = (float*)xcalloc(l.nweights, sizeof(float));

    l.biases = (float*)xcalloc(n, sizeof(float));
    l.bias_updates = (float*)xcalloc(n, sizeof(float));

    l.output = (float*)xcalloc(static_cast<size_t>(total_batch) * l.outputs, sizeof(float));
    l.delta = (float*)xcalloc(static_cast<size_t>(total_batch) * l.outputs, sizeof(float));
    if (dcnv4_needs_activation_input(l.activation)) {
        l.activation_input = (float*)xcalloc(static_cast<size_t>(total_batch) * l.outputs, sizeof(float));
    }

    if (batch_normalize) {
        l.scales = (float*)xcalloc(n, sizeof(float));
        l.scale_updates = (float*)xcalloc(n, sizeof(float));
        for (int i = 0; i < n; ++i) l.scales[i] = 1;

        l.mean = (float*)xcalloc(n, sizeof(float));
        l.variance = (float*)xcalloc(n, sizeof(float));
        l.mean_delta = (float*)xcalloc(n, sizeof(float));
        l.variance_delta = (float*)xcalloc(n, sizeof(float));

        l.rolling_mean = (float*)xcalloc(n, sizeof(float));
        l.rolling_variance = (float*)xcalloc(n, sizeof(float));
        l.x = (float*)xcalloc(static_cast<size_t>(total_batch) * l.outputs, sizeof(float));
        l.x_norm = (float*)xcalloc(static_cast<size_t>(total_batch) * l.outputs, sizeof(float));
    }

    // DCNv4 p_offset: group * size * size * 3 (2 offsets + 1 mask)
    // We need to ensure padded_offset_dim % 8 == 0 for DCNv4 CUDA kernel
    const int K = dcnv4_kernel_points(l);
    const int offset_filters_raw = groups * K * 3;
    const int padded_offset_dim = ((offset_filters_raw + 7) / 8) * 8;

    // We'll reuse offset_weights etc. for p_offset
    const size_t offset_weights_size = static_cast<size_t>(padded_offset_dim) * c * size * size;
    l.offset_weights = (float*)xcalloc(offset_weights_size, sizeof(float));
    l.offset_weight_updates = (float*)xcalloc(offset_weights_size, sizeof(float));
    l.offset_biases = (float*)xcalloc(padded_offset_dim, sizeof(float));
    l.offset_bias_updates = (float*)xcalloc(padded_offset_dim, sizeof(float));

    // Initialize mask (attention weight) biases to 1.0 so the layer starts as
    // a standard local aggregation. Without this, non-softmax DCNv4 produces zero
    // output from zero-initialized masks, killing all gradient flow.
    // Layout per group: [2*K offsets, K masks]. Initialize the mask portion.
    for (int g = 0; g < groups; ++g) {
        for (int k = 0; k < K; ++k) {
            l.offset_biases[g * K * 3 + K * 2 + k] = softmax ? 0.0f : 1.0f;
        }
    }

    l.offsets = (float*)xcalloc(static_cast<size_t>(total_batch) * l.out_h * l.out_w * padded_offset_dim, sizeof(float));
    l.offset_deltas = (float*)xcalloc(static_cast<size_t>(total_batch) * l.out_h * l.out_w * padded_offset_dim, sizeof(float));

    l.forward = forward_dcnv4_layer;
    l.backward = backward_dcnv4_layer;
    l.update = update_dcnv4_layer;

#ifdef DARKNET_GPU
    l.forward_gpu = forward_dcnv4_layer_gpu;
    l.backward_gpu = backward_dcnv4_layer_gpu;
    l.update_gpu = update_dcnv4_layer_gpu;

    if (cfg_and_state.gpu_index >= 0) {
        l.weights_gpu = cuda_make_array(l.weights, l.nweights);
        l.weight_updates_gpu = cuda_make_array(l.weight_updates, l.nweights);
        l.biases_gpu = cuda_make_array(l.biases, n);
        l.bias_updates_gpu = cuda_make_array(l.bias_updates, n);
        l.output_gpu = cuda_make_array(l.output, static_cast<size_t>(total_batch) * l.outputs);
        l.delta_gpu = cuda_make_array(l.delta, static_cast<size_t>(total_batch) * l.outputs);
        if (l.activation_input) {
            l.activation_input_gpu = cuda_make_array(l.activation_input, static_cast<size_t>(total_batch) * l.outputs);
        }

        if (batch_normalize) {
            l.scales_gpu = cuda_make_array(l.scales, n);
            l.scale_updates_gpu = cuda_make_array(l.scale_updates, n);

            l.mean_gpu = cuda_make_array(l.mean, n);
            l.variance_gpu = cuda_make_array(l.variance, n);

            l.rolling_mean_gpu = cuda_make_array(l.rolling_mean, n);
            l.rolling_variance_gpu = cuda_make_array(l.rolling_variance, n);

            l.mean_delta_gpu = cuda_make_array(l.mean_delta, n);
            l.variance_delta_gpu = cuda_make_array(l.variance_delta, n);

            l.x_gpu = cuda_make_array(l.output, static_cast<size_t>(total_batch) * l.outputs);
            l.x_norm_gpu = cuda_make_array(l.output, static_cast<size_t>(total_batch) * l.outputs);

#ifdef CUDNN
            if (l.normTensorDesc == nullptr) {
                CHECK_CUDNN(cudnnCreateTensorDescriptor(&l.normTensorDesc));
            }
            if (l.normDstTensorDesc == nullptr) {
                CHECK_CUDNN(cudnnCreateTensorDescriptor(&l.normDstTensorDesc));
            }
            if (l.normDstTensorDescF16 == nullptr) {
                CHECK_CUDNN(cudnnCreateTensorDescriptor(&l.normDstTensorDescF16));
            }
            CHECK_CUDNN(cudnnSetTensor4dDescriptor(l.normTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, l.out_c, 1, 1));
            CHECK_CUDNN(cudnnSetTensor4dDescriptor(l.normDstTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, l.batch, l.out_c, l.out_h, l.out_w));
            CHECK_CUDNN(cudnnSetTensor4dDescriptor(l.normDstTensorDescF16, CUDNN_TENSOR_NCHW, CUDNN_DATA_HALF, l.batch, l.out_c, l.out_h, l.out_w));
#endif
        }

        l.offset_weights_gpu = cuda_make_array(l.offset_weights, offset_weights_size);
        l.offset_weight_updates_gpu = cuda_make_array(l.offset_weight_updates, offset_weights_size);
#ifdef CUDNN_HALF
        l.offset_weights_gpu16 = cuda_make_array(nullptr, offset_weights_size / 2 + 1);
        l.offset_weight_updates_gpu16 = cuda_make_array(nullptr, offset_weights_size / 2 + 1);
#endif
        l.offset_biases_gpu = cuda_make_array(l.offset_biases, padded_offset_dim);
        l.offset_bias_updates_gpu = cuda_make_array(l.offset_bias_updates, padded_offset_dim);
        l.offsets_gpu = cuda_make_array(l.offsets, static_cast<size_t>(total_batch) * l.out_h * l.out_w * padded_offset_dim);
        l.offset_deltas_gpu = cuda_make_array(l.offset_deltas, static_cast<size_t>(total_batch) * l.out_h * l.out_w * padded_offset_dim);

#ifdef CUDNN
        dcnv4_setup_cudnn_offset_descriptors(l);
#endif
    }
#endif

    *cfg_and_state.output << "dcnv4 " << size << " x " << size << " / " << stride_x << " x " << stride_y << ", " << n << " filters" << std::endl;

    return l;
}

void forward_dcnv4_layer(Darknet::Layer & l, Darknet::NetworkState state)
{
    TAT(TATPARMS);

    const size_t workspace_floats = (get_dcnv4_workspace_size(l) + sizeof(float) - 1) / sizeof(float);
    std::vector<float> local_workspace;
    float * workspace = state.workspace;
    if (workspace == nullptr)
    {
        local_workspace.resize(workspace_floats);
        workspace = local_workspace.data();
    }

    std::memset(l.output, 0, static_cast<size_t>(l.batch) * l.outputs * sizeof(float));

    dcnv4_predict_offsets_cpu(l, state, workspace);
    dcnv4_deformable_forward_cpu(l, state);

    if (l.batch_normalize) {
        forward_batchnorm_layer(l, state);
    } else {
        add_bias(l.output, l.biases, l.batch, l.n, l.out_h * l.out_w);
    }

    dcnv4_activate_cpu(l);
}

void backward_dcnv4_layer(Darknet::Layer & l, Darknet::NetworkState state)
{
    (void)l;
    (void)state;
    Darknet::display_warning_msg("backward_dcnv4_layer: CPU training path is not implemented; use GPU for DCNv4 training\n");
}

void update_dcnv4_layer(Darknet::Layer & l, int batch, float learning_rate, float momentum, float decay)
{
    TAT(TATPARMS);

    const float rate = (learning_rate * l.learning_rate_scale) / batch;
    const float offset_rate = rate * 0.01f;
    const int padded_offset_dim = dcnv4_padded_offset_dim(l);
    const int offset_nweights = l.c * padded_offset_dim * l.size * l.size;

    axpy_cpu(offset_nweights, -decay * batch, l.offset_weights, 1, l.offset_weight_updates, 1);
    axpy_cpu(offset_nweights, offset_rate, l.offset_weight_updates, 1, l.offset_weights, 1);
    scal_cpu(offset_nweights, momentum, l.offset_weight_updates, 1);

    axpy_cpu(padded_offset_dim, offset_rate, l.offset_bias_updates, 1, l.offset_biases, 1);
    scal_cpu(padded_offset_dim, momentum, l.offset_bias_updates, 1);

    if (l.scales) {
        axpy_cpu(l.n, rate, l.scale_updates, 1, l.scales, 1);
        scal_cpu(l.n, momentum, l.scale_updates, 1);
    } else {
        axpy_cpu(l.n, rate, l.bias_updates, 1, l.biases, 1);
        scal_cpu(l.n, momentum, l.bias_updates, 1);
    }
}

void resize_dcnv4_layer(Darknet::Layer * l, int w, int h)
{
    if (l == nullptr) return;

    l->h = h;
    l->w = w;
    l->out_w = (w + 2 * l->pad - l->dilation * (l->size - 1) - 1) / l->stride_x + 1;
    l->out_h = (h + 2 * l->pad - l->dilation * (l->size - 1) - 1) / l->stride_y + 1;
    if (l->out_w <= 0 || l->out_h <= 0) {
        darknet_fatal_error(DARKNET_LOC,
            "DCNv4 layer %d resize produced invalid output size %dx%d from input=%dx%d, size=%d, stride=%dx%d, dilation=%d, padding=%d",
            l->index, l->out_w, l->out_h, w, h, l->size, l->stride_x, l->stride_y, l->dilation, l->pad);
    }
    l->out_c = l->n;
    l->outputs = l->out_h * l->out_w * l->out_c;
    l->inputs = l->w * l->h * l->c;
    l->workspace_size = get_dcnv4_workspace_size(*l);

    const int padded_offset_dim = dcnv4_padded_offset_dim(*l);
    const int total_batch = l->batch * std::max(1, l->steps);
    const size_t output_count = static_cast<size_t>(total_batch) * l->outputs;
    const size_t offset_count = static_cast<size_t>(total_batch) * l->out_h * l->out_w * padded_offset_dim;

    l->output = (float*)xrealloc(l->output, output_count * sizeof(float));
    l->delta = (float*)xrealloc(l->delta, output_count * sizeof(float));
    l->offsets = (float*)xrealloc(l->offsets, offset_count * sizeof(float));
    l->offset_deltas = (float*)xrealloc(l->offset_deltas, offset_count * sizeof(float));

    if (dcnv4_needs_activation_input(l->activation)) {
        l->activation_input = (float*)xrealloc(l->activation_input, output_count * sizeof(float));
    }

    if (l->batch_normalize) {
        l->x = (float*)xrealloc(l->x, output_count * sizeof(float));
        l->x_norm = (float*)xrealloc(l->x_norm, output_count * sizeof(float));
    }

#ifdef DARKNET_GPU
    if (cfg_and_state.gpu_index >= 0) {
        if (l->output_gpu) cuda_free(l->output_gpu);
        if (l->delta_gpu) cuda_free(l->delta_gpu);
        if (l->offsets_gpu) cuda_free(l->offsets_gpu);
        if (l->offset_deltas_gpu) cuda_free(l->offset_deltas_gpu);
        l->output_gpu = cuda_make_array(l->output, output_count);
        l->delta_gpu = cuda_make_array(l->delta, output_count);
        l->offsets_gpu = cuda_make_array(l->offsets, offset_count);
        l->offset_deltas_gpu = cuda_make_array(l->offset_deltas, offset_count);

        if (dcnv4_needs_activation_input(l->activation)) {
            if (l->activation_input_gpu) cuda_free(l->activation_input_gpu);
            l->activation_input_gpu = cuda_make_array(l->activation_input, output_count);
        }

        if (l->batch_normalize) {
            if (l->x_gpu) cuda_free(l->x_gpu);
            if (l->x_norm_gpu) cuda_free(l->x_norm_gpu);
            l->x_gpu = cuda_make_array(l->x, output_count);
            l->x_norm_gpu = cuda_make_array(l->x_norm, output_count);
#ifdef CUDNN
            if (l->normTensorDesc == nullptr) CHECK_CUDNN(cudnnCreateTensorDescriptor(&l->normTensorDesc));
            if (l->normDstTensorDesc == nullptr) CHECK_CUDNN(cudnnCreateTensorDescriptor(&l->normDstTensorDesc));
            if (l->normDstTensorDescF16 == nullptr) CHECK_CUDNN(cudnnCreateTensorDescriptor(&l->normDstTensorDescF16));
            CHECK_CUDNN(cudnnSetTensor4dDescriptor(l->normTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, l->out_c, 1, 1));
            CHECK_CUDNN(cudnnSetTensor4dDescriptor(l->normDstTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, l->batch, l->out_c, l->out_h, l->out_w));
            CHECK_CUDNN(cudnnSetTensor4dDescriptor(l->normDstTensorDescF16, CUDNN_TENSOR_NCHW, CUDNN_DATA_HALF, l->batch, l->out_c, l->out_h, l->out_w));
#endif
        }
#ifdef CUDNN
        dcnv4_setup_cudnn_offset_descriptors(*l);
#endif
    }
#endif
}
