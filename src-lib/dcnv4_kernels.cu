#include "darknet_internal.hpp"
#include "dcnv4_layer.hpp"
#include "blas.hpp"
#include "activations.hpp"
#include "convolutional_layer.hpp"
#include "im2col.hpp"
#include "col2im.hpp"
#include "gemm.hpp"
#include "permute_kernels.hpp"

#include <algorithm>

#define CUDA_KERNEL_LOOP(i, n)                                                 \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n);                 \
       i += blockDim.x * gridDim.x)

typedef float opmath_t;

__global__ void nhwc_bilinear_upsample_kernel(
    const float *__restrict__ src, float *__restrict__ dst,
    int B, int C, int H_src, int W_src, int H_dst, int W_dst,
    int stride_h, int stride_w);

__global__ void nhwc_bilinear_upsample_backward_kernel(
    const float *__restrict__ d_dst, float *__restrict__ d_src,
    int B, int C, int H_src, int W_src, int H_dst, int W_dst,
    int stride_h, int stride_w);

namespace
{
    static bool dcnv4_use_tensor_op(const Darknet::NetworkState &state)
    {
#ifdef DARKNET_GPU_CUDA
        return state.net.cudnn_half || state.net.cudnn_bf16;
#else
        (void)state;
        return false;
#endif
    }

#if defined(CUDNN) && defined(CUDNN_HALF)
    static int dcnv4_cudnn_16bit_mode(const Darknet::NetworkState & state)
    {
        return state.net.cudnn_bf16 ? DARKNET_CUDNN_16BIT_BF16 : DARKNET_CUDNN_16BIT_HALF;
    }

    static void dcnv4_ensure_16bit_workspace(float ** ptr, size_t * max_size, const size_t required_elements)
    {
        if (ptr == nullptr || max_size == nullptr)
        {
            darknet_fatal_error(DARKNET_LOC, "DCNv4 cuDNN 16-bit workspace is not initialized");
        }
        if (*max_size < required_elements)
        {
            *max_size = required_elements;
            if (*ptr) cuda_free(*ptr);
            CHECK_CUDA(cudaMalloc((void **)ptr, required_elements * sizeof(short)));
        }
    }
#endif

#ifdef CUDNN
    static bool dcnv4_has_cudnn_offset_descriptors(const Darknet::Layer & l)
    {
        return l.srcTensorDesc != nullptr && l.dstTensorDesc != nullptr &&
               l.dsrcTensorDesc != nullptr && l.ddstTensorDesc != nullptr &&
               l.weightDesc != nullptr && l.dweightDesc != nullptr &&
               l.convDesc != nullptr;
    }

    static bool dcnv4_use_cudnn_16bit_offset(const Darknet::Layer & l, const Darknet::NetworkState & state, const int padded_offset_dim)
    {
#ifdef CUDNN_HALF
        const bool requested = state.net.cudnn_half || state.net.cudnn_bf16;
        return requested && l.offset_weights_gpu16 != nullptr && l.offset_weight_updates_gpu16 != nullptr &&
               l.srcTensorDesc16 != nullptr && l.dstTensorDesc16 != nullptr &&
               l.dsrcTensorDesc16 != nullptr && l.ddstTensorDesc16 != nullptr &&
               l.weightDesc16 != nullptr && l.dweightDesc16 != nullptr &&
               padded_offset_dim % 8 == 0 && l.c % 8 == 0;
#else
        (void)l;
        (void)state;
        (void)padded_offset_dim;
        return false;
#endif
    }

    static bool dcnv4_predict_offsets_cudnn_gpu(
        Darknet::Layer & l,
        Darknet::NetworkState state,
        float * scratch,
        const int batch,
        const int padded_offset_dim,
        const int out_h,
        const int out_w,
        const int threads)
    {
        if (!dcnv4_has_cudnn_offset_descriptors(l))
        {
            return false;
        }

        const int d_stride = std::max(1, l.d_stride);
        const int H_c = (out_h + d_stride - 1) / d_stride;
        const int W_c = (out_w + d_stride - 1) / d_stride;
        const int spatial_source = H_c * W_c;
        const size_t source_count = (size_t)batch * padded_offset_dim * spatial_source;
        float *offset_nchw = scratch;
        float *coarse_offsets_nhwc = offset_nchw + source_count;
        const float alpha = 1.0f;
        const float beta = 0.0f;

        const bool use_16bit = dcnv4_use_cudnn_16bit_offset(l, state, padded_offset_dim);
        if (use_16bit)
        {
#ifdef CUDNN_HALF
            const int mode = dcnv4_cudnn_16bit_mode(state);
            if (l.cudnn_16bit_mode != mode)
            {
                set_dcnv4_cudnn_16bit_mode(&l, mode);
            }
            const size_t input16_size = (size_t)batch * l.c * l.h * l.w;
            dcnv4_ensure_16bit_workspace(state.net.input16_gpu, state.net.max_input16_size, std::max(input16_size, source_count));
            dcnv4_ensure_16bit_workspace(state.net.output16_gpu, state.net.max_output16_size, source_count);
            float *input16 = *state.net.input16_gpu;
            float *output16 = *state.net.output16_gpu;
            const size_t offset_weights_size = (size_t)padded_offset_dim * l.c * l.size * l.size;
            cuda_convert_f32_to_cudnn_16bit(state.input, input16_size, input16, mode);
            cuda_convert_f32_to_cudnn_16bit(l.offset_weights_gpu, offset_weights_size, l.offset_weights_gpu16, mode);
            const cudnnStatus_t status = cudnnConvolutionForward(cudnn_handle(),
                &alpha,
                l.srcTensorDesc16,
                input16,
                l.weightDesc16,
                l.offset_weights_gpu16,
                l.convDesc,
                l.fw_algo16,
                nullptr,
                0,
                &beta,
                l.dstTensorDesc16,
                output16);
            if (status != CUDNN_STATUS_SUCCESS)
            {
                return false;
            }
            cuda_convert_cudnn_16bit_to_f32(output16, source_count, offset_nchw, mode);
#endif
        }
        else
        {
            const cudnnStatus_t status = cudnnConvolutionForward(cudnn_handle(),
                &alpha,
                l.srcTensorDesc,
                state.input,
                l.weightDesc,
                l.offset_weights_gpu,
                l.convDesc,
                l.fw_algo,
                nullptr,
                0,
                &beta,
                l.dstTensorDesc,
                offset_nchw);
            if (status != CUDNN_STATUS_SUCCESS)
            {
                return false;
            }
        }

        add_bias_gpu(offset_nchw, l.offset_biases_gpu, batch, padded_offset_dim, spatial_source);
        if (d_stride > 1)
        {
            nchw_to_nhwc_gpu(offset_nchw, coarse_offsets_nhwc, batch, padded_offset_dim, H_c, W_c);
            const int total_up = batch * out_h * out_w * padded_offset_dim;
            nhwc_bilinear_upsample_kernel<<<(total_up + threads - 1) / threads, threads, 0, get_cuda_stream()>>>(
                coarse_offsets_nhwc, l.offsets_gpu, batch, padded_offset_dim, H_c, W_c, out_h, out_w,
                d_stride, d_stride);
            CHECK_CUDA(cudaPeekAtLastError());
        }
        else
        {
            nchw_to_nhwc_gpu(offset_nchw, l.offsets_gpu, batch, padded_offset_dim, out_h, out_w);
        }
        return true;
    }

    static bool dcnv4_backprop_offsets_cudnn_gpu(
        Darknet::Layer & l,
        Darknet::NetworkState state,
        const float * grad_offset_nhwc,
        float * scratch,
        float * grad_im_nchw,
        const int batch,
        const int padded_offset_dim,
        const int out_h,
        const int out_w,
        const int threads)
    {
        if (!dcnv4_has_cudnn_offset_descriptors(l))
        {
            return false;
        }

        const int d_stride = std::max(1, l.d_stride);
        const int H_c = (out_h + d_stride - 1) / d_stride;
        const int W_c = (out_w + d_stride - 1) / d_stride;
        const int spatial_source = H_c * W_c;
        const size_t source_count = (size_t)batch * spatial_source * padded_offset_dim;

        float *grad_source_nhwc = scratch;
        float *grad_source_nchw = grad_source_nhwc + source_count;
        if (d_stride > 1)
        {
            cudaMemsetAsync(grad_source_nhwc, 0, source_count * sizeof(float), get_cuda_stream());
            const int total_down = batch * out_h * out_w * padded_offset_dim;
            nhwc_bilinear_upsample_backward_kernel<<<(total_down + threads - 1) / threads, threads, 0, get_cuda_stream()>>>(
                grad_offset_nhwc, grad_source_nhwc,
                batch, padded_offset_dim, H_c, W_c, out_h, out_w,
                d_stride, d_stride);
            CHECK_CUDA(cudaPeekAtLastError());
            nhwc_to_nchw_gpu(grad_source_nhwc, grad_source_nchw, batch, padded_offset_dim, H_c, W_c);
        }
        else
        {
            grad_source_nchw = scratch;
            nhwc_to_nchw_gpu(const_cast<float *>(grad_offset_nhwc), grad_source_nchw, batch, padded_offset_dim, out_h, out_w);
        }

        const float one = 1.0f;
        const float zero = 0.0f;
        const bool use_16bit = dcnv4_use_cudnn_16bit_offset(l, state, padded_offset_dim);
        if (use_16bit)
        {
#ifdef CUDNN_HALF
            const int mode = dcnv4_cudnn_16bit_mode(state);
            if (l.cudnn_16bit_mode != mode)
            {
                set_dcnv4_cudnn_16bit_mode(&l, mode);
            }
            const size_t input16_size = (size_t)batch * l.c * l.h * l.w;
            const size_t input16_workspace_size = state.delta ? input16_size * 2 : input16_size;
            dcnv4_ensure_16bit_workspace(state.net.input16_gpu, state.net.max_input16_size, input16_workspace_size);
            dcnv4_ensure_16bit_workspace(state.net.output16_gpu, state.net.max_output16_size, source_count);
            char *input16_bytes = reinterpret_cast<char *>(*state.net.input16_gpu);
            float *input16 = reinterpret_cast<float *>(input16_bytes);
            float *grad_input16 = reinterpret_cast<float *>(input16_bytes + input16_size * sizeof(short));
            float *grad16 = *state.net.output16_gpu;
            const size_t offset_weights_size = (size_t)padded_offset_dim * l.c * l.size * l.size;
            cuda_convert_f32_to_cudnn_16bit(state.input, input16_size, input16, mode);
            cuda_convert_f32_to_cudnn_16bit(grad_source_nchw, source_count, grad16, mode);
            cuda_convert_f32_to_cudnn_16bit(l.offset_weights_gpu, offset_weights_size, l.offset_weights_gpu16, mode);
            cuda_convert_f32_to_cudnn_16bit(l.offset_weight_updates_gpu, offset_weights_size, l.offset_weight_updates_gpu16, mode);

            if (state.delta)
            {
                cudnnStatus_t status = cudnnConvolutionBackwardData(cudnn_handle(),
                    &one,
                    l.weightDesc16,
                    l.offset_weights_gpu16,
                    l.ddstTensorDesc16,
                    grad16,
                    l.convDesc,
                    l.bd_algo16,
                    nullptr,
                    0,
                    &zero,
                    l.dsrcTensorDesc16,
                    grad_input16);
                if (status != CUDNN_STATUS_SUCCESS)
                {
                    return false;
                }
            }

            cudnnStatus_t status = cudnnConvolutionBackwardFilter(cudnn_handle(),
                &one,
                l.srcTensorDesc16,
                input16,
                l.ddstTensorDesc16,
                grad16,
                l.convDesc,
                l.bf_algo16,
                nullptr,
                0,
                &one,
                l.dweightDesc16,
                l.offset_weight_updates_gpu16);
            if (status != CUDNN_STATUS_SUCCESS)
            {
                return false;
            }
            cuda_convert_cudnn_16bit_to_f32(l.offset_weight_updates_gpu16, offset_weights_size, l.offset_weight_updates_gpu, mode);
            backward_bias_gpu(l.offset_bias_updates_gpu, grad_source_nchw, batch, padded_offset_dim, spatial_source);

            if (state.delta)
            {
                cuda_convert_cudnn_16bit_to_f32(grad_input16, input16_size, grad_im_nchw, mode);
                axpy_ongpu((int)input16_size, 1.0f, grad_im_nchw, 1, state.delta, 1);
            }
            return true;
#endif
        }

        if (state.delta)
        {
            cudnnStatus_t status = cudnnConvolutionBackwardData(cudnn_handle(),
                &one,
                l.weightDesc,
                l.offset_weights_gpu,
                l.ddstTensorDesc,
                grad_source_nchw,
                l.convDesc,
                l.bd_algo,
                nullptr,
                0,
                &zero,
                l.dsrcTensorDesc,
                grad_im_nchw);
            if (status != CUDNN_STATUS_SUCCESS)
            {
                return false;
            }
        }

        cudnnStatus_t status = cudnnConvolutionBackwardFilter(cudnn_handle(),
            &one,
            l.srcTensorDesc,
            state.input,
            l.ddstTensorDesc,
            grad_source_nchw,
            l.convDesc,
            l.bf_algo,
            nullptr,
            0,
            &one,
            l.dweightDesc,
            l.offset_weight_updates_gpu);
        if (status != CUDNN_STATUS_SUCCESS)
        {
            return false;
        }
        backward_bias_gpu(l.offset_bias_updates_gpu, grad_source_nchw, batch, padded_offset_dim, spatial_source);

        if (state.delta)
        {
            axpy_ongpu((int)((size_t)batch * l.c * l.h * l.w), 1.0f, grad_im_nchw, 1, state.delta, 1);
        }
        return true;
    }
#endif
}

__global__ void add_bias_nhwc_kernel(float *__restrict__ output, const float *__restrict__ biases, int batch, int spatial, int channels)
{
    const int total = batch * spatial * channels;
    for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < total; index += blockDim.x * gridDim.x) {
        output[index] += biases[index % channels];
    }
}

// Bilinear upsample on the coarse output lattice: NHWC [B, H_src, W_src, C] -> [B, H_dst, W_dst, C]
// Coarse offsets are predicted with im2col stride=(layer_stride * d_stride), so destination
// output coordinate (h_d, w_d) maps to source coordinate (h_d / d_stride, w_d / d_stride).
// Border clamping is used for the last partial interval when H_dst/W_dst is not divisible by d_stride.
__global__ void nhwc_bilinear_upsample_kernel(
    const float *__restrict__ src, float *__restrict__ dst,
    int B, int C, int H_src, int W_src, int H_dst, int W_dst,
    int stride_h, int stride_w)
{
    const int total = B * H_dst * W_dst * C;
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total; idx += blockDim.x * gridDim.x) {
        const int c   = idx % C;
        const int w_d = (idx / C) % W_dst;
        const int h_d = (idx / C / W_dst) % H_dst;
        const int b   = idx / C / W_dst / H_dst;

        const float h_s = (float)h_d / (float)max(1, stride_h);
        const float w_s = (float)w_d / (float)max(1, stride_w);

        const int h0_raw = (int)floorf(h_s);
        const int w0_raw = (int)floorf(w_s);
        const int h1_raw = h0_raw + 1;
        const int w1_raw = w0_raw + 1;

        const float lh = h_s - (float)h0_raw;
        const float lw = w_s - (float)w0_raw;
        const float hh = 1.0f - lh;
        const float hw = 1.0f - lw;

        const int h0 = min(max(h0_raw, 0), H_src - 1);
        const int h1 = min(max(h1_raw, 0), H_src - 1);
        const int w0 = min(max(w0_raw, 0), W_src - 1);
        const int w1 = min(max(w1_raw, 0), W_src - 1);

        const float *src_b = src + ((b * H_src) * W_src) * C;
        dst[idx] =
            hh * hw * src_b[((h0 * W_src + w0) * C) + c] +
            hh * lw * src_b[((h0 * W_src + w1) * C) + c] +
            lh * hw * src_b[((h1 * W_src + w0) * C) + c] +
            lh * lw * src_b[((h1 * W_src + w1) * C) + c];
    }
}

// Backward through lattice-aligned bilinear upsample: scatter gradients from
// [B, H_dst, W_dst, C] -> [B, H_src, W_src, C].
__global__ void nhwc_bilinear_upsample_backward_kernel(
    const float *__restrict__ d_dst, float *__restrict__ d_src,
    int B, int C, int H_src, int W_src, int H_dst, int W_dst,
    int stride_h, int stride_w)
{
    const int total = B * H_dst * W_dst * C;
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total; idx += blockDim.x * gridDim.x) {
        const int c   = idx % C;
        const int w_d = (idx / C) % W_dst;
        const int h_d = (idx / C / W_dst) % H_dst;
        const int b   = idx / C / W_dst / H_dst;

        const float h_s = (float)h_d / (float)max(1, stride_h);
        const float w_s = (float)w_d / (float)max(1, stride_w);

        const int h0_raw = (int)floorf(h_s);
        const int w0_raw = (int)floorf(w_s);
        const int h1_raw = h0_raw + 1;
        const int w1_raw = w0_raw + 1;

        const float lh = h_s - (float)h0_raw;
        const float lw = w_s - (float)w0_raw;
        const float hh = 1.0f - lh;
        const float hw = 1.0f - lw;

        const int h0 = min(max(h0_raw, 0), H_src - 1);
        const int h1 = min(max(h1_raw, 0), H_src - 1);
        const int w0 = min(max(w0_raw, 0), W_src - 1);
        const int w1 = min(max(w1_raw, 0), W_src - 1);

        float *d_src_b = d_src + ((b * H_src) * W_src) * C;
        const float g = d_dst[idx];
        atomicAdd(&d_src_b[((h0 * W_src + w0) * C) + c], hh * hw * g);
        atomicAdd(&d_src_b[((h0 * W_src + w1) * C) + c], hh * lw * g);
        atomicAdd(&d_src_b[((h1 * W_src + w0) * C) + c], lh * hw * g);
        atomicAdd(&d_src_b[((h1 * W_src + w1) * C) + c], lh * lw * g);
    }
}

// Bilinear sampling for c_per_thread adjacent channels at position (h_px, w_px).
// NHWC layout: channels at the same (h, w) are contiguous → base_ptr+0..c_per_thread-1
// are stride-1 accesses, enabling coalesced loads when c_per_thread>1.
// Boundary conditions are the same for all channels → hoisted outside the inner loop.
template <typename scalar_t, typename transfer_t, int c_per_thread>
__device__ __forceinline__ void dcnv4_im2col_bilinear_gpu(
    opmath_t out_reg_array[], const scalar_t *__restrict__ p_value, const int height,
    const int width, const opmath_t h_px, const opmath_t w_px,
    const opmath_t attn, const int w_stride, const int base_ptr) {

  const int h_low = floorf(h_px);
  const int w_low = floorf(w_px);
  const int h_high = h_low + 1;
  const int w_high = w_low + 1;

  // Zero-padding mode: if the entire bilinear footprint is outside the image, skip all channel work.
  if (h_high < 0 || h_low >= height || w_high < 0 || w_low >= width) return;

  const opmath_t lh = h_px - h_low;
  const opmath_t lw = w_px - w_low;
  const opmath_t hh = 1 - lh;
  const opmath_t hw = 1 - lw;

  const opmath_t w1 = hh * hw, w2 = hh * lw, w3 = lh * hw, w4 = lh * lw;

  const int h_stride = width * w_stride;
  const int h_low_ptr_offset  = h_low  * h_stride;
  const int h_high_ptr_offset = h_high * h_stride;
  const int w_low_ptr_offset  = w_low  * w_stride;
  const int w_high_ptr_offset = w_high * w_stride;

  const bool all_valid = (h_low >= 0 && h_high < height && w_low >= 0 && w_high < width);
  if (all_valid) {
#pragma unroll
    for (int c = 0; c < c_per_thread; c++) {
      const int ptr = base_ptr + c;
      out_reg_array[c] += attn * (
          w1 * (opmath_t)(p_value[h_low_ptr_offset  + w_low_ptr_offset  + ptr]) +
          w2 * (opmath_t)(p_value[h_low_ptr_offset  + w_high_ptr_offset + ptr]) +
          w3 * (opmath_t)(p_value[h_high_ptr_offset + w_low_ptr_offset  + ptr]) +
          w4 * (opmath_t)(p_value[h_high_ptr_offset + w_high_ptr_offset + ptr]));
    }
    return;
  }

  const bool valid_h_low  = (h_low  >= 0 && h_low  < height);
  const bool valid_h_high = (h_high >= 0 && h_high < height);
  const bool valid_w_low  = (w_low  >= 0 && w_low  < width);
  const bool valid_w_high = (w_high >= 0 && w_high < width);

#pragma unroll
  for (int c = 0; c < c_per_thread; c++) {
    const int ptr = base_ptr + c;
    if (valid_h_low  && valid_w_low)
      out_reg_array[c] += attn * w1 * (opmath_t)(p_value[h_low_ptr_offset  + w_low_ptr_offset  + ptr]);
    if (valid_h_low  && valid_w_high)
      out_reg_array[c] += attn * w2 * (opmath_t)(p_value[h_low_ptr_offset  + w_high_ptr_offset + ptr]);
    if (valid_h_high && valid_w_low)
      out_reg_array[c] += attn * w3 * (opmath_t)(p_value[h_high_ptr_offset + w_low_ptr_offset  + ptr]);
    if (valid_h_high && valid_w_high)
      out_reg_array[c] += attn * w4 * (opmath_t)(p_value[h_high_ptr_offset + w_high_ptr_offset + ptr]);
  }
}

template <typename scalar_t, int d_stride, typename transfer_t, int K, bool softmax,
          int kernel_h_c = 0, int kernel_w_c = 0,
          int stride_h_c = 0, int stride_w_c = 0,
          int dilation_h_c = 0, int dilation_w_c = 0,
          int remove_center_c = -1>
__global__ void dcnv4_forward_kernel_gpu(
    const scalar_t *__restrict__ p_value, const scalar_t *__restrict__ p_offset, scalar_t *__restrict__ p_output,
    const int B, const int G, const int D, const int Q, const int kernel_h,
    const int kernel_w, const int stride_h, const int stride_w, const int pad_h,
    const int pad_w, const int dilation_h, const int dilation_w,
    const int height_in, const int width_in, const int height_out,
    const int width_out, const opmath_t offset_scale, const int remove_center,
    const int block_multiplier, const int padded_offset_dim) {

  const int kernel_h_eff = kernel_h_c ? kernel_h_c : kernel_h;
  const int kernel_w_eff = kernel_w_c ? kernel_w_c : kernel_w;
  const int stride_h_eff = stride_h_c ? stride_h_c : stride_h;
  const int stride_w_eff = stride_w_c ? stride_w_c : stride_w;
  const int dilation_h_eff = dilation_h_c ? dilation_h_c : dilation_h;
  const int dilation_w_eff = dilation_w_c ? dilation_w_c : dilation_w;
  const bool remove_center_eff = (remove_center_c >= 0) ? (remove_center_c != 0) : (remove_center != 0);

  const int pixel_idx = blockIdx.x * block_multiplier + threadIdx.z;
  if (pixel_idx >= B * Q) return;

  const int bi = pixel_idx / Q;
  const int qi = pixel_idx % Q;

  const int di_s = threadIdx.x * d_stride;
  const int gi = threadIdx.y;

  opmath_t p_mask_shm[K];
  opmath_t p_out_shm[d_stride];
  for (int i=0; i < d_stride; ++i) p_out_shm[i] = 0.0f;

  const scalar_t *p_offset_ptr = p_offset + (bi * Q + qi) * padded_offset_dim + gi * K * 3;

  for (int i=0; i < K; i++){
    p_mask_shm[i] = (opmath_t)(*(p_offset_ptr + K * 2 + i));
  }

  if (softmax) {
      opmath_t softmax_max = -1e10f;
      opmath_t softmax_sum = 0.0f;
      for (int j = 0; j < K; j++) softmax_max = fmaxf(softmax_max, p_mask_shm[j]);
      for (int j = 0; j < K; j++) {
        opmath_t exp_results = expf(p_mask_shm[j] - softmax_max);
        p_mask_shm[j] = exp_results;
        softmax_sum += exp_results;
      }
      for (int j = 0; j < K; j++) p_mask_shm[j] /= (softmax_sum + 1e-6f);
  }

  const int w_stride = G * D;
  const int base_ptr = gi * D + di_s;
  const scalar_t *p_value_ptr = p_value + (bi * (height_in * width_in)) * (G * D);

  const int p0_w = ((dilation_w_eff * (kernel_w_eff - 1)) >> 1) - pad_w + (qi % width_out) * stride_w_eff;
  const int p0_h = ((dilation_h_eff * (kernel_h_eff - 1)) >> 1) - pad_h + (qi / width_out) * stride_h_eff;
  const opmath_t p0_w_ = (opmath_t)p0_w - ((dilation_w_eff * (kernel_w_eff - 1)) >> 1) * offset_scale;
  const opmath_t p0_h_ = (opmath_t)p0_h - ((dilation_h_eff * (kernel_h_eff - 1)) >> 1) * offset_scale;
  const int center_h = kernel_h_eff / 2;
  const int center_w = kernel_w_eff / 2;

  int offset_idx = 0;
  int mask_idx = 0;
  for (int i = 0; i < kernel_w_eff; ++i) {
    for (int j = 0; j < kernel_h_eff; ++j) {
      if (i != center_w || j != center_h || !remove_center_eff) {
        const opmath_t w_im = p0_w_ + (i * dilation_w_eff + (opmath_t)p_offset_ptr[offset_idx]) * offset_scale;
        const opmath_t h_im = p0_h_ + (j * dilation_h_eff + (opmath_t)p_offset_ptr[offset_idx + 1]) * offset_scale;
        const opmath_t attn = p_mask_shm[mask_idx];

        dcnv4_im2col_bilinear_gpu<scalar_t, transfer_t, d_stride>(
            p_out_shm, p_value_ptr, height_in, width_in, h_im, w_im, attn,
            w_stride, base_ptr);
        
        offset_idx += 2;
        mask_idx += 1;
      }
    }
  }

  int out_idx = ((bi * Q + qi) * G + gi) * D + di_s;
#pragma unroll
  for (int ds = 0; ds < d_stride; ds++) {
    p_output[out_idx + ds] = (scalar_t)p_out_shm[ds];
  }
}

template <int DS, bool SM, int KVAL,
          int kernel_h_c = 0, int kernel_w_c = 0,
          int stride_h_c = 0, int stride_w_c = 0,
          int dilation_h_c = 0, int dilation_w_c = 0,
          int remove_center_c = -1>
static void dcnv4_launch_forward_kernel(
    const Darknet::Layer & l,
    const float *input_nhwc,
    const float *offsets,
    float *output_nhwc,
    const int batch,
    const int G,
    const int D,
    const int Q,
    const int out_h,
    const int out_w,
    const int block_multiplier,
    const int padded_offset_dim,
    const dim3 num_blocks,
    const dim3 num_threads)
{
    dcnv4_forward_kernel_gpu<float, DS, float, KVAL, SM,
        kernel_h_c, kernel_w_c, stride_h_c, stride_w_c,
        dilation_h_c, dilation_w_c, remove_center_c>
        <<<num_blocks, num_threads, 0, get_cuda_stream()>>>(
            input_nhwc, offsets, output_nhwc,
            batch, G, D, Q,
            l.size, l.size,
            l.stride_y, l.stride_x,
            l.pad, l.pad,
            l.dilation, l.dilation,
            l.h, l.w,
            out_h, out_w,
            l.offset_scale,
            l.remove_center,
            block_multiplier,
            padded_offset_dim);
}

template <int DS, bool SM>
static bool dcnv4_try_launch_forward_static(
    const Darknet::Layer & l,
    const float *input_nhwc,
    const float *offsets,
    float *output_nhwc,
    const int batch,
    const int G,
    const int D,
    const int Q,
    const int out_h,
    const int out_w,
    const int block_multiplier,
    const int padded_offset_dim,
    const dim3 num_blocks,
    const dim3 num_threads)
{
    if (l.dilation != 1 || l.stride_x != l.stride_y)
    {
        return false;
    }

    const int stride = l.stride_x;
    if (l.size == 3)
    {
        if (stride == 1)
        {
            if (l.remove_center) dcnv4_launch_forward_kernel<DS, SM, 8, 3, 3, 1, 1, 1, 1, 1>(l, input_nhwc, offsets, output_nhwc, batch, G, D, Q, out_h, out_w, block_multiplier, padded_offset_dim, num_blocks, num_threads);
            else dcnv4_launch_forward_kernel<DS, SM, 9, 3, 3, 1, 1, 1, 1, 0>(l, input_nhwc, offsets, output_nhwc, batch, G, D, Q, out_h, out_w, block_multiplier, padded_offset_dim, num_blocks, num_threads);
            return true;
        }
        if (stride == 2)
        {
            if (l.remove_center) dcnv4_launch_forward_kernel<DS, SM, 8, 3, 3, 2, 2, 1, 1, 1>(l, input_nhwc, offsets, output_nhwc, batch, G, D, Q, out_h, out_w, block_multiplier, padded_offset_dim, num_blocks, num_threads);
            else dcnv4_launch_forward_kernel<DS, SM, 9, 3, 3, 2, 2, 1, 1, 0>(l, input_nhwc, offsets, output_nhwc, batch, G, D, Q, out_h, out_w, block_multiplier, padded_offset_dim, num_blocks, num_threads);
            return true;
        }
    }
    else if (l.size == 5)
    {
        if (stride == 1)
        {
            if (l.remove_center) dcnv4_launch_forward_kernel<DS, SM, 24, 5, 5, 1, 1, 1, 1, 1>(l, input_nhwc, offsets, output_nhwc, batch, G, D, Q, out_h, out_w, block_multiplier, padded_offset_dim, num_blocks, num_threads);
            else dcnv4_launch_forward_kernel<DS, SM, 25, 5, 5, 1, 1, 1, 1, 0>(l, input_nhwc, offsets, output_nhwc, batch, G, D, Q, out_h, out_w, block_multiplier, padded_offset_dim, num_blocks, num_threads);
            return true;
        }
        if (stride == 2)
        {
            if (l.remove_center) dcnv4_launch_forward_kernel<DS, SM, 24, 5, 5, 2, 2, 1, 1, 1>(l, input_nhwc, offsets, output_nhwc, batch, G, D, Q, out_h, out_w, block_multiplier, padded_offset_dim, num_blocks, num_threads);
            else dcnv4_launch_forward_kernel<DS, SM, 25, 5, 5, 2, 2, 1, 1, 0>(l, input_nhwc, offsets, output_nhwc, batch, G, D, Q, out_h, out_w, block_multiplier, padded_offset_dim, num_blocks, num_threads);
            return true;
        }
    }
    else if (l.size == 7)
    {
        if (stride == 1)
        {
            if (l.remove_center) dcnv4_launch_forward_kernel<DS, SM, 48, 7, 7, 1, 1, 1, 1, 1>(l, input_nhwc, offsets, output_nhwc, batch, G, D, Q, out_h, out_w, block_multiplier, padded_offset_dim, num_blocks, num_threads);
            else dcnv4_launch_forward_kernel<DS, SM, 49, 7, 7, 1, 1, 1, 1, 0>(l, input_nhwc, offsets, output_nhwc, batch, G, D, Q, out_h, out_w, block_multiplier, padded_offset_dim, num_blocks, num_threads);
            return true;
        }
        if (stride == 2)
        {
            if (l.remove_center) dcnv4_launch_forward_kernel<DS, SM, 48, 7, 7, 2, 2, 1, 1, 1>(l, input_nhwc, offsets, output_nhwc, batch, G, D, Q, out_h, out_w, block_multiplier, padded_offset_dim, num_blocks, num_threads);
            else dcnv4_launch_forward_kernel<DS, SM, 49, 7, 7, 2, 2, 1, 1, 0>(l, input_nhwc, offsets, output_nhwc, batch, G, D, Q, out_h, out_w, block_multiplier, padded_offset_dim, num_blocks, num_threads);
            return true;
        }
    }

    return false;
}

template <int DS, bool SM>
static void dcnv4_launch_forward_dispatch(
    const Darknet::Layer & l,
    const int K,
    const float *input_nhwc,
    const float *offsets,
    float *output_nhwc,
    const int batch,
    const int G,
    const int D,
    const int Q,
    const int out_h,
    const int out_w,
    const int block_multiplier,
    const int padded_offset_dim,
    const dim3 num_blocks,
    const dim3 num_threads)
{
    if (dcnv4_try_launch_forward_static<DS, SM>(l, input_nhwc, offsets, output_nhwc, batch, G, D, Q, out_h, out_w, block_multiplier, padded_offset_dim, num_blocks, num_threads))
    {
        return;
    }

    switch (K) {
        case  4: dcnv4_launch_forward_kernel<DS, SM,  4>(l, input_nhwc, offsets, output_nhwc, batch, G, D, Q, out_h, out_w, block_multiplier, padded_offset_dim, num_blocks, num_threads); break;
        case  8: dcnv4_launch_forward_kernel<DS, SM,  8>(l, input_nhwc, offsets, output_nhwc, batch, G, D, Q, out_h, out_w, block_multiplier, padded_offset_dim, num_blocks, num_threads); break;
        case  9: dcnv4_launch_forward_kernel<DS, SM,  9>(l, input_nhwc, offsets, output_nhwc, batch, G, D, Q, out_h, out_w, block_multiplier, padded_offset_dim, num_blocks, num_threads); break;
        case 15: dcnv4_launch_forward_kernel<DS, SM, 15>(l, input_nhwc, offsets, output_nhwc, batch, G, D, Q, out_h, out_w, block_multiplier, padded_offset_dim, num_blocks, num_threads); break;
        case 16: dcnv4_launch_forward_kernel<DS, SM, 16>(l, input_nhwc, offsets, output_nhwc, batch, G, D, Q, out_h, out_w, block_multiplier, padded_offset_dim, num_blocks, num_threads); break;
        case 24: dcnv4_launch_forward_kernel<DS, SM, 24>(l, input_nhwc, offsets, output_nhwc, batch, G, D, Q, out_h, out_w, block_multiplier, padded_offset_dim, num_blocks, num_threads); break;
        case 25: dcnv4_launch_forward_kernel<DS, SM, 25>(l, input_nhwc, offsets, output_nhwc, batch, G, D, Q, out_h, out_w, block_multiplier, padded_offset_dim, num_blocks, num_threads); break;
        case 48: dcnv4_launch_forward_kernel<DS, SM, 48>(l, input_nhwc, offsets, output_nhwc, batch, G, D, Q, out_h, out_w, block_multiplier, padded_offset_dim, num_blocks, num_threads); break;
        case 49: dcnv4_launch_forward_kernel<DS, SM, 49>(l, input_nhwc, offsets, output_nhwc, batch, G, D, Q, out_h, out_w, block_multiplier, padded_offset_dim, num_blocks, num_threads); break;
        default: Darknet::display_warning_msg("DCNv4: unsupported K=" + std::to_string(K) + ", skipping forward pass\n"); break;
    }
}

void forward_dcnv4_layer_gpu(Darknet::Layer & l, Darknet::NetworkState state)
{
    int K = l.size * l.size;
    if (l.remove_center) K -= 1;
    int offset_filters_raw = l.groups * K * 3;
    int padded_offset_dim = ((offset_filters_raw + 7) / 8) * 8;
    int out_h = l.out_h;
    int out_w = l.out_w;
    int spatial_out = out_h * out_w;
    int batch = l.batch;
    const bool use_tensor_op = dcnv4_use_tensor_op(state);

    // 2. Prepare layout-transformed buffers in workspace
    float *input_nhwc = state.workspace;
    size_t input_nhwc_size = (size_t)batch * l.c * l.h * l.w;
    float *output_nhwc = input_nhwc + input_nhwc_size;
    size_t output_nhwc_size = (size_t)batch * l.n * spatial_out;
    float *scratch = output_nhwc + output_nhwc_size;
    
    // 1. Predict p_offset
    // When l.d_stride > 1: predict at coarse spatial resolution [B, H_c, W_c, padded_offset_dim]
    // then bilinearly upsample to full [B, out_h, out_w, padded_offset_dim].
    // When l.d_stride == 1: predict directly at full resolution.
    int threads = 512;
    int K_gemm = l.c * l.size * l.size;

    bool offsets_predicted = false;
#ifdef CUDNN
    offsets_predicted = dcnv4_predict_offsets_cudnn_gpu(l, state, scratch, batch, padded_offset_dim, out_h, out_w, threads);
#endif
    if (!offsets_predicted) {
        if (l.d_stride > 1) {
            const int H_c = (out_h + l.d_stride - 1) / l.d_stride;
            const int W_c = (out_w + l.d_stride - 1) / l.d_stride;
            const int spatial_coarse = H_c * W_c;
            const size_t coarse_im2col_floats = (size_t)l.c * l.size * l.size * spatial_coarse;
            float *coarse_im2col   = scratch;
            float *coarse_offsets  = scratch + coarse_im2col_floats; // [B, spatial_coarse, padded_offset_dim]

            for (int b = 0; b < batch; ++b) {
                float *im        = state.input + b * l.c * l.h * l.w;
                float *coarse_b  = coarse_offsets + b * spatial_coarse * padded_offset_dim;
                // im2col at coarse stride = stride * d_stride
                im2col_gpu_ext(im, l.c, l.h, l.w, l.size, l.size,
                               l.pad, l.pad,
                               l.stride_y * l.d_stride, l.stride_x * l.d_stride,
                               l.dilation, l.dilation, coarse_im2col);
                gemm_ongpu_tensor_op(1, 1, spatial_coarse, padded_offset_dim, K_gemm, 1,
                           coarse_im2col, spatial_coarse,
                           l.offset_weights_gpu, K_gemm,
                           0, coarse_b, padded_offset_dim,
                           use_tensor_op);
            }
            // Add bias to coarse offsets
            int n_bias_c = batch * spatial_coarse * padded_offset_dim;
            add_bias_nhwc_kernel<<<(n_bias_c + threads - 1) / threads, threads, 0, get_cuda_stream()>>>(
                coarse_offsets, l.offset_biases_gpu, batch, spatial_coarse, padded_offset_dim);
            // Upsample coarse -> full offset field
            int total_up = batch * out_h * out_w * padded_offset_dim;
            nhwc_bilinear_upsample_kernel<<<(total_up + threads - 1) / threads, threads, 0, get_cuda_stream()>>>(
                coarse_offsets, l.offsets_gpu, batch, padded_offset_dim, H_c, W_c, out_h, out_w,
                l.d_stride, l.d_stride);
            CHECK_CUDA(cudaPeekAtLastError());
        } else {
            // Full-resolution offset prediction
            for (int b = 0; b < batch; ++b) {
                float *im  = state.input + b * l.c * l.h * l.w;
                float *off = l.offsets_gpu + b * spatial_out * padded_offset_dim;
                im2col_gpu_ext(im, l.c, l.h, l.w, l.size, l.size,
                               l.pad, l.pad,
                               l.stride_y, l.stride_x, l.dilation, l.dilation, scratch);
                gemm_ongpu_tensor_op(1, 1, spatial_out, padded_offset_dim, K_gemm, 1,
                           scratch, spatial_out, l.offset_weights_gpu, K_gemm, 0, off, padded_offset_dim,
                           use_tensor_op);
            }
            int n_bias = batch * spatial_out * padded_offset_dim;
            add_bias_nhwc_kernel<<<(n_bias + threads - 1) / threads, threads, 0, get_cuda_stream()>>>(
                l.offsets_gpu, l.offset_biases_gpu, batch, spatial_out, padded_offset_dim);
        }
    }

    nchw_to_nhwc_gpu(state.input, input_nhwc, batch, l.c, l.h, l.w);

    // 3. Deformable Attention Kernel
    int G = l.groups;
    int D = l.n / G;
    int Q = spatial_out;
    int block_thread = l.block_thread;
    // d_stride controls per-thread ILP in the D dimension.
    // Use 4-wide channel tiles when possible, otherwise fall back to 2 or 1.
    const int d_stride = (D % 4 == 0 && D >= 4) ? 4 : ((D % 2 == 0 && D >= 2) ? 2 : 1);
    int block_multiplier = block_thread / (D / d_stride) / G;
    if (block_multiplier <= 0) block_multiplier = 1;

    dim3 num_blocks((batch * Q + block_multiplier - 1) / block_multiplier);
    dim3 num_threads(D / d_stride, G, block_multiplier);

    if (d_stride == 4) {
        if (l.softmax) dcnv4_launch_forward_dispatch<4, true>(l, K, input_nhwc, l.offsets_gpu, output_nhwc, batch, G, D, Q, out_h, out_w, block_multiplier, padded_offset_dim, num_blocks, num_threads);
        else           dcnv4_launch_forward_dispatch<4, false>(l, K, input_nhwc, l.offsets_gpu, output_nhwc, batch, G, D, Q, out_h, out_w, block_multiplier, padded_offset_dim, num_blocks, num_threads);
    } else if (d_stride == 2) {
        if (l.softmax) dcnv4_launch_forward_dispatch<2, true>(l, K, input_nhwc, l.offsets_gpu, output_nhwc, batch, G, D, Q, out_h, out_w, block_multiplier, padded_offset_dim, num_blocks, num_threads);
        else           dcnv4_launch_forward_dispatch<2, false>(l, K, input_nhwc, l.offsets_gpu, output_nhwc, batch, G, D, Q, out_h, out_w, block_multiplier, padded_offset_dim, num_blocks, num_threads);
    } else {
        if (l.softmax) dcnv4_launch_forward_dispatch<1, true>(l, K, input_nhwc, l.offsets_gpu, output_nhwc, batch, G, D, Q, out_h, out_w, block_multiplier, padded_offset_dim, num_blocks, num_threads);
        else           dcnv4_launch_forward_dispatch<1, false>(l, K, input_nhwc, l.offsets_gpu, output_nhwc, batch, G, D, Q, out_h, out_w, block_multiplier, padded_offset_dim, num_blocks, num_threads);
    }
    CHECK_CUDA(cudaPeekAtLastError());

    // 4. Return to NCHW
    nhwc_to_nchw_gpu(output_nhwc, l.output_gpu, batch, l.n, out_h, out_w);

    if (l.batch_normalize) {
        forward_batchnorm_layer_gpu(l, state);
    } else {
        add_bias_gpu(l.output_gpu, l.biases_gpu, batch, l.n, spatial_out);
    }
    if (l.activation == SWISH) activate_array_swish_ongpu(l.output_gpu, l.outputs * batch, l.activation_input_gpu, l.output_gpu);
    else if (l.activation == MISH) activate_array_mish_ongpu(l.output_gpu, l.outputs * batch, l.activation_input_gpu, l.output_gpu);
    else if (l.activation == HARD_MISH) activate_array_hard_mish_ongpu(l.output_gpu, l.outputs * batch, l.activation_input_gpu, l.output_gpu);
    else if (l.activation == EML) activate_array_eml_ongpu(l.output_gpu, l.outputs * batch, l.activation_input_gpu, l.output_gpu);
    else if (l.activation == NORM_CHAN) activate_array_normalize_channels_ongpu(l.output_gpu, l.outputs * batch, l.batch, l.out_c, spatial_out, l.output_gpu);
    else if (l.activation == NORM_CHAN_SOFTMAX) activate_array_normalize_channels_softmax_ongpu(l.output_gpu, l.outputs * batch, l.batch, l.out_c, spatial_out, l.output_gpu, 0);
    else if (l.activation == NORM_CHAN_SOFTMAX_MAXVAL) activate_array_normalize_channels_softmax_ongpu(l.output_gpu, l.outputs * batch, l.batch, l.out_c, spatial_out, l.output_gpu, 1);
    else if (l.activation != LINEAR) activate_array_ongpu(l.output_gpu, l.outputs * batch, l.activation);
}

__global__ void dcnv4_backward_kernel_gpu_simple(
    const float *__restrict__ p_value, const float *__restrict__ p_offset, const float *__restrict__ grad_output,
    const int B, const int G, const int D, const int Q, const int kernel_h,
    const int kernel_w, const int stride_h, const int stride_w, const int pad_h,
    const int pad_w, const int dilation_h, const int dilation_w,
    const int height_in, const int width_in, const int height_out,
    const int width_out, const float offset_scale, const int remove_center,
    float *__restrict__ grad_im, float *__restrict__ grad_offset, const int padded_offset_dim, bool softmax)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int total_pixels = B * Q * G;
    if (index >= total_pixels) return;

    int g = index % G;
    int q = (index / G) % Q;
    int b = index / (G * Q);

    int K = kernel_h * kernel_w;
    if (remove_center) K -= 1;

    const float *p_offset_ptr = p_offset + (b * Q + q) * padded_offset_dim + g * K * 3;
    float *grad_offset_ptr = grad_offset + (b * Q + q) * padded_offset_dim + g * K * 3;

    // Max K = 7*7 = 49; 9 was only safe for 3x3 kernels
    float p_mask[49];
    for (int i = 0; i < K; i++) {
        p_mask[i] = p_offset_ptr[K * 2 + i];
    }
    if (softmax) {
        float max_val = -1e10f;
        float sum = 0.0f;
        for (int i = 0; i < K; i++) max_val = fmaxf(max_val, p_mask[i]);
        for (int i = 0; i < K; i++) {
            p_mask[i] = expf(p_mask[i] - max_val);
            sum += p_mask[i];
        }
        for (int i = 0; i < K; i++) p_mask[i] /= (sum + 1e-6f);
    }

    const int p0_w = ((dilation_w * (kernel_w - 1)) >> 1) - pad_w + (q % width_out) * stride_w;
    const int p0_h = ((dilation_h * (kernel_h - 1)) >> 1) - pad_h + (q / width_out) * stride_h;
    const float p0_w_ = (float)p0_w - ((dilation_w * (kernel_w - 1)) >> 1) * offset_scale;
    const float p0_h_ = (float)p0_h - ((dilation_h * (kernel_h - 1)) >> 1) * offset_scale;
    const int center_h = kernel_h / 2;
    const int center_w = kernel_w / 2;

    int offset_idx = 0;
    int mask_idx = 0;

    for (int i = 0; i < kernel_w; ++i) {
        for (int j = 0; j < kernel_h; ++j) {
            if (i != center_w || j != center_h || !remove_center) {
                float offset_w_val = p_offset_ptr[offset_idx];
                float offset_h_val = p_offset_ptr[offset_idx + 1];
                float w_im = p0_w_ + (i * dilation_w + offset_w_val) * offset_scale;
                float h_im = p0_h_ + (j * dilation_h + offset_h_val) * offset_scale;
                float attn = p_mask[mask_idx];

                int h_low = floorf(h_im);
                int w_low = floorf(w_im);
                int h_high = h_low + 1;
                int w_high = w_low + 1;
                float lh = h_im - h_low;
                float lw = w_im - w_low;
                float hh = 1.0f - lh;
                float hw = 1.0f - lw;
                float w1 = hh * hw, w2 = hh * lw, w3 = lh * hw, w4 = lh * lw;

                float grad_w_weight_sum = 0.0f;
                float grad_h_weight_sum = 0.0f;
                float current_val_grad_sum = 0.0f;

                for (int d = 0; d < D; d++) {
                    float top_grad = grad_output[((b * Q + q) * G + g) * D + d];
                    float p_val1 = 0, p_val2 = 0, p_val3 = 0, p_val4 = 0;

                    int c_idx = g * D + d;
                    if (h_low >= 0 && h_low < height_in && w_low >= 0 && w_low < width_in) {
                        int idx = ((b * height_in + h_low) * width_in + w_low) * G * D + c_idx;
                        p_val1 = p_value[idx];
                        atomicAdd(&grad_im[idx], top_grad * attn * w1);
                    }
                    if (h_low >= 0 && h_low < height_in && w_high >= 0 && w_high < width_in) {
                        int idx = ((b * height_in + h_low) * width_in + w_high) * G * D + c_idx;
                        p_val2 = p_value[idx];
                        atomicAdd(&grad_im[idx], top_grad * attn * w2);
                    }
                    if (h_high >= 0 && h_high < height_in && w_low >= 0 && w_low < width_in) {
                        int idx = ((b * height_in + h_high) * width_in + w_low) * G * D + c_idx;
                        p_val3 = p_value[idx];
                        atomicAdd(&grad_im[idx], top_grad * attn * w3);
                    }
                    if (h_high >= 0 && h_high < height_in && w_high >= 0 && w_high < width_in) {
                        int idx = ((b * height_in + h_high) * width_in + w_high) * G * D + c_idx;
                        p_val4 = p_value[idx];
                        atomicAdd(&grad_im[idx], top_grad * attn * w4);
                    }

                    float grad_w_weight = -hh * p_val1 + hh * p_val2 - lh * p_val3 + lh * p_val4;
                    float grad_h_weight = -hw * p_val1 - lw * p_val2 + hw * p_val3 + lw * p_val4;

                    grad_w_weight_sum += grad_w_weight * top_grad;
                    grad_h_weight_sum += grad_h_weight * top_grad;

                    float current_val = w1 * p_val1 + w2 * p_val2 + w3 * p_val3 + w4 * p_val4;
                    current_val_grad_sum += current_val * top_grad;
                }

                grad_offset_ptr[offset_idx] = grad_w_weight_sum * offset_scale * attn;
                grad_offset_ptr[offset_idx + 1] = grad_h_weight_sum * offset_scale * attn;
                grad_offset_ptr[K * 2 + mask_idx] = current_val_grad_sum;

                offset_idx += 2;
                mask_idx += 1;
            }
        }
    }

    if (softmax) {
        float sum = 0.0f;
        for (int i = 0; i < K; i++) {
            sum += grad_offset_ptr[K * 2 + i] * p_mask[i];
        }
        for (int i = 0; i < K; i++) {
            grad_offset_ptr[K * 2 + i] = p_mask[i] * (grad_offset_ptr[K * 2 + i] - sum);
        }
    }
}

template <int K, bool softmax,
          int kernel_h_c = 0, int kernel_w_c = 0,
          int stride_h_c = 0, int stride_w_c = 0,
          int dilation_h_c = 0, int dilation_w_c = 0,
          int remove_center_c = -1>
__global__ void dcnv4_backward_kernel_gpu_block(
    const float *__restrict__ p_value, const float *__restrict__ p_offset, const float *__restrict__ grad_output,
    const int B, const int G, const int D, const int Q, const int kernel_h,
    const int kernel_w, const int stride_h, const int stride_w, const int pad_h,
    const int pad_w, const int dilation_h, const int dilation_w,
    const int height_in, const int width_in, const int height_out,
    const int width_out, const float offset_scale, const int remove_center,
    float *__restrict__ grad_im, float *__restrict__ grad_offset, const int padded_offset_dim)
{
    const int index = blockIdx.x;
    const int total_pixels = B * Q * G;
    if (index >= total_pixels) return;

    const int kernel_h_eff = kernel_h_c ? kernel_h_c : kernel_h;
    const int kernel_w_eff = kernel_w_c ? kernel_w_c : kernel_w;
    const int stride_h_eff = stride_h_c ? stride_h_c : stride_h;
    const int stride_w_eff = stride_w_c ? stride_w_c : stride_w;
    const int dilation_h_eff = dilation_h_c ? dilation_h_c : dilation_h;
    const int dilation_w_eff = dilation_w_c ? dilation_w_c : dilation_w;
    const bool remove_center_eff = (remove_center_c >= 0) ? (remove_center_c != 0) : (remove_center != 0);

    const int g = index % G;
    const int q = (index / G) % Q;
    const int b = index / (G * Q);
    const int tid = threadIdx.x;

    const float *p_offset_ptr = p_offset + (b * Q + q) * padded_offset_dim + g * K * 3;
    float *grad_offset_ptr = grad_offset + (b * Q + q) * padded_offset_dim + g * K * 3;

    extern __shared__ float shared[];
    float *red_w = shared;
    float *red_h = red_w + blockDim.x;
    float *red_m = red_h + blockDim.x;
    float *p_mask = red_m + blockDim.x;

    if (tid < K)
    {
        p_mask[tid] = p_offset_ptr[K * 2 + tid];
    }
    __syncthreads();

    if (softmax && tid == 0)
    {
        float max_val = -1e10f;
        float sum = 0.0f;
        for (int i = 0; i < K; i++) max_val = fmaxf(max_val, p_mask[i]);
        for (int i = 0; i < K; i++)
        {
            p_mask[i] = expf(p_mask[i] - max_val);
            sum += p_mask[i];
        }
        const float inv_sum = 1.0f / (sum + 1e-6f);
        for (int i = 0; i < K; i++) p_mask[i] *= inv_sum;
    }
    __syncthreads();

    const int p0_w = ((dilation_w_eff * (kernel_w_eff - 1)) >> 1) - pad_w + (q % width_out) * stride_w_eff;
    const int p0_h = ((dilation_h_eff * (kernel_h_eff - 1)) >> 1) - pad_h + (q / width_out) * stride_h_eff;
    const float p0_w_ = (float)p0_w - ((dilation_w_eff * (kernel_w_eff - 1)) >> 1) * offset_scale;
    const float p0_h_ = (float)p0_h - ((dilation_h_eff * (kernel_h_eff - 1)) >> 1) * offset_scale;
    const int center_h = kernel_h_eff / 2;
    const int center_w = kernel_w_eff / 2;

    int offset_idx = 0;
    int mask_idx = 0;

    for (int i = 0; i < kernel_w_eff; ++i)
    {
        for (int j = 0; j < kernel_h_eff; ++j)
        {
            if (i != center_w || j != center_h || !remove_center_eff)
            {
                const float offset_w_val = p_offset_ptr[offset_idx];
                const float offset_h_val = p_offset_ptr[offset_idx + 1];
                const float w_im = p0_w_ + (i * dilation_w_eff + offset_w_val) * offset_scale;
                const float h_im = p0_h_ + (j * dilation_h_eff + offset_h_val) * offset_scale;
                const float attn = p_mask[mask_idx];

                const int h_low = floorf(h_im);
                const int w_low = floorf(w_im);
                const int h_high = h_low + 1;
                const int w_high = w_low + 1;
                const float lh = h_im - h_low;
                const float lw = w_im - w_low;
                const float hh = 1.0f - lh;
                const float hw = 1.0f - lw;
                const float w1 = hh * hw;
                const float w2 = hh * lw;
                const float w3 = lh * hw;
                const float w4 = lh * lw;

                float grad_w_weight_sum = 0.0f;
                float grad_h_weight_sum = 0.0f;
                float current_val_grad_sum = 0.0f;

                for (int d = tid; d < D; d += blockDim.x)
                {
                    const float top_grad = grad_output[((b * Q + q) * G + g) * D + d];
                    float p_val1 = 0.0f;
                    float p_val2 = 0.0f;
                    float p_val3 = 0.0f;
                    float p_val4 = 0.0f;

                    const int c_idx = g * D + d;
                    if (h_low >= 0 && h_low < height_in && w_low >= 0 && w_low < width_in)
                    {
                        const int idx = ((b * height_in + h_low) * width_in + w_low) * G * D + c_idx;
                        p_val1 = p_value[idx];
                        atomicAdd(&grad_im[idx], top_grad * attn * w1);
                    }
                    if (h_low >= 0 && h_low < height_in && w_high >= 0 && w_high < width_in)
                    {
                        const int idx = ((b * height_in + h_low) * width_in + w_high) * G * D + c_idx;
                        p_val2 = p_value[idx];
                        atomicAdd(&grad_im[idx], top_grad * attn * w2);
                    }
                    if (h_high >= 0 && h_high < height_in && w_low >= 0 && w_low < width_in)
                    {
                        const int idx = ((b * height_in + h_high) * width_in + w_low) * G * D + c_idx;
                        p_val3 = p_value[idx];
                        atomicAdd(&grad_im[idx], top_grad * attn * w3);
                    }
                    if (h_high >= 0 && h_high < height_in && w_high >= 0 && w_high < width_in)
                    {
                        const int idx = ((b * height_in + h_high) * width_in + w_high) * G * D + c_idx;
                        p_val4 = p_value[idx];
                        atomicAdd(&grad_im[idx], top_grad * attn * w4);
                    }

                    const float grad_w_weight = -hh * p_val1 + hh * p_val2 - lh * p_val3 + lh * p_val4;
                    const float grad_h_weight = -hw * p_val1 - lw * p_val2 + hw * p_val3 + lw * p_val4;
                    grad_w_weight_sum += grad_w_weight * top_grad;
                    grad_h_weight_sum += grad_h_weight * top_grad;

                    const float current_val = w1 * p_val1 + w2 * p_val2 + w3 * p_val3 + w4 * p_val4;
                    current_val_grad_sum += current_val * top_grad;
                }

                red_w[tid] = grad_w_weight_sum;
                red_h[tid] = grad_h_weight_sum;
                red_m[tid] = current_val_grad_sum;
                __syncthreads();

                for (int step = blockDim.x >> 1; step > 0; step >>= 1)
                {
                    if (tid < step)
                    {
                        red_w[tid] += red_w[tid + step];
                        red_h[tid] += red_h[tid + step];
                        red_m[tid] += red_m[tid + step];
                    }
                    __syncthreads();
                }

                if (tid == 0)
                {
                    grad_offset_ptr[offset_idx] = red_w[0] * offset_scale * attn;
                    grad_offset_ptr[offset_idx + 1] = red_h[0] * offset_scale * attn;
                    grad_offset_ptr[K * 2 + mask_idx] = red_m[0];
                }
                __syncthreads();

                offset_idx += 2;
                mask_idx += 1;
            }
        }
    }

    if (softmax && tid == 0)
    {
        float sum = 0.0f;
        for (int i = 0; i < K; i++)
        {
            sum += grad_offset_ptr[K * 2 + i] * p_mask[i];
        }
        for (int i = 0; i < K; i++)
        {
            grad_offset_ptr[K * 2 + i] = p_mask[i] * (grad_offset_ptr[K * 2 + i] - sum);
        }
    }

    (void)height_out;
}

static int dcnv4_backward_block_threads(const int D)
{
    int threads = 32;
    while (threads < D && threads < 256)
    {
        threads <<= 1;
    }
    return threads;
}

template <int KVAL, bool SM,
          int kernel_h_c = 0, int kernel_w_c = 0,
          int stride_h_c = 0, int stride_w_c = 0,
          int dilation_h_c = 0, int dilation_w_c = 0,
          int remove_center_c = -1>
static void dcnv4_launch_backward_block_kernel(
    const Darknet::Layer & l,
    const float *input_nhwc,
    const float *offsets,
    const float *grad_output_nhwc,
    const int batch,
    const int G,
    const int D,
    const int Q,
    const int out_h,
    const int out_w,
    float *grad_im_nhwc,
    float *grad_offset_nhwc,
    const int padded_offset_dim,
    const int total_pixels)
{
    const int threads = dcnv4_backward_block_threads(D);
    const size_t shared_size = static_cast<size_t>(3 * threads + KVAL) * sizeof(float);
    dcnv4_backward_kernel_gpu_block<KVAL, SM,
        kernel_h_c, kernel_w_c, stride_h_c, stride_w_c,
        dilation_h_c, dilation_w_c, remove_center_c>
        <<<total_pixels, threads, shared_size, get_cuda_stream()>>>(
            input_nhwc, offsets, grad_output_nhwc,
            batch, G, D, Q,
            l.size, l.size,
            l.stride_y, l.stride_x,
            l.pad, l.pad,
            l.dilation, l.dilation,
            l.h, l.w,
            out_h, out_w,
            l.offset_scale,
            l.remove_center,
            grad_im_nhwc,
            grad_offset_nhwc,
            padded_offset_dim);
}

template <bool SM>
static bool dcnv4_try_launch_backward_static(
    const Darknet::Layer & l,
    const float *input_nhwc,
    const float *offsets,
    const float *grad_output_nhwc,
    const int batch,
    const int G,
    const int D,
    const int Q,
    const int out_h,
    const int out_w,
    float *grad_im_nhwc,
    float *grad_offset_nhwc,
    const int padded_offset_dim,
    const int total_pixels)
{
    if (l.dilation != 1 || l.stride_x != l.stride_y)
    {
        return false;
    }

    const int stride = l.stride_x;
    if (l.size == 3)
    {
        if (stride == 1)
        {
            if (l.remove_center) dcnv4_launch_backward_block_kernel<8, SM, 3, 3, 1, 1, 1, 1, 1>(l, input_nhwc, offsets, grad_output_nhwc, batch, G, D, Q, out_h, out_w, grad_im_nhwc, grad_offset_nhwc, padded_offset_dim, total_pixels);
            else dcnv4_launch_backward_block_kernel<9, SM, 3, 3, 1, 1, 1, 1, 0>(l, input_nhwc, offsets, grad_output_nhwc, batch, G, D, Q, out_h, out_w, grad_im_nhwc, grad_offset_nhwc, padded_offset_dim, total_pixels);
            return true;
        }
        if (stride == 2)
        {
            if (l.remove_center) dcnv4_launch_backward_block_kernel<8, SM, 3, 3, 2, 2, 1, 1, 1>(l, input_nhwc, offsets, grad_output_nhwc, batch, G, D, Q, out_h, out_w, grad_im_nhwc, grad_offset_nhwc, padded_offset_dim, total_pixels);
            else dcnv4_launch_backward_block_kernel<9, SM, 3, 3, 2, 2, 1, 1, 0>(l, input_nhwc, offsets, grad_output_nhwc, batch, G, D, Q, out_h, out_w, grad_im_nhwc, grad_offset_nhwc, padded_offset_dim, total_pixels);
            return true;
        }
    }
    else if (l.size == 5)
    {
        if (stride == 1)
        {
            if (l.remove_center) dcnv4_launch_backward_block_kernel<24, SM, 5, 5, 1, 1, 1, 1, 1>(l, input_nhwc, offsets, grad_output_nhwc, batch, G, D, Q, out_h, out_w, grad_im_nhwc, grad_offset_nhwc, padded_offset_dim, total_pixels);
            else dcnv4_launch_backward_block_kernel<25, SM, 5, 5, 1, 1, 1, 1, 0>(l, input_nhwc, offsets, grad_output_nhwc, batch, G, D, Q, out_h, out_w, grad_im_nhwc, grad_offset_nhwc, padded_offset_dim, total_pixels);
            return true;
        }
        if (stride == 2)
        {
            if (l.remove_center) dcnv4_launch_backward_block_kernel<24, SM, 5, 5, 2, 2, 1, 1, 1>(l, input_nhwc, offsets, grad_output_nhwc, batch, G, D, Q, out_h, out_w, grad_im_nhwc, grad_offset_nhwc, padded_offset_dim, total_pixels);
            else dcnv4_launch_backward_block_kernel<25, SM, 5, 5, 2, 2, 1, 1, 0>(l, input_nhwc, offsets, grad_output_nhwc, batch, G, D, Q, out_h, out_w, grad_im_nhwc, grad_offset_nhwc, padded_offset_dim, total_pixels);
            return true;
        }
    }
    else if (l.size == 7)
    {
        if (stride == 1)
        {
            if (l.remove_center) dcnv4_launch_backward_block_kernel<48, SM, 7, 7, 1, 1, 1, 1, 1>(l, input_nhwc, offsets, grad_output_nhwc, batch, G, D, Q, out_h, out_w, grad_im_nhwc, grad_offset_nhwc, padded_offset_dim, total_pixels);
            else dcnv4_launch_backward_block_kernel<49, SM, 7, 7, 1, 1, 1, 1, 0>(l, input_nhwc, offsets, grad_output_nhwc, batch, G, D, Q, out_h, out_w, grad_im_nhwc, grad_offset_nhwc, padded_offset_dim, total_pixels);
            return true;
        }
        if (stride == 2)
        {
            if (l.remove_center) dcnv4_launch_backward_block_kernel<48, SM, 7, 7, 2, 2, 1, 1, 1>(l, input_nhwc, offsets, grad_output_nhwc, batch, G, D, Q, out_h, out_w, grad_im_nhwc, grad_offset_nhwc, padded_offset_dim, total_pixels);
            else dcnv4_launch_backward_block_kernel<49, SM, 7, 7, 2, 2, 1, 1, 0>(l, input_nhwc, offsets, grad_output_nhwc, batch, G, D, Q, out_h, out_w, grad_im_nhwc, grad_offset_nhwc, padded_offset_dim, total_pixels);
            return true;
        }
    }

    return false;
}

template <bool SM>
static bool dcnv4_launch_backward_dispatch(
    const Darknet::Layer & l,
    const int K,
    const float *input_nhwc,
    const float *offsets,
    const float *grad_output_nhwc,
    const int batch,
    const int G,
    const int D,
    const int Q,
    const int out_h,
    const int out_w,
    float *grad_im_nhwc,
    float *grad_offset_nhwc,
    const int padded_offset_dim,
    const int total_pixels)
{
    if (D < 8 || total_pixels <= 0)
    {
        return false;
    }
    if (dcnv4_try_launch_backward_static<SM>(l, input_nhwc, offsets, grad_output_nhwc, batch, G, D, Q, out_h, out_w, grad_im_nhwc, grad_offset_nhwc, padded_offset_dim, total_pixels))
    {
        return true;
    }

    switch (K) {
        case  4: dcnv4_launch_backward_block_kernel< 4, SM>(l, input_nhwc, offsets, grad_output_nhwc, batch, G, D, Q, out_h, out_w, grad_im_nhwc, grad_offset_nhwc, padded_offset_dim, total_pixels); return true;
        case  8: dcnv4_launch_backward_block_kernel< 8, SM>(l, input_nhwc, offsets, grad_output_nhwc, batch, G, D, Q, out_h, out_w, grad_im_nhwc, grad_offset_nhwc, padded_offset_dim, total_pixels); return true;
        case  9: dcnv4_launch_backward_block_kernel< 9, SM>(l, input_nhwc, offsets, grad_output_nhwc, batch, G, D, Q, out_h, out_w, grad_im_nhwc, grad_offset_nhwc, padded_offset_dim, total_pixels); return true;
        case 15: dcnv4_launch_backward_block_kernel<15, SM>(l, input_nhwc, offsets, grad_output_nhwc, batch, G, D, Q, out_h, out_w, grad_im_nhwc, grad_offset_nhwc, padded_offset_dim, total_pixels); return true;
        case 16: dcnv4_launch_backward_block_kernel<16, SM>(l, input_nhwc, offsets, grad_output_nhwc, batch, G, D, Q, out_h, out_w, grad_im_nhwc, grad_offset_nhwc, padded_offset_dim, total_pixels); return true;
        case 24: dcnv4_launch_backward_block_kernel<24, SM>(l, input_nhwc, offsets, grad_output_nhwc, batch, G, D, Q, out_h, out_w, grad_im_nhwc, grad_offset_nhwc, padded_offset_dim, total_pixels); return true;
        case 25: dcnv4_launch_backward_block_kernel<25, SM>(l, input_nhwc, offsets, grad_output_nhwc, batch, G, D, Q, out_h, out_w, grad_im_nhwc, grad_offset_nhwc, padded_offset_dim, total_pixels); return true;
        case 48: dcnv4_launch_backward_block_kernel<48, SM>(l, input_nhwc, offsets, grad_output_nhwc, batch, G, D, Q, out_h, out_w, grad_im_nhwc, grad_offset_nhwc, padded_offset_dim, total_pixels); return true;
        case 49: dcnv4_launch_backward_block_kernel<49, SM>(l, input_nhwc, offsets, grad_output_nhwc, batch, G, D, Q, out_h, out_w, grad_im_nhwc, grad_offset_nhwc, padded_offset_dim, total_pixels); return true;
        default: return false;
    }
}

void backward_dcnv4_layer_gpu(Darknet::Layer & l, Darknet::NetworkState state) 
{
    int K = l.size * l.size;
    if (l.remove_center) K -= 1;
    int offset_filters_raw = l.groups * K * 3;
    int padded_offset_dim = ((offset_filters_raw + 7) / 8) * 8;
    int out_h = l.out_h;
    int out_w = l.out_w;
    int spatial_out = out_h * out_w;
    int batch = l.batch;
    const bool use_tensor_op = dcnv4_use_tensor_op(state);
    
    // We expect the delta from the next layer in l.delta_gpu (NCHW)
    if (l.activation == SWISH) gradient_array_swish_ongpu(l.output_gpu, l.outputs * batch, l.activation_input_gpu, l.delta_gpu);
    else if (l.activation == MISH) gradient_array_mish_ongpu(l.outputs * batch, l.activation_input_gpu, l.delta_gpu);
    else if (l.activation == HARD_MISH) gradient_array_hard_mish_ongpu(l.outputs * batch, l.activation_input_gpu, l.delta_gpu);
    else if (l.activation == EML) gradient_array_eml_ongpu(l.outputs * batch, l.activation_input_gpu, l.delta_gpu);
    else if (l.activation == NORM_CHAN_SOFTMAX || l.activation == NORM_CHAN_SOFTMAX_MAXVAL) gradient_array_normalize_channels_softmax_ongpu(l.output_gpu, l.outputs * batch, l.batch, l.out_c, spatial_out, l.delta_gpu);
    else if (l.activation == NORM_CHAN) gradient_array_normalize_channels_ongpu(l.output_gpu, l.outputs * batch, l.batch, l.out_c, spatial_out, l.delta_gpu);
    else gradient_array_ongpu(l.output_gpu, l.outputs * batch, l.activation, l.delta_gpu);
    
    if (l.batch_normalize) {
        backward_batchnorm_layer_gpu(l, state);
    } else {
        backward_bias_gpu(l.bias_updates_gpu, l.delta_gpu, batch, l.n, spatial_out);
    }

    float *input_nhwc = state.workspace;
    size_t input_nhwc_size = (size_t)batch * l.c * l.h * l.w;
    float *grad_output_nhwc = input_nhwc + input_nhwc_size;
    size_t output_nhwc_size = (size_t)batch * l.n * spatial_out;
    float *grad_im_nhwc = grad_output_nhwc + output_nhwc_size;
    float *grad_offset_nhwc = grad_im_nhwc + input_nhwc_size;
    size_t offsets_nhwc_size = (size_t)batch * spatial_out * padded_offset_dim;
    float *grad_im_nchw = grad_offset_nhwc + offsets_nhwc_size;
    float *scratch = grad_im_nchw + input_nhwc_size;

    nchw_to_nhwc_gpu(state.input, input_nhwc, batch, l.c, l.h, l.w);
    nchw_to_nhwc_gpu(l.delta_gpu, grad_output_nhwc, batch, l.n, out_h, out_w);
    
    cudaMemsetAsync(grad_im_nhwc, 0, input_nhwc_size * sizeof(float), get_cuda_stream());
    cudaMemsetAsync(grad_offset_nhwc, 0, offsets_nhwc_size * sizeof(float), get_cuda_stream());
    
    int G = l.groups;
    int D = l.n / G;
    int Q = spatial_out;
    int total_pixels = batch * Q * G;
    int threads = 512;

    bool block_backward_launched = false;
    if (l.softmax) block_backward_launched = dcnv4_launch_backward_dispatch<true>(
        l, K, input_nhwc, l.offsets_gpu, grad_output_nhwc, batch, G, D, Q, out_h, out_w,
        grad_im_nhwc, grad_offset_nhwc, padded_offset_dim, total_pixels);
    else block_backward_launched = dcnv4_launch_backward_dispatch<false>(
        l, K, input_nhwc, l.offsets_gpu, grad_output_nhwc, batch, G, D, Q, out_h, out_w,
        grad_im_nhwc, grad_offset_nhwc, padded_offset_dim, total_pixels);

    if (!block_backward_launched)
    {
        int blocks = (total_pixels + threads - 1) / threads;
        dcnv4_backward_kernel_gpu_simple<<<blocks, threads, 0, get_cuda_stream()>>>(
            input_nhwc, l.offsets_gpu, grad_output_nhwc, batch, G, D, Q,
            l.size, l.size, l.stride_y, l.stride_x, l.pad, l.pad, l.dilation, l.dilation,
            l.h, l.w, out_h, out_w, l.offset_scale, l.remove_center,
            grad_im_nhwc, grad_offset_nhwc, padded_offset_dim, l.softmax > 0);
    }
    CHECK_CUDA(cudaPeekAtLastError());

    if (state.delta) {
        nhwc_to_nchw_gpu(grad_im_nhwc, grad_im_nchw, batch, l.c, l.h, l.w);
        axpy_ongpu((int)input_nhwc_size, 1.0f, grad_im_nchw, 1, state.delta, 1);
    }
    
    // Backprop through offset prediction subnet (weight + bias updates).
    // When d_stride > 1: backprop through bilinear upsample first to get coarse grad,
    // then use coarse im2col for the weight update GEMM.
    int K_gemm = l.c * l.size * l.size;
    int threads_bwd = 512;

#ifdef CUDNN
    if (dcnv4_backprop_offsets_cudnn_gpu(l, state, grad_offset_nhwc, scratch, grad_im_nchw, batch, padded_offset_dim, out_h, out_w, threads_bwd))
    {
        return;
    }
#endif

    if (l.d_stride > 1) {
        const int H_c = (out_h + l.d_stride - 1) / l.d_stride;
        const int W_c = (out_w + l.d_stride - 1) / l.d_stride;
        const int spatial_coarse = H_c * W_c;
        const size_t coarse_im2col_floats = (size_t)l.c * l.size * l.size * spatial_coarse;

        // grad_coarse layout: [B, spatial_coarse, padded_offset_dim]
        float *coarse_im2col  = scratch;
        float *grad_coarse    = scratch + coarse_im2col_floats;

        // Downsample gradient: grad_offset_nhwc [B, out_h, out_w, C] → grad_coarse [B, H_c, W_c, C]
        cudaMemsetAsync(grad_coarse, 0, (size_t)batch * spatial_coarse * padded_offset_dim * sizeof(float), get_cuda_stream());
        int total_down = batch * out_h * out_w * padded_offset_dim;
        nhwc_bilinear_upsample_backward_kernel<<<(total_down + threads_bwd - 1) / threads_bwd, threads_bwd, 0, get_cuda_stream()>>>(
            grad_offset_nhwc, grad_coarse,
            batch, padded_offset_dim, H_c, W_c, out_h, out_w,
            l.d_stride, l.d_stride);
        CHECK_CUDA(cudaPeekAtLastError());

        for (int b = 0; b < batch; ++b) {
            float *im         = state.input + b * l.c * l.h * l.w;
            float *grad_off_b = grad_coarse + b * spatial_coarse * padded_offset_dim;
            // Re-run im2col at coarse stride to get the column matrix for weight update
            im2col_gpu_ext(im, l.c, l.h, l.w, l.size, l.size,
                           l.pad, l.pad,
                           l.stride_y * l.d_stride, l.stride_x * l.d_stride,
                           l.dilation, l.dilation, coarse_im2col);
            gemm_ongpu_tensor_op(1, 1, padded_offset_dim, K_gemm, spatial_coarse, 1,
                       grad_off_b, padded_offset_dim,
                       coarse_im2col, spatial_coarse,
                       1, l.offset_weight_updates_gpu, K_gemm,
                       use_tensor_op);
            backward_bias_spatial_gpu(l.offset_bias_updates_gpu, grad_off_b, padded_offset_dim, spatial_coarse);

            if (state.delta) {
                float *col_delta = coarse_im2col;
                float *input_delta_b = grad_im_nchw + (size_t)b * l.c * l.h * l.w;
                fill_ongpu(l.c * l.h * l.w, 0.0f, input_delta_b, 1);
                gemm_ongpu_tensor_op(1, 1, K_gemm, spatial_coarse, padded_offset_dim, 1.0f,
                           l.offset_weights_gpu, K_gemm,
                           grad_off_b, padded_offset_dim,
                           0.0f, col_delta, spatial_coarse,
                           use_tensor_op);
                col2im_gpu_ext(col_delta, l.c, l.h, l.w, l.size, l.size,
                               l.pad, l.pad,
                               l.stride_y * l.d_stride, l.stride_x * l.d_stride,
                               l.dilation, l.dilation, input_delta_b);
                axpy_ongpu(l.c * l.h * l.w, 1.0f, input_delta_b, 1,
                           state.delta + (size_t)b * l.c * l.h * l.w, 1);
            }
        }
    } else {
        for (int b = 0; b < batch; ++b) {
            float *im      = state.input + b * l.c * l.h * l.w;
            float *grad_off = grad_offset_nhwc + b * spatial_out * padded_offset_dim;
            im2col_gpu_ext(im, l.c, l.h, l.w, l.size, l.size,
                           l.pad, l.pad,
                           l.stride_y, l.stride_x, l.dilation, l.dilation, scratch);
            gemm_ongpu_tensor_op(1, 1, padded_offset_dim, K_gemm, spatial_out, 1,
                       grad_off, padded_offset_dim, scratch, spatial_out,
                       1, l.offset_weight_updates_gpu, K_gemm,
                       use_tensor_op);
            backward_bias_spatial_gpu(l.offset_bias_updates_gpu, grad_off, padded_offset_dim, spatial_out);

            if (state.delta) {
                float *col_delta = scratch;
                float *input_delta_b = grad_im_nchw + (size_t)b * l.c * l.h * l.w;
                fill_ongpu(l.c * l.h * l.w, 0.0f, input_delta_b, 1);
                gemm_ongpu_tensor_op(1, 1, K_gemm, spatial_out, padded_offset_dim, 1.0f,
                       l.offset_weights_gpu, K_gemm,
                       grad_off, padded_offset_dim,
                       0.0f, col_delta, spatial_out,
                       use_tensor_op);
                col2im_gpu_ext(col_delta, l.c, l.h, l.w, l.size, l.size,
                               l.pad, l.pad,
                               l.stride_y, l.stride_x, l.dilation, l.dilation,
                               input_delta_b);
                axpy_ongpu(l.c * l.h * l.w, 1.0f, input_delta_b, 1,
                           state.delta + (size_t)b * l.c * l.h * l.w, 1);
            }
        }
    }
}

void push_dcnv4_layer(Darknet::Layer & l)
{
    int K = l.size * l.size;
    if (l.remove_center) K -= 1;
    int offset_filters_raw = l.groups * K * 3;
    int padded_offset_dim = ((offset_filters_raw + 7) / 8) * 8;
    size_t offset_weights_size = (size_t)padded_offset_dim * l.c * l.size * l.size;

    cuda_push_array(l.weights_gpu, l.weights, l.nweights);
    cuda_push_array(l.biases_gpu, l.biases, l.n);
    cuda_push_array(l.offset_weights_gpu, l.offset_weights, offset_weights_size);
    cuda_push_array(l.offset_biases_gpu, l.offset_biases, padded_offset_dim);

    if (l.batch_normalize) {
        cuda_push_array(l.scales_gpu, l.scales, l.n);
        cuda_push_array(l.rolling_mean_gpu, l.rolling_mean, l.n);
        cuda_push_array(l.rolling_variance_gpu, l.rolling_variance, l.n);
    }
}

void pull_dcnv4_layer(Darknet::Layer & l)
{
    int K = l.size * l.size;
    if (l.remove_center) K -= 1;
    int offset_filters_raw = l.groups * K * 3;
    int padded_offset_dim = ((offset_filters_raw + 7) / 8) * 8;
    size_t offset_weights_size = (size_t)padded_offset_dim * l.c * l.size * l.size;

    cuda_pull_array(l.weights_gpu, l.weights, l.nweights);
    cuda_pull_array(l.biases_gpu, l.biases, l.n);
    cuda_pull_array(l.offset_weights_gpu, l.offset_weights, offset_weights_size);
    cuda_pull_array(l.offset_biases_gpu, l.offset_biases, padded_offset_dim);

    if (l.batch_normalize) {
        cuda_pull_array(l.scales_gpu, l.scales, l.n);
        cuda_pull_array(l.rolling_mean_gpu, l.rolling_mean, l.n);
        cuda_pull_array(l.rolling_variance_gpu, l.rolling_variance, l.n);
    }
}

void update_dcnv4_layer_gpu(Darknet::Layer & l, int batch, float learning_rate, float momentum, float decay, float loss_scale) 
{
    float rate = (learning_rate * l.learning_rate_scale) / (batch * loss_scale);
    float offset_rate = rate * 0.01f;
    
    int K = l.size * l.size;
    if (l.remove_center) K -= 1;
    int offset_filters_raw = l.groups * K * 3;
    int padded_offset_dim = ((offset_filters_raw + 7) / 8) * 8;
    int offset_nweights = l.c * padded_offset_dim * l.size * l.size;

    axpy_ongpu(offset_nweights, -decay * batch * loss_scale, l.offset_weights_gpu, 1, l.offset_weight_updates_gpu, 1);
    axpy_ongpu(offset_nweights, offset_rate, l.offset_weight_updates_gpu, 1, l.offset_weights_gpu, 1);
    scal_ongpu(offset_nweights, momentum, l.offset_weight_updates_gpu, 1);

    axpy_ongpu(padded_offset_dim, offset_rate, l.offset_bias_updates_gpu, 1, l.offset_biases_gpu, 1);
    scal_ongpu(padded_offset_dim, momentum, l.offset_bias_updates_gpu, 1);
    
    if (l.batch_normalize) {
        axpy_ongpu(l.n, rate, l.scale_updates_gpu, 1, l.scales_gpu, 1);
        scal_ongpu(l.n, momentum, l.scale_updates_gpu, 1);
    } else {
        axpy_ongpu(l.n, rate, l.bias_updates_gpu, 1, l.biases_gpu, 1);
        scal_ongpu(l.n, momentum, l.bias_updates_gpu, 1);
    }
}
