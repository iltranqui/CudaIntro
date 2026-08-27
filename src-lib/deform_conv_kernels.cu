#include "darknet_internal.hpp"
#include "deform_conv_layer.hpp"
#include "blas.hpp"
#include "activations.hpp"
#include "im2col.hpp"
#include "gemm.hpp"
#include <iomanip>

namespace
{
    static auto & cfg_and_state = Darknet::CfgAndState::get();
}

__device__ int nan_detected_flag = 0;
__global__ void check_nan_kernel(const float* arr, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        if (isnan(arr[i]) || isinf(arr[i])) {
            atomicExch(&nan_detected_flag, 1);
        }
    }
}

static void check_nan_gpu(const char* step_name, const float* d_arr, int n, int layer_idx) {
    int h_flag = 0;
    cudaMemcpyToSymbol(nan_detected_flag, &h_flag, sizeof(int));
    int threads = 512;
    int blocks = (n + threads - 1) / threads;
    check_nan_kernel<<<blocks, threads, 0, get_cuda_stream()>>>(d_arr, n);
    cudaDeviceSynchronize();
    cudaMemcpyFromSymbol(&h_flag, nan_detected_flag, sizeof(int));
    if (h_flag) {
        const std::string layer_label = Darknet::layer_type_diagnostic_label(Darknet::ELayerType::DEFORM_CONV);
        printf("[%s layer] NaN/Inf detected at layer %d, step: %s\n", layer_label.c_str(), layer_idx, step_name);
        // We do not abort so we don't crash, but it will print!
    }
}

/**
 * @brief CUDA kernel for computing offsets in deformable convolution
 *
 * This kernel computes the offset field for deformable convolution.
 */
__global__ void compute_offsets_kernel(float *input, float *weights, float *biases, float *output,
                                      int batch, int channels, int height, int width,
                                      int out_h, int out_w, int size, int pad, int stride_x, int stride_y,
                                      int dilation, int offset_filters)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int spatial = out_h * out_w;
    if (index >= batch * offset_filters * spatial) return;

    // Decompose the linear thread index into batch, offset filter, and spatial (y, x) coordinates
    int b = index / (offset_filters * spatial);
    int rem = index % (offset_filters * spatial);
    int k = rem / spatial;
    int pos = rem % spatial;
    int i = pos / out_w;
    int j = pos % out_w;

    // Initialize with bias and calculate the base window position in the input
    float sum = biases[k];
    int h_offset = -pad + i * stride_y;
    int w_offset = -pad + j * stride_x;

    // Standard convolution loop: iterate over input channels and the kernel window
    int c, h, w, kh, kw;
    for (c = 0; c < channels; ++c) {
        for (kh = 0; kh < size; ++kh) {
            for (kw = 0; kw < size; ++kw) {
                h = h_offset + kh * dilation;
                w = w_offset + kw * dilation;

                // Accumulate weighted input values if within spatial boundaries
                if (h >= 0 && h < height && w >= 0 && w < width) {
                    int input_index = ((b * channels + c) * height + h) * width + w;
                    int weight_index = ((k * channels + c) * size + kh) * size + kw;
                    sum += input[input_index] * weights[weight_index];
                }
            }
        }
    }

    output[index] = sum;
}

/**
 * @brief CUDA device function for bilinear interpolation
 */
__device__ float bilinear_interpolate_gpu(const float* data, int h, int w, float y, float x)
{
    if (y < -1.0f || y > h || x < -1.0f || x > w) return 0.0f;

    int y_low = floorf(y);
    int x_low = floorf(x);
    int y_high = y_low + 1;
    int x_high = x_low + 1;

    float ly = y - y_low;
    float lx = x - x_low;
    float hy = 1.0f - ly;
    float hx = 1.0f - lx;

    float v1 = (y_low >= 0 && y_low < h && x_low >= 0 && x_low < w) ? data[y_low * w + x_low] : 0.0f;
    float v2 = (y_low >= 0 && y_low < h && x_high >= 0 && x_high < w) ? data[y_low * w + x_high] : 0.0f;
    float v3 = (y_high >= 0 && y_high < h && x_low >= 0 && x_low < w) ? data[y_high * w + x_low] : 0.0f;
    float v4 = (y_high >= 0 && y_high < h && x_high >= 0 && x_high < w) ? data[y_high * w + x_high] : 0.0f;

    return hy * hx * v1 + hy * lx * v2 + ly * hx * v3 + ly * lx * v4;
}

/**
 * @brief CUDA device function to compute bilinear interpolation gradients
 *
 * Computes gradients of bilinear interpolation w.r.t. the sampling coordinates.
 * Used for backpropagating through the offset field in deformable convolution.
 *
 * @param data Input feature map data
 * @param h Input height
 * @param w Input width
 * @param y Vertical sampling position
 * @param x Horizontal sampling position
 * @param grad_h Output gradient w.r.t. y coordinate
 * @param grad_w Output gradient w.r.t. x coordinate
 */
__device__ void get_bilinear_gradient_gpu(
    const float* data, int h, int w, float y, float x,
    float* grad_h, float* grad_w)
{
    if (y < -1.0f || y > h || x < -1.0f || x > w) {
        *grad_h = *grad_w = 0.0f;
        return;
    }

    int y_low = floorf(y);
    int x_low = floorf(x);
    int y_high = y_low + 1;
    int x_high = x_low + 1;

    float ly = y - y_low;
    float lx = x - x_low;
    float hy = 1.0f - ly;
    float hx = 1.0f - lx;

    // Sample 4 corners with bounds checking
    float v1 = (y_low >= 0 && x_low >= 0 && y_low < h && x_low < w)
               ? data[y_low * w + x_low] : 0.0f;
    float v2 = (y_low >= 0 && x_high < w && y_low < h && x_high >= 0)
               ? data[y_low * w + x_high] : 0.0f;
    float v3 = (y_high < h && x_low >= 0 && y_high >= 0 && x_low < w)
               ? data[y_high * w + x_low] : 0.0f;
    float v4 = (y_high < h && x_high < w && y_high >= 0 && x_high >= 0)
               ? data[y_high * w + x_high] : 0.0f;

    // Gradient w.r.t. y: d(bilinear)/dy
    *grad_h = (-hx) * v1 + (-lx) * v2 + hx * v3 + lx * v4;
    // Gradient w.r.t. x: d(bilinear)/dx
    *grad_w = (-hy) * v1 + hy * v2 + (-ly) * v3 + ly * v4;
}

/**
 * @brief CUDA kernel for deformable convolution forward pass (DCNv2 with mask support)
 *
 * TEACHING MOMENT: Inside the Deformable Kernel (the "Heart" of DCN)
 * This is the GPU code that runs for every single output pixel. It's like
 * a normal convolution kernel, but with "flexible" sampling.
 *
 * STEP-BY-STEP DATA FLOW:
 * 1.  Identify where we are in the batch and output feature map.
 * 2.  Start with the base (rigid) grid for our 3x3 (or other size) kernel.
 * 3.  Look up the (dy, dx) offsets we predicted in the previous step.
 * 4.  Calculate exactly where in the input image we need to look (dy + dx).
 * 5.  Sample that fractional pixel using Bilinear Interpolation.
 * 6.  Weight it by the modulation mask (if DCNv2) and the conv weights.
 * 7.  Sum it all up and store the result!
 */
__global__ void deform_conv_kernel(float *input, float *weights, float *biases, float *offsets,
                                  float *masks, float *output,
                                  int batch, int channels, int height, int width,
                                  int out_c, int out_h, int out_w, int size, int pad,
                                  int stride_x, int stride_y, int dilation, int groups, int use_mask, int use_bias)
{
    // Step 1: Who am I? (Calculate global thread index)
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= batch * out_c * out_h * out_w) return;

    // Step 2: Identify output location and corresponding input group/channels
    int b = index / (out_c * out_h * out_w);
    int c = (index % (out_c * out_h * out_w)) / (out_h * out_w);
    int i = (index % (out_h * out_w)) / out_w;
    int j = index % out_w;

    int group = c / (out_c / groups);
    int group_channels = channels / groups;

    // Step 3: Use pre-computed bias if available
    float sum = (use_bias && biases) ? biases[c] : 0.0f;
    int h_base = -pad + i * stride_y;
    int w_base = -pad + j * stride_x;

    // Step 4: Deformable sampling: for each input channel and each kernel point
    int spatial = out_h * out_w;
    int local_pos = i * out_w + j;
    int K = size * size;
    // Offsets stored as [B, 2*K, spatial] by compute_offsets_kernel
    int offset_base = b * (K * 2) * spatial;
    int mask_base = b * K * spatial;

    for (int g_c = 0; g_c < group_channels; ++g_c) {
        int c_in = group * group_channels + g_c;
        const float* im_ptr = input + (b * channels + c_in) * height * width;

        for (int kh = 0; kh < size; ++kh) {
            for (int kw = 0; kw < size; ++kw) {
                int k_idx = kh * size + kw;

                // Step 5: FETCH OFFSETS - Fetch learned offsets for this kernel position and spatial location
                int offset_h_idx = offset_base + (k_idx * 2) * spatial + local_pos;
                int offset_w_idx = offset_base + (k_idx * 2 + 1) * spatial + local_pos;
                float offset_h = offsets[offset_h_idx];
                float offset_w = offsets[offset_w_idx];

                // Step 6: FRACTIONAL POSITION - Compute fractional sampling coordinate
                float h_im = h_base + kh * dilation + offset_h;
                float w_im = w_base + kw * dilation + offset_w;

                // Step 7: BILINEAR INTERPOLATION - Sample between input grid points
                float val = bilinear_interpolate_gpu(im_ptr, height, width, h_im, w_im);

                // Step 8: MODULATION (DCNv2) - Apply modulation mask if provided
                if (use_mask && masks) {
                    int mask_idx = mask_base + k_idx * spatial + local_pos;
                    val *= masks[mask_idx];
                }

                // Step 9: APPLY WEIGHTS - Standard convolution weight multiplication
                int weight_index = ((c * group_channels + g_c) * size + kh) * size + kw;
                sum += val * weights[weight_index];
            }
        }
    }

    // Step 10: Store final result
    output[index] = sum;
}

/**
 * @brief GPU kernel for applying sigmoid to mask values
 */
__global__ void sigmoid_kernel(float *data, int n)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < n) {
        data[index] = 1.0f / (1.0f + expf(-data[index]));
    }
}

/**
 * @brief GPU kernel to compute sum and sum of squares for instance normalization
 *
 * Uses parallel reduction within blocks, then atomicAdd across blocks.
 */
__global__ void compute_offset_stats_kernel(const float *offsets, int n, float *sum_out, float *sum_sq_out)
{
    __shared__ float sdata[BLOCK];
    __shared__ float sdata_sq[BLOCK];

    int tid = threadIdx.x;
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    // Load and accumulate
    float val = (index < n) ? offsets[index] : 0.0f;
    sdata[tid] = val;
    sdata_sq[tid] = val * val;
    __syncthreads();

    // Reduction within block
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
            sdata_sq[tid] += sdata_sq[tid + s];
        }
        __syncthreads();
    }

    // Write block result
    if (tid == 0) {
        atomicAdd(sum_out, sdata[0]);
        atomicAdd(sum_sq_out, sdata_sq[0]);
    }
}

/**
 * @brief GPU kernel to apply instance normalization to offsets
 *
 * Normalizes: x = (x - mean) / sqrt(var + eps)
 */
__global__ void instance_norm_offset_kernel(float *offsets, int n, float mean, float inv_std)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < n) {
        offsets[index] = (offsets[index] - mean) * inv_std;
    }
}

/**
 * @brief GPU helper matching the centered-LHTAN offset mapping used on CPU.
 *
 * The mapping stays linear in [-max_offset, max_offset] and keeps a small
 * 0.001 slope outside that range so saturated offsets can still recover.
 */
__device__ __forceinline__ float centered_lhtan_activate_gpu(float x)
{
    float res = x;
    if (x < 0.0f) {
        res = 0.001f * x;
    } else if (x > 1.0f) {
        res = 0.001f * (x - 1.0f) + 1.0f;
    }
    return res;
}

__global__ void clamp_offsets_kernel(float *offsets, int n, float max_offset)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < n) {
        const float raw_offset = offsets[index];
        const float scaled = raw_offset / (2.0f * max_offset) + 0.5f;
        const float normalized = 2.0f * centered_lhtan_activate_gpu(scaled) - 1.0f;
        offsets[index] = normalized * max_offset;
    }
}

/**
 * @brief GPU kernel for applying the centered-LHTAN derivative to offset gradients.
 *
 * The centered-LHTAN mapping is identity in the main interval and has a small
 * 0.001 slope outside, so the backward pass only needs a piecewise multiplier.
 *
 * @param grad_offset Offset gradients to modify in-place
 * @param clamped_offset Offset values after the centered-LHTAN mapping
 * @param n Total number of offset elements
 * @param max_offset Maximum offset value of the linear interval
 */
__global__ void lhtan_gradient_kernel(float *grad_offset, const float *clamped_offset,
                                      int n, float max_offset)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < n) {
        const float offset = clamped_offset[index];
        const float lhtan_deriv = (offset < -max_offset || offset > max_offset) ? 0.001f : 1.0f;
        grad_offset[index] *= lhtan_deriv;
    }
}

/**
 * @brief GPU kernel for bias updates with spatial-major layout
 */
__global__ void backward_bias_spatial_kernel(float *bias_updates, const float *delta, int spatial, int filters)
{
    __shared__ float part[BLOCK];
    int filter = blockIdx.x;
    int p = threadIdx.x;
    float sum = 0.0f;
    for (int i = p; i < spatial; i += BLOCK) {
        sum += delta[i * filters + filter];
    }
    part[p] = sum;
    __syncthreads();
    if (p == 0) {
        float total = 0.0f;
        for (int i = 0; i < BLOCK; ++i) {
            total += part[i];
        }
        bias_updates[filter] += total;
    }
}

void backward_bias_spatial_gpu(float *bias_updates, const float *delta, int filters, int spatial)
{
    backward_bias_spatial_kernel<<<filters, BLOCK, 0, get_cuda_stream()>>>(bias_updates, delta, spatial, filters);
    CHECK_CUDA(cudaPeekAtLastError());
}

/**
 * @brief GPU kernel for deformable im2col
 *
 * Converts input into a column matrix using learned offsets (and masks for DCNv2).
 */
__global__ void deformable_im2col_gpu_kernel(
    const float* data_im, const float* data_offset, const float* data_mask,
    int channels, int height, int width,
    int kernel_h, int kernel_w,
    int pad_h, int pad_w,
    int stride_h, int stride_w,
    int dilation_h, int dilation_w,
    int out_h, int out_w,
    float* data_col)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int kernel_size = kernel_h * kernel_w;
    int total = channels * kernel_size * out_h * out_w;
    if (index >= total) return;

    int w_col = index % out_w;
    int h_col = (index / out_w) % out_h;
    int c_col = index / (out_w * out_h);
    int c_im = c_col / kernel_size;
    int k_idx = c_col % kernel_size;
    int kh = k_idx / kernel_w;
    int kw = k_idx % kernel_w;

    int pos_idx = h_col * out_w + w_col;
    int spatial = out_h * out_w;
    int offset_h_idx = (k_idx * 2) * spatial + pos_idx;
    int offset_w_idx = (k_idx * 2 + 1) * spatial + pos_idx;
    int mask_idx = k_idx * spatial + pos_idx;

    float offset_h = data_offset[offset_h_idx];
    float offset_w = data_offset[offset_w_idx];
    float mask_val = data_mask ? data_mask[mask_idx] : 1.0f;

    float h_in = h_col * stride_h - pad_h + kh * dilation_h + offset_h;
    float w_in = w_col * stride_w - pad_w + kw * dilation_w + offset_w;

    const float* im_ptr = data_im + c_im * height * width;
    float val = bilinear_interpolate_gpu(im_ptr, height, width, h_in, w_in);
    data_col[(c_col * out_h + h_col) * out_w + w_col] = val * mask_val;
}

static void deformable_im2col_gpu(
    const float* data_im, const float* data_offset, const float* data_mask,
    int channels, int height, int width,
    int kernel_h, int kernel_w,
    int pad_h, int pad_w,
    int stride_h, int stride_w,
    int dilation_h, int dilation_w,
    int out_h, int out_w,
    float* data_col)
{
    const int threads = BLOCK;
    const int total = channels * kernel_h * kernel_w * out_h * out_w;
    const int blocks = (total + threads - 1) / threads;
    deformable_im2col_gpu_kernel<<<blocks, threads, 0, get_cuda_stream()>>>(
        data_im, data_offset, data_mask,
        channels, height, width,
        kernel_h, kernel_w,
        pad_h, pad_w,
        stride_h, stride_w,
        dilation_h, dilation_w,
        out_h, out_w,
        data_col);
    CHECK_CUDA(cudaPeekAtLastError());
}

/**
 * @brief GPU kernel for computing gradients w.r.t. offsets and masks (DCNv2)
 *
 * Computes gradients by differentiating through bilinear interpolation.
 * One thread per (batch, output_position, kernel_position) element.
 */
__global__ void deformable_col2im_coord_gpu_kernel(
    const float* col_grad, const float* input, const float* offsets, const float* masks,
    int batch, int channels, int height, int width,
    int kernel_h, int kernel_w, int pad_h, int pad_w,
    int stride_h, int stride_w, int dilation_h, int dilation_w,
    int out_h, int out_w, int use_mask,
    float* grad_offset, float* grad_mask)
{
    int kernel_size = kernel_h * kernel_w;
    int total = batch * out_h * out_w * kernel_size;
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= total) return;

    int k_idx = index % kernel_size;
    int w_col = (index / kernel_size) % out_w;
    int h_col = (index / kernel_size / out_w) % out_h;
    int b = index / kernel_size / out_w / out_h;

    int kh = k_idx / kernel_w;
    int kw = k_idx % kernel_w;

    int pos_idx = (b * out_h + h_col) * out_w + w_col;
    int spatial = out_h * out_w;
    int offset_h_idx = (k_idx * 2) * spatial + pos_idx;
    int offset_w_idx = (k_idx * 2 + 1) * spatial + pos_idx;
    int mask_idx = k_idx * spatial + pos_idx;

    float offset_h = offsets[offset_h_idx];
    float offset_w = offsets[offset_w_idx];
    float mask_val = (use_mask && masks) ? masks[mask_idx] : 1.0f;

    float h_in = h_col * stride_h - pad_h + kh * dilation_h + offset_h;
    float w_in = w_col * stride_w - pad_w + kw * dilation_w + offset_w;

    float grad_h_acc = 0.0f, grad_w_acc = 0.0f, grad_m_acc = 0.0f;

    for (int c = 0; c < channels; ++c) {
        const float* im_ptr = input + (b * channels + c) * height * width;
        int c_col = c * kernel_size + k_idx;
        float col_g = col_grad[c_col * (out_h * out_w) + h_col * out_w + w_col];

        // 1. Compute gradients of the bilinear interpolation w.r.t. sampling coordinates (h, w)
        float dh, dw;
        get_bilinear_gradient_gpu(im_ptr, height, width, h_in, w_in, &dh, &dw);

        // 2. Accumulate gradients for offsets (chain rule: dLoss/dOffset = dLoss/dSample * dSample/dOffset)
        grad_h_acc += col_g * mask_val * dh;
        grad_w_acc += col_g * mask_val * dw;

        // 3. Accumulate gradients for the modulation mask (dLoss/dMask = dLoss/dSample * SampleValue)
        if (use_mask && grad_mask) {
            float val = bilinear_interpolate_gpu(im_ptr, height, width, h_in, w_in);
            grad_m_acc += col_g * val;
        }
    }

    // Atomic additions are required because multiple channels/kernel points contribute to the same offset gradient
    atomicAdd(&grad_offset[offset_h_idx], grad_h_acc);
    atomicAdd(&grad_offset[offset_w_idx], grad_w_acc);
    if (use_mask && grad_mask) {
        // Apply sigmoid derivative for the mask gradient path
        atomicAdd(&grad_mask[mask_idx], grad_m_acc * mask_val * (1.0f - mask_val));
    }
}

/**
 * @brief GPU kernel for distributing gradients back to input (deformable col2im)
 *
 * Inverse of deformable im2col - distributes column gradients to input positions
 * using learned offsets with bilinear interpolation weights.
 */
__global__ void deformable_col2im_gpu_kernel(
    const float* col_grad, const float* offsets, const float* masks,
    int batch, int channels, int height, int width,
    int kernel_h, int kernel_w, int pad_h, int pad_w,
    int stride_h, int stride_w, int dilation_h, int dilation_w,
    int out_h, int out_w, int use_mask,
    float* grad_input)
{
    int kernel_size = kernel_h * kernel_w;
    int col_channels = channels * kernel_size;
    int total = batch * col_channels * out_h * out_w;
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= total) return;

    int w_col = index % out_w;
    int h_col = (index / out_w) % out_h;
    int c_col = (index / out_w / out_h) % col_channels;
    int b = index / out_w / out_h / col_channels;

    int kw = c_col % kernel_w;
    int kh = (c_col / kernel_w) % kernel_h;
    int c_im = c_col / kernel_size;

    int pos_idx = (b * out_h + h_col) * out_w + w_col;
    int k_idx = kh * kernel_w + kw;
    int spatial = out_h * out_w;
    int offset_h_idx = (k_idx * 2) * spatial + pos_idx;
    int offset_w_idx = (k_idx * 2 + 1) * spatial + pos_idx;
    int mask_idx = k_idx * spatial + pos_idx;

    float offset_h = offsets[offset_h_idx];
    float offset_w = offsets[offset_w_idx];
    float mask_val = (use_mask && masks) ? masks[mask_idx] : 1.0f;

    float h_in = h_col * stride_h - pad_h + kh * dilation_h + offset_h;
    float w_in = w_col * stride_w - pad_w + kw * dilation_w + offset_w;

    float grad = col_grad[c_col * (out_h * out_w) + h_col * out_w + w_col] * mask_val;

    if (h_in > -1.0f && h_in < height && w_in > -1.0f && w_in < width) {
        int y_low = floorf(h_in), x_low = floorf(w_in);
        int y_high = y_low + 1, x_high = x_low + 1;
        float ly = h_in - y_low, lx = w_in - x_low;
        float hy = 1.0f - ly, hx = 1.0f - lx;

        float* im_ptr = grad_input + (b * channels + c_im) * height * width;
        if (y_low >= 0 && x_low >= 0 && y_low < height && x_low < width)
            atomicAdd(&im_ptr[y_low * width + x_low], grad * hy * hx);
        if (y_low >= 0 && x_high < width && x_high >= 0 && y_low < height)
            atomicAdd(&im_ptr[y_low * width + x_high], grad * hy * lx);
        if (y_high < height && x_low >= 0 && y_high >= 0 && x_low < width)
            atomicAdd(&im_ptr[y_high * width + x_low], grad * ly * hx);
        if (y_high < height && x_high < width && y_high >= 0 && x_high >= 0)
            atomicAdd(&im_ptr[y_high * width + x_high], grad * ly * lx);
    }
}

/**
 * @brief Debug helper to print offset statistics (only when --trace is enabled)
 */
static void print_deform_stats_gpu(const char* label, float* data_gpu, int count, int layer_idx)
{
    float* data_cpu = (float*)xcalloc(count, sizeof(float));
    cuda_pull_array(data_gpu, data_cpu, count);

    float sum = 0, sum_sq = 0, min_val = FLT_MAX, max_val = -FLT_MAX;
    int nonzero = 0;
    for (int i = 0; i < count; i++) {
        float v = data_cpu[i];
        sum += v;
        sum_sq += v * v;
        if (v < min_val) min_val = v;
        if (v > max_val) max_val = v;
        if (fabsf(v) > 1e-6f) nonzero++;
    }
    float mean = sum / count;
    float variance = sum_sq / count - mean * mean;
    float std = variance > 0 ? sqrtf(variance) : 0;

    *cfg_and_state.output
        << "deform_conv L" << layer_idx << " " << label
        << ": mean=" << std::fixed << std::setprecision(6) << mean
        << " std=" << std
        << " min=" << min_val
        << " max=" << max_val
        << " nonzero=" << nonzero << "/" << count
        << std::endl;

    free(data_cpu);
}

/**
 * @brief Forward pass for deformable convolutional layer (GPU version)
 *
 * TEACHING MOMENT: Let's follow the data through the GPU forward pass!
 * 1. OFFSET PREDICTION: We run a kernel to compute where each sampling point
 *    should move based on the current input.
 * 2. STABILIZATION (CRITICAL!): Deformable sampling can easily explode or
 *    produce NaN values if the offsets get too large. We normalize and clamp them.
 * 3. MASK COMPUTATION (DCNv2): If enabled, we compute a 0-1 "importance" mask
 *    for each kernel point.
 * 4. THE CONVOLUTION: Finally, we call the deformable kernel which does the
 *    actual bilinear sampling, masking, and weight multiplication.
 *
 * Implements DCNv2 when use_mask=1 (modulated deformable convolution).
 */
void forward_deform_conv_layer_gpu(Darknet::Layer &l, Darknet::NetworkState state)
{
    TAT(TATPARMS);

    int offset_filters = 2 * l.size * l.size;
    int mask_filters = l.size * l.size;
    int batch = l.batch;
    int out_h = l.out_h;
    int out_w = l.out_w;
    int spatial_size = out_h * out_w;
    int threads = 512;

    // STEP 1: PREVENT EXPLOSIONS
    // We start by sanitizing the input. If previous layers have NaNs or huge values,
    // the offset convolution will fail immediately.
    int input_size = batch * l.c * l.h * l.w;
    fix_nan_and_inf(state.input, input_size);
    constrain_ongpu(input_size, 10000.0f, state.input, 1);

    // STEP 2: PREDICT THE OFFSETS
    // This kernel acts like a mini convolutional layer that outputs (dy, dx)
    // for every filter point (e.g., 3x3=9 points, so 18 channels).
    int num_offset_elements = batch * spatial_size * offset_filters;
    int blocks = (num_offset_elements + threads - 1) / threads;

    compute_offsets_kernel<<<blocks, threads, 0, get_cuda_stream()>>>(
        state.input, l.offset_weights_gpu, l.offset_biases_gpu, l.offsets_gpu,
        batch, l.c, l.h, l.w, out_h, out_w, l.size, l.pad * l.dilation, l.stride_x, l.stride_y,
        l.dilation, offset_filters
    );
    CHECK_CUDA(cudaPeekAtLastError());
    check_nan_gpu("forward: after compute_offsets_kernel", l.offsets_gpu, num_offset_elements, l.index);

    // STEP 3: NORMALIZE THE OFFSETS (IF THEY'RE TOO BIG)
    // Deformable convolutions are very sensitive. If they move too far from
    // the original pixel, the gradients can become unstable. We use Instance Norm
    // here to pull extreme offsets back toward the center of the window.
    fix_nan_and_inf(l.offsets_gpu, num_offset_elements);

    float max_offset = (float)(l.size * l.dilation * 2);
    float threshold = max_offset;
    constrain_ongpu(num_offset_elements, 10000.0f, l.offsets_gpu, 1);

    {
        float *d_sum = state.workspace;
        float *d_sum_sq = state.workspace + 1;
        CHECK_CUDA(cudaMemsetAsync(d_sum, 0, sizeof(float), get_cuda_stream()));
        CHECK_CUDA(cudaMemsetAsync(d_sum_sq, 0, sizeof(float), get_cuda_stream()));

        compute_offset_stats_kernel<<<blocks, threads, 0, get_cuda_stream()>>>(
            l.offsets_gpu, num_offset_elements, d_sum, d_sum_sq
        );
        CHECK_CUDA(cudaPeekAtLastError());

        float h_sum, h_sum_sq;
        CHECK_CUDA(cudaMemcpy(&h_sum, d_sum, sizeof(float), cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(&h_sum_sq, d_sum_sq, sizeof(float), cudaMemcpyDeviceToHost));

        float mean = h_sum / num_offset_elements;
        float var = h_sum_sq / num_offset_elements - mean * mean;
        float std = sqrtf(var + 1e-5f);

        if (std > threshold || fabsf(mean) > threshold) {
            float inv_std = 1.0f / std;
            instance_norm_offset_kernel<<<blocks, threads, 0, get_cuda_stream()>>>(
                l.offsets_gpu, num_offset_elements, mean, inv_std
            );
            CHECK_CUDA(cudaPeekAtLastError());
        }
    }

    // SOFT-CLAMP: Ensure offsets don't "shoot off" into infinity.
    clamp_offsets_kernel<<<blocks, threads, 0, get_cuda_stream()>>>(
        l.offsets_gpu, num_offset_elements, max_offset
    );
    CHECK_CUDA(cudaPeekAtLastError());
    check_nan_gpu("forward: after clamp_offsets_kernel", l.offsets_gpu, num_offset_elements, l.index);

    // STEP 4: PREDICT THE MASKS (DCNv2 ONLY)
    // This predicts how much to "trust" each sampled point. Sigmoid maps the output to [0, 1].
    if (l.use_mask) {
        compute_offsets_kernel<<<blocks, threads, 0, get_cuda_stream()>>>(
            state.input, l.mask_weights_gpu, l.mask_biases_gpu, l.masks_gpu,
            batch, l.c, l.h, l.w, out_h, out_w, l.size, l.pad * l.dilation, l.stride_x, l.stride_y,
            l.dilation, mask_filters
        );
        CHECK_CUDA(cudaPeekAtLastError());

        int num_mask_elements = batch * spatial_size * mask_filters;
        fix_nan_and_inf(l.masks_gpu, num_mask_elements);

        blocks = (num_mask_elements + threads - 1) / threads;
        sigmoid_kernel<<<blocks, threads, 0, get_cuda_stream()>>>(l.masks_gpu, num_mask_elements);
        CHECK_CUDA(cudaPeekAtLastError());
        check_nan_gpu("forward: after sigmoid_kernel (masks)", l.masks_gpu, num_mask_elements, l.index);
    }

    // STEP 5: FINAL DEFORMABLE CONVOLUTION
    // This is the main event! It reads the offsets and masks we just computed,
    // does the fractional grid sampling, and produces the output feature map.
    int num_output_elements = batch * l.out_c * spatial_size;
    blocks = (num_output_elements + threads - 1) / threads;

    int use_bias = l.batch_normalize ? 0 : 1;
    deform_conv_kernel<<<blocks, threads, 0, get_cuda_stream()>>>(
        state.input, l.weights_gpu, l.biases_gpu, l.offsets_gpu,
        l.use_mask ? l.masks_gpu : nullptr, l.output_gpu,
        batch, l.c, l.h, l.w, l.out_c, out_h, out_w, l.size, l.pad * l.dilation,
        l.stride_x, l.stride_y, l.dilation, l.groups, l.use_mask, use_bias
    );
    CHECK_CUDA(cudaPeekAtLastError());
    check_nan_gpu("forward: after deform_conv_kernel", l.output_gpu, num_output_elements, l.index);

    // STEP 6: WRAP UP
    // We finalize the output by applying Batch Norm and the Activation function.
    fix_nan_and_inf(l.output_gpu, num_output_elements);
    constrain_ongpu(num_output_elements, 10000.0f, l.output_gpu, 1);

    if (l.batch_normalize) {
        forward_batchnorm_layer_gpu(l, state);
    }

    activate_array_ongpu(l.output_gpu, l.outputs * batch, l.activation);
    fix_nan_and_inf(l.output_gpu, l.outputs * batch);
}

/**
 * @brief Backward pass for deformable convolutional layer (GPU version)
 *
 * TEACHING MOMENT: Reversing the data flow!
 * 1. DELTA CLEANUP: We start with the gradient (delta) from the next layer.
 * 2. ACTIVATION & BN: Reverse the non-linear activation and batch normalization.
 * 3. IM2COL: Re-create the same "sampled columns" we used in the forward pass.
 * 4. WEIGHT UPDATES: Multiply the deltas by the sampled columns to update
 *    the main convolution weights.
 * 5. COORDINATE BACKPROP: This is the hard part! We calculate how changing
 *    the OFFSETS would have changed the final result.
 * 6. OFFSET WEIGHT UPDATES: Finally, update the small network that predicts
 *    the offsets.
 */
void backward_deform_conv_layer_gpu(Darknet::Layer &l, Darknet::NetworkState state)
{
    TAT(TATPARMS);

    int out_h = l.out_h;
    int out_w = l.out_w;
    int offset_filters = 2 * l.size * l.size;
    int mask_filters = l.size * l.size;
    int kernel_size = l.size * l.size;
    int spatial = out_h * out_w;
    int threads = 512;

    // STEP 1: INITIAL CLEANUP
    // Fix NaNs in the incoming gradient (delta). If we don't, everything will become NaN.
    fix_nan_and_inf(l.delta_gpu, l.outputs * l.batch);

    // STEP 2: ACTIVATION GRADIENT
    // Calculate how the activation function (like Leaky ReLU) affects the gradient.
    gradient_array_ongpu(l.output_gpu, l.outputs * l.batch, l.activation, l.delta_gpu);

    if (state.net.try_fix_nan) {
        reset_nan_and_inf(l.delta_gpu, l.outputs * l.batch);
        constrain_ongpu(l.outputs * l.batch, 1.0f, l.delta_gpu, 1);
    }
    check_nan_gpu("backward: initial delta", l.delta_gpu, l.outputs * l.batch, l.index);

    // STEP 3: BATCH NORM GRADIENT
    // Reverse the normalization process if this layer uses it.
    if (l.batch_normalize) {
        backward_batchnorm_layer_gpu(l, state);
    } else {
        backward_bias_gpu(l.bias_updates_gpu, l.delta_gpu, l.batch, l.n, spatial);
    }

    // STEP 4: PREPARE GRADIENT BUFFERS
    fill_ongpu(l.batch * spatial * offset_filters, 0, l.offset_deltas_gpu, 1);
    if (l.use_mask) {
        fill_ongpu(l.batch * spatial * mask_filters, 0, l.mask_deltas_gpu, 1);
    }

    int m = l.n / l.groups;
    int n = l.size * l.size * l.c / l.groups;
    int k = spatial;
    int group_channels = l.c / l.groups;
    int group_input_size = group_channels * l.h * l.w;

    // STEP 5: LOOP THROUGH THE BATCH
    for (int b = 0; b < l.batch; ++b) {
        float *input = state.input + b * l.c * l.h * l.w;
        float *offset = l.offsets_gpu + b * spatial * offset_filters;
        float *mask = l.use_mask ? (l.masks_gpu + b * spatial * mask_filters) : nullptr;
        float *offset_delta = l.offset_deltas_gpu + b * spatial * offset_filters;
        float *mask_delta = l.use_mask ? (l.mask_deltas_gpu + b * spatial * mask_filters) : nullptr;

        for (int g = 0; g < l.groups; ++g) {
            float *im = input + g * group_channels * l.h * l.w;
            float *delta_out = l.delta_gpu + (b * l.groups + g) * m * k;
            float *weights = l.weights_gpu + g * l.nweights / l.groups;
            float *weight_updates = l.weight_updates_gpu + g * l.nweights / l.groups;
            float *col = state.workspace;

            // STEP 6: RE-SAMPLE THE INPUT
            // We need the same column matrix we had in the forward pass to update the weights.
            deformable_im2col_gpu(im, offset, mask,
                                  group_channels, l.h, l.w,
                                  l.size, l.size,
                                  l.pad * l.dilation, l.pad * l.dilation,
                                  l.stride_y, l.stride_x,
                                  l.dilation, l.dilation,
                                  out_h, out_w, col);

            // STEP 7: MAIN WEIGHT UPDATES
            // dy * input^T (Standard convolution update rule).
            if (!state.net.adversarial && !l.train_only_bn) {
                gemm_ongpu(0, 1, m, n, k, 1, delta_out, k, col, k, 1, weight_updates, n);
            }

            // STEP 8: PROJECT GRADIENTS BACK TO SAMPLED SPACE
            // weights^T * dy (Projecting gradients back to the sampled pixels).
            gemm_ongpu(1, 0, n, k, m, 1, weights, n, delta_out, k, 0, col, k);
            reset_nan_and_inf(col, n * k);

            // STEP 9: BACKPROP THROUGH THE SAMPLER (THE COORDINATE GRADIENTS)
            // This is the "chain rule" part that tells the offset network how to
            // adjust the dy/dx values to improve the loss.
            int num_coord = spatial * kernel_size;
            int blocks = (num_coord + threads - 1) / threads;
            deformable_col2im_coord_gpu_kernel<<<blocks, threads, 0, get_cuda_stream()>>>(
                col, im, offset, mask,
                1, group_channels, l.h, l.w, l.size, l.size,
                l.pad * l.dilation, l.pad * l.dilation,
                l.stride_y, l.stride_x, l.dilation, l.dilation,
                out_h, out_w, l.use_mask,
                offset_delta, mask_delta
            );
            CHECK_CUDA(cudaPeekAtLastError());
            check_nan_gpu("backward: after deformable_col2im_coord", offset_delta, spatial * offset_filters, l.index);

            // Stabilize the gradients for offsets/masks.
            reset_nan_and_inf(offset_delta, spatial * offset_filters);
            constrain_ongpu(spatial * offset_filters, 1.0f, offset_delta, 1);
            if (l.use_mask && mask_delta) {
                reset_nan_and_inf(mask_delta, spatial * mask_filters);
                constrain_ongpu(spatial * mask_filters, 1.0f, mask_delta, 1);
            }

            // STEP 10: BACKPROP TO THE PREVIOUS LAYER
            // We need to pass the gradients down so earlier layers can learn too.
            if (state.delta) {
                float *im_delta = state.delta + (b * l.groups + g) * group_input_size;
                fill_ongpu(group_input_size, 0, im_delta, 1);
                int num_col = group_channels * kernel_size * spatial;
                blocks = (num_col + threads - 1) / threads;
                deformable_col2im_gpu_kernel<<<blocks, threads, 0, get_cuda_stream()>>>(
                    col, offset, mask,
                    1, group_channels, l.h, l.w, l.size, l.size,
                    l.pad * l.dilation, l.pad * l.dilation,
                    l.stride_y, l.stride_x, l.dilation, l.dilation,
                    out_h, out_w, l.use_mask,
                    im_delta
                );
                CHECK_CUDA(cudaPeekAtLastError());

                if (state.net.try_fix_nan) {
                    reset_nan_and_inf(im_delta, group_input_size);
                    constrain_ongpu(group_input_size, 10.0f, im_delta, 1);
                }
            }
        }

        // STEP 11: APPLY LHTAN DERIVATIVE TO OFFSETS
        // Remember we soft-clamped the offsets? We must account for that here.
        float max_offset = (float)(l.size * l.dilation * 2);
        int num_offset_elements = spatial * offset_filters;
        int lhtan_blocks = (num_offset_elements + threads - 1) / threads;
        lhtan_gradient_kernel<<<lhtan_blocks, threads, 0, get_cuda_stream()>>>(
            offset_delta, offset, num_offset_elements, max_offset
        );
        CHECK_CUDA(cudaPeekAtLastError());

        // STEP 12: UPDATE THE OFFSET PREDICTION NETWORK
        // We calculate the gradients for the weights inside the internal
        // offset/mask convolutional filters.
        if (!state.net.adversarial && !l.train_only_bn) {
            im2col_gpu_ext(input,
                           l.c, l.h, l.w,
                           l.size, l.size,
                           l.pad * l.dilation, l.pad * l.dilation,
                           l.stride_y, l.stride_x,
                           l.dilation, l.dilation,
                           state.workspace);

            int offset_k = l.c * l.size * l.size;
            gemm_ongpu(0, 1, offset_filters, offset_k, k, 1,
                       offset_delta, k, state.workspace, k, 1,
                       l.offset_weight_updates_gpu, offset_k);
            backward_bias_gpu(l.offset_bias_updates_gpu, offset_delta, 1, offset_filters, k);
            check_nan_gpu("backward: after offset GEMM", l.offset_weight_updates_gpu, offset_k, l.index);

            if (l.use_mask) {
                gemm_ongpu(0, 1, mask_filters, offset_k, k, 1,
                           mask_delta, k, state.workspace, k, 1,
                           l.mask_weight_updates_gpu, offset_k);
                backward_bias_gpu(l.mask_bias_updates_gpu, mask_delta, 1, mask_filters, k);
            }
        }
    }

    // STEP 13: FINAL STABILIZATION
    // One last check for NaNs before we're done with the backward pass.
    int offset_nweights = l.c * offset_filters * l.size * l.size;
    reset_nan_and_inf(l.offset_weight_updates_gpu, offset_nweights);
    reset_nan_and_inf(l.offset_bias_updates_gpu, offset_filters);
    constrain_ongpu(offset_nweights, 1.0f, l.offset_weight_updates_gpu, 1);
    constrain_ongpu(offset_filters, 1.0f, l.offset_bias_updates_gpu, 1);
    if (l.use_mask) {
        int mask_nweights = l.c * mask_filters * l.size * l.size;
        reset_nan_and_inf(l.mask_weight_updates_gpu, mask_nweights);
        reset_nan_and_inf(l.mask_bias_updates_gpu, mask_filters);
        constrain_ongpu(mask_nweights, 1.0f, l.mask_weight_updates_gpu, 1);
        constrain_ongpu(mask_filters, 1.0f, l.mask_bias_updates_gpu, 1);
    }
}

/**
 * @brief Update weights for deformable convolutional layer (GPU version)
 *
 * Updates all learnable parameters with SGD + momentum + weight decay:
 * - Main convolution weights and biases
 * - Offset convolution weights and biases
 * - DCNv2: Mask convolution weights and biases
 * - Batch normalization scales (if enabled)
 */
void update_deform_conv_layer_gpu(Darknet::Layer &l, int batch, float learning_rate, float momentum, float decay, float loss_scale)
{
    TAT(TATPARMS);

    float rate = (learning_rate * l.learning_rate_scale) / (batch * loss_scale);
    // Offset/mask weights use much lower learning rate (0.01x for stability)
    float offset_rate = rate * 0.01f;
    int offset_filters = 2 * l.size * l.size;
    int mask_filters = l.size * l.size;
    int offset_nweights = l.c * offset_filters * l.size * l.size;
    int mask_nweights = l.c * mask_filters * l.size * l.size;

    // Fix NaN/Inf in weights (like regular conv)
    reset_nan_and_inf(l.weight_updates_gpu, l.nweights);
    fix_nan_and_inf(l.weights_gpu, l.nweights);

    reset_nan_and_inf(l.offset_weight_updates_gpu, offset_nweights);
    fix_nan_and_inf(l.offset_weights_gpu, offset_nweights);
    if (l.use_mask) {
        reset_nan_and_inf(l.mask_weight_updates_gpu, mask_nweights);
        fix_nan_and_inf(l.mask_weights_gpu, mask_nweights);
    }

    // Main convolution weights
    axpy_ongpu(l.nweights, -decay * batch * loss_scale, l.weights_gpu, 1, l.weight_updates_gpu, 1);
    axpy_ongpu(l.nweights, rate, l.weight_updates_gpu, 1, l.weights_gpu, 1);
    scal_ongpu(l.nweights, momentum, l.weight_updates_gpu, 1);

    // Main convolution biases
    axpy_ongpu(l.n, rate, l.bias_updates_gpu, 1, l.biases_gpu, 1);
    scal_ongpu(l.n, momentum, l.bias_updates_gpu, 1);

    // Offset convolution weights and biases (0.1x learning rate)
    axpy_ongpu(offset_nweights, -decay * batch * loss_scale, l.offset_weights_gpu, 1, l.offset_weight_updates_gpu, 1);
    axpy_ongpu(offset_nweights, offset_rate, l.offset_weight_updates_gpu, 1, l.offset_weights_gpu, 1);
    scal_ongpu(offset_nweights, momentum, l.offset_weight_updates_gpu, 1);

    axpy_ongpu(offset_filters, offset_rate, l.offset_bias_updates_gpu, 1, l.offset_biases_gpu, 1);
    scal_ongpu(offset_filters, momentum, l.offset_bias_updates_gpu, 1);

    // DCNv2: Mask convolution weights and biases (0.1x learning rate)
    if (l.use_mask) {
        axpy_ongpu(mask_nweights, -decay * batch * loss_scale, l.mask_weights_gpu, 1, l.mask_weight_updates_gpu, 1);
        axpy_ongpu(mask_nweights, offset_rate, l.mask_weight_updates_gpu, 1, l.mask_weights_gpu, 1);
        scal_ongpu(mask_nweights, momentum, l.mask_weight_updates_gpu, 1);

        axpy_ongpu(mask_filters, offset_rate, l.mask_bias_updates_gpu, 1, l.mask_biases_gpu, 1);
        scal_ongpu(mask_filters, momentum, l.mask_bias_updates_gpu, 1);
    }

    // Batch normalization scales
    if (l.batch_normalize) {
        axpy_ongpu(l.n, rate, l.scale_updates_gpu, 1, l.scales_gpu, 1);
        scal_ongpu(l.n, momentum, l.scale_updates_gpu, 1);
    }
}

/**
 * @brief Push weights from CPU to GPU for deformable convolutional layer
 */
void push_deform_conv_layer(Darknet::Layer &l)
{
    int offset_filters = 2 * l.size * l.size;
    int mask_filters = l.size * l.size;
    int offset_nweights = l.c * offset_filters * l.size * l.size;
    int mask_nweights = l.c * mask_filters * l.size * l.size;

    cuda_push_array(l.weights_gpu, l.weights, l.nweights);
    cuda_push_array(l.biases_gpu, l.biases, l.n);

    cuda_push_array(l.offset_weights_gpu, l.offset_weights, offset_nweights);
    cuda_push_array(l.offset_biases_gpu, l.offset_biases, offset_filters);

    if (l.use_mask) {
        cuda_push_array(l.mask_weights_gpu, l.mask_weights, mask_nweights);
        cuda_push_array(l.mask_biases_gpu, l.mask_biases, mask_filters);
    }

    if (l.batch_normalize) {
        cuda_push_array(l.scales_gpu, l.scales, l.n);
        cuda_push_array(l.rolling_mean_gpu, l.rolling_mean, l.n);
        cuda_push_array(l.rolling_variance_gpu, l.rolling_variance, l.n);
    }
}

/**
 * @brief Pull weights from GPU to CPU for deformable convolutional layer
 */
void pull_deform_conv_layer(Darknet::Layer &l)
{
    int offset_filters = 2 * l.size * l.size;
    int mask_filters = l.size * l.size;
    int offset_nweights = l.c * offset_filters * l.size * l.size;
    int mask_nweights = l.c * mask_filters * l.size * l.size;

    cuda_pull_array(l.weights_gpu, l.weights, l.nweights);
    cuda_pull_array(l.biases_gpu, l.biases, l.n);

    cuda_pull_array(l.offset_weights_gpu, l.offset_weights, offset_nweights);
    cuda_pull_array(l.offset_biases_gpu, l.offset_biases, offset_filters);

    if (l.use_mask) {
        cuda_pull_array(l.mask_weights_gpu, l.mask_weights, mask_nweights);
        cuda_pull_array(l.mask_biases_gpu, l.mask_biases, mask_filters);
    }

    if (l.batch_normalize) {
        cuda_pull_array(l.scales_gpu, l.scales, l.n);
        cuda_pull_array(l.rolling_mean_gpu, l.rolling_mean, l.n);
        cuda_pull_array(l.rolling_variance_gpu, l.rolling_variance, l.n);
    }
}
