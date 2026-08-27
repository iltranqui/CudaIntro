#include "darknet_internal.hpp"
#include "deform_conv_layer.hpp"
#include "activations.hpp"
#include "gemm.hpp"
#include "blas.hpp"
#include "batchnorm_layer.hpp"
#include "im2col.hpp"

namespace
{
    static auto & cfg_and_state = Darknet::CfgAndState::get();
    constexpr float kDeformFiniteClamp = 10000.0f;
    constexpr float kDeformSigmoidClamp = 80.0f;
    constexpr double kDeformMinStddev = 1e-12;

    inline float sanitize_and_clamp_value(float value, float abs_limit = kDeformFiniteClamp)
    {
        if (!std::isfinite(value))
        {
            return 0.0f;
        }
        if (value > abs_limit)
        {
            return abs_limit;
        }
        if (value < -abs_limit)
        {
            return -abs_limit;
        }
        return value;
    }

    void sanitize_and_clamp_array(float *data, int n, float abs_limit = kDeformFiniteClamp)
    {
        for (int i = 0; i < n; ++i)
        {
            data[i] = sanitize_and_clamp_value(data[i], abs_limit);
        }
    }

    void compute_mean_and_stddev(const float *data, int n, double &mean_out, double &stddev_out)
    {
        if (n <= 0)
        {
            mean_out = 0.0;
            stddev_out = 0.0;
            return;
        }

        double sum = 0.0;
        double sum_sq = 0.0;
        for (int i = 0; i < n; ++i)
        {
            const double v = data[i];
            sum += v;
            sum_sq += v * v;
        }

        mean_out = sum / static_cast<double>(n);
        double variance = sum_sq / static_cast<double>(n) - mean_out * mean_out;
        if (variance < 0.0)
        {
            variance = 0.0;
        }
        stddev_out = std::sqrt(variance);
    }

    inline float stable_sigmoid(float value)
    {
        if (!std::isfinite(value))
        {
            return 0.5f;
        }

        if (value >= 0.0f)
        {
            const float clamped = (value > kDeformSigmoidClamp) ? kDeformSigmoidClamp : value;
            const float z = expf(-clamped);
            return 1.0f / (1.0f + z);
        }

        const float clamped = (value < -kDeformSigmoidClamp) ? -kDeformSigmoidClamp : value;
        const float z = expf(clamped);
        return z / (1.0f + z);
    }

    inline float centered_lhtan_offset(float raw_offset, float max_offset)
    {
        if (!std::isfinite(raw_offset) || !std::isfinite(max_offset) || max_offset <= 0.0f)
        {
            return 0.0f;
        }

        const float scaled = raw_offset / (2.0f * max_offset) + 0.5f;
        const float normalized = 2.0f * lhtan_activate(scaled) - 1.0f;
        return sanitize_and_clamp_value(normalized * max_offset);
    }

    inline float centered_lhtan_gradient_from_offset(float offset, float max_offset)
    {
        if (!std::isfinite(offset) || !std::isfinite(max_offset) || max_offset <= 0.0f)
        {
            return 0.0f;
        }

        return (offset < -max_offset || offset > max_offset) ? 0.001f : 1.0f;
    }

#ifdef CUDNN
    void setup_deform_conv_batchnorm_cudnn(Darknet::Layer *l)
    {
        if (l->normTensorDesc == nullptr) {
            CHECK_CUDNN(cudnnCreateTensorDescriptor(&l->normTensorDesc));
        }
        if (l->normDstTensorDesc == nullptr) {
            CHECK_CUDNN(cudnnCreateTensorDescriptor(&l->normDstTensorDesc));
        }
        if (l->normDstTensorDescF16 == nullptr) {
            CHECK_CUDNN(cudnnCreateTensorDescriptor(&l->normDstTensorDescF16));
        }

        CHECK_CUDNN(cudnnSetTensor4dDescriptor(l->normTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, l->out_c, 1, 1));
        CHECK_CUDNN(cudnnSetTensor4dDescriptor(l->normDstTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, l->batch, l->out_c, l->out_h, l->out_w));
        CHECK_CUDNN(cudnnSetTensor4dDescriptor(l->normDstTensorDescF16, CUDNN_TENSOR_NCHW, CUDNN_DATA_HALF, l->batch, l->out_c, l->out_h, l->out_w));
    }
#endif
}

/**
 * @brief Create a deformable convolutional layer
 *
 * This function initializes a deformable convolutional layer with the specified parameters.
 * Deformable convolutions extend standard convolutions by adding learnable offsets to the
 * sampling grid, allowing the network to adaptively adjust its receptive field.
 */

/**
 * @brief Get workspace size for deformable convolutional layer
 *
 * Calculates the workspace size needed for forward/backward pass.
 * Needs to accommodate the largest im2col matrix: (c * kernel_size) x (out_h * out_w)
 */
size_t get_deform_conv_workspace_size(const Darknet::Layer & l)
{
    // The offset convolution uses full channels for im2col
    size_t workspace_size = (size_t)l.out_h * l.out_w * l.size * l.size * l.c * sizeof(float);
    return workspace_size;
}

/**
 * @brief Create a deformable convolutional layer
 *
 * This function initializes a deformable convolutional layer with the specified parameters.
 * Deformable convolutions extend standard convolutions by adding learnable offsets to the
 * sampling grid, allowing the network to adaptively adjust its receptive field.
 */
Darknet::Layer make_deform_conv_layer(int batch, int steps, int h, int w, int c, int n, int groups,
                                      int size, int stride_x, int stride_y, int dilation,
                                      int padding, ACTIVATION activation, int batch_normalize,
                                      int binary, int xnor, int adam, int use_bin_output,
                                      int index, int antialiasing, Darknet::Layer *share_layer,
                                      int assisted_excitation, int train, int use_mask)
{
    TAT(TATPARMS);

    int total_batch = batch * steps;
    Darknet::Layer l = { (Darknet::ELayerType)0 };
    l.type = Darknet::ELayerType::DEFORM_CONV;
    l.train = train;

    if (xnor) groups = 1;   // disable groups for XNOR-net
    if (groups < 1) groups = 1;

    l.groups = groups;
    l.batch = batch;
    l.h = h;
    l.w = w;
    l.c = c;
    l.n = n;
    l.binary = binary;
    l.xnor = xnor;
    l.use_bin_output = use_bin_output;
    l.batch_normalize = batch_normalize;
    l.antialiasing = antialiasing;
    l.assisted_excitation = assisted_excitation;
    l.size = size;
    l.stride_x = stride_x;
    l.stride_y = stride_y;
    l.dilation = dilation;
    l.pad = padding;
    l.index = index;

    // Deformable convolution specific parameters
    l.deform = 1;  // Mark as deformable convolution
    l.extra = 1;   // Use extra field to indicate this is a deformable convolution

    l.out_w = (w + 2 * padding - dilation * (size - 1) - 1) / stride_x + 1;
    l.out_h = (h + 2 * padding - dilation * (size - 1) - 1) / stride_y + 1;
    l.out_c = n;
    l.outputs = l.out_h * l.out_w * l.out_c;
    l.inputs = l.w * l.h * l.c;
    l.activation = activation;
    l.nweights = (c / groups) * n * size * size;

    l.workspace_size = get_deform_conv_workspace_size(l);

    // Allocate memory for weights and biases
    // Main convolution weights: (c/groups) * n * size * size
    l.weights = (float*)xcalloc(l.nweights, sizeof(float));
    l.weight_updates = (float*)xcalloc(l.nweights, sizeof(float));

    // He initialization for main weights (like regular conv)
    float scale = sqrt(2.0f / (size * size * c / groups));
    for (int i = 0; i < l.nweights; ++i) {
        l.weights[i] = scale * rand_uniform_weight_init(-1.0f, 1.0f);
    }

    l.biases = (float*)xcalloc(n, sizeof(float));
    l.bias_updates = (float*)xcalloc(n, sizeof(float));

    // Allocate memory for outputs and deltas
    l.output = (float*)xcalloc(total_batch * l.outputs, sizeof(float));
    l.delta = (float*)xcalloc(total_batch * l.outputs, sizeof(float));

    // Allocate memory for deformable convolution fields
    // Offset conv weights: offset_filters * (c * size * size)
    // Input to offset conv is the SAME feature map (c channels), kernel size is same (size * size)
    int offset_filters = 2 * size * size;
    size_t offset_weights_size = (size_t)offset_filters * c * size * size;

    l.offset_weights = (float*)xcalloc(offset_weights_size, sizeof(float));
    l.offset_weight_updates = (float*)xcalloc(offset_weights_size, sizeof(float));
    l.offset_biases = (float*)xcalloc(offset_filters, sizeof(float));
    l.offset_bias_updates = (float*)xcalloc(offset_filters, sizeof(float));

    // Small initialization for offset weights (so offsets start near zero)
    // This follows DCN paper recommendation - start with standard conv behavior
    float offset_scale = 0.001f;
    for (size_t i = 0; i < offset_weights_size; ++i) {
        l.offset_weights[i] = offset_scale * rand_uniform_weight_init(-1.0f, 1.0f);
    }
    // Offset biases stay zero so initial offsets are zero
    l.offsets = (float*)xcalloc(total_batch * l.out_h * l.out_w * offset_filters, sizeof(float));
    l.offset_deltas = (float*)xcalloc(total_batch * l.out_h * l.out_w * offset_filters, sizeof(float));

    // DCNv2: Allocate modulation mask arrays when use_mask=1
    l.use_mask = use_mask;
    int mask_filters = size * size;  // K*K mask values per output position
    size_t mask_weights_size = (size_t)mask_filters * c * size * size;

    if (use_mask) {
        l.mask_weights = (float*)xcalloc(mask_weights_size, sizeof(float));
        l.mask_weight_updates = (float*)xcalloc(mask_weights_size, sizeof(float));
        l.mask_biases = (float*)xcalloc(mask_filters, sizeof(float));
        l.mask_bias_updates = (float*)xcalloc(mask_filters, sizeof(float));
        l.masks = (float*)xcalloc(total_batch * l.out_h * l.out_w * mask_filters, sizeof(float));
        l.mask_deltas = (float*)xcalloc(total_batch * l.out_h * l.out_w * mask_filters, sizeof(float));

        // Small initialization for mask weights (so masks start at 0.5 after sigmoid)
        float mask_scale = 0.001f;
        for (size_t i = 0; i < mask_weights_size; ++i) {
            l.mask_weights[i] = mask_scale * rand_uniform_weight_init(-1.0f, 1.0f);
        }
        // Mask biases stay zero so initial masks are 0.5 (sigmoid(0) = 0.5)
    }

    // Set forward and backward functions
    l.forward = forward_deform_conv_layer;
    l.backward = backward_deform_conv_layer;
    l.update = update_deform_conv_layer;

    // Batch normalization
    if (batch_normalize) {
        l.scales = (float*)xcalloc(n, sizeof(float));
        l.scale_updates = (float*)xcalloc(n, sizeof(float));
        for (int i = 0; i < n; ++i) {
            l.scales[i] = 1;
        }

        l.mean = (float*)xcalloc(n, sizeof(float));
        l.variance = (float*)xcalloc(n, sizeof(float));

        l.mean_delta = (float*)xcalloc(n, sizeof(float));
        l.variance_delta = (float*)xcalloc(n, sizeof(float));

        l.rolling_mean = (float*)xcalloc(n, sizeof(float));
        l.rolling_variance = (float*)xcalloc(n, sizeof(float));
        l.x = (float*)xcalloc(total_batch * l.outputs, sizeof(float));
        l.x_norm = (float*)xcalloc(total_batch * l.outputs, sizeof(float));
    }

    // CUDA setup
#ifdef DARKNET_GPU
    l.forward_gpu = forward_deform_conv_layer_gpu;
    l.backward_gpu = backward_deform_conv_layer_gpu;
    l.update_gpu = update_deform_conv_layer_gpu;

    if (cfg_and_state.gpu_index >= 0)
    {
        // Allocate GPU memory
        l.weights_gpu = cuda_make_array(l.weights, l.nweights);
        l.weight_updates_gpu = cuda_make_array(l.weight_updates, l.nweights);

        l.biases_gpu = cuda_make_array(l.biases, n);
        l.bias_updates_gpu = cuda_make_array(l.bias_updates, n);

        l.output_gpu = cuda_make_array(l.output, total_batch * l.outputs);
        l.delta_gpu = cuda_make_array(l.delta, total_batch * l.outputs);

        // Allocate GPU memory for deformable convolution fields
        l.offset_weights_gpu = cuda_make_array(l.offset_weights, offset_weights_size);
        l.offset_weight_updates_gpu = cuda_make_array(l.offset_weight_updates, offset_weights_size);
        l.offset_biases_gpu = cuda_make_array(l.offset_biases, offset_filters);
        l.offset_bias_updates_gpu = cuda_make_array(l.offset_bias_updates, offset_filters);
        l.offsets_gpu = cuda_make_array(l.offsets, total_batch * l.out_h * l.out_w * offset_filters);
        l.offset_deltas_gpu = cuda_make_array(l.offset_deltas, total_batch * l.out_h * l.out_w * offset_filters);

        // DCNv2: GPU memory for modulation mask arrays
        if (use_mask) {
            l.mask_weights_gpu = cuda_make_array(l.mask_weights, mask_weights_size);
            l.mask_weight_updates_gpu = cuda_make_array(l.mask_weight_updates, mask_weights_size);
            l.mask_biases_gpu = cuda_make_array(l.mask_biases, mask_filters);
            l.mask_bias_updates_gpu = cuda_make_array(l.mask_bias_updates, mask_filters);
            l.masks_gpu = cuda_make_array(l.masks, total_batch * l.out_h * l.out_w * mask_filters);
            l.mask_deltas_gpu = cuda_make_array(l.mask_deltas, total_batch * l.out_h * l.out_w * mask_filters);
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

            l.x_gpu = cuda_make_array(l.output, total_batch * l.outputs);
            l.x_norm_gpu = cuda_make_array(l.output, total_batch * l.outputs);

#ifdef CUDNN
            setup_deform_conv_batchnorm_cudnn(&l);
#endif
        }
    }
#endif

    // Log layer creation with DCN version info
    *cfg_and_state.output << "deform_conv" << (use_mask ? "v2" : "v1") << " " << size << " x " << size
                          << " / " << stride_x << " x " << stride_y << ", " << n << " filters, "
                          << l.inputs << " inputs, " << l.outputs << " outputs" << std::endl;

    return l;
}

/**
 * @brief Resize a deformable convolutional layer
 *
 * This function resizes a deformable convolutional layer to handle inputs of a different size.
 */
void resize_deform_conv_layer(Darknet::Layer *l, int w, int h)
{
    TAT(TATPARMS);

    l->w = w;
    l->h = h;

    l->out_w = (w + 2 * l->pad - l->dilation * (l->size - 1) - 1) / l->stride_x + 1;
    l->out_h = (h + 2 * l->pad - l->dilation * (l->size - 1) - 1) / l->stride_y + 1;

    l->outputs = l->out_h * l->out_w * l->out_c;
    l->inputs = l->w * l->h * l->c;

    int total_batch = l->batch;

    // Resize output and delta arrays
    l->output = (float*)xrealloc(l->output, total_batch * l->outputs * sizeof(float));
    l->delta = (float*)xrealloc(l->delta, total_batch * l->outputs * sizeof(float));

    // Resize deformable convolution arrays
    int offset_filters = 2 * l->size * l->size;
    l->offsets = (float*)xrealloc(l->offsets, total_batch * l->out_h * l->out_w * offset_filters * sizeof(float));
    l->offset_deltas = (float*)xrealloc(l->offset_deltas, total_batch * l->out_h * l->out_w * offset_filters * sizeof(float));

    // DCNv2: Resize mask arrays if using modulation
    int mask_filters = l->size * l->size;
    if (l->use_mask) {
        l->masks = (float*)xrealloc(l->masks, total_batch * l->out_h * l->out_w * mask_filters * sizeof(float));
        l->mask_deltas = (float*)xrealloc(l->mask_deltas, total_batch * l->out_h * l->out_w * mask_filters * sizeof(float));
    }

    // Resize batch normalization arrays if needed
    if (l->batch_normalize) {
        l->x = (float*)xrealloc(l->x, total_batch * l->outputs * sizeof(float));
        l->x_norm = (float*)xrealloc(l->x_norm, total_batch * l->outputs * sizeof(float));
    }

#ifdef DARKNET_GPU
    // Resize GPU arrays
    cuda_free(l->output_gpu);
    cuda_free(l->delta_gpu);

    l->output_gpu = cuda_make_array(l->output, total_batch * l->outputs);
    l->delta_gpu = cuda_make_array(l->delta, total_batch * l->outputs);

    // Resize GPU arrays for deformable convolution
    cuda_free(l->offsets_gpu);
    cuda_free(l->offset_deltas_gpu);

    l->offsets_gpu = cuda_make_array(l->offsets, total_batch * l->out_h * l->out_w * offset_filters);
    l->offset_deltas_gpu = cuda_make_array(l->offset_deltas, total_batch * l->out_h * l->out_w * offset_filters);

    // DCNv2: Resize GPU mask arrays
    if (l->use_mask) {
        cuda_free(l->masks_gpu);
        cuda_free(l->mask_deltas_gpu);
        l->masks_gpu = cuda_make_array(l->masks, total_batch * l->out_h * l->out_w * mask_filters);
        l->mask_deltas_gpu = cuda_make_array(l->mask_deltas, total_batch * l->out_h * l->out_w * mask_filters);
    }

    if (l->batch_normalize) {
        cuda_free(l->x_gpu);
        cuda_free(l->x_norm_gpu);

        l->x_gpu = cuda_make_array(l->output, total_batch * l->outputs);
        l->x_norm_gpu = cuda_make_array(l->output, total_batch * l->outputs);
    }

#ifdef CUDNN
    if (l->batch_normalize) {
        setup_deform_conv_batchnorm_cudnn(l);
    }
#endif
#endif
}

/**
 * @brief Bilinear interpolation for sampling at fractional positions
 *
 * Samples input at (y, x) using bilinear interpolation between 4 neighboring pixels.
 * This is critical for deformable convolution where offsets produce fractional positions.
 *
 * @param data Input feature map (channel x height x width)
 * @param h Input height
 * @param w Input width
 * @param y Vertical sample position (can be fractional)
 * @param x Horizontal sample position (can be fractional)
 * @return Interpolated value at (y, x), or 0 if outside bounds
 */
inline float bilinear_interpolate_cpu(const float* data, int h, int w, float y, float x)
{
    // Return 0 for positions completely outside input bounds
    if (y < -1.0f || y > h || x < -1.0f || x > w) return 0.0f;

    // Compute integer floor coordinates
    int y_low = static_cast<int>(floorf(y));
    int x_low = static_cast<int>(floorf(x));
    int y_high = y_low + 1;
    int x_high = x_low + 1;

    // Compute interpolation weights (distance from floor)
    float ly = y - y_low;
    float lx = x - x_low;
    float hy = 1.0f - ly;
    float hx = 1.0f - lx;

    // Sample 4 corners with complete bounds checking
    float v1 = (y_low >= 0 && y_low < h && x_low >= 0 && x_low < w) ? data[y_low * w + x_low] : 0.0f;
    float v2 = (y_low >= 0 && y_low < h && x_high >= 0 && x_high < w) ? data[y_low * w + x_high] : 0.0f;
    float v3 = (y_high >= 0 && y_high < h && x_low >= 0 && x_low < w) ? data[y_high * w + x_low] : 0.0f;
    float v4 = (y_high >= 0 && y_high < h && x_high >= 0 && x_high < w) ? data[y_high * w + x_high] : 0.0f;

    // Bilinear interpolation formula
    return hy * hx * v1 + hy * lx * v2 + ly * hx * v3 + ly * lx * v4;
}

/**
 * @brief Deformable im2col - transforms input using learned offsets
 *
 * Like im2col but samples at offset positions using bilinear interpolation.
 * For DCNv2: multiplies by mask values for modulated sampling.
 *
 * @param data_im Input feature map (c x h x w)
 * @param data_offset Offset field (out_h x out_w x 2*K*K) containing (dy, dx) per kernel point
 * @param data_mask Modulation mask (out_h x out_w x K*K) or nullptr for DCNv1
 * @param channels Input channels
 * @param height Input height
 * @param width Input width
 * @param kernel_h Kernel height
 * @param kernel_w Kernel width
 * @param pad_h Padding height
 * @param pad_w Padding width
 * @param stride_h Stride height
 * @param stride_w Stride width
 * @param dilation_h Dilation height
 * @param dilation_w Dilation width
 * @param out_h Output height
 * @param out_w Output width
 * @param data_col Output column matrix (c*K*K x out_h*out_w)
 */
void deformable_im2col_cpu(
    const float* data_im,
    const float* data_offset,
    const float* data_mask,
    int channels, int height, int width,
    int kernel_h, int kernel_w,
    int pad_h, int pad_w,
    int stride_h, int stride_w,
    int dilation_h, int dilation_w,
    int out_h, int out_w,
    float* data_col)
{
    // data_col layout: (channels * kernel_h * kernel_w) x (out_h * out_w)
    int kernel_size = kernel_h * kernel_w;
    int col_channels = channels * kernel_size;

    for (int c_col = 0; c_col < col_channels; ++c_col) {
        // Decompose column channel index into input channel and kernel position
        int w_offset = c_col % kernel_w;
        int h_offset = (c_col / kernel_w) % kernel_h;
        int c_im = c_col / kernel_size;

        // Pointer to this channel in input
        const float* data_im_ptr = data_im + c_im * height * width;

        for (int h_col = 0; h_col < out_h; ++h_col) {
            for (int w_col = 0; w_col < out_w; ++w_col) {
                // Base sampling position (without offset)
                float h_in = h_col * stride_h - pad_h + h_offset * dilation_h;
                float w_in = w_col * stride_w - pad_w + w_offset * dilation_w;

                // Get offset index for this kernel position and output location
                int pos_idx = h_col * out_w + w_col;
                int k_idx = h_offset * kernel_w + w_offset;
                int offset_h_idx = (k_idx * 2) * (out_h * out_w) + pos_idx;
                int offset_w_idx = (k_idx * 2 + 1) * (out_h * out_w) + pos_idx;
                float offset_h = data_offset[offset_h_idx];
                float offset_w = data_offset[offset_w_idx];

                // Apply offsets to sampling position
                h_in += offset_h;
                w_in += offset_w;

                // Sample using bilinear interpolation
                float val = bilinear_interpolate_cpu(data_im_ptr, height, width, h_in, w_in);

                // DCNv2: Apply modulation mask if provided
                if (data_mask) {
                    int mask_idx = k_idx * (out_h * out_w) + pos_idx;
                    val *= data_mask[mask_idx];
                }

                // Store in column matrix
                data_col[c_col * (out_h * out_w) + h_col * out_w + w_col] = val;
            }
        }
    }
}

/**
 * @brief Bilinear interpolation gradient - distributes gradient to 4 neighboring pixels
 *
 * Computes gradient contribution from sampling position (y, x) back to the input.
 * Called during backward pass to propagate gradients through deformable sampling.
 *
 * @param data_delta Output delta to accumulate gradients into (h x w)
 * @param h Input height
 * @param w Input width
 * @param y Vertical sample position
 * @param x Horizontal sample position
 * @param grad Gradient value to distribute
 */
inline void bilinear_interpolate_gradient_cpu(float* data_delta, int h, int w, float y, float x, float grad)
{
    if (y < -1.0f || y > h || x < -1.0f || x > w) return;

    int y_low = static_cast<int>(floorf(y));
    int x_low = static_cast<int>(floorf(x));
    int y_high = y_low + 1;
    int x_high = x_low + 1;

    float ly = y - y_low;
    float lx = x - x_low;
    float hy = 1.0f - ly;
    float hx = 1.0f - lx;

    // Distribute gradient to 4 corners (atomic adds for thread safety)
    if (y_low >= 0 && x_low >= 0 && y_low < h && x_low < w)
        data_delta[y_low * w + x_low] += hy * hx * grad;
    if (y_low >= 0 && x_high < w && y_low < h && x_high >= 0)
        data_delta[y_low * w + x_high] += hy * lx * grad;
    if (y_high < h && x_low >= 0 && y_high >= 0 && x_low < w)
        data_delta[y_high * w + x_low] += ly * hx * grad;
    if (y_high < h && x_high < w && y_high >= 0 && x_high >= 0)
        data_delta[y_high * w + x_high] += ly * lx * grad;
}

/**
 * @brief Deformable col2im - backward pass for input gradients
 *
 * Distributes gradients from column matrix back to input using learned offsets.
 * Inverse of deformable_im2col_cpu.
 */
void deformable_col2im_cpu(
    const float* data_col,
    const float* data_offset,
    const float* data_mask,
    int channels, int height, int width,
    int kernel_h, int kernel_w,
    int pad_h, int pad_w,
    int stride_h, int stride_w,
    int dilation_h, int dilation_w,
    int out_h, int out_w,
    float* grad_im)
{
    int kernel_size = kernel_h * kernel_w;
    int col_channels = channels * kernel_size;

    for (int c_col = 0; c_col < col_channels; ++c_col) {
        int w_offset = c_col % kernel_w;
        int h_offset = (c_col / kernel_w) % kernel_h;
        int c_im = c_col / kernel_size;

        float* grad_im_ptr = grad_im + c_im * height * width;

        for (int h_col = 0; h_col < out_h; ++h_col) {
            for (int w_col = 0; w_col < out_w; ++w_col) {
                float h_in = h_col * stride_h - pad_h + h_offset * dilation_h;
                float w_in = w_col * stride_w - pad_w + w_offset * dilation_w;

                int pos_idx = h_col * out_w + w_col;
                int k_idx = h_offset * kernel_w + w_offset;
                int offset_h_idx = (k_idx * 2) * (out_h * out_w) + pos_idx;
                int offset_w_idx = (k_idx * 2 + 1) * (out_h * out_w) + pos_idx;
                h_in += data_offset[offset_h_idx];
                w_in += data_offset[offset_w_idx];

                float grad = data_col[c_col * (out_h * out_w) + h_col * out_w + w_col];
                if (data_mask) {
                    int mask_idx = k_idx * (out_h * out_w) + pos_idx;
                    grad *= data_mask[mask_idx];
                }

                bilinear_interpolate_gradient_cpu(grad_im_ptr, height, width, h_in, w_in, grad);
            }
        }
    }
}

/**
 * @brief Compute gradients w.r.t. offsets and masks (DCNv2)
 *
 * Computes gradients for offset field by differentiating through bilinear interpolation.
 * For DCNv2, also computes gradients for modulation masks.
 */
void deformable_col2im_coord_cpu(
    const float* data_col,
    const float* data_im,
    const float* data_offset,
    const float* data_mask,
    int channels, int height, int width,
    int kernel_h, int kernel_w,
    int pad_h, int pad_w,
    int stride_h, int stride_w,
    int dilation_h, int dilation_w,
    int out_h, int out_w,
    float* grad_offset,
    float* grad_mask)
{
    int kernel_size = kernel_h * kernel_w;

    for (int h_col = 0; h_col < out_h; ++h_col) {
        for (int w_col = 0; w_col < out_w; ++w_col) {
            for (int kh = 0; kh < kernel_h; ++kh) {
                for (int kw = 0; kw < kernel_w; ++kw) {
                    int k_idx = kh * kernel_w + kw;
                    int pos_idx = h_col * out_w + w_col;
                    int offset_h_idx = (k_idx * 2) * (out_h * out_w) + pos_idx;
                    int offset_w_idx = (k_idx * 2 + 1) * (out_h * out_w) + pos_idx;
                    int mask_idx = k_idx * (out_h * out_w) + pos_idx;

                    float h_in = h_col * stride_h - pad_h + kh * dilation_h;
                    float w_in = w_col * stride_w - pad_w + kw * dilation_w;
                    h_in += data_offset[offset_h_idx];
                    w_in += data_offset[offset_w_idx];

                    float mask_val = data_mask ? data_mask[mask_idx] : 1.0f;
                    float grad_h = 0.0f, grad_w = 0.0f, grad_m = 0.0f;

                    if (h_in > -1.0f && h_in < height && w_in > -1.0f && w_in < width) {
                        int y_low = static_cast<int>(floorf(h_in));
                        int x_low = static_cast<int>(floorf(w_in));
                        int y_high = y_low + 1;
                        int x_high = x_low + 1;

                        float ly = h_in - y_low;
                        float lx = w_in - x_low;
                        float hy = 1.0f - ly;
                        float hx = 1.0f - lx;

                        // Accumulate gradients across all input channels
                        for (int c = 0; c < channels; ++c) {
                            const float* im_ptr = data_im + c * height * width;
                            int c_col = c * kernel_size + k_idx;
                            float col_grad = data_col[c_col * (out_h * out_w) + h_col * out_w + w_col];

                            float v1 = (y_low >= 0 && x_low >= 0) ? im_ptr[y_low * width + x_low] : 0.0f;
                            float v2 = (y_low >= 0 && x_high < width) ? im_ptr[y_low * width + x_high] : 0.0f;
                            float v3 = (y_high < height && x_low >= 0) ? im_ptr[y_high * width + x_low] : 0.0f;
                            float v4 = (y_high < height && x_high < width) ? im_ptr[y_high * width + x_high] : 0.0f;

                            // Gradient w.r.t. h (y coordinate)
                            grad_h += col_grad * mask_val * ((-hx) * v1 + (-lx) * v2 + hx * v3 + lx * v4);
                            // Gradient w.r.t. w (x coordinate)
                            grad_w += col_grad * mask_val * ((-hy) * v1 + hy * v2 + (-ly) * v3 + ly * v4);
                            // Gradient w.r.t. mask (sampled value)
                            if (grad_mask) {
                                float val = hy * hx * v1 + hy * lx * v2 + ly * hx * v3 + ly * lx * v4;
                                grad_m += col_grad * val;
                            }
                        }
                    }

                    grad_offset[offset_h_idx] += grad_h;
                    grad_offset[offset_w_idx] += grad_w;
                    if (grad_mask) {
                        // Gradient through sigmoid: grad * mask * (1 - mask)
                        grad_mask[mask_idx] += grad_m * mask_val * (1.0f - mask_val);
                    }
                }
            }
        }
    }
}

/**
 * @brief Forward pass for deformable convolutional layer (CPU version)
 *
 * This function performs the forward pass for a deformable convolutional layer on CPU.
 * Implements DCNv2 when use_mask=1 (modulated deformable convolution).
 *
 * Forward pass steps:
 * 1. Compute offset field: conv(input, offset_weights) + offset_biases
 * 2. DCNv2 only: Compute mask field: sigmoid(conv(input, mask_weights) + mask_biases)
 * 3. Sample input using deformable_im2col with learned offsets (and masks for DCNv2)
 * 4. GEMM: output = weights @ sampled_columns
 * 5. Apply batch normalization or add biases
 * 6. Apply activation function
 */
void forward_deform_conv_layer(Darknet::Layer &l, Darknet::NetworkState state)
{
    TAT(TATPARMS);

    int out_h = l.out_h;
    int out_w = l.out_w;
    int offset_filters = 2 * l.size * l.size;  // (dy, dx) for each kernel position
    int mask_filters = l.size * l.size;         // one scalar per kernel position

    // Fix NaN/Inf and constrain input from previous layers
    // Use larger constraint (10000) to preserve signal while preventing overflow
    // Normal activations are ~1000, explosions reach 400K+
    int input_size = l.batch * l.c * l.h * l.w;
    sanitize_and_clamp_array(state.input, input_size);

    // Initialize output to zero
    fill_cpu(l.outputs * l.batch, 0, l.output, 1);

    // GEMM dimensions for main convolution
    int m = l.n / l.groups;                      // Number of filters per group
    int n = out_h * out_w;                       // Spatial size of output
    int k = l.size * l.size * l.c / l.groups;   // Size of each filter

    for (int b = 0; b < l.batch; ++b) {
        float *input = state.input + b * l.c * l.h * l.w;
        float *offset = l.offsets + b * out_h * out_w * offset_filters;
        float *mask = l.use_mask ? (l.masks + b * out_h * out_w * mask_filters) : nullptr;

        // Step 1: Compute the offset field by convolving the input feature map
        // This predicts (dy, dx) for each kernel point at each output spatial location.
        im2col_cpu_ext(input, l.c, l.h, l.w, l.size, l.size,
                       l.pad * l.dilation, l.pad * l.dilation,
                       l.stride_y, l.stride_x, l.dilation, l.dilation,
                       state.workspace);
        gemm_cpu(0, 0, offset_filters, n, l.c * l.size * l.size, 1,
             l.offset_weights, l.c * l.size * l.size,
             state.workspace, n, 0, offset, n);
        add_bias(offset, l.offset_biases, 1, offset_filters, n);

        // Limit offsets to prevent numerical instability and sampling outside image bounds
        float max_offset = (float)(l.size * l.dilation * 2);
        int offset_spatial = out_h * out_w * offset_filters;

        sanitize_and_clamp_array(offset, offset_spatial);

        // Optional: Perform instance-like normalization if offsets explode
        const float threshold = max_offset;
        double mean = 0.0;
        double stddev = 0.0;
        compute_mean_and_stddev(offset, offset_spatial, mean, stddev);

        if (stddev > kDeformMinStddev &&
            (stddev > static_cast<double>(threshold) || std::fabs(mean) > static_cast<double>(threshold))) {
            const float mean_f = static_cast<float>(mean);
            const float inv_std = 1.0f / static_cast<float>(stddev);
            for (int i = 0; i < offset_spatial; ++i) {
                offset[i] = (offset[i] - mean_f) * inv_std;
            }
        }

        sanitize_and_clamp_array(offset, offset_spatial);

        // Apply soft-clamping mapping
        for (int i = 0; i < offset_spatial; ++i) {
            offset[i] = centered_lhtan_offset(offset[i], max_offset);
        }

        // Step 2: DCNv2 - Compute the modulation mask (scalars in [0, 1])
        if (l.use_mask) {
            gemm_cpu(0, 0, mask_filters, n, l.c * l.size * l.size, 1,
                 l.mask_weights, l.c * l.size * l.size,
                 state.workspace, n, 0, mask, n);
            add_bias(mask, l.mask_biases, 1, mask_filters, n);
            sanitize_and_clamp_array(mask, mask_filters * n);
            for (int i = 0; i < mask_filters * n; ++i) {
                mask[i] = stable_sigmoid(mask[i]);
            }
        }

        // Step 3: Deformable Convolution via im2col + GEMM
        // Sample input at (grid + offset) using bilinear interpolation, then apply weights.
        for (int g = 0; g < l.groups; ++g) {
            float *im = input + g * (l.c / l.groups) * l.h * l.w;
            float *a = l.weights + g * l.nweights / l.groups;
            float *col = state.workspace;
            float *c = l.output + b * l.outputs + g * m * n;

            // Perform deformable im2col to populate the column matrix
            deformable_im2col_cpu(im, offset, mask,
                                  l.c / l.groups, l.h, l.w,
                                  l.size, l.size,
                                  l.pad * l.dilation, l.pad * l.dilation,
                                  l.stride_y, l.stride_x,
                                  l.dilation, l.dilation,
                                  out_h, out_w, col);

            // Matrix multiplication for the main convolution operation
            gemm_cpu(0, 0, m, n, k, 1, a, k, col, n, 1, c, n);
        }
    }

    // Fix NaN/Inf and constrain output before batch norm
    // This prevents explosion from propagating through batch norm (which would kill gradients)
    int output_size = l.outputs * l.batch;
    sanitize_and_clamp_array(l.output, output_size);

    // Step 5: Batch normalization or bias addition
    if (l.batch_normalize) {
        forward_batchnorm_layer(l, state);
    } else {
        add_bias(l.output, l.biases, l.batch, l.n, out_h * out_w);
    }

    // Step 6: Apply activation function
    activate_array(l.output, l.outputs * l.batch, l.activation);

    // Final NaN/Inf fix
    sanitize_and_clamp_array(l.output, output_size);
}

/**
 * @brief Backward pass for deformable convolutional layer (CPU version)
 *
 * Computes gradients for all learnable parameters and propagates to previous layer.
 *
 * Backward pass steps:
 * 1. Gradient through activation function
 * 2. Batch normalization backward (or bias updates)
 * 3. Main convolution weight updates via deformable_im2col + GEMM
 * 4. Offset/mask gradient computation via deformable_col2im_coord
 * 5. Offset/mask weight gradients via GEMM
 * 6. Input gradient propagation via deformable_col2im
 */
void backward_deform_conv_layer(Darknet::Layer &l, Darknet::NetworkState state)
{
    TAT(TATPARMS);

    int out_h = l.out_h;
    int out_w = l.out_w;
    int offset_filters = 2 * l.size * l.size;
    int mask_filters = l.size * l.size;

    // GEMM dimensions
    int m = l.n / l.groups;                      // Filters per group
    int n_gemm = l.size * l.size * l.c / l.groups;  // Filter size
    int k_gemm = out_h * out_w;                  // Spatial output size

    // Backpass step 1: sanitize the incoming top gradient before any derivative math.
    // ALWAYS sanitize incoming deltas - NaN from YOLO layers will poison entire backward pass
    sanitize_and_clamp_array(l.delta, l.outputs * l.batch);

    // Backpass step 2: propagate gradients through the activation function.
    gradient_array(l.output, l.outputs * l.batch, l.activation, l.delta);

    // Backpass step 3: run batch-normalization backward if this layer uses BN.
    if (l.batch_normalize) {
        backward_batchnorm_layer(l, state);
    } else {
        // Backpass step 4: otherwise accumulate bias gradients directly from the output delta.
        backward_bias(l.bias_updates, l.delta, l.batch, l.n, k_gemm);
    }

    for (int b = 0; b < l.batch; ++b) {
        float *input = state.input + b * l.c * l.h * l.w;
        float *offset = l.offsets + b * out_h * out_w * offset_filters;
        float *mask = l.use_mask ? (l.masks + b * out_h * out_w * mask_filters) : nullptr;
        float *offset_delta = l.offset_deltas + b * out_h * out_w * offset_filters;
        float *mask_delta = l.use_mask ? (l.mask_deltas + b * out_h * out_w * mask_filters) : nullptr;

        // Backpass step 5: clear the offset-gradient buffer for this batch before accumulation.
        fill_cpu(out_h * out_w * offset_filters, 0, offset_delta, 1);
        if (mask_delta) {
            // Backpass step 6: clear the mask-gradient buffer for this batch before accumulation.
            fill_cpu(out_h * out_w * mask_filters, 0, mask_delta, 1);
        }

        for (int g = 0; g < l.groups; ++g) {
            float *im = input + g * (l.c / l.groups) * l.h * l.w;
            float *delta_out = l.delta + b * l.outputs + g * m * k_gemm;
            float *weight_updates = l.weight_updates + g * l.nweights / l.groups;
            float *weights = l.weights + g * l.nweights / l.groups;
            float *col = state.workspace;

            // Backpass step 7: reconstruct the sampled columns used in forward from saved offsets/masks.
            deformable_im2col_cpu(im, offset, mask,
                                  l.c / l.groups, l.h, l.w,
                                  l.size, l.size,
                                  l.pad * l.dilation, l.pad * l.dilation,
                                  l.stride_y, l.stride_x,
                                  l.dilation, l.dilation,
                                  out_h, out_w, col);

            // Backpass step 8: accumulate main convolution weight gradients via delta * sampled_columns^T.
            gemm_cpu(0, 1, m, n_gemm, k_gemm, 1, delta_out, k_gemm, col, k_gemm, 1, weight_updates, n_gemm);

            // Backpass step 9: project output gradients back into sampled-column space with weights^T * delta.
            float *col_grad = state.workspace;
            gemm_cpu(1, 0, n_gemm, k_gemm, m, 1, weights, n_gemm, delta_out, k_gemm, 0, col_grad, k_gemm);

            // Backpass step 10: backprop through the deformable sampler into offset and mask gradients.
            deformable_col2im_coord_cpu(col_grad, im, offset, mask,
                                        l.c / l.groups, l.h, l.w,
                                        l.size, l.size,
                                        l.pad * l.dilation, l.pad * l.dilation,
                                        l.stride_y, l.stride_x,
                                        l.dilation, l.dilation,
                                        out_h, out_w, offset_delta, mask_delta);

            if (state.delta) {
                float *im_delta = state.delta + b * l.c * l.h * l.w + g * (l.c / l.groups) * l.h * l.w;
                // Backpass step 11: scatter sampled-column gradients back to the previous layer input.
                deformable_col2im_cpu(col_grad, offset, mask,
                                      l.c / l.groups, l.h, l.w,
                                      l.size, l.size,
                                      l.pad * l.dilation, l.pad * l.dilation,
                                      l.stride_y, l.stride_x,
                                      l.dilation, l.dilation,
                                      out_h, out_w, im_delta);
            }
        }

        // Apply the centered-LHTAN derivative to offset gradients (matching GPU behavior).
        // The mapping is identity inside [-max_offset, max_offset] and 0.001-slope outside.
        float max_offset = (float)(l.size * l.dilation * 2);
        int offset_spatial = out_h * out_w * offset_filters;
        // Backpass step 12: sanitize the accumulated offset gradient buffer before the centered-LHTAN derivative.
        sanitize_and_clamp_array(offset_delta, offset_spatial);
        if (mask_delta) {
            // Backpass step 13: sanitize the accumulated mask gradient buffer before mask-weight backprop.
            sanitize_and_clamp_array(mask_delta, out_h * out_w * mask_filters);
        }
        for (int i = 0; i < offset_spatial; ++i) {
            const float lhtan_deriv = centered_lhtan_gradient_from_offset(offset[i], max_offset);
            offset_delta[i] = sanitize_and_clamp_value(offset_delta[i] * lhtan_deriv);
        }

        // Backpass step 14: rebuild the offset/mask-convolution receptive fields with a standard im2col.
        im2col_cpu_ext(input, l.c, l.h, l.w, l.size, l.size,
                       l.pad * l.dilation, l.pad * l.dilation,
                       l.stride_y, l.stride_x, l.dilation, l.dilation,
                       state.workspace);

        // Backpass step 15: accumulate offset-kernel gradients from offset_delta * input_columns^T.
        gemm_cpu(0, 1, offset_filters, l.c * l.size * l.size, out_h * out_w, 1,
             offset_delta, out_h * out_w, state.workspace, out_h * out_w, 1,
             l.offset_weight_updates, l.c * l.size * l.size);
        // Backpass step 16: accumulate offset-bias gradients from the offset delta map.
        backward_bias(l.offset_bias_updates, offset_delta, 1, offset_filters, out_h * out_w);

        if (l.use_mask) {
            // Backpass step 17: accumulate mask-kernel gradients from mask_delta * input_columns^T.
            gemm_cpu(0, 1, mask_filters, l.c * l.size * l.size, out_h * out_w, 1,
                 mask_delta, out_h * out_w, state.workspace, out_h * out_w, 1,
                 l.mask_weight_updates, l.c * l.size * l.size);
            // Backpass step 18: accumulate mask-bias gradients from the mask delta map.
            backward_bias(l.mask_bias_updates, mask_delta, 1, mask_filters, out_h * out_w);
        }
    }

    // Backpass step 19: sanitize main convolution weight updates before the optimizer step.
    sanitize_and_clamp_array(l.weight_updates, l.nweights);
    // Backpass step 20: sanitize main bias updates before the optimizer step.
    sanitize_and_clamp_array(l.bias_updates, l.n);
    // Backpass step 21: sanitize offset-kernel updates before the optimizer step.
    sanitize_and_clamp_array(l.offset_weight_updates, l.c * offset_filters * l.size * l.size);
    // Backpass step 22: sanitize offset-bias updates before the optimizer step.
    sanitize_and_clamp_array(l.offset_bias_updates, offset_filters);
    // Backpass step 23: sanitize stored offset deltas before they are reused or inspected.
    sanitize_and_clamp_array(l.offset_deltas, l.batch * out_h * out_w * offset_filters);
    if (l.use_mask) {
        // Backpass step 24: sanitize mask-kernel updates before the optimizer step.
        sanitize_and_clamp_array(l.mask_weight_updates, l.c * mask_filters * l.size * l.size);
        // Backpass step 25: sanitize mask-bias updates before the optimizer step.
        sanitize_and_clamp_array(l.mask_bias_updates, mask_filters);
        // Backpass step 26: sanitize stored mask deltas before they are reused or inspected.
        sanitize_and_clamp_array(l.mask_deltas, l.batch * out_h * out_w * mask_filters);
    }
    if (state.delta) {
        // Backpass step 27: sanitize the gradient passed to the previous layer.
        sanitize_and_clamp_array(state.delta, l.batch * l.c * l.h * l.w);
    }
}

/**
 * @brief Update weights for deformable convolutional layer (CPU version)
 *
 * Updates all learnable parameters with SGD + momentum + weight decay:
 * - Main convolution weights and biases
 * - Offset convolution weights and biases
 * - DCNv2: Mask convolution weights and biases
 * - Batch normalization scales (if enabled)
 */
void update_deform_conv_layer(Darknet::Layer &l, int batch, float learning_rate, float momentum, float decay)
{
    TAT(TATPARMS);

    float rate = learning_rate / batch;
    // Offset/mask weights need much lower learning rate (0.01x to prevent explosion)
    float offset_rate = rate * 0.01f;
    int nweights = l.nweights;
    int offset_filters = 2 * l.size * l.size;
    int mask_filters = l.size * l.size;
    int offset_nweights = l.c * offset_filters * l.size * l.size;
    int mask_nweights = l.c * mask_filters * l.size * l.size;

    // Main convolution weights: weight_updates -= decay * weights; weights += rate * updates
    axpy_cpu(nweights, -decay * batch, l.weights, 1, l.weight_updates, 1);
    axpy_cpu(nweights, rate, l.weight_updates, 1, l.weights, 1);
    scal_cpu(nweights, momentum, l.weight_updates, 1);

    // Main convolution biases
    axpy_cpu(l.n, rate, l.bias_updates, 1, l.biases, 1);
    scal_cpu(l.n, momentum, l.bias_updates, 1);

    // Offset convolution weights and biases (0.1x learning rate)
    axpy_cpu(offset_nweights, -decay * batch, l.offset_weights, 1, l.offset_weight_updates, 1);
    axpy_cpu(offset_nweights, offset_rate, l.offset_weight_updates, 1, l.offset_weights, 1);
    scal_cpu(offset_nweights, momentum, l.offset_weight_updates, 1);

    axpy_cpu(offset_filters, offset_rate, l.offset_bias_updates, 1, l.offset_biases, 1);
    scal_cpu(offset_filters, momentum, l.offset_bias_updates, 1);

    // DCNv2: Mask convolution weights and biases (0.1x learning rate)
    if (l.use_mask) {
        axpy_cpu(mask_nweights, -decay * batch, l.mask_weights, 1, l.mask_weight_updates, 1);
        axpy_cpu(mask_nweights, offset_rate, l.mask_weight_updates, 1, l.mask_weights, 1);
        scal_cpu(mask_nweights, momentum, l.mask_weight_updates, 1);

        axpy_cpu(mask_filters, offset_rate, l.mask_bias_updates, 1, l.mask_biases, 1);
        scal_cpu(mask_filters, momentum, l.mask_bias_updates, 1);
    }

    // Batch normalization scales
    if (l.batch_normalize) {
        axpy_cpu(l.n, rate, l.scale_updates, 1, l.scales, 1);
        scal_cpu(l.n, momentum, l.scale_updates, 1);
    }
}

/**
 * @brief Free batch normalization resources for deformable convolutional layer
 *
 * This function frees the memory allocated for batch normalization in a deformable convolutional layer.
 */
void free_deform_conv_batchnorm(Darknet::Layer *l)
{
    if (l->batch_normalize) {
        free(l->scales);
        free(l->scale_updates);
        free(l->mean);
        free(l->variance);
        free(l->mean_delta);
        free(l->variance_delta);
        free(l->rolling_mean);
        free(l->rolling_variance);
        free(l->x);
        free(l->x_norm);

#ifdef DARKNET_GPU
        cuda_free(l->scales_gpu);
        cuda_free(l->scale_updates_gpu);
        cuda_free(l->mean_gpu);
        cuda_free(l->variance_gpu);
        cuda_free(l->mean_delta_gpu);
        cuda_free(l->variance_delta_gpu);
        cuda_free(l->rolling_mean_gpu);
        cuda_free(l->rolling_variance_gpu);
        cuda_free(l->x_gpu);
        cuda_free(l->x_norm_gpu);
#endif
    }
}

/**
 * @brief Denormalize weights in a deformable convolutional layer
 *
 * This function denormalizes the weights in a deformable convolutional layer by
 * incorporating the batch normalization parameters into the weights.
 */
void denormalize_deform_conv_layer(Darknet::Layer &l)
{
    if (l.batch_normalize) {
        int f;
        for (f = 0; f < l.n; ++f) {
            l.biases[f] = l.biases[f] - l.scales[f] * l.rolling_mean[f] / (sqrtf(l.rolling_variance[f] + .00001f));

            const int filter_size = l.size * l.size * l.c / l.groups;
            int i;
            for (i = 0; i < filter_size; ++i) {
                int w_index = f * filter_size + i;
                l.weights[w_index] = l.weights[w_index] * l.scales[f] / (sqrtf(l.rolling_variance[f] + .00001f));
            }
        }

        free_deform_conv_batchnorm(&l);
    }
}

/**
 * @brief Set workspace size limit for deformable convolutional layer
 *
 * This function sets a limit on the workspace size for a deformable convolutional layer.
 */
void set_deform_conv_workspace_limit(Darknet::Layer *l, size_t workspace_size_limit)
{
    // Limit the workspace size for this layer
    if (workspace_size_limit > 0 && l->workspace_size > workspace_size_limit) {
        l->workspace_size = workspace_size_limit;
    }
}

/**
 * @brief Add biases to the output of a deformable convolutional layer
 *
 * This function adds biases to the output of a deformable convolutional layer.
 */
void add_deform_bias(float *output, float *biases, int batch, int n, int size)
{
    int i, j, k;
    for (i = 0; i < batch; ++i) {
        for (j = 0; j < n; ++j) {
            for (k = 0; k < size; ++k) {
                output[(i * n + j) * size + k] += biases[j];
            }
        }
    }
}

/**
 * @brief Compute gradients for biases in a deformable convolutional layer
 *
 * This function computes the gradients for biases in a deformable convolutional layer.
 */
void backward_deform_bias(float *bias_updates, float *delta, int batch, int n, int size)
{
    int i, j, k;
    for (i = 0; i < batch; ++i) {
        for (j = 0; j < n; ++j) {
            for (k = 0; k < size; ++k) {
                bias_updates[j] += delta[(i * n + j) * size + k];
            }
        }
    }
}

/**
 * @brief Get an image representation of the deformable convolutional layer's output
 *
 * This function returns an image representation of the deformable convolutional layer's output.
 */
Darknet::Image get_deform_conv_image(const Darknet::Layer &l)
{
    int h = l.out_h;
    int w = l.out_w;
    int c = l.out_c;
    return Darknet::float_to_image(w, h, c, l.output);
}

/**
 * @brief Get an image representation of the deformable convolutional layer's delta
 *
 * This function returns an image representation of the deformable convolutional layer's delta.
 */
Darknet::Image get_deform_conv_delta(const Darknet::Layer &l)
{
    int h = l.out_h;
    int w = l.out_w;
    int c = l.out_c;
    return Darknet::float_to_image(w, h, c, l.delta);
}

/**
 * @brief Get an image representation of a specific filter in the deformable convolutional layer
 *
 * This function returns an image representation of a specific filter in the deformable convolutional layer.
 */
Darknet::Image get_deform_conv_weight(const Darknet::Layer &l, int i)
{
    int h = l.size;
    int w = l.size;
    int c = l.c / l.groups;
    return Darknet::float_to_image(w, h, c, l.weights + i * h * w * c);
}

/**
 * @brief Calculate the output height of a deformable convolutional layer
 *
 * This function calculates the output height of a deformable convolutional layer.
 */
int deform_conv_out_height(const Darknet::Layer &l)
{
    return (l.h + 2 * l.pad - l.dilation * (l.size - 1) - 1) / l.stride_y + 1;
}

/**
 * @brief Calculate the output width of a deformable convolutional layer
 *
 * This function calculates the output width of a deformable convolutional layer.
 */
int deform_conv_out_width(const Darknet::Layer &l)
{
    return (l.w + 2 * l.pad - l.dilation * (l.size - 1) - 1) / l.stride_x + 1;
}

/**
 * @brief Rescale weights in a deformable convolutional layer
 *
 * This function rescales the weights in a deformable convolutional layer.
 */
void rescale_deform_weights(Darknet::Layer &l, float scale, float trans)
{
    int i, j, k;
    int n = l.n;
    int size = l.size;
    int c = l.c / l.groups;
    float *weights = l.weights;

    for (i = 0; i < n; ++i) {
        for (j = 0; j < c; ++j) {
            for (k = 0; k < size * size; ++k) {
                weights[i * c * size * size + j * size * size + k] *= scale;
                weights[i * c * size * size + j * size * size + k] += trans;
            }
        }
    }
}

/**
 * @brief Convert RGB weights to BGR in a deformable convolutional layer
 *
 * This function converts RGB weights to BGR in a deformable convolutional layer.
 * This is useful for processing images in BGR format (like OpenCV).
 */
void rgbgr_deform_weights(const Darknet::Layer &l)
{
    int i;
    int size = l.size;
    int c = l.c;
    int n = l.n;
    if (c % 3 != 0) return;

    for (i = 0; i < n; ++i) {
        int j = 0;
        float *swap = l.weights + i * size * size * c;
        while (j < c) {
            float tmp = swap[j];
            swap[j] = swap[j + 2];
            swap[j + 2] = tmp;
            j += 3;
        }
    }
}

/**
 * @brief Forward pass for assisted excitation in deformable convolutional layer
 *
 * This function performs the forward pass for assisted excitation in a deformable convolutional layer.
 * Assisted excitation is a technique to enhance feature activation.
 */
void assisted_excitation_deform_forward(Darknet::Layer &l, Darknet::NetworkState state)
{
    // This is a placeholder implementation
    // In a real implementation, this would implement assisted excitation
    // For now, we'll just call the standard forward function
    forward_deform_conv_layer(l, state);
}

/**
 * @brief Visualize filters in a deformable convolutional layer
 *
 * This function creates a visualization of the filters in a deformable convolutional layer.
 */
Darknet::Image *visualize_deform_conv_layer(const Darknet::Layer &l, const char *window, Darknet::Image *prev_weights)
{
    int width = l.size * l.size * l.c / l.groups;
    int height = l.n;
    float min = FLT_MAX;
    float max = -FLT_MAX;

    Darknet::Image *single_weights = (Darknet::Image *)xcalloc(1, sizeof(Darknet::Image));
    *single_weights = make_image(width, height, 1);

    int i, j;
    for (i = 0; i < height; ++i) {
        for (j = 0; j < width; ++j) {
            float val = l.weights[i * width + j];
            if (val > max) max = val;
            if (val < min) min = val;
            single_weights->data[i * width + j] = val;
        }
    }

    Darknet::normalize_image(*single_weights);

    return single_weights;
}

#ifdef DARKNET_GPU
/**
 * @brief Add biases to the output of a deformable convolutional layer (GPU version)
 *
 * This function adds biases to the output of a deformable convolutional layer on GPU.
 */
void add_deform_bias_gpu(float *output, float *biases, int batch, int n, int size)
{
    // This is a placeholder implementation
    // In a real implementation, this would use a CUDA kernel
    // For now, we'll just call the CPU version and copy the data
    float *output_cpu = (float *)xcalloc(batch * n * size, sizeof(float));
    float *biases_cpu = (float *)xcalloc(n, sizeof(float));

    cuda_pull_array(output, output_cpu, batch * n * size);
    cuda_pull_array(biases, biases_cpu, n);

    add_deform_bias(output_cpu, biases_cpu, batch, n, size);

    cuda_push_array(output, output_cpu, batch * n * size);

    free(output_cpu);
    free(biases_cpu);
}

/**
 * @brief Compute gradients for biases in a deformable convolutional layer (GPU version)
 *
 * This function computes the gradients for biases in a deformable convolutional layer on GPU.
 */
void backward_deform_bias_gpu(float *bias_updates, float *delta, int batch, int n, int size)
{
    // This is a placeholder implementation
    // In a real implementation, this would use a CUDA kernel
    // For now, we'll just call the CPU version and copy the data
    float *bias_updates_cpu = (float *)xcalloc(n, sizeof(float));
    float *delta_cpu = (float *)xcalloc(batch * n * size, sizeof(float));

    cuda_pull_array(bias_updates, bias_updates_cpu, n);
    cuda_pull_array(delta, delta_cpu, batch * n * size);

    backward_deform_bias(bias_updates_cpu, delta_cpu, batch, n, size);

    cuda_push_array(bias_updates, bias_updates_cpu, n);

    free(bias_updates_cpu);
    free(delta_cpu);
}

/**
 * @brief Forward pass for assisted excitation in deformable convolutional layer (GPU version)
 *
 * This function performs the forward pass for assisted excitation in a deformable convolutional layer on GPU.
 */
void assisted_excitation_deform_forward_gpu(Darknet::Layer &l, Darknet::NetworkState state)
{
    // This is a placeholder implementation
    // In a real implementation, this would implement assisted excitation on GPU
    // For now, we'll just call the standard forward function
    forward_deform_conv_layer_gpu(l, state);
}
#endif
