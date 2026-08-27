#pragma once

#include "darknet_internal.hpp"

/**
 * @file dcnv4_layer.hpp
 * @brief Deformable Convolution v4 (DCNv4) Layer Implementation
 *
 * DCNv4 is a spatial aggregation operator that learns deformable sampling offsets
 * to adaptively aggregate information from irregular spatial positions. Unlike standard
 * convolutions with fixed grids, DCNv4 learns where to sample from.
 *
 * ============================================================================
 * CRITICAL CONSTRAINT
 * ============================================================================
 * DCNv4 preserves the channel dimension: output filters MUST equal input channels.
 * If filters != input_channels, they are forced to match and a warning is logged.
 * To change channel dimensions, use a separate 1x1 convolutional layer before/after.
 *
 * ============================================================================
 * USAGE IN .cfg FILES
 * ============================================================================
 *
 * Basic example:
 *
 *   [dcnv4]
 *   filters=64              # Must equal input channels (spatial aggregation only)
 *   size=3                  # Kernel size (3, 5, 7, etc.)
 *   stride=1                # Stride (or use stride_x/stride_y)
 *   stride_x=1              # Horizontal stride (overrides stride if set)
 *   stride_y=1              # Vertical stride (overrides stride if set)
 *   dilation=1              # Dilation factor
 *   padding=1               # Explicit padding, or use pad=1 for size/2
 *   pad=1                   # Shorthand: pad=1 → padding=size/2
 *   groups=1                # Group convolution (default: 1 = standard)
 *   batch_normalize=0       # Enable batch normalization (0 or 1)
 *   activation=leaky        # Activation function
 *   offset_scale=1.0        # Scaling of learned offsets (default: 1.0)
 *   remove_center=0         # Exclude center offset (0 or 1, default: 0)
 *   d_stride=8              # Stride for offset field prediction (default: 8)
 *                           # 1=per-pixel, 8=sparse, higher=coarser
 *   block_thread=256        # CUDA block thread count (default: 256)
 *   softmax=0               # Normalize offsets via softmax (0 or 1, default: 0)
 *
 * Real-world examples (from YOLOv4-tiny variant in production):
 *
 *   # DCNv4 in deep backbone (256-CSP branch)
 *   [dcnv4]
 *   batch_normalize=1
 *   filters=128
 *   size=3
 *   stride=1
 *   pad=1
 *   activation=leaky
 *
 *   # DCNv4 in bottleneck (512 backbone stem)
 *   [dcnv4]
 *   batch_normalize=1
 *   filters=512
 *   size=3
 *   stride=1
 *   pad=1
 *   activation=leaky
 *
 *   # DCNv4 in neck (512 neck conv)
 *   [dcnv4]
 *   batch_normalize=1
 *   filters=512
 *   size=3
 *   stride=1
 *   pad=1
 *   activation=leaky
 *
 * ============================================================================
 * PARAMETER DESCRIPTIONS
 * ============================================================================
 *
 * Standard Convolution Parameters:
 *   - filters: Output channels (MUST equal input channels)
 *   - size: Kernel height/width
 *   - stride_x, stride_y: Horizontal and vertical stride
 *   - dilation: Spacing between kernel samples
 *   - padding: Zero-padding on input
 *   - groups: Group convolution (groups=c means depthwise)
 *   - batch_normalize: Apply batch normalization after aggregation
 *   - activation: Post-aggregation activation function
 *
 * DCNv4-Specific Parameters:
 *   - offset_scale: Multiplicative scale for learned offsets (default: 1.0).
 *                   Controls the range of spatial deformation.
 *
 *   - remove_center: If 1, excludes the center position from the sampling grid.
 *                    Useful for efficient 1x1 or center-excluding variants.
 *
 *   - d_stride: Stride of the offset prediction field (default: 8).
 *               Lower values (1): per-pixel offsets, more expressive but slower.
 *               Higher values (8, 16): coarser offsets, faster inference.
 *
 *   - block_thread: CUDA thread block size (default: 256).
 *                   Tunable for GPU performance optimization.
 *
 *   - softmax: If 1, normalizes offsets via softmax (creates attention-like masks).
 *              If 0, offsets are learned freely.
 *
 * ============================================================================
 * IMPLEMENTATION NOTES
 * ============================================================================
 *
 * - DCNv4 allocates dummy weights for file compatibility but does not use them
 *   in forward/backward/update operations.
 * - The layer performs spatial aggregation only; channel mixing requires separate
 *   1x1 convolutions.
 * - GPU computation is required for training/inference; CPU fallback available.
 * - Workspace is allocated for im2col and intermediate buffers.
 *
 */

size_t get_dcnv4_workspace_size(const Darknet::Layer & l);

void push_dcnv4_layer(Darknet::Layer & l);
void pull_dcnv4_layer(Darknet::Layer & l);

#ifdef DARKNET_GPU
void forward_dcnv4_layer_gpu(Darknet::Layer & l, Darknet::NetworkState state);
void backward_dcnv4_layer_gpu(Darknet::Layer & l, Darknet::NetworkState state);
void update_dcnv4_layer_gpu(Darknet::Layer & l, int batch, float learning_rate, float momentum, float decay, float loss_scale);
#ifdef CUDNN
void set_dcnv4_cudnn_16bit_mode(Darknet::Layer * l, int mode);
#endif
#endif

Darknet::Layer make_dcnv4_layer(int batch, int steps, int h, int w, int c, int n, int groups,
                               int size, int stride_x, int stride_y, int dilation,
                               int padding, ACTIVATION activation, int batch_normalize,
                               float offset_scale, int remove_center, int d_stride, int block_thread, int softmax,
                               int index, int train);

void forward_dcnv4_layer(Darknet::Layer & l, Darknet::NetworkState state);
void backward_dcnv4_layer(Darknet::Layer & l, Darknet::NetworkState state);
void update_dcnv4_layer(Darknet::Layer & l, int batch, float learning_rate, float momentum, float decay);
void resize_dcnv4_layer(Darknet::Layer * l, int w, int h);
