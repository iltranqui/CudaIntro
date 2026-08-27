#pragma once

#include "darknet_internal.hpp"

/**
 * @brief Deformable Convolutional Layer (DCNv1 and DCNv2)
 *
 * ╔══════════════════════════════════════════════════════════════════════════════════════╗
 * ║             DEFORMABLE CONVOLUTION — THE ART OF ADAPTIVE SAMPLING                   ║
 * ╚══════════════════════════════════════════════════════════════════════════════════════╝
 *
 *  ACT 1 — THE CORE INSIGHT: RIGID vs. ADAPTIVE GRIDS
 *  ─────────────────────────────────────────────────────────────────────────────────────
 *
 *  A standard 3×3 conv always samples the same 9 positions relative to each output pixel.
 *  DCN LEARNS where to look. The sampling grid bends to chase edges and object structure.
 *
 *  ┌───────────────────────────────┐    ┌───────────────────────────────────────────────┐
 *  │   STANDARD CONV (rigid stamp) │    │   DEFORMABLE CONV (elastic probe)             │
 *  │                               │    │                                               │
 *  │   ░░░▒▒▓▓███▓▓▒▒░░░           │    │   ░░░▒▒▓▓███▓▓▒▒░░░     Offset field:        │
 *  │   ░░▒▓▓█████████▓▒░           │    │   ░░▒▓▓█████████▓▒░     (Δy,Δx per point)    │
 *  │   ░░▒▒▓▓▓█████▓▓▒▒░           │    │   ░░▒▒▓▓▓█████▓▓▒▒░    ┌───┬───┬───┐        │
 *  │   ░░░░░░░░░░░░░░░░░           │    │   ░░░░░░░░░░░░░░░░░    │ ↗ │ ↑ │ ↖ │        │
 *  │                               │    │                         ├───┼───┼───┤        │
 *  │   Kernel always lands here:   │    │   Kernel bends to edge: │ → │ ○ │ ← │        │
 *  │                               │    │                         ├───┼───┼───┤        │
 *  │   ┌───┬───┬───┐               │    │    .      .      .      │ ↘ │ ↓ │ ↙ │        │
 *  │   │ ● │ ● │ ● │               │    │        ●      ●         └───┴───┴───┘        │
 *  │   ├───┼───┼───┤               │    │             ●                                │
 *  │   │ ● │ ● │ ● │               │    │        ●         ●   ← points cluster on     │
 *  │   ├───┼───┼───┤               │    │    ●        ●       ●    object boundaries!  │
 *  │   │ ● │ ● │ ● │               │    │        ●         ●                           │
 *  │   └───┴───┴───┘               │    │             ●                                │
 *  │   Misses the object edge!     │    │   Adapts to feature geometry!                │
 *  └───────────────────────────────┘    └───────────────────────────────────────────────┘
 *
 *
 *  ACT 2 — FORWARD PASS: FIVE STAGES OF ADAPTIVE PERCEPTION
 *  ─────────────────────────────────────────────────────────────────────────────────────
 *
 *  ┌─────────────────────┐
 *  │  STAGE 1: PERCEIVE  │  Input [B,C,H,W] ──▶ Offset Conv (3×3) ──▶ Raw Offsets [B, 2·K², H, W]
 *  └─────────────────────┘                                                ↑
 *                                               K² = kernel_h × kernel_w (2 values per point: Δy, Δx)
 *                              If use_mask=1: also predicts Mask [B, K², H, W]  (one scalar per point)
 *
 *  ┌───────────────────────┐
 *  │  STAGE 2: STABILIZE   │  Raw offsets can grow huge; clamp to prevent degenerate grids:
 *  └───────────────────────┘  Δy_k = clamp(Δy_k, -max_offset, +max_offset)
 *                             Δx_k = clamp(Δx_k, -max_offset, +max_offset)
 *
 *  ┌────────────────────────────────────┐
 *  │  STAGE 3: BUILD SAMPLE COORDINATES │  For each output (p_y, p_x) and kernel point k=(k_y, k_x):
 *  └────────────────────────────────────┘
 *       sample_y = p_y · stride + k_y · dilation + Δy_k
 *       sample_x = p_x · stride + k_x · dilation + Δx_k
 *                                                    └── fractional! not on the pixel grid
 *
 *  ┌────────────────────────────────────┐
 *  │  STAGE 4: BILINEAR INTERPOLATION   │  Reading between integer pixels using 4-neighbor blending:
 *  └────────────────────────────────────┘
 *
 *       Let sy = sample_y,  sx = sample_x
 *       Let dy = sy - ⌊sy⌋,  dx = sx - ⌊sx⌋     (fractional parts, both in [0,1))
 *
 *        ⌊sy⌋ →  ○─────────────────────○  ← ⌊sy⌋
 *                │                     │
 *         sy ──▶ │          ★          │  ★ = fractional sample point
 *                │   dy↕        dx→    │
 *        ⌈sy⌉ →  ○─────────────────────○
 *                ↑                     ↑
 *              ⌊sx⌋                  ⌈sx⌉
 *
 *       w_tl = (1-dy)(1-dx)    w_tr = (1-dy)(dx)     ← top    row weights
 *       w_bl = (  dy)(1-dx)    w_br = (  dy)(dx)     ← bottom row weights
 *
 *       value = w_tl·I[⌊sy⌋,⌊sx⌋] + w_tr·I[⌊sy⌋,⌈sx⌉]
 *             + w_bl·I[⌈sy⌉,⌊sx⌋] + w_br·I[⌈sy⌉,⌈sx⌉]
 *
 *       ► The gradient ∂value/∂Δy and ∂value/∂Δx exist analytically — this is what
 *         enables the offset network to be trained end-to-end by backpropagation!
 *
 *  ┌─────────────────────────────┐
 *  │  STAGE 5: ACCUMULATE OUTPUT │
 *  └─────────────────────────────┘
 *       DCNv1:  out[b,n,p] = Σ_c Σ_k  W[n,c,k] · sample(input[b,c], p+k+Δk)
 *       DCNv2:  out[b,n,p] = Σ_c Σ_k  W[n,c,k] · sample(input[b,c], p+k+Δk) · m[b,k,p]
 *                                                                               └─ mask ─┘
 *
 *       DCNv1 vs. DCNv2 — What the modulation mask adds:
 *       ┌──────────────────────────────────┬───────────────────────────────────────────┐
 *       │  DCNv1  (use_mask=0)             │  DCNv2  (use_mask=1)                      │
 *       ├──────────────────────────────────┼───────────────────────────────────────────┤
 *       │  Learns WHERE  to look           │  Learns WHERE  to look  (offset network)  │
 *       │  All 9 samples equally trusted   │  Learns HOW MUCH to trust (mask network)  │
 *       │                                  │                                           │
 *       │  ●  ●  ●   (weight=1 each)       │  ●₀.₉  ●₀.₁  ●₀.₈  (per-point salience) │
 *       │  ●  ●  ●                         │  ●₀.₃  ●₀.₉  ●₀.₅                        │
 *       │  ●  ●  ●                         │  ●₀.₇  ●₀.₄  ●₀.₂                        │
 *       │                                  │                                           │
 *       │  Learns geometry only            │  Learns geometry + importance             │
 *       └──────────────────────────────────┴───────────────────────────────────────────┘
 *
 *
 *  ACT 3 — BACKWARD PASS: THREE GRADIENT STREAMS
 *  ─────────────────────────────────────────────────────────────────────────────────────
 *
 *   ∂L/∂output  (arriving from the next layer)
 *        │
 *        ├──────────────────────────────────────────────────────────────────────────────
 *        │                                                                              │
 *        ▼                                                                              │
 *   ┌─────────────────────────────┐                                                    │
 *   │  STREAM A: ∂L/∂W            │  Standard GEMM backward. The workspace column-     │
 *   │  (main conv weights)        │  buffer (sampled+reordered input) is already ready.│
 *   └─────────────────────────────┘                                                    │
 *        │                                                                              │
 *        ▼                                                                              │
 *   ┌─────────────────────────────┐                                                    │
 *   │  STREAM B: ∂L/∂input        │  Reverse bilinear interpolation: scatter δ back    │
 *   │  (to previous layer)        │  to the 4 integer neighbors of each sample point,  │
 *   │                             │  weighted by w_tl, w_tr, w_bl, w_br.               │
 *   └─────────────────────────────┘                                                    │
 *        │                                                                              │
 *        ▼                                                                              │
 *   ┌─────────────────────────────┐                                                    │
 *   │  STREAM C: ∂L/∂offset       │  ∂L/∂Δy_k = Σ_c Σ_n  δ[n] · W[n,c,k] · ∂I/∂y    │
 *   │  (offset network gradient)  │  ∂L/∂Δx_k = Σ_c Σ_n  δ[n] · W[n,c,k] · ∂I/∂x    │
 *   │                             │                                                    │
 *   │  KEY: ∂I/∂y is the pixel    │  This gradient teaches the offset subnet to move   │
 *   │  gradient at sample point   │  sampling points toward high-contrast regions!      │
 *   └─────────────────────────────┘                                                    │
 *        │                                                                              │
 *        │  DCNv2 also computes:                                                        │
 *        │  ∂L/∂m_k = Σ_c Σ_n  δ[n] · W[n,c,k] · sample(input[c], p+k+Δk)            │
 *        └──────────────────────────────────────────────────────────────────────────────┘
 *
 * ╔══════════════════════════════════════════════════════════════════════════════════════╗
 * ║  "NOW I GET IT!": The offset subnet sees the raw input and learns to produce        ║
 * ║  (Δy,Δx) values that pull sampling points toward edges and structure. The mask      ║
 * ║  subnet learns to suppress irrelevant background samples. Together they implement   ║
 * ║  a form of spatial attention with no explicit supervision — just end-to-end loss.   ║
 * ╚══════════════════════════════════════════════════════════════════════════════════════╝
 *
 * ### CFG File Usage:
 * To use this layer in a Darknet .cfg file:
 * ```cfg
 * [deformable_convolutional]
 * batch_normalize=1     ; 1 to use batch normalization, 0 otherwise
 * filters=64            ; number of output filters
 * size=3                ; kernel size (e.g., 3x3)
 * stride=1              ; stride of the convolution
 * pad=1                 ; padding (usually size/2)
 * activation=leaky      ; activation function
 * use_mask=1            ; 1 for DCNv2 (with modulation mask), 0 for DCNv1 (offsets only)
 * ```
 */

#ifdef DARKNET_GPU
/** @brief GPU forward pass for deformable convolution. Handles DCNv1 and DCNv2 logic via CUDA kernels. */
void forward_deform_conv_layer_gpu(Darknet::Layer & l, Darknet::NetworkState state);
/** @brief GPU backward pass. Computes gradients for main weights, offsets, and masks using coordinate backprop. */
void backward_deform_conv_layer_gpu(Darknet::Layer & l, Darknet::NetworkState state);
/** @brief GPU weight update. Applies gradients to weights, biases, offsets, and masks with stabilization. */
void update_deform_conv_layer_gpu(Darknet::Layer & l, int batch, float learning_rate, float momentum, float decay, float loss_scale);

/** @brief Synchronizes layer weights from CPU to GPU memory. */
void push_deform_conv_layer(Darknet::Layer & l);
/** @brief Synchronizes layer weights from GPU to CPU memory. */
void pull_deform_conv_layer(Darknet::Layer & l);

/** @brief GPU helper to add biases to the convolutional output. */
void add_deform_bias_gpu(float *output, float *biases, int batch, int n, int size);
/** @brief GPU helper to accumulate bias gradients from deltas. */
void backward_deform_bias_gpu(float *bias_updates, float *delta, int batch, int n, int size);
#endif

/** @brief Frees memory associated with batch normalization for this layer. */
void free_deform_conv_batchnorm(Darknet::Layer *l);

/** @brief Returns the required workspace size in bytes for im2col operations. */
size_t get_deform_conv_workspace_size(const Darknet::Layer & l);

/**
 * @brief Factory function to initialize a new deformable convolutional layer.
 * @param use_mask If 1, enables DCNv2 (modulated deformable convolution).
 */
Darknet::Layer make_deform_conv_layer(int batch, int steps, int h, int w, int c, int n, int groups,
                                     int size, int stride_x, int stride_y, int dilation,
                                     int padding, ACTIVATION activation, int batch_normalize,
                                     int binary, int xnor, int adam, int use_bin_output,
                                     int index, int antialiasing, Darknet::Layer *share_layer,
                                     int assisted_excitation, int train, int use_mask = 1);

/** @brief Merges batch normalization parameters into weights/biases for inference optimization. */
void denormalize_deform_conv_layer(Darknet::Layer & l);
/** @brief Sets a maximum limit on the workspace memory used by this layer. */
void set_deform_conv_workspace_limit(Darknet::Layer *l, size_t workspace_size_limit);
/** @brief Resizes the layer buffers to handle a new input resolution. */
void resize_deform_conv_layer(Darknet::Layer * l, int w, int h);
/** @brief CPU forward pass implementation using bilinear interpolation and GEMM. */
void forward_deform_conv_layer(Darknet::Layer & l, Darknet::NetworkState state);
/** @brief CPU weight update using standard SGD with momentum and weight decay. */
void update_deform_conv_layer(Darknet::Layer & l, int batch, float learning_rate, float momentum, float decay);
/** @brief Creates a visualization of the layer's weights for debugging. */
Darknet::Image *visualize_deform_conv_layer(const Darknet::Layer & l, const char * window, Darknet::Image * prev_weights);

/** @brief CPU backward pass. Propagates gradients through the deformable sampling grid. */
void backward_deform_conv_layer(Darknet::Layer & l, Darknet::NetworkState state);

/** @brief CPU helper to add biases to the output feature map. */
void add_deform_bias(float *output, float *biases, int batch, int n, int size);
/** @brief CPU helper to calculate bias updates. */
void backward_deform_bias(float *bias_updates, float *delta, int batch, int n, int size);

/** @brief Returns the output feature map as a Darknet Image structure. */
Darknet::Image get_deform_conv_image(const Darknet::Layer & l);
/** @brief Returns the gradient (delta) map as a Darknet Image structure. */
Darknet::Image get_deform_conv_delta(const Darknet::Layer & l);
/** @brief Returns a specific filter's weights as an image. */
Darknet::Image get_deform_conv_weight(const Darknet::Layer & l, int i);

/** @brief Calculates output height based on input, padding, stride, and dilation. */
int deform_conv_out_height(const Darknet::Layer & l);
/** @brief Calculates output width based on input, padding, stride, and dilation. */
int deform_conv_out_width(const Darknet::Layer & l);
/** @brief Rescales weights and adds a translation factor. */
void rescale_deform_weights(Darknet::Layer & l, float scale, float trans);
/** @brief Swaps RGB weight channels to BGR for compatibility with BGR input. */
void rgbgr_deform_weights(const Darknet::Layer & l);
/** @brief Placeholders for assisted excitation feature. */
void assisted_excitation_deform_forward(Darknet::Layer & l, Darknet::NetworkState state);
void assisted_excitation_deform_forward_gpu(Darknet::Layer & l, Darknet::NetworkState state);
