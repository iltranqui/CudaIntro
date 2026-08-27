# Deformable Convolution Layer (DCNv1 / DCNv2)

**Status**: COMPLETE  
**Files**: `src-lib/deform_conv_layer.{hpp,cpp}`, `src-lib/deform_conv_kernels.cu`  
**CFG key**: `[deformable_convolutional]`

---

## What It Is

Standard convolution samples a fixed K×K grid. Deformable convolution **learns where to sample**. The offset subnet predicts (Δy, Δx) for every kernel point, bending the sampling grid toward object boundaries and relevant structure.

DCNv2 extends DCNv1 with a **modulation mask** — a per-point scalar that learns to suppress irrelevant background samples.

Reference: [Deformable Convolutional Networks (Dai et al., 2017)](https://arxiv.org/abs/1703.06211) and [DCNv2 (Zhu et al., 2019)](https://arxiv.org/abs/1811.11168).

---

## Architecture

```
Input [B, C, H, W]
       │
       ├──▶ Offset Conv (1 or 3×3, shared input) ──▶ Raw Offsets [B, 2·K², H', W']
       │                                              (+ Mask [B, K², H', W'] if use_mask=1)
       │
       ├──▶ Clamp offsets to [-max_offset, +max_offset]
       │
       ├──▶ Build sample coords: sample_y = p_y·stride + k_y·dilation + Δy_k
       │                         sample_x = p_x·stride + k_x·dilation + Δx_k
       │
       ├──▶ Bilinear interpolation at fractional coords
       │       value = w_tl·I[y0,x0] + w_tr·I[y0,x1] + w_bl·I[y1,x0] + w_br·I[y1,x1]
       │
       └──▶ GEMM accumulate: out[b,n,p] = Σ_c Σ_k W[n,c,k] · sample(input[b,c], p+k+Δk) [· mask_k]
```

---

## Key Parameters

| CFG Parameter    | Default | Description |
|------------------|---------|-------------|
| `filters`        | —       | Output channels N |
| `size`           | 3       | Kernel size (K×K) |
| `stride`         | 1       | Spatial stride |
| `dilation`       | 1       | Dilation factor |
| `pad`            | size/2  | Zero padding |
| `use_mask`       | 1       | 0=DCNv1 (offsets only), 1=DCNv2 (offsets + modulation mask) |
| `batch_normalize`| 0       | Enable BN |
| `activation`     | leaky   | Post-BN activation |

---

## Memory Layout

| Buffer | Shape | Purpose |
|--------|-------|---------|
| `weights` | [N, C/groups, K, K] | Main conv weights |
| `offset_weights` | [2·K², C, 1, 1] | Offset subnet weights |
| `offset_biases` | [2·K²] | Offset subnet biases |
| `mask_weights` | [K², C, 1, 1] | Mask subnet weights (use_mask=1 only) |
| `workspace` | [C·K²·H'·W'] | im2col column buffer |
| `offset_output` | [B, 2·K², H', W'] | Predicted offsets |
| `mask_output` | [B, K², H', W'] | Predicted masks (use_mask=1 only) |

---

## Forward Pass (CPU)

`forward_deform_conv_layer()` in `deform_conv_layer.cpp`:

1. Run offset conv: `gemm` with `offset_weights` → `offset_output`
2. Clamp: `clamp(offset_output, -max_offset, max_offset)`
3. If `use_mask=1`: sigmoid(`mask_output`)
4. `deform_im2col_cpu()`: builds column buffer using bilinear interp at offset coords
5. `gemm(weights, col_buffer)` → output
6. Add biases, apply activation

---

## Backward Pass (3 Streams)

`backward_deform_conv_layer()`:

| Stream | Target | How |
|--------|--------|-----|
| A | `∂L/∂W` (main weights) | Standard GEMM with pre-built col buffer |
| B | `∂L/∂input` (prev layer) | Reverse bilinear scatter: add δ·w to 4 integer neighbors |
| C | `∂L/∂offset` (offset net) | `∂I/∂y` and `∂I/∂x` pixel gradients at sample points |
| D (v2) | `∂L/∂mask` | δ · W · sampled_input at each point |

---

## CFG Example

```cfg
[deformable_convolutional]
batch_normalize=1
filters=64
size=3
stride=1
pad=1
dilation=1
activation=leaky
use_mask=1
```

---

## Known Issues / Constraints

- `max_offset` is heuristic (default ~3× kernel size). Too large → degenerate grids. Too small → no adaptation.
- `try_fix_nan` is auto-enabled when deform layer detected in network (offset gradients can produce NaN early in training).
- DCNv1 (`use_mask=0`) path is rarely tested. Run a sanity check if using it.
- Offset subnet shares the same input as main conv. This is intentional — offsets are predicted from context, not output.

---

## Relation to DCNv3 / DCNv4

| | DCNv1 | DCNv2 | DCNv3 | DCNv4 |
|---|---|---|---|---|
| Per-point offset | ✓ | ✓ | ✓ (grouped) | ✓ (grouped) |
| Modulation mask | ✗ | ✓ | Softmax weights | Fused weight |
| Weight matrix | Full conv | Full conv | Depthwise-style | Aggregation only |
| Channel change | ✓ | ✓ | ✗ (forced C=N) | ✗ (forced C=N) |
| GPU kernel style | im2col+GEMM | im2col+GEMM | grouped im2col | Flash-fused |
