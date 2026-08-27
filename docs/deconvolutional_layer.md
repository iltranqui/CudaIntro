# Deconvolutional Layer (Transposed Convolution)

**Status**: PARTIAL — CPU forward/backward functional; BN missing; GPU path needs verification  
**Files**: `src-lib/deconvolutional_layer.{hpp,cpp}`, `src-lib/deconvolutional_kernels.cu`  
**Reference source**: `darknet.CG/src/deconvolutional_layer.{c,h}`, `unet_darknet/src/`  
**CFG key**: `[deconvolutional]`

---

## What It Is

Transposed convolution (also called "deconv" or "fractionally strided convolution") inverts the spatial size reduction of a standard convolution. It is the standard upsampling primitive in:

- **U-Net** decoder paths (feature map → double spatial size)
- **FPN / PANet** top-down pathways
- **Generative** architectures

It is **not** a true inverse of convolution mathematically — it is a full convolution on a zero-interleaved (stride-upsampled) input.

---

## Output Size Formula

```
out_h = stride * (h - 1) + size
out_w = stride * (w - 1) + size
```

Example: 13×13 input, size=4, stride=2 → `2*(13-1)+4 = 28` → 28×28 output.

**Note**: No padding parameter supported in current CFG parser. The parser will `darknet_fatal_error` if `pad` or `padding` keys are present.

---

## Architecture

```
Input [B, C, H, W]
       │
       ├──▶ col2im (inverse of im2col): scatter input through transposed weight matrix
       │       effectively: zero-interleave rows/cols, then standard conv with flipped weights
       │
       ├──▶ Add biases
       │
       └──▶ Apply activation
```

Internally uses `col2im_cpu()` + `gemm(TRANSPOSE_weights, input_as_col)`.

---

## Key Parameters

| CFG Parameter | Default | Description |
|---------------|---------|-------------|
| `filters`     | —       | Output channels N |
| `size`        | 4       | Kernel size |
| `stride`      | 2       | Upsampling factor |
| `activation`  | leaky   | Post-activation |
| `batch_normalize` | ✗ NOT SUPPORTED | **Missing — needs implementation** |

---

## Memory Layout

| Buffer | Shape | Purpose |
|--------|-------|---------|
| `weights` | [C, N, size, size] | Note: C×N order (transposed from standard conv) |
| `weight_updates` | [C, N, size, size] | Gradient accumulator |
| `biases` | [N] | Output bias |
| `workspace` | [C·size²·out_h·out_w] | col2im work buffer |
| `output` | [B, N, out_h, out_w] | Output feature map |
| `delta` | [B, N, out_h, out_w] | Gradient map |

---

## Forward Pass (CPU)

`forward_deconvolutional_layer()`:

1. For each batch item:
   - `gemm(TRANSPOSE, weights, input_item)` → col buffer (transposed GEMM)
   - `col2im_cpu(col_buffer, ...)` → scattered into output
2. Add biases
3. Apply activation in-place

---

## Backward Pass (CPU)

`backward_deconvolutional_layer()`:

1. Activation backward (in-place on delta)
2. Bias update: accumulate delta → `bias_updates`
3. Weight update: `gemm(input, col_buffer^T)` → `weight_updates`
4. Delta to prev layer: `im2col(delta)` → col → `gemm(weights, col)` → prev delta

---

## GPU Path

`deconvolutional_kernels.cu` provides:
- `forward_deconvolutional_layer_gpu()`
- `backward_deconvolutional_layer_gpu()`
- `update_deconvolutional_layer_gpu()`
- `push_deconvolutional_layer()` / `pull_deconvolutional_layer()`

**Current state**: Ported from `darknet.CG` reference. Needs verification against `unet_darknet/src/deconvolutional_kernels.cu` for correctness. Push/pull sync verified present.

---

## Missing: Batch Normalization

Standard conv layers in Darknet support `batch_normalize=1`. Deconv does not yet.

**What needs adding**:
1. In `make_deconvolutional_layer()`: allocate BN buffers (`scales`, `rolling_mean`, `rolling_variance`, etc.)
2. In `forward_deconvolutional_layer()`: call `forward_batchnorm_layer()` after GEMM, before activation
3. In `backward_deconvolutional_layer()`: call `backward_batchnorm_layer()` before delta propagation
4. In CFG parser: read `batch_normalize` key (currently ignored)

---

## CFG Example

```cfg
[deconvolutional]
filters=128
size=4
stride=2
activation=leaky
```

---

## Known Issues / Constraints

1. **No `pad` support** — parser hard-errors if `pad` key present. Output size fully determined by stride and size.
2. **No `batch_normalize`** — forces use of standalone `[batchnorm]` layer if needed.
3. **No `dilation`** — field exists in Layer struct but ignored here.
4. **No `groups`** — all-to-all channels only.
5. Weight init uses `scale = 1/sqrt(size*size*c)` — same as standard conv.

---

## Reference Sources for Porting

| Feature | Reference file |
|---------|----------------|
| CPU forward/backward | `unet_darknet/src/deconvolutional_layer.c` |
| GPU kernels | `unet_darknet/src/deconvolutional_kernels.cu` |
| BN integration | `src-lib/convolutional_layer.cpp` (copy BN block pattern) |
| Col2im primitive | `src-lib/col2im.hpp` + `col2im.cpp` |
