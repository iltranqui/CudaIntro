# Darknet Deform-Branch TODO

Branch: `deform`  
Last updated: 2026-04-14  
See `LAYERS_PLAN.md` for architecture context and component breakdown.

Legend: `[P1]` critical / correctness · `[P2]` important / perf · `[P3]` nice-to-have

---

## DCNv4 — `src-lib/dcnv4_layer.cpp` + `dcnv4_kernels.cu`

### Correctness / Guards

- [P1] `make_dcnv4_layer()`: add `groups must divide C` guard  
  → `if (c % groups != 0) darknet_fatal_error(...)` before output size calculation  
  File: `dcnv4_layer.cpp:64`

- [P1] `make_dcnv4_layer()`: round `block_thread` to nearest power-of-2 ≤ 1024  
  → `int bt = 1; while (bt < block_thread) bt <<= 1; if (bt > 1024) bt = 1024;`  
  File: `dcnv4_layer.cpp:59`

### Features

- [P2] **Spatial `d_stride` coarsening** — offset field predicted at `[B, H/d_stride, W/d_stride, padded_offset_dim]`, then bilinearly upsampled to full `[B, out_h, out_w, padded_offset_dim]`  
  Forward steps:
  1. Compute `H_c = ceil(out_h/d_stride)`, `W_c = ceil(out_w/d_stride)`
  2. `im2col_gpu_ext` at stride `stride_y*d_stride`, `stride_x*d_stride` → coarse col
  3. GEMM → coarse offsets `[B, H_c*W_c, padded_offset_dim]`  
  4. Add bias → apply new `nhwc_bilinear_upsample_kernel` to `l.offsets_gpu`  
  Backward: add `nhwc_bilinear_upsample_backward_kernel` (atomic scatter) before GEMM weight update  
  File: `dcnv4_kernels.cu:forward_dcnv4_layer_gpu` and `backward_dcnv4_layer_gpu`  
  Workspace note: coarse im2col fits in existing scratch (3× headroom); no resize needed when `d_stride ≤ 3`

- [P3] CPU fallback forward/backward — for debugging on CPU-only machines  
  File: `dcnv4_layer.cpp:forward_dcnv4_layer`, `backward_dcnv4_layer`

- [P3] Test: run forward on identical input, compare against `papers/dcnv4/` reference output

---

## DCNv3 — `src-lib/dcnv3_layer.cpp` + `dcnv3_kernels.cu`

All items below are HIGH COMPLEXITY. Reference code lives in `papers/dcnv3/`.

- [P2] Port `papers/dcnv3/cuda/dcnv3_im2col_cuda.cuh` → GPU forward kernel in `dcnv3_kernels.cu`  
  Key: softmax-normalized per-group attention over K² sampling points; no fused kernel (use explicit im2col)

- [P2] Port col2im gradient path → `backward_dcnv3_layer_gpu()`  
  Reference: `papers/dcnv3/cuda/dcnv3_cuda.cu`

- [P2] Wire `update_dcnv3_layer_gpu()`: SGD on `offset_weights_gpu`, `offset_biases_gpu`, BN scales

- [P3] CPU fallback: port `papers/dcnv3/cpu/dcnv3_cpu.cpp` → `forward_dcnv3_layer()`

- [P3] CPU backward: gradient through softmax-weighted bilinear (needed for CPU training)

- [P3] Integration test: verify offset convergence on 16×16 toy feature map

---

## Transformer — `src-lib/transformer_kernels.cu` + `transformer_layer.cpp`

### Performance

- [P2] **LayerNorm backward `dgamma`/`dbeta`**: reduce `atomicAdd` contention  
  Current: `total_tokens` threads each `atomicAdd` to C buckets  
  Fix option A — 2-pass: kernel 1 stores per-token `[total_tokens, C]` contributions, kernel 2 column-sums via warp reduction  
  Fix option B — warp-per-column: one warp per channel j, each lane strides over `total_tokens/32` rows, single `atomicAdd` per block  
  File: `transformer_kernels.cu:transformer_layernorm_backward_kernel:137`

- [P2] **`transformer_sum_rows_kernel`**: replace per-element `atomicAdd` with column-wise warp reduction  
  Pattern: one warp per column j, lane strides rows → 5 shuffle rounds → 1 `atomicAdd`  
  File: `transformer_kernels.cu:419`  
  Impact: used 4× in backward (ffn_b2, ffn_b1, wo_bias, qkv_bias updates)

### Features

- [P2] Add gradient norm monitoring per transformer layer (log `||grad||` to `cfg_and_state.output`)  
  File: `transformer_kernels.cu:backward_transformer_layer_gpu`, near end

- [P3] Verify `max_grad_norm=5.0` is set in all training configs that use `[transformer]`  
  Files: `LegoGears_transformer.cfg`, any other cfg with `[transformer]` layers

---

## ViT — `src-lib/vit_layer.cpp` + `vit_kernels.cu`

### Critical

- [DONE] **Bilinear pos-embed interpolation in `resize_vit_layer()`**  
  Fixed: `vit_pos_embed_bilinear_resize_kernel` in `vit_kernels.cu` — center-aligned coords, gather pattern, no atomics  
  Old GPU embed interpolated → synced to CPU → old buffer freed; gradient accum reset  
  File: `vit_layer.cpp:661-673`, `vit_kernels.cu:517-570`

### Features

- [P2] Add masked attention for padding tokens  
  When `H×W` is padded to `ceil(H/patch)×ceil(W/patch)`, pad tokens should not attend to real tokens  
  Approach: compute binary mask at layer init, add `−∞` before softmax (same as transformer shift mask)  
  File: `vit_kernels.cu:vit_attention_kernel`

- [P3] Memory assertion: `T = H×W` at max resolution must satisfy `T² × heads × batch × 4 bytes ≤ GPU_MEM / 4`  
  Add `if (T > 1024) darknet_fatal_error(...)` guard with configurable threshold  
  File: `vit_layer.cpp:make_vit_layer`

- [P3] Benchmark: measure GPU memory at 13×13 (T=169) vs 26×26 (T=676) feature scales and document in `docs/vit_layer.md`

---

## Deconvolutional — `src-lib/deconvolutional_layer.cpp` + `deconvolutional_kernels.cu`

All items LOW RISK (isolated, no cross-layer dependencies).

- [P2] Add `batch_normalize` support to `make_deconvolutional_layer()`  
  → allocate `scales`, `mean`, `variance`, `rolling_*` arrays  
  → allocate GPU equivalents and cuDNN descriptors  
  File: `deconvolutional_layer.cpp:make_deconvolutional_layer`

- [P2] Add BN forward/backward hooks:  
  `forward_deconvolutional_layer_gpu()` → call `forward_batchnorm_layer_gpu()` when `l.batch_normalize`  
  `backward_deconvolutional_layer_gpu()` → call `backward_batchnorm_layer_gpu()`  
  File: `deconvolutional_kernels.cu`

- [P2] Fix `push_deconvolutional_layer()` / `pull_deconvolutional_layer()`:  
  → sync `scales_gpu`, `rolling_mean_gpu`, `rolling_variance_gpu` when `batch_normalize`  
  File: `deconvolutional_layer.cpp`

- [P2] Add CFG parser support for `batch_normalize=1` in `parse_deconvolutional_section()`  
  File: `darknet_cfg.cpp` — find the `[deconvolutional]` section parser

- [P3] Verify GPU kernels: diff `deconvolutional_kernels.cu` against `3rdparty/unet_darknet/` reference  
  Known gap: `push_/pull_` may not sync all weights

---

## Cross-cutting

- [P2] `dcnv4_layer.cpp` + `dcnv3_layer.cpp`: add `workspace_size` update in `resize_*_layer()` if output dims change
- [P3] All GPU layers: audit `cudaMemsetAsync` vs `fill_ongpu` consistency (some use one, some the other)
- [P3] Add `static_assert(sizeof(float) == 4)` guard in CUDA files that assume 32-bit float

---

## Won't Fix / Out of Scope

- DCNv4 spatial d_stride > 8: workspace would exceed 3× budget; requires explicit `l.coarse_offsets_gpu` allocation
- ViT global attention at 52×52+: T² blows GPU memory; use Swin transformer instead
- DCNv3 CPU backward with full softmax Jacobian: O(K²) per point; not tractable for large K
