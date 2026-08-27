# Darknet Advanced Layers — Implementation Plan

Branch: `deform`

## Directory Tree

```
src-lib/
├── deform_conv_layer.hpp / .cpp          # DCNv1 / DCNv2  (COMPLETE)
├── deform_conv_kernels.cu                # GPU kernels     (COMPLETE)
├── deconvolutional_layer.hpp / .cpp      # Transposed conv (PARTIAL — no BN, no full GPU)
├── deconvolutional_kernels.cu            # GPU kernels     (PARTIAL)
├── transformer_layer.hpp / .cpp          # Swin windowed   (FUNCTIONAL — training unstable)
├── transformer_kernels.cu                # GPU kernels     (FUNCTIONAL)
├── vit_layer.hpp / .cpp                  # ViT global      (FUNCTIONAL — memory risk at hi-res)
├── vit_kernels.cu                        # GPU kernels     (FUNCTIONAL)
├── graph_conv_layer.hpp / .cpp           # Local graph conv (FUNCTIONAL)
├── graph_conv_kernels.cu                 # GPU kernels      (FUNCTIONAL)
├── dcnv3_layer.hpp / .cpp                # DCNv3           (STUB — no real backward)
├── dcnv3_kernels.cu                      # GPU kernels     (STUB — placeholders only)
├── dcnv4_layer.hpp / .cpp                # DCNv4           (STUB — no real backward)
├── dcnv4_kernels.cu                      # GPU kernels     (STUB — placeholders only)
│
papers/
├── dcnv3/                                # Reference CUDA from InternImage repo
│   ├── cpu/dcnv3_cpu.{h,cpp}
│   └── cuda/dcnv3_{im2col,cuda}.{cuh,cu,h}
└── dcnv4/                                # Reference CUDA from Flash-Deformable repo
    ├── cuda/flash_deform_{im2col,col2im,attn}_cuda.{cuh,cu,h}
    └── dcnv4.h

docs/
├── LAYERS_PLAN.md                        # This file
├── deform_conv_layer.md
├── deconvolutional_layer.md
├── transformer_layer.md
├── vit_layer.md
├── graph_conv_layer.md
├── dcnv3_layer.md
└── dcnv4_layer.md
```

---

## Layer Status Matrix

| Layer              | CFG Key                     | make_*  | CPU fwd | CPU bwd | GPU fwd | GPU bwd | BN  | Status     |
|--------------------|-----------------------------|---------|---------|---------|---------|---------|-----|------------|
| Deformable (DCNv2) | `[deformable_convolutional]`| ✓       | ✓       | ✓       | ✓       | ✓       | ✓   | COMPLETE   |
| Deconvolutional    | `[deconvolutional]`         | ✓       | ✓       | ✓       | ✓ (partial) | ✓ (partial) | ✗ | PARTIAL |
| Transformer (Swin) | `[transformer]`             | ✓       | ✓       | ✓       | ✓ (opt) | ✓ (opt) | N/A | FUNCTIONAL |
| ViT (Global)       | `[vit]`                     | ✓       | ✓       | ✓       | ✓       | ✓       | N/A | FUNCTIONAL |
| Graph Conv         | `[graph_conv]`              | ✓       | ✓       | ✓       | ✓       | ✓       | ✓   | FUNCTIONAL |
| DCNv3              | `[dcnv3]`                   | ✓       | STUB    | STUB    | STUB    | STUB    | ✓   | STUB       |
| DCNv4              | `[dcnv4]`                   | ✓       | STUB    | STUB    | ✓       | ✓       | ✓   | GPU-ONLY   |

---

## Component Breakdown

### 1. Deformable Conv (DCNv1 / DCNv2) — `deform_conv_layer`

**Role**: Adaptive spatial sampling. Learns (Δy, Δx) offsets per kernel point; DCNv2 also learns per-point modulation masks.

**Dependencies**:
- `gemm.hpp` — weight matrix multiply
- `im2col.hpp` — column buffer construction
- `batchnorm_layer.hpp` — BN support
- `deform_conv_kernels.cu` — bilinear interpolation, offset backprop

**Constraints/Open Questions**:
- `max_offset` clamping value is heuristic — needs tuning per task
- DCNv1 mode (`use_mask=0`) rarely tested; verify mask-free path

---

### 2. Deconvolutional Layer — `deconvolutional_layer`

**Role**: Transposed convolution (upsampling). Maps small feature maps to larger via fractional-stride convolution. Foundation for decoder paths in U-Net / FPN.

**Dependencies**:
- `col2im.hpp` — inverse im2col for transposed conv
- `gemm.hpp` — weight multiply
- `deconvolutional_kernels.cu` — GPU path

**Constraints/Open Questions**:
- No `batch_normalize` support — needs BN fields, forward/backward BN hooks
- GPU path from darknet.CG reference: verify correctness, add `push_/pull_` sync
- No `dilation` support; no `groups` support
- No padding parameter in CFG parser (hard error on `pad=1`)

---

### 3. Transformer Layer (Swin) — `transformer_layer`

**Role**: Windowed multi-head self-attention. Divides feature map into size×size windows; optionally shifts by size/2 for cross-window communication.

**Dependencies**:
- `transformer_kernels.cu` — QKV projections, softmax, attention accumulation
- LayerNorm (inline in kernels), FFN (GELU/leaky MLP)

**GPU Optimizations (applied)**:
- All 6 attention GEMMs use `cublasSgemmStridedBatched` — single dispatch per op vs N sequential calls
- LayerNorm forward uses warp-shuffle reduction (32 threads/token, `__shfl_xor_sync`), `rsqrtf`
- See `docs/transformer_layer.md` → GPU Implementation Notes for details

**Constraints/Open Questions**:
- `max_grad_norm` MUST be 5.0 (not 1.0) — see `memory/feedback_swin_grad_norm.md`
- Training with SGD is suboptimal (paper uses AdamW); LR needs careful warmup
- Window padding for non-divisible H/W must be undone before residual add
- LayerNorm backward `dgamma/dbeta` uses `atomicAdd` — correct, some contention at large batch×windows

---

### 4. ViT Layer (Global) — `vit_layer`

**Role**: Full global self-attention over all T = H×W spatial tokens. Absolute positional embeddings, single transformer block per layer.

**Dependencies**:
- `vit_kernels.cu` — global QKV, softmax over T×T, pos embed interp

**Constraints/Open Questions**:
- Memory: `vit_attn_scores` = [B, heads, T, T]. At 416×416, T=173056 → T²=30B. **Only use deep in network (13×13 or 26×26 feature maps).**
- `resize_vit_layer()` must bicubic-interpolate positional embeddings when network resizes
- No masking/padding mask support

---

### 5. DCNv3 — `dcnv3_layer`

**Role**: Grouped deformable depthwise convolution with softmax-normalized attention weights. Each group learns independent offsets and attention over K² sampling points. From InternImage.

**Dependencies**:
- `papers/dcnv3/` — reference CUDA kernels (`dcnv3_im2col_cuda.cuh`, `dcnv3_cuda.cu`)
- `batchnorm_layer.hpp`
- `gemm.hpp` (linear projection after aggregation)

**Constraints/Open Questions**:
- Channel dim locked: `filters` forced to equal input channels (aggregation operator)
- Need to port `dcnv3_im2col_cuda.cuh` bilinear sampling + softmax normalization into `dcnv3_kernels.cu`
- CPU backward is a stub — needs analytic gradient through softmax-weighted bilinear
- Groups must divide channels evenly; group_channels = C / groups

---

### 6. DCNv4 — `dcnv4_layer`

**Role**: Flash-accelerated deformable aggregation. Combines per-point offsets + weights in a single fused CUDA kernel (no explicit im2col buffer). Optional `remove_center`, `softmax`, and sparse `d_stride` for efficiency.

**Dependencies**:
- `batchnorm_layer.hpp`, `gemm.hpp`, `im2col.hpp`

**GPU Optimizations (applied)**:
- Forward kernel dispatches `d_stride=2` ILP when `D = n/groups` is even: each thread accumulates 2 adjacent channels per sampling point, halving thread count and improving arithmetic density
- Bilinear function `dcnv4_im2col_bilinear_gpu` loops over `c_per_thread` channels with boundary checks hoisted; NHWC layout makes adjacent channel accesses stride-1 (coalesced)
- Expanded K dispatch: supports all kernel sizes 2×2 through 7×7 (K=4,8,9,15,16,24,25,48,49)
- Fixed `p_mask[9]` → `p_mask[49]` in backward (correctness fix for K > 9)

**Constraints/Open Questions**:
- Channel dim locked: `filters` forced to equal input channels
- `offset_filters` padded to multiple of 8 for CUDA alignment — do not change
- Flash kernel requires CUDA ≥ 11.0
- Config `d_stride > 1` (spatial coarsening) not yet implemented; offsets predicted at full resolution
- `block_thread` must be power-of-2 and ≤ 1024

---

## Implementation Plan

### Phase 1 — Deconvolutional Layer (LOW RISK, ISOLATED)

- [ ] Add `batch_normalize` support to `make_deconvolutional_layer()`
- [ ] Add BN forward/backward hooks in `forward_/backward_deconvolutional_layer()`
- [ ] Verify GPU kernels against `unet_darknet/` reference (diff `deconvolutional_kernels.cu`)
- [ ] Fix `push_/pull_deconvolutional_layer()` for full weight sync
- [ ] Add CFG support for `batch_normalize=1` in `parse_deconvolutional_section()`
- [ ] Test: U-Net style decoder with 2× upsampling

### Phase 2 — DCNv3 CPU/GPU (HIGH COMPLEXITY)

- [ ] Port `papers/dcnv3/cpu/dcnv3_cpu.cpp` → CPU forward `forward_dcnv3_layer()`
- [ ] Implement CPU backward: gradient through softmax + bilinear interpolation
- [ ] Port `papers/dcnv3/cuda/dcnv3_im2col_cuda.cuh` → `dcnv3_kernels.cu` forward kernel
- [ ] Port `dcnv3_cuda.cu` col2im path → backward GPU kernel
- [ ] Wire `update_dcnv3_layer_gpu()` with SGD + BN parameter update
- [ ] Test: verify offset convergence on small feature map (16×16)

### Phase 3 — DCNv4 GPU (FUNCTIONAL — remaining TODOs)

- [x] Forward GPU kernel: flash deformable sampling with im2col offset prediction
- [x] Backward GPU kernel: gradient w.r.t. input, offsets, attention masks
- [x] K dispatch expanded (2×2 through 7×7, with/without center removal)
- [x] d_stride=2 ILP dispatch for even-D groups
- [x] Fixed backward p_mask stack overflow for K > 9
- [ ] CPU fallback forward/backward (for debugging without GPU)
- [ ] Spatial `d_stride` coarsening: predict offsets at H/d_stride × W/d_stride then bilinear upsample
- [ ] Test: compare outputs against papers/dcnv4 reference on identical input

### Phase 4 — Transformer / ViT Training Stability (ONGOING)

- [ ] Confirm `max_grad_norm=5.0` is enforced (not 1.0) in training configs
- [ ] Add gradient norm monitoring output for transformer layers
- [x] Implement bilinear pos-embed interpolation in `resize_vit_layer()` — `vit_pos_embed_bilinear_resize_kernel` added to `vit_kernels.cu`, old embed interpolated before free
- [ ] Add masked attention support for ViT (pad tokens should not attend)
- [ ] Benchmark ViT memory at 13×13 vs 26×26 feature scales

---

## Key Risks

| Risk | Layer | Mitigation |
|------|-------|------------|
| DCNv3/DCNv4 gradient instability | DCNv3, DCNv4 | offset_scale clamping, verify softmax numerical stability |
| Transposed conv output size mismatch | Deconv | formula: out = stride*(in-1) + size; verify against reference |
| ViT T² memory blow-up | ViT | restrict to deep layers only; add assertion on T |
| Swin window shift mask errors | Transformer | unit test shifted vs non-shifted attention sums |
| BF16 precision in im2col paths | All | see `memory/feedback_bf16_precision_rules.md` |
