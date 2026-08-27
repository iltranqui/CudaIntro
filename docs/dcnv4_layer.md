# DCNv4 Layer (Flash-Accelerated Deformable Aggregation)

**Status**: GPU-ONLY — forward/backward functional; CPU stubs only; spatial d_stride implemented  
**Files**: `src-lib/dcnv4_layer.{hpp,cpp}`, `src-lib/dcnv4_kernels.cu`  
**Reference**: `papers/dcnv4/` (Flash-Deformable CUDA kernels)  
**CFG key**: `[dcnv4]`

---

## What It Is

DCNv4 is the latest evolution of deformable convolution. It improves over DCNv3 with:

1. **Fused CUDA kernel** — no intermediate im2col buffer; sampling, weighting, and accumulation happen in a single pass (hence "Flash" in the reference implementation)
2. **Unified offset + weight prediction** — offset_filters = `G × K² × 3` (Δy, Δx, weight per point, per group), padded to multiple of 8 for CUDA alignment
3. **Optional center exclusion** (`remove_center=1`) — skips the center K point to enforce peripheral context
4. **Sparse offset field** (`d_stride`) — offsets predicted at coarser spatial resolution for efficiency
5. **Optional softmax normalization** (`softmax=1`) — same as DCNv3's group-wise softmax

Like DCNv3, it is a **spatial aggregation operator only** — channel dimension is fixed (C = N forced).

Reference: [Flash-DCNv4 / InternImage v2 (2023)](https://arxiv.org/abs/2211.05778).

---

## Architecture

```
Input [B, C, H, W]
       │
       ├──▶ Offset prediction (at d_stride spatial resolution):
       │       offset_field [B, H'', W'', G·K_eff·3]
       │       K_eff = K² - (1 if remove_center else 0)
       │       offset_dim = G · K_eff · 3   padded to multiple of 8
       │       H'' = H' / d_stride   (coarser if d_stride > 1)
       │
       ├──▶ Upsample offset field back to H'×W' (bilinear, if d_stride > 1)
       │
       ├──▶ Split offset_field: Δy [G,K_eff], Δx [G,K_eff], weight [G,K_eff]
       │       if softmax=1: softmax(weight, dim=K_eff) per group
       │       scale Δy, Δx by offset_scale
       │
       ├──▶ Flash grouped deformable sampling (single fused CUDA kernel):
       │       For each output position p, each group g, each active point k:
       │           sample_y = p_y·stride + k_y·dilation + Δy[g,k]
       │           sample_x = p_x·stride + k_x·dilation + Δx[g,k]
       │           val = bilinear(input[g·Cg:(g+1)·Cg], sample_y, sample_x)
       │           output[g·Cg:(g+1)·Cg] += weight[g,k] · val
       │
       ├──▶ Optional BN + activation
       │
       └──▶ Output [B, C, H', W']
```

---

## Key Parameters

| CFG Parameter | Default | Description |
|---------------|---------|-------------|
| `filters` | — | **Must equal input channels C** |
| `size` | 3 | Kernel size K |
| `stride` | 1 | Spatial stride (or `stride_x` / `stride_y`) |
| `dilation` | 1 | Dilation |
| `pad` | size/2 | Padding |
| `groups` | 1 | Deformable groups G. C must be divisible by G. |
| `offset_scale` | 1.0 | Scale applied to offset predictions |
| `remove_center` | 0 | Skip center point of kernel (forces peripheral sampling) |
| `d_stride` | 8 | Offset field downsample factor. 1=per-pixel, 8=sparse/coarse. |
| `block_thread` | 256 | CUDA block thread count. Must be power-of-2, ≤ 1024. |
| `softmax` | 0 | 0=sigmoid-scaled weights, 1=softmax-normalized weights |
| `batch_normalize` | 0 | BN after aggregation |
| `activation` | leaky | Activation |

---

## Offset Dimension Formula

```cpp
int K     = size * size;
int K_eff = remove_center ? (K - 1) : K;
int offset_filters_raw = groups * K_eff * 3;           // Δy + Δx + weight
int offset_filters     = ((offset_filters_raw + 7) / 8) * 8;  // pad to 8
```

This padding is **required** for CUDA memory alignment in the flash kernel.

---

## Memory Layout

| Buffer | Shape | Purpose |
|--------|-------|---------|
| `weights` | [(C/G)·N·K·K] | Legacy conv weights (may be unused in pure aggregation; verify) |
| `offset_weights` | [offset_filters, C, 1, 1] | Combined Δy+Δx+weight prediction subnet |
| `offset_biases` | [offset_filters] | Offset subnet biases |
| `workspace` | dynamic | Input/output NHWC buffers, offset buffer (×3 overallocation) |

Workspace formula (from `get_dcnv4_workspace_size()`):
```
3 × (input_nhwc + output_nhwc + offsets_nhwc + im2col)
```

---

## Current State

`dcnv4_layer.cpp` has:
- `make_dcnv4_layer()` — complete: allocates all buffers, handles BN, offset subnet
- `forward_dcnv4_layer()` — CPU stub (warning only; use GPU path)
- `backward_dcnv4_layer()` — CPU stub
- `update_dcnv4_layer()` — CPU stub

`dcnv4_kernels.cu` has:
- `forward_dcnv4_layer_gpu()` — **functional**: im2col offset prediction + flash deformable sampling
- `backward_dcnv4_layer_gpu()` — **functional**: bilinear grad w.r.t. input, offsets, and attention masks
- `update_dcnv4_layer_gpu()` — SGD on offset weights/biases + BN scales

---

## Implementation Plan

### Step 1: CPU Fallback Forward (for debugging)

1. Run offset subnet: `gemm(offset_weights, input)` → raw offsets [B, H'', W'', offset_filters]
2. If `d_stride > 1`: upsample offset field to [B, H', W', offset_filters]
3. Split into Δy, Δx, weight tensors
4. If `softmax=1`: apply softmax over K_eff dim per group
5. Scale Δy, Δx by `offset_scale`
6. For each position, group, active point: bilinear sample + weighted accumulate

### Step 2: GPU Forward (Flash Kernel)

Port `papers/dcnv4/cuda/flash_deform_im2col_cuda.cuh`:

Key design: single kernel launch where each CUDA thread handles one (position, group) pair.
- Shared memory preloads: input tiles around sampling neighborhood
- Warp-level reduction for the K_eff accumulation
- `block_thread` controls occupancy vs register pressure tradeoff

### Step 3: GPU Backward

Port `papers/dcnv4/cuda/flash_deform_col2im_cuda.cuh`:
- Gradient w.r.t. input: atomic adds at bilinear neighbors
- Gradient w.r.t. Δy, Δx: pixel gradient `∂I/∂y`, `∂I/∂x`
- Gradient w.r.t. weight: accumulated bilinear value at each sample point

### Step 4: d_stride Offset Upsample

If `d_stride > 1`, the offset field is predicted at `[H/d_stride, W/d_stride]` and bilinearly upsampled to `[H', W']`. The backward must propagate through this upsample (bilinear backward / average pooling).

---

## CFG Example

```cfg
# DCNv4 in deep backbone (128 channels, groups=4, sparse offsets)
[dcnv4]
batch_normalize=1
filters=128
size=3
stride=1
pad=1
groups=4
offset_scale=1.0
remove_center=0
d_stride=1
block_thread=256
softmax=0
activation=leaky

# DCNv4 with center removal + softmax (closer to original paper)
[dcnv4]
batch_normalize=1
filters=256
size=3
stride=1
pad=1
groups=8
offset_scale=0.5
remove_center=1
d_stride=1
block_thread=256
softmax=1
activation=gelu
```

---

## GPU Implementation Notes

### d_stride (ILP, not spatial)
The `d_stride` template parameter in `dcnv4_forward_kernel_gpu` controls **per-thread channel ILP** — not the spatial offset coarseness from the config. When `D % 2 == 0`, the kernel dispatches `d_stride=2`: each thread processes 2 adjacent channels per sampling point using a compile-time unrolled inner loop in `dcnv4_im2col_bilinear_gpu`. In NHWC layout, adjacent channel elements at the same spatial position are contiguous → stride-1 memory access, enabling coalesced loads.

> **Note**: the config parameter `d_stride` (offset field spatial stride) is currently unused in the forward/backward — offsets are predicted at full spatial resolution. Implementing spatial downsampling + upsample is a future step.

### Kernel dispatch
Forward kernel dispatches over `(d_stride, K, softmax)`. Supported K values:

| Kernel | K_full | K_center_removed |
|--------|--------|-----------------|
| 2×2 | 4 | 3 |
| 3×3 | 9 | 8 |
| 4×4 | 16 | 15 |
| 5×5 | 25 | 24 |
| 7×7 | 49 | 48 |

Unsupported K prints a warning and skips the forward pass.

### Bilinear sampling
`dcnv4_im2col_bilinear_gpu<scalar_t, transfer_t, c_per_thread>` loops over `c_per_thread` adjacent channels with boundary checks hoisted outside the inner `#pragma unroll` loop. For `d_stride=1`, `c_per_thread=1` (identical to original behavior).

---

## Known Issues / Constraints

1. **CPU path unimplemented** — GPU required for training/inference
2. **CUDA version requirement** — flash kernel requires CUDA ≥ 11.0
3. **d_stride spatial downsampling not implemented** — config `d_stride > 1` is accepted but offsets are still predicted at full resolution; the spatial coarsening + bilinear upsample is a TODO
4. **offset_filters padding to 8** — critical for CUDA alignment; if changed, kernel reads will be wrong
5. **block_thread** — must be power-of-2; no guard in `make_dcnv4_layer()`
6. **groups must divide C** — no guard; silent wrong behavior otherwise
7. **remove_center indexing** — skipping center point changes K_eff; kernel maps active indices correctly by skipping `(kernel_w/2, kernel_h/2)`

---

## Relation to DCNv2 and DCNv3

| | DCNv2 | DCNv3 | DCNv4 |
|---|---|---|---|
| Groups | 1 | G | G |
| Offset format | (Δy,Δx) × K² | (Δy,Δx) × G×K² + softmax weight | (Δy,Δx,w) × G×K_eff (fused) |
| Weight predict | Separate sigmoid | Separate softmax | Fused with offset |
| Channel change | Yes | No | No |
| im2col buffer | Explicit | Explicit per group | No (flash fused) |
| Center skip | No | No | Optional |
| Sparse offsets | No | No | `d_stride` |
| CUDA requirement | CUDA 9+ | CUDA 10+ | CUDA 11+ (flash ops) |
