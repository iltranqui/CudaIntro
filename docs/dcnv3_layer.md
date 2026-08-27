# DCNv3 Layer (Grouped Deformable Depthwise Convolution)

**Status**: STUB — `make_*` and structure complete; forward/backward are non-functional placeholders  
**Files**: `src-lib/dcnv3_layer.{hpp,cpp}`, `src-lib/dcnv3_kernels.cu`  
**Reference**: `papers/dcnv3/` (InternImage CUDA kernels)  
**CFG key**: `[dcnv3]`

---

## What It Is

DCNv3 is the spatial aggregation operator from **InternImage**. It generalizes DCNv2 by:

1. **Grouping** — G independent groups, each with its own offsets and attention weights
2. **Softmax normalization** — attention weights (not sigmoid masks) — sum-to-1 across K² points per group
3. **Depthwise-style aggregation** — no separate weight matrix per output filter; aggregation only

This means DCNv3 **cannot change the channel dimension**. Filters = input channels always. Use a 1×1 conv before/after to project channels.

Reference: [InternImage (Wang et al., 2022)](https://arxiv.org/abs/2211.05778).

---

## Architecture

```
Input [B, C, H, W]
       │
       ├──▶ Linear projection: C → C  (input_proj_weights)
       │
       ├──▶ For each group g (g=0..G-1), channels = C/G:
       │       offset_net: [B, H', W', G·K²·2] = predicted (Δy, Δx) per point per group
       │       attn_net:   [B, H', W', G·K²]   = predicted attention weights per point
       │       softmax(attn, dim=K²) → normalized weights sum to 1 within each group
       │
       ├──▶ Grouped deformable sampling:
       │       For group g, channels c in [g·Cg, (g+1)·Cg):
       │           sample K² points using group g's offsets (bilinear interp)
       │           aggregate: output_c = Σ_k  attn_g_k · sample(input_c, p+k+Δk)
       │
       ├──▶ Linear projection: C → C  (output_proj_weights)
       │
       ├──▶ Optional BN + activation
       │
       └──▶ Output [B, C, H', W']
```

---

## Key Parameters

| CFG Parameter | Default | Description |
|---------------|---------|-------------|
| `filters` | — | **Must equal input channels C** (enforced, warning if mismatch) |
| `size` | 3 | Kernel size K (K×K points sampled) |
| `stride` | 1 | Spatial stride |
| `dilation` | 1 | Dilation |
| `pad` | size/2 | Padding |
| `groups` | 1 | Number of deformable groups G. C must be divisible by G. |
| `offset_scale` | 1.0 | Scale applied to predicted offsets before sampling |
| `batch_normalize` | 0 | BN after aggregation |
| `activation` | leaky | Activation |

---

## Memory Layout

| Buffer | Shape | Purpose |
|--------|-------|---------|
| `weights` | [(C/G)·N·K·K] | Aggregation weights (currently: C/G × N × K × K, but N=C) |
| `offset_weights` | [G·K²·2, C, 1, 1] | Offset subnet weights |
| `offset_biases` | [G·K²·2] | Offset subnet biases |
| `attn_weights` | [G·K², C, 1, 1] | Attention weight subnet |
| `attn_biases` | [G·K²] | Attention biases |
| `workspace` | dynamic | Input/output NHWC buffers, offset/attn buffers, im2col buffer |

Workspace formula (from `get_dcnv3_workspace_size()`):
```
3 × (input_nhwc + output_nhwc + offsets_nhwc + masks_nhwc + im2col)
```

---

## Current State (STUB)

`dcnv3_layer.cpp` has:
- `make_dcnv3_layer()` — complete: allocates all buffers, BN params, offset/attn subnets
- `forward_dcnv3_layer()` — **STUB**: calls `memset(output, 0)`, no actual computation
- `backward_dcnv3_layer()` — **STUB**: returns immediately
- `update_dcnv3_layer()` — partial SGD update

`dcnv3_kernels.cu` has:
- GPU functions declared but bodies are placeholders

---

## Implementation Plan

### Step 1: CPU Forward

Port from `papers/dcnv3/cpu/dcnv3_cpu.cpp`:

1. Run offset subnet: `gemm(offset_weights, input)` → offsets [B, H', W', G·K²·2]
2. Run attn subnet: `gemm(attn_weights, input)` → raw attn
3. Apply softmax over K² dim per group → normalized attn weights
4. Scale offsets by `offset_scale`
5. For each group, each output position, each K² point:
   - Compute sample coord: `y = p_y·stride + k_y·dilation + offset_y`
   - Bilinear interpolate input at (y, x) for channels `[g·Cg : (g+1)·Cg]`
   - Multiply by attn weight, accumulate → output
6. Output linear projection

### Step 2: CPU Backward

Gradients needed:
- `∂L/∂attn_weights`: softmax backward + accumulation
- `∂L/∂offset_weights`: `∂I/∂y` and `∂I/∂x` at sample points (same as DCNv2 Stream C)
- `∂L/∂input`: reverse bilinear scatter (same as DCNv2 Stream B)
- Softmax backward: standard cross-entropy-style grad through softmax

### Step 3: GPU Forward

Port `papers/dcnv3/cuda/dcnv3_im2col_cuda.cuh` into `dcnv3_kernels.cu`:
- Kernel: per output pixel, per group, sample K² points + apply attn weights
- Softmax in kernel (log-sum-exp trick for numerical stability)

### Step 4: GPU Backward

Port `papers/dcnv3/cuda/dcnv3_cuda.cu` col2im path:
- Atomic adds back to input grad
- Offset grad via pixel gradient interpolation

---

## CFG Example

```cfg
[dcnv3]
filters=128
size=3
stride=1
pad=1
groups=4
offset_scale=1.0
batch_normalize=1
activation=gelu
```

---

## Known Issues / Constraints

1. **No CPU implementation yet** — must port from `papers/dcnv3/` reference
2. **Softmax stability** — raw attn logits can be large; need temperature scaling or clipping before softmax
3. **Workspace size** — `×3` multiplier is a conservative overallocation; may need tuning
4. **groups must divide C** — no guard in current code; silent wrong behavior otherwise
5. **Weight matrix semantics** — `l.nweights = (c/groups) * n * size * size`. With `n=c` forced, this is `c²/groups * K²` which may be larger than needed for pure depthwise aggregation. Verify against InternImage reference.

---

## Relation to DCNv2 and DCNv4

| | DCNv2 | DCNv3 | DCNv4 |
|---|---|---|---|
| Groups | 1 | G (configurable) | G (configurable) |
| Attention type | Sigmoid mask | Softmax (sum=1) | Fused weight |
| Weight matrix | Full conv | Input/output proj only | None (aggregation) |
| CUDA style | im2col + GEMM | im2col per group | Flash fused kernel |
| Channel change | Yes | No (C=N forced) | No (C=N forced) |
