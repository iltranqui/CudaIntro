# Transformer Layer (Swin-Style Windowed Attention)

**Status**: FUNCTIONAL — training observed but unstable under some configs  
**Files**: `src-lib/transformer_layer.{hpp,cpp}`, `src-lib/transformer_kernels.cu`  
**CFG key**: `[transformer]`

---

## What It Is

A drop-in replacement for `[conv]` that uses **multi-head self-attention** instead of a fixed-weight convolution. Implements the Swin Transformer's core idea: divide the feature map into **non-overlapping windows** of size×size tokens and run attention only within each window.

Cross-window communication uses **cyclic shift** (`shift=1`): the feature map is rolled by size/2 before windowing so that adjacent windows in the shifted partition overlap with non-adjacent windows in the original.

Reference: [Swin Transformer (Liu et al., 2021)](https://arxiv.org/abs/2103.14030).

---

## Architecture (Forward Pass)

```
Input [B, C, H, W]
       │
       ├──▶ Pad H,W to multiples of window size (if needed)
       │
       ├──▶ [shift=1] Cyclic shift by size/2 in both dims
       │
       ├──▶ Partition into windows: [B·Nw, T=size², C]   (Nw = padded_H/size * padded_W/size)
       │
       ├──▶ LayerNorm (LN1)
       │
       ├──▶ QKV projections: Q,K,V each [B·Nw, T, d_head·heads]
       │
       ├──▶ Multi-head self-attention within each window:
       │       scores = Q·K^T / sqrt(d_head)    [B·Nw, heads, T, T]
       │       [shift=1] Apply attention mask (zero out cross-window leakage)
       │       attn = softmax(scores)
       │       out = attn · V                    [B·Nw, T, d_head·heads]
       │
       ├──▶ Output projection (linear)
       │
       ├──▶ Residual add (+ input, with proj if C≠N)
       │
       ├──▶ LayerNorm (LN2)
       │
       ├──▶ FFN: Linear(N → ffn_ratio·N) → activation → Linear(ffn_ratio·N → N)
       │
       ├──▶ Residual add
       │
       ├──▶ [shift=1] Reverse cyclic shift
       │
       ├──▶ Unpad to original H,W
       │
       └──▶ Output [B, N, H, W]
```

---

## Key Parameters

| CFG Parameter | Default | Description |
|---------------|---------|-------------|
| `filters` | — | Output channels N. If N ≠ C, adds residual projection linear layer. |
| `size` | 7 | Window size M. Windows are M×M tokens. H,W padded to multiples of M. |
| `heads` | 4 | Number of attention heads h. d_head = N / h. |
| `shift` | 0 | 0 = regular windows, 1 = shifted windows (cyclic shift by M/2). |
| `ffn_ratio` | 4 | FFN hidden dim multiplier. FFN hidden = ffn_ratio × N. |
| `activation` | gelu | Activation inside FFN. |

---

## Memory Layout

| Buffer | Shape | Purpose |
|--------|-------|---------|
| `qkv_weights` | [3·N, N] | Combined QKV projection weights |
| `proj_weights` | [N, N] | Output projection weights |
| `ffn1_weights` | [ffn_ratio·N, N] | FFN first linear |
| `ffn2_weights` | [N, ffn_ratio·N] | FFN second linear |
| `ln1_{gamma,beta}` | [N] | LayerNorm 1 params |
| `ln2_{gamma,beta}` | [N] | LayerNorm 2 params |
| `res_proj_weights` | [N, C] | Residual projection (only if C≠N) |
| `workspace` | dynamic | QKV activations, attention scores [B·Nw, heads, T, T] |

---

## Training Notes

**Critical**: `max_grad_norm` must be **5.0**, not 1.0.  
Setting it to 1.0 consistently breaks transformer training by over-clipping gradients during warmup.  
See `memory/feedback_swin_grad_norm.md`.

```cfg
[net]
max_grad_norm=5.0
learning_rate=0.0000261
burn_in=1000
momentum=0.92
decay=0.0005
```

Paper uses AdamW with decoupled weight decay 0.05 and cosine LR schedule. SGD can work but requires conservative LR and longer warmup.

---

## Window Size Constraint

Feature map dimensions must be divisible by `size`. If not, the layer pads to the next multiple internally and removes padding before the residual add.

**Check**: After adding residual, unpad must happen **before** adding to the (unpadded) residual, or shapes mismatch.

---

## Shifted Window Attention Mask

When `shift=1`, the cyclic-shifted windows contain tokens from non-adjacent positions in the original map. An attention mask of −∞ is applied to prevent these tokens from attending to each other.

The mask is computed once at layer init (for fixed H,W) and reused. On `resize_transformer_layer()`, the mask must be recomputed.

---

## CFG Example

Standard Swin-style pair (regular + shifted):

```cfg
[transformer]
filters=128
size=7
heads=4
shift=0
ffn_ratio=4
activation=gelu

[transformer]
filters=128
size=7
heads=4
shift=1
ffn_ratio=4
activation=gelu
```

---

## GPU Implementation Notes

### Attention GEMM (`transformer_kernels.cu`)
All four attention matrix products use `gemm_ongpu_strided_batched` (→ `cublasSgemmStridedBatched`). This replaces sequential per-`(window, head)` `cublasSgemm` calls with a single batched dispatch:

| Operation | Formula | Stride A | Stride B | Stride C |
|-----------|---------|----------|----------|----------|
| Q·K^T | scores = Q @ K^T / √d | T·d | T·d | T·T |
| A·V | out = scores @ V | T·T | T·d | T·d |
| scores^T·dH | dV = scores^T @ dH | T·T | T·d | T·d |
| dH·V^T | dS = dH @ V^T | T·d | T·d | T·T |
| dS·K | dQ = dS @ K | T·T | T·d | T·d |
| dS^T·Q | dK = dS^T @ Q | T·T | T·d | T·d |

With `batch=8`, `nW=4`, `heads=4` → `num_batches=128`: this reduces **128 sequential cuBLAS calls → 1** per operation.

### LayerNorm (`transformer_layernorm_forward_kernel`)
Uses a **warp-shuffle reduction** (32 threads per token):
- Mean: each lane sums its strided elements, then 5 `__shfl_xor_sync` rounds reduce to warp sum
- Variance: same pattern over `(xi - m)^2`
- Normalize: each lane writes its `c_per_thread = C/32` output elements
- `rsqrtf` replaces `1/sqrtf` (single hardware instruction)

Launch: `ceil(total_tokens * 32 / TRANS_BLOCK)` blocks, 256 threads each (8 tokens per block).

### LayerNorm backward (`transformer_layernorm_backward_kernel`)
`dx` computation is warp-independent (no sync needed). `dgamma`/`dbeta` accumulation still uses `atomicAdd` — correct but contended when `total_tokens` is large. Future improvement: stage per-token contributions and reduce separately.

---

## Known Issues / Constraints

1. **Gradient instability** — `max_grad_norm=5.0` is mandatory; see memory file.
2. **Padding removal timing** — unpad must happen after attn but before residual add. Bug-prone on resize.
3. **Window size vs feature map** — small feature maps (e.g., 7×7 with size=7) produce Nw=1 window — attention degenerates to global. Fine but wasteful.
4. **No relative positional bias** — original Swin uses relative position bias in attention scores. Currently implemented via `tf_rel_pos_bias_gpu`.
5. **SGD vs AdamW** — Darknet uses SGD. Transformer layers prefer AdamW. Use very low LR with high momentum (see Training Notes).
6. **Mixed-size networks** — if transformer is followed by a conv that expects un-padded H,W, ensure transformer output is properly unpadded.
7. **LayerNorm backward dgamma/dbeta** — uses atomicAdd per channel per token; contended for large `total_tokens * C`. Not a correctness issue.
