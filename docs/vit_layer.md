# Vision Transformer Layer (Global Attention)

**Status**: FUNCTIONAL — patch-based, memory-intensive, learned/sinusoidal pos-embed support
**Files**: `src-lib/vit_layer.{hpp,cpp}`, `src-lib/vit_kernels.cu`  
**CFG key**: `[vit]`

---

## What It Is

A patch-based **Vision Transformer (ViT)** encoder block with **global** multi-head self-attention over all T = (H/P)×(W/P) patch tokens. Every token can attend to every other token — no windowing.

Unlike the `[transformer]` (Swin) layer which restricts attention to local windows, `[vit]` runs full O(T²) attention. This makes it expensive but gives maximum receptive field.

Reference: [An Image is Worth 16×16 Words (Dosovitskiy et al., 2021)](https://arxiv.org/abs/2010.11929).

---

## CRITICAL: Memory Constraint

Attention scores buffer: `[B, heads, T, T]` where T = (H / patch_size) × (W / patch_size).

| Feature map / patch | T | T² per head | At B=4, heads=4 |
|---|---|---|---|
| 26×26 / P=1 | 676 | 456,976 | ~460M floats ≈ 1.7 GB |
| 52×52 / P=2 | 676 | 456,976 | ~460M floats ≈ 1.7 GB |
| 416×416 / P=16 | 676 | 456,976 | ~460M floats ≈ 1.7 GB |
| 416×416 / P=1 | 173,056 | **3×10¹⁰** | **Out of memory** |

**Rule**: keep the patch-token count small. Downsample before `[vit]`, increase `patch_size`, or both.

---

## Architecture (Forward Pass)

```
Input [B, C, H, W]
       │
       ├──▶ Patchify: [B, T, P·P·C]  where T = (H/P) × (W/P)
       │
       ├──▶ Patch embedding: Linear(P·P·C → N)
       │
       ├──▶ Add positional embeddings: token_i = token_i + pos_embed_i  [T, N]
       │
       ├──▶ LayerNorm (LN1)
       │
       ├──▶ QKV projections: Q,K,V each [B, T, d_head·heads]
       │
       ├──▶ Global multi-head self-attention:
       │       scores = Q·K^T / sqrt(d_head)    [B, heads, T, T]
       │       attn = softmax(scores, dim=-1)
       │       context = attn · V               [B, heads, T, d_head]
       │       out = concat heads → project     [B, T, N]
       │
       ├──▶ Residual add (+ input; with projection if C≠N)
       │
       ├──▶ LayerNorm (LN2)
       │
       ├──▶ FFN: Linear(N → mlp_dim) → activation → Linear(mlp_dim → N)
       │
       ├──▶ Residual add
       │
       └──▶ Reshape to [B, N, H/P, W/P]
```

---

## Key Parameters

| CFG Parameter | Default | Description |
|---------------|---------|-------------|
| `dim` / `filters` | 128 | Output token width N. `dim` is the SimpleViT-style alias. |
| `patch_size` | 1 | Patch edge length P. Input H and W must be divisible by P. |
| `heads` | 4 | Attention heads. d_head = N / heads. Must divide N evenly. |
| `mlp_dim` | `dim * ffn_ratio` | FFN hidden width. |
| `ffn_ratio` | 4 | Backward-compatible shorthand for `mlp_dim = dim * ffn_ratio`. |
| `pos_embed` | learned | `learned` absolute embedding or fixed 2D `sinusoidal` SimpleViT-style embedding. |
| `pos_init` | random | Learned positional embedding init: `random` small values or `zero` for detector drop-in starts. |
| `activation` | gelu | FFN activation |

No `size` or `shift` parameter — attention is always global.

---

## Positional Embeddings

Positional embeddings: `vit_pos_embed[T, N]` — one embedding vector per patch-grid position.

**Learned init behavior**: `pos_init=zero` keeps the ViT from adding full-scale position features at step zero. This is preferred when inserting `[vit]` into an existing detector cfg.

**Learned resize behavior**: When `resize_vit_layer()` is called (network resize with `random=1`), T changes. Learned pos_embed is **bilinearly interpolated** (center-aligned coordinates) from the old patch grid to the new patch grid via `vit_pos_embed_bilinear_resize_kernel` in `vit_kernels.cu`.

**Sinusoidal resize behavior**: `pos_embed=sinusoidal` regenerates the fixed 2D embedding for the new patch grid and does not train positional embedding weights.

---

## Memory Layout

| Buffer | Shape | Purpose |
|--------|-------|---------|
| `vit_patch_embed` | [N, P·P·C] | Patch projection |
| `vit_pos_embed` | [T, N] | Learned or fixed positional embeddings |
| `qkv_weights` | [3·N, N] | Combined QKV projection |
| `proj_weights` | [N, N] | Output projection |
| `ffn1_weights` | [mlp_dim, N] | FFN first linear |
| `ffn2_weights` | [N, mlp_dim] | FFN second linear |
| `ln1_{gamma,beta}` | [N] | LN1 params |
| `ln2_{gamma,beta}` | [N] | LN2 params |
| `scales` | [12] | Trainable mHC residual mixer params saved in weights version 0.2.6+ |
| `vit_attn_scores` | [B, heads, T, T] | **THE BIG ONE — see memory table above** |
| `vit_attn_probs` | [B, heads, T, T] | Post-softmax (may alias scores buffer) |

---

## Training Notes

Same optimizer constraints as `[transformer]`:
- SGD is suboptimal; paper uses AdamW
- Warmup is critical — LR cold-start causes attention score explosion
- Gradient clipping: `max_grad_norm=5.0`

ViT is more sensitive to weight init than conv layers. If training diverges in first 100 iterations, check:
1. LR is 10-100× smaller than typical conv training LR
2. Positional embeddings initialized to small normal (not zeros)
3. QKV weights initialized with smaller std (e.g., `1/sqrt(C)`)

---

## CFG Example

```cfg
# After several conv + maxpool stages — feature map should be 13x13 or smaller

[vit]
dim=256
patch_size=2
heads=4
mlp_dim=512
pos_embed=learned
pos_init=zero
activation=gelu
```

---

## Comparison: ViT vs Transformer (Swin)

| | `[vit]` (global) | `[transformer]` (Swin) |
|---|---|---|
| Attention scope | All T tokens | Within M×M window |
| Memory for scores | O(T²) | O(T·M²/T) = O(M²) |
| Best resolution | ≤26×26 feature maps | Any (early layers ok) |
| Cross-region info | Every layer | Every 2 layers (shift) |
| Positional encoding | Learned absolute or fixed 2D sinusoidal | Relative bias (not yet impl) |
| Padding handling | None needed | Pad to multiple of M |

---

## Known Issues / Constraints

1. **Memory explosion at early layers** — hard limit: T ≤ ~700 for reasonable GPU memory at batch=4.
2. **No padding mask** — all T tokens treated as valid; no support for variable-size inputs within a batch.
3. **Learned pos embed resize** — bilinearly interpolates from old to new patch grid; sinusoidal embeddings are regenerated.
4. **Heads must divide N** — guarded at layer creation.
5. **No classifier head** — this is a feature-map encoder block. SimpleViT `num_classes` and pooling belong in a later head/layer.
