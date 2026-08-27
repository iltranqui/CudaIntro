# ViT Layer — Debug & Tuning Notes

Empirical notes from training Darknet's `[vit]` layer on the LegoGears
small-object detector. What works, what breaks, why.

---

## 1. Best config (99.05% mAP)

```cfg
[vit]
dim=512
patch_size=1
heads=8
mlp_dim=2048
pos_embed=learned
pos_init=zero
activation=gelu

[convolutional] filters=512 size=3 stride=2 pad=1   # downsample 14x10 -> 7x5
[convolutional] filters=512 size=1 stride=1 pad=1
```

Single ViT, P=1 (per-token attention at full 14×10), learned pos.
Downsample lives in the conv after, not in the ViT.
Survived both LR-schedule steps (2200, 2600) with no collapse.

---

## 2. Hard rules

| Rule                                                                 | Why                                                              |
|----------------------------------------------------------------------|------------------------------------------------------------------|
| Use `pos_embed=learned pos_init=zero` at `dim>=512`                  | Sinusoidal: ~85% of channels are near-zero on small grids (§4.1) |
| Stay at **1 ViT** unless using the historical 2-ViT (P=2 → P=1) baseline | Latent instabilities compound per ViT (§5)                        |
| Use **odd P** for same-res (`stride=1 pad=(P-1)/2`)                  | Triggers true identity init. Even P falls into box-blur init      |
| Keep `T = (H * W) / stride² <= 200`                                  | Attention buffer is `B·heads·T²·4` bytes (§3)                     |
| Eval `_best.weights`, never `_last.weights`                          | If training collapses near end, `_last` is poisoned               |

---

## 3. Stride / pad invariants

From `vit_validate_shape`:

| Mode               | P   | S  | pad      | constraint                       | out                 |
|--------------------|-----|----|----------|----------------------------------|---------------------|
| Classic            | P   | P  | 0        | `H % P == 0 && W % P == 0`       | `H/P × W/P`         |
| Same-res           | odd | 1  | (P-1)/2  | none                             | `H × W`             |
| Overlap downsample | any | <P | any      | `(H+2·pad-P) % S == 0`           | `(H+2·pad-P)/S + 1` |

Even P + same-res = impossible (no symmetric pad).

**Attention cost** (`B=32, heads=8`):

| Map  | T   | Attn buffer per ViT |
|------|-----|---------------------|
| 7×5  | 35  | ~1.2 MB             |
| 14×10| 140 | ~20 MB              |
| 26×26| 676 | ~470 MB             |
| 52×52|2704 | ~7.5 GB             |

Same-res ViT patch-embed GEMM scales as `P²·C` — cheap vs attention.

---

## 4. What does NOT work

### 4.1 Sinusoidal pos at small grid + large dim

`vit_fill_2d_sinusoidal_pos_embed` uses `div = exp(-log(10000)·band/quarter)`
with `quarter = C/4 = 128`. For `band > ~30`, `div < 1e-2`, so
`sin(pos · div) ≈ 0` for ~85% of 512 channels on 14×10 grids.
Attention has nothing to discriminate tokens by → softmax flattens →
permutation-invariant garbage. mAP collapses to ~0%.

Fix: `pos_embed=learned pos_init=zero`.

### 4.2 Even P (no identity init)

`vit_init_dropin_patch_embed` falls into box-blur for even P
(`1/(P²)` weight on every position in the P×P window). Network at t=0 has
a P×P average-pool inserted vs no-ViT baseline → gradient must "undo the
blur" before learning anything useful.

**Confirmed empirically (2026-05-23)**: P=2 classic (S=2, pad=0, 14×10→7×5)
on LegoGears collapsed to mAP=6.89%. AP>0 on big objects (gears), TP=0 at
conf=0.25 across all classes. Small objects (<16px: red light, pin, center)
went to 0 AP — 2×2 avg-pool init annihilates their signal at iter 0, head
never recovers. Box-blur not just suboptimal — actively destructive when
small objects dominate the dataset.

Only `P=1` (classic, init = identity) or odd P (same-res, init = true
identity at center pixel) is safe.

### 4.3 Stacks of 3+ ViTs

Even with identity init, 3+ ViTs collapses or never reaches usable mAP.
See §5 for why.

### 4.4 P=3 ≥ P=1 (no gain, slightly worse on rare classes)

P=3 same-res adds 2M patch-embed params (512×4608 vs 512×512). On this
dataset: 97.73% mAP vs P=1's 99.05%. Upstream conv stack already gives
each token a wide effective RF — 3×3 patchify input is redundant. Extra
params mildly hurt the rarest class (red light).

### 4.5 "Bigger P = more global attention" — wrong

Attention is global at **any P**: every token attends to every other.
P only controls:
1. Token count `T = H·W / stride²` (attention cost)
2. Per-token input dim `K = P²·C` (richer QKV input)
3. Hard spatial pooling before attention

If global mixing is the goal, P=1 already has it.

---

## 5. Why stacks collapse (latent bugs)

Single ViT survives. 2+ ViTs hit collapse around the iter-2200 LR step.
Same code; different outcome because per-ViT instability sources compound:

| Source                                | Per-ViT contribution                        |
|---------------------------------------|---------------------------------------------|
| mHC residual logits (12/ViT)          | Branch coefficients can co-rise across ViTs |
| LN γ/β (no weight decay applied)      | Drifts unbounded with steady-sign gradient  |
| Momentum buffer at LR-schedule step   | Post-step overshoot, per ViT's worth        |
| Stacked attention non-linearity       | Per-layer perturbation × geometric          |

Latent bugs still in code:
- mHC "0.1× LR" partially defeated by momentum → effective ~1.25× lr
  (`vit_layer.cpp:1740`, `vit_kernels.cu:1437`)
- LN γ/β never get `-decay·gamma` term (`vit_layer.cpp:1745-1754`)
- Darknet update-then-decay momentum overshoots after LR cut

Single ViT fits inside the error budget. 2+ doesn't.

---

## 6. NaN-source cheat sheet

| Symptom                                          | Likely cause                                                |
|--------------------------------------------------|-------------------------------------------------------------|
| `AP ≈ 0` across all classes from iter 1         | Pos embed dead (sinusoidal on small grid) — use learned     |
| `AP > 0` but `TP = 0` at conf=0.25               | Head starved; reduce ViT depth                              |
| Late-training collapse after LR-step (iter 2200) | Latent §5 bugs; reduce ViT count or remove LR schedule      |
| NaN at yolo layer, propagates back               | CIoU `v = (4/π²)·(atan(gt_w/gt_h) - atan(p_w/p_h))²` on degenerate aug box. Not a ViT bug; tighten clamps as mitigation |
| Small objects vanish when adding ViTs            | Box-blur init (even P) or P too large in downsampler        |
| OOM on attn buffer                               | Check `T = H·W / S²` against §3 table                       |

Current clamps (`vit_layer.cpp:18-21`, mirrored `vit_kernels.cu:18-21`):
```
VIT_ATTENTION_QK_CLAMP    = 128
VIT_ATTENTION_SCORE_CLAMP =  10
VIT_FEATURE_CLAMP         =  10
VIT_GRAD_CLAMP            =  20
```

---

## 7. Files for stride/pad support

- `darknet_layers.hpp` — `vit_patch_stride`, `vit_patch_pad` fields
- `darknet_cfg.cpp` — parser reads `patch_stride`, `patch_pad`
- `vit_layer.hpp` — `make_vit_layer` signature
- `vit_layer.cpp` — validate, patchify CPU, scatter backward, identity
  init, resize, diagnostics
- `vit_kernels.cu` — patchify + delta-to-spatial kernels, launch sites at
  lines 795, 1271, 1293
- `src-test/test_vit_layer.cpp` — 6 `make_vit_layer` call sites

Cfg without `patch_stride`/`patch_pad` → bit-exact identical to pre-patch
layer.

---

## 7b. Tucker attention as depth-stable alternative

Confirmed 2026-05-23: 4 stacked `[tucker_attention]` layers train without
collapse. Layout:

| #   | Slot         | ch  | size | rank | heads |
|-----|--------------|-----|------|------|-------|
| T1  | 28×20        | 256 | 5    | 32   | 4     |
| T3  | 28×20 (stack)| 256 | 7    | 32   | 4     |
| T2  | 14×10        | 512 | 5    | 64   | 8     |
| T4  | 14×10 (stack)| 512 | 7    | 64   | 8     |

ViT collapsed at depth=3+ same-scale (§5). Tucker survives depth=2 at each
of 2 scales = 4 total attention layers.

Why: low-rank Q/K/V/O factors (rank=c/8) bound per-layer singular values,
capping amplification. Full attention has no such cap → §5 compounding.
Tucker is the depth-stable choice; ViT is the single-layer-global-mixing
choice.

Strict contract: tucker requires `filters == input channels` at construct
time (`tucker_attention_layer.cpp:208`). ViT silently falls back to
random init when `l.n != l.c`. Two different ergonomics.

Example: `cfg/LegoGears_tucker.cfg`.

---

## 8. Confirmed scaling: ViT at higher resolution

Single ViT P=1 dim=128 at stride-8 (28×20, T=560) trains successfully
(2026-05-23). 4× more tokens than the deep-slot baseline, no collapse.

Implication: single-ViT failure mode is depth/stacking, not attention
surface size. Identity init + learned-zero pos handles large T fine.

Confounder: dim dropped 512→128 to match channel count at this depth.
Smaller dim = smaller momentum mass = less post-LR-step overshoot. Cannot
isolate "more tokens" from "smaller dim" without further test.

Next test idea: P=1 ViTs at multiple resolutions (stride-8 dim=128 +
stride-16 dim=512). Distributes ViTs across scales instead of stacking at
same scale → may dodge §5 compounding.

---

## 9. Open questions

- Fix LN γ/β weight decay → does the 2+ ViT collapse go away?
- Replace `scal(momentum, scale_updates)` with explicit slow buffer for
  mHC → does the "0.1× LR" mitigation actually work then?
- Asymmetric padding API for even P same-res (P=4 on 7×5) — worth it?
- Clamp `v` in CIoU loss to stop yolo-side NaN poisoning all upstream layers?
- Multi-scale ViT placement (§8 idea) vs same-scale stack — does
  distributing across resolutions avoid the §5 collapse?
