# TODO — Layer HTML Diagram Pages

> See `html_diagrams_plan.md` for full design spec (SVG style, color palette, animation pattern).

## Files to Create

- [x] `docs/transformer_layer.html` — **30 steps** (canonical template, build first)
- [x] `docs/vit_layer.html`          — **24 steps** + O(T²) memory callout
- [x] `docs/dcnv4_layer.html`        — **9 steps**  + C=N constraint callout
- [x] `docs/deconvolutional_layer.html` — **7 steps** + BN-not-supported warning
- [x] `docs/clifford_layer.html`     — **14 steps** + wedge/inner algebra callout
- [x] `docs/graph_conv_layer.html`   — **14 steps** + credit-overlap diagram
- [x] `docs/index.html`              — card grid linking all 6 pages

---

## Per-File Step Counts (from .hpp checkpoints)

| File | Steps | Source |
|---|---|---|
| transformer_layer.html | 30 | `src-lib/transformer_layer.hpp` [01]–[30] |
| vit_layer.html | 24 | `src-lib/vit_layer.hpp` [01]–[24] |
| dcnv4_layer.html | 9 | `src-lib/dcnv4_layer.hpp` + `docs/dcnv4_layer.md` |
| deconvolutional_layer.html | 7 | `src-lib/deconvolutional_layer.hpp` + `docs/deconvolutional_layer.md` |
| clifford_layer.html | 14 | `src-lib/clifford_layer.hpp` + `src-lib/clifford_layer.cpp` |
| graph_conv_layer.html | 14 | `src-lib/graph_conv_layer.hpp` [01]–[14] |

---

## Diagram Rules (quick ref)

- Tensors → **3D isometric box** (3 polygon faces: front / top / right)
- Operations → **flat rounded rect**, color by type (amber=LN, purple=attn, green=FFN, gray=residual, red=deform)
- Attention matrices [T×T] → **flat 2D heatmap grid** (not 3D)
- Arrows → SVG `<line>` with `stroke-dashoffset` animation on scroll
- Animation trigger → `IntersectionObserver` adds `.visible` class → CSS transitions fire

## Color Palette

| Role | Hex |
|---|---|
| Input tensor | `#FF8C55` / `#FFAB7F` / `#D4623A` (front/top/right) |
| Intermediate tensor | `#5BA4E8` / `#85BEF0` / `#3A7DC4` |
| Output tensor | `#3ECFCF` / `#72E0E0` / `#1EA5A5` |
| LayerNorm op | `#F5A623` |
| Attention op | `#9B59B6` |
| FFN/projection op | `#27AE60` |
| Residual op | `#95A5A6` |
| Deformation/offset | `#E74C3C` |

---

## Verification Checklist (run after each file)

- [x] Opens in browser without server
- [x] All `.step` cards animate on scroll
- [x] Nav links to sibling pages work
- [x] `grep -c 'class="step"'` matches expected count
- [x] Renders readable at <768px width
