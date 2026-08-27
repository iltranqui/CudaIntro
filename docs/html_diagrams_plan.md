# Plan: Layer Documentation HTML Pages

## Context

Six novel neural-network layers have been implemented in `src-lib/` and described in rich detail (with ASCII forward-pass checkpoints, symbol legends, CFG tables) in their `.hpp` headers and in `docs/*.md` markdown files.  The goal is to turn that existing knowledge into interactive, self-contained HTML pages — one per layer — that explain each numbered forward-pass step with an animated SVG diagram, so that a reader can scroll through and visually trace the data flow without opening source code.

---

## Project Tree

```
docs/
├── index.html                     ← overview + nav card grid (NEW)
├── transformer_layer.html         ← Swin windowed attention    (NEW)
├── vit_layer.html                 ← ViT global attention       (NEW)
├── dcnv4_layer.html               ← Deformable Conv v4         (NEW)
├── deconvolutional_layer.html     ← Transposed convolution     (NEW)
├── clifford_layer.html            ← Clifford algebra layer     (NEW)
└── graph_conv_layer.html          ← Graph convolutional        (NEW)

(existing .md files are unchanged, used as source of truth)
```

---

## Component Breakdown

### Shared HTML Template (inlined per file)

**Role:** Every HTML file is fully self-contained — no external CSS or JS files.

**Structure per file:**
```
<head>  ← inline <style> with all CSS + CSS @keyframes
<body>
  <header>  ← layer name, one-line tagline, nav links to sibling pages
  <section.quickref>  ← CFG param table + symbol legend table
  <section.forward>   ← per-step cards (main content)
  <section.backward>  ← backward pass summary + gradient-highway SVG
  <footer>
  <script>  ← IntersectionObserver wiring + SVG arrow animation trigger
```

**Scroll animation mechanism:**
- Each `.step` card starts with `opacity:0; transform:translateY(30px)`
- `IntersectionObserver` (threshold 0.15) adds class `.visible` when card enters viewport
- `.visible` triggers CSS transition: `opacity→1; transform→none` (300 ms ease-out)
- SVG arrows use `stroke-dasharray / stroke-dashoffset` reset to 0 on `.visible`
- Tensor boxes use `animation: fadeScale 0.4s ease` on `.visible`

**Color palette (consistent across all 6 files):**
| Role | Color |
|---|---|
| Input tensor | `#FF6B35` (orange) |
| Output tensor | `#00CED1` (teal) |
| Intermediate tensor | `#4A90D9` (blue) |
| Normalization op | `#F5A623` (amber) |
| Attention op | `#9B59B6` (purple) |
| FFN / projection op | `#27AE60` (green) |
| Residual path | `#95A5A6` (gray) |
| Deformation / offset | `#E74C3C` (red) |

---

### transformer_layer.html — 30 steps

Source: `src-lib/transformer_layer.hpp` (checkpoints [01]–[30] + symbol legend)

Steps grouped into visual phases (each phase rendered as a colored section break):

| Phase | Steps | Key SVG elements |
|---|---|---|
| Setup | 01–05 | [B,C,H,W]→pad→[B,C,Hp,Wp] box-to-box with pad cells shown |
| Spatial transform | 06–08 | cyclic shift arrow grid; Wn window grid; token tape |
| Attention | 09–17 | LN box; 3 Q/K/V projections; head-split tree; T×T score heatmap; softmax curve; V-mix |
| Output+FFN | 18–25 | head-concat; Wo; residual Y-merge; LN; FC1→act→FC2; residual Y-merge |
| Spatial restore | 26–30 | token→window reshape; tile grid; reverse-shift arrow; crop; output box |

---

### vit_layer.html — 24 steps

Source: `src-lib/vit_layer.hpp` (checkpoints [01]–[24])

| Phase | Steps | Key SVG elements |
|---|---|---|
| Tokenize | 01–04 | 2-D grid→1-D tape flatten; pos-embed addition |
| LN + QKV | 05–09 | LN; 3 linear projections; head split |
| Global attention | 10–13 | dense T×T score matrix (full grid shown for T=6); softmax; V-mix |
| Head merge + residual | 14–16 | concat; Wo; Y-merge |
| FFN + residual | 17–22 | LN; FC1; act; FC2; Y-merge |
| Reshape output | 23–24 | token tape → 2-D grid |

Critical callout box: **O(T²) memory warning** with a table showing T² at 13×13, 26×26, 52×52.

---

### dcnv4_layer.html — 9 steps

Source: `src-lib/dcnv4_layer.hpp` + `docs/dcnv4_layer.md`

| Step | SVG elements |
|---|---|
| 01 input | [B,C,H,W] box |
| 02 offset subnet | small conv subnet → [B, G·K²·3, H, W] offset tensor |
| 03 unpack offsets | offset tensor split into Δx, Δy, weight per sample point |
| 04 deformed sampling grid | regular grid warped into deformed positions (animated dots moving) |
| 05 bilinear interpolation | 2×2 neighbor grid + bilinear weight quad |
| 06 weighted aggregation | K² sampled features → weighted sum |
| 07 BN | normalize box |
| 08 activation | act box |
| 09 output | [B,C,H,W] box |

Callout: **C=N forced** constraint box. d_stride sparse field diagram.

---

### deconvolutional_layer.html — 7 steps

Source: `src-lib/deconvolutional_layer.hpp` + `docs/deconvolutional_layer.md`

| Step | SVG elements |
|---|---|
| 01 input | [B,C,H,W] |
| 02 output size formula | formula box: out_h = stride·(H−1)+size |
| 03 spread (transpose) | sparse input→expanded grid with zeros inserted between samples |
| 04 weight kernel (transposed) | flipped kernel applied via col2im |
| 05 GEMM | matrix multiply box |
| 06 bias + activation | boxes |
| 07 output | [B,N,out_H,out_W] (larger than input) |

Callout: **BN not yet supported** warning box.

---

### clifford_layer.html — 14 steps

Source: `src-lib/clifford_layer.hpp` + `src-lib/clifford_layer.cpp` (lines 1–200)

| Phase | Steps | Key SVG elements |
|---|---|---|
| Local DWConv | 01–02 | input; depthwise conv(s) if dwconv enabled |
| Wedge path | 03–04 | cyclic channel shift by wedge schedule; antisymmetric product xi ∧ x_{i+s} |
| Inner path | 05–06 | cyclic shift by inner schedule; symmetric dot product |
| Combine | 07 | concat wedge+inner → [B,C_raw,H,W] |
| Global FFN | 08 | optional 1×1 across all channels |
| Higher order | 09 | optional higher-order local interaction |
| Layer scale | 10 | per-channel learned scale multiply |
| Drop path | 11 | stochastic depth mask (train only) |
| Residual | 12 | Y-merge with shortcut |
| Activation | 13 | act box |
| Output | 14 | [B,N,H,W] |

Geometric insight callout: wedge ↔ antisymmetric (contrast/edges), inner ↔ symmetric (alignment/coherence).

---

### graph_conv_layer.html — 14 steps

Source: `src-lib/graph_conv_layer.hpp`

| Phase | Steps | Key SVG elements |
|---|---|---|
| Setup | 01–02 | input; output lattice grid |
| Gather | 03–04 | center pixel graph star; K² neighbors radiating outward |
| Edge weights | 05–07 | attention logit formula; softmax bar chart; uniform fallback |
| Aggregation | 08 | weighted sum of K² feature vectors → aggr_i |
| Projection | 09–11 | W_neigh GEMM; optional W_self GEMM; additive merge |
| Normalize + activate | 12–13 | BN; act |
| Output | 14 | [B,N,out_H,out_W] |

Callout: **credit-assignment overlap** diagram — one pixel receiving gradients from 9 different neighborhoods.

---

### index.html — Overview Page

A card grid: one card per layer with:
- Layer name + badge color
- 1-sentence description
- Key hyperparameters list
- Link to the layer's HTML page

---

## Implementation Plan

### Step 1 — Build shared HTML template (1 file as prototype)
Write `transformer_layer.html` first as the canonical template:
- Inline CSS with all color variables, `.step` card layout, animation keyframes
- IntersectionObserver JS snippet
- SVG arrow animation pattern (stroke-dashoffset)
- All 30 steps with SVG diagrams

### Step 2 — vit_layer.html (24 steps)
Reuse same CSS/JS pattern. Add O(T²) callout box.

### Step 3 — dcnv4_layer.html (9 steps)
Include deformed-grid SVG animation (dots moving from regular to irregular positions).

### Step 4 — deconvolutional_layer.html (7 steps)
Upsampling animation: input grid spacing expands to show stride insertion.

### Step 5 — clifford_layer.html (14 steps)
Include channel-shift ring diagram showing cyclic offset.

### Step 6 — graph_conv_layer.html (14 steps)
Include graph star SVG with edge-weight bar chart.

### Step 7 — index.html
Card grid linking all 6 pages.

---

## SVG Per-Step Diagram Style — 3D Isometric Tensor Boxes

Tensors are drawn as **3D perspective rectangular prisms** in the style of classic CNN architecture diagrams (e.g. VGGNet, ResNet paper figures): front face shows H×W, right-side face shows channel depth C, top face visible to complete the 3D effect.

### Isometric box geometry (SVG polygon)

For a box of visual width `w`, height `h`, depth `d` (all in SVG units), at origin `(x0, y0)`:
```
  top face:    (x0,y0)  (x0+d,y0-d/2)  (x0+w+d,y0-d/2)  (x0+w,y0)
  front face:  (x0,y0)  (x0+w,y0)      (x0+w,y0+h)       (x0,y0+h)
  right face:  (x0+w,y0)(x0+w+d,y0-d/2)(x0+w+d,y0+h-d/2)(x0+w,y0+h)
```

### CSS for 3D face colors and animation:

```css
.face-front.input   { fill: #FF8C55; stroke: #c0622a; stroke-width:1.5; }
.face-top.input     { fill: #FFAB7F; stroke: #c0622a; stroke-width:1.5; }
.face-right.input   { fill: #D4623A; stroke: #c0622a; stroke-width:1.5; }

.face-front.intermediate { fill: #5BA4E8; stroke: #2c6fab; stroke-width:1.5; }
.face-top.intermediate   { fill: #85BEF0; stroke: #2c6fab; stroke-width:1.5; }
.face-right.intermediate { fill: #3A7DC4; stroke: #2c6fab; stroke-width:1.5; }

.face-front.output  { fill: #3ECFCF; stroke: #1a9e9e; stroke-width:1.5; }
.face-top.output    { fill: #72E0E0; stroke: #1a9e9e; stroke-width:1.5; }
.face-right.output  { fill: #1EA5A5; stroke: #1a9e9e; stroke-width:1.5; }

.op.normalization { fill: #F5A623; stroke: #c07800; }
.op.attention     { fill: #9B59B6; stroke: #6c3483; }
.op.ffn           { fill: #27AE60; stroke: #1a7a42; }
.op.residual      { fill: #95A5A6; stroke: #707b7c; }
.op.deformation   { fill: #E74C3C; stroke: #a93226; }

.arrow {
  stroke: #555; stroke-width: 2;
  stroke-dasharray: 120; stroke-dashoffset: 120;
  transition: stroke-dashoffset 0.5s ease 0.3s;
}
.step.visible .arrow { stroke-dashoffset: 0; }
```

### Box sizing convention:
| Dimension shown | SVG visual width `w` | SVG depth `d` |
|---|---|---|
| Channel count | proportional, min 20 px | represents C |
| H×W (spatial) | fixed 100 px | — |
| Depth/steps | 24 px default | shrinks if C small |

For attention score matrices [T×T]: drawn as a **flat 2D grid** (not 3D box) with colored cells (heat-map style).

---

## Verification

1. Open each HTML file in a browser (Firefox/Chrome) — no server needed
2. Scroll through: every `.step` card should animate in
3. Resize window to <768px — layout should remain readable
4. Check nav links between pages work (relative href)
5. Validate: `grep -c 'class="step"'` each file to confirm step count matches .hpp checkpoints
6. For transformer: 30 steps; vit: 24; dcnv4: 9; deconv: 7; clifford: 14; graph_conv: 14
