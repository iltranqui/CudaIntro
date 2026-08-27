#pragma once

#include "darknet_internal.hpp"

/**
 * @brief Swin-style Windowed Transformer Layer
 *
 * ╔══════════════════════════════════════════════════════════════════════════════════════╗
 * ║              WINDOWED TRANSFORMER — ATTEND LOCALLY, COMMUNICATE GLOBALLY            ║
 * ╚══════════════════════════════════════════════════════════════════════════════════════╝
 *
 * Implements a transformer block that can serve as a drop-in substitute for [conv]:
 * - Windowed multi-head self-attention within size×size non-overlapping windows
 * - Shifted windows (cyclic shift by size/2) for cross-window communication
 * - Full transformer block: LayerNorm → MHSA → mHC residual mix → LayerNorm → FFN → mHC residual mix
 * - mHC residual mixers are initialized near identity so inserted transformer blocks start conservatively
 *
 * Config syntax:
 *   [transformer]
 *   filters=128       # output channels (N). If C != N, a residual projection is added.
 *   size=7            # window size (M). Windows are MxM non-overlapping patches.
 *   heads=4           # number of attention heads (h). d = N/h channels per head.
 *   shift=0           # 0=regular windows, 1=shift by M/2 for cross-window communication.
 *   ffn_ratio=4       # FFN expansion ratio (r). Hidden dim = r*N.
 *   activation=gelu   # FFN internal activation (gelu, leaky, mish, etc.)
 *
 * Network-level parameters affecting transformer training:
 *   [net]
 *   batch=16          # batch size (paper uses 1024; small batches increase gradient variance)
 *   momentum=0.92     # SGD momentum (paper uses AdamW, not SGD)
 *   decay=0.0005      # weight decay, coupled with SGD (paper uses decoupled 0.05 with AdamW)
 *   learning_rate=0.0000261  # base LR
 *   burn_in=1000      # LR warmup steps (paper uses 20-epoch cosine warmup)
 *   loss_scale=1.0    # gradient loss scaling factor
 *
 * Example: LegoGears_transformer.cfg (5-class object detection, 224x160)
 *   - Layer 12: [transformer] filters=64  size=7 heads=4 shift=0 (28x20x64, C==N)
 *   - Layer 13: [transformer] filters=64  size=7 heads=4 shift=1 (28x20x64, C==N, shifted)
 *   - Layer 18: [transformer] filters=256 size=7 heads=4 shift=0 (14x10x256, C==N)
 *   - Layer 35: [transformer] filters=256 size=7 heads=4 shift=0 (14x10x384→256, C!=N, res_proj)
 *
 *  FORWARD PASS — DATA FLOW
 *
 *   Input feature map
 *   [B,C,H,W]
 *       │
 *       ├──▶ pad to multiples of window size
 *       │       (only if H or W is not divisible by size)
 *       │
 *       ├──▶ optional cyclic shift by size/2
 *       │       lets later windows see across old boundaries
 *       │
 *       ├──▶ partition into non-overlapping windows
 *       │
 *       │      ╭──── window 0 ────╮ ╭──── window 1 ────╮
 *       │      │ x x x x x x x    │ │ x x x x x x x    │
 *       │      │ x x x x x x x    │ │ x x x x x x x    │
 *       │      │ x x x x x x x    │ │ x x x x x x x    │
 *       │      ╰──────────────────╯ ╰──────────────────╯
 *       │
 *       ├──▶ flatten each window into tokens [B*num_windows, T=size^2, C]
 *       │
 *       ├──▶ LN1
 *       │
 *       ├──▶ Q,K,V projections
 *       │       q = xWq, k = xWk, v = xWv
 *       │
 *       ├──▶ multi-head self-attention inside each window only
 *       │
 *       │      tokens in one window
 *       │      t0  t1  t2  ...  tT-1
 *       │      │ \ │ \ │
 *       │      │  \│  \│    local all-to-all attention
 *       │      │  /│  /│    no direct links to other windows here
 *       │      │ / │ / │
 *       │      ▼   ▼   ▼
 *       │
 *       ├──▶ output projection
 *       │
 *       ├──▶ mHC residual mix
 *       │       x ≈ shortcut at initialization, with a small trainable attention branch
 *       │
 *       ├──▶ LN2
 *       │
 *       ├──▶ FFN / MLP
 *       │       C ─▶ ffn_ratio*C ─▶ activation ─▶ C
 *       │
 *       ├──▶ mHC residual mix
 *       │       x ≈ previous token state at initialization, with a small trainable FFN branch
 *       │
 *       ├──▶ unpartition windows back to spatial grid
 *       │
 *       ├──▶ reverse cyclic shift
 *       │
 *       ├──▶ crop away temporary padding
 *       │
 *       └──▶ Output [B,N,H,W]
 *
 *  BACKWARD PASS — GRADIENT FLOW
 *
 *   dL/dOutput
 *       │
 *       ├──▶ crop grad restore padded frame
 *       ├──▶ reverse-unshift grad
 *       ├──▶ repartition grad into windows
 *       │
 *       ├──▶ residual split at FFN output
 *       │      dX_total
 *       │       ├──▶ skip branch --------------------------┐
 *       │       └──▶ LN2 -> FFN backward -> dX_ffn        │
 *       │                                                 ▼
 *       │      accumulate: dX_after_attn = dX_skip + dX_ffn
 *       │
 *       ├──▶ residual split at attention output
 *       │      dX_after_attn
 *       │       ├──▶ skip branch --------------------------┐
 *       │       └──▶ LN1 -> attn backward                 │
 *       │              -> dV, dK, dQ                      │
 *       │              -> softmax backward                │
 *       │              -> projection weight grads         │
 *       │              -> token grads per window          │
 *       │                                                 ▼
 *       │      accumulate: dX_window = dX_skip + dX_attn
 *       │
 *       ├──▶ merge window grads back to image layout
 *       ├──▶ undo initial shift/pad transforms
 *       └──▶ dL/dInput [B,C,H,W]
 *
 *  KEY IDEA
 *
 *   Pass 1: attention is dense inside a small window.
 *   Pass 2: shifted windows move the borders, so information crosses windows.
 *   Repeating blocks builds long-range context without paying full global-attention cost.
 *
 *  ─────────────────────────────────────────────────────────────────────────────────────
 *  SYMBOL LEGEND
 *  ─────────────────────────────────────────────────────────────────────────────────────
 *
 *   B  = batch size
 *   C  = input channel count
 *   N  = output / embedding channel count
 *   H  = input height
 *   W  = input width
 *   Hp = padded height
 *   Wp = padded width
 *   M  = window size (`size` in cfg)
 *   S  = cyclic shift amount = M/2 when `shift=1`
 *   T  = tokens per window = M*M
 *   Wn = number of windows per image = (Hp/M) * (Wp/M)
 *   h  = number of attention heads
 *   d  = channels per head = N / h
 *   r  = FFN expansion ratio
 *
 *   Think of the layer as operating on two coordinate systems:
 *
 *      spatial grid                          token grid per window
 *
 *      [y,x,c]                               [window_id, token_id, channel]
 *
 *      image-like layout                     sequence-like layout
 *      easier for padding/shift              easier for LN/QKV/attention
 *
 *   The forward pass alternates between those worlds:
 *
 *      spatial ──partition──▶ token ──attention/MLP──▶ token ──merge──▶ spatial
 *
 *  ─────────────────────────────────────────────────────────────────────────────────────
 *  GEOMETRY BEFORE ATTENTION
 *  ─────────────────────────────────────────────────────────────────────────────────────
 *
 *   Suppose H=10, W=13, M=4.
 *
 *   The transformer cannot form clean 4x4 windows until the map is padded:
 *
 *      original: 10 x 13
 *      padded:   12 x 16
 *
 *   Why?
 *
 *      10 mod 4 = 2   → need 2 extra rows
 *      13 mod 4 = 1   → need 3 extra cols
 *
 *   Visual:
 *
 *      before padding                    after padding
 *
 *      ┌───────────────────────┐         ┌────────────────────────────┐
 *      │ real real real real   │         │ real real real real pad    │
 *      │ real real real real   │         │ real real real real pad    │
 *      │ real real real real   │         │ real real real real pad    │
 *      │ real real real real   │         │ real real real real pad    │
 *      │ real real real real   │         │ real real real real pad    │
 *      │ real real real real   │   ==>   │ real real real real pad    │
 *      │ real real real real   │         │ real real real real pad    │
 *      │ real real real real   │         │ real real real real pad    │
 *      │ real real real real   │         │ real real real real pad    │
 *      │ real real real real   │         │ real real real real pad    │
 *      └───────────────────────┘         │ pad  pad  pad  pad  pad    │
 *                                        │ pad  pad  pad  pad  pad    │
 *                                        └────────────────────────────┘
 *
 *   Padding is not the interesting part of the model.
 *   It is a mechanical prerequisite so every window has exactly T=M*M tokens.
 *
 *  ─────────────────────────────────────────────────────────────────────────────────────
 *  WINDOW PARTITION AS A VIEW CHANGE
 *  ─────────────────────────────────────────────────────────────────────────────────────
 *
 *   After padding, the spatial map is chopped into non-overlapping MxM windows.
 *
 *      padded feature map
 *
 *      ╔════╤════╦════╤════╦════╤════╦════╤════╗
 *      ║ w0 │ w0 ║ w1 │ w1 ║ w2 │ w2 ║ w3 │ w3 ║
 *      ╟────┼────╫────┼────╫────┼────╫────┼────╢
 *      ║ w0 │ w0 ║ w1 │ w1 ║ w2 │ w2 ║ w3 │ w3 ║
 *      ╠════╪════╬════╪════╬════╪════╬════╪════╣
 *      ║ w4 │ w4 ║ w5 │ w5 ║ w6 │ w6 ║ w7 │ w7 ║
 *      ╟────┼────╫────┼────╫────┼────╫────┼────╢
 *      ║ w4 │ w4 ║ w5 │ w5 ║ w6 │ w6 ║ w7 │ w7 ║
 *      ╠════╪════╬════╪════╬════╪════╬════╪════╣
 *      ║ w8 │ w8 ║ w9 │ w9 ║ wa │ wa ║ wb │ wb ║
 *      ╟────┼────╫────┼────╫────┼────╫────┼────╢
 *      ║ w8 │ w8 ║ w9 │ w9 ║ wa │ wa ║ wb │ wb ║
 *      ╚════╧════╩════╧════╩════╧════╩════╧════╝
 *
 *   Each window becomes a short token sequence:
 *
 *      window w5
 *
 *      ┌────┬────┐
 *      │ t0 │ t1 │
 *      ├────┼────┤   flatten row-major  ==>  [t0, t1, t2, t3]
 *      │ t2 │ t3 │
 *      └────┴────┘
 *
 *   The whole tensor changes shape conceptually from:
 *
 *      [B, N, Hp, Wp]
 *
 *   to:
 *
 *      [B, Wn, T, N]
 *
 *   and often internally to:
 *
 *      [B*Wn, T, N]
 *
 *   because attention is the same kernel repeated independently per window.
 *
 *  ─────────────────────────────────────────────────────────────────────────────────────
 *  WHY SHIFTED WINDOWS EXIST
 *  ─────────────────────────────────────────────────────────────────────────────────────
 *
 *   Non-shifted windows are cheap, but they isolate information at fixed borders.
 *
 *   Example with M=4:
 *
 *      block A | block B
 *      --------+--------
 *      pixels in A cannot directly attend to pixels in B in this layer
 *
 *   Shifted windows solve that without global attention:
 *
 *      Layer k        : partition on original grid
 *      Layer k + 1    : shift by S=M/2, then partition again
 *
 *   Picture:
 *
 *      original partition
 *
 *      ╔════╤════╦════╤════╗
 *      ║ A  │ A  ║ B  │ B  ║
 *      ╟────┼────╫────┼────╢
 *      ║ A  │ A  ║ B  │ B  ║
 *      ╠════╪════╬════╪════╣
 *      ║ C  │ C  ║ D  │ D  ║
 *      ╟────┼────╫────┼────╢
 *      ║ C  │ C  ║ D  │ D  ║
 *      ╚════╧════╩════╧════╝
 *
 *      after cyclic shift by one cell in both directions
 *
 *      ╔════╤════╦════╤════╗
 *      ║ D  │ C  ║ C  │ D  ║
 *      ╟────┼────╫────┼────╢
 *      ║ B  │ A  ║ A  │ B  ║
 *      ╠════╪════╬════╪════╣
 *      ║ B  │ A  ║ A  │ B  ║
 *      ╟────┼────╫────┼────╢
 *      ║ D  │ C  ║ C  │ D  ║
 *      ╚════╧════╩════╧════╝
 *
 *   Now one shifted window contains pieces from A, B, C, and D.
 *   That is the whole trick:
 *
 *      local attention + alternating boundaries = gradual global mixing
 *
 *  ─────────────────────────────────────────────────────────────────────────────────────
 *  INSIDE ONE WINDOW: TOKEN STORYBOARD
 *  ─────────────────────────────────────────────────────────────────────────────────────
 *
 *   Consider one window with T=4 tokens and N=8 channels.
 *
 *      input tokens
 *
 *      t0 = [c0 c1 c2 c3 c4 c5 c6 c7]
 *      t1 = [c0 c1 c2 c3 c4 c5 c6 c7]
 *      t2 = [c0 c1 c2 c3 c4 c5 c6 c7]
 *      t3 = [c0 c1 c2 c3 c4 c5 c6 c7]
 *
 *   Step A: LayerNorm
 *
 *      each token normalized across channel dimension
 *
 *      for each token ti:
 *         mean_i = average(ti[:])
 *         var_i  = average((ti[:] - mean_i)^2)
 *         ln_i   = gamma * (ti - mean_i) / sqrt(var_i + eps) + beta
 *
 *   Important:
 *
 *      normalization does NOT mix tokens with other tokens
 *      normalization only rescales channels within a single token
 *
 *   Step B: Linear projections
 *
 *      Q = LN(X) * Wq
 *      K = LN(X) * Wk
 *      V = LN(X) * Wv
 *
 *      shapes:
 *
 *      X    : [T, N]
 *      Wq   : [N, N]
 *      Wk   : [N, N]
 *      Wv   : [N, N]
 *      Q/K/V: [T, N]
 *
 *   Step C: split heads
 *
 *      if h=2 and N=8, then d=4
 *
 *      Q -> [head0 | head1]
 *      K -> [head0 | head1]
 *      V -> [head0 | head1]
 *
 *      head0 sees channels [0..3]
 *      head1 sees channels [4..7]
 *
 *   Step D: compute attention scores per head
 *
 *      score(i,j) = dot(q_i, k_j) / sqrt(d)
 *
 *      for T=4 this yields a 4x4 matrix per head
 *
 *      rows    = query token asking "who matters to me?"
 *      columns = key token answering "how relevant am I?"
 *
 *  ─────────────────────────────────────────────────────────────────────────────────────
 *  ATTENTION MATRIX AS A HEAT MAP
 *  ─────────────────────────────────────────────────────────────────────────────────────
 *
 *   Before softmax:
 *
 *      head 0 scores
 *
 *           k0    k1    k2    k3
 *        ┌────────────────────────┐
 *      q0│ 0.2   1.1  -0.3   0.5  │
 *      q1│ 0.0   0.8   0.9  -0.2  │
 *      q2│-0.4   0.1   1.5   1.0  │
 *      q3│ 0.7   0.2   0.4   1.8  │
 *        └────────────────────────┘
 *
 *   After softmax row-wise:
 *
 *           k0    k1    k2    k3
 *        ┌────────────────────────┐
 *      q0│ 0.16  0.39  0.10  0.25 │
 *      q1│ 0.14  0.31  0.34  0.21 │
 *      q2│ 0.07  0.11  0.44  0.38 │
 *      q3│ 0.17  0.10  0.12  0.61 │
 *        └────────────────────────┘
 *
 *   Reading one row:
 *
 *      q2 attends 44% to k2 and 38% to k3
 *
 *   Interpretation:
 *
 *      token 2 found tokens 2 and 3 most informative for its update
 *
 *   Each row sums to 1.
 *   Each row is a routing distribution for one query token.
 *
 *  ─────────────────────────────────────────────────────────────────────────────────────
 *  VALUE MIXING
 *  ─────────────────────────────────────────────────────────────────────────────────────
 *
 *   The attention weights do not produce the final output alone.
 *   They are used to mix value vectors.
 *
 *      out_i = sum_j attn(i,j) * v_j
 *
 *   For one query row:
 *
 *      out_2 =
 *          0.07 * v0
 *        + 0.11 * v1
 *        + 0.44 * v2
 *        + 0.38 * v3
 *
 *   So the attention matrix answers:
 *
 *      "how much of each value vector should this token borrow?"
 *
 *   Multi-head attention means this borrowing happens several times in parallel,
 *   each head using a different learned subspace.
 *
 *  ─────────────────────────────────────────────────────────────────────────────────────
 *  MERGING HEADS BACK TOGETHER
 *  ─────────────────────────────────────────────────────────────────────────────────────
 *
 *   After attention, head outputs are concatenated:
 *
 *      head0_out: [T, d]
 *      head1_out: [T, d]
 *      ...
 *      headh_out: [T, d]
 *
 *      concat(head0_out, head1_out, ..., headh_out) -> [T, N]
 *
 *   Then an output projection mixes information across heads:
 *
 *      attn_out = concat_heads * Wo
 *
 *   Spatially:
 *
 *      per-window result
 *
 *      [token0']
 *      [token1']
 *      [token2']
 *      [token3']
 *
 *   This is still in token layout.
 *   The next residual block and FFN operate in that same token layout.
 *
 *  ─────────────────────────────────────────────────────────────────────────────────────
 *  RESIDUAL BLOCK 1
 *  ─────────────────────────────────────────────────────────────────────────────────────
 *
 *   Transformer blocks preserve a clean highway for gradients and features:
 *
 *      x0 = window_tokens_before_LN1
 *      x1 = MHSA(LN1(x0))
 *      x2 = x0 + x1
 *
 *   Why add the shortcut?
 *
 *      1. the model can keep old information if attention is unhelpful
 *      2. gradients can flow around complex sub-blocks
 *      3. training deep stacks becomes much more stable
 *
 *   ASCII view:
 *
 *      x0 ───────────────────────────────┐
 *       │                                │
 *       └──▶ LN1 ─▶ MHSA ─▶ proj ───────▶ + ───▶ x2
 *
 *  ─────────────────────────────────────────────────────────────────────────────────────
 *  FEED-FORWARD NETWORK
 *  ─────────────────────────────────────────────────────────────────────────────────────
 *
 *   After attention, each token is processed independently by an MLP:
 *
 *      token by token:
 *
 *      [N] ─▶ [rN] ─▶ activation ─▶ [N]
 *
 *   Example when N=64 and r=4:
 *
 *      64 channels
 *         │
 *         ├── linear up-projection to 256
 *         ├── GELU / other activation
 *         └── linear down-projection to 64
 *
 *   Key contrast:
 *
 *      attention mixes tokens with tokens
 *      FFN mixes channels with channels
 *
 *   Together they provide:
 *
 *      attention = communication across positions
 *      FFN       = nonlinear feature transformation at each position
 *
 *  ─────────────────────────────────────────────────────────────────────────────────────
 *  RESIDUAL BLOCK 2
 *  ─────────────────────────────────────────────────────────────────────────────────────
 *
 *      x2 ───────────────────────────────┐
 *       │                                │
 *       └──▶ LN2 ─▶ FC1 ─▶ Act ─▶ FC2 ─▶ + ───▶ x3
 *
 *   Final token output per window:
 *
 *      x3 = x2 + FFN(LN2(x2))
 *
 *   Only after this second residual block do we merge windows back to spatial form.
 *
 *  ─────────────────────────────────────────────────────────────────────────────────────
 *  MERGING TOKENS BACK TO THE IMAGE
 *  ─────────────────────────────────────────────────────────────────────────────────────
 *
 *   Each window sequence is reshaped back into an MxM patch.
 *
 *      [t0 t1 t2 t3]  ->  ┌────┬────┐
 *                          │ t0 │ t1 │
 *                          ├────┼────┤
 *                          │ t2 │ t3 │
 *                          └────┴────┘
 *
 *   Then all windows are tiled back into the padded spatial map.
 *
 *      windows
 *      w0 w1 w2
 *      w3 w4 w5
 *      w6 w7 w8
 *
 *   become:
 *
 *      ╔════╤════╦════╤════╦════╤════╗
 *      ║ w0 │ w0 ║ w1 │ w1 ║ w2 │ w2 ║
 *      ╟────┼────╫────┼────╫────┼────╢
 *      ║ w0 │ w0 ║ w1 │ w1 ║ w2 │ w2 ║
 *      ╠════╪════╬════╪════╬════╪════╣
 *      ║ w3 │ w3 ║ w4 │ w4 ║ w5 │ w5 ║
 *      ╟────┼────╫────┼────╫────┼────╢
 *      ║ w3 │ w3 ║ w4 │ w4 ║ w5 │ w5 ║
 *      ╠════╪════╬════╪════╬════╪════╣
 *      ║ w6 │ w6 ║ w7 │ w7 ║ w8 │ w8 ║
 *      ╟────┼────╫────┼────╫────┼────╢
 *      ║ w6 │ w6 ║ w7 │ w7 ║ w8 │ w8 ║
 *      ╚════╧════╩════╧════╩════╧════╝
 *
 *   Finally:
 *
 *      if shift was applied at the start, undo it now
 *      if padding was added at the start, crop it now
 *
 *  ─────────────────────────────────────────────────────────────────────────────────────
 *  FORWARD PASS AS NUMBERED CHECKPOINTS
 *  ─────────────────────────────────────────────────────────────────────────────────────
 *
 *   [01] receive input tensor [B,C,H,W]
 *   [02] project / align channels to N if the implementation requires it
 *   [03] compute Hp, Wp for padding
 *   [04] allocate / reuse temporary padded buffer
 *   [05] copy input into padded buffer
 *   [06] apply cyclic shift if enabled
 *   [07] compute number of windows Wn
 *   [08] flatten each MxM patch into T tokens
 *   [09] apply LN1 to each token
 *   [10] project Q
 *   [11] project K
 *   [12] project V
 *   [13] split into heads
 *   [14] compute scaled dot-product scores per head
 *   [15] apply attention masking if the implementation uses it
 *   [16] softmax rows of score matrices
 *   [17] mix values using those weights
 *   [18] concatenate heads
 *   [19] apply output projection
 *   [20] add first residual
 *   [21] apply LN2
 *   [22] apply FFN first linear
 *   [23] apply FFN activation
 *   [24] apply FFN second linear
 *   [25] add second residual
 *   [26] reshape tokens back to windows
 *   [27] tile windows back to padded spatial layout
 *   [28] reverse cyclic shift
 *   [29] crop to original H,W
 *   [30] write output tensor [B,N,H,W]
 *
 *  ─────────────────────────────────────────────────────────────────────────────────────
 *  MICRO EXAMPLE: TWO HEADS, ONE WINDOW
 *  ─────────────────────────────────────────────────────────────────────────────────────
 *
 *   Let:
 *
 *      M = 2
 *      T = 4 tokens
 *      N = 4 channels
 *      h = 2 heads
 *      d = 2 channels per head
 *
 *   Token sequence entering attention:
 *
 *      t0 = [ 1,  0,  2,  1]
 *      t1 = [ 0,  1,  1,  2]
 *      t2 = [ 2,  1,  0,  1]
 *      t3 = [ 1,  2,  1,  0]
 *
 *   Head 0 sees first two channels:
 *
 *      t0h0 = [1,0]
 *      t1h0 = [0,1]
 *      t2h0 = [2,1]
 *      t3h0 = [1,2]
 *
 *   Head 1 sees last two channels:
 *
 *      t0h1 = [2,1]
 *      t1h1 = [1,2]
 *      t2h1 = [0,1]
 *      t3h1 = [1,0]
 *
 *   Even before learning exact weights, you can imagine:
 *
 *      head0 specializing in one pattern family
 *      head1 specializing in another
 *
 *   This is why multiple heads are useful:
 *
 *      they let the model route information by several notions of similarity at once
 *
 *  ─────────────────────────────────────────────────────────────────────────────────────
 *  BACKWARD PASS DECOMPOSED
 *  ─────────────────────────────────────────────────────────────────────────────────────
 *
 *   Backprop through this layer is easiest to understand if we reverse the checkpoints.
 *
 *   Start:
 *
 *      g_out = dL/d(output)
 *
 *   Reverse [30]..[26]:
 *
 *      g_spatial = gradient on [B,N,H,W]
 *      pad back to [B,N,Hp,Wp] if forward cropped
 *      shift back into shifted coordinate system if forward unshifted last
 *      repartition into window-token layout [B*Wn, T, N]
 *
 *   Reverse [25]:
 *
 *      x3 = x2 + ffn_out
 *
 *      so:
 *
 *      g_x2_from_skip = g_x3
 *      g_ffn_out      = g_x3
 *
 *      the residual duplicates gradient into two branches
 *
 *   Reverse [24] [23] [22] [21]:
 *
 *      g_ffn_out
 *        └── FC2 backward -> g_act, grad_W2, grad_b2
 *             └── activation backward -> g_fc1
 *                  └── FC1 backward -> g_ln2, grad_W1, grad_b1
 *                       └── LN2 backward -> g_x2_from_ffn, grad_gamma2, grad_beta2
 *
 *   Accumulate at x2:
 *
 *      g_x2 = g_x2_from_skip + g_x2_from_ffn
 *
 *   Reverse [20]:
 *
 *      x2 = x0 + attn_out
 *
 *      g_x0_from_skip = g_x2
 *      g_attn_out     = g_x2
 *
 *   Reverse [19]:
 *
 *      output projection backward gives:
 *
 *      g_concat_heads
 *      grad_Wo
 *      grad_bo
 *
 *   Reverse [18]:
 *
 *      split g_concat_heads back into per-head gradients
 *
 *   Reverse [17]:
 *
 *      ctx = attn * V
 *
 *      produces:
 *
 *      g_attn = g_ctx * V^T
 *      g_V    = attn^T * g_ctx
 *
 *   Reverse [16]:
 *
 *      attn = softmax(scores)
 *
 *      softmax backward is row-wise
 *
 *      for each query row:
 *
 *         g_scores = J_softmax^T * g_attn_row
 *
 *      where J_softmax is the implicit Jacobian of the row distribution
 *
 *   Reverse [14]:
 *
 *      scores = QK^T / sqrt(d)
 *
 *      yields:
 *
 *      g_Q = g_scores * K / sqrt(d)
 *      g_K = g_scores^T * Q / sqrt(d)
 *
 *   Reverse [13] [12] [11] [10]:
 *
 *      combine g_Q, g_K, g_V across heads
 *      pass them through the Q/K/V linear layers
 *
 *      obtain:
 *
 *      g_ln1_from_Q
 *      g_ln1_from_K
 *      g_ln1_from_V
 *      grad_Wq, grad_bq
 *      grad_Wk, grad_bk
 *      grad_Wv, grad_bv
 *
 *   Accumulate:
 *
 *      g_ln1 = g_ln1_from_Q + g_ln1_from_K + g_ln1_from_V
 *
 *   Reverse [09]:
 *
 *      LN1 backward returns:
 *
 *      g_x0_from_attn
 *      grad_gamma1
 *      grad_beta1
 *
 *   Accumulate at x0:
 *
 *      g_x0 = g_x0_from_skip + g_x0_from_attn
 *
 *   Reverse [08] [07] [06] [05] [04] [03] [02] [01]:
 *
 *      merge token gradients back to padded image
 *      reverse initial cyclic shift
 *      drop gradients that correspond only to pad cells
 *      map back to input layout [B,C,H,W]
 *
 *  ─────────────────────────────────────────────────────────────────────────────────────
 *  RESIDUALS AS GRADIENT HIGHWAYS
 *  ─────────────────────────────────────────────────────────────────────────────────────
 *
 *   Two additive shortcuts matter enormously during training.
 *
 *      g_x3
 *       ├── directly to x2
 *       └── through FFN path to x2
 *
 *      g_x2
 *       ├── directly to x0
 *       └── through attention path to x0
 *
 *   Without those shortcuts, every gradient would be forced through LN,
 *   matrix multiplies, softmax, and activation nonlinearities.
 *
 *   With them, there is always a direct additive route.
 *
 *  ─────────────────────────────────────────────────────────────────────────────────────
 *  WHAT ACTUALLY GETS LEARNED?
 *  ─────────────────────────────────────────────────────────────────────────────────────
 *
 *   Parameter families typically include:
 *
 *      LN1 gamma, beta
 *      Wq, bq
 *      Wk, bk
 *      Wv, bv
 *      Wo, bo
 *      LN2 gamma, beta
 *      Wfc1, bfc1
 *      Wfc2, bfc2
 *
 *   During backprop, each of those collects gradients from every window in every batch.
 *
 *   Intuition:
 *
 *      Wq / Wk learn which token features should seek or offer relevance
 *      Wv      learns what information is worth transporting
 *      Wo      learns how to recombine head-specific messages
 *      FFN     learns richer per-token feature transformations
 *
 *  ─────────────────────────────────────────────────────────────────────────────────────
 *  COMPUTE COMPLEXITY
 *  ─────────────────────────────────────────────────────────────────────────────────────
 *
 *   For one window:
 *
 *      attention score cost ~ O(T*T*d) per head
 *
 *   For the whole image:
 *
 *      O(B * Wn * h * T*T*d)
 *
 *   Since T=M*M:
 *
 *      cost grows quadratically with the window area, not with the full image area
 *
 *   This is the central engineering win over full global attention at early layers.
 *
 *  ─────────────────────────────────────────────────────────────────────────────────────
 *  MEMORY MENTAL MODEL
 *  ─────────────────────────────────────────────────────────────────────────────────────
 *
 *   The largest intermediate objects are commonly:
 *
 *      window tokens          [B*Wn, T, N]
 *      Q/K/V                  [B*Wn, h, T, d]
 *      attention scores       [B*Wn, h, T, T]
 *      attention probabilities[B*Wn, h, T, T]
 *      FFN hidden             [B*Wn, T, rN]
 *
 *   If M grows, the T*T score tensors become the expensive part.
 *
 *   Small M:
 *
 *      cheaper, more local
 *
 *   Large M:
 *
 *      broader context, more expensive
 *
 *  ─────────────────────────────────────────────────────────────────────────────────────
 *  DEBUGGING GUIDE
 *  ─────────────────────────────────────────────────────────────────────────────────────
 *
 *   If shapes do not line up:
 *
 *      check N % heads == 0
 *      check window size > 0
 *      check padded dims are divisible by window size
 *      check shift is either 0 or M/2 style logic expected by the implementation
 *
 *   If outputs become NaN:
 *
 *      inspect LN eps
 *      inspect softmax overflow / score scaling
 *      inspect initialization of Q/K/V projections
 *      inspect FFN activation for extreme values
 *
 *   If loss does not decrease:
 *
 *      verify windows are partitioned and merged in inverse-consistent order
 *      verify the reverse shift exactly matches the forward shift
 *      verify residual branches are added, not overwritten
 *      verify gradients from both residual branches are accumulated
 *
 *  ─────────────────────────────────────────────────────────────────────────────────────
 *  MENTAL SUMMARY
 *  ─────────────────────────────────────────────────────────────────────────────────────
 *
 *   A convolution says:
 *
 *      "use the same local weighted stencil everywhere"
 *
 *   A windowed transformer says:
 *
 *      "inside this local patch, let every token decide dynamically which peers matter"
 *
 *   A shifted-window transformer adds:
 *
 *      "next layer, move the patch boundaries so information crosses former borders"
 *
 *   That is the whole play in one line:
 *
 *      dynamic local routing + alternating neighborhoods = scalable contextual mixing
 *
 * ### CFG File Usage:
 * Intended `.cfg` stanza for this layer:
 * ```cfg
 * [transformer]
 * filters=256         ; output embedding channels
 * size=7              ; window size (tokens per window = 7x7)
 * heads=8             ; number of attention heads
 * shift=0             ; 0 = regular windows, 1 = shifted windows
 * ffn_ratio=4         ; FFN hidden width = filters * 4
 * activation=gelu     ; FFN activation
 * ```
 *
 * Practical meaning of each field:
 *
 *   `filters`
 *      output channel count after the attention output projection and FFN.
 *      If `filters ==` previous layer channels, the first residual shortcut is active.
 *      If `filters !=` previous layer channels, the block still runs, but the first
 *      residual add becomes projection-only because the channel counts no longer match.
 *
 *   `size`
 *      window size. `size=7` means each attention region contains `49` tokens.
 *      Larger windows increase local context and cost. Smaller windows are cheaper
 *      but communicate across less area per block.
 *
 *   `heads`
 *      number of attention heads. The implementation requires the INPUT channel count
 *      from the previous layer to be divisible by `heads`.
 *
 *   `shift`
 *      `0` keeps regular non-overlapping windows.
 *      `1` enables the Swin-style cyclic shift by `size/2`, so tokens near old window
 *      borders can mix with tokens from neighboring windows.
 *
 *   `ffn_ratio`
 *      expansion ratio of the feed-forward network.
 *      Hidden width = `filters * ffn_ratio`.
 *
 *   `activation`
 *      activation used in the FFN. `gelu` is the natural default for transformer blocks.
 *
 * Recommended placement in a backbone:
 *
 *   place it after an image-like layer that outputs `[C,H,W]`
 *   prefer stages where spatial resolution is already moderately reduced
 *   keep `filters` equal to the incoming channel count for the cleanest residual behavior
 *   alternate paired blocks such as `shift=0` then `shift=1`
 *
 * Minimal example inside a network:
 * ```cfg
 * [convolutional]
 * batch_normalize=1
 * filters=256
 * size=3
 * stride=2
 * pad=1
 * activation=leaky
 *
 * [transformer]
 * filters=256
 * size=7
 * heads=8
 * shift=0
 * ffn_ratio=4
 * activation=gelu
 *
 * [transformer]
 * filters=256
 * size=7
 * heads=8
 * shift=1
 * ffn_ratio=4
 * activation=gelu
 * ```
 *
 * Read that stack as:
 *
 *   conv/downsample       -> build stronger local features
 *   transformer shift=0   -> dense attention inside fixed windows
 *   transformer shift=1   -> move the window borders and mix across them
 *
 * Constraints worth remembering:
 *
 *   previous layer must output image-shaped data
 *   previous layer channels must be divisible by `heads`
 *   `size` must be >= 1
 *   padding to a multiple of `size` is handled internally by the layer
 *
 * Parser note:
 *
 *   the cfg reader dispatches `[transformer]` sections to
 *   `parse_transformer_section()`, which forwards these fields into
 *   `make_transformer_layer()`.
 */


struct TransformerWorkspaceLayout
{
	size_t spatial0;
	size_t spatial1;
	size_t token_c0;
	size_t token_c1;
	size_t token_c2;
	size_t token_n0;
	size_t token_n1;
	size_t token_n2;
	size_t token_3c;
	size_t ffn;
	size_t head0;
	size_t head1;
	size_t head2;
	size_t head3;
	size_t head4;
	size_t scores;
	size_t total;
};

inline TransformerWorkspaceLayout make_transformer_workspace_layout(int batch, int c, int n, int h, int w,
	int window_size, int heads, int ffn_ratio)
{
	const size_t Hp = static_cast<size_t>(h);
	const size_t Wp = static_cast<size_t>(w);
	const size_t T = static_cast<size_t>(window_size) * static_cast<size_t>(window_size);
	const size_t nW = (Hp / static_cast<size_t>(window_size)) * (Wp / static_cast<size_t>(window_size));
	const size_t total_windows = static_cast<size_t>(batch) * nW;
	const size_t spatial_max = static_cast<size_t>(batch) * static_cast<size_t>(std::max(c, n)) * Hp * Wp;
	const size_t token_c = total_windows * T * static_cast<size_t>(c);
	const size_t token_n = total_windows * T * static_cast<size_t>(n);
	const size_t token_3c = total_windows * T * static_cast<size_t>(3 * c);
	const size_t ffn = total_windows * T * static_cast<size_t>(n * ffn_ratio);
	const size_t qkv_heads = total_windows * T * static_cast<size_t>(c);
	const size_t scores = total_windows * static_cast<size_t>(heads) * T * T;

	TransformerWorkspaceLayout layout{};
	size_t offset = 0;
	layout.spatial0 = offset; offset += spatial_max;
	layout.spatial1 = offset; offset += spatial_max;
	layout.token_c0 = offset; offset += token_c;
	layout.token_c1 = offset; offset += token_c;
	layout.token_c2 = offset; offset += token_c;
	layout.token_n0 = offset; offset += token_n;
	layout.token_n1 = offset; offset += token_n;
	layout.token_n2 = offset; offset += token_n;
	layout.token_3c = offset; offset += token_3c;
	layout.ffn = offset; offset += ffn;
	layout.head0 = offset; offset += qkv_heads;
	layout.head1 = offset; offset += qkv_heads;
	layout.head2 = offset; offset += qkv_heads;
	layout.head3 = offset; offset += qkv_heads;
	layout.head4 = offset; offset += qkv_heads;
	layout.scores = offset; offset += scores;
	layout.total = offset;
	return layout;
}

#ifdef DARKNET_GPU
void forward_transformer_layer_gpu(Darknet::Layer & l, Darknet::NetworkState state);
void backward_transformer_layer_gpu(Darknet::Layer & l, Darknet::NetworkState state);
void update_transformer_layer_gpu(Darknet::Layer & l, int batch, float learning_rate, float momentum, float decay, float loss_scale);

void push_transformer_layer(Darknet::Layer & l);
void pull_transformer_layer(Darknet::Layer & l);
#endif

Darknet::Layer make_transformer_layer(int batch, int h, int w, int c, int n,
	int size, int heads, int shift, int ffn_ratio, ACTIVATION activation, int index, int train);

void resize_transformer_layer(Darknet::Layer * l, int w, int h);
void forward_transformer_layer(Darknet::Layer & l, Darknet::NetworkState state);
void backward_transformer_layer(Darknet::Layer & l, Darknet::NetworkState state);
void update_transformer_layer(Darknet::Layer & l, int batch, float learning_rate, float momentum, float decay);
