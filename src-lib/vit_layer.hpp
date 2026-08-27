#pragma once

#include "darknet_internal.hpp"

/* ── Empirical notes (LegoGears, 2026-05-23) ────────────────────────────────
 * Best single-ViT: P=1 dim=128 at 28x20 (T=560) — works. Also P=1 dim=512 at
 * 14x10 (T=140) = 99.05% mAP. See docs/VIT_DEBUG.md.
 *
 * RULES:
 * - P=1 OR odd-P same-res (S=1, pad=(P-1)/2) for true identity init.
 * - Even P → box-blur init: KILLS small objects (P=2 classic = 6.89% mAP).
 * - pos_embed=learned pos_init=zero at dim>=512 on small grids.
 *   Sinusoidal: ~85% channels near-zero → softmax flattens → mAP→0.
 * - Single ViT only. 3+ same-scale stacks collapse (latent bugs compound:
 *   LN γ/β no decay, mHC momentum, post-LR-cut overshoot). Use tucker for depth.
 * - dim != input_ch → identity init silently skipped, falls to random init.
 *   Layer still works but loses drop-in property.
 * - Eval _best.weights, not _last (late collapses poison _last).
 *
 * SHAPE:
 *   classic:     stride=P pad=0,         needs H%P==0 && W%P==0, out H/P × W/P
 *   same-res:    stride=1 pad=(P-1)/2,   P must be odd,           out H × W
 *   overlap:     any,                    (H+2pad-P)%S==0,         (H+2pad-P)/S+1
 *
 * COST: attn buffer = B·heads·T²·4 bytes. T=140 → 20MB. T=560 → 320MB.
 *
 * Inserting [vit] mid-stack shifts later layers → update literal route indices.
 * ─────────────────────────────────────────────────────────────────────────── */

/**
 * @brief Vision Transformer (Global) Layer
 *
 * ╔══════════════════════════════════════════════════════════════════════════════════════╗
 * ║                 VISION TRANSFORMER — EVERY TOKEN CAN SEE EVERY TOKEN                ║
 * ╚══════════════════════════════════════════════════════════════════════════════════════╝
 *
 * Implements a patch-based ViT encoder block that can serve as a drop-in substitute for [conv]:
 * - Global multi-head self-attention over all T = (H/P) x (W/P) patch tokens
 * - Full transformer block: PosEmbed + X -> LayerNorm -> MHSA -> Residual -> LayerNorm -> FFN -> Residual
 *
 * Config syntax:
 *   [vit]
 *   dim=128           # output token width (alias: filters)
 *   patch_size=2      # patch edge length P
 *   heads=4           # attention heads
 *   mlp_dim=512       # FFN hidden width (alias: filters * ffn_ratio)
 *   pos_embed=learned # learned or sinusoidal
 *   pos_init=zero     # learned PE init: random or zero
 *   activation=gelu   # FFN internal activation
 *
 * Note on Absolute Positional Embeddings vs Network Resizing:
 *   Darknet supports dynamic network resizing (random=1 in .cfg or via resize_network). 
 *   Since T = (H/P) x (W/P) changes when the network resizes, learned vit_pos_embed
 *   needs to be bidimensionally interpolated from its base trained patch grid to the
 *   new patch grid inside resize_vit_layer().  SimpleViT-style sinusoidal embeddings
 *   are regenerated instead.
 *
 * Note on Attention Complexity:
 *   Since attention is global, the vit_attn_scores buffer size is [B, heads, T, T]. 
 *   If an image is 416x416, T = 173056, meaning T^2 is incredibly huge. Therefore, Global 
 *   ViT layers in Darknet should only be placed deep in the network (e.g., after several 
 *   downsampling convolutional or maxpool layers where H and W are small, like 13x13 or 26x26), 
 *   unlike the windowed Swin approach which handles earlier higher-resolution layers efficiently.
 *
 *  FORWARD PASS — GLOBAL ATTENTION
 *
 *   Input [B,C,H,W]
 *       │
 *       ├──▶ patchify into T = (H/P)*(W/P) tokens
 *       │
 *       │      [x00 x01 x02 ...]
 *       │      [x10 x11 x12 ...]   ──flatten──▶   [t0 t1 t2 ... tT-1]
 *       │      [x20 x21 x22 ...]
 *       │
 *       ├──▶ add learned or fixed 2D sinusoidal positional embeddings
 *       │       token_i = token_i + pos_i
 *       │
 *       ├──▶ LN1
 *       ├──▶ Q,K,V projections
 *       │
 *       ├──▶ global multi-head self-attention
 *       │
 *       │      t0 ────────────────┐
 *       │      t1 ────────────┐   │
 *       │      t2 ────────┐   │   │   every token attends to every token
 *       │      ...        │   │   │
 *       │      tT-1 ◀─────┴───┴───┘
 *       │
 *       │      scores = QK^T / sqrt(d_head)
 *       │      probs  = softmax(scores)
 *       │      ctx    = probs * V
 *       │
 *       ├──▶ output projection
 *       ├──▶ residual add
 *       ├──▶ LN2
 *       ├──▶ FFN / MLP
 *       │       N ─▶ mlp_dim ─▶ activation ─▶ N
 *       ├──▶ residual add
 *       ├──▶ reshape tokens back to [B,N,H/P,W/P]
 *       └──▶ Output [B,N,H/P,W/P]
 *
 *  BACKWARD PASS — GLOBAL CREDIT ASSIGNMENT
 *
 *   dL/dOutput
 *       │
 *       ├──▶ reshape to token gradients [B,T,N]
 *       │
 *       ├──▶ residual split at FFN
 *       │      dX_total
 *       │       ├──▶ skip branch --------------------------┐
 *       │       └──▶ LN2 -> FFN backward -> dX_ffn        │
 *       │                                                 ▼
 *       │      accumulate: dX_after_attn = dX_skip + dX_ffn
 *       │
 *       ├──▶ residual split at attention block
 *       │      dX_after_attn
 *       │       ├──▶ skip branch --------------------------┐
 *       │       └──▶ LN1 -> attention backward            │
 *       │              -> dV, dK, dQ                      │
 *       │              -> softmax(scores) backward        │
 *       │              -> grad QK^T                       │
 *       │              -> grad projection weights         │
 *       │                                                 ▼
 *       │      accumulate: dX_tokens = dX_skip + dX_attn
 *       │
 *       ├──▶ positional embedding grads
 *       │       dPos receives the same token-aligned gradient stream
 *       │
 *       └──▶ reshape token grads back to dL/dInput [B,C,H,W]
 *
 *  KEY IDEA
 *
 *   Swin says: "talk inside the room."
 *   ViT says:  "everyone joins the same call."
 *   That gives maximum context, but the T×T attention matrix grows quadratically.
 *
 *  ─────────────────────────────────────────────────────────────────────────────────────
 *  SYMBOL LEGEND
 *  ─────────────────────────────────────────────────────────────────────────────────────
 *
 *   B  = batch size
 *   C  = input channel count
 *   N  = token / embedding dimension (`filters`)
 *   H  = feature map height
 *   W  = feature map width
 *   T  = number of spatial tokens = H * W
 *   h  = number of attention heads
 *   d  = channels per head = N / h
 *   r  = FFN expansion ratio
 *
 *   Two coordinate systems matter here:
 *
 *      image coordinates      : (y, x, channel)
 *      token coordinates      : (token_id, channel)
 *
 *   ViT spends most of its time in token coordinates.
 *
 *  ─────────────────────────────────────────────────────────────────────────────────────
 *  FROM IMAGE GRID TO TOKEN TAPE
 *  ─────────────────────────────────────────────────────────────────────────────────────
 *
 *   Start with a feature map:
 *
 *      [B,C,H,W]
 *
 *   For one image and one channel slice, a 2D map looks like:
 *
 *      x00  x01  x02  x03
 *      x10  x11  x12  x13
 *      x20  x21  x22  x23
 *
 *   Flattening reorders spatial positions into a 1D sequence:
 *
 *      token 0 = spatial (0,0)
 *      token 1 = spatial (0,1)
 *      token 2 = spatial (0,2)
 *      token 3 = spatial (0,3)
 *      token 4 = spatial (1,0)
 *      ...
 *
 *   Visually:
 *
 *      grid                               sequence
 *
 *      ┌────┬────┬────┬────┐             ┌────┬────┬────┬────┬────┬────┐
 *      │ t0 │ t1 │ t2 │ t3 │             │ t0 │ t1 │ t2 │ t3 │ t4 │ ...│
 *      ├────┼────┼────┼────┤    ───▶      └────┴────┴────┴────┴────┴────┘
 *      │ t4 │ t5 │ t6 │ t7 │
 *      ├────┼────┼────┼────┤
 *      │ t8 │ t9 │ta  │tb  │
 *      └────┴────┴────┴────┘
 *
 *   Each token still carries N channels after any required input projection.
 *
 *  ─────────────────────────────────────────────────────────────────────────────────────
 *  POSITIONAL EMBEDDINGS: PUTTING GEOGRAPHY BACK IN
 *  ─────────────────────────────────────────────────────────────────────────────────────
 *
 *   Flattening destroys explicit 2D neighborhood structure.
 *   Without help, token 0 and token 50 are just two vectors in a list.
 *
 *   Positional embeddings repair that by attaching location identity:
 *
 *      token_i <- token_i + pos_i
 *
 *   Think of it like adding an address label to each token:
 *
 *      raw token        : "what am I?"
 *      position vector  : "where am I?"
 *      combined token   : "what am I, at this location?"
 *
 *   Tiny example:
 *
 *      feature token t5  = [0.2, 1.1, -0.4, 0.8]
 *      pos embed  p5     = [0.9, 0.1,  0.0, 0.3]
 *      sum               = [1.1, 1.2, -0.4, 1.1]
 *
 *   That sum is what enters attention.
 *
 *  ─────────────────────────────────────────────────────────────────────────────────────
 *  GLOBAL ATTENTION: EVERY TOKEN TALKS TO EVERY TOKEN
 *  ─────────────────────────────────────────────────────────────────────────────────────
 *
 *   This is the defining move of the layer.
 *
 *   If T=6, the communication graph looks like:
 *
 *      t0 ─────────────────────────────────────────┐
 *      t1 ─────────────────────────────────────┐   │
 *      t2 ────────────────────────────────┐    │   │
 *      t3 ───────────────────────────┐    │    │   │
 *      t4 ──────────────────────┐    │    │    │   │
 *      t5 ◀─────────────────────┴────┴────┴────┴───┘
 *
 *   In practice, that means a dense T x T score matrix per head.
 *
 *   Compare:
 *
 *      CNN receptive field     = local unless many layers accumulate
 *      windowed transformer    = local within each window per layer
 *      global ViT              = whole token set in one layer
 *
 *   This is powerful and expensive.
 *
 *  ─────────────────────────────────────────────────────────────────────────────────────
 *  Q, K, V PROJECTIONS
 *  ─────────────────────────────────────────────────────────────────────────────────────
 *
 *   After LN1:
 *
 *      X_ln shape = [B, T, N]
 *
 *   Three learned linear maps are applied:
 *
 *      Q = X_ln * Wq
 *      K = X_ln * Wk
 *      V = X_ln * Wv
 *
 *   Shapes:
 *
 *      Wq, Wk, Wv : [N, N]
 *      Q, K, V    : [B, T, N]
 *
 *   Split into heads:
 *
 *      [B, T, N] -> [B, h, T, d]
 *
 *   Each head gets its own view of similarity and message content.
 *
 *  ─────────────────────────────────────────────────────────────────────────────────────
 *  SCORES MATRIX
 *  ─────────────────────────────────────────────────────────────────────────────────────
 *
 *   For one head:
 *
 *      scores = Q * K^T / sqrt(d)
 *
 *   If T=5:
 *
 *             k0    k1    k2    k3    k4
 *         ┌───────────────────────────────┐
 *      q0 │ 0.3   1.2  -0.1   0.7   0.5   │
 *      q1 │ 0.0   0.6   0.8   0.4  -0.2   │
 *      q2 │ 1.4   0.1   0.2   0.0   1.0   │
 *      q3 │-0.3   0.2   1.1   0.9   0.8   │
 *      q4 │ 0.5   0.4   0.0   1.6   0.2   │
 *         └───────────────────────────────┘
 *
 *   Read row q2:
 *
 *      token 2 is asking which keys matter to it
 *
 *   Read column k4:
 *
 *      how available token 4 is as relevant context to all queries
 *
 *  ─────────────────────────────────────────────────────────────────────────────────────
 *  SOFTMAX TURNS SCORES INTO ROUTING PROBABILITIES
 *  ─────────────────────────────────────────────────────────────────────────────────────
 *
 *   Softmax is applied row-wise:
 *
 *      probs(q_i, :)
 *
 *   Each row becomes a probability distribution over all tokens.
 *
 *   Example after softmax:
 *
 *             k0    k1    k2    k3    k4
 *         ┌───────────────────────────────┐
 *      q0 │ 0.12  0.29  0.08  0.19  0.15  │
 *      q1 │ 0.14  0.21  0.26  0.18  0.09  │
 *      q2 │ 0.38  0.10  0.11  0.09  0.25  │
 *      q3 │ 0.08  0.13  0.31  0.25  0.23  │
 *      q4 │ 0.14  0.12  0.08  0.45  0.11  │
 *         └───────────────────────────────┘
 *
 *   Interpretation:
 *
 *      q4 sends 45% of its attention budget to token 3
 *
 *   Since every row sees the entire sequence, a token can reach any location immediately.
 *
 *  ─────────────────────────────────────────────────────────────────────────────────────
 *  VALUE MIXING
 *  ─────────────────────────────────────────────────────────────────────────────────────
 *
 *   The value vectors carry the information being transported:
 *
 *      ctx = probs * V
 *
 *   For one row:
 *
 *      ctx_4 =
 *          0.14 * v0 +
 *          0.12 * v1 +
 *          0.08 * v2 +
 *          0.45 * v3 +
 *          0.11 * v4
 *
 *   So the output token is not copied from a single location.
 *   It is a learned mixture of all token values.
 *
 *   That is why attention behaves like content-dependent routing.
 *
 *  ─────────────────────────────────────────────────────────────────────────────────────
 *  MULTI-HEAD PARALLELISM
 *  ─────────────────────────────────────────────────────────────────────────────────────
 *
 *   Different heads can specialize:
 *
 *      head 0: local texture similarity
 *      head 1: long horizontal alignments
 *      head 2: object-part co-occurrence
 *      head 3: background suppression
 *
 *   Whether they actually learn that depends on data and optimization,
 *   but the architecture permits several simultaneous attention patterns.
 *
 *   After each head produces [B,T,d], the outputs are concatenated:
 *
 *      [head0 | head1 | ... | head(h-1)] -> [B,T,N]
 *
 *   Then `Wo` mixes those channels across heads.
 *
 *  ─────────────────────────────────────────────────────────────────────────────────────
 *  RESIDUAL + FFN SUBLAYERS
 *  ─────────────────────────────────────────────────────────────────────────────────────
 *
 *   The transformer block has two major residual sections.
 *
 *   First residual:
 *
 *      x0 = tokens + pos
 *      x1 = MHSA(LN1(x0))
 *      x2 = x0 + x1
 *
 *   Second residual:
 *
 *      x3 = FFN(LN2(x2))
 *      x4 = x2 + x3
 *
 *   FFN acts independently on every token:
 *
 *      [N] -> [rN] -> activation -> [N]
 *
 *   Attention mixes positions.
 *   FFN mixes channels.
 *
 *  ─────────────────────────────────────────────────────────────────────────────────────
 *  COMPLETE FORWARD CHECKPOINTS
 *  ─────────────────────────────────────────────────────────────────────────────────────
 *
 *   [01] receive input [B,C,H,W]
 *   [02] flatten spatial positions into T tokens
 *   [03] align / project channels to N if needed
 *   [04] add absolute positional embeddings
 *   [05] cache pre-attention residual x0
 *   [06] apply LN1
 *   [07] compute Q
 *   [08] compute K
 *   [09] compute V
 *   [10] split Q/K/V into h heads
 *   [11] compute scaled score matrices QK^T / sqrt(d)
 *   [12] softmax scores over key dimension
 *   [13] multiply probabilities by V
 *   [14] concatenate heads
 *   [15] apply output projection Wo
 *   [16] add first residual
 *   [17] cache pre-FFN residual x2
 *   [18] apply LN2
 *   [19] apply FFN linear 1
 *   [20] apply activation
 *   [21] apply FFN linear 2
 *   [22] add second residual
 *   [23] reshape tokens back to [B,N,H,W]
 *   [24] write output
 *
 *  ─────────────────────────────────────────────────────────────────────────────────────
 *  MICRO EXAMPLE: WHY GLOBAL ATTENTION CHANGES BEHAVIOR
 *  ─────────────────────────────────────────────────────────────────────────────────────
 *
 *   Suppose token t0 represents a small object corner in the top-left.
 *   Suppose token t47 represents another corner from the same object in the bottom-right.
 *
 *   A local operator may need many layers before those two positions influence each other.
 *
 *   A ViT layer can do this immediately:
 *
 *      q0 strongly matches k47
 *
 *      therefore
 *
 *      t0 can directly borrow value information from t47 in one attention step
 *
 *   This is the core expressive advantage of global attention.
 *
 *  ─────────────────────────────────────────────────────────────────────────────────────
 *  WHY THE COST BLOWS UP
 *  ─────────────────────────────────────────────────────────────────────────────────────
 *
 *   The expensive object is the score matrix:
 *
 *      [B, h, T, T]
 *
 *   If H=W=13:
 *
 *      T = 169
 *      T*T = 28,561
 *
 *   If H=W=26:
 *
 *      T = 676
 *      T*T = 456,976
 *
 *   If H=W=52:
 *
 *      T = 2704
 *      T*T = 7,311,616
 *
 *   That growth is quadratic in the number of spatial sites.
 *   This is why full ViT attention is usually placed at smaller feature maps.
 *
 *  ─────────────────────────────────────────────────────────────────────────────────────
 *  BACKWARD PASS: REVERSE THE WHOLE CONVERSATION
 *  ─────────────────────────────────────────────────────────────────────────────────────
 *
 *   Start with:
 *
 *      g_out = dL/d(output) with shape [B,N,H,W]
 *
 *   Reverse reshape:
 *
 *      g_tokens = [B,T,N]
 *
 *   Reverse second residual:
 *
 *      x4 = x2 + ffn_out
 *
 *      so:
 *
 *      g_x2_from_skip = g_tokens
 *      g_ffn_out      = g_tokens
 *
 *   Reverse FFN:
 *
 *      g_ffn_out
 *        └── FC2 backward -> g_act, grad_W2, grad_b2
 *             └── activation backward -> g_fc1
 *                  └── FC1 backward -> g_ln2, grad_W1, grad_b1
 *                       └── LN2 backward -> g_x2_from_ffn, grad_gamma2, grad_beta2
 *
 *   Accumulate:
 *
 *      g_x2 = g_x2_from_skip + g_x2_from_ffn
 *
 *   Reverse first residual:
 *
 *      x2 = x0 + attn_out
 *
 *      g_x0_from_skip = g_x2
 *      g_attn_out     = g_x2
 *
 *   Reverse output projection:
 *
 *      g_concat_heads, grad_Wo, grad_bo
 *
 *   Reverse head concat:
 *
 *      split g_concat_heads into g_ctx for each head
 *
 *   Reverse value mixing:
 *
 *      ctx = probs * V
 *
 *      gives:
 *
 *      g_probs = g_ctx * V^T
 *      g_V     = probs^T * g_ctx
 *
 *   Reverse softmax:
 *
 *      g_scores = softmax_backward(g_probs)
 *
 *   Reverse score computation:
 *
 *      scores = QK^T / sqrt(d)
 *
 *      gives:
 *
 *      g_Q = g_scores * K / sqrt(d)
 *      g_K = g_scores^T * Q / sqrt(d)
 *
 *   Reverse Q/K/V projections:
 *
 *      backprop through Wq, Wk, Wv
 *
 *      accumulate:
 *
 *      g_ln1_from_Q
 *      g_ln1_from_K
 *      g_ln1_from_V
 *
 *      grad_Wq, grad_bq
 *      grad_Wk, grad_bk
 *      grad_Wv, grad_bv
 *
 *   Reverse LN1:
 *
 *      g_x0_from_attn, grad_gamma1, grad_beta1
 *
 *   Accumulate at x0:
 *
 *      g_x0 = g_x0_from_skip + g_x0_from_attn
 *
 *   Reverse positional addition:
 *
 *      g_tokens_before_pos = g_x0
 *      grad_pos_embed     += g_x0
 *
 *   Reverse flatten:
 *
 *      reshape token gradients back to [B,C,H,W] aligned input layout
 *
 *  ─────────────────────────────────────────────────────────────────────────────────────
 *  POSITIONAL EMBEDDING GRADIENTS
 *  ─────────────────────────────────────────────────────────────────────────────────────
 *
 *   Since forward does:
 *
 *      x0 = token + pos
 *
 *   backward does:
 *
 *      g_token += g_x0
 *      g_pos   += g_x0
 *
 *   Each position vector learns from the exact token slot where it was added.
 *
 *   If the layer is resized and positional embeddings are interpolated,
 *   the implementation must keep that mapping coherent across training/inference.
 *
 *  ─────────────────────────────────────────────────────────────────────────────────────
 *  WHAT EACH PARAMETER LEARNS
 *  ─────────────────────────────────────────────────────────────────────────────────────
 *
 *   Positional embeddings:
 *
 *      encode spatial identity and relative location cues in an absolute table
 *
 *   Wq:
 *
 *      learns what kinds of content should ask for context
 *
 *   Wk:
 *
 *      learns what kinds of content should advertise themselves as useful
 *
 *   Wv:
 *
 *      learns what information should be carried once a connection is chosen
 *
 *   Wo:
 *
 *      learns how to recombine head-specific contexts
 *
 *   FFN weights:
 *
 *      learn token-local nonlinear feature transformations after context mixing
 *
 *  ─────────────────────────────────────────────────────────────────────────────────────
 *  MEMORY MAP
 *  ─────────────────────────────────────────────────────────────────────────────────────
 *
 *   Important temporary tensors often include:
 *
 *      token sequence        [B,T,N]
 *      positional sum        [B,T,N]
 *      Q/K/V                 [B,h,T,d]
 *      scores                [B,h,T,T]
 *      probs                 [B,h,T,T]
 *      attention context     [B,h,T,d]
 *      concat heads          [B,T,N]
 *      FFN hidden            [B,T,rN]
 *
 *   The `[B,h,T,T]` tensors dominate memory when T is large.
 *
 *  ─────────────────────────────────────────────────────────────────────────────────────
 *  DEBUGGING CHECKLIST
 *  ─────────────────────────────────────────────────────────────────────────────────────
 *
 *   If shapes fail:
 *
 *      verify N % heads == 0
 *      verify patchify order matches reshape-back order
 *      verify positional embedding table matches current patch grid
 *
 *   If memory explodes:
 *
 *      inspect T = (H/P)*(W/P)
 *      inspect head count
 *      inspect whether attention scores are stored in unnecessary precision
 *
 *   If outputs look spatially scrambled:
 *
 *      inspect patchify indexing
 *      inspect unflatten indexing
 *      inspect positional interpolation after resize
 *
 *   If gradients are wrong:
 *
 *      verify accumulation from both residual branches
 *      verify softmax backward is applied along the key axis
 *      verify Q/K score gradients use the correct transpose order
 *
 *  ─────────────────────────────────────────────────────────────────────────────────────
 *  MENTAL SUMMARY
 *  ─────────────────────────────────────────────────────────────────────────────────────
 *
 *   ViT turns an image feature map into a meeting room of tokens.
 *
 *   Every token:
 *
 *      keeps its own content
 *      carries a positional identity
 *      asks all other tokens for useful context
 *      receives a weighted blend of their value vectors
 *      passes through an FFN
 *      returns to the spatial grid richer than before
 *
 *   The strength of the layer is immediate long-range communication.
 *   The cost of the layer is the dense T x T conversation matrix.
 *
 * ### CFG File Usage:
 * To use this layer in a Darknet `.cfg` file:
 * ```cfg
 * [vit]
 * dim=512             ; output embedding channels (alias: filters)
 * patch_size=2        ; patch edge length P
 * heads=8             ; number of attention heads
 * mlp_dim=2048        ; FFN hidden width (ffn_ratio=4 is still accepted)
 * pos_embed=learned    ; trainable absolute positional embedding
 * pos_init=zero        ; stable detector drop-in start
 * activation=gelu     ; FFN activation
 * ```
 *
 * Practical meaning of each field:
 *
 *   `dim` / `filters`
 *      output channel count after attention and FFN.
 *      If `filters ==` previous layer channels, the first residual shortcut is active.
 *      If `filters !=` previous layer channels, the first shortcut is skipped and the
 *      block behaves more like "attention projection + FFN" than a full same-width block.
 *
 *   `heads`
 *      number of global attention heads.
 *      The implementation requires `dim`/`filters` to be divisible by `heads`.
 *
 *   `mlp_dim` / `ffn_ratio`
 *      direct feed-forward hidden width, or the legacy expansion ratio shorthand.
 *      If both are specified, `mlp_dim` must equal `dim * ffn_ratio`.
 *
 *   `pos_embed`
 *      `learned` keeps the original trainable absolute embedding. `sinusoidal`
 *      uses the fixed 2D SimpleViT-style embedding and does not update it.
 *
 *   `pos_init`
 *      `zero` starts learned positional embeddings with no additive perturbation,
 *      which is safer when inserting ViT into an existing detector cfg.
 *
 *   `activation`
 *      activation used inside the FFN. `gelu` is the default transformer-style choice.
 *
 * Recommended placement:
 *
 *   use this layer deep in the backbone or neck, not near the raw image
 *   prefer small feature maps such as `13x13`, `20x20`, or `26x26`
 *   keep `filters` equal to the incoming channel count when possible
 *   avoid stacking many global ViT blocks at high resolution because attention cost is quadratic in the patch-token count
 *
 * Good example: place it after several downsampling stages
 * ```cfg
 * [convolutional]
 * batch_normalize=1
 * filters=512
 * size=3
 * stride=2
 * pad=1
 * activation=leaky
 *
 * [vit]
 * dim=512
 * patch_size=2
 * heads=8
 * mlp_dim=2048
 * pos_embed=learned
 * pos_init=zero
 * activation=gelu
 *
 * [convolutional]
 * batch_normalize=1
 * filters=512
 * size=1
 * stride=1
 * pad=1
 * activation=leaky
 * ```
 *
 * Read that placement as:
 *
 *   downsample first  -> reduce token count T = (H/P)*(W/P)
 *   apply global ViT  -> let every remaining token communicate with every other token
 *   continue with conv -> fuse the enriched features back into the CNN pipeline
 *
 * Constraints worth remembering:
 *
 *   previous layer must output image-shaped data
 *   `dim`/`filters` must be divisible by `heads`
 *   positional embeddings resize with the patch grid, but very frequent resizing still changes the token lattice
 *   memory grows with `[batch, heads, T, T]`, so resolution dominates runtime and RAM
 */

#ifdef DARKNET_GPU
void forward_vit_layer_gpu(Darknet::Layer & l, Darknet::NetworkState state);
void backward_vit_layer_gpu(Darknet::Layer & l, Darknet::NetworkState state);
void update_vit_layer_gpu(Darknet::Layer & l, int batch, float learning_rate, float momentum, float decay, float loss_scale);

void push_vit_layer(Darknet::Layer & l);
void pull_vit_layer(Darknet::Layer & l);
void resize_vit_pos_embed_gpu(const float *old_embed_gpu, float *new_embed_gpu,
	int old_H, int old_W, int new_H, int new_W, int C);
#endif

/// @param patch_size    kernel size P of the patch window (>=1)
/// @param patch_stride  stride between patch centers (>=1). Pass patch_size for classic non-overlap.
/// @param patch_pad     zero-padding around the input (>=0). Pass 0 for classic non-overlap.
Darknet::Layer make_vit_layer(int batch, int h, int w, int c, int n,
	int patch_size, int patch_stride, int patch_pad,
	int heads, int ffn_ratio, int mlp_dim, int pos_embed_type, int pos_init_type,
	ACTIVATION activation, int index, int train);

void resize_vit_layer(Darknet::Layer * l, int w, int h);
void forward_vit_layer(Darknet::Layer & l, Darknet::NetworkState state);
void backward_vit_layer(Darknet::Layer & l, Darknet::NetworkState state);
void backward_vit_attention_tail(Darknet::Layer & l, Darknet::NetworkState state, const float *d_attn_out, const float *d_skip);
void update_vit_layer(Darknet::Layer & l, int batch, float learning_rate, float momentum, float decay);
