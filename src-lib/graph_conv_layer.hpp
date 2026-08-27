#pragma once

#include "darknet_internal.hpp"

/**
 * @brief Graph Convolutional Layer
 *
 * ╔══════════════════════════════════════════════════════════════════════════════════════╗
 * ║              GRAPH CONVOLUTION — PIXELS THAT VOTE ON EACH OTHER                     ║
 * ╚══════════════════════════════════════════════════════════════════════════════════════╝
 *
 *  THE CORE INSIGHT: Standard convolution gives every neighbor equal weight.
 *  Graph convolution lets similar neighbors shout louder. Each pixel dynamically
 *  decides who its most relevant "friends" are — and listens to them more.
 *
 *  ─────────────────────────────────────────────────────────────────────────────────────
 *  STEP 1 — THE GRAPH: Image as a Social Network
 *  ─────────────────────────────────────────────────────────────────────────────────────
 *
 *   Pixel Grid (values)      Standard conv weights      GCN attention weights (learned)
 *   ╔═════╤═════╤═════╗      ┌───────┬───────┬───────┐  ┌────────┬────────┬────────┐
 *   ║  8  │  2  │  7  ║      │  1/9  │  1/9  │  1/9  │  │  0.04  │  0.31  │  0.05  │
 *   ╠═════╪═════╪═════╣      ├───────┼───────┼───────┤  ├────────┼────────┼────────┤
 *   ║  1  │  9  │  8  ║ ──▶  │  1/9  │  1/9  │  1/9  │  │  0.02  │   —    │  0.38  │
 *   ╠═════╪═════╪═════╣      ├───────┼───────┼───────┤  ├────────┼────────┼────────┤
 *   ║  3  │  7  │  6  ║      │  1/9  │  1/9  │  1/9  │  │  0.03  │  0.25  │  0.07  │
 *   ╚═════╧═════╧═════╝      └───────┴───────┴───────┘  └────────┴────────┴────────┘
 *    Center pixel = 9          Mean pooling (static)      Attention (dynamic):
 *                              All equal, ignores          similar pixels (7, 8, 8)
 *                              feature similarity          get HIGHER weights!
 *
 *  ─────────────────────────────────────────────────────────────────────────────────────
 *  STEP 2 — ATTENTION (edge_mode=1): How Edge Weights Are Computed
 *  ─────────────────────────────────────────────────────────────────────────────────────
 *
 *   For center pixel i and neighbor j, the edge weight e_ij is computed:
 *
 *   center   ──▶  feature_i = [f₁, f₂, ..., f_C]    (C-dimensional feature vector)
 *   neighbor ──▶  feature_j = [g₁, g₂, ..., g_C]
 *
 *   Attention logit:    a_ij = Σ_c  (feature_i[c] + feature_j[c]) · w_attn[c]
 *                                                                    └──────────┘
 *                                                        small learned kernel (1×1×C)
 *
 *   Attention weight:   e_ij = softmax_j(a_ij) = exp(a_ij) / Σ_{k ∈ N(i)} exp(a_ik)
 *
 *   SOFTMAX NORMALIZATION ensures Σ_j e_ij = 1  (all neighbor weights sum to exactly 1):
 *
 *   Raw logits:     [ 2.1,  0.3,  1.8,  0.9,  2.0,  0.4,  0.5,  1.7,  2.2 ]  (9 values)
 *                     ↓     ↓     ↓     ↓     ↓     ↓     ↓     ↓     ↓
 *   After softmax:  [ 0.31, 0.04, 0.22, 0.08, 0.26, 0.05, 0.06, 0.20, 0.33 ]  (sums to 1)
 *                     ▲                        ▲                        ▲
 *                  similar                  similar                 most similar
 *                  → high weight            → high weight           → highest weight
 *
 *   Mean mode (edge_mode=0): skip the attention kernel, set all e_ij = 1/K².
 *   Faster to train but loses the ability to discriminate neighbors by similarity.
 *
 *  ─────────────────────────────────────────────────────────────────────────────────────
 *  STEP 3 — NEIGHBORHOOD AGGREGATION: The Weighted Conversation
 *  ─────────────────────────────────────────────────────────────────────────────────────
 *
 *   For center node i, collect a weighted sum of all neighbor features:
 *
 *       aggr_i = Σ_{j ∈ N(i)}  e_ij · feature_j
 *
 *   Visualized for a single feature channel (center pixel = 9):
 *
 *           Pixel values         Attention weights       Weighted contribution
 *           ┌────┬────┬────┐     ┌──────┬──────┬──────┐  ┌──────┬──────┬──────┐
 *           │  8 │  2 │  7 │  ×  │ 0.04 │ 0.31 │ 0.05 │ =│ 0.32 │ 0.62 │ 0.35 │
 *           ├────┼────┼────┤     ├──────┼──────┼──────┤  ├──────┼──────┼──────┤
 *           │  1 │  9 │  8 │     │ 0.02 │  —   │ 0.38 │  │ 0.02 │  —   │ 3.04 │
 *           ├────┼────┼────┤     ├──────┼──────┼──────┤  ├──────┼──────┼──────┤
 *           │  3 │  7 │  6 │     │ 0.03 │ 0.25 │ 0.07 │  │ 0.09 │ 1.75 │ 0.42 │
 *           └────┴────┴────┘     └──────┴──────┴──────┘  └──────┴──────┴──────┘
 *            (neighbors)          (attn weights)           aggr_i = Σ = 6.61
 *                                                          (biased toward similar pixels!)
 *
 *  ─────────────────────────────────────────────────────────────────────────────────────
 *  STEP 4 — FEATURE TRANSFORMATION: The Linear Projection
 *  ─────────────────────────────────────────────────────────────────────────────────────
 *
 *   The raw aggregate is projected into output space via a learned weight matrix:
 *
 *       h_i  =  aggr_i  ×  W_neigh
 *               [C_in]     [C_in, C_out]  →  output: [C_out]
 *
 *   This is equivalent to a standard 1×1 convolution applied to the aggregated features.
 *
 *  ─────────────────────────────────────────────────────────────────────────────────────
 *  STEP 5 — SELF-CONNECTION (use_self=1): Don't Forget Your Own Identity!
 *  ─────────────────────────────────────────────────────────────────────────────────────
 *
 *   Without self-connection, the center pixel is "washed out" by its neighbors.
 *   Adding a separate self-transform preserves the pixel's own representation:
 *
 *       output_i  =  h_i  +  feature_i × W_self
 *                    └──────────────────────────── neighborhood contribution
 *                                  └────────────── center's own separate projection
 *
 *   ╔══════════════╦══════════════════════════════════════════════════════╗
 *   ║  use_self=0  ║  output = Σ_j e_ij · f_j · W_neigh                 ║
 *   ╠══════════════╬══════════════════════════════════════════════════════╣
 *   ║  use_self=1  ║  output = Σ_j e_ij · f_j · W_neigh + f_i · W_self  ║
 *   ╚══════════════╩══════════════════════════════════════════════════════╝
 *
 *  ─────────────────────────────────────────────────────────────────────────────────────
 *  COMPLETE DATA FLOW
 *  ─────────────────────────────────────────────────────────────────────────────────────
 *
 *  Input [B,C,H,W]
 *       │
 *       ├──▶ Attention subnet (1×1 conv) ──▶ logits [B, K², H, W] ──▶ softmax ──▶ e_ij
 *       │                                                                          │
 *       ├──▶ Neighbor features ──▶ weighted aggregate (× e_ij) ──▶ × W_neigh ──▶ ┐ │
 *       │                                                                         ├─┴─▶ + ──▶ BN ──▶ Act ──▶ Output
 *       └──▶ Self features (use_self=1) ──────────────────────── × W_self ──────▶ ┘
 *
 * ╔══════════════════════════════════════════════════════════════════════════════════════╗
 * ║  "NOW I GET IT!": Unlike standard conv (equal weights for all neighbors), GCN       ║
 * ║  amplifies similar neighbors and suppresses different ones. After training, pixels   ║
 * ║  in flat regions average smoothly; pixels near boundaries amplify their strong       ║
 * ║  edge-neighbor signals — emergent sharpening with no explicit supervision!           ║
 * ╚══════════════════════════════════════════════════════════════════════════════════════╝
 *
 *  BACKWARD PASS — WHO GETS BLAMED FOR THE OUTPUT?
 *  ─────────────────────────────────────────────────────────────────────────────────────
 *
 *   dL/dOutput
 *       │
 *       ├──▶ activation backward
 *       ├──▶ batch-norm backward (if enabled)
 *       │
 *       ├──▶ split gradient by additive merge
 *       │
 *       │      dY
 *       │      ├──▶ neighbor branch ──▶ grad W_neigh
 *       │      │                        grad aggr_i
 *       │      │                        grad neighbor features
 *       │      │
 *       │      └──▶ self branch (use_self=1) ──▶ grad W_self
 *       │                                   └──▶ grad center features
 *       │
 *       ├──▶ if edge_mode=1:
 *       │      grad aggr_i
 *       │        ├──▶ grad e_ij from weighted sum
 *       │        ├──▶ softmax backward
 *       │        └──▶ grad attention logits/kernel
 *       │
 *       └──▶ accumulate all paths into dL/dInput
 *
 *   The critical detail: each input pixel receives gradient from
 *   1. its own self-projection,
 *   2. all neighborhoods where it was sampled as a neighbor,
 *   3. the attention mechanism that decided how loudly it should vote.
 *
 *  ─────────────────────────────────────────────────────────────────────────────────────
 *  SYMBOL LEGEND
 *  ─────────────────────────────────────────────────────────────────────────────────────
 *
 *   B   = batch size
 *   C   = input channels
 *   N   = output channels / filters
 *   H,W = input height and width
 *   K   = neighborhood size (`size` in cfg)
 *   R   = neighborhood radius = floor(K/2) when symmetric
 *   s   = stride
 *   p   = padding
 *   i   = center pixel / node index
 *   j   = neighbor index inside the KxK stencil
 *   f_i = feature vector at center node i
 *   f_j = feature vector at neighbor j
 *   e_ij= edge weight from neighbor j to center i
 *   aggr_i = weighted neighbor aggregate for node i
 *
 *   The graph is not stored as a giant sparse matrix here in the conceptual view.
 *   Instead, it is generated locally from image neighborhoods:
 *
 *      every output site = one center node + one local KxK neighborhood
 *
 *  ─────────────────────────────────────────────────────────────────────────────────────
 *  SPATIAL VIEW VS GRAPH VIEW
 *  ─────────────────────────────────────────────────────────────────────────────────────
 *
 *   Spatial CNN thinking:
 *
 *      "take a local patch and apply a fixed kernel"
 *
 *   Graph thinking:
 *
 *      "the center node asks nearby nodes for information, but it does not trust
 *       every neighbor equally"
 *
 *   Same patch, different interpretation:
 *
 *      image patch
 *
 *      ┌────┬────┬────┐
 *      │ n0 │ n1 │ n2 │
 *      ├────┼────┼────┤
 *      │ n3 │  i │ n4 │
 *      ├────┼────┼────┤
 *      │ n5 │ n6 │ n7 │
 *      └────┴────┴────┘
 *
 *      graph star centered at i
 *
 *             n0   n1   n2
 *               \  |  /
 *                \ | /
 *          n3 ---- i ---- n4
 *                / | \
 *               /  |  \
 *             n5   n6   n7
 *
 *   In mean mode:
 *
 *      every incoming arrow has the same weight
 *
 *   In attention mode:
 *
 *      every incoming arrow gets its own learned confidence
 *
 *  ─────────────────────────────────────────────────────────────────────────────────────
 *  FORWARD PASS AS A PIPELINE
 *  ─────────────────────────────────────────────────────────────────────────────────────
 *
 *   [01] receive input [B,C,H,W]
 *   [02] define output lattice from stride, padding, dilation
 *   [03] for each output site, gather the KxK neighborhood
 *   [04] for each center-neighbor pair, form features for scoring
 *   [05] if edge_mode=1, compute attention logits
 *   [06] normalize logits with softmax over the neighborhood
 *   [07] if edge_mode=0, use uniform weights 1/(K*K)
 *   [08] compute weighted aggregate aggr_i = sum_j e_ij * f_j
 *   [09] apply neighbor projection W_neigh
 *   [10] if use_self=1, apply self projection W_self to f_i
 *   [11] add projected neighbor and self branches
 *   [12] apply batch norm if enabled
 *   [13] apply activation
 *   [14] write output [B,N,H_out,W_out]
 *
 *   This is the same computation repeated at every output coordinate.
 *
 *  ─────────────────────────────────────────────────────────────────────────────────────
 *  NEIGHBORHOOD EXTRACTION
 *  ─────────────────────────────────────────────────────────────────────────────────────
 *
 *   Imagine K=3 and stride=1.
 *
 *   For a center pixel at (y,x), the neighborhood coordinates are:
 *
 *      (y-1,x-1)  (y-1,x)  (y-1,x+1)
 *      (y,  x-1)  (y,  x)  (y,  x+1)
 *      (y+1,x-1)  (y+1,x)  (y+1,x+1)
 *
 *   If padding is active, out-of-bounds accesses are handled by the layer's padding rule.
 *
 *   The neighborhood can be imagined as a stack of C-channel vectors:
 *
 *      n0 = [c0 c1 c2 ... cC-1]
 *      n1 = [c0 c1 c2 ... cC-1]
 *      ...
 *      n8 = [c0 c1 c2 ... cC-1]
 *
 *   The center node is one of those positions.
 *
 *   If `graph_valid_mask_zero` is enabled in the implementation logic,
 *   padded / invalid samples may be explicitly suppressed to zero contribution.
 *
 *  ─────────────────────────────────────────────────────────────────────────────────────
 *  EDGE SCORING IN ATTENTION MODE
 *  ─────────────────────────────────────────────────────────────────────────────────────
 *
 *   The layer first decides how strongly the center should listen to each neighbor.
 *
 *   One conceptual scoring pattern is:
 *
 *      pair_ij = combine(f_i, f_j)
 *
 *   In this header's explanation, the combination is illustrated as:
 *
 *      a_ij = sum_c (f_i[c] + f_j[c]) * w_attn[c]
 *
 *   This has an important effect:
 *
 *      if the center and neighbor activate similar learned channels strongly,
 *      their score tends to increase
 *
 *   Visual intuition:
 *
 *      center feature:   [ 2  0  5  1 ]
 *      neighbor A:       [ 2  1  4  1 ]   -> similar
 *      neighbor B:       [ 0  4  0  3 ]   -> dissimilar
 *
 *      score(center,A) > score(center,B)
 *
 *   So the graph is adaptive:
 *
 *      the same spatial offset can be important in one region and ignored in another
 *
 *  ─────────────────────────────────────────────────────────────────────────────────────
 *  SOFTMAX AS COMPETITION INSIDE THE STENCIL
 *  ─────────────────────────────────────────────────────────────────────────────────────
 *
 *   Once logits are computed, the neighborhood enters a competition:
 *
 *      weights = softmax([a_i0, a_i1, ..., a_i(K*K-1)])
 *
 *   This means:
 *
 *      neighbors are not scored independently
 *      they are scored relative to each other
 *
 *   If one logit becomes much larger, it can dominate the whole local message.
 *
 *   Example:
 *
 *      logits   = [ 0.1, 0.3, 2.0, 0.0, 0.2, 0.1, -0.2, 0.4, 0.0 ]
 *      softmax  = [ 0.07,0.08,0.45,0.06,0.07,0.07,0.05,0.09,0.06 ]
 *
 *   Read it as:
 *
 *      "neighbor 2 carries nearly half of the message budget for this center node"
 *
 *   In mean mode the budget is split evenly:
 *
 *      each neighbor gets exactly 1/9 when K=3
 *
 *  ─────────────────────────────────────────────────────────────────────────────────────
 *  AGGREGATION AS MESSAGE PASSING
 *  ─────────────────────────────────────────────────────────────────────────────────────
 *
 *   After weights are known, the center collects a message:
 *
 *      aggr_i = sum_j e_ij * f_j
 *
 *   Expand this for K=3:
 *
 *      aggr_i =
 *          e_i0 * n0 +
 *          e_i1 * n1 +
 *          e_i2 * n2 +
 *          e_i3 * n3 +
 *          e_i4 * n4 +
 *          e_i5 * n5 +
 *          e_i6 * n6 +
 *          e_i7 * n7 +
 *          e_i8 * n8
 *
 *   This is vector-valued addition.
 *   Every term is a full C-dimensional feature vector.
 *
 *   What changes from site to site is:
 *
 *      1. which neighbors are present
 *      2. what their features are
 *      3. what attention weights they receive
 *
 *  ─────────────────────────────────────────────────────────────────────────────────────
 *  NEIGHBOR PROJECTION VS SELF PROJECTION
 *  ─────────────────────────────────────────────────────────────────────────────────────
 *
 *   The aggregate lives in the input feature space [C].
 *   It is projected into output space [N]:
 *
 *      neigh_out_i = aggr_i * W_neigh
 *
 *   If `use_self=1`, the center's original feature also gets its own projection:
 *
 *      self_out_i = f_i * W_self
 *
 *   Then:
 *
 *      pre_bn_i = neigh_out_i + self_out_i
 *
 *   Why have a separate self branch?
 *
 *      because the center pixel's identity is often too important to mix
 *      into the generic neighborhood average
 *
 *   It acts like a learned skip connection local to each node.
 *
 *  ─────────────────────────────────────────────────────────────────────────────────────
 *  COMPLETE FORWARD MAP FOR ONE PIXEL
 *  ─────────────────────────────────────────────────────────────────────────────────────
 *
 *      center feature f_i
 *           │
 *           ├──▶ attention scorer with each neighbor ──▶ logits a_ij ──▶ softmax ──┐
 *           │                                                                       │
 *      neighbors f_j ────────────────────────────────────────────────────────────────┤
 *           │                                                                       ▼
 *           └──────────────────────────── weighted sum aggr_i = Σ e_ij f_j ──▶ W_neigh
 *
 *      center feature f_i ───────────────────────────────────────────────────▶ W_self
 *
 *      W_neigh branch + W_self branch ──▶ BN ──▶ activation ──▶ output_i
 *
 *   The local graph is rebuilt for every spatial site.
 *
 *  ─────────────────────────────────────────────────────────────────────────────────────
 *  WHY THIS CAN SHARPEN EDGES
 *  ─────────────────────────────────────────────────────────────────────────────────────
 *
 *   Near a semantic boundary, the neighborhood might contain two very different regions:
 *
 *      left side  = object
 *      right side = background
 *
 *   A fixed convolution averages both unless deeper features learn to compensate.
 *
 *   A graph-conv attention rule can do this instead:
 *
 *      center on object
 *      assign high weight to object-like neighbors
 *      assign low weight to background-like neighbors
 *
 *   Result:
 *
 *      aggregation becomes boundary-aware
 *
 *   That is why this layer often feels like:
 *
 *      "a convolution that learned to respect feature similarity"
 *
 *  ─────────────────────────────────────────────────────────────────────────────────────
 *  BACKWARD PASS AS REVERSED MESSAGE PASSING
 *  ─────────────────────────────────────────────────────────────────────────────────────
 *
 *   Begin with:
 *
 *      g_out = dL/d(output)
 *
 *   Reverse activation:
 *
 *      g_bn = g_out * activation'(pre_act)
 *
 *   Reverse BN if present:
 *
 *      g_pre_bn, grad_bn_scale, grad_bn_bias
 *
 *   Reverse branch addition:
 *
 *      g_neigh_out = g_pre_bn
 *      g_self_out  = g_pre_bn   if use_self=1
 *
 *   Reverse neighbor projection:
 *
 *      g_aggr_i   = g_neigh_out * W_neigh^T
 *      grad_W_neigh += aggr_i^T * g_neigh_out
 *
 *   Reverse self projection:
 *
 *      g_f_i_from_self = g_self_out * W_self^T
 *      grad_W_self    += f_i^T * g_self_out
 *
 *   Reverse aggregation:
 *
 *      aggr_i = sum_j e_ij * f_j
 *
 *      gives:
 *
 *      g_f_j_from_aggr += e_ij * g_aggr_i
 *      g_e_ij          += dot(g_aggr_i, f_j)
 *
 *   Reverse softmax if attention mode:
 *
 *      g_logits = softmax_backward(g_e)
 *
 *   Reverse attention scorer:
 *
 *      send gradient into:
 *      - attention kernel weights
 *      - center features f_i
 *      - neighbor features f_j
 *
 *   Final input gradient at one node accumulates several streams:
 *
 *      from being the center node
 *      from being a neighbor to nearby centers
 *      from the self branch
 *      from the attention scorer
 *
 *  ─────────────────────────────────────────────────────────────────────────────────────
 *  CREDIT ASSIGNMENT OVERLAP
 *  ─────────────────────────────────────────────────────────────────────────────────────
 *
 *   One pixel can influence many outputs.
 *
 *   Example with K=3 and stride=1:
 *
 *      a single input pixel may appear in up to 9 neighboring stencils
 *
 *   Therefore, during backprop:
 *
 *      its final gradient is the sum of all those contributions
 *
 *   ASCII view:
 *
 *           out00  out01  out02
 *             \      |      /
 *              \     |     /
 *               \    |    /
 *                  pixel p
 *               /    |    \
 *              /     |     \
 *           out10  out11  out12
 *
 *   Each output that sampled `p` can send gradient back to it.
 *
 *  ─────────────────────────────────────────────────────────────────────────────────────
 *  PARAMETER GROUPS
 *  ─────────────────────────────────────────────────────────────────────────────────────
 *
 *   Depending on configuration, the learnable pieces include:
 *
 *      W_neigh
 *      b_neigh
 *      W_self
 *      b_self
 *      attention kernel / scorer weights
 *      BN scale / BN bias
 *
 *   Their roles:
 *
 *      W_neigh   = how aggregated neighbor evidence is interpreted
 *      W_self    = how the center preserves its own identity
 *      W_attn    = how similarity / relevance is measured
 *      BN params = how channel distributions are renormalized after fusion
 *
 *  ─────────────────────────────────────────────────────────────────────────────────────
 *  COMPLEXITY INTUITION
 *  ─────────────────────────────────────────────────────────────────────────────────────
 *
 *   Standard KxK convolution:
 *
 *      fixed weights, fixed neighbor importance
 *
 *   Graph conv with attention:
 *
 *      fixed projection weights
 *      plus dynamic edge weights per spatial site
 *
 *   So relative to ordinary conv, graph conv adds:
 *
 *      local scoring cost
 *      softmax cost
 *      more temporary storage for logits / weights
 *
 *   The reward is adaptivity:
 *
 *      the layer can decide that a diagonal neighbor matters a lot here,
 *      but almost not at all one pixel away
 *
 *  ─────────────────────────────────────────────────────────────────────────────────────
 *  DEBUGGING CHECKLIST
 *  ─────────────────────────────────────────────────────────────────────────────────────
 *
 *   If outputs look oversmoothed:
 *
 *      inspect whether edge_mode accidentally fell back to mean mode
 *      inspect whether softmax logits are all nearly equal
 *      inspect whether self branch is disabled when it should be enabled
 *
 *   If outputs are unstable:
 *
 *      inspect softmax saturation
 *      inspect BN statistics
 *      inspect large attention kernel values
 *
 *   If gradients seem too small:
 *
 *      verify neighbor contributions are accumulated from every stencil
 *      verify softmax backward is implemented over the correct neighborhood axis
 *      verify invalid padded positions are masked consistently in forward and backward
 *
 *  ─────────────────────────────────────────────────────────────────────────────────────
 *  MENTAL SUMMARY
 *  ─────────────────────────────────────────────────────────────────────────────────────
 *
 *   Convolution says:
 *
 *      "all nearby pixels vote with fixed importance"
 *
 *   Graph convolution says:
 *
 *      "all nearby pixels may vote, but their volume depends on feature similarity"
 *
 *   Add the self branch and the final message becomes:
 *
 *      "listen to your neighbors, but keep your own identity too"
 *
 *  ─────────────────────────────────────────────────────────────────────────
 *  LAYER ARCHITECTURE AT A GLANCE
 *  ─────────────────────────────────────────────────────────────────────────
 *
 *   Input X [B, C, H, W]
 *        │
 *        │  ┌───────────────────────────────────────────────────────────────┐
 *        │  │  GATHER PHASE  (irregular — one thread per output node)        │
 *        │  │                                                                 │
 *        ├──┤  center pixel → graph_ref  [B, C, out_H, out_W]              │
 *        │  │                                                                 │
 *        ├──┤  K² neighbors ──┬── edge_mode=0: weight = 1/valid_count        │
 *        │  │                 └── edge_mode=1: logit = bias                  │
 *        │  │                                        + W_ref·ref             │
 *        │  │                                        + W_nbr·neighbor        │
 *        │  │                         → softmax over K² logits              │
 *        │  │                         → graph_alpha [B, G, out_H, out_W, K²]│
 *        │  │                                                                 │
 *        │  │  graph_agg = Σ_k  alpha_k · x_k  [B, C, out_H, out_W]       │
 *        │  └───────────────────────────────────────────────────────────────┘
 *        │
 *        │  ┌───────────────────────────────────────────────────────────────┐
 *        │  │  PROJECT PHASE  (regular — cuBLAS GEMM)                        │
 *        │  │                                                                 │
 *        │  │  out = W_neighbor @ graph_agg          [N, out_H*out_W]       │
 *        │  │  IF use_self: out += W_self @ graph_ref                        │
 *        │  └───────────────────────────────────────────────────────────────┘
 *        │
 *        │  ┌───────────────────────────────────────────────────────────────┐
 *        │  │  NORMALISE & ACTIVATE                                           │
 *        │  │  BN (or +bias)  →  activation  →  Y [B, N, out_H, out_W]    │
 *        │  └───────────────────────────────────────────────────────────────┘
 *
 *  Key point: the GATHER phase is data-dependent and irregular (variable valid
 *  neighbor counts, dynamic alpha weights); the PROJECT phase is a fixed matrix
 *  multiply.  Keeping them separate lets GEMM / cuBLAS handle 90% of FLOPS.
 *
 *  ─────────────────────────────────────────────────────────────────────────
 *  FORMAL PSEUDOCODE
 *  ─────────────────────────────────────────────────────────────────────────
 *
 *  INPUT:  X [B, C, H, W]         (NCHW feature map, C = groups * cpg)
 *  OUTPUT: Y [B, N, out_H, out_W] (NCHW, N = groups * npg)
 *
 *  ── STEP 1 — GATHER ───────────────────────────────────────────────────
 *    FOR b IN [0, B):
 *      FOR g IN [0, groups):                       # independent channel groups
 *        FOR (oy, ox) IN output grid:
 *
 *          (ref_y, ref_x) ← center input coord for (oy, ox)
 *          ref ← X[b, g*cpg : (g+1)*cpg, ref_y, ref_x]  # center features
 *          agg ← 0                                         # will hold Σ alpha·x
 *
 *          # Score each of the K² spatial neighbors
 *          FOR k IN [0, K²):
 *            (iy, ix) ← neighbor_coord(oy, ox, k)
 *            IF (iy, ix) out of bounds:
 *              logits[k] ← −∞;  valid[k] ← 0;  CONTINUE
 *            valid[k] ← 1
 *
 *            IF edge_mode = ATTENTION:
 *              logits[k] ← bias_k
 *                         + Σ_c  W_ref[k, c] · ref[c]
 *                         + Σ_c  W_nbr[k, c] · X[b, g*cpg+c, iy, ix]
 *            ELSE (UNIFORM):
 *              logits[k] ← 0
 *
 *          # Normalise edge weights
 *          valid_count ← #{k : valid[k]}
 *          IF edge_mode = ATTENTION:
 *            alpha[k] ← exp(logits[k] − max_logit) / Σ_{valid j} exp(logits[j] − max_logit)
 *          ELSE:
 *            alpha[k] ← 1 / valid_count  (uniform)
 *
 *          # Accumulate weighted neighbor features
 *          FOR k IN [0, K²):
 *            IF valid[k]:
 *              agg ← agg + alpha[k] · X[b, g*cpg : (g+1)*cpg, iy_k, ix_k]
 *
 *          graph_ref [b, g*cpg :, oy, ox] ← ref
 *          graph_agg [b, g*cpg :, oy, ox] ← agg
 *          graph_alpha[b, g, oy, ox, :]   ← alpha
 *
 *  ── STEP 2 — PROJECT ──────────────────────────────────────────────────
 *    FOR b IN [0, B), g IN [0, groups):
 *      # Reinterpret graph buffers as matrices [cpg, spatial]
 *      output[b, g] ← W_neighbor[g]  @  graph_agg[b, g]   # [npg, spatial]
 *      IF use_self:
 *        output[b, g] ← output[b, g] + W_self[g]  @  graph_ref[b, g]
 *
 *  ── STEP 3 — NORMALISE & ACTIVATE ────────────────────────────────────
 *    IF batch_normalize: Y ← BN(output)
 *    ELSE:               Y ← output + biases
 *    Y ← activation(Y)
 *
 *  ─────────────────────────────────────────────────────────────────────────
 *  BACKWARD PSEUDOCODE (overview)
 *  ─────────────────────────────────────────────────────────────────────────
 *
 *  ── STAGE A — DENSE (GEMM) ───────────────────────────────────────────
 *    d_agg      ← W_neighbor^T  @  delta_out          # gradient into graph_agg
 *    dW_neighbor += delta_out  @  graph_agg^T
 *    IF use_self:
 *      d_ref_self ← W_self^T @ delta_out
 *      dW_self   += delta_out @ graph_ref^T
 *
 *  ── STAGE B — GRAPH-SPECIFIC (per output node) ───────────────────────
 *    FOR each (b, g, oy, ox):
 *      # Gradient from aggregation: agg = Σ_k alpha_k · x_k
 *      FOR k IN valid neighbors:
 *        d_alpha[k] ← dot(d_agg[oy,ox], X[:, iy_k, ix_k])
 *        delta_input[:, iy_k, ix_k] += alpha[k] · d_agg[oy,ox]
 *
 *      IF edge_mode = ATTENTION:
 *        # Softmax backward: d_logit_k = alpha_k · (d_alpha_k − Σ_j alpha_j · d_alpha_j)
 *        sum_term ← Σ_j alpha[j] · d_alpha[j]
 *        FOR k IN valid neighbors:
 *          d_logit[k] ← alpha[k] · (d_alpha[k] − sum_term)
 *          # Propagate into edge kernel and input features
 *          dW_ref[k]  += d_logit[k] · ref
 *          dW_nbr[k]  += d_logit[k] · X[:, iy_k, ix_k]
 *          delta_input[:, ref_y, ref_x] += d_logit[k] · W_ref[k]
 *          delta_input[:, iy_k, ix_k]  += d_logit[k] · W_nbr[k]
 *
 *      IF use_self:
 *        delta_input[:, ref_y, ref_x] += d_ref_self[oy, ox]
 *
 * ### CFG File Usage:
 * To use this layer in a Darknet .cfg file:
 * ```cfg
 * [graph_convolutional]
 * batch_normalize=1     ; 1 to use batch normalization
 * filters=64            ; number of output filters
 * size=3                ; neighborhood size (3x3 grid)
 * stride=1              ; spatial stride
 * padding=1             ; padding
 * activation=leaky      ; activation function
 * edge_mode=1           ; 0 for simple averaging, 1 for learned edge weights (attention)
 * use_self=1            ; 1 to include the center pixel in the aggregation
 * ```
 *
 * ### Clifford Shift Schedules:
 * The `[clifford]` layer has a related geometric-topology control that is useful
 * to document here because it follows the same "local neighbors are not all used
 * in the same way" idea.
 *
 * In Clifford interaction, there are two product streams:
 *
 *   inner  = scalar / dot-like interaction
 *   wedge  = exterior / antisymmetric interaction
 *
 * Old shared behavior:
 * ```cfg
 * [clifford]
 * cli_mode=full
 * shifts=1,2,4,8
 * ```
 *
 * This means both streams sample the same cyclic channel offsets.
 *
 * Decoupled behavior:
 * ```cfg
 * [clifford]
 * cli_mode=full
 * shifts_wedge=1,2,4,8
 * shifts_inner=1,3,5
 * ```
 *
 * Intended meaning:
 *
 *   shifts_wedge = which channel offsets feed the structural / rotational wedge path
 *   shifts_inner = which channel offsets feed the coherent / scalar inner path
 *
 * Why split them?
 *
 *   wedge tends to emphasize contrast, boundaries, and antisymmetric structure
 *   inner tends to emphasize alignment, similarity, and coherent channel mixing
 *
 * So different offset sets let the model look for structure and coherence at
 * different channel distances instead of forcing both products to reuse the same
 * topology.
 *
 * Compatibility rules:
 *
 *   `shifts=` still works and keeps the old shared behavior
 *   if only one side is specified, the other side inherits a sensible default
 *   if `shifts_inner` and `shifts_wedge` are identical, the implementation
 *   collapses back to the shared layout
 */

#ifdef DARKNET_GPU
/** @brief GPU forward pass for graph convolution. Performs neighborhood aggregation and feature transformation in parallel. */
void forward_graph_conv_layer_gpu(Darknet::Layer & l, Darknet::NetworkState state);
/** @brief GPU backward pass. Computes gradients for node features and dynamic edge weights. */
void backward_graph_conv_layer_gpu(Darknet::Layer & l, Darknet::NetworkState state);
/** @brief GPU weight update. Updates transformation matrices and edge prediction kernels. */
void update_graph_conv_layer_gpu(Darknet::Layer & l, int batch, float learning_rate, float momentum, float decay, float loss_scale);

/** @brief Synchronizes graph weights from CPU to GPU. */
void push_graph_conv_layer(Darknet::Layer & l);
/** @brief Synchronizes graph weights from GPU to CPU. */
void pull_graph_conv_layer(Darknet::Layer & l);
#endif

/** @brief Returns the required workspace size (typically zero for this implementation as it uses direct kernels). */
size_t get_graph_conv_workspace_size(const Darknet::Layer & l);

/**
 * @brief Factory function to initialize a new graph convolutional layer.
 * @param graph_edge_mode 0: Mean pooling neighbors, 1: Softmax attention over neighbors.
 * @param graph_use_self 1: Learn a separate transformation for the center pixel.
 */
Darknet::Layer make_graph_conv_layer(int batch, int steps, int h, int w, int c, int n, int groups,
	int size, int stride_x, int stride_y, int dilation, int padding, ACTIVATION activation,
	int batch_normalize, int binary, int xnor, int adam, int use_bin_output, int index,
	int antialiasing, Darknet::Layer * share_layer, int assisted_excitation, int train,
	int graph_edge_mode, int graph_use_self, int graph_valid_mask_zero);

/** @brief Resizes graph runtime buffers to handle a new input resolution. */
void resize_graph_conv_layer(Darknet::Layer * l, int w, int h);
/** @brief CPU forward pass implementation. Aggregates neighbor features and applies linear transforms. */
void forward_graph_conv_layer(Darknet::Layer & l, Darknet::NetworkState state);
/** @brief CPU backward pass. Propagates gradients through the aggregation and attention mechanisms. */
void backward_graph_conv_layer(Darknet::Layer & l, Darknet::NetworkState state);
/** @brief CPU weight update using SGD with momentum. */
void update_graph_conv_layer(Darknet::Layer & l, int batch, float learning_rate, float momentum, float decay);

/** @brief Calculates output height of the graph feature map. */
int graph_conv_out_height(const Darknet::Layer & l);
/** @brief Calculates output width of the graph feature map. */
int graph_conv_out_width(const Darknet::Layer & l);
