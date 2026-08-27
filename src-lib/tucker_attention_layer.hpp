#pragma once

#include "darknet_internal.hpp"

/* ── Empirical notes (LegoGears, 2026-05-23) ────────────────────────────────
 * - Strict: filters == input channels (line ~208). Mismatch = fatal at build.
 * - Depth-stable: 4 stacked layers train cleanly (vs ViT collapses at 3+).
 *   Low-rank Q/K/V/O caps per-layer singular values → bounded amplification.
 * - Tested layout (cfg/LegoGears_tucker.cfg): 2 at 28x20x256 (size=5,7 rank=32)
 *   + 2 at 14x10x512 (size=5,7 rank=64). Multi-scale + same-scale depth both OK.
 * - rank default = c/8. heads typical = ch/64. size = spatial mixing kernel.
 * - When you need attention depth → tucker. When you need single global pass → vit.
 * ─────────────────────────────────────────────────────────────────────────── */

Darknet::Layer make_tucker_attention_layer(int batch, int h, int w, int c, int n,
	int size, int heads, int rank_q, int rank_k, int rank_v, int rank_o,
	ACTIVATION activation, int index, int train);

void resize_tucker_attention_layer(Darknet::Layer *l, int w, int h);
void forward_tucker_attention_layer(Darknet::Layer & l, Darknet::NetworkState state);
void backward_tucker_attention_layer(Darknet::Layer & l, Darknet::NetworkState state);
void update_tucker_attention_layer(Darknet::Layer & l, int batch, float learning_rate, float momentum, float decay);

#ifdef DARKNET_GPU
void forward_tucker_attention_layer_gpu(Darknet::Layer & l, Darknet::NetworkState state);
void backward_tucker_attention_layer_gpu(Darknet::Layer & l, Darknet::NetworkState state);
void update_tucker_attention_layer_gpu(Darknet::Layer & l, int batch, float learning_rate, float momentum, float decay, float loss_scale);
#endif
