#pragma once

#include "darknet_internal.hpp"

// Shift values are treated as cyclic channel offsets and normalized into `[0, c)`
// when the layer is constructed. `shifts` remains the wedge/default schedule, while
// `inner_shifts` optionally decouples the scalar pathway from the wedge pathway.
Darknet::Layer make_clifford_layer(
	int batch, int h, int w, int c, int n,
	const int *shifts, int num_shifts,
	const int *inner_shifts, int num_inner_shifts,
	int ctx_mode, int cli_mode, int gffn_mode, int higher_mode,
	int dwconv_size, int num_dwconv,
	ACTIVATION activation, float drop_path, float layerscale_init,
	int index, int train);

void forward_clifford_layer(Darknet::Layer & l, Darknet::NetworkState state);
void backward_clifford_layer(Darknet::Layer & l, Darknet::NetworkState state);
void update_clifford_layer(Darknet::Layer & l, int batch, float lr, float momentum, float decay);
void resize_clifford_layer(Darknet::Layer * l, int w, int h);

void save_clifford_weights(Darknet::Layer & l, FILE *fp);
size_t load_clifford_weights(Darknet::Layer & l, FILE *fp);

#ifdef DARKNET_GPU
void forward_clifford_layer_gpu(Darknet::Layer & l, Darknet::NetworkState state);
void backward_clifford_layer_gpu(Darknet::Layer & l, Darknet::NetworkState state);
void update_clifford_layer_gpu(Darknet::Layer & l, int batch, float lr, float momentum, float decay, float loss_scale);
void push_clifford_layer(Darknet::Layer & l);
void pull_clifford_layer(Darknet::Layer & l);
#endif
