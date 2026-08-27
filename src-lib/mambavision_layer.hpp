#pragma once

#include "darknet_internal.hpp"

/**
 * @brief MambaVision mixer block.
 *
 * Config syntax:
 *   [mambavision]
 *   filters=128       # output channels; defaults to input channels
 *   state=8           # selective-scan state size
 *   conv_size=3       # regular depthwise 1D convolution kernel over H*W tokens
 *   dt_rank=0         # 0 means ceil(filters / 16)
 *   ffn_ratio=4       # FFN expansion ratio
 *   activation=gelu   # FFN activation
 *
 * The layer follows the MambaVision block described in arXiv:2407.08083:
 * LayerNorm -> MambaVision mixer -> residual -> LayerNorm -> FFN -> residual.
 */

Darknet::Layer make_mambavision_layer(int batch, int h, int w, int c, int n,
	int d_state, int conv_size, int dt_rank, int ffn_ratio, ACTIVATION activation, int index, int train);

void resize_mambavision_layer(Darknet::Layer * l, int w, int h);
void forward_mambavision_layer(Darknet::Layer & l, Darknet::NetworkState state);
void backward_mambavision_layer(Darknet::Layer & l, Darknet::NetworkState state);
void update_mambavision_layer(Darknet::Layer & l, int batch, float learning_rate, float momentum, float decay);

#ifdef DARKNET_GPU
void forward_mambavision_layer_gpu(Darknet::Layer & l, Darknet::NetworkState state);
void backward_mambavision_layer_gpu(Darknet::Layer & l, Darknet::NetworkState state);
void update_mambavision_layer_gpu(Darknet::Layer & l, int batch, float learning_rate, float momentum, float decay, float loss_scale);
void push_mambavision_layer(Darknet::Layer & l);
void pull_mambavision_layer(Darknet::Layer & l);
#endif
