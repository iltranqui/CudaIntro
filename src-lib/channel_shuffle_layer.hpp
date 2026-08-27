// Reference: https://github.com/gmayday1997/darknet.CG/tree/master/src/channel_shuffle.h
#pragma once

#include "darknet_internal.hpp"

Darknet::Layer make_channel_shuffle_layer(int batch, int w, int h, int c, int groups);
void resize_channel_shuffle_layer(Darknet::Layer *l, int h, int w);
void forward_channel_shuffle_layer(Darknet::Layer & l, Darknet::NetworkState state);
void backward_channel_shuffle_layer(Darknet::Layer & l, Darknet::NetworkState state);

#ifdef DARKNET_GPU
void channel_shuffle_ongpu(int count, float *output, float *input, int group_row, int group_column, int feature_map_size, int spatial_size);
void forward_channel_shuffle_layer_gpu(Darknet::Layer & l, Darknet::NetworkState state);
void backward_channel_shuffle_layer_gpu(Darknet::Layer & l, Darknet::NetworkState state);
#endif
