// Reference: https://github.com/gmayday1997/darknet.CG/tree/master/src/channel_slice.h
#pragma once

#include "darknet_internal.hpp"

Darknet::Layer make_channel_slice_layer(
	int batch,
	int w,
	int h,
	int c,
	int begin_slice_point,
	int end_slice_point,
	int axis,
	int n,
	int *input_layers,
	int *input_sizes);

void resize_channel_slice_layer(Darknet::Layer *l, Darknet::Network *net);
void forward_channel_slice_layer(Darknet::Layer & l, Darknet::NetworkState state);
void backward_channel_slice_layer(Darknet::Layer & l, Darknet::NetworkState state);

#ifdef DARKNET_GPU
void channel_slice_ongpu(int count, float *output, float *input, int batch_size, int spatial_size, int input_slice_axis, int output_slice_axis, int begin_slice_axis, int forward);
void forward_channel_slice_layer_gpu(Darknet::Layer & l, Darknet::NetworkState state);
void backward_channel_slice_layer_gpu(Darknet::Layer & l, Darknet::NetworkState state);
#endif
