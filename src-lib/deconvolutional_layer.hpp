// Reference: https://github.com/gmayday1997/darknet.CG/tree/master/src/deconvolutional_layer.h
#pragma once

#include "darknet_internal.hpp"

Darknet::Layer make_deconvolutional_layer(int batch, int h, int w, int c, int n, int size, int stride, ACTIVATION activation);
void resize_deconvolutional_layer(Darknet::Layer *l, int h, int w);
void forward_deconvolutional_layer(Darknet::Layer & l, Darknet::NetworkState state);
void backward_deconvolutional_layer(Darknet::Layer & l, Darknet::NetworkState state);
void update_deconvolutional_layer(Darknet::Layer & l, int batch, float learning_rate, float momentum, float decay);

Darknet::Image get_deconvolutional_image(const Darknet::Layer & l);
Darknet::Image get_deconvolutional_delta(const Darknet::Layer & l);
int deconvolutional_out_height(const Darknet::Layer & l);
int deconvolutional_out_width(const Darknet::Layer & l);

#ifdef DARKNET_GPU
void forward_deconvolutional_layer_gpu(Darknet::Layer & l, Darknet::NetworkState state);
void backward_deconvolutional_layer_gpu(Darknet::Layer & l, Darknet::NetworkState state);
void update_deconvolutional_layer_gpu(Darknet::Layer & l, int batch, float learning_rate, float momentum, float decay, float loss_scale);
void push_deconvolutional_layer(Darknet::Layer & l);
void pull_deconvolutional_layer(Darknet::Layer & l);
#endif
