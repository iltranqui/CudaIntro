#pragma once

#include "modern_yolo_layer.hpp"

namespace Darknet
{
	Darknet::Layer make_yolox_layer(int batch, int w, int h, int classes, int max_boxes);

	void resize_yolox_layer(Darknet::Layer * l, int w, int h);

	void forward_yolox_layer(Darknet::Layer & l, Darknet::NetworkState state);
	void backward_yolox_layer(Darknet::Layer & l, Darknet::NetworkState state);

	int yolox_num_detections(const Darknet::Layer & l, float thresh);
	int get_yolox_detections(const Darknet::Layer & l, int w, int h, int netw, int neth, float thresh, int * map, int relative, Darknet::Detection * dets, int letter);

#ifdef DARKNET_GPU
	void forward_yolox_layer_gpu(Darknet::Layer & l, Darknet::NetworkState state);
	void backward_yolox_layer_gpu(Darknet::Layer & l, Darknet::NetworkState state);
#endif
}
