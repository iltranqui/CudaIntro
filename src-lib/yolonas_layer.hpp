#pragma once

#include "modern_yolo_layer.hpp"

namespace Darknet
{
	Darknet::Layer make_yolonas_layer(int batch, int w, int h, int classes, int max_boxes, int reg_max = 16);

	void resize_yolonas_layer(Darknet::Layer * l, int w, int h);

	void forward_yolonas_layer(Darknet::Layer & l, Darknet::NetworkState state);
	void backward_yolonas_layer(Darknet::Layer & l, Darknet::NetworkState state);

	int yolonas_num_detections(const Darknet::Layer & l, float thresh);
	int get_yolonas_detections(const Darknet::Layer & l, int w, int h, int netw, int neth, float thresh, int * map, int relative, Darknet::Detection * dets, int letter);

#ifdef DARKNET_GPU
	void forward_yolonas_layer_gpu(Darknet::Layer & l, Darknet::NetworkState state);
	void backward_yolonas_layer_gpu(Darknet::Layer & l, Darknet::NetworkState state);
#endif
}
