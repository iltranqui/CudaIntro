#include "yolonas_layer.hpp"

namespace Darknet
{
	Darknet::Layer make_yolonas_layer(int batch, int w, int h, int classes, int max_boxes, int reg_max)
	{
		return make_modern_yolo_layer(batch, w, h, classes, max_boxes, ModernYoloHeadKind::YOLONAS, reg_max);
	}

	void resize_yolonas_layer(Darknet::Layer * l, int w, int h)
	{
		resize_modern_yolo_layer(l, w, h);
	}

	void forward_yolonas_layer(Darknet::Layer & l, Darknet::NetworkState state)
	{
		forward_modern_yolo_layer(l, state, ModernYoloHeadKind::YOLONAS);
	}

	void backward_yolonas_layer(Darknet::Layer & l, Darknet::NetworkState state)
	{
		backward_modern_yolo_layer(l, state);
	}

	int yolonas_num_detections(const Darknet::Layer & l, float thresh)
	{
		return modern_yolo_num_detections(l, thresh);
	}

	int get_yolonas_detections(const Darknet::Layer & l, int w, int h, int netw, int neth, float thresh, int * map, int relative, Darknet::Detection * dets, int letter)
	{
		return get_modern_yolo_detections(l, w, h, netw, neth, thresh, map, relative, dets, letter, ModernYoloHeadKind::YOLONAS);
	}

#ifdef DARKNET_GPU
	void forward_yolonas_layer_gpu(Darknet::Layer & l, Darknet::NetworkState state)
	{
		forward_modern_yolo_layer_gpu(l, state, ModernYoloHeadKind::YOLONAS);
	}

	void backward_yolonas_layer_gpu(Darknet::Layer & l, Darknet::NetworkState state)
	{
		backward_modern_yolo_layer_gpu(l, state);
	}
#endif
}
