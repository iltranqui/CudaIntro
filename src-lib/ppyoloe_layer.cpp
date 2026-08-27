#include "ppyoloe_layer.hpp"

namespace Darknet
{
	Darknet::Layer make_ppyoloe_layer(int batch, int w, int h, int classes, int max_boxes, int reg_max)
	{
		return make_modern_yolo_layer(batch, w, h, classes, max_boxes, ModernYoloHeadKind::PPYOLOE, reg_max);
	}

	void resize_ppyoloe_layer(Darknet::Layer * l, int w, int h)
	{
		resize_modern_yolo_layer(l, w, h);
	}

	void forward_ppyoloe_layer(Darknet::Layer & l, Darknet::NetworkState state)
	{
		forward_modern_yolo_layer(l, state, ModernYoloHeadKind::PPYOLOE);
	}

	void backward_ppyoloe_layer(Darknet::Layer & l, Darknet::NetworkState state)
	{
		backward_modern_yolo_layer(l, state);
	}

	int ppyoloe_num_detections(const Darknet::Layer & l, float thresh)
	{
		return modern_yolo_num_detections(l, thresh);
	}

	int get_ppyoloe_detections(const Darknet::Layer & l, int w, int h, int netw, int neth, float thresh, int * map, int relative, Darknet::Detection * dets, int letter)
	{
		return get_modern_yolo_detections(l, w, h, netw, neth, thresh, map, relative, dets, letter, ModernYoloHeadKind::PPYOLOE);
	}

#ifdef DARKNET_GPU
	void forward_ppyoloe_layer_gpu(Darknet::Layer & l, Darknet::NetworkState state)
	{
		forward_modern_yolo_layer_gpu(l, state, ModernYoloHeadKind::PPYOLOE);
	}

	void backward_ppyoloe_layer_gpu(Darknet::Layer & l, Darknet::NetworkState state)
	{
		backward_modern_yolo_layer_gpu(l, state);
	}
#endif
}
