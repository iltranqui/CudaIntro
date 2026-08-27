#pragma once

#include "darknet_internal.hpp"

/** @file
 * Darknet-compatible modern anchor-free YOLO heads.
 *
 * This layer family mirrors the public shape of yolo_layer.hpp while exposing
 * three head behaviours:
 *
 * - YOLOX: anchor-free cxcywh, decoupled-object/class logits, SimOTA-style
 *   dynamic assignment within one feature map.
 * - PP-YOLOE: anchor-free ltrb distribution regression, Task-Aligned
 *   assignment, Varifocal-style quality targets.
 * - YOLO-NAS-style: anchor-free ltrb distribution regression with
 *   quality-aware duplicate down-weighting intended to make training more
 *   NMS-aware without copying Deci's licensed model architecture/weights.
 *
 * Output layouts expected from the preceding [convolutional] layer:
 *
 *   YOLOX:
 *     filters = classes + 5
 *     channels = [tx, ty, tw, th, objectness, class_0 ... class_N]
 *
 *   PP-YOLOE / YOLO-NAS-style:
 *     filters = 4 * (reg_max + 1) + 1 + classes
 *     channels = [l_bins..., t_bins..., r_bins..., b_bins..., quality, classes...]
 *
 * Labels use the same standard Darknet detection truth layout as yolo_layer:
 *
 *   class_id cx cy w h [track_id]
 *
 * All coordinates are normalized to [0, 1].
 */

namespace Darknet
{
	enum class ModernYoloHeadKind
	{
		YOLOX = 0,
		PPYOLOE = 1,
		YOLONAS = 2
	};

	const char * modern_yolo_head_name(ModernYoloHeadKind kind);

	Darknet::Layer make_modern_yolo_layer(
		int batch,
		int w,
		int h,
		int classes,
		int max_boxes,
		ModernYoloHeadKind kind,
		int reg_max = 16);

	void resize_modern_yolo_layer(Darknet::Layer * l, int w, int h);

	void forward_modern_yolo_layer(Darknet::Layer & l, Darknet::NetworkState state, ModernYoloHeadKind kind);

	void backward_modern_yolo_layer(Darknet::Layer & l, Darknet::NetworkState state);

	int modern_yolo_num_detections(const Darknet::Layer & l, float thresh);
	int modern_yolo_num_detections_batch(const Darknet::Layer & l, float thresh, int batch);
	int get_modern_yolo_detections(const Darknet::Layer & l, int w, int h, int netw, int neth, float thresh, int * map, int relative, Darknet::Detection * dets, int letter, ModernYoloHeadKind kind);
	int get_modern_yolo_detections_batch(const Darknet::Layer & l, int w, int h, int netw, int neth, float thresh, int * map, int relative, Darknet::Detection * dets, int letter, int batch, ModernYoloHeadKind kind);
	void correct_modern_yolo_boxes(Darknet::Detection * dets, int n, int w, int h, int netw, int neth, int relative, int letter);

#ifdef DARKNET_GPU
	void forward_modern_yolo_layer_gpu(Darknet::Layer & l, Darknet::NetworkState state, ModernYoloHeadKind kind);

	void backward_modern_yolo_layer_gpu(Darknet::Layer & l, Darknet::NetworkState state);
#endif
}
