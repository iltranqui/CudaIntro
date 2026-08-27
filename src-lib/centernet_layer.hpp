#pragma once

#include "darknet_internal.hpp"

/** @file
 * CenterNet-style anchor-free detection head for axis-aligned boxes.
 *
 * Expected preceding [convolutional] layout:
 *
 *   filters = classes + 4
 *   channels = [class_heatmap_0 ... class_heatmap_N, log_w, log_h, off_x, off_y]
 *
 * Labels use the standard Darknet detection truth layout:
 *
 *   class_id cx cy w h [track_id]
 *
 * Decode contract:
 *
 *   cx = (cell_x + 0.5 + off_x) / heatmap_w
 *   cy = (cell_y + 0.5 + off_y) / heatmap_h
 *   w  = exp(log_w) / heatmap_w
 *   h  = exp(log_h) / heatmap_h
 *
 * Training target contract:
 *
 *   The class heatmap is a CenterNet center target.  By default this implementation
 *   uses an adaptive anisotropic Gaussian instead of the original isotropic disk.
 *   For AABB boxes this means the Gaussian is axis-aligned and stretched according
 *   to the object's width/height ratio:
 *
 *     base_radius = max(center_min_radius, gaussian_radius(w_cells, h_cells))
 *     radius_x    = round(base_radius * sqrt(w_cells / h_cells))
 *     radius_y    = round(base_radius * sqrt(h_cells / w_cells))
 *
 *   This is intentionally stronger for elongated or tiny objects: a thin 2x12 px
 *   object does not receive the same circular center blob as a 6x6 px object.
 *   The heatmap still predicts a single center peak; the ellipse only changes the
 *   supervision field around that peak.
 *
 * Tiny-object weighting:
 *
 *   The old hard threshold boost is replaced by a continuous multiplier:
 *
 *     weight = clamp(small_ref_size / min(box_w_px, box_h_px), 1, small_boost)
 *
 *   Example with small_ref_size=32 and small_boost=8:
 *
 *     2 px -> 8x, 4 px -> 8x, 8 px -> 4x, 16 px -> 2x, 32+ px -> 1x
 *
 * Relevant cfg keys:
 *
 *   center_min_radius       Minimum output-heatmap radius.
 *   anisotropic_gaussian    1 = ellipse target, 0 = original circular target.
 *   small_ref_size          Pixel reference size used by continuous tiny weighting.
 *   small_boost             Maximum total tiny-object multiplier.
 *   scale_min_px/max_px     Optional FPN scale gate based on max object side.
 *   gaussian_iou            CenterNet radius overlap target.
 *   focal_alpha/beta        Center heatmap focal-loss exponents.
 */

namespace Darknet
{
	Darknet::Layer make_centernet_layer(int batch, int w, int h, int classes, int max_boxes);

	void resize_centernet_layer(Darknet::Layer * l, int w, int h);

	void forward_centernet_layer(Darknet::Layer & l, Darknet::NetworkState state);
	void backward_centernet_layer(Darknet::Layer & l, Darknet::NetworkState state);

	int centernet_num_detections(const Darknet::Layer & l, float thresh);
	int centernet_num_detections_batch(const Darknet::Layer & l, float thresh, int batch);

	int get_centernet_detections(
		const Darknet::Layer & l,
		int w,
		int h,
		int netw,
		int neth,
		float thresh,
		int * map,
		int relative,
		Darknet::Detection * dets,
		int letter);

	int get_centernet_detections_batch(
		const Darknet::Layer & l,
		int w,
		int h,
		int netw,
		int neth,
		float thresh,
		int * map,
		int relative,
		Darknet::Detection * dets,
		int letter,
		int batch);

#ifdef DARKNET_GPU
	void forward_centernet_layer_gpu(Darknet::Layer & l, Darknet::NetworkState state);
	void backward_centernet_layer_gpu(Darknet::Layer & l, Darknet::NetworkState state);
#endif
}
