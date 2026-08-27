#pragma once

#include "darknet_internal.hpp"

#ifdef DARKNET_GPU_CUDA

namespace Darknet
{
	enum class YoloCudaLaunchStatus
	{
		launched,
		unsupported,
		recoverable_failure,
	};

	/** Fused classic-YOLO tensor copy and activation.
	 *
	 * The input and output use Darknet's anchor-major NCHW-like YOLO layout.
	 * For classic coordinates, x/y and objectness/classes are logistic activated;
	 * w/h remain logits.  For new_coords, the legacy raw-value behaviour is kept.
	 */
	void yolo_activate_output_gpu(
		const float * input_gpu,
		float * output_gpu,
		int batch,
		int anchors,
		int width,
		int height,
		int classes,
		float scale_x_y,
		bool new_coords);

	/** Launch the supported classic-YOLO loss path without synchronizing.
	 *
	 * Setup and allocation errors are reported as recoverable failures before
	 * any loss kernel is launched.  Once launched, CUDA execution errors are
	 * fatal because the CUDA context may no longer be safe for CPU fallback.
	 */
	YoloCudaLaunchStatus forward_yolo_training_cuda(
		Darknet::Layer & layer,
		Darknet::NetworkState state,
		const char ** reason);

	/** Complete all pending YOLO metric copies with one stream synchronization. */
	void finalize_yolo_training_cuda(Darknet::Network & net);

	/** Drop shape-dependent CUDA scratch.  It is recreated lazily on next use. */
	void resize_yolo_training_cuda(Darknet::Layer & layer);

	/** Release all CUDA and pinned-host resources owned by the YOLO layer. */
	void release_yolo_training_cuda(Darknet::Layer & layer);
}

#endif // DARKNET_GPU_CUDA
