#pragma once

#include <cstddef>

namespace Darknet
{
	/// FP4-quantization for Tucker attention has no external scale (cuDNN Frontend's
	/// block-scale op computes it internally per call -- same limitation already
	/// accepted for conv FP4), and fp4_gemm_execute() takes plain FP32 row-major
	/// operands directly rather than a pre-quantized buffer.  These helpers convert
	/// the attention layer's __half Q/K/V/scores/context/gradient buffers into the
	/// FP32 layouts fp4_gemm_execute() needs, batched over (head, window) slices.

	/// Plain elementwise __half -> float cast, no reshape.  Used for operands whose
	/// native (T, D) layout already matches what fp4_gemm_execute() needs (Q/K in the
	/// forward scores GEMM, V in the dAttn backward GEMM).
	void fp4_half_to_float_gpu(const void * src_half, size_t count, float * dst, int batch = 1, size_t src_stride = 0, size_t dst_stride = 0);

	/// __half -> float cast with the column dimension zero-padded to cols_pad --
	/// NVFP4's 16-element reduction block requires the k dimension to be a multiple
	/// of 16, which T (window_size^2) generally is not.  Used for (T, T) tensors
	/// (attn, d_scores) consumed as the un-transposed reduction-over-T operand.
	void fp4_pad_cols_half_to_float_gpu(const void * src_half, int rows, int cols, int cols_pad, float * dst, int batch = 1, size_t src_stride = 0, size_t dst_stride = 0);

	/// __half -> float cast, transposed, with the (now-column) reduction dimension
	/// zero-padded to rows_pad.  Used for (T, D) tensors that must become the
	/// (D, key_pad) "B-transposed" operand (V, K, Q, d_context depending on which
	/// backward GEMM), and for (T, T) tensors used transposed (d_scores in dK).
	void fp4_transpose_pad_cols_half_to_float_gpu(const void * src_half, int rows, int cols, int rows_pad, float * dst, int batch = 1, size_t src_stride = 0, size_t dst_stride = 0);

	/// float -> __half cast with a multiplicative scale folded in (fp4_gemm_execute
	/// has no alpha parameter, so the 1/sqrt(D) attention scale for dQ/dK must be
	/// applied here; pass scale = 1.0f for the plain-cast case).
	void fp4_scale_cast_float_to_half_gpu(const float * src, size_t count, float scale, void * dst_half, int batch = 1, size_t src_stride = 0, size_t dst_stride = 0);
}
