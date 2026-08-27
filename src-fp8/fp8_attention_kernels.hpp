#pragma once

#include <cstddef>

namespace Darknet
{
	/// FP8-quantize a batched, half-precision (__half) rowmajor (rows x cols) tensor into
	/// (rows x cols_pad), applying a precomputed scale.  Mirrors
	/// fp8_quantize_rowmajor_pad_cols_gpu() in fp8_kernels.hpp, but for a __half source --
	/// needed because Tucker attention's CUBLAS_HALF forward path keeps Q/K/V in __half,
	/// never converting back to float before the attention GEMMs.
	void fp8_quantize_half_rowmajor_pad_cols_gpu(const void * src_half, int rows, int cols, int cols_pad, const float * scale_gpu, void * dst_fp8, size_t dst_ld = 0, int batch = 1, size_t src_stride = 0, size_t dst_stride = 0);

	/// Same as above but transposes into (cols x rows_pad) -- used to produce the
	/// "B operand, transposed" layout cuBLASLt's FP8 TN-only matmul requires (see
	/// Fp8GemmSpec's doc comment in fp8_gemm.hpp).
	void fp8_quantize_half_transpose_rowmajor_pad_cols_gpu(const void * src_half, int rows, int cols, int rows_pad, const float * scale_gpu, void * dst_fp8, int batch = 1, size_t src_stride = 0, size_t dst_stride = 0);

	/// Accumulate max-abs over a batched half-precision buffer (flat, `count` = rows*cols*batch)
	/// into *amax_gpu.  Used during calibration; mirrors fp8_accumulate_amax_gpu() for a
	/// __half source.
	void fp8_accumulate_amax_half_gpu(const void * src_half, size_t count, float * amax_gpu);

	/// FP8-quantize a batched, half-precision rowmajor (rows x cols) tensor into a fully
	/// padded (rows_pad x cols_pad) destination -- both dimensions zero-padded, needed for
	/// cuBLASLt FP8's "A operand" (output_rows and reduction_pad must both be multiples of
	/// 16; unlike the B operand, A has no unconstrained dimension to skip padding on).
	void fp8_quantize_half_pad_rows_cols_gpu(const void * src_half, int rows, int cols, int rows_pad, int cols_pad, const float * scale_gpu, void * dst_fp8, int batch = 1, size_t src_stride = 0, size_t dst_stride = 0);

	/// Same as above but for a float32 source (used for the post-softmax attention weights,
	/// which run through cudnnSoftmaxForward's FP32 path before being re-quantized for the
	/// attn@V GEMM).
	void fp8_quantize_pad_rows_cols_gpu(const float * src, int rows, int cols, int rows_pad, int cols_pad, const float * scale_gpu, void * dst_fp8, int batch = 1, size_t src_stride = 0, size_t dst_stride = 0);

	/// Dequantizes the scores GEMM's raw FP32 cuBLASLt output (A=K row-padded to key_pad, B=Q
	/// with T unconstrained) directly into the tightly packed FP16 buffer cudnnSoftmaxForward's
	/// existing tensor descriptor expects, dropping the key_pad-T garbage columns produced by
	/// K's zero-padded rows. See Fp8GemmSpec's doc comment (fp8_gemm.hpp) for why the raw
	/// column-major D buffer is already byte-identical to row-major (t_query, key_pad).
	void fp8_dequant_compact_scores_half_gpu(const float * src, int t_query, int t_key, int key_pad, float score_scale, void * dst_half, int batch = 1, size_t src_stride = 0, size_t dst_stride = 0);
}
