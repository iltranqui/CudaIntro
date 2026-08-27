#pragma once

#include <cstddef>
#include <cstdint>

namespace Darknet
{
	/// Pack already-scaled FP32 values as two E2M1 nibbles per byte.
	/// These conversion kernels intentionally work on Ada and do not invoke FP4 tensor cores.
	void fp4_pack_e2m1_gpu(const float * input_gpu, size_t element_count, uint8_t * packed_gpu);
	void fp4_pack_e2m1_stochastic_gpu(
		const float * input_gpu, size_t element_count, uint64_t seed, uint8_t * packed_gpu);

	void fp4_transpose_rowmajor_gpu(const float * input_gpu, int rows, int columns, float * output_gpu);
	void fp4_copy_matrix_columns_gpu(const float * input_gpu, int rows, int columns,
		int output_columns, int column_offset, float * output_gpu);
	void fp4_pack_batch_rows_gpu(const float * input_gpu, int batch, int rows, int columns, float * output_gpu);

	/// Storage required by cuBLASLt's packed E2M1 matrix and tiled UE4M3
	/// scale-factor layout.  `outer` is M for operand A or N for operand B;
	/// `reduction` is the contiguous K dimension quantized in blocks of 16.
	size_t fp4_cublaslt_packed_bytes(int outer, int reduction);
	size_t fp4_cublaslt_scale_bytes(int outer, int reduction);

	/// Convert an FP32 row-major [outer, K] matrix into the NVFP4 representation
	/// consumed by cuBLASLt: two E2M1 values per byte plus one unsigned E4M3
	/// dequantization scale per 16 adjacent K values in NVIDIA's 128x4 tiles.
	/// `reduction` must be a positive multiple of 16.
	bool fp4_quantize_cublaslt_gpu(const float * input_gpu, int outer, int reduction,
		uint8_t * packed_gpu, uint8_t * scales_gpu);

	/// Same NVFP4 representation as fp4_quantize_cublaslt_gpu(), but converts a
	/// NCHW activation tensor directly to the [spatial, channels] right-operand
	/// layout consumed by a following compatible 1x1 convolution.  It lets a
	/// launch-time precision chain retain the producer output in NVFP4 instead
	/// of quantizing the next layer's FP32 input again. `channels` must be a
	/// positive multiple of 16.
	bool fp4_quantize_nchw_to_cublaslt_gpu(const float * input_nchw_gpu,
		int batch, int channels, int height, int width,
		uint8_t * packed_gpu, uint8_t * scales_gpu);

	/// Same as fp4_quantize_nchw_to_cublaslt_gpu(), but folds a final ReLU into
	/// the pack.  It is used only on a safe Conv->Conv relay edge.
	bool fp4_relu_quantize_nchw_to_cublaslt_gpu(float * input_output_nchw_gpu,
		int batch, int channels, int height, int width,
		uint8_t * packed_gpu, uint8_t * scales_gpu);

	/// Calibration-only running-max reduction over a device float buffer, used to
	/// measure per-layer input activation |amax| across `darknet detector calibrate
	/// -fp4` images. Same generic max-abs reduction as FP8's calibration helpers --
	/// nothing FP4-format-specific here, just kept in this translation unit so it
	/// links whenever DARKNET_HAS_FP4 is enabled without depending on FP8 sources.
	void fp4_clear_amax_gpu(float * amax_gpu);
	void fp4_accumulate_amax_gpu(const float * src, size_t count, float * amax_gpu);
	float fp4_pull_amax_gpu(float * amax_gpu);
}
