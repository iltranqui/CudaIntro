#pragma once

#include <cstddef>

namespace Darknet
{
	size_t fp8_tensor_bytes(size_t elements);
	size_t fp8_rowmajor_pad_cols_bytes(int rows, int cols_pad);
	size_t fp8_rowmajor_pad_rows_bytes(int rows_pad, int cols);

	enum class Fp8Im2colQuantizeKind
	{
		Generic,
		Conv3x3Pad1Stride1,
		Conv3x3Pad1Stride2
	};

	enum class Fp8DgradEpilogueKind
	{
		Generic,
		Direct1x1,
		Conv3x3Stride1Pad1
	};

	Fp8Im2colQuantizeKind fp8_im2col_quantize_kind(
		int kernel_h, int kernel_w,
		int pad_h, int pad_w,
		int stride_h, int stride_w,
		int dilation_h, int dilation_w);
	Fp8DgradEpilogueKind fp8_dgrad_epilogue_kind(
		int kernel_h, int kernel_w,
		int pad_h, int pad_w,
		int stride_h, int stride_w,
		int dilation_h, int dilation_w,
		int height, int width,
		int height_col, int width_col);

	/// dst_ld (0 = cols_pad) is the row stride of the destination in elements; a larger stride lets
	/// several launches tile per-image stripes of one wide matrix (batch folded into the k dimension)
	/// batch/src_stride/dst_stride: grid z iterates `batch` images; per-image pointers advance by the
	/// strides (floats for src, fp8 elements for dst) so a whole chunk quantizes in ONE launch
	void fp8_quantize_rowmajor_pad_cols_gpu(const float * src, int rows, int cols, int cols_pad, const float * scale_gpu, void * dst_fp8, size_t dst_ld = 0, int batch = 1, size_t src_stride = 0, size_t dst_stride = 0);
	void fp8_quantize_transpose_rowmajor_pad_cols_gpu(const float * src, int rows, int cols, int rows_pad, const float * scale_gpu, void * dst_fp8, int batch = 1, size_t src_stride = 0, size_t dst_stride = 0);
	void fp8_quantize_transpose_rowmajor_pad_rows_gpu(const float * src, int rows, int cols, int rows_pad, const float * scale_gpu, void * dst_fp8);
	void fp8_quantize_rowmajor_pad_cols_record_amax_gpu(const float * src, int rows, int cols, int cols_pad, const float * scale_gpu, void * dst_fp8, float * amax_gpu);
	void fp8_quantize_rowmajor_pad_rows_record_amax_gpu(const float * src, int rows, int cols, int rows_pad, const float * scale_gpu, void * dst_fp8, float * amax_gpu);
	void fp8_quantize_transpose_rowmajor_pad_cols_record_amax_gpu(const float * src, int rows, int cols, int rows_pad, const float * scale_gpu, void * dst_fp8, float * amax_gpu, int batch = 1, size_t src_stride = 0, size_t dst_stride = 0);
	void fp8_quantize_transpose_rowmajor_pad_rows_record_amax_gpu(const float * src, int rows, int cols, int rows_pad, const float * scale_gpu, void * dst_fp8, float * amax_gpu);
	void fp8_im2col_quantize_rowmajor_pad_rows_record_amax_gpu(
		const float * data_im,
		int channels, int height, int width,
		int kernel_h, int kernel_w,
		int pad_h, int pad_w,
		int stride_h, int stride_w,
		int dilation_h, int dilation_w,
		int rows_pad,
		const float * scale_gpu,
		void * dst_fp8,
		float * amax_gpu);
	void fp8_im2col_quantize_transpose_rowmajor_pad_rows_record_amax_gpu(
		const float * data_im,
		int channels, int height, int width,
		int kernel_h, int kernel_w,
		int pad_h, int pad_w,
		int stride_h, int stride_w,
		int dilation_h, int dilation_w,
		int cols_pad,
		const float * scale_gpu,
		void * dst_fp8,
		float * amax_gpu);
	void fp8_im2col_quantize_transpose_rowmajor_pad_cols_gpu(
		const float * data_im,
		int channels, int height, int width,
		int kernel_h, int kernel_w,
		int pad_h, int pad_w,
		int stride_h, int stride_w,
		int dilation_h, int dilation_w,
		int rows_pad,
		const float * scale_gpu,
		void * dst_fp8,
		int batch = 1, size_t src_stride = 0, size_t dst_stride = 0);
	void fp8_im2col_quantize_transpose_rowmajor_pad_cols_record_amax_gpu(
		const float * data_im,
		int channels, int height, int width,
		int kernel_h, int kernel_w,
		int pad_h, int pad_w,
		int stride_h, int stride_w,
		int dilation_h, int dilation_w,
		int rows_pad,
		const float * scale_gpu,
		void * dst_fp8,
		float * amax_gpu,
		int batch = 1, size_t src_stride = 0, size_t dst_stride = 0);
	void fp8_im2col_quantize_rowmajor_pad_cols_gpu(
		const float * data_im,
		int channels, int height, int width,
		int kernel_h, int kernel_w,
		int pad_h, int pad_w,
		int stride_h, int stride_w,
		int dilation_h, int dilation_w,
		int cols_pad,
		const float * scale_gpu,
		void * dst_fp8,
		size_t dst_ld = 0,
		int batch = 1, size_t src_stride = 0, size_t dst_stride = 0);
	void fp8_quantize_e5m2_transpose_rowmajor_pad_cols_record_amax_gpu(const float * src, int rows, int cols, int rows_pad, const float * scale_gpu, void * dst_fp8, float * amax_gpu, int batch = 1, size_t src_stride = 0, size_t dst_stride = 0);
	/// one read of dy -> both backward layouts: dst_wgrad = row-major (rows x cols) stripe padded to
	/// cols_pad with row stride wgrad_ld, dst_dgrad = transposed (cols x rows_pad); amax recorded once.
	/// Either destination may be null to skip that layout. Grid z iterates `batch` images.
	void fp8_quantize_e5m2_dual_layout_record_amax_gpu(
		const float * src, int rows, int cols, int cols_pad, int rows_pad,
		const float * scale_gpu,
		void * dst_wgrad_fp8, size_t wgrad_ld,
		void * dst_dgrad_fp8,
		float * amax_gpu,
		int batch = 1, size_t src_stride = 0, size_t wgrad_stride = 0, size_t dgrad_stride = 0);
	/// one read of the FP32 weights -> both GEMM operand layouts (row-major pad-cols + transposed pad-cols)
	void fp8_quantize_dual_layout_weights_gpu(
		const float * src, int filters, int kernel, int kernel_pad, int filters_pad,
		const float * scale_gpu, void * dst_rowmajor_fp8, void * dst_transposed_fp8);
	void fp8_quantize_triple_layout_weights_gpu(
		const float * src_kcrs,
		int filters,
		int channels,
		int kernel_h,
		int kernel_w,
		int kernel_pad,
		int filters_pad,
		const float * scale_gpu,
		void * dst_rowmajor_fp8,
		void * dst_transposed_fp8,
		void * dst_krsc_fp8);
	void fp8_colmajor_output_accumulate_rowmajor_gpu(
		const void * src_colmajor, int rows, int cols, bool src_bf16, float alpha, float * dst_rowmajor);
	void fp8_colmajor_output_to_nchw_delta_gpu(
		const void * src_colmajor,
		int batch,
		int channels,
		int height,
		int width,
		int kernel_h,
		int kernel_w,
		int pad_h,
		int pad_w,
		int stride_h,
		int stride_w,
		int dilation_h,
		int dilation_w,
		bool src_bf16,
		float * delta_nchw);
	void fp8_quantize_e5m2_record_amax_gpu(const float * src, size_t count, const float * scale_gpu, void * dst_fp8, float * amax_gpu);
	void fp8_quantize_e5m2_rowmajor_pad_cols_record_amax_gpu(const float * src, int rows, int cols, int cols_pad, const float * scale_gpu, void * dst_fp8, float * amax_gpu, size_t dst_ld = 0);
	void fp8_quantize_e5m2_rowmajor_pad_rows_record_amax_gpu(const float * src, int rows, int cols, int rows_pad, const float * scale_gpu, void * dst_fp8, float * amax_gpu);
	void fp8_quantize_e5m2_rowmajor_pad_rows_cols_record_amax_gpu(const float * src, int rows, int cols, int rows_pad, int cols_pad, const float * scale_gpu, void * dst_fp8, float * amax_gpu);
	void fp8_quantize_nchw_to_nhwc_gpu(const float * src, int batch, int channels, int height, int width, const float * scale_gpu, void * dst_fp8);
	void fp8_quantize_nchw_to_nhwc_record_amax_gpu(const float * src, int batch, int channels, int height, int width, const float * scale_gpu, void * dst_fp8, float * amax_gpu);
	/// Fused final ReLU plus E4M3 NHWC relay pack.  `src_dst` is updated in
	/// place so the normal FP32 output semantics remain intact for diagnostics
	/// and non-convolutional consumers.
	bool fp8_relu_quantize_nchw_to_nhwc_gpu(float * src_dst, int batch, int channels, int height, int width, const float * scale_gpu, void * dst_fp8, float * amax_gpu = nullptr);
	void fp8_quantize_weights_krsc_gpu(const float * src_kcrs, int filters, int channels, int kernel_h, int kernel_w, const float * scale_gpu, void * dst_fp8_krsc);
	void fp8_nhwc_output_to_nchw_gpu(const void * src, int batch, int channels, int height, int width, bool src_bf16, const float * bias, float * dst);
	/// number of floats to allocate for a device-side delayed-scaling state buffer (history + write index)
	size_t fp8_scale_state_floats();
	/// per-tensor delayed-scaling descriptor for the batched update below
	struct Fp8ScaleUpdate
	{
		float * amax_gpu = nullptr;
		float * state_gpu = nullptr;
		float format_max = 0.0f;
		int margin = 0;
		float * scale_gpu = nullptr;
	};
	/// up to 3 delayed-scaling updates in ONE kernel launch (one block per tensor); null amax entries are skipped
	void fp8_delayed_scale_update3_gpu(const Fp8ScaleUpdate & a, const Fp8ScaleUpdate & b, const Fp8ScaleUpdate & c);
	void fp8_clear_amax_gpu(float * amax_gpu);
	void fp8_accumulate_amax_gpu(const float * src, size_t count, float * amax_gpu);
	float fp8_pull_amax_gpu(float * amax_gpu);
}
