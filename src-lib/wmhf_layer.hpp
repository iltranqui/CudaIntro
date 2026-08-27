#pragma once

#include "darknet_internal.hpp"

/**
 * Wavelet-Mamba High-frequency Fusion layer: [wmhf]
 *
 * One Darknet cfg layer that internally contains:
 *   - pre_pw:       1x1 projection C_in -> filters
 *   - local_dw3/5/7 depthwise local detail branches
 *   - local_mix_pw: 1x1 local branch mixer
 *   - Haar DWT/IWT global branch with 4-direction EMA scan
 *   - hf_gate_pw:   1x1 high-frequency gate
 *   - fuse_pw:      1x1 final branch fusion
 *
 * Parent shape is preserved: B x C x H x W -> B x filters x H x W.
 *
 * Required integration in darknet_internal.hpp:
 *   - Add ELayerType::WMHF, or use whatever enum name you prefer.
 * Required parser integration:
 *   - Map cfg block [wmhf] to make_wmhf_layer().
 * Required resize integration:
 *   - Call resize_wmhf_layer(&l, new_w, new_h) from resize_network().
 */

namespace Darknet
{
	Layer make_wmhf_layer(
		int batch,
		int h,
		int w,
		int c,
		int filters,
		float identity_ratio,
		float local_ratio,
		float freq_scale,
		int shortcut,
		ACTIVATION activation,
		int batch_normalize,
		int adam,
		int index,
		int train);

	void resize_wmhf_layer(Layer * l, int w, int h);
	void forward_wmhf_layer(Layer & l, NetworkState state);
	void backward_wmhf_layer(Layer & l, NetworkState state);
	void update_wmhf_layer(Layer & l, int batch, float learning_rate, float momentum, float decay);
	void free_wmhf_layer(Layer & l);

#ifdef DARKNET_GPU
	void forward_wmhf_layer_gpu(Layer & l, NetworkState state);
	void backward_wmhf_layer_gpu(Layer & l, NetworkState state);
	void update_wmhf_layer_gpu(Layer & l, int batch, float learning_rate, float momentum, float decay, float loss_scale);
#endif
}

#ifdef DARKNET_GPU
void wmhf_extract_channels_ongpu(int count, const float * input, float * output, int batch, int in_c, int out_c, int begin_c, int spatial);
void wmhf_insert_channels_ongpu(int count, const float * input, float * output, int batch, int out_c, int in_c, int begin_c, int spatial, float scale);
void wmhf_local_concat_ongpu(int count, const float * a, const float * b, const float * c, float * out, int channels, int spatial);
void wmhf_local_concat_backward_ongpu(int count, const float * cat_delta, float * da, float * db, float * dc, int channels, int spatial);
void wmhf_fuse_concat_ongpu(int count, const float * projected, const float * local, const float * global, float * out, int id_c, int local_c, int global_c, int spatial);
void wmhf_fuse_concat_backward_ongpu(int count, const float * cat_delta, float * projected_delta, float * local_delta, float * global_delta, int id_c, int local_c, int global_c, int spatial);

void wmhf_dwt_ongpu(int count, const float * input, float * ll, float * lh, float * hl, float * hh, int batch, int channels, int h, int w, int h2, int w2);
void wmhf_dwt_backward_ongpu(int count, const float * dll, const float * dlh, const float * dhl, const float * dhh, float * input_delta, int batch, int channels, int h, int w, int h2, int w2);
void wmhf_iwt_ongpu(int count, const float * ll, const float * lh, const float * hl, const float * hh, float * output, int batch, int channels, int h, int w, int h2, int w2);
void wmhf_iwt_backward_ongpu(int count, const float * output_delta, float * dll, float * dlh, float * dhl, float * dhh, int batch, int channels, int h, int w, int h2, int w2);

void wmhf_scan4_forward_ongpu(int sequences, const float * input, const float * weights, float * output, int batch, int channels, int h, int w);
void wmhf_scan4_backward_ongpu(int sequences, const float * input, const float * output_delta, const float * weights, float * weight_updates, float * input_delta, int batch, int channels, int h, int w);

void wmhf_hf_energy_upsample_ongpu(int count, const float * lh, const float * hl, const float * hh, float * e_up, int batch, int channels, int h, int w, int h2, int w2);
void wmhf_hf_energy_upsample_backward_ongpu(int count, const float * e_up_delta, const float * lh, const float * hl, const float * hh, float * dlh, float * dhl, float * dhh, int batch, int channels, int h, int w, int h2, int w2);

void wmhf_apply_gate_forward_ongpu(int count, const float * fuse, const float * gate, const float * projected, const float * shortcut, float * out, float freq_scale, int use_shortcut);
void wmhf_apply_gate_backward_ongpu(int count, const float * delta, const float * gate, const float * projected, float * fuse_delta, float * gate_delta, float * projected_delta, float * shortcut_delta, float freq_scale, int use_shortcut);
#endif
