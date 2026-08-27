#include "option_list.hpp"
#ifdef DARKNET_HAS_FP8
#include "fp8_calibration.hpp"
#endif
#ifdef DARKNET_HAS_FP4
#include "fp4_calibration.hpp"
#endif
#include "darknet_internal.hpp"

#include <algorithm>
#include <cmath>

namespace
{
	static auto & cfg_and_state = Darknet::CfgAndState::get();
	static constexpr int k_wmhf_sub_count = 7;

#ifdef DARKNET_HAS_FP8
	static void load_fp8_calibration_sidecar(Darknet::Network * net, const char * filename)
	{
		TAT(TATPARMS);

		if (net == nullptr || filename == nullptr || !net->fp8_inference)
		{
			return;
		}

		const auto sidecar_path = Darknet::fp8_scales_sidecar_path(filename);
		Darknet::Fp8CalibrationTable table;
		if (!Darknet::fp8_read_calibration_scales(sidecar_path, table))
		{
			Darknet::display_warning_msg("FP8 requested but calibration sidecar \"" + sidecar_path.string() + "\" is missing or invalid; run `darknet detector calibrate` first. Disabling FP8.\n");
			net->fp8_inference = 0;
			return;
		}

		int conv_layers = 0;
		int loaded_layers = 0;
		for (int idx = 0; idx < net->n; ++idx)
		{
			Darknet::Layer & l = net->layers[idx];
			if (l.type != Darknet::ELayerType::CONVOLUTIONAL)
			{
				continue;
			}

			conv_layers += 1;
			const auto iter = table.find(idx);
			if (iter == table.end())
			{
				continue;
			}

			l.fp8_activation_amax_host = iter->second.amax;
			l.fp8_input_scale_host = iter->second.scale;
			l.fp8_scales_loaded = 1;
			loaded_layers += 1;
		}

		if (loaded_layers == 0)
		{
			Darknet::display_warning_msg("FP8 requested but no convolutional calibration scales matched this network; disabling FP8.\n");
			net->fp8_inference = 0;
			return;
		}
		if (loaded_layers < conv_layers)
		{
			Darknet::display_warning_msg("FP8 calibration sidecar \"" + sidecar_path.string() + "\" covers " + std::to_string(loaded_layers) + " of " + std::to_string(conv_layers) + " convolutional layers; uncovered layers will use the existing inference path.\n");
		}
	}
#endif

#ifdef DARKNET_HAS_FP4
	// Unlike FP8's sidecar, this one is not yet load-bearing for FP4 inference --
	// the cuDNN Frontend block-scale-quantize op computes its scales internally
	// per call with no exposed override (see calibrate_detector_fp4() in
	// detector.cpp). Loading it here just populates the diagnostic host fields;
	// a missing or partial sidecar is not an error and never disables FP4.
	static void load_fp4_calibration_sidecar(Darknet::Network * net, const char * filename)
	{
		TAT(TATPARMS);

		if (net == nullptr || filename == nullptr || !net->fp4_inference)
		{
			return;
		}

		const auto sidecar_path = Darknet::fp4_scales_sidecar_path(filename);
		Darknet::Fp4CalibrationTable table;
		if (!Darknet::fp4_read_calibration_scales(sidecar_path, table))
		{
			return;
		}

		for (int idx = 0; idx < net->n; ++idx)
		{
			Darknet::Layer & l = net->layers[idx];
			if (l.type != Darknet::ELayerType::CONVOLUTIONAL)
			{
				continue;
			}

			const auto iter = table.find(idx);
			if (iter == table.end())
			{
				continue;
			}

			l.fp4_activation_amax_host = iter->second.amax;
			l.fp4_input_scale_host = iter->second.scale;
			l.fp4_scales_loaded = 1;
		}
	}
#endif


	/// @returns the total number of bytes read
	static inline size_t xfread(void * dst, const size_t size, const size_t count, std::FILE * fp, const std::string & description = "")
	{
		TAT(TATPARMS);

		if (dst == nullptr)
		{
			darknet_fatal_error(DARKNET_LOC, "attempting to load %lu %s, but destination pointer is NULL", count, description.c_str());
		}
		const auto items_read = std::fread(dst, size, count, fp);
		if (items_read != count)
		{
			Darknet::display_warning_msg(
				"The .weights file does not match the .cfg file (not enough fields to read in the weights).\n"
				"Normally this means the .weights file was corrupted, or you've mixed up which .cfg file goes with which .weights file.\n"
				"Another common problem is if you edit your .names file or .cfg file and you forget to re-train your network.\n");

			darknet_fatal_error(DARKNET_LOC, "expected to read %lu fields, but only read %lu", count, items_read);
		}

		if (cfg_and_state.is_trace)
		{
			*cfg_and_state.output << "-> read " << count << " x " << (size * 8) << "-bit values (" << Darknet::size_to_IEC_string(size * count) << ")" << (description.empty() ? "" : " as " + description) << std::endl;
		}

		return size * count;
	}
}

void save_convolutional_weights(Darknet::Layer & l, FILE *fp);
void save_deconvolutional_weights(Darknet::Layer & l, FILE *fp);
void save_tucker_attention_weights(Darknet::Layer & l, FILE *fp);
void save_wmhf_weights(Darknet::Layer & l, FILE *fp);


Darknet::Network parse_network_cfg(const char * filename)
{
	TAT(TATPARMS);

	return parse_network_cfg_custom(filename, 0, 0);
}


Darknet::Network parse_network_cfg_custom(const char * filename, int batch, int time_steps)
{
	TAT(TATPARMS);

	if (filename == nullptr or filename[0] == '\0')
	{
		darknet_fatal_error(DARKNET_LOC, "expected a .cfg filename but got a NULL filename instead");
	}

	// V3 JAZZ:  we now use the new CfgFile class to load configuration

	Darknet::CfgFile cfg_file(filename);
	cfg_file.create_network(batch, time_steps);

	if (cfg_and_state.is_trace)
	{
		Darknet::dump(cfg_file);
	}

	return cfg_file.net;
}


void save_convolutional_weights_binary(Darknet::Layer & l, FILE *fp)
{
	TAT(TATPARMS);

#ifdef DARKNET_GPU
	if (cfg_and_state.gpu_index >= 0)
	{
		pull_convolutional_layer(l);
	}
#endif
	int size = (l.c/l.groups)*l.size*l.size;
	binarize_weights(l.weights, l.n, size, l.binary_weights);
	int i, j, k;
	fwrite(l.biases, sizeof(float), l.n, fp);
	if (l.batch_normalize)
	{
		fwrite(l.scales, sizeof(float), l.n, fp);
		fwrite(l.rolling_mean, sizeof(float), l.n, fp);
		fwrite(l.rolling_variance, sizeof(float), l.n, fp);
	}
	for (i = 0; i < l.n; ++i)
	{
		float mean = l.binary_weights[i*size];
		if (mean < 0)
		{
			mean = -mean;
		}
		fwrite(&mean, sizeof(float), 1, fp);
		for (j = 0; j < size/8; ++j)
		{
			int index = i*size + j*8;
			unsigned char c = 0;
			for (k = 0; k < 8; ++k)
			{
				if (j*8 + k >= size)
				{
					break;
				}
				if (l.binary_weights[index + k] > 0)
				{
					c = (c | 1<<k);
				}
			}
			fwrite(&c, sizeof(char), 1, fp);
		}
	}
}

void save_vit_weights(Darknet::Layer & l, FILE *fp)
{
	TAT(TATPARMS);

#ifdef DARKNET_GPU
	if (cfg_and_state.gpu_index >= 0)
	{
		pull_vit_layer(l);
	}
#endif

	const int patch_dim = l.vit_patch_size * l.vit_patch_size * l.c;
	fwrite(l.vit_patch_embed, sizeof(float), l.out_c * patch_dim, fp);
	fwrite(l.vit_patch_bias, sizeof(float), l.out_c, fp);
	fwrite(l.biases, sizeof(float), l.nbiases, fp);
	fwrite(l.vit_ln1_gamma, sizeof(float), l.out_c, fp);
	fwrite(l.vit_ln1_beta, sizeof(float), l.out_c, fp);
	fwrite(l.weights, sizeof(float), l.nweights, fp);
	fwrite(l.vit_wo, sizeof(float), l.out_c * l.out_c, fp);
	fwrite(l.vit_wo_bias, sizeof(float), l.out_c, fp);
	fwrite(l.vit_ln2_gamma, sizeof(float), l.out_c, fp);
	fwrite(l.vit_ln2_beta, sizeof(float), l.out_c, fp);
	fwrite(l.vit_ffn_w1, sizeof(float), l.out_c * l.vit_mlp_dim, fp);
	fwrite(l.vit_ffn_b1, sizeof(float), l.vit_mlp_dim, fp);
	fwrite(l.vit_ffn_w2, sizeof(float), l.out_c * l.vit_mlp_dim, fp);
	fwrite(l.vit_ffn_b2, sizeof(float), l.out_c, fp);
	fwrite(l.vit_pos_embed, sizeof(float), l.out_h * l.out_w * l.out_c, fp);
	fwrite(l.scales, sizeof(float), 12, fp);
}

void save_tucker_attention_weights(Darknet::Layer & l, FILE *fp)
{
	TAT(TATPARMS);

#ifdef DARKNET_GPU
	if (cfg_and_state.gpu_index >= 0)
	{
		cuda_pull_array(l.weights_gpu, l.weights, l.nweights);
		cuda_pull_array(l.biases_gpu, l.biases, l.nbiases);
	}
#endif

	fwrite(l.biases, sizeof(float), l.nbiases, fp);
	fwrite(l.weights, sizeof(float), l.nweights, fp);
}

void save_mambavision_weights(Darknet::Layer & l, FILE *fp)
{
	TAT(TATPARMS);

#ifdef DARKNET_GPU
	if (cfg_and_state.gpu_index >= 0)
	{
		pull_mambavision_layer(l);
	}
#endif

	const int D = l.n / 2;
	const int P = l.mv_dt_rank + 2 * l.mv_d_state;
	const int ffn_hidden = l.n * l.mv_ffn_ratio;

	fwrite(l.weights, sizeof(float), l.nweights, fp);
	fwrite(l.biases, sizeof(float), l.nbiases, fp);
	fwrite(l.mv_conv_x, sizeof(float), D * l.mv_conv_size, fp);
	fwrite(l.mv_conv_x_bias, sizeof(float), D, fp);
	fwrite(l.mv_conv_z, sizeof(float), D * l.mv_conv_size, fp);
	fwrite(l.mv_conv_z_bias, sizeof(float), D, fp);
	fwrite(l.mv_x_proj, sizeof(float), P * D, fp);
	fwrite(l.mv_dt_proj, sizeof(float), D * l.mv_dt_rank, fp);
	fwrite(l.mv_dt_bias, sizeof(float), D, fp);
	fwrite(l.mv_A_log, sizeof(float), D * l.mv_d_state, fp);
	fwrite(l.mv_D, sizeof(float), D, fp);
	fwrite(l.mv_out_proj, sizeof(float), l.n * l.n, fp);
	fwrite(l.mv_out_bias, sizeof(float), l.n, fp);
	if (l.c != l.n) fwrite(l.mv_res_proj, sizeof(float), l.n * l.c, fp);
	fwrite(l.mv_ln1_gamma, sizeof(float), l.c, fp);
	fwrite(l.mv_ln1_beta, sizeof(float), l.c, fp);
	fwrite(l.mv_ln2_gamma, sizeof(float), l.n, fp);
	fwrite(l.mv_ln2_beta, sizeof(float), l.n, fp);
	fwrite(l.mv_ffn_w1, sizeof(float), ffn_hidden * l.n, fp);
	fwrite(l.mv_ffn_b1, sizeof(float), ffn_hidden, fp);
	fwrite(l.mv_ffn_w2, sizeof(float), l.n * ffn_hidden, fp);
	fwrite(l.mv_ffn_b2, sizeof(float), l.n, fp);
}

void save_clifford_weights(Darknet::Layer & l, FILE *fp)
{
	TAT(TATPARMS);

#ifdef DARKNET_GPU
	if (cfg_and_state.gpu_index >= 0)
	{
		pull_clifford_layer(l);
	}
#endif

	fwrite(l.cli_ln_gamma, sizeof(float), l.c, fp);
	fwrite(l.cli_ln_beta, sizeof(float), l.c, fp);
	fwrite(l.cli_w_det, sizeof(float), l.c * l.c, fp);
	fwrite(l.cli_b_det, sizeof(float), l.c, fp);

	for (int i = 0; i < l.cli_num_dwconv; ++i)
	{
		save_convolutional_weights(l.cli_dwconv[i], fp);
	}

	fwrite(l.cli_w_proj, sizeof(float), l.c * l.cli_proj_in_dim, fp);
	fwrite(l.cli_b_proj, sizeof(float), l.c, fp);
	fwrite(l.cli_w_gate, sizeof(float), l.c * 2 * l.c, fp);
	fwrite(l.cli_b_gate, sizeof(float), l.c, fp);
	fwrite(l.cli_layer_scale, sizeof(float), l.c, fp);

	if (l.cli_gffn_mode != 0)
	{
		fwrite(l.cli_w_proj_g, sizeof(float), l.c * l.cli_proj_in_dim, fp);
		fwrite(l.cli_b_proj_g, sizeof(float), l.c, fp);
		fwrite(l.cli_w_gate_g, sizeof(float), l.c * 2 * l.c, fp);
		fwrite(l.cli_b_gate_g, sizeof(float), l.c, fp);
	}
}

void save_shortcut_weights(Darknet::Layer & l, FILE *fp)
{
	TAT(TATPARMS);

#ifdef DARKNET_GPU
	if (cfg_and_state.gpu_index >= 0)
	{
		pull_shortcut_layer(l);
		*cfg_and_state.output << std::endl << "pull_shortcut_layer" << std::endl;
	}
#endif
	for (int i = 0; i < l.nweights; ++i)
	{
		*cfg_and_state.output << " " << l.weights[i] << ", ";
	}
	*cfg_and_state.output << "l.nweights=" << l.nweights << std::endl << std::endl;

	int num = l.nweights;
	fwrite(l.weights, sizeof(float), num, fp);
}

void save_deform_conv_weights(Darknet::Layer & l, FILE *fp)
{
	TAT(TATPARMS);

	int offset_filters = 2 * l.size * l.size;
	int mask_filters = l.size * l.size;
	size_t offset_weights_size = (size_t)offset_filters * l.c * l.size * l.size;
	size_t mask_weights_size = (size_t)mask_filters * l.c * l.size * l.size;

#ifdef DARKNET_GPU
	if (cfg_and_state.gpu_index >= 0) pull_deform_conv_layer(l);
#endif

	fwrite(l.biases, sizeof(float), l.n, fp);
	if (l.batch_normalize)
	{
		fwrite(l.scales			, sizeof(float), l.n, fp);
		fwrite(l.rolling_mean		, sizeof(float), l.n, fp);
		fwrite(l.rolling_variance	, sizeof(float), l.n, fp);
	}
	fwrite(l.weights, sizeof(float), l.nweights, fp);
	fwrite(l.offset_weights, sizeof(float), offset_weights_size, fp);
	fwrite(l.offset_biases, sizeof(float), offset_filters, fp);
	if (l.use_mask)
	{
		fwrite(l.mask_weights, sizeof(float), mask_weights_size, fp);
		fwrite(l.mask_biases, sizeof(float), mask_filters, fp);
	}
}

void save_deconvolutional_weights(Darknet::Layer & l, FILE *fp)
{
	TAT(TATPARMS);

#ifdef DARKNET_GPU
	if (cfg_and_state.gpu_index >= 0)
	{
		pull_deconvolutional_layer(l);
	}
#endif

	fwrite(l.biases, sizeof(float), l.n, fp);
	fwrite(l.weights, sizeof(float), l.nweights, fp);
}

void save_dcnv4_weights(Darknet::Layer & l, FILE *fp)
{
	TAT(TATPARMS);

	int K = l.size * l.size;
	if (l.remove_center) K -= 1;
	int offset_filters_raw = l.groups * K * 3;
	int padded_offset_dim = ((offset_filters_raw + 7) / 8) * 8;
	size_t offset_weights_size = (size_t)padded_offset_dim * l.c * l.size * l.size;

#ifdef DARKNET_GPU
	if (cfg_and_state.gpu_index >= 0) pull_dcnv4_layer(l);
#endif

	fwrite(l.biases, sizeof(float), l.n, fp);
	if (l.batch_normalize)
	{
		fwrite(l.scales			, sizeof(float), l.n, fp);
		fwrite(l.rolling_mean		, sizeof(float), l.n, fp);
		fwrite(l.rolling_variance	, sizeof(float), l.n, fp);
	}
	fwrite(l.weights, sizeof(float), l.nweights, fp);
	fwrite(l.offset_weights, sizeof(float), offset_weights_size, fp);
	fwrite(l.offset_biases, sizeof(float), padded_offset_dim, fp);
}

void save_convolutional_weights(Darknet::Layer & l, FILE *fp)
{
	TAT(TATPARMS);

	if (l.binary)
	{
		//save_convolutional_weights_binary(l, fp);
		//return;
	}
#ifdef DARKNET_GPU
	if (cfg_and_state.gpu_index >= 0)
	{
		pull_convolutional_layer(l);
	}
#endif
	int num = l.nweights;
	fwrite(l.biases, sizeof(float), l.n, fp);
	if (l.batch_normalize)
	{
		fwrite(l.scales, sizeof(float), l.n, fp);
		fwrite(l.rolling_mean, sizeof(float), l.n, fp);
		fwrite(l.rolling_variance, sizeof(float), l.n, fp);
	}
	fwrite(l.weights, sizeof(float), num, fp);

	//if (l.adam){
	//    fwrite(l.m, sizeof(float), num, fp);
	//    fwrite(l.v, sizeof(float), num, fp);
	//}
}

void save_wmhf_weights(Darknet::Layer & l, FILE *fp)
{
	TAT(TATPARMS);

	for (int i = 0; i < k_wmhf_sub_count; ++i)
	{
		save_convolutional_weights(l.input_layer[i], fp);
	}

#ifdef DARKNET_GPU
	if (cfg_and_state.gpu_index >= 0 and l.weights_gpu)
	{
		cuda_pull_array(l.weights_gpu, l.weights, l.nweights);
	}
#endif
	fwrite(l.weights, sizeof(float), l.nweights, fp);
}

void save_convolutional_weights_ema(Darknet::Layer & l, FILE *fp)
{
	TAT(TATPARMS);

	if (l.binary)
	{
		//save_convolutional_weights_binary(l, fp);
		//return;
	}
#ifdef DARKNET_GPU
	if (cfg_and_state.gpu_index >= 0)
	{
		pull_convolutional_layer(l);
	}
#endif
	int num = l.nweights;
	fwrite(l.biases_ema, sizeof(float), l.n, fp);
	if (l.batch_normalize)
	{
		fwrite(l.scales_ema, sizeof(float), l.n, fp);
		fwrite(l.rolling_mean, sizeof(float), l.n, fp);
		fwrite(l.rolling_variance, sizeof(float), l.n, fp);
	}
	fwrite(l.weights_ema, sizeof(float), num, fp);
	//if (l.adam){
	//    fwrite(l.m, sizeof(float), num, fp);
	//    fwrite(l.v, sizeof(float), num, fp);
	//}
}

void save_graph_conv_weights(Darknet::Layer & l, FILE *fp)
{
	TAT(TATPARMS);

#ifdef DARKNET_GPU
	if (cfg_and_state.gpu_index >= 0)
	{
		pull_graph_conv_layer(l);
	}
#endif

	fwrite(l.biases, sizeof(float), l.n, fp);
	if (l.batch_normalize)
	{
		fwrite(l.scales, sizeof(float), l.n, fp);
		fwrite(l.rolling_mean, sizeof(float), l.n, fp);
		fwrite(l.rolling_variance, sizeof(float), l.n, fp);
	}
	fwrite(l.weights, sizeof(float), l.nweights, fp);

	if (l.graph_use_self)
	{
		fwrite(l.graph_self_weights, sizeof(float), l.n * l.graph_cpg, fp);
	}

	if (l.graph_edge_mode == 1)
	{
		fwrite(l.graph_edge_kernel, sizeof(float), l.groups * l.graph_k * (2 * l.graph_cpg), fp);
		fwrite(l.graph_edge_biases, sizeof(float), l.groups * l.graph_k, fp);
	}
}

void save_batchnorm_weights(Darknet::Layer & l, FILE *fp)
{
	TAT(TATPARMS);

#ifdef DARKNET_GPU
	if (cfg_and_state.gpu_index >= 0)
	{
		pull_batchnorm_layer(l);
	}
#endif
	fwrite(l.biases, sizeof(float), l.c, fp);
	fwrite(l.scales, sizeof(float), l.c, fp);
	fwrite(l.rolling_mean, sizeof(float), l.c, fp);
	fwrite(l.rolling_variance, sizeof(float), l.c, fp);
}

void save_connected_weights(Darknet::Layer & l, FILE *fp)
{
	TAT(TATPARMS);

#ifdef DARKNET_GPU
	if (cfg_and_state.gpu_index >= 0)
	{
		pull_connected_layer(l);
	}
#endif
	fwrite(l.biases, sizeof(float), l.outputs, fp);
	fwrite(l.weights, sizeof(float), l.outputs*l.inputs, fp);
	if (l.batch_normalize)
	{
		fwrite(l.scales, sizeof(float), l.outputs, fp);
		fwrite(l.rolling_mean, sizeof(float), l.outputs, fp);
		fwrite(l.rolling_variance, sizeof(float), l.outputs, fp);
	}
}


void save_weights_upto(const Darknet::Network & net, const char *filename, int cutoff, int save_ema)
{
	TAT(TATPARMS);

#ifdef DARKNET_GPU
	if (net.gpu_index >= 0)
	{
		cuda_set_device(net.gpu_index);
	}
#endif

	*cfg_and_state.output << "Saving weights to " << Darknet::in_colour(Darknet::EColour::kBrightMagenta, filename) << " ... ";

	FILE *fp = fopen(filename, "wb");
	if (not fp)
	{
		file_error(filename, DARKNET_LOC);
	}

	const int major = DARKNET_WEIGHTS_VERSION_MAJOR;
	const int minor = DARKNET_WEIGHTS_VERSION_MINOR;
	const int revision = DARKNET_WEIGHTS_VERSION_PATCH;

	fwrite(&major, sizeof(int), 1, fp);
	fwrite(&minor, sizeof(int), 1, fp);
	fwrite(&revision, sizeof(int), 1, fp);
	(*net.seen) = get_current_iteration(net) * net.batch * net.subdivisions; // remove this line, when you will save to weights-file both: seen & cur_iteration
	fwrite(net.seen, sizeof(uint64_t), 1, fp);

	for (int i = 0; i < net.n && i < cutoff; ++i)
	{
		Darknet::Layer & l = net.layers[i];
		if (l.type == Darknet::ELayerType::CONVOLUTIONAL && l.share_layer == NULL)
		{
			if (save_ema)
			{
				save_convolutional_weights_ema(l, fp);
			}
			else
			{
				save_convolutional_weights(l, fp);
			}
		}
		if (l.type == Darknet::ELayerType::EML_CONV)
		{
			if (save_ema)
			{
				save_convolutional_weights_ema(*(l.input_layer), fp);
				save_convolutional_weights_ema(*(l.self_layer), fp);
			}
			else
			{
				save_convolutional_weights(*(l.input_layer), fp);
				save_convolutional_weights(*(l.self_layer), fp);
			}
		}
		if (l.type == Darknet::ELayerType::DECONVOLUTIONAL)
		{
			save_deconvolutional_weights(l, fp);
		}
		if (l.type == Darknet::ELayerType::GRAPH_CONV && l.share_layer == NULL)
		{
			save_graph_conv_weights(l, fp);
		}
		if (l.type == Darknet::ELayerType::VIT)
		{
			save_vit_weights(l, fp);
		}
		if (l.type == Darknet::ELayerType::TUCKER_ATTENTION)
		{
			save_tucker_attention_weights(l, fp);
		}
		if (l.type == Darknet::ELayerType::MAMBAVISION)
		{
			save_mambavision_weights(l, fp);
		}
		if (l.type == Darknet::ELayerType::CLIFFORD)
		{
			save_clifford_weights(l, fp);
		}
		if (l.type == Darknet::ELayerType::WMHF)
		{
			save_wmhf_weights(l, fp);
		}
		if (l.type == Darknet::ELayerType::RECURSIVE_BLOCK)
		{
			for (int j = 0; j < l.rb_body_count; ++j)
			{
				Darknet::Layer & bl = l.rb_body[j];
				if (bl.type == Darknet::ELayerType::CONVOLUTIONAL)
				{
					save_convolutional_weights(bl, fp);
				}
				else if (bl.type == Darknet::ELayerType::CONNECTED)
				{
					save_connected_weights(bl, fp);
				}
				// Stateless body types (maxpool, avgpool, upsample, etc.) have no weights to save
			}
		}
		if (l.type == Darknet::ELayerType::DEFORM_CONV && l.share_layer == NULL)
		{
			save_deform_conv_weights(l, fp);
		}
		if (l.type == Darknet::ELayerType::DCNV4 && l.share_layer == NULL)
		{
			save_dcnv4_weights(l, fp);
		}
		if (l.type == Darknet::ELayerType::DETR_DECODER)
		{
			save_detr_decoder_weights(l, fp);
		}
		if (l.type == Darknet::ELayerType::SHORTCUT && l.nweights > 0)
		{
			save_shortcut_weights(l, fp);
		}
		if (l.type == Darknet::ELayerType::CONNECTED)
		{
			save_connected_weights(l, fp);
		}
		if (l.type == Darknet::ELayerType::RNN)
		{
			save_connected_weights(*(l.input_layer), fp);
			save_connected_weights(*(l.self_layer), fp);
			save_connected_weights(*(l.output_layer), fp);
		}
		if (l.type == Darknet::ELayerType::LSTM)
		{
			save_connected_weights(*(l.wf), fp);
			save_connected_weights(*(l.wi), fp);
			save_connected_weights(*(l.wg), fp);
			save_connected_weights(*(l.wo), fp);
			save_connected_weights(*(l.uf), fp);
			save_connected_weights(*(l.ui), fp);
			save_connected_weights(*(l.ug), fp);
			save_connected_weights(*(l.uo), fp);
		}
		if (l.type == Darknet::ELayerType::CRNN)
		{
			save_convolutional_weights(*(l.input_layer), fp);
			save_convolutional_weights(*(l.self_layer), fp);
			save_convolutional_weights(*(l.output_layer), fp);
		}
	}
	const long file_size = ftell(fp);
	fclose(fp);
	*cfg_and_state.output << Darknet::in_colour(Darknet::EColour::kBrightMagenta, std::to_string(file_size / 1024 / 1024) + " MB") << " (" << file_size << " bytes)" << std::endl;
}


void save_weights(const Darknet::Network & net, const char *filename)
{
	TAT(TATPARMS);

	save_weights_upto(net, filename, net.n, 0);
}


void transpose_matrix(float *a, int rows, int cols)
{
	TAT(TATPARMS);

	float* transpose = (float*)xcalloc(rows * cols, sizeof(float));

	for (int x = 0; x < rows; ++x)
	{
		for (int y = 0; y < cols; ++y)
		{
			transpose[y*rows + x] = a[x*cols + y];
		}
	}
	memcpy(a, transpose, rows*cols*sizeof(float));
	free(transpose);
}


size_t load_connected_weights(Darknet::Layer & l, FILE *fp, int transpose)
{
	TAT(TATPARMS);

	size_t bytes_read = 0;

	bytes_read += xfread(l.biases	, sizeof(float), l.outputs				, fp, "biases"	);
	bytes_read += xfread(l.weights	, sizeof(float), l.outputs * l.inputs	, fp, "weights"	);
	if (transpose)
	{
		transpose_matrix(l.weights, l.inputs, l.outputs);
	}
	if (l.batch_normalize && (not l.dontloadscales))
	{
		bytes_read += xfread(l.scales			, sizeof(float), l.outputs, fp, "scales"			);
		bytes_read += xfread(l.rolling_mean		, sizeof(float), l.outputs, fp, "rolling mean"		);
		bytes_read += xfread(l.rolling_variance	, sizeof(float), l.outputs, fp, "rolling variance"	);
	}
#ifdef DARKNET_GPU
	if (cfg_and_state.gpu_index >= 0)
	{
		push_connected_layer(l);
	}
#endif

	return bytes_read;
}

size_t load_convolutional_weights(Darknet::Layer & l, FILE *fp)
{
	TAT(TATPARMS);

	size_t bytes_read = 0;
	const int num = l.nweights;

	bytes_read += xfread(l.biases, sizeof(float), l.n, fp, "biases");
	if (l.batch_normalize && (not l.dontloadscales))
	{
		bytes_read += xfread(l.scales			, sizeof(float), l.n, fp, "scales"			);
		bytes_read += xfread(l.rolling_mean		, sizeof(float), l.n, fp, "rolling mean"	);
		bytes_read += xfread(l.rolling_variance	, sizeof(float), l.n, fp, "rolling variance");
	}
	bytes_read += xfread(l.weights, sizeof(float), num, fp, "weights");

	if (l.flipped)
	{
		transpose_matrix(l.weights, (l.c / l.groups) * l.size * l.size, l.n);
	}
	//if (l.binary) binarize_weights(l.weights, l.n, (l.c/l.groups)*l.size*l.size, l.weights);
#ifdef DARKNET_GPU
	if (cfg_and_state.gpu_index >= 0)
	{
		push_convolutional_layer(l);
	}
#endif

	return bytes_read;
}


size_t load_wmhf_weights(Darknet::Layer & l, FILE *fp)
{
	TAT(TATPARMS);

	size_t bytes_read = 0;
	for (int i = 0; i < k_wmhf_sub_count; ++i)
	{
		bytes_read += load_convolutional_weights(l.input_layer[i], fp);
	}

	bytes_read += xfread(l.weights, sizeof(float), l.nweights, fp, "wmhf scan weights");

#ifdef DARKNET_GPU
	if (cfg_and_state.gpu_index >= 0 and l.weights_gpu)
	{
		cuda_push_array(l.weights_gpu, l.weights, l.nweights);
	}
#endif

	return bytes_read;
}

size_t load_deconvolutional_weights(Darknet::Layer & l, FILE *fp)
{
	TAT(TATPARMS);

	size_t bytes_read = 0;

	bytes_read += xfread(l.biases, sizeof(float), l.n, fp, "biases");
	bytes_read += xfread(l.weights, sizeof(float), l.nweights, fp, "weights");

#ifdef DARKNET_GPU
	if (cfg_and_state.gpu_index >= 0)
	{
		push_deconvolutional_layer(l);
	}
#endif

	return bytes_read;
}

size_t load_deform_conv_weights(Darknet::Layer & l, FILE *fp)
{
	TAT(TATPARMS);

	size_t bytes_read = 0;
	int offset_filters = 2 * l.size * l.size;
	int mask_filters = l.size * l.size;
	size_t offset_weights_size = (size_t)offset_filters * l.c * l.size * l.size;
	size_t mask_weights_size = (size_t)mask_filters * l.c * l.size * l.size;

	bytes_read += xfread(l.biases, sizeof(float), l.n, fp, "biases");
	if (l.batch_normalize && (not l.dontloadscales))
	{
		bytes_read += xfread(l.scales			, sizeof(float), l.n, fp, "scales"			);
		bytes_read += xfread(l.rolling_mean		, sizeof(float), l.n, fp, "rolling mean"	);
		bytes_read += xfread(l.rolling_variance	, sizeof(float), l.n, fp, "rolling variance");
	}
	bytes_read += xfread(l.weights, sizeof(float), l.nweights, fp, "weights");
	
	bytes_read += xfread(l.offset_weights, sizeof(float), offset_weights_size, fp, "offset weights");
	bytes_read += xfread(l.offset_biases, sizeof(float), offset_filters, fp, "offset biases");
	
	if (l.use_mask)
	{
		bytes_read += xfread(l.mask_weights, sizeof(float), mask_weights_size, fp, "mask weights");
		bytes_read += xfread(l.mask_biases, sizeof(float), mask_filters, fp, "mask biases");
	}

#ifdef DARKNET_GPU
	if (cfg_and_state.gpu_index >= 0)
	{
		push_deform_conv_layer(l);
	}
#endif

	return bytes_read;
}

size_t load_dcnv4_weights(Darknet::Layer & l, FILE *fp)
{
	TAT(TATPARMS);

	size_t bytes_read = 0;
	int K = l.size * l.size;
	if (l.remove_center) K -= 1;
	int offset_filters_raw = l.groups * K * 3;
	int padded_offset_dim = ((offset_filters_raw + 7) / 8) * 8;
	size_t offset_weights_size = (size_t)padded_offset_dim * l.c * l.size * l.size;

	bytes_read += xfread(l.biases, sizeof(float), l.n, fp, "biases");
	if (l.batch_normalize && (not l.dontloadscales))
	{
		bytes_read += xfread(l.scales			, sizeof(float), l.n, fp, "scales"			);
		bytes_read += xfread(l.rolling_mean		, sizeof(float), l.n, fp, "rolling mean"	);
		bytes_read += xfread(l.rolling_variance	, sizeof(float), l.n, fp, "rolling variance");
	}
	bytes_read += xfread(l.weights, sizeof(float), l.nweights, fp, "weights");
	
	bytes_read += xfread(l.offset_weights, sizeof(float), offset_weights_size, fp, "offset weights");
	bytes_read += xfread(l.offset_biases, sizeof(float), padded_offset_dim, fp, "offset biases");

#ifdef DARKNET_GPU
	if (cfg_and_state.gpu_index >= 0)
	{
		push_dcnv4_layer(l);
	}
#endif

	return bytes_read;
}

size_t load_graph_conv_weights(Darknet::Layer & l, FILE *fp)
{
	TAT(TATPARMS);

	size_t bytes_read = 0;

	bytes_read += xfread(l.biases, sizeof(float), l.n, fp, "graph biases");
	if (l.batch_normalize && (not l.dontloadscales))
	{
		bytes_read += xfread(l.scales, sizeof(float), l.n, fp, "graph scales");
		bytes_read += xfread(l.rolling_mean, sizeof(float), l.n, fp, "graph rolling mean");
		bytes_read += xfread(l.rolling_variance, sizeof(float), l.n, fp, "graph rolling variance");
	}
	bytes_read += xfread(l.weights, sizeof(float), l.nweights, fp, "graph weights");

	if (l.graph_use_self)
	{
		bytes_read += xfread(l.graph_self_weights, sizeof(float), l.n * l.graph_cpg, fp, "graph self weights");
	}

	if (l.graph_edge_mode == 1)
	{
		bytes_read += xfread(l.graph_edge_kernel, sizeof(float), l.groups * l.graph_k * (2 * l.graph_cpg), fp, "graph edge kernel");
		bytes_read += xfread(l.graph_edge_biases, sizeof(float), l.groups * l.graph_k, fp, "graph edge biases");
	}

#ifdef DARKNET_GPU
	if (cfg_and_state.gpu_index >= 0)
	{
		push_graph_conv_layer(l);
	}
#endif

	return bytes_read;
}

size_t load_vit_weights(Darknet::Layer & l, FILE *fp, bool has_mhc_scales)
{
	TAT(TATPARMS);

	size_t bytes_read = 0;

	const int patch_dim = l.vit_patch_size * l.vit_patch_size * l.c;
	bytes_read += xfread(l.vit_patch_embed, sizeof(float), l.out_c * patch_dim, fp, "vit patch_embed");
	bytes_read += xfread(l.vit_patch_bias, sizeof(float), l.out_c, fp, "vit patch_bias");
	bytes_read += xfread(l.biases, sizeof(float), l.nbiases, fp, "vit biases");
	bytes_read += xfread(l.vit_ln1_gamma, sizeof(float), l.out_c, fp, "vit ln1_gamma");
	bytes_read += xfread(l.vit_ln1_beta, sizeof(float), l.out_c, fp, "vit ln1_beta");
	bytes_read += xfread(l.weights, sizeof(float), l.nweights, fp, "vit weights");
	bytes_read += xfread(l.vit_wo, sizeof(float), l.out_c * l.out_c, fp, "vit wo");
	bytes_read += xfread(l.vit_wo_bias, sizeof(float), l.out_c, fp, "vit wo_bias");
	bytes_read += xfread(l.vit_ln2_gamma, sizeof(float), l.out_c, fp, "vit ln2_gamma");
	bytes_read += xfread(l.vit_ln2_beta, sizeof(float), l.out_c, fp, "vit ln2_beta");
	bytes_read += xfread(l.vit_ffn_w1, sizeof(float), l.out_c * l.vit_mlp_dim, fp, "vit ffn_w1");
	bytes_read += xfread(l.vit_ffn_b1, sizeof(float), l.vit_mlp_dim, fp, "vit ffn_b1");
	bytes_read += xfread(l.vit_ffn_w2, sizeof(float), l.out_c * l.vit_mlp_dim, fp, "vit ffn_w2");
	bytes_read += xfread(l.vit_ffn_b2, sizeof(float), l.out_c, fp, "vit ffn_b2");
	bytes_read += xfread(l.vit_pos_embed, sizeof(float), l.out_h * l.out_w * l.out_c, fp, "vit pos_embed");
	if (has_mhc_scales)
	{
		bytes_read += xfread(l.scales, sizeof(float), 12, fp, "vit mhc scales");
	}
	else
	{
		Darknet::display_warning_msg(
			"Loading old [vit] weights without mHC residual mixer scales. "
			"The layer will keep its identity-start residual mixer defaults; re-save after training to preserve learned ViT branch mixing.\n");
	}

	if (l.vit_pos_embed_type == 1)
	{
		const int quarter = std::max(1, l.out_c / 4);
		for (int y = 0; y < l.out_h; ++y)
		{
			for (int x = 0; x < l.out_w; ++x)
			{
				float *row = l.vit_pos_embed + (y * l.out_w + x) * l.out_c;
				for (int c = 0; c < l.out_c; ++c)
				{
					const int band = c / 4;
					const float div = std::exp(-std::log(10000.0f) * static_cast<float>(band) / static_cast<float>(quarter));
					switch (c % 4)
					{
						case 0: row[c] = std::sin(static_cast<float>(y) * div); break;
						case 1: row[c] = std::cos(static_cast<float>(y) * div); break;
						case 2: row[c] = std::sin(static_cast<float>(x) * div); break;
						default: row[c] = std::cos(static_cast<float>(x) * div); break;
					}
				}
			}
		}
	}

#ifdef DARKNET_GPU
	if (cfg_and_state.gpu_index >= 0)
	{
		push_vit_layer(l);
	}
#endif

	return bytes_read;
}

size_t load_tucker_attention_weights(Darknet::Layer & l, FILE *fp)
{
	TAT(TATPARMS);

	size_t bytes_read = 0;
	bytes_read += xfread(l.biases, sizeof(float), l.nbiases, fp, "tucker_attention biases");
	bytes_read += xfread(l.weights, sizeof(float), l.nweights, fp, "tucker_attention weights");

#ifdef DARKNET_GPU
	if (cfg_and_state.gpu_index >= 0)
	{
		cuda_push_array(l.weights_gpu, l.weights, l.nweights);
		cuda_push_array(l.biases_gpu, l.biases, l.nbiases);
	}
#endif

	return bytes_read;
}

size_t load_mambavision_weights(Darknet::Layer & l, FILE *fp)
{
	TAT(TATPARMS);

	size_t bytes_read = 0;
	const int D = l.n / 2;
	const int P = l.mv_dt_rank + 2 * l.mv_d_state;
	const int ffn_hidden = l.n * l.mv_ffn_ratio;

	bytes_read += xfread(l.weights, sizeof(float), l.nweights, fp, "mambavision in_proj weights");
	bytes_read += xfread(l.biases, sizeof(float), l.nbiases, fp, "mambavision in_proj biases");
	bytes_read += xfread(l.mv_conv_x, sizeof(float), D * l.mv_conv_size, fp, "mambavision conv_x");
	bytes_read += xfread(l.mv_conv_x_bias, sizeof(float), D, fp, "mambavision conv_x_bias");
	bytes_read += xfread(l.mv_conv_z, sizeof(float), D * l.mv_conv_size, fp, "mambavision conv_z");
	bytes_read += xfread(l.mv_conv_z_bias, sizeof(float), D, fp, "mambavision conv_z_bias");
	bytes_read += xfread(l.mv_x_proj, sizeof(float), P * D, fp, "mambavision x_proj");
	bytes_read += xfread(l.mv_dt_proj, sizeof(float), D * l.mv_dt_rank, fp, "mambavision dt_proj");
	bytes_read += xfread(l.mv_dt_bias, sizeof(float), D, fp, "mambavision dt_bias");
	bytes_read += xfread(l.mv_A_log, sizeof(float), D * l.mv_d_state, fp, "mambavision A_log");
	bytes_read += xfread(l.mv_D, sizeof(float), D, fp, "mambavision D");
	bytes_read += xfread(l.mv_out_proj, sizeof(float), l.n * l.n, fp, "mambavision out_proj");
	bytes_read += xfread(l.mv_out_bias, sizeof(float), l.n, fp, "mambavision out_bias");
	if (l.c != l.n) bytes_read += xfread(l.mv_res_proj, sizeof(float), l.n * l.c, fp, "mambavision res_proj");
	bytes_read += xfread(l.mv_ln1_gamma, sizeof(float), l.c, fp, "mambavision ln1_gamma");
	bytes_read += xfread(l.mv_ln1_beta, sizeof(float), l.c, fp, "mambavision ln1_beta");
	bytes_read += xfread(l.mv_ln2_gamma, sizeof(float), l.n, fp, "mambavision ln2_gamma");
	bytes_read += xfread(l.mv_ln2_beta, sizeof(float), l.n, fp, "mambavision ln2_beta");
	bytes_read += xfread(l.mv_ffn_w1, sizeof(float), ffn_hidden * l.n, fp, "mambavision ffn_w1");
	bytes_read += xfread(l.mv_ffn_b1, sizeof(float), ffn_hidden, fp, "mambavision ffn_b1");
	bytes_read += xfread(l.mv_ffn_w2, sizeof(float), l.n * ffn_hidden, fp, "mambavision ffn_w2");
	bytes_read += xfread(l.mv_ffn_b2, sizeof(float), l.n, fp, "mambavision ffn_b2");

#ifdef DARKNET_GPU
	if (cfg_and_state.gpu_index >= 0)
	{
		push_mambavision_layer(l);
	}
#endif

	return bytes_read;
}

size_t load_clifford_weights(Darknet::Layer & l, FILE *fp)
{
	TAT(TATPARMS);

	size_t bytes_read = 0;

	bytes_read += xfread(l.cli_ln_gamma, sizeof(float), l.c, fp, "clifford ln_gamma");
	bytes_read += xfread(l.cli_ln_beta, sizeof(float), l.c, fp, "clifford ln_beta");
	bytes_read += xfread(l.cli_w_det, sizeof(float), l.c * l.c, fp, "clifford w_det");
	bytes_read += xfread(l.cli_b_det, sizeof(float), l.c, fp, "clifford b_det");

	for (int i = 0; i < l.cli_num_dwconv; ++i)
	{
		bytes_read += load_convolutional_weights(l.cli_dwconv[i], fp);
	}

	bytes_read += xfread(l.cli_w_proj, sizeof(float), l.c * l.cli_proj_in_dim, fp, "clifford w_proj");
	bytes_read += xfread(l.cli_b_proj, sizeof(float), l.c, fp, "clifford b_proj");
	bytes_read += xfread(l.cli_w_gate, sizeof(float), l.c * 2 * l.c, fp, "clifford w_gate");
	bytes_read += xfread(l.cli_b_gate, sizeof(float), l.c, fp, "clifford b_gate");
	bytes_read += xfread(l.cli_layer_scale, sizeof(float), l.c, fp, "clifford layer_scale");

	if (l.cli_gffn_mode != 0)
	{
		bytes_read += xfread(l.cli_w_proj_g, sizeof(float), l.c * l.cli_proj_in_dim, fp, "clifford w_proj_g");
		bytes_read += xfread(l.cli_b_proj_g, sizeof(float), l.c, fp, "clifford b_proj_g");
		bytes_read += xfread(l.cli_w_gate_g, sizeof(float), l.c * 2 * l.c, fp, "clifford w_gate_g");
		bytes_read += xfread(l.cli_b_gate_g, sizeof(float), l.c, fp, "clifford b_gate_g");
	}

#ifdef DARKNET_GPU
	if (cfg_and_state.gpu_index >= 0)
	{
		push_clifford_layer(l);
	}
#endif

	return bytes_read;
}

size_t load_shortcut_weights(Darknet::Layer & l, FILE *fp)
{
	TAT(TATPARMS);

	size_t bytes_read = 0;
	int num = l.nweights;

	bytes_read += xfread(l.weights, sizeof(float), num, fp, "weights");

#ifdef DARKNET_GPU
	if (cfg_and_state.gpu_index >= 0)
	{
		push_shortcut_layer(l);
	}
#endif

	return bytes_read;
}


void load_weights_upto(Darknet::Network * net, const char * filename, int cutoff)
{
	TAT(TATPARMS);

	if (net			== nullptr or
		filename	== nullptr or
		filename[0]	== '\0')
	{
		// nothing we can do
		Darknet::display_warning_msg("Cannot load weights due to NULL configuration or weights filename.\n");
		return;
	}

	if (net->details == nullptr)
	{
		darknet_fatal_error(DARKNET_LOC, "network structure was not created correctly (details pointer is null!?)");
	}
	net->details->weights_path = filename;

	if (cfg_and_state.is_verbose)
	{
		*cfg_and_state.output << "Loading weights from \"" << filename << "\""
			<< " (" << Darknet::size_to_IEC_string(std::filesystem::file_size(filename)) << ")"
			<< std::endl;
	}

#ifdef DARKNET_GPU
	if (net->gpu_index >= 0)
	{
		cuda_set_device(net->gpu_index);
	}
#endif

	FILE *fp = fopen(filename, "rb");
	if (not fp)
	{
		file_error(filename, DARKNET_LOC);
	}

	int major;
	int minor;
	int revision;
	xfread(&major	, sizeof(int), 1, fp, "major version number");
	xfread(&minor	, sizeof(int), 1, fp, "minor version number");
	xfread(&revision, sizeof(int), 1, fp, "patch version number");

	if ((major * 10 + minor) >= 2)
	{
		uint64_t iseen = 0;
		xfread(&iseen, sizeof(uint64_t), 1, fp, "images seen during training");
		*net->seen = iseen;
	}
	else
	{
		uint32_t iseen = 0;
		xfread(&iseen, sizeof(uint32_t), 1, fp, "images seen during training");
		*net->seen = iseen;
	}

	*net->cur_iteration = get_current_batch(*net);
	int transpose = (major > 1000) || (minor > 1000);

	size_t layers_with_weights = 0;
	size_t total_bytes_read = 0;

	for (int i = 0; i < net->n && i < cutoff; ++i)
	{
		Darknet::Layer & l = net->layers[i];
		if (l.dontload)
		{
			if (cfg_and_state.is_trace)
			{
				*cfg_and_state.output << "=> layer #" << i << " (" << Darknet::to_string(l.type) << "): dontload is set" << std::endl;
			}
			continue;
		}

		// also see Darknet::ONNXExport::populate_graph_initializers()
		switch(l.type)
		{
			case Darknet::ELayerType::CONVOLUTIONAL:
			{
				size_t bytes_read = 0;
				if (l.share_layer == NULL)
				{
					if (cfg_and_state.is_trace)
					{
						*cfg_and_state.output << "=> layer #" << i << " (" << Darknet::to_string(l.type) << "): loading convolutional weights" << std::endl;
					}
					layers_with_weights ++;
					bytes_read += load_convolutional_weights(l, fp);
				}
				if (cfg_and_state.is_trace)
				{
					*cfg_and_state.output << "-> layer #" << i << " (" << Darknet::to_string(l.type) << "): loaded " << Darknet::size_to_IEC_string(bytes_read) << std::endl;
				}
				total_bytes_read += bytes_read;
				break;
			}
			case Darknet::ELayerType::EML_CONV:
			{
				size_t bytes_read = 0;
				if (cfg_and_state.is_trace)
				{
					*cfg_and_state.output << "=> layer #" << i << " (" << Darknet::to_string(l.type) << "): loading EML convolutional weights" << std::endl;
				}
				layers_with_weights ++;
				bytes_read += load_convolutional_weights(*(l.input_layer), fp);
				bytes_read += load_convolutional_weights(*(l.self_layer), fp);
				if (cfg_and_state.is_trace)
				{
					*cfg_and_state.output << "-> layer #" << i << " (" << Darknet::to_string(l.type) << "): loaded " << Darknet::size_to_IEC_string(bytes_read) << std::endl;
				}
				total_bytes_read += bytes_read;
				break;
			}
			case Darknet::ELayerType::DECONVOLUTIONAL:
			{
				size_t bytes_read = 0;
				if (cfg_and_state.is_trace)
				{
					*cfg_and_state.output << "=> layer #" << i << " (" << Darknet::to_string(l.type) << "): loading deconvolutional weights" << std::endl;
				}
				layers_with_weights ++;
				bytes_read += load_deconvolutional_weights(l, fp);
				if (cfg_and_state.is_trace)
				{
					*cfg_and_state.output << "-> layer #" << i << " (" << Darknet::to_string(l.type) << "): loaded " << Darknet::size_to_IEC_string(bytes_read) << std::endl;
				}
				total_bytes_read += bytes_read;
				break;
			}
			case Darknet::ELayerType::GRAPH_CONV:
			{
				size_t bytes_read = 0;
				if (l.share_layer == NULL)
				{
					if (cfg_and_state.is_trace)
					{
						*cfg_and_state.output << "=> layer #" << i << " (" << Darknet::to_string(l.type) << "): loading graph_conv weights" << std::endl;
					}
					layers_with_weights ++;
					bytes_read += load_graph_conv_weights(l, fp);
				}
				if (cfg_and_state.is_trace)
				{
					*cfg_and_state.output << "-> layer #" << i << " (" << Darknet::to_string(l.type) << "): loaded " << Darknet::size_to_IEC_string(bytes_read) << std::endl;
				}
				total_bytes_read += bytes_read;
				break;
			}
			case Darknet::ELayerType::VIT:
			{
				size_t bytes_read = 0;
				if (cfg_and_state.is_trace)
				{
					*cfg_and_state.output << "=> layer #" << i << " (" << Darknet::to_string(l.type) << "): loading vit weights" << std::endl;
				}
				layers_with_weights ++;
				const bool has_mhc_scales = (major > DARKNET_WEIGHTS_VERSION_MAJOR) ||
					(major == DARKNET_WEIGHTS_VERSION_MAJOR && minor > DARKNET_WEIGHTS_VERSION_MINOR) ||
					(major == DARKNET_WEIGHTS_VERSION_MAJOR && minor == DARKNET_WEIGHTS_VERSION_MINOR &&
						revision >= DARKNET_WEIGHTS_VERSION_PATCH);
				bytes_read += load_vit_weights(l, fp, has_mhc_scales);
				if (cfg_and_state.is_trace)
				{
					*cfg_and_state.output << "-> layer #" << i << " (" << Darknet::to_string(l.type) << "): loaded " << Darknet::size_to_IEC_string(bytes_read) << std::endl;
				}
				total_bytes_read += bytes_read;
				break;
			}
			case Darknet::ELayerType::TUCKER_ATTENTION:
			{
				size_t bytes_read = 0;
				if (cfg_and_state.is_trace)
				{
					*cfg_and_state.output << "=> layer #" << i << " (" << Darknet::to_string(l.type) << "): loading tucker_attention weights" << std::endl;
				}
				layers_with_weights ++;
				bytes_read += load_tucker_attention_weights(l, fp);
				if (cfg_and_state.is_trace)
				{
					*cfg_and_state.output << "-> layer #" << i << " (" << Darknet::to_string(l.type) << "): loaded " << Darknet::size_to_IEC_string(bytes_read) << std::endl;
				}
				total_bytes_read += bytes_read;
				break;
			}
			case Darknet::ELayerType::MAMBAVISION:
			{
				size_t bytes_read = 0;
				if (cfg_and_state.is_trace)
				{
					*cfg_and_state.output << "=> layer #" << i << " (" << Darknet::to_string(l.type) << "): loading mambavision weights" << std::endl;
				}
				layers_with_weights ++;
				bytes_read += load_mambavision_weights(l, fp);
				if (cfg_and_state.is_trace)
				{
					*cfg_and_state.output << "-> layer #" << i << " (" << Darknet::to_string(l.type) << "): loaded " << Darknet::size_to_IEC_string(bytes_read) << std::endl;
				}
				total_bytes_read += bytes_read;
				break;
			}
			case Darknet::ELayerType::CLIFFORD:
			{
				size_t bytes_read = 0;
				if (cfg_and_state.is_trace)
				{
					*cfg_and_state.output << "=> layer #" << i << " (" << Darknet::to_string(l.type) << "): loading clifford weights" << std::endl;
				}
				layers_with_weights ++;
				bytes_read += load_clifford_weights(l, fp);
				if (cfg_and_state.is_trace)
				{
					*cfg_and_state.output << "-> layer #" << i << " (" << Darknet::to_string(l.type) << "): loaded " << Darknet::size_to_IEC_string(bytes_read) << std::endl;
				}
				total_bytes_read += bytes_read;
				break;
			}
			case Darknet::ELayerType::WMHF:
			{
				size_t bytes_read = 0;
				if (cfg_and_state.is_trace)
				{
					*cfg_and_state.output << "=> layer #" << i << " (" << Darknet::to_string(l.type) << "): loading wmhf weights" << std::endl;
				}
				layers_with_weights ++;
				bytes_read += load_wmhf_weights(l, fp);
				if (cfg_and_state.is_trace)
				{
					*cfg_and_state.output << "-> layer #" << i << " (" << Darknet::to_string(l.type) << "): loaded " << Darknet::size_to_IEC_string(bytes_read) << std::endl;
				}
				total_bytes_read += bytes_read;
				break;
			}
			case Darknet::ELayerType::RECURSIVE_BLOCK:
			{
				size_t bytes_read = 0;
				if (cfg_and_state.is_trace)
				{
					*cfg_and_state.output << "=> layer #" << i << " (" << Darknet::to_string(l.type) << "): loading recursive_block weights" << std::endl;
				}
				layers_with_weights ++;
				for (int j = 0; j < l.rb_body_count; ++j)
				{
					Darknet::Layer & bl = l.rb_body[j];
					if (bl.type == Darknet::ELayerType::CONVOLUTIONAL)
					{
						bytes_read += load_convolutional_weights(bl, fp);
					}
					else if (bl.type == Darknet::ELayerType::CONNECTED)
					{
						bytes_read += load_connected_weights(bl, fp, 0);
					}
					// Stateless body types have no weights to load
				}
				if (cfg_and_state.is_trace)
				{
					*cfg_and_state.output << "-> layer #" << i << " (" << Darknet::to_string(l.type) << "): loaded " << Darknet::size_to_IEC_string(bytes_read) << std::endl;
				}
				total_bytes_read += bytes_read;
				break;
			}
			case Darknet::ELayerType::DEFORM_CONV:
			{
				size_t bytes_read = 0;
				if (cfg_and_state.is_trace)
				{
					*cfg_and_state.output << "=> layer #" << i << " (" << Darknet::to_string(l.type) << "): loading deformable weights" << std::endl;
				}
				layers_with_weights ++;
				bytes_read += load_deform_conv_weights(l, fp);
				if (cfg_and_state.is_trace)
				{
					*cfg_and_state.output << "-> layer #" << i << " (" << Darknet::to_string(l.type) << "): loaded " << Darknet::size_to_IEC_string(bytes_read) << std::endl;
				}
				total_bytes_read += bytes_read;
				break;
			}
			case Darknet::ELayerType::DCNV4:
			{
				size_t bytes_read = 0;
				if (cfg_and_state.is_trace)
				{
					*cfg_and_state.output << "=> layer #" << i << " (" << Darknet::to_string(l.type) << "): loading dcnv4 weights" << std::endl;
				}
				layers_with_weights ++;
				bytes_read += load_dcnv4_weights(l, fp);
				if (cfg_and_state.is_trace)
				{
					*cfg_and_state.output << "-> layer #" << i << " (" << Darknet::to_string(l.type) << "): loaded " << Darknet::size_to_IEC_string(bytes_read) << std::endl;
				}
				total_bytes_read += bytes_read;
				break;
			}
			case Darknet::ELayerType::DETR_DECODER:
			{
				size_t bytes_read = 0;
				if (cfg_and_state.is_trace)
				{
					*cfg_and_state.output << "=> layer #" << i << " (" << Darknet::to_string(l.type) << "): loading detr_decoder weights" << std::endl;
				}
				layers_with_weights ++;
				bytes_read += load_detr_decoder_weights(l, fp);
				if (cfg_and_state.is_trace)
				{
					*cfg_and_state.output << "-> layer #" << i << " (" << Darknet::to_string(l.type) << "): loaded " << Darknet::size_to_IEC_string(bytes_read) << std::endl;
				}
				total_bytes_read += bytes_read;
				break;
			}
			case Darknet::ELayerType::SHORTCUT:
			{
				size_t bytes_read = 0;
				if (l.nweights > 0)
				{
					if (cfg_and_state.is_trace)
					{
						*cfg_and_state.output << "=> layer #" << i << " (" << Darknet::to_string(l.type) << "): loading shortcut weights" << std::endl;
					}
					layers_with_weights ++;
					bytes_read += load_shortcut_weights(l, fp);
				}
				if (cfg_and_state.is_trace)
				{
					*cfg_and_state.output << "-> layer #" << i << " (" << Darknet::to_string(l.type) << "): loaded " << Darknet::size_to_IEC_string(bytes_read) << std::endl;
				}
				total_bytes_read += bytes_read;
				break;
			}
			case Darknet::ELayerType::CONNECTED:
			{
				size_t bytes_read = 0;
				if (cfg_and_state.is_trace)
				{
					*cfg_and_state.output << "=> layer #" << i << " (" << Darknet::to_string(l.type) << "): loading connected weights" << std::endl;
				}
				layers_with_weights ++;
				bytes_read += load_connected_weights(l, fp, transpose);
				if (cfg_and_state.is_trace)
				{
					*cfg_and_state.output << "-> layer #" << i << " (" << Darknet::to_string(l.type) << "): loaded " << Darknet::size_to_IEC_string(bytes_read) << std::endl;
				}
				total_bytes_read += bytes_read;
				break;
			}
			case Darknet::ELayerType::CRNN:
			{
				size_t bytes_read = 0;
				if (cfg_and_state.is_trace)
				{
					*cfg_and_state.output << "=> layer #" << i << " (" << Darknet::to_string(l.type) << "): loading convolutional weights" << std::endl;
				}
				layers_with_weights ++;
				bytes_read += load_convolutional_weights(*(l.input_layer)	, fp);
				bytes_read += load_convolutional_weights(*(l.self_layer)	, fp);
				bytes_read += load_convolutional_weights(*(l.output_layer)	, fp);
				if (cfg_and_state.is_trace)
				{
					*cfg_and_state.output << "-> layer #" << i << " (" << Darknet::to_string(l.type) << "): loaded " << Darknet::size_to_IEC_string(bytes_read) << std::endl;
				}
				total_bytes_read += bytes_read;
				break;
			}
			case Darknet::ELayerType::RNN:
			{
				size_t bytes_read = 0;
				if (cfg_and_state.is_trace)
				{
					*cfg_and_state.output << "=> layer #" << i << " (" << Darknet::to_string(l.type) << "): loading connected weights" << std::endl;
				}
				layers_with_weights ++;
				bytes_read += load_connected_weights(*(l.input_layer)	, fp, transpose);
				bytes_read += load_connected_weights(*(l.self_layer)	, fp, transpose);
				bytes_read += load_connected_weights(*(l.output_layer)	, fp, transpose);
				if (cfg_and_state.is_trace)
				{
					*cfg_and_state.output << "-> layer #" << i << " (" << Darknet::to_string(l.type) << "): loaded " << Darknet::size_to_IEC_string(bytes_read) << std::endl;
				}
				total_bytes_read += bytes_read;
				break;
			}
			case Darknet::ELayerType::LSTM:
			{
				size_t bytes_read = 0;
				if (cfg_and_state.is_trace)
				{
					*cfg_and_state.output << "=> layer #" << i << " (" << Darknet::to_string(l.type) << "): loading connected weights" << std::endl;
				}
				layers_with_weights ++;
				bytes_read += load_connected_weights(*(l.wf), fp, transpose);
				bytes_read += load_connected_weights(*(l.wi), fp, transpose);
				bytes_read += load_connected_weights(*(l.wg), fp, transpose);
				bytes_read += load_connected_weights(*(l.wo), fp, transpose);
				bytes_read += load_connected_weights(*(l.uf), fp, transpose);
				bytes_read += load_connected_weights(*(l.ui), fp, transpose);
				bytes_read += load_connected_weights(*(l.ug), fp, transpose);
				bytes_read += load_connected_weights(*(l.uo), fp, transpose);
				if (cfg_and_state.is_trace)
				{
					*cfg_and_state.output << "-> layer #" << i << " (" << Darknet::to_string(l.type) << "): loaded " << Darknet::size_to_IEC_string(bytes_read) << std::endl;
				}
				total_bytes_read += bytes_read;
				break;
			}
			default:
			{
				// this layer does not have weights to load
				if (cfg_and_state.is_trace)
				{
					*cfg_and_state.output << "=> layer #" << i << " (" << Darknet::to_string(l.type) << "): no weights to load" << std::endl;
				}
				continue;
			}
		}

		if (feof(fp))
		{
			Darknet::display_warning_msg("premature end-of-file reached while loading weights " + std::string(filename) + "\n");
			break;
		}
	}

	// if everything has gone well, there will be zero bytes left to read at this point
	const auto position = ftell(fp);
	const auto filesize = std::filesystem::file_size(filename);
	if (position != filesize and cutoff >= net->n)
	{
		Darknet::display_warning_msg(
			"The .weights file does not match the .cfg file (weights file is larger than expected as described in the configuration).\n"
			"Normally this means the .weights file was corrupted, or you've mixed up which .cfg file goes with which .weights file.\n"
			"Another common problem is if you edit your .names file or .cfg file and you forget to re-train your network.\n");

		darknet_fatal_error(DARKNET_LOC, "failure detected while reading weights (fn=%s, layers=%d, pos=%lu, filesize=%lu)", filename, net->n, position, filesize);
	}

	if (cfg_and_state.is_verbose)
	{
		*cfg_and_state.output << "Loaded " << Darknet::size_to_IEC_string(total_bytes_read) << " in weights for " << layers_with_weights << " of " << net->n << " layers from " << filename << std::endl;
	}

	fclose(fp);

	return;
}


void load_weights(Darknet::Network * net, const char * filename)
{
	TAT(TATPARMS);

	load_weights_upto(net, filename, net->n);
#ifdef DARKNET_HAS_FP8
	load_fp8_calibration_sidecar(net, filename);
#endif
#ifdef DARKNET_HAS_FP4
	load_fp4_calibration_sidecar(net, filename);
#endif
}


// load network & force - set batch size
DarknetNetworkPtr load_network_custom(const char * cfg, const char * weights, int clear, int batch)
{
	TAT(TATPARMS);

	if (cfg_and_state.is_verbose)
	{
		*cfg_and_state.output << "Loading configuration from \"" << cfg << "\"" << std::endl;
	}

	Darknet::Network * net = (Darknet::Network*)xcalloc(1, sizeof(Darknet::Network));
	*net = parse_network_cfg_custom(cfg, batch, 1);
	load_weights(net, weights);
	fuse_conv_batchnorm(*net);

	/** @todo V3 Some code seems to also call this next function, and some not.  This was not originally called here, but
	 * I copied it from several other code locations.  Need to invetigate whether or not it should be here.  2024-08-03
	 */
	calculate_binary_weights(net);

	if (clear)
	{
		(*net->seen) = 0;
		(*net->cur_iteration) = 0;
	}

	return net;
}


// load network & get batch size from cfg-file
DarknetNetworkPtr load_network(const char * cfg, const char * weights, int clear)
{
	TAT(TATPARMS);

	if (cfg_and_state.is_verbose)
	{
		*cfg_and_state.output << "Loading configuration from \"" << cfg << "\"" << std::endl;
	}

	Darknet::Network* net = (Darknet::Network*)xcalloc(1, sizeof(Darknet::Network));
	*net = parse_network_cfg(cfg);
	load_weights(net, weights);

	/// @todo V3 why do we not call fuse_conv_batchnorm() here?

	if (clear)
	{
		(*net->seen) = 0;
		(*net->cur_iteration) = 0;
	}

	return net;
}


void Darknet::load_names(Darknet::NetworkPtr ptr, const std::filesystem::path & filename)
{
	TAT(TATPARMS);

	if (cfg_and_state.is_verbose)
	{
		*cfg_and_state.output << "Loading names from \"" << filename.string() << "\"" << std::endl;
	}

	if (not std::filesystem::exists(filename))
	{
		darknet_fatal_error(DARKNET_LOC, "expected a .names file but got a bad filename instead: \"%s\"", filename.string().c_str());
	}

	Darknet::Network * net = reinterpret_cast<Darknet::Network *>(ptr);
	if (net == nullptr)
	{
		darknet_fatal_error(DARKNET_LOC, "cannot set .names to \"%s\" when network pointer is null", filename.string().c_str());
	}

	if (net->details == nullptr)
	{
		darknet_fatal_error(DARKNET_LOC, "network structure was not created correctly (details pointer is null!?)");
	}

	net->details->names_path = filename;
	net->details->class_names.clear();

	std::string line;
	std::ifstream ifs(filename);
	while (std::getline(ifs, line))
	{
		// strip whitespace at the end of line, which should help us ignore \n and \r\n problems between Windows and Linux
		Darknet::trim(line);

		if (line.empty())
		{
			Darknet::display_error_msg("The .names file appears to contain a blank line.\n");
		}

		net->details->class_names.push_back(line);
	}

	if (net->layers[net->n - 1].classes != net->details->class_names.size())
	{
		darknet_fatal_error(DARKNET_LOC, "mismatch between number of classes in %s and the number of lines in %s", net->details->cfg_path.string().c_str(), net->details->names_path.string().c_str());
	}

	assign_default_class_colours(net);

	return;
}


void Darknet::assign_default_class_colours(Darknet::Network * net)
{
	TAT(TATPARMS);

	if (net == nullptr)
	{
		darknet_fatal_error(DARKNET_LOC, "cannot assign class colours when the network pointer is null");
	}

	if (net->details == nullptr)
	{
		darknet_fatal_error(DARKNET_LOC, "network structure was not created correctly (details pointer is null!?)");
	}

	if (net->n < 1)
	{
		darknet_fatal_error(DARKNET_LOC, "no network layers exist (was the network loaded?)");
	}

	const auto number_of_classes = net->layers[net->n - 1].classes;

	const bool class_names_are_blank = net->details->class_names.empty();
	if (class_names_are_blank)
	{
		// we may not have the network names available, so create fake labels we can use
		net->details->class_names.reserve(number_of_classes);
		for (int i = 0; i < number_of_classes; i++)
		{
			net->details->class_names.push_back("#" + std::to_string(i));
		}
	}

	if (number_of_classes != net->details->class_names.size())
	{
		darknet_fatal_error(DARKNET_LOC, "last layer indicates %d classes, but %ld classes exist", number_of_classes, net->details->class_names.size());
	}

	// assign a colour to each class

	net->details->text_colours	.clear();
	net->details->class_colours	.clear();
	net->details->text_colours	.reserve(number_of_classes);
	net->details->class_colours	.reserve(number_of_classes);

	for (int i = 0; i < number_of_classes; i++)
	{
		const int offset = i * 123457 % number_of_classes;
		const int r = std::min(255.0f, std::round(256.0f * Darknet::get_color(2, offset, number_of_classes)));
		const int g = std::min(255.0f, std::round(256.0f * Darknet::get_color(1, offset, number_of_classes)));
		const int b = std::min(255.0f, std::round(256.0f * Darknet::get_color(0, offset, number_of_classes)));

		const cv::Scalar background = CV_RGB(r, g, b);
		const cv::Scalar foreground = CV_RGB(0, 0, 0);

		net->details->class_colours.push_back(background);
		net->details->text_colours.push_back(foreground);

		if (cfg_and_state.is_verbose and not class_names_are_blank)
		{
			std::string colour_start	= "";
			std::string colour_end		= "";
			if (cfg_and_state.colour_is_enabled)
			{
				// use 32-bit colour to show the actual class colour
				colour_start	= "\033[0;38;2;" + std::to_string(r) + ";" + std::to_string(g) + ";" + std::to_string(b) + "m";
				colour_end		= "\033[0m";
			}

			*cfg_and_state.output
				<< "-> class #" << i << " will use colour "
				<< colour_start
				<< "0x"
				<< std::setw(2) << std::setfill('0') << std::hex << r
				<< std::setw(2) << std::setfill('0') << std::hex << g
				<< std::setw(2) << std::setfill('0') << std::hex << b
				<< std::setw(0) << std::setfill(' ') << std::dec
				<< colour_end
				<< " = "
				<< colour_start
				<< net->details->class_names.at(i)
				<< colour_end
				<< std::endl;
		}
	}

	return;
}
