#include "darknet_internal.hpp"
#include "yolo_layer_cuda.hpp"
#ifdef DARKNET_HAS_FP4
#include "fp4_gemm.hpp"
#endif
#ifdef DARKNET_HAS_FP8
#include "fp8_conv.hpp"
#include "fp8_gemm.hpp"
#endif

namespace
{
	void static inline free_and_clear(uint32_t* & ptr)
	{
		TAT(TATPARMS);

		if (ptr)
		{
			free(ptr);
			ptr = nullptr;
		}

		return;
	}

	void static inline free_and_clear(float* & ptr)
	{
		TAT(TATPARMS);

		if (ptr)
		{
			free(ptr);
			ptr = nullptr;
		}

		return;
	}

	void static inline free_and_clear(float** & array)
	{
		TAT(TATPARMS);

		if (array)
		{
			/** @todo Isn't this an array?  Should the array be freed?
			 *
			free_and_clear(*array);
			 */
			free(array);
			array = nullptr;
		}

		return;
	}

	void static inline free_and_clear(int* & ptr)
	{
		TAT(TATPARMS);

		if (ptr)
		{
			free(ptr);
			ptr = nullptr;
		}

		return;
	}

	void static inline free_and_clear(char* & ptr)
	{
		TAT(TATPARMS);

		if (ptr)
		{
			free(ptr);
			ptr = nullptr;
		}

		return;
	}

	void static inline free_sublayer(Darknet::Layer* & l)
	{
		TAT(TATPARMS);

		if (l)
		{
			free_layer(*l);
			free(l);
			l = nullptr;
		}

		return;
	}

	void static inline free_sublayer_array(Darknet::Layer* & layers, const int count)
	{
		TAT(TATPARMS);

		if (layers)
		{
			for (int i = 0; i < count; ++i)
			{
				free_layer(layers[i]);
			}
			free(layers);
			layers = nullptr;
		}

		return;
	}

	#ifdef DARKNET_GPU
	void static inline cuda_free_and_clear(float* & ptr)
	{
		TAT(TATPARMS);

		if (ptr)
		{
			cuda_free(ptr);
			ptr = nullptr;
		}

		return;
	}

	void static inline cuda_free_and_clear(int* & ptr)
	{
		TAT(TATPARMS);

		if (ptr)
		{
			cudaFree(ptr);
			ptr = nullptr;
		}

		return;
	}

	void static inline cuda_free_and_clear(char* & ptr)
	{
		TAT(TATPARMS);

		if (ptr)
		{
			cudaFree(ptr);
			ptr = nullptr;
		}

		return;
	}
	#endif
}


void free_layer(Darknet::Layer & l)
{
	TAT(TATPARMS);

	free_layer_custom(l, 0);
}


void free_layer_custom(Darknet::Layer & l, int keep_cudnn_desc)
{
	TAT(TATPARMS);

	if (l.share_layer != nullptr)
	{
		return;	// don't free shared layers
	}

#ifdef DARKNET_GPU_CUDA
	if (l.yolo_training_gpu_context)
	{
		Darknet::release_yolo_training_cuda(l);
	}
#endif

	if (l.antialiasing)
	{
		free_sublayer(l.input_layer);
	}

	if (l.type == Darknet::ELayerType::CRNN)
	{
		free_sublayer(l.input_layer);
		free_sublayer(l.self_layer);
		free_sublayer(l.output_layer);
		l.output		= nullptr;
		l.delta			= nullptr;
#ifdef DARKNET_GPU
		l.output_gpu	= nullptr;
		l.delta_gpu		= nullptr;
#endif // DARKNET_GPU
	}

	if (l.type == Darknet::ELayerType::EML_CONV)
	{
		free_sublayer(l.input_layer);
		free_sublayer(l.self_layer);
	}

	if (l.type == Darknet::ELayerType::MAMBAVISION)
	{
		free_sublayer(l.mv_in_proj_layer);
		free_sublayer(l.mv_conv_x_layer);
		free_sublayer(l.mv_conv_z_layer);
		free_sublayer(l.mv_x_proj_layer);
		free_sublayer(l.mv_dt_proj_layer);
		free_sublayer(l.mv_out_proj_layer);
		free_sublayer(l.mv_res_proj_layer);
		free_sublayer(l.mv_ffn1_layer);
		free_sublayer(l.mv_ffn2_layer);
	}

	if (l.type == Darknet::ELayerType::WMHF)
	{
		Darknet::free_wmhf_layer(l);
		return;
	}

	if (l.type == Darknet::ELayerType::DROPOUT)
	{
		if (l.rand)						free_and_clear(l.rand);
#ifdef DARKNET_GPU
		if (l.rand_gpu)					cuda_free_and_clear(l.rand_gpu);
		if (l.drop_blocks_scale)		cuda_free_host(l.drop_blocks_scale);
		l.drop_blocks_scale = nullptr;
		if (l.drop_blocks_scale_gpu)	cuda_free_and_clear(l.drop_blocks_scale_gpu);
#endif
		return;
	}

	if (l.mask)							free_and_clear(l.mask);
	if (l.classes_multipliers)			free_and_clear(l.classes_multipliers);
	if (l.cweights)						free_and_clear(l.cweights);
	if (l.indexes)						free_and_clear(l.indexes);
	if (l.input_layers)					free_and_clear(l.input_layers);
	if (l.input_sizes)					free_and_clear(l.input_sizes);
	if (l.layers_output)				free_and_clear(l.layers_output);
	if (l.layers_delta)					free_and_clear(l.layers_delta);
	if (l.map)							free_and_clear(l.map);
	if (l.rand)							free_and_clear(l.rand);
	if (l.cost)							free_and_clear(l.cost);
	if (l.labels && !l.detection)		free_and_clear(l.labels);
	if (l.class_ids && !l.detection)	free_and_clear(l.class_ids);
	if (l.cos_sim)						free_and_clear(l.cos_sim);
	if (l.exp_cos_sim)					free_and_clear(l.exp_cos_sim);
	if (l.p_constrastive)				free_and_clear(l.p_constrastive);
	if (l.embedding_output)				free_and_clear(l.embedding_output);
	if (l.state)						free_and_clear(l.state);
	if (l.prev_state)					free_and_clear(l.prev_state);
	if (l.forgot_state)					free_and_clear(l.forgot_state);
	if (l.forgot_delta)					free_and_clear(l.forgot_delta);
	if (l.state_delta)					free_and_clear(l.state_delta);
	if (l.concat)						free_and_clear(l.concat);
	if (l.concat_delta)					free_and_clear(l.concat_delta);
	if (l.binary_weights)				free_and_clear(l.binary_weights);
	if (l.biases)						free_and_clear(l.biases);
	if (l.bias_updates)					free_and_clear(l.bias_updates);
	if (l.scales)						free_and_clear(l.scales);
	if (l.scale_updates)				free_and_clear(l.scale_updates);
	if (l.biases_ema)					free_and_clear(l.biases_ema);
	if (l.scales_ema)					free_and_clear(l.scales_ema);
	if (l.weights_ema)					free_and_clear(l.weights_ema);
	if (l.weights)						free_and_clear(l.weights);
	if (l.weight_updates)				free_and_clear(l.weight_updates);
	if (l.graph_self_weights)			free_and_clear(l.graph_self_weights);
	if (l.graph_self_weight_updates)	free_and_clear(l.graph_self_weight_updates);
	if (l.graph_edge_kernel)			free_and_clear(l.graph_edge_kernel);
	if (l.graph_edge_kernel_updates)	free_and_clear(l.graph_edge_kernel_updates);
	if (l.graph_edge_biases)			free_and_clear(l.graph_edge_biases);
	if (l.graph_edge_bias_updates)		free_and_clear(l.graph_edge_bias_updates);
	if (l.graph_ref)					free_and_clear(l.graph_ref);
	if (l.graph_agg)					free_and_clear(l.graph_agg);
	if (l.graph_alpha)					free_and_clear(l.graph_alpha);
	if (l.graph_valid)					free_and_clear(l.graph_valid);
	if (l.align_bit_weights)			free_and_clear(l.align_bit_weights);
	if (l.mean_arr)						free_and_clear(l.mean_arr);

	if (l.tf_wo)						free_and_clear(l.tf_wo);
	if (l.tf_wo_updates)				free_and_clear(l.tf_wo_updates);
	if (l.tf_wo_bias)					free_and_clear(l.tf_wo_bias);
	if (l.tf_wo_bias_updates)			free_and_clear(l.tf_wo_bias_updates);
	if (l.tf_ln1_gamma)				free_and_clear(l.tf_ln1_gamma);
	if (l.tf_ln1_gamma_updates)		free_and_clear(l.tf_ln1_gamma_updates);
	if (l.tf_ln1_beta)				free_and_clear(l.tf_ln1_beta);
	if (l.tf_ln1_beta_updates)		free_and_clear(l.tf_ln1_beta_updates);
	if (l.tf_ln2_gamma)				free_and_clear(l.tf_ln2_gamma);
	if (l.tf_ln2_gamma_updates)		free_and_clear(l.tf_ln2_gamma_updates);
	if (l.tf_ln2_beta)				free_and_clear(l.tf_ln2_beta);
	if (l.tf_ln2_beta_updates)		free_and_clear(l.tf_ln2_beta_updates);
	if (l.tf_ffn_w1)				free_and_clear(l.tf_ffn_w1);
	if (l.tf_ffn_w1_updates)		free_and_clear(l.tf_ffn_w1_updates);
	if (l.tf_ffn_b1)				free_and_clear(l.tf_ffn_b1);
	if (l.tf_ffn_b1_updates)		free_and_clear(l.tf_ffn_b1_updates);
	if (l.tf_ffn_w2)				free_and_clear(l.tf_ffn_w2);
	if (l.tf_ffn_w2_updates)		free_and_clear(l.tf_ffn_w2_updates);
	if (l.tf_ffn_b2)				free_and_clear(l.tf_ffn_b2);
	if (l.tf_ffn_b2_updates)		free_and_clear(l.tf_ffn_b2_updates);
	if (l.tf_rel_pos_bias)			free_and_clear(l.tf_rel_pos_bias);
	if (l.tf_rel_pos_bias_updates)	free_and_clear(l.tf_rel_pos_bias_updates);
	if (l.tf_rel_pos_index)			free_and_clear(l.tf_rel_pos_index);
	if (l.tf_res_proj)				free_and_clear(l.tf_res_proj);
	if (l.tf_res_proj_updates)		free_and_clear(l.tf_res_proj_updates);
	if (l.tf_qkv_out)				free_and_clear(l.tf_qkv_out);
	if (l.tf_attn_scores)			free_and_clear(l.tf_attn_scores);
	if (l.tf_attn_out)				free_and_clear(l.tf_attn_out);
	if (l.tf_ffn_hidden)			free_and_clear(l.tf_ffn_hidden);
	if (l.tf_ln1_mean)				free_and_clear(l.tf_ln1_mean);
	if (l.tf_ln1_var)				free_and_clear(l.tf_ln1_var);
	if (l.tf_ln2_mean)				free_and_clear(l.tf_ln2_mean);
	if (l.tf_ln2_var)				free_and_clear(l.tf_ln2_var);
	if (l.tf_ln1_xhat)				free_and_clear(l.tf_ln1_xhat);
	if (l.tf_ln2_xhat)				free_and_clear(l.tf_ln2_xhat);
	if (l.tf_pre_res1)				free_and_clear(l.tf_pre_res1);
	if (l.tf_pre_res2)				free_and_clear(l.tf_pre_res2);
	if (l.tf_windowed_input)		free_and_clear(l.tf_windowed_input);
	if (l.tf_attn_mask)				free_and_clear(l.tf_attn_mask);
	if (l.tf_workspace)				free_and_clear(l.tf_workspace);

	if (l.vit_patch_embed)				free_and_clear(l.vit_patch_embed);
	if (l.vit_patch_embed_updates)		free_and_clear(l.vit_patch_embed_updates);
	if (l.vit_patch_bias)				free_and_clear(l.vit_patch_bias);
	if (l.vit_patch_bias_updates)		free_and_clear(l.vit_patch_bias_updates);
	if (l.vit_wo)						free_and_clear(l.vit_wo);
	if (l.vit_wo_updates)				free_and_clear(l.vit_wo_updates);
	if (l.vit_wo_bias)					free_and_clear(l.vit_wo_bias);
	if (l.vit_wo_bias_updates)			free_and_clear(l.vit_wo_bias_updates);
	if (l.vit_ln1_gamma)				free_and_clear(l.vit_ln1_gamma);
	if (l.vit_ln1_gamma_updates)		free_and_clear(l.vit_ln1_gamma_updates);
	if (l.vit_ln1_beta)					free_and_clear(l.vit_ln1_beta);
	if (l.vit_ln1_beta_updates)			free_and_clear(l.vit_ln1_beta_updates);
	if (l.vit_ln2_gamma)				free_and_clear(l.vit_ln2_gamma);
	if (l.vit_ln2_gamma_updates)		free_and_clear(l.vit_ln2_gamma_updates);
	if (l.vit_ln2_beta)					free_and_clear(l.vit_ln2_beta);
	if (l.vit_ln2_beta_updates)			free_and_clear(l.vit_ln2_beta_updates);
	if (l.vit_ffn_w1)					free_and_clear(l.vit_ffn_w1);
	if (l.vit_ffn_w1_updates)			free_and_clear(l.vit_ffn_w1_updates);
	if (l.vit_ffn_b1)					free_and_clear(l.vit_ffn_b1);
	if (l.vit_ffn_b1_updates)			free_and_clear(l.vit_ffn_b1_updates);
	if (l.vit_ffn_w2)					free_and_clear(l.vit_ffn_w2);
	if (l.vit_ffn_w2_updates)			free_and_clear(l.vit_ffn_w2_updates);
	if (l.vit_ffn_b2)					free_and_clear(l.vit_ffn_b2);
	if (l.vit_ffn_b2_updates)			free_and_clear(l.vit_ffn_b2_updates);
	if (l.vit_pos_embed)				free_and_clear(l.vit_pos_embed);
	if (l.vit_pos_embed_updates)		free_and_clear(l.vit_pos_embed_updates);
	if (l.vit_qkv_out)					free_and_clear(l.vit_qkv_out);
	if (l.vit_attn_scores)				free_and_clear(l.vit_attn_scores);
	if (l.vit_attn_out)					free_and_clear(l.vit_attn_out);
	if (l.vit_ffn_hidden)				free_and_clear(l.vit_ffn_hidden);
	if (l.vit_ln1_mean)					free_and_clear(l.vit_ln1_mean);
	if (l.vit_ln1_var)					free_and_clear(l.vit_ln1_var);
	if (l.vit_ln2_mean)					free_and_clear(l.vit_ln2_mean);
	if (l.vit_ln2_var)					free_and_clear(l.vit_ln2_var);
	if (l.vit_ln1_xhat)					free_and_clear(l.vit_ln1_xhat);
	if (l.vit_ln2_xhat)					free_and_clear(l.vit_ln2_xhat);
	if (l.vit_pre_res1)					free_and_clear(l.vit_pre_res1);
	if (l.vit_pre_res2)					free_and_clear(l.vit_pre_res2);

	if (l.tucker_q_latent)				free_and_clear(l.tucker_q_latent);
	if (l.tucker_k_latent)				free_and_clear(l.tucker_k_latent);
	if (l.tucker_v_latent)				free_and_clear(l.tucker_v_latent);
	if (l.tucker_q)						free_and_clear(l.tucker_q);
	if (l.tucker_k)						free_and_clear(l.tucker_k);
	if (l.tucker_v)						free_and_clear(l.tucker_v);
	if (l.tucker_scores)				free_and_clear(l.tucker_scores);
	if (l.tucker_context)				free_and_clear(l.tucker_context);
	if (l.tucker_windowed_input)		free_and_clear(l.tucker_windowed_input);
	if (l.tucker_gpu_input_cpu)			free_and_clear(l.tucker_gpu_input_cpu);

	if (l.mv_conv_x)					free_and_clear(l.mv_conv_x);
	if (l.mv_conv_x_updates)			free_and_clear(l.mv_conv_x_updates);
	if (l.mv_conv_x_bias)				free_and_clear(l.mv_conv_x_bias);
	if (l.mv_conv_x_bias_updates)		free_and_clear(l.mv_conv_x_bias_updates);
	if (l.mv_conv_z)					free_and_clear(l.mv_conv_z);
	if (l.mv_conv_z_updates)			free_and_clear(l.mv_conv_z_updates);
	if (l.mv_conv_z_bias)				free_and_clear(l.mv_conv_z_bias);
	if (l.mv_conv_z_bias_updates)		free_and_clear(l.mv_conv_z_bias_updates);
	if (l.mv_x_proj)					free_and_clear(l.mv_x_proj);
	if (l.mv_x_proj_updates)			free_and_clear(l.mv_x_proj_updates);
	if (l.mv_dt_proj)					free_and_clear(l.mv_dt_proj);
	if (l.mv_dt_proj_updates)			free_and_clear(l.mv_dt_proj_updates);
	if (l.mv_dt_bias)					free_and_clear(l.mv_dt_bias);
	if (l.mv_dt_bias_updates)			free_and_clear(l.mv_dt_bias_updates);
	if (l.mv_A_log)						free_and_clear(l.mv_A_log);
	if (l.mv_A_log_updates)				free_and_clear(l.mv_A_log_updates);
	if (l.mv_D)							free_and_clear(l.mv_D);
	if (l.mv_D_updates)					free_and_clear(l.mv_D_updates);
	if (l.mv_out_proj)					free_and_clear(l.mv_out_proj);
	if (l.mv_out_proj_updates)			free_and_clear(l.mv_out_proj_updates);
	if (l.mv_out_bias)					free_and_clear(l.mv_out_bias);
	if (l.mv_out_bias_updates)			free_and_clear(l.mv_out_bias_updates);
	if (l.mv_res_proj)					free_and_clear(l.mv_res_proj);
	if (l.mv_res_proj_updates)			free_and_clear(l.mv_res_proj_updates);
	if (l.mv_ln1_gamma)					free_and_clear(l.mv_ln1_gamma);
	if (l.mv_ln1_gamma_updates)			free_and_clear(l.mv_ln1_gamma_updates);
	if (l.mv_ln1_beta)					free_and_clear(l.mv_ln1_beta);
	if (l.mv_ln1_beta_updates)			free_and_clear(l.mv_ln1_beta_updates);
	if (l.mv_ln2_gamma)					free_and_clear(l.mv_ln2_gamma);
	if (l.mv_ln2_gamma_updates)			free_and_clear(l.mv_ln2_gamma_updates);
	if (l.mv_ln2_beta)					free_and_clear(l.mv_ln2_beta);
	if (l.mv_ln2_beta_updates)			free_and_clear(l.mv_ln2_beta_updates);
	if (l.mv_ffn_w1)					free_and_clear(l.mv_ffn_w1);
	if (l.mv_ffn_w1_updates)			free_and_clear(l.mv_ffn_w1_updates);
	if (l.mv_ffn_b1)					free_and_clear(l.mv_ffn_b1);
	if (l.mv_ffn_b1_updates)			free_and_clear(l.mv_ffn_b1_updates);
	if (l.mv_ffn_w2)					free_and_clear(l.mv_ffn_w2);
	if (l.mv_ffn_w2_updates)			free_and_clear(l.mv_ffn_w2_updates);
	if (l.mv_ffn_b2)					free_and_clear(l.mv_ffn_b2);
	if (l.mv_ffn_b2_updates)			free_and_clear(l.mv_ffn_b2_updates);
	if (l.mv_tokens)					free_and_clear(l.mv_tokens);
	if (l.mv_ln1_out)					free_and_clear(l.mv_ln1_out);
	if (l.mv_ln1_mean)					free_and_clear(l.mv_ln1_mean);
	if (l.mv_ln1_var)					free_and_clear(l.mv_ln1_var);
	if (l.mv_ln1_xhat)					free_and_clear(l.mv_ln1_xhat);
	if (l.mv_in_proj_out)				free_and_clear(l.mv_in_proj_out);
	if (l.mv_x_conv_pre)				free_and_clear(l.mv_x_conv_pre);
	if (l.mv_x_conv)					free_and_clear(l.mv_x_conv);
	if (l.mv_z_conv_pre)				free_and_clear(l.mv_z_conv_pre);
	if (l.mv_z_conv)					free_and_clear(l.mv_z_conv);
	if (l.mv_x_proj_out)				free_and_clear(l.mv_x_proj_out);
	if (l.mv_dt_pre)					free_and_clear(l.mv_dt_pre);
	if (l.mv_dt)						free_and_clear(l.mv_dt);
	if (l.mv_scan_state)				free_and_clear(l.mv_scan_state);
	if (l.mv_scan_out)					free_and_clear(l.mv_scan_out);
	if (l.mv_mixer_cat)					free_and_clear(l.mv_mixer_cat);
	if (l.mv_mixer_out)					free_and_clear(l.mv_mixer_out);
	if (l.mv_pre_res2)					free_and_clear(l.mv_pre_res2);
	if (l.mv_ln2_out)					free_and_clear(l.mv_ln2_out);
	if (l.mv_ln2_mean)					free_and_clear(l.mv_ln2_mean);
	if (l.mv_ln2_var)					free_and_clear(l.mv_ln2_var);
	if (l.mv_ln2_xhat)					free_and_clear(l.mv_ln2_xhat);
	if (l.mv_ffn_hidden)				free_and_clear(l.mv_ffn_hidden);
	if (l.mv_gpu_input_cpu)				free_and_clear(l.mv_gpu_input_cpu);

	if (l.cli_shifts)					free_and_clear(l.cli_shifts);
	if (l.cli_shifts_inner)				free_and_clear(l.cli_shifts_inner);
	if (l.cli_dwconv)					free_sublayer_array(l.cli_dwconv, l.cli_num_dwconv);
	if (l.cli_w_det)					free_and_clear(l.cli_w_det);
	if (l.cli_w_det_updates)			free_and_clear(l.cli_w_det_updates);
	if (l.cli_b_det)					free_and_clear(l.cli_b_det);
	if (l.cli_b_det_updates)			free_and_clear(l.cli_b_det_updates);
	if (l.cli_w_proj)					free_and_clear(l.cli_w_proj);
	if (l.cli_w_proj_updates)			free_and_clear(l.cli_w_proj_updates);
	if (l.cli_b_proj)					free_and_clear(l.cli_b_proj);
	if (l.cli_b_proj_updates)			free_and_clear(l.cli_b_proj_updates);
	if (l.cli_w_gate)					free_and_clear(l.cli_w_gate);
	if (l.cli_w_gate_updates)			free_and_clear(l.cli_w_gate_updates);
	if (l.cli_b_gate)					free_and_clear(l.cli_b_gate);
	if (l.cli_b_gate_updates)			free_and_clear(l.cli_b_gate_updates);
	if (l.cli_ln_gamma)					free_and_clear(l.cli_ln_gamma);
	if (l.cli_ln_gamma_updates)			free_and_clear(l.cli_ln_gamma_updates);
	if (l.cli_ln_beta)					free_and_clear(l.cli_ln_beta);
	if (l.cli_ln_beta_updates)			free_and_clear(l.cli_ln_beta_updates);
	if (l.cli_layer_scale)				free_and_clear(l.cli_layer_scale);
	if (l.cli_layer_scale_updates)		free_and_clear(l.cli_layer_scale_updates);
	if (l.cli_w_proj_g)					free_and_clear(l.cli_w_proj_g);
	if (l.cli_w_proj_g_updates)			free_and_clear(l.cli_w_proj_g_updates);
	if (l.cli_b_proj_g)					free_and_clear(l.cli_b_proj_g);
	if (l.cli_b_proj_g_updates)			free_and_clear(l.cli_b_proj_g_updates);
	if (l.cli_w_gate_g)					free_and_clear(l.cli_w_gate_g);
	if (l.cli_w_gate_g_updates)			free_and_clear(l.cli_w_gate_g_updates);
	if (l.cli_b_gate_g)					free_and_clear(l.cli_b_gate_g);
	if (l.cli_b_gate_g_updates)			free_and_clear(l.cli_b_gate_g_updates);
	if (l.cli_ln_out)					free_and_clear(l.cli_ln_out);
	if (l.cli_ln_mean)					free_and_clear(l.cli_ln_mean);
	if (l.cli_ln_var)					free_and_clear(l.cli_ln_var);
	if (l.cli_ln_xhat)					free_and_clear(l.cli_ln_xhat);
	if (l.cli_z_det)					free_and_clear(l.cli_z_det);
	if (l.cli_z_ctx)					free_and_clear(l.cli_z_ctx);
	if (l.cli_z_ctx_pre_diff)			free_and_clear(l.cli_z_ctx_pre_diff);
	if (l.cli_g_raw)					free_and_clear(l.cli_g_raw);
		if (l.cli_g_feat)					free_and_clear(l.cli_g_feat);
		if (l.cli_gate_alpha)				free_and_clear(l.cli_gate_alpha);
		if (l.cli_gate_pre_sigmoid)			free_and_clear(l.cli_gate_pre_sigmoid);
		if (l.cli_vb_feat)					free_and_clear(l.cli_vb_feat);
		if (l.cli_hmix)						free_and_clear(l.cli_hmix);
	if (l.cli_drop_mask)				free_and_clear(l.cli_drop_mask);
	if (l.cli_global_ctx)				free_and_clear(l.cli_global_ctx);
	if (l.cli_g_raw_g)					free_and_clear(l.cli_g_raw_g);
	if (l.cli_g_feat_g)					free_and_clear(l.cli_g_feat_g);
	if (l.cli_gate_alpha_g)				free_and_clear(l.cli_gate_alpha_g);
	if (l.cli_gate_pre_sigmoid_g)		free_and_clear(l.cli_gate_pre_sigmoid_g);

#ifdef DARKNET_GPU
	if (l.delta && l.delta_pinned)
	{
		CHECK_CUDA(cudaFreeHost(l.delta));
		l.delta = nullptr;
	}

	if (l.output && l.output_pinned)
	{
		CHECK_CUDA(cudaFreeHost(l.output));
		l.output = nullptr;
	}
#endif  // DARKNET_GPU

	if (l.delta)						free_and_clear(l.delta);
	if (l.output)						free_and_clear(l.output);
	if (l.activation_input)				free_and_clear(l.activation_input);
	if (l.squared)						free_and_clear(l.squared);
	if (l.norms)						free_and_clear(l.norms);
	if (l.spatial_mean)					free_and_clear(l.spatial_mean);
	if (l.mean)							free_and_clear(l.mean);
	if (l.variance)						free_and_clear(l.variance);
	if (l.mean_delta)					free_and_clear(l.mean_delta);
	if (l.variance_delta)				free_and_clear(l.variance_delta);
	if (l.rolling_mean)					free_and_clear(l.rolling_mean);
	if (l.rolling_variance)				free_and_clear(l.rolling_variance);
	if (l.x)							free_and_clear(l.x);
	if (l.x_norm)						free_and_clear(l.x_norm);
	if (l.m)							free_and_clear(l.m);
	if (l.v)							free_and_clear(l.v);
	if (l.z_cpu)						free_and_clear(l.z_cpu);
	if (l.r_cpu)						free_and_clear(l.r_cpu);
	if (l.binary_input)					free_and_clear(l.binary_input);
	if (l.bin_re_packed_input)			free_and_clear(l.bin_re_packed_input);
	if (l.t_bit_input)					free_and_clear(l.t_bit_input);
	if (l.loss)							free_and_clear(l.loss);
	if (l.offset_weights)				free_and_clear(l.offset_weights);
	if (l.offset_weight_updates)			free_and_clear(l.offset_weight_updates);
	if (l.offset_biases)					free_and_clear(l.offset_biases);
	if (l.offset_bias_updates)			free_and_clear(l.offset_bias_updates);
	if (l.offsets)						free_and_clear(l.offsets);
	if (l.offset_deltas)					free_and_clear(l.offset_deltas);
	if (l.mask_weights)					free_and_clear(l.mask_weights);
	if (l.mask_weight_updates)			free_and_clear(l.mask_weight_updates);
	if (l.mask_biases)					free_and_clear(l.mask_biases);
	if (l.mask_bias_updates)				free_and_clear(l.mask_bias_updates);
	if (l.masks)						free_and_clear(l.masks);
	if (l.mask_deltas)					free_and_clear(l.mask_deltas);

	// CONV-LSTM
	if (l.f_cpu)						free_and_clear(l.f_cpu);
	if (l.i_cpu)						free_and_clear(l.i_cpu);
	if (l.g_cpu)						free_and_clear(l.g_cpu);
	if (l.o_cpu)						free_and_clear(l.o_cpu);
	if (l.c_cpu)						free_and_clear(l.c_cpu);
	if (l.h_cpu)						free_and_clear(l.h_cpu);
	if (l.temp_cpu)						free_and_clear(l.temp_cpu);
	if (l.temp2_cpu)					free_and_clear(l.temp2_cpu);
	if (l.temp3_cpu)					free_and_clear(l.temp3_cpu);
	if (l.dc_cpu)						free_and_clear(l.dc_cpu);
	if (l.dh_cpu)						free_and_clear(l.dh_cpu);
	if (l.prev_state_cpu)				free_and_clear(l.prev_state_cpu);
	if (l.prev_cell_cpu)				free_and_clear(l.prev_cell_cpu);
	if (l.stored_c_cpu)					free_and_clear(l.stored_c_cpu);
	if (l.stored_h_cpu)					free_and_clear(l.stored_h_cpu);
	if (l.cell_cpu)						free_and_clear(l.cell_cpu);

#ifdef DARKNET_GPU
	if (l.indexes_gpu)					cuda_free((float *)l.indexes_gpu);
	if (l.contrast_p_gpu)				cuda_free((float *)l.contrast_p_gpu);
	l.indexes_gpu = nullptr;
	l.contrast_p_gpu = nullptr;
	if (l.z_gpu)						cuda_free_and_clear(l.z_gpu);
	if (l.r_gpu)						cuda_free_and_clear(l.r_gpu);
	if (l.m_gpu)						cuda_free_and_clear(l.m_gpu);
	if (l.v_gpu)						cuda_free_and_clear(l.v_gpu);
	if (l.forgot_state_gpu)				cuda_free_and_clear(l.forgot_state_gpu);
	if (l.forgot_delta_gpu)				cuda_free_and_clear(l.forgot_delta_gpu);
	if (l.state_gpu)					cuda_free_and_clear(l.state_gpu);
	if (l.state_delta_gpu)				cuda_free_and_clear(l.state_delta_gpu);
	if (l.gate_gpu)						cuda_free_and_clear(l.gate_gpu);
	if (l.gate_delta_gpu)				cuda_free_and_clear(l.gate_delta_gpu);
	if (l.save_gpu)						cuda_free_and_clear(l.save_gpu);
	if (l.save_delta_gpu)				cuda_free_and_clear(l.save_delta_gpu);
	if (l.concat_gpu)					cuda_free_and_clear(l.concat_gpu);
	if (l.concat_delta_gpu)				cuda_free_and_clear(l.concat_delta_gpu);
	if (l.binary_input_gpu)				cuda_free_and_clear(l.binary_input_gpu);
	if (l.binary_weights_gpu)			cuda_free_and_clear(l.binary_weights_gpu);
	if (l.mean_gpu)						cuda_free_and_clear(l.mean_gpu);
	if (l.variance_gpu)					cuda_free_and_clear(l.variance_gpu);
	if (l.m_cbn_avg_gpu)				cuda_free_and_clear(l.m_cbn_avg_gpu);
	if (l.v_cbn_avg_gpu)				cuda_free_and_clear(l.v_cbn_avg_gpu);
	if (l.rolling_mean_gpu)				cuda_free_and_clear(l.rolling_mean_gpu);
	if (l.rolling_variance_gpu)			cuda_free_and_clear(l.rolling_variance_gpu);
	if (l.variance_delta_gpu)			cuda_free_and_clear(l.variance_delta_gpu);
	if (l.mean_delta_gpu)				cuda_free_and_clear(l.mean_delta_gpu);
	if (l.x_norm_gpu)					cuda_free_and_clear(l.x_norm_gpu);

	// assisted excitation
	if (l.gt_gpu)						cuda_free_and_clear(l.gt_gpu);
	if (l.a_avg_gpu)					cuda_free_and_clear(l.a_avg_gpu);

	if (l.align_bit_weights_gpu)		cuda_free((float *)l.align_bit_weights_gpu);
	l.align_bit_weights_gpu = nullptr;

	if (l.mean_arr_gpu)					cuda_free_and_clear(l.mean_arr_gpu);
	if (l.align_workspace_gpu)			cuda_free_and_clear(l.align_workspace_gpu);
	if (l.transposed_align_workspace_gpu) cuda_free_and_clear(l.transposed_align_workspace_gpu);

	if (l.tf_wo_gpu)					cuda_free_and_clear(l.tf_wo_gpu);
	if (l.tf_wo_updates_gpu)			cuda_free_and_clear(l.tf_wo_updates_gpu);
	if (l.tf_wo_bias_gpu)				cuda_free_and_clear(l.tf_wo_bias_gpu);
	if (l.tf_wo_bias_updates_gpu)		cuda_free_and_clear(l.tf_wo_bias_updates_gpu);
	if (l.tf_ln1_gamma_gpu)			cuda_free_and_clear(l.tf_ln1_gamma_gpu);
	if (l.tf_ln1_gamma_updates_gpu)	cuda_free_and_clear(l.tf_ln1_gamma_updates_gpu);
	if (l.tf_ln1_beta_gpu)				cuda_free_and_clear(l.tf_ln1_beta_gpu);
	if (l.tf_ln1_beta_updates_gpu)		cuda_free_and_clear(l.tf_ln1_beta_updates_gpu);
	if (l.tf_ln2_gamma_gpu)			cuda_free_and_clear(l.tf_ln2_gamma_gpu);
	if (l.tf_ln2_gamma_updates_gpu)	cuda_free_and_clear(l.tf_ln2_gamma_updates_gpu);
	if (l.tf_ln2_beta_gpu)				cuda_free_and_clear(l.tf_ln2_beta_gpu);
	if (l.tf_ln2_beta_updates_gpu)		cuda_free_and_clear(l.tf_ln2_beta_updates_gpu);
	if (l.tf_ffn_w1_gpu)				cuda_free_and_clear(l.tf_ffn_w1_gpu);
	if (l.tf_ffn_w1_updates_gpu)		cuda_free_and_clear(l.tf_ffn_w1_updates_gpu);
	if (l.tf_ffn_b1_gpu)				cuda_free_and_clear(l.tf_ffn_b1_gpu);
	if (l.tf_ffn_b1_updates_gpu)		cuda_free_and_clear(l.tf_ffn_b1_updates_gpu);
	if (l.tf_ffn_w2_gpu)				cuda_free_and_clear(l.tf_ffn_w2_gpu);
	if (l.tf_ffn_w2_updates_gpu)		cuda_free_and_clear(l.tf_ffn_w2_updates_gpu);
	if (l.tf_ffn_b2_gpu)				cuda_free_and_clear(l.tf_ffn_b2_gpu);
	if (l.tf_ffn_b2_updates_gpu)		cuda_free_and_clear(l.tf_ffn_b2_updates_gpu);
	if (l.tf_rel_pos_bias_gpu)			cuda_free_and_clear(l.tf_rel_pos_bias_gpu);
	if (l.tf_rel_pos_bias_updates_gpu)	cuda_free_and_clear(l.tf_rel_pos_bias_updates_gpu);
	if (l.tf_rel_pos_index_gpu)			cuda_free((float *)l.tf_rel_pos_index_gpu);
	l.tf_rel_pos_index_gpu = nullptr;
	if (l.tf_res_proj_gpu)				cuda_free_and_clear(l.tf_res_proj_gpu);
	if (l.tf_res_proj_updates_gpu)		cuda_free_and_clear(l.tf_res_proj_updates_gpu);
	if (l.tf_qkv_out_gpu)				cuda_free_and_clear(l.tf_qkv_out_gpu);
	if (l.tf_attn_scores_gpu)			cuda_free_and_clear(l.tf_attn_scores_gpu);
	if (l.tf_attn_out_gpu)				cuda_free_and_clear(l.tf_attn_out_gpu);
	if (l.tf_ffn_hidden_gpu)			cuda_free_and_clear(l.tf_ffn_hidden_gpu);
	if (l.tf_ln1_mean_gpu)				cuda_free_and_clear(l.tf_ln1_mean_gpu);
	if (l.tf_ln1_var_gpu)				cuda_free_and_clear(l.tf_ln1_var_gpu);
	if (l.tf_ln2_mean_gpu)				cuda_free_and_clear(l.tf_ln2_mean_gpu);
	if (l.tf_ln2_var_gpu)				cuda_free_and_clear(l.tf_ln2_var_gpu);
	if (l.tf_ln1_xhat_gpu)				cuda_free_and_clear(l.tf_ln1_xhat_gpu);
	if (l.tf_ln2_xhat_gpu)				cuda_free_and_clear(l.tf_ln2_xhat_gpu);
	if (l.tf_pre_res1_gpu)				cuda_free_and_clear(l.tf_pre_res1_gpu);
	if (l.tf_pre_res2_gpu)				cuda_free_and_clear(l.tf_pre_res2_gpu);
	if (l.tf_windowed_input_gpu)		cuda_free_and_clear(l.tf_windowed_input_gpu);
	if (l.tf_attn_mask_gpu)				cuda_free_and_clear(l.tf_attn_mask_gpu);
	if (l.tf_gpu_workspace)				cuda_free_and_clear(l.tf_gpu_workspace);

	if (l.vit_patch_embed_gpu)			cuda_free_and_clear(l.vit_patch_embed_gpu);
	if (l.vit_patch_embed_updates_gpu)	cuda_free_and_clear(l.vit_patch_embed_updates_gpu);
	if (l.vit_patch_bias_gpu)			cuda_free_and_clear(l.vit_patch_bias_gpu);
	if (l.vit_patch_bias_updates_gpu)	cuda_free_and_clear(l.vit_patch_bias_updates_gpu);
	if (l.vit_wo_gpu)					cuda_free_and_clear(l.vit_wo_gpu);
	if (l.vit_wo_updates_gpu)			cuda_free_and_clear(l.vit_wo_updates_gpu);
	if (l.vit_wo_bias_gpu)				cuda_free_and_clear(l.vit_wo_bias_gpu);
	if (l.vit_wo_bias_updates_gpu)		cuda_free_and_clear(l.vit_wo_bias_updates_gpu);
	if (l.vit_ln1_gamma_gpu)			cuda_free_and_clear(l.vit_ln1_gamma_gpu);
	if (l.vit_ln1_gamma_updates_gpu)	cuda_free_and_clear(l.vit_ln1_gamma_updates_gpu);
	if (l.vit_ln1_beta_gpu)				cuda_free_and_clear(l.vit_ln1_beta_gpu);
	if (l.vit_ln1_beta_updates_gpu)		cuda_free_and_clear(l.vit_ln1_beta_updates_gpu);
	if (l.vit_ln2_gamma_gpu)			cuda_free_and_clear(l.vit_ln2_gamma_gpu);
	if (l.vit_ln2_gamma_updates_gpu)	cuda_free_and_clear(l.vit_ln2_gamma_updates_gpu);
	if (l.vit_ln2_beta_gpu)				cuda_free_and_clear(l.vit_ln2_beta_gpu);
	if (l.vit_ln2_beta_updates_gpu)		cuda_free_and_clear(l.vit_ln2_beta_updates_gpu);
	if (l.vit_ffn_w1_gpu)				cuda_free_and_clear(l.vit_ffn_w1_gpu);
	if (l.vit_ffn_w1_updates_gpu)		cuda_free_and_clear(l.vit_ffn_w1_updates_gpu);
	if (l.vit_ffn_b1_gpu)				cuda_free_and_clear(l.vit_ffn_b1_gpu);
	if (l.vit_ffn_b1_updates_gpu)		cuda_free_and_clear(l.vit_ffn_b1_updates_gpu);
	if (l.vit_ffn_w2_gpu)				cuda_free_and_clear(l.vit_ffn_w2_gpu);
	if (l.vit_ffn_w2_updates_gpu)		cuda_free_and_clear(l.vit_ffn_w2_updates_gpu);
	if (l.vit_ffn_b2_gpu)				cuda_free_and_clear(l.vit_ffn_b2_gpu);
	if (l.vit_ffn_b2_updates_gpu)		cuda_free_and_clear(l.vit_ffn_b2_updates_gpu);
	if (l.vit_pos_embed_gpu)			cuda_free_and_clear(l.vit_pos_embed_gpu);
	if (l.vit_pos_embed_updates_gpu)	cuda_free_and_clear(l.vit_pos_embed_updates_gpu);
	if (l.vit_qkv_out_gpu)				cuda_free_and_clear(l.vit_qkv_out_gpu);
	if (l.vit_attn_scores_gpu)			cuda_free_and_clear(l.vit_attn_scores_gpu);
	if (l.vit_attn_out_gpu)				cuda_free_and_clear(l.vit_attn_out_gpu);
	if (l.vit_ffn_hidden_gpu)			cuda_free_and_clear(l.vit_ffn_hidden_gpu);
	if (l.vit_ln1_mean_gpu)				cuda_free_and_clear(l.vit_ln1_mean_gpu);
	if (l.vit_ln1_var_gpu)				cuda_free_and_clear(l.vit_ln1_var_gpu);
	if (l.vit_ln2_mean_gpu)				cuda_free_and_clear(l.vit_ln2_mean_gpu);
	if (l.vit_ln2_var_gpu)				cuda_free_and_clear(l.vit_ln2_var_gpu);
		if (l.vit_ln1_xhat_gpu)				cuda_free_and_clear(l.vit_ln1_xhat_gpu);
		if (l.vit_ln2_xhat_gpu)				cuda_free_and_clear(l.vit_ln2_xhat_gpu);
		if (l.vit_pre_res1_gpu)				cuda_free_and_clear(l.vit_pre_res1_gpu);
		if (l.vit_pre_res2_gpu)				cuda_free_and_clear(l.vit_pre_res2_gpu);
		if (l.vit_patch_tokens_gpu)			cuda_free_and_clear(l.vit_patch_tokens_gpu);
		if (l.vit_patch_delta_gpu)			cuda_free_and_clear(l.vit_patch_delta_gpu);
		if (l.tucker_q_latent_gpu)			cuda_free_and_clear(l.tucker_q_latent_gpu);
		if (l.tucker_k_latent_gpu)			cuda_free_and_clear(l.tucker_k_latent_gpu);
		if (l.tucker_v_latent_gpu)			cuda_free_and_clear(l.tucker_v_latent_gpu);
		if (l.tucker_q_gpu)					cuda_free_and_clear(l.tucker_q_gpu);
		if (l.tucker_k_gpu)					cuda_free_and_clear(l.tucker_k_gpu);
		if (l.tucker_v_gpu)					cuda_free_and_clear(l.tucker_v_gpu);
		if (l.tucker_scores_gpu)			cuda_free_and_clear(l.tucker_scores_gpu);
		if (l.tucker_context_gpu)			cuda_free_and_clear(l.tucker_context_gpu);
		if (l.tucker_windowed_input_gpu)	cuda_free_and_clear(l.tucker_windowed_input_gpu);
		if (l.fp8_tucker_q_gpu)				cuda_free_and_clear(l.fp8_tucker_q_gpu);
		if (l.fp8_tucker_k_gpu)				cuda_free_and_clear(l.fp8_tucker_k_gpu);
		if (l.fp8_tucker_attn_gpu)			cuda_free_and_clear(l.fp8_tucker_attn_gpu);
		if (l.fp8_tucker_v_t_gpu)			cuda_free_and_clear(l.fp8_tucker_v_t_gpu);
		if (l.fp8_tucker_q_amax_gpu)			cuda_free_and_clear(l.fp8_tucker_q_amax_gpu);
		if (l.fp8_tucker_q_scale_gpu)		cuda_free_and_clear(l.fp8_tucker_q_scale_gpu);
		if (l.fp8_tucker_k_amax_gpu)			cuda_free_and_clear(l.fp8_tucker_k_amax_gpu);
		if (l.fp8_tucker_k_scale_gpu)		cuda_free_and_clear(l.fp8_tucker_k_scale_gpu);
		if (l.fp8_tucker_attn_amax_gpu)		cuda_free_and_clear(l.fp8_tucker_attn_amax_gpu);
		if (l.fp8_tucker_attn_scale_gpu)	cuda_free_and_clear(l.fp8_tucker_attn_scale_gpu);
		if (l.fp8_tucker_v_amax_gpu)			cuda_free_and_clear(l.fp8_tucker_v_amax_gpu);
		if (l.fp8_tucker_v_scale_gpu)		cuda_free_and_clear(l.fp8_tucker_v_scale_gpu);
		if (l.fp8_tucker_scores_gemm_plan)
		{
			Darknet::fp8_gemm_plan_destroy(static_cast<Darknet::Fp8GemmPlan *>(l.fp8_tucker_scores_gemm_plan));
			l.fp8_tucker_scores_gemm_plan = nullptr;
		}
		if (l.fp8_tucker_context_gemm_plan)
		{
			Darknet::fp8_gemm_plan_destroy(static_cast<Darknet::Fp8GemmPlan *>(l.fp8_tucker_context_gemm_plan));
			l.fp8_tucker_context_gemm_plan = nullptr;
		}
		if (l.vit_tmp_token_c1_gpu)		cuda_free_and_clear(l.vit_tmp_token_c1_gpu);
		if (l.vit_tmp_token_c2_gpu)		cuda_free_and_clear(l.vit_tmp_token_c2_gpu);
		if (l.vit_tmp_token_n1_gpu)		cuda_free_and_clear(l.vit_tmp_token_n1_gpu);
		if (l.vit_tmp_token_n2_gpu)		cuda_free_and_clear(l.vit_tmp_token_n2_gpu);
		if (l.vit_tmp_token_n3_gpu)		cuda_free_and_clear(l.vit_tmp_token_n3_gpu);
		if (l.vit_tmp_ffn_hidden_gpu)		cuda_free_and_clear(l.vit_tmp_ffn_hidden_gpu);
		if (l.vit_tmp_head1_gpu)			cuda_free_and_clear(l.vit_tmp_head1_gpu);
		if (l.vit_tmp_head2_gpu)			cuda_free_and_clear(l.vit_tmp_head2_gpu);
		if (l.vit_tmp_head3_gpu)			cuda_free_and_clear(l.vit_tmp_head3_gpu);
		if (l.vit_tmp_head4_gpu)			cuda_free_and_clear(l.vit_tmp_head4_gpu);
		if (l.vit_tmp_head5_gpu)			cuda_free_and_clear(l.vit_tmp_head5_gpu);
		if (l.vit_tmp_scores_gpu)		cuda_free_and_clear(l.vit_tmp_scores_gpu);

		if (l.mv_ln1_gamma_gpu)			cuda_free_and_clear(l.mv_ln1_gamma_gpu);
		if (l.mv_ln1_gamma_updates_gpu)	cuda_free_and_clear(l.mv_ln1_gamma_updates_gpu);
		if (l.mv_ln1_beta_gpu)			cuda_free_and_clear(l.mv_ln1_beta_gpu);
		if (l.mv_ln1_beta_updates_gpu)	cuda_free_and_clear(l.mv_ln1_beta_updates_gpu);
		if (l.mv_ln2_gamma_gpu)			cuda_free_and_clear(l.mv_ln2_gamma_gpu);
		if (l.mv_ln2_gamma_updates_gpu)	cuda_free_and_clear(l.mv_ln2_gamma_updates_gpu);
		if (l.mv_ln2_beta_gpu)			cuda_free_and_clear(l.mv_ln2_beta_gpu);
		if (l.mv_ln2_beta_updates_gpu)	cuda_free_and_clear(l.mv_ln2_beta_updates_gpu);
		if (l.mv_A_log_gpu)				cuda_free_and_clear(l.mv_A_log_gpu);
		if (l.mv_A_log_updates_gpu)		cuda_free_and_clear(l.mv_A_log_updates_gpu);
		if (l.mv_D_gpu)					cuda_free_and_clear(l.mv_D_gpu);
		if (l.mv_D_updates_gpu)			cuda_free_and_clear(l.mv_D_updates_gpu);
		if (l.mv_tokens_gpu)				cuda_free_and_clear(l.mv_tokens_gpu);
		if (l.mv_ln1_out_gpu)			cuda_free_and_clear(l.mv_ln1_out_gpu);
		if (l.mv_ln1_mean_gpu)			cuda_free_and_clear(l.mv_ln1_mean_gpu);
		if (l.mv_ln1_var_gpu)			cuda_free_and_clear(l.mv_ln1_var_gpu);
		if (l.mv_ln1_xhat_gpu)			cuda_free_and_clear(l.mv_ln1_xhat_gpu);
		if (l.mv_dt_pre_gpu)				cuda_free_and_clear(l.mv_dt_pre_gpu);
		if (l.mv_dt_gpu)					cuda_free_and_clear(l.mv_dt_gpu);
		if (l.mv_scan_state_gpu)			cuda_free_and_clear(l.mv_scan_state_gpu);
		if (l.mv_scan_out_gpu)			cuda_free_and_clear(l.mv_scan_out_gpu);
		if (l.mv_mixer_cat_gpu)			cuda_free_and_clear(l.mv_mixer_cat_gpu);
		if (l.mv_pre_res2_gpu)			cuda_free_and_clear(l.mv_pre_res2_gpu);
		if (l.mv_ln2_out_gpu)			cuda_free_and_clear(l.mv_ln2_out_gpu);
		if (l.mv_ln2_mean_gpu)			cuda_free_and_clear(l.mv_ln2_mean_gpu);
		if (l.mv_ln2_var_gpu)			cuda_free_and_clear(l.mv_ln2_var_gpu);
		if (l.mv_ln2_xhat_gpu)			cuda_free_and_clear(l.mv_ln2_xhat_gpu);
		if (l.mv_tmp_token_c_gpu)		cuda_free_and_clear(l.mv_tmp_token_c_gpu);
		if (l.mv_tmp_token_n_gpu)		cuda_free_and_clear(l.mv_tmp_token_n_gpu);
		if (l.mv_tmp_token_p_gpu)		cuda_free_and_clear(l.mv_tmp_token_p_gpu);
		if (l.mv_tmp_bdt_gpu)			cuda_free_and_clear(l.mv_tmp_bdt_gpu);
		if (l.mv_tmp_bdt2_gpu)			cuda_free_and_clear(l.mv_tmp_bdt2_gpu);
		if (l.mv_tmp_ffn_gpu)			cuda_free_and_clear(l.mv_tmp_ffn_gpu);

		if (l.cli_shifts_gpu)				cuda_free_and_clear(l.cli_shifts_gpu);
		if (l.cli_shifts_inner_gpu)		cuda_free_and_clear(l.cli_shifts_inner_gpu);
		if (l.cli_w_det_gpu)				cuda_free_and_clear(l.cli_w_det_gpu);
		if (l.cli_w_det_updates_gpu)		cuda_free_and_clear(l.cli_w_det_updates_gpu);
		if (l.cli_b_det_gpu)				cuda_free_and_clear(l.cli_b_det_gpu);
		if (l.cli_b_det_updates_gpu)		cuda_free_and_clear(l.cli_b_det_updates_gpu);
		if (l.cli_w_proj_gpu)				cuda_free_and_clear(l.cli_w_proj_gpu);
		if (l.cli_w_proj_updates_gpu)		cuda_free_and_clear(l.cli_w_proj_updates_gpu);
		if (l.cli_b_proj_gpu)				cuda_free_and_clear(l.cli_b_proj_gpu);
		if (l.cli_b_proj_updates_gpu)		cuda_free_and_clear(l.cli_b_proj_updates_gpu);
		if (l.cli_w_gate_gpu)				cuda_free_and_clear(l.cli_w_gate_gpu);
		if (l.cli_w_gate_updates_gpu)		cuda_free_and_clear(l.cli_w_gate_updates_gpu);
		if (l.cli_b_gate_gpu)				cuda_free_and_clear(l.cli_b_gate_gpu);
		if (l.cli_b_gate_updates_gpu)		cuda_free_and_clear(l.cli_b_gate_updates_gpu);
		if (l.cli_ln_gamma_gpu)			cuda_free_and_clear(l.cli_ln_gamma_gpu);
		if (l.cli_ln_gamma_updates_gpu)	cuda_free_and_clear(l.cli_ln_gamma_updates_gpu);
		if (l.cli_ln_beta_gpu)				cuda_free_and_clear(l.cli_ln_beta_gpu);
		if (l.cli_ln_beta_updates_gpu)		cuda_free_and_clear(l.cli_ln_beta_updates_gpu);
		if (l.cli_layer_scale_gpu)			cuda_free_and_clear(l.cli_layer_scale_gpu);
		if (l.cli_layer_scale_updates_gpu)	cuda_free_and_clear(l.cli_layer_scale_updates_gpu);
		if (l.cli_w_proj_g_gpu)			cuda_free_and_clear(l.cli_w_proj_g_gpu);
		if (l.cli_w_proj_g_updates_gpu)	cuda_free_and_clear(l.cli_w_proj_g_updates_gpu);
		if (l.cli_b_proj_g_gpu)			cuda_free_and_clear(l.cli_b_proj_g_gpu);
		if (l.cli_b_proj_g_updates_gpu)	cuda_free_and_clear(l.cli_b_proj_g_updates_gpu);
		if (l.cli_w_gate_g_gpu)			cuda_free_and_clear(l.cli_w_gate_g_gpu);
		if (l.cli_w_gate_g_updates_gpu)	cuda_free_and_clear(l.cli_w_gate_g_updates_gpu);
		if (l.cli_b_gate_g_gpu)			cuda_free_and_clear(l.cli_b_gate_g_gpu);
		if (l.cli_b_gate_g_updates_gpu)	cuda_free_and_clear(l.cli_b_gate_g_updates_gpu);
		if (l.cli_ln_out_gpu)				cuda_free_and_clear(l.cli_ln_out_gpu);
		if (l.cli_ln_mean_gpu)			cuda_free_and_clear(l.cli_ln_mean_gpu);
		if (l.cli_ln_var_gpu)				cuda_free_and_clear(l.cli_ln_var_gpu);
		if (l.cli_ln_xhat_gpu)			cuda_free_and_clear(l.cli_ln_xhat_gpu);
		if (l.cli_z_det_gpu)				cuda_free_and_clear(l.cli_z_det_gpu);
		if (l.cli_z_ctx_gpu)				cuda_free_and_clear(l.cli_z_ctx_gpu);
		if (l.cli_z_ctx_pre_diff_gpu)		cuda_free_and_clear(l.cli_z_ctx_pre_diff_gpu);
		if (l.cli_g_raw_gpu)				cuda_free_and_clear(l.cli_g_raw_gpu);
		if (l.cli_g_feat_gpu)				cuda_free_and_clear(l.cli_g_feat_gpu);
		if (l.cli_gate_alpha_gpu)			cuda_free_and_clear(l.cli_gate_alpha_gpu);
		if (l.cli_gate_pre_sigmoid_gpu)	cuda_free_and_clear(l.cli_gate_pre_sigmoid_gpu);
		if (l.cli_vb_feat_gpu)				cuda_free_and_clear(l.cli_vb_feat_gpu);
		if (l.cli_hmix_gpu)				cuda_free_and_clear(l.cli_hmix_gpu);
		if (l.cli_drop_mask_gpu)			cuda_free_and_clear(l.cli_drop_mask_gpu);
		if (l.cli_global_ctx_gpu)			cuda_free_and_clear(l.cli_global_ctx_gpu);
		if (l.cli_g_raw_g_gpu)			cuda_free_and_clear(l.cli_g_raw_g_gpu);
		if (l.cli_g_feat_g_gpu)			cuda_free_and_clear(l.cli_g_feat_g_gpu);
		if (l.cli_gate_alpha_g_gpu)		cuda_free_and_clear(l.cli_gate_alpha_g_gpu);
		if (l.cli_gate_pre_sigmoid_g_gpu)	cuda_free_and_clear(l.cli_gate_pre_sigmoid_g_gpu);

		if (l.weights_gpu)					cuda_free_and_clear(l.weights_gpu);
		if (l.weight_updates_gpu)			cuda_free_and_clear(l.weight_updates_gpu);
		if (l.graph_self_weights_gpu)		cuda_free_and_clear(l.graph_self_weights_gpu);
		if (l.graph_self_weight_updates_gpu) cuda_free_and_clear(l.graph_self_weight_updates_gpu);
		if (l.graph_self_weights_gpu16)		cuda_free_and_clear(l.graph_self_weights_gpu16);
		if (l.graph_self_weight_updates_gpu16) cuda_free_and_clear(l.graph_self_weight_updates_gpu16);
	if (l.graph_edge_kernel_gpu)		cuda_free_and_clear(l.graph_edge_kernel_gpu);
	if (l.graph_edge_kernel_gpu16)	cuda_free_and_clear(l.graph_edge_kernel_gpu16);
	if (l.graph_edge_kernel_updates_gpu) cuda_free_and_clear(l.graph_edge_kernel_updates_gpu);
	if (l.graph_edge_biases_gpu)		cuda_free_and_clear(l.graph_edge_biases_gpu);
	if (l.graph_edge_bias_updates_gpu)	cuda_free_and_clear(l.graph_edge_bias_updates_gpu);
		if (l.graph_ref_gpu)				cuda_free_and_clear(l.graph_ref_gpu);
		if (l.graph_agg_gpu)				cuda_free_and_clear(l.graph_agg_gpu);
		if (l.graph_alpha_gpu)				cuda_free_and_clear(l.graph_alpha_gpu);
		if (l.graph_valid_gpu)				cuda_free_and_clear(l.graph_valid_gpu);
		if (l.weight_deform_gpu)			cuda_free_and_clear(l.weight_deform_gpu);
		if (l.offset_weights_gpu)		cuda_free_and_clear(l.offset_weights_gpu);
		if (l.offset_weight_updates_gpu)	cuda_free_and_clear(l.offset_weight_updates_gpu);
		if (l.offset_weights_gpu16)		cuda_free_and_clear(l.offset_weights_gpu16);
		if (l.offset_weight_updates_gpu16) cuda_free_and_clear(l.offset_weight_updates_gpu16);
		if (l.offset_biases_gpu)			cuda_free_and_clear(l.offset_biases_gpu);
		if (l.offset_bias_updates_gpu)	cuda_free_and_clear(l.offset_bias_updates_gpu);
		if (l.offsets_gpu)				cuda_free_and_clear(l.offsets_gpu);
		if (l.offset_deltas_gpu)			cuda_free_and_clear(l.offset_deltas_gpu);
		if (l.mask_weights_gpu)			cuda_free_and_clear(l.mask_weights_gpu);
		if (l.mask_weight_updates_gpu)	cuda_free_and_clear(l.mask_weight_updates_gpu);
		if (l.mask_biases_gpu)			cuda_free_and_clear(l.mask_biases_gpu);
		if (l.mask_bias_updates_gpu)		cuda_free_and_clear(l.mask_bias_updates_gpu);
		if (l.masks_gpu)				cuda_free_and_clear(l.masks_gpu);
		if (l.mask_deltas_gpu)			cuda_free_and_clear(l.mask_deltas_gpu);
	if (l.weights_gpu16)				cuda_free_and_clear(l.weights_gpu16);
	if (l.weight_updates_gpu16)			cuda_free_and_clear(l.weight_updates_gpu16);
#ifdef DARKNET_HAS_FP4
	if (l.fp4_gemm_plan)
	{
		Darknet::fp4_gemm_plan_destroy(static_cast<Darknet::Fp4GemmPlan *>(l.fp4_gemm_plan));
		l.fp4_gemm_plan = nullptr;
	}
	if (l.fp4_wgrad_gemm_plan)
	{
		Darknet::fp4_gemm_plan_destroy(static_cast<Darknet::Fp4GemmPlan *>(l.fp4_wgrad_gemm_plan));
		l.fp4_wgrad_gemm_plan = nullptr;
	}
	if (l.fp4_dgrad_gemm_plan)
	{
		Darknet::fp4_gemm_plan_destroy(static_cast<Darknet::Fp4GemmPlan *>(l.fp4_dgrad_gemm_plan));
		l.fp4_dgrad_gemm_plan = nullptr;
	}
	if (l.fp4_relay_gpu) cuda_free_and_clear(l.fp4_relay_gpu);
	if (l.fp4_relay_scales_gpu) cuda_free_and_clear(l.fp4_relay_scales_gpu);
	if (l.fp4_amax_gpu) cuda_free_and_clear(l.fp4_amax_gpu);
#endif
#ifdef DARKNET_HAS_FP8
	if (l.weights_fp8_gpu)				cuda_free_and_clear(l.weights_fp8_gpu);
	if (l.weights_fp8_t_gpu)			cuda_free_and_clear(l.weights_fp8_t_gpu);
	if (l.weights_fp8_nhwc_gpu)			cuda_free_and_clear(l.weights_fp8_nhwc_gpu);
	if (l.fp8_weight_scale_gpu)			cuda_free_and_clear(l.fp8_weight_scale_gpu);
	if (l.fp8_input_scale_gpu)			cuda_free_and_clear(l.fp8_input_scale_gpu);
	if (l.fp8_dy_scale_gpu)				cuda_free_and_clear(l.fp8_dy_scale_gpu);
	if (l.fp8_input_amax_gpu)			cuda_free_and_clear(l.fp8_input_amax_gpu);
	if (l.fp8_dy_amax_gpu)				cuda_free_and_clear(l.fp8_dy_amax_gpu);
	if (l.fp8_amax_gpu)					cuda_free_and_clear(l.fp8_amax_gpu);
	if (l.fp8_weight_scale_state_gpu)	cuda_free_and_clear(l.fp8_weight_scale_state_gpu);
	if (l.fp8_input_scale_state_gpu)	cuda_free_and_clear(l.fp8_input_scale_state_gpu);
	if (l.fp8_dy_scale_state_gpu)		cuda_free_and_clear(l.fp8_dy_scale_state_gpu);
	if (l.fp8_relay_gpu)				cuda_free_and_clear(l.fp8_relay_gpu);
	if (l.fp8_relay_amax_gpu)			cuda_free_and_clear(l.fp8_relay_amax_gpu);
	if (l.fp8_gemm_plan)
	{
		Darknet::fp8_gemm_plan_destroy(static_cast<Darknet::Fp8GemmPlan *>(l.fp8_gemm_plan));
		l.fp8_gemm_plan = nullptr;
	}
	if (l.fp8_wgrad_gemm_plan)
	{
		Darknet::fp8_gemm_plan_destroy(static_cast<Darknet::Fp8GemmPlan *>(l.fp8_wgrad_gemm_plan));
		l.fp8_wgrad_gemm_plan = nullptr;
	}
	if (l.fp8_dgrad_gemm_plan)
	{
		Darknet::fp8_gemm_plan_destroy(static_cast<Darknet::Fp8GemmPlan *>(l.fp8_dgrad_gemm_plan));
		l.fp8_dgrad_gemm_plan = nullptr;
	}
	if (l.fp8_conv_fwd_plan)
	{
		Darknet::fp8_conv_plan_destroy(static_cast<Darknet::Fp8ConvPlan *>(l.fp8_conv_fwd_plan));
		l.fp8_conv_fwd_plan = nullptr;
	}
	if (l.fp8_conv_wgrad_plan)
	{
		Darknet::fp8_conv_plan_destroy(static_cast<Darknet::Fp8ConvPlan *>(l.fp8_conv_wgrad_plan));
		l.fp8_conv_wgrad_plan = nullptr;
	}
	if (l.fp8_conv_dgrad_plan)
	{
		Darknet::fp8_conv_plan_destroy(static_cast<Darknet::Fp8ConvPlan *>(l.fp8_conv_dgrad_plan));
		l.fp8_conv_dgrad_plan = nullptr;
	}
#endif
	if (l.biases_gpu)					cuda_free_and_clear(l.biases_gpu);
	if (l.bias_updates_gpu)				cuda_free_and_clear(l.bias_updates_gpu);
	if (l.scales_gpu)					cuda_free_and_clear(l.scales_gpu);
	if (l.scale_updates_gpu)			cuda_free_and_clear(l.scale_updates_gpu);
	if (l.input_antialiasing_gpu)		cuda_free_and_clear(l.input_antialiasing_gpu);
	if (l.optimized_memory < 2)
	{
		if (l.x_gpu)					cuda_free_and_clear(l.x_gpu);
		if (l.output_gpu)				cuda_free_and_clear(l.output_gpu);
		if (l.output_avg_gpu)			cuda_free_and_clear(l.output_avg_gpu);
		if (l.activation_input_gpu)		cuda_free_and_clear(l.activation_input_gpu);
	}

	if (l.delta_gpu && (l.optimized_memory < 1 || (l.keep_delta_gpu && l.optimized_memory < 3)))
	{
		cuda_free_and_clear(l.delta_gpu);
	}

	if (l.cos_sim_gpu)					cuda_free_and_clear(l.cos_sim_gpu);
	if (l.rand_gpu)						cuda_free_and_clear(l.rand_gpu);
	if (l.squared_gpu)					cuda_free_and_clear(l.squared_gpu);
	if (l.norms_gpu)					cuda_free_and_clear(l.norms_gpu);
	if (l.input_sizes_gpu)				cuda_free((float*)l.input_sizes_gpu);
	if (l.layers_output_gpu)			cuda_free((float*)l.layers_output_gpu);
	if (l.layers_delta_gpu)				cuda_free((float*)l.layers_delta_gpu);
	l.input_sizes_gpu	= nullptr;
	l.layers_output_gpu	= nullptr;
	l.layers_delta_gpu	= nullptr;

	// CONV-LSTM
	if (l.f_gpu)						cuda_free_and_clear(l.f_gpu);
	if (l.i_gpu)						cuda_free_and_clear(l.i_gpu);
	if (l.g_gpu)						cuda_free_and_clear(l.g_gpu);
	if (l.o_gpu)						cuda_free_and_clear(l.o_gpu);
	if (l.c_gpu)						cuda_free_and_clear(l.c_gpu);
	if (l.h_gpu)						cuda_free_and_clear(l.h_gpu);
	if (l.bottelneck_hi_gpu)			cuda_free_and_clear(l.bottelneck_hi_gpu);
	if (l.bottelneck_delta_gpu)			cuda_free_and_clear(l.bottelneck_delta_gpu);
	if (l.temp_gpu)						cuda_free_and_clear(l.temp_gpu);
	if (l.temp2_gpu)					cuda_free_and_clear(l.temp2_gpu);
	if (l.temp3_gpu)					cuda_free_and_clear(l.temp3_gpu);
	if (l.dc_gpu)						cuda_free_and_clear(l.dc_gpu);
	if (l.dh_gpu)						cuda_free_and_clear(l.dh_gpu);
	if (l.prev_state_gpu)				cuda_free_and_clear(l.prev_state_gpu);
	if (l.prev_cell_gpu)				cuda_free_and_clear(l.prev_cell_gpu);
	if (l.stored_c_gpu)					cuda_free_and_clear(l.stored_c_gpu);
	if (l.stored_h_gpu)					cuda_free_and_clear(l.stored_h_gpu);
	if (l.last_prev_state_gpu)			cuda_free_and_clear(l.last_prev_state_gpu);
	if (l.last_prev_cell_gpu)			cuda_free_and_clear(l.last_prev_cell_gpu);
	if (l.cell_gpu)						cuda_free_and_clear(l.cell_gpu);

#ifdef CUDNN   // shouldn't be used for -map
	if (!keep_cudnn_desc)
	{
		if (l.srcTensorDesc)		CHECK_CUDNN(cudnnDestroyTensorDescriptor(l.srcTensorDesc));
		if (l.dstTensorDesc)		CHECK_CUDNN(cudnnDestroyTensorDescriptor(l.dstTensorDesc));
		if (l.srcTensorDesc16)		CHECK_CUDNN(cudnnDestroyTensorDescriptor(l.srcTensorDesc16));
		if (l.dstTensorDesc16)		CHECK_CUDNN(cudnnDestroyTensorDescriptor(l.dstTensorDesc16));
		if (l.dsrcTensorDesc)		CHECK_CUDNN(cudnnDestroyTensorDescriptor(l.dsrcTensorDesc));
		if (l.ddstTensorDesc)		CHECK_CUDNN(cudnnDestroyTensorDescriptor(l.ddstTensorDesc));
		if (l.dsrcTensorDesc16)		CHECK_CUDNN(cudnnDestroyTensorDescriptor(l.dsrcTensorDesc16));
		if (l.ddstTensorDesc16)		CHECK_CUDNN(cudnnDestroyTensorDescriptor(l.ddstTensorDesc16));
		if (l.normTensorDesc)		CHECK_CUDNN(cudnnDestroyTensorDescriptor(l.normTensorDesc));
		if (l.normDstTensorDesc)	CHECK_CUDNN(cudnnDestroyTensorDescriptor(l.normDstTensorDesc));
		if (l.normDstTensorDescF16)	CHECK_CUDNN(cudnnDestroyTensorDescriptor(l.normDstTensorDescF16));

		if (l.weightDesc)			CHECK_CUDNN(cudnnDestroyFilterDescriptor(l.weightDesc));
		if (l.weightDesc16)			CHECK_CUDNN(cudnnDestroyFilterDescriptor(l.weightDesc16));
		if (l.dweightDesc)			CHECK_CUDNN(cudnnDestroyFilterDescriptor(l.dweightDesc));
		if (l.dweightDesc16)		CHECK_CUDNN(cudnnDestroyFilterDescriptor(l.dweightDesc16));

		if (l.convDesc)				CHECK_CUDNN(cudnnDestroyConvolutionDescriptor(l.convDesc));

		if (l.poolingDesc)			CHECK_CUDNN(cudnnDestroyPoolingDescriptor(l.poolingDesc));

		//cudnnConvolutionFwdAlgo_t fw_algo, fw_algo16;
		//cudnnConvolutionBwdDataAlgo_t bd_algo, bd_algo16;
		//cudnnConvolutionBwdFilterAlgo_t bf_algo, bf_algo16;
	}
#endif  // CUDNN

#endif  // DARKNET_GPU
}
