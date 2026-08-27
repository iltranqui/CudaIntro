/* Darknet/YOLO:  https://codeberg.org/CCodeRun/darknet
 * Copyright 2024-2026 Stephane Charette
 */

#pragma once

#ifndef __cplusplus
#error "The Darknet/YOLO project requires a C++ compiler."
#endif

/** @file
 * Defines the layer structure and includes all of the different layer include files.
 */

#include "darknet_internal.hpp"


namespace Darknet
{
	struct Layer final
	{
		Darknet::ELayerType type; ///< @see @ref Darknet::to_string()
		ACTIVATION activation;
		ACTIVATION lstm_activation;
		COST_TYPE cost_type;

		void(*forward)		(Layer & l, Darknet::NetworkState network_state);
		void(*backward)		(Layer & l, Darknet::NetworkState network_state);
		void(*update)		(Layer & l, int, float, float, float);
		void(*forward_gpu)	(Layer & l, Darknet::NetworkState network_state);
		void(*backward_gpu)	(Layer & l, Darknet::NetworkState network_state);
		void(*update_gpu)	(Layer & l, int, float, float, float, float);

		Layer *share_layer;
		int train;
		int avgpool;
		int batch_normalize;
		int shortcut;
		int batch;
		int dynamic_minibatch;
		int forced;
		int flipped;
		int inputs;
		int outputs;
		float mean_alpha;
		int nweights; ///< number of floats stored in @ref weights
		int nbiases; ///< unused?  Seems to be no references to this in the codebase.
		int extra;
		int truths;
		int h; ///< input height
		int w; ///< input width
		int c; ///< input channels
		int out_h;	///< output height
		int out_w;	///< output width
		int out_c;	///< output channels
		int n; ///< number of anchors, masks (?), weights (?); for example, with YOLOv4-tiny this is set to @p 3
		int max_boxes;
		int truth_size;
		int groups;
		int group_id;
		int axis;
		int begin_slice_point;
		int end_slice_point;
		int size;
		int side;
		int stride;
		int stride_x;
		int stride_y;
		int dilation;
		int antialiasing;
		int maxpool_depth;
		int maxpool_zero_nonmax;
		int out_channels;
		float reverse;
		int coordconv;
		int flatten;
		int spatial;
		int pad;
		int sqrt;
		int flip;
		int index; ///< layer number starting at zero ([net] does not count)
		int scale_wh;
		int binary;
		int xnor;
		int peephole;
		int use_bin_output;

		// DETR decoder head (detr_decoder_layer) ----------------------------
		int detr_queries;			///< number of object queries (set size)
		int detr_heads;				///< attention heads (reserved; v1 uses 1 effective head)
		int detr_ffn;				///< FFN hidden dimension
		float detr_cls_weight;		///< classification loss weight
		float detr_l1_weight;		///< L1 box loss weight
		float detr_giou_weight;		///< GIoU loss weight (reserved in v1)
		float detr_noobj_weight;		///< unmatched-query classification negative weight
		float * detr_workspace;		///< host staging buffer (reserved)
		float * detr_workspace_gpu;	///< single GPU scratch arena carved into per-stage buffers
		int keep_delta_gpu;
		int cudnn_16bit_mode; ///< 0=FP16, 1=BF16 for the existing *16 cuDNN descriptors/buffers.
		int optimized_memory;
		int steps;
		int history_size;
		int bottleneck;
		float time_normalizer;
		int state_constrain;
		int hidden;
		int truth;
		float smooth;
		float dot;
		int deform;
		int grad_centr;
		int sway;
		int rotate;
		int stretch;
		int stretch_sway;
		float angle;
		float jitter;
		float resize;
		float saturation;
		float exposure;
		float shift;
		float ratio;
		float learning_rate_scale;
		float clip;
		int focal_loss;
		float *classes_multipliers;
		float label_smooth_eps;
		int noloss;
		int softmax;
		int classes;
		int detection;
		int embedding_layer_id;
		float *embedding_output;
		int embedding_size;
		float sim_thresh;
		int track_history_size;
		int dets_for_track;
		int dets_for_show;
		float track_ciou_norm;
		int coords;
		int background;
		int rescore;
		int objectness;
		int does_cost;
		int joint;
		int noadjust;
		int reorg;
		int log;
		int tanh;
		int *mask;
		int total;
		float bflops;

		int adam;
		float B1;
		float B2;
		float eps;

		int t;

		float alpha;
		float beta;
		float kappa;

		float coord_scale;
		float object_scale;
		float noobject_scale;
		float mask_scale;
		float class_scale;
		int bias_match;
		float random;
		float ignore_thresh;
		float truth_thresh;
		float iou_thresh;
		float thresh;
		float focus;
		int classfix;
		int absolute;
		int assisted_excitation;

		int onlyforward;
		int stopbackward;
		int train_only_bn;
		int dont_update;
		int burnin_update;
		int dontload;
		int dontsave;
		int dontloadscales;
		int numload;

		float temperature;
		float probability;
		float dropblock_size_rel;
		int dropblock_size_abs;
		int dropblock;
		float scale;

		int receptive_w;
		int receptive_h;
		int receptive_w_scale;
		int receptive_h_scale;

		char  * cweights; ///< @todo V5: possibly unused?
		int   * indexes;
		int   * input_layers;
		int   * input_sizes;
		float **layers_output;
		float **layers_delta;
		WEIGHTS_TYPE_T weights_type;
		WEIGHTS_NORMALIZATION_T weights_normalization;
		int   * map;
		int   * counts;
		float ** sums;
		float * rand;
		float * cost;
		int *labels;
		int *class_ids;
		int contrastive_neg_max; ///< @todo V5: possibly unused?

		// OMD/metric-geometry auxiliary loss fields.
		float omd_margin;
		float omd_positive_weight;
		float omd_negative_weight;
		float omd_background_weight;
		float omd_small_threshold;
		float omd_small_boost;
		float omd_eps;
		int omd_max_samples;
		int omd_background_samples;
		int omd_pool;

		float *cos_sim;
		float *exp_cos_sim;
		float *p_constrastive;
		contrastive_params *contrast_p_gpu;
		float * state;
		float * prev_state;
		float * forgot_state; ///< @todo V5: unused?
		float * forgot_delta; ///< @todo V5: unused?
		float * state_delta; ///< @todo V5: possibly unused?
		float * combine_cpu_unused; ///< @todo V5: unused?
		float * combine_delta_cpu_unused; ///< @todo V5: unused?

		float *concat;
		float *concat_delta;

		float *binary_weights;

		float *biases;			/// biases loaded here by @ref load_convolutional_weights() and @ref load_connected_weights(), see @ref n
		float *bias_updates;

		float *scales;			/// scales loaded here by @ref load_convolutional_weights() when @ref batch_normalize is set
		float *scale_updates;

		float *weights_ema;
		float *biases_ema;
		float *scales_ema;

		float *weights;			/// weights loaded here by @ref load_convolutional_weights(), @ref load_connected_weights(), and @ref load_shortcut_weights()
		float *weight_updates;

		float scale_x_y;
		int objectness_smooth;
		int new_coords;
		int show_details;
		float max_delta;
		float uc_normalizer;
		float iou_normalizer;
		float obj_normalizer;
		float cls_normalizer;
		// modern anchor-free head (yolox/ppyoloe/yolonas) tunables
		float center_radius;       // SimOTA / TAL candidate center radius, grid cells
		int   assign_topk;         // SimOTA dynamic-k pool size (yolox) or TAL top-k (ppyoloe/yolonas)
		float tal_alpha;
		float tal_beta;
		float nas_duplicate_decay; // 0 disables
		float vfl_gamma;           // 0 disables varifocal weighting (future phase)
		int   yolox_soft_label;    // 0 = hard 1.0 class target
		int   yolox_ignore_neg;    // 1 = keep ignore_thresh negative suppression
		float box_loss_weight;
		float dfl_loss_weight;
		int   score_mode;          // 0 = obj*cls, 1 = cls-only (future phase)
		int   atss_warmup_iters;
		int   l1_final_iters;
		float delta_normalizer;

		// CenterNet-style point/heatmap detection fields.
		int ct_min_radius;			///< minimum Gaussian radius on the output heatmap for tiny objects
		int ct_peak_nms;			///< use 3x3 local-maximum filtering at inference
		int ct_anisotropic_gaussian;	///< 1=adaptive axis-aligned elliptical Gaussian, 0=circular Gaussian
		float ct_small_threshold;	///< legacy/diagnostic object min-side threshold in input pixels
		float ct_small_ref_size;	///< reference pixel size for continuous tiny-object weighting
		float ct_small_boost;		///< maximum total multiplier for tiny center/regression loss
		float ct_scale_min_px;	///< min max-side size in input pixels assigned to this head
		float ct_scale_max_px;	///< max max-side size in input pixels assigned to this head
		float ct_gaussian_iou;		///< CenterNet Gaussian radius overlap target
		float ct_focal_alpha;		///< heatmap focal alpha
		float ct_focal_beta;		///< heatmap Gaussian-halo negative beta
		float ct_hm_normalizer;	///< heatmap loss normalizer
		float ct_wh_normalizer;	///< width/height log-size loss normalizer
		float ct_off_normalizer;	///< sub-cell center offset loss normalizer

		IOU_LOSS iou_loss;
		IOU_LOSS iou_thresh_kind;
		NMS_KIND nms_kind;
		float beta_nms;
		YOLO_POINT yolo_point;

		char *align_bit_weights_gpu;
		float *mean_arr_gpu;
		float *align_workspace_gpu;
		float *transposed_align_workspace_gpu;
		int align_workspace_size;

		char *align_bit_weights;
		float *mean_arr;
		int align_bit_weights_size;
		int lda_align;
		int new_lda;
		int bit_align;

		float *col_image;
		float * delta;
		float * output;
		float * activation_input;
		int delta_pinned;
		int output_pinned;
		float * loss;
		float * squared;
		float * norms;

		float * spatial_mean;
		float * mean;
		float * variance;

		float * mean_delta;
		float * variance_delta;

		float * rolling_mean;		/// rolling means loaded here by @ref load_convolutional_weights() and @ref load_connected_weights() when @ref batch_normalize is set
		float * rolling_variance;	/// rolling variance loaded here by @ref load_convolutional_weights() and @ref load_connected_weights() when @ref batch_normalize is set

		float * x;
		float * x_norm;

		float * m;
		float * v;

		float * bias_m;
		float * bias_v;
		float * scale_m;
		float * scale_v;

		float *z_cpu;
		float *r_cpu;
		float *h_cpu;
		float *stored_h_cpu;
		float * prev_state_cpu;

		float *temp_cpu;
		float *temp2_cpu;
		float *temp3_cpu;

		float *dh_cpu;
		float *hh_cpu;
		float *prev_cell_cpu;
		float *cell_cpu;
		float *f_cpu;
		float *i_cpu;
		float *g_cpu;
		float *o_cpu;
		float *c_cpu;
		float *stored_c_cpu; ///< @todo V5: unused?
		float *dc_cpu;

		float *binary_input;
		uint32_t *bin_re_packed_input;
		char *t_bit_input;

		Layer *input_layer;
		Layer *self_layer;
		Layer *output_layer;
		Layer *mv_in_proj_layer;
		Layer *mv_conv_x_layer;
		Layer *mv_conv_z_layer;
		Layer *mv_x_proj_layer;
		Layer *mv_dt_proj_layer;
		Layer *mv_out_proj_layer;
		Layer *mv_res_proj_layer;
		Layer *mv_ffn1_layer;
		Layer *mv_ffn2_layer;

		Layer *reset_layer_unused; ///< @todo V5: unused?
		Layer *update_layer_unused; ///< @todo V5: unused?
		Layer *state_layer_unused; ///< @todo V5: unused?

		Layer *input_gate_layer_unused; ///< @todo V5: unused?
		Layer *state_gate_layer_unused; ///< @todo V5: unused?
		Layer *input_save_layer_unused; ///< @todo V5: unused?
		Layer *state_save_layer_unused; ///< @todo V5: unused?
		Layer *input_state_layer_unused; ///< @todo V5: unused?
		Layer *state_state_layer_unused; ///< @todo V5: unused?

		Layer *input_z_layer_unused; ///< @todo V5: unused?
		Layer *state_z_layer_unused; ///< @todo V5: unused?

		Layer *input_r_layer_unused; ///< @todo V5: unused?
		Layer *state_r_layer_unused; ///< @todo V5: unused?

		Layer *input_h_layer_unused; ///< @todo V5: unused?
		Layer *state_h_layer_unused; ///< @todo V5: unused?

		// "They are mostly relevant during training and not in the prediction/output phase."

		Layer *wz_unused; ///< @todo V5: unused?
		Layer *uz_unused; ///< @todo V5: unused?
		Layer *wr_unused; ///< @todo V5: unused?
		Layer *ur_unused; ///< @todo V5: unused?
		Layer *wh_unused; ///< @todo V5: unused?
		Layer *uh_unused; ///< @todo V5: unused?
		Layer *uo; ///< used in lstm (update for output gate?)
		Layer *wo; ///< used in lstm (weights for output forget gate?)
		Layer *vo_unused; ///< @todo V5: unused?
		Layer *uf; ///< used in lstm (update for forget gate?)
		Layer *wf; ///< used in lstm (weights for forget gate?)
		Layer *vf_unused; ///< @todo V5: unused?
		Layer *ui; ///< used in lstm (update input gate?)
		Layer *wi; ///< used in lstm (weight for input connections?)
		Layer *vi_unused; ///< @todo V5: unused?
		Layer *ug; ///< used in lstm (update gradient?)
		Layer *wg; ///< used in lstm (weight gradient?)

		Darknet::Tree *softmax_tree;

		size_t workspace_size;

		int *indexes_gpu;

		int stream;
		int wait_stream_id;

		float *z_gpu;
		float *r_gpu;
		float *h_gpu;
		float *stored_h_gpu;
		float *bottelneck_hi_gpu;
		float *bottelneck_delta_gpu;

		float *temp_gpu;
		float *temp2_gpu;
		float *temp3_gpu;

		float *dh_gpu;
		float *hh_gpu; ///< @todo V5: possibly unused?
		float *prev_cell_gpu;
		float *prev_state_gpu;
		float *last_prev_state_gpu; ///< @todo V5: unused?
		float *last_prev_cell_gpu; ///< @todo V5: unused?
		float *cell_gpu;
		float *f_gpu;
		float *i_gpu;
		float *g_gpu;
		float *o_gpu;
		float *c_gpu;
		float *stored_c_gpu; ///< @todo V5: possibly unused?
		float *dc_gpu;

		// adam
		float *m_gpu;
		float *v_gpu;
		float *bias_m_gpu; ///< @todo V5: possibly unused?
		float *scale_m_gpu; ///< @todo V5: possibly unused?
		float *bias_v_gpu; ///< @todo V5: possibly unused?
		float *scale_v_gpu; ///< @todo V5: possibly unused?

		float * combine_gpu; ///< @todo V5: unused?
		float * combine_delta_gpu; ///< @todo V5: unused?

		float * forgot_state_gpu; ///< @todo V5: possibly unused?
		float * forgot_delta_gpu; ///< @todo V5: possibly unused?
		float * state_gpu;
		float * state_delta_gpu; ///< @todo V5: possibly unused?
		float * gate_gpu; ///< @todo V5: possibly unused?
		float * gate_delta_gpu; ///< @todo V5: possibly unused?
		float * save_gpu; ///< @todo V5: possibly unused?
		float * save_delta_gpu; ///< @todo V5: unused?
		float * concat_gpu; ///< @todo V5: unused?
		float * concat_delta_gpu;  ///< @todo V5: unused?

		float *binary_input_gpu;
		float *binary_weights_gpu;
		float *bin_conv_shortcut_in_gpu; ///< @todo V5: possibly unused?
		float *bin_conv_shortcut_out_gpu; ///< @todo V5: possibly unused?

		float * mean_gpu;
		float * variance_gpu;
		float * m_cbn_avg_gpu; ///< @todo V5: possibly unused?
		float * v_cbn_avg_gpu; ///< @todo V5: possibly unused?

		float * rolling_mean_gpu;
		float * rolling_variance_gpu;

		float * variance_delta_gpu;
		float * mean_delta_gpu;

		float * col_image_gpu;

		float * x_gpu;
		float * x_norm_gpu;
		float * weights_gpu;
		float * weight_updates_gpu;
		float * weight_deform_gpu;
		float * weight_change_gpu;

		float * weights_gpu16;
		float * weight_updates_gpu16;
		void * fp4_gemm_plan;
		void * fp4_wgrad_gemm_plan;
		void * fp4_dgrad_gemm_plan;
		size_t fp4_workspace_size;
		int fp4_eligible;
		int fp4_train_eligible;
		int fp4_weights_prepacked;
		char * fp4_relay_gpu;
		char * fp4_relay_scales_gpu;
		size_t fp4_relay_packed_bytes;
		size_t fp4_relay_scale_bytes;
		int fp4_relay_next_layer;
		int fp4_relay_source_layer;
		int fp4_relay_valid;
		/// Calibration-only: accumulates this layer's input activation |amax| across
		/// the images passed to `darknet detector calibrate -fp4`. Written to the
		/// .fp4scales sidecar as a diagnostic/audit artifact -- unlike FP8, the
		/// cuDNN Frontend block-scale-quantize op used by the FP4 GEMM path computes
		/// its scales entirely internally with no exposed override, so this value is
		/// not yet consumed by the forward pass. See fp4_calibration.hpp.
		float * fp4_amax_gpu;
		float fp4_activation_amax_host;
		float fp4_input_scale_host;
		int fp4_scales_loaded;
		// Persistent FP8 activation edge in the consumer's NHWC E4M3 layout.
		// The FP32 output remains available for ordinary Darknet layers; a direct
		// convolutional consumer reads this buffer and skips its own re-quantize.
		char * fp8_relay_gpu;
		float * fp8_relay_amax_gpu;
		size_t fp8_relay_bytes;
		int fp8_relay_next_layer;
		int fp8_relay_source_layer;
		int fp8_relay_valid;
		int fp8_relay_enabled;
		int fp8_relay_saturation_fallback;
		int fp8_graph_activation_fused;
		char * weights_fp8_gpu;
		char * weights_fp8_t_gpu;
		char * weights_fp8_nhwc_gpu;
		float * fp8_weight_scale_gpu;
		float * fp8_input_scale_gpu;
		float * fp8_dy_scale_gpu;
		float * fp8_input_amax_gpu;
		float * fp8_dy_amax_gpu;
		float * fp8_amax_gpu;
		/// device-side delayed-scaling state (16-entry amax history + write index); avoids per-iteration GPU->CPU syncs
		float * fp8_weight_scale_state_gpu;
		float * fp8_input_scale_state_gpu;
		float * fp8_dy_scale_state_gpu;
		void * fp8_gemm_plan;
		void * fp8_wgrad_gemm_plan;
		void * fp8_dgrad_gemm_plan;
		void * fp8_conv_fwd_plan;
		void * fp8_conv_wgrad_plan;
		void * fp8_conv_dgrad_plan;
		size_t fp8_workspace_size;
		int fp8_eligible;
		int fp8_train_eligible;
		int fp8_scales_loaded;
		int fp8_k_pad;
		/// images per strided-batched FP8 GEMM (chunk size the forward/dgrad plans were built for)
		int fp8_forward_batch;
		int fp8_dgrad_batch;
		/// images folded into the reduction dimension of one wgrad GEMM
		int fp8_wgrad_batch;
		/// wgrad GEMM writes directly into the row-major weight update buffer
		int fp8_wgrad_direct_update;
		/// 1x1 dgrad GEMM writes directly into the NCHW input-delta buffer
		int fp8_dgrad_direct_update;
		/// set only after the current backward pass recorded a full-batch dY amax
		int fp8_dy_amax_valid;
		float fp8_weight_scale_host;
		float fp8_input_scale_host;
		float fp8_dy_scale_host;
		float fp8_weight_amax_history[16];
		float fp8_input_amax_history[16];
		float fp8_dy_amax_history[16];
		int fp8_weight_amax_next;
		int fp8_input_amax_next;
		int fp8_dy_amax_next;
		int fp8_weight_amax_count;
		int fp8_input_amax_count;
		int fp8_dy_amax_count;
		float fp8_activation_amax_host;

		float * biases_gpu;
		float * bias_updates_gpu;
		float * bias_change_gpu;

		float * scales_gpu;
		float * scale_updates_gpu;
		float * scale_change_gpu;

		float * input_antialiasing_gpu;
		float * output_gpu;
		float * output_avg_gpu;
		float * activation_input_gpu;
		float * loss_gpu;
		float * delta_gpu;
		/// Opaque shape-dependent workspace for the classic YOLO CUDA training loss.
		void * yolo_training_gpu_context;
		/// Set after a recoverable setup failure to avoid retrying allocations every iteration.
		int yolo_training_gpu_disabled;
		float * cos_sim_gpu;
		float * rand_gpu;
		float * drop_blocks_scale;
		float * drop_blocks_scale_gpu;
		float * squared_gpu;
		float * norms_gpu;

		float *gt_gpu;
		float *a_avg_gpu;

		int *input_sizes_gpu;
		float **layers_output_gpu;
		float **layers_delta_gpu;

		// Graph Convolution fields
		int graph_k;                    ///< neighborhood size == size * size
		int graph_edge_mode;            ///< 0=uniform valid mean, 1=learned pairwise softmax
		int graph_use_self;             ///< include self branch
		int graph_valid_mask_zero;      ///< invalid neighbors are masked out
		int graph_cpg;                  ///< channels per group
		int graph_npg;                  ///< output channels per group

		float *graph_self_weights;          ///< [n, c/groups]
		float *graph_self_weight_updates;
		float *graph_edge_kernel;           ///< [groups, graph_k, 2 * c/groups] for [ref, neighbor]
		float *graph_edge_kernel_updates;
		float *graph_edge_biases;           ///< [groups, graph_k]
		float *graph_edge_bias_updates;

		float *graph_ref;                   ///< [batch, c, out_h, out_w]
		float *graph_agg;                   ///< [batch, c, out_h, out_w]
		float *graph_alpha;                 ///< [batch, groups, out_h, out_w, graph_k]
		float *graph_valid;                 ///< [batch, groups, out_h, out_w, graph_k], 0 or 1

		float *graph_self_weights_gpu;
		float *graph_self_weight_updates_gpu;
		float *graph_self_weights_gpu16;
		float *graph_self_weight_updates_gpu16;
		float *graph_edge_kernel_gpu;
		float *graph_edge_kernel_gpu16;
		float *graph_edge_kernel_updates_gpu;
		float *graph_edge_biases_gpu;
		float *graph_edge_bias_updates_gpu;
		float *graph_ref_gpu;
		float *graph_agg_gpu;
		float *graph_alpha_gpu;
		float *graph_valid_gpu;

		// Deformable Convolution (DCNv2) fields
		int use_mask;                    ///< 1=DCNv2 with modulation, 0=DCNv1 offsets only

		// Offset convolution (predicts 2*K*K offsets per output position)
		float *offset_weights;           ///< Offset conv weights: c * 2*K*K
		float *offset_weight_updates;
		float *offset_biases;            ///< Offset conv biases: 2*K*K
		float *offset_bias_updates;
		float *offsets;                  ///< Computed offsets: batch * out_h * out_w * 2*K*K
		float *offset_deltas;            ///< Gradients for offsets

		// DCNv2 mask convolution (predicts K*K modulation weights per output position)
		float *mask_weights;             ///< Mask conv weights: c * K*K
		float *mask_weight_updates;
		float *mask_biases;              ///< Mask conv biases: K*K
		float *mask_bias_updates;
		float *masks;                    ///< Computed masks: batch * out_h * out_w * K*K
		float *mask_deltas;              ///< Gradients for masks

		// GPU versions of deformable convolution arrays
		float *offset_weights_gpu;
		float *offset_weight_updates_gpu;
		float *offset_weights_gpu16;
		float *offset_weight_updates_gpu16;
		float *offset_biases_gpu;
		float *offset_bias_updates_gpu;
		float *offsets_gpu;
		float *offset_deltas_gpu;

		float *mask_weights_gpu;
		float *mask_weight_updates_gpu;
		float *mask_biases_gpu;
		float *mask_bias_updates_gpu;
		float *masks_gpu;
		float *mask_deltas_gpu;

		// DCNv4 specific fields
		float offset_scale;
		int remove_center;
		int d_stride;
		int block_thread;

		// Transformer (Swin-style) fields
		int tf_heads;				///< number of attention heads
		int tf_head_dim;			///< C / heads (derived)
		int tf_ffn_ratio;			///< FFN expansion ratio
		int tf_shift;				///< 0=no shift, 1=shift by size/2
		int tf_window_size;			///< window size (may differ from l.size if we repurpose)
		int tf_pad_h;				///< padded height (nearest multiple of window size)
		int tf_pad_w;				///< padded width

		// Transformer weights (CPU) — QKV uses l.weights / l.biases
		float *tf_wo;				float *tf_wo_updates;
		float *tf_wo_bias;			float *tf_wo_bias_updates;
		float *tf_ln1_gamma;		float *tf_ln1_gamma_updates;
		float *tf_ln1_beta;			float *tf_ln1_beta_updates;
		float *tf_ln2_gamma;		float *tf_ln2_gamma_updates;
		float *tf_ln2_beta;			float *tf_ln2_beta_updates;
		float *tf_ffn_w1;			float *tf_ffn_w1_updates;
		float *tf_ffn_b1;			float *tf_ffn_b1_updates;
		float *tf_ffn_w2;			float *tf_ffn_w2_updates;
		float *tf_ffn_b2;			float *tf_ffn_b2_updates;
		float *tf_rel_pos_bias;		float *tf_rel_pos_bias_updates;
		int   *tf_rel_pos_index;	///< [T, T] precomputed index table
		float *tf_res_proj;			float *tf_res_proj_updates;	///< [N, C] residual projection when C != N

		// Transformer runtime buffers (saved for backward)
		float *tf_qkv_out;			///< [B*nW, T, 3C]
		float *tf_attn_scores;		///< [B*nW, heads, T, T] post-softmax
		float *tf_attn_out;			///< [B*nW, T, C] pre-output-projection
		float *tf_ffn_hidden;		///< [B*nW*T, N*ratio] pre-activation
		float *tf_ln1_mean;			///< saved for LN backward
		float *tf_ln1_var;
		float *tf_ln2_mean;
		float *tf_ln2_var;
		float *tf_ln1_xhat;		///< normalized input (pre-scale) for LN1 backward
		float *tf_ln2_xhat;		///< normalized input for LN2 backward
		float *tf_pre_res1;			///< saved input for residual 1 backward
		float *tf_pre_res2;			///< saved input for residual 2 backward
		float *tf_windowed_input;	///< [B*nW, T, C] after window partition
		float *tf_attn_mask;		///< [nW, T, T] shift attention mask
		float *tf_workspace;		///< scratch workspace for allocation-free forward/backward
		size_t tf_workspace_size;	///< workspace size in floats

		// Transformer GPU arrays
		float *tf_wo_gpu;			float *tf_wo_updates_gpu;
		float *tf_wo_bias_gpu;		float *tf_wo_bias_updates_gpu;
		float *tf_ln1_gamma_gpu;	float *tf_ln1_gamma_updates_gpu;
		float *tf_ln1_beta_gpu;		float *tf_ln1_beta_updates_gpu;
		float *tf_ln2_gamma_gpu;	float *tf_ln2_gamma_updates_gpu;
		float *tf_ln2_beta_gpu;		float *tf_ln2_beta_updates_gpu;
		float *tf_ffn_w1_gpu;		float *tf_ffn_w1_updates_gpu;
		float *tf_ffn_b1_gpu;		float *tf_ffn_b1_updates_gpu;
		float *tf_ffn_w2_gpu;		float *tf_ffn_w2_updates_gpu;
		float *tf_ffn_b2_gpu;		float *tf_ffn_b2_updates_gpu;
		float *tf_rel_pos_bias_gpu;	float *tf_rel_pos_bias_updates_gpu;
		int   *tf_rel_pos_index_gpu;
		float *tf_res_proj_gpu;		float *tf_res_proj_updates_gpu;

		float *tf_qkv_out_gpu;
		float *tf_attn_scores_gpu;
		float *tf_attn_out_gpu;
		float *tf_ffn_hidden_gpu;
		float *tf_ln1_mean_gpu;
		float *tf_ln1_var_gpu;
		float *tf_ln2_mean_gpu;
		float *tf_ln2_var_gpu;
		float *tf_ln1_xhat_gpu;
		float *tf_ln2_xhat_gpu;
		float *tf_pre_res1_gpu;
		float *tf_pre_res2_gpu;
		float *tf_windowed_input_gpu;
		float *tf_attn_mask_gpu;
		float *tf_gpu_workspace;
		size_t tf_gpu_workspace_size;

		// ViT fields (config)
		int vit_patch_size;			// patch edge length P; token grid is H/P x W/P
		int vit_patch_stride;		// stride between patch centers (defaults to vit_patch_size)
		int vit_patch_pad;			// zero-padding around input (defaults to 0)
		int vit_heads;				// number of attention heads
		int vit_head_dim;			// filters / heads (derived)
		int vit_ffn_ratio;			// FFN expansion ratio
		int vit_mlp_dim;			// FFN hidden width; SimpleViT calls this mlp_dim
		int vit_pos_embed_type;		// 0=learned absolute, 1=fixed 2D sinusoidal
		int vit_pos_init_type;		// learned PE init: 0=small random, 1=zero

		// ViT weights (CPU)
		float *vit_patch_embed;		float *vit_patch_embed_updates;
		float *vit_patch_bias;		float *vit_patch_bias_updates;
		float *vit_wo;				float *vit_wo_updates;
		float *vit_wo_bias;			float *vit_wo_bias_updates;
		float *vit_ln1_gamma;		float *vit_ln1_gamma_updates;
		float *vit_ln1_beta;		float *vit_ln1_beta_updates;
		float *vit_ln2_gamma;		float *vit_ln2_gamma_updates;
		float *vit_ln2_beta;		float *vit_ln2_beta_updates;
		float *vit_ffn_w1;			float *vit_ffn_w1_updates;
		float *vit_ffn_b1;			float *vit_ffn_b1_updates;
		float *vit_ffn_w2;			float *vit_ffn_w2_updates;
		float *vit_ffn_b2;			float *vit_ffn_b2_updates;
		float *vit_pos_embed;		float *vit_pos_embed_updates;

		// ViT runtime buffers (saved for backward pass)
		float *vit_qkv_out;			// [B, T, 3C]
		float *vit_attn_scores;		// [B, heads, T, T] post-softmax
		float *vit_attn_out;		// [B, T, C] pre-projection
		float *vit_ffn_hidden;		// [B*T, mlp_dim] pre-activation
		float *vit_ln1_mean;		// saved for LN backward
		float *vit_ln1_var;
		float *vit_ln2_mean;
		float *vit_ln2_var;
		float *vit_ln1_xhat;		// normalized input for LN1 backward
		float *vit_ln2_xhat;		// normalized input for LN2 backward
		float *vit_pre_res1;		// saved input for residual 1 backward
		float *vit_pre_res2;		// saved input for residual 2 backward

		// ViT GPU counterparts
		float *vit_patch_embed_gpu;		float *vit_patch_embed_updates_gpu;
		float *vit_patch_bias_gpu;		float *vit_patch_bias_updates_gpu;
		float *vit_wo_gpu;				float *vit_wo_updates_gpu;
		float *vit_wo_bias_gpu;			float *vit_wo_bias_updates_gpu;
		float *vit_ln1_gamma_gpu;		float *vit_ln1_gamma_updates_gpu;
		float *vit_ln1_beta_gpu;		float *vit_ln1_beta_updates_gpu;
		float *vit_ln2_gamma_gpu;		float *vit_ln2_gamma_updates_gpu;
		float *vit_ln2_beta_gpu;		float *vit_ln2_beta_updates_gpu;
		float *vit_ffn_w1_gpu;			float *vit_ffn_w1_updates_gpu;
		float *vit_ffn_b1_gpu;			float *vit_ffn_b1_updates_gpu;
		float *vit_ffn_w2_gpu;			float *vit_ffn_w2_updates_gpu;
		float *vit_ffn_b2_gpu;			float *vit_ffn_b2_updates_gpu;
		float *vit_pos_embed_gpu;		float *vit_pos_embed_updates_gpu;

		float *vit_qkv_out_gpu;
		float *vit_attn_scores_gpu;
		float *vit_attn_out_gpu;
		float *vit_ffn_hidden_gpu;
		float *vit_ln1_mean_gpu;
		float *vit_ln1_var_gpu;
		float *vit_ln2_mean_gpu;
		float *vit_ln2_var_gpu;
		float *vit_ln1_xhat_gpu;
		float *vit_ln2_xhat_gpu;
		float *vit_pre_res1_gpu;
		float *vit_pre_res2_gpu;
		float *vit_patch_tokens_gpu;
		float *vit_patch_delta_gpu;
		float *vit_tmp_token_c1_gpu;
		float *vit_tmp_token_c2_gpu;
		float *vit_tmp_token_n1_gpu;
		float *vit_tmp_token_n2_gpu;
		float *vit_tmp_token_n3_gpu;
		float *vit_tmp_ffn_hidden_gpu;
		float *vit_tmp_head1_gpu;
		float *vit_tmp_head2_gpu;
		float *vit_tmp_head3_gpu;
		float *vit_tmp_head4_gpu;
		float *vit_tmp_head5_gpu;
		float *vit_tmp_scores_gpu;

		int tucker_heads;
		int tucker_head_dim;
		int tucker_rank_q;
		int tucker_rank_k;
		int tucker_rank_v;
		int tucker_rank_o;
		int tucker_window_size;
		int tucker_pad_h;
		int tucker_pad_w;

		float *tucker_q_latent;
		float *tucker_k_latent;
		float *tucker_v_latent;
		float *tucker_q;
		float *tucker_k;
		float *tucker_v;
		float *tucker_scores;
		float *tucker_context;
		float *tucker_windowed_input;

		float *tucker_q_latent_gpu;
		float *tucker_k_latent_gpu;
		float *tucker_v_latent_gpu;
		float *tucker_q_gpu;
		float *tucker_k_gpu;
		float *tucker_v_gpu;
		float *tucker_scores_gpu;
		float *tucker_context_gpu;
		float *tucker_windowed_input_gpu;
		float *tucker_gpu_input_cpu;

		// Tucker attention FP8: opt-in, additive quantization of the Q@K^T and attn@V
		// batched GEMMs only.  Softmax stays FP32.  See tucker_attention_kernels.cu's
		// DARKNET_TUCKER_USE_CUBLAS_HALF forward path.
		int fp8_tucker_attention;
		int fp8_tucker_scales_loaded;

		// Scores GEMM (Q@K^T) is wired as A=K (row-padded to key_pad, the T dimension
		// rounded up to a multiple of 16), B=Q (col-padded only, T is B's unconstrained
		// dimension) -- this makes the column-major D output byte-identical to a
		// row-major (T_query, key_pad) buffer, so only a small compaction+cast kernel
		// (drop the key_pad-T garbage columns) is needed to land in the existing tightly
		// packed FP16 scores buffer feeding cudnnSoftmaxForward unchanged.
		char *fp8_tucker_q_gpu;      // FP8 quantized Q -- B operand, row-major (T, D), col-padded only
		char *fp8_tucker_k_gpu;      // FP8 quantized K -- A operand, row-major (key_pad, D)
		char *fp8_tucker_attn_gpu;   // FP8 quantized post-softmax attn -- B operand, row-major (T, key_pad)
		char *fp8_tucker_v_t_gpu;    // FP8 quantized V, transposed -- A operand, row-major (D, key_pad)

		float *fp8_tucker_q_amax_gpu;      float *fp8_tucker_q_scale_gpu;
		float *fp8_tucker_k_amax_gpu;      float *fp8_tucker_k_scale_gpu;
		float *fp8_tucker_attn_amax_gpu;   float *fp8_tucker_attn_scale_gpu;
		float *fp8_tucker_v_amax_gpu;      float *fp8_tucker_v_scale_gpu;

		float fp8_tucker_q_amax_host;      float fp8_tucker_q_scale_host;
		float fp8_tucker_k_amax_host;      float fp8_tucker_k_scale_host;
		float fp8_tucker_attn_amax_host;   float fp8_tucker_attn_scale_host;
		float fp8_tucker_v_amax_host;      float fp8_tucker_v_scale_host;

		void *fp8_tucker_scores_gemm_plan;   // Darknet::Fp8GemmPlan* for Q@K^T (A=K, B=Q)
		void *fp8_tucker_context_gemm_plan;  // Darknet::Fp8GemmPlan* for attn@V (A=V^T, B=attn)
		float *fp8_tucker_scores_out_gpu;    // raw FP32 GEMM1 output, column-major (key_pad, T) per batch item
		float *fp8_tucker_context_out_gpu;   // raw FP32 GEMM2 output, column-major (D, T) per batch item == row-major (T, D)
		void *fp8_tucker_lt_workspace_gpu;   // shared cuBLASLt scratch for both GEMMs

		// Tucker attention FP4: opt-in, additive, diagnostic-only (fp4_gemm_execute()
		// has no external scale pointer -- NVFP4 block-scale is computed internally
		// per call, same limitation already accepted for conv FP4).  Unlike FP8, the
		// GEMM API takes plain FP32 row-major operands directly and quantizes them
		// itself, so no persistent quantized buffers are needed -- only FP32 scratch
		// for the __half -> float conversion.  Only 2 plan shapes exist across all
		// six GEMM roles (forward Q@K^T/attn@V, backward dAttn/dV/dQ/dK), reused by
		// shape rather than one plan per role.  See tucker_attention_kernels.cu.
		int fp4_tucker_attention;

		void *fp4_tucker_scores_gemm_plan;    // Darknet::Fp4GemmPlan*, shape (T, T, D):     Q@K^T fwd, dContext@V^T bwd
		void *fp4_tucker_context_gemm_plan;   // Darknet::Fp4GemmPlan*, shape (T, D, key_pad): attn@V fwd, dV/dQ/dK bwd
		void *fp4_tucker_scores_lt_workspace;
		void *fp4_tucker_context_lt_workspace;

		float *fp4_tucker_a_gpu;    // FP32 scratch, batched (T, D) or (T, key_pad) -- the un-transposed operand
		float *fp4_tucker_b_gpu;    // FP32 scratch, batched (T, D) or (D, key_pad) -- the transposed operand
		float *fp4_tucker_out_gpu;  // FP32 scratch, batched (T, T) or (T, D) -- raw GEMM output before cast to half

		// MambaVision fields (config)
		int mv_d_state;
		int mv_dt_rank;
		int mv_conv_size;
		int mv_ffn_ratio;

		// MambaVision weights (CPU).  The input projection uses l.weights/l.biases.
		float *mv_conv_x;			float *mv_conv_x_updates;
		float *mv_conv_x_bias;		float *mv_conv_x_bias_updates;
		float *mv_conv_z;			float *mv_conv_z_updates;
		float *mv_conv_z_bias;		float *mv_conv_z_bias_updates;
		float *mv_x_proj;			float *mv_x_proj_updates;
		float *mv_dt_proj;			float *mv_dt_proj_updates;
		float *mv_dt_bias;			float *mv_dt_bias_updates;
		float *mv_A_log;			float *mv_A_log_updates;
		float *mv_D;				float *mv_D_updates;
		float *mv_out_proj;			float *mv_out_proj_updates;
		float *mv_out_bias;			float *mv_out_bias_updates;
		float *mv_res_proj;			float *mv_res_proj_updates;
		float *mv_ln1_gamma;		float *mv_ln1_gamma_updates;
		float *mv_ln1_beta;			float *mv_ln1_beta_updates;
		float *mv_ln2_gamma;		float *mv_ln2_gamma_updates;
		float *mv_ln2_beta;			float *mv_ln2_beta_updates;
		float *mv_ffn_w1;			float *mv_ffn_w1_updates;
		float *mv_ffn_b1;			float *mv_ffn_b1_updates;
		float *mv_ffn_w2;			float *mv_ffn_w2_updates;
		float *mv_ffn_b2;			float *mv_ffn_b2_updates;

		// MambaVision runtime buffers (saved for backward)
		float *mv_tokens;
		float *mv_ln1_out;
		float *mv_ln1_mean;
		float *mv_ln1_var;
		float *mv_ln1_xhat;
		float *mv_in_proj_out;
		float *mv_x_conv_pre;
		float *mv_x_conv;
		float *mv_z_conv_pre;
		float *mv_z_conv;
		float *mv_x_proj_out;
		float *mv_dt_pre;
		float *mv_dt;
		float *mv_scan_state;
		float *mv_scan_out;
		float *mv_mixer_cat;
		float *mv_mixer_out;
		float *mv_pre_res2;
		float *mv_ln2_out;
		float *mv_ln2_mean;
		float *mv_ln2_var;
		float *mv_ln2_xhat;
		float *mv_ffn_hidden;

		// MambaVision GPU buffers
		float *mv_gpu_input_cpu;
		float *mv_A_log_gpu;				float *mv_A_log_updates_gpu;
		float *mv_D_gpu;					float *mv_D_updates_gpu;
		float *mv_ln1_gamma_gpu;			float *mv_ln1_gamma_updates_gpu;
		float *mv_ln1_beta_gpu;			float *mv_ln1_beta_updates_gpu;
		float *mv_ln2_gamma_gpu;			float *mv_ln2_gamma_updates_gpu;
		float *mv_ln2_beta_gpu;			float *mv_ln2_beta_updates_gpu;

		float *mv_tokens_gpu;
		float *mv_ln1_out_gpu;
		float *mv_ln1_mean_gpu;
		float *mv_ln1_var_gpu;
		float *mv_ln1_xhat_gpu;
		float *mv_dt_pre_gpu;
		float *mv_dt_gpu;
		float *mv_scan_state_gpu;
		float *mv_scan_out_gpu;
		float *mv_mixer_cat_gpu;
		float *mv_pre_res2_gpu;
		float *mv_ln2_out_gpu;
		float *mv_ln2_mean_gpu;
		float *mv_ln2_var_gpu;
		float *mv_ln2_xhat_gpu;
		float *mv_tmp_token_c_gpu;
		float *mv_tmp_token_n_gpu;
		float *mv_tmp_token_p_gpu;
		float *mv_tmp_bdt_gpu;
		float *mv_tmp_bdt2_gpu;
		float *mv_tmp_ffn_gpu;

		// Clifford / Geometric Algebra fields
		int cli_num_shifts;
		int *cli_shifts;
		int *cli_shifts_gpu;
		int cli_num_shifts_inner;
		int *cli_shifts_inner;
		int *cli_shifts_inner_gpu;
			int cli_ctx_mode;
			int cli_interaction_mode;
			int cli_gffn_mode;
			int cli_higher_mode;
			int cli_proj_in_dim;
		float cli_drop_path;
		float cli_layerscale_init;

		int cli_num_dwconv;
		Layer *cli_dwconv;

		float *cli_w_det;               float *cli_w_det_updates;
		float *cli_b_det;               float *cli_b_det_updates;
		float *cli_w_proj;              float *cli_w_proj_updates;
		float *cli_b_proj;              float *cli_b_proj_updates;
		float *cli_w_gate;              float *cli_w_gate_updates;
		float *cli_b_gate;              float *cli_b_gate_updates;
		float *cli_ln_gamma;            float *cli_ln_gamma_updates;
		float *cli_ln_beta;             float *cli_ln_beta_updates;
		float *cli_layer_scale;         float *cli_layer_scale_updates;

		float *cli_w_proj_g;            float *cli_w_proj_g_updates;
		float *cli_b_proj_g;            float *cli_b_proj_g_updates;
		float *cli_w_gate_g;            float *cli_w_gate_g_updates;
		float *cli_b_gate_g;            float *cli_b_gate_g_updates;

		float *cli_ln_out;
		float *cli_ln_mean;
		float *cli_ln_var;
		float *cli_ln_xhat;
		float *cli_z_det;
		float *cli_z_ctx;
		float *cli_z_ctx_pre_diff;
		float *cli_g_raw;
			float *cli_g_feat;
			float *cli_gate_alpha;
			float *cli_gate_pre_sigmoid;
			float *cli_vb_feat;
			float *cli_hmix;
		float *cli_drop_mask;

		float *cli_global_ctx;
		float *cli_g_raw_g;
		float *cli_g_feat_g;
		float *cli_gate_alpha_g;
		float *cli_gate_pre_sigmoid_g;

		float *cli_w_det_gpu;           float *cli_w_det_updates_gpu;
		float *cli_b_det_gpu;           float *cli_b_det_updates_gpu;
		float *cli_w_proj_gpu;          float *cli_w_proj_updates_gpu;
		float *cli_b_proj_gpu;          float *cli_b_proj_updates_gpu;
		float *cli_w_gate_gpu;          float *cli_w_gate_updates_gpu;
		float *cli_b_gate_gpu;          float *cli_b_gate_updates_gpu;
		float *cli_ln_gamma_gpu;        float *cli_ln_gamma_updates_gpu;
		float *cli_ln_beta_gpu;         float *cli_ln_beta_updates_gpu;
		float *cli_layer_scale_gpu;     float *cli_layer_scale_updates_gpu;

		float *cli_w_proj_g_gpu;        float *cli_w_proj_g_updates_gpu;
		float *cli_b_proj_g_gpu;        float *cli_b_proj_g_updates_gpu;
		float *cli_w_gate_g_gpu;        float *cli_w_gate_g_updates_gpu;
		float *cli_b_gate_g_gpu;        float *cli_b_gate_g_updates_gpu;

		float *cli_ln_out_gpu;
		float *cli_ln_mean_gpu;
		float *cli_ln_var_gpu;
		float *cli_ln_xhat_gpu;
		float *cli_z_det_gpu;
		float *cli_z_ctx_gpu;
		float *cli_z_ctx_pre_diff_gpu;
		float *cli_g_raw_gpu;
			float *cli_g_feat_gpu;
			float *cli_gate_alpha_gpu;
			float *cli_gate_pre_sigmoid_gpu;
			float *cli_vb_feat_gpu;
			float *cli_hmix_gpu;
		float *cli_drop_mask_gpu;

		float *cli_global_ctx_gpu;
		float *cli_g_raw_g_gpu;
		float *cli_g_feat_g_gpu;
		float *cli_gate_alpha_g_gpu;
		float *cli_gate_pre_sigmoid_g_gpu;

		// Recursive block (rb_*): arbitrary body applied N times with shared weights
		Layer * rb_body;           // array of body sub-layers [rb_body_count]
		int     rb_body_count;     // number of body sub-layers
		int     rb_loops;          // forward loop count (backward only through final pass)
		float   rb_body_scale;     // scale applied to body output F(h)
		float   rb_residual_scale; // scale applied to recurrent residual h
		float   rb_injection_scale;// scale applied to original input injection e
		float * rb_last_input;     // h_{T-1}: input snapshot to the final body pass (for backward)
		float * rb_last_input_gpu; // GPU mirror of rb_last_input

		// --- recursive_block Ouroboros Stage 1 + Stage 2 fields ---
		int   rb_ouroboros;              // 0=legacy static, 1=Stage 1 FiLM+gate, 2=Stage 2 Conv-LoRA + FiLM+gate
		int   rb_controller_input_c;     // CPU cache: c + 1
		int   rb_controller_output_c;    // CPU cache: 2*max(c,out_c) + Stage 2 LoRA diag count
		int   rb_controller_gpu_input_c; // GPU allocation cache
		int   rb_controller_gpu_output_c;// GPU allocation cache
		float rb_gate_bias;              // default -2.0
		float rb_gamma_clip;             // default 4.0; <=0 disables clamp
		float rb_gate_clip;              // default 30.0; <=0 disables clamp

		float * rb_controller_input;     // batch * rb_controller_input_c
		float * rb_controller_output;    // batch * rb_controller_output_c
		float * rb_controller_delta;     // batch * rb_controller_output_c
		float * rb_candidate;            // batch * max(inputs, outputs)
		float * rb_candidate_delta;      // batch * max(inputs, outputs)
		int     rb_candidate_size;       // CPU allocated element count for candidate buffers

#ifdef DARKNET_GPU
		float * rb_controller_input_gpu;
		float * rb_controller_output_gpu;
		float * rb_controller_delta_gpu;
		float * rb_candidate_gpu;
		float * rb_candidate_delta_gpu;
		int     rb_candidate_gpu_size;   // GPU allocated element count for candidate buffers
#endif

		// Stage 2 Conv-LoRA
		int   rb_lora_rank;                   // e.g. 8 or 16; 0 disables LoRA
		float rb_lora_alpha;                  // LoRA scaling; effective = alpha / rank
		int   rb_lora_freeze_base;            // skip base body updates when true
		float rb_lora_diag_clip;              // default 4.0; <=0 disables clamp on diag values
		float rb_lora_diag_init;              // default 1.0; controller diag bias at init
		int   rb_lora_adapters;               // number of eligible conv body layers wrapped
		int   rb_lora_total_rank;             // rb_lora_adapters * rb_lora_rank
		int   rb_lora_diag_offset;            // controller output index where LoRA diag values start
		int   rb_lora_configured_body_count;  // resize/config cache
		int             * rb_lora_body_adapter;   // [rb_body_count] body layer -> adapter index (-1 = none)
		int             * rb_lora_body_indices;   // [rb_lora_adapters] adapter -> body layer index
		int             * rb_lora_rank_offsets;   // [rb_lora_adapters] adapter -> diag rank offset
		Darknet::Layer  * rb_lora_A;              // [rb_lora_adapters] Conv_A: input -> rank
		Darknet::Layer  * rb_lora_B;              // [rb_lora_adapters] Conv_B: rank -> out_c
		float          ** rb_lora_scaled;          // [rb_lora_adapters] host: diag_t * Conv_A(x)
		float          ** rb_lora_scaled_delta;    // [rb_lora_adapters] host: dL/d scaled
		int             * rb_lora_scaled_sizes;    // [rb_lora_adapters] allocated element counts

#ifdef DARKNET_GPU
		float          ** rb_lora_scaled_gpu;       // [rb_lora_adapters] device pointers (host array)
		float          ** rb_lora_scaled_delta_gpu; // [rb_lora_adapters] device pointers (host array)
		int             * rb_lora_scaled_gpu_sizes; // [rb_lora_adapters] allocated element counts
#endif

#ifdef CUDNN
		cudnnTensorDescriptor_t srcTensorDesc, dstTensorDesc;
		cudnnTensorDescriptor_t srcTensorDesc16, dstTensorDesc16;
		cudnnTensorDescriptor_t dsrcTensorDesc, ddstTensorDesc;
		cudnnTensorDescriptor_t dsrcTensorDesc16, ddstTensorDesc16;
		cudnnTensorDescriptor_t normTensorDesc, normDstTensorDesc, normDstTensorDescF16;
		cudnnFilterDescriptor_t weightDesc, weightDesc16;
		cudnnFilterDescriptor_t dweightDesc, dweightDesc16;
		cudnnConvolutionDescriptor_t convDesc;
		cudnnConvolutionFwdAlgo_t fw_algo, fw_algo16;
		cudnnConvolutionBwdDataAlgo_t bd_algo, bd_algo16;
		cudnnConvolutionBwdFilterAlgo_t bf_algo, bf_algo16;
		cudnnPoolingDescriptor_t poolingDesc;
#else
		// pad the structure so it has the same size regardless of CPU, CUDA, CUDNN, etc.
		void* srcTensorDesc, *dstTensorDesc;
		void* srcTensorDesc16, *dstTensorDesc16;
		void* dsrcTensorDesc, *ddstTensorDesc;
		void* dsrcTensorDesc16, *ddstTensorDesc16;
		void* normTensorDesc, *normDstTensorDesc, *normDstTensorDescF16;
		void* weightDesc, *weightDesc16;
		void* dweightDesc, *dweightDesc16;
		void* convDesc;
		UNUSED_ENUM_TYPE fw_algo, fw_algo16;
		UNUSED_ENUM_TYPE bd_algo, bd_algo16;
		UNUSED_ENUM_TYPE bf_algo, bf_algo16;
		void* poolingDesc;
#endif  // CUDNN
	};
}


#include "avgpool_layer.hpp"
#include "batchnorm_layer.hpp"
#include "channel_shuffle_layer.hpp"
#include "channel_slice_layer.hpp"
#include "connected_layer.hpp"
#include "convolutional_layer.hpp"
#include "cost_layer.hpp"
#include "crnn_layer.hpp"
#include "deconvolutional_layer.hpp"
#include "dropout_layer.hpp"
#include "gaussian_yolo_layer.hpp"
#include "lstm_layer.hpp"
#include "maxpool_layer.hpp"
#include "region_layer.hpp"
#include "reorg_layer.hpp"
#include "rnn_layer.hpp"
#include "route_layer.hpp"
#include "sam_layer.hpp"
#include "scale_channels_layer.hpp"
#include "shortcut_layer.hpp"
#include "softmax_layer.hpp"
#include "transformer_layer.hpp"
#include "clifford_layer.hpp"
#include "vit_layer.hpp"
#include "mambavision_layer.hpp"
#include "tucker_attention_layer.hpp"
#include "upsample_layer.hpp"
#include "yolo_layer.hpp"
#include "detr_decoder_layer.hpp"
#include "graph_conv_layer.hpp"
#include "deform_conv_layer.hpp"
#include "dcnv4_layer.hpp"
#include "recursive_block_layer.hpp"
#include "wmhf_layer.hpp"
