#pragma once

#include "darknet_internal.hpp"

/**
 * @file detr_decoder_layer.hpp
 * @brief DETR-style query-based detection decoder head (RF-DETR substrate, v1).
 *
 * This is darknet's first *sparse set-prediction* detection head. Unlike the dense
 * anchor / anchor-free YOLO heads, it carries a fixed set of learnable object queries
 * that first self-attend to each other (so queries can specialize / de-duplicate
 * against one another), then cross-attend to the incoming feature map ("memory"),
 * and each emit one (class-logits, box) prediction. Training uses an optimal
 * (Hungarian / Kuhn-Munkres) one-to-one matcher between the Q predictions and the
 * ground-truth boxes, with a sigmoid **focal** classification loss (Deformable-DETR /
 * RF-DETR; gamma=2, alpha=0.25) plus L1 + GIoU box loss. Focal loss down-weights the many
 * easy background negatives so the sparse matched positives are not swamped.
 *
 * ============================================================================
 * SCOPE OF THIS VERSION (v1)
 * ============================================================================
 *  - Multi-head self-attention among the Q query embeddings (`heads` heads, each of
 *    dimension D/heads), residual-added onto the raw embedding, followed by a single
 *    cross-attention block (scaled dot-product) + FFN + cls/box heads.
 *  - Sinusoidal (non-learned) positional encoding added to the memory keys; the
 *    learnable query embedding doubles as the query positional code.
 *  - Learnable per-query **reference points** (RF-DETR / DAB-DETR spatial prior): each query
 *    owns a 4-vector added to the box pre-activation, box = sigmoid(Wb*ffn + bb + ref_q).
 *    cx,cy are grid-spread at init so queries start at distinct locations (a YOLO-like prior
 *    that speeds localization); still one-to-one set prediction / anchor-free (no tiled anchors).
 *  - Backbone features receive gradient on both the CPU and GPU paths (`state.delta`
 *    is accumulated into during `backward_detr_decoder_layer`/`_gpu`), so decoder and
 *    backbone train jointly. Self-attention among queries never touches the backbone
 *    gradient -- it is purely a function of the query embeddings and their weights.
 *  - The GPU path is a real CUDA implementation (`detr_decoder_kernels.cu`, GEMM +
 *    custom elementwise/softmax kernels) for forward/backward; only the matching/loss
 *    computation (`detr_decoder_loss`) runs on the host.
 *
 * Planned next steps (see plan): multi-layer decoder with auxiliary losses,
 * multi-scale deformable cross-attention (built on the DCNv4 bilinear sampler).
 *
 * ============================================================================
 * USAGE IN .cfg FILES
 * ============================================================================
 *
 *   [detr_decoder]
 *   queries=100            # number of object queries (set size)
 *   classes=80            # number of object classes
 *   heads=8               # self-attention heads among queries (D must be divisible by heads)
 *   ffn=512               # FFN hidden dimension
 *   max_boxes=90          # max GT boxes per image
 *   cls_weight=1.0        # classification loss weight
 *   l1_weight=5.0         # L1 box loss weight
 *   giou_weight=2.0       # GIoU loss weight (combined with L1 for box regression)
 *   noobj_weight=1.0      # unmatched-query class-negative weight
 *
 * The input feature map provides the model dimension D = input channels, and the
 * memory tokens N = input_h * input_w.
 */

Darknet::Layer make_detr_decoder_layer(int batch, int h, int w, int c,
		int queries, int classes, int heads, int ffn, int max_boxes,
		float cls_weight, float l1_weight, float giou_weight, float noobj_weight,
		int index, int train);

void forward_detr_decoder_layer(Darknet::Layer & l, Darknet::NetworkState state);
void backward_detr_decoder_layer(Darknet::Layer & l, Darknet::NetworkState state);
void update_detr_decoder_layer(Darknet::Layer & l, int batch, float learning_rate, float momentum, float decay);
void resize_detr_decoder_layer(Darknet::Layer * l, int w, int h);

void save_detr_decoder_weights(const Darknet::Layer & l, FILE * fp);
size_t load_detr_decoder_weights(Darknet::Layer & l, FILE * fp);

/** Hungarian-matched set loss shared by the CPU and GPU forward paths.
 * Reads predictions from @p l.output, writes the (negative-gradient) output-space
 * deltas into @p l.delta, and returns the total scalar loss. @p truth is the base
 * pointer of the per-image ground-truth boxes (batch-major, stride @p l.truths).
 */
float detr_decoder_loss(Darknet::Layer & l, const float * truth);

int detr_decoder_num_detections(const Darknet::Layer & l, float thresh);
int detr_decoder_num_detections_batch(const Darknet::Layer & l, float thresh, int batch);
int get_detr_decoder_detections(const Darknet::Layer & l, int w, int h, int netw, int neth, float thresh, int * map, int relative, Darknet::Detection * dets, int letter);
int get_detr_decoder_detections_batch(const Darknet::Layer & l, int w, int h, int netw, int neth, float thresh, int * map, int relative, Darknet::Detection * dets, int letter, int batch);

#ifdef DARKNET_GPU
void detr_decoder_setup_gpu(Darknet::Layer & l);		///< allocate GPU params + scratch, push initial weights
void detr_decoder_resize_gpu(Darknet::Layer & l);		///< reallocate token-dependent GPU scratch
void forward_detr_decoder_layer_gpu(Darknet::Layer & l, Darknet::NetworkState state);
void backward_detr_decoder_layer_gpu(Darknet::Layer & l, Darknet::NetworkState state);
void update_detr_decoder_layer_gpu(Darknet::Layer & l, int batch, float learning_rate, float momentum, float decay, float loss_scale);
#endif
