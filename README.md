# Darknet `deform` Branch

This branch is a research/development branch on top of Darknet V6 "Winston". It adds newer spatial, sequence, graph, transformer, and image-warp layers while keeping the normal Darknet training and inference workflow.

The important rule for this branch is compatibility: existing YOLO configs should keep using standard Darknet data loading, YOLO heads, anchors, losses, NMS, and detection APIs unless a config explicitly inserts an experimental layer.

## What Is In This Branch

- YOLO-compatible Telescope-style hyperbolic foveation: `[hyperbolic_foveation]`
- MambaVision mixer blocks: `[mambavision]`, `[mambavision_block]`
- Deformable convolution variants: `[deform_conv]`, `[deformable_convolutional]`, `[dcnv3]`, `[dcnv4]`
- Transformer and ViT blocks: `[transformer]`, `[transformer_block]`, `[vit]`
- Graph, Clifford, Tucker, EML, and recursive-block experiments
- CUDA training paths for the layers currently being validated

Expected local branch:

```sh
git branch --show-current
# deform
```

## Build

Use the normal CMake build:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

For a focused library check while developing layers:

```sh
cmake --build build --target darknetobjlib -j2
```

For CUDA training, build with CUDA enabled in the same way this workspace is configured. The current development environment has been tested on Ubuntu/WSL with an NVIDIA RTX 4060 Laptop GPU.

## Run

Standard Darknet commands still apply:

```sh
./build/src-cli/darknet detector train <data.data> <cfg> <initial.weights>
./build/src-cli/darknet detector map <data.data> <cfg> <weights>
./build/src-cli/darknet detector test <data.data> <cfg> <weights> <image>
```

Example configs in this branch:

| Config | Purpose |
| --- | --- |
| `cfg/yolov4_tiny_hf.cfg` | YOLOv4-tiny with Telescope-style foveation before the backbone. |
| `cfg/yolov4_tiny_mamba.cfg` | YOLOv4-tiny with MambaVision blocks. |
| `cfg/yolov4_tiny_tucker.cfg` | YOLOv4-tiny with Tucker attention. |
| `cfg/yolov4_tiny_eml.cfg` | YOLOv4-tiny with EML convolution. |
| `cfg/yolov4-tiny-recursive-block.cfg` | YOLOv4-tiny with recursive block structure. |
| `cfg/LegoGears_clifford.cfg` | Clifford-layer object detection experiment. |

## Telescope-Style Foveation Layer

`[hyperbolic_foveation]` is a YOLO-compatible image-warp layer inspired by Telescope. It does not replace YOLO with Deformable DETR and does not change YOLO output tensors, anchors, classes, NMS, or public detection APIs.

Instead, the layer keeps YOLO unaware of the foveation internals:

1. The dataloader provides the original image and original YOLO labels.
2. The foveation layer predicts or uses warp parameters.
3. The image is warped into the same network input size, for example `416x416`.
4. Training labels are rewritten into the warped image coordinate system before YOLO loss runs.
5. YOLO trains normally on the warped image and warped labels.
6. During inference, YOLO predictions are inverse-warped back into original image coordinates before output/NMS handling.

This is why the layer transforms both pixels and GT boxes. Warping only the image would make YOLO train against labels that no longer cover the object pixels.

### Foveation Geometry

The layer uses normalized `[-1,1]` coordinates internally. For each box:

```text
original box center: c = (x, y)
warped center:       c' = Phi(c)
warped size:         local Jacobian J_Phi(c) applied to the box width/height axes
```

Inference applies the inverse:

```text
YOLO warped prediction center: c'
original center:               c = Phi_inverse(c')
original size:                 inverse Jacobian at c applied to predicted width/height axes
```

If `strength=0`, the layer has an explicit identity bypass: image data is copied, the sampling grid is identity, and boxes are returned unchanged. This protects standard YOLO behavior when no transform is applied.

### Foveation Parameters

The layer currently supports four warp parameters:

| Parameter | Meaning |
| --- | --- |
| `cx`, `cy` | Foveation center in normalized `[-1,1]` coordinates. |
| `R` | Radius of the foveated region in normalized coordinates. |
| `strength` | Warp gate in `[0,1]`; `0` is identity, larger values apply more foveation. |

`alpha` and `p_exp` are fixed cfg parameters controlling the shape of the radial map. This is still a YOLO-compatible Telescope-inspired implementation, not a full reproduction of the Telescope detector architecture.

### Foveation CFG Keys

Typical section:

```ini
[hyperbolic_foveation]
params=learned
dontload=1
init_cx=0.0
init_cy=-0.1
init_radius=0.75
init_strength=0.05
small_box_px=14
small_box_strength=0.85
strength_loss=1.0
tiny_center_loss=1.0
tiny_radius_loss=0.25
tiny_strength_loss=1.0
strength_zero_loss=1.0
hf_lr_mult=1.0
debug=1
debug_interval=100
alpha=2.0
p_exp=2.0
downsample=8
pred_hidden=64
inverse_method=newton
pred_source_layer=-1
warp_source=previous
nr_iters=10
nr_eta=0.5
```

Key behavior:

| Key | Default | Meaning |
| --- | --- | --- |
| `params` | `global`/empty | `fixed` uses global params, `global`/`trainable` trains global params, `learned` predicts per-image params. |
| `pred_hidden` | `0`, or `64` for `params=learned` | Hidden width of the per-image parameter predictor. |
| `pred_source_layer` | `-1` | Predictor source. `-1` uses the incoming tensor; other values reference an earlier layer. |
| `warp_source` | `previous` | `previous` warps the incoming tensor; `network_input` predicts from a branch but warps the original image. |
| `inverse_method` | `newton` | `newton` uses the analytic Jacobian; `fixed_point` uses the paper Eq. 4 residual update. |
| `small_box_px` | `14` | A GT box is tiny only when both width and height are below this pixel threshold. |
| `small_box_strength` | `0.85` | Tiny-box target value for `strength`. |
| `strength_loss` | `1.0` | Backward-compatible alias used as the default for tiny center/strength losses. |
| `tiny_center_loss` | `strength_loss` | Auxiliary target weight for `cx` and `cy`. |
| `tiny_radius_loss` | `0.25 * strength_loss` | Auxiliary target weight for `R`. |
| `tiny_strength_loss` | `strength_loss` | Auxiliary target weight for moving `strength` toward `small_box_strength` on tiny-box images. |
| `strength_zero_loss` | `0.0` | Regularizer that prefers `strength=0`; set in `cfg/yolov4_tiny_hf.cfg` to make warping justify itself. |
| `hf_lr_mult` | `1.0` | Learning-rate multiplier for foveation parameters/predictor weights. |
| `debug` | `0` | Enables aggregate foveation diagnostics. |
| `debug_interval` | `100` | Print debug summaries every N training iterations when `debug=1`. |
| `nr_iters`, `nr_eta` | `10`, `0.5` | Iteration count and step for inverse point solve. |

Optional head conditioning can be added with `[foveation_film]` before a YOLO head or head-side convolution:

```ini
[foveation_film]
embed_dim=256
backprop_to_hf=1
film_init_scale=0.0
```

The layer maps foveation parameters to per-channel FiLM scale and shift. With `film_init_scale=0.0`, it starts as an exact identity layer.

Tiny-box auxiliary gradients are normalized by the number of batch images that contain tiny boxes, so update size is less sensitive to batch composition.

### Debug Output

With `debug=1`, the layer prints periodic aggregate summaries instead of per-image spam. The summary includes:

- mean/std/min/max of `cx`, `cy`, `R`, and `strength`
- number of tiny boxes and tiny-box images in the batch
- tiny target mean
- parameter-target RMSE
- approximate magnification at tiny-box centers

Use this to verify that `params=learned` is not collapsing to constant values and that tiny-box target error trends down during training.

## Layer Status

Status means the state of this `deform` branch, not upstream Darknet in general.

| Layer / cfg section | Status | Notes |
| --- | --- | --- |
| `[convolutional]`, `[conv]` | Works | Core Darknet convolution. Supports grouped/depthwise usage through `groups`. |
| `[connected]`, `[conn]` | Works | Core fully connected layer. Used directly and as sublayers in newer modules. |
| `[route]` | Works | Core YOLO graph wiring and tensor concatenation/slicing. |
| `[shortcut]` | Works | Core residual/skip wiring. |
| `[maxpool]`, `[max]` | Works | Core YOLO pooling. |
| `[upsample]` | Works | Core YOLO upsample/downsample path. |
| `[yolo]` | Works | Standard YOLO detection head. The foveation work intentionally leaves this unchanged. |
| `[hyperbolic_foveation]`, `[hf]` | Experimental, active | YOLO-compatible Telescope-style image warp. Transforms training labels and inverse-transforms detections. |
| `[mambavision]`, `[mambavision_block]` | Works, active | Current mature branch focus. Forward/backward/update run on GPU and reuse connected/convolutional sublayers. |
| `[deconvolutional]` | Partial | Transposed convolution for decoder/upsampling paths. No batch-normalize support in this port. |
| `[deform_conv]`, `[deformable_convolutional]` | Experimental | DCNv1/DCNv2-style adaptive sampling layer. GPU path exists; keep validating before production use. |
| `[dcnv3]`, `[dcnv3_convolutional]` | Not ready | Parser/layer skeleton exists, but it is not considered validated. |
| `[dcnv4]`, `[dcnv4_convolutional]` | Experimental GPU-only | GPU path exists for deformable aggregation; CPU path is not the intended route. |
| `[transformer]`, `[transformer_block]` | Experimental | Swin-style windowed attention. Functional but needs more end-to-end YOLO training validation. |
| `[vit]` | Experimental | Global attention over all spatial tokens. Functional, but memory grows quadratically with spatial size. |
| `[graph_conv]`, `[graph_convolutional]` | Experimental | Graph-style local/global feature mixing. Not part of the currently validated training path. |
| `[clifford]`, `[clifford_block]` | Experimental / unstable | Algebraic feature-mixing layer. Broad test runs have shown crashes outside narrow experiments. |
| `[eml_convolutional]`, `[eml_conv]` | Experimental | Binary EML convolution block. Not currently validated for the main branch goal. |
| `[tucker_attention]`, `[tucker_attn]` | Experimental | Compact factorized attention insertion used by `cfg/yolov4_tiny_tucker.cfg`. |
| `[recursive_block]` | Experimental | Loops a body sequence with shared weights and requires the body to preserve feature dimensions. |
| `[Gaussian_yolo]` | Works, legacy | Present for older configs. Not the focus of this branch. |
| `[region]` | Works, legacy | Present for older YOLO configs. Not the focus of this branch. |
| `[avgpool]`, `[softmax]`, `[cost]`, `[dropout]` | Works, legacy | Mostly used by old classification configs. |
| `[local_avgpool]`, `[reorg3d]`, `[scale_channels]`, `[sam]`, `[contrastive]` | Works, limited | Existing specialty layers. Not actively developed here. |
| `[rnn]`, `[lstm]`, `[crnn]` | Legacy / not branch focus | Present for old non-YOLO configs. Not part of current validation. |

## What The Layers Are

This section describes the layer types conceptually. It is meant to answer “what does this cfg section do?” directly in this README.

### Core YOLO Layers

`[convolutional]` is the standard feature extractor layer. It applies learned filters over the image or feature map. With `groups`, it can also act like grouped or depthwise convolution. Most YOLO backbones are built mostly from this layer.

`[connected]` is a fully connected layer. In detection configs it is less common as a top-level layer, but newer modules reuse connected-style projections internally for token mixing and feed-forward blocks.

`[route]` wires tensors together. It can forward one previous layer, concatenate multiple layers, or select grouped channel slices. YOLO necks use it heavily to join shallow and deep features.

`[shortcut]` adds residual connections. It lets a block learn a correction while preserving the incoming feature signal, which stabilizes deeper networks.

`[maxpool]` reduces spatial resolution or expands receptive field by taking local maxima. It is part of many classic YOLO backbones.

`[upsample]` increases spatial resolution, usually so deeper semantic features can be fused with shallower high-resolution features in a YOLO neck.

`[yolo]` is the detection head. It decodes anchor-based predictions, computes the normal YOLO training loss, and produces detections. The Telescope/foveation work intentionally does not change this layer.

### Telescope / Foveation

`[hyperbolic_foveation]` is an image-space warp layer placed before the detector backbone. It learns or uses foveation parameters, warps the input image, rewrites GT boxes into the warped coordinate system during training, and inverse-warps detections during inference.

Use it when small or distant objects need extra effective resolution while preserving a normal YOLO detector downstream. It is not a new detector head. YOLO still sees a regular fixed-size tensor and trains with its usual loss.

Important behavior:

- `strength=0` is the identity path: image and boxes stay unchanged.
- `strength_zero_loss` makes training prefer the identity path unless foveation helps enough.
- Tiny-box supervision can pull `cx`, `cy`, `R`, and `strength` toward boxes below `small_box_px`.
- Labels must be transformed together with the image, otherwise YOLO receives boxes that no longer match the pixels.

### MambaVision

`[mambavision]` and `[mambavision_block]` are sequence-mixer blocks adapted to image feature maps. They flatten or rearrange spatial features into token-like sequences, apply MambaVision-style mixing, and return an image-shaped feature map.

Use them as backbone or neck replacements when convolution alone may not capture enough long-range context. Compared with global attention, Mamba-style sequence mixing aims to provide wider context with better scaling.

In this branch, MambaVision is one of the more mature experimental paths. It uses Darknet-style connected and convolutional sublayers where practical, plus custom CUDA for layout, layernorm, selective scan, residual routing, and glue logic.

### Deformable Convolution

`[deform_conv]` and `[deformable_convolutional]` are convolution layers with learned sampling offsets. A normal convolution samples a fixed grid around each pixel. A deformable convolution learns where each grid point should move.

Use it when object shape, pose, or local geometry varies enough that a fixed kernel grid is too rigid. It is still convolution-like: it produces a feature map and can replace selected convolutional layers.

DCNv1 learns offsets. DCNv2-style behavior also uses modulation masks so each sampled point can be weighted.

### DCNv3 and DCNv4

`[dcnv3]` and `[dcnv4]` are newer deformable aggregation designs. They are related to deformable convolution, but the sampling and weighting rules are more specialized than classic DCNv1/DCNv2.

Treat them as research layers in this branch:

- DCNv3 is not ready for normal training use here.
- DCNv4 has a GPU path and is intended as a GPU-only experiment.

Use these only when you are specifically testing deformable aggregation behavior, not as a default YOLO improvement.

### Deconvolutional

`[deconvolutional]` is transposed convolution. It learns to upsample a feature map, unlike `[upsample]`, which usually performs a fixed resizing operation.

Use it in decoder-style networks or feature pyramid experiments where the model should learn how to reconstruct higher-resolution features.

Current limitations:

- no `batch_normalize` support
- `pad` and `padding` are rejected by the parser
- no dilation or groups support

### Transformer

`[transformer]` and `[transformer_block]` are Swin-style windowed attention blocks. They split the feature map into local windows, run self-attention inside each window, and use shifted windows to exchange information across window boundaries.

Use them when you want attention-based local context without the full memory cost of global attention. They are best placed where the feature map is not too large.

The layer is functional, but training stability and optimizer settings still need careful validation in YOLO configs.

### ViT

`[vit]` is global Vision Transformer attention over all spatial tokens in the feature map. Every token can attend to every other token.

Use it only at small spatial sizes, such as deep backbone stages, because memory and compute grow quadratically with the number of tokens. A `13x13` feature map is much safer than a large early feature map.

### Graph Convolution

`[graph_conv]` and `[graph_convolutional]` are graph-style feature aggregation layers. Instead of using a fixed convolution grid only, they model feature relationships through graph-like neighbor aggregation.

Use this for experiments where the important relationship between features may not be purely local or grid-aligned. It is not part of the stable training path yet.

### Clifford

`[clifford]` and `[clifford_block]` are algebraic feature-mixing layers. They use operations inspired by Clifford/geometric algebra, such as wedge-like and inner-product-like channel interactions.

Use them for exploratory feature mixing where you want structured interactions between channel groups. They are experimental and have shown instability outside narrow configs.

### Tucker Attention

`[tucker_attention]` and `[tucker_attn]` are compact attention layers that factor projections through low-rank Tucker-style structure. The goal is to add attention-like mixing with fewer parameters or less compute than a full dense attention block.

Use it as a small attention insertion in a YOLO backbone or neck when full transformer blocks are too heavy.

### EML Convolution

`[eml_convolutional]` and `[eml_conv]` are experimental binary/operator-style convolution blocks. They explore alternative elementary mixing operations rather than standard dense convolution.

Use them only for architecture experiments. They are not currently a validated replacement for normal convolution in the main training path.

### Recursive Block

`[recursive_block]` repeats a body sequence of layers multiple times while reusing the same body weights. It is similar in spirit to applying the same block recurrently over depth.

Use it when you want more iterative refinement without increasing parameters as much as duplicating the body layers. The body must preserve dimensions so each loop can feed the next.

### Legacy And Specialty Layers

`[Gaussian_yolo]` and `[region]` are older detection heads kept for compatibility with legacy configs.

`[avgpool]`, `[softmax]`, `[cost]`, and `[dropout]` mostly support older classification-style networks.

`[local_avgpool]`, `[reorg3d]`, `[scale_channels]`, `[sam]`, and `[contrastive]` are specialty layers used by specific configs or older experiments. They are available, but they are not the focus of this branch.

`[rnn]`, `[lstm]`, and `[crnn]` support older recurrent/non-YOLO configs. They remain in the codebase but are not part of the current detection-layer work.

## Verification Notes

The current code builds with:

```sh
cmake --build build --target darknetobjlib -j2
```

The full test suite is not a clean signal for this branch yet because some unrelated experimental layers still have failures and some local build trees do not expose the test target. For MambaVision work, validate with a focused training or mAP run using `cfg/yolov4_tiny_mamba.cfg`. For foveation work, validate with `cfg/yolov4_tiny_hf.cfg`, `debug=1`, and a dataset containing boxes below `small_box_px`.
