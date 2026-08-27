# Graph Convolutional Layer

**Status**: FUNCTIONAL -- CPU/GPU forward and backward, Darknet weight IO, resize support
**Files**: `src-lib/graph_conv_layer.hpp`, `src-lib/graph_conv_layer.cpp`, `src-lib/graph_conv_kernels.cu`
**CFG keys**: `[graph_conv]`, `[graph_convolutional]`

---

## What It Is

`[graph_conv]` is a local message-passing layer for image feature maps. It keeps the same spatial idea as a convolution, but treats each output position as a node that receives messages from a `size x size` neighborhood.

The layer has two edge modes:

| Mode | CFG | Behavior |
|------|-----|----------|
| Uniform mean | `edge_mode=0` | Average valid neighbors, then project channels. |
| Learned attention | `edge_mode=1` | Compute one softmax-normalized edge weight per neighbor from center and neighbor features. |

The output is:

```text
neighbor = W_neigh * sum_j(alpha_ij * x_j)
self     = W_self  * x_i                 # only when use_self=1
output   = activation(BN_or_bias(neighbor + self))
```

This makes the layer useful where a fixed convolution kernel is too rigid: local edges, boundaries, and textured regions can learn different neighbor weights than flat regions.

---

## CFG Example

```cfg
[graph_conv]
batch_normalize=1
filters=128
size=3
stride=1
pad=1
edge_mode=1
use_self=1
activation=swish
```

Use `edge_mode=0` as a cheaper first test. Use `edge_mode=1` when you want feature-dependent local attention.

---

## Parameters

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `filters` | `1` | Output channels. |
| `groups` | `1` | Splits input and output channels into independent graph projections. `C` and `filters` must both divide by `groups`. |
| `size` | `1` | Odd neighborhood size. `3` means a 3x3 local graph. |
| `stride`, `stride_x`, `stride_y` | `1` | Output lattice stride. |
| `dilation` | `1` | Spacing between neighbors in the local graph. Forced to `1` for `size=1`. |
| `pad` / `padding` | `0` | `pad=1` uses `size/2`, matching common Darknet convolution cfgs. |
| `batch_normalize` | `0` | Applies BN before activation. `cbn=1` maps to Darknet's BN mode `2`. |
| `edge_mode` / `graph_edge_mode` | `1` | `0` = valid-neighbor mean, `1` = learned attention softmax. |
| `use_self` / `graph_use_self` | `1` | Adds a separate center-pixel projection. |
| `valid_mask_zero` / `graph_valid_mask_zero` | `1` | Required currently; invalid padding neighbors are masked out. |
| `activation` | `logistic` | Any activation accepted by `get_activation_from_name()`. |

Unsupported options currently fail at layer creation: `binary`, `xnor`, `antialiasing`, `share_layer`, `assisted_excitation`, and `adam`.

---

## File Guide

### `src-lib/graph_conv_layer.hpp`

The header is the API and usage reference:

- Long-form conceptual walkthrough of graph aggregation and backward credit assignment.
- Public CPU/GPU function declarations.
- Output-shape helpers: `graph_conv_out_height()` and `graph_conv_out_width()`.
- Factory contract: `make_graph_conv_layer(...)`, including `graph_edge_mode`, `graph_use_self`, and `graph_valid_mask_zero`.

When changing the cfg surface, update the header's CFG usage block and this guide together.

### `src-lib/graph_conv_layer.cpp`

The CPU implementation owns:

- Layer construction, validation, allocation, and resize.
- Runtime buffers:
  - `graph_ref`: center feature at each output position.
  - `graph_agg`: aggregated neighbor feature before channel projection.
  - `graph_alpha`: normalized edge weights.
  - `graph_valid`: padding mask for valid neighbors.
- CPU forward:
  - Build `graph_ref`, `graph_agg`, `graph_alpha`, and `graph_valid`.
  - Project `graph_agg` through `weights` with GEMM.
  - Optionally add `graph_self_weights * graph_ref`.
  - Apply bias/BN and activation.
- CPU backward:
  - Backprop projection weights and self weights.
  - Backprop aggregation into input pixels.
  - For `edge_mode=1`, run softmax backward and update edge kernels/biases.
- SGD update for main projection, self projection, and learned edge parameters.

The parser lives in `src-lib/darknet_cfg.cpp::parse_graph_conv_section()`. Weight save/load is in `src-lib/weights.cpp`.

### `src-lib/graph_conv_kernels.cu`

The CUDA implementation mirrors the CPU path:

- One CUDA thread handles one `(batch, group, out_y, out_x)` graph node for aggregation.
- `graph_conv_forward_kernel` writes `graph_ref`, `graph_agg`, `graph_alpha`, and `graph_valid`.
- Dense channel projection is intentionally handled by cuBLAS GEMM after graph aggregation.
- `graph_conv_backward_kernel` computes graph-specific gradients, including attention softmax backward.
- `forward_graph_conv_layer_gpu()` and `backward_graph_conv_layer_gpu()` orchestrate kernels, GEMMs, BN, activation, and gradient accumulation.
- `push_graph_conv_layer()` / `pull_graph_conv_layer()` synchronize main, self, edge, BN, and activation buffers.

Implementation constants:

| Constant | Purpose |
|----------|---------|
| `GRAPH_LOCAL_REF_CACHE_MAX=32` | Cache small center feature vectors inside one thread. |
| `GRAPH_LOCAL_LOGITS_MAX=49` | Cache logits locally through 7x7 neighborhoods; larger kernels recompute. |

---

## Shape And Memory

For input `[B, C, H, W]`:

```text
out_h = (H + 2*pad - dilation*(size - 1) - 1) / stride_y + 1
out_w = (W + 2*pad - dilation*(size - 1) - 1) / stride_x + 1
output = [B, filters, out_h, out_w]
graph_ref / graph_agg = [B, C, out_h, out_w]
graph_alpha / graph_valid = [B, groups, out_h, out_w, size*size]
```

Parameter shapes:

```text
weights             = [groups, filters/groups, C/groups]
graph_self_weights  = [groups, filters/groups, C/groups]       # when use_self=1
graph_edge_kernel   = [groups, size*size, 2*(C/groups)]         # when edge_mode=1
graph_edge_biases   = [groups, size*size]                       # when edge_mode=1
```

---

## LegoGears Graph Config

`cfg/LegoGears_graph.cfg` is the graph-convolution sibling of `cfg/LegoGears_clifford.cfg`.

It keeps the Clifford cfg's route topology and detection heads, but replaces every standard convolution and Clifford feature-mixing block with a graph layer. Spatial 3x3 blocks use attention graph convolution:

```cfg
[graph_conv]
batch_normalize=1
filters=<same channel count as the Clifford block>
size=3
stride=1
pad=1
edge_mode=1
use_self=1
activation=swish
```

That preserves layer count and route indexes while changing the local feature mixer.

Pointwise 1x1 projections, including the YOLO logit projections, use:

```cfg
[graph_conv]
filters=<same channel count as the original conv>
size=1
stride=1
pad=1
edge_mode=0
use_self=0
activation=<same activation as the original conv>
```

That keeps the 1x1 layers as pointwise graph projections instead of adding a second self branch.

---

## Validation Commands

Focused graph layer tests:

```bash
./build/src-test/darknet_tests --gtest_filter='GraphConv*'
```

Cfg parse/ops smoke test:

```bash
./build/src-cli/darknet ops cfg/LegoGears_graph.cfg
```

If the binary is stale, rebuild first:

```bash
cmake --build build --target darknet_tests darknetcli -j 8
```
