# Recursive Block Layer

**Status**: COMPLETE  
**Files**: `src-lib/recursive_block_layer.{hpp,cpp}`, `src-lib/recursive_block_layer_gpu.cu`  
**CFG key**: `[recursive_block]`

Reference: [Recurrent Depth Transformers (2024)](https://arxiv.org/html/2512.24601v2)

---

## What It Is

A `[recursive_block]` wraps an arbitrary sequence of Darknet layers (the **body**) and applies that body **N times in a loop**, reusing the exact same weights each pass.

The body can be any size-preserving combination of `[convolutional]`, `[maxpool]`, `[avgpool]`, `[upsample]`, `[connected]`, `[dropout]`, or `[channel_shuffle]` layers — as long as the spatial dimensions and channel count come out the same as they went in.

This is sometimes called **Recurrent Depth**, **Looped Transformer**, or **Universal Transformer** style — the idea being that the same learned transformation is applied iteratively rather than once, letting the network "think harder" about a feature map without adding extra parameters.

---

## Core Idea

Compare a standard 4-conv stack versus a 1-conv recursive block with `loops=4`:

```
Standard (4 × parameters):      Recursive (1 × parameters, 4 × compute):

input → conv1 → conv2            input ──┐
      → conv3 → conv4 → output           │
                                  h₀ = input
                                  h₁ = F(h₀) + h₀ + input
                                  h₂ = F(h₁) + h₁ + input
                                  h₃ = F(h₂) + h₂ + input
                                  h₄ = F(h₃) + h₃ + input → output
```

`F` is the body. Every pass uses the same weights. The network learns *one good refinement step* and applies it repeatedly.

---

## Architecture

```
Input e  [B, C, H, W]
  │
  │  h₀ = e
  │
  │  for t = 0 .. loops-1:
  │    body_out = Body(h_t)           ← shared weights, same F every iteration
  │    h_{t+1} = body_out + h_t       ← residual connection through time
  │             [+ e]                 ← optional injection of original input (injection=1)
  │
  └─▶  output = h_{loops}  [B, C, H, W]
```

**Injection** (`injection=1`, default): the original input `e` is added at every loop iteration, not just the first. This acts as a constant "signal" that prevents the hidden state from drifting too far from the input and helps gradient flow during training.

---

## Training: 4 Forward Passes, 1 Backward Pass

During training with `loops=4`, the layer performs:

- **4 forward passes** (each applies the body once)
- **1 backward pass** (only through the final loop iteration)

This is called **truncated BPTT with k=1** (truncated backpropagation through time). It is the standard approach for recurrent-depth models because:

1. **Memory**: Only `h_{T-1}` (input to the last pass) is stored. Memory cost is O(1), not O(loops).
2. **Stability**: Full BPTT through all loops creates gradients equivalent to an RNN of depth T, causing vanishing/exploding gradients. Truncation avoids this.
3. **Practical convergence**: The body still receives gradient from the final application of every weight. The earlier passes act as "inference steps" whose effect propagates indirectly through the output.

The previous network layer receives gradient with respect to `h_{T-1}` rather than the original input. This is a biased but effective approximation — exactly how equilibrium-style and PonderNet-style iterative refinement models are trained.

---

## CFG Format

```ini
[recursive_block]
loops=4          ; number of forward passes (default: 4)
injection=1      ; add original input at each step (default: 1)
body=2           ; number of following sections that form the body

[convolutional]
batch_normalize=1
filters=256
size=3
pad=1            ; pad=1 with size=3, stride=1 keeps spatial dims unchanged
activation=leaky

[convolutional]
batch_normalize=1
filters=256
size=3
pad=1
activation=leaky

[convolutional]  ; ← this is a normal top-level layer again (body=2 ended above)
...
```

The `body=N` parameter tells the parser to treat the next `N` cfg sections as the body. Those sections are **not** added to the main network layer list — they are sub-layers owned by the recursive block.

---

## Parameters

| Parameter   | Default | Description |
|-------------|---------|-------------|
| `loops`     | 4       | Number of times the body is applied (forward passes). |
| `injection` | 1       | If 1, add the original input to `h_t` at every loop iteration. |
| `body`      | 1       | Number of cfg sections that follow and define the body sequence. |

---

## Size Constraint

The body **must be size-preserving**: the output spatial dimensions (H, W) and channel count (C) must equal the input. If they differ, the layer cannot loop and Darknet will report an error at startup.

**Valid body layers** (all must keep H × W × C unchanged):
- `[convolutional]` with `stride=1`, `pad=size//2` (e.g. `size=3, pad=1`)
- `[channel_shuffle]`
- `[dropout]`
- `[connected]` (if input and output size match)

**Invalid body layers** (change spatial size):
- `[maxpool]`, `[avgpool]`, `[upsample]` — these change H or W unless paired to cancel out

---

## Weight Count

A recursive block with `loops=4` and a body of one `3×3×256` conv has exactly the same parameter count as **one** `3×3×256` conv — not four. The loops add compute, not parameters.

This makes it a parameter-efficient way to add depth at inference time, which can be especially useful when the bottleneck is model size rather than inference budget.

---

## Example: Backbone Drop-in

Replace a plain conv stack with a recursive block in a YOLOv4-tiny backbone:

```ini
# Before: 2 separate convs (2× parameters)
[convolutional]
batch_normalize=1
filters=32
size=3
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=32
size=3
pad=1
activation=leaky

# After: 1 shared conv applied 4× (0.5× parameters, 2× compute)
[recursive_block]
loops=4
injection=1
body=1

[convolutional]
batch_normalize=1
filters=32
size=3
pad=1
activation=leaky
```

A full example network is in `cfg/yolov4-tiny-recursive-block.cfg`.

---

## Implementation Notes

- **Forward**: runs a C++ loop over the body layers `loops` times. GPU forward calls each body layer's `forward_gpu` function pointer.
- **Backward**: runs the body backward exactly once (final pass only). Each body layer's `weight_updates` accumulate normally.
- **Weights save/load**: the recursive block writes/reads its body layers' weights in order. The file format is identical to saving those layers at the top level.
- **Resize**: when the network input size changes, each body layer is resized and `rb_last_input` is reallocated.
