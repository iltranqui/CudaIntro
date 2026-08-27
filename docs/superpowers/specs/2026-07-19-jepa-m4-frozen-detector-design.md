# JEPA M4 Frozen Detector Design

## Goal

Add a native Darknet detector whose prefix is the exact LegoGears JEPA encoder,
whose YOLO head alone trains, and whose pretrained result is compared with an
otherwise identical random-encoder control.

## Architecture

The detector cfg repeats `src-jepa/configs/jepa-encoder-tiny.cfg` layer-for-layer,
then adds a 30-channel linear convolution (`3 * (5 + 5)`) and one YOLO layer.
`stopbackward=1` is placed on the final encoder ViT, which is Darknet's native
boundary for freezing all earlier layers while leaving later layers trainable.

`darknet jepa detect` parses an encoder cfg and the detector cfg before training.
It validates that the detector prefix has the same layer types, input/output
shapes, and named tensor sizes as the encoder. Only after validation does stock
Darknet load the encoder checkpoint with `load_weights_upto(..., encoder.n)` for
each detector replica. The random control uses the same detector call without a
checkpoint or prefix load.

## Data and Anchors

The detector train/valid manifests contain only JPEGs with adjacent YOLO label
files. Anchor calculation uses a separate manifest with one image-shaped path
for each of the 238 annotation files; `calc_anchors` needs only the derived label
path, so suffix annotations without a physical JPEG still participate safely.
The native Darknet k-means result at the canonical 32x32 encoder geometry is
copied into the detector cfg.

## Metrics

The command enables stock detector `-map` behavior. Darknet's validation path
already reports per-class/global precision, recall, F1, AP, and mean AP. No
parallel metric implementation is added.

## Error Handling

- Missing encoder cfg/checkpoint in pretrained mode is an actionable error.
- `--random-encoder` and a pretrained weights request are mutually exclusive.
- Prefix layer count/type/shape/tensor mismatches fail before any checkpoint is
  read into the detector.
- A checkpoint that cannot fill the audited prefix fails through the native
  weight reader.

## Verification

CPU tests cover cfg parsing, exact and mismatched prefixes, prefix-only loading,
the random/pretrained option contract, and native k-means/assignment seams where
they can be exercised without a GPU. The CPU suite is run, then both CPU and
CUDA-bearing JEPA/Darknet targets are compiled with `-j 4`. Training, one-step
GPU freeze immutability, and the JEPA-versus-random mAP comparison remain runtime
work because this host has no CUDA driver.

