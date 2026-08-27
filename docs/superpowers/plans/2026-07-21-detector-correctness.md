# Detector Correctness and Documentation Implementation Plan

> **For agentic workers:** Use test-first edits. Code agents may not build or run
> the test suite. The root agent integrates all edits, then performs the only
> project build and test run.

**Goal:** Correct the audited DETR, modern-head, and Hyperbolic integration
defects and publish a source-oriented comparison of the detector layer families.

**Architecture:** Keep paper-inspired heads compatible while making their
runtime contracts explicit. Shared detector routing will use layer semantics
rather than tensor-shape heuristics; Hyperbolic geometry will have one ordering
for training and inference; DETR safety checks will fail before invalid memory
access.

**Tech stack:** C++17/CUDA, Darknet cfg parser/runtime, GoogleTest, Markdown.

## Global Constraints

- Preserve unrelated worktree changes.
- Agents edit code only; no CMake build, test execution, commit, or planning edits.
- Agents may use targeted `g++ -fsyntax-only` checks.
- The root agent builds only after every agent edit is integrated and reviewed.
- Every root build/check uses `systemd-run --user --scope -p MemoryHigh=3G`.
- Every build uses at most four threads; use `-j 4`.
- Full official YOLO-NAS, PP-YOLOE, RF-DETR, and Telescope ports are out of scope.

## Relevant Tree

```text
darknet/
|-- cfg/
|   |-- rfdetr-lite.cfg
|   |-- LegoGears_detr.cfg
|   |-- LegoGears_yolonas*.cfg
|   |-- LegoGears_ppyoloe*.cfg
|   `-- yolov4-tiny-obb-anchorless.cfg
|-- docs/
|   |-- TELESCOPE_PLAN.md
|   `-- DETECTOR_LAYER_GUIDE.md
|-- src-lib/
|   |-- detr_decoder_layer.{cpp,hpp}
|   |-- detr_decoder_kernels.cu
|   |-- modern_yolo_layer.{cpp,hpp}
|   |-- yolonas_layer.{cpp,hpp}
|   |-- ppyoloe_layer.{cpp,hpp}
|   |-- obb_anchorless_layer.{cpp,hpp}
|   |-- hyperbolic_foveation_layer.{cpp,hpp}
|   |-- hyperbolic_foveation_layer_gpu.cu
|   |-- foveation_film_layer.{cpp,hpp}
|   |-- darknet_cfg.cpp
|   |-- darknet_network.cpp
|   |-- detector.cpp
|   |-- detector_map.cpp
|   |-- layer.cpp
|   `-- weights.cpp
`-- src-test/
    |-- test_custom_layers.cpp
    |-- test_modern_yolo_heads.cpp
    `-- test_hyperbolic_foveation.cpp
```

## Tasks

### Task 1: DETR safety and loss correctness

- [x] Add regression coverage for `G > Q`, invalid truth, and focal gradients.
- [x] Fail with a precise diagnostic before Hungarian matching when `G > Q`.
- [x] Validate truth classes and finite normalized boxes.
- [x] Implement the complete sigmoid focal derivative.
- [x] Make CUDA loss scaling and workspace destruction consistent.
- [x] Correct stale RF-DETR-lite cfg comments.

### Task 2: Modern-head and shared detector routing

- [x] Stop treating DFL `coords` as proof of OBB geometry.
- [x] Validate PP-YOLOE and YOLO-NAS input channel counts.
- [x] Add and route batch-aware PP-YOLOE/YOLO-NAS extraction.
- [x] Recognize `DETR_DECODER` consistently in validation scans.
- [x] Skip NMS for DETR set prediction and honor class maps.
- [x] Remove duplicate Hyperbolic inverse calls from shared inference paths.

### Task 3: Hyperbolic geometry and state lifecycle

- [x] Make fixed mode freeze predictor and global parameters.
- [x] Move the nonlinear inverse before letterbox/image correction through the
      shared raw normalized-box extraction APIs.
- [x] Ensure DETR and modern DFL detector regression gradients reach the nearest
      preceding Hyperbolic layer through box and parameter Jacobians.
- [x] Initialize `original_input` consistently on CPU.
- [x] Rebuild predictor allocations during resize.
- [x] Correct anisotropic L8 documentation and add a FiLM example.

### Task 4: Detector layer documentation

- [x] Create `docs/DETECTOR_LAYER_GUIDE.md`.
- [x] Compare `DETR_DECODER`, `YOLONAS`, `PPYOLOE`, and
      `OBB_ANCHORLESS`/`yolo_anchorless` side by side.
- [x] Document cfg syntax, output layout, assignment/loss, post-processing,
      source ownership, paper relationship, limitations, and sample cfgs.
- [x] Explicitly explain that `[yolo_anchorless]` aliases the oriented
      `OBB_ANCHORLESS` layer in this repository.

### Task 5: Integrated verification

- [x] Review the combined diff for overlapping or unrelated edits.
- [x] Build `darknet_tests` and `darknetcli` with `MemoryHigh=3G`, `-j 4`.
- [x] Run focused DETR, modern-head, Hyperbolic, and cfg tests.
- [x] Run the full test suite and classify failures outside the edited paths.
- [x] Run `git diff --check` and record exact evidence.
