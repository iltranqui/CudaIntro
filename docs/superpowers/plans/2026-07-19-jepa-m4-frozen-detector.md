# JEPA M4 Frozen Detector Implementation Plan

> **For agentic workers:** Execute inline only. The user explicitly forbids subagents. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and compile the native frozen-backbone LegoGears JEPA detector and its random-encoder control.

**Architecture:** Reuse stock Darknet detector training, cutoff weight loading, anchor k-means, YOLO assignment, and mAP reporting. Add a JEPA-owned prefix auditor and a narrow detector initialization hook rather than a second detector implementation.

**Tech Stack:** C++23, CUDA/CUDNN build targets, CMake, GoogleTest, Darknet cfg/data formats.

## Global Constraints

- No subagents.
- No Git writes, staging, or commits.
- Every compile command uses exactly `-j 4`.
- Do not use `systemd-run`.
- GPU runtime execution is not required on this host.

---

### Task 1: Prefix contract and loader

**Files:**
- Create: `src-jepa/src/detection.hpp`
- Create: `src-jepa/src/detection.cpp`
- Create: `src-jepa/tests/test_detection.cpp`
- Modify: `src-jepa/CMakeLists.txt`
- Modify: `src-jepa/tests/CMakeLists.txt`

**Interfaces:**
- Produces: prefix validation and audited encoder cutoff for the detector trainer.

- [ ] Write tests that parse the canonical encoder and an exact detector prefix.
- [ ] Write mismatch tests for layer type and tensor/shape differences.
- [ ] Build the focused test and confirm it fails because the API is absent.
- [ ] Implement structural and named-tensor prefix validation.
- [ ] Rebuild and confirm the focused tests pass.

### Task 2: Stock detector initialization hook

**Files:**
- Modify: `src-lib/detector.cpp`
- Modify: `src-lib/darknet_network.hpp`
- Test: `src-jepa/tests/test_detection.cpp`

**Interfaces:**
- Consumes: audited positive encoder layer count and checkpoint path.
- Produces: detector training with `load_weights_upto` applied to every replica.

- [ ] Add a failing prefix-load test proving head tensors remain unchanged.
- [ ] Add an overload/options seam to the existing trainer.
- [ ] Load the checkpoint only through the audited cutoff.
- [ ] Run the focused test and stock CPU tests.

### Task 3: Detector cfg, manifests, and anchors

**Files:**
- Create: `cfg/LegoGears_jepa_detect.cfg`
- Create: `data/LegoGears_jepa/LegoGears_jepa.names`
- Create: `data/LegoGears_jepa/LegoGears_jepa_detect.data`
- Create: `data/LegoGears_jepa/LegoGears_jepa_anchors.data`
- Create: `data/LegoGears_jepa/detector_train.txt`
- Create: `data/LegoGears_jepa/detector_valid.txt`
- Create: `data/LegoGears_jepa/anchor_images.txt`

**Interfaces:**
- Consumes: all 238 annotation `.txt` files and the 90 paired JPEGs.
- Produces: a parseable 5-class, 3-anchor detector configuration.

- [ ] Generate deterministic train/valid and 238-label anchor manifests.
- [ ] Run `darknet detector calcanchors ... -num_of_clusters 3 -width 32 -height 32`.
- [ ] Put the exact native anchors in the cfg.
- [ ] Parse the cfg in a CPU test and assert the freeze boundary/head dimensions.

### Task 4: JEPA detection command and random control

**Files:**
- Modify: `src-jepa/src/cli.cpp`
- Modify: `src-jepa/src/detection.cpp/.hpp`
- Test: `src-jepa/tests/test_detection.cpp`

**Interfaces:**
- Produces: `darknet jepa detect DATA CFG --encoder CFG --weights FILE` and `--random-encoder`.

- [ ] Add option-validation tests for pretrained and random modes.
- [ ] Add help text and parse the M4 command.
- [ ] Invoke stock detector training with `calc_map=1`.
- [ ] Print the selected control mode and audited prefix details before training.

### Task 5: Native anchor and assignment tests

**Files:**
- Modify or create: `src-test/test_m4_anchors.cpp`
- Modify: `src-test/CMakeLists.txt`

**Interfaces:**
- Consumes: Darknet native `do_kmeans` and YOLO anchor-selection helpers.

- [ ] Add a tiny deterministic synthetic width/height clustering test.
- [ ] Add a best-IoU anchor selection test if the existing helper is public and CPU-safe.
- [ ] Run the focused CPU tests.

### Task 6: Compile and verify

**Files:**
- Modify: `tasks/todo.md`, `task_plan.md`, and `notes.md` with exact evidence.

- [ ] Build `jepa`, `jepa_tests`, `darknetcli`, `darknet_tests`, and JEPA apps in the CPU tree with `-j 4`.
- [ ] Run the focused and full applicable CPU suites.
- [ ] Build the same CUDA-bearing targets in `build` with `-j 4`.
- [ ] Run `git diff --check` and review every changed file.
- [ ] Record GPU runtime TODOs without claiming training results.

