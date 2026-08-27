# FP4 and FP8 in darknet

Reference documentation for the low-precision GEMM paths in this codebase,
consolidated from the investigations that fixed the FP4 quantize kernel and
CMake feature-detection bugs, and the FP8 shape-contract fix that followed.
Facts are cited by `file:line`; anything not directly verified against
hardware is marked **unverified**.

## 1. Overview

| | FP4 (NVFP4 / E2M1) | FP8 (E4M3 / E5M2) |
|---|---|---|
| Format | Block-scaled: 16-element blocks, FP8 E4M3 scale per block | Per-tensor scaled: one FP32 scale per whole tensor |
| Hardware | Blackwell tensor cores only (SM 100/103/110/120/121); quantize kernels have a software fallback that runs on any arch | Compute capability >= 8.9 (Ada and newer), `Darknet::fp8_gemm_supported()` (`src-fp8/fp8_gemm.cpp:309-327`) |
| Layer types | Convolutional, Tucker attention (inference only) | Convolutional, Tucker attention (inference only) |
| Inference | Yes | Yes |
| Training | Not implemented | Yes — forward, weight-gradient, and data-gradient GEMMs (`src-lib/convolutional_layer.cpp:1325-1476`) |
| Calibration | `calibrate_detector_fp4` (`src-lib/detector.cpp:1432`) — sidecar written, diagnostic only; does not yet change inference numerics | `calibrate_detector_fp8` (`src-lib/detector.cpp:1330`) — sidecar consumed at inference time |

Neither format is used by `connected_layer.cpp`/`.hpp` — both are
convolution-only (+ Tucker-attention-inference-only) features.

Training policy for this implementation is narrower than generic format support:
Ampere keeps the existing FP16 path; Ada SM89 uses the optimized FP8 convolution
path; Hopper is outside this change; Blackwell training FP4 remains future scope.
The SM89 path requires both an `89`/`89-real` build target and an SM89 runtime GPU.

## 2. FP8 architecture

### GEMM plan model
- `Fp8GemmPlan` is built once per layer at network setup
  (`fp8_setup_convolutional_layer` for inference,
  `fp8_setup_convolutional_layer_training` for training —
  `src-lib/convolutional_layer.cpp:1261-1476`) and reused for every
  forward/backward call afterward. It is torn down once, in
  `fp8_release_convolutional_layer`.
- Three GEMMs exist per training-eligible conv layer, selected via
  `Fp8TrainingGemm` (`src-fp8/fp8_gemm.hpp:11-17`, spec builder
  `src-fp8/fp8_gemm.cpp:28-64`):
  - `Forward` — A=weights (E4M3), B=input (E4M3)
  - `WeightGradient` — A=dy (E5M2), B=input (E4M3)
  - `WeightGradientDirectUpdate` — swapped-operand FP32 update, default on SM89
  - `DataGradient` — A=weights (E4M3), B=dy (E5M2)
  - `DataGradientDirectUpdate` — A=batched dy^T (E5M2), B=broadcast weights
    (E4M3), writes FP32 directly into NCHW `state.delta`; restricted to exact
    1x1/stride-1/pad-0/dilation-1 convolution geometry
- `cublasLt` only supports the TN case for FP8 matmuls. This codebase
  exploits row-major/column-major bit-equivalence to avoid an explicit
  transpose (see the buffer-contract comment at `src-fp8/fp8_gemm.hpp:25-40`).

### Mixed precision
FP32 "master" weights (`l.weights_gpu`) are retained and updated by the
optimizer every step in FP32/BF16. A disposable FP8 view
(`l.weights_fp8_gpu`, `l.weights_fp8_t_gpu`, `l.weights_fp8_nhwc_gpu`) is
regenerated from the master every iteration by
`fp8_requantize_convolutional_training_weights`
(`src-lib/convolutional_kernels.cu:2284-2370`, called from
`update_convolutional_layer_gpu` at `convolutional_kernels.cu:2455`). Scale
factors use NVIDIA-Transformer-Engine-style **delayed scaling**: a rolling
history of recent amax values (`kFp8AmaxHistoryLength`,
`Darknet::fp8_delayed_scaling_record_amax`) rather than a synchronous
fresh amax, computed entirely on-device via a fused kernel
(`fp8_delayed_scale_update3_gpu`) to avoid per-iteration host syncs.
When cuDNN FP8 forward is available on SM89, the row-major forward GEMM view and
forward cuBLASLt plan are omitted; only KRSC forward weights and transposed dgrad
weights are regenerated.

### Training gates and exclusions
- `use_fp8_training` requires `state.net.fp8_training`, a warmup period
  (`get_current_iteration(state.net) >= state.net.fp8_warmup_iters`), and
  `l.fp8_train_eligible` (`src-lib/convolutional_kernels.cu:481-487`).
- The backward pass can be forced back onto cuDNN's normal precision via
  `Darknet::fp8_backward_mode_from_env() == Fp8BackwardMode::Cudnn`
  (`convolutional_kernels.cu:697`) — an opt-in escape hatch, not the
  default.
- Two conv layers are always excluded from FP8 **training** regardless of
  shape, for accuracy/stability, not shape validity: the first
  FP8-eligible conv layer (`first_train_conv`) and any conv layer feeding
  the detection head (`layer_feeds_detection_head`) —
  `src-lib/darknet_network.cpp:2598-2609`.
- The previous staged paths remain available with
  `DARKNET_FP8_DISABLE_SM89_LAYOUT_PRUNING=1`,
  `DARKNET_FP8_DISABLE_DIRECT_WGRAD_UPDATE=1`, and
  `DARKNET_FP8_DISABLE_DIRECT_DGRAD_UPDATE=1`. These are setup-time controls;
  set them before loading the network.

### Calibration pipeline
`calibrate_detector_fp8` (`src-lib/detector.cpp:1330-1429`), invoked
automatically at the end of training when `net.fp8_training` is set
(`detector.cpp:755-765`, added in commit `3209e3b2`):
1. Reloads the just-saved weights into a fresh network with
   `fp8_calibrating=1`, fuses conv+batchnorm.
2. Runs a **one-shot, forward-only** inference pass over the calibration
   image list — no gradients. During each forward call,
   `forward_convolutional_layer_gpu_fp8` accumulates a running per-layer,
   per-tensor amax of real activations (`Darknet::fp8_accumulate_amax_gpu`,
   `convolutional_kernels.cu:472-475`).
3. Converts the accumulated amax to a scale
   (`Darknet::fp8_scale_from_amax`) and writes a
   `{layer_index, amax, scale}` table to a sidecar file,
   `<weights>.fp8scales` (`Darknet::fp8_write_calibration_scales`).
4. The sidecar is loaded back at inference time
   (`src-lib/weights.cpp:30,1617`) and gates `fp8_setup_convolutional_layer`
   via `l.fp8_scales_loaded` (`convolutional_layer.cpp:1266`).

A second, unrelated calibration mechanism exists for inference-time relay
fusion: `fp8_activation_calibration_pending` /
`fp8_finalize_network_activation_calibration`
(`src-lib/darknet_network.cpp:650-700`) — a single eager (non-CUDA-graph)
first inference frame used to measure inter-layer relay-link activation
stats before enabling fused FP8 relays. Not tied to training.

### Quantize / dequantize kernels
`src-fp8/fp8_kernels.cu` kernels already use 256/512-thread launches (not a
small fixed-block pattern) and avoid redundant global reads: most kernels
read each source element exactly once per thread; the layout-transpose
kernels (`fp8_quantize_dual_layout_weights_kernel`,
`fp8_quantize_triple_layout_weights_kernel`,
`fp8_quantize_nchw_to_nhwc_kernel`, `fp8_kernels.cu:1140`) cache a 32x33
tile (33 = 32 padded by one float to dodge the shared-memory bank conflict
a plain `[32][32]` transpose would hit) in `__shared__` memory and reuse it
for the second (transposed) write — `fp8_kernels.cu:1137-1139` documents
this as already having replaced an older strided-read version, on the
**input** side (quantizing the NCHW activation into the NHWC FP8 GEMM
operand).

**2026-08-27 finding — the output side had the same bug the input side was
already fixed for.** `fp8_nhwc_output_to_nchw_gpu` (dequantizes/reformats
the GEMM's NHWC output back to NCHW, once per FP8 conv layer per frame) was
still calling the naive one-thread-per-element form:
`src[((n*height+h)*width+w)*channels+c]` — every thread in a warp reads an
address `channels` floats apart, i.e. a full-cache-line-per-4-useful-bytes
gather at high channel counts (~3% DRAM efficiency at 512 channels). Same
class of bug as the pre-fix input kernel; just never got the same fix
applied to its mirror.

Found while investigating a single-image-vs-video timing discrepancy on
LegoGears_v2 (224x160, batch 1): a one-shot `darknet_01_inference_images`
run showed FP8 *slower* than FP16 (217ms vs 119ms predict) purely from
first-frame delayed-scaling calibration cost (`FP8-NHWC relay links=9
(first-frame scale guard pending)` in the startup banner), while the
390-frame `darknet_04_process_videos` run showed FP8 correctly ahead once
warmed up (125.97 FPS vs 92.11 FPS; `forward_convolutional_layer_gpu`
0.1ms/call fp8 vs 0.3ms/call fp16, averaged over 8190 calls). Reading the
dequant kernel during that investigation is what surfaced the gather-stride
asymmetry against the already-tiled input path.

At this network's tiny per-layer tensor sizes (max 512 channels x 35
spatial elements) the naive gather stayed cheap in absolute terms
(`fp8_nhwc_output_to_nchw_gpu` measured at ~16us/call, ~13% of frame budget
combined with the input-side quantize step) — not the current bottleneck —
but the inefficiency scales with channel count and would matter more at
larger input resolutions or a heavier backbone.

**Fix applied**: replaced `fp8_nhwc_f32_to_nchw_kernel` /
`fp8_nhwc_bf16_to_nchw_kernel` with one templated
`fp8_nhwc_to_nchw_tiled_kernel<SrcT>` (`fp8_kernels.cu:1340`, next to
`fp8_quantize_nchw_to_nhwc_kernel`), mirroring the same 32x33-tile pattern:
NHWC load coalesced on `tx`=channel, NCHW store coalesced on `tx`=spatial,
transposed through shared memory in between, with the per-channel bias add
folded into the store loop as before. `fp8_nhwc_output_to_nchw_gpu`'s
launch config (`fp8_kernels.cu:~2300`) now matches the quantize side's 3D
grid (`(spatial+31)/32, (channels+31)/32, batch`) instead of the flat 1D
`fp8_gridsize_256` grid it used before.

**Not yet verified**: the `darknet/build` tree had no `CMakeCache.txt`
(needs a fresh `cmake` configure); the reconfigure+rebuild was interrupted
before running. Still needs: a clean build, a re-run of the LegoGears_v2
`darknet_04_process_videos` benchmark to confirm no regression in
detection output (object count / avg objects-per-frame should stay ~1950 /
~5.0), and ideally a benchmark at a larger resolution/channel count where
the fix should show a measurable FPS delta (expected negligible at
224x160, since the old kernel was already ~16us/call here).

## 3. FP4 architecture

### Quantize kernels (`src-fp4/fp4_kernels.cu`)
`quantize_cublaslt_kernel` and `quantize_nchw_to_cublaslt_kernel` batch
`kQuantizeWarpsPerCta = 4` reduction blocks per CTA (one warp per block,
`fp4_kernels.cu:146`), with each pack-phase lane reusing its phase-1
register value via `__shfl_down_sync` instead of re-reading global memory.
Host launchers (`fp4_kernels.cu:406,426,447`) compute
`reduction_block_groups` and launch `32 * kQuantizeWarpsPerCta` threads
per CTA. Verified byte-for-byte identical to the pre-fix kernel across
production K/M shapes and boundary cases (single-K-block, non-multiple-of-4
tail warps) via an isolated `nvcc` harness; confirmed to emit genuine
`sm_100`/`sm_120` device code via `cuobjdump`.

### Two backends, two different shape gates
`fp4_gemm_plan_create` tries the fast cuBLASLt direct path first; if its
gate fails, it still returns a working plan using the cuDNN Frontend graph
path alone — only if *both* fail does a layer become fully FP4-ineligible.

- **cuBLASLt direct path** — `setup_cublaslt_nvfp4`, gate at
  `src-fp4/fp4_gemm.cpp:78`:
  ```cpp
  if (plan.spec.batch != 1 || k % 32 != 0 || m % 8 != 0 || n % 8 != 0) return false;
  ```
- **cuDNN Frontend graph path** — gate at `src-fp4/fp4_gemm.cpp:300`:
  ```cpp
  spec.reduction <= 0 || spec.reduction % 16 != 0
  ```
  only `K` (the block-scale reduction axis) is checked; `M`/`N` are
  unconstrained for this path.

For conv layers, `K = c*size*size`, `M = filters`, `N = out_h*out_w`
(`convolutional_layer.cpp`'s `fp4_setup_convolutional_layer`). Every YOLO
net's first conv layer (`K=3*3*3=27`) fails even the `%16` gate (raw
3-channel input — expected, not a bug). Every YOLO detection head
(`filters=255`, odd) fails the cuBLASLt path's `M%8` but still gets FP4 via
the cuDNN Frontend fallback — reported as `fp4-shape-slow`, not
`fp4-shape-no`, by the eligibility display below.

### cuDNN Frontend API contract (verified against the vendored headers)
`Block_scale_quantize_attributes` / `Block_scale_dequantize_attributes`
(`third_party/cudnn-frontend/include/cudnn_frontend/graph_properties.h:2474-2576`).
Key facts:
- Quantize output `scale` tensor's dim = input dim with the chosen axis
  divided by `block_size` — **integer division**. `graph.validate()` does
  **not** reject a non-multiple axis; it silently truncates
  (`floor(K/16)` instead of `ceil(K/16)`) instead of erroring. This is why
  the codebase's own `reduction % 16 != 0` guard is load-bearing, not
  redundant — confirmed via a handle-free `graph.validate()` call using the
  real API (`K=27` and `K=33` both "pass" validate() with a truncated scale
  dim).
- Dequantize output must be a **virtual** tensor (feed directly into
  another graph op, e.g. `matmul`) — cannot be a graph-level output
  (`node/block_scale_dequantize.h:26-40`).
- No M/N/K alignment beyond the block-size-divisibility rule above is
  documented anywhere in the frontend headers. The vendored sample
  `samples/cpp/matmul/blackwell_nvfp4_mxfp8_block_scale_matmul.cpp:82-140`
  exercises non-power-of-two, non-block-size-multiple M/N shapes
  successfully (M=137, N=268/272, K=160).

## 4. Shape/eligibility gate reference table

| Gate | Format/path | Constraint | Status |
|---|---|---|---|
| cuDNN Frontend axis divisibility | FP4 | `K % 16 == 0` | **Confirmed real** — cuDNN Frontend headers, `graph.validate()` truncation behavior |
| cuBLASLt direct-path K | FP4 | `K % 32 == 0` | **Unverified** — no cuBLASLt header/doc mentions E2M1 alignment at all (grep found zero FP4 mentions in `cublasLt.h`); plausible common ancestor is cuBLASLt's *generic* Tensor Core performance guideline (`(m*CtypeSize)%16==0`, stated as a perf recommendation, not a hard requirement), not something specific to FP4 |
| cuBLASLt direct-path M/N | FP4 | `M % 8 == 0`, `N % 8 == 0` | **Unverified** — same as above |
| cuBLASLt direct-path batch | FP4 | `batch == 1` | Confirmed by code inspection — training/batch>1 always falls to the cuDNN Frontend graph path for FP4 conv |
| FP8 reduction padding | FP8 | `reduction_pad % 16 == 0` | **Confirmed enforced** — every caller rounds via `fp8_round_up_to_16()` (`src-fp8/fp8_gemm.cpp:9-12,62`); `fp8_gemm_plan_create_ex` checks `reduction_pad >= reduction` |
| FP8 output_rows (M) | FP8 | previously documented as `% 16` | **Corrected** — not enforced anywhere in code (`fp8_gemm.cpp:349-360`), and no vendored cuBLASLt documentation supports the requirement for an FP8 (E4M3/E5M2) TN matmul. Doc comment at `fp8_gemm.hpp:25-40` fixed to stop overstating this. Actual behavior for non-multiple-of-16 `output_rows` (e.g. `filters=255`) is decided solely by `cublasLtMatmulAlgoGetHeuristic()` at plan-creation time — **unverified on real hardware this session** |

No padding or splitting scheme exists for the *actual* GEMM data in either
format — only scale/metadata buffers are ever rounded up. Detection heads
and stem layers being excluded from the fast paths is treated as
acceptable (these are accuracy-critical or trivially cheap layers), not
something to force-fit via padding.

## 5. Per-layer eligibility display

`Darknet::format_layer_summary` (`src-lib/darknet_format_and_colour.cpp`)
prints two independent static-shape-check columns for every convolutional
layer in a `.cfg` (used by the `darknet cfg` display path,
`darknet_cfg.cpp:1196` — this runs before weights are loaded or the GPU is
initialized, so it can only report what's knowable from the cfg alone):

- **`fp4-shape-no` / `fp4-shape-slow` / `fp4-shape-ok`** — mirrors the two
  FP4 gates above. `-no`: fails even the cuDNN Frontend `K%16` gate (or
  fails a structural precondition: grouped, binary/xnor, shared weights).
  `-slow`: passes `K%16` but not the stricter cuBLASLt `K%32,M%8,N%8`
  set — gets FP4 via the cuDNN Frontend fallback only. `-ok`: passes both.
- **`fp8-no` / `fp8-shape-ok`** — FP8 has no confirmed shape-based
  rejection (see table above), so this column only reports the structural
  gate mirrored from `fp8_convolutional_candidate()`
  (`convolutional_layer.cpp:1137-1150`: type, `groups==1`, not
  binary/xnor, no `share_layer`). A `fp8-shape-ok` layer can still fail to
  run FP8 at runtime for two reasons this display cannot see: GPU compute
  capability < 8.9, or no `.fp8scales` calibration sidecar loaded
  (`l.fp8_scales_loaded`).

## 6. Open questions

- Whether cuBLASLt's FP4 `K%32/M%8/N%8` constants are a real hardware
  requirement or this codebase's conservative guess — needs either real
  Blackwell execution or a documented NVIDIA source; unresolved.
- Whether FP8's `output_rows` (M) alignment matters in practice on real
  Ada/Hopper/Blackwell hardware — the doc comment has been corrected to
  stop claiming a requirement that isn't enforced, but whether
  `cublasLtMatmulAlgoGetHeuristic()` actually accepts or silently degrades
  for odd `M` (e.g. `filters=255`) has not been observed on real hardware.
- No FP4-vs-FP8 GEMM throughput benchmark exists in this codebase.
  `src-cli/precision_benchmark.cpp` reports FP4/FP8 eligible-layer *counts*
  and dispatch counters, not comparative timing. External "4-6x faster
  than BF16" FP4 figures found in the literature are FP4-vs-BF16, not
  FP4-vs-FP8, and do not support a direct throughput comparison claim.
- The FP4 cuBLASLt path's output C/D layout uses an operand-swap trick
  (computing `C^T` in column-major to get the row-major `[M,N]` output
  Darknet expects, `src-fp4/fp4_gemm.cpp`) that has not been cross-checked
  against FP8's equivalent layout handling for correctness parity —
  flagged as a plausible candidate for divergent behavior, not confirmed
  either way.
