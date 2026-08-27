# How to Develop CUDA Kernels, Manage Memory, and Benchmark in Darknet

A playbook distilled from the FP8 optimization campaign (2026-07-05/06, `fp8` branch,
RTX 4060 Laptop sm_89, WSL2, CUDA 13.2, cuDNN 9.17). Written for future sessions:
follow this and you skip a day of rediscovery.

---

## 1. The loop that works

```
profile → rank buckets → change ONE thing → bitwise test → re-profile → keep or revert
```

Concrete case history from this campaign (GPU kernel total, same 15-iter benchmark):

| Step | Total | Verdict |
|---|---:|---|
| FP8 im2col + cuBLASLt baseline | 1603 ms | start |
| + cuDNN graph FP8 fused forward | 1435 ms | keep |
| + fused dgrad/wgrad epilogues + batched launches | ~1100 ms | keep |
| + x4 packed stores + tiled NCHW→NHWC | 1005 ms | keep (beats 1201 ms default) |
| + shared-mem tiled 3x3 im2col | 1057 ms | **REVERT** — regression |

Lessons in that last row: measure BEFORE keeping. The tiled im2col "obviously" fixed a
9x read amplification — but those reads were already in L1/L2, so the tile only added
sync overhead. Intuition about redundant *reads* often loses to the cache; redundant
*writes* and extra full-tensor *passes* are the reliable wins.

### What actually paid, ranked
1. **Removing whole passes** (fused epilogues: GEMM-out→convert→col2im→axpy became one
   kernel; dual-layout quantize: one read → two layouts) — biggest, always wins.
2. **Launch-count reduction** (grid-z batching over images: O(batch)→O(1) per layer;
   3 single-thread scale updates → 1 kernel). WSL2 launch latency makes this bigger
   than on native Linux. 36k launches/15iter of `simple_copy`+`axpy` cost ~163 ms.
3. **Store widening** (4 FP8 bytes packed into one `uint32` store): 167→94 ms on the
   im2col-quantize. 1-byte scalar stores waste 3/4 of every transaction.
4. **Read coalescing via 32x33 shared tile** — only when access is genuinely strided
   (NCHW→NHWC channel-strided reads: 41→22 ms). NOT when reads are cached (see revert).

---

## 2. Profiling recipe (copy-paste)

Benchmark config: copy LegoGears cfg, set `max_batches=15` (past the FP8 warmup of 4).

```bash
# profile (background it; ~2 min)
nsys profile -o run_prof --force-overwrite true --stats=false -t cuda \
  ./build/src-cli/darknet detector train \
  /home/kerrigan/datasets/LegoGears_v2/LegoGears_v2.data bench.cfg -dont_show -fp8 \
  > train.log 2>&1

# kernel ranking
nsys stats -r cuda_gpu_kern_sum --format csv run_prof.nsys-rep | head -20

# one-number total for A/B comparison
nsys stats -r cuda_gpu_kern_sum --format csv run_prof.nsys-rep \
  | awk -F, 'NR>1 && $2 ~ /^[0-9]/ {s+=$2} END {printf "%.1f ms\n", s/1e6}'

# wall-clock per iteration (skip iters 1-4: FP16 warmup, FP8 engages at iter 5)
grep -oE "train=[0-9.]+ milliseconds" train.log | tail -10
```

Reference totals (LegoGears 224x160 batch 64, 15 iters): default TF32 = 1201 ms,
FP8 as of 2026-07-06 = 1005 ms (~178 ms/iter vs 206 default).
Raw artifacts preserved in `tasks/fp8_bench_nsys/` and `tasks/fp8_bench_outputs/`.

Interpretation rules:
- `Instances` column ≈ launch count. Thousands of <10 µs kernels = launch-latency bound.
- A GEMM at 1-2% of total means precision changes are irrelevant; attack the wrappers.
- Compare against the SAME nsys workload; the profiler itself adds ~5-10%.
- Env switches for A/B from one binary: `DARKNET_FP8_DISABLE_CUDNN_CONV=1`,
  `DARKNET_FP8_DISABLE_FUSED_WGRAD_ACCUM=1`, `DARKNET_FP8_DISABLE_FUSED_DGRAD_ACCUM=1`,
  `DARKNET_FP8_DISABLE_SM89_LAYOUT_PRUNING=1`,
  `DARKNET_FP8_DISABLE_DIRECT_WGRAD_UPDATE=1`,
  `DARKNET_FP8_DISABLE_DIRECT_DGRAD_UPDATE=1`,
  `DARKNET_FP8_DEBUG=1` (prints per-layer engine/fallback choices).

### SM89 pass-elimination A/B

Use the same data, cfg, and initial checkpoint for every run. Set `max_batches=55`
(iterations 1-4 warm up; measure 5-55) and repeat each process five times.

```bash
# optimized Ada FP8
DARKNET_FP8_DEBUG=1 nsys profile -o fp8_sm89_fast --force-overwrite true --stats=false -t cuda \
  ./build/src-cli/darknet detector train \
  /home/kerrigan/datasets/LegoGears_v2/LegoGears_v2.data bench_fp8.cfg \
  /home/kerrigan/datasets/LegoGears_v2/LegoGears_v2_1000.weights -dont_show -fp8

# same binary, previous staged FP8 dataflow
DARKNET_FP8_DEBUG=1 \
DARKNET_FP8_DISABLE_SM89_LAYOUT_PRUNING=1 \
DARKNET_FP8_DISABLE_DIRECT_WGRAD_UPDATE=1 \
DARKNET_FP8_DISABLE_DIRECT_DGRAD_UPDATE=1 \
nsys profile -o fp8_sm89_staged --force-overwrite true --stats=false -t cuda \
  ./build/src-cli/darknet detector train \
  /home/kerrigan/datasets/LegoGears_v2/LegoGears_v2.data bench_fp8.cfg \
  /home/kerrigan/datasets/LegoGears_v2/LegoGears_v2_1000.weights -dont_show -fp8

# FP16 reference: bench_fp16.cfg has fp8_training=0 and cudnn_half=1; omit -fp8
./build/src-cli/darknet detector train \
  /home/kerrigan/datasets/LegoGears_v2/LegoGears_v2.data bench_fp16.cfg \
  /home/kerrigan/datasets/LegoGears_v2/LegoGears_v2_1000.weights -dont_show
```

Acceptance: optimized FP8 must reduce median post-warmup end-to-end step time by
at least 10% versus staged FP8. Profiles must also show the removed row-major
weight store and eligible 1x1 dgrad/wgrad conversion passes are absent.

---

## 3. Kernel-writing patterns (house style, `src-lib/fp8_kernels.cu`)

- **Grid geometry**: 1-D kernels use `fp8_gridsize_256(total)` + 256 threads; y-dim
  spillover is built in. Batch over images with `grid.z` + `src_stride`/`dst_stride`
  params (defaulted to 1/0/0 so old call sites compile unchanged). `gridDim.z ≤ 65535`.
- **Transpose/layout kernels**: 32x33 shared tile (`tile[32][33]` — the +1 kills bank
  conflicts), block (32,8), each thread walks 4 tile rows. See
  `fp8_quantize_e5m2_dual_layout_amax_kernel` for the canonical form.
- **amax fusion**: never do a separate amax pass; fuse block-reduce + `atomicMax` on
  `__float_as_int` into the quantize kernel. CAUTION: `fp8_record_block_amax` reduces
  over `blockDim.x` — with a 2-D (32,8) block you must inline a fixed-256 reduction
  (bug found twice).
- **Packed stores**: 4 FP8 values → one `uint32`. Requires every dst offset component
  (ld, per-image stride, col origin) % 4 == 0 — all pads here are multiples of 16, so
  gate with a `can_x4` check and keep the scalar kernel as fallback.
- **Sanitize in-kernel**: non-finite → 0.0f before quantize, always.
- **Enqueue-only**: no syncs, no mallocs, no host reads in anything under the training
  loop. Delayed-scaling state lives on-device (17 floats: 16-ring + index).
- After every launch: `CHECK_CUDA(cudaPeekAtLastError());`

### Opaque-plan wrapper pattern (for vendor libs)
`fp8_gemm.cpp` (cuBLASLt) and `fp8_conv.cpp` (cuDNN graph / cudnn-frontend) both follow:
opaque struct + `plan_create(spec, scale_ptrs...)` doing ALL heuristics/finalization at
setup (nullptr on unsupported → caller falls back), execute() enqueue-only, per-op
runtime fallback seams. Copy this shape for any new backend.

---

## 4. Memory / workspace rules

- **No per-call cudaMalloc.** Layer-lifetime buffers are allocated in
  `fp8_setup_convolutional_training_layer` (convolutional_layer.cpp) and freed in BOTH
  `fp8_release_convolutional_layer` AND `layer.cpp::free_layer_custom()` (mirror or leak).
- **Scratch comes from the shared net workspace** via offsets computed with
  `fp8_align_workspace_offset`. THE trap: the sizing computation in
  convolutional_layer.cpp must mirror the runtime offset computation in
  convolutional_kernels.cu **exactly** — they are two copies of the same layout. When
  you change one, change the other in the same commit (grep "must stay in sync").
- When adding an alternative path, size workspace as **max(old layout, new layout)** so
  runtime fallback stays safe.
- Persisting data across phases (e.g. dy^T quantized once in the wgrad loop, reused by
  dgrad): give it its own region at offset 0 and put per-phase scratch after it; guard
  with a `*_ready` flag for the fallback ordering.
- 8 GB card: batch-64 whole-batch buffers are fine for FP8 (1 byte/elem) but audit
  anything FP32 whole-batch.

---

## 5. Correctness verification

- **Bitwise regression tests** (`src-test/test_fp8_kernels.cpp`): new kernel output
  byte-compared against the reference composition (e.g. `im2col_gpu_ext` + quantize).
  These caught real bugs; write one per new kernel, include a >200k-element case to
  cover the multi-block grid path and a tail case.
- Run: `systemd-run --user --scope -p MemoryHigh=3G ./build/src-test/darknet_tests '--gtest_filter=Fp8Kernels.*:Fp8Conv.*:Fp8Gemm.*:Fp8Scaling.*'`
- **Loss parity**: 100-iter LegoGears run vs default from same starting weights — same
  loss band is the bar (exact overlay impossible: loader/augmentation not pinned).
- Numerics recipe (don't change casually): E4M3 fwd / E5M2 grads, 16-deep delayed
  scaling, dy margin 1, warmup 4 iters, first conv + head feeders excluded, mid-batch
  GEMM failure = fatal (partial-accumulation double-count guard).

---

## 6. Environment gotchas (this machine)

- **Build**: `cmake --build` is BLOCKED by a hook (regex `cmake.*build`). Use
  `systemd-run --user --scope -p MemoryHigh=3G make -C build darknetcli darknet_tests -j4`.
  Max 4 threads. ccache is not installed (configure without launcher flags).
  Configure takes ~3 min.
- **WSL GPU stack dies mid-session sometimes**: symptom = ANY CUDA binary segfaults at
  startup (even `nvidia-smi`; crash is in ld.so relocating `/usr/lib/wsl/lib/libcuda.so.1`).
  Not your code. Fix: `wsl --shutdown` from Windows. Diagnose with
  `nvidia-smi; echo $?` → 139 = dead stack.
- **Ada sm_89 hardware facts**: FP8 tensor FLOPs == FP16 at FP32 accumulate (wins come
  from fused kernels/bandwidth, not the GEMM). cuDNN FP8 conv engines: fprop YES,
  wgrad/dgrad NO (`SM80 || SM120` required — probe evidence in
  `tasks/fp8_conv_probe_2026-07-05.txt`). cuBLASLt FP8 matmul is TN-only, m/k % 16.
- Foreground `sleep` is blocked in this harness; background long commands and let the
  completion notification resume you.

---

## 7. Current state + open work

- FP8 training beats default: results log `tasks/fp8_cudnn_conv_benchmark_2026-07-05.md`.
- Next FP8-adjacent lever: BF16/TF32 cuDNN backward hybrid —
  `tasks/fp8_forward_cudnn_backward_plan.md` (phase 1 = flagged early-return in
  `backward_convolutional_layer_gpu_fp8`; caller already falls through per-op).
- Generic (non-FP8) backlog: `tasks/darknet_generic_perf_backlog.md` — the
  simple_copy/axpy launch storm (36k launches, ~163 ms/15iter) is the biggest item.
- Test-only FP8 kernel wrappers (pad-rows family etc.) kept for coverage; candidates
  for deletion with their tests if the API surface needs shrinking.
