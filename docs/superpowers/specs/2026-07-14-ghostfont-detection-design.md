# Ghost-Font Video Detection — Design Note

Date: 2026-07-14 · Branch: `fp8` · Status: implemented (code only, not yet built)

## Goal

Detect text rendered in the mixfont "ghost font" style in videos using darknet.
The network input is **not** plain RGB: each frame is fed as 4 channels — RGB
plus a motion channel computed with OpenCV from frame i vs frame i−1.  Frame i
is additionally conditioned on previous frames through `[crnn]` recurrence and
a `[tucker_attention]` block.

## Ghost-font research

The mixfont "ghost" face (https://www.mixfont.com/ghost-font) is a hollow /
outlined decorative typeface: thin double-stroke outlines, weak fill.  In
video, ghost-font text can additionally be *temporally* hidden.  Analysis of
the reference clip `ghost-message_v2.mp4` (1280×720, 30 fps, 6.2 s, random-dot
noise field, 6-px grid / 3-px marks) found **two** hidden layers:

1. **Static density watermark** — revealed by a plain temporal mean
   (`ffmpeg tmix=50..128 + boxblur + normalize`): reads "WRITTEN IN GHOST
   FONT".  A decoy.
2. **Motion-direction message** — the real message "HELLO HUMAN": background
   dots drift vertically one way, text-masked dots the opposite way at
   ±4 px/frame.  Invisible per-frame, invisible to plain pairwise diffs and to
   long averages; human vision sees it while playing (motion segregation).

Verified decode: `D+ = |shift(gray_i, +4px) − gray_{i−1}|`,
`D− = |shift(gray_i, −4px) − gray_{i−1}|`, accumulate ~100 frames, `D+ − D−`
→ clean white-on-black "HELLO HUMAN" (`ghost-message_v2_HELLO_HUMAN.png`).

## The motion channel (4th input plane)

Default mode `dirmatch`, identical math in `tools/make_ghostfont_dataset.py`
and `src-examples/darknet_20_ghostfont_video.cpp` (keep them in lock-step):

```
D+  = |shift(gray_i, +SHIFT px vertical) − gray_{i−1}|      SHIFT default 4
D−  = |shift(gray_i, −SHIFT px vertical) − gray_{i−1}|
raw = D+ − D−                    (signed)
acc = 0.9·acc + 0.1·raw          (EMA, ~10-20 frames of SNR build-up)
out = NORM_MINMAX(blur(acc,5×5)) (uint8; 128 = neutral before warmup)
```

Frames 0-1 of each video: neutral plane, empty labels, no detections; output
starts at frame 3.  Accumulator and `[crnn]` state reset per video.
Alternative modes in the dataset tool: `pairdiff` (Sobel-magnitude absdiff,
for genuinely moving targets) and `ema` (running-mean deviation, for static
density watermarks).

## What was changed in darknet

| File | Change |
|------|--------|
| `src-lib/image_opencv.cpp` | `load_rgb_mat_image`: accept `channels==4` (`IMREAD_UNCHANGED` + guard). `image_data_augmentation`: HSV jitter on RGB planes only (4th plane never photometrically augmented); fog restores the motion plane; 4-ch final conversion via `Darknet::mat_to_image`. |
| `src-lib/darknet_image.cpp` | `Darknet::load_image`: `IMREAD_UNCHANGED` + `BGRA2RGBA` branch for `channels==4` (needed by `-map` validation). |
| `src-lib/darknet_network.cpp` | `reset_network_state`: proper CRNN reset — zeroes `l.state` (CPU+GPU, correct `hidden·batch·(steps+1)` size) and the self layer's output.  Entry point: `reset_rnn()`. |
| `cfg/yolov4-tiny-ghostfont.cfg` | New: channels=4, `[tucker_attention]` on the deepest 512-ch map, `[crnn]` (output=hidden=256, shortcut=1) before each YOLO head, classes=1, `time_steps=8 track=1 augment_speed=3 sequential_subdivisions=8`, mosaic/mixup off. |
| `cfg/yolov4-tiny-ghostfont-infer.cfg` | Inference variant: batch=1, subdivisions=1, time_steps=1, track=0. |
| `tools/make_ghostfont_dataset.py` | New: labeled videos → 4-channel BGRA PNGs + YOLO txts + temporally-ordered train/valid lists (split by video, tail padding against cross-video sequence contamination). |
| `src-examples/darknet_20_ghostfont_video.cpp` | New: stateful video inference (per-video `reset_rnn`, motion accumulator, warmup frames, `Darknet::Image` predict overload — the `cv::Mat` overload strips the 4th plane). |

## Training

```
python3 tools/make_ghostfont_dataset.py --videos VIDS --labels LABELS --out ghostfont_data
darknet detector train -map -dont_show ghostfont_data/ghostfont.data cfg/yolov4-tiny-ghostfont.cfg
```

Notes: train from scratch (no 4-channel pretrained weights).  `-map` numbers
will be pessimistic — validation loads single non-sequential images so CRNN
state carry-over is garbage; final evaluation is the video app.  If CRNN goes
NaN: `try_fix_nan=1`, `shortcut=0`, single crnn block, lower LR; on the fp8
branch force fp32 on crnn sublayers if unstable.

## Inference

```
darknet_20_ghostfont_video ghostfont.names cfg/yolov4-tiny-ghostfont-infer.cfg weights.weights video.mp4
```

## Ensemble inference (phase 2)

Ghost videos vary: drift polarity can invert (messages 2/3 vs v2 needed the
opposite subtraction order) and drift speed differs.  A hardcoded decode
setting misses messages, so `darknet_20_ghostfont_video` is a self-tuning
ensemble:

1. **Hypothesis grid**: axis (vertical; horizontal with `--horizontal`) ×
   shift px/frame (`--shifts 2,3,4,6,8`) × polarity (±) — 10 hypotheses default.
2. **Pass 1 (scan)**: each hypothesis runs darknet on the first `--scan-frames`
   (default 90) frames; score = Σ detection confidences ≥ 0.25.  Recurrent
   state and motion accumulator reset per run.
3. **Pass 2 (refine)**: top `--top` (default 3) positive-scoring hypotheses
   re-run on the full video, per-frame predictions collected.
4. **Stacking**: per frame, boxes of all runs are greedily clustered at IoU
   ≥ `--iou` (0.5); cluster confidence = mean member confidence × support/K
   (support = distinct hypotheses agreeing); keep support ≥ 2 or single-run
   boxes with confidence ≥ 0.6; cluster rect = confidence-weighted average.
5. **Output**: `<video>_output.m4v` (voted boxes drawn), stdout score table,
   `<video>_ghost_report.json` (grid + scores + winner + per-frame voted
   detections).  Exit code 2 when no hypothesis detects anything.

## Verification checklist (pending — build was intentionally skipped)

1. Build; confirm `darknet_20_ghostfont_video` target appears (examples CMake GLOBs `*.cpp`).
2. Loader smoke test: inference on one RGBA PNG with the new cfg — no fatal error.
3. Tiny overfit on one short video (`max_batches=500 burn_in=50 time_steps=4 batch=8 subdivisions=2`) — loss collapses; video app shows boxes from frame 3.
4. State persistence: video app with vs without `reset_rnn` between two videos — early detections must differ.
