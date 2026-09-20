#!/usr/bin/env python3
"""Emit the `smart_resize` truth table for `qwen3vl_image.rs`.

⛔ `smart_resize` BELOW IS THE PUBLISHED FUNCTION, PASTED VERBATIM from
`transformers/models/qwen2_vl/image_processing_qwen2_vl.py` (Qwen3-VL has no image processor of its
own; its preprocessor_config.json names `Qwen2VLImageProcessorFast`). It is NOT re-derived here —
re-deriving it in the oracle is exactly how an oracle comes to agree with a wrong implementation.

⛔ THE TRAP THIS TABLE EXISTS FOR: Python's `round()` is BANKER'S rounding (half to EVEN), so
`round(2.5) == 2`. Rust's `f64::round()` is half AWAY FROM ZERO, so `2.5f64.round() == 3.0`. At
factor 32 every input of the form 32k+16 sits exactly on a half, and the two rules disagree on
alternate ones (h=80 -> 64 vs 96, h=144 -> 128 vs 160). Nothing downstream notices: both answers are
valid grids and the model runs.

    python3 qwen3vl_smart_resize.py <out.csv>
"""
import math, sys

# ---- verbatim, do not edit ------------------------------------------------------------------
def smart_resize(
    height: int, width: int, factor: int = 28, min_pixels: int = 56 * 56, max_pixels: int = 14 * 14 * 4 * 1280
):
    if max(height, width) / min(height, width) > 200:
        raise ValueError(
            f"absolute aspect ratio must be smaller than 200, got {max(height, width) / min(height, width)}"
        )
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar
# ---------------------------------------------------------------------------------------------

# This checkpoint: patch_size 16 * merge_size 2 = 32, and its own preprocessor_config.json.
FACTOR, MIN_PX, MAX_PX = 32, 4096, 1310720

cases = []
# every exact-half input on both axes — where banker's rounding and away-from-zero part company
cases += [(32 * k + 16, 32 * j + 16) for k in range(0, 9) for j in range(0, 5)]
# ordinary sizes, including ones that trip each branch
cases += [(224, 224), (448, 448), (1080, 1920), (1920, 1080), (4000, 3000), (64, 64),
          (33, 33), (31, 31), (1, 1), (17, 4000), (4000, 17), (100, 100), (96, 96),
          (1280, 720), (720, 1280), (2048, 2048), (8, 8), (7, 9), (512, 341)]
# the aspect-ratio refusal, from both sides
cases += [(1, 201), (201, 1), (1, 200), (200, 1)]

out = sys.argv[1]
n_err = 0
with open(out, "w") as f:
    f.write("h,w,out_h,out_w\n")
    for (h, w) in sorted(set(cases)):
        try:
            oh, ow = smart_resize(h, w, FACTOR, MIN_PX, MAX_PX)
            f.write(f"{h},{w},{oh},{ow}\n")
        except ValueError:
            n_err += 1
            f.write(f"{h},{w},ERR,ERR\n")
print(f"wrote {out}: {len(set(cases))} cases, {n_err} refusals")
# show the rows where the two rounding rules differ, so the table's purpose is visible
print("\nrows where banker's rounding matters (python round vs away-from-zero):")
for (h, w) in sorted(set(cases)):
    if h % FACTOR == FACTOR // 2:
        b = round(h / FACTOR) * FACTOR
        a = math.floor(h / FACTOR + 0.5) * FACTOR
        if a != b:
            print(f"  h={h}: python round -> {b}, away-from-zero -> {a}")
            break
