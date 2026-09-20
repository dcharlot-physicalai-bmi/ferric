#!/usr/bin/env python3
"""Emit bicubic-resize reference cases for `qwen3vl_image::resize_cubic`.

⛔ THE ORACLE IS PILLOW ITSELF, not a reimplementation — `Image.resize(..., Image.BICUBIC)` on a
mode="F" array, which runs Pillow's own C resampler in double precision. A hand-written reference
here would be my own arithmetic checking my own arithmetic.

⚠ Pillow's BICUBIC is the Keys kernel with **a = -0.5**, established by impulse response (residual
0.000e+00 vs a=-0.5, 3.693e-2 vs a=-0.75), NOT assumed. PyTorch/torchvision — and HuggingFace's own
`_interpolation_axis_taps_weights` — use **a = -0.75**. These fixtures therefore pin the SUPPORT-SCALED
SEPARABLE MACHINERY (which is shared) at a = -0.5; the kernel constant is tested separately as a
closed form. torchvision is not installed on this machine, so no claim is made here about matching it.

    python3 qwen3vl_resize_ref.py <out-dir>
"""
import struct, sys, numpy as np
from PIL import Image

out = sys.argv[1]

def src(h, w, seed):
    # deterministic, high-frequency content — a smooth ramp would hide a wrong kernel entirely
    y, x = np.mgrid[0:h, 0:w]
    return ((np.sin(y * 0.7 + seed) * np.cos(x * 0.9 - seed) * 120 + 128
             + ((y * 13 + x * 7) % 31)).astype(np.float32))

def dump(path, a):
    a = np.ascontiguousarray(a, dtype="<f4").ravel()
    with open(path, "wb") as f:
        f.write(struct.pack("<I", a.size)); f.write(a.tobytes())

# (src_h, src_w, dst_h, dst_w) — downscale is the case Qwen3-VL actually hits, but upscale and
# non-integer ratios exercise the support/offset arithmetic differently.
CASES = [(37, 53, 32, 32), (64, 64, 96, 96), (100, 80, 32, 64),
         (48, 48, 48, 48), (17, 9, 32, 32), (128, 96, 32, 32)]
idx = []
for i, (sh, sw, dh, dw) in enumerate(CASES):
    a = src(sh, sw, i)
    got = np.asarray(Image.fromarray(a, mode="F").resize((dw, dh), Image.BICUBIC), dtype=np.float32)
    assert got.shape == (dh, dw), got.shape
    dump(f"{out}/resize{i}.src", a)
    dump(f"{out}/resize{i}.dst", got)
    idx.append(f"{sh},{sw},{dh},{dw}")
    print(f"  case {i}: {sh}x{sw} -> {dh}x{dw}  range [{got.min():.3f}, {got.max():.3f}]")
open(f"{out}/resize_cases.csv", "w").write("src_h,src_w,dst_h,dst_w\n" + "\n".join(idx) + "\n")
print(f"wrote {len(CASES)} cases to {out}")
