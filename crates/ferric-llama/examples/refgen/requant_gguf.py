#!/usr/bin/env python3
"""Quantize an F32 GGUF of the AUTHORS' weights to Q8_0, or to a Q4_K/Q6_K mix shaped like Q4_K_M, so
that a quantized file's distance to the authors is PURE QUANTIZATION.

⛔ WHY THIS EXISTS. The published `Qwen/Qwen2.5-0.5B-Instruct-GGUF` q8_0 file is not a quantization of
the checkpoint the authors serve today: its layer weights differ from `transformers`' by 15-52% relative
L2 (blk.0.attn_q 0.435, blk.23.attn_v 0.516, blk.12.ffn_up 0.151) while token_embd agrees to Q8_0
noise (0.0055). A band measured on that file is mostly a revision difference, not quantization — the
q8_0 file sat at mean |dlogit| 0.36 from the authors where a Q8_0 of their own weights sits at 0.058.

    python3 requant_gguf.py <authors-F32.gguf> <out.gguf> q8_0|q4km

Block LAYOUTS are ggml's (interop: dequantize_row_q8_0 / _q4_K / _q6_K read these bytes). The SEARCH is
not llama.cpp's: plain min/max per sub-block, no refinement — so the q4km band here is WIDER than a
llama.cpp Q4_K_M (Q4_K rel L2 ~8% per tensor here), which is fine for a band and wrong for a
quality claim. q4km policy: Q6_K for output.weight and for ffn_down/attn_v in even layers, Q4_K for
every other 2-D weight, Q8_0 where a row is not a multiple of 256. Norms and biases stay F32.
"""
import struct, sys
import numpy as np

src, dst, policy = sys.argv[1], sys.argv[2], sys.argv[3]
assert policy in ("q8_0", "q4km"), policy
d = open(src, "rb").read()
o = 0
def u32():
    global o; v = struct.unpack_from("<I", d, o)[0]; o += 4; return v
def u64():
    global o; v = struct.unpack_from("<Q", d, o)[0]; o += 8; return v
def rstr():
    global o; n = u64(); v = d[o:o + n]; o += n; return v
SIZES = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
def skip(t):
    global o
    if t in SIZES: o += SIZES[t]; return
    if t == 8: rstr(); return
    if t == 9:
        et = u32(); n = u64()
        for _ in range(n): skip(et)
        return
    raise SystemExit(f"unknown GGUF value type {t}")

assert d[:4] == b"GGUF"; o = 4; ver = u32(); nt = u64(); nkv = u64()
kv0, align, ftype_off = o, 32, None
for _ in range(nkv):
    k = rstr(); t = u32()
    if k == b"general.alignment": align = struct.unpack_from("<I", d, o)[0]
    if k == b"general.file_type": ftype_off = o
    skip(t)
kv = bytearray(d[kv0:o])   # the metadata is copied verbatim; only file_type is rewritten
infos = []
for _ in range(nt):
    name = rstr().decode(); nd = u32(); dims = [u64() for _ in range(nd)]; ty = u32(); off = u64()
    infos.append([name, dims, ty, off])
base = (o + align - 1) // align * align

def h16(a): return np.asarray(a, np.float64).astype(np.float16)
def safe_div(a, b): return np.where(b != 0, a / np.where(b != 0, b, 1), 0.0)

def q8_0(x):
    b = x.reshape(-1, 32); dd = np.abs(b).max(1) / 127.0
    q = np.clip(np.round(safe_div(b, dd[:, None])), -127, 127).astype(np.int8)
    out = np.zeros((b.shape[0], 34), np.uint8)
    out[:, :2] = h16(dd).view(np.uint8).reshape(-1, 2); out[:, 2:] = q.view(np.uint8)
    return out.tobytes()

def q4_k(x):
    sb = x.reshape(-1, 8, 32); nb = sb.shape[0]
    mn = np.minimum(sb.min(2), 0.0); scale = (sb.max(2) - mn) / 15.0; minv = -mn
    dd, dm = scale.max(1) / 63.0, minv.max(1) / 63.0
    d16, m16 = h16(dd).astype(np.float64), h16(dm).astype(np.float64)
    sc = np.clip(np.round(safe_div(scale, d16[:, None])), 0, 63).astype(np.int64)
    mi = np.clip(np.round(safe_div(minv, m16[:, None])), 0, 63).astype(np.int64)
    es, em = d16[:, None] * sc, m16[:, None] * mi
    q = np.clip(np.round(safe_div(sb + em[:, :, None], es[:, :, None])), 0, 15).astype(np.uint8)
    scb = np.zeros((nb, 12), np.int64)          # inverse of ggml's get_scale_min_k4
    for j in range(4):
        scb[:, j] = sc[:, j] | ((sc[:, j + 4] >> 4) << 6)
        scb[:, j + 4] = mi[:, j] | ((mi[:, j + 4] >> 4) << 6)
        scb[:, j + 8] = (sc[:, j + 4] & 0xF) | ((mi[:, j + 4] & 0xF) << 4)
    out = np.zeros((nb, 144), np.uint8)
    out[:, 0:2] = h16(dd).view(np.uint8).reshape(-1, 2); out[:, 2:4] = h16(dm).view(np.uint8).reshape(-1, 2)
    out[:, 4:16] = scb.astype(np.uint8)
    for j in range(4):                            # sub-blocks 2j (low nibble) and 2j+1 (high) share bytes
        out[:, 16 + 32 * j:16 + 32 * j + 32] = q[:, 2 * j] | (q[:, 2 * j + 1] << 4)
    return out.tobytes()

def q6_k(x):
    sb = x.reshape(-1, 16, 16); nb = sb.shape[0]
    ext = np.take_along_axis(sb, np.abs(sb).argmax(2)[:, :, None], 2)[:, :, 0]
    scale = -ext / 32.0                           # the extreme value maps to q = -32
    dd = np.abs(scale).max(1) / 127.0; d16 = h16(dd).astype(np.float64)
    sc = np.clip(np.round(safe_div(scale, d16[:, None])), -128, 127).astype(np.int64)
    q = (np.clip(np.round(safe_div(sb, (d16[:, None] * sc)[:, :, None])), -32, 31).astype(np.int64) + 32).reshape(nb, 256)
    ql = np.zeros((nb, 128), np.int64); qh = np.zeros((nb, 64), np.int64); l = np.arange(32)
    for n in range(2):                            # ggml's dequantize_row_q6_K, inverted
        q1, q2, q3, q4 = (q[:, 128 * n + l + 32 * i] for i in range(4))
        ql[:, 64 * n + l] = (q1 & 0xF) | ((q3 & 0xF) << 4)
        ql[:, 64 * n + l + 32] = (q2 & 0xF) | ((q4 & 0xF) << 4)
        qh[:, 32 * n + l] = (q1 >> 4) | ((q2 >> 4) << 2) | ((q3 >> 4) << 4) | ((q4 >> 4) << 6)
    out = np.zeros((nb, 210), np.uint8)
    out[:, :128] = ql.astype(np.uint8); out[:, 128:192] = qh.astype(np.uint8)
    out[:, 192:208] = sc.astype(np.int8).view(np.uint8); out[:, 208:210] = h16(dd).view(np.uint8).reshape(-1, 2)
    return out.tobytes()

TY = {"q8_0": (8, q8_0, 32), "q4_k": (12, q4_k, 256), "q6_k": (14, q6_k, 256)}
def choose(name, dims):
    if len(dims) != 2 or not name.endswith(".weight") or "norm" in name: return None
    if policy == "q8_0": want = "q8_0"
    else:
        layer = int(name.split(".")[1]) if name.startswith("blk.") else -1
        even_big = ("ffn_down" in name or "attn_v" in name) and layer % 2 == 0
        want = "q6_k" if name == "output.weight" or even_big else "q4_k"
    return want if dims[0] % TY[want][2] == 0 else "q8_0"

blobs, counts = [], {}
for inf in infos:
    name, dims, ty, off = inf
    n = int(np.prod(dims))
    if ty != 0: raise SystemExit(f"{name}: the source must be F32 (found type {ty})")
    raw = d[base + off: base + off + 4 * n]
    want = choose(name, dims)
    if want:
        inf[2], fn, _ = TY[want]
        raw = fn(np.frombuffer(raw, np.float32).astype(np.float64))
    counts[want or "f32"] = counts.get(want or "f32", 0) + 1
    blobs.append(raw)
if ftype_off is not None:
    rel = ftype_off - kv0
    kv[rel:rel + 4] = struct.pack("<I", 7 if policy == "q8_0" else 15)   # MOSTLY_Q8_0 / MOSTLY_Q4_K_M
out = bytearray(b"GGUF" + struct.pack("<IQQ", ver, nt, nkv)) + kv
pad = lambda n: (n + align - 1) // align * align
offs, off = [], 0
for b in blobs: offs.append(off); off += pad(len(b))
for (name, dims, ty, _), o2 in zip(infos, offs):   # offsets are relative to the data section
    nb = name.encode(); out += struct.pack("<Q", len(nb)) + nb + struct.pack("<I", len(dims))
    for x in dims: out += struct.pack("<Q", x)
    out += struct.pack("<IQ", ty, o2)
out += b"\0" * (pad(len(out)) - len(out))
for b in blobs: out += b + b"\0" * (pad(len(b)) - len(b))
open(dst, "wb").write(out)
print(dst, counts, f"{len(out) / 1e6:.1f} MB")
