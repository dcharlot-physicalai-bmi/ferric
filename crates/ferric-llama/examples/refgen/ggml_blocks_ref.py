#!/usr/bin/env python3
"""GGUF block fixtures decoded by THE AUTHORS' OWN ggml — `libggml-base`, through ctypes.

llama.cpp DEFINES the IQ and NVFP4 formats (it is the authors' code, not a peer), so the oracle for
`ferric_gguf::more_quants` is `ggml_get_type_traits(t)->to_float` in the shipped library — not a
re-reading of the C, and not gguf-py (which is checked here as a SECOND reading and must agree too).

Two kinds of fixture per type, each `<name>.raw` (block bytes) + `<name>.f32` (ggml's decode):

  * `<type>_real`  — the first blocks of a real tensor in a PUBLISHED GGUF (unsloth / bartowski /
                     a published NVFP4 file), read by byte range from the file.
  * `<type>_grid`  — blocks CRAFTED so every grid entry of the type's codebook is read at least once,
                     under varied scales and sign patterns (NVFP4: `_scales`, all 256 scale bytes x all
                     16 codes in both nibbles). A transcription error in one grid row cannot hide in a
                     real tensor that never happens to index it.

  <python> ggml_blocks_ref.py <out_dir> [--lib /opt/homebrew/lib/libggml-base.dylib]
"""
import argparse
import ctypes
import os
import struct
import sys

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--lib", default="/opt/homebrew/lib/libggml-base.dylib")
ap.add_argument("--blocks", type=int, default=16)
A = ap.parse_args()
os.makedirs(A.out, exist_ok=True)

lib = ctypes.CDLL(A.lib)


class Traits(ctypes.Structure):
    _fields_ = [("type_name", ctypes.c_char_p), ("blck_size", ctypes.c_int64), ("blck_size_interleave", ctypes.c_int64),
                ("type_size", ctypes.c_size_t), ("is_quantized", ctypes.c_bool),
                ("to_float", ctypes.c_void_p), ("from_float_ref", ctypes.c_void_p)]


lib.ggml_get_type_traits.restype = ctypes.POINTER(Traits)
lib.ggml_get_type_traits.argtypes = [ctypes.c_int]
TO_FLOAT = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int64)
TYPES = {"iq2_xs": 17, "iq1_s": 19, "iq3_s": 21, "iq2_s": 22, "iq1_m": 29, "nvfp4": 40}


def ggml_decode(ty, raw):
    tr = lib.ggml_get_type_traits(ty).contents
    n = len(raw) // tr.type_size * tr.blck_size
    assert len(raw) % tr.type_size == 0, (ty, len(raw), tr.type_size)
    buf = ctypes.create_string_buffer(bytes(raw), len(raw))
    out = (ctypes.c_float * n)()
    TO_FLOAT(tr.to_float)(ctypes.cast(buf, ctypes.c_void_p), out, n)
    return np.frombuffer(out, dtype=np.float32).copy(), tr


def gguf_tensors(path):
    """name -> (ggml type, n elements, absolute byte offset). Minimal GGUF v3 header parse."""
    f = open(path, "rb")
    def u32(): return struct.unpack("<I", f.read(4))[0]
    def u64(): return struct.unpack("<Q", f.read(8))[0]
    def rstr(): return f.read(u64()).decode("utf-8", "replace")
    sizes = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
    def skip(t):
        if t in sizes: f.read(sizes[t]); return
        if t == 8: rstr(); return
        if t == 9:
            et = u32(); n = u64()
            for _ in range(n): skip(et)
    assert f.read(4) == b"GGUF"
    u32(); nt = u64(); nkv = u64()
    align = 32
    for _ in range(nkv):
        k = rstr(); t = u32()
        if k == "general.alignment" and t == 4: align = u32()
        else: skip(t)
    infos = []
    for _ in range(nt):
        name = rstr(); nd = u32(); dims = [u64() for _ in range(nd)]; ty = u32(); off = u64()
        infos.append((name, ty, int(np.prod(dims)), off))
    start = f.tell()
    start = (start + align - 1) // align * align
    return {n: (ty, ne, start + off) for n, ty, ne, off in infos}, path


HUB = os.path.expanduser("~/.cache/huggingface/hub")
def hub(repo, fname):
    d = os.path.join(HUB, "models--" + repo.replace("/", "--"), "snapshots")
    for s in sorted(os.listdir(d)):
        p = os.path.join(d, s, fname)
        if os.path.exists(p): return p
    raise SystemExit(f"{repo}/{fname} not downloaded")


# Published files the real blocks are taken from (first tensor of each type found, in file order).
SOURCES = [
    ("unsloth/Qwen3-0.6B-GGUF", "Qwen3-0.6B-UD-IQ1_S.gguf"),
    ("unsloth/Qwen3-0.6B-GGUF", "Qwen3-0.6B-UD-IQ1_M.gguf"),
    ("unsloth/Qwen3-0.6B-GGUF", "Qwen3-0.6B-UD-IQ2_M.gguf"),
    ("bartowski/Qwen_Qwen3-0.6B-GGUF", "Qwen_Qwen3-0.6B-IQ2_M.gguf"),
    ("bartowski/Qwen_Qwen3-0.6B-GGUF", "Qwen_Qwen3-0.6B-IQ3_XS.gguf"),
    ("AIconjured/Qwen3-Embedding-0.6B-Q8-NVFP4", "Qwen3-Embedding-0.6B-NVFP4.gguf"),
]
found = {}
for repo, fname in SOURCES:
    tens, path = gguf_tensors(hub(repo, fname))
    for name, (ty, ne, off) in sorted(tens.items(), key=lambda kv: kv[1][2]):
        for tn, tid in TYPES.items():
            if ty == tid and tn not in found:
                found[tn] = (repo, fname, name, path, off)

try:
    import gguf.quants as gq
    from gguf.constants import GGMLQuantizationType as QT
except ImportError:
    gq = None


def emit(name, ty, raw, meta):
    vals, tr = ggml_decode(ty, raw)
    open(os.path.join(A.out, f"{name}.raw"), "wb").write(bytes(raw))
    open(os.path.join(A.out, f"{name}.f32"), "wb").write(vals.astype("<f4").tobytes())
    second = ""
    if gq is not None:
        py = gq.dequantize(np.frombuffer(bytes(raw), dtype=np.uint8), QT(ty)).astype(np.float32).ravel()
        same = np.array_equal(py.view(np.uint32), vals.view(np.uint32))
        second = f"gguf-py {'AGREES bit for bit' if same else 'DIFFERS: ' + str(int((py.view(np.uint32) != vals.view(np.uint32)).sum())) + ' values'}"
    nz = int((vals != 0).sum())
    print(f"{name:16s} {tr.type_name.decode():7s} {len(raw)//tr.type_size:4d} blocks  nonzero {nz}/{len(vals)}  {second}  {meta}")


for tn, ty in TYPES.items():
    if tn not in found:
        raise SystemExit(f"no published tensor of type {tn} in {SOURCES}")
    repo, fname, tname, path, off = found[tn]
    tr = lib.ggml_get_type_traits(ty).contents
    with open(path, "rb") as f:
        f.seek(off)
        raw = f.read(tr.type_size * A.blocks)
    emit(f"{tn}_real", ty, raw, f"from {repo}/{fname} :: {tname}")

rng = np.random.default_rng(20260928)


def f16_bytes(x):
    return np.float16(x).tobytes()


def rand_f16(lo=-8, hi=-2):
    # a normal f16 magnitude spread over binades, random sign-free (scales are positive in practice)
    return float(np.float16(2.0 ** rng.uniform(lo, hi)))


# ---- crafted every-grid-entry sweeps -------------------------------------------------------------
def iq2_xs_grid():
    blocks = []
    for b in range(512 // 32):
        d = f16_bytes(rand_f16())
        qs = []
        for k in range(32):
            e = b * 32 + k
            qs.append(e | (int(rng.integers(0, 128)) << 9))
        scales = rng.integers(0, 256, 8, dtype=np.uint8).tobytes()
        blocks.append(d + np.array(qs, dtype="<u2").tobytes() + scales)
    return b"".join(blocks)


def iq2_s_grid():
    blocks = []
    for b in range(1024 // 32):
        d = f16_bytes(rand_f16())
        qs, qh = bytearray(32), bytearray(8)
        for k in range(32):
            e = b * 32 + k
            ib32, l = k // 4, k % 4
            qs[k] = e & 0xff
            qh[ib32] |= ((e >> 8) & 3) << (2 * l)
        signs = rng.integers(0, 256, 32, dtype=np.uint8).tobytes()
        scales = rng.integers(0, 256, 8, dtype=np.uint8).tobytes()
        blocks.append(d + bytes(qs) + signs + bytes(qh) + scales)
    return b"".join(blocks)


def iq3_s_grid():
    blocks = []
    for b in range(512 // 64):
        d = f16_bytes(rand_f16())
        qs, qh = bytearray(64), bytearray(8)
        for k in range(64):
            e = b * 64 + k
            ib32, rem = k // 8, k % 8          # 8 lookups per 32 values: (l, first/second)
            l, second = rem // 2, rem % 2
            qs[ib32 * 8 + 2 * l + second] = e & 0xff
            # grid1 high bit at qh bit (2l), grid2 at bit (2l+1): (qh << (8-2l)) & 256 / (qh << (7-2l)) & 256
            qh[ib32] |= ((e >> 8) & 1) << (2 * l + second)
        signs = rng.integers(0, 256, 32, dtype=np.uint8).tobytes()
        scales = rng.integers(0, 256, 4, dtype=np.uint8).tobytes()
        blocks.append(d + bytes(qs) + bytes(qh) + signs + scales)
    return b"".join(blocks)


def iq1_s_grid():
    blocks = []
    for b in range(2048 // 32):
        d = f16_bytes(rand_f16())
        qs = bytearray(32)
        qh = [0] * 8
        for k in range(32):
            e = b * 32 + k
            ib, l = k // 4, k % 4
            qs[k] = e & 0xff
            qh[ib] |= ((e >> 8) & 7) << (3 * l)
        for ib in range(8):
            qh[ib] |= int(rng.integers(0, 8)) << 12 | int(rng.integers(0, 2)) << 15
        blocks.append(d + bytes(qs) + np.array(qh, dtype="<u2").tobytes())
    return b"".join(blocks)


def iq1_m_grid():
    blocks = []
    for b in range(2048 // 32):
        qs, qh = bytearray(32), bytearray(16)
        for k in range(32):
            e = b * 32 + k
            ib, l = k // 4, k % 4
            qs[k] = e & 0xff
            h = (e >> 8) & 7
            qh[2 * ib + l // 2] |= h << (4 * (l % 2))
        for i in range(16):
            qh[i] |= int(rng.integers(0, 2)) << 3 | int(rng.integers(0, 2)) << 7
        # the f16 super-scale is the top nibble of each of the four u16 scale words
        u = np.float16(rand_f16()).view(np.uint16).item()
        sc = []
        for kk in range(4):
            low12 = int(rng.integers(0, 4096))
            sc.append(low12 | (((u >> (4 * kk)) & 0xf) << 12))
        blocks.append(bytes(qs) + bytes(qh) + np.array(sc, dtype="<u2").tobytes())
    return b"".join(blocks)


def nvfp4_scales():
    blocks = []
    for b in range(64):
        d = bytes([(4 * b + s) & 0xff for s in range(4)])       # every scale byte once
        qs = bytes(((j % 16) | ((15 - (j + b) % 16) << 4)) & 0xff for j in range(32))
        blocks.append(d + qs)
    return b"".join(blocks)


for name, ty, raw in [("iq2_xs_grid", 17, iq2_xs_grid()), ("iq2_s_grid", 22, iq2_s_grid()),
                      ("iq3_s_grid", 21, iq3_s_grid()), ("iq1_s_grid", 19, iq1_s_grid()),
                      ("iq1_m_grid", 29, iq1_m_grid()), ("nvfp4_scales", 40, nvfp4_scales())]:
    emit(name, ty, raw, "crafted: every grid entry / every scale byte")
print(f"libggml-base: {A.lib}")
