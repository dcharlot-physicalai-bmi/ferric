#!/usr/bin/env python3
"""Quantized checkpoints dequantized by THEIR DEFINING LIBRARIES — weights, and the authors' model on them.

Two jobs, one per subcommand:

  weights <checkpoint> <out.bin> [--native]
      Every quantized linear's weight [out, in], dequantized by the format's own library at float32:
        gptq               gptqmodel.utils.model_dequant.convert_gptq_file   (GPTQModel's dequantizer)
                           + with --native: each module's runtime dequantize_weight() (float16, as
                             transformers + GPTQModel run it) — compared as round_f16(ferric)
        awq                gptqmodel's vendored AutoAWQ `dequantize_gemm` (reverse_awq_order), scales
                           taken at float32; ALSO gptqmodel's offline `convert_awq_file`, recorded
                           under `offline:` names (it skips the AWQ order — see ferric_load::quant::awq)
        compressed-tensors the library's compressor `decompress` (scale upcast to float32)
        fp8                transformers' own load (dtype=float32 -> Fp8Dequantize on any machine
                           without an FP8 GPU)
        modelopt           nvidia-modelopt NVFP4QTensor.dequantize(fast=False)
        gguf               libggml-base's `to_float` (the authors of the GGUF types)
      written as `u32 name_len, name, u64 n, f32[n]`, names in GGUF form (blk.N.attn_q.weight), so the
      gate compares them against `quant_logits --weights` bit for bit.

  logits <checkpoint> <out.json.gz> --base <hf model id for config/tokenizer>
      The AUTHORS' model code (transformers AutoModelForCausalLM, the checkpoint's own architecture)
      carrying exactly those dequantized weights, run at float32 AND float64 — the noise floor — on
      the same text lm_floor_ref.py uses. Records, per position: a fixed seeded 128-id vocabulary sample
      (full precision), the full row's sum of squares, the float64 top-10.

<checkpoint> is a Hub id / local dir (safetensors formats) or a .gguf path.
"""
import argparse
import ctypes
import glob
import gzip
import json
import os
import random
import struct
import sys

import numpy as np
import torch

TEXT = ("The heron stood motionless in the shallows while the tide turned. A fisherman on the far bank "
        "counted his nets twice, then a third time, because the numbers never agreed. Measurements, he "
        "thought, are only as honest as the instrument and the person reading it. In 1854 John Snow "
        "mapped cholera deaths around a single water pump on Broad Street; the pattern was plain once the "
        "points were on paper. def mean(xs): return sum(xs) / len(xs)  # an empty list divides by zero. "
        "Über den Wolken muss die Freiheit wohl grenzenlos sein. 月が綺麗ですね。 The answer is 42.")

HF2GGUF = {"self_attn.q_proj": "attn_q", "self_attn.k_proj": "attn_k", "self_attn.v_proj": "attn_v",
           "self_attn.o_proj": "attn_output", "mlp.gate_proj": "ffn_gate", "mlp.up_proj": "ffn_up",
           "mlp.down_proj": "ffn_down"}


def gguf_name(prefix):
    """model.layers.3.self_attn.q_proj -> blk.3.attn_q.weight"""
    parts = prefix.split(".")
    i = parts.index("layers")
    return f"blk.{parts[i + 1]}.{HF2GGUF['.'.join(parts[i + 2:])]}.weight"


def snapshot(ck):
    if os.path.isdir(ck):
        return ck
    from huggingface_hub import snapshot_download
    return snapshot_download(ck, allow_patterns=["*.json", "*.safetensors"])


def shards(d):
    return sorted(glob.glob(os.path.join(d, "*.safetensors")))


def qconfig(d):
    cfg = json.load(open(os.path.join(d, "config.json")))
    return cfg, cfg.get("quantization_config") or cfg.get("compression_config") or {}


# ---------------------------------------------------------------- per-format library dequantizers
def deq_gptq(d, native=False):
    from gptqmodel.utils.model_dequant import convert_gptq_file
    cfg, qc = qconfig(d)
    out = {}
    for f in shards(d):
        t = convert_gptq_file(__import__("pathlib").Path(f), torch.float32, qc, "cpu")
        for k, v in t.items():
            if k.endswith(".weight") and ".layers." in k and any(s in k for s in HF2GGUF):
                out[gguf_name(k[:-len(".weight")])] = v.float()
    lib = "gptqmodel %s convert_gptq_file (float32)" % __import__("gptqmodel").__version__
    if native:
        from transformers import AutoModelForCausalLM
        m = AutoModelForCausalLM.from_pretrained(d, dtype=torch.float32, device_map="cpu")
        for n, mod in m.named_modules():
            if hasattr(mod, "dequantize_weight") and ".layers." in n:
                out["native:" + gguf_name(n)] = mod.dequantize_weight().t().contiguous()
        lib += " + runtime %s.dequantize_weight()" % type(mod).__name__
    return out, lib


def deq_awq(d):
    from gptqmodel.quantization.awq.utils.packing_utils import dequantize_gemm
    from gptqmodel.utils.model_dequant import convert_awq_file
    from safetensors import safe_open
    cfg, qc = qconfig(d)
    gs = qc.get("group_size", qc.get("q_group_size", 128))
    out = {}
    for f in shards(d):
        with safe_open(f, "pt") as r:
            keys = set(r.keys())
            for k in sorted(keys):
                if not k.endswith(".qweight"):
                    continue
                p = k[:-len(".qweight")]
                w = dequantize_gemm(r.get_tensor(p + ".qweight"), r.get_tensor(p + ".qzeros"),
                                    r.get_tensor(p + ".scales").float(), 4, gs)
                out[gguf_name(p)] = w.t().contiguous().float()
        t = convert_awq_file(__import__("pathlib").Path(f), torch.float32, "cpu")
        for k, v in t.items():
            if k.endswith(".weight") and ".layers." in k and any(s in k for s in HF2GGUF):
                out["offline:" + gguf_name(k[:-len(".weight")])] = v.float()
    return out, "gptqmodel %s vendored AutoAWQ dequantize_gemm (scales float32); offline: convert_awq_file" % __import__("gptqmodel").__version__


def deq_ct(d):
    import compressed_tensors as ct
    from compressed_tensors.compressors.base import BaseCompressor
    from compressed_tensors.quantization import QuantizationScheme
    from safetensors import safe_open
    cfg, qc = qconfig(d)
    groups = qc["config_groups"]
    if len(groups) != 1:
        raise SystemExit("multi-group configs: extend this reference")
    g = next(iter(groups.values()))
    scheme = QuantizationScheme(targets=g["targets"], weights=g["weights"], input_activations=None, output_activations=None)
    fmt = g.get("format") or qc["format"]
    # 0.19 registers classes with classmethod decompress(state_dict, scheme); 0.11 instances with
    # decompress_weight(compressed_data, quantization_args). Both are the library's own entry points.
    try:
        comp = BaseCompressor.get_value_from_registry(fmt)
    except Exception:
        comp = BaseCompressor.load_from_registry(fmt)
    tensors = {}
    for f in shards(d):
        with safe_open(f, "pt") as r:
            for k in r.keys():
                tensors[k] = r.get_tensor(k)
    prefixes = sorted({k.rsplit(".", 1)[0] for k in tensors if k.endswith((".weight_packed", ".weight_scale"))})
    out = {}
    for p in prefixes:
        if not any(s in p for s in HF2GGUF):
            continue
        sd = {k[len(p) + 1:]: v for k, v in tensors.items() if k.startswith(p + ".") and k[len(p) + 1:].startswith("weight")}
        # the library's arithmetic, run at float32: its _dequantize computes in the SCALE's dtype
        if sd["weight_scale"].dtype in (torch.bfloat16, torch.float16):
            sd["weight_scale"] = sd["weight_scale"].float()
        if fmt == "nvfp4-pack-quantized":
            from compressed_tensors.compressors.nvfp4.helpers import unpack_fp4_from_uint8
            from compressed_tensors.quantization.lifecycle.forward import dequantize
            packed = sd["weight_packed"]
            un = unpack_fp4_from_uint8(packed, packed.shape[0], packed.shape[1] * 2, dtype=torch.float32)
            w = dequantize(x_q=un, scale=sd["weight_scale"].to(torch.float32), args=scheme.weights,
                           global_scale=sd["weight_global_scale"], dtype=torch.float32)
        else:
            if hasattr(comp, "decompress_weight"):
                inst = comp() if isinstance(comp, type) else comp
                w = inst.decompress_weight(sd, scheme.weights)
            else:
                w = comp.decompress(sd, scheme)["weight"]
        out[gguf_name(p)] = w.float()
    return out, "compressed-tensors %s %s decompress (scale float32)" % (ct.__version__, fmt)


def deq_fp8(d):
    """transformers' own FP8 dequantizer — the op its loader attaches on any machine without an FP8 GPU —
    called per (weight, weight_scale_inv) pair with the float32 output dtype a float32 load asks for."""
    from transformers.integrations.finegrained_fp8 import Fp8Dequantize
    from safetensors import safe_open
    import transformers
    op = Fp8Dequantize(None)
    out = {}
    for f in shards(d):
        with safe_open(f, "pt") as r:
            for k in sorted(r.keys()):
                if not k.endswith(".weight_scale_inv"):
                    continue
                p = k[:-len("_scale_inv")]
                if any(s in p for s in HF2GGUF):
                    out[gguf_name(p[:-len(".weight")])] = op._dequantize_one(r.get_tensor(p), r.get_tensor(k), torch.float32).float()
    return out, "transformers %s Fp8Dequantize._dequantize_one (float32 output)" % transformers.__version__


def deq_modelopt(d):
    import modelopt
    from modelopt.torch.quantization.qtensor.nvfp4_tensor import NVFP4QTensor
    from safetensors import safe_open
    out = {}
    for f in shards(d):
        with safe_open(f, "pt") as r:
            for k in sorted(r.keys()):
                if not k.endswith(".weight_scale_2"):
                    continue
                p = k[:-len(".weight_scale_2")]
                if not any(s in p for s in HF2GGUF):
                    continue
                q = r.get_tensor(p + ".weight")
                shape = (q.shape[0], q.shape[1] * 2)
                qt = NVFP4QTensor(torch.Size(shape), torch.float32, q)
                w = qt.dequantize(dtype=torch.float32, fast=False, scale=r.get_tensor(p + ".weight_scale"),
                                  double_scale=r.get_tensor(p + ".weight_scale_2"), block_sizes={-1: 16})
                out[gguf_name(p)] = w.float()
    return out, "nvidia-modelopt %s NVFP4QTensor.dequantize(fast=False)" % modelopt.__version__


# ---------------------------------------------------------------- GGUF through libggml
class Traits(ctypes.Structure):
    _fields_ = [("type_name", ctypes.c_char_p), ("blck_size", ctypes.c_int64), ("blck_size_interleave", ctypes.c_int64),
                ("type_size", ctypes.c_size_t), ("is_quantized", ctypes.c_bool),
                ("to_float", ctypes.c_void_p), ("from_float_ref", ctypes.c_void_p)]


def gguf_all(path, lib_path="/opt/homebrew/lib/libggml-base.dylib"):
    lib = ctypes.CDLL(lib_path)
    lib.ggml_get_type_traits.restype = ctypes.POINTER(Traits)
    lib.ggml_get_type_traits.argtypes = [ctypes.c_int]
    TO = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int64)
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
    u32(); nt = u64(); nkv = u64(); align = 32
    for _ in range(nkv):
        k = rstr(); t = u32()
        if k == "general.alignment" and t == 4: align = u32()
        else: skip(t)
    infos = []
    for _ in range(nt):
        name = rstr(); nd = u32(); dims = [u64() for _ in range(nd)]; ty = u32(); off = u64()
        infos.append((name, dims, ty, off))
    base = (f.tell() + align - 1) // align * align
    out, types = {}, {}
    for name, dims, ty, off in infos:
        n = int(np.prod(dims))
        f.seek(base + off)
        if ty == 0:
            v = np.frombuffer(f.read(4 * n), dtype="<f4").copy()
        else:
            tr = lib.ggml_get_type_traits(ty).contents
            raw = f.read(n // tr.blck_size * tr.type_size)
            buf = ctypes.create_string_buffer(raw, len(raw))
            o = (ctypes.c_float * n)()
            TO(tr.to_float)(ctypes.cast(buf, ctypes.c_void_p), o, n)
            v = np.frombuffer(o, dtype=np.float32).copy()
            types[name] = tr.type_name.decode()
        shape = list(reversed(dims))
        out[name] = torch.from_numpy(v.reshape(shape) if len(shape) > 1 else v)
    return out, types


GG2HF = {"attn_norm": "input_layernorm", "attn_q": "self_attn.q_proj", "attn_k": "self_attn.k_proj",
         "attn_v": "self_attn.v_proj", "attn_output": "self_attn.o_proj", "attn_q_norm": "self_attn.q_norm",
         "attn_k_norm": "self_attn.k_norm", "ffn_norm": "post_attention_layernorm", "ffn_gate": "mlp.gate_proj",
         "ffn_up": "mlp.up_proj", "ffn_down": "mlp.down_proj"}


def gguf_to_hf(name):
    if name == "token_embd.weight": return "model.embed_tokens.weight"
    if name == "output_norm.weight": return "model.norm.weight"
    if name == "output.weight": return "lm_head.weight"
    p = name.split(".")
    assert p[0] == "blk", name
    return f"model.layers.{p[1]}.{GG2HF[p[2]]}.{p[3]}"


# ---------------------------------------------------------------- dispatch
def dequant_all(ck, native=False):
    if ck.endswith(".gguf"):
        w, types = gguf_all(ck)
        q = {k: v for k, v in w.items() if k in types and k.startswith("blk.") and types[k] not in ("f16", "bf16")
             and k.rsplit(".", 2)[-2] in ("attn_q", "attn_k", "attn_v", "attn_output", "ffn_gate", "ffn_up", "ffn_down")}
        return q, "libggml-base 0.25.3 to_float (%s)" % ", ".join(sorted(set(types[k] for k in q)))
    d = snapshot(ck)
    cfg, qc = qconfig(d)
    method = qc.get("quant_method")
    if os.path.exists(os.path.join(d, "hf_quant_config.json")) or qc.get("quant_library") == "modelopt":
        return deq_modelopt(d)
    return {"gptq": lambda: deq_gptq(d, native), "awq": lambda: deq_awq(d), "compressed-tensors": lambda: deq_ct(d),
            "fp8": lambda: deq_fp8(d)}[method]()


def write_bin(path, tensors):
    with open(path, "wb") as f:
        for k in sorted(tensors):
            v = tensors[k].detach().to(torch.float32).contiguous().numpy().ravel()
            b = k.encode()
            f.write(struct.pack("<I", len(b))); f.write(b)
            f.write(struct.pack("<Q", v.size)); f.write(v.astype("<f4").tobytes())


def dense_state(ck, qw):
    """Every tensor of the dense model: quantized linears from `qw` (GGUF names), the rest as stored."""
    sd = {}
    if ck.endswith(".gguf"):
        w, types = gguf_all(ck)
        for k, v in w.items():
            sd[gguf_to_hf(k)] = v
        return sd
    from safetensors import safe_open
    d = snapshot(ck)
    for f in shards(d):
        with safe_open(f, "pt") as r:
            for k in r.keys():
                t = r.get_slice(k)
                if t.get_dtype() in ("F32", "F16", "BF16") and not k.endswith(("_scale", "_scale_inv", "scale_2", "scales", "input_global_scale", "weight_global_scale", "input_scale")):
                    sd[k] = r.get_tensor(k).float()
    for g, v in qw.items():
        if g.startswith(("native:", "offline:")):
            continue
        p = g.split(".")
        sd[f"model.layers.{p[1]}.{GG2HF[p[2]]}.weight"] = v
    return sd


def logits(ck, out, base):
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    import transformers
    qw, lib = dequant_all(ck)
    sd = dense_state(ck, qw)
    tok = AutoTokenizer.from_pretrained(base)
    ids = tok(TEXT)["input_ids"]
    cfg = AutoConfig.from_pretrained(base)
    for k in ("quantization_config", "compression_config"):
        if hasattr(cfg, k): delattr(cfg, k)
    rng = random.Random(1234)
    sample = sorted(rng.sample(range(cfg.vocab_size), 128))
    res = {}
    for dt in (torch.float32, torch.float64):
        cfg.dtype = dt
        m = AutoModelForCausalLM.from_config(cfg, attn_implementation="eager").to(dt)
        m.eval()
        missing = [k for k in m.state_dict() if k not in sd and not (k == "lm_head.weight" and cfg.tie_word_embeddings)]
        if missing:
            raise SystemExit(f"weights missing for {missing[:5]}")
        m.load_state_dict({k: v.to(dt) for k, v in sd.items() if k in m.state_dict()}, strict=False)
        if cfg.tie_word_embeddings:
            m.tie_weights()
        with torch.no_grad():
            lg = m(torch.tensor([ids])).logits[0].to(torch.float64)
        res[str(dt)] = lg
        del m
    rows32, rows64 = res["torch.float32"], res["torch.float64"]
    fx = {"model": os.path.basename(ck) if ck.endswith(".gguf") else ck, "base": base, "dequantizer": lib, "transformers": transformers.__version__, "torch": torch.__version__,
          "attn_implementation": "eager", "ids": ids, "sample_ids": sample, "positions": list(range(len(ids))),
          "float32": [], "float64": []}
    for t in range(len(ids)):
        for key, rows in (("float32", rows32), ("float64", rows64)):
            r = rows[t]
            # 9 significant digits: every f32 value round-trips, and the f64 rows lose ~5e-9 relative —
            # three orders below any floor measured here.
            e = {"sample": [float("%.9g" % float(r[i])) for i in sample], "ssq": float((r * r).sum())}
            if key == "float64":
                top = torch.topk(r, 10)
                e["top"] = [[int(i), float(v)] for v, i in zip(top.values, top.indices)]
            fx[key].append(e)
    with gzip.open(out, "wt") as f:
        json.dump(fx, f)
    d = max(max(abs(a - b) for a, b in zip(x["sample"], y["sample"])) for x, y in zip(fx["float32"], fx["float64"]))
    print(f"{ck}: {len(ids)} tokens, {lib}; float32 vs float64 max |d| over the sample {d:.3e} -> {out}")


def fixture(ck, out_dir, module, keep):
    """A committed unit-test fixture: ONE module of a real checkpoint, sliced to `keep` output channels
    (columns for GPTQ/AWQ, which pack along the output in qzeros/qweight; rows otherwise), written as a
    mini checkpoint, plus what the library dequantizes THAT mini checkpoint to."""
    from safetensors import safe_open
    from safetensors.torch import save_file
    d = snapshot(ck)
    cfg, qc = qconfig(d)
    os.makedirs(out_dir, exist_ok=True)
    t = {}
    for f in shards(d):
        with safe_open(f, "pt") as r:
            for k in r.keys():
                if k.startswith(module + "."):
                    t[k[len(module) + 1:]] = r.get_tensor(k)
    method = qc.get("quant_method")
    modelopt = os.path.exists(os.path.join(d, "hf_quant_config.json")) or qc.get("quant_library") == "modelopt"
    sl = {}
    if method in ("gptq",):
        bits = qc["bits"]
        for k, v in t.items():
            if k == "qweight" or k == "scales": sl[k] = v[:, :keep]
            elif k == "qzeros": sl[k] = v[:, :keep * bits // 32]
            elif k == "bias": sl[k] = v[:keep]
            else: sl[k] = v
    elif method == "awq":
        for k, v in t.items():
            if k in ("qweight", "qzeros"): sl[k] = v[:, :keep // 8]
            elif k == "scales": sl[k] = v[:, :keep]
            elif k == "bias": sl[k] = v[:keep]
            else: sl[k] = v
    else:
        g = next(iter(qc.get("config_groups", {"x": {"weights": {}}}).values()))["weights"] if not modelopt and method != "fp8" else {}
        bits = g.get("num_bits", 8)
        for k, v in t.items():
            if k == "weight_shape": sl[k] = torch.tensor([keep, int(v[1])], dtype=v.dtype)
            elif k == "weight_zero_point" and v.dtype == torch.int32: sl[k] = v[:keep * bits // 32]
            elif k in ("weight_scale_inv",) or (k == "weight_scale" and g.get("strategy") == "block"):
                bh = cfg.get("quantization_config", {}).get("weight_block_size", [128])[0] if method == "fp8" else g["block_structure"][0]
                sl[k] = v[: (keep + bh - 1) // bh]
            elif v.ndim >= 1 and v.shape[0] > 1 and k != "weight_g_idx": sl[k] = v[:keep]
            else: sl[k] = v
    sl = {f"{module}.{k}": v.contiguous() for k, v in sl.items()}
    save_file(sl, os.path.join(out_dir, "model.safetensors"))
    json.dump({"quantization_config": qc} if not modelopt else {"quantization_config": qc}, open(os.path.join(out_dir, "config.json"), "w"), indent=1)
    if modelopt and os.path.exists(os.path.join(d, "hf_quant_config.json")):
        import shutil; shutil.copy(os.path.join(d, "hf_quant_config.json"), out_dir)
    w, lib = dequant_all(out_dir)
    w = {k: v for k, v in w.items() if ":" not in k}
    write_bin(os.path.join(out_dir, "ref.bin"), w)
    import re
    m = re.search(r"models--([^/]+?)--([^/]+)/snapshots/([0-9a-f]+)", d)
    src = {"source": f"{m.group(1)}/{m.group(2)}", "revision": m.group(3)} if m else {"source": ck}
    json.dump({**src, "module": module, "keep": keep, "dequantizer": lib}, open(os.path.join(out_dir, "source.json"), "w"), indent=1)
    sz = sum(os.path.getsize(os.path.join(out_dir, x)) for x in os.listdir(out_dir))
    print(f"{ck} :: {module}[:{keep}] -> {out_dir} ({sz // 1024} KiB), {lib}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["weights", "logits", "fixture"])
    ap.add_argument("checkpoint")
    ap.add_argument("out")
    ap.add_argument("--native", action="store_true")
    ap.add_argument("--base", default=None)
    ap.add_argument("--module", default="model.layers.0.self_attn.q_proj")
    ap.add_argument("--keep", type=int, default=32)
    A = ap.parse_args()
    if A.cmd == "weights":
        w, lib = dequant_all(A.checkpoint, A.native)
        write_bin(A.out, w)
        print(f"{A.checkpoint}: {len(w)} tensors, {lib} -> {A.out}")
    elif A.cmd == "fixture":
        fixture(A.checkpoint, A.out, A.module, A.keep)
    else:
        logits(A.checkpoint, A.out, A.base or A.checkpoint)
