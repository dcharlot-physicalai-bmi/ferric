#!/usr/bin/env python3
"""Independent SmolVLA f32 reference.

Reads the bytes written by `ferric-smolvla network-forward` and reruns the
LeRobot a0eb860d sample_actions math in PyTorch. BF16 weights are cast to
float32 for the comparable path. A separate BF16 matmul is attempted and
reported on its own. This script is not the inference runtime.
"""

import argparse
import json
import math
import struct
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open


def load_f32(path):
    raw = Path(path).read_bytes()
    return torch.tensor(struct.unpack("<" + "f" * (len(raw) // 4), raw))


def load_u32(path):
    raw = Path(path).read_bytes()
    return list(struct.unpack("<" + "I" * (len(raw) // 4), raw))


def err(a, b):
    d = (a.float() - b.float()).abs()
    return {"max_abs": float(d.max()), "rmse": float(torch.sqrt((d * d).mean())), "n": int(d.numel())}


def gelu_tanh(x):
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x.pow(3))))


def pixel_shuffle(x, scale):
    b, seq, dim = x.shape
    side = int(seq**0.5)
    x = x.view(b, side, side, dim)
    x = x.view(b, side, side // scale, dim * scale)
    x = x.permute(0, 2, 1, 3)
    x = x.reshape(b, side // scale, side // scale, dim * scale * scale)
    x = x.permute(0, 2, 1, 3)
    return x.reshape(b, seq // (scale * scale), dim * scale * scale)


def resize_with_pad(img, height, width):
    # img: [1,3,h,w] in [0,1]
    _, _, h, w = img.shape
    if h == height and w == width:
        return img
    ratio = max(w / width, h / height)
    rh = int(h / ratio)
    rw = int(w / ratio)
    out = F.interpolate(img, size=(rh, rw), mode="bilinear", align_corners=False)
    return F.pad(out, (width - rw, 0, height - rh, 0), value=0.0)


U = 2.0 ** -23
ROPE_MEASURED = {"max_norm_abs": 0.0, "max_angle_abs": 0.0, "pairs": 0}


def gamma(n):
    nu = n * U
    return nu / (1.0 - nu)


def observe_rope(x, y, positions, base=10000.0):
    """Same derived RoPE invariant as ferric_smolvla::rope_invariant_holds."""
    dh = x.shape[-1]
    half = dh // 2
    x0 = x[..., :half].double()
    x1 = x[..., half:].double()
    y0 = y[..., :half].double()
    y1 = y[..., half:].double()
    n_in = torch.sqrt(x0 * x0 + x1 * x1)
    n_out = torch.sqrt(y0 * y0 + y1 * y1)
    norm_err = (n_out - n_in).abs()
    norm_bound = (2.0 ** 0.5) * gamma(5) * (x0.abs() + x1.abs())
    if torch.any(norm_err > norm_bound):
        raise RuntimeError(f"reference rope norm {float(norm_err.max())} exceeds {float(norm_bound.max())}")
    freq = base ** (torch.arange(half, dtype=torch.float64) * (2.0 / dh))
    expected = positions.to(torch.float64)[:, None] / freq[None, :]
    expected = expected[None, :, None, :]
    n2 = x0 * x0 + x1 * x1
    determined = n2 > U
    c_hat = torch.where(determined, (x0 * y0 + x1 * y1) / n2.clamp_min(1e-30), torch.zeros_like(n2))
    s_hat = torch.where(determined, (x0 * y1 - x1 * y0) / n2.clamp_min(1e-30), torch.zeros_like(n2))
    recovered = torch.atan2(s_hat, c_hat)
    delta = torch.remainder(recovered - expected + torch.pi, 2 * torch.pi) - torch.pi
    ang_err = torch.where(determined, delta.abs(), torch.zeros_like(delta))
    ang_bound = gamma(5) * (1.0 + expected.abs())
    if torch.any(ang_err > ang_bound):
        raise RuntimeError(f"reference rope angle {float(ang_err.max())} exceeds {ang_bound}")
    ROPE_MEASURED["max_norm_abs"] = max(ROPE_MEASURED["max_norm_abs"], float(norm_err.max()))
    ROPE_MEASURED["max_angle_abs"] = max(ROPE_MEASURED["max_angle_abs"], float(ang_err.max()))
    ROPE_MEASURED["pairs"] += int(determined.sum())


def rope(x, positions, base=10000.0):
    # x [1, seq, heads, dh]
    dh = x.shape[-1]
    half = dh // 2
    freq = base ** (torch.arange(half, device=x.device, dtype=torch.float32) * (2.0 / dh))
    ang = positions.to(torch.float32)[:, None] / freq[None, :]
    sin = torch.sin(ang)[:, None, :]
    cos = torch.cos(ang)[:, None, :]
    x1, x2 = x[..., :half], x[..., half:]
    y = torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)
    observe_rope(x, y, positions, base)
    return y


def linear(x, w, b=None):
    return F.linear(x.to(dtype=w.dtype), w, None if b is None else b.to(dtype=w.dtype))


class Ref:
    def __init__(self, path, dtype):
        self.f = safe_open(path, framework="pt")
        self.dtype = dtype
        self.cache = {}

    def t(self, name):
        if name not in self.cache:
            w = self.f.get_tensor(name)
            if w.dtype == torch.bfloat16 and self.dtype == torch.bfloat16:
                self.cache[name] = w
            else:
                self.cache[name] = w.float()
        return self.cache[name]

    def vision(self, pixels):
        w = self.t("model.vlm_with_expert.vlm.model.vision_model.embeddings.patch_embedding.weight")
        b = self.t("model.vlm_with_expert.vlm.model.vision_model.embeddings.patch_embedding.bias")
        hidden = F.conv2d(pixels.to(w.dtype), w, b, stride=16)
        hidden = hidden.flatten(2).transpose(1, 2)
        pos = self.t("model.vlm_with_expert.vlm.model.vision_model.embeddings.position_embedding.weight")
        hidden = hidden + pos
        patch = hidden[0, 0].detach().clone()
        layer0 = None
        for i in range(12):
            p = f"model.vlm_with_expert.vlm.model.vision_model.encoder.layers.{i}"
            residual = hidden
            n = F.layer_norm(hidden, (768,), self.t(f"{p}.layer_norm1.weight"), self.t(f"{p}.layer_norm1.bias"), 1e-6)
            q = linear(n, self.t(f"{p}.self_attn.q_proj.weight"), self.t(f"{p}.self_attn.q_proj.bias"))
            k = linear(n, self.t(f"{p}.self_attn.k_proj.weight"), self.t(f"{p}.self_attn.k_proj.bias"))
            v = linear(n, self.t(f"{p}.self_attn.v_proj.weight"), self.t(f"{p}.self_attn.v_proj.bias"))
            hidden = attn(q, k, v, 12, 12, None) 
            hidden = linear(hidden, self.t(f"{p}.self_attn.out_proj.weight"), self.t(f"{p}.self_attn.out_proj.bias"))
            hidden = hidden + residual
            residual = hidden
            n = F.layer_norm(hidden, (768,), self.t(f"{p}.layer_norm2.weight"), self.t(f"{p}.layer_norm2.bias"), 1e-6)
            hidden = linear(n, self.t(f"{p}.mlp.fc1.weight"), self.t(f"{p}.mlp.fc1.bias"))
            hidden = gelu_tanh(hidden)
            hidden = linear(hidden, self.t(f"{p}.mlp.fc2.weight"), self.t(f"{p}.mlp.fc2.bias"))
            hidden = hidden + residual
            if i == 0:
                layer0 = hidden[0, 0].detach().clone()
        hidden = F.layer_norm(
            hidden,
            (768,),
            self.t("model.vlm_with_expert.vlm.model.vision_model.post_layernorm.weight"),
            self.t("model.vlm_with_expert.vlm.model.vision_model.post_layernorm.bias"),
            1e-6,
        )
        post = hidden[0, 0].detach().clone()
        shuffled = pixel_shuffle(hidden, 4)
        tokens = linear(shuffled, self.t("model.vlm_with_expert.vlm.model.connector.modality_projection.proj.weight"))
        return tokens, patch, layer0, post, tokens[0, 0].detach().clone()


def attn(q, k, v, hq, hkv, mask):
    # q,k,v: [1, seq, heads*dh]
    b, sq, qd = q.shape
    sk = k.shape[1]
    dh = qd // hq
    q = q.view(b, sq, hq, dh).transpose(1, 2)
    k = k.view(b, sk, hkv, dh).transpose(1, 2)
    v = v.view(b, sk, hkv, dh).transpose(1, 2)
    if hkv != hq:
        k = k.repeat_interleave(hq // hkv, dim=1)
        v = v.repeat_interleave(hq // hkv, dim=1)
    scores = torch.matmul(q.float(), k.float().transpose(-1, -2)) * (dh**-0.5)
    if mask is not None:
        scores = scores.masked_fill(~mask[:, None, :, :], torch.finfo(torch.float32).min)
    probs = torch.softmax(scores, dim=-1).to(v.dtype)
    out = torch.matmul(probs, v.float() if v.dtype == torch.float32 else v)
    if v.dtype != torch.float32:
        out = out.to(v.dtype)
    return out.transpose(1, 2).reshape(b, sq, hq * dh)


def rms(x, w, eps=1e-5):
    return x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + eps) * w


def swiglu(ref, prefix, x, dim, inter):
    n = rms(x, ref.t(f"{prefix}.post_attention_layernorm.weight"))
    gate = linear(n, ref.t(f"{prefix}.mlp.gate_proj.weight"))
    up = linear(n, ref.t(f"{prefix}.mlp.up_proj.weight"))
    return linear(F.silu(gate) * up, ref.t(f"{prefix}.mlp.down_proj.weight"))


def text_stack(ref, hidden, mask, pos):
    cache = []
    k0 = None
    seq = hidden.shape[1]
    for layer in range(16):
        p = f"model.vlm_with_expert.vlm.model.text_model.layers.{layer}"
        n = rms(hidden, ref.t(f"{p}.input_layernorm.weight"))
        q = linear(n, ref.t(f"{p}.self_attn.q_proj.weight")).view(1, seq, 15, 64)
        k = linear(n, ref.t(f"{p}.self_attn.k_proj.weight")).view(1, seq, 5, 64)
        v = linear(n, ref.t(f"{p}.self_attn.v_proj.weight")).view(1, seq, 5, 64)
        q = rope(q, pos)
        k = rope(k, pos)
        if layer == 0:
            k0 = k[0, 0].reshape(-1).detach().clone()
        qh = q.reshape(1, seq, 15 * 64)
        kh = k.reshape(1, seq, 5 * 64)
        vh = v.reshape(1, seq, 5 * 64)
        out = attn(qh, kh, vh, 15, 5, mask)
        out = linear(out, ref.t(f"{p}.self_attn.o_proj.weight"))
        residual = hidden + out
        hidden = residual + swiglu(ref, p, residual, 960, 2560)
        cache.append((k.detach(), v.detach()))
    return cache, k0


def sinusoidal(time, dim=720):
    half = dim // 2
    frac = torch.linspace(0, 1, half, dtype=torch.float64)
    period = 0.004 * (4.0 / 0.004) ** frac
    ang = (1.0 / period) * 2 * math.pi * time
    return torch.cat([torch.sin(ang), torch.cos(ang)]).float()


def denoise(ref, cache, pad, x, time, prefix_valid):
    action = linear(x, ref.t("model.action_in_proj.weight"), ref.t("model.action_in_proj.bias"))
    time_row = sinusoidal(time).view(1, 1, 720).expand_as(action)
    fused = torch.cat([action, time_row], dim=-1)
    mid = F.silu(linear(fused, ref.t("model.action_time_mlp_in.weight"), ref.t("model.action_time_mlp_in.bias")))
    hidden = linear(mid, ref.t("model.action_time_mlp_out.weight"), ref.t("model.action_time_mlp_out.bias"))
    offset = int(pad.sum())
    pos = torch.arange(50) + offset
    pos0 = torch.arange(50)
    prefix = pad.numel()
    self_mask = torch.zeros(1, 50, prefix + 50, dtype=torch.bool)
    self_mask[0, :, :prefix] = pad
    for i in range(50):
        self_mask[0, i, prefix : prefix + i + 1] = True
    cross = pad.view(1, 1, prefix).expand(1, 50, prefix)
    for layer in range(16):
        p = f"model.vlm_with_expert.lm_expert.layers.{layer}"
        n = rms(hidden, ref.t(f"{p}.input_layernorm.weight"))
        q = linear(n, ref.t(f"{p}.self_attn.q_proj.weight"))
        if layer % 2 == 0:
            k = linear(n, ref.t(f"{p}.self_attn.k_proj.weight")).view(1, 50, 5, 64)
            v = linear(n, ref.t(f"{p}.self_attn.v_proj.weight")).view(1, 50, 5, 64)
            qv = rope(q.view(1, 50, 15, 64), pos).reshape(1, 50, 15 * 64)
            k = rope(k, pos)
            kk = torch.cat([cache[layer][0], k], dim=1).reshape(1, prefix + 50, 5 * 64)
            vv = torch.cat([cache[layer][1], v], dim=1).reshape(1, prefix + 50, 5 * 64)
            out = attn(qv, kk, vv, 15, 5, self_mask)
        else:
            pk = cache[layer][0].reshape(1, prefix, 320)
            pv = cache[layer][1].reshape(1, prefix, 320)
            k = linear(pk, ref.t(f"{p}.self_attn.k_proj.weight"))
            v = linear(pv, ref.t(f"{p}.self_attn.v_proj.weight"))
            qv = rope(q.view(1, 50, 15, 64), pos0).reshape(1, 50, 15 * 64)
            out = attn(qv, k, v, 15, 5, cross)
        out = linear(out, ref.t(f"{p}.self_attn.o_proj.weight"))
        residual = hidden + out
        hidden = residual + swiglu(ref, p, residual, 720, 2048)
    hidden = rms(hidden, ref.t("model.vlm_with_expert.lm_expert.norm.weight"))
    return linear(hidden, ref.t("model.action_out_proj.weight"), ref.t("model.action_out_proj.bias"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--trace-dir", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dtype", choices=["f32", "bf16"], default="f32")
    args = ap.parse_args()
    d = Path(args.trace_dir)
    images = load_f32(d / "images.f32").view(3, 3, 256, 256)
    state = load_f32(d / "state.f32")
    noise = load_f32(d / "noise.f32").view(1, 50, 32)
    tokens = load_u32(d / "tokens.u32")
    mask_bytes = (d / "mask.u8").read_bytes()
    task = (d / "task.txt").read_text()
    rust_prepared = load_f32(d / "prepared_image0.f32").view(3, 512, 512)

    prepared = []
    for i in range(3):
        img = resize_with_pad(images[i : i + 1], 512, 512)
        prepared.append(img * 2 - 1)
    prep_err = err(prepared[0][0], rust_prepared)

    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(args.tokenizer)
    tok.enable_padding(length=48, pad_id=2, pad_token="<|im_end|>", direction="right")
    tok.enable_truncation(max_length=48)
    ids = tok.encode(task).ids
    token_match = ids == tokens

    dtype = torch.float32 if args.dtype == "f32" else torch.bfloat16
    ref = Ref(args.weights, dtype)
    snaps = []
    embs = []
    for img in prepared:
        tokens_i, patch, layer0, post, conn = ref.vision(img)
        embs.append(tokens_i * math.sqrt(960))
        snaps.append((patch, layer0, post, conn))
    lang = ref.t("model.vlm_with_expert.vlm.model.text_model.embed_tokens.weight")[tokens].unsqueeze(0) * math.sqrt(960)
    state32 = torch.zeros(1, 32)
    state32[0, : state.numel()] = state
    state_tok = linear(state32, ref.t("model.state_proj.weight"), ref.t("model.state_proj.bias"))
    prefix = torch.cat(embs + [lang, state_tok.unsqueeze(0) if state_tok.ndim == 2 else state_tok], dim=1)
    # lang is [1,48,960], state_tok is [1,960]
    if state_tok.ndim == 2:
        prefix = torch.cat(embs + [lang, state_tok.unsqueeze(1)], dim=1)
    lang_mask = torch.tensor([[b != 0 for b in mask_bytes]], dtype=torch.bool)
    pad = torch.cat([
        torch.ones(1, 64 * 3, dtype=torch.bool),
        lang_mask,
        torch.ones(1, 1, dtype=torch.bool),
    ], dim=1)
    att = torch.zeros(1, pad.shape[1], dtype=torch.int32)
    att[0, -1] = 1
    cum = torch.cumsum(att, dim=1)
    att2 = (cum[:, None, :] <= cum[:, :, None]) & (pad[:, None, :] & pad[:, :, None])
    pos = torch.cumsum(pad.to(torch.int32), dim=1) - 1
    cache, k0 = text_stack(ref, prefix, att2, pos[0])
    x = noise.clone()
    dt = -1.0 / 10
    velocity0 = None
    for step in range(10):
        time = 1.0 + step * dt
        v = denoise(ref, cache, pad[0], x, time, int(pad.sum()))
        if step == 0:
            velocity0 = v[0, 0].detach().clone()
        x = x + dt * v
    profile = {
        "dtype": args.dtype,
        "weight_cast": "bf16 expanded to f32, f32 tensors left as f32" if args.dtype == "f32" else "bf16 matmul kept in torch.bfloat16; f32 projection weights stay f32; compared after cast to f32",
        "input_class": "synthetic_protocol",
        "executable": False,
        "token_ids_match_tokenizer_json": token_match,
        "prepared_image0": prep_err,
        "patch_token0": err(snaps[0][0], load_f32(d / "patch_token0.f32")),
        "vision_l0_token0": err(snaps[0][1], load_f32(d / "vision_l0_token0.f32")),
        "vision_post_token0": err(snaps[0][2], load_f32(d / "vision_post_token0.f32")),
        "connector_token0": err(snaps[0][3], load_f32(d / "connector_token0.f32")),
        "state_token": err(state_tok[0], load_f32(d / "state_token.f32")),
        "text_l0_k_token0": err(k0, load_f32(d / "text_l0_k_token0.f32")),
        "velocity0": err(velocity0, load_f32(d / "velocity0.f32")),
        "raw_actions": err(x[0].reshape(-1), load_f32(d / "raw_actions.f32")),
    }
    profile["rope_invariant"] = {
        "max_norm_abs": ROPE_MEASURED["max_norm_abs"],
        "max_angle_abs": ROPE_MEASURED["max_angle_abs"],
        "pairs_checked": ROPE_MEASURED["pairs"],
        "holds": True,
        "evidence": "measured on this reference forward",
    }
    Path(args.out).write_text(json.dumps(profile, indent=2))
    print(json.dumps(profile, indent=2))


if __name__ == "__main__":
    main()
