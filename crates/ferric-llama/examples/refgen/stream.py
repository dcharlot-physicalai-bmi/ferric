"""The authors' model, run ONE DECODER LAYER AT A TIME — for checkpoints whose float32 reference does not fit.

A 9B model in float32 is 36 GB; with the runtime under test also resident, a 48 GB machine cannot hold
both. Nothing about the reference requires the whole model at once: the authors' `forward` walks its
layers in order and carries only the hidden state between them. So the model is built as a META-device
skeleton (structure, no storage), and each decoder layer is replaced by a proxy that, when the authors'
own loop calls it:

  1. constructs the authors' layer class on the CPU in float32,
  2. loads its weights from the checkpoint files, strict=True, upcast from the stored dtype (exact),
  3. runs it, and frees it.

The model's own forward — masks, positions, the layer loop, the final norm — runs unchanged around the
proxies. The embedding is gathered for the prompt's rows only, and the LM head is applied in vocabulary
chunks. Peak memory is one layer plus one head chunk, a few GB.

⛔ NAME MAPPING IS WHERE THIS CAN GO WRONG, silently: a composite checkpoint nests its text model
(`model.language_model.layers.3...`) where the text-only class says `model.layers.3...`, and several keys
share a tail (`model.language_model.norm.weight`, `model.visual.merger.norm.weight`, `mtp.norm.weight`).
The rule is refload.py's — exact name, else a key ending in the name minus its first component, among
those sharing that first component — plus one tie-break: the SHORTEST prefix wins (the text model sits
directly under `model.`, deeper ones belong to other towers). Anything still ambiguous REFUSES. Every
load is strict, so a parameter the mapping could not fill is an error, never a random init.
"""
import gc
import glob
import json
import os

import torch
from safetensors import safe_open


class Checkpoint:
    """The checkpoint's safetensors files, by key, read lazily in their stored dtype."""

    def __init__(self, snapshot):
        idx = os.path.join(snapshot, "model.safetensors.index.json")
        if os.path.exists(idx):
            wm = json.load(open(idx))["weight_map"]
            files = sorted(set(wm.values()))
        else:
            files = [os.path.basename(p) for p in glob.glob(os.path.join(snapshot, "*.safetensors"))]
            wm = None
        self.handles = {f: safe_open(os.path.join(snapshot, f), "pt") for f in files}
        self.where = wm or {k: f for f, h in self.handles.items() for k in h.keys()}
        self.keys = list(self.where)

    def resolve(self, name):
        if name in self.where:
            return name
        first, tail = (name.split(".", 1) + [""])[:2]
        cands = [k for k in self.keys if k.endswith("." + tail) or k == tail]
        if len(cands) > 1:
            cands = [k for k in cands if k.startswith(first + ".")] or cands
        if len(cands) > 1:
            depth = lambda k: k[: len(k) - len(tail)].count(".")
            best = min(depth(k) for k in cands)
            cands = [k for k in cands if depth(k) == best]
        if len(cands) != 1:
            raise SystemExit(f"cannot map parameter {name!r} to one checkpoint key (candidates {cands[:4]}) — refusing")
        return cands[0]

    def get(self, key):
        return self.handles[self.where[key]].get_tensor(key)


class StreamedLayer(torch.nn.Module):
    """Stands in for decoder layer `i`: builds the authors' layer, loads it, runs it, frees it."""

    def __init__(self, cls, config, i, prefix, ckpt, report):
        super().__init__()
        self._cls, self._config, self._i, self._prefix, self._ckpt, self._report = cls, config, i, prefix, ckpt, report

    def forward(self, *args, **kwargs):
        layer = self._cls(self._config, self._i).to(torch.float32)
        sd = {}
        for name, t in layer.state_dict().items():
            key = self._ckpt.resolve(self._prefix + name)
            v = self._ckpt.get(key)
            if tuple(v.shape) != tuple(t.shape):
                raise SystemExit(f"{key}: checkpoint shape {tuple(v.shape)} != layer's {tuple(t.shape)} — refusing")
            sd[name] = v.to(torch.float32)
        layer.load_state_dict(sd, strict=True)
        # By value, AS LOADED — after load_state_dict, before the forward (the lesson in refload.py).
        live = layer.state_dict()
        bad = [n for n, v in sd.items() if not torch.equal(live[n], v)]
        if bad:
            raise SystemExit(f"layer {self._i}: {len(bad)} tensors differ from the checkpoint after loading ({bad[:3]})")
        layer.eval()
        if any(m.training for m in layer.modules()):
            raise SystemExit(f"layer {self._i} has a module in training mode — refusing")
        with torch.no_grad():
            out = layer(*args, **kwargs)
        self._report["layers_streamed"] += 1
        self._report["tensors_loaded"] += len(sd)
        del layer, sd, live
        gc.collect()
        return out


def streamed_logits(auto_cls, text_config, snapshot, ids):
    """Logits [T, V] of the authors' model for `ids`, with one decoder layer resident at a time.
    Returns (logits, report, skeleton) — the skeleton carries the config and class for the record."""
    ckpt = Checkpoint(snapshot)
    report = {"mode": "streamed, one decoder layer resident", "layers_streamed": 0, "tensors_loaded": 0}
    with torch.device("meta"):
        skel = auto_cls.from_config(text_config, attn_implementation="eager")
    skel.eval()
    body = skel.model
    layers = body.layers
    if len(layers) != text_config.num_hidden_layers:
        raise SystemExit(f"skeleton has {len(layers)} layers, config says {text_config.num_hidden_layers}")
    prefix = next(n for n, m in skel.named_modules() if m is layers) + "."
    for i in range(len(layers)):
        layers[i] = StreamedLayer(type(layers[i]), text_config, i, f"{prefix}{i}.", ckpt, report)
    # The small modules the forward touches outside the layers: rebuilt on the CPU and loaded, strict.
    body.rotary_emb = type(body.rotary_emb)(config=text_config)          # non-persistent buffers: computed
    body.norm = type(body.norm)(text_config.hidden_size, eps=text_config.rms_norm_eps)
    body.norm.load_state_dict({"weight": ckpt.get(ckpt.resolve(prefix[: -len("layers.")] + "norm.weight")).float()}, strict=True)
    emb_key = ckpt.resolve(prefix[: -len("layers.")] + "embed_tokens.weight")
    emb = ckpt.get(emb_key)[torch.tensor(ids)].to(torch.float32)[None]
    report["embedding"] = emb_key
    with torch.no_grad():
        h = body(inputs_embeds=emb, use_cache=False).last_hidden_state[0]
    head_key = ckpt.resolve("lm_head.weight") if not getattr(text_config, "tie_word_embeddings", False) else emb_key
    W = ckpt.get(head_key)
    report["lm_head"] = head_key
    V = W.shape[0]
    logits = torch.empty(h.shape[0], V, dtype=torch.float32)
    with torch.no_grad():
        for c in range(0, V, 16384):
            logits[:, c:c + 16384] = h @ W[c:c + 16384].to(torch.float32).T
    report["checkpoint_keys"] = len(ckpt.keys)
    if report["layers_streamed"] != len(layers):
        raise SystemExit(f"{report['layers_streamed']} of {len(layers)} layers ran — the forward skipped some; refusing")
    return logits, report, skel
