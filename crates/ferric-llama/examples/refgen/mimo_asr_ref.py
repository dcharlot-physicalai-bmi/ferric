#!/usr/bin/env python3
"""MiMo-V2.5-ASR from the MODEL AUTHORS' OWN CODE, waveform to logits, stage by stage.

The authors' implementation is Xiaomi's repository (github.com/XiaomiMiMo/MiMo-V2.5-ASR): `MimoAudio`
(`src/mimo_audio/mimo_audio.py`) drives `MiMoAudioTokenizer` (audio -> 8 RVQ code channels at 25 Hz)
and `MiMoAudioForCausalLM` (codes -> a 4-frame patch encoder -> a Qwen2 LM -> text). Every stage below
runs THEIR functions, in the environment their requirements.txt pins (torch 2.6.0, torchaudio 2.6.0,
transformers 4.49.0), refused otherwise:

  resample  their `resample_audio_if_needed` (torchaudio sinc resampler), 16 kHz -> 24 kHz
  mel       their `wav2mel` (torchaudio MelSpectrogram, |X|, log clipped at 1e-7)
  tokenizer their `AudioEncoder`: 2 convs, 32 transformer layers, a skip, 2x pooling, then RVQ codes
  prompt    their `get_asr_sft_prompt` + `InputSegment.to_input_id` (text and 8 code channels)
  embeds    their `_prepare_input_embeds`: code embeddings -> the 6-layer patch encoder -> downcast
  LM        their `Qwen2Model`, STREAMED one decoder layer at a time (refgen/stream.py), and lm_head

⭐ PRECISION: THE AUTHORS' DEPLOYED WEIGHTS, AT FLOAT32 ARITHMETIC. Both checkpoints are stored in float32,
and the authors' code rounds every weight to bfloat16 when it loads them (`from_pretrained(torch_dtype=
torch.bfloat16)`, `mimo_audio_tokenizer.eval().bfloat16()`; the RVQ codebooks are then cast back with
`quantizer.float()`, still carrying the rounding). Here their `__init__` runs as written, so their own
casts do that rounding; the tokenizer is then upcast to float32 (exact) and the LM streamed at float32
from the same rounded values. Their bfloat16 ARITHMETIC is run separately (`codes_bf16`) to measure how
many codes their own deployment flips: RVQ codes are an argmin, and a near-tie is decided by rounding.

⛔ Deviations, each recorded in the fixture:
  - `flash_attn` is shimmed (shims/flash_attn): per-sequence softmax attention, refusing any causal or
    windowed call. The encoder uses neither.
  - `random.choice` over their prompt templates is pinned to the first English template; the language
    tag is "<english>", as their demo sends for English.
Their modelling code itself runs UNMODIFIED.

⛔⛔ AND IT KEEPS ITS bfloat16 ISLANDS AT FLOAT32. `_prepare_input_embeds` sums the code embeddings into a
buffer it creates as `torch.bfloat16`; that bf16 tensor then makes the patch transformer's first RMSNorm
round its output (`weight * x.to(input_dtype)`) and its rotary module return `cos.to(x.dtype)` — at any
weight dtype. A first version of this harness replaced that literal with the model's dtype, saying their
code could not run at float32. It runs; the "correction" had silently removed all three roundings from
the reference, and Ferric was matched to a model the authors never run. An adversarial review found it.

⛔ THE PATCH ENCODER IS BIDIRECTIONAL ONLY UNDER SDPA. Their code asks for it with `is_causal=False`,
which transformers 4.49 forwards to SDPA; the eager path would build a causal mask and ignore it. The
generator asserts SDPA and PROVES the attention is bidirectional (a causal run must differ).

⛔ THE HARNESS IS ITSELF CHECKED: `--selftest` builds a tiny random MiMoAudioForCausalLM with the authors'
class, gives it weights that are NOT bf16-representable (so the rounding on load is exercised), and requires
the streamed path to reproduce their own `forward`: the last row's hidden state through their `lm_head`
bit for bit, and the harness's own chunked head (the one the fixture's logits come from) to within the
~3e-6 a BLAS kernel's row count can move a product. It runs before every fixture.

    <venv>/python mimo_asr_ref.py --src <checkout> <asr_dir> <tokenizer_dir> <audio_16k.wav> | gzip -9 > fx.json.gz
    <venv>/python mimo_asr_ref.py --src <checkout> --selftest
    <venv>/python mimo_asr_ref.py --src <checkout> --greedy <asr_dir> <tokenizer_dir> <audio.wav> <id,id,...>
"""
import gc
import json
import os
import random
import subprocess
import sys
import tempfile
import wave

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "shims"))   # flash_attn -> the shim
sys.path.insert(0, HERE)
import stream  # noqa: E402

PINNED = {"torch": "2.6.0", "torchaudio": "2.6.0", "transformers": "4.49.0"}
TAP_LAYERS = [0, 15, 31]             # tokenizer encoder layers whose outputs are recorded
EN_TAG = "<english>"


def die(msg):
    raise SystemExit(f"⛔ {msg} — refusing to emit a fixture")


def import_authors(src):
    import torchaudio
    import transformers
    got = {"torch": torch.__version__.split("+")[0], "torchaudio": torchaudio.__version__.split("+")[0],
           "transformers": transformers.__version__}
    if got != PINNED:
        die(f"environment {got} is not the authors' pinned {PINNED}")
    sys.path.insert(0, src)
    import flash_attn
    if not os.path.abspath(flash_attn.__file__).startswith(os.path.join(HERE, "shims")):
        die(f"flash_attn resolved to {flash_attn.__file__}, not the shim")
    from src.mimo_audio import mimo_audio as MA
    from src.mimo_audio import modeling_mimo_audio as MM
    commit = subprocess.run(["git", "-C", src, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    return MA, MM, {**got, "authors_commit": commit or "unknown"}


def read_wav16(path):
    with wave.open(path, "rb") as w:
        if w.getnchannels() != 1 or w.getsampwidth() != 2:
            die(f"{path}: need mono 16-bit PCM")
        rate = w.getframerate()
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float32) / 32768.0
    return torch.from_numpy(pcm), rate


def meta_skeleton(MM, asr_dir, args):
    """MiMoAudioForCausalLM built on the META device: structure only; nothing of the 30 GB is read."""
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(asr_dir)
    cfg._attn_implementation = "sdpa"
    with torch.device("meta"):
        m = MM.MiMoAudioForCausalLM(cfg, args)
    for name, sub in [("model", m.model), ("input_local_transformer", m.input_local_transformer)]:
        if sub.config._attn_implementation != "sdpa":
            die(f"{name} runs {sub.config._attn_implementation}, not sdpa — the patch encoder would go causal")
    return m.eval()


def materialize(module, prefix, ckpt, dtype, round_to):
    """Rebuild a meta submodule on the CPU and load it strict from the checkpoint, rounded like theirs."""
    module.to_empty(device="cpu")
    sd = {}
    for name, t in module.state_dict().items():
        key = prefix + name
        if key not in ckpt.where:
            continue   # non-persistent buffers (rotary inv_freq) are recomputed below
        v = ckpt.get(key)
        if tuple(v.shape) != tuple(t.shape):
            die(f"{key}: checkpoint {tuple(v.shape)} != module {tuple(t.shape)}")
        sd[name] = (v.to(round_to) if round_to is not None else v).to(dtype)
    missing = [n for n in module.state_dict() if n not in sd and "inv_freq" not in n]
    if missing:
        die(f"{prefix}: no checkpoint value for {missing[:3]}")
    module.load_state_dict(sd, strict=False)
    module.to(dtype)
    for n, v in sd.items():
        if not torch.equal(module.state_dict()[n], v):
            die(f"{prefix}{n} differs from the checkpoint after loading")
    # recompute rotary tables on the CPU (their buffers are non-persistent)
    for sub in module.modules():
        if hasattr(sub, "inv_freq") and hasattr(sub, "rope_init_fn"):
            inv, sub.attention_scaling = sub.rope_init_fn(sub.config, "cpu")
            sub.inv_freq = inv
            sub.original_inv_freq = inv
    return len(sd)


def streamed_lm(skel, ckpt, input_ids, dtype, round_to, report, taps=None):
    """Their `_prepare_input_embeds` + `Qwen2Model` forward with one decoder layer resident, and lm_head.
    Returns logits [T_groups, V]."""
    for name, prefix in [("speech_embeddings", "speech_embeddings."),
                         ("input_local_transformer", "input_local_transformer."),
                         ("speech_group_downcast", "speech_group_downcast."),
                         ("embed_tokens", "model.embed_tokens."), ("norm", "model.norm."),
                         ("rotary_emb", "model.rotary_emb.")]:
        mod = getattr(skel.model, name) if name in ("embed_tokens", "norm", "rotary_emb") else getattr(skel, name)
        report["tensors_loaded"] += materialize(mod, prefix, ckpt, dtype, round_to)
    layers = skel.model.layers
    for i in range(len(layers)):
        if not isinstance(layers[i], stream.StreamedLayer):
            layers[i] = stream.StreamedLayer(type(layers[i]), skel.model.config, i, f"model.layers.{i}.", ckpt,
                                             report, dtype, round_to)
    hooks = []
    if taps is not None:
        hooks.append(skel.input_local_transformer.register_forward_hook(
            lambda m, a, o: taps.__setitem__("patch_encoder", o.last_hidden_state.detach().clone())))
    with torch.no_grad():
        emb = skel._prepare_input_embeds(input_ids)
        if taps is not None:
            taps["inputs_embeds"] = emb[0].detach().clone()
        T = emb.shape[1]
        h = skel.model(inputs_embeds=emb, attention_mask=torch.ones(1, T, dtype=torch.bool),
                       position_ids=torch.arange(T)[None], use_cache=False, return_dict=True).last_hidden_state[0]
        W = ckpt.get("lm_head.weight")
        logits = torch.empty(T, W.shape[0], dtype=dtype)
        for c in range(0, W.shape[0], 16384):
            w = W[c:c + 16384]
            logits[:, c:c + 16384] = torch.nn.functional.linear(h, (w.to(round_to) if round_to is not None else w).to(dtype))
    if taps is not None:
        taps["hidden"] = h.detach().clone()
    for hk in hooks:
        hk.remove()
    if report["layers_streamed"] < len(layers):
        die(f"{report['layers_streamed']} of {len(layers)} layers ran")
    return logits


def selftest(MM):
    """Tiny random MiMoAudioForCausalLM: the streamed path must equal the authors' forward exactly."""
    from safetensors.torch import save_file
    from transformers import Qwen2Config
    torch.manual_seed(0)
    cfg = Qwen2Config(hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=4,
                      num_key_value_heads=2, vocab_size=300, rope_theta=640000.0, rms_norm_eps=1e-6,
                      max_position_embeddings=4096, head_dim=16, speech_vocab_size="17-17-9-9",
                      speech_zeroemb_idx="16-16-8-8", delay_pattern="0-1-2-3", group_size=4, audio_channels=4,
                      local_dim=32, local_layers=1, local_attn_heads=4, local_ffn_dim=64, input_local_layers=2,
                      input_local_dim=32, input_full_attention=True, attention_bias=True)
    cfg._attn_implementation = "sdpa"
    args = MM.MiMoAudioArguments(model_name_or_path="tiny", sosp_idx=290, eosp_idx=291, empty_idx=292,
                                 sostm_idx=293, eostm_idx=294, eot_idx=295)
    full = MM.MiMoAudioForCausalLM(cfg, args).eval()
    raw = {}
    with torch.no_grad():
        for n, p in full.named_parameters():
            # float32 values that are NOT bf16-representable: the file holds these, and both paths must
            # round them on load exactly as the authors' `from_pretrained(torch_dtype=bfloat16)` does
            raw[n] = (torch.randn_like(p) * 0.5 + (1.0 if n.endswith("norm.weight") else 0.0)).contiguous()
            p.copy_(raw[n].to(torch.bfloat16).float())
    if all(torch.equal(v, v.to(torch.bfloat16).float()) for v in raw.values()):
        die("self-test weights are all bf16-representable — the rounding path would go unexercised")
    G = 9   # groups: 2 text, sosp, 4 speech, eosp, 1 text
    text = [5, 6, 290] + [292] * 4 + [291, 7]
    ids = torch.full((1, 5, G * 4), -100, dtype=torch.long)
    for g, t in enumerate(text):
        ids[0, 0, g * 4] = t
        for c, (empty, vs) in enumerate(zip([16, 16, 8, 8], [16, 16, 8, 8])):
            ids[0, 1 + c, g * 4:(g + 1) * 4] = torch.randint(0, vs, (4,)) if t == 292 else empty
    with torch.no_grad():
        want = full(input_ids=ids, attention_mask=torch.ones(1, G, dtype=torch.bool),
                    position_ids=torch.arange(G)[None]).text_logits[0, -1]
    with tempfile.TemporaryDirectory() as d:
        sd = {k: v.contiguous() for k, v in full.state_dict().items()}
        sd.update(raw)                                    # the file holds the UNROUNDED weights
        save_file(sd, os.path.join(d, "model.safetensors"))
        ckpt = stream.Checkpoint(d)
        with torch.device("meta"):
            skel = MM.MiMoAudioForCausalLM(cfg, args).eval()
        rep = {"layers_streamed": 0, "tensors_loaded": 0}
        t = {}
        lg = streamed_lm(skel, ckpt, ids, torch.float32, torch.bfloat16, rep, t)
    # (1) the wiring, exactly: the streamed hidden state's last row through the authors' own head call
    got = full.lm_head(t["hidden"][-1:])[0]
    d = (got - want).abs().max().item()
    if d != 0.0:
        die(f"harness self-test: streamed != the authors' forward by {d:.3e}")
    # (2) the harness's OWN head — chunked, from the file, rounded — which is what the fixture records. A
    # 9-row product may round differently from their 1-row one (measured 2.9e-6), so a tolerance here,
    # and the rows before the last are checked against the same head applied row by row.
    dh = (lg[-1] - want).abs().max().item()
    rows = torch.stack([full.lm_head(t["hidden"][i:i + 1])[0] for i in range(lg.shape[0])])
    dr = (lg - rows).abs().max().item()
    if dh > 1e-5 * max(1.0, want.abs().max().item()) or dr > 1e-5 * max(1.0, rows.abs().max().item()):
        die(f"harness self-test: the fixture's head differs from the authors' (last row {dh:.3e}, all rows {dr:.3e})")
    return {"tiny_config": "LM 2 layers, patch encoder 2 layers, 4 code channels, 9 groups; weights not bf16-representable",
            "max_abs_last_logit_diff": d, "fixture_head_last_row": dh, "fixture_head_all_rows": dr}


def rows_record(x, cols):
    g = lambda v, dd: float(f"{float(v):.{dd}g}")
    x = x.detach().to(torch.float64)
    return {"sum": [g(v, 10) for v in x.sum(1)], "ssq": [g(v, 10) for v in (x * x).sum(1)],
            "sample": [[g(v, 8) for v in r] for r in x[:, cols]]}


def main():
    argv = sys.argv[1:]
    if "--src" not in argv:
        die("--src <MiMo-V2.5-ASR checkout> is required")
    i = argv.index("--src"); src = argv[i + 1]; del argv[i:i + 2]
    MA, MM, env = import_authors(src)
    corrections = []   # their modelling code runs unmodified (see the module docstring)
    st = selftest(MM)
    if argv == ["--selftest"]:
        print(json.dumps({"selftest": st, "env": env})); return
    greedy = argv[0] == "--greedy"
    if greedy:
        argv = argv[1:]
    asr_dir, tok_dir, wav_path = argv[:3]

    # ---- their MimoAudio.__init__, with the 30 GB LM kept on the META device ---------------------------
    skel_holder = {}
    real_tok_from_pretrained = MA.MiMoAudioTokenizer.from_pretrained

    def lm_from_pretrained(path, args=None, **kw):
        skel_holder["m"] = meta_skeleton(MM, path, args)
        return skel_holder["m"]
    MA.MiMoAudioForCausalLM.from_pretrained = staticmethod(lm_from_pretrained)
    MA.MiMoAudioTokenizer.from_pretrained = staticmethod(lambda p, **kw: real_tok_from_pretrained(p, **kw))
    import contextlib
    with contextlib.redirect_stdout(sys.stderr):        # their code prints; stdout carries the fixture
        A = MA.MimoAudio(asr_dir, tok_dir, device="cpu")    # their init: tokenizer .bfloat16(), mel transform
    skel = skel_holder["m"]
    tok_model = A.mimo_audio_tokenizer
    if next(tok_model.parameters()).dtype != torch.bfloat16:
        die("their init did not leave the tokenizer in bfloat16")
    del tok_model.decoder                                # never used by ASR
    gc.collect()

    pcm, rate = read_wav16(wav_path)
    wav24 = A.resample_audio_if_needed(pcm, rate)        # their resampler

    def encode_codes(bf16):
        tok_model.to(torch.bfloat16 if bf16 else torch.float32)   # bf16 -> f32 is exact
        taps = {}
        hooks = [tok_model.encoder.layers[0].register_forward_pre_hook(
                     lambda m, a: taps.__setitem__("conv_out", a[0].detach().float().clone()))]
        for li in TAP_LAYERS:
            hooks.append(tok_model.encoder.layers[li].register_forward_hook(
                lambda m, a, o, li=li: taps.__setitem__(f"enc_layer{li}", o.detach().float().clone())))
        hooks.append(tok_model.encoder.layer_norm.register_forward_hook(
            lambda m, a, o: taps.__setitem__("enc_norm", o.detach().float().clone())))
        hooks.append(tok_model.encoder.down_sample_norm.register_forward_hook(
            lambda m, a, o: taps.__setitem__("pooled", o.detach().float().clone())))
        mels = []
        orig = A.wav2mel
        A.wav2mel = lambda w: (lambda m: (mels.append(m.detach().clone()), m)[1])(orig(w))
        random.choice = lambda seq: seq[0]               # pin the template
        with torch.no_grad(), contextlib.redirect_stdout(sys.stderr):
            ids = A.get_asr_sft_prompt(wav24, audio_tag=EN_TAG)
        A.wav2mel = orig
        for h in hooks:
            h.remove()
        return ids, taps, mels

    ids16, _, _ = encode_codes(bf16=True)                # their deployment arithmetic
    ids, taps, mels = encode_codes(bf16=False)           # the same weights at float32
    if len(mels) != 1:
        die(f"{len(mels)} mel chunks: this fixture covers one 30 s chunk")
    text_ch = ids[0, ::A.group_size]
    speech = text_ch == A.empty_token
    codes = ids[1:, :].reshape(A.audio_channels, -1, A.group_size)[:, speech, :].reshape(A.audio_channels, -1)
    codes16 = ids16[1:, :].reshape(A.audio_channels, -1, A.group_size)[:, speech, :].reshape(A.audio_channels, -1)
    if not torch.equal(ids16[0], ids[0]):
        die("the text channel depends on arithmetic precision")

    ckpt = stream.Checkpoint(asr_dir)
    report = {"mode": "LM streamed at float32 from bf16-rounded weights, one decoder layer resident",
              "layers_streamed": 0, "tensors_loaded": 0}
    if greedy:
        cont = [int(x) for x in argv[3].split(",")]
        empty = torch.tensor(A.speech_zeroemb_idx, dtype=ids.dtype)
        ext = [ids]
        for t in cont[:-1]:
            g = torch.empty(A.audio_channels + 1, A.group_size, dtype=ids.dtype)
            g[0, :] = t; g[1:, :] = empty[:, None]
            ext.append(g)
        full_ids = torch.cat(ext, dim=1)
        lg = streamed_lm(skel, ckpt, full_ids[None], torch.float32, torch.bfloat16, report)
        n_prompt = ids.shape[1] // A.group_size
        tail = lg[n_prompt - 1:]
        top2 = torch.topk(tail, 2, dim=1)
        am = top2.indices[:, 0].tolist()
        json.dump({"env": env, "harness_selftest": st, "corrections": corrections, "load": report,
                   "audio": os.path.basename(wav_path), "prompt_text_ids": text_ch.tolist(), "continuation": cont,
                   "authors_argmax": am, "margin": [round(float(a - b), 5) for a, b in top2.values.tolist()],
                   "agrees": [a == c for a, c in zip(am, cont)],
                   "decoded": A.tokenizer.decode(cont)}, sys.stdout)
        return

    lm_taps = {}
    logits = streamed_lm(skel, ckpt, ids[None], torch.float32, torch.bfloat16, report, lm_taps)

    # ⛔ prove the patch encoder attended bidirectionally: the same group, causal, must differ
    with torch.no_grad():
        x = torch.randn(1, A.group_size, skel.input_local_config.hidden_size)
        full_o = skel.input_local_transformer(inputs_embeds=x, is_causal=False).last_hidden_state
        causal_o = skel.input_local_transformer(inputs_embeds=x, is_causal=True).last_hidden_state
    if torch.equal(full_o[0, 0], causal_o[0, 0]):
        die("the patch encoder's first frame ignores the later frames — it ran causal")

    # ---- ⭐ THE NOISE FLOOR: the same code at float64, same (bf16-rounded) weights, same audio ------------
    # Their explicit float32 casts stay where they wrote them (rope angles, RMSNorm variance), and so do
    # the patch encoder's bf16 islands. The tokenizer's attention scores are float32 because the shim
    # carries the flash-attn KERNEL's arithmetic over (it has no float64 mode) — not a cast of theirs.
    # Their `encode()` casts to float32 before the quantizer, so the float64 RVQ is done here by their
    # EuclideanCodebook.quantize formula on the same rounded codebooks.
    import copy
    tok_model.to(torch.float64)
    taps64, hooks = {}, []
    enc = tok_model.encoder
    hooks.append(enc.layers[0].register_forward_pre_hook(lambda m, a: taps64.__setitem__("conv_out", a[0].detach().clone())))
    for li in TAP_LAYERS:
        hooks.append(enc.layers[li].register_forward_hook(
            lambda m, a, o, li=li: taps64.__setitem__(f"enc_layer{li}", o.detach().clone())))
    hooks.append(enc.layer_norm.register_forward_hook(lambda m, a, o: taps64.__setitem__("enc_norm", o.detach().clone())))
    hooks.append(enc.down_sample_norm.register_forward_hook(lambda m, a, o: taps64.__setitem__("pooled", o.detach().clone())))
    wav24_64 = A.resample_audio_if_needed(pcm.double(), rate)
    mt64 = copy.deepcopy(A.mel_transform).double()
    mel64 = torch.log(torch.clip(mt64(wav24_64[None]), min=1e-7)).squeeze().transpose(0, 1)
    with torch.no_grad():
        enc.encode(input_features=mel64, input_lens=torch.tensor([mel64.shape[0]]), use_quantizer=False)
        r = taps64["pooled"].clone(); codes64 = []; margins64 = []
        for layer in enc.quantizer.vq.layers[:A.audio_channels]:
            e = layer._codebook.embed.double()
            dist = -(r.pow(2).sum(1, keepdim=True) - 2 * r @ e.t() + e.pow(2).sum(1)[None])
            top2 = torch.topk(dist, 2, dim=1)
            idx = top2.indices[:, 0]
            codes64.append(idx.tolist()); margins64.append((top2.values[:, 0] - top2.values[:, 1]).tolist())
            r = r - e[idx]
    for h in hooks:
        h.remove()
    taps64["mel"] = mel64
    n25 = taps["pooled"].shape[0]
    codes64_agree = float((torch.tensor(codes64) == codes[:, :n25]).float().mean())
    # the LM at float64 on the float32 run's ids, so the floor measures arithmetic alone
    skel64 = meta_skeleton(MM, asr_dir, skel.args)
    rep64 = {"mode": "LM streamed at float64 from bf16-rounded weights", "layers_streamed": 0, "tensors_loaded": 0}
    lm64 = {}
    logits64 = streamed_lm(skel64, ckpt, ids[None], torch.float64, torch.bfloat16, rep64, lm64)

    rng = random.Random(20260926)
    stages = {}
    mel = mels[0].transpose(0, 1)                         # [frames, n_mels]
    # every row, 8 columns (12 elsewhere): per-row sum/ssq cover the whole row, the samples locate errors
    stages["mel"] = {"shape": list(mel.shape), "cols": sorted(rng.sample(range(mel.shape[1]), 16))[::2]}
    stages["mel"].update(rows_record(mel, stages["mel"]["cols"]))
    for name, t in taps.items():
        cols = sorted(random.Random(f"{name}/20260926").sample(range(t.shape[1]), 24))[::2]
        stages[name] = {"shape": list(t.shape), "cols": cols, **rows_record(t, cols)}
    pe = lm_taps["patch_encoder"]                         # [groups, 4, 1024]: speech groups only are defined
    pe = pe[speech].reshape(-1, pe.shape[-1])
    cols = sorted(random.Random("patch_encoder/20260926").sample(range(pe.shape[1]), 24))[::2]
    stages["patch_encoder"] = {"shape": list(pe.shape), "cols": cols, **rows_record(pe, cols)}
    emb = lm_taps["inputs_embeds"]
    cols = sorted(random.Random("inputs_embeds/20260926").sample(range(emb.shape[1]), 24))[::2]
    stages["inputs_embeds"] = {"shape": list(emb.shape), "cols": cols, **rows_record(emb, cols)}

    stages64 = {}
    for name, t in taps64.items():
        stages64[name] = rows_record(t, stages[name]["cols"])
    pe64 = lm64["patch_encoder"][speech].reshape(-1, lm64["patch_encoder"].shape[-1])
    stages64["patch_encoder"] = rows_record(pe64, stages["patch_encoder"]["cols"])
    stages64["inputs_embeds"] = rows_record(lm64["inputs_embeds"], stages["inputs_embeds"]["cols"])

    V = logits.shape[1]
    sample = sorted(random.Random(20260924).sample(range(V), 128))

    def logit_rows(lg):
        rows = []
        for t in range(lg.shape[0]):
            r = lg[t]
            top = torch.topk(r, 10)
            rows.append({"top": [[int(i), round(float(v), 6)] for v, i in zip(top.values, top.indices)],
                         "sample": [round(float(r[i]), 6) for i in sample],
                         "sum": float(r.double().sum()), "ssq": float((r.double() ** 2).sum())})
        return rows
    rows = logit_rows(logits)
    wg = lambda v: float(f"{float(v):.8g}")
    pos = sorted(random.Random("wav24/20260926").sample(range(wav24.shape[0]), 256))
    json.dump({
        "model": "XiaomiMiMo/MiMo-V2.5-ASR + XiaomiMiMo/MiMo-Audio-Tokenizer",
        "asr_snapshot": os.path.basename(os.path.normpath(asr_dir)),
        "tokenizer_snapshot": os.path.basename(os.path.normpath(tok_dir)),
        "code": "the authors' repository (MimoAudio, MiMoAudioTokenizer, MiMoAudioForCausalLM), their pinned env",
        "env": env, "shims": ["flash_attn.flash_attn_varlen_func -> per-sequence softmax attention (refgen/shims/flash_attn)"],
        "corrections": corrections, "harness_selftest": st, "load": report,
        "precision": "weights rounded to bfloat16 by the authors' own load casts; arithmetic float32",
        "prompt": {"template": "asr_en_templates[0]", "audio_tag": EN_TAG},
        "audio": os.path.basename(wav_path), "rate_in": rate, "samples_in": int(pcm.shape[0]),
        "pcm_sha256": __import__("hashlib").sha256((pcm * 32768).round().to(torch.int16).numpy().astype("<i2").tobytes()).hexdigest(),
        "wav24": {"n": int(wav24.shape[0]), "sum": float(wav24.double().sum()),
                  "ssq": float((wav24.double() ** 2).sum()), "pos": pos, "values": [wg(wav24[p]) for p in pos]},
        "text_ids": text_ch.tolist(), "is_speech": speech.int().tolist(),
        "frames_25hz": int(taps["pooled"].shape[0]),
        "codes": codes.tolist(),
        "codes_bf16_agreement": float((codes16 == codes).float().mean()),
        "codes_bf16_disagree_per_channel": (codes16 != codes).sum(1).tolist(),
        "stages": stages, "vocab": V, "sample_ids": sample, "rows": rows,
        "float64": {"note": "the same code at float64 (their explicit float32 casts kept; RVQ by their formula)",
                    "load": rep64, "codes": codes64, "codes_agree_with_f32": codes64_agree,
                    "min_margin_per_channel": [round(min(m), 6) for m in margins64],
                    "stages": stages64, "rows": logit_rows(logits64)},
    }, sys.stdout)


if __name__ == "__main__":
    main()
