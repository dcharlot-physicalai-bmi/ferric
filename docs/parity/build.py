#!/usr/bin/env python3
"""Merge the parity audit (27 Sept 2026) into one ranked matrix: docs/parity/MATRIX.md.

Inputs, all in this folder:
  features.tsv        the feature list the audit used (S serve, E engine, F format, B backend, M model,
                      T tuning, O ops)
  peer_serve.json     14 serving peers (Ollama, llama-server, vLLM, SGLang, LM Studio, mistral.rs,
                      LocalAI, oMLX, mlx-lm, Jan, llamafile, KTransformers, Lemonade, Docker Model Runner)
                      plus what 13 front-end apps require (X-APPS)
  peer_engine.json    18 engine peers, cloned at pinned commits; every evidence URL points into them
  peer_tuning.json    fine-tuning / quantisation / distributed peers
  ferric_serve.json   Ferric's serve + ops features at 230edb9, with file:line evidence
  ferric_engine.json  Ferric's engine / format / backend / model / tuning features at 230edb9

A peer HAS a feature only where its docs or source, read during the audit, say so. `partial` counts half.
`inherited` (a wrapper documenting nothing but running an engine that has it) counts as have.

Ferric's statuses are the audit's, then UPDATED below for commits made after it, each citing the commit.

    python3 docs/parity/build.py   # rewrites docs/parity/MATRIX.md
"""
import csv, json, os

D = os.path.dirname(os.path.abspath(__file__))
feats = {r["id"]: r for r in csv.DictReader(open(f"{D}/features.tsv"), delimiter="\t")}
load = lambda f: json.load(open(f"{D}/{f}"))
peer = {}
for f in ["peer_serve.json", "peer_engine.json", "peer_tuning.json"]:
    for x in load(f)["features"]:
        peer[x["id"]] = x
fer = {}
for f in ["ferric_serve.json", "ferric_engine.json"]:
    for x in load(f)["features"]:
        fer[x["id"]] = x

# ── Ferric changes after the audit (HEAD 230edb9) ────────────────────────────────────────────────────────
UPDATES = {
    "S01": ("verified", "3f6089f: content-part arrays read (were silently empty); every sampling field honoured; tested over real sockets, batched == serial"),
    "S03": ("verified", "361cddb + da736c9: encoders (nomic-bert, bge) served beside the chat model; nomic tokenisation EQUAL to the authors', text→vector cos ≥ 0.99999975 over HTTP"),
    "S06": ("implemented", "40d4a54: /api/tags, show, ps, version, chat, generate, embed, embeddings (NDJSON, options, format, tools); serial path only"),
    "S12": ("verified", "3f6089f: logprobs / top_logprobs, raw-distribution log-softmax; unit-tested normalisation"),
    "S13": ("verified", "f8d6360: n 1..16 on chat + completions; choice i = the request with seed + i, so choice 0 IS the n = 1 answer (live: byte-equal, choice 2 = a seed+2 request, streamed = non-streamed per index); out-of-range n, n with tools, streamed completion n > 1 refused"),
    "S14": ("verified", "3f6089f: stop strings cut before the match, streamed tails held back; batched == serial over sockets"),
    "S15": ("verified", "3f6089f: temperature, top_p, top_k, min_p, presence/frequency/repeat penalty, seed; default path pinned token-for-token to the old sampler"),
    "S23": ("implemented", "da736c9: lists every loaded model (chat + embedder)"),
    "S28": ("implemented", "3f6089f: query strings no longer 404 (/health?x=1)"),
    "S35": ("implemented", "3f6089f: a prompt past <arch>.context_length is a 400 naming both numbers; no max_tokens = until EOS or the context"),
    "E19": ("partial", "87c1f6b linear + 210382c: Phi-3 LongRoPE (short/long tables switched at the authors' rule, attn_factor, cache recomputed across the switch) = the authors within 1.5-2.3x their f32 floor; decoder angles built on the host, 30k-position error 18-109x → 1.6-2.3x the floor; dynamic NTK still refused by name"),
    "M06": ("verified", "361cddb: nomic-bert (nomic-embed-text) verified against its authors at every stage; registry row added"),
    "S07": ("verified", "d185918: /v1/completions streams; tool calls stream as an SSE delta"),
    "S17": ("verified", "b0b4448: the GGUF's own Jinja template via minijinja = HF apply_chat_template — 23/23 templates byte-identical; token ids equal on Qwen2.5, Gemma-3, Phi-3.5, Llama-3.2 (scripts/chat_template_conformance.py, chat_ids_conformance.sh)"),
    "S08": ("verified", "dda7557: each family's own tool-call syntax (Hermes, Qwen3.5 XML, Gemma-4, LFM2, Mistral, DeepSeek, Llama python_tag); live: 5 families return the same call"),
    "S11": ("verified", "1c3fe7c: reasoning_content / Ollama thinking from the template's markers; streamed"),
    "E22": ("verified", "59906e7: CUDA graphs now used — the native decode step is one captured graph replayed for every token of every sequence (all per-token inputs in one pinned step block; FERRIC_CUDA_NO_GRAPH=1 = the eager twin, bit-identical over 400 steps in a SEPARATE graph); +1-2% on the 4050 (host logits +6-11%); plus the WGSL fusion codegen (410 -> 290 dispatches/token)"),
    "S32": ("verified", "bundled chat page at /ui (and / for browsers): streaming, stop, reasoning, model picker, per-answer tok/s + joules; scripts/ui_check.py drives it in headless Chrome and checks the page's answer and token count against the API"),
    "S36": ("verified", "OpenAI Batch API (files + batches + cancel), each line through the live request path; scripts/batch_api_check.py drives it with the official openai SDK: answers == live requests, error file, validation codes, cancel, usage + joules sums"),
    "F02": ("verified", "feat/formats: IQ1_S, IQ1_M, IQ2_XS, IQ2_S, IQ3_S decode = libggml-base 0.25.3 to_float bit for bit (7 whole files + 5 published + every-grid-entry blocks), raw-block GPU kernel; model 0.80-1.83x the authors' f32-vs-f64 floor (scripts/quant_formats_conformance.sh)"),
    "F04": ("verified", "feat/formats: block-scaled FP8 (Qwen3-0.6B-FP8) = transformers 5.17 Fp8Dequantize bit for bit, packed kernel; model 1.25x floor; compressed-tensors FP8 channel/block 0.93x. Activations stay f32 (no FP8 compute)"),
    "F05": ("verified", "feat/formats: GPTQ int2/3/4/8, act-order, v1/v2 = GPTQModel 7.5.0 bit for bit on 8 checkpoints, packed kernel; model Int4 1.03x / act-order 1.73x / 3-bit 0.97x of floor; g_idx-ignored control 99,594x"),
    "F06": ("verified", "feat/formats: AWQ GEMM = AutoAWQ dequantize_gemm (as vendored by GPTQModel) bit for bit; model 1.12x floor; nibble-order control 496,803x (GPTQModel's OFFLINE AWQ converter found wrong on 240M of 358M values)"),
    "F08": ("verified", "feat/formats: NVFP4 GGUF (type 40) = libggml-base bit for bit (1.36x floor) and ModelOpt NVFP4 safetensors = nvidia-modelopt 0.47 (0.98x); packed kernels"),
    "F15": ("verified", "feat/formats: compressed-tensors int4 sym/asym, int8 channel, FP8 channel/block, NVFP4 = compressed-tensors 0.19 (0.11 for actorder=group) bit for bit on 7 checkpoints; model 0.93-1.44x floor; 3-bit refused (layout ambiguous by version)"),
    "O08": ("partial", "libferric now wraps ferric-serve's engine (every registered arch, templates, samplers, constraints, energy; was: every GGUF loaded as dense Qwen3) + a Python ctypes binding; scripts/ffi_check.py: binding == HTTP on Qwen2.5 / Llama-3.2 / Qwen3. Go/Swift/Zig/C on the same symbols; no JS/Kotlin package"),
    "O09": ("verified", "OTLP/HTTP traces, json + protobuf, one span per request with GenAI semconv attributes and the request's joules; traceparent continued; scripts/otlp_conformance.sh decodes every export with opentelemetry-proto and checks spans against responses"),
    "O01": ("verified", "944f580: joules per request on every route, ∫(P−P_idle)/n(t) on the accelerator rails; unit-tested attribution; live batching −40% J/token"),
    "S05": ("verified", "d3adbe1: Anthropic /v1/messages + count_tokens; the official anthropic SDK works unmodified (tools, round trip, stream, stop_sequence)"),
    "S04": ("verified", "d3adbe1: OpenAI Responses API incl. previous_response_id and streaming; the official openai SDK works unmodified"),
    "S19": ("verified", "64d495c: /v1/audio/transcriptions over the NeMo-verified Parakeet (WAV; json/text/verbose_json); OpenAI SDK checked"),
    "S26": ("verified", "64d495c: --api-key (Bearer or x-api-key); 401 otherwise"),
    "S22": ("implemented", "15e605c: /tokenize, /detokenize in llama-server and vLLM shapes"),
    "S27": ("implemented", "15e605c: Prometheus /metrics incl. ferric_energy_joules_total"),
    "S10": ("verified", "2323cf7: GBNF (a port of llama.cpp's grammar engine: 70/70 + 69/69 of its own integration strings, 9/9 build checks), regex (= Python re.fullmatch on 982 strings), choice; every peer spelling (grammar, guided_*, structured_outputs, response_format) on chat and completions; trie mask"),
    "S16": ("verified", "07cf840: DRY, XTC, top-nσ, typical-p, Mirostat v2, logit_bias (OpenAI + llama-server forms) — each equal to the code that defines it (text-generation-webui's classes, transformers' TypicalLogitsWarper) on 332 recorded cases; live DRY breaks a greedy loop"),
    "E10": ("implemented", "5154e47: prompt-lookup (n-gram) drafts on any dense model, FERRIC_LOOKUP=k; answers identical 6/6, 7.3 tokens/forward on copying — ⚠ not yet faster: a few-row verify forward runs the prefill matmul path (~34 ms vs ~10 ms decode)"),
    "S30": ("verified", "0f0605d: prompt caching across requests on the dense runtime (radix-indexed PrefixCache, 16-token chunks, keyed on the LoRA selection and LongRoPE table); follow-up turns 5.2 s → 0.3 s and 73 J → 2.7 J, answers identical (scripts/prefix_cache_check.py)"),
    "E03": ("verified", "0f0605d: the radix PrefixCache is the server's (dense runtime; hybrid/MoE runtimes keep the one-slot MTP path); entries keyed on adapter + LongRoPE table (8a42c8b, 210382c)"),
    "B03": ("verified", "5a1f7a6: + quantized prefill on the M5 matrix units (FERRIC_QGEMM, opt-in: Q8_0/Q5_0/Q4_K/Q6_K tiles → fp16 → mpp matmul2d), prefill 10-20x (0.5B Q8_0 ~500 → 5,000-10,200 tok/s at 512), KL to the authors equal to the portable path's; 9.7-17x fewer GPU mJ/prompt token where attributable"),
    "E06": ("verified", "64e4b06: flash prefill now also serves a block CONTINUING a cache (prefix-cache suffixes, prefill chunks) — was t==s only; float64 reference at offsets to 2100, mutation-checked"),
    "O02": ("implemented", "63dd4aa: `ferric bench <model>` — prompt-processing and generation tok/s + J/token over any served GGUF, ranges over repetitions; no pp/tg sweep grid yet"),
    "E04": ("verified", "64e4b06: chunked prefill interleaved with decode (512 tokens/step while others stream); live 6,587-token prompt: worst stream stall 88.5 s → 2.6 s, answer identical; socket test mutation-checked"),
    "S24": ("verified", "e0bf215: any GGUF in the model directory loads when a request names it (stem, owner/repo:tag, path); per-model batches, one shared meter; Ollama keep_alive/ttl, LRU eviction under --max-models and a memory budget, command-line model pinned; 10 socket tests, 8 mutation-checked; live: 4 models by 3 name forms, concurrent two-model answers equal solo"),
    "S29": ("verified", "15e605c: a disconnected client frees its batch slot (socket test; mutation-checked: 14.8 s without); 5667ff5: serial streams too — hang up at 1 s → 2-60 tokens instead of 800 on four routes, counted"),
    "S18": ("verified", "5667ff5 + 53f3fe7: images on every dialect for Qwen2.5-VL/Qwen3-VL checkpoints; MiMo-Embodied over HTTP = the authors' 83 prompt ids and 64/64 greedy tokens (PNG); JPEG decoded to PIL's exact pixels (libjpeg-turbo 3.1.4.1 path), incl. the q92 file whose answer diverged — equal input, so the verified answer"),
    "S33": ("verified", "63dd4aa: `ferric` run/chat/pull/list/show/ps/stop/rm/bench/serve; 30 tests vs an Ollama-API mock, 8/8 mutations; live with the multi-model server"),
    "S25": ("verified", "63dd4aa: `ferric pull owner/repo[:QUANT]` — split GGUFs, resume (Range asserted), progress; real HF pull sha256 = the repo's LFS hash"),
    "E25": ("verified", "8a42c8b: forward_batch applies EACH ROW's LoRA selection; rows a/b/none match PEFT's adapter_names mixed batch at 0.77x its float32 floor (scripts/lora_conformance.sh; crossed-rows control 26k x). Dense runtime only; not yet exposed by ferric-serve"),
    "S37": ("verified", "8a42c8b engine + bddbae0: ferric-serve --lora name=path; select by `model: \"<adapter>\"` (vLLM) or `lora: [{id|name, scale}]` (llama-server); over HTTP every spelling = HF PEFT's own greedy continuation, 9 concurrent mixed base/syn_a/syn_b rows each = PEFT (scripts/serve_lora_conformance.sh; 2 wiring mutations caught)"),
    "T09": ("verified", "8a42c8b: examples/finetune_lora_peft trains LoRA pairs and writes a PEFT adapter (+ GGUF); PEFT loading only the files reproduces Ferric's in-memory logits at 0.93x its float32 floor (scripts/lora_roundtrip.sh)"),
    "T01": ("partial", "8a42c8b: genuine LoRA A/B on q_proj/v_proj with PEFT's scaling, exported as a PEFT adapter; still example-level (qwen2/qwen3 blocks reconstructed by hand)"),
    "B01": ("partial", "59906e7 (merged feat/cuda2 d910414): native NVIDIA decode as ONE CUDA graph replayed per token + host logits, split-K (flash-decoding) attention, int8 tensor-core prefill GEMM v3 (activations as three int8 digits of a 22-bit fixed point: exact in-tile sums), tiled prefill attention, rope angles from the host table. RTX 4050 tok/s decode at 512 ctx: Qwen2.5-0.5B Q4_K_M 239 (llama.cpp 281, 85%), Llama-3.2-1B Q4_K_M 134 (80%), Q8_0 223 (89%), Qwen3-0.6B Q5_K_M 162 (80%); prefill 33-45% of llama.cpp (keeps three activation digits where llama.cpp quantises to one). Gates: cuda_conformance + cuda_rope_conformance on 4 files (rope 2e-5..2.5e-4 at 30000), 21 kernel + 3 Rust mutations caught, PTX byte-identical to nvcc 12.8. Still opt-in (FERRIC_CUDA); no batched decode natively, MoE/hybrid, YaRN/LongRoPE, f16 KV stay on Vulkan"),
}
for k, (st, why) in UPDATES.items():
    fer.setdefault(k, {"id": k})
    fer[k] = {**fer[k], "status": st, "update": why}

def prevalence(p):
    if not p:
        return None
    have = set(p.get("have", [])) | set(p.get("inherited", []))
    part = set(p.get("partial", [])) - have
    total = have | part | set(p.get("lack", [])) | set(p.get("unknown", [])) | set(p.get("unclear", [])) \
            | set(p.get("unverified", [])) | set(p.get("not_applicable", []))
    n = len(total - set(p.get("not_applicable", [])))
    return (len(have) + 0.5 * len(part), n, sorted(have))

ORDER = {"absent": 0, "partial": 1, "implemented": 2, "verified": 3}
rows = []
for fid, f in feats.items():
    pv = prevalence(peer.get(fid))
    fs = fer.get(fid, {})
    rows.append({"id": fid, "group": f["group"], "feature": f["feature"], "prev": pv,
                 "status": fs.get("status", "unknown"), "update": fs.get("update", ""),
                 "note": (fs.get("notes") or "").replace("\n", " ").replace("|", "/")})

def frac(r):
    return r["prev"][0] / r["prev"][1] if r["prev"] and r["prev"][1] else 0.0

out = []
w = out.append
w("# Ferric feature parity — 27 September 2026\n")
w("Generated by `docs/parity/build.py` from the audit data in this folder. A peer **has** a feature only where "
  "its docs or source, read during the audit, say so (evidence URLs in the JSON; the 18 engine peers were "
  "cloned at pinned commits). Ferric's status is `verified` (a test or gate that fails without it), "
  "`implemented` (the code path exists end to end, nothing checks it), `partial`, or `absent`, each with "
  "file:line evidence in `ferric_*.json` — and updated for later commits, which the Update column cites.\n")
counts = {}
for r in rows:
    counts[r["status"]] = counts.get(r["status"], 0) + 1
w("**Ferric, " + str(len(rows)) + " features:** " + ", ".join(f"{counts.get(s, 0)} {s}" for s in ["verified", "implemented", "partial", "absent"]) + ".\n")

w("## Gaps, ranked by how many peers have them\n")
w("Every feature Ferric lacks or has only in part, most widely shipped first. Prevalence is peers with it "
  "(partial = ½) over peers checked for it.\n")
w("| id | feature | peers with it | Ferric | what is missing / what changed |")
w("|---|---|---|---|---|")
gaps = [r for r in rows if r["status"] in ("absent", "partial")]
gaps.sort(key=lambda r: (-frac(r), ORDER.get(r["status"], 9), r["id"]))
for r in gaps:
    pv = f"{r['prev'][0]:g}/{r['prev'][1]}" if r["prev"] else "—"
    w(f"| {r['id']} | {r['feature']} | {pv} | {r['status']} | {(r['update'] or r['note'])[:220]} |")

w("\n## Where Ferric stands that peers do not\n")
w("| id | feature | peers with it | Ferric |")
w("|---|---|---|---|")
for r in sorted([r for r in rows if r["status"] in ("verified", "implemented") and frac(r) <= 0.25 and r["prev"]], key=frac):
    w(f"| {r['id']} | {r['feature']} | {r['prev'][0]:g}/{r['prev'][1]} | {r['status']} |")
w("\nNot on the feature list, and measured by the audit: **none of the 14 serving peers reports energy per "
  "request or per token** (O01; the closest are Lemonade's utilisation endpoint and exo's power sampler on "
  "bench runs), and **only vLLM and SGLang document a deterministic mode — on the same hardware and "
  "version**; bit-identity ACROSS vendors is claimed by none of the 18 engine peers.\n")

w("## The whole matrix\n")
for g in ["serve", "engine", "format", "backend", "model", "tuning", "ops"]:
    w(f"### {g}\n")
    w("| id | feature | peers with it | Ferric | note |")
    w("|---|---|---|---|---|")
    for r in [r for r in rows if r["group"] == g]:
        pv = f"{r['prev'][0]:g}/{r['prev'][1]}" if r["prev"] else "—"
        w(f"| {r['id']} | {r['feature']} | {pv} | {r['status']} | {(r['update'] or r['note'])[:160]} |")
    w("")

apps = next((a for a in load("peer_serve.json").get("added", []) if a.get("id") == "X-APPS"), None)
if apps:
    w("## What the front-ends require of a backend\n")
    w("From each app's own source (X-APPS in peer_serve.json).\n")
    for a in apps.get("apps", apps.get("per_app", [])) if isinstance(apps.get("apps", apps.get("per_app")), list) else []:
        name = a.get("app") or a.get("name")
        need = a.get("minimum_drop_in") or a.get("notes") or ""
        if name:
            w(f"- **{name}** — {str(need)[:300]}")
open(f"{D}/MATRIX.md", "w").write("\n".join(out) + "\n")
print(f"{len(rows)} features; gaps {len(gaps)}; wrote {D}/MATRIX.md")
