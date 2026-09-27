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
    "S13": ("absent", "3f6089f: n > 1 is now a 400 naming the field, not a silent single choice"),
    "S14": ("verified", "3f6089f: stop strings cut before the match, streamed tails held back; batched == serial over sockets"),
    "S15": ("verified", "3f6089f: temperature, top_p, top_k, min_p, presence/frequency/repeat penalty, seed; default path pinned token-for-token to the old sampler"),
    "S23": ("implemented", "da736c9: lists every loaded model (chat + embedder)"),
    "S28": ("implemented", "3f6089f: query strings no longer 404 (/health?x=1)"),
    "S35": ("implemented", "3f6089f: a prompt past <arch>.context_length is a 400 naming both numbers; no max_tokens = until EOS or the context"),
    "E19": ("partial", "87c1f6b: linear applied; any declared type a runtime does not apply is REFUSED (was silently unscaled); longrope/NTK absent"),
    "M06": ("verified", "361cddb: nomic-bert (nomic-embed-text) verified against its authors at every stage; registry row added"),
    "S07": ("verified", "d185918: /v1/completions streams; tool calls stream as an SSE delta"),
    "S17": ("verified", "b0b4448: the GGUF's own Jinja template via minijinja = HF apply_chat_template — 23/23 templates byte-identical; token ids equal on Qwen2.5, Gemma-3, Phi-3.5, Llama-3.2 (scripts/chat_template_conformance.py, chat_ids_conformance.sh)"),
    "S08": ("verified", "dda7557: each family's own tool-call syntax (Hermes, Qwen3.5 XML, Gemma-4, LFM2, Mistral, DeepSeek, Llama python_tag); live: 5 families return the same call"),
    "S11": ("verified", "1c3fe7c: reasoning_content / Ollama thinking from the template's markers; streamed"),
    "O01": ("verified", "944f580: joules per request on every route, ∫(P−P_idle)/n(t) on the accelerator rails; unit-tested attribution; live batching −40% J/token"),
    "S05": ("verified", "d3adbe1: Anthropic /v1/messages + count_tokens; the official anthropic SDK works unmodified (tools, round trip, stream, stop_sequence)"),
    "S04": ("verified", "d3adbe1: OpenAI Responses API incl. previous_response_id and streaming; the official openai SDK works unmodified"),
    "S19": ("verified", "64d495c: /v1/audio/transcriptions over the NeMo-verified Parakeet (WAV; json/text/verbose_json); OpenAI SDK checked"),
    "S26": ("verified", "64d495c: --api-key (Bearer or x-api-key); 401 otherwise"),
    "S22": ("implemented", "15e605c: /tokenize, /detokenize in llama-server and vLLM shapes"),
    "S27": ("implemented", "15e605c: Prometheus /metrics incl. ferric_energy_joules_total"),
    "S29": ("verified", "15e605c: a disconnected client frees its batch slot (socket test; mutation-checked: 14.8 s without)"),
    "E25": ("verified", "8a42c8b: forward_batch applies EACH ROW's LoRA selection; rows a/b/none match PEFT's adapter_names mixed batch at 0.77x its float32 floor (scripts/lora_conformance.sh; crossed-rows control 26k x). Dense runtime only; not yet exposed by ferric-serve"),
    "S37": ("partial", "8a42c8b: engine API only — Qwen3::upload_lora(PEFT dir or llama.cpp GGUF) + Cache::set_adapters per request, merged or unmerged, 0.4-1.5x PEFT's floor on 4 adapters incl. a published one; ferric-serve does not expose it yet"),
    "T09": ("verified", "8a42c8b: examples/finetune_lora_peft trains LoRA pairs and writes a PEFT adapter (+ GGUF); PEFT loading only the files reproduces Ferric's in-memory logits at 0.93x its float32 floor (scripts/lora_roundtrip.sh)"),
    "T01": ("partial", "8a42c8b: genuine LoRA A/B on q_proj/v_proj with PEFT's scaling, exported as a PEFT adapter; still example-level (qwen2/qwen3 blocks reconstructed by hand)"),
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
