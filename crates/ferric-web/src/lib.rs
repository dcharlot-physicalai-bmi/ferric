//! Ferric web — proves the SAME pure-Rust kernels (ferric-core) run in the browser on WebGPU.
//! The matmul WGSL, the Context, the readback — all identical to native; only the target changes.
use ferric_core::{demo, matmul_cpu, max_abs_diff, Context};
use ferric_gguf::{parse, GgufSource, Meta};
use ferric_llama::qwen3::{Cache, Qwen3};
use ferric_tensor::Tensor;
use ferric_tokenizer::Bpe;
use std::collections::HashMap;
use std::sync::Arc;
use wasm_bindgen::prelude::*;

/// Scheduler worker entrypoint: the native fabric sends an op frame (op · dims · A · B, same format
/// as Device::Remote) over a WebSocket; this executes it on the tab's WebGPU and returns the result
/// bytes. That makes this browser tab a device in the heterogeneous fabric (Device::BrowserWorker).
#[wasm_bindgen]
pub async fn ferric_worker_exec(input: Vec<u8>) -> Vec<u8> {
    console_error_panic_hook::set_once();
    let ru32 = |o: usize| u32::from_le_bytes([input[o], input[o + 1], input[o + 2], input[o + 3]]) as usize;
    let op = input[0];
    let dims = [ru32(1), ru32(5), ru32(9), ru32(13)];
    let mut off = 17;
    let la = ru32(off); off += 4;
    let a: Vec<f32> = input[off..off + la * 4].chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
    off += la * 4;
    let lb = ru32(off); off += 4;
    let b: Vec<f32> = input[off..off + lb * 4].chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();

    let ctx = Arc::new(Context::new().await.unwrap());
    let out = if op == 0 {
        let (batch, m, k, n) = (dims[0], dims[1], dims[2], dims[3]);
        Tensor::from_vec(&ctx, &a, &[batch, m, k]).matmul(&Tensor::from_vec(&ctx, &b, &[k, n])).to_vec().await
    } else {
        let (rows, inn, outn) = (dims[0], dims[1], dims[2]);
        Tensor::from_vec(&ctx, &a, &[rows, inn]).matmul(&Tensor::from_vec(&ctx, &b, &[inn, outn])).relu().to_vec().await
    };
    out.iter().flat_map(|f| f.to_le_bytes()).collect()
}

// Same deterministic matrices as the native example, so the browser result must match.
fn gen(m: u32, k: u32, n: u32) -> (Vec<f32>, Vec<f32>) {
    let a: Vec<f32> = (0..(m * k) as usize).map(|i| ((i * 7 % 13) as f32 - 6.0) * 0.1).collect();
    let b: Vec<f32> = (0..(k * n) as usize).map(|i| ((i * 5 % 11) as f32 - 5.0) * 0.1).collect();
    (a, b)
}

/// Runs the Ferric matmul on the browser's WebGPU and validates against the CPU reference.
/// Returns "backend|maxdiff|first6".
#[wasm_bindgen]
pub async fn ferric_matmul_demo(m: u32, k: u32, n: u32) -> std::result::Result<String, JsValue> {
    console_error_panic_hook::set_once();
    let ctx = Context::new().await.map_err(|e| JsValue::from_str(&e))?;
    let (a, b) = gen(m, k, n);
    let gpu = ctx.matmul(&a, &b, m, k, n).await.map_err(|e| JsValue::from_str(&e))?;
    let cpu = matmul_cpu(&a, &b, m as usize, k as usize, n as usize);
    let diff = max_abs_diff(&gpu, &cpu);
    Ok(format!("{:?}|{:.3e}|{:?}", ctx.backend, diff, &gpu[..6.min(gpu.len())]))
}

/// Runs the full Ferric transformer LM in the browser on WebGPU: greedy-generates `steps` tokens from
/// a comma-separated prompt of token ids, and validates the prefill logits against the in-wasm CPU
/// reference. Returns a JSON string {backend, prompt, generated, layers, logit_diff, ms}.
#[wasm_bindgen]
pub async fn ferric_lm_demo(prompt: String, steps: usize) -> std::result::Result<String, JsValue> {
    console_error_panic_hook::set_once();
    let ids: Vec<u32> = prompt.split(',').filter_map(|s| s.trim().parse().ok()).map(|v: u32| v % demo::VOCAB as u32).collect();
    if ids.is_empty() {
        return Err(JsValue::from_str("no valid token ids in prompt"));
    }
    let ctx = Context::new().await.map_err(|e| JsValue::from_str(&e))?;
    let t0 = js_sys::Date::now();
    let generated = demo::generate(&ctx, &ids, steps).await.map_err(|e| JsValue::from_str(&e))?;
    let ms = js_sys::Date::now() - t0;
    // correctness, in the browser: GPU prefill logits vs the CPU reference (same math, wasm CPU)
    let gpu = demo::logits(&ctx, &ids).await.map_err(|e| JsValue::from_str(&e))?;
    let diff = max_abs_diff(&gpu, &demo::logits_cpu(&ids));
    let js = |v: &[u32]| v.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(",");
    Ok(format!(
        "{{\"backend\":\"{:?}\",\"adapter\":\"{}\",\"prompt\":[{}],\"generated\":[{}],\"layers\":{},\"vocab\":{},\"logit_diff\":{:.3e},\"ms\":{:.1}}}",
        ctx.backend, ctx.adapter_name, js(&ids), js(&generated), demo::N_LAYERS, demo::VOCAB, diff, ms
    ))
}

/// **The moat, demonstrated.** Runs a real PrismML Ternary Bonsai (Qwen3 dense, Q2_0) *entirely in
/// the browser tab* on WebGPU — the exact same `Qwen3` code that runs natively, compiled to wasm32.
/// `model` is the fetched .gguf bytes; greedily continues `prompt` for `steps` tokens.
/// Returns JSON {backend, adapter, layers, vocab, prompt, text, load_ms, prompt_ms, decode_ms}.
#[wasm_bindgen]
pub async fn bonsai_generate(model: Vec<u8>, prompt: String, steps: usize) -> std::result::Result<String, JsValue> {
    console_error_panic_hook::set_once();
    let err = |e: String| JsValue::from_str(&e);

    let t_load = js_sys::Date::now();
    let g = parse(model).map_err(err)?;
    let (bpe, toks) = build_bpe(&g)?;
    let u2b = gpt2_byte_decoder();
    let detok = |ids: &[u32]| -> String {
        let s: String = ids.iter().map(|&i| toks.get(i as usize).cloned().unwrap_or_default()).collect();
        String::from_utf8_lossy(&s.chars().filter_map(|c| u2b.get(&c).copied()).collect::<Vec<u8>>()).into_owned()
    };

    let ctx = Arc::new(Context::new().await.map_err(err)?);
    let m = Qwen3::load(&ctx, &g).map_err(err)?;
    let c = &m.cfg;
    let load_ms = js_sys::Date::now() - t_load;

    let ids = encode_with_bos(&g, &bpe, &prompt);
    if ids.is_empty() { return Err(err("prompt encoded to zero tokens".into())); }

    let mut cache = Cache::new(c);
    let mut seq = ids.clone();
    let argmax = |row: &[f32]| (0..c.n_vocab).max_by(|&a, &b| row[a].partial_cmp(&row[b]).unwrap()).unwrap() as u32;

    let t_prompt = js_sys::Date::now();
    let mut prompt_ms = 0.0;
    for step in 0..steps {
        let logits = if step == 0 { m.forward_cached(&ids, &mut cache) } else { m.forward_cached(&seq[seq.len() - 1..], &mut cache) };
        let v = logits.to_vec().await;
        let next = argmax(&v[v.len() - c.n_vocab..]);
        if step == 0 { prompt_ms = js_sys::Date::now() - t_prompt; }
        seq.push(next);
        if next == 151645 || next == 151643 { break; }
    }
    let decode_ms = (js_sys::Date::now() - t_prompt - prompt_ms) / (steps.max(1) as f64);
    let text = detok(&seq);
    let esc = |s: &str| s.replace('\\', "\\\\").replace('"', "\\\"").replace('\n', "\\n");
    Ok(format!(
        "{{\"backend\":\"{:?}\",\"adapter\":\"{}\",\"layers\":{},\"vocab\":{},\"prompt\":\"{}\",\"text\":\"{}\",\"load_ms\":{:.0},\"prompt_ms\":{:.0},\"decode_ms\":{:.1}}}",
        ctx.backend, ctx.adapter_name, c.n_layer, c.n_vocab, esc(&prompt), esc(&text), load_ms, prompt_ms, decode_ms
    ))
}

/// Streaming variant: same model, but invokes `on_token(text)` as each token is produced and
/// `on_token` may also carry progress. Returns final JSON stats. `on_token` is a JS function
/// `(kind: string, payload: string) => void` — kind ∈ {"status","token","done"}.
#[wasm_bindgen]
pub async fn bonsai_stream(model: Vec<u8>, prompt: String, steps: usize, on_token: js_sys::Function) -> std::result::Result<String, JsValue> {
    console_error_panic_hook::set_once();
    let err = |e: String| JsValue::from_str(&e);
    let emit = |kind: &str, payload: &str| { let _ = on_token.call2(&JsValue::NULL, &JsValue::from_str(kind), &JsValue::from_str(payload)); };

    emit("status", "parsing model…");
    let t_load = js_sys::Date::now();
    let g = parse(model).map_err(err)?;
    let (bpe, toks) = build_bpe(&g)?;
    let u2b = gpt2_byte_decoder();
    let detok = |ids: &[u32]| -> String {
        let s: String = ids.iter().map(|&i| toks.get(i as usize).cloned().unwrap_or_default()).collect();
        String::from_utf8_lossy(&s.chars().filter_map(|c| u2b.get(&c).copied()).collect::<Vec<u8>>()).into_owned()
    };

    emit("status", "uploading weights to GPU…");
    let ctx = Arc::new(Context::new().await.map_err(err)?);
    let m = Qwen3::load(&ctx, &g).map_err(err)?;
    let c = &m.cfg;
    let load_ms = js_sys::Date::now() - t_load;
    emit("status", &format!("ready · {} layers · {:?} · {:.0}ms", c.n_layer, ctx.backend, load_ms));

    let ids = encode_with_bos(&g, &bpe, &prompt);
    if ids.is_empty() { return Err(err("prompt encoded to zero tokens".into())); }

    let mut cache = Cache::new(c);
    let mut seq = ids.clone();
    let argmax = |row: &[f32]| (0..c.n_vocab).max_by(|&a, &b| row[a].partial_cmp(&row[b]).unwrap()).unwrap() as u32;
    let t_prompt = js_sys::Date::now();
    let mut prompt_ms = 0.0;
    // Detokenize the whole generated run each step and emit the new suffix — robust to BPE tokens
    // that split a multi-byte UTF-8 char across steps (the partial simply completes next step).
    let mut emitted = String::new();
    for step in 0..steps {
        let logits = if step == 0 { m.forward_cached(&ids, &mut cache) } else { m.forward_cached(&seq[seq.len() - 1..], &mut cache) };
        let v = logits.to_vec().await;
        let next = argmax(&v[v.len() - c.n_vocab..]);
        if step == 0 { prompt_ms = js_sys::Date::now() - t_prompt; }
        seq.push(next);
        let full = detok(&seq[ids.len()..]);
        if let Some(delta) = full.strip_prefix(&emitted) {
            if !delta.is_empty() { emit("token", delta); }
        }
        emitted = full;
        if next == 151645 || next == 151643 { break; }
    }
    let decode_ms = (js_sys::Date::now() - t_prompt - prompt_ms) / (steps.max(1) as f64);
    let stats = format!(
        "{{\"backend\":\"{:?}\",\"layers\":{},\"vocab\":{},\"load_ms\":{:.0},\"prompt_ms\":{:.0},\"decode_ms\":{:.1}}}",
        ctx.backend, c.n_layer, c.n_vocab, load_ms, prompt_ms, decode_ms
    );
    emit("done", &stats);
    Ok(stats)
}

/// **Guided decoding in the browser** — the moat as a live demo. Runs the same Qwen3-on-WebGPU
/// generation but constrains the sampler with `ferric_agent::guide` (the exact code the native server
/// uses), so the output is guaranteed valid JSON — schema-conformant if `schema` is a JSON-Schema
/// string, else a well-formed JSON object. Because it masks the same logits, an on-device browser tab
/// and the datacenter produce the *same deterministic* constrained output. Streams deltas via
/// `on_token(kind, payload)` (kind ∈ {"status","token","done"}); returns the final JSON string.
#[wasm_bindgen]
pub async fn bonsai_generate_json(model: Vec<u8>, prompt: String, steps: usize, schema: String, on_token: js_sys::Function) -> std::result::Result<String, JsValue> {
    console_error_panic_hook::set_once();
    let err = |e: String| JsValue::from_str(&e);
    let emit = |kind: &str, payload: &str| { let _ = on_token.call2(&JsValue::NULL, &JsValue::from_str(kind), &JsValue::from_str(payload)); };
    emit("status", "parsing model…");
    let g = parse(model).map_err(err)?;
    let (bpe, toks) = build_bpe(&g)?;
    let u2b = gpt2_byte_decoder();
    let detok = |ids: &[u32]| -> String {
        let s: String = ids.iter().map(|&i| toks.get(i as usize).cloned().unwrap_or_default()).collect();
        String::from_utf8_lossy(&s.chars().filter_map(|c| u2b.get(&c).copied()).collect::<Vec<u8>>()).into_owned()
    };
    // Each token's raw bytes (None = special token → disallowed under the constraint, except EOS).
    let token_bytes: Vec<Option<Vec<u8>>> = toks.iter().map(|t| {
        let mut b = Vec::with_capacity(t.len());
        for ch in t.chars() { match u2b.get(&ch) { Some(&x) => b.push(x), None => return None } }
        Some(b)
    }).collect();

    emit("status", "uploading weights to GPU…");
    let ctx = Arc::new(Context::new().await.map_err(err)?);
    let m = Qwen3::load(&ctx, &g).map_err(err)?;
    let c = &m.cfg;
    emit("status", &format!("ready · {:?}", ctx.backend));

    let sch_prog = ferric_agent::guide::compile_str(&schema);
    let mut guide = match &sch_prog {
        Some(prog) => ferric_agent::guide::Guide::Schema(ferric_agent::guide::Schema::new(prog)),
        None => ferric_agent::guide::Guide::Json(ferric_agent::guide::Json::object()),
    };
    let eos = |t: u32| t == 151645 || t == 151643;
    let ids = encode_with_bos(&g, &bpe, &prompt);
    if ids.is_empty() { return Err(err("prompt encoded to zero tokens".into())); }
    let mut cache = Cache::new(c);
    let mut seq = ids.clone();
    let mut emitted = String::new();
    for step in 0..steps {
        let logits = if step == 0 { m.forward_cached(&ids, &mut cache) } else { m.forward_cached(&seq[seq.len() - 1..], &mut cache) };
        let v = logits.to_vec().await;
        let row = &v[v.len() - c.n_vocab..];
        // Masked argmax: highest-logit token whose bytes keep the JSON valid (EOS only once complete).
        let can_stop = guide.can_stop();
        let (mut best, mut best_l) = (None, f32::NEG_INFINITY);
        for i in 0..c.n_vocab {
            let ok = if eos(i as u32) { can_stop }
                else { match &token_bytes[i] { Some(b) if !b.is_empty() => { let mut a = guide; b.iter().all(|&ch| a.step(ch)) } _ => false } };
            if ok && row[i] > best_l { best_l = row[i]; best = Some(i as u32); }
        }
        let next = match best { Some(t) => t, None => break };
        if eos(next) { break; }
        if let Some(b) = &token_bytes[next as usize] { for &ch in b { guide.step(ch); } }
        seq.push(next);
        let full = detok(&seq[ids.len()..]);
        if let Some(delta) = full.strip_prefix(&emitted) { if !delta.is_empty() { emit("token", delta); } }
        emitted = full;
    }
    emit("done", &emitted);
    Ok(emitted)
}

/// Cross-fabric proof: run the SAME Qwen3 forward the native binary runs, on the browser's WebGPU,
/// and return a deterministic fingerprint of the last-position logits (top-8 + fixed probes + sum).
/// Compared against `run_qwen3 --dump` (native Metal), this proves the moat — bit-comparable numerics
/// for a real 1.7B ternary LLM — not just for a lone matmul.
#[wasm_bindgen]
pub async fn bonsai_logits(model: Vec<u8>, prompt: String) -> std::result::Result<String, JsValue> {
    console_error_panic_hook::set_once();
    let err = |e: String| JsValue::from_str(&e);
    let g = parse(model).map_err(err)?;
    let (bpe, _toks) = build_bpe(&g)?;
    let ctx = Arc::new(Context::new().await.map_err(err)?);
    let m = Qwen3::load(&ctx, &g).map_err(err)?;
    let nv = m.cfg.n_vocab;
    let ids = encode_with_bos(&g, &bpe, &prompt);
    let v = m.forward_cached(&ids, &mut Cache::new(&m.cfg)).to_vec().await;
    let row = &v[v.len() - nv..];
    let mut idx: Vec<usize> = (0..nv).collect();
    idx.sort_by(|&a, &b| row[b].partial_cmp(&row[a]).unwrap());
    let top: Vec<String> = idx.iter().take(8).map(|&i| format!("[{i},{:.5}]", row[i])).collect();
    let sum: f64 = row.iter().map(|&x| x as f64).sum();
    Ok(format!(
        "{{\"ids\":{:?},\"argmax\":{},\"top\":[{}],\"probe\":[{:.5},{:.5},{:.5},{:.5}],\"sum\":{:.3}}}",
        ids, idx[0], top.join(","), row[0], row[100], row[1000], row[10000], sum
    ))
}

/// Build the exact BPE (+ the raw token strings for detok) from a GGUF's embedded tokenizer.
fn build_bpe(g: &impl GgufSource) -> std::result::Result<(Bpe, Vec<String>), JsValue> {
    let err = |e: &str| JsValue::from_str(e);
    let toks: Vec<String> = match g.metadata().get("tokenizer.ggml.tokens") {
        Some(Meta::Arr(a)) => a.iter().map(|m| if let Meta::Str(s) = m { s.clone() } else { String::new() }).collect(),
        _ => return Err(err("no tokens")),
    };
    let vocab: HashMap<String, u32> = toks.iter().enumerate().map(|(i, t)| (t.clone(), i as u32)).collect();
    let merges: Vec<(String, String)> = match g.metadata().get("tokenizer.ggml.merges") {
        Some(Meta::Arr(a)) => a.iter().filter_map(|m| if let Meta::Str(s) = m { s.split_once(' ').map(|(x, y)| (x.into(), y.into())) } else { None }).collect(),
        _ => return Err(err("no merges")),
    };
    Ok((Bpe::new(vocab, &merges), toks))
}

/// Encode `prompt`, prepending BOS when the GGUF asks for it (Llama-3 does; Qwen doesn't) — Llama
/// models degrade badly without BOS. Default: add BOS when a bos id exists and add_bos isn't false.
fn encode_with_bos(g: &impl GgufSource, bpe: &Bpe, prompt: &str) -> Vec<u32> {
    let bos = match g.metadata().get("tokenizer.ggml.bos_token_id") { Some(Meta::U(v)) => Some(*v as u32), _ => None };
    let add = match g.metadata().get("tokenizer.ggml.add_bos_token") { Some(Meta::Bool(b)) => *b, _ => bos.is_some() };
    let mut ids = bpe.encode(prompt);
    if add { if let Some(b) = bos { ids.insert(0, b); } }
    ids
}

/// GPT-2 byte↔printable-unicode map, inverted — vocab entry chars back to raw bytes.
fn gpt2_byte_decoder() -> HashMap<char, u8> {
    let mut m = HashMap::new();
    let mut n = 0u32;
    for b in 0u32..256 {
        let printable = (0x21..=0x7e).contains(&b) || (0xa1..=0xac).contains(&b) || (0xae..=0xff).contains(&b);
        let c = if printable { b } else { let c = 256 + n; n += 1; c };
        m.insert(char::from_u32(c).unwrap(), b as u8);
    }
    m
}
