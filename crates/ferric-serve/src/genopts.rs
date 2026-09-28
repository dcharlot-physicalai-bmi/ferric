//! **What one request asks of a generation, and the pieces every generation loop shares.**
//!
//! The server has three generation loops — the serial `Engine::generate`, the MTP speculative
//! `generate_spec` and the batched `step` — and before this module each read `max_tokens` and
//! `temperature` and nothing else. `stop`, `seed`, `top_p`, `top_k`, `n` and `logprobs` were accepted and
//! silently ignored, `finish_reason` was `"stop"` even when `max_tokens` cut the answer off, and an
//! OpenAI content-part array (`[{"type":"text","text":...}]`, what most chat front-ends send) became an
//! empty string — the model answered a prompt the user never wrote. Everything here is shared, so the
//! three loops cannot disagree about what a request meant.
//!
//! A parameter this server cannot honour is a 400 naming it, never a silent default: a client can act on
//! an error; it cannot act on an answer to a different question.
use serde_json::{json, Value};

/// Sampling controls. Defaults keep Ferric's historical behaviour: temperature 0 (greedy, deterministic)
/// and, when sampling, nucleus 0.95 over a fixed-seed RNG — so a request that sets none of these gets
/// byte-identical output to before.
#[derive(Clone, Debug)]
pub(crate) struct Sampling {
    pub temperature: f32,
    pub top_p: f32,
    /// 0 = off.
    pub top_k: usize,
    /// 0 = off. Keeps tokens whose probability is at least `min_p` x the most likely one's.
    pub min_p: f32,
    /// OpenAI semantics: subtracted once per distinct generated token / per occurrence.
    pub presence_penalty: f32,
    pub frequency_penalty: f32,
    /// llama.cpp / Ollama semantics: logits of recently seen tokens divided (if > 0) or multiplied
    /// (if < 0) by this. 1.0 = off. Window `repeat_last_n` over prompt tail + generated.
    pub repeat_penalty: f32,
    pub repeat_last_n: usize,
    // ── Extended samplers (parity gap S16). Each runs only when set, so the default path is untouched. ──
    /// OpenAI `logit_bias`: added to a token's logit before anything else; `-inf` bans it.
    pub logit_bias: Vec<(u32, f32)>,
    /// DRY (p-e-w): penalise continuing a sequence that already occurred. 0 = off.
    pub dry_multiplier: f32,
    pub dry_base: f32,
    pub dry_allowed_length: usize,
    /// Token ids that break a DRY match (the tokenizer's last id for "a" + each breaker string).
    pub dry_breakers: Vec<u32>,
    /// The breaker strings as sent (resolved to ids against the model's tokenizer in `Engine::gen_opts`).
    pub dry_breaker_strings: Vec<String>,
    /// Tokens of context DRY looks at; 0 = all.
    pub dry_range: usize,
    /// XTC (p-e-w): with probability `xtc_probability`, remove every token above `xtc_threshold` but the
    /// least likely of them — unless that would remove a newline or EOS (`xtc_specials`).
    pub xtc_threshold: f32,
    pub xtc_probability: f32,
    pub xtc_specials: Vec<u32>,
    /// Locally typical sampling (Meister et al.; transformers' TypicalLogitsWarper). 1.0 = off.
    pub typical_p: f32,
    /// Top-nσ: keep logits within n standard deviations of the maximum. 0 = off.
    pub top_n_sigma: f32,
    /// Mirostat v2 (target surprise `tau`, learning rate `eta`). 0 = off.
    pub mirostat: u8,
    pub mirostat_tau: f32,
    pub mirostat_eta: f32,
    /// Mirostat's running threshold for THIS sequence (NaN until the first step sets it to 2·tau).
    pub mirostat_mu: std::cell::Cell<f32>,
}

impl Default for Sampling {
    fn default() -> Self {
        Sampling { temperature: 0.0, top_p: 0.95, top_k: 0, min_p: 0.0, presence_penalty: 0.0,
                   frequency_penalty: 0.0, repeat_penalty: 1.0, repeat_last_n: 64,
                   logit_bias: Vec::new(), dry_multiplier: 0.0, dry_base: 1.75, dry_allowed_length: 2,
                   dry_breakers: Vec::new(), dry_breaker_strings: vec!["\n".into(), ":".into(), "\"".into(), "*".into()],
                   dry_range: 0, xtc_threshold: 0.1, xtc_probability: 0.0, xtc_specials: Vec::new(), typical_p: 1.0,
                   top_n_sigma: 0.0, mirostat: 0, mirostat_tau: 5.0, mirostat_eta: 0.1, mirostat_mu: std::cell::Cell::new(f32::NAN) }
    }
}

/// Everything a request asks of one generation.
#[derive(Clone, Debug)]
pub(crate) struct GenOpts {
    /// `None` = until a stop token or the context runs out (what OpenAI, llama-server and Ollama do
    /// when the client sends no limit). The caller resolves it against the model's context.
    pub max_tokens: Option<usize>,
    pub sampling: Sampling,
    /// Initial RNG state. Without a `seed` it is Ferric's fixed default, so sampled output is still
    /// reproducible request to request.
    pub rng: u64,
    pub stop: Vec<String>,
    pub logprobs: bool,
    pub top_logprobs: usize,
    /// Internal, never from a request: decode WITH special tokens, because this model's reasoning
    /// markers are specials the visible decode would drop (Gemma 4).
    pub with_specials: bool,
    /// Internal: the request's image, decoded and planned (`vision`). It rides with the request rather
    /// than on the engine, because two images of one size produce the same prompt ids.
    pub image: Option<std::sync::Arc<crate::vision::MmInput>>,
    /// `logit_bias` entries keyed by TEXT (llama-server): every token of the text gets the bias, resolved
    /// against the model's tokenizer in `Engine::gen_opts`.
    pub logit_bias_text: Vec<(String, f32)>,
    /// Internal: the LoRA adapters this request runs with (`Engine::gen_opts`); empty = the base model.
    pub lora: Lora,
}

/// A request's adapter selection: each adapter uploaded once, and its multiplier.
#[derive(Clone, Default)]
pub(crate) struct Lora(pub Vec<(std::sync::Arc<ferric_llama::lora::DeviceLora>, f32)>);
impl std::fmt::Debug for Lora {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_list().entries(self.0.iter().map(|(d, s)| format!("{}@{s}", d.id))).finish()
    }
}

pub(crate) const DEFAULT_RNG: u64 = 0x2545_F491_4F6C_DD1D;

impl Default for GenOpts {
    fn default() -> Self {
        GenOpts { max_tokens: None, sampling: Sampling::default(), rng: DEFAULT_RNG, stop: Vec::new(),
                  logprobs: false, top_logprobs: 0, with_specials: false, image: None, lora: Lora::default(), logit_bias_text: Vec::new() }
    }
}

fn f32_in(req: &Value, k: &str, lo: f32, hi: f32) -> Result<Option<f32>, String> {
    match &req[k] {
        Value::Null => Ok(None),
        v => match v.as_f64() {
            Some(x) if (lo as f64..=hi as f64).contains(&x) => Ok(Some(x as f32)),
            _ => Err(format!("`{k}` must be a number in [{lo}, {hi}], got {v}")),
        },
    }
}

/// splitmix64 — spreads a small user seed (0, 1, 42) over the whole state so xorshift starts well.
fn seed_state(seed: u64) -> u64 {
    let mut z = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    let z = z ^ (z >> 31);
    if z == 0 { DEFAULT_RNG } else { z }
}

impl GenOpts {
    /// Read a `/v1/chat/completions` (`chat`) or `/v1/completions` request. Accepts the OpenAI names,
    /// plus `top_k`, `min_p`, `repeat_penalty` and `repeat_last_n` as llama-server and Ollama clients
    /// send them.
    pub fn from_req(req: &Value, chat: bool) -> Result<GenOpts, String> {
        let mut o = GenOpts::default();
        // `max_completion_tokens` is OpenAI's current name; `max_tokens` the older one.
        for k in ["max_completion_tokens", "max_tokens"] {
            match &req[k] {
                Value::Null => {}
                v => match v.as_u64() {
                    Some(n) if n > 0 => { o.max_tokens = Some(n as usize); break; }
                    _ => return Err(format!("`{k}` must be a positive integer, got {v}")),
                },
            }
        }
        match &req["n"] {
            Value::Null => {}
            v if v.as_u64() == Some(1) => {}
            v => return Err(format!("`n` = {v}: this server returns one choice per request; send n requests")),
        }
        let s = &mut o.sampling;
        if let Some(t) = f32_in(req, "temperature", 0.0, 5.0)? { s.temperature = t; }
        if let Some(p) = f32_in(req, "top_p", 0.0, 1.0)? {
            if p <= 0.0 { return Err("`top_p` must be greater than 0".into()); }
            s.top_p = p;
        }
        match &req["top_k"] {
            Value::Null => {}
            v => match v.as_i64() {
                Some(k) if k >= 0 => s.top_k = k as usize,
                Some(-1) => s.top_k = 0, // vLLM spells "off" as -1
                _ => return Err(format!("`top_k` must be a non-negative integer, got {v}")),
            },
        }
        if let Some(m) = f32_in(req, "min_p", 0.0, 1.0)? { s.min_p = m; }
        if let Some(p) = f32_in(req, "presence_penalty", -2.0, 2.0)? { s.presence_penalty = p; }
        if let Some(p) = f32_in(req, "frequency_penalty", -2.0, 2.0)? { s.frequency_penalty = p; }
        if let Some(p) = f32_in(req, "repeat_penalty", 0.0, 10.0)? {
            if p <= 0.0 { return Err("`repeat_penalty` must be greater than 0 (1.0 = off)".into()); }
            s.repeat_penalty = p;
        }
        if let Some(n) = req["repeat_last_n"].as_u64() { s.repeat_last_n = n as usize; }
        // logit_bias: OpenAI's {"<id>": bias}, or llama-server's [[id, bias | false], …] (false = ban). A string
        // key is a piece of text, resolved against the tokenizer in `Engine::gen_opts`.
        match &req["logit_bias"] {
            Value::Null => {}
            Value::Object(m) => for (k, v) in m {
                let id: u32 = k.parse().map_err(|_| format!("logit_bias key {k:?} is not a token id"))?;
                s.logit_bias.push((id, bias_of(v)?));
            },
            Value::Array(a) => for x in a {
                match (&x[0], &x[1]) {
                    (Value::Number(n), b) => s.logit_bias.push((n.as_u64().ok_or("logit_bias token ids are non-negative integers")? as u32, bias_of(b)?)),
                    (Value::String(_), _) => o.logit_bias_text.push((x[0].as_str().unwrap().to_string(), bias_of(&x[1])?)),
                    _ => return Err(format!("logit_bias entries are [token, bias], got {x}")),
                }
            },
            v => return Err(format!("`logit_bias` must be an object or an array, got {v}")),
        }
        if let Some(m) = f32_in(req, "dry_multiplier", 0.0, 100.0)? { s.dry_multiplier = m; }
        if let Some(b) = f32_in(req, "dry_base", 1.0, 100.0)? { s.dry_base = b; }
        if let Some(n) = req["dry_allowed_length"].as_u64() { s.dry_allowed_length = n as usize; }
        match &req["dry_penalty_last_n"] { Value::Null => {}, v => s.dry_range = v.as_i64().filter(|&n| n >= -1).map(|n| n.max(0) as usize).ok_or("`dry_penalty_last_n` must be -1, 0 or positive")? }
        match &req["dry_sequence_breakers"] {
            Value::Null => {}
            Value::Array(a) => s.dry_breaker_strings = a.iter().map(|x| x.as_str().map(String::from).ok_or("`dry_sequence_breakers` must be strings")).collect::<Result<_, _>>()?,
            v => return Err(format!("`dry_sequence_breakers` must be an array of strings, got {v}")),
        }
        if let Some(t) = f32_in(req, "xtc_threshold", 0.0, 1.0)? { s.xtc_threshold = t; }
        if let Some(p) = f32_in(req, "xtc_probability", 0.0, 1.0)? { s.xtc_probability = p; }
        if let Some(p) = f32_in(req, "typical_p", 0.0, 1.0)? { if p <= 0.0 { return Err("`typical_p` must be greater than 0 (1.0 = off)".into()); } s.typical_p = p; }
        if let Some(n) = f32_in(req, "top_n_sigma", -1.0, 100.0)? { s.top_n_sigma = n.max(0.0); }
        match &req["mirostat"] {
            Value::Null => {}
            v => match v.as_u64() {
                Some(0) => s.mirostat = 0,
                Some(2) => s.mirostat = 2,
                Some(1) => return Err("`mirostat` 1 is not served; mirostat 2 is (text-generation-webui's, which defines the implementation checked here)".into()),
                _ => return Err(format!("`mirostat` must be 0 or 2, got {v}")),
            },
        }
        if let Some(t) = f32_in(req, "mirostat_tau", 0.0, 100.0)? { s.mirostat_tau = t; }
        if let Some(e) = f32_in(req, "mirostat_eta", 0.0, 10.0)? { s.mirostat_eta = e; }
        match &req["seed"] {
            Value::Null => {}
            v => match v.as_i64() {
                Some(x) => o.rng = seed_state(x as u64),
                None => return Err(format!("`seed` must be an integer, got {v}")),
            },
        }
        o.stop = match &req["stop"] {
            Value::Null => Vec::new(),
            Value::String(x) => vec![x.clone()],
            Value::Array(a) => {
                let mut v = Vec::with_capacity(a.len());
                for x in a { match x.as_str() { Some(t) => v.push(t.to_string()), None => return Err("`stop` must be a string or an array of strings".into()) } }
                v
            }
            v => return Err(format!("`stop` must be a string or an array of strings, got {v}")),
        };
        o.stop.retain(|x| !x.is_empty());
        if o.stop.len() > 16 { return Err("`stop` takes at most 16 sequences".into()); }
        if chat {
            o.logprobs = req["logprobs"].as_bool().unwrap_or(false);
            match &req["top_logprobs"] {
                Value::Null => {}
                v => match v.as_u64() {
                    Some(n) if n <= 20 => { o.top_logprobs = n as usize; if n > 0 && !o.logprobs {
                        return Err("`top_logprobs` needs `logprobs: true`".into()); } }
                    _ => return Err(format!("`top_logprobs` must be an integer in [0, 20], got {v}")),
                },
            }
        } else {
            // Completions: `logprobs` is the NUMBER of alternatives (0 = the chosen token only).
            match &req["logprobs"] {
                Value::Null => {}
                v => match v.as_u64() {
                    Some(n) if n <= 5 => { o.logprobs = true; o.top_logprobs = n as usize; }
                    _ => return Err(format!("`logprobs` must be an integer in [0, 5] for completions, got {v}")),
                },
            }
        }
        Ok(o)
    }
}

/// A `logit_bias` value: a number in [-100, 100] (OpenAI's range), or `false` (llama-server: ban the token).
fn bias_of(v: &Value) -> Result<f32, String> {
    match v {
        Value::Bool(false) => Ok(f32::NEG_INFINITY),
        v => v.as_f64().filter(|b| (-100.0..=100.0).contains(b)).map(|b| b as f32)
            .ok_or_else(|| format!("a logit_bias value must be a number in [-100, 100] or false, got {v}")),
    }
}

// ─────────────────────────────── Extended samplers (each a pure function) ───────────────────────────────
// Checked against the code that DEFINES each one (tests/fixtures/samplers: text-generation-webui's
// sampler_hijack.py for DRY, XTC, top-nσ and Mirostat v2 — p-e-w introduced DRY and XTC there — and
// transformers' TypicalLogitsWarper for typical-p), on recorded logits and contexts.

/// DRY, as `DRYLogitsProcessor.__call__`: for each earlier occurrence of the last token, extend the match
/// backwards (at most 50, stopping at a breaker); the token that followed it is penalised by
/// `multiplier · base^(length − allowed)` when the longest such match reaches `allowed`.
pub(crate) fn apply_dry(scores: &mut [f32], context: &[u32], multiplier: f32, base: f32, allowed: usize, breakers: &[u32], range: usize) {
    let ids = if range > 0 && context.len() > range { &context[context.len() - range..] } else { context };
    let Some(&last) = ids.last() else { return };
    if breakers.contains(&last) { return; }
    let mut lengths: std::collections::HashMap<u32, usize> = std::collections::HashMap::new();
    for i in 0..ids.len() - 1 {
        if ids[i] != last { continue; }
        let next = ids[i + 1];
        if breakers.contains(&next) { continue; }
        let mut len = 1usize;
        while len < 50 {
            if len > i { break; }
            let j = i - len;
            let prev = ids[ids.len() - (len + 1)];
            if ids[j] != prev || breakers.contains(&prev) { break; }
            len += 1;
        }
        let e = lengths.entry(next).or_insert(0);
        *e = (*e).max(len);
    }
    for (t, len) in lengths {
        if len >= allowed {
            if let Some(x) = scores.get_mut(t as usize) { *x -= multiplier * base.powi((len - allowed) as i32); }
        }
    }
}

/// Top-nσ, as `TopNSigmaLogitsWarper`: the threshold is max − n·std, with the standard deviation (unbiased)
/// taken over the WHOLE row with removed (non-finite) entries counted as 0 — torch.std over masked_fill.
pub(crate) fn top_n_sigma_keep(scores: &[f32], n: f32) -> Vec<bool> {
    let max = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let v: Vec<f64> = scores.iter().map(|&x| if x.is_finite() { x as f64 } else { 0.0 }).collect();
    let mean = v.iter().sum::<f64>() / v.len() as f64;
    let std = (v.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (v.len().max(2) - 1) as f64).sqrt();
    let thr = max as f64 - n as f64 * std;
    scores.iter().map(|&x| x as f64 >= thr).collect()
}

/// Locally typical sampling, as transformers' `TypicalLogitsWarper` on the candidates `cand` (logits already
/// temperature-scaled in `s`): keep the tokens whose surprise is closest to the entropy until `mass` is
/// covered. Returns the kept candidates in `cand`'s order.
pub(crate) fn typical_filter(cand: &[usize], s: &[f32], mass: f32) -> Vec<usize> {
    let maxv = cand.iter().map(|&i| s[i] as f64).fold(f64::NEG_INFINITY, f64::max);
    let lse = maxv + cand.iter().map(|&i| (s[i] as f64 - maxv).exp()).sum::<f64>().ln();
    let norm: Vec<f64> = cand.iter().map(|&i| s[i] as f64 - lse).collect();
    let ent: f64 = -norm.iter().map(|&n| n * n.exp()).sum::<f64>();
    let shifted: Vec<f64> = norm.iter().map(|&n| (-n - ent).abs()).collect();
    let mut order: Vec<usize> = (0..cand.len()).collect();
    order.sort_by(|&a, &b| shifted[a].partial_cmp(&shifted[b]).unwrap_or(std::cmp::Ordering::Equal));
    let (mut cum, mut last) = (0f64, 0usize);
    for &o in &order { cum += norm[o].exp(); if cum < mass as f64 { last += 1; } }
    let last = last.min(order.len() - 1);
    let bound = shifted[order[last]];
    let keep_first = order[0];
    cand.iter().enumerate().filter(|&(k, _)| shifted[k] <= bound || k == keep_first).map(|(_, &i)| i).collect()
}

/// XTC's action, as `XTCLogitsWarper` once its random check has passed: among `cand` (sorted by descending
/// probability `p`, renormalised over `cand`), remove every token whose successor is at or above the
/// threshold — all above it but the least likely — unless a special token (newline, EOS) would go.
pub(crate) fn xtc_filter(cand: &[usize], p: &[f32], threshold: f32, specials: &[u32]) -> Option<Vec<usize>> {
    let tot: f32 = cand.iter().map(|&i| p[i]).sum();
    let remove: Vec<bool> = (0..cand.len()).map(|k| k + 1 < cand.len() && p[cand[k + 1]] / tot >= threshold).collect();
    if cand.iter().zip(&remove).any(|(&i, &r)| r && specials.contains(&(i as u32))) { return None; }
    Some(cand.iter().zip(&remove).filter(|(_, r)| !**r).map(|(&i, _)| i).collect())
}

/// Mirostat v2's truncation, as `MirostatLogitsWarper`: in descending probability (`probs_sorted`,
/// normalised), keep up to the first token whose surprise −log2 p exceeds `mu` (at least one).
pub(crate) fn mirostat_k(probs_sorted: &[f64], mu: f64) -> usize {
    for (i, &c) in probs_sorted.iter().enumerate() {
        if c > 0.0 && -c.log2() > mu { return if i == 0 { 1 } else { i }; }
    }
    probs_sorted.len()
}

/// Mirostat v2's update after drawing a token of probability `p_chosen` (within the truncated, renormalised set).
pub(crate) fn mirostat_update(mu: f64, p_chosen: f64, tau: f64, eta: f64) -> f64 { mu - eta * (-p_chosen.log2() - tau) }

/// xorshift64 — the generator Ferric has always used, kept so default sampled output does not move.
fn next_unit(rng: &mut u64) -> f32 {
    *rng ^= *rng << 13; *rng ^= *rng >> 7; *rng ^= *rng << 17;
    (*rng >> 11) as f32 / (1u64 << 53) as f32
}

/// **One token from one row of logits.** `prompt` is the whole prompt (its last `repeat_last_n` tokens feed
/// `repeat_penalty`, all of it DRY); `generated` feeds those and the OpenAI penalties. Temperature 0 is argmax
/// of the adjusted row. With every optional control off the sampled path reproduces the historical
/// `sample_top_p` exactly, RNG stream included; each extended sampler (logit_bias, DRY, top-nσ, typical-p,
/// Mirostat v2, XTC) runs only when set, in text-generation-webui's default order.
pub(crate) fn sample(row: &[f32], s: &Sampling, prompt: &[u32], generated: &[u32], rng: &mut u64) -> u32 {
    let prompt_tail = &prompt[prompt.len().saturating_sub(s.repeat_last_n)..];
    let mut owned: Option<Vec<f32>> = None;
    if !s.logit_bias.is_empty() {
        let r = owned.get_or_insert_with(|| row.to_vec());
        for &(t, b) in &s.logit_bias { if let Some(x) = r.get_mut(t as usize) { *x += b; } }
    }
    if s.presence_penalty != 0.0 || s.frequency_penalty != 0.0 || s.repeat_penalty != 1.0 {
        let r = owned.get_or_insert_with(|| row.to_vec());
        if s.presence_penalty != 0.0 || s.frequency_penalty != 0.0 {
            let mut counts = std::collections::HashMap::<u32, u32>::new();
            for &t in generated { *counts.entry(t).or_default() += 1; }
            for (&t, &c) in &counts {
                if let Some(x) = r.get_mut(t as usize) { *x -= s.presence_penalty + s.frequency_penalty * c as f32; }
            }
        }
        if s.repeat_penalty != 1.0 && s.repeat_last_n > 0 {
            let seen: std::collections::HashSet<u32> = prompt_tail.iter().chain(generated.iter())
                .rev().take(s.repeat_last_n).copied().collect();
            for t in seen {
                if let Some(x) = r.get_mut(t as usize) { *x = if *x > 0.0 { *x / s.repeat_penalty } else { *x * s.repeat_penalty }; }
            }
        }
    }
    if s.dry_multiplier > 0.0 {
        let r = owned.get_or_insert_with(|| row.to_vec());
        let ctx: Vec<u32> = prompt.iter().chain(generated.iter()).copied().collect();
        apply_dry(r, &ctx, s.dry_multiplier, s.dry_base, s.dry_allowed_length, &s.dry_breakers, s.dry_range);
    }
    let row: &[f32] = owned.as_deref().unwrap_or(row);
    let argmax = || (0..row.len()).max_by(|&a, &b| row[a].partial_cmp(&row[b]).unwrap_or(std::cmp::Ordering::Equal)).unwrap_or(0) as u32;
    if s.temperature <= 0.0 { return argmax(); }
    let maxl = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    if !maxl.is_finite() { return argmax(); }
    let probs: Vec<f32> = row.iter().map(|&l| ((l - maxl) / s.temperature).exp()).collect();
    let sum: f32 = probs.iter().sum();
    let mut idx: Vec<usize> = (0..row.len()).collect();
    idx.sort_by(|&a, &b| probs[b].partial_cmp(&probs[a]).unwrap_or(std::cmp::Ordering::Equal));
    // Top-nσ is scale-free, so taking it on the unscaled row equals taking it after temperature.
    let sigma_cut = s.top_n_sigma > 0.0;
    if sigma_cut { let keep = top_n_sigma_keep(row, s.top_n_sigma); idx.retain(|&i| keep[i]); }
    if s.top_k > 0 { idx.truncate(s.top_k); }
    if s.min_p > 0.0 {
        let floor = s.min_p * probs[idx[0]];
        idx.retain(|&i| probs[i] >= floor);
    }
    // Renormalise over what survived top-k / min-p / top-nσ (nothing, with all off — then `sum` is unchanged).
    let sum = if s.top_k > 0 || s.min_p > 0.0 || sigma_cut { idx.iter().map(|&i| probs[i]).sum::<f32>() } else { sum };
    // nucleus: smallest prefix whose mass reaches top_p
    let (mut cum, mut cut) = (0.0f32, idx.len());
    for (k, &i) in idx.iter().enumerate() { cum += probs[i] / sum; if cum >= s.top_p { cut = k + 1; break; } }
    if s.typical_p >= 1.0 && s.mirostat == 0 && s.xtc_probability <= 0.0 {
        let r = next_unit(rng) * cum;
        let (mut acc, mut pick) = (0.0f32, idx[0]);
        for &i in &idx[..cut] { acc += probs[i] / sum; if acc >= r { pick = i; break; } }
        return pick as u32;
    }
    // The extended tail: typical-p, then Mirostat v2 (which draws itself), then XTC, then one draw.
    let mut cand: Vec<usize> = idx[..cut].to_vec();
    let scaled: Vec<f32> = row.iter().map(|&l| l / s.temperature).collect();
    if s.typical_p < 1.0 { cand = typical_filter(&cand, &scaled, s.typical_p); }
    if s.mirostat == 2 {
        let mu = if s.mirostat_mu.get().is_nan() { 2.0 * s.mirostat_tau } else { s.mirostat_mu.get() };
        let m = cand.iter().map(|&i| scaled[i] as f64).fold(f64::NEG_INFINITY, f64::max);
        let z: f64 = cand.iter().map(|&i| (scaled[i] as f64 - m).exp()).sum();
        let ps: Vec<f64> = cand.iter().map(|&i| (scaled[i] as f64 - m).exp() / z).collect();
        let k = mirostat_k(&ps, mu as f64);
        let zk: f64 = ps[..k].iter().sum();
        let r = next_unit(rng) as f64;
        let (mut acc, mut pick) = (0.0f64, 0usize);
        for j in 0..k { acc += ps[j] / zk; if acc >= r { pick = j; break; } pick = j; }
        s.mirostat_mu.set(mirostat_update(mu as f64, ps[pick] / zk, s.mirostat_tau as f64, s.mirostat_eta as f64) as f32);
        return cand[pick] as u32;
    }
    if s.xtc_probability > 0.0 && next_unit(rng) < s.xtc_probability {
        if let Some(kept) = xtc_filter(&cand, &probs, s.xtc_threshold, &s.xtc_specials) { cand = kept; }
    }
    let tot: f32 = cand.iter().map(|&i| probs[i]).sum();
    let r = next_unit(rng) * tot;
    let (mut acc, mut pick) = (0.0f32, cand[0]);
    for &i in &cand { acc += probs[i]; if acc >= r { pick = i; break; } }
    pick as u32
}

/// log-softmax of the RAW model row at `tok`, and the `top` most likely alternatives — the model's own
/// distribution, before penalties or temperature (what vLLM reports by default).
pub(crate) fn logprobs_of(row: &[f32], tok: u32, top: usize) -> (f32, Vec<(u32, f32)>) {
    let maxl = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let lse = maxl + row.iter().map(|&l| ((l - maxl) as f64).exp()).sum::<f64>().ln() as f32;
    let lp = row.get(tok as usize).map(|&l| l - lse).unwrap_or(f32::NEG_INFINITY);
    let mut alts = Vec::new();
    if top > 0 {
        let mut idx: Vec<usize> = (0..row.len()).collect();
        let k = top.min(idx.len());
        idx.select_nth_unstable_by(k - 1, |&a, &b| row[b].partial_cmp(&row[a]).unwrap_or(std::cmp::Ordering::Equal));
        idx.truncate(k);
        idx.sort_by(|&a, &b| row[b].partial_cmp(&row[a]).unwrap_or(std::cmp::Ordering::Equal));
        alts = idx.into_iter().map(|i| (i as u32, row[i] - lse)).collect();
    }
    (lp, alts)
}

/// One chat `logprobs.content[]` entry.
pub(crate) fn logprob_entry(piece: &dyn Fn(u32) -> (String, Vec<u8>), tok: u32, lp: f32, alts: &[(u32, f32)]) -> Value {
    let (t, b) = piece(tok);
    json!({"token": t, "logprob": lp, "bytes": b,
           "top_logprobs": alts.iter().map(|&(a, l)| { let (t, b) = piece(a); json!({"token": t, "logprob": l, "bytes": b}) }).collect::<Vec<_>>()})
}

/// **Stop strings and streaming, together.** Holds back any tail of the decoded text that could still
/// become a stop string, cuts the text at the first stop string that completes, and reports whether it
/// did. OpenAI returns the text BEFORE the stop string and never the stop string itself.
pub(crate) struct Emitter {
    stop: Vec<String>,
    /// The decoded text of every accepted token, cut at a stop string once one completes.
    pub text: String,
    /// Bytes of `text` already released to the client.
    sent: usize,
    pub hit_stop: bool,
    /// Which stop string ended it (Anthropic's `stop_sequence` names it).
    pub hit_str: Option<String>,
}

impl Emitter {
    pub fn new(stop: &[String]) -> Emitter { Emitter { stop: stop.to_vec(), text: String::new(), sent: 0, hit_stop: false, hit_str: None } }

    /// `full` is the decode of every accepted token so far. Returns the text that may be released now.
    /// Mirrors the old loop's guard: nothing is released until the decode grows on a char boundary, so a
    /// multi-byte character split across tokens is never cut.
    pub fn update(&mut self, full: &str) -> Option<String> {
        if self.hit_stop { return None; }
        if !(full.len() > self.text.len() && full.is_char_boundary(self.text.len())) { return None; }
        self.text = full.to_string();
        if self.stop.is_empty() { return self.release(self.text.len()); }
        // Earliest completed stop string. A stop string can only newly complete in the unreleased tail
        // plus the longest stop string before it.
        let longest = self.stop.iter().map(|s| s.len()).max().unwrap_or(0);
        let mut from = self.sent.saturating_sub(longest);
        while !self.text.is_char_boundary(from) { from -= 1; }
        let hit = self.stop.iter().filter_map(|s| self.text[from..].find(s.as_str()).map(|p| (from + p, s.clone()))).min_by_key(|(p, _)| *p);
        if let Some((p, which)) = hit {
            self.hit_stop = true;
            self.hit_str = Some(which);
            self.text.truncate(p);
            return self.release(p.max(self.sent));
        }
        // Hold back the longest suffix that is a proper prefix of some stop string.
        let mut hold = 0;
        for s in &self.stop {
            for k in (1..s.len().min(self.text.len() + 1)).rev() {
                if s.is_char_boundary(k) && self.text.ends_with(&s[..k]) { hold = hold.max(k); break; }
            }
        }
        let mut safe = self.text.len() - hold;
        while !self.text.is_char_boundary(safe) { safe -= 1; }
        self.release(safe)
    }

    fn release(&mut self, upto: usize) -> Option<String> {
        if upto <= self.sent { return None; }
        let d = self.text[self.sent..upto].to_string();
        self.sent = upto;
        Some(d)
    }

    /// End of generation: whatever was held back for a stop string that never completed.
    pub fn flush(&mut self) -> Option<String> { let n = self.text.len(); self.release(n) }
}

/// **Reasoning apart from the answer.** A thinking model writes its reasoning between markers its chat
/// template defines — `<think>…</think>` (Qwen3/3.5, DeepSeek-R1, MiMo, Nemotron) or
/// `<|channel>thought…<channel|>` (Gemma 4) — and 12 of 14 serving peers return it as `reasoning_content`
/// rather than inside the answer. Fed the decode WITH special tokens (Gemma's markers are specials), in
/// pieces as they stream; `push` returns (reasoning, content) text safe to release, holding back any tail
/// that could still become a marker.
pub(crate) struct ReasoningSplit {
    open: String,
    close: String,
    inside: bool,
    buf: String,
    pub reasoning: String,
    pub content: String,
}

impl ReasoningSplit {
    /// `started` = the prompt already opened the block (a template that ends its generation prompt with
    /// `<think>\n`): everything up to the close marker is reasoning.
    pub fn new(open: &str, close: &str, started: bool) -> ReasoningSplit {
        ReasoningSplit { open: open.into(), close: close.into(), inside: started, buf: String::new(),
                         reasoning: String::new(), content: String::new() }
    }

    pub fn push(&mut self, piece: &str) -> (String, String) {
        self.buf.push_str(piece);
        let (mut r, mut c) = (String::new(), String::new());
        loop {
            let marker = if self.inside { self.close.clone() } else { self.open.clone() };
            if let Some(p) = self.buf.find(&marker) {
                let head: String = self.buf[..p].to_string();
                if self.inside { r.push_str(&head) } else { c.push_str(&head) }
                self.buf = self.buf[p + marker.len()..].to_string();
                self.inside = !self.inside;
                continue;
            }
            // Keep the longest tail that is a proper prefix of the marker; release the rest.
            let mut hold = 0;
            for k in (1..marker.len().min(self.buf.len() + 1)).rev() {
                if marker.is_char_boundary(k) && self.buf.ends_with(&marker[..k]) { hold = k; break; }
            }
            let mut cut = self.buf.len() - hold;
            while !self.buf.is_char_boundary(cut) { cut -= 1; }
            let out: String = self.buf[..cut].to_string();
            self.buf = self.buf[cut..].to_string();
            if self.inside { r.push_str(&out) } else { c.push_str(&out) }
            break;
        }
        // The first content after the block usually opens with the template's newlines; they are framing.
        if !self.content.is_empty() || !c.trim_start().is_empty() || self.inside {
            if self.content.is_empty() { c = c.trim_start().to_string(); }
        } else { c.clear(); }
        if self.reasoning.is_empty() { r = r.trim_start().to_string(); }
        self.reasoning.push_str(&r);
        self.content.push_str(&c);
        (r, c)
    }

    /// End of generation: whatever was held back belongs where the stream stood.
    pub fn finish(&mut self) -> (String, String) {
        let rest = std::mem::take(&mut self.buf);
        if self.inside { self.reasoning.push_str(&rest); (rest, String::new()) } else { self.content.push_str(&rest); (String::new(), rest) }
    }
}

/// OpenAI message `content`: a string, null, or an array of parts. Text parts are joined with a newline
/// (vLLM's rule). Image and audio parts are refused by name — this path feeds text-only prompts, and
/// dropping an image silently answers a question the user did not ask.
pub(crate) fn content_text(c: &Value) -> Result<String, String> {
    match c {
        Value::Null => Ok(String::new()),
        Value::String(s) => Ok(s.clone()),
        Value::Array(parts) => {
            let mut out: Vec<String> = Vec::with_capacity(parts.len());
            for p in parts {
                match p["type"].as_str() {
                    Some("text") | Some("input_text") | Some("output_text") =>
                        out.push(p["text"].as_str().ok_or("a text content part has no `text` string")?.to_string()),
                    Some(t @ ("image_url" | "input_image" | "image" | "input_audio" | "audio" | "file")) =>
                        return Err(format!("content part of type `{t}`: this server feeds text-only prompts; \
                                            images and audio are not accepted here yet")),
                    Some(t) => return Err(format!("unknown content part type `{t}`")),
                    None => return Err("a content part has no `type`".into()),
                }
            }
            Ok(out.join("\n"))
        }
        v => Err(format!("message `content` must be a string, null or an array of parts, got {v}")),
    }
}

#[cfg(test)]
mod tests {
    /// Each extended sampler against the code that defines it (tests/fixtures/samplers/reference.json.gz:
    /// text-generation-webui's classes copied verbatim, transformers' TypicalLogitsWarper), on seeded logits.
    #[test]
    fn extended_samplers_match_the_code_that_defines_them() {
        use std::io::Read;
        let gz = std::fs::read(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/samplers/reference.json.gz")).unwrap();
        let mut js = String::new();
        flate2::read::GzDecoder::new(&gz[..]).read_to_string(&mut js).unwrap();
        let fx: Value = serde_json::from_str(&js).unwrap();
        let c = &fx["cases"];
        let f32s = |v: &Value| -> Vec<f32> { v.as_array().unwrap().iter().map(|x| x.as_f64().map(|f| f as f32).unwrap_or(f32::NEG_INFINITY)).collect() };
        let u32s = |v: &Value| -> Vec<u32> { v.as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as u32).collect() };
        let bools = |v: &Value| -> Vec<bool> { v.as_array().unwrap().iter().map(|x| x.as_bool().unwrap()).collect() };
        let by_prob = |l: &[f32]| { let mut o: Vec<usize> = (0..l.len()).collect(); o.sort_by(|&a, &b| l[b].partial_cmp(&l[a]).unwrap()); o };
        let mut n = 0;
        for d in c["dry"].as_array().unwrap() {
            let mut l = f32s(&d["logits"]);
            apply_dry(&mut l, &u32s(&d["context"]), d["multiplier"].as_f64().unwrap() as f32, d["base"].as_f64().unwrap() as f32,
                      d["allowed_length"].as_u64().unwrap() as usize, &u32s(&d["breakers"]), d["range"].as_u64().unwrap() as usize);
            let want = f32s(&d["out"]);
            let err = l.iter().zip(&want).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
            assert!(err < 1e-4, "DRY off by {err}: {d}"); n += 1;
        }
        for d in c["xtc"].as_array().unwrap() {
            let l = f32s(&d["logits"]);
            let m = l.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let p: Vec<f32> = l.iter().map(|&x| (x - m).exp()).collect();
            let cand = by_prob(&l);
            let kept = xtc_filter(&cand, &p, d["threshold"].as_f64().unwrap() as f32, &u32s(&d["specials"])).unwrap_or(cand.clone());
            let mut mask = vec![false; l.len()];
            for i in kept { mask[i] = true; }
            assert_eq!(mask, bools(&d["kept"]), "XTC: {d}"); n += 1;
        }
        for d in c["typical"].as_array().unwrap() {
            let l = f32s(&d["logits"]);
            let kept = typical_filter(&by_prob(&l), &l, d["mass"].as_f64().unwrap() as f32);
            let mut mask = vec![false; l.len()];
            for i in kept { mask[i] = true; }
            assert_eq!(mask, bools(&d["kept"]), "typical: {d}"); n += 1;
        }
        for d in c["top_n_sigma"].as_array().unwrap() {
            let l = f32s(&d["logits"]);
            let keep: Vec<bool> = top_n_sigma_keep(&l, d["n"].as_f64().unwrap() as f32).iter().zip(&l).map(|(&k, x)| k && x.is_finite()).collect();
            assert_eq!(keep, bools(&d["kept"]), "top-n-sigma: {d}"); n += 1;
        }
        for d in c["mirostat"].as_array().unwrap() {
            let (tau, eta) = (d["tau"].as_f64().unwrap(), d["eta"].as_f64().unwrap());
            for st in d["steps"].as_array().unwrap() {
                let l = f32s(&st["logits"]);
                let o = by_prob(&l);
                let m = l[o[0]] as f64;
                let z: f64 = o.iter().map(|&i| (l[i] as f64 - m).exp()).sum();
                let ps: Vec<f64> = o.iter().map(|&i| (l[i] as f64 - m).exp() / z).collect();
                let mu = st["mu_before"].as_f64().unwrap();
                let k = mirostat_k(&ps, mu);
                assert_eq!(k as u64, st["k"].as_u64().unwrap(), "mirostat truncation: {st}");
                let rank = o.iter().position(|&i| i as u64 == st["chosen"].as_u64().unwrap()).unwrap();
                let zk: f64 = ps[..k].iter().sum();
                let mu2 = mirostat_update(mu, ps[rank] / zk, tau, eta);
                assert!((mu2 - st["mu_after"].as_f64().unwrap()).abs() < 1e-4, "mirostat mu {mu2} vs {st}");
                n += 1;
            }
        }
        assert_eq!(n, 332, "the fixture changed size");
    }

    #[test]
    fn extended_sampler_parameters_are_read_and_bad_ones_refused() {
        let o = GenOpts::from_req(&json!({"logit_bias": {"5": -100, "7": 2.5}, "dry_multiplier": 0.8, "xtc_probability": 0.5,
            "typical_p": 0.9, "top_n_sigma": 1.0, "mirostat": 2, "mirostat_tau": 4.0}), true).unwrap();
        let s = &o.sampling;
        assert_eq!((s.logit_bias.len(), s.dry_multiplier, s.xtc_probability, s.typical_p, s.top_n_sigma, s.mirostat, s.mirostat_tau), (2, 0.8, 0.5, 0.9, 1.0, 2, 4.0));
        let o = GenOpts::from_req(&json!({"logit_bias": [[5, false], ["hello", 1.0]]}), true).unwrap();
        assert_eq!(o.sampling.logit_bias, vec![(5, f32::NEG_INFINITY)]);
        assert_eq!(o.logit_bias_text, vec![("hello".to_string(), 1.0)]);
        for bad in [json!({"logit_bias": {"x": 1}}), json!({"logit_bias": {"5": 101}}), json!({"mirostat": 1}), json!({"typical_p": 0}),
                    json!({"dry_sequence_breakers": "\n"})] {
            assert!(GenOpts::from_req(&bad, true).is_err(), "{bad} must be refused");
        }
        // Off by default, so the default path stays the historical sampler (pinned elsewhere).
        let d = Sampling::default();
        assert!(d.logit_bias.is_empty() && d.dry_multiplier == 0.0 && d.xtc_probability == 0.0 && d.typical_p == 1.0 && d.top_n_sigma == 0.0 && d.mirostat == 0);
    }

    /// Mirostat keeps its threshold per sequence and moves it toward the target surprise.
    #[test]
    fn mirostat_state_is_per_sequence_and_moves() {
        let o = GenOpts::from_req(&json!({"mirostat": 2, "temperature": 1.0, "mirostat_tau": 3.0}), true).unwrap();
        let row: Vec<f32> = (0..32).map(|i| (i as f32 * 0.37).sin() * 3.0).collect();
        let mut rng = DEFAULT_RNG;
        assert!(o.sampling.mirostat_mu.get().is_nan());
        let _ = sample(&row, &o.sampling, &[], &[], &mut rng);
        let mu1 = o.sampling.mirostat_mu.get();
        assert!(mu1.is_finite() && mu1 != 6.0, "mu starts at 2*tau and moves after one step: {mu1}");
        let fresh = o.clone();
        let other = GenOpts::from_req(&json!({"mirostat": 2, "temperature": 1.0, "mirostat_tau": 3.0}), true).unwrap();
        assert!(other.sampling.mirostat_mu.get().is_nan(), "a new request starts fresh");
        assert_eq!(fresh.sampling.mirostat_mu.get(), mu1, "a clone carries its own copy");
    }

    use super::*;

    fn run(e: &mut Emitter, pieces: &[&str]) -> String {
        let mut full = String::new();
        let mut out = String::new();
        for p in pieces { full.push_str(p); if let Some(d) = e.update(&full) { out.push_str(&d); } if e.hit_stop { break; } }
        if let Some(d) = e.flush() { out.push_str(&d); }
        out
    }

    #[test]
    fn a_stop_string_split_across_tokens_is_cut_and_never_released() {
        let mut e = Emitter::new(&["\nUser:".to_string()]);
        let out = run(&mut e, &["Hello", " there", "\nUs", "er:", " more"]);
        assert_eq!(out, "Hello there");
        assert!(e.hit_stop);
        assert_eq!(e.text, "Hello there");
    }

    #[test]
    fn a_held_back_prefix_that_never_completes_is_released_at_the_end() {
        let mut e = Emitter::new(&["STOP".to_string()]);
        let out = run(&mut e, &["abc", "ST", "O"]);
        assert_eq!(out, "abcSTO");
        assert!(!e.hit_stop);
    }

    #[test]
    fn with_no_stop_strings_every_delta_is_released_immediately() {
        let mut e = Emitter::new(&[]);
        assert_eq!(e.update("ab").as_deref(), Some("ab"));
        assert_eq!(e.update("abcd").as_deref(), Some("cd"));
        assert_eq!(e.flush(), None);
    }

    #[test]
    fn the_earliest_of_several_stop_strings_wins() {
        let mut e = Emitter::new(&["zz".to_string(), "b".to_string()]);
        let out = run(&mut e, &["aaabzz"]);
        assert_eq!(out, "aaa");
    }

    fn split_all(sp: &mut ReasoningSplit, pieces: &[&str]) -> (String, String) {
        let (mut r, mut c) = (String::new(), String::new());
        for p in pieces { let (a, b) = sp.push(p); r.push_str(&a); c.push_str(&b); }
        let (a, b) = sp.finish(); r.push_str(&a); c.push_str(&b);
        (r, c)
    }

    #[test]
    fn reasoning_is_split_from_the_answer_across_any_token_boundary() {
        let mut sp = ReasoningSplit::new("<think>", "</think>", false);
        let (r, c) = split_all(&mut sp, &["<th", "ink>\nLet me ", "add: 2+2=4.\n</thi", "nk>\n\nThe answer is 4."]);
        assert_eq!((r.as_str(), c.as_str()), ("Let me add: 2+2=4.\n", "The answer is 4."));
        // a prompt that opened the block: everything up to the close is reasoning
        let mut sp = ReasoningSplit::new("<think>", "</think>", true);
        assert_eq!(split_all(&mut sp, &["hmm</think>Four."]), ("hmm".to_string(), "Four.".to_string()));
        // Gemma 4's channel markers
        let mut sp = ReasoningSplit::new("<|channel>thought", "<channel|>", false);
        assert_eq!(split_all(&mut sp, &["<|channel>thought\nParis is the capital.<channel|>Paris"]),
                   ("Paris is the capital.".to_string(), "Paris".to_string()));
        // no block at all: all content
        let mut sp = ReasoningSplit::new("<think>", "</think>", false);
        assert_eq!(split_all(&mut sp, &["Just ", "an answer."]), (String::new(), "Just an answer.".to_string()));
    }

    #[test]
    fn content_parts_are_joined_and_images_refused() {
        let v = json!([{"type": "text", "text": "What is"}, {"type": "text", "text": "the capital of France?"}]);
        assert_eq!(content_text(&v).unwrap(), "What is\nthe capital of France?");
        assert_eq!(content_text(&json!("plain")).unwrap(), "plain");
        assert_eq!(content_text(&Value::Null).unwrap(), "");
        let img = json!([{"type": "text", "text": "what is this"}, {"type": "image_url", "image_url": {"url": "data:..."}}]);
        assert!(content_text(&img).unwrap_err().contains("image_url"));
    }

    #[test]
    fn unsupported_or_malformed_parameters_are_refused_not_ignored() {
        assert!(GenOpts::from_req(&json!({"n": 2}), true).unwrap_err().contains("`n`"));
        assert!(GenOpts::from_req(&json!({"top_p": 0}), true).is_err());
        assert!(GenOpts::from_req(&json!({"temperature": -1}), true).is_err());
        assert!(GenOpts::from_req(&json!({"stop": [1, 2]}), true).is_err());
        assert!(GenOpts::from_req(&json!({"top_logprobs": 3}), true).is_err());
        let o = GenOpts::from_req(&json!({"max_completion_tokens": 7, "stop": "x", "seed": 1, "top_k": 5}), true).unwrap();
        assert_eq!(o.max_tokens, Some(7));
        assert_eq!(o.stop, vec!["x".to_string()]);
        assert_eq!(o.sampling.top_k, 5);
        assert_ne!(o.rng, DEFAULT_RNG);
        assert_eq!(GenOpts::from_req(&json!({}), true).unwrap().max_tokens, None);
    }

    /// The historical sampler, verbatim, so the default path can be checked against it.
    fn old_sample_top_p(row: &[f32], temp: f32, top_p: f32, rng: &mut u64) -> u32 {
        let maxl = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let probs: Vec<f32> = row.iter().map(|&l| ((l - maxl) / temp).exp()).collect();
        let sum: f32 = probs.iter().sum();
        let mut idx: Vec<usize> = (0..row.len()).collect();
        idx.sort_by(|&a, &b| probs[b].partial_cmp(&probs[a]).unwrap());
        let (mut cum, mut cut) = (0.0f32, idx.len());
        for (k, &i) in idx.iter().enumerate() { cum += probs[i] / sum; if cum >= top_p { cut = k + 1; break; } }
        *rng ^= *rng << 13; *rng ^= *rng >> 7; *rng ^= *rng << 17;
        let r = (*rng >> 11) as f32 / (1u64 << 53) as f32 * cum;
        let (mut acc, mut pick) = (0.0f32, idx[0]);
        for &i in &idx[..cut] { acc += probs[i] / sum; if acc >= r { pick = i; break; } }
        pick as u32
    }

    #[test]
    fn default_sampling_reproduces_the_historical_sampler_token_for_token() {
        let row: Vec<f32> = (0..500).map(|i| ((i * 7919 % 101) as f32) * 0.05).collect();
        let s = Sampling { temperature: 0.8, ..Sampling::default() };
        let (mut a, mut b) = (DEFAULT_RNG, DEFAULT_RNG);
        let mut differs_from_argmax = false;
        for _ in 0..200 {
            let x = sample(&row, &s, &[], &[], &mut a);
            let y = old_sample_top_p(&row, 0.8, 0.95, &mut b);
            assert_eq!(x, y, "the default path must not move sampled output");
            differs_from_argmax |= x != sample(&row, &Sampling::default(), &[], &[], &mut 1);
        }
        assert!(differs_from_argmax, "the row must actually be sampled, or this compares argmax to argmax");
    }

    #[test]
    fn top_k_one_and_min_p_one_are_argmax_and_penalties_move_the_choice() {
        let row = vec![0.0, 3.0, 2.9, 1.0];
        let mut rng = DEFAULT_RNG;
        for _ in 0..50 {
            assert_eq!(sample(&row, &Sampling { temperature: 1.0, top_k: 1, ..Sampling::default() }, &[], &[], &mut rng), 1);
            assert_eq!(sample(&row, &Sampling { temperature: 1.0, min_p: 1.0, ..Sampling::default() }, &[], &[], &mut rng), 1);
        }
        let s = Sampling { presence_penalty: 0.5, ..Sampling::default() };
        assert_eq!(sample(&row, &s, &[], &[1], &mut rng), 2, "a presence penalty on token 1 must hand argmax to token 2");
    }

    #[test]
    fn logprobs_are_a_normalised_log_distribution() {
        let row = vec![1.0f32, 2.0, 3.0];
        let (lp, alts) = logprobs_of(&row, 2, 2);
        let total: f64 = row.iter().map(|&l| ((l - 3.0) as f64).exp()).sum();
        assert!((lp as f64 - (-(total.ln()))).abs() < 1e-6);
        assert_eq!(alts.iter().map(|a| a.0).collect::<Vec<_>>(), vec![2, 1]);
        let mass: f64 = row.iter().enumerate().map(|(i, _)| (logprobs_of(&row, i as u32, 0).0 as f64).exp()).sum();
        assert!((mass - 1.0).abs() < 1e-6);
    }
}
