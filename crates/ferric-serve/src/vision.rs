//! **Images in chat requests** — on the vision-language models whose image path is verified against
//! their authors, served from the authors' own checkpoint directory.
//!
//! Parity gap S18 (images over the API). The towers existed and were verified — Qwen2.5-VL on
//! MiMo-Embodied-7B from image file to logits and a 64-token greedy continuation
//! (`scripts/vl_conformance.sh`), Qwen3-VL from image file to final hidden state (`qwen3vl_e2e`) — but only
//! examples could reach them. `ferric-serve <checkpoint-dir>` (or a request naming `owner/repo` in the
//! Hugging Face cache) now serves such a model, and an image in a chat request goes through the same calls
//! those checks run:
//!
//! ```text
//! bytes (data: URL / base64) → images::decode (PIL-equivalent) → qwen3vl_image::preprocess (the authors'
//! resize and patching) → tower.encode_patches → splice into the prompt at its <|image_pad|> run →
//! forward_embeds_mm with the authors' 3-D mRoPE positions (qwen3vl_rope::rope_index) → decode steps at
//! position index + delta (NOT the cache index — see `decode`).
//! ```
//!
//! The prompt comes from the checkpoint's OWN `chat_template.json` / `tokenizer_config.json` and
//! `tokenizer.json`, not from a GGUF copy of them.
//!
//! ⛔ An image travels in the request's `GenOpts`, never in state on the `Engine`: two images of the same
//! size give IDENTICAL token ids (the same run of `<|image_pad|>`), so anything keyed on ids — the prefix
//! cache above all — could answer one request about another request's picture. Image prompts are never
//! seeded from, or offered to, the prefix cache, and never batched.
//!
//! Limits, refused by name: one image per request (the verified path; Qwen3-VL's deepstack features land
//! at one image's rows), no video, no remote URLs (the server does not fetch: send a `data:` URL).
use crate::{byte_decoder, genopts, images, ollama, template, Engine, Model, Shared};
use ferric_llama::qwen3::{Cache, Qwen3};
use ferric_llama::qwen3vl_image::{preprocess, PreprocCfg, KEYS_DEFAULT};
use ferric_tensor::Tensor;
use serde_json::{json, Value};
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

/// The checkpoint `model_type`s whose image path is verified here.
pub(crate) const VL_TYPES: &[&str] = &["qwen2_5_vl", "qwen3_vl"];

pub(crate) enum Tower {
    Qwen25(ferric_llama::qwen25vl_vision::VisionTower),
    Qwen3(ferric_llama::qwen3vl_vision::VisionTower),
}

/// A vision tower and what feeding it needs.
pub(crate) struct Vision {
    tower: Tower,
    pcfg: PreprocCfg,
    img_tok: u32,
}

/// One request's image, decoded and planned: what the prefill splices in.
#[derive(Debug)]
pub(crate) struct MmInput {
    px: Vec<f32>,
    gh: usize,
    gw: usize,
    /// Where its `<|image_pad|>` run starts in the prompt, and how long it is.
    start: usize,
    n: usize,
}

/// Base64, standard or URL-safe alphabet, whitespace and padding ignored.
fn b64(s: &str) -> Result<Vec<u8>, String> {
    let v = |c: u8| -> Result<u32, String> { Ok(match c {
        b'A'..=b'Z' => c - b'A', b'a'..=b'z' => c - b'a' + 26, b'0'..=b'9' => c - b'0' + 52,
        b'+' | b'-' => 62, b'/' | b'_' => 63, _ => return Err(format!("not base64 (byte {c:#04x})")),
    } as u32) };
    let s: Vec<u8> = s.bytes().filter(|c| !c.is_ascii_whitespace() && *c != b'=').collect();
    let mut out = Vec::with_capacity(s.len() * 3 / 4);
    for q in s.chunks(4) {
        if q.len() == 1 { return Err("truncated base64".into()); }
        let mut n = 0u32;
        for (i, &c) in q.iter().enumerate() { n |= v(c)? << (18 - 6 * i); }
        out.extend([(n >> 16) as u8, (n >> 8) as u8, n as u8].into_iter().take(q.len() - 1));
    }
    Ok(out)
}

/// An image's bytes from any dialect's spelling of it: OpenAI `image_url` (`{"url": …}` or a bare string),
/// the Responses API's `input_image`, Anthropic's `image` with a base64 `source`, or a bare base64 string
/// (Ollama's `images`). `data:` URLs and base64 only.
fn image_bytes(p: &Value) -> Result<Vec<u8>, String> {
    let url = p["image_url"]["url"].as_str().or_else(|| p["image_url"].as_str()).or_else(|| p.as_str());
    if let Some(u) = url {
        if u.starts_with("http://") || u.starts_with("https://") || u.starts_with("file:") {
            return Err("image URLs are not fetched by this server; send the image inline as a data: URL (base64)".into());
        }
        let data = u.strip_prefix("data:").map(|r| r.split_once(',').map(|(_, d)| d).unwrap_or("")).unwrap_or(u);
        return b64(data);
    }
    match p["source"]["type"].as_str() {
        Some("base64") => b64(p["source"]["data"].as_str().ok_or("an image source has no `data`")?),
        Some(t) => Err(format!("image source of type `{t}`: send base64")),
        None => Err("an image part carries no image (expected image_url.url, a data: URL, or source.data)".into()),
    }
}

fn is_image_part(p: &Value) -> bool { matches!(p["type"].as_str(), Some("image_url" | "input_image" | "image")) }

/// Messages as the template reads them, and the images they carry, in order. Text-only messages are left
/// exactly as the text path leaves them; a message with an image keeps its parts, the image as
/// `{"type": "image"}` — the spelling Qwen's templates test for — and never its bytes.
fn split_images(messages: &[Value]) -> Result<(Vec<Value>, Vec<Vec<u8>>), String> {
    let (mut out, mut imgs) = (Vec::with_capacity(messages.len()), Vec::new());
    for (i, m) in messages.iter().enumerate() {
        let ollama_imgs = m["images"].as_array().filter(|a| !a.is_empty());
        let parts = m["content"].as_array().filter(|a| a.iter().any(is_image_part));
        if ollama_imgs.is_none() && parts.is_none() { out.push(m.clone()); continue; }
        let mut content: Vec<Value> = Vec::new();
        // Ollama's `images` sit beside the text; they go first, as in the authors' examples.
        for b in ollama_imgs.into_iter().flatten() {
            imgs.push(image_bytes(b).map_err(|e| format!("messages[{i}].images: {e}"))?);
            content.push(json!({"type": "image"}));
        }
        match parts {
            Some(ps) => for (k, p) in ps.iter().enumerate() {
                if is_image_part(p) {
                    imgs.push(image_bytes(p).map_err(|e| format!("messages[{i}].content[{k}]: {e}"))?);
                    content.push(json!({"type": "image"}));
                } else {
                    let t = genopts::content_text(&Value::Array(vec![p.clone()])).map_err(|e| format!("messages[{i}].content[{k}]: {e}"))?;
                    content.push(json!({"type": "text", "text": t}));
                }
            },
            None => {
                let t = genopts::content_text(&m["content"]).map_err(|e| format!("messages[{i}]: {e}"))?;
                if !t.is_empty() { content.push(json!({"type": "text", "text": t})); }
            }
        }
        let mut m2 = m.clone();
        if let Some(o) = m2.as_object_mut() { o.remove("images"); }
        m2["content"] = Value::Array(content);
        out.push(m2);
    }
    Ok((out, imgs))
}

impl Engine {
    /// A chat prompt on a vision model: the template renders one `<|image_pad|>` per image, and that one
    /// token becomes the run of `n` the tower's output fills. `Ok(None)` = the conversation has no image
    /// (then it takes the text path unchanged).
    pub(crate) fn vision_prompt(&self, messages: &[Value], tools: Option<&[Value]>, kwargs: &serde_json::Map<String, Value>)
        -> Result<Option<(Vec<u32>, Arc<MmInput>)>, String>
    {
        let (msgs, imgs) = split_images(messages)?;
        if imgs.is_empty() { return Ok(None); }
        let Some(v) = &self.vision else {
            return Err(format!("{} does not read images; serve a vision model ({}) from its checkpoint directory",
                               self.name, VL_TYPES.join(", ")));
        };
        if imgs.len() > 1 {
            return Err(format!("{} images in this conversation: one image per request is served (the verified path)", imgs.len()));
        }
        let t = self.chat_template.as_ref().ok_or("this checkpoint has no chat template to place the image")?;
        let msgs: Vec<Value> = msgs.iter().map(crate::prepare_for_template).collect();
        let ids = self.encode_special(&t.render(&msgs, tools, true, kwargs)?);
        let at: Vec<usize> = ids.iter().enumerate().filter(|(_, x)| **x == v.img_tok).map(|(i, _)| i).collect();
        if at.len() != 1 {
            return Err(format!("the chat template placed {} image tokens for 1 image", at.len()));
        }
        let img = images::decode(&imgs[0])?;
        let (px, plan) = preprocess(&img, &v.pcfg, KEYS_DEFAULT)?;
        let n = plan.tokens(v.pcfg.merge);
        let mut out = Vec::with_capacity(ids.len() + n);
        out.extend_from_slice(&ids[..at[0]]);
        out.extend(std::iter::repeat_n(v.img_tok, n));
        out.extend_from_slice(&ids[at[0] + 1..]);
        Ok(Some((out, Arc::new(MmInput { px, gh: plan.grid_h, gw: plan.grid_w, start: at[0], n }))))
    }

    /// The prefill of an image prompt: the tower, the splice, the authors' mRoPE positions. Returns the
    /// last row's logits and `delta`, the offset every later position carries.
    pub(crate) fn vision_prefill(&self, ids: &[u32], mm: &MmInput, cache: &mut Cache) -> Result<(Vec<f32>, i64), String> {
        let (Some(v), Model::Dense(m)) = (&self.vision, &self.model) else { return Err("not a vision model".into()) };
        let row_w = 3 * v.pcfg.temporal_patch * v.pcfg.patch * v.pcfg.patch;
        let pxt = Tensor::from_vec(&self.ctx, &mm.px, &[mm.gh * mm.gw, row_w]);
        let (rows, deep) = match &v.tower {
            Tower::Qwen25(t) => (t.encode_patches(&pxt, mm.gh, mm.gw, &mut Vec::new())?, Vec::new()),
            Tower::Qwen3(t) => { let (p, d, _) = t.encode_patches(&pxt, mm.gh, mm.gw)?; (p, d) }
        };
        if rows.shape[0] != mm.n { return Err(format!("the tower gave {} rows for {} image tokens", rows.shape[0], mm.n)); }
        let types: Vec<u8> = ids.iter().map(|&x| (x == v.img_tok) as u8).collect();
        let ri = ferric_llama::qwen3vl_rope::rope_index(&types, &[(1, mm.gh, mm.gw)], v.pcfg.merge)?;
        let mut mrope: Vec<u32> = Vec::with_capacity(4 * ids.len());
        for c in [&ri.t, &ri.h, &ri.w] { mrope.extend(c.iter().map(|&x| x as u32)); }
        mrope.extend(std::iter::repeat_n(0u32, ids.len()));
        let x = m.splice_image_embeds(&m.embed_tokens(ids), mm.start, &rows);
        let h = m.forward_embeds_mm(&x, cache, &mrope, &deep, mm.start, &mut Vec::new());
        let last = h.narrow(0, ids.len() - 1, 1).contiguous();
        Ok((pollster::block_on(m.logits_from_normed(&last).to_vec()), ri.delta))
    }

    /// One decode step after an image. ⛔ The position is `index + delta`, NOT the cache index: an image
    /// of `n` tokens advances the sequence position by `max(h, w) / merge`, so the text after it sits
    /// behind its index. Rotating by the index is fluent and wrong (vl_ask's `decode_cachepos` control).
    pub(crate) fn vision_decode(&self, tok: u32, pos: i64, cache: &mut Cache) -> Vec<f32> {
        let Model::Dense(m) = &self.model else { unreachable!("vision models are dense") };
        let p = pos as u32;
        let h = m.forward_embeds_mm(&m.embed_tokens(&[tok]), cache, &[p, p, p, 0], &[], 0, &mut Vec::new());
        pollster::block_on(m.logits_from_normed(&h).to_vec())
    }
}

/// Whether a directory is a checkpoint this module serves.
pub(crate) fn vl_model_type(dir: &Path) -> Option<String> {
    let c: Value = serde_json::from_slice(&std::fs::read(dir.join("config.json")).ok()?).ok()?;
    let t = c["model_type"].as_str()?.to_string();
    VL_TYPES.contains(&t.as_str()).then_some(t)
}

/// The weight files' total size, parameter count and commonest dtype, from the safetensors headers.
pub(crate) fn weights_summary(dir: &Path) -> (u64, u64, String) {
    let (mut bytes, mut params) = (0u64, 0u64);
    let mut by: HashMap<String, u64> = HashMap::new();
    for e in std::fs::read_dir(dir).into_iter().flatten().flatten() {
        let p = e.path();
        if p.extension().is_none_or(|x| x != "safetensors") { continue; }
        bytes += std::fs::metadata(&p).map(|m| m.len()).unwrap_or(0);
        let Ok(mut f) = std::fs::File::open(&p) else { continue };
        use std::io::Read;
        let mut n = [0u8; 8];
        if f.read_exact(&mut n).is_err() { continue; }
        let mut h = vec![0u8; u64::from_le_bytes(n).min(1 << 26) as usize];
        if f.read_exact(&mut h).is_err() { continue; }
        let Ok(Value::Object(hdr)) = serde_json::from_slice::<Value>(&h) else { continue };
        for (k, v) in hdr {
            if k == "__metadata__" { continue; }
            let n: u64 = v["shape"].as_array().map(|s| s.iter().filter_map(|d| d.as_u64()).product()).unwrap_or(0);
            params += n;
            *by.entry(v["dtype"].as_str().unwrap_or("?").to_string()).or_default() += n;
        }
    }
    (bytes, params, by.into_iter().max_by_key(|(_, n)| *n).map(|(t, _)| t).unwrap_or_default())
}

/// What `/api/tags` and `/api/show` say about a checkpoint directory.
pub(crate) fn card(name: &str, dir: &Path) -> Option<ollama::Card> {
    let c: Value = serde_json::from_slice(&std::fs::read(dir.join("config.json")).ok()?).ok()?;
    let t = c.get("text_config").unwrap_or(&c);
    let (size, params, quant) = weights_summary(dir);
    Some(ollama::Card {
        name: name.to_string(), path: dir.to_string_lossy().into_owned(),
        arch: c["model_type"].as_str().unwrap_or("").to_string(), params, size, quant,
        modified: std::fs::metadata(dir.join("config.json")).and_then(|m| m.modified()).unwrap_or(std::time::UNIX_EPOCH),
        context: t["max_position_embeddings"].as_u64().or(c["max_position_embeddings"].as_u64()).unwrap_or(4096) as usize,
        embedding: false, template: chat_template_of(dir).unwrap_or_default(),
        dim: t["hidden_size"].as_u64().or(c["hidden_size"].as_u64()).unwrap_or(0) as usize,
    })
}

/// The authors' chat template: `chat_template.json`, else `chat_template.jinja`, else `tokenizer_config.json`.
fn chat_template_of(dir: &Path) -> Option<String> {
    let rd = |f: &str| std::fs::read(dir.join(f)).ok();
    if let Some(v) = rd("chat_template.json").and_then(|b| serde_json::from_slice::<Value>(&b).ok()) {
        if let Some(s) = v["chat_template"].as_str() { return Some(s.to_string()); }
    }
    if let Some(b) = rd("chat_template.jinja") { return String::from_utf8(b).ok(); }
    let tc: Value = serde_json::from_slice(&rd("tokenizer_config.json")?).ok()?;
    match &tc["chat_template"] {
        Value::String(s) => Some(s.clone()),
        Value::Array(a) => a.iter().find(|t| t["name"] == "default").and_then(|t| t["template"].as_str()).map(String::from),
        _ => None,
    }
}

impl Engine {
    /// Build an engine from the authors' checkpoint DIRECTORY: their safetensors (through `ferric_load::hf`),
    /// their `tokenizer.json` and their chat template, plus the vision tower.
    pub(crate) fn load_hf_in(shared: &Shared, dir: &str, name: String) -> Result<Engine, String> {
        let d = Path::new(dir);
        let mt = vl_model_type(d).ok_or_else(|| format!("{dir}: not a checkpoint of a type served here ({})", VL_TYPES.join(", ")))?;
        let ctx = shared.ctx.clone();
        let hf = ferric_load::hf::HfCheckpoint::open(d)?;
        let model = Qwen3::load(&ctx, &hf)?;
        let tj = std::fs::read(d.join("tokenizer.json")).map_err(|e| format!("tokenizer.json: {e}"))?;
        // Both served types are Qwen checkpoints: byte-level BPE with Qwen2's pre-tokenizer.
        let bpe = ferric_tokenizer::Bpe::from_tokenizer_json_with_pre(&tj, ferric_tokenizer::Pre::Qwen2)?;
        let tv: Value = serde_json::from_slice(&tj).map_err(|e| format!("tokenizer.json: {e}"))?;
        let mut tokens: Vec<String> = vec![String::new(); model.cfg.n_vocab];
        let put = |tokens: &mut Vec<String>, id: usize, s: &str| { if id >= tokens.len() { tokens.resize(id + 1, String::new()); } tokens[id] = s.to_string(); };
        for (t, id) in tv["model"]["vocab"].as_object().ok_or("tokenizer.json: no model.vocab")? {
            put(&mut tokens, id.as_u64().unwrap_or(0) as usize, t);
        }
        // Every added token is atomic to the authors' tokenizer, special or not (`<tool_call>` is not).
        let mut specials: Vec<(String, u32)> = Vec::new();
        for a in tv["added_tokens"].as_array().into_iter().flatten() {
            let (Some(c), Some(id)) = (a["content"].as_str(), a["id"].as_u64()) else { continue };
            put(&mut tokens, id as usize, c);
            specials.push((c.to_string(), id as u32));
        }
        specials.sort_by_key(|(s, _)| std::cmp::Reverse(s.len()));
        let id_of = |s: &str| specials.iter().find(|(t, _)| t == s).map(|(_, i)| *i);
        let cfg: Value = serde_json::from_slice(&std::fs::read(d.join("config.json")).map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
        let img_tok = cfg["image_token_id"].as_u64().map(|x| x as u32).or_else(|| id_of("<|image_pad|>"))
            .ok_or("config.json: no image_token_id")?;
        let gencfg: Value = std::fs::read(d.join("generation_config.json")).ok().and_then(|b| serde_json::from_slice(&b).ok()).unwrap_or(Value::Null);
        let mut eos: Vec<u32> = match &gencfg["eos_token_id"] {
            Value::Array(a) => a.iter().filter_map(|x| x.as_u64().map(|x| x as u32)).collect(),
            Value::Number(n) => n.as_u64().map(|x| vec![x as u32]).unwrap_or_default(),
            _ => Vec::new(),
        };
        for t in ["<|im_end|>", "<|endoftext|>"] { if let Some(e) = id_of(t) { if !eos.contains(&e) { eos.push(e); } } }
        let template = chat_template_of(d).ok_or("no chat template in the checkpoint")?;
        let tc: Value = std::fs::read(d.join("tokenizer_config.json")).ok().and_then(|b| serde_json::from_slice(&b).ok()).unwrap_or(Value::Null);
        let tok_s = |k: &str| tc[k].as_str().or_else(|| tc[k]["content"].as_str()).unwrap_or("").to_string();
        let chat_template = Some(template::ChatTemplate::compile(&template, &tok_s("bos_token"), &tok_s("eos_token"))?);
        let u2b = byte_decoder();
        let token_bytes: Vec<Option<Vec<u8>>> = tokens.iter().map(|t| {
            let mut b = Vec::with_capacity(t.len());
            for c in t.chars() { match u2b.get(&c) { Some(&x) => b.push(x), None => return None } }
            Some(b)
        }).collect();
        let tower = match mt.as_str() {
            "qwen2_5_vl" => Tower::Qwen25(ferric_llama::qwen25vl_vision::VisionTower::load(&ctx, dir)?),
            _ => Tower::Qwen3(ferric_llama::qwen3vl_vision::VisionTower::load(&ctx, dir)?),
        };
        let vision = Some(Vision { tower, pcfg: PreprocCfg::load(dir)?, img_tok });
        let card = card(&name, d).ok_or("config.json unreadable")?;
        let reasoning_markers = template.contains("<think>").then(|| ("<think>".to_string(), "</think>".to_string()));
        let spec_gate = std::sync::Mutex::new(crate::specgate::SpecGate::new(model.cfg.n_layer));
        let n_ctx = card.context;
        let (im_start, im_end) = (id_of("<|im_start|>"), id_of("<|im_end|>"));
        Ok(Engine {
            ctx, model: Model::Dense(model), aux: shared.aux.clone(), card, chat_template, reasoning_markers,
            energy: shared.energy.clone(), responses: shared.responses.clone(), metrics: shared.metrics.clone(),
            // Image prompts never use it (see the module docs); a text turn on this model may.
            prefix_cache: None, spec_gate, bpe, spm: None, add_space_prefix: false, tokens, u2b, im_start, im_end,
            bos_id: None, add_bos: false, eos_id: eos.first().copied(), add_eos: false, pooling: None, eos, name,
            token_bytes, specials, rstrip_after: Default::default(), template, prefix: std::cell::RefCell::new(None),
            n_ctx, vision, adapters: Vec::new(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn images_are_found_in_every_dialects_spelling_and_urls_are_refused() {
        let png = b"\x89PNG\r\n\x1a\nrest";
        let enc = "iVBORw0KGgpyZXN0"; // base64 of the eight PNG magic bytes + "rest"
        assert_eq!(b64(enc).unwrap(), png);
        for part in [json!({"type": "image_url", "image_url": {"url": format!("data:image/png;base64,{enc}")}}),
                     json!({"type": "image_url", "image_url": format!("data:image/png;base64,{enc}")}),
                     json!({"type": "input_image", "image_url": format!("data:image/png;base64,{enc}")}),
                     json!({"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": enc}})] {
            let (msgs, imgs) = split_images(&[json!({"role": "user", "content": [part, {"type": "text", "text": "what is this?"}]})]).unwrap();
            assert_eq!(imgs, vec![png.to_vec()]);
            assert_eq!(msgs[0]["content"], json!([{"type": "image"}, {"type": "text", "text": "what is this?"}]), "the bytes never reach the template");
        }
        let (msgs, imgs) = split_images(&[json!({"role": "user", "content": "what is this?", "images": [enc]})]).unwrap();
        assert_eq!((imgs.len(), &msgs[0]["content"]), (1, &json!([{"type": "image"}, {"type": "text", "text": "what is this?"}])));
        assert!(msgs[0].get("images").is_none());
        let (msgs, imgs) = split_images(&[json!({"role": "user", "content": "plain"})]).unwrap();
        assert!(imgs.is_empty() && msgs[0]["content"] == json!("plain"), "a text-only message is left exactly as it was");
        let e = split_images(&[json!({"role": "user", "content": [{"type": "image_url", "image_url": {"url": "https://x/y.png"}}]})]).unwrap_err();
        assert!(e.contains("not fetched"), "{e}");
    }
}
