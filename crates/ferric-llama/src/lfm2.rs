//! **Liquid AI LFM2 / LFM2.5** — the conv/attention hybrid, with real incremental state.
//!
//! Both mixers are recurrent in the sense that matters for decode: attention carries a KV cache, and
//! the short-conv carries `l_cache - 1` timesteps of the *pre-convolution* gated signal. Carrying
//! both is what makes a 128k-context model actually usable — re-running the prefix per token is
//! O(n²) and turns a long context into a number on a spec sheet.
//!
//! Per-token cost here is O(1) in the conv layers (a 2-row state, independent of sequence length)
//! and O(s) in the 8 attention layers, against O(n) layers × O(n) tokens for a prefix re-run.
//!
//! ```text
//! conv block:  BCx = in_proj(norm(h)); B,C,x = chunks 0,1,2
//!              bx  = B ⊙ x                      (no activation anywhere in this block)
//!              y   = C ⊙ conv(concat(state, bx))[-t:] ; state = last L-1 rows of concat(state, bx)
//!              out = out_proj(y)
//! attn block:  q,k,v → per-head QK RMSNorm → RoPE at the ABSOLUTE position → KV cache → GQA
//! every block: + SwiGLU FFN, both residual
//! final:       token_embd_norm (the FINAL norm despite its name) → tied head
//! ```
use ferric_core::Context;
use ferric_gguf::{GgufSource, Meta};
use ferric_tensor::kvquant::{KvStore, KvqFmt};
use ferric_tensor::{append2, nn, KvBuf, QMatrix, Tensor};
use std::sync::Arc;

pub struct Cfg {
    pub n_layer: usize, pub d: usize, pub n_head: usize, pub head_dim: usize,
    pub n_vocab: usize, pub eps: f32, pub rope_base: f32, pub conv_l: usize,
    /// Per-block KV head count; `0` marks a short-conv block. Read from the ARRAY, never a scalar.
    pub kv: Vec<usize>,
}

enum Mixer {
    Conv { in_proj: QMatrix, conv: Tensor, out_proj: QMatrix },
    Attn { q: QMatrix, k: QMatrix, v: QMatrix, o: QMatrix, q_norm: Tensor, k_norm: Tensor, n_kv: usize },
}

struct Block {
    norm: Tensor, mixer: Mixer, ffn_norm: Tensor,
    gate: QMatrix, up: QMatrix, down: QMatrix,
}

/// Per-sequence decode state: KV for attention blocks, a rolling window for conv blocks.
pub struct Cache {
    pub pos: usize,
    kv: Vec<(KvBuf, KvBuf)>,
    /// Block-quantized twin of `kv`. **Empty unless KV quantization is on**, and when it is on `kv` is
    /// the empty one — holding both would spend the memory the quantization exists to save.
    q: Vec<(KvStore, KvStore)>,
    fmt: Option<KvqFmt>,
    /// `[l_cache-1, d]` of the PRE-convolution gated signal `B⊙x` — the reference stores the signal,
    /// not the conv output, and storing the wrong one produces plausible drift rather than an error.
    ///
    /// Never quantized: it is bounded by `l_cache`, not by context length. See [`Cache::with_kvq`].
    conv: Vec<Option<Tensor>>,
}

impl Cache {
    /// Default (f32) cache, unless `FERRIC_KVQ` asks otherwise. See [`Cache::with_kvq`].
    pub fn new(cfg: &Cfg) -> Cache { Cache::with_kvq(cfg, crate::qwen3::kvq_from_env()) }

    /// A cache whose attention K/V rows are stored as `fmt` quantization blocks — `None` for f32.
    ///
    /// **Opt-in, and it must stay opt-in**: this trades accuracy for memory, so it is the caller's
    /// choice and never a silent change to an existing run.
    ///
    /// ⚠ **The conv state is NOT quantized and is not counted by [`Cache::kv_bytes`].** LFM2 carries
    /// two kinds of per-sequence state and only one of them is KV: the conv blocks hold an
    /// `[l_cache-1, d]` rolling window of the pre-convolution signal, which is bounded by `l_cache`
    /// (3 on the 1.2B) rather than by context length. It does not grow with the sequence, so it is not
    /// what a long context spends memory on, and quantizing a 2-row window would trade real accuracy
    /// for nothing.
    pub fn with_kvq(cfg: &Cfg, fmt: Option<KvqFmt>) -> Cache {
        let conv = (0..cfg.n_layer).map(|_| None).collect();
        match fmt {
            None => Cache {
                pos: 0,
                kv: (0..cfg.n_layer).map(|_| (KvBuf::default(), KvBuf::default())).collect(),
                q: Vec::new(),
                fmt: None,
                conv,
            },
            Some(f) => Cache {
                pos: 0,
                kv: Vec::new(),
                // Sized `n_layer`, not `n_attn`: every index here is a BLOCK index, and a compacted
                // vector would need an attention-block renumbering that nothing else in this file does.
                // The conv blocks' slots stay empty and cost nothing until appended to.
                // K may be token-GROUPED (FERRIC_KVQ_K_AXIS=grouped, via the same predicate the
                // dense runtime uses so the env var means one thing everywhere); V stays per-block —
                // the measurement is asymmetric and grouping V buys ~nothing for a staging tail.
                // Motivation on THIS runtime: block-K q4_0 diverged from f32 at generated token 3.
                q: (0..cfg.n_layer).map(|_| (crate::qwen3::k_store_from_env(f), KvStore::block(f))).collect(),
                fmt: Some(f),
                conv,
            },
        }
    }

    /// Explicit KV configuration — the constructor a browser must use: `Cache::new`/`with_kvq` read
    /// the axis from the environment, and a wasm32 tab HAS no environment, so the env path silently
    /// pins a tab to block-K. Same reasoning as `qwen3::Cache::with_kv_config`.
    pub fn with_kv_config(cfg: &Cfg, fmt: Option<KvqFmt>, grouped_k: bool) -> Cache {
        let mut c = Cache::with_kvq(cfg, fmt);
        if let Some(f) = fmt {
            if grouped_k {
                c.q = (0..cfg.n_layer).map(|_| (KvStore::grouped(f), KvStore::block(f))).collect();
            }
        }
        c
    }

    /// The KV-cache quantization format in force, or `None` for f32.
    pub fn kvq_fmt(&self) -> Option<KvqFmt> { self.fmt }

    /// **Device bytes the attention K/V caches occupy right now.** Excludes the conv state, which is
    /// bounded by `l_cache` and does not grow with context — see [`Cache::with_kvq`].
    pub fn kv_bytes(&self) -> usize {
        match self.fmt {
            None => self.kv.iter().map(|(k, v)| (k.len() * k.width() + v.len() * v.width()) * 4).sum(),
            Some(_) => self.q.iter().map(|(k, v)| k.bytes() + v.bytes()).sum(),
        }
    }

    /// Live K/V bytes, ignoring allocated slack — what the FORMAT buys, as opposed to what the device
    /// is holding. `kv_bytes()` counts allocated capacity and both stores grow by doubling, so
    /// `kv_bytes() / kv_f32_bytes()` understates the format ratio by up to 2x. See
    /// [`ferric_tensor::kvquant::QKvCache::live_bytes`].
    pub fn kv_live_bytes(&self) -> usize {
        match self.fmt {
            None => self.kv_bytes(),
            Some(_) => self.q.iter().map(|(k, v)| k.live_bytes() + v.live_bytes()).sum(),
        }
    }

    /// What the same live rows would cost as f32 — `kv_bytes()`'s denominator, on either kind of cache.
    pub fn kv_f32_bytes(&self) -> usize {
        match self.fmt {
            None => self.kv_bytes(),
            Some(_) => self.q.iter().map(|(k, v)| k.f32_bytes() + v.f32_bytes()).sum(),
        }
    }

}

pub struct Lfm2 {
    ctx: Arc<Context>,
    pub cfg: Cfg,
    blocks: Vec<Block>,
    out_norm: Tensor,
    head: QMatrix,
    embd_ty: u32,
    embd_raw: Vec<u8>,
}

impl Lfm2 {
    pub fn load(ctx: &Arc<Context>, g: &impl GgufSource) -> Result<Lfm2, String> {
        let md = g.metadata();
        let u = |k: &str| match md.get(&format!("lfm2.{k}")) { Some(Meta::U(v)) => Ok(*v as usize), _ => Err(format!("missing lfm2.{k}")) };
        let f = |k: &str| match md.get(&format!("lfm2.{k}")) { Some(Meta::F(v)) => Ok(*v as f32), _ => Err(format!("missing lfm2.{k}")) };
        let n_layer = u("block_count")?;
        let d = u("embedding_length")?;
        let n_head = u("attention.head_count")?;
        let eps = f("attention.layer_norm_rms_epsilon")?;
        let rope_base = f("rope.freq_base")?;
        let conv_l = u("shortconv.l_cache")?;
        let kv: Vec<usize> = match md.get("lfm2.attention.head_count_kv") {
            Some(Meta::Arr(v)) => v.iter().map(|m| match m { Meta::U(x) => *x as usize, Meta::I(x) => *x as usize, _ => 0 }).collect(),
            Some(Meta::U(v)) => vec![*v as usize; n_layer],
            _ => return Err("no lfm2.attention.head_count_kv".into()),
        };
        if kv.len() != n_layer { return Err(format!("schedule covers {} of {n_layer} blocks", kv.len())); }
        let head_dim = d / n_head;
        let n_vocab = g.tensor("token_embd.weight").ok_or("no token_embd")?.dims[1] as usize;

        let qm = |name: &str| -> QMatrix {
            let t = g.tensor(name).unwrap_or_else(|| panic!("missing {name}"));
            let (ty, rows, cols) = (t.ggml_type, t.dims[1] as usize, t.dims[0] as usize);
            if QMatrix::block_bytes(ty).is_some() {
                QMatrix::from_bytes(ctx, &g.raw(name).unwrap(), ty, rows, cols).unwrap()
            } else {
                QMatrix::from_dense(ctx, &g.dequant(name).unwrap(), rows, cols)
            }
        };
        let ft = |name: &str, shape: &[usize]| Tensor::from_vec(ctx, &g.dequant(name).unwrap(), shape);

        let blocks = (0..n_layer).map(|il| {
            let b = |s: &str| format!("blk.{il}.{s}");
            let n_kv = kv[il];
            let mixer = if n_kv > 0 {
                Mixer::Attn {
                    q: qm(&b("attn_q.weight")), k: qm(&b("attn_k.weight")),
                    v: qm(&b("attn_v.weight")), o: qm(&b("attn_output.weight")),
                    q_norm: ft(&b("attn_q_norm.weight"), &[head_dim]),
                    k_norm: ft(&b("attn_k_norm.weight"), &[head_dim]),
                    n_kv,
                }
            } else {
                // GGUF [L, d] dequantises row-major to [d, L] = the [C, L] the conv kernel wants.
                Mixer::Conv {
                    in_proj: qm(&b("shortconv.in_proj.weight")),
                    conv: ft(&b("shortconv.conv.weight"), &[d, conv_l]),
                    out_proj: qm(&b("shortconv.out_proj.weight")),
                }
            };
            Block {
                norm: ft(&b("attn_norm.weight"), &[d]), mixer,
                ffn_norm: ft(&b("ffn_norm.weight"), &[d]),
                gate: qm(&b("ffn_gate.weight")), up: qm(&b("ffn_up.weight")), down: qm(&b("ffn_down.weight")),
            }
        }).collect();

        Ok(Lfm2 {
            ctx: ctx.clone(),
            cfg: Cfg { n_layer, d, n_head, head_dim, n_vocab, eps, rope_base, conv_l, kv },
            blocks,
            // `token_embd_norm` is the FINAL norm, not a norm on the embeddings — llama.cpp maps it
            // through a dedicated enum whose comment reads "fix for wrong tensor name".
            out_norm: ft("token_embd_norm.weight", &[d]),
            head: if g.tensor("output.weight").is_some() { qm("output.weight") } else { qm("token_embd.weight") },
            embd_ty: g.tensor("token_embd.weight").unwrap().ggml_type,
            embd_raw: g.raw("token_embd.weight")?,
        })
    }

    fn embed(&self, tokens: &[u32]) -> Tensor {
        let d = self.cfg.d;
        let row = ferric_gguf::type_size(self.embd_ty, d).expect("embd type");
        let mut v = Vec::with_capacity(tokens.len() * d);
        for &t in tokens {
            let o = t as usize * row;
            v.extend(ferric_gguf::deq_raw(&self.embd_raw[o..o + row], d, self.embd_ty).expect("embed row"));
        }
        Tensor::from_vec(&self.ctx, &v, &[tokens.len(), d])
    }

    /// Feed `tokens`, carrying attention KV and conv state in `cache`. Prompt once, then one token
    /// per step — the per-token cost does not grow with how much has already been fed.
    pub fn forward(&self, tokens: &[u32], cache: &mut Cache) -> Tensor {
        self.forward_hidden_cached(tokens, cache).matmul_q(&self.head)
    }

    /// The final normed hidden state, before the output head — what LAST-pooling embedding
    /// references pool over. Split out rather than duplicated so the served path and the embedding
    /// path cannot drift: `forward` is this plus one matmul.
    pub fn forward_hidden_cached(&self, tokens: &[u32], cache: &mut Cache) -> Tensor {
        let (d, n_head, head_dim, eps) = (self.cfg.d, self.cfg.n_head, self.cfg.head_dim, self.cfg.eps);
        let t = tokens.len();
        let pos = cache.pos;
        let mut x = self.embed(tokens);

        for (il, blk) in self.blocks.iter().enumerate() {
            let h = x.rmsnorm(&blk.norm, eps);
            let op = match &blk.mixer {
                Mixer::Attn { q, k, v, o, q_norm, k_norm, n_kv } => {
                    // RoPE at the ABSOLUTE position, so a cached decode step rotates by `pos`, not 0.
                    let qh = h.matmul_q(q).reshape(&[t * n_head, head_dim]).rmsnorm(q_norm, eps)
                        .reshape(&[t, n_head * head_dim]).rope(n_head, head_dim, self.cfg.rope_base, pos);
                    let kh = h.matmul_q(k).reshape(&[t * n_kv, head_dim]).rmsnorm(k_norm, eps)
                        .reshape(&[t, n_kv * head_dim]).rope(*n_kv, head_dim, self.cfg.rope_base, pos);
                    let vh = h.matmul_q(v);
                    // One `&mut` to the tuple, then split into its two disjoint fields: the borrow
                    // checker cannot see that .0 and .1 do not alias when reached through two separate
                    // index expressions.
                    // The quantized arm is the same contract with a different store: `QKvCache::append`
                    // quantizes exactly the rows handed to it and writes them at their flat block
                    // offset, so a one-token step touches `width/32` blocks and re-reads nothing, and
                    // `dequantize` then materialises this block's whole history as a transient f32
                    // window that attention reads and the block drops. Only ONE block's K and V exist
                    // in f32 at a time, against all `n_attn` of them persistently before.
                    let (kc, vc) = match cache.fmt {
                        None => { let e = &mut cache.kv[il]; append2(&self.ctx, &mut e.0, &kh, &mut e.1, &vh) }
                        Some(_) => {
                            let e = &mut cache.q[il];
                            e.0.append(&self.ctx, &kh);
                            e.1.append(&self.ctx, &vh);
                            (e.0.dequantize(&self.ctx), e.1.dequantize(&self.ctx))
                        }
                    };
                    let att = if t == 1 {
                        nn::decode_attention(&qh, &kc, &vc, n_head, *n_kv, 0.0)
                    } else {
                        // Serves full prefill AND a chunk against a longer cache.
                        nn::chunked_attention(&qh, &kc, &vc, n_head, *n_kv, 0.0)
                    };
                    att.matmul_q(o)
                }
                Mixer::Conv { in_proj, conv, out_proj } => {
                    let bcx = h.matmul_q(in_proj);
                    let b = bcx.narrow(1, 0, d).contiguous();
                    let c = bcx.narrow(1, d, d).contiguous();
                    let xx = bcx.narrow(1, 2 * d, d).contiguous();
                    let bx = b.mul(&xx);                       // no activation in this block
                    let keep = self.cfg.conv_l - 1;
                    // Prepend the carried window. The conv is causal with zero padding, so rows
                    // [0, keep) of the concatenated signal see a wrong (zero) history — they are
                    // discarded; only the `t` new rows are kept, and those see the real history.
                    let full = match &cache.conv[il] {
                        Some(st) => st.cat(&bx, 0),
                        None => Tensor::from_vec(&self.ctx, &vec![0f32; keep * d], &[keep, d]).cat(&bx, 0),
                    };
                    let n = full.shape[0];
                    let conv_out = full.depthwise_conv1d_causal(conv, self.cfg.conv_l).narrow(0, n - t, t).contiguous();
                    // Carry the last `keep` rows of the PRE-conv signal, matching the reference.
                    cache.conv[il] = Some(full.narrow(0, n - keep, keep).contiguous());
                    c.mul(&conv_out).matmul_q(out_proj)
                }
            };
            x = x.add(&op);
            let hn = x.rmsnorm(&blk.ffn_norm, eps);
            x = x.add(&hn.matmul_q(&blk.gate).silu().mul(&hn.matmul_q(&blk.up)).matmul_q(&blk.down));
        }
        cache.pos += t;
        x.rmsnorm(&self.out_norm, eps)
    }

    /// One block's mixer for **N independent sequences, one token each**.
    ///
    /// Both arms have the same shape: the projections run ONCE over `[N, d]` — that is the entire win,
    /// because decode is weight-streaming and a projection's cost is reading its weights, not the row it
    /// multiplies. What stays a loop is the *state*, and LFM2 has two kinds of it:
    ///
    ///   - **attention**: sequence `i` attends its own KV history at its own position, and those
    ///     histories differ in length, so there is no one call that serves all N.
    ///   - **conv**: sequence `i` carries its own `l_cache - 1` rolling window of the pre-conv signal, so
    ///     the depthwise conv reads that row's window and no other. Two rows at the same position still
    ///     have different windows — the window is history, not position — so this loop cannot be folded
    ///     away by aligning positions either.
    ///
    /// Neither loop touches a weight, which is why leaving them per-sequence costs nothing that matters:
    /// they move a few KiB of per-sequence state, against ~700 MB of weights now read once instead of N
    /// times.
    fn mixer_batch(&self, h: &Tensor, mixer: &Mixer, caches: &mut [&mut Cache], il: usize, positions: &[u32]) -> Tensor {
        let (d, n_head, head_dim, eps) = (self.cfg.d, self.cfg.n_head, self.cfg.head_dim, self.cfg.eps);
        let n = h.shape[0];
        debug_assert_eq!(n, caches.len(), "one row per sequence");
        match mixer {
            Mixer::Attn { q, k, v, o, q_norm, k_norm, n_kv } => {
                // `rope_at_ex(.., None, false)` is the SAME kernel and the SAME pairing the solo path's
                // `rope` reaches (`rope_full` with interleaved=false and no per-dimension scale); only
                // where the angle's position comes from differs — one position per row instead of
                // `offset + i`. LFM2 declares no `rope_freqs` and no NORM pairing, so there is no scaled
                // or interleaved variant here for the two paths to disagree about. If one is ever added
                // it must be added to BOTH: the precedent is `rope_at` shipping NEOX-only-and-unscaled,
                // which made batched decode diverge from the solo path with no error and fluent output.
                let rope_rows = |x: Tensor, heads: usize|
                    x.rope_at_ex(heads, head_dim, self.cfg.rope_base, positions, None, false);
                let qh = rope_rows(h.matmul_q(q).reshape(&[n * n_head, head_dim]).rmsnorm(q_norm, eps)
                                    .reshape(&[n, n_head * head_dim]), n_head);
                let kh = rope_rows(h.matmul_q(k).reshape(&[n * n_kv, head_dim]).rmsnorm(k_norm, eps)
                                    .reshape(&[n, n_kv * head_dim]), *n_kv);
                let vh = h.matmul_q(v);
                let mut outs = Vec::with_capacity(n);
                for (i, c) in caches.iter_mut().enumerate() {
                    // Row `i` of the batched projection goes into cache `i` and nowhere else. This narrow
                    // is the only thing keeping the sequences apart; getting the index wrong appends one
                    // sequence's key to another's history, which reads as fluent drift, never an error.
                    let (ki, vi) = (kh.narrow(0, i, 1), vh.narrow(0, i, 1));
                    // Both stores index by SEQUENCE then layer, so batching needs no fork — each row
                    // appends to its own cache exactly as the solo path does. `narrow` on dim 0 keeps
                    // `strides[1] == 1`, so `QKvCache::append` reads it in place without packing.
                    let (kc, vc) = match c.fmt {
                        None => { let e = &mut c.kv[il]; append2(&self.ctx, &mut e.0, &ki, &mut e.1, &vi) }
                        Some(_) => {
                            let e = &mut c.q[il];
                            e.0.append(&self.ctx, &ki);
                            e.1.append(&self.ctx, &vi);
                            (e.0.dequantize(&self.ctx), e.1.dequantize(&self.ctx))
                        }
                    };
                    // Always the single-query kernel: batching advances every sequence by exactly one
                    // token, so `t == 1` per row by construction even though the batch has N rows.
                    let qi = qh.narrow(0, i, 1).contiguous();
                    outs.push(nn::decode_attention(&qi, &kc, &vc, n_head, *n_kv, 0.0));
                }
                let att = outs.iter().skip(1).fold(outs[0].clone(), |a, t| a.cat(t, 0));
                att.matmul_q(o)
            }
            Mixer::Conv { in_proj, conv, out_proj } => {
                let bcx = h.matmul_q(in_proj);                   // <- batched: the win
                let b = bcx.narrow(1, 0, d).contiguous();
                let c = bcx.narrow(1, d, d).contiguous();
                let xx = bcx.narrow(1, 2 * d, d).contiguous();
                let bx = b.mul(&xx);                             // no activation in this block
                let keep = self.cfg.conv_l - 1;
                let mut outs = Vec::with_capacity(n);
                for (i, cache) in caches.iter_mut().enumerate() {
                    // `full` is exactly `conv_l` rows: this row's carried window then this row's new
                    // signal. The conv is causal, so its LAST row — the only one kept — sees precisely
                    // those `conv_l` timesteps of THIS sequence. A batch-wide concatenation would let
                    // row `i`'s output reach back into row `i-1`'s tail instead.
                    let row = bx.narrow(0, i, 1);
                    let full = match &cache.conv[il] {
                        Some(st) => st.cat(&row, 0),
                        // Only on a cold cache (no prefill yet); the zero window is what the causal conv
                        // would have padded with anyway.
                        None => Tensor::from_vec(&self.ctx, &vec![0f32; keep * d], &[keep, d]).cat(&row, 0),
                    };
                    let nf = full.shape[0];
                    outs.push(full.depthwise_conv1d_causal(conv, self.cfg.conv_l).narrow(0, nf - 1, 1).contiguous());
                    // Carry the PRE-conv signal, per sequence, exactly as the solo path does. Storing the
                    // conv OUTPUT here — or one shared window for the whole batch — drifts plausibly
                    // instead of failing.
                    cache.conv[il] = Some(full.narrow(0, nf - keep, keep).contiguous());
                }
                let conv_out = outs.iter().skip(1).fold(outs[0].clone(), |a, t| a.cat(t, 0));
                c.mul(&conv_out).matmul_q(out_proj)
            }
        }
    }

    /// **Batched decode** — advance N independent sequences by ONE token each, in one forward pass.
    ///
    /// `tokens[i]` is the next token for `caches[i]`. Returns `[N, n_vocab]` logits; row `i` belongs to
    /// sequence `i`. The sequences need not be at the same position or the same length.
    ///
    /// Row `i` is **token-identical** to decoding sequence `i` alone through [`Self::forward`]. Batching
    /// changes only how the work is scheduled, never the result — and that has to be checked by direct
    /// comparison, because the failure has no symptom: a path that crossed sequences (one shared RoPE
    /// position, a conv window read from the wrong row, KV appended to the wrong cache) still returns
    /// finite logits and still writes fluent text. `examples/lfm2_batched_decode.rs` re-runs every
    /// sequence solo and compares token ids.
    pub fn forward_batch(&self, tokens: &[u32], caches: &mut [&mut Cache]) -> Tensor {
        assert_eq!(tokens.len(), caches.len(), "forward_batch: one token per sequence");
        assert!(!tokens.is_empty(), "forward_batch needs at least one sequence");
        // Every cache in a batch must agree on representation: `mixer_batch` branches on `c.fmt` per
        // sequence, so a mixed batch is not wrong, but it is a sign the caller built them from
        // different configs and the memory accounting downstream would be meaningless.
        assert!(caches.windows(2).all(|w| w[0].fmt == w[1].fmt),
                "forward_batch got a mix of quantized and f32 caches in one batch: {:?}",
                caches.iter().map(|c| c.fmt).collect::<Vec<_>>());
        let eps = self.cfg.eps;
        // Snapshot every row's absolute position ONCE, before any layer runs. Each row ropes at its own
        // `cache.pos`, and `pos` is advanced once at the end — reading it inside the layer loop would
        // give the same answer today and silently stop doing so the moment a layer touched the cache.
        let positions: Vec<u32> = caches.iter().map(|c| c.pos as u32).collect();
        let mut x = self.embed(tokens);
        for (il, blk) in self.blocks.iter().enumerate() {
            let h = x.rmsnorm(&blk.norm, eps);
            let op = self.mixer_batch(&h, &blk.mixer, caches, il, &positions);
            x = x.add(&op);
            let hn = x.rmsnorm(&blk.ffn_norm, eps);
            // The FFN is the largest weight read in the block and it batches with no per-row state at
            // all: rmsnorm, SwiGLU and the three projections are row-independent.
            x = x.add(&hn.matmul_q(&blk.gate).silu().mul(&hn.matmul_q(&blk.up)).matmul_q(&blk.down));
        }
        for c in caches.iter_mut() { c.pos += 1; }
        x.rmsnorm(&self.out_norm, eps).matmul_q(&self.head)
    }
}

#[cfg(test)]
mod kvq_tests {
    use super::*;

    /// 16 blocks in LFM2.5-1.2B's shape: 10 conv, 6 attention.
    fn cfg() -> Cfg {
        let kv: Vec<usize> = (0..16).map(|i| if i % 3 == 2 { 8 } else { 0 }).collect();
        Cfg { n_layer: 16, d: 2048, n_head: 32, head_dim: 64, n_vocab: 65536, eps: 1e-5,
              rope_base: 1e6, conv_l: 3, kv }
    }

    /// The two stores are never both populated, and the quantized one is indexed by BLOCK.
    ///
    /// LFM2 is the case where a "compact the vector to attention blocks only" optimisation is
    /// tempting and wrong: every index in this file is a block index, so a compacted vector needs a
    /// renumbering nothing else does. The empty conv slots cost nothing until appended to.
    #[test]
    fn a_quantized_cache_holds_no_f32_rows_and_is_indexed_by_block() {
        let c = cfg();
        let n_attn = c.kv.iter().filter(|&&k| k > 0).count();
        assert_eq!(n_attn, 5, "fixture sanity: the schedule must actually contain attention blocks");

        let f = Cache::with_kvq(&c, None);
        assert!(f.kvq_fmt().is_none());
        assert_eq!(f.kv.len(), c.n_layer);
        assert!(f.q.is_empty(), "an f32 cache must not allocate the quantized twin");

        for fmt in KvqFmt::ALL {
            let q = Cache::with_kvq(&c, Some(fmt));
            assert_eq!(q.kvq_fmt(), Some(fmt));
            assert!(q.kv.is_empty(), "{}: the f32 vector must be EMPTY, not merely unused — anything \
                                      reading it gets zero rows and no error", fmt.name());
            assert_eq!(q.q.len(), c.n_layer,
                       "{}: one slot per BLOCK, not per attention block", fmt.name());
            // Both kinds carry the conv state, and it is never quantized: bounded by l_cache, not by
            // context, so it is not what a long context spends memory on.
            assert_eq!(q.conv.len(), c.n_layer, "{}: conv state is per block on both paths", fmt.name());
            assert_eq!(q.kv_bytes(), 0, "{}: an untouched cache costs nothing", fmt.name());
        }
    }
}
