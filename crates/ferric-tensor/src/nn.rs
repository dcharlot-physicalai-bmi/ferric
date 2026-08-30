//! Transformer building blocks expressed ON the general tensor runtime. The point of unification:
//! attention is not a bespoke kernel — it's `reshape → batched matmul → softmax → matmul`, i.e. the
//! general ops. RMSNorm/softmax/RoPE are fused fast-paths (methods on `Tensor`) but produce exactly
//! what composing primitives would. One substrate; the model is an expression in it.

use crate::Tensor;

/// Linear y = x·W in the [in,out] weight convention (no bias) — just a matmul.
pub fn linear(x: &Tensor, w: &Tensor) -> Tensor { x.matmul(w) }

/// Linear in the HF convention: W is stored [out, in]; y = x·Wᵀ (direct, no transpose materialized).
pub fn linear_hf(x: &Tensor, w: &Tensor) -> Tensor { x.matmul_bt(w) }

/// Weight-quantized HF linear: W is a per-row int4/int8 [out,in]; y = x·Wᵀ, W dequantized on the fly.
pub fn linear_hf_q(x: &Tensor, w: &crate::QRow) -> Tensor { x.matmul_qweight(w) }

/// Factored (low-rank / 2-core tensor-network) HF linear. Instead of one dense `W` [out,in], the
/// layer carries two cores `u` [out,r] and `v` [r,in] with `W ≈ u·v`. Computed as `y = (x·vᵀ)·uᵀ`
/// on the SAME `matmul_bt` GPU kernel the dense path uses — so the saving is end-to-end, not a
/// standalone kernel: the dense layer touches `out·in` weights and multiplies, the factored layer
/// touches `r·(out+in)` of each. At r ≪ min(out,in) that is the compression ratio in both memory
/// and MACs, on whatever backend the runtime is already on. Correct only when the layer is (near)
/// low-rank — which, per the topic's bench 1, means TRAINING the factored form, not squeezing a
/// dense one. No intermediate is materialized beyond the [rows,r] bottleneck activation.
pub fn linear_factored(x: &Tensor, u: &Tensor, v: &Tensor) -> Tensor {
    x.matmul_bt(v).matmul_bt(u)
}

/// Factored HF linear with the activation fused into the OUTER projection's epilogue (one extra
/// kernel over `linear_factored`, no separate activation pass). The inner projection stays linear
/// (the bottleneck is not a nonlinearity); the outer `u` matmul applies `act` in its epilogue —
/// act: 0 identity, 1 relu, 2 silu, 3 gelu, 4 sigmoid. This is the drop-in for a factored MLP
/// hidden layer or gate: `silu(x·Wᵀ)` becomes `linear_factored_act(x, u, v, 2)`.
pub fn linear_factored_act(x: &Tensor, u: &Tensor, v: &Tensor, act: u32) -> Tensor {
    x.matmul_bt(v).matmul_bt_act(u, act)
}

/// Causal multi-head attention with grouped-query attention, composed from general ops (+ fused
/// softmax). q is [T, n_heads·dh]; k/v are [T, n_kv_heads·dh]. Returns [T, n_heads·dh].
pub fn causal_attention(q: &Tensor, k: &Tensor, v: &Tensor, n_heads: usize, n_kv_heads: usize, softcap: f32) -> Tensor {
    let t = q.shape[0];
    let d = q.shape[1];
    let dh = d / n_heads;
    let g = n_heads / n_kv_heads;
    let scale = 1.0 / (dh as f32).sqrt();
    let qh = q.reshape(&[t, n_heads, dh]).permute(&[1, 0, 2]).contiguous(); // [nh, T, dh]
    // K/V: [T, nkv·dh] → [nkv, T, dh] → repeat each kv head g times → [nh, T, dh]
    let kv_heads = |x: &Tensor| {
        let hx = x.reshape(&[t, n_kv_heads, dh]).permute(&[1, 0, 2]).contiguous(); // [nkv, T, dh]
        hx.reshape(&[n_kv_heads, 1, t, dh]).broadcast_to(&[n_kv_heads, g, t, dh]).reshape(&[n_heads, t, dh])
    };
    let (kh, vh) = (kv_heads(k), kv_heads(v));
    let scores = softcapped(qh.matmul(&kh.transpose(2, 1)).mul(&q.scalar(scale)), softcap); // [nh, T, T]
    let probs = scores.add(&causal_mask(q, t)).softmax(2);             // masked softmax over keys
    let ctx = probs.matmul(&vh);                                       // [nh, T, dh]
    ctx.permute(&[1, 0, 2]).reshape(&[t, d])
}

/// **Softmax with a learnable attention sink.** `scores` is `[n_heads, T, S]`; `sinks` is one raw
/// scalar per head, `[n_heads]`.
///
/// The sink is one extra logit that participates in the max and adds one term to the denominator,
/// and then contributes no value vector. So each row's probabilities sum to strictly LESS than one:
///
/// ```text
///   M   = max( max_j l_j , s_h )
///   Z   = Σ_j exp(l_j − M) + exp(s_h − M)
///   p_j = exp(l_j − M) / Z            with   Σ_j p_j = 1 − exp(s_h − M)/Z  <  1
/// ```
///
/// ⚠ **Do not renormalise.** The missing mass is the mechanism: it is how a head declines to
/// attend to anything, and restoring it to one deletes the feature while leaving every shape,
/// every sum-to-one intuition and every downstream assert intact.
///
/// ⚠ **`s_h` is raw.** It is not multiplied by the attention scale, not masked, not position- or
/// token-dependent, and never rotated. At the usual init of `0.0` this is exactly the classic
/// "+1 in the denominator".
///
/// This is implemented as "append the sink as one more key column, softmax, discard that column",
/// which is not an approximation of the definition above — it IS the definition, and it reuses the
/// existing numerically-stable softmax rather than adding a second kernel that would then have to
/// be kept bit-identical to the first on every fabric.
pub fn softmax_with_sinks(scores: &Tensor, sinks: &Tensor) -> Tensor {
    let (nh, t, s) = (scores.shape[0], scores.shape[1], scores.shape[2]);
    assert_eq!(sinks.numel(), nh, "one sink per head: {} heads, {} sinks", nh, sinks.numel());
    let col = sinks.reshape(&[nh, 1, 1]).broadcast_to(&[nh, t, 1]).contiguous();
    scores.cat(&col, 2).softmax(2).narrow(2, 0, s).contiguous()
}

/// [`causal_attention`] with a per-head learnable sink. `sinks` is `[n_heads]`.
///
/// A head whose sink is large and positive shrinks its whole output toward zero; a large negative
/// one recovers ordinary attention exactly. Both limits are checked in this module's tests, because
/// "it produces plausible numbers" is what a wrong sink also does.
pub fn causal_attention_sinks(q: &Tensor, k: &Tensor, v: &Tensor, n_heads: usize, n_kv_heads: usize,
                              sinks: &Tensor, softcap: f32) -> Tensor {
    let t = q.shape[0];
    let d = q.shape[1];
    let dh = d / n_heads;
    let g = n_heads / n_kv_heads;
    let scale = 1.0 / (dh as f32).sqrt();
    let qh = q.reshape(&[t, n_heads, dh]).permute(&[1, 0, 2]).contiguous();
    let kv_heads = |x: &Tensor| {
        let hx = x.reshape(&[t, n_kv_heads, dh]).permute(&[1, 0, 2]).contiguous();
        hx.reshape(&[n_kv_heads, 1, t, dh]).broadcast_to(&[n_kv_heads, g, t, dh]).reshape(&[n_heads, t, dh])
    };
    let (kh, vh) = (kv_heads(k), kv_heads(v));
    let scores = softcapped(qh.matmul(&kh.transpose(2, 1)).mul(&q.scalar(scale)), softcap);
    // The mask is added BEFORE the sink column is appended, so a masked-out key contributes nothing
    // while the sink still does — which is what makes the first row's output well defined.
    let probs = softmax_with_sinks(&scores.add(&causal_mask(q, t)), sinks);
    probs.matmul(&vh).permute(&[1, 0, 2]).reshape(&[t, d])
}

/// Gemma-2 attention-logit softcapping (`cap·tanh(x/cap)` over the scores before softmax); identity
/// when `cap == 0`.
fn softcapped(scores: Tensor, cap: f32) -> Tensor { if cap > 0.0 { scores.softcap(cap) } else { scores } }

/// Sliding-window causal attention (Gemma's local layers): query `i` attends to keys `(i-window, i]`.
/// `window == 0` is full causal. Masking older keys in the full cache is identical to a rolling window
/// cache (they contribute 0), so this is exact — just not memory-optimized.
pub fn causal_attention_win(q: &Tensor, k: &Tensor, v: &Tensor, n_heads: usize, n_kv_heads: usize, window: usize, softcap: f32) -> Tensor {
    let t = q.shape[0];
    // The cache may hold MORE rows than this query block: chunked prefill feeds `t` new rows against
    // a cache already `s` long. This used to reshape k/v to [t, ..] unconditionally, which panics on
    // numel the moment t != s — so windowed models could not be chunk-prefilled at all, while the
    // non-windowed path already handled it. Query i sits at absolute position `off + i`.
    let s = k.shape[0];
    assert!(s >= t, "cache ({s}) cannot be shorter than the query block ({t})");
    let off = s - t;
    let d = q.shape[1];
    let dh = d / n_heads;
    let g = n_heads / n_kv_heads;
    let scale = 1.0 / (dh as f32).sqrt();
    let qh = q.reshape(&[t, n_heads, dh]).permute(&[1, 0, 2]).contiguous();
    let kv_heads = |x: &Tensor| {
        let hx = x.reshape(&[s, n_kv_heads, dh]).permute(&[1, 0, 2]).contiguous();
        hx.reshape(&[n_kv_heads, 1, s, dh]).broadcast_to(&[n_kv_heads, g, s, dh]).reshape(&[n_heads, s, dh])
    };
    let (kh, vh) = (kv_heads(k), kv_heads(v));
    let scores = softcapped(qh.matmul(&kh.transpose(2, 1)).mul(&q.scalar(scale)), softcap);
    let probs = scores.add(&sliding_causal_mask_off(q, t, s, off, window)).softmax(2);
    probs.matmul(&vh).permute(&[1, 0, 2]).reshape(&[t, d])
}

/// Sliding-window single-query decode: the new query (at position S−1) attends to the last `window`
/// cached keys only. `window == 0` or `window >= S` → no masking (identical to `decode_attention`).
pub fn decode_attention_win(q: &Tensor, k: &Tensor, v: &Tensor, n_heads: usize, n_kv_heads: usize, window: usize, softcap: f32) -> Tensor {
    let s = k.shape[0];
    if window == 0 || window >= s { return decode_attention(q, k, v, n_heads, n_kv_heads, softcap); }
    let d = q.shape[1];
    let dh = d / n_heads;
    let g = n_heads / n_kv_heads;
    let scale = 1.0 / (dh as f32).sqrt();
    let qh = q.reshape(&[1, n_heads, dh]).permute(&[1, 0, 2]).contiguous();
    let kv_heads = |x: &Tensor| {
        let hx = x.reshape(&[s, n_kv_heads, dh]).permute(&[1, 0, 2]).contiguous();
        hx.reshape(&[n_kv_heads, 1, s, dh]).broadcast_to(&[n_kv_heads, g, s, dh]).reshape(&[n_heads, s, dh])
    };
    let (kh, vh) = (kv_heads(k), kv_heads(v));
    let scores = softcapped(qh.matmul(&kh.transpose(2, 1)).mul(&q.scalar(scale)), softcap); // [nh, 1, S]
    // Mask every key older than the window from the query at S−1.
    let mut m = vec![0.0f32; s];
    for j in 0..(s - window) { m[j] = -1e30; }
    let mask = Tensor::from_vec(&q.ctx_arc(), &m, &[1, 1, s]);
    probs_matmul(scores.add(&mask).softmax(2), &vh, n_heads, d)
}

fn probs_matmul(probs: Tensor, vh: &Tensor, n_heads: usize, d: usize) -> Tensor {
    let _ = n_heads;
    probs.matmul(vh).permute(&[1, 0, 2]).reshape(&[1, d])
}

/// Additive banded mask: −inf where `j > i` (future) or `i − j >= window` (older than the window).
fn sliding_causal_mask(like: &Tensor, t: usize, window: usize) -> Tensor {
    sliding_causal_mask_off(like, t, t, 0, window)
}

/// Banded causal mask for a query block that starts at absolute position `off` against `s` cached
/// keys. Query `i` is at position `off + i`; key `j` at position `j`. `off == 0` (t == s) reproduces
/// the whole-history mask exactly.
fn sliding_causal_mask_off(like: &Tensor, t: usize, s: usize, off: usize, window: usize) -> Tensor {
    let mut m = vec![0.0f32; t * s];
    for i in 0..t {
        let qi = off + i;
        for j in 0..s {
            if j > qi || (window > 0 && qi - j >= window) { m[i * s + j] = -1e30; }
        }
    }
    Tensor::from_vec(&like.ctx_arc(), &m, &[t, s])
}

/// Incremental-decode attention against a KV cache (one new query token vs all cached keys/values).
/// q is [1, n_heads·dh]; k/v are the cache [S, n_kv_heads·dh]. No mask (cache precedes the query).
/// Composed from general ops — the KV-cache decode path, no bespoke kernel.
pub fn decode_attention(q: &Tensor, k: &Tensor, v: &Tensor, n_heads: usize, n_kv_heads: usize, softcap: f32) -> Tensor {
    let dh = q.shape[1] / n_heads;
    // Fused single-pass kernel collapses the ~12-dispatch composed path into one
    // workgroup-per-head kernel; keys stream in chunks with online softmax, so any cache length works.
    // The fused kernel has no softcap, so a softcapped model (Gemma-2, always dh>128) takes the composed path.
    if dh <= 128 && softcap == 0.0 {
        return q.fused_decode_attention(k, v, n_heads, n_kv_heads, dh);
    }
    decode_attention_composed(q, k, v, n_heads, n_kv_heads, softcap)
}

/// The composed (multi-dispatch) single-query attention — reference for the fused kernel and the
/// fallback for long contexts. reshape/permute/matmul/softmax/matmul with GQA broadcast.
pub fn decode_attention_composed(q: &Tensor, k: &Tensor, v: &Tensor, n_heads: usize, n_kv_heads: usize, softcap: f32) -> Tensor {
    let d = q.shape[1];
    let dh = d / n_heads;
    let s = k.shape[0];
    let g = n_heads / n_kv_heads;
    let scale = 1.0 / (dh as f32).sqrt();
    let qh = q.reshape(&[1, n_heads, dh]).permute(&[1, 0, 2]).contiguous(); // [nh, 1, dh]
    let kv_heads = |x: &Tensor| {
        let hx = x.reshape(&[s, n_kv_heads, dh]).permute(&[1, 0, 2]).contiguous(); // [nkv, S, dh]
        hx.reshape(&[n_kv_heads, 1, s, dh]).broadcast_to(&[n_kv_heads, g, s, dh]).reshape(&[n_heads, s, dh])
    };
    let (kh, vh) = (kv_heads(k), kv_heads(v)); // [nh, S, dh]
    let probs = softcapped(qh.matmul(&kh.transpose(2, 1)).mul(&q.scalar(scale)), softcap).softmax(2); // [nh, 1, S]
    probs.matmul(&vh).permute(&[1, 0, 2]).reshape(&[1, d]) // [nh,1,dh] → [1,d]
}

/// Bidirectional (non-causal) multi-head attention — JEPA/ViT encoders. No causal mask; every
/// position attends to all others. q [T, n_heads·dh]; k/v [T, n_kv_heads·dh]. Returns [T, n_heads·dh].
pub fn bidirectional_attention(q: &Tensor, k: &Tensor, v: &Tensor, n_heads: usize, n_kv_heads: usize) -> Tensor {
    let (t, d) = (q.shape[0], q.shape[1]);
    let dh = d / n_heads;
    let g = n_heads / n_kv_heads;
    let scale = 1.0 / (dh as f32).sqrt();
    let qh = q.reshape(&[t, n_heads, dh]).permute(&[1, 0, 2]).contiguous();
    let kv_heads = |x: &Tensor| {
        let hx = x.reshape(&[t, n_kv_heads, dh]).permute(&[1, 0, 2]).contiguous();
        hx.reshape(&[n_kv_heads, 1, t, dh]).broadcast_to(&[n_kv_heads, g, t, dh]).reshape(&[n_heads, t, dh])
    };
    let (kh, vh) = (kv_heads(k), kv_heads(v));
    let probs = qh.matmul(&kh.transpose(2, 1)).mul(&q.scalar(scale)).softmax(2); // no mask
    probs.matmul(&vh).permute(&[1, 0, 2]).reshape(&[t, d])
}

/// Rectangular full (non-causal) attention: `Tq` queries over `Tk` keys/values, no mask — the
/// diffusion-tower stream of Cosmos's dual-stream joint attention (Q_DM over [K_AR; K_DM]).
pub fn full_attention_kv(q: &Tensor, k: &Tensor, v: &Tensor, n_heads: usize, n_kv_heads: usize) -> Tensor {
    let (tq, d) = (q.shape[0], q.shape[1]);
    let tk = k.shape[0];
    let dh = d / n_heads;
    let g = n_heads / n_kv_heads;
    let scale = 1.0 / (dh as f32).sqrt();
    let qh = q.reshape(&[tq, n_heads, dh]).permute(&[1, 0, 2]).contiguous();
    let kv = |x: &Tensor| {
        let hx = x.reshape(&[tk, n_kv_heads, dh]).permute(&[1, 0, 2]).contiguous();
        hx.reshape(&[n_kv_heads, 1, tk, dh]).broadcast_to(&[n_kv_heads, g, tk, dh]).reshape(&[n_heads, tk, dh])
    };
    let (kh, vh) = (kv(k), kv(v));
    let probs = qh.matmul(&kh.transpose(2, 1)).mul(&q.scalar(scale)).softmax(2); // [nh, Tq, Tk]
    probs.matmul(&vh).permute(&[1, 0, 2]).reshape(&[tq, d])
}

/// Like [`full_attention_kv`] but with an arbitrary ADDITIVE attention mask `mask` of shape `[Tq, Tk]`
/// (0 = allowed, −∞ = masked), added to the scaled scores before softmax. Broadcasts over heads.
/// Used for V-JEPA 2-AC's block-causal frame attention.
pub fn masked_attention_kv(q: &Tensor, k: &Tensor, v: &Tensor, mask: &Tensor, n_heads: usize, n_kv_heads: usize) -> Tensor {
    let (tq, d) = (q.shape[0], q.shape[1]);
    let tk = k.shape[0];
    let dh = d / n_heads;
    let g = n_heads / n_kv_heads;
    let scale = 1.0 / (dh as f32).sqrt();
    let qh = q.reshape(&[tq, n_heads, dh]).permute(&[1, 0, 2]).contiguous();
    let kv = |x: &Tensor| {
        let hx = x.reshape(&[tk, n_kv_heads, dh]).permute(&[1, 0, 2]).contiguous();
        hx.reshape(&[n_kv_heads, 1, tk, dh]).broadcast_to(&[n_kv_heads, g, tk, dh]).reshape(&[n_heads, tk, dh])
    };
    let (kh, vh) = (kv(k), kv(v));
    let scores = qh.matmul(&kh.transpose(2, 1)).mul(&q.scalar(scale)); // [nh, Tq, Tk]
    let scores = scores.add(&mask.reshape(&[1, tq, tk]).broadcast_to(&[n_heads, tq, tk]));
    let probs = scores.softmax(2);
    probs.matmul(&vh).permute(&[1, 0, 2]).reshape(&[tq, d])
}

/// Additive causal mask [T,T]: 0 on/below the diagonal, −∞ above (broadcasts over heads on add).
/// Causal attention where the queries are the **last `t_q` positions** of a longer history.
///
/// The case between full prefill (`t_q == t_kv`) and single-token decode (`t_q == 1`), and Ferric had
/// neither a kernel nor a mask for it. Two capabilities need exactly this:
///
/// - **prefix caching** — the shared prompt's KV is already in the cache, so only the new suffix is
///   prefilled against a much longer history;
/// - **chunked prefill** — a long prompt processed in bounded pieces instead of one quadratic pass, which
///   is what keeps peak activation memory flat.
///
/// Query `i` sits at absolute position `offset + i`, where `offset = t_kv − t_q`, and attends to keys
/// `0 ..= offset + i`. Getting that offset wrong is a silent bug: the model still produces fluent text,
/// having simply attended to the wrong span.
pub fn chunked_attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    n_heads: usize,
    n_kv_heads: usize,
    softcap: f32,
) -> Tensor {
    let (tq, d) = (q.shape[0], q.shape[1]);
    let tkv = k.shape[0];
    assert!(tkv >= tq, "chunked_attention: {tq} queries against a shorter {tkv}-key history");
    if tq == tkv { return causal_attention(q, k, v, n_heads, n_kv_heads, softcap); }
    let dh = d / n_heads;
    let g = n_heads / n_kv_heads;
    let scale = 1.0 / (dh as f32).sqrt();
    let qh = q.reshape(&[tq, n_heads, dh]).permute(&[1, 0, 2]).contiguous(); // [nh, tq, dh]
    let kv_heads = |x: &Tensor| {
        let hx = x.reshape(&[tkv, n_kv_heads, dh]).permute(&[1, 0, 2]).contiguous(); // [nkv, tkv, dh]
        hx.reshape(&[n_kv_heads, 1, tkv, dh]).broadcast_to(&[n_kv_heads, g, tkv, dh])
            .reshape(&[n_heads, tkv, dh])
    };
    let (kh, vh) = (kv_heads(k), kv_heads(v));
    let scores = softcapped(qh.matmul(&kh.transpose(2, 1)).mul(&q.scalar(scale)), softcap); // [nh, tq, tkv]
    let probs = scores.add(&offset_causal_mask(q, tq, tkv)).softmax(2);
    let ctx = probs.matmul(&vh);                                                            // [nh, tq, dh]
    ctx.permute(&[1, 0, 2]).reshape(&[tq, d])
}

/// Mask for [`chunked_attention`]: row `i` may see keys up to `(t_kv − t_q) + i`.
fn offset_causal_mask(like: &Tensor, tq: usize, tkv: usize) -> Tensor {
    let off = tkv - tq;
    let mut m = vec![0.0f32; tq * tkv];
    for i in 0..tq {
        for j in (off + i + 1)..tkv {
            m[i * tkv + j] = -1e30;
        }
    }
    Tensor::from_vec(&like.ctx_arc(), &m, &[tq, tkv])
}

/// A `[t, t]` causal mask broadcast to `[heads, t, t]`, additive (`0` visible, `−1e30` hidden).
///
/// The per-head shape exists because attention scores are `[heads, T, S]` and adding a `[T, T]` mask
/// to them relies on broadcasting rules that differ between op sets. Materialising the head axis
/// makes the alignment a shape the compiler checks rather than a convention the reader has to trust.
pub fn causal_mask_hw(like: &Tensor, heads: usize, t: usize) -> Tensor {
    causal_mask(like, t).reshape(&[1, t, t]).broadcast_to(&[heads, t, t]).contiguous()
}

fn causal_mask(like: &Tensor, t: usize) -> Tensor {
    let mut m = vec![0.0f32; t * t];
    for i in 0..t {
        for j in (i + 1)..t {
            m[i * t + j] = -1e30;
        }
    }
    Tensor::from_vec(&like.ctx_arc(), &m, &[t, t])
}

// ── Gated-delta-net prep fusion (Qwen3.5/3.6 hybrid decode) ─────────────────────────────────────
// The GDN mixer's pre-processing was ~15 small dispatches (cat/conv/narrow/silu ×2 copies, l2norm ×2,
// tile ×2, softplus chain, sigmoid, cat) between the in_proj matmul and the delta rule — pure
// dispatch overhead at decode. These three kernels replace all of it. Formulas replicate the unary /
// l2norm kernels bit-for-bit (same serial reduction order, same stable softplus, `1/max(sqrt(ss),eps)`).

const GDN_CONV_WGSL: &str = r#"
@group(0) @binding(0) var<storage,read>        proj: array<f32>;   // [T, pw] — [qkv(cd) | z | alpha | beta]
@group(0) @binding(1) var<storage,read>        prev: array<f32>;   // [pad, cd] carried conv tail
@group(0) @binding(2) var<storage,read>        w:    array<f32>;   // [cd, L]
@group(0) @binding(3) var<storage,read_write>  conv: array<f32>;   // [T, cd] silu(conv(cat(prev,qkv)))[pad..]
@group(0) @binding(4) var<storage,read_write>  tail: array<f32>;   // [pad, cd] last pad rows of the stream
@group(0) @binding(5) var<storage,read_write>  v:    array<f32>;   // [T, d_inner] = conv cols kd2..
@group(0) @binding(6) var<uniform>             info: array<vec4<u32>, 2>; // t,cd,pad,l | pw,d_inner,kd2,stride
fn stream(i: u32, c: u32) -> f32 {
    let pad = info[0].z; let cd = info[0].y; let pw = info[1].x;
    if (i < pad) { return prev[i * cd + c]; }
    return proj[(i - pad) * pw + c];
}
fn convsilu(row: u32, c: u32) -> f32 {
    let l = info[0].w;
    var acc = 0.0;
    for (var k: u32 = 0u; k < l; k = k + 1u) { acc = acc + w[c * l + k] * stream(row + k, c); }
    return acc / (1.0 + exp(-acc));
}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x + gid.y * info[1].w;
    let t = info[0].x; let cd = info[0].y; let pad = info[0].z;
    let di = info[1].y; let kd2 = info[1].z;
    let n1 = t * cd; let n2 = pad * cd;
    if (idx < n1) {
        conv[idx] = convsilu(idx / cd, idx % cd);
    } else if (idx < n1 + n2) {
        let e = idx - n1;
        tail[e] = stream(t + e / cd, e % cd);
    } else if (idx < n1 + n2 + t * di) {
        let e = idx - n1 - n2;
        v[e] = convsilu(e / di, kd2 + e % di);
    }
}
"#;

const GDN_GATE_WGSL: &str = r#"
@group(0) @binding(0) var<storage,read>        proj: array<f32>;
@group(0) @binding(1) var<storage,read>        dtb:  array<f32>;   // [nv]
@group(0) @binding(2) var<storage,read>        a:    array<f32>;   // [nv] — already -exp(A_log)
@group(0) @binding(3) var<storage,read_write>  gb:   array<f32>;   // [T, nv, 2] = (g, β)
@group(0) @binding(4) var<uniform>             info: vec4<u32>;    // t, nv, alpha_off, pw
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x; let t = info.x; let nv = info.y; let ao = info.z; let pw = info.w;
    if (idx >= t * nv) { return; }
    let row = idx / nv; let hv = idx % nv;
    let ar = proj[row * pw + ao + hv] + dtb[hv];
    let g = a[hv] * (max(ar, 0.0) + log(1.0 + exp(-abs(ar))));   // stable softplus, then plain multiply
    let br = proj[row * pw + ao + nv + hv];
    gb[idx * 2u] = g;
    gb[idx * 2u + 1u] = 1.0 / (1.0 + exp(-br));
}
"#;

const GDN_QK_WGSL: &str = r#"
@group(0) @binding(0) var<storage,read>        conv: array<f32>;   // [T, cd]
@group(0) @binding(1) var<storage,read_write>  q:    array<f32>;   // [T, nv·dk] l2normed·scale, tiled
@group(0) @binding(2) var<storage,read_write>  kk:   array<f32>;   // [T, nv·dk] l2normed, tiled
@group(0) @binding(3) var<uniform>             info: array<vec4<u32>, 2>; // t,nk,dk,rep | cd,scale,eps,_
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let t = info[0].x; let nk = info[0].y; let dk = info[0].z; let rep = info[0].w;
    let cd = info[1].x; let scale = bitcast<f32>(info[1].y); let eps = bitcast<f32>(info[1].z);
    if (idx >= t * nk * 2u) { return; }
    let plane = idx / (t * nk);              // 0 = q, 1 = k
    let r = (idx % (t * nk)) / nk; let h = idx % nk;
    let base = r * cd + plane * (nk * dk) + h * dk;
    var ss = 0.0;
    for (var j: u32 = 0u; j < dk; j = j + 1u) { let x = conv[base + j]; ss = ss + x * x; }
    let inv = 1.0 / max(sqrt(ss), eps);      // same clamp as the l2norm kernel
    let s = select(1.0, scale, plane == 0u); // 1/√dv folded into q only
    let orow = r * (rep * nk * dk);
    for (var ri: u32 = 0u; ri < rep; ri = ri + 1u) {
        let ob = orow + (ri * nk + h) * dk;  // tiled: v-head = ri·nk + h  (head % nk broadcast)
        for (var j: u32 = 0u; j < dk; j = j + 1u) {
            let val = conv[base + j] * inv * s;
            if (plane == 0u) { q[ob + j] = val; } else { kk[ob + j] = val; }
        }
    }
}
"#;

/// Fused GDN conv stage: silu(causal depthwise conv over [carried tail; qkv-part-of-proj]) plus the
/// carried tail for the next step and the (conv'd) V block — one dispatch for what was five.
pub fn gdn_conv(proj: &Tensor, prev: &Tensor, w: &Tensor, cd: usize, kernel_l: usize, d_inner: usize, kd2: usize) -> (Tensor, Tensor, Tensor) {
    let (ctx, p) = (&proj.ctx, proj.contiguous());
    let t = p.shape[0]; let pw = p.shape[1]; let pad = kernel_l - 1;
    let conv = crate::empty(ctx, t * cd);
    let tail = crate::empty(ctx, pad * cd);
    let v = crate::empty(ctx, t * d_inner);
    let n = t * cd + pad * cd + t * d_inner;
    let (grid, stride) = crate::groups2d(n);
    crate::run(ctx, GDN_CONV_WGSL, "gdn_conv",
        &[p.buf.as_ref(), prev.contiguous().buf.as_ref(), w.contiguous().buf.as_ref(), &conv, &tail, &v,
          &crate::unibuf(ctx, &[t as u32, cd as u32, pad as u32, kernel_l as u32, pw as u32, d_inner as u32, kd2 as u32, stride])],
        grid);
    (Tensor::from_parts(ctx, conv, vec![t, cd]), Tensor::from_parts(ctx, tail, vec![pad, cd]), Tensor::from_parts(ctx, v, vec![t, d_inner]))
}

/// Fused GDN gate pack: (g, β) per v-head from the proj's alpha/beta columns — one dispatch for four.
pub fn gdn_gate(proj: &Tensor, dt_bias: &Tensor, a: &Tensor, nv: usize, alpha_off: usize) -> Tensor {
    let (ctx, p) = (&proj.ctx, proj.contiguous());
    let t = p.shape[0]; let pw = p.shape[1];
    let gb = crate::empty(ctx, t * nv * 2);
    crate::run(ctx, GDN_GATE_WGSL, "gdn_gate",
        &[p.buf.as_ref(), dt_bias.contiguous().buf.as_ref(), a.contiguous().buf.as_ref(), &gb,
          &crate::unibuf(ctx, &[t as u32, nv as u32, alpha_off as u32, pw as u32])],
        crate::groups(t * nv));
    Tensor::from_parts(ctx, gb, vec![t, nv, 2])
}

/// Fused GDN q/k: per-head L2 norm (+ q's 1/√dv) and the tiled head-broadcast — one dispatch for six.
pub fn gdn_qk(conv: &Tensor, nk: usize, dk: usize, rep: usize, cd: usize, scale: f32, eps: f32) -> (Tensor, Tensor) {
    let (ctx, c) = (&conv.ctx, conv.contiguous());
    let t = c.shape[0]; let nv = rep * nk;
    let q = crate::empty(ctx, t * nv * dk);
    let k = crate::empty(ctx, t * nv * dk);
    crate::run(ctx, GDN_QK_WGSL, "gdn_qk",
        &[c.buf.as_ref(), &q, &k,
          &crate::unibuf(ctx, &[t as u32, nk as u32, dk as u32, rep as u32, cd as u32, scale.to_bits(), eps.to_bits(), 0])],
        crate::groups(t * nk * 2));
    (Tensor::from_parts(ctx, q, vec![t, nv, dk]), Tensor::from_parts(ctx, k, vec![t, nv, dk]))
}

const GDN_POST_WGSL: &str = r#"
@group(0) @binding(0) var<storage,read>        o:    array<f32>;   // [T, nv, dv] delta-rule output
@group(0) @binding(1) var<storage,read>        proj: array<f32>;   // z gate read in place from the in_proj
@group(0) @binding(2) var<storage,read>        norm: array<f32>;   // [dv]
@group(0) @binding(3) var<storage,read_write>  outp: array<f32>;   // [T, nv·dv]
@group(0) @binding(4) var<uniform>             info: array<vec4<u32>, 2>; // t,nv,dv,z_off | pw,eps,_,_
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let r = gid.x; let t = info[0].x; let nv = info[0].y; let dv = info[0].z;
    if (r >= t * nv) { return; }
    let zo = info[0].w; let pw = info[1].x; let eps = bitcast<f32>(info[1].y);
    let base = r * dv;
    var ms = 0.0;
    for (var j: u32 = 0u; j < dv; j = j + 1u) { let v = o[base + j]; ms = ms + v * v; }
    let inv = 1.0 / sqrt(ms / f32(dv) + eps);       // same mean+eps clamp as the rmsnorm kernel
    let zb = (r / nv) * pw + zo + (r % nv) * dv;
    for (var j: u32 = 0u; j < dv; j = j + 1u) {
        let z = proj[zb + j];
        outp[base + j] = o[base + j] * inv * norm[j] * (z / (1.0 + exp(-z)));
    }
}
"#;

/// Fused GDN post: gated RMSNorm over head_v_dim — rmsnorm(o)·silu(z), z read in place from the
/// in_proj columns — one dispatch for the narrow/silu/mul/reshape chain. Returns [T, nv·dv].
pub fn gdn_post(o: &Tensor, proj: &Tensor, norm: &Tensor, z_off: usize, eps: f32) -> Tensor {
    let (ctx, oc) = (&o.ctx, o.contiguous());
    let (t, nv, dv) = (oc.shape[0], oc.shape[1], oc.shape[2]);
    let p = proj.contiguous();
    let out = crate::empty(ctx, t * nv * dv);
    crate::run(ctx, GDN_POST_WGSL, "gdn_post",
        &[oc.buf.as_ref(), p.buf.as_ref(), norm.contiguous().buf.as_ref(), &out,
          &crate::unibuf(ctx, &[t as u32, nv as u32, dv as u32, z_off as u32, p.shape[1] as u32, eps.to_bits(), 0, 0])],
        crate::groups(t * nv));
    Tensor::from_parts(ctx, out, vec![t, nv * dv])
}

/// Rectangular causal attention for speculative-verify forwards: `t` new queries against a cache of
/// `s ≥ t` keys/values whose last `t` rows are the queries' own positions. Query `i` attends keys
/// `j ≤ s−t+i`. Identical math to `causal_attention` when `s == t`.
pub fn causal_attention_kv(q: &Tensor, k: &Tensor, v: &Tensor, n_heads: usize, n_kv_heads: usize, softcap: f32) -> Tensor {
    let t = q.shape[0];
    let d = q.shape[1];
    let s = k.shape[0];
    let dh = d / n_heads;
    let g = n_heads / n_kv_heads;
    let scale = 1.0 / (dh as f32).sqrt();
    let qh = q.reshape(&[t, n_heads, dh]).permute(&[1, 0, 2]).contiguous(); // [nh, t, dh]
    let kv_heads = |x: &Tensor| {
        let hx = x.reshape(&[s, n_kv_heads, dh]).permute(&[1, 0, 2]).contiguous();
        hx.reshape(&[n_kv_heads, 1, s, dh]).broadcast_to(&[n_kv_heads, g, s, dh]).reshape(&[n_heads, s, dh])
    };
    let (kh, vh) = (kv_heads(k), kv_heads(v));
    let scores = softcapped(qh.matmul(&kh.transpose(2, 1)).mul(&q.scalar(scale)), softcap); // [nh, t, s]
    let mut m = vec![0.0f32; t * s];
    for i in 0..t {
        for j in (s - t + i + 1)..s { m[i * s + j] = -1e30; }
    }
    let mask = Tensor::from_vec(&q.ctx_arc(), &m, &[t, s]);
    let probs = scores.add(&mask).softmax(2);
    probs.matmul(&vh).permute(&[1, 0, 2]).reshape(&[t, d])
}

/// Causal attention where the **value head is narrower than the query/key head**.
///
/// Every other helper here derives one head width from `q.shape[1] / n_heads` and uses it for K and V
/// alike. That is wrong for MLA: DeepSeek-V2 carries a 192-wide query/key head (128 non-positional +
/// 64 RoPE) against a 128-wide value head. Reusing a same-width helper does not fail — it reshapes V
/// to the wrong stride and reads value lanes belonging to the neighbouring head.
///
/// `k`/`v` may be longer than `q` (a decode step or a prefill chunk against a filled cache); the
/// query block is taken to sit at the end, at offset `s - t`.
pub fn causal_attention_split(q: &Tensor, k: &Tensor, v: &Tensor, n_heads: usize,
                              qk_dim: usize, v_dim: usize, softcap: f32) -> Tensor {
    attend_split(q, k, v, n_heads, qk_dim, v_dim, None, softcap)
}

/// [`causal_attention_split`] with a per-head learnable sink, `[n_heads]`. See
/// [`softmax_with_sinks`] for what a sink does and what must not be done to it.
pub fn causal_attention_split_sinks(q: &Tensor, k: &Tensor, v: &Tensor, n_heads: usize,
                                    qk_dim: usize, v_dim: usize, sinks: &Tensor, softcap: f32) -> Tensor {
    attend_split(q, k, v, n_heads, qk_dim, v_dim, Some(sinks), softcap)
}

fn attend_split(q: &Tensor, k: &Tensor, v: &Tensor, n_heads: usize,
                qk_dim: usize, v_dim: usize, sinks: Option<&Tensor>, softcap: f32) -> Tensor {
    let t = q.shape[0];
    let s = k.shape[0];
    assert_eq!(q.shape[1], n_heads * qk_dim, "q width must be n_heads * qk_dim");
    assert_eq!(k.shape[1], n_heads * qk_dim, "k width must be n_heads * qk_dim");
    assert_eq!(v.shape[1], n_heads * v_dim, "v width must be n_heads * v_dim");
    assert!(s >= t, "cache ({s}) shorter than the query block ({t})");
    let scale = 1.0 / (qk_dim as f32).sqrt();
    let qh = q.reshape(&[t, n_heads, qk_dim]).permute(&[1, 0, 2]).contiguous();  // [nh, t, qk]
    let kh = k.reshape(&[s, n_heads, qk_dim]).permute(&[1, 0, 2]).contiguous();  // [nh, s, qk]
    let vh = v.reshape(&[s, n_heads, v_dim]).permute(&[1, 0, 2]).contiguous();   // [nh, s, vd]
    let scores = softcapped(qh.matmul(&kh.transpose(2, 1)).mul(&q.scalar(scale)), softcap);
    // Query row i is at absolute position off+i, so it may attend to keys 0..=off+i.
    let off = s - t;
    let mut m = vec![0.0f32; t * s];
    for i in 0..t {
        for j in (off + i + 1)..s { m[i * s + j] = -1e30; }
    }
    let mask = Tensor::from_vec(&q.ctx_arc(), &m, &[t, s]);
    let masked = scores.add(&mask);
    let probs = match sinks { None => masked.softmax(2), Some(sk) => softmax_with_sinks(&masked, sk) };
    probs.matmul(&vh).permute(&[1, 0, 2]).reshape(&[t, n_heads * v_dim])
}

/// Chunked windowed attention must equal whole-history windowed attention.
///
/// This is the property that makes chunked prefill safe on a sliding-window model: feeding the rows
/// in slices against a growing cache has to produce the SAME output as feeding them at once. It
/// failed before by panicking on numel rather than returning a wrong answer, which was lucky.
#[cfg(test)]
mod windowed_chunk_tests {
    use super::*;
    use std::sync::Arc;

    #[test]
    fn chunked_equals_whole_history() {
        let Ok(ctx) = pollster::block_on(ferric_core::Context::new()) else {
            eprintln!("SKIPPED chunked windowed attention: no GPU");
            return;
        };
        let ctx = Arc::new(ctx);
        let (t, nh, nkv, dh, window) = (12usize, 4usize, 2usize, 8usize, 5usize);
        let d = nh * dh;
        let mk = |n: usize, seed: f32, cols: usize| {
            let v: Vec<f32> = (0..n * cols).map(|i| ((i as f32 + seed) * 0.13).sin()).collect();
            Tensor::from_vec(&ctx, &v, &[n, cols])
        };
        let (q, k, v) = (mk(t, 1.0, d), mk(t, 2.0, nkv * dh), mk(t, 3.0, nkv * dh));

        let whole = pollster::block_on(causal_attention_win(&q, &k, &v, nh, nkv, window, 0.0).to_vec());

        // Same rows, fed in chunks against a cache that grows: q chunk [off, off+len) sees k/v[0, off+len).
        let mut chunked: Vec<f32> = Vec::new();
        let mut off = 0usize;
        while off < t {
            let len = 5.min(t - off);
            let s = off + len;
            let qc = q.narrow(0, off, len).contiguous();
            let kc = k.narrow(0, 0, s).contiguous();
            let vc = v.narrow(0, 0, s).contiguous();
            chunked.extend(pollster::block_on(causal_attention_win(&qc, &kc, &vc, nh, nkv, window, 0.0).to_vec()));
            off = s;
        }
        assert_eq!(chunked.len(), whole.len());
        let scale = whole.iter().fold(1e-6f32, |a, &x| a.max(x.abs()));
        let err = chunked.iter().zip(&whole).fold(0f32, |a, (&c, &w)| a.max((c - w).abs())) / scale;
        eprintln!("windowed chunked vs whole: rel max|Δ| {err:.3e} over {} values", whole.len());
        assert!(err < 1e-5, "chunked windowed attention differs from whole-history by {err:.3e}");
    }
}

#[cfg(test)]
mod sink_tests {
    use super::*;
    use ferric_core::Context;
    use std::sync::Arc;

    macro_rules! ctx_or_skip {
        () => { match pollster::block_on(Context::new()) { Ok(c) => Arc::new(c), Err(_) => { eprintln!("no GPU context — skipping"); return } } };
    }
    fn get(t: &Tensor) -> Vec<f32> { pollster::block_on(t.to_vec()) }

    const NH: usize = 3;
    const T: usize = 2;
    const S: usize = 5;

    fn logits(ctx: &Arc<Context>) -> (Tensor, Vec<f32>) {
        let mut s = 12345u64;
        let v: Vec<f32> = (0..NH * T * S).map(|_| {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((s >> 33) as f32 / (1u64 << 30) as f32) - 2.0
        }).collect();
        (Tensor::from_vec(ctx, &v, &[NH, T, S]), v)
    }

    /// The definition, checked term by term against host arithmetic — including that the row sums
    /// to strictly less than one by exactly the sink's own share.
    #[test]
    fn sinked_softmax_matches_the_definition() {
        let ctx = ctx_or_skip!();
        let (sc, lv) = logits(&ctx);
        let sv = [-1.5f32, 0.0, 2.25];
        let p = get(&softmax_with_sinks(&sc, &Tensor::from_vec(&ctx, &sv, &[NH])));

        for h in 0..NH {
            for t in 0..T {
                let l = &lv[(h * T + t) * S..(h * T + t) * S + S];
                let m = l.iter().fold(sv[h], |a, &b| a.max(b));
                let z: f32 = l.iter().map(|v| (v - m).exp()).sum::<f32>() + (sv[h] - m).exp();
                let mut sum = 0.0;
                for j in 0..S {
                    let want = (l[j] - m).exp() / z;
                    let got = p[(h * T + t) * S + j];
                    assert!((got - want).abs() < 1e-6, "head {h} row {t} key {j}: {got} vs {want}");
                    sum += got;
                }
                let deficit = (sv[h] - m).exp() / z;
                assert!((1.0 - sum - deficit).abs() < 1e-6,
                        "head {h} row {t}: rows must sum to 1 − sink share, got {sum} vs {}", 1.0 - deficit);
                assert!(sum < 1.0, "a sinked row must never sum to one — that is the mechanism");
            }
        }
        // A larger sink must take more mass. If this is flat, the sink is being ignored or
        // renormalised away, and every per-element check above would still pass on a wrong impl
        // that simply dropped the extra column before the softmax rather than after it.
        let mass = |h: usize| 1.0 - (0..S).map(|j| p[h * T * S + j]).sum::<f32>();
        assert!(mass(0) < mass(1) && mass(1) < mass(2),
                "sink mass must increase with the sink: {:?}", (mass(0), mass(1), mass(2)));
    }

    /// A very negative sink is ordinary softmax, exactly. This is the control: if it does NOT
    /// coincide, the sink is leaking into the numerator or the max.
    #[test]
    fn a_deeply_negative_sink_is_plain_softmax() {
        let ctx = ctx_or_skip!();
        let (sc, _) = logits(&ctx);
        let plain = get(&sc.softmax(2));
        let sunk = get(&softmax_with_sinks(&sc, &Tensor::from_vec(&ctx, &[-60.0; NH], &[NH])));
        for (a, b) in plain.iter().zip(&sunk) {
            assert!((a - b).abs() < 1e-6, "sink at −60 should vanish: {a} vs {b}");
        }
    }

    /// At the shipped init of 0.0 this is the classic "+1 in the denominator".
    #[test]
    fn a_zero_sink_is_the_plus_one_denominator() {
        let ctx = ctx_or_skip!();
        let (sc, lv) = logits(&ctx);
        let p = get(&softmax_with_sinks(&sc, &Tensor::from_vec(&ctx, &[0.0; NH], &[NH])));
        for h in 0..NH { for t in 0..T {
            let l = &lv[(h * T + t) * S..(h * T + t) * S + S];
            let z: f32 = l.iter().map(|v| v.exp()).sum::<f32>() + 1.0;
            for j in 0..S {
                let want = l[j].exp() / z;
                assert!((p[(h * T + t) * S + j] - want).abs() < 1e-6, "not the +1 denominator");
            }
        }}
    }

    /// A large positive sink shrinks a head's whole output toward zero — the head declining to
    /// attend. With V bounded, the output norm must fall below the sink's own share of the mass.
    #[test]
    fn a_large_sink_shrinks_the_head_output() {
        let ctx = ctx_or_skip!();
        let (dh, nh) = (4usize, 2usize);
        let n = T * nh * dh;
        let q = Tensor::from_vec(&ctx, &vec![0.3f32; n], &[T, nh * dh]);
        let k = Tensor::from_vec(&ctx, &vec![0.3f32; n], &[T, nh * dh]);
        let v = Tensor::from_vec(&ctx, &vec![1.0f32; n], &[T, nh * dh]);
        let open = get(&causal_attention_sinks(&q, &k, &v, nh, nh, &Tensor::from_vec(&ctx, &[-60.0; 2], &[2]), 0.0));
        let shut = get(&causal_attention_sinks(&q, &k, &v, nh, nh, &Tensor::from_vec(&ctx, &[20.0; 2], &[2]), 0.0));
        for x in &open { assert!((x - 1.0).abs() < 1e-4, "no sink over all-ones V must give 1.0, got {x}") }
        for x in &shut { assert!(x.abs() < 1e-4, "a sink of +20 must shut the head, got {x}") }
    }

    /// The sink is per-head, so one head's sink must not touch another's row. A scalar broadcast
    /// over all heads keeps every shape and would pass all the tests above if they used one value.
    #[test]
    fn sinks_are_per_head_not_shared() {
        let ctx = ctx_or_skip!();
        let (sc, _) = logits(&ctx);
        let p = get(&softmax_with_sinks(&sc, &Tensor::from_vec(&ctx, &[-60.0, 0.0, 3.0], &[NH])));
        let mass: Vec<f32> = (0..NH).map(|h| 1.0 - (0..S).map(|j| p[h * T * S + j]).sum::<f32>()).collect();
        assert!(mass[0] < 1e-5, "head 0 has no sink and should keep all its mass, lost {}", mass[0]);
        assert!(mass[2] > 0.5, "head 2's sink should take most of the mass, took {}", mass[2]);
    }
}
