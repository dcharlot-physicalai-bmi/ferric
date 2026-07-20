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

/// Causal multi-head attention with grouped-query attention, composed from general ops (+ fused
/// softmax). q is [T, n_heads·dh]; k/v are [T, n_kv_heads·dh]. Returns [T, n_heads·dh].
pub fn causal_attention(q: &Tensor, k: &Tensor, v: &Tensor, n_heads: usize, n_kv_heads: usize) -> Tensor {
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
    let scores = qh.matmul(&kh.transpose(2, 1)).mul(&q.scalar(scale)); // [nh, T, T]
    let probs = scores.add(&causal_mask(q, t)).softmax(2);             // masked softmax over keys
    let ctx = probs.matmul(&vh);                                       // [nh, T, dh]
    ctx.permute(&[1, 0, 2]).reshape(&[t, d])
}

/// Sliding-window causal attention (Gemma's local layers): query `i` attends to keys `(i-window, i]`.
/// `window == 0` is full causal. Masking older keys in the full cache is identical to a rolling window
/// cache (they contribute 0), so this is exact — just not memory-optimized.
pub fn causal_attention_win(q: &Tensor, k: &Tensor, v: &Tensor, n_heads: usize, n_kv_heads: usize, window: usize) -> Tensor {
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
    let scores = qh.matmul(&kh.transpose(2, 1)).mul(&q.scalar(scale));
    let probs = scores.add(&sliding_causal_mask(q, t, window)).softmax(2);
    probs.matmul(&vh).permute(&[1, 0, 2]).reshape(&[t, d])
}

/// Sliding-window single-query decode: the new query (at position S−1) attends to the last `window`
/// cached keys only. `window == 0` or `window >= S` → no masking (identical to `decode_attention`).
pub fn decode_attention_win(q: &Tensor, k: &Tensor, v: &Tensor, n_heads: usize, n_kv_heads: usize, window: usize) -> Tensor {
    let s = k.shape[0];
    if window == 0 || window >= s { return decode_attention(q, k, v, n_heads, n_kv_heads); }
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
    let scores = qh.matmul(&kh.transpose(2, 1)).mul(&q.scalar(scale)); // [nh, 1, S]
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
    let mut m = vec![0.0f32; t * t];
    for i in 0..t {
        for j in 0..t {
            if j > i || (window > 0 && i - j >= window) { m[i * t + j] = -1e30; }
        }
    }
    Tensor::from_vec(&like.ctx_arc(), &m, &[t, t])
}

/// Incremental-decode attention against a KV cache (one new query token vs all cached keys/values).
/// q is [1, n_heads·dh]; k/v are the cache [S, n_kv_heads·dh]. No mask (cache precedes the query).
/// Composed from general ops — the KV-cache decode path, no bespoke kernel.
pub fn decode_attention(q: &Tensor, k: &Tensor, v: &Tensor, n_heads: usize, n_kv_heads: usize) -> Tensor {
    let dh = q.shape[1] / n_heads;
    // Fused single-pass kernel collapses the ~12-dispatch composed path into one
    // workgroup-per-head kernel; keys stream in chunks with online softmax, so any cache length works.
    if dh <= 128 {
        return q.fused_decode_attention(k, v, n_heads, n_kv_heads, dh);
    }
    decode_attention_composed(q, k, v, n_heads, n_kv_heads)
}

/// The composed (multi-dispatch) single-query attention — reference for the fused kernel and the
/// fallback for long contexts. reshape/permute/matmul/softmax/matmul with GQA broadcast.
pub fn decode_attention_composed(q: &Tensor, k: &Tensor, v: &Tensor, n_heads: usize, n_kv_heads: usize) -> Tensor {
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
    let probs = qh.matmul(&kh.transpose(2, 1)).mul(&q.scalar(scale)).softmax(2); // [nh, 1, S]
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

/// Additive causal mask [T,T]: 0 on/below the diagonal, −∞ above (broadcasts over heads on add).
fn causal_mask(like: &Tensor, t: usize) -> Tensor {
    let mut m = vec![0.0f32; t * t];
    for i in 0..t {
        for j in (i + 1)..t {
            m[i * t + j] = -1e30;
        }
    }
    Tensor::from_vec(&like.ctx_arc(), &m, &[t, t])
}
