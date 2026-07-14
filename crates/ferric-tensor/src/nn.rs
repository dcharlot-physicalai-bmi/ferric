//! Transformer building blocks expressed ON the general tensor runtime. The point of unification:
//! attention is not a bespoke kernel — it's `reshape → batched matmul → softmax → matmul`, i.e. the
//! general ops. RMSNorm/softmax/RoPE are fused fast-paths (methods on `Tensor`) but produce exactly
//! what composing primitives would. One substrate; the model is an expression in it.

use crate::Tensor;

/// Linear y = x·W in the [in,out] weight convention (no bias) — just a matmul.
pub fn linear(x: &Tensor, w: &Tensor) -> Tensor { x.matmul(w) }

/// Linear in the HF convention: W is stored [out, in]; y = x·Wᵀ.
pub fn linear_hf(x: &Tensor, w: &Tensor) -> Tensor { x.matmul(&w.transpose(w.rank() - 1, w.rank() - 2)) }

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
