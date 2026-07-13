//! Transformer building blocks expressed ON the general tensor runtime. The point of unification:
//! attention is not a bespoke kernel — it's `reshape → batched matmul → softmax → matmul`, i.e. the
//! general ops. RMSNorm/softmax/RoPE are fused fast-paths (methods on `Tensor`) but produce exactly
//! what composing primitives would. One substrate; the model is an expression in it.

use crate::Tensor;

/// Linear y = x·W in the [in,out] weight convention (no bias) — just a matmul.
pub fn linear(x: &Tensor, w: &Tensor) -> Tensor { x.matmul(w) }

/// Causal multi-head attention, composed entirely from general ops (+ a fused softmax).
/// q/k/v are [T, n_heads·head_dim] with heads contiguous per row. Returns [T, n_heads·head_dim].
pub fn causal_attention(q: &Tensor, k: &Tensor, v: &Tensor, n_heads: usize) -> Tensor {
    let t = q.shape[0];
    let d = q.shape[1];
    let dh = d / n_heads;
    let scale = 1.0 / (dh as f32).sqrt();
    // [T, d] → [n_heads, T, head_dim]
    let heads = |x: &Tensor| x.reshape(&[t, n_heads, dh]).permute(&[1, 0, 2]).contiguous();
    let (qh, kh, vh) = (heads(q), heads(k), heads(v));
    let scores = qh.matmul(&kh.transpose(2, 1)).mul(&q.scalar(scale)); // [h, T, T]
    let probs = scores.add(&causal_mask(q, t)).softmax(2);             // masked softmax over keys
    let ctx = probs.matmul(&vh);                                       // [h, T, head_dim]
    ctx.permute(&[1, 0, 2]).reshape(&[t, d])
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
