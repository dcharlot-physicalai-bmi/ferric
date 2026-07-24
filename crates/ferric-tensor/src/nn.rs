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

/// Gemma-2 attention-logit softcapping (`cap·tanh(x/cap)` over the scores before softmax); identity
/// when `cap == 0`.
fn softcapped(scores: Tensor, cap: f32) -> Tensor { if cap > 0.0 { scores.softcap(cap) } else { scores } }

/// Sliding-window causal attention (Gemma's local layers): query `i` attends to keys `(i-window, i]`.
/// `window == 0` is full causal. Masking older keys in the full cache is identical to a rolling window
/// cache (they contribute 0), so this is exact — just not memory-optimized.
pub fn causal_attention_win(q: &Tensor, k: &Tensor, v: &Tensor, n_heads: usize, n_kv_heads: usize, window: usize, softcap: f32) -> Tensor {
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
    let probs = scores.add(&sliding_causal_mask(q, t, window)).softmax(2);
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
