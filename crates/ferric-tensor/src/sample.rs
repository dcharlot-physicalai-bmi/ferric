//! **Token selection on the device: the O(vocab) work moves to the GPU, the exact arithmetic stays where
//! the reference does it.**
//!
//! # What this replaces
//!
//! Every decode step read the whole logits row back (608 KB on a 152k vocabulary, 0.45 ms measured) and
//! sampled it on the host, where the sampled path computes 152k `exp`s and SORTS all 152k probabilities
//! (`ferric-serve` `genopts::sample`): 6.0 ms per row at T=0.8 and 12.8 ms at T=1.0 on the dev Mac
//! (loaded), i.e. more than the forward that produced the row. Peers sample on the device (vLLM/SGLang
//! with FlashInfer's sort-free rejection sampling) — read as specs, not oracles.
//!
//! # Why the device does not simply return the sampled id
//!
//! The requirement is that the chosen token be IDENTICAL to the host sampler's for the same logits and
//! RNG state. Greedy is exact on the device: a max is exact arithmetic, and the tie rule (the LAST index
//! among equal maxima, what `Iterator::max_by` returns) is reproduced. Temperature sampling is not: its
//! result depends on every `exp` bit (the sequential `Σ probs` and the sorted cumulative sums), and the
//! host's `f32::exp` is the platform libm — MEASURED on the dev Mac to differ from the correctly rounded
//! value on 1,193,247 of the 1,120,927,745 floats in [-104, 0] (0.106%), and from ARM's published expf on
//! 1,198,768. No portable device kernel reproduces an unpublished libm bit for bit, and replacing the
//! host's `exp` would move the reference this must match. So the work splits:
//!
//! * the device computes the row's (penalized) maximum, approximate probabilities and, from them, a
//!   REDUCED ROW: every element that can change the host's sequential `Σ probs` (an addition of
//!   `p < ulp(acc)/2` cannot change `acc`, and `acc` is at least the running maximum of what came
//!   before it), plus every element the nucleus / top-k / min-p prefix can reach;
//! * the host runs the reference sampler itself on that reduced row, with its own `exp`.
//!
//! Every exclusion is conservative by a stated bound `eps` on |p_host/p_device − 1| (WGSL's accuracy for
//! `/` and `exp`, plus the penalty division; see `K2`), so the host's arithmetic over the reduced row is
//! the same sequence of operations on the same values as over the full row. What the device cannot
//! prove it says: a row with a NaN, a non-finite maximum, an `eps` above 1e-2, or a reduced row larger
//! than [`OUT_MAX`] comes back [`RowOut::Fallback`] and the caller reads that row. Measured on 120 real
//! Qwen2.5-0.5B rows (`ferric-llama/examples/sample_sets.rs`): the reduced row is a median 2.5k elements
//! at T=0.8 and 11k at T=1.0 (the nucleus itself: 5 and 16), against 151,936.
//!
//! Multi-row: every kernel takes `rows` requests at once (a batched decode step, a prompt-lookup verify
//! forward), one workgroup row per request, one command buffer, one readback.
use crate::{empty, flush_batch, run, unibuf, Context, Tensor};
use std::sync::Arc;

/// Most reduced-row entries one row may return (id, raw logit). Beyond it: [`RowOut::Fallback`].
pub const OUT_MAX: usize = 32768;
const HDR: usize = 8;
const NBUCKET: usize = 1024;

/// An approximate adjustment of one token's logit, for the device's FILTER only. The host applies the
/// exact adjustments to the reduced row itself; this lets the device rank an adjusted token about right.
/// Applied in the host sampler's order: `add`, `sub`, `rep`, `sub2`.
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub struct Patch {
    pub token: u32,
    /// `x + add` (OpenAI `logit_bias`; `-inf` bans the token).
    pub add: Option<f32>,
    /// `x - sub` (OpenAI presence/frequency).
    pub sub: Option<f32>,
    /// `x > 0 ? x / rep : x * rep` (llama.cpp repeat penalty).
    pub rep: Option<f32>,
    /// `x - sub2` (DRY).
    pub sub2: Option<f32>,
}

/// What one row asks for. `temperature <= 0` is greedy.
#[derive(Clone, Debug, Default)]
pub struct RowReq {
    pub row: usize,
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: usize,
    pub min_p: f32,
    /// Sorted by token, one entry per token.
    pub patches: Vec<Patch>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum RowOut {
    /// Greedy with no penalties: the host's `argmax` of the row, exactly.
    Id(u32),
    /// The reduced row: token ids in increasing order and their RAW (unpenalized) logits. `tau_hi`
    /// bounds from above the host probability of every token NOT listed (sampled rows; 0 for greedy).
    Reduced { ids: Vec<u32>, raw: Vec<f32>, tau_hi: f32, eps: f32 },
    /// The device could not prove a reduced row sufficient: read this row and sample it on the host.
    Fallback,
}

// ------------------------------------------------------------------------------------------------
// Kernels. Specs: 8 u32 per request [row, mode(0 greedy|1 sample), T, top_p, top_k, min_p, patch_off,
// patch_n]. Patches: 6 u32 [token, flags(1 add|2 sub|4 rep|8 sub2), add, sub, rep, sub2], sorted by token.
// `work` per request: [0..16) row stats (max, eps), then NB x 8 block partials.
// `outb` per request: [status(0 pending|1 id|2 reduced|3 fallback), count, tau_hi, eps, id, ..] + pairs.
// ------------------------------------------------------------------------------------------------

const COMMON: &str = r#"
const WG: u32 = 256u;
fn neg_inf() -> f32 { return bitcast<f32>(0xff800000u); }
fn spec(q: u32, k: u32) -> u32 { return specs[q * 8u + k]; }
// The row value the HOST sampler sees at token i: raw, or the penalized value — approximately, since
// WGSL `/` is only guaranteed to 2.5 ulp, which the eps bound covers.
fn yval(q: u32, i: u32, raw: f32) -> vec2<f32> {      // (value, 1 if penalized)
    let off = spec(q, 6u); let n = spec(q, 7u);
    var lo = 0u; var hi = n;
    loop {
        if (lo >= hi) { break; }
        let mid = (lo + hi) / 2u;
        if (patches[(off + mid) * 6u] < i) { lo = mid + 1u; } else { hi = mid; }
    }
    if (lo < n && patches[(off + lo) * 6u] == i) {
        let b = (off + lo) * 6u; let fl = patches[b + 1u];
        var v = raw;
        if ((fl & 1u) != 0u) { v = v + bitcast<f32>(patches[b + 2u]); }
        if ((fl & 2u) != 0u) { v = v - bitcast<f32>(patches[b + 3u]); }
        if ((fl & 4u) != 0u) { let r = bitcast<f32>(patches[b + 4u]); v = select(v * r, v / r, v > 0.0); }
        if ((fl & 8u) != 0u) { v = v - bitcast<f32>(patches[b + 5u]); }
        return vec2<f32>(v, 1.0);
    }
    return vec2<f32>(raw, 0.0);
}
// (v2, i2) replaces (v1, i1) when larger, or EQUAL at a later index: `Iterator::max_by` keeps the last
// of equal maxima, and -0.0 == +0.0 compares equal there exactly as it does here.
fn better(v2: f32, i2: u32, v1: f32, i1: u32) -> bool { return v2 > v1 || (v2 == v1 && i2 > i1); }
"#;

/// K1: per 1024-element block — max and last argmax of the penalized row, the same over UNPENALIZED
/// tokens only (greedy-with-penalties hands the host that one plus the penalized tokens), a NaN flag,
/// and the largest |value| among penalized tokens (for eps).
const K1: &str = r#"
@group(0) @binding(0) var<storage,read> logits: array<f32>;
@group(0) @binding(1) var<storage,read> specs: array<u32>;
@group(0) @binding(2) var<storage,read> patches: array<u32>;
@group(0) @binding(3) var<storage,read_write> work: array<u32>;
@group(0) @binding(4) var<uniform> info: vec4<u32>;   // V, NB, OUT_MAX, rows
__COMMON__
var<workgroup> sv: array<f32, 256>; var<workgroup> si: array<u32, 256>;
var<workgroup> su: array<f32, 256>; var<workgroup> sj: array<u32, 256>;
var<workgroup> sn: array<u32, 256>; var<workgroup> sp: array<f32, 256>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let V = info.x; let NB = info.y; let q = wg.y; let b = wg.x; let t = lid.x;
    let base = spec(q, 0u) * V;
    var mv = neg_inf(); var mi = 0u; var uv = neg_inf(); var ui = 0u; var nan = 0u; var pabs = 0.0;
    for (var k: u32 = 0u; k < 4u; k = k + 1u) {
        let i = b * 1024u + k * 256u + t;
        if (i < V) {
            let raw = logits[base + i];
            let yv = yval(q, i, raw); let y = yv.x;
            if (y != y || raw != raw) { nan = 1u; }
            else {
                if (better(y, i, mv, mi)) { mv = y; mi = i; }
                if (yv.y == 0.0) { if (better(y, i, uv, ui)) { uv = y; ui = i; } }
                // A banned token (-inf) is exact on both sides; only finite adjusted values carry error.
                else if (abs(y) <= 3.4028235e38) { pabs = max(pabs, max(abs(y), abs(raw))); }
            }
        }
    }
    sv[t] = mv; si[t] = mi; su[t] = uv; sj[t] = ui; sn[t] = nan; sp[t] = pabs;
    workgroupBarrier();
    for (var s: u32 = 128u; s > 0u; s = s >> 1u) {
        if (t < s) {
            if (better(sv[t + s], si[t + s], sv[t], si[t])) { sv[t] = sv[t + s]; si[t] = si[t + s]; }
            if (better(su[t + s], sj[t + s], su[t], sj[t])) { su[t] = su[t + s]; sj[t] = sj[t + s]; }
            sn[t] = sn[t] | sn[t + s]; sp[t] = max(sp[t], sp[t + s]);
        }
        workgroupBarrier();
    }
    if (t == 0u) {
        let o = q * (16u + NB * 8u) + 16u + b * 8u;
        work[o] = bitcast<u32>(sv[0]); work[o + 1u] = si[0]; work[o + 2u] = bitcast<u32>(su[0]); work[o + 3u] = sj[0];
        work[o + 4u] = sn[0]; work[o + 5u] = bitcast<u32>(sp[0]);
    }
}
"#;

/// K2: per request — combine the blocks; answer greedy rows; for sampled rows fix the max and eps, zero
/// the histogram, and mark the row pending.
const K2: &str = r#"
@group(0) @binding(0) var<storage,read> logits: array<f32>;
@group(0) @binding(1) var<storage,read> specs: array<u32>;
@group(0) @binding(2) var<storage,read> patches: array<u32>;
@group(0) @binding(3) var<storage,read_write> work: array<u32>;
@group(0) @binding(4) var<storage,read_write> hist: array<u32>;
@group(0) @binding(5) var<storage,read_write> outb: array<u32>;
@group(0) @binding(6) var<uniform> info: vec4<u32>;
__COMMON__
var<workgroup> sv: array<f32, 256>; var<workgroup> si: array<u32, 256>;
var<workgroup> su: array<f32, 256>; var<workgroup> sj: array<u32, 256>;
var<workgroup> sn: array<u32, 256>; var<workgroup> sp: array<f32, 256>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let V = info.x; let NB = info.y; let OUTN = info.z; let q = wg.y; let t = lid.x;
    let wb = q * (16u + NB * 8u);
    var mv = neg_inf(); var mi = 0u; var uv = neg_inf(); var ui = 0u; var nan = 0u; var pabs = 0.0;
    for (var b: u32 = t; b < NB; b = b + WG) {
        let o = wb + 16u + b * 8u;
        let v = bitcast<f32>(work[o]); let i = work[o + 1u];
        if (better(v, i, mv, mi)) { mv = v; mi = i; }
        let u = bitcast<f32>(work[o + 2u]); let j = work[o + 3u];
        if (better(u, j, uv, ui)) { uv = u; ui = j; }
        nan = nan | work[o + 4u]; pabs = max(pabs, bitcast<f32>(work[o + 5u]));
    }
    sv[t] = mv; si[t] = mi; su[t] = uv; sj[t] = ui; sn[t] = nan; sp[t] = pabs;
    workgroupBarrier();
    for (var s: u32 = 128u; s > 0u; s = s >> 1u) {
        if (t < s) {
            if (better(sv[t + s], si[t + s], sv[t], si[t])) { sv[t] = sv[t + s]; si[t] = si[t + s]; }
            if (better(su[t + s], sj[t + s], su[t], sj[t])) { su[t] = su[t + s]; sj[t] = sj[t + s]; }
            sn[t] = sn[t] | sn[t + s]; sp[t] = max(sp[t], sp[t + s]);
        }
        workgroupBarrier();
    }
    let ob = q * (8u + 2u * OUTN);
    let mode = spec(q, 1u); let pn = spec(q, 7u);
    let maxl = sv[0];
    let temp = bitcast<f32>(spec(q, 2u));
    // eps bounds |p_host / p_device - 1|. Unpenalized, a = (y - max)/T differs only by WGSL `/`
    // (<= 2.5 ulp) against the host's correctly rounded one; |a| <= 104 wherever p is not 0, so that
    // is <= 3·2^-24·104 = 1.9e-5 in p. The device `exp` is bounded by WGSL at (3 + 2|a|) ulp = 1.3e-5,
    // the host's by 1 ulp. 1e-4 covers those 3x over. A penalized value (and the max, if penalized)
    // adds the penalty division's 3 ulp of |y|, divided by T.
    let eps = 1.0e-4 + 6.0 * 5.9604645e-8 * (sp[0] + abs(maxl)) / temp * 1.01;
    if (t == 0u) {
        work[wb] = bitcast<u32>(maxl); work[wb + 1u] = bitcast<u32>(eps);
        for (var k: u32 = 0u; k < 8u; k = k + 1u) { outb[ob + k] = 0u; }
        if (sn[0] != 0u) { outb[ob] = 3u; }                                   // a NaN anywhere: fallback
        else if (mode == 0u && pn == 0u) { outb[ob] = 1u; outb[ob + 4u] = si[0]; }
        else if (mode == 0u) {
            // greedy with penalties: the last unpenalized maximum + every penalized token, raw
            let off = spec(q, 6u); let rb = spec(q, 0u) * V;
            outb[ob] = 2u; outb[ob + 1u] = pn + 1u;
            outb[ob + 8u] = sj[0]; outb[ob + 9u] = bitcast<u32>(logits[rb + sj[0]]);
            for (var k: u32 = 0u; k < pn; k = k + 1u) {
                let tk = patches[(off + k) * 6u];
                outb[ob + 10u + 2u * k] = tk; outb[ob + 11u + 2u * k] = bitcast<u32>(logits[rb + tk]);
            }
        }
        // sampled: a non-finite max takes the host's argmax branch; an eps too loose to filter with
        else if (!(abs(maxl) <= 3.4028235e38) || !(temp > 0.0) || eps > 1.0e-2) { outb[ob] = 3u; }
        else { outb[ob] = 0u; }                                                // pending
    }
    for (var b: u32 = t; b < 1024u; b = b + WG) { hist[q * 1024u + b] = 0u; }
}
"#;

/// K3: sampled rows — p = exp((y - max)/T) for every token, a 1024-bucket histogram of p's float bits
/// (monotone in p), and the per-block sum of p.
const K3: &str = r#"
@group(0) @binding(0) var<storage,read> logits: array<f32>;
@group(0) @binding(1) var<storage,read> specs: array<u32>;
@group(0) @binding(2) var<storage,read> patches: array<u32>;
@group(0) @binding(3) var<storage,read_write> work: array<u32>;
@group(0) @binding(4) var<storage,read_write> hist: array<atomic<u32>>;
@group(0) @binding(5) var<storage,read_write> prob: array<f32>;
@group(0) @binding(6) var<storage,read> outb: array<u32>;
@group(0) @binding(7) var<uniform> info: vec4<u32>;
__COMMON__
var<workgroup> lh: array<atomic<u32>, 1024>;
var<workgroup> ss: array<f32, 256>;
var<workgroup> st_s: u32;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let V = info.x; let NB = info.y; let OUTN = info.z; let q = wg.y; let b = wg.x; let t = lid.x;
    if (t == 0u) { st_s = outb[q * (8u + 2u * OUTN)]; }
    if (workgroupUniformLoad(&st_s) != 0u) { return; }           // not a pending sampled row
    let wb = q * (16u + NB * 8u);
    let maxl = bitcast<f32>(work[wb]); let temp = bitcast<f32>(spec(q, 2u));
    for (var k: u32 = t; k < 1024u; k = k + WG) { atomicStore(&lh[k], 0u); }
    workgroupBarrier();
    let base = spec(q, 0u) * V;
    var ps = 0.0;
    for (var k: u32 = 0u; k < 4u; k = k + 1u) {
        let i = b * 1024u + k * 256u + t;
        if (i < V) {
            let y = yval(q, i, logits[base + i]).x;
            let p = exp((y - maxl) / temp);
            prob[q * V + i] = p;
            atomicAdd(&lh[min(bitcast<u32>(p) >> 20u, 1023u)], 1u);
            ps = ps + p;
        }
    }
    ss[t] = ps;
    workgroupBarrier();
    for (var k: u32 = t; k < 1024u; k = k + WG) { let c = atomicLoad(&lh[k]); if (c > 0u) { atomicAdd(&hist[q * 1024u + k], c); } }
    for (var s: u32 = 128u; s > 0u; s = s >> 1u) {
        if (t < s) { ss[t] = ss[t] + ss[t + s]; }
        workgroupBarrier();
    }
    if (t == 0u) { work[wb + 16u + b * 8u + 7u] = bitcast<u32>(ss[0]); }
}
"#;

/// K4: sampled rows — choose the candidate threshold from the histogram, then keep every token that
/// can change the host's `Σ p` or that the sorted prefix can reach, written in index order.
const K4: &str = r#"
@group(0) @binding(0) var<storage,read> logits: array<f32>;
@group(0) @binding(1) var<storage,read> specs: array<u32>;
@group(0) @binding(2) var<storage,read> work: array<u32>;
@group(0) @binding(3) var<storage,read> hist: array<u32>;
@group(0) @binding(4) var<storage,read> prob: array<f32>;
@group(0) @binding(5) var<storage,read_write> outb: array<u32>;
@group(0) @binding(6) var<uniform> info: vec4<u32>;
const WG: u32 = 256u;
fn spec(q: u32, k: u32) -> u32 { return specs[q * 8u + k]; }
fn lower(b: u32) -> f32 { return bitcast<f32>(b << 20u); }
fn upper(b: u32) -> f32 { return bitcast<f32>((b + 1u) << 20u); }
// Can adding p (the host's value) change the host's running sum? The sum is at least the host's
// running max of what came before, itself at least lb·(1 - eps); an addition changes it only if
// p >= ulp(acc)/2. So drop p only when even its upper bound is below half an ulp of that lower bound.
// Below 2^-100 keep everything (the denormal range, where relative bounds say nothing).
fn relevant(p: f32, lb: f32, eps: f32) -> bool {
    let lbl = lb * (1.0 - eps);
    if (lbl < 7.888609e-31) { return true; }
    let half = bitcast<f32>((((bitcast<u32>(lbl) >> 23u) & 0xffu) - 24u) << 23u);
    return p * (1.0 + eps) + 1.1754944e-38 >= half;
}
var<workgroup> cm: array<f32, 256>;
var<workgroup> cc: array<u32, 256>;
var<workgroup> tau_s: f32;
var<workgroup> st_s: u32;
var<workgroup> tot_s: u32;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let V = info.x; let NB = info.y; let OUTN = info.z; let q = wg.y; let t = lid.x;
    let ob = q * (8u + 2u * OUTN);
    if (t == 0u) { st_s = outb[ob]; }
    if (workgroupUniformLoad(&st_s) != 0u) { return; }
    let wb = q * (16u + NB * 8u);
    let eps = bitcast<f32>(work[wb + 1u]);
    let top_p = bitcast<f32>(spec(q, 3u)); let top_k = spec(q, 4u); let min_p = bitcast<f32>(spec(q, 5u));
    if (t == 0u) {
        var sum = 0.0;
        for (var b: u32 = 0u; b < NB; b = b + 1u) { sum = sum + bitcast<f32>(work[wb + 16u + b * 8u + 7u]); }
        // bt: the lowest bucket kept. Walk down from the top and stop at the first bucket that satisfies
        // the mode, then keep ONE bucket more (a bucket spans 2^(1/8) > (1+eps)/(1-eps)), so everything
        // the criterion counted lies strictly above tau_hi on the host as well. The host re-checks.
        let hb = q * 1024u;
        var tail = 0.0;                                   // Σ count·upper: an upper bound of the mass below
        for (var b: u32 = 0u; b < 1024u; b = b + 1u) { tail = tail + f32(hist[hb + b]) * upper(b); }
        var bt = 0u; var cnt = 0u; var found = false;
        let mb = bitcast<u32>(min_p) >> 20u;
        for (var bb: u32 = 1024u; bb > 0u; bb = bb - 1u) {
            let b = bb - 1u;
            let c = hist[hb + b];
            cnt = cnt + c; tail = tail - f32(c) * upper(b);
            if (top_k > 0u) { if (cnt >= top_k) { bt = b; found = true; break; } }
            else if (min_p > 0.0) { if (b <= mb) { bt = b; found = true; break; } }
            else if (max(tail, 0.0) * 1.001 <= (1.0 - top_p) * sum * (1.0 - 2.0 * eps) - 1.0e-3 * sum) { bt = b; found = true; break; }
        }
        if (bt > 0u) { bt = bt - 1u; }
        var tau = lower(bt);
        if (!found) { tau = 0.0; }
        tau_s = tau;
        outb[ob + 2u] = bitcast<u32>(tau * (1.0 + eps) * 1.0000002 + 1.1754944e-38);
        outb[ob + 3u] = bitcast<u32>(eps);
    }
    workgroupBarrier();
    let tau = tau_s;
    // Only plain nucleus sampling normalizes by the FULL sum; top-k and min-p renormalize over their
    // survivors, which all sit above tau — so only there must the reduced row carry the sum's terms.
    let full_sum = top_k == 0u && !(min_p > 0.0);
    let ch = (V + WG - 1u) / WG;
    let i0 = min(t * ch, V); let i1 = min(i0 + ch, V);
    var m = 0.0;
    for (var i: u32 = i0; i < i1; i = i + 1u) { m = max(m, prob[q * V + i]); }
    cm[t] = m;
    workgroupBarrier();
    if (t == 0u) { var run = 0.0; for (var k: u32 = 0u; k < WG; k = k + 1u) { let v = cm[k]; cm[k] = run; run = max(run, v); } }
    workgroupBarrier();
    var lb = cm[t]; var n = 0u;
    for (var i: u32 = i0; i < i1; i = i + 1u) {
        let p = prob[q * V + i];
        if (p >= tau || (full_sum && relevant(p, lb, eps))) { n = n + 1u; }
        lb = max(lb, p);
    }
    cc[t] = n;
    workgroupBarrier();
    if (t == 0u) { var run = 0u; for (var k: u32 = 0u; k < WG; k = k + 1u) { let v = cc[k]; cc[k] = run; run = run + v; } tot_s = run; }
    let tot = workgroupUniformLoad(&tot_s);
    if (tot > OUTN) { if (t == 0u) { outb[ob] = 3u; } return; }
    var w = cc[t]; lb = cm[t];
    let rb = spec(q, 0u) * V;
    for (var i: u32 = i0; i < i1; i = i + 1u) {
        let p = prob[q * V + i];
        if (p >= tau || (full_sum && relevant(p, lb, eps))) {
            outb[ob + 8u + 2u * w] = i; outb[ob + 9u + 2u * w] = bitcast<u32>(logits[rb + i]);
            w = w + 1u;
        }
        lb = max(lb, p);
    }
    if (t == 0u) { outb[ob] = 2u; outb[ob + 1u] = tot; }
}
"#;

fn storage(ctx: &Context, data: &[u32]) -> wgpu::Buffer {
    use wgpu::util::DeviceExt;
    ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("sample"), contents: bytemuck::cast_slice(if data.is_empty() { &[0u32] } else { data }),
        usage: wgpu::BufferUsages::STORAGE,
    })
}

/// Select a token (or reduce the row) for each request, all requests in one set of dispatches and one
/// readback. `logits` is `[rows, vocab]`; each request names its row.
pub fn select_rows(logits: &Tensor, reqs: &[RowReq]) -> Vec<RowOut> {
    if reqs.is_empty() { return Vec::new(); }
    let ctx: &Arc<Context> = &logits.ctx;
    let lg = logits.contiguous();
    let v = *lg.shape.last().expect("logits rank >= 1");
    let nb = v.div_ceil(1024);
    let nq = reqs.len();
    let mut specs = Vec::with_capacity(nq * 8);
    let mut patches: Vec<u32> = Vec::new();
    for r in reqs {
        assert!((r.row + 1) * v <= lg.numel(), "row {} outside the logits", r.row);
        let off = patches.len() / 6;
        for w in r.patches.windows(2) { assert!(w[0].token < w[1].token, "patches must be sorted by token, one entry each"); }
        for p in &r.patches {
            let fl = (p.add.is_some() as u32) | ((p.sub.is_some() as u32) << 1) | ((p.rep.is_some() as u32) << 2) | ((p.sub2.is_some() as u32) << 3);
            patches.extend([p.token, fl, p.add.unwrap_or(0.0).to_bits(), p.sub.unwrap_or(0.0).to_bits(),
                            p.rep.unwrap_or(1.0).to_bits(), p.sub2.unwrap_or(0.0).to_bits()]);
        }
        specs.extend([r.row as u32, (r.temperature > 0.0) as u32, r.temperature.to_bits(), r.top_p.to_bits(),
                      r.top_k as u32, r.min_p.to_bits(), off as u32, r.patches.len() as u32]);
    }
    let any_sampled = reqs.iter().any(|r| r.temperature > 0.0);
    let specs_b = storage(ctx, &specs);
    let patch_b = storage(ctx, &patches);
    let work = empty(ctx, nq * (16 + nb * 8));
    let hist = empty(ctx, nq * NBUCKET);
    let outn = if any_sampled { OUT_MAX.min(v) } else { reqs.iter().map(|r| r.patches.len() + 1).max().unwrap_or(1) };
    let outb = empty(ctx, nq * (HDR + 2 * outn));
    let info = unibuf(ctx, &[v as u32, nb as u32, outn as u32, nq as u32]);
    let common = |k: &str| k.replace("__COMMON__", COMMON);
    let prob = any_sampled.then(|| empty(ctx, nq * v));
    crate::batch(ctx, || {
        run(ctx, &common(K1), "smp_max", &[lg.buf.as_ref(), &specs_b, &patch_b, &work, &info], (nb as u32, nq as u32, 1));
        run(ctx, &common(K2), "smp_reduce", &[lg.buf.as_ref(), &specs_b, &patch_b, &work, &hist, &outb, &info], (1, nq as u32, 1));
        if let Some(prob) = &prob {
            run(ctx, &common(K3), "smp_prob", &[lg.buf.as_ref(), &specs_b, &patch_b, &work, &hist, prob, &outb, &info], (nb as u32, nq as u32, 1));
            run(ctx, K4, "smp_select", &[lg.buf.as_ref(), &specs_b, &work, &hist, prob, &outb, &info], (1, nq as u32, 1));
        }
    });
    read_outs(ctx, &outb, nq, outn)
}

/// Map the output once and read, per request, the header and only the entries it holds.
fn read_outs(ctx: &Context, outb: &wgpu::Buffer, nq: usize, outn: usize) -> Vec<RowOut> {
    flush_batch(ctx);
    let stride = HDR + 2 * outn;
    let bytes = (nq * stride * 4) as u64;
    let staging = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("sample.staging"), size: bytes, usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut enc = ctx.device.create_command_encoder(&Default::default());
    enc.copy_buffer_to_buffer(outb, 0, &staging, 0, bytes);
    ctx.queue.submit([enc.finish()]);
    let (tx, rx) = flume::bounded(1);
    staging.slice(..).map_async(wgpu::MapMode::Read, move |r| { let _ = tx.send(r); });
    let _ = ctx.device.poll(wgpu::PollType::wait_indefinitely());
    rx.recv().expect("map callback").expect("map");
    let data = staging.slice(..).get_mapped_range().expect("mapped");
    let w: &[u32] = bytemuck::cast_slice(&data);
    let outs = (0..nq).map(|q| {
        let h = &w[q * stride..(q + 1) * stride];
        match h[0] {
            1 => RowOut::Id(h[4]),
            2 => {
                let n = h[1] as usize;
                let (mut ids, mut raw) = (Vec::with_capacity(n), Vec::with_capacity(n));
                for k in 0..n { ids.push(h[HDR + 2 * k]); raw.push(f32::from_bits(h[HDR + 2 * k + 1])); }
                RowOut::Reduced { ids, raw, tau_hi: f32::from_bits(h[2]), eps: f32::from_bits(h[3]) }
            }
            _ => RowOut::Fallback,
        }
    }).collect();
    drop(data);
    staging.unmap();
    outs
}
