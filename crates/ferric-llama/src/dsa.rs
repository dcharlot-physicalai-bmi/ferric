//! **DeepSeek Sparse Attention** — the lightning indexer, its top-k selection, and the index cache.
//!
//! Dense attention over a 1M-token context is not affordable, so hyv4 scores every cached position
//! with a cheap auxiliary attention and lets only the best `top_k` of them into the real one. The
//! indexer is 32 heads of width 128 against a **single shared key per position**, and it is cheap
//! for two reasons that are easy to lose in a port:
//!
//! * its queries are read off the *existing* `q_a` LoRA latent, so it never projects the 6144-wide
//!   hidden state a second time;
//! * only 21 of 78 layers own indexer weights at all. The other 57 reuse a preceding layer's
//!   selection — see [`IndexSchedule`].
//!
//! There is **no softmax anywhere in the indexer**. The score exists to rank positions, nothing
//! more, and normalising it would be a plausible-looking change with no meaning.
//!
//! ## Three inputs from two places
//!
//! ⚠ The query comes from `qr`, the 2048-wide post-norm `q_a` latent. The key and the per-head
//! weights come from `cur`, the 6144-wide post-`attn_norm` hidden state. Routing all three from one
//! source is the obvious tidy-up and silently changes the model.
//!
//! ## The trap that a prefill test cannot see
//!
//! ⚠ `iw[h, t]` is indexed by the **query token**, not the key position. During a full-context
//! prefill `T == M`, so `iw[h,t]` and a wrongly written `iw[h,j]` have identical shapes and produce
//! a plausible score matrix. It diverges only when `T != M` — a decode step, or chunked prefill.
//! Every test here therefore uses `T != M`.

use ferric_tensor::Tensor;

/// Shapes and constants for one layer's lightning indexer.
#[derive(Debug, Clone, Copy)]
pub struct IndexerCfg {
    /// `attention.indexer.head_count` — 32 on the 770B checkpoint.
    pub n_heads: usize,
    /// `attention.indexer.key_length` — 128. Split as `head_dim − rope_dim` nope, `rope_dim` rope.
    pub head_dim: usize,
    /// RoPE width, shared with the main attention path (`rope.dimension_count`, 64).
    pub rope_dim: usize,
    /// `attention.indexer.top_k` — 2048. Positions outside the top `top_k` are masked out of the
    /// real attention entirely.
    pub top_k: usize,
    /// Epsilon for the key LayerNorm.
    pub eps: f32,
    /// Must match the main attention path's convention: the indexer and the MLA share one RoPE
    /// width and one frequency base, so a disagreement here is a disagreement about positions.
    pub rope_interleaved: bool,
}

impl IndexerCfg {
    /// Width of the non-positional half of an indexer head.
    pub fn nope_dim(&self) -> usize { self.head_dim - self.rope_dim }

    /// The single factor folded into the per-head weights.
    ///
    /// The reference divides the dot product by `sqrt(head_dim)` and the weights by
    /// `sqrt(n_heads)`. ReLU is positively homogeneous — `relu(ax) = a·relu(x)` for `a > 0` — so
    /// pulling both through to one multiply *after* the ReLU is exact, not an approximation. On the
    /// real shapes it is `1/sqrt(128·32) = 1/64`.
    pub fn weight_scale(&self) -> f32 { 1.0 / ((self.head_dim * self.n_heads) as f32).sqrt() }
}

/// One layer's indexer weights. Only the 21 `is_full` layers have any.
pub struct IndexerWeights {
    /// `indexer.attn_q_b` — `[n_heads * head_dim, q_lora_rank]`. Consumes the `q_a` latent.
    pub q_b: Tensor,
    /// `indexer.attn_k` — `[head_dim, hidden]`. ONE key head, shared by all 32 query heads.
    pub k: Tensor,
    /// `indexer.k_norm.weight` — `[head_dim]`.
    pub k_norm_w: Tensor,
    /// `indexer.k_norm.bias` — `[head_dim]`.
    ///
    /// ⚠ The presence of a bias is the tell. This is a **true LayerNorm** — mean-centred, biased
    /// variance, affine with a bias — and it is the only one in the whole hyv4 graph; everything
    /// else is RMSNorm. RMSNorm over the same vector has the same shape and the same parameter
    /// count if the bias is ignored, so nothing downstream can tell.
    pub k_norm_b: Tensor,
    /// `indexer.proj` — `[n_heads, hidden]`. One scalar per (query token, head), from the hidden
    /// state, with no dependence on the key position at all.
    pub proj: Tensor,
}

pub struct Indexer {
    pub cfg: IndexerCfg,
    pub w: IndexerWeights,
}

impl Indexer {
    pub fn new(cfg: IndexerCfg, w: IndexerWeights) -> Self { Self { cfg, w } }

    fn rope_split(&self, x: &Tensor, t: usize, heads: usize, cos: &Tensor, sin: &Tensor) -> Tensor {
        let (n, r) = (self.cfg.nope_dim(), self.cfg.rope_dim);
        let nope = x.narrow(2, 0, n).contiguous();
        let pe = x.narrow(2, n, r).contiguous();
        let pe = if self.cfg.rope_interleaved {
            pe.reshape(&[t * heads, r / 2, 2]).transpose(1, 2).contiguous().reshape(&[t, heads, r])
        } else { pe };
        let pe = pe.reshape(&[t, heads * r]).apply_rope_costable(cos, sin, heads, r).reshape(&[t, heads, r]);
        nope.cat(&pe, 2)
    }

    /// The indexer key for each of `cur`'s positions: `[T, head_dim]`, ready to be appended to the
    /// index cache.
    ///
    /// ⚠ The LayerNorm covers the **full 128-d key, before** the nope/rope split. Normalising after
    /// the rotation, or only the nope half, gives the same shape and a different key.
    pub fn keys(&self, cur: &Tensor, cos: &Tensor, sin: &Tensor) -> Tensor {
        let t = cur.shape[0];
        let dk = self.cfg.head_dim;
        let ik = cur
            .matmul_bt(&self.w.k)
            .layernorm(&self.w.k_norm_w, &self.w.k_norm_b, self.cfg.eps)
            .reshape(&[t, 1, dk]);
        self.rope_split(&ik, t, 1, cos, sin).reshape(&[t, dk])
    }

    /// The per-query indexer queries: `[T, n_heads, head_dim]`. `qr` is the **post-norm `q_a`
    /// latent** the main attention already computed, not the hidden state.
    pub fn queries(&self, qr: &Tensor, cos: &Tensor, sin: &Tensor) -> Tensor {
        let (t, h, dk) = (qr.shape[0], self.cfg.n_heads, self.cfg.head_dim);
        let iq = qr.matmul_bt(&self.w.q_b).reshape(&[t, h, dk]);
        self.rope_split(&iq, t, h, cos, sin)
    }

    /// The per-(token, head) weights, `[T, n_heads]`, already carrying [`IndexerCfg::weight_scale`].
    ///
    /// From `cur`, the hidden state — not from `qr`, and not from anything key-side.
    pub fn head_weights(&self, cur: &Tensor) -> Tensor {
        let w = cur.matmul_bt(&self.w.proj);
        w.mul(&w.scalar(self.cfg.weight_scale()))
    }

    /// `score[t, j] = Σ_h iw[t,h] · relu( q[t,h] · k[j] )` — `[T, M]`.
    ///
    /// `keys` is the whole index cache, `[M, head_dim]`, so `M` is the number of visible positions
    /// and need not equal `T`. The ReLU is on the dot product **only**: never on `iw`, which is
    /// signed, and never on the summed score.
    pub fn scores(&self, qr: &Tensor, cur: &Tensor, keys: &Tensor, cos: &Tensor, sin: &Tensor) -> Tensor {
        let (t, h, dk) = (qr.shape[0], self.cfg.n_heads, self.cfg.head_dim);
        let m = keys.shape[0];
        let q = self.queries(qr, cos, sin).permute(&[1, 0, 2]).contiguous(); // [h, t, dk]
        let k = keys.reshape(&[1, m, dk]).broadcast_to(&[h, m, dk]).contiguous();
        let dots = q.matmul(&k.transpose(2, 1)).relu();                     // [h, t, m]
        let iw = self.head_weights(cur).transpose(0, 1).contiguous()        // [h, t]
            .reshape(&[h, t, 1]).broadcast_to(&[h, t, m]);
        dots.mul(&iw).sum(&[0], false)                                      // [t, m]
    }
}

/// The additive attention mask implied by a score matrix: `0` for the `top_k` best visible
/// positions of each query, `−1e30` everywhere else.
///
/// `offset` is the absolute position of query row 0, so query `i` may see keys `0..=offset+i`.
///
/// The causal mask is applied **before** the selection, which is what makes the ordering safe:
/// ReLU output is non-negative, so every visible position scores `≥ 0 > −1e30` and can never be
/// outranked by a masked one. A token's own position is therefore always selected while `k ≥ 1`.
///
/// ⚠ The selection runs on the host. `M` can be a million, and a device-side top-k is the
/// optimisation this wants — but the ranking is the semantics and the kernel is not, so the
/// semantics land first and say so.
pub fn top_k_mask(scores: &Tensor, k: usize, offset: usize) -> Tensor {
    let (t, m) = (scores.shape[0], scores.shape[1]);
    let s = pollster::block_on(scores.to_vec());
    let mut mask = vec![-1e30f32; t * m];
    let mut idx: Vec<usize> = Vec::with_capacity(m);
    for i in 0..t {
        let last = (offset + i).min(m - 1);
        idx.clear();
        idx.extend(0..=last);
        let keep = k.min(idx.len());
        // `select_nth_unstable_by` is a partial sort: the ranking below the cut does not matter,
        // only membership does, and paying O(M log M) per query row for an order nothing reads
        // would be the whole cost of the thing this is meant to make cheap.
        if keep < idx.len() {
            idx.select_nth_unstable_by(keep - 1, |&a, &b| s[i * m + b].total_cmp(&s[i * m + a]));
            idx.truncate(keep);
        }
        for &j in &idx { mask[i * m + j] = 0.0 }
    }
    Tensor::from_vec(&scores.ctx_arc(), &mask, &[t, m])
}

/// Which layer's selection each layer uses.
///
/// Only `is_full` layers run an indexer; every other layer reuses **the most recent preceding**
/// full layer's index set. Not the nearest one, and never a following one — on the real pattern
/// `[0, 1, 5, 9, …, 77]` that makes layer 2 reuse layer **1**, not layer 0, because layer 1
/// overwrites layer 0's selection before layer 2 is reached.
///
/// The sharing groups are therefore `{0}`, `{1,2,3,4}`, `{5,6,7,8}`, …, `{73,74,75,76}`, `{77}` —
/// two singletons and nineteen fours.
#[derive(Debug, Clone)]
pub struct IndexSchedule {
    is_full: Vec<bool>,
    source: Vec<usize>,
}

impl IndexSchedule {
    /// `is_full` comes verbatim from `hyv4.attention.indexer.is_full`.
    ///
    /// Layer 0 must be full: nothing precedes it, so a model whose first layer only reuses has no
    /// selection to reuse and is rejected here rather than reading uninitialised state later.
    pub fn new(is_full: Vec<bool>) -> Result<Self, String> {
        if is_full.is_empty() { return Err("indexer.is_full is empty".into()) }
        if !is_full[0] {
            return Err("indexer.is_full[0] is false: layer 0 has no preceding layer to share a \
                        selection from, so this checkpoint cannot be run".into());
        }
        let mut source = Vec::with_capacity(is_full.len());
        let mut last = 0;
        for (il, &f) in is_full.iter().enumerate() {
            if f { last = il }
            source.push(last);
        }
        Ok(Self { is_full, source })
    }

    pub fn n_layers(&self) -> usize { self.is_full.len() }
    pub fn is_full(&self, il: usize) -> bool { self.is_full[il] }
    /// The layer whose index set layer `il` attends under.
    pub fn source(&self, il: usize) -> usize { self.source[il] }
    /// The layers that own indexer weights, and the only ones that touch the index cache.
    pub fn full_layers(&self) -> Vec<usize> {
        (0..self.n_layers()).filter(|&i| self.is_full[i]).collect()
    }
    /// Layers grouped by the selection they share.
    pub fn groups(&self) -> Vec<Vec<usize>> {
        self.full_layers().iter()
            .map(|&f| (0..self.n_layers()).filter(|&i| self.source[i] == f).collect())
            .collect()
    }
    /// Index-cache slots a naive per-layer allocation would reserve, against what is live.
    ///
    /// Only full layers ever write a key, so allocating `n_layers` slots wastes the difference. At
    /// 1M context and f16 keys that is 21 layers × 5.63 GiB rather than 78 × the same per-layer
    /// cost — the kind of over-allocation that looks like a memory requirement rather than a bug.
    pub fn live_cache_layers(&self) -> usize { self.full_layers().len() }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferric_core::Context;
    use std::sync::Arc;

    const H: usize = 4;
    const DK: usize = 8;
    const R: usize = 4;
    const HID: usize = 6;
    const LQ: usize = 5;
    // ⚠ T != M throughout. With T == M the query-indexed head weights and a key-indexed
    // misreading have identical shapes, and every test below would pass on the wrong one.
    const T: usize = 3;
    const M: usize = 7;

    fn cfg() -> IndexerCfg {
        IndexerCfg { n_heads: H, head_dim: DK, rope_dim: R, top_k: 4, eps: 1e-5, rope_interleaved: false }
    }

    macro_rules! ctx_or_skip {
        () => { match pollster::block_on(Context::new()) { Ok(c) => Arc::new(c), Err(_) => { eprintln!("no GPU context — skipping"); return } } };
    }

    fn rnd(ctx: &Arc<Context>, shape: &[usize], seed: u64) -> (Tensor, Vec<f32>) {
        let n: usize = shape.iter().product();
        let mut s = seed;
        let v: Vec<f32> = (0..n).map(|_| {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((s >> 33) as f32 / (1u64 << 31) as f32) - 1.0
        }).collect();
        (Tensor::from_vec(ctx, &v, shape), v)
    }

    fn indexer(ctx: &Arc<Context>) -> Indexer {
        Indexer::new(cfg(), IndexerWeights {
            q_b: rnd(ctx, &[H * DK, LQ], 1).0,
            k: rnd(ctx, &[DK, HID], 2).0,
            k_norm_w: rnd(ctx, &[DK], 3).0,
            k_norm_b: rnd(ctx, &[DK], 4).0,
            proj: rnd(ctx, &[H, HID], 5).0,
        })
    }
    fn get(t: &Tensor) -> Vec<f32> { pollster::block_on(t.to_vec()) }

    /// The scoring function, term by term, against host arithmetic — with `T != M`, so a head
    /// weight read by key position instead of query position cannot survive.
    #[test]
    fn the_score_matches_the_definition() {
        let ctx = ctx_or_skip!();
        let ix = indexer(&ctx);
        let (qr, _) = rnd(&ctx, &[T, LQ], 10);
        let (cur, _) = rnd(&ctx, &[T, HID], 11);
        let (keys, _) = rnd(&ctx, &[M, DK], 12);
        let (cos, _) = rnd(&ctx, &[T.max(M), R], 13);
        let (sin, _) = rnd(&ctx, &[T.max(M), R], 14);

        let q = get(&ix.queries(&qr, &cos, &sin));   // [T, H, DK]
        let iw = get(&ix.head_weights(&cur));        // [T, H]
        let kv = get(&keys);
        let sc = get(&ix.scores(&qr, &cur, &keys, &cos, &sin));
        assert_eq!(sc.len(), T * M);

        // Three readings of "where does the ReLU go", computed on the host from the same q, iw and
        // keys. Only the first is the format; the other two are the plausible misreadings, and the
        // guards below fail if this particular draw cannot tell them apart — otherwise the
        // comparison would be about the random numbers rather than about the code.
        let mut on_dot = vec![0.0f32; T * M];
        let mut on_score = vec![0.0f32; T * M];
        let mut no_relu = vec![0.0f32; T * M];
        for t in 0..T {
            for j in 0..M {
                let (mut a, mut c) = (0.0f32, 0.0f32);
                for h in 0..H {
                    let dot: f32 = (0..DK).map(|d| q[(t * H + h) * DK + d] * kv[j * DK + d]).sum();
                    a += iw[t * H + h] * dot.max(0.0);
                    c += iw[t * H + h] * dot;
                }
                on_dot[t * M + j] = a;
                on_score[t * M + j] = c.max(0.0);
                no_relu[t * M + j] = c;
            }
        }
        for t in 0..T { for j in 0..M {
            let (got, want) = (sc[t * M + j], on_dot[t * M + j]);
            assert!((got - want).abs() < 2e-5 * want.abs().max(1.0), "score[{t},{j}]: {got} vs {want}");
        }}
        let sep = |o: &[f32]| on_dot.iter().zip(o).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        assert!(sep(&on_score) > 1e-3, "this draw cannot distinguish ReLU-on-dot from ReLU-on-score");
        assert!(sep(&no_relu) > 1e-3, "this draw cannot distinguish ReLU-on-dot from no ReLU at all");
    }

    /// The scale is `1/sqrt(head_dim · n_heads)`, folded into the weights after the ReLU. That fold
    /// is exact only because ReLU is positively homogeneous — this checks the number, and that it
    /// is applied once rather than twice.
    ///
    /// ⛔ Testing the constant is not testing the fold. `the_score_matches_the_definition` builds its
    /// reference by calling `head_weights()`, so it is blind to what happens inside — a mutation
    /// applying the scale twice left it green. The second half here compares against the raw
    /// projection instead.
    #[test]
    fn the_weight_scale_is_folded_once() {
        let c = cfg();
        assert!((c.weight_scale() - 1.0 / (32.0f32).sqrt()).abs() < 1e-7);
        let real = IndexerCfg { n_heads: 32, head_dim: 128, rope_dim: 64, top_k: 2048, eps: 1e-5, rope_interleaved: false };
        assert!((real.weight_scale() - 0.015625).abs() < 1e-9, "the real shapes give exactly 1/64");
        assert_eq!(real.nope_dim(), 64);

        let ctx = ctx_or_skip!();
        let ix = indexer(&ctx);
        let (cur, _) = rnd(&ctx, &[T, HID], 60);
        let got = get(&ix.head_weights(&cur));
        let raw = get(&cur.matmul_bt(&ix.w.proj));
        assert_eq!(got.len(), T * H);
        for (i, (g, r)) in got.iter().zip(&raw).enumerate() {
            let want = r * c.weight_scale();
            assert!((g - want).abs() < 1e-6 * want.abs().max(1.0),
                    "head weight {i}: {g} vs {want} — the scale is applied {} times",
                    if (g - want * c.weight_scale()).abs() < 1e-6 { "twice" } else { "some other number of" });
        }
        assert!(raw.iter().any(|v| v.abs() > 1e-2), "the projection is ~zero; this proves nothing");
    }

    /// The key norm is a true LayerNorm — mean-centred, with a bias. An RMSNorm has the same shape
    /// and the same weight vector, so only the values separate them.
    #[test]
    fn the_key_norm_is_a_layernorm_not_an_rmsnorm() {
        let ctx = ctx_or_skip!();
        let ix = indexer(&ctx);
        // A constant-offset input: RMSNorm keeps the offset (it only divides), LayerNorm removes it.
        let cur = Tensor::from_vec(&ctx, &vec![1.0f32; T * HID], &[T, HID]);
        let (cos, sin) = (Tensor::from_vec(&ctx, &vec![1.0; T * R], &[T, R]),
                          Tensor::from_vec(&ctx, &vec![0.0; T * R], &[T, R]));
        let k = get(&ix.keys(&cur, &cos, &sin));

        // With cos=1,sin=0 the rotation is the identity, so the key is exactly the normed vector.
        // ⚠ That also means this test CANNOT see where the norm sits relative to the rotation —
        // `the_key_norm_runs_before_the_rotation` covers that, and had to, because a mutation moving
        // the norm after the RoPE left this one green.
        // (x−mean)/sqrt(var+eps)·w + b. Every row of `cur` is identical, so every key row is too.
        let w = get(&ix.w.k_norm_w);
        let b = get(&ix.w.k_norm_b);
        let pre = get(&cur.matmul_bt(&ix.w.k));
        for t in 0..T {
            let row = &pre[t * DK..(t + 1) * DK];
            let mu: f32 = row.iter().sum::<f32>() / DK as f32;
            let var: f32 = row.iter().map(|v| (v - mu) * (v - mu)).sum::<f32>() / DK as f32;
            let inv = 1.0 / (var + 1e-5).sqrt();
            for d in 0..DK {
                let want = (row[d] - mu) * inv * w[d] + b[d];
                assert!((k[t * DK + d] - want).abs() < 1e-4, "key[{t},{d}] is not a LayerNorm");
                // And it is NOT the RMSNorm of the same vector.
                let rms = (row.iter().map(|v| v * v).sum::<f32>() / DK as f32 + 1e-5).sqrt();
                let rmsnorm = row[d] / rms * w[d];
                if (want - rmsnorm).abs() > 1e-3 {
                    assert!((k[t * DK + d] - rmsnorm).abs() > 1e-4, "key[{t},{d}] matches RMSNorm");
                }
            }
        }
    }

    /// ⚠ The LayerNorm is applied BEFORE the rotation. Under the identity rotation the two orders
    /// coincide exactly, which is how the test above — written with `cos=1, sin=0` so the expected
    /// value stays hand-computable — passed a mutation that moved the norm after the RoPE. This one
    /// uses a real rotation and builds the wrong order out of the same public ops, so the only
    /// difference between the two arms is where the norm sits.
    #[test]
    fn the_key_norm_runs_before_the_rotation() {
        let ctx = ctx_or_skip!();
        let ix = indexer(&ctx);
        let (cur, _) = rnd(&ctx, &[T, HID], 40);
        // A genuine rotation: doubled tables, so entry i and i+half share an angle.
        let half = R / 2;
        let ang: Vec<f32> = (0..T * R).map(|n| 0.3 + 0.7 * ((n / R) as f32) + 0.11 * ((n % R % half) as f32)).collect();
        let cos = Tensor::from_vec(&ctx, &ang.iter().map(|a| a.cos()).collect::<Vec<_>>(), &[T, R]);
        let sin = Tensor::from_vec(&ctx, &ang.iter().map(|a| a.sin()).collect::<Vec<_>>(), &[T, R]);

        let right = get(&ix.keys(&cur, &cos, &sin));

        let n = DK - R;
        let pre = cur.matmul_bt(&ix.w.k).reshape(&[T, 1, DK]);
        let rotated = {
            let nope = pre.narrow(2, 0, n).contiguous();
            let pe = pre.narrow(2, n, R).contiguous().reshape(&[T, R])
                .apply_rope_costable(&cos, &sin, 1, R).reshape(&[T, 1, R]);
            nope.cat(&pe, 2).reshape(&[T, DK])
        };
        let wrong = get(&rotated.layernorm(&ix.w.k_norm_w, &ix.w.k_norm_b, 1e-5));

        let sep = right.iter().zip(&wrong).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        assert!(sep > 1e-3, "norm-before-rope and norm-after-rope agree ({sep}); the rotation is too \
                             close to the identity for this test to mean anything");
        // ...and confirm the rotation was real, or the line above is about nothing.
        let unrot = get(&pre.reshape(&[T, DK]));
        let moved = (0..T).flat_map(|t| (n..DK).map(move |d| (t, d)))
            .map(|(t, d)| (get(&rotated)[t * DK + d] - unrot[t * DK + d]).abs()).fold(0.0f32, f32::max);
        assert!(moved > 1e-3, "the RoPE table used here is effectively the identity");
    }

    /// The query path, pinned independently of the score.
    ///
    /// ⛔ `the_score_matches_the_definition` calls `queries()` to build its own reference, so it is
    /// blind to everything inside it — a mutation that negated the queries left it green. The three
    /// checks here need no host replica of Ferric's RoPE convention: an identity rotation must be a
    /// passthrough (pinning the reshape and that nothing else is applied), a real rotation must
    /// leave the nope half bit-identical (pinning WHICH half rotates and where the split is), and
    /// the rotation must preserve each `(i, i+half)` pair's norm (pinning that it is a rotation).
    #[test]
    fn the_query_path_rotates_only_the_positional_half() {
        let ctx = ctx_or_skip!();
        let ix = indexer(&ctx);
        let (qr, _) = rnd(&ctx, &[T, LQ], 50);
        let n = DK - R;
        let half = R / 2;

        let ones = Tensor::from_vec(&ctx, &vec![1.0f32; T * R], &[T, R]);
        let zeros = Tensor::from_vec(&ctx, &vec![0.0f32; T * R], &[T, R]);
        let ident = get(&ix.queries(&qr, &ones, &zeros));
        let raw = get(&qr.matmul_bt(&ix.w.q_b));
        for (i, (a, b)) in ident.iter().zip(&raw).enumerate() {
            assert!((a - b).abs() < 1e-5, "identity rotation is not a passthrough at {i}: {a} vs {b}");
        }

        let ang: Vec<f32> = (0..T * R).map(|k| 0.4 + 0.9 * ((k / R) as f32) + 0.17 * ((k % R % half) as f32)).collect();
        let cos = Tensor::from_vec(&ctx, &ang.iter().map(|a| a.cos()).collect::<Vec<_>>(), &[T, R]);
        let sin = Tensor::from_vec(&ctx, &ang.iter().map(|a| a.sin()).collect::<Vec<_>>(), &[T, R]);
        let rot = get(&ix.queries(&qr, &cos, &sin));

        let mut moved = 0.0f32;
        for t in 0..T { for h in 0..H {
            let base = (t * H + h) * DK;
            for d in 0..n {
                assert!((rot[base + d] - ident[base + d]).abs() < 1e-5,
                        "the nope half of head {h} moved at {d} — the split is in the wrong place");
            }
            for i in 0..half {
                let (a0, b0) = (ident[base + n + i], ident[base + n + i + half]);
                let (a1, b1) = (rot[base + n + i], rot[base + n + i + half]);
                moved = moved.max((a1 - a0).abs());
                let (r0, r1) = ((a0 * a0 + b0 * b0).sqrt(), (a1 * a1 + b1 * b1).sqrt());
                assert!((r0 - r1).abs() < 1e-4 * r0.max(1.0),
                        "pair ({i},{}) of head {h} is not norm-preserving: {r0} -> {r1}", i + half);
            }
        }}
        assert!(moved > 1e-3, "the rotation did nothing; the checks above would pass on a no-op");
    }

    /// The mask keeps exactly the top-k visible positions, and causality wins over score: a
    /// high-scoring future position must never be selected.
    #[test]
    fn selection_is_causal_first_then_top_k() {
        let ctx = ctx_or_skip!();
        // Row i's scores rise with j, so the best-scoring keys are always the FUTURE ones.
        let s: Vec<f32> = (0..T * M).map(|n| (n % M) as f32).collect();
        let scores = Tensor::from_vec(&ctx, &s, &[T, M]);
        let k = 2usize;
        let mask = get(&top_k_mask(&scores, k, 0));

        for i in 0..T {
            let kept: Vec<usize> = (0..M).filter(|&j| mask[i * M + j] == 0.0).collect();
            assert!(kept.iter().all(|&j| j <= i), "row {i} selected a future key: {kept:?}");
            assert_eq!(kept.len(), k.min(i + 1), "row {i} kept {kept:?}, want {} of them", k.min(i + 1));
            assert!(kept.contains(&i), "row {i} must always keep its own position");
            for j in 0..M {
                assert!(mask[i * M + j] == 0.0 || mask[i * M + j] <= -1e29, "mask must be 0 or −inf");
            }
        }
        // Highest visible scores win among the visible ones.
        let kept: Vec<usize> = (0..M).filter(|&j| mask[2 * M + j] == 0.0).collect();
        assert_eq!(kept, vec![1, 2], "row 2 should keep its two highest visible keys");
    }

    /// `offset` places query row 0 at an absolute position — the chunked-prefill and decode case,
    /// where the query block is shorter than the cache. Ignoring it makes every query think it is
    /// at the start of the sequence.
    #[test]
    fn the_offset_moves_the_causal_horizon() {
        let ctx = ctx_or_skip!();
        let scores = Tensor::from_vec(&ctx, &vec![1.0f32; 1 * M], &[1, M]);
        let mask = get(&top_k_mask(&scores, M, 4));
        let kept: Vec<usize> = (0..M).filter(|&j| mask[j] == 0.0).collect();
        assert_eq!(kept, vec![0, 1, 2, 3, 4], "a query at absolute position 4 sees keys 0..=4");
    }

    /// The reuse rule, against the real 78-layer pattern. Most recent PRECEDING full layer — so
    /// layer 2 shares layer 1's set, not layer 0's, and layer 76 shares layer 73's.
    #[test]
    fn the_schedule_reuses_the_most_recent_preceding_full_layer() {
        let mut full = vec![false; 78];
        full[0] = true; full[1] = true;
        for l in (5..78).step_by(4) { full[l] = true }
        let sch = IndexSchedule::new(full).unwrap();

        assert_eq!(sch.full_layers(),
                   vec![0, 1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45, 49, 53, 57, 61, 65, 69, 73, 77],
                   "the shipped is_full positions");
        assert_eq!(sch.live_cache_layers(), 21, "only 21 of 78 layers ever write an indexer key");

        assert_eq!(sch.source(0), 0);
        assert_eq!(sch.source(1), 1);
        assert_eq!(sch.source(2), 1, "layer 2 reuses layer 1, NOT layer 0");
        assert_eq!(sch.source(4), 1);
        assert_eq!(sch.source(5), 5);
        assert_eq!(sch.source(6), 5, "layer 6 reuses layer 5");
        assert_eq!(sch.source(76), 73);
        assert_eq!(sch.source(77), 77);
        for il in 0..78 {
            assert!(sch.source(il) <= il, "layer {il} looks forward");
            assert!(sch.is_full(sch.source(il)), "layer {il} sources from a non-full layer");
        }

        let g = sch.groups();
        assert_eq!(g.len(), 21);
        assert_eq!(g[0], vec![0], "layer 0 is a singleton");
        assert_eq!(g[1], vec![1, 2, 3, 4]);
        assert_eq!(g[20], vec![77], "layer 77 is a singleton");
        assert_eq!(g.iter().map(|v| v.len()).sum::<usize>(), 78);
        assert_eq!(g.iter().filter(|v| v.len() == 4).count(), 19, "nineteen fours and two singletons");
    }

    /// A checkpoint whose layer 0 only reuses has nothing to reuse. Refuse at construction rather
    /// than attend under an uninitialised selection.
    #[test]
    fn a_schedule_whose_first_layer_is_not_full_is_refused() {
        let mut full = vec![false; 8];
        full[1] = true;
        let e = IndexSchedule::new(full).unwrap_err();
        assert!(e.contains("is_full[0]"), "unhelpful error: {e}");
        assert!(IndexSchedule::new(vec![]).is_err());
    }
}
