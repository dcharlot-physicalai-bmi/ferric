//! **Nemotron-H** — NVIDIA's Mamba-2 / attention / MLP hybrid (`general.architecture = nemotron_h`).
//!
//! The first genuinely non-transformer runtime in this crate. Of 42 blocks in Nemotron-3-Nano-4B,
//! **four** carry attention; the rest are Mamba-2 state-space mixers and MLPs. Sequence mixing is a
//! recurrence, not a quadratic attention matrix, which is the whole point of the family.
//!
//! ## The schedule is DATA, not a pattern
//!
//! Which block is which is not derivable from the index. The file states it as two arrays:
//!
//! ```text
//! feed_forward_length  [0, 12544, 0, 12544, 0, ...]   0 => not an MLP block
//! attention.head_count_kv [0,0,...,8,...,8,...]       0 => not an attention block
//! ```
//!
//! A scalar accessor returns `Err` on an array and would fall back to a default, collapsing a
//! per-block schedule into a uniform one — every block becoming an MLP, or every block attention.
//! Nothing would error; the model would simply be a different model. `gemma4` carries the same trap
//! for its sliding-window pattern, and this reads the arrays for the same reason.
//!
//! ## The Mamba-2 mixer, reconciled against the file
//!
//! Every dimension below was checked against Nemotron-3-Nano-4B rather than taken from a paper:
//!
//! ```text
//! ssm_in.weight  [3136 -> 17504]   z 7680 | x 7680 | B,C 2048 | dt 96   (sums to 17504)
//! ssm_conv1d     [4, 9728]         conv covers xBC only  (7680 + 2048 = 9728), NOT z
//! ssm_norm       [960, 8]          grouped RMSNorm, 8 groups x 960 = 7680 = inner
//! ssm_a, ssm_d   [1, 96]           one per SSM head; head_dim = 7680/96 = 80
//! ```
//!
//! ## The conventions, READ OFF the reference graph
//!
//! `llama-eval-callback` on Nemotron-3-Nano-4B, 2 tokens. Every line below is a shape from that dump,
//! not a reading of the Mamba-2 paper — these are exactly the choices `ssm_scan`'s doc warns are
//! "a checkpoint convention" whose wrong value is fluent and wrong.
//!
//! ```text
//! MUL_MAT  ssm_in {3136 -> 17504}          split: z 7680 | xBC 9728 | dt 96
//! SSM_CONV over xBC only  {5,9728}x{4,9728}   z and dt BYPASS the conv
//! ADD      conv1d.bias
//! SILU                                     applied to ALL of xBC, so x is silu'd BEFORE the scan
//!          views of it: x {80,96,T}  B {128,8,T}  C {128,8,T}
//! ADD      dt + ssm_dt.bias  {96,T}        dt comes straight from ssm_in, NOT through the conv
//! SSM_SCAN
//! MUL      x * ssm_d {1,96}   ->  ADD "mamba2_y_add_d"
//! SWIGLU   (z, y_with_d)                   z is the FIRST operand: silu(z) * y
//! RESHAPE  {80,96,T} -> {960,8,T}          then RMS_NORM + MUL ssm_norm — the grouped norm
//! MUL_MAT  ssm_out {7680 -> 3136}  ->  ADD residual
//! ```
//!
//! ### Reconciling with Ferric's kernel
//!
//! `Tensor::ssm_scan` states its own contract in its bindings and it differs from ggml's in one way
//! that matters: `da` is **`exp(dt*A)` already formed** and `dt` is **already softplus'd**, so those
//! two are the caller's job; and the kernel adds `D*x` **internally** (`acc + dv * xv`), where ggml
//! does it externally as the `mamba2_y_add_d` node above. So the port passes silu'd `x` plus
//! `d_skip` and must NOT repeat that add — doing both would double the skip term, which is finite,
//! fluent and wrong.
//!
//! Group mapping is contiguous: the kernel computes `heads_per_group = n_head / n_group` and
//! `group = head / heads_per_group`, i.e. heads 0..11 → group 0 for the 4B's 96 heads over 8 groups.
//!
//! ### ⛔ There is NO RoPE, and the metadata says otherwise
//!
//! The file declares `rope.dimension_count = 78` and `rope.scaling.finetuned`, and **neither is
//! used**: the reference graph contains **zero** ROPE ops. Attention here is position-free — the
//! Mamba-2 layers carry position — and the attention blocks go straight
//! `MUL_MAT → RESHAPE → PERMUTE → FLASH_ATTN_EXT`.
//!
//! Trusting the metadata would have applied partial RoPE over 78 of 128 head dims and produced
//! fluent, wrong text with every shape assertion still passing. A declared key is evidence of what a
//! converter wrote, not of what the model does; only the graph settles it.
//!
//! Attention itself is plain GQA: Q `{128, 40}`, K/V `{128, 8}`, head_dim 128.
//!
//! ⚠ The remaining risk is **convention, not shape**: whether `ssm_a` is used directly or
//! exponentiated, whether `dt` takes a softplus and where its bias lands, and how B/C map onto the 8
//! groups. `Tensor::ssm_scan`'s own doc flags these as "a checkpoint convention — getting them wrong
//! yields finite, fluent, wrong output". They are therefore resolved against `llama-eval-callback`
//! per-op, never by reasoning, and until each one is pinned this architecture is registered
//! `Status::Parts` rather than `Verified`.
use ferric_core::Context;
use ferric_gguf::{GgufSource, Meta};
use ferric_tensor::{QMatrix, Tensor};
use std::sync::Arc;

/// What one block does. Read from the file's arrays, never inferred from the index.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlockKind {
    /// Mamba-2 state-space mixer.
    Ssm,
    /// Gated MLP.
    Ffn,
    /// Grouped-query attention. Four of 42 in the 4B.
    Attn,
}

pub struct Cfg {
    pub n_layer: usize,
    pub d: usize,
    pub n_vocab: usize,
    pub eps: f32,
    /// One entry per block, in order.
    pub kind: Vec<BlockKind>,
    /// Per-block MLP width; 0 where the block is not an MLP.
    pub n_ff: Vec<usize>,
    /// Per-block KV head count; 0 where the block is not attention.
    pub n_kv: Vec<usize>,
    pub n_head: usize,
    pub head_dim: usize,
    // ---- SSM ----
    pub ssm_inner: usize,
    pub ssm_state: usize,
    pub ssm_groups: usize,
    pub ssm_heads: usize,
    pub ssm_conv: usize,
    /// `ssm_inner / ssm_heads`.
    pub ssm_head_dim: usize,
}

/// Read a metadata value that may be a scalar OR a per-block array, as an array of `n`.
///
/// The distinction is the point: `Some(Meta::U)` broadcast to every block is correct for a uniform
/// model and catastrophic for a scheduled one, so an array is never silently reduced and a scalar is
/// never silently expanded past what the caller asked for.
fn per_block(md: &std::collections::HashMap<String, Meta>, key: &str, n: usize) -> Result<Vec<usize>, String> {
    match md.get(key) {
        Some(Meta::Arr(a)) => {
            let v: Vec<usize> = a.iter().map(|m| match m {
                Meta::U(x) => *x as usize, Meta::I(x) => *x as usize, _ => 0,
            }).collect();
            if v.len() != n {
                return Err(format!("{key} covers {} of {n} blocks", v.len()));
            }
            Ok(v)
        }
        Some(Meta::U(x)) => Ok(vec![*x as usize; n]),
        _ => Err(format!("missing {key}")),
    }
}

impl Cfg {
    pub fn from_gguf(g: &impl GgufSource) -> Result<Cfg, String> {
        let md = g.metadata();
        let u = |k: &str| match md.get(&format!("nemotron_h.{k}")) {
            Some(Meta::U(v)) => Ok(*v as usize), _ => Err(format!("missing nemotron_h.{k}")),
        };
        let f = |k: &str| match md.get(&format!("nemotron_h.{k}")) {
            Some(Meta::F(v)) => Ok(*v as f32), _ => Err(format!("missing nemotron_h.{k}")),
        };
        let n_layer = u("block_count")?;
        let n_ff = per_block(md, "nemotron_h.feed_forward_length", n_layer)?;
        let n_kv = per_block(md, "nemotron_h.attention.head_count_kv", n_layer)?;

        // Classify from the two arrays. A block is attention if it has KV heads, an MLP if it has a
        // width, and an SSM otherwise — and a block claiming BOTH is a schedule this loader cannot
        // represent, so it is refused rather than silently resolved by arm order.
        let mut kind = Vec::with_capacity(n_layer);
        for i in 0..n_layer {
            kind.push(match (n_kv[i] > 0, n_ff[i] > 0) {
                (true, true) => return Err(format!(
                    "block {i} declares both {} KV heads and an MLP width of {}; this loader gives a \
                     block exactly one role", n_kv[i], n_ff[i])),
                (true, false) => BlockKind::Attn,
                (false, true) => BlockKind::Ffn,
                (false, false) => BlockKind::Ssm,
            });
        }

        let ssm_inner = u("ssm.inner_size")?;
        let ssm_heads = u("ssm.time_step_rank")?;
        if ssm_heads == 0 || ssm_inner % ssm_heads != 0 {
            return Err(format!("ssm.inner_size {ssm_inner} is not divisible by ssm.time_step_rank {ssm_heads}"));
        }
        Ok(Cfg {
            d: u("embedding_length")?,
            n_vocab: u("vocab_size").or_else(|_| {
                g.tensor("token_embd.weight").map(|t| t.dims[1] as usize).ok_or("no vocab_size".to_string())
            })?,
            eps: f("attention.layer_norm_rms_epsilon")?,
            n_head: u("attention.head_count")?,
            head_dim: u("attention.key_length")?,
            ssm_state: u("ssm.state_size")?,
            ssm_groups: u("ssm.group_count")?,
            ssm_conv: u("ssm.conv_kernel")?,
            ssm_head_dim: ssm_inner / ssm_heads,
            ssm_inner, ssm_heads, n_layer, kind, n_ff, n_kv,
        })
    }

    /// Blocks of each kind, for a load-time receipt. A schedule that silently collapsed shows up here
    /// as 42/0/0 rather than as wrong text a thousand tokens later.
    pub fn schedule(&self) -> (usize, usize, usize) {
        let c = |k: BlockKind| self.kind.iter().filter(|x| **x == k).count();
        (c(BlockKind::Ssm), c(BlockKind::Ffn), c(BlockKind::Attn))
    }
}


/// Weights for one SSM block. `ssm_a`/`ssm_d` stay host-side: they are per-head scalars (96 floats)
/// consumed as `exp(dt*A)` per timestep, so uploading them as tensors would buy nothing.
struct SsmBlock {
    norm: Tensor,
    in_w: QMatrix,
    conv_w: Tensor,
    conv_b: Tensor,
    dt_b: Tensor,
    a: Vec<f32>,
    d: Tensor,
    gnorm: Tensor,
    out_w: QMatrix,
}

struct FfnBlock { norm: Tensor, up: QMatrix, down: QMatrix }

/// Four of 42 blocks. No RoPE, no positional term of any kind — see the module header.
struct AttnBlock { norm: Tensor, q: QMatrix, k: QMatrix, v: QMatrix, o: QMatrix }

pub struct NemotronH {
    ctx: Arc<Context>,
    pub cfg: Cfg,
    tok_embd: Vec<f32>,
    out_norm: Tensor,
    head: QMatrix,
    ssm: Vec<Option<SsmBlock>>,
    ffn: Vec<Option<FfnBlock>>,
    attn: Vec<Option<AttnBlock>>,
}

fn t1(ctx: &Arc<Context>, g: &impl GgufSource, name: &str) -> Result<Tensor, String> {
    let i = g.tensor(name).ok_or_else(|| format!("no {name}"))?;
    let n: usize = i.dims.iter().product::<u64>() as usize;
    Ok(Tensor::from_vec(ctx, &g.dequant(name)?, &[1, n]))
}

impl NemotronH {
    pub fn load(ctx: &Arc<Context>, g: &impl GgufSource) -> Result<NemotronH, String> {
        let cfg = Cfg::from_gguf(g)?;
        // Same helper every other runtime uses: native packed kernel when one exists for the stored
        // format, dense fallback otherwise. Reusing it rather than re-deriving keeps a new
        // architecture from quietly disagreeing with the rest of the crate about quant handling.
        let qm = |n: &str| crate::qwen35::qm(ctx, g, n);
        let (mut ssm, mut ffn, mut attn) = (Vec::new(), Vec::new(), Vec::new());
        for il in 0..cfg.n_layer {
            let n = |s: &str| format!("blk.{il}.{s}");
            match cfg.kind[il] {
                BlockKind::Ssm => {
                    // conv1d is stored [L, C]; depthwise_conv1d_causal wants [C, L]. The transpose is
                    // explicit because a silently mis-shaped kernel still convolves — it just mixes
                    // the wrong taps, which is finite, fluent and wrong.
                    let ci = g.tensor(&n("ssm_conv1d.weight")).ok_or("no ssm_conv1d.weight")?;
                    let (kl, ch) = (ci.dims[0] as usize, ci.dims[1] as usize);
                    let cw = Tensor::from_vec(ctx, &g.dequant(&n("ssm_conv1d.weight"))?, &[ch, kl]);
                    ssm.push(Some(SsmBlock {
                        norm: t1(ctx, g, &n("attn_norm.weight"))?,
                        in_w: qm(&n("ssm_in.weight"))?,
                        conv_w: cw,
                        conv_b: t1(ctx, g, &n("ssm_conv1d.bias"))?,
                        dt_b: t1(ctx, g, &n("ssm_dt.bias"))?,
                        a: g.dequant(&n("ssm_a"))?,
                        d: t1(ctx, g, &n("ssm_d"))?,
                        gnorm: t1(ctx, g, &n("ssm_norm.weight"))?,
                        out_w: qm(&n("ssm_out.weight"))?,
                    }));
                    ffn.push(None); attn.push(None);
                }
                BlockKind::Ffn => {
                    ffn.push(Some(FfnBlock {
                        norm: t1(ctx, g, &n("attn_norm.weight"))?,
                        up: qm(&n("ffn_up.weight"))?,
                        down: qm(&n("ffn_down.weight"))?,
                    }));
                    ssm.push(None); attn.push(None);
                }
                BlockKind::Attn => {
                    attn.push(Some(AttnBlock {
                        norm: t1(ctx, g, &n("attn_norm.weight"))?,
                        q: qm(&n("attn_q.weight"))?, k: qm(&n("attn_k.weight"))?,
                        v: qm(&n("attn_v.weight"))?, o: qm(&n("attn_output.weight"))?,
                    }));
                    ssm.push(None); ffn.push(None);
                }
            }
        }
        Ok(NemotronH {
            ctx: ctx.clone(),
            tok_embd: g.dequant("token_embd.weight")?,
            out_norm: t1(ctx, g, "output_norm.weight")?,
            head: qm("output.weight")?,
            cfg, ssm, ffn, attn,
        })
    }

    /// Full forward over `ids`, returning logits `[T, n_vocab]` plus per-op trace checkpoints.
    ///
    /// The trace exists from the first version rather than being added after a divergence, which is
    /// the ordering the BERT port earned the hard way: nine hypotheses against whole-model outputs,
    /// then a per-op trace that found the answer on its first run.
    pub fn forward_traced(&self, ids: &[u32]) -> Result<(Tensor, Vec<(String, Tensor)>), String> {
        let c = &self.cfg;
        let (d, t) = (c.d, ids.len());
        let mut e = vec![0f32; t * d];
        for (p, &id) in ids.iter().enumerate() {
            let src = (id as usize) * d;
            if src + d > self.tok_embd.len() { return Err(format!("token {id} outside the {}-row table", c.n_vocab)); }
            e[p * d..(p + 1) * d].copy_from_slice(&self.tok_embd[src..src + d]);
        }
        let mut h = Tensor::from_vec(&self.ctx, &e, &[t, d]);
        let trace = std::env::var("FERRIC_NEMO_TRACE").ok().as_deref() == Some("1");
        let mut tr: Vec<(String, Tensor)> = Vec::new();
        if trace { tr.push(("embd".into(), h.clone())); }

        for il in 0..c.n_layer {
            let out = match c.kind[il] {
                BlockKind::Ssm => {
                    let b = self.ssm[il].as_ref().ok_or("ssm block missing")?;
                    let n = h.rmsnorm(&b.norm, c.eps);
                    self.mamba2(b, &n, t)?
                }
                BlockKind::Ffn => {
                    let b = self.ffn[il].as_ref().ok_or("ffn block missing")?;
                    let n = h.rmsnorm(&b.norm, c.eps);
                    // ReLU² — Nemotron's MLP, not SwiGLU: one up projection, squared ReLU, one down.
                    // A gated MLP would need a gate tensor this checkpoint does not carry.
                    let up = n.matmul_q(&b.up);
                    let act = up.relu();
                    act.mul(&act).matmul_q(&b.down)
                }
                BlockKind::Attn => {
                    let b = self.attn[il].as_ref().ok_or("attn block missing")?;
                    let n = h.rmsnorm(&b.norm, c.eps);
                    self.attention(b, &n, t)?
                }
            };
            h = h.add(&out);
            if trace { tr.push((format!("blk{il}.{:?}", c.kind[il]), h.clone())); }
        }
        let h = h.rmsnorm(&self.out_norm, c.eps);
        Ok((h.contiguous().matmul_q(&self.head), tr))
    }

    pub fn forward(&self, ids: &[u32]) -> Result<Tensor, String> {
        self.forward_traced(ids).map(|(l, _)| l)
    }

    /// Grouped-query attention with **no positional encoding at all**. See the module header: the
    /// reference graph has zero ROPE ops despite the file declaring `rope.dimension_count`.
    fn attention(&self, b: &AttnBlock, h: &Tensor, _t: usize) -> Result<Tensor, String> {
        let c = &self.cfg;
        let nkv = c.n_kv.iter().copied().find(|&x| x > 0).ok_or("no KV head count")?;
        let q = h.matmul_q(&b.q);
        let k = h.matmul_q(&b.k);
        let v = h.matmul_q(&b.v);
        // The crate's shared causal attention: GQA, scaling and the mask in one place. Hand-rolling
        // it here would be a fourth copy of the same arithmetic to keep in agreement, and the mask is
        // the one term whose absence reads as perfectly fluent output.
        let y = ferric_tensor::nn::causal_attention(&q, &k, &v, c.n_head, nkv, 0.0);
        Ok(y.contiguous().matmul_q(&b.o))
    }

    /// One Mamba-2 mixer. Sequencing and every convention here was read off `llama-eval-callback`;
    /// see the module header for the dump it came from.
    fn mamba2(&self, b: &SsmBlock, h: &Tensor, t: usize) -> Result<Tensor, String> {
        let c = &self.cfg;
        let (inner, ng, ns, nh, hp) = (c.ssm_inner, c.ssm_groups, c.ssm_state, c.ssm_heads, c.ssm_head_dim);
        let zxbcdt = h.contiguous().matmul_q(&b.in_w);          // [T, 17504]
        let z   = zxbcdt.narrow(1, 0, inner).contiguous();
        let xbc = zxbcdt.narrow(1, inner, inner + 2 * ng * ns).contiguous();
        let dt  = zxbcdt.narrow(1, inner + inner + 2 * ng * ns, nh).contiguous();

        // conv over xBC ONLY — z and dt bypass it — then bias, then SILU over the WHOLE xBC, which is
        // why the x fed to the scan (and to the D skip) is the silu'd one.
        let xbc = b.conv_w_apply(&xbc, c.ssm_conv).add(&b.conv_b).silu();
        let x = xbc.narrow(1, 0, inner).contiguous();
        let bb = xbc.narrow(1, inner, ng * ns).contiguous();
        let cc = xbc.narrow(1, inner + ng * ns, ng * ns).contiguous();

        // dt: bias, then softplus. `ssm_scan` wants dt ALREADY softplus'd and `da` already formed as
        // exp(dt*A) — its bindings say so, and that is the half ggml keeps outside its own kernel.
        let dts = dt.add(&b.dt_b).softplus();
        let dtv = pollster::block_on(dts.to_vec());
        let mut da = vec![0f32; t * nh];
        for ti in 0..t { for hd in 0..nh { da[ti * nh + hd] = (dtv[ti * nh + hd] * b.a[hd]).exp(); } }
        let da = Tensor::from_vec(&self.ctx, &da, &[t, nh]);
        let h0 = Tensor::from_vec(&self.ctx, &vec![0f32; nh * hp * ns], &[nh * hp * ns]);

        // The kernel adds D*x internally (`acc + dv*xv`), where ggml emits it as `mamba2_y_add_d`.
        // Adding it again here would DOUBLE the skip term.
        let y = x.ssm_scan(&da, &dts, &bb, &cc, &b.d, &h0, nh, hp, ns, ng);

        // Gate then grouped norm, in that order — the dump is SWIGLU(z, y) followed by RMS_NORM.
        let gated = z.silu().mul(&y);
        // Grouped norm in TWO steps, because the weight is not the same width as the normalised row.
        // The reference is `RMS_NORM({960, 8, T})` — each 960-wide group normalised on its own, with
        // NO weight — followed by `MUL` against the full `{960, 8}` = 7680 tensor, whose values differ
        // per group. Passing the 7680-wide weight to `rmsnorm` over 960-wide rows reads the wrong
        // slice for seven groups out of eight; measured, that put block 0 at -175.8 against the
        // reference's -2.73.
        let gw = b.gnorm.broadcast_to(&[t, inner]);
        let normed = gated.reshape(&[t * ng, inner / ng])
            .rmsnorm_weightless(c.eps)
            .reshape(&[t, inner])
            .mul(&gw);
        Ok(normed.contiguous().matmul_q(&b.out_w))
    }
}

impl SsmBlock {
    fn conv_w_apply(&self, x: &Tensor, l: usize) -> Tensor { x.depthwise_conv1d_causal(&self.conv_w, l) }
}
