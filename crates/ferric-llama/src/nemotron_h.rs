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
//! ⚠ The remaining risk is **convention, not shape**: whether `ssm_a` is used directly or
//! exponentiated, whether `dt` takes a softplus and where its bias lands, and how B/C map onto the 8
//! groups. `Tensor::ssm_scan`'s own doc flags these as "a checkpoint convention — getting them wrong
//! yields finite, fluent, wrong output". They are therefore resolved against `llama-eval-callback`
//! per-op, never by reasoning, and until each one is pinned this architecture is registered
//! `Status::Parts` rather than `Verified`.
use ferric_gguf::{GgufSource, Meta};

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
