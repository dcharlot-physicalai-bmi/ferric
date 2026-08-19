//! The encoder forward pass: patches in, latents out.
//!
//! Pre-norm blocks, bidirectional (unmasked) attention because an encoder may look both ways
//! within a window, SwiGLU feed-forward, and a linear head down to the quantizer's latent width.
//! The attention itself is [`ferric_tensor::nn::bidirectional_attention`] rather than a second
//! implementation of the same thing.
//!
//! ## What a test can and cannot decide here
//!
//! This is the first module in the crate that needs weights, so it is the first whose output
//! cannot be checked against its own inverse. Three things are still decidable, and all three are
//! checked below:
//!
//! 1. **The allocated parameters equal the arithmetic.** [`EncoderConfig::params`] claims a count;
//!    the tensors actually built must sum to it. This closes the loop between the sizing argument
//!    and the code, and it is the check that catches a layer silently not being built.
//! 2. **The pass is bit-exact across runs.** A tokenizer that returns different tokens for the same
//!    signal is unusable regardless of accuracy, and floating-point non-determinism on a GPU is
//!    real rather than theoretical.
//! 3. **It stays finite** on inputs that saturate, which is where a missing norm shows up.
//!
//! What is NOT decided: whether these are the right weights, or whether this tower matches the one
//! it is sized against. No reference weights were located, so nothing here is compared to them.

use crate::encoder::{sinusoidal_positions, EncoderConfig};
use crate::patch::PatchError;
use ferric_tensor::nn::bidirectional_attention;
use ferric_core::Context;
use ferric_tensor::Tensor;
use std::sync::Arc;

const EPS: f32 = 1e-5;

/// One pre-norm block.
pub struct Block {
    pub attn_norm: Tensor,
    pub wq: Tensor,
    pub wk: Tensor,
    pub wv: Tensor,
    pub wo: Tensor,
    pub ffn_norm: Tensor,
    pub w_gate: Tensor,
    pub w_up: Tensor,
    pub w_down: Tensor,
}

/// Every tensor the encoder owns.
pub struct EncoderWeights {
    pub cfg: EncoderConfig,
    /// `[d_model, patch_len]`, HF layout: a row per output feature.
    pub patch_embed: Tensor,
    pub blocks: Vec<Block>,
    pub out_norm: Tensor,
    /// `[latent_dim, d_model]`.
    pub latent_head: Tensor,
}

/// Deterministic, seeded fill. No RNG crate and no thread-local state, so the same seed produces
/// the same weights on every machine and a determinism test means what it says.
/// Re-exported for sibling modules that build their own towers with the same initialisation.
pub(crate) fn fill_pub(seed: u64, n: usize, scale: f32) -> Vec<f32> { fill(seed, n, scale) }

fn fill(seed: u64, n: usize, scale: f32) -> Vec<f32> {
    let mut s = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    (0..n)
        .map(|_| {
            // splitmix64
            s = s.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = s;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^= z >> 31;
            // Map to (-1, 1) then scale. 24 bits is exactly f32's mantissa.
            let u = ((z >> 40) as f32) / (1u32 << 24) as f32;
            (u * 2.0 - 1.0) * scale
        })
        .collect()
}

impl EncoderWeights {
    /// Build a tower with deterministic pseudo-random weights.
    ///
    /// These are **not trained**. The constructor exists so the shape of the computation can be
    /// exercised and its determinism and parameter count checked; it makes no claim about outputs.
    pub fn deterministic(ctx: &Arc<Context>, cfg: EncoderConfig, seed: u64) -> Result<Self, PatchError> {
        cfg.validate()?;
        let d = cfg.d_model;
        // 1/sqrt(fan_in), the usual scale, so activations neither vanish nor blow up across depth.
        let s = |fan_in: usize| 1.0 / (fan_in as f32).sqrt();
        let mut k = seed;
        let mut next = |n: usize, scale: f32, shape: &[usize]| {
            k = k.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            Tensor::from_vec(ctx, &fill(k, n, scale), shape)
        };
        let patch_embed = next(d * cfg.patch_len, s(cfg.patch_len), &[d, cfg.patch_len]);
        let mut blocks = Vec::with_capacity(cfg.n_layers);
        for _ in 0..cfg.n_layers {
            blocks.push(Block {
                attn_norm: Tensor::from_vec(ctx, &vec![1.0f32; d], &[d]),
                wq: next(d * d, s(d), &[d, d]),
                wk: next(d * d, s(d), &[d, d]),
                wv: next(d * d, s(d), &[d, d]),
                wo: next(d * d, s(d), &[d, d]),
                ffn_norm: Tensor::from_vec(ctx, &vec![1.0f32; d], &[d]),
                w_gate: next(cfg.d_ff * d, s(d), &[cfg.d_ff, d]),
                w_up: next(cfg.d_ff * d, s(d), &[cfg.d_ff, d]),
                w_down: next(d * cfg.d_ff, s(cfg.d_ff), &[d, cfg.d_ff]),
            });
        }
        Ok(Self {
            cfg,
            patch_embed,
            blocks,
            out_norm: Tensor::from_vec(ctx, &vec![1.0f32; d], &[d]),
            latent_head: next(cfg.latent_dim * d, s(d), &[cfg.latent_dim, d]),
        })
    }

    /// Sum of every allocated element. Compared against [`EncoderConfig::params`] in the tests, so
    /// the sizing argument and the built model cannot drift apart.
    pub fn allocated(&self) -> usize {
        let n = |t: &Tensor| t.shape.iter().product::<usize>();
        n(&self.patch_embed)
            + n(&self.out_norm)
            + n(&self.latent_head)
            + self
                .blocks
                .iter()
                .map(|b| {
                    n(&b.attn_norm)
                        + n(&b.wq)
                        + n(&b.wk)
                        + n(&b.wv)
                        + n(&b.wo)
                        + n(&b.ffn_norm)
                        + n(&b.w_gate)
                        + n(&b.w_up)
                        + n(&b.w_down)
                })
                .sum::<usize>()
    }

    /// `[t, patch_len]` of normalized patches → `[t, latent_dim]` of latents for the quantizer.
    pub fn forward(&self, ctx: &Arc<Context>, patches: &Tensor) -> Result<Tensor, PatchError> {
        let cfg = self.cfg;
        if patches.shape.len() != 2 || patches.shape[1] != cfg.patch_len {
            return Err(PatchError::Ragged { len: patches.shape.iter().product(), channels: cfg.patch_len });
        }
        let t = patches.shape[0];

        let mut x = patches.matmul_bt(&self.patch_embed);
        // Position is added, not concatenated, so the residual width never changes.
        let pe = Tensor::from_vec(ctx, &sinusoidal_positions(t, cfg.d_model)?, &[t, cfg.d_model]);
        x = x.add(&pe);

        for b in &self.blocks {
            let h = x.rmsnorm(&b.attn_norm, EPS);
            let (q, k, v) = (h.matmul_bt(&b.wq), h.matmul_bt(&b.wk), h.matmul_bt(&b.wv));
            // n_kv_heads == n_heads: no grouped-query sharing in an encoder this small.
            let a = bidirectional_attention(&q, &k, &v, cfg.n_heads, cfg.n_heads);
            x = x.add(&a.matmul_bt(&b.wo));

            let h = x.rmsnorm(&b.ffn_norm, EPS);
            let gate = h.matmul_bt(&b.w_gate).silu();
            let up = h.matmul_bt(&b.w_up);
            x = x.add(&gate.mul(&up).matmul_bt(&b.w_down));
        }

        Ok(x.rmsnorm(&self.out_norm, EPS).matmul_bt(&self.latent_head))
    }
}

impl EncoderWeights {
    /// Flatten to named tensors for [`crate::Weights`], in `params_flat` order.
    ///
    /// Names carry the layer index so a file is readable without knowing the order, and so a
    /// mismatched load fails by NAME rather than by silently taking the wrong tensor.
    pub fn to_weights(&self) -> crate::Weights {
        let mut w = crate::Weights::new();
        let names = Self::tensor_names(self.cfg.n_layers);
        for (n, t) in names.iter().zip(self.params_flat()) {
            w.push(n.clone(), &t.shape, pollster::block_on(t.to_vec()));
        }
        w
    }

    /// The canonical name for every tensor, in `params_flat` order.
    pub fn tensor_names(n_layers: usize) -> Vec<String> {
        let mut v = vec!["patch_embed".to_string()];
        for l in 0..n_layers {
            for part in ["attn_norm", "wq", "wk", "wv", "wo", "ffn_norm", "w_gate", "w_up", "w_down"] {
                v.push(format!("block.{l}.{part}"));
            }
        }
        v.push("out_norm".into());
        v.push("latent_head".into());
        v
    }

    /// Rebuild from a loaded file, checking every shape against the configuration.
    pub fn from_weights(
        ctx: &Arc<Context>,
        cfg: EncoderConfig,
        w: &crate::Weights,
    ) -> Result<Self, crate::StoreError> {
        let d = cfg.d_model;
        let g = |name: &str, want: &[usize]| w.tensor(ctx, name, want);
        let mut blocks = Vec::with_capacity(cfg.n_layers);
        for l in 0..cfg.n_layers {
            let p = |s: &str| format!("block.{l}.{s}");
            blocks.push(Block {
                attn_norm: g(&p("attn_norm"), &[d])?,
                wq: g(&p("wq"), &[d, d])?,
                wk: g(&p("wk"), &[d, d])?,
                wv: g(&p("wv"), &[d, d])?,
                wo: g(&p("wo"), &[d, d])?,
                ffn_norm: g(&p("ffn_norm"), &[d])?,
                w_gate: g(&p("w_gate"), &[cfg.d_ff, d])?,
                w_up: g(&p("w_up"), &[cfg.d_ff, d])?,
                w_down: g(&p("w_down"), &[d, cfg.d_ff])?,
            });
        }
        Ok(Self {
            cfg,
            patch_embed: g("patch_embed", &[d, cfg.patch_len])?,
            blocks,
            out_norm: g("out_norm", &[d])?,
            latent_head: g("latent_head", &[cfg.latent_dim, d])?,
        })
    }
}

/// The decoder tower: latents back to patches.
///
/// Mirrors [`EncoderWeights`] exactly except at the two ends — it takes the quantizer's latent
/// width in and emits a patch out, where the encoder does the reverse. That asymmetry is why
/// `EncoderConfig::params` cannot simply be doubled to size an autoencoder, and why the sizing
/// test in `encoder.rs` swaps those two terms explicitly rather than assuming symmetry.
pub struct DecoderWeights {
    pub cfg: EncoderConfig,
    /// `[d_model, latent_dim]`.
    pub latent_up: Tensor,
    pub blocks: Vec<Block>,
    pub out_norm: Tensor,
    /// `[patch_len, d_model]`.
    pub patch_head: Tensor,
}

impl DecoderWeights {
    pub fn deterministic(ctx: &Arc<Context>, cfg: EncoderConfig, seed: u64) -> Result<Self, PatchError> {
        cfg.validate()?;
        let d = cfg.d_model;
        let s = |fan_in: usize| 1.0 / (fan_in as f32).sqrt();
        let mut k = seed;
        let mut next = |n: usize, scale: f32, shape: &[usize]| {
            k = k.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            Tensor::from_vec(ctx, &fill(k, n, scale), shape)
        };
        let latent_up = next(d * cfg.latent_dim, s(cfg.latent_dim), &[d, cfg.latent_dim]);
        let mut blocks = Vec::with_capacity(cfg.n_layers);
        for _ in 0..cfg.n_layers {
            blocks.push(Block {
                attn_norm: Tensor::from_vec(ctx, &vec![1.0f32; d], &[d]),
                wq: next(d * d, s(d), &[d, d]),
                wk: next(d * d, s(d), &[d, d]),
                wv: next(d * d, s(d), &[d, d]),
                wo: next(d * d, s(d), &[d, d]),
                ffn_norm: Tensor::from_vec(ctx, &vec![1.0f32; d], &[d]),
                w_gate: next(cfg.d_ff * d, s(d), &[cfg.d_ff, d]),
                w_up: next(cfg.d_ff * d, s(d), &[cfg.d_ff, d]),
                w_down: next(d * cfg.d_ff, s(cfg.d_ff), &[d, cfg.d_ff]),
            });
        }
        Ok(Self {
            cfg,
            latent_up,
            blocks,
            out_norm: Tensor::from_vec(ctx, &vec![1.0f32; d], &[d]),
            patch_head: next(cfg.patch_len * d, s(d), &[cfg.patch_len, d]),
        })
    }

    /// Elements actually allocated. The decoder's own count, not the encoder's.
    pub fn allocated(&self) -> usize {
        let n = |t: &Tensor| t.shape.iter().product::<usize>();
        n(&self.latent_up)
            + n(&self.out_norm)
            + n(&self.patch_head)
            + self.blocks.iter().map(|b| {
                n(&b.attn_norm) + n(&b.wq) + n(&b.wk) + n(&b.wv) + n(&b.wo)
                    + n(&b.ffn_norm) + n(&b.w_gate) + n(&b.w_up) + n(&b.w_down)
            }).sum::<usize>()
    }

    /// `[t, latent_dim]` dequantized latents → `[t, patch_len]` reconstructed patches.
    pub fn forward(&self, ctx: &Arc<Context>, latents: &Tensor) -> Result<Tensor, PatchError> {
        let cfg = self.cfg;
        if latents.shape.len() != 2 || latents.shape[1] != cfg.latent_dim {
            return Err(PatchError::Ragged { len: latents.shape.iter().product(), channels: cfg.latent_dim });
        }
        let t = latents.shape[0];
        let mut x = latents.matmul_bt(&self.latent_up);
        let pe = Tensor::from_vec(ctx, &sinusoidal_positions(t, cfg.d_model)?, &[t, cfg.d_model]);
        x = x.add(&pe);
        for b in &self.blocks {
            let h = x.rmsnorm(&b.attn_norm, EPS);
            let (q, k, v) = (h.matmul_bt(&b.wq), h.matmul_bt(&b.wk), h.matmul_bt(&b.wv));
            let a = bidirectional_attention(&q, &k, &v, cfg.n_heads, cfg.n_heads);
            x = x.add(&a.matmul_bt(&b.wo));
            let h = x.rmsnorm(&b.ffn_norm, EPS);
            let gate = h.matmul_bt(&b.w_gate).silu();
            let up = h.matmul_bt(&b.w_up);
            x = x.add(&gate.mul(&up).matmul_bt(&b.w_down));
        }
        Ok(x.rmsnorm(&self.out_norm, EPS).matmul_bt(&self.patch_head))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Fsq, Patcher, RevIn};

    /// A GPU context, or a LOUD failure.
    ///
    /// The repository's older tests print "no GPU — skipping" and return, which reports `ok` for a
    /// test that never ran — a green board that means nothing. Here a missing context fails unless
    /// it is waived deliberately with `FERRIC_NO_GPU=1`, so skipping is an act rather than a
    /// default.
    fn ctx() -> Option<Arc<Context>> {
        match pollster::block_on(Context::new()) {
            Ok(c) => Some(Arc::new(c)),
            Err(e) => {
                if std::env::var("FERRIC_NO_GPU").is_ok() {
                    eprintln!("FERRIC_NO_GPU set; skipping deliberately ({e:?})");
                    None
                } else {
                    panic!("no GPU context ({e:?}). This test measures nothing without one. \
                            Set FERRIC_NO_GPU=1 to waive it on purpose.");
                }
            }
        }
    }

    fn tiny() -> EncoderConfig {
        EncoderConfig { patch_len: 8, d_model: 32, n_layers: 2, n_heads: 4, d_ff: 64, latent_dim: 5 }
    }

    /// THE LOOP-CLOSING CHECK. `params()` is an argument about size; this is the model that got
    /// built. A block silently not appended, or a matrix allocated at the wrong width, shows up
    /// here and nowhere else.
    #[test]
    fn allocated_parameters_equal_the_arithmetic() {
        let Some(ctx) = ctx() else { return };
        for cfg in [tiny(), EncoderConfig::signal_4m()] {
            let w = EncoderWeights::deterministic(&ctx, cfg, 7).unwrap();
            assert_eq!(
                w.allocated(),
                cfg.params().total(),
                "allocated tensors disagree with EncoderConfig::params for {cfg:?}"
            );
        }
    }

    #[test]
    fn forward_produces_one_latent_per_patch() {
        let Some(ctx) = ctx() else { return };
        let cfg = tiny();
        let w = EncoderWeights::deterministic(&ctx, cfg, 1).unwrap();
        let t = 12usize;
        let x = Tensor::from_vec(&ctx, &fill(3, t * cfg.patch_len, 1.0), &[t, cfg.patch_len]);
        let y = w.forward(&ctx, &x).unwrap();
        assert_eq!(y.shape, vec![t, cfg.latent_dim]);
    }

    /// Bit-exact, not approximately equal. A tokenizer that returns different token ids for the
    /// same signal is unusable however accurate it is, and this is the property a determinism
    /// receipt would later be asserting.
    #[test]
    fn forward_is_bit_exact_across_repeated_runs() {
        let Some(ctx) = ctx() else { return };
        let cfg = tiny();
        let w = EncoderWeights::deterministic(&ctx, cfg, 11).unwrap();
        let x = Tensor::from_vec(&ctx, &fill(5, 16 * cfg.patch_len, 1.0), &[16, cfg.patch_len]);
        let a = pollster::block_on(w.forward(&ctx, &x).unwrap().to_vec());
        for run in 0..4 {
            let b = pollster::block_on(w.forward(&ctx, &x).unwrap().to_vec());
            assert_eq!(a, b, "forward pass was not bit-exact on run {run}");
        }
    }

    /// Same seed, same weights; different seed, different weights. Without the first half the
    /// determinism test above is trivially satisfiable by a constant.
    #[test]
    fn the_seeded_initialiser_is_reproducible_and_not_constant() {
        let Some(ctx) = ctx() else { return };
        let cfg = tiny();
        let a = EncoderWeights::deterministic(&ctx, cfg, 42).unwrap();
        let b = EncoderWeights::deterministic(&ctx, cfg, 42).unwrap();
        let c = EncoderWeights::deterministic(&ctx, cfg, 43).unwrap();
        let g = |w: &EncoderWeights| pollster::block_on(w.latent_head.to_vec());
        assert_eq!(g(&a), g(&b), "same seed produced different weights");
        assert_ne!(g(&a), g(&c), "different seeds produced identical weights");
    }

    /// POSITION MUST ACTUALLY ENTER THE COMPUTATION, and nothing else here can tell.
    ///
    /// Unmasked attention is permutation-EQUIVARIANT: with no positional signal, reversing the
    /// patch order reverses the output exactly and changes nothing else. The model would be a bag
    /// of patches — it could describe *what* is in a window and never *when* — and every other test
    /// in this file still passes, because shapes, determinism, finiteness and parameter counts are
    /// all blind to it. Found by mutation: deleting `x = x.add(&pe)` broke no test until this one.
    ///
    /// So: reverse the input and assert the output is NOT the reversed original.
    #[test]
    fn reversing_the_patch_order_is_not_merely_reversing_the_output() {
        let Some(ctx) = ctx() else { return };
        let cfg = tiny();
        let w = EncoderWeights::deterministic(&ctx, cfg, 17).unwrap();
        let t = 10usize;
        let raw = fill(21, t * cfg.patch_len, 1.0);

        let fwd = pollster::block_on(
            w.forward(&ctx, &Tensor::from_vec(&ctx, &raw, &[t, cfg.patch_len])).unwrap().to_vec(),
        );

        // Same patches, reversed order.
        let mut rev_in = Vec::with_capacity(raw.len());
        for i in (0..t).rev() {
            rev_in.extend_from_slice(&raw[i * cfg.patch_len..(i + 1) * cfg.patch_len]);
        }
        let rev_out = pollster::block_on(
            w.forward(&ctx, &Tensor::from_vec(&ctx, &rev_in, &[t, cfg.patch_len])).unwrap().to_vec(),
        );

        // What a position-blind encoder would produce: the original latents, reversed.
        let mut equivariant = Vec::with_capacity(fwd.len());
        for i in (0..t).rev() {
            equivariant.extend_from_slice(&fwd[i * cfg.latent_dim..(i + 1) * cfg.latent_dim]);
        }

        let max_dev = rev_out
            .iter()
            .zip(&equivariant)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_dev > 1e-3,
            "the encoder is permutation-equivariant (max deviation {max_dev:e}): position is not \
             reaching the computation, so this is a bag of patches rather than a sequence"
        );
    }


    #[test]
    fn the_decoder_allocates_what_its_own_arithmetic_claims() {
        let Some(ctx) = ctx() else { return };
        for cfg in [tiny(), EncoderConfig::signal_4m()] {
            let d = DecoderWeights::deterministic(&ctx, cfg, 3).unwrap();
            assert_eq!(d.allocated(), cfg.decoder_params(), "decoder disagrees with decoder_params");
        }
    }

    /// The two towers ARE the same size, and the reason is worth pinning down because the
    /// plausible-sounding opposite is what I assumed first.
    ///
    /// Each tower holds BOTH end matrices — the encoder embeds a patch and emits a latent, the
    /// decoder lifts a latent and emits a patch — so the same two shapes appear in both, transposed.
    /// This is a construction, not a coincidence, so it is asserted across several configurations
    /// including ones where patch_len and latent_dim are far apart.
    #[test]
    fn the_two_towers_are_the_same_size_by_construction() {
        for cfg in [
            EncoderConfig::signal_4m(),
            tiny(),
            EncoderConfig { patch_len: 128, d_model: 64, n_layers: 3, n_heads: 8, d_ff: 128, latent_dim: 2 },
            EncoderConfig { patch_len: 2, d_model: 64, n_layers: 1, n_heads: 2, d_ff: 96, latent_dim: 64 },
        ] {
            assert_eq!(cfg.decoder_params(), cfg.params().total(), "asymmetry appeared for {cfg:?}");
        }
    }

    #[test]
    fn the_decoder_returns_one_patch_per_latent_and_is_bit_exact() {
        let Some(ctx) = ctx() else { return };
        let cfg = tiny();
        let d = DecoderWeights::deterministic(&ctx, cfg, 5).unwrap();
        let t = 9usize;
        let lat = Tensor::from_vec(&ctx, &fill(31, t * cfg.latent_dim, 1.0), &[t, cfg.latent_dim]);
        let a = d.forward(&ctx, &lat).unwrap();
        assert_eq!(a.shape, vec![t, cfg.patch_len]);
        let av = pollster::block_on(a.to_vec());
        for _ in 0..3 {
            assert_eq!(pollster::block_on(d.forward(&ctx, &lat).unwrap().to_vec()), av);
        }
        assert!(av.iter().all(|v| v.is_finite()));
        assert!(d.forward(&ctx, &Tensor::from_vec(&ctx, &vec![0.0; t * 3], &[t, 3])).is_err());
    }

    /// THE FULL AUTOENCODER PATH: signal → tokens → signal. Shapes and finiteness only.
    ///
    /// The reconstruction ERROR is deliberately not asserted. These weights are untrained, so the
    /// output is noise, and a threshold here would either be vacuous or would quietly become a
    /// quality claim about a model that has never seen data.
    #[test]
    fn the_full_autoencoder_path_returns_a_signal_of_the_right_length() {
        let Some(ctx) = ctx() else { return };
        let cfg = tiny();
        let enc = EncoderWeights::deterministic(&ctx, cfg, 1).unwrap();
        let dec = DecoderWeights::deterministic(&ctx, cfg, 2).unwrap();
        let q = Fsq::signal_15bit();

        let n = cfg.patch_len * 24;
        let raw: Vec<f32> = (0..n).map(|i| (i as f32 * 0.05).sin() * 4.0 + 1.5).collect();
        let rev = RevIn::fit(&raw, 1).unwrap();
        let p = Patcher::contiguous(cfg.patch_len).unwrap();
        let patches = p.patchify(&rev.apply(&raw).unwrap()).unwrap();
        let t = patches.len() / cfg.patch_len;

        let lat = pollster::block_on(
            enc.forward(&ctx, &Tensor::from_vec(&ctx, &patches, &[t, cfg.patch_len])).unwrap().to_vec(),
        );
        // Quantize, then dequantize: the round trip a decoder actually receives.
        let mut deq = Vec::with_capacity(lat.len());
        for i in 0..t {
            let code = q.quantize(&lat[i * cfg.latent_dim..(i + 1) * cfg.latent_dim]).unwrap();
            deq.extend(q.dequantize(&code).unwrap());
        }
        let out = pollster::block_on(
            dec.forward(&ctx, &Tensor::from_vec(&ctx, &deq, &[t, cfg.latent_dim])).unwrap().to_vec(),
        );
        assert_eq!(out.len(), t * cfg.patch_len);
        assert!(out.iter().all(|v| v.is_finite()));

        let signal = rev.invert(&p.unpatchify(&out).unwrap()).unwrap();
        assert_eq!(signal.len(), p.covered(raw.len()));
        assert!(signal.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn forward_stays_finite_on_saturating_input() {
        let Some(ctx) = ctx() else { return };
        let cfg = tiny();
        let w = EncoderWeights::deterministic(&ctx, cfg, 2).unwrap();
        for amp in [0.0f32, 1e-6, 1.0, 1e3, 1e6] {
            let x = Tensor::from_vec(&ctx, &vec![amp; 8 * cfg.patch_len], &[8, cfg.patch_len]);
            let y = pollster::block_on(w.forward(&ctx, &x).unwrap().to_vec());
            assert!(y.iter().all(|v| v.is_finite()), "non-finite latent at amplitude {amp}");
        }
    }

    #[test]
    fn a_mis_shaped_input_is_refused_rather_than_reinterpreted() {
        let Some(ctx) = ctx() else { return };
        let cfg = tiny();
        let w = EncoderWeights::deterministic(&ctx, cfg, 4).unwrap();
        let wrong = Tensor::from_vec(&ctx, &vec![0.0; 8 * 7], &[8, 7]);
        assert!(w.forward(&ctx, &wrong).is_err());
    }

    /// THE WHOLE PIPELINE, end to end: a raw multi-channel signal becomes token ids.
    /// normalize → patch → encode → quantize → pack. Every token must be a legal id.
    #[test]
    fn a_raw_signal_becomes_valid_token_ids() {
        let Some(ctx) = ctx() else { return };
        let cfg = tiny();
        let w = EncoderWeights::deterministic(&ctx, cfg, 9).unwrap();
        let q = Fsq::signal_15bit();
        assert_eq!(q.dim(), cfg.latent_dim);

        // One channel, 40 patches' worth, with a flat stretch in the middle so the front end's
        // stuck-sensor path is exercised by the end-to-end test too.
        let n = cfg.patch_len * 40;
        let raw: Vec<f32> = (0..n)
            .map(|i| if (n / 3..n / 2).contains(&i) { 3.3 } else { (i as f32 * 0.07).sin() * 2.0 + 3.3 })
            .collect();

        let rev = RevIn::fit(&raw, 1).unwrap();
        let norm = rev.apply(&raw).unwrap();
        let p = Patcher::contiguous(cfg.patch_len).unwrap();
        let patches = p.patchify(&norm).unwrap();
        let t = patches.len() / cfg.patch_len;

        let x = Tensor::from_vec(&ctx, &patches, &[t, cfg.patch_len]);
        let lat = pollster::block_on(w.forward(&ctx, &x).unwrap().to_vec());
        assert_eq!(lat.len(), t * cfg.latent_dim);
        assert!(lat.iter().all(|v| v.is_finite()), "encoder emitted a non-finite latent");

        let mut ids = Vec::with_capacity(t);
        for i in 0..t {
            let code = q.quantize(&lat[i * cfg.latent_dim..(i + 1) * cfg.latent_dim]).unwrap();
            let id = q.to_index(&code).unwrap();
            assert!(id < q.codebook_size());
            assert_eq!(q.from_index(id).unwrap(), code, "token {i} did not round-trip");
            ids.push(id);
        }
        assert_eq!(ids.len(), t);
    }
}

// ---------------------------------------------------------------------------------------------
// The same tower, differentiably.
//
// Two implementations of one computation is exactly where drift hides, so the Var path below is
// held to the Tensor path numerically: `the_var_tower_matches_the_tensor_tower` runs both over the
// same weights and requires agreement. Without that test this file would be the most likely place
// in the crate for a silent divergence, because both halves would keep passing their own tests.
// ---------------------------------------------------------------------------------------------

use ferric_tensor::autograd::Var;

/// Flat parameter order, shared by [`EncoderWeights::params_flat`] and [`forward_var`].
///
/// One ordering defined once. If the two ever disagree the tower trains on shuffled weights and
/// nothing reports it — the shapes still line up per layer.
fn linear_v(x: &Var, w: &Var) -> Var {
    x.matmul(&w.transpose(1, 0))
}

/// Unmasked multi-head attention over `[t, d]`, built from differentiable primitives.
fn attention_v(q: &Var, k: &Var, v: &Var, t: usize, d: usize, n_heads: usize, ctx: &Arc<Context>) -> Var {
    let dh = d / n_heads;
    let heads = |x: &Var| x.reshape(&[t, n_heads, dh]).transpose(1, 0).contiguous();
    let (qh, kh, vh) = (heads(q), heads(k), heads(v));
    let scale = Var::leaf(Tensor::from_vec(ctx, &[1.0 / (dh as f32).sqrt()], &[1]))
        .broadcast_to(&[n_heads, t, t]);
    let probs = qh.matmul(&kh.transpose(2, 1)).mul(&scale).softmax(2);
    probs.matmul(&vh).transpose(1, 0).contiguous().reshape(&[t, d])
}

impl EncoderWeights {
    /// Every parameter tensor in the order [`forward_var`] expects.
    pub fn params_flat(&self) -> Vec<Tensor> {
        let mut v = vec![self.patch_embed.clone()];
        for b in &self.blocks {
            v.extend([
                b.attn_norm.clone(), b.wq.clone(), b.wk.clone(), b.wv.clone(), b.wo.clone(),
                b.ffn_norm.clone(), b.w_gate.clone(), b.w_up.clone(), b.w_down.clone(),
            ]);
        }
        v.push(self.out_norm.clone());
        v.push(self.latent_head.clone());
        v
    }
}

/// The encoder forward pass over autograd variables.
///
/// `params` must be in [`EncoderWeights::params_flat`] order. Returns `[t, latent_dim]`.
pub fn forward_var(
    ctx: &Arc<Context>,
    cfg: EncoderConfig,
    params: &[Var],
    patches: &Var,
) -> Result<Var, PatchError> {
    // 1 patch embedding + 9 per block + out_norm + latent_head.
    let expect = 3 + 9 * cfg.n_layers;
    if params.len() != expect {
        return Err(PatchError::Ragged { len: params.len(), channels: expect });
    }
    if patches.value().shape.len() != 2 || patches.value().shape[1] != cfg.patch_len {
        return Err(PatchError::Ragged {
            len: patches.value().shape.iter().product(),
            channels: cfg.patch_len,
        });
    }
    let t = patches.value().shape[0];

    let mut x = linear_v(patches, &params[0]);
    let pe = Var::leaf(Tensor::from_vec(
        ctx,
        &sinusoidal_positions(t, cfg.d_model)?,
        &[t, cfg.d_model],
    ));
    x = x.add(&pe);

    for l in 0..cfg.n_layers {
        let p = &params[1 + 9 * l..1 + 9 * (l + 1)];
        let h = x.rmsnorm(&p[0], EPS);
        let (q, k, v) = (linear_v(&h, &p[1]), linear_v(&h, &p[2]), linear_v(&h, &p[3]));
        let a = attention_v(&q, &k, &v, t, cfg.d_model, cfg.n_heads, ctx);
        x = x.add(&linear_v(&a, &p[4]));

        let h = x.rmsnorm(&p[5], EPS);
        let gate = linear_v(&h, &p[6]).silu();
        let up = linear_v(&h, &p[7]);
        x = x.add(&linear_v(&gate.mul(&up), &p[8]));
    }
    let n = params.len();
    Ok(linear_v(&x.rmsnorm(&params[n - 2], EPS), &params[n - 1]))
}

impl DecoderWeights {
    /// Every parameter tensor in the order [`decoder_forward_var`] expects.
    pub fn params_flat(&self) -> Vec<Tensor> {
        let mut v = vec![self.latent_up.clone()];
        for b in &self.blocks {
            v.extend([
                b.attn_norm.clone(), b.wq.clone(), b.wk.clone(), b.wv.clone(), b.wo.clone(),
                b.ffn_norm.clone(), b.w_gate.clone(), b.w_up.clone(), b.w_down.clone(),
            ]);
        }
        v.push(self.out_norm.clone());
        v.push(self.patch_head.clone());
        v
    }
}

/// The decoder forward pass over autograd variables. `[t, latent_dim]` in, `[t, patch_len]` out.
pub fn decoder_forward_var(
    ctx: &Arc<Context>,
    cfg: EncoderConfig,
    params: &[Var],
    latents: &Var,
) -> Result<Var, PatchError> {
    let expect = 3 + 9 * cfg.n_layers;
    if params.len() != expect {
        return Err(PatchError::Ragged { len: params.len(), channels: expect });
    }
    if latents.value().shape.len() != 2 || latents.value().shape[1] != cfg.latent_dim {
        return Err(PatchError::Ragged {
            len: latents.value().shape.iter().product(),
            channels: cfg.latent_dim,
        });
    }
    let t = latents.value().shape[0];
    let mut x = linear_v(latents, &params[0]);
    let pe = Var::leaf(Tensor::from_vec(ctx, &sinusoidal_positions(t, cfg.d_model)?, &[t, cfg.d_model]));
    x = x.add(&pe);
    for l in 0..cfg.n_layers {
        let p = &params[1 + 9 * l..1 + 9 * (l + 1)];
        let h = x.rmsnorm(&p[0], EPS);
        let (q, k, v) = (linear_v(&h, &p[1]), linear_v(&h, &p[2]), linear_v(&h, &p[3]));
        let a = attention_v(&q, &k, &v, t, cfg.d_model, cfg.n_heads, ctx);
        x = x.add(&linear_v(&a, &p[4]));
        let h = x.rmsnorm(&p[5], EPS);
        let gate = linear_v(&h, &p[6]).silu();
        let up = linear_v(&h, &p[7]);
        x = x.add(&linear_v(&gate.mul(&up), &p[8]));
    }
    let n = params.len();
    Ok(linear_v(&x.rmsnorm(&params[n - 2], EPS), &params[n - 1]))
}

#[cfg(test)]
mod var_tests {
    use super::*;

    fn ctx() -> Option<Arc<Context>> {
        match pollster::block_on(Context::new()) {
            Ok(c) => Some(Arc::new(c)),
            Err(e) => {
                if std::env::var("FERRIC_NO_GPU").is_ok() {
                    eprintln!("FERRIC_NO_GPU set; skipping deliberately ({e:?})");
                    None
                } else {
                    panic!("no GPU context ({e:?}). Set FERRIC_NO_GPU=1 to waive this on purpose.");
                }
            }
        }
    }

    fn tiny() -> EncoderConfig {
        EncoderConfig { patch_len: 8, d_model: 32, n_layers: 2, n_heads: 4, d_ff: 64, latent_dim: 5 }
    }

    /// THE TEST THAT KEEPS THE TWO TOWERS ONE TOWER.
    ///
    /// Same weights, same input, two implementations. They must agree to floating-point tolerance,
    /// or the model that trains is not the model that runs — and both halves would keep passing
    /// their own shape, determinism and parameter tests while quietly computing different things.
    #[test]
    fn the_var_tower_matches_the_tensor_tower() {
        let Some(ctx) = ctx() else { return };
        for cfg in [tiny(), EncoderConfig { n_layers: 1, n_heads: 1, ..tiny() }] {
            let w = EncoderWeights::deterministic(&ctx, cfg, 13).unwrap();
            let t = 11usize;
            let xs = fill(77, t * cfg.patch_len, 1.0);
            let x = Tensor::from_vec(&ctx, &xs, &[t, cfg.patch_len]);

            let a = pollster::block_on(w.forward(&ctx, &x).unwrap().to_vec());
            let vars: Vec<Var> = w.params_flat().into_iter().map(Var::leaf).collect();
            let b = pollster::block_on(
                forward_var(&ctx, cfg, &vars, &Var::leaf(x.clone())).unwrap().value().to_vec(),
            );

            assert_eq!(a.len(), b.len());
            let worst = a.iter().zip(&b).map(|(p, q)| (p - q).abs()).fold(0.0f32, f32::max);
            assert!(worst < 2e-4, "the two towers disagree by {worst:e} for {cfg:?}");
        }
    }

    /// A gradient must reach EVERY parameter. A weight that receives none is a weight that never
    /// trains, and the loss still falls because the others compensate.

    /// The decoder's two implementations must agree for the same reason the encoder's must.
    #[test]
    fn the_var_decoder_matches_the_tensor_decoder() {
        let Some(ctx) = ctx() else { return };
        let cfg = tiny();
        let w = DecoderWeights::deterministic(&ctx, cfg, 31).unwrap();
        let t = 9usize;
        let lat = Tensor::from_vec(&ctx, &fill(88, t * cfg.latent_dim, 1.0), &[t, cfg.latent_dim]);
        let a = pollster::block_on(w.forward(&ctx, &lat).unwrap().to_vec());
        let vars: Vec<Var> = w.params_flat().into_iter().map(Var::leaf).collect();
        let b = pollster::block_on(
            decoder_forward_var(&ctx, cfg, &vars, &Var::leaf(lat.clone())).unwrap().value().to_vec(),
        );
        let worst = a.iter().zip(&b).map(|(p, q)| (p - q).abs()).fold(0.0f32, f32::max);
        assert!(worst < 2e-4, "the two decoders disagree by {worst:e}");
    }


    /// THE PROPERTY THAT MAKES PUBLISHED WEIGHTS MEAN ANYTHING.
    ///
    /// Save a model, load it back, and the TOKENS must be identical — not the loss, not the
    /// activations approximately, the actual token ids. A weight file that round-trips to a
    /// nearly-identical model produces a nearly-identical tokenizer, and "nearly identical" is
    /// exactly what the determinism receipt exists to refuse.
    #[test]
    fn a_model_loaded_from_a_file_produces_identical_tokens() {
        let Some(ctx) = ctx() else { return };
        let cfg = tiny();
        let a = EncoderWeights::deterministic(&ctx, cfg, 99).unwrap();

        let bytes = a.to_weights().to_bytes();
        let loaded = crate::Weights::from_bytes(&bytes).unwrap();
        let b = EncoderWeights::from_weights(&ctx, cfg, &loaded).unwrap();

        let t = 13usize;
        let x = Tensor::from_vec(&ctx, &fill(404, t * cfg.patch_len, 1.0), &[t, cfg.patch_len]);
        let la = pollster::block_on(a.forward(&ctx, &x).unwrap().to_vec());
        let lb = pollster::block_on(b.forward(&ctx, &x).unwrap().to_vec());
        assert_eq!(la, lb, "the loaded model computed different latents");

        let q = crate::Fsq::signal_15bit();
        let toks = |l: &[f32]| -> Vec<u32> {
            (0..t).map(|i| q.to_index(&q.quantize(&l[i * cfg.latent_dim..(i + 1) * cfg.latent_dim]).unwrap()).unwrap()).collect()
        };
        assert_eq!(toks(&la), toks(&lb), "the loaded model produced different tokens");
    }

    /// A file for a DIFFERENT configuration must be refused by shape, not loaded and run.
    #[test]
    fn weights_for_another_configuration_are_refused() {
        let Some(ctx) = ctx() else { return };
        let a = EncoderWeights::deterministic(&ctx, tiny(), 1).unwrap();
        let w = a.to_weights();
        let wider = EncoderConfig { d_model: 64, ..tiny() };
        assert!(EncoderWeights::from_weights(&ctx, wider, &w).is_err(), "a mismatched file loaded");
        let deeper = EncoderConfig { n_layers: 3, ..tiny() };
        assert!(EncoderWeights::from_weights(&ctx, deeper, &w).is_err(), "a missing layer went unnoticed");
    }

    #[test]
    fn tensor_names_match_the_flat_order_one_for_one() {
        let Some(ctx) = ctx() else { return };
        let cfg = tiny();
        let a = EncoderWeights::deterministic(&ctx, cfg, 2).unwrap();
        let names = EncoderWeights::tensor_names(cfg.n_layers);
        let flat = a.params_flat();
        assert_eq!(names.len(), flat.len(), "names and tensors disagree in count");
        let w = a.to_weights();
        for (n, t) in names.iter().zip(&flat) {
            assert_eq!(&w.get(n).unwrap().1, &t.shape, "{n} stored with the wrong shape");
        }
    }

    #[test]
    fn every_parameter_receives_a_gradient() {
        let Some(ctx) = ctx() else { return };
        let cfg = tiny();
        let w = EncoderWeights::deterministic(&ctx, cfg, 21).unwrap();
        let vars: Vec<Var> = w.params_flat().into_iter().map(Var::leaf).collect();
        let t = 7usize;
        let x = Var::leaf(Tensor::from_vec(&ctx, &fill(5, t * cfg.patch_len, 1.0), &[t, cfg.patch_len]));

        forward_var(&ctx, cfg, &vars, &x).unwrap().sum_all().backward();
        for (i, v) in vars.iter().enumerate() {
            let g = v.grad().unwrap_or_else(|| panic!("parameter {i} received no gradient"));
            let gv = pollster::block_on(g.to_vec());
            assert!(gv.iter().any(|x| x.abs() > 0.0), "parameter {i} received an all-zero gradient");
        }
    }

    #[test]
    fn the_flat_parameter_order_matches_what_the_forward_expects() {
        let Some(ctx) = ctx() else { return };
        let cfg = tiny();
        let w = EncoderWeights::deterministic(&ctx, cfg, 3).unwrap();
        let flat = w.params_flat();
        assert_eq!(flat.len(), 3 + 9 * cfg.n_layers);
        assert_eq!(flat.iter().map(|t| t.shape.iter().product::<usize>()).sum::<usize>(), w.allocated());
        let vars: Vec<Var> = flat.into_iter().map(Var::leaf).collect();
        // One short must be refused rather than silently reinterpreted.
        assert!(forward_var(&ctx, cfg, &vars[..vars.len() - 1],
                            &Var::leaf(Tensor::from_vec(&ctx, &vec![0.0; 8], &[1, 8]))).is_err());
    }
}
