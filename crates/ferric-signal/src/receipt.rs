//! A determinism receipt for a token stream.
//!
//! The claim this makes checkable is narrow and, as far as this review located, unoffered anywhere
//! else in the sensor-language landscape: **these tokens came from that signal, under this exact
//! configuration, and you can recompute it from the file.**
//!
//! Two digests, and the split is the whole design:
//!
//! - **`spec_digest`** covers everything that DETERMINES the token stream — the signal, the
//!   normalization statistics, the patching, the encoder shape and its weights, the quantizer
//!   levels, the vocabulary layout. Two runs with the same spec digest were asked the same question.
//! - **`token_digest`** covers what came out.
//!
//! So a mismatch is diagnosable rather than merely alarming. Same spec, different tokens means the
//! COMPUTATION diverged, and that is a question worth answering — a different GPU, a different
//! kernel, a reordered reduction. Different spec means you compared two different things and the
//! receipt says so before anyone theorises about numerics.
//!
//! **The platform is deliberately OUTSIDE the spec digest.** It is provenance. If it were inside,
//! every machine would produce a different spec digest and the comparison that matters — did the
//! same question get the same answer on different hardware — could never be posed.
//!
//! The output rides as flat key/value pairs, which is how `ferroscope` carries a receipt inside an
//! MCAP recording. Interoperating on the format rather than by linking the two build graphs is
//! deliberate: a shared serialization is cheaper to keep honest than a shared dependency.

use crate::encoder::EncoderConfig;
use crate::sha256::{hex, Sha256};

mod tag {
    pub const STR: u8 = 1;
    pub const U64: u8 = 2;
    pub const F32: u8 = 3;
    pub const PAIR: u8 = 4;
}

/// Length-prefixed and tagged. Without both, `("ab", "c")` and `("a", "bc")` hash identically and
/// two different specs quietly share a digest.
fn feed(h: &mut Sha256, t: u8, body: &[u8]) {
    h.update(&[t]);
    h.update(&(body.len() as u64).to_le_bytes());
    h.update(body);
}
fn feed_str(h: &mut Sha256, s: &str) {
    feed(h, tag::STR, s.as_bytes());
}
fn feed_u64(h: &mut Sha256, v: u64) {
    feed(h, tag::U64, &v.to_le_bytes());
}
fn feed_f32s(h: &mut Sha256, v: &[f32]) {
    let mut body = Vec::with_capacity(v.len() * 4);
    for x in v {
        // Bit pattern, not a formatted number: -0.0 and 0.0 are different inputs and a receipt
        // must not pretend otherwise. NaN payloads are preserved for the same reason.
        body.extend_from_slice(&x.to_bits().to_le_bytes());
    }
    feed(h, tag::F32, &body);
}
fn feed_pairs(h: &mut Sha256, pairs: &[(String, String)]) {
    let mut sorted: Vec<&(String, String)> = pairs.iter().collect();
    sorted.sort();
    let mut body = Vec::new();
    for (k, v) in sorted {
        body.extend_from_slice(&(k.len() as u64).to_le_bytes());
        body.extend_from_slice(k.as_bytes());
        body.extend_from_slice(&(v.len() as u64).to_le_bytes());
        body.extend_from_slice(v.as_bytes());
    }
    feed(h, tag::PAIR, &body);
}

/// Everything that determines a token stream.
#[derive(Debug, Clone, PartialEq)]
pub struct TokenSpec {
    /// Digest of the raw signal, before any normalization.
    pub signal_digest: String,
    /// Channel count of that signal.
    pub channels: u64,
    /// Per-channel means and scales actually used. These are measured from the data, so two runs
    /// that normalized differently produced different tokens for the same samples.
    pub revin_mean: Vec<f32>,
    pub revin_scale: Vec<f32>,
    pub patch_len: u64,
    pub stride: u64,
    /// Encoder shape, flattened.
    pub encoder: Vec<(String, String)>,
    /// Digest of the encoder weights. A run that swapped weights is a different run.
    pub weights_digest: String,
    /// FSQ level list.
    pub fsq_levels: Vec<u64>,
    /// Text vocabulary length, which fixes where signal ids begin.
    pub text_vocab: u64,
    pub build: String,
}

impl TokenSpec {
    pub fn from_parts(
        signal_digest: impl Into<String>,
        channels: usize,
        revin: (&[f32], &[f32]),
        patch_len: usize,
        stride: usize,
        cfg: &EncoderConfig,
        weights_digest: impl Into<String>,
        fsq_levels: &[u32],
        text_vocab: u32,
    ) -> Self {
        Self {
            signal_digest: signal_digest.into(),
            channels: channels as u64,
            revin_mean: revin.0.to_vec(),
            revin_scale: revin.1.to_vec(),
            patch_len: patch_len as u64,
            stride: stride as u64,
            encoder: vec![
                ("patch_len".into(), cfg.patch_len.to_string()),
                ("d_model".into(), cfg.d_model.to_string()),
                ("n_layers".into(), cfg.n_layers.to_string()),
                ("n_heads".into(), cfg.n_heads.to_string()),
                ("d_ff".into(), cfg.d_ff.to_string()),
                ("latent_dim".into(), cfg.latent_dim.to_string()),
            ],
            weights_digest: weights_digest.into(),
            fsq_levels: fsq_levels.iter().map(|&l| l as u64).collect(),
            text_vocab: text_vocab as u64,
            build: env!("CARGO_PKG_VERSION").to_string(),
        }
    }

    pub fn digest(&self) -> String {
        let mut h = Sha256::new();
        feed_str(&mut h, "ferric-signal.TokenSpec.v1");
        feed_str(&mut h, &self.signal_digest);
        feed_u64(&mut h, self.channels);
        feed_f32s(&mut h, &self.revin_mean);
        feed_f32s(&mut h, &self.revin_scale);
        feed_u64(&mut h, self.patch_len);
        feed_u64(&mut h, self.stride);
        feed_pairs(&mut h, &self.encoder);
        feed_str(&mut h, &self.weights_digest);
        for l in &self.fsq_levels {
            feed_u64(&mut h, *l);
        }
        feed_u64(&mut h, self.text_vocab);
        feed_str(&mut h, &self.build);
        hex(&h.finish())
    }
}

/// A recomputable claim about a token stream.
#[derive(Debug, Clone, PartialEq)]
pub struct TokenReceipt {
    pub spec: TokenSpec,
    pub spec_digest: String,
    pub token_digest: String,
    pub tokens: u64,
    /// Latents that were not finite before quantization. Non-zero is a finding, not a footnote:
    /// FSQ clamps, so a NaN latent still yields a legal-looking token id.
    pub non_finite_latents: u64,
    /// Provenance only, deliberately outside [`TokenSpec::digest`].
    pub platform: String,
}

impl TokenReceipt {
    pub fn new(spec: TokenSpec, ids: &[u32], non_finite_latents: u64, platform: impl Into<String>) -> Self {
        let mut h = Sha256::new();
        feed_str(&mut h, "ferric-signal.TokenStream.v1");
        feed_u64(&mut h, ids.len() as u64);
        let mut body = Vec::with_capacity(ids.len() * 4);
        for id in ids {
            body.extend_from_slice(&id.to_le_bytes());
        }
        feed(&mut h, tag::U64, &body);
        Self {
            spec_digest: spec.digest(),
            spec,
            token_digest: hex(&h.finish()),
            tokens: ids.len() as u64,
            non_finite_latents,
            platform: platform.into(),
        }
    }

    /// Flat key/value form, the shape `ferroscope` carries inside an MCAP recording.
    pub fn to_pairs(&self) -> Vec<(String, String)> {
        let mut kv = vec![
            ("ferric-signal.version".into(), "1".into()),
            ("spec_digest".into(), self.spec_digest.clone()),
            ("token_digest".into(), self.token_digest.clone()),
            ("tokens".into(), self.tokens.to_string()),
            ("non_finite_latents".into(), self.non_finite_latents.to_string()),
            ("platform".into(), self.platform.clone()),
            ("signal_digest".into(), self.spec.signal_digest.clone()),
            ("channels".into(), self.spec.channels.to_string()),
            ("patch_len".into(), self.spec.patch_len.to_string()),
            ("stride".into(), self.spec.stride.to_string()),
            ("weights_digest".into(), self.spec.weights_digest.clone()),
            ("text_vocab".into(), self.spec.text_vocab.to_string()),
            ("build".into(), self.spec.build.clone()),
            (
                "fsq_levels".into(),
                self.spec.fsq_levels.iter().map(|l| l.to_string()).collect::<Vec<_>>().join(","),
            ),
        ];
        for (k, v) in &self.spec.encoder {
            kv.push((format!("encoder.{k}"), v.clone()));
        }
        kv
    }
}

/// What comparing two receipts actually tells you.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Agreement {
    /// Same question, same answer.
    Identical,
    /// Same question, different answer. The computation diverged: a question, not a verdict.
    ComputationDiverged,
    /// Different question. Comparing the outputs would be meaningless.
    DifferentSpec,
}

/// Compare two receipts.
///
/// The order matters: a spec mismatch is reported FIRST, because "the tokens differ" is not a
/// finding when the two runs were not asked the same thing, and reporting it as one sends people
/// hunting for numerical causes that do not exist.
pub fn agree(a: &TokenReceipt, b: &TokenReceipt) -> Agreement {
    agreement(&a.spec_digest, &a.token_digest, &b.spec_digest, &b.token_digest)
}

/// The same rule, over digests alone.
///
/// A verifier reading two receipts out of two files has the digests and nothing else, and it must
/// reach the identical verdict a caller holding both `TokenReceipt` values would. Keeping the
/// ordering in ONE function is what stops the command line and the library from drifting into two
/// slightly different definitions of agreement.
pub fn agreement(spec_a: &str, token_a: &str, spec_b: &str, token_b: &str) -> Agreement {
    if spec_a != spec_b {
        Agreement::DifferentSpec
    } else if token_a != token_b {
        Agreement::ComputationDiverged
    } else {
        Agreement::Identical
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sha256::sha256;

    fn spec() -> TokenSpec {
        TokenSpec::from_parts(
            hex(&sha256(b"a signal")),
            2,
            (&[1.0, 2.0][..], &[0.5, 0.25][..]),
            16,
            16,
            &EncoderConfig::signal_4m(),
            hex(&sha256(b"weights")),
            &[8, 8, 8, 8, 8],
            32_000,
        )
    }

    /// THE CONCATENATION AMBIGUITY, tested where it can actually bite.
    ///
    /// This is the classic way a digest silently collides, and my first version of this test could
    /// not have caught it: I varied two fields that are not adjacent in `digest()`, and with ASCII
    /// bodies the TAG BYTE already separates them (`01 61 62 01 63` vs `01 61 01 62 63`). Removing
    /// the length prefix broke nothing, which is how I learned the test was decorative.
    ///
    /// The prefix earns its place when a body CONTAINS the tag byte. Then
    /// `("\x01", "x")` and `("", "\x01x")` both feed `01 01 01 78` and collide exactly.
    /// Digest inputs here are hex strings today, but a spec field is a `String` and nothing stops a
    /// future one carrying arbitrary bytes.
    #[test]
    fn two_field_sequences_cannot_feed_identical_bytes() {
        fn two(a: &str, b: &str) -> String {
            let mut h = Sha256::new();
            feed_str(&mut h, a);
            feed_str(&mut h, b);
            hex(&h.finish())
        }
        assert_ne!(two("\u{1}", "x"), two("", "\u{1}x"), "a field boundary was lost");
        assert_ne!(two("ab", "c"), two("a", "bc"));
        assert_ne!(two("", "a"), two("a", ""));
    }

    #[test]
    fn the_same_spec_digests_identically_every_time() {
        let d = spec().digest();
        for _ in 0..16 {
            assert_eq!(spec().digest(), d);
        }
    }

    /// Every field must be load-bearing. A field that does not change the digest is a field a run
    /// can differ on while claiming to be the same run.
    #[test]
    fn every_field_changes_the_digest() {
        let base = spec().digest();
        let mut cases: Vec<(&str, TokenSpec)> = Vec::new();
        let mut s = spec(); s.signal_digest = "other".into(); cases.push(("signal", s));
        let mut s = spec(); s.channels = 3; cases.push(("channels", s));
        let mut s = spec(); s.revin_mean[0] = 1.5; cases.push(("revin mean", s));
        let mut s = spec(); s.revin_scale[1] = 0.26; cases.push(("revin scale", s));
        let mut s = spec(); s.patch_len = 32; cases.push(("patch_len", s));
        let mut s = spec(); s.stride = 8; cases.push(("stride", s));
        let mut s = spec(); s.encoder[1].1 = "512".into(); cases.push(("encoder d_model", s));
        let mut s = spec(); s.weights_digest = "other".into(); cases.push(("weights", s));
        let mut s = spec(); s.fsq_levels[0] = 5; cases.push(("fsq levels", s));
        let mut s = spec(); s.text_vocab = 50_000; cases.push(("text vocab", s));
        let mut s = spec(); s.build = "9.9.9".into(); cases.push(("build", s));
        for (name, s) in cases {
            assert_ne!(s.digest(), base, "changing {name} did not change the spec digest");
        }
    }

    /// Floats are digested as BIT PATTERNS, not as formatted text.
    ///
    /// Signed zero alone does not prove this — `Display` already prints `0` and `-0` differently,
    /// so a text-formatting implementation passes that case, as a surviving mutant showed me. What
    /// text cannot represent is a NaN PAYLOAD: every NaN formats as "NaN". Bit patterns also make
    /// the receipt independent of how a future Rust release chooses to format a float, which for a
    /// digest that must be stable across toolchains is the more important property.
    #[test]
    fn floats_are_digested_by_bit_pattern_not_by_formatting() {
        let mut a = spec();
        let mut b = spec();
        a.revin_scale[0] = f32::from_bits(0x7fc0_0001);
        b.revin_scale[0] = f32::from_bits(0x7fc0_0002);
        assert!(a.revin_scale[0].is_nan() && b.revin_scale[0].is_nan());
        assert_eq!(format!("{}", a.revin_scale[0]), format!("{}", b.revin_scale[0]));
        assert_ne!(a.digest(), b.digest(), "two distinct bit patterns share a digest");

        let mut c = spec();
        let mut d = spec();
        c.revin_mean[0] = 0.0;
        d.revin_mean[0] = -0.0;
        assert_ne!(c.digest(), d.digest(), "0.0 and -0.0 produced the same digest");
    }

    /// THE POINT OF THE SPLIT. The same question must digest the same on every machine, or the
    /// comparison that matters can never be posed.
    #[test]
    fn the_platform_is_outside_the_spec_digest() {
        let ids = [1u32, 2, 3];
        let a = TokenReceipt::new(spec(), &ids, 0, "aarch64-darwin / metal");
        let b = TokenReceipt::new(spec(), &ids, 0, "x86_64-linux / cuda");
        assert_eq!(a.spec_digest, b.spec_digest, "platform leaked into the spec digest");
        assert_eq!(agree(&a, &b), Agreement::Identical);
    }

    #[test]
    fn a_changed_token_changes_the_token_digest() {
        let base = TokenReceipt::new(spec(), &[1, 2, 3, 4], 0, "p");
        for (i, ids) in [
            vec![9u32, 2, 3, 4],
            vec![1, 2, 3, 9],
            vec![1, 2, 3],
            vec![1, 2, 3, 4, 5],
            vec![2, 1, 3, 4],
        ]
        .into_iter()
        .enumerate()
        {
            let r = TokenReceipt::new(spec(), &ids, 0, "p");
            assert_ne!(r.token_digest, base.token_digest, "case {i} collided");
        }
    }

    /// A spec mismatch is reported BEFORE a token mismatch, because "the tokens differ" is not a
    /// finding when the two runs were never asked the same question.
    #[test]
    fn a_spec_mismatch_outranks_a_token_mismatch() {
        let a = TokenReceipt::new(spec(), &[1, 2, 3], 0, "p");
        let mut other = spec();
        other.stride = 8;
        let b = TokenReceipt::new(other, &[9, 9, 9], 0, "p");
        assert_eq!(agree(&a, &b), Agreement::DifferentSpec);

        let c = TokenReceipt::new(spec(), &[9, 9, 9], 0, "p");
        assert_eq!(agree(&a, &c), Agreement::ComputationDiverged);
    }

    /// The digest-only path and the receipt path must never disagree, or a file-based verifier
    /// and an in-process one would reach different verdicts about the same two runs.
    #[test]
    fn the_digest_only_rule_matches_the_receipt_rule() {
        let a = TokenReceipt::new(spec(), &[1, 2, 3], 0, "p");
        let mut other = spec();
        other.stride = 8;
        for b in [
            TokenReceipt::new(spec(), &[1, 2, 3], 0, "q"),
            TokenReceipt::new(spec(), &[9, 9, 9], 0, "q"),
            TokenReceipt::new(other, &[1, 2, 3], 0, "q"),
        ] {
            assert_eq!(
                agree(&a, &b),
                agreement(&a.spec_digest, &a.token_digest, &b.spec_digest, &b.token_digest)
            );
        }
    }

    /// The pairs are what actually travels. Every digest a verifier needs must survive the
    /// flattening, or the receipt is unusable in the file it rides in.
    #[test]
    fn the_flat_form_carries_everything_a_verifier_needs() {
        let r = TokenReceipt::new(spec(), &[1, 2, 3], 2, "aarch64-darwin");
        let kv = r.to_pairs();
        let get = |k: &str| kv.iter().find(|(a, _)| a == k).map(|(_, v)| v.clone());
        assert_eq!(get("spec_digest"), Some(r.spec_digest.clone()));
        assert_eq!(get("token_digest"), Some(r.token_digest.clone()));
        assert_eq!(get("tokens"), Some("3".into()));
        assert_eq!(get("non_finite_latents"), Some("2".into()));
        assert_eq!(get("fsq_levels"), Some("8,8,8,8,8".into()));
        assert_eq!(get("encoder.d_model"), Some("256".into()));
        assert_eq!(get("platform"), Some("aarch64-darwin".into()));
        assert!(kv.iter().all(|(k, _)| !k.is_empty()));
    }
}
