//! # ferric-signal
//!
//! Open sensor-language tokenization for physical signals.
//!
//! ## Why this crate exists
//!
//! A sensor-language model reads a stream of measurements and a stream of words through the same
//! embedding table. That only works if a signal can be turned into tokens that live in the same
//! vocabulary as text, and the tokenizer is the piece everything else is built on: the decoder,
//! the training data, the evaluation. In the published landscape this review located, the
//! forecasting foundation models are open but do not speak language, and the models that do speak
//! language about sensors are either narrow (wearable IMU) or reachable only through an API.
//!
//! The tokenizer is also, by a wide margin, the *small* part. A discrete bottleneck over five
//! dimensions with eight levels each spans 32,768 codes and needs no codebook at all. So the
//! component that makes the rest possible is the one that costs least to open, which is the whole
//! argument for building it here.
//!
//! ## What is in place
//!
//! - [`fsq`] — Finite Scalar Quantization. The discrete bottleneck, with round-trip and
//!   bijection verified exhaustively over the entire code space rather than sampled.
//! - [`cost`] — exact operations and bytes per token, and why a per-token figure without its
//!   window length is underspecified.
//! - [`encoder`] — tower shape, exact parameter accounting against the published 9.5M, and the
//!   parameter-free positional encoding.
//! - [`language`] — mixed text/signal sequences: the piece that makes this sensor-LANGUAGE
//!   rather than a sensor codec.
//! - [`patch`] — the signal front end: patching, and reversible per-channel normalization whose
//!   inverse is checked directly rather than against a reference model.
//! - [`tower`] — encoder and decoder forward passes, each on both the `Tensor` and `Var`
//!   backends, held to each other numerically so the model that trains is the model that runs.
//! - [`receipt`] — a recomputable claim that these tokens came from that signal, in the flat
//!   key/value form `ferroscope` carries inside an MCAP recording.
//! - [`synth`] — parameterized synthetic physical processes with known ground truth, the data
//!   generator behind every measured number in this crate.
//! - [`store`] — saving and loading weights, so a trained model can leave the process that
//!   trained it and a receipt can digest real bytes rather than a seed.
//! - [`train`] — the straight-through estimator, without which the gradient through the discrete
//!   bottleneck is exactly zero and the encoder silently never learns.
//! - [`vocab`] — the hybrid vocabulary: signal codes and text tokens in one contiguous id space,
//!   so a decoder does one lookup and never has to know which modality a token came from.
//!
//! ## What is not
//!
//! **There are no published weights.** The towers run, are the right size, and train end to end
//! through the bottleneck — 25.5 dB on synthetic physics in a single run — but nothing here has
//! been trained on real sensor data, and nothing has been compared against a reference
//! implementation's outputs, because no reference weights were located. Training the language half
//! additionally needs sensor-text PAIRS, which is a data question rather than an engineering one.
//!
//! **Figures in this crate's documentation and examples are single runs unless a seed count is
//! given.** Held-out accuracy at these sample sizes carries a standard deviation of roughly six
//! points across seeds, so a difference smaller than that is not a measurement. The README reports
//! the one comparison that was run with five seeds; treat everything else as unresolved at that
//! level.
//!
//! Reconstruction quality is asserted nowhere in the test suite. A threshold over untrained
//! weights would either be vacuous or would quietly become a quality claim.

pub mod cost;
pub mod encoder;
pub mod fsq;
pub mod language;
pub mod patch;
pub mod receipt;
pub mod sha256;
pub mod store;
pub mod synth;
pub mod tower;
pub mod train;
pub mod vocab;

pub use cost::{vocab_cost, TokenCost, VocabCost};
pub use encoder::{EncoderConfig, ParamBreakdown, sinusoidal_positions};
pub use fsq::{Fsq, FsqError};
pub use receipt::{agree, agreement, Agreement, TokenReceipt, TokenSpec};
pub use language::{cross_entropy, embed_var, lm_forward_var, Example, SensorLm, Sequencer, Span, Task};
pub use patch::{PatchError, Patcher, RevIn};
pub use tower::{decoder_forward_var, forward_var, Block, DecoderWeights, EncoderWeights};
pub use store::{StoreError, Weights};
pub use train::{mse, straight_through};
pub use vocab::{HybridVocab, TokenKind, VocabError};
