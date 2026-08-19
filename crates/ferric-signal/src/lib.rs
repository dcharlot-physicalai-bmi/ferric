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
//! - [`store`] — saving and loading weights, so a trained model can leave the process that
//!   trained it and a receipt can digest real bytes rather than a seed.
//! - [`train`] — the straight-through estimator, without which the gradient through the discrete
//!   bottleneck is exactly zero and the encoder silently never learns.
//! - [`vocab`] — the hybrid vocabulary: signal codes and text tokens in one contiguous id space,
//!   so a decoder does one lookup and never has to know which modality a token came from.
//!
//! ## What is not
//!
//! **There are no published weights.** The towers run, are the right size, train end to end through
//! the bottleneck, and reach 25.5 dB on synthetic physics — but nothing here has been trained on
//! real sensor data, and nothing has been compared against a reference implementation's outputs,
//! because no reference weights were located. Training the language half additionally needs
//! sensor-text PAIRS, which is a data question rather than an engineering one.
//!
//! Reconstruction quality is therefore asserted nowhere in the test suite. A threshold over
//! untrained weights would either be vacuous or would quietly become a quality claim.

pub mod cost;
pub mod encoder;
pub mod fsq;
pub mod language;
pub mod patch;
pub mod receipt;
pub mod sha256;
pub mod store;
pub mod tower;
pub mod train;
pub mod vocab;

pub use cost::TokenCost;
pub use encoder::{EncoderConfig, ParamBreakdown, sinusoidal_positions};
pub use fsq::{Fsq, FsqError};
pub use receipt::{agree, agreement, Agreement, TokenReceipt, TokenSpec};
pub use language::{cross_entropy, lm_forward_var, Example, SensorLm, Sequencer, Span, Task};
pub use patch::{PatchError, Patcher, RevIn};
pub use tower::{decoder_forward_var, forward_var, Block, DecoderWeights, EncoderWeights};
pub use store::{StoreError, Weights};
pub use train::{mse, straight_through};
pub use vocab::{HybridVocab, TokenKind, VocabError};
