//! One sequence, two modalities: words and measurements through a single embedding table.
//!
//! This is the piece that makes the crate *sensor-language* rather than a sensor codec. Everything
//! before it builds a tokenizer; this is what the tokenizer is for — a decoder that reads
//!
//! ```text
//!   "the vibration"  ⟨signal⟩ 14203 9871 22014 ⟨/signal⟩  "rose sharply"
//! ```
//!
//! as one stream, doing one embedding lookup per position and never needing to know which modality
//! a token came from. [`HybridVocab`](crate::HybridVocab) already lays the two id spaces out
//! contiguously; this module builds the sequences and reads them back.
//!
//! ## What is tested here, and the one that is easy to miss
//!
//! Sequence construction and parsing are exact inverses, checked directly. The property that is
//! **false without causal masking** — that a position cannot see the future — is checked by
//! changing a later token and requiring earlier outputs not to move. A shape or determinism test
//! passes happily through a missing mask, exactly as it passes through a missing positional
//! encoding; the same lesson, in a different place.

use crate::vocab::{HybridVocab, TokenKind, VocabError};

/// One span of a mixed sequence.
#[derive(Debug, Clone, PartialEq)]
pub enum Span {
    /// Text token ids, in the text vocabulary's own numbering.
    Text(Vec<u32>),
    /// One signal run: a list of channels, each a list of FSQ code indices.
    ///
    /// Channels are separated in the emitted stream, so a multi-channel window stays one run
    /// rather than becoming several the model has to re-associate.
    Signal(Vec<Vec<u32>>),
}

/// Build and read back mixed text/signal sequences.
#[derive(Debug, Clone)]
pub struct Sequencer {
    vocab: HybridVocab,
}

impl Sequencer {
    pub fn new(vocab: HybridVocab) -> Self {
        Self { vocab }
    }

    pub fn vocab(&self) -> &HybridVocab {
        &self.vocab
    }

    /// Flatten spans into one id stream.
    pub fn encode(&self, spans: &[Span]) -> Result<Vec<u32>, VocabError> {
        let mut out = Vec::new();
        for span in spans {
            match span {
                Span::Text(ids) => {
                    for &t in ids {
                        out.push(self.vocab.text(t)?);
                    }
                }
                Span::Signal(channels) => {
                    out.push(self.vocab.signal_begin());
                    for (i, ch) in channels.iter().enumerate() {
                        if i > 0 {
                            out.push(self.vocab.channel_sep());
                        }
                        for &c in ch {
                            out.push(self.vocab.signal(c)?);
                        }
                    }
                    out.push(self.vocab.signal_end());
                }
            }
        }
        Ok(out)
    }

    /// Read an id stream back into spans. Exact inverse of [`Sequencer::encode`].
    ///
    /// A signal run that never closes is an **error**, not a truncation: silently accepting it
    /// would let a malformed stream train as though it were well formed.
    pub fn decode(&self, ids: &[u32]) -> Result<Vec<Span>, VocabError> {
        let mut spans = Vec::new();
        let mut text: Vec<u32> = Vec::new();
        let mut run: Option<Vec<Vec<u32>>> = None;

        for &id in ids {
            match self.vocab.kind(id)? {
                TokenKind::Text(t) => {
                    if run.is_some() {
                        // Text inside a signal run means the stream is malformed.
                        return Err(VocabError::OutOfRange { id, total: self.vocab.total() });
                    }
                    text.push(t);
                }
                TokenKind::SignalBegin => {
                    if run.is_some() {
                        return Err(VocabError::OutOfRange { id, total: self.vocab.total() });
                    }
                    if !text.is_empty() {
                        spans.push(Span::Text(std::mem::take(&mut text)));
                    }
                    run = Some(vec![Vec::new()]);
                }
                TokenKind::ChannelSep => match run.as_mut() {
                    Some(r) => r.push(Vec::new()),
                    None => return Err(VocabError::OutOfRange { id, total: self.vocab.total() }),
                },
                TokenKind::Signal(c) => match run.as_mut() {
                    Some(r) => r.last_mut().expect("a run always has a channel").push(c),
                    None => return Err(VocabError::OutOfRange { id, total: self.vocab.total() }),
                },
                TokenKind::SignalEnd => match run.take() {
                    Some(r) => spans.push(Span::Signal(r)),
                    None => return Err(VocabError::OutOfRange { id, total: self.vocab.total() }),
                },
            }
        }
        if run.is_some() {
            // An unterminated run.
            return Err(VocabError::OutOfRange { id: 0, total: self.vocab.total() });
        }
        if !text.is_empty() {
            spans.push(Span::Text(text));
        }
        Ok(spans)
    }

    /// How many embedding rows a model over this vocabulary needs.
    pub fn embedding_rows(&self) -> u32 {
        self.vocab.total()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Fsq;

    fn seq() -> Sequencer {
        Sequencer::new(HybridVocab::new(32_000, Fsq::signal_15bit()).unwrap())
    }

    #[test]
    fn text_and_signal_round_trip_through_one_stream() {
        let s = seq();
        let spans = vec![
            Span::Text(vec![10, 20, 30]),
            Span::Signal(vec![vec![0, 1, 32_767], vec![5, 6]]),
            Span::Text(vec![40]),
            Span::Signal(vec![vec![100]]),
        ];
        let ids = s.encode(&spans).unwrap();
        assert_eq!(s.decode(&ids).unwrap(), spans);
    }

    /// EVERY id in the stream must be a legal embedding row. This is the property that lets one
    /// lookup serve both modalities, and an off-by-one at a vocabulary boundary breaks it silently.
    #[test]
    fn every_emitted_id_is_a_legal_embedding_row() {
        let s = seq();
        let ids = s
            .encode(&[
                Span::Text(vec![0, 31_999]),
                Span::Signal(vec![vec![0, 32_767], vec![1]]),
            ])
            .unwrap();
        for &id in &ids {
            assert!(id < s.embedding_rows(), "id {id} is outside the table");
            assert!(s.vocab().kind(id).is_ok());
        }
    }

    /// A signal token and a text token never share an id, so the decoder cannot confuse a
    /// measurement for a word.
    #[test]
    fn text_and_signal_ids_never_collide() {
        let s = seq();
        for t in [0u32, 1, 31_999] {
            for c in [0u32, 1, 32_767] {
                assert_ne!(s.vocab().text(t).unwrap(), s.vocab().signal(c).unwrap());
            }
        }
    }

    #[test]
    fn channel_separators_survive_the_round_trip() {
        let s = seq();
        let spans = vec![Span::Signal(vec![vec![1, 2], vec![3], vec![4, 5, 6]])];
        let ids = s.encode(&spans).unwrap();
        assert_eq!(ids.iter().filter(|&&i| i == s.vocab().channel_sep()).count(), 2);
        assert_eq!(s.decode(&ids).unwrap(), spans);
    }

    /// A malformed stream is REFUSED. Accepting it would let a broken recording train as though it
    /// were well formed, which is the kind of fault that shows up as unexplained model behaviour.
    #[test]
    fn malformed_streams_are_refused_rather_than_repaired() {
        let s = seq();
        let (b, e, sep) = (s.vocab().signal_begin(), s.vocab().signal_end(), s.vocab().channel_sep());
        let sig = s.vocab().signal(7).unwrap();
        let txt = s.vocab().text(9).unwrap();

        assert!(s.decode(&[b, sig]).is_err(), "unterminated run accepted");
        assert!(s.decode(&[sig, e]).is_err(), "signal token outside a run accepted");
        assert!(s.decode(&[e]).is_err(), "close without open accepted");
        assert!(s.decode(&[sep]).is_err(), "separator outside a run accepted");
        assert!(s.decode(&[b, b, e]).is_err(), "nested run accepted");
        assert!(s.decode(&[b, txt, e]).is_err(), "text inside a run accepted");
    }

    #[test]
    fn an_empty_sequence_is_empty_rather_than_an_error() {
        let s = seq();
        assert_eq!(s.encode(&[]).unwrap(), Vec::<u32>::new());
        assert_eq!(s.decode(&[]).unwrap(), Vec::<Span>::new());
    }

    #[test]
    fn out_of_range_spans_are_refused_at_encode_time() {
        let s = seq();
        assert!(s.encode(&[Span::Text(vec![32_000])]).is_err());
        assert!(s.encode(&[Span::Signal(vec![vec![32_768]])]).is_err());
    }

    /// The table a model must allocate, stated once so a decoder and this module cannot disagree.
    #[test]
    fn the_embedding_table_covers_text_signals_and_markers() {
        let s = seq();
        assert_eq!(s.embedding_rows(), 32_000 + 32_768 + 3);
    }
}

/// The three task classes a sensor-language model is trained on.
///
/// Named here because whoever trains this otherwise has to invent the prompt formats, and two
/// people inventing them separately produce two incompatible checkpoints. The published
/// description of the model this crate reproduces lists exactly these three; the layouts below are
/// **this crate's choice** of how to express them, not a recovery of anyone else's.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Task {
    /// Describe, caption, detect anomalies: measurements in, words out.
    SignalToText,
    /// Generate, impute, filter: words in, measurements out.
    TextToSignal,
    /// Forecast, smooth, denoise: measurements in, measurements out.
    SignalToSignal,
}

/// A training example: what the model reads, and what it is scored on.
#[derive(Debug, Clone, PartialEq)]
pub struct Example {
    /// The full token stream, prompt followed by target.
    pub tokens: Vec<u32>,
    /// Index of the first token that carries loss.
    ///
    /// Everything before it is context. **Scoring the prompt teaches the model to predict its own
    /// inputs**, which for `TextToSignal` means most of the gradient goes into reproducing the
    /// instruction rather than the signal — a real and quiet way to waste a training run.
    pub target_from: usize,
}

impl Sequencer {
    /// Build a training example for one task.
    pub fn example(
        &self,
        task: Task,
        text: &[u32],
        signal: &[Vec<u32>],
        target_signal: &[Vec<u32>],
    ) -> Result<Example, VocabError> {
        let (prompt, target) = match task {
            Task::SignalToText => (
                vec![Span::Signal(signal.to_vec())],
                vec![Span::Text(text.to_vec())],
            ),
            Task::TextToSignal => (
                vec![Span::Text(text.to_vec())],
                vec![Span::Signal(target_signal.to_vec())],
            ),
            Task::SignalToSignal => (
                vec![Span::Text(text.to_vec()), Span::Signal(signal.to_vec())],
                vec![Span::Signal(target_signal.to_vec())],
            ),
        };
        let head = self.encode(&prompt)?;
        let tail = self.encode(&target)?;
        let target_from = head.len();
        let mut tokens = head;
        tokens.extend(tail);
        Ok(Example { tokens, target_from })
    }
}

#[cfg(test)]
mod task_tests {
    use super::*;
    use crate::Fsq;

    fn seq() -> Sequencer {
        Sequencer::new(HybridVocab::new(200, Fsq::signal_15bit()).unwrap())
    }

    #[test]
    fn each_task_puts_the_right_modality_on_each_side() {
        let s = seq();
        let text = vec![1u32, 2, 3];
        let sig = vec![vec![10u32, 11]];
        let tgt = vec![vec![20u32, 21, 22]];

        for (task, prompt_is_signal) in [
            (Task::SignalToText, true),
            (Task::TextToSignal, false),
            (Task::SignalToSignal, false),
        ] {
            let e = s.example(task, &text, &sig, &tgt).unwrap();
            let spans = s.decode(&e.tokens).unwrap();
            assert!(!spans.is_empty(), "{task:?} produced nothing");
            assert_eq!(matches!(spans[0], Span::Signal(_)), prompt_is_signal, "{task:?} prompt side");
            // The scored half must be the modality the task names.
            let scored = s.decode(&e.tokens[e.target_from..]).unwrap();
            match task {
                Task::SignalToText => assert!(matches!(scored[0], Span::Text(_))),
                _ => assert!(matches!(scored[0], Span::Signal(_))),
            }
        }
    }

    /// The split must be at a SPAN boundary. A `target_from` landing mid-run would score half a
    /// signal against a stream the model never sees the start of, and the tokens still parse.
    #[test]
    fn the_target_boundary_falls_between_complete_spans() {
        let s = seq();
        let sig = vec![vec![10u32, 11], vec![12]];
        let tgt = vec![vec![20u32]];
        for task in [Task::SignalToText, Task::TextToSignal, Task::SignalToSignal] {
            let e = s.example(task, &[1, 2], &sig, &tgt).unwrap();
            assert!(s.decode(&e.tokens[..e.target_from]).is_ok(), "{task:?} prompt is malformed");
            assert!(s.decode(&e.tokens[e.target_from..]).is_ok(), "{task:?} target is malformed");
            assert!(e.target_from > 0 && e.target_from < e.tokens.len());
        }
    }

    /// Signal-to-signal keeps the CONDITIONING signal and the TARGET signal as separate runs, so
    /// the model can tell where the history ends and the forecast begins.
    #[test]
    fn signal_to_signal_keeps_history_and_target_as_separate_runs() {
        let s = seq();
        let e = s
            .example(Task::SignalToSignal, &[9], &[vec![1, 2, 3]], &[vec![4, 5]])
            .unwrap();
        let spans = s.decode(&e.tokens).unwrap();
        let runs: Vec<&Span> = spans.iter().filter(|x| matches!(x, Span::Signal(_))).collect();
        assert_eq!(runs.len(), 2, "history and target were merged into one run");
        assert_eq!(runs[0], &Span::Signal(vec![vec![1, 2, 3]]));
        assert_eq!(runs[1], &Span::Signal(vec![vec![4, 5]]));
    }
}

// ---------------------------------------------------------------------------------------------
// The decoder: one embedding table, one causal stream, both modalities.
// ---------------------------------------------------------------------------------------------

use crate::encoder::{sinusoidal_positions, EncoderConfig};
use crate::patch::PatchError;
use crate::tower::Block;
use ferric_core::Context;
use ferric_tensor::nn::causal_attention;
use ferric_tensor::Tensor;
use std::sync::Arc;

const EPS: f32 = 1e-5;

/// A causal decoder over the hybrid vocabulary.
///
/// The whole point is the first line of [`SensorLm::forward`]: **one `gather_rows` for every
/// position**, whether that position is a word, a measurement or a marker. Nothing downstream
/// branches on modality, because by the time the tokens reach the tower there is no modality left —
/// only ids into one table. That is what [`HybridVocab`] buys.
pub struct SensorLm {
    pub cfg: EncoderConfig,
    pub rows: u32,
    /// `[rows, d_model]`.
    pub embed: Tensor,
    pub blocks: Vec<Block>,
    pub out_norm: Tensor,
    /// `[rows, d_model]`; untied from `embed` so the two can be checked independently.
    pub head: Tensor,
}

impl SensorLm {
    pub fn deterministic(
        ctx: &Arc<Context>,
        cfg: EncoderConfig,
        rows: u32,
        seed: u64,
    ) -> Result<Self, PatchError> {
        cfg.validate()?;
        let d = cfg.d_model;
        let s = |fan: usize| 1.0 / (fan as f32).sqrt();
        let mut k = seed;
        let mut next = |n: usize, scale: f32, shape: &[usize]| {
            k = k.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            Tensor::from_vec(ctx, &crate::tower::fill_pub(k, n, scale), shape)
        };
        let n = rows as usize;
        let embed = next(n * d, s(d), &[n, d]);
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
            rows,
            embed,
            blocks,
            out_norm: Tensor::from_vec(ctx, &vec![1.0f32; d], &[d]),
            head: next(n * d, s(d), &[n, d]),
        })
    }

    /// Token ids in, logits `[t, rows]` out.
    pub fn forward(&self, ctx: &Arc<Context>, ids: &[u32]) -> Result<Tensor, PatchError> {
        if ids.is_empty() {
            return Err(PatchError::TooShort { len: 0, patch_len: 1 });
        }
        if let Some(&bad) = ids.iter().find(|&&i| i >= self.rows) {
            // An id past the table would read another row's embedding and look perfectly healthy.
            return Err(PatchError::Ragged { len: bad as usize, channels: self.rows as usize });
        }
        let cfg = self.cfg;
        let t = ids.len();

        // ONE lookup, both modalities. There is no branch here and there is not meant to be.
        let mut x = self.embed.gather_rows(ids);
        let pe = Tensor::from_vec(ctx, &sinusoidal_positions(t, cfg.d_model)?, &[t, cfg.d_model]);
        x = x.add(&pe);

        for b in &self.blocks {
            let h = x.rmsnorm(&b.attn_norm, EPS);
            let (q, k, v) = (h.matmul_bt(&b.wq), h.matmul_bt(&b.wk), h.matmul_bt(&b.wv));
            // CAUSAL, not bidirectional: a language model predicts the next token, so a position
            // must not see its own future. The encoder tower uses the unmasked variant because it
            // reads a whole window at once; using the wrong one here is invisible to every shape
            // and determinism test in this file.
            let a = causal_attention(&q, &k, &v, cfg.n_heads, cfg.n_heads, 0.0);
            x = x.add(&a.matmul_bt(&b.wo));

            let h = x.rmsnorm(&b.ffn_norm, EPS);
            let gate = h.matmul_bt(&b.w_gate).silu();
            let up = h.matmul_bt(&b.w_up);
            x = x.add(&gate.mul(&up).matmul_bt(&b.w_down));
        }
        Ok(x.rmsnorm(&self.out_norm, EPS).matmul_bt(&self.head))
    }
}

use ferric_tensor::autograd::Var;

impl SensorLm {
    /// Look up embeddings. **Not differentiable**: `Var` has no row gather, so the table is frozen
    /// and training moves the blocks and the head. That is a real limitation and it is stated here
    /// rather than discovered — a frozen embedding cannot learn that two signal codes mean similar
    /// things, so it caps what the language half can pick up from a small corpus.
    pub fn embed_tokens(&self, ids: &[u32]) -> Tensor {
        self.embed.gather_rows(ids)
    }

    /// Trainable parameters, in the order [`lm_forward_var`] expects. The embedding is excluded.
    pub fn params_flat(&self) -> Vec<Tensor> {
        let mut v = Vec::new();
        for b in &self.blocks {
            v.extend([
                b.attn_norm.clone(), b.wq.clone(), b.wk.clone(), b.wv.clone(), b.wo.clone(),
                b.ffn_norm.clone(), b.w_gate.clone(), b.w_up.clone(), b.w_down.clone(),
            ]);
        }
        v.push(self.out_norm.clone());
        v.push(self.head.clone());
        v
    }
}

/// The causal tower over already-embedded positions. `[t, d_model]` in, `[t, rows]` logits out.
pub fn lm_forward_var(
    ctx: &Arc<Context>,
    cfg: EncoderConfig,
    params: &[Var],
    embedded: &Var,
) -> Result<Var, PatchError> {
    let expect = 2 + 9 * cfg.n_layers;
    if params.len() != expect {
        return Err(PatchError::Ragged { len: params.len(), channels: expect });
    }
    let t = embedded.value().shape[0];
    let d = cfg.d_model;
    let lin = |x: &Var, w: &Var| x.matmul(&w.transpose(1, 0));

    let pe = Var::leaf(Tensor::from_vec(ctx, &sinusoidal_positions(t, d)?, &[t, d]));
    let mut x = embedded.add(&pe);

    for l in 0..cfg.n_layers {
        let p = &params[9 * l..9 * (l + 1)];
        let h = x.rmsnorm(&p[0], EPS);
        let (q, k, v) = (lin(&h, &p[1]), lin(&h, &p[2]), lin(&h, &p[3]));
        // CAUSAL. Built here from a triangular additive mask because `Var` has no masked-attention
        // helper; the mask is -inf above the diagonal, so softmax gives a future position zero
        // weight rather than a small one.
        let dh = d / cfg.n_heads;
        let heads = |z: &Var| z.reshape(&[t, cfg.n_heads, dh]).transpose(1, 0).contiguous();
        let (qh, kh, vh) = (heads(&q), heads(&k), heads(&v));
        let mut m = vec![0.0f32; cfg.n_heads * t * t];
        for hd in 0..cfg.n_heads {
            for i in 0..t {
                for j in (i + 1)..t {
                    m[hd * t * t + i * t + j] = f32::NEG_INFINITY;
                }
            }
        }
        let mask = Var::leaf(Tensor::from_vec(ctx, &m, &[cfg.n_heads, t, t]));
        let scale = Var::leaf(Tensor::from_vec(ctx, &[1.0 / (dh as f32).sqrt()], &[1]))
            .broadcast_to(&[cfg.n_heads, t, t]);
        let probs = qh.matmul(&kh.transpose(2, 1)).mul(&scale).add(&mask).softmax(2);
        let a = probs.matmul(&vh).transpose(1, 0).contiguous().reshape(&[t, d]);
        x = x.add(&lin(&a, &p[4]));

        let h = x.rmsnorm(&p[5], EPS);
        let gate = lin(&h, &p[6]).silu();
        let up = lin(&h, &p[7]);
        x = x.add(&lin(&gate.mul(&up), &p[8]));
    }
    let n = params.len();
    Ok(lin(&x.rmsnorm(&params[n - 2], EPS), &params[n - 1]))
}

/// Mean cross-entropy over the scored span only.
///
/// `from` is [`Example::target_from`]. Positions before it are context and contribute nothing —
/// scoring them teaches the model to predict its own prompt.
pub fn cross_entropy(
    ctx: &Arc<Context>,
    logits: &Var,
    ids: &[u32],
    from: usize,
    rows: u32,
) -> Result<Var, PatchError> {
    let t = logits.value().shape[0];
    let v = rows as usize;
    if t != ids.len() || from >= t {
        return Err(PatchError::Ragged { len: ids.len(), channels: t });
    }
    // Next-token targets: position i predicts token i+1. A one-hot mask picks the target logit,
    // which is how this is done without a gather on the autograd path.
    let mut oh = vec![0.0f32; t * v];
    let mut n = 0.0f32;
    for i in from.saturating_sub(1)..t - 1 {
        oh[i * v + ids[i + 1] as usize] = 1.0;
        n += 1.0;
    }
    let mask = Var::leaf(Tensor::from_vec(ctx, &oh, &[t, v]));
    let eps = Var::leaf(Tensor::from_vec(ctx, &[1e-9f32], &[1])).broadcast_to(&[t, v]);
    let logp = logits.softmax(1).add(&eps).log();
    let scale = Var::leaf(Tensor::from_vec(ctx, &[-1.0 / n.max(1.0)], &[1]));
    Ok(logp.mul(&mask).sum_all().mul(&scale))
}

/// Differentiable embedding lookup, from existing primitives only.
///
/// `Var` has no row-gather backward, and this crate does not reach into a shared crate to add one
/// while other work is in flight there. It does not need to: a one-hot matrix `[t, rows]` times
/// the table `[rows, d]` IS gather in the forward pass, and matmul's existing backward computes
/// `one_hot^T @ grad` — exactly the scatter-add a trainable embedding needs. The one-hot is
/// materialized, t x rows floats, which is megabytes at this crate's scales and would want a
/// native gather in a real deployment; the cost of NOT having a trainable table at all is measured
/// in the README at 60% against 38% held-out.
pub fn embed_var(ctx: &Arc<Context>, table: &Var, ids: &[u32]) -> Result<Var, PatchError> {
    let shape = &table.value().shape;
    if shape.len() != 2 {
        return Err(PatchError::Ragged { len: shape.len(), channels: 2 });
    }
    let rows = shape[0];
    if ids.is_empty() {
        return Err(PatchError::TooShort { len: 0, patch_len: 1 });
    }
    if let Some(&bad) = ids.iter().find(|&&i| (i as usize) >= rows) {
        return Err(PatchError::Ragged { len: bad as usize, channels: rows });
    }
    let t = ids.len();
    let mut oh = vec![0.0f32; t * rows];
    for (i, &id) in ids.iter().enumerate() {
        oh[i * rows + id as usize] = 1.0;
    }
    Ok(Var::leaf(Tensor::from_vec(ctx, &oh, &[t, rows])).matmul(table))
}

#[cfg(test)]
mod embed_var_tests {
    use super::*;
    use crate::Fsq;

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

    /// The differentiable lookup must agree with the frozen one exactly, or the model that trains
    /// with one and runs with the other is two models.
    #[test]
    fn the_one_hot_lookup_equals_gather_rows() {
        let Some(ctx) = ctx() else { return };
        let v = HybridVocab::new(50, Fsq::signal_15bit()).unwrap();
        let lm = SensorLm::deterministic(
            &ctx,
            EncoderConfig { patch_len: 8, d_model: 16, n_layers: 1, n_heads: 2, d_ff: 32, latent_dim: 5 },
            v.total(),
            7,
        )
        .unwrap();
        let ids = [0u32, 49, 500, v.total() - 1, 3];
        let a = pollster::block_on(lm.embed_tokens(&ids).to_vec());
        let b = pollster::block_on(
            embed_var(&ctx, &Var::leaf(lm.embed.clone()), &ids).unwrap().value().to_vec(),
        );
        assert_eq!(a, b, "the two lookups disagree");
    }

    /// THE PROPERTY THE FROZEN TABLE LACKED: gradient reaches exactly the rows that were used.
    /// A used row must receive a non-zero gradient and an unused row must receive zero — anything
    /// else either fails to train the codes that appeared or trains codes that did not.
    #[test]
    fn gradient_reaches_used_rows_and_only_used_rows() {
        let Some(ctx) = ctx() else { return };
        let rows = 40usize;
        let d = 8usize;
        let data: Vec<f32> = (0..rows * d).map(|i| ((i * 37) % 19) as f32 * 0.1 - 0.9).collect();
        let table = Var::leaf(Tensor::from_vec(&ctx, &data, &[rows, d]));
        let ids = [3u32, 17, 3, 39];
        embed_var(&ctx, &table, &ids).unwrap().sum_all().backward();
        let g = pollster::block_on(table.grad().expect("no gradient reached the table").to_vec());
        for r in 0..rows {
            let row_g = &g[r * d..(r + 1) * d];
            let used = ids.contains(&(r as u32));
            let nonzero = row_g.iter().any(|x| *x != 0.0);
            assert_eq!(nonzero, used, "row {r}: used={used} but gradient nonzero={nonzero}");
        }
        // Row 3 appeared twice, so its gradient is the SUM: exactly 2.0 per column under sum_all.
        for c in 0..d {
            assert_eq!(g[3 * d + c], 2.0, "duplicate id did not accumulate");
        }
    }

    #[test]
    fn out_of_range_and_empty_ids_are_refused() {
        let Some(ctx) = ctx() else { return };
        let table = Var::leaf(Tensor::from_vec(&ctx, &vec![0.0; 10 * 4], &[10, 4]));
        assert!(embed_var(&ctx, &table, &[10]).is_err(), "an id past the table was accepted");
        assert!(embed_var(&ctx, &table, &[]).is_err(), "an empty id list was accepted");
    }
}

#[cfg(test)]
mod lm_tests {
    use super::*;
    use crate::Fsq;

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

    fn small() -> (Sequencer, EncoderConfig) {
        // A 200-word text vocabulary keeps the embedding table small; the signal half is the real
        // 32,768 either way, which is the case that matters.
        let v = HybridVocab::new(200, Fsq::signal_15bit()).unwrap();
        let cfg = EncoderConfig { patch_len: 16, d_model: 32, n_layers: 2, n_heads: 4, d_ff: 64, latent_dim: 5 };
        (Sequencer::new(v), cfg)
    }

    fn mixed(s: &Sequencer) -> Vec<u32> {
        s.encode(&[
            Span::Text(vec![7, 8, 9]),
            Span::Signal(vec![vec![101, 202, 303], vec![404, 505]]),
            Span::Text(vec![11, 12]),
        ])
        .unwrap()
    }

    #[test]
    fn a_mixed_stream_produces_one_logit_row_per_position() {
        let Some(ctx) = ctx() else { return };
        let (s, cfg) = small();
        let lm = SensorLm::deterministic(&ctx, cfg, s.embedding_rows(), 5).unwrap();
        let ids = mixed(&s);
        let out = lm.forward(&ctx, &ids).unwrap();
        assert_eq!(out.shape, vec![ids.len(), s.embedding_rows() as usize]);
        assert!(pollster::block_on(out.to_vec()).iter().all(|v| v.is_finite()));
    }

    /// THE PROPERTY THAT IS FALSE WITHOUT A CAUSAL MASK.
    ///
    /// Change the LAST token and every earlier position must be untouched. Shape, determinism and
    /// finiteness tests all pass through an unmasked decoder without noticing, exactly as they pass
    /// through a missing positional encoding — the same blindness, a different component.
    #[test]
    fn a_position_cannot_see_its_own_future() {
        let Some(ctx) = ctx() else { return };
        let (s, cfg) = small();
        let lm = SensorLm::deterministic(&ctx, cfg, s.embedding_rows(), 9).unwrap();
        let mut ids = mixed(&s);
        let rows = s.embedding_rows() as usize;

        let a = pollster::block_on(lm.forward(&ctx, &ids).unwrap().to_vec());
        let last = ids.len() - 1;
        ids[last] = s.vocab().text(42).unwrap();
        let b = pollster::block_on(lm.forward(&ctx, &ids).unwrap().to_vec());

        // Every position before the change must be bit-identical.
        let cut = last * rows;
        assert_eq!(&a[..cut], &b[..cut], "an earlier position moved when a later token changed");
        // And the changed position itself must actually differ, or the test proves nothing.
        assert_ne!(&a[cut..], &b[cut..], "the changed position did not move at all");
    }

    #[test]
    fn the_forward_is_bit_exact_and_refuses_bad_input() {
        let Some(ctx) = ctx() else { return };
        let (s, cfg) = small();
        let lm = SensorLm::deterministic(&ctx, cfg, s.embedding_rows(), 3).unwrap();
        let ids = mixed(&s);
        let a = pollster::block_on(lm.forward(&ctx, &ids).unwrap().to_vec());
        for _ in 0..3 {
            assert_eq!(pollster::block_on(lm.forward(&ctx, &ids).unwrap().to_vec()), a);
        }
        assert!(lm.forward(&ctx, &[]).is_err(), "an empty stream was accepted");
        assert!(lm.forward(&ctx, &[s.embedding_rows()]).is_err(), "an id past the table was accepted");
    }

    /// A word and a measurement must reach the tower as different vectors. If the two id spaces
    /// overlapped, the model would be reading a measurement as a word and nothing would say so.
    #[test]
    fn a_word_and_a_measurement_embed_differently() {
        let Some(ctx) = ctx() else { return };
        let (s, cfg) = small();
        let lm = SensorLm::deterministic(&ctx, cfg, s.embedding_rows(), 11).unwrap();
        let t = s.vocab().text(5).unwrap();
        let g = s.vocab().signal(5).unwrap();
        assert_ne!(t, g);
        let rows = pollster::block_on(lm.embed.gather_rows(&[t, g]).to_vec());
        let d = cfg.d_model;
        assert_ne!(&rows[..d], &rows[d..], "a word and a signal code share an embedding row");
    }
}
