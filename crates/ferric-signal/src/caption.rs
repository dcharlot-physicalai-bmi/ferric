//! Training a decoder to say, in words, what a signal is — and the three things that have to be
//! reported alongside whatever it says.
//!
//! This is the piece that makes the crate sensor-LANGUAGE rather than a sensor codec: a caption is
//! several words, one per label axis, and each axis is scored at its own position. It lives here
//! rather than in one corpus example because two corpora already need it and a training loop that
//! exists twice gets fixed once.
//!
//! ## What this module knows that cost something to learn
//!
//! **Presentation order is part of the protocol.** Condition-ordered corpora are the norm in this
//! field. Walking one in corpus order, at batch 1, was on its own enough to collapse a decoder to
//! a constant — every axis at its majority baseline, standard deviation 0.000 across seeds — and
//! that null was published as evidence about the tokenizer. [`train_captions`] shuffles by
//! default and takes `sequential` only so the failure can be reproduced.
//!
//! **A batch is accumulated, not stacked.** [`crate::lm_forward_var`] takes `[t, d]` with no batch
//! axis, and stacking two examples would let one attend to the other. Gradients are summed
//! instead, so every example is still encoded on its own.
//!
//! **What the model SAID is reported, not just whether it was right.** A decoder answering the
//! same word for every held-out example scores that word's frequency and reads as a weak learner
//! in an accuracy column. [`SeedResult::distinct`] counts the words it actually emitted, so that
//! failure announces itself.
//!
//! **The vocabulary is sized to the corpus.** [`compact`] restricts the signal vocabulary to codes
//! the TRAINING examples use — 32,768 rows down to a few thousand, at equal accuracy, because
//! every unvisited row is an untrained embedding and a class in the output softmax.

use crate::{
    cross_entropy, embed_var, lm_forward_var, EncoderConfig, Example, PatchError, SensorLm,
    Sequencer, Span, VocabError,
};
use ferric_core::Context;
use ferric_tensor::autograd::Var;
use ferric_tensor::optim::Adam;
use ferric_tensor::Tensor;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use crate::bench::shuffled;

/// Why a training run could not be set up. Kept as two named cases rather than one string,
/// because a vocabulary that cannot represent a caption and a tensor of the wrong shape are
/// different mistakes with different fixes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CaptionError {
    /// A caption word or signal code outside the vocabulary it was encoded against.
    Vocab(VocabError),
    /// A tensor shape the towers refused.
    Shape(PatchError),
}

impl std::fmt::Display for CaptionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Vocab(e) => write!(f, "vocabulary: {e:?}"),
            Self::Shape(e) => write!(f, "shape: {e:?}"),
        }
    }
}

impl std::error::Error for CaptionError {}

impl From<VocabError> for CaptionError {
    fn from(e: VocabError) -> Self {
        Self::Vocab(e)
    }
}

impl From<PatchError> for CaptionError {
    fn from(e: PatchError) -> Self {
        Self::Shape(e)
    }
}

/// What one seed produced: held-out accuracy per axis, and whether the model said anything.
pub struct SeedResult {
    pub acc: Vec<f64>,
    /// Distinct words the model actually emitted at each axis position across the held-out set.
    /// **1 means it answered the same thing every time.**
    pub distinct: Vec<usize>,
    /// Share of held-out predictions at each axis position that were not a valid word for that
    /// axis at all — a signal code, an end marker, or another axis's word.
    ///
    /// **"Wrong answer" and "not answering the question" are different failures.** An accuracy of
    /// exactly 0.0% is the signature of the second: a model that had settled on the wrong caption
    /// word would still be right whenever that word happened to be correct, so it lands on that
    /// word's frequency rather than on nothing. Zero means the argmax is leaving the caption
    /// vocabulary, and no amount of staring at an accuracy column says so.
    pub off_axis: Vec<f64>,
}

/// One word per (axis, value) pair, plus a terminator at id 0.
///
/// A caption is then one word per axis and an end marker, and the axes stay separable at scoring
/// time — which is what lets a result say WHICH axis a decoder can read rather than reporting one
/// blended number.
pub fn build_words(axis_names: &[&str], labels: &[Vec<i32>]) -> (Vec<String>, Vec<Vec<u32>>) {
    let n_axes = axis_names.len();
    let mut words = vec!["<end>".to_string()];
    let mut per_axis: Vec<Vec<i32>> = vec![Vec::new(); n_axes];
    for row in labels {
        for (a, &v) in row.iter().enumerate() {
            if !per_axis[a].contains(&v) {
                per_axis[a].push(v);
            }
        }
    }
    for a in 0..n_axes {
        per_axis[a].sort_unstable();
        for v in &per_axis[a] {
            words.push(format!("{}={}", axis_names[a], v));
        }
    }
    let idx = |a: usize, v: i32| -> u32 {
        let mut base = 1usize;
        for k in 0..a {
            base += per_axis[k].len();
        }
        (base + per_axis[a].iter().position(|&x| x == v).unwrap()) as u32
    };
    let caps: Vec<Vec<u32>> = labels
        .iter()
        .map(|row| {
            let mut c: Vec<u32> = (0..n_axes).map(|a| idx(a, row[a])).collect();
            c.push(0);
            c
        })
        .collect();
    (words, caps)
}

/// Compact the signal vocabulary down to the codes the TRAINING examples actually use.
///
/// Returns the remapped documents, the new codebook size, and the share of held-out tokens that
/// fell outside the training vocabulary and were mapped to one reserved id.
///
/// **Built from the training examples only.** A vocabulary fitted to every code the held-out
/// examples contain has already been told something about them; an unsupervised step is still
/// leakage when the number being reported is held-out accuracy. The miss rate is returned rather
/// than hidden, because a compaction that quietly discards a third of the held-out signal is not a
/// free win.
pub fn compact(
    docs: &[Vec<Vec<u32>>],
    train: &[usize],
    held: &[usize],
) -> (Vec<Vec<Vec<u32>>>, u32, f64) {
    let mut seen: Vec<u32> = train
        .iter()
        .flat_map(|&i| docs[i].iter().flatten().copied())
        .collect::<HashSet<u32>>()
        .into_iter()
        .collect();
    seen.sort_unstable();
    let map: HashMap<u32, u32> = seen.iter().enumerate().map(|(k, &c)| (c, k as u32)).collect();
    let unk = seen.len() as u32;
    let remapped: Vec<Vec<Vec<u32>>> = docs
        .iter()
        .map(|runs| {
            runs.iter()
                .map(|r| r.iter().map(|c| map.get(c).copied().unwrap_or(unk)).collect())
                .collect()
        })
        .collect();
    let (mut miss, mut total) = (0usize, 0usize);
    for &i in held {
        for run in &docs[i] {
            for c in run {
                total += 1;
                if !map.contains_key(c) {
                    miss += 1;
                }
            }
        }
    }
    (remapped, unk + 1, miss as f64 / total.max(1) as f64 * 100.0)
}

/// Train a signal-to-text decoder for one seed and score every axis on the held-out set.
#[allow(clippy::too_many_arguments)]
pub fn train_captions(
    ctx: &Arc<Context>,
    seq: &Sequencer,
    rows_tokens: &[Vec<Vec<u32>>],
    caps: &[Vec<u32>],
    train_idx: &[usize],
    held_idx: &[usize],
    steps: usize,
    batch: usize,
    lm_cfg: EncoderConfig,
    sequential: bool,
    n_axes: usize,
    seed: u64,
) -> Result<SeedResult, CaptionError> {
    let rows = seq.embedding_rows();
    let lm_cfg = lm_cfg;
    let cfg = lm_cfg;
    let lm = SensorLm::deterministic(ctx, cfg, rows, seed)?;
    // The embedding table trains with the rest: a signal code the corpus never visited is an
    // untrained row, and this corpus visits only a fraction of the code space.
    let mut params: Vec<Tensor> = std::iter::once(lm.embed.clone()).chain(lm.params_flat()).collect();
    let mut opt = Adam::new(&params, 2e-3);

    let build = |i: usize| -> Result<Example, CaptionError> {
        let prompt = seq.encode(&[Span::Signal(rows_tokens[i].clone())])?;
        let target = seq.encode(&[Span::Text(caps[i].clone())])?;
        let target_from = prompt.len();
        let mut tokens = prompt;
        tokens.extend(target);
        Ok(Example { tokens, target_from })
    };

    // GRADIENT ACCUMULATION over `batch` examples per optimizer step.
    //
    // `lm_forward_var` takes `[t, d]`, with no batch axis, and giving it two examples stacked
    // would let one attend to the other — the same coupling the tokenizer path already had to
    // correct. Summing gradients instead leaves every example encoded on its own and only
    // averages what the optimizer sees. At batch 1 the gradient from a single 600-token sequence
    // with a six-word target is noisy enough that the cheapest descent direction is the label
    // marginal, which is exactly the constant-output failure the `distinct` column below counts.
    // Presentation ORDER is a variable here, not a detail. This corpus is laid out by
    // experimental condition, so walking it in index order feeds the decoder long runs of
    // near-identical labels. `sequential` reproduces that walk so the two can be compared.
    let mut order: Vec<usize> = if sequential {
        (0..train_idx.len()).collect()
    } else {
        shuffled(train_idx.len(), seed ^ 0x5EED)
    };
    let mut cursor = 0usize;
    for _ in 0..steps {
        let mut acc: Vec<Tensor> = Vec::new();
        for _ in 0..batch.max(1) {
            if cursor >= order.len() {
                if !sequential {
                    order = shuffled(train_idx.len(), seed ^ 0x5EED ^ cursor as u64);
                }
                cursor = 0;
            }
            let i = train_idx[order[cursor]];
            cursor += 1;
            let e = build(i)?;
            let vars: Vec<Var> = params.iter().cloned().map(Var::leaf).collect();
            let emb = embed_var(ctx, &vars[0], &e.tokens)?;
            let logits = lm_forward_var(ctx, cfg, &vars[1..], &emb)?;
            let loss = cross_entropy(ctx, &logits, &e.tokens, e.target_from, rows)?;
            loss.backward();
            let grads: Vec<Tensor> = vars.iter().map(|v| v.grad().expect("no gradient")).collect();
            acc = if acc.is_empty() {
                grads
            } else {
                acc.iter()
                    .zip(grads.iter())
                    .map(|(a, g)| Var::leaf(a.clone()).add(&Var::leaf(g.clone())).value().clone())
                    .collect()
            };
        }
        let inv = 1.0 / batch.max(1) as f32;
        let grads: Vec<Tensor> = acc
            .iter()
            .map(|a| {
                let sc = Var::leaf(Tensor::from_vec(ctx, &[inv], &[1])).broadcast_to(&a.shape);
                Var::leaf(a.clone()).mul(&sc).value().clone()
            })
            .collect();
        opt.step(&mut params, &grads);
    }

    // Score each axis at its own caption position: the model sees the signal plus the caption
    // words before this axis, and must produce this axis's word.
    // Which embedding rows are legal answers at each caption position, derived from the captions
    // themselves so no extra plumbing can fall out of step with `build_words`.
    let mut legal: Vec<HashSet<usize>> = vec![HashSet::new(); n_axes];
    for c in caps {
        for (a, w) in c.iter().take(n_axes).enumerate() {
            legal[a].insert(seq.vocab().text(*w)? as usize);
        }
    }

    let vars: Vec<Var> = params.iter().cloned().map(Var::leaf).collect();
    let mut right = vec![0usize; n_axes];
    let mut off = vec![0usize; n_axes];
    // What the model EMITTED, not just whether it was right. A model that answers the same word
    // for every held-out cycle scores the frequency of that word and looks like a weak learner in
    // an accuracy column; counting distinct predictions says outright that it never varied.
    let mut said: Vec<HashSet<usize>> = vec![HashSet::new(); n_axes];
    for &i in held_idx {
        let e = build(i)?;
        for a in 0..n_axes {
            let upto = e.target_from + a;
            let emb = embed_var(ctx, &vars[0], &e.tokens[..upto])?;
            let logits =
                pollster::block_on(lm_forward_var(ctx, cfg, &vars[1..], &emb)?.value().to_vec());
            let last = (upto - 1) * rows as usize;
            let row = &logits[last..last + rows as usize];
            // An empty logits row would mean an embedding table with no rows, which the
            // vocabulary makes impossible; naming it beats a silent zero.
            let best = row
                .iter()
                .enumerate()
                .max_by(|x, y| x.1.total_cmp(y.1))
                .map(|(i, _)| i)
                .ok_or(PatchError::Ragged { len: 0, channels: rows as usize })?;
            said[a].insert(best);
            if !legal[a].contains(&best) {
                off[a] += 1;
            }
            if best == seq.vocab().text(caps[i][a])? as usize {
                right[a] += 1;
            }
        }
    }
    let n = held_idx.len().max(1) as f64;
    Ok(SeedResult {
        acc: right.iter().map(|&r| r as f64 / n * 100.0).collect(),
        distinct: said.iter().map(|s| s.len()).collect(),
        off_axis: off.iter().map(|&o| o as f64 / n * 100.0).collect(),
    })
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_caption_is_one_word_per_axis_plus_a_terminator() {
        let labels = vec![vec![3, 100], vec![20, 100], vec![3, 80]];
        let (words, caps) = build_words(&["cooler", "valve"], &labels);
        // id 0 is the terminator, then cooler's two values, then valve's two.
        assert_eq!(words[0], "<end>");
        assert_eq!(&words[1..], &["cooler=3", "cooler=20", "valve=80", "valve=100"]);
        for (c, row) in caps.iter().zip(&labels) {
            assert_eq!(c.len(), row.len() + 1);
            assert_eq!(*c.last().unwrap(), 0, "every caption ends with the terminator");
        }
        // The words a caption points at must spell its own labels back.
        for (c, row) in caps.iter().zip(&labels) {
            assert_eq!(words[c[0] as usize], format!("cooler={}", row[0]));
            assert_eq!(words[c[1] as usize], format!("valve={}", row[1]));
        }
    }

    /// Values are sorted per axis, so a caption vocabulary does not depend on the order examples
    /// happened to arrive in — two runs over a shuffled corpus must build the same words.
    #[test]
    fn the_caption_vocabulary_does_not_depend_on_example_order() {
        let a = vec![vec![3, 100], vec![20, 80], vec![100, 90]];
        let b = vec![vec![100, 90], vec![3, 100], vec![20, 80]];
        let (wa, _) = build_words(&["x", "y"], &a);
        let (wb, _) = build_words(&["x", "y"], &b);
        assert_eq!(wa, wb);
    }

    /// The axes must stay separable: two examples differing on ONE axis differ in exactly one
    /// caption position. If they did not, a per-axis score would be measuring something blended.
    #[test]
    fn changing_one_label_changes_exactly_one_caption_word() {
        let labels = vec![vec![0, 5, 9], vec![1, 5, 9]];
        let (_, caps) = build_words(&["a", "b", "c"], &labels);
        let diff = caps[0].iter().zip(&caps[1]).filter(|(x, y)| x != y).count();
        assert_eq!(diff, 1);
    }

    fn docs(v: &[&[&[u32]]]) -> Vec<Vec<Vec<u32>>> {
        v.iter().map(|d| d.iter().map(|r| r.to_vec()).collect()).collect()
    }

    #[test]
    fn compaction_densely_renumbers_the_codes_training_actually_used() {
        // Training uses 5, 900 and 30000; held-out also uses 77, which training never saw.
        let d = docs(&[&[&[900, 5]], &[&[30000]], &[&[77, 5]]]);
        let (out, size, miss) = compact(&d, &[0, 1], &[2]);
        // Three training codes plus one reserved id for everything unseen.
        assert_eq!(size, 4);
        // Sorted, so 5 -> 0, 900 -> 1, 30000 -> 2, and anything else -> 3.
        assert_eq!(out[0], vec![vec![1, 0]]);
        assert_eq!(out[1], vec![vec![2]]);
        assert_eq!(out[2], vec![vec![3, 0]], "an unseen code goes to the reserved id");
        // One of the held-out example's two tokens was unseen.
        assert!((miss - 50.0).abs() < 1e-9, "miss rate was {miss}");
    }

    /// **The miss rate is the number that says whether compaction was free.** A vocabulary built
    /// from training that does not cover the held-out set is discarding held-out signal, and the
    /// accuracy that follows would be reported as if it had not.
    #[test]
    fn a_held_out_set_of_entirely_unseen_codes_reports_a_full_miss() {
        let d = docs(&[&[&[1, 2]], &[&[500, 501]]]);
        let (_, _, miss) = compact(&d, &[0], &[1]);
        assert!((miss - 100.0).abs() < 1e-9, "miss rate was {miss}");
    }

    /// Compaction must not merge two documents that were different, or the corpus quietly loses
    /// examples to collisions and every count after it is wrong.
    #[test]
    fn compaction_keeps_distinct_documents_distinct() {
        let d = docs(&[&[&[10, 20]], &[&[20, 10]], &[&[10, 10]]]);
        let (out, _, _) = compact(&d, &[0, 1, 2], &[]);
        assert_ne!(out[0], out[1]);
        assert_ne!(out[0], out[2]);
        assert_ne!(out[1], out[2]);
    }

    /// Compaction is built from the TRAINING examples only. A vocabulary fitted to codes that only
    /// the held-out set contains has already been told something about it.
    #[test]
    fn compaction_never_reserves_a_row_for_a_code_only_the_held_out_set_uses() {
        let d = docs(&[&[&[1]], &[&[2]], &[&[3]], &[&[4]]]);
        let (_, size, _) = compact(&d, &[0, 1], &[2, 3]);
        assert_eq!(size, 3, "two training codes plus one reserved id, and nothing for 3 or 4");
    }
}
