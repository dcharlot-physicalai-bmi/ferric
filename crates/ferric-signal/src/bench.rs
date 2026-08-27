//! The instruments a labelled-corpus result has to be read through.
//!
//! These live in the library rather than in one example because every corpus ingest needs the same
//! four, and a probe that exists twice gets fixed once. They are cheap, they run on CPU, and none
//! of them is a model — they are the things that decide whether a model's number means anything.
//!
//! ## Why each one exists
//!
//! **[`majority`]** — "at chance" and "predicting the training prior" look identical in an accuracy
//! column and are different failures. A five-class problem where one class holds 55% of the data
//! makes a constant predictor look like more than twice chance. This crate published a null once
//! whose whole legibility came from having this column.
//!
//! **[`nb_probe`]** — a classifier with no capacity to speak, reading the same tokens a language
//! model reads. When a decoder is emitting a constant, "the tokens carry nothing" and "the decoder
//! cannot extract it" are indistinguishable from the outside, and they call for opposite work. The
//! probe separates them in one run.
//!
//! **[`permutation_control`]** — the probe is only evidence if the same probe on the same features
//! fails when the labels are permuted. Run once at ten held-out examples this control reached
//! **+40 points**, as large as the effect being reported; at a hundred it sits near zero. A probe
//! quoted without its control is an accuracy column with extra steps.
//!
//! **[`shuffled`]** — deterministic presentation order. Condition-ordered corpora are the norm in
//! this field, and walking one in file order is enough on its own to collapse a decoder to a
//! constant.

use std::collections::{HashMap, HashSet};

/// Deterministic Fisher-Yates, so "shuffled" is reproducible and a reported number can be rerun.
pub fn shuffled(n: usize, seed: u64) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..n).collect();
    let mut s = seed;
    let mut next = move || {
        s = s.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = s;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    };
    for i in (1..n).rev() {
        let j = (next() % (i as u64 + 1)) as usize;
        idx.swap(i, j);
    }
    idx
}

/// Accuracy of always answering the most frequent TRAINING label, scored on the held-out set.
///
/// The training majority, not the overall one: a baseline that peeks at the held-out distribution
/// is not a baseline.
pub fn majority(labels: &[i32], train: &[usize], held: &[usize]) -> f64 {
    if held.is_empty() {
        return f64::NAN;
    }
    let mut counts: HashMap<i32, usize> = HashMap::new();
    for &i in train {
        *counts.entry(labels[i]).or_insert(0) += 1;
    }
    // Ties broken by the smaller label so the baseline is deterministic.
    let maj = counts
        .iter()
        .max_by_key(|&(&v, &c)| (c, std::cmp::Reverse(v)))
        .map(|(&v, _)| v)
        .unwrap_or(0);
    held.iter().filter(|&&i| labels[i] == maj).count() as f64 / held.len() as f64 * 100.0
}

/// Chance accuracy: one over the number of distinct labels present.
pub fn chance(labels: &[i32]) -> f64 {
    let n = labels.iter().collect::<HashSet<_>>().len().max(1);
    100.0 / n as f64
}

/// Multinomial naive Bayes over (run, token) counts — does the token stream separate the classes
/// at all, independent of whether a model can be trained to say so?
///
/// `docs[i]` is one example: a list of runs, each a list of token ids. **A feature is (run index,
/// token), not the token alone** — the same code from a pressure channel and a temperature channel
/// is not the same evidence, and pooling them throws away the one thing a multi-channel corpus
/// has.
///
/// Laplace smoothing over the vocabulary observed in training. Held-out tokens never seen in
/// training contribute the smoothed floor to every class, which is the right behaviour: they carry
/// no evidence either way.
pub fn nb_probe(docs: &[Vec<Vec<u32>>], labels: &[i32], train: &[usize], held: &[usize]) -> f64 {
    if held.is_empty() || train.is_empty() {
        return f64::NAN;
    }
    let feat = |run: usize, tok: u32| -> u64 { (run as u64) << 32 | tok as u64 };
    let classes: Vec<i32> = {
        let mut v: Vec<i32> = train.iter().map(|&i| labels[i]).collect();
        v.sort_unstable();
        v.dedup();
        v
    };
    let mut counts: Vec<HashMap<u64, f64>> = vec![HashMap::new(); classes.len()];
    let mut totals = vec![0.0f64; classes.len()];
    let mut vocab: HashSet<u64> = HashSet::new();
    for &i in train {
        let k = classes.iter().position(|&v| v == labels[i]).unwrap();
        for (r, run) in docs[i].iter().enumerate() {
            for &t in run {
                *counts[k].entry(feat(r, t)).or_insert(0.0) += 1.0;
                totals[k] += 1.0;
                vocab.insert(feat(r, t));
            }
        }
    }
    let v = vocab.len() as f64;
    let alpha = 1.0f64;
    let mut right = 0usize;
    for &i in held {
        let mut best = (f64::NEG_INFINITY, 0usize);
        for k in 0..classes.len() {
            let denom = totals[k] + alpha * v;
            let mut lp = 0.0f64;
            for (r, run) in docs[i].iter().enumerate() {
                for &t in run {
                    let n = counts[k].get(&feat(r, t)).copied().unwrap_or(0.0);
                    lp += ((n + alpha) / denom).ln();
                }
            }
            if lp > best.0 {
                best = (lp, k);
            }
        }
        if classes[best.1] == labels[i] {
            right += 1;
        }
    }
    right as f64 / held.len() as f64 * 100.0
}

/// How far above its own majority baseline the probe gets when the example-to-label assignment is
/// PERMUTED, over several permutations, reported as the worst case.
///
/// Permuting the labels and not the features is what leaves everything else — split, class
/// balance, and any correlation structure inside a label — exactly as it was. The worst of several
/// is reported because one lucky permutation is not a control.
///
/// **Worst-of-N grows with N, and that is the direction to err in.** Five rounds on one corpus gave
/// +9.0 points on an axis where a different five gave +2.0 — a control that swings five points
/// between draws is under-sampled, and under-sampling a control makes results look better than
/// they are. Twenty is the default the examples use; the number of rounds belongs next to the
/// figure, because "worst of five" and "worst of twenty" are not the same quantity.
///
/// A control near zero is what makes a positive probe mean something. **Read it before reading the
/// probe**, and read it against the same held-out size: it shrinks as n grows and is worth tens of
/// points at small n.
pub fn permutation_control(
    docs: &[Vec<Vec<u32>>],
    labels: &[i32],
    train: &[usize],
    held: &[usize],
    rounds: usize,
    seed: u64,
) -> f64 {
    let mut worst = f64::NEG_INFINITY;
    for k in 0..rounds.max(1) {
        let order = shuffled(labels.len(), seed ^ (k as u64).wrapping_mul(0x9E37_79B9));
        let permuted: Vec<i32> = order.iter().map(|&i| labels[i]).collect();
        let acc = nb_probe(docs, &permuted, train, held);
        let maj = majority(&permuted, train, held);
        worst = worst.max(acc - maj);
    }
    worst
}

/// A NON-LEARNED RECONSTRUCTION BASELINE AT A MATCHED BIT RATE.
///
/// An FSQ tokenizer over 32,768 codes spends **15 bits per patch**. This spends the same 15 bits in
/// a way that involves no training at all — 7 bits to name which of 128 DCT coefficients is
/// largest, 8 bits to quantize its value — and reconstructs from that one coefficient. Returns the
/// mean squared error against the patch it was given.
///
/// The 7-bit index assumes a 128-sample patch, which is where the budgets line up exactly; at other
/// patch lengths the index costs `log2(n)` and the comparison is approximate. Callers are expected
/// to say which they are doing.
///
/// It exists because reconstruction was reported in this crate for a long time with no baseline at
/// all, while every classification figure carried three. Decibels against a signal's own variance
/// say how much of it survived; they do not say whether a scheme that learned nothing would have
/// done as well. **Adding this reversed two of four rows** the first time it was run: a
/// 9.5M-parameter trained tokenizer lost to one untrained coefficient on one corpus and tied on
/// another. A corpus where the learned model barely beats one coefficient is telling you about the
/// SIGNAL and not about the training.
pub fn dct_baseline(patch: &[f32]) -> f64 {
    let n = patch.len();
    // DCT-II, O(n^2). n is 128 here and this runs once per held-out patch, not in a training loop.
    let mut coef = vec![0.0f64; n];
    for (k, c) in coef.iter_mut().enumerate() {
        let mut acc = 0.0f64;
        for (i, &x) in patch.iter().enumerate() {
            acc += x as f64
                * ((std::f64::consts::PI / n as f64) * (i as f64 + 0.5) * k as f64).cos();
        }
        *c = acc;
    }
    let (best, &peak) = coef
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.abs().total_cmp(&b.1.abs()))
        .unwrap();
    // 8 bits for the value, over the range a coefficient of a unit-scale patch cannot exceed.
    //
    // THE BOUND IS `n`, NOT `sqrt(n)`. By Cauchy-Schwarz a DCT-II coefficient is at most
    // `sqrt(n * sum(x^2))`, and a RevIn-normalised patch has `sum(x^2) = n`, so the bound is `n`.
    // The first version of this used `4*sqrt(n)` — 45 where the coefficient of a pure cosine is
    // `n/2` = 64 — so every strong coefficient CLAMPED, and a pure cosine came back at 10.7 dB
    // instead of 40+. That understates the baseline, which flatters every model measured against
    // it, which is the direction a baseline must never be wrong in.
    let span = n as f64;
    let q = ((peak / span).clamp(-1.0, 1.0) * 127.0).round() / 127.0 * span;
    // Inverse DCT-III of the single retained coefficient.
    let mut se = 0.0f64;
    for (i, &x) in patch.iter().enumerate() {
        let r = 2.0 / n as f64
            * q
            * ((std::f64::consts::PI / n as f64) * (i as f64 + 0.5) * best as f64).cos();
        se += (x as f64 - r) * (x as f64 - r);
    }
    se / n as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Documents whose tokens name their own class, so a working probe must be perfect and a
    /// broken one cannot be.
    fn separable(n: usize, classes: usize) -> (Vec<Vec<Vec<u32>>>, Vec<i32>) {
        let mut docs = Vec::new();
        let mut labels = Vec::new();
        for i in 0..n {
            let c = i % classes;
            docs.push(vec![vec![c as u32; 8]]);
            labels.push(c as i32);
        }
        (docs, labels)
    }

    /// A quarter held out, chosen by a deterministic shuffle rather than by `i % 4`.
    ///
    /// THE FIRST VERSION OF THIS HELPER TOOK EVERY FOURTH INDEX, and the separable fixture below
    /// assigned classes with `i % 4`. The two periods aligned, so every held-out example carried
    /// the one class that never appeared in training and the probe scored exactly 0.0 — which
    /// reads as a broken probe and was a broken fixture. Same shape as a condition-ordered corpus
    /// walked in corpus order: a periodic split lines up with periodic structure in the data.
    /// [`balanced`] now asserts against it directly.
    fn split(n: usize) -> (Vec<usize>, Vec<usize>) {
        let order = shuffled(n, 4242);
        let cut = n / 4;
        let mut held: Vec<usize> = order[..cut].to_vec();
        let mut train: Vec<usize> = order[cut..].to_vec();
        held.sort_unstable();
        train.sort_unstable();
        (train, held)
    }

    /// Every class must appear on BOTH sides of a split, or the test measures something other than
    /// what it says. Asserted rather than assumed, because assuming it is what broke this file.
    fn balanced(labels: &[i32], train: &[usize], held: &[usize]) {
        let tr: HashSet<i32> = train.iter().map(|&i| labels[i]).collect();
        let hd: HashSet<i32> = held.iter().map(|&i| labels[i]).collect();
        assert_eq!(tr, hd, "split does not carry the same classes on both sides");
        assert!(tr.len() > 1, "a single-class split tests nothing");
    }

    #[test]
    fn the_probe_is_perfect_on_perfectly_separable_tokens() {
        let (docs, labels) = separable(80, 4);
        let (tr, hd) = split(80);
        balanced(&labels, &tr, &hd);
        assert_eq!(nb_probe(&docs, &labels, &tr, &hd), 100.0);
    }

    /// THE INSTRUMENT'S OWN SELF-TEST. On documents whose tokens name their class, the probe is
    /// perfect and permuting the labels must destroy that. A control that stayed high on features
    /// which certainly carry the label would mean the control is broken, and every starred result
    /// it had ever waved through would be unsupported.
    #[test]
    fn the_permutation_control_collapses_on_features_that_certainly_carry_the_label() {
        let (docs, labels) = separable(400, 4);
        let (tr, hd) = split(400);
        balanced(&labels, &tr, &hd);
        let effect = nb_probe(&docs, &labels, &tr, &hd) - majority(&labels, &tr, &hd);
        let ctl = permutation_control(&docs, &labels, &tr, &hd, 5, 99);
        assert!(effect > 60.0, "effect was only {effect:+.1} points");
        // The criterion is the one a result is read by: the effect must be well clear of what
        // permuted labels can produce. Not "the control is near zero" — this fixture has a
        // vocabulary of FOUR tokens, so a permuted-label classifier can memorise all of it and the
        // control runs around +20. On a real corpus with thousands of codes it is far smaller.
        // A control this size next to an effect this size is still a result; a control this size
        // next to a +25 effect would not be, which is the comparison to make every time.
        assert!(effect > ctl * 2.0, "control {ctl:+.1} against an effect of {effect:+.1}");
    }

    /// THE CONTROL IS A FUNCTION OF THE HELD-OUT SIZE, and that is the whole reason it is reported
    /// next to the effect rather than assumed to be zero. Measured on this crate's own corpus: at
    /// ten held-out examples a permuted-label control reached **+40 points**, as large as the
    /// effect being reported; at a hundred it sat at +1 to +9. Here the same shrinkage is asserted
    /// directly, on features that carry NOTHING, where the control is the only thing being
    /// measured.
    #[test]
    fn the_control_shrinks_as_the_held_out_set_grows() {
        let noise = |n: usize| -> (Vec<Vec<Vec<u32>>>, Vec<i32>) {
            let order = shuffled(n, 77);
            let docs = (0..n)
                .map(|i| vec![(0..6).map(|k| ((order[i] * 31 + k * 7) % 97) as u32).collect()])
                .collect();
            (docs, (0..n).map(|i| (i % 4) as i32).collect())
        };
        let at = |n: usize| {
            let (docs, labels) = noise(n);
            let (tr, hd) = split(n);
            permutation_control(&docs, &labels, &tr, &hd, 5, 5)
        };
        let small = at(40);
        let large = at(1200);
        assert!(
            large < small,
            "control at n=300 held out was {large:+.1}, at n=10 it was {small:+.1}"
        );
    }

    /// A feature that is the same for every example carries nothing, and the probe must land on
    /// the majority baseline rather than above it.
    #[test]
    fn the_probe_sits_at_majority_when_every_document_is_identical() {
        let n = 120;
        let docs: Vec<Vec<Vec<u32>>> = (0..n).map(|_| vec![vec![7u32; 5]]).collect();
        // Deliberately unbalanced, so majority and chance differ and the distinction is testable.
        let labels: Vec<i32> = (0..n).map(|i| if i % 5 == 0 { 1 } else { 0 }).collect();
        let (tr, hd) = split(n);
        let maj = majority(&labels, &tr, &hd);
        let acc = nb_probe(&docs, &labels, &tr, &hd);
        assert!(maj > 70.0, "majority should be high here, was {maj}");
        assert!((acc - maj).abs() < 1e-9, "probe {acc} against majority {maj}");
    }

    /// The baseline is the TRAINING majority. A baseline computed on the held-out labels is not a
    /// baseline, and would be silently right whenever the split happened to be balanced.
    #[test]
    fn the_majority_baseline_reads_the_training_split_not_the_held_out_one() {
        //     index: 0 1 2 3
        //     label: 0 0 0 1     train = 0,1,2  held = 3
        let labels = vec![0, 0, 0, 1];
        let tr = vec![0, 1, 2];
        let hd = vec![3];
        // Training majority is 0; the held-out example is 1; so the baseline scores ZERO here.
        // A baseline peeking at the held-out set would answer 1 and score 100.
        assert_eq!(majority(&labels, &tr, &hd), 0.0);
    }

    #[test]
    fn chance_is_one_over_the_number_of_classes_present() {
        assert!((chance(&[0, 1, 2, 0, 1, 2]) - 100.0 / 3.0).abs() < 1e-9);
        assert_eq!(chance(&[5, 5, 5]), 100.0);
    }

    /// A per-channel feature must beat a pooled one when the SAME token means different things in
    /// different channels — which is the situation in every multi-channel corpus.
    #[test]
    fn features_are_channel_tagged_so_the_same_token_in_two_channels_is_two_features() {
        // Class 0: channel A says 1, channel B says 2. Class 1: the reverse. Pooled counts are
        // identical for both classes; only the channel tag separates them.
        let n = 80;
        let mut docs = Vec::new();
        let mut labels = Vec::new();
        for i in 0..n {
            if i % 2 == 0 {
                docs.push(vec![vec![1u32; 6], vec![2u32; 6]]);
                labels.push(0);
            } else {
                docs.push(vec![vec![2u32; 6], vec![1u32; 6]]);
                labels.push(1);
            }
        }
        let (tr, hd) = split(n);
        balanced(&labels, &tr, &hd);
        assert_eq!(nb_probe(&docs, &labels, &tr, &hd), 100.0);
        // Pooling the two runs into one makes the two classes' token counts IDENTICAL, so the
        // probe loses all of it and falls to answering a constant. It does not land exactly on the
        // majority baseline: with the likelihoods tied, the class with fewer training tokens wins
        // on its smaller denominator, and that need not be the majority class. Constant is the
        // claim; which constant is an artifact.
        let pooled: Vec<Vec<Vec<u32>>> =
            docs.iter().map(|d| vec![d.iter().flatten().copied().collect()]).collect();
        let acc = nb_probe(&pooled, &labels, &tr, &hd);
        assert!(acc < 65.0, "pooled probe scored {acc}, so pooling did not destroy the signal");
    }

    #[test]
    fn shuffling_is_a_permutation_and_is_reproducible() {
        let a = shuffled(500, 7);
        assert_eq!(a, shuffled(500, 7));
        assert_ne!(a, shuffled(500, 8));
        let mut sorted = a.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, (0..500).collect::<Vec<_>>());
    }

    /// A PURE COSINE IS EXACTLY ONE DCT COEFFICIENT, so a one-coefficient baseline must reconstruct
    /// it almost perfectly. If the transform or its inverse were mis-normalised the error would be
    /// large here, the baseline would be understated, and every model compared against it would
    /// look better than it is.
    #[test]
    fn one_coefficient_reconstructs_a_pure_cosine() {
        let n = 128usize;
        for k in [1usize, 5, 17, 60] {
            let patch: Vec<f32> = (0..n)
                .map(|i| {
                    ((std::f64::consts::PI / n as f64) * (i as f64 + 0.5) * k as f64).cos() as f32
                })
                .collect();
            let var = {
                let m = patch.iter().sum::<f32>() as f64 / n as f64;
                patch.iter().map(|&v| (v as f64 - m) * (v as f64 - m)).sum::<f64>() / n as f64
            };
            let mse = dct_baseline(&patch);
            let snr = 10.0 * (var / mse.max(1e-18)).log10();
            assert!(snr > 20.0, "k={k}: only {snr:.1} dB from one coefficient on a pure cosine");
        }
    }

    /// And it must NOT reconstruct broadband noise, which is the other half of the claim: the
    /// baseline is weak where a signal has no dominant component, so a model that beats it there is
    /// doing real work.
    #[test]
    fn one_coefficient_barely_touches_broadband_noise() {
        let n = 128usize;
        let order = shuffled(n, 31337);
        let patch: Vec<f32> =
            (0..n).map(|i| ((order[i] as f32 / n as f32) - 0.5) * 2.0).collect();
        let m = patch.iter().sum::<f32>() as f64 / n as f64;
        let var = patch.iter().map(|&v| (v as f64 - m) * (v as f64 - m)).sum::<f64>() / n as f64;
        let snr = 10.0 * (var / dct_baseline(&patch).max(1e-18)).log10();
        assert!(snr < 6.0, "one coefficient captured {snr:.1} dB of broadband noise");
    }

    /// The budgets line up exactly at a 128-sample patch: 7 bits of index plus 8 of value is 15,
    /// and an FSQ codebook of 32,768 is 15. Asserted so a future change to either side has to
    /// notice that it broke the comparison.
    #[test]
    fn the_baseline_spends_the_same_fifteen_bits_as_the_codebook() {
        let index_bits = (128f64).log2() as u32;
        let value_bits = 8u32;
        assert_eq!(index_bits + value_bits, 15);
        assert_eq!(crate::Fsq::signal_15bit().codebook_size(), 1 << 15);
    }

    /// A constant patch has no variance to recover and must not produce a NaN or a negative error.
    #[test]
    fn a_flat_patch_is_handled_rather_than_dividing_by_nothing() {
        let mse = dct_baseline(&vec![0.7f32; 128]);
        assert!(mse.is_finite() && mse >= 0.0, "flat patch gave {mse}");
    }
}
