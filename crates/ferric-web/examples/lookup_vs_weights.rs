//! **Is it cheaper to look a fact up than to know it?** — the small-model thesis, graded.
//!
//! The claim this exists to test is the one that decides whether browser-native AI is a curiosity or
//! the default: that a model small enough to ship to a tab, with the corpus *outside* its weights,
//! answers more questions correctly than a model many times larger answering from memory. If that
//! holds, parameter count stops being the axis of progress for a large class of real work.
//!
//! It is a claim that is very easy to rig, so the construction is deliberately hostile to itself:
//!
//! - **The facts are real and checkable**, not invented for this file. A model that genuinely knows
//!   them can win from weights alone, which is the only way the closed-book arm is a real baseline.
//! - **Every answer-bearing passage is flanked by two topically adjacent distractors** — Chernobyl
//!   beside Three Mile Island and Fukushima, gold beside silver and platinum. Retrieval has to pick
//!   the right one of three near neighbours, so it can genuinely fail, and it does.
//! - **Passages are phrased away from the questions.** The corpus says "suffered a catastrophic power
//!   excursion during a late-night safety test on 26 April 1986"; the question asks "in which year".
//! - **No positional shortcut can pass.** In the invented-facts corpus every passage carries three
//!   numbers (the value, a revision, a catalogue entry) and only half lead with the answer, so
//!   "emit the first number you see" scores 11/22, as does "emit the last". Before that fix the
//!   answer was always the leading number and a pure copying heuristic would have scored 22/22 —
//!   the bench would have measured copying and reported it as comprehension.
//! - **Both arms are graded by the same word-boundary matcher** against the same answer key, and the
//!   per-task outcomes are printed, so any reader can check the grader rather than trust the tally.
//!
//! ## The prediction, registered before the first run
//!
//! The retrieval arm wins on accuracy, and the interesting failure is not retrieval missing the
//! passage but the small model failing to *extract* the answer from a passage it was handed — an
//! extraction ceiling, not a search one. If instead the large model wins outright, the thesis as
//! stated is wrong for factual recall and this file says so.
//!
//! ## What this can and cannot measure on this machine
//!
//! Joules: **not on AC power**, and `ferric-joule` refuses to invent them rather than reporting a
//! TDP guess wearing a measurement's clothes. So the energy arm returns `None` here and the run
//! reports the half that IS measurable anywhere — the success rates, which are precisely the
//! denominator any future joules figure gets divided by. On a machine with a live sensor the same
//! run yields a full `Saving` through `compare_tasks`; the code path is identical either way.
//!
//!   cargo run -p ferric-web --example lookup_vs_weights --release -- \
//!       <small.gguf> <big.gguf> web/qa.tsv web/qa_corpus.txt
use ferric_joule::{grade_tasks, Meter};

/// One graded question.
struct Q {
    ask: String,
    /// Accepted answers, any one of which counts. Alternatives are `|`-separated in the key file.
    accept: Vec<String>,
}

/// Word-boundary, case-insensitive containment.
///
/// A plain `contains` is wrong here and quietly inflates both arms: the answer `2` is inside `1986`,
/// `212` and `2007`, all of which appear in this corpus, so a substring grader scores the smallest
/// prime correct whenever the model emits any of a dozen unrelated numbers. Both arms are graded by
/// this same function, so the residual strictness — a right answer in unexpected words scores zero —
/// costs both of them, and it costs the closed-book arm slightly more, which is the direction that
/// works against the thesis rather than for it.
fn hit(text: &str, needle: &str) -> bool {
    let (t, n) = (text.to_lowercase(), needle.to_lowercase());
    let tb = t.as_bytes();
    let mut from = 0;
    while let Some(i) = t[from..].find(&n).map(|x| x + from) {
        let before_ok = i == 0 || !(tb[i - 1] as char).is_alphanumeric();
        let j = i + n.len();
        let after_ok = j >= tb.len() || !(tb[j] as char).is_alphanumeric();
        if before_ok && after_ok { return true; }
        from = i + 1;
    }
    false
}

fn graded(text: &str, q: &Q) -> bool { q.accept.iter().any(|a| hit(text, a)) }

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let (mut d, mut na, mut nb) = (0.0f32, 0.0f32, 0.0f32);
    for i in 0..a.len() { d += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i]; }
    if na <= 0.0 || nb <= 0.0 { 0.0 } else { d / (na.sqrt() * nb.sqrt()) }
}

fn main() { pollster::block_on(run()); }

async fn run() {
    let a: Vec<String> = std::env::args().collect();
    let small_p = a.get(1).expect("usage: lookup_vs_weights <small.gguf> <big.gguf> <qa.tsv> <corpus.txt> [n_gen]");
    let big_p = a.get(2).expect("big.gguf");
    let qa_p = a.get(3).expect("qa.tsv");
    let corpus_p = a.get(4).expect("corpus.txt");
    let n_gen: usize = a.get(5).and_then(|s| s.parse().ok()).unwrap_or(28);
    // Optional SEPARATE retriever checkpoint. When absent the generator embeds its own corpus, which
    // is what the first run of this bench did and is why the retriever collapsed: `FerricModel::embed`
    // pools the last hidden state, and its own doc warns that doing so on a checkpoint not TRAINED
    // for embedding "hands back plausible cosine scores that mean nothing". The guard in that method
    // checks the runtime kind (Dense), not whether the weights were trained for the task, so a
    // generative model passes it silently. Measured on the first run: mean top1-top2 margin 0.0107,
    // 13 distinct passages chosen for 22 questions.
    let retr_p = a.get(6).map(String::as_str);

    let qa_text = std::fs::read_to_string(qa_p).expect("read qa");
    let qs: Vec<Q> = qa_text.lines().filter(|l| !l.trim().is_empty()).map(|l| {
        let (ask, acc) = l.split_once('\t').expect("qa.tsv is question<TAB>answer|answer");
        Q { ask: ask.trim().into(), accept: acc.split('|').map(|s| s.trim().to_string()).collect() }
    }).collect();
    assert!(!qs.is_empty(), "no questions");

    let corpus_text = std::fs::read_to_string(corpus_p).expect("read corpus");
    let chunks: Vec<String> = corpus_text.split("\n\n").map(|c| c.trim().to_string())
        .filter(|c| !c.is_empty()).collect();
    assert!(chunks.len() > qs.len(), "a corpus with no distractors cannot test retrieval");

    let (sm_bytes, bg_bytes) = (std::fs::metadata(small_p).unwrap().len(), std::fs::metadata(big_p).unwrap().len());
    println!("lookup vs weights — {} questions, {} corpus chunks", qs.len(), chunks.len());
    println!("  candidate (weights OUTSIDE): {small_p}  {:.0} MB + corpus", sm_bytes as f64 / 1e6);
    println!("  baseline  (weights INSIDE):  {big_p}  {:.0} MB, closed book", bg_bytes as f64 / 1e6);
    let retr_bytes = retr_p.map(|r| std::fs::metadata(r).unwrap().len()).unwrap_or(0);
    // The candidate must be charged for EVERY byte it ships. A separate retriever is a second model
    // in the tab, and quietly comparing only the generator against the baseline would be the same
    // class of error as a baseline measured at 3.4% utilisation.
    println!("  ratio: the baseline ships {:.1}x the candidate's total bytes ({:.0} MB incl. retriever)\n",
             bg_bytes as f64 / (sm_bytes + retr_bytes) as f64, (sm_bytes + retr_bytes) as f64 / 1e6);

    // ---- candidate arm: one small model as BOTH retriever and generator ----
    let small = ferric_web::FerricModel::load(std::fs::read(small_p).expect("read small")).await.expect("load small");
    let retriever = match retr_p {
        Some(rp) => Some(ferric_web::FerricModel::load(std::fs::read(rp).expect("read retriever")).await.expect("load retriever")),
        None => None,
    };
    let embed_with = retriever.as_ref().unwrap_or(&small);
    match retr_p {
        Some(rp) => println!("  retriever: {rp}  {:.0} MB (separate, trained for embedding)",
                             std::fs::metadata(rp).unwrap().len() as f64 / 1e6),
        None => println!("  retriever: the generator itself, pooling its own last hidden state"),
    }
    eprintln!("embedding {} chunks...", chunks.len());
    let t_embed = std::time::Instant::now();
    let mut vecs: Vec<Vec<f32>> = Vec::with_capacity(chunks.len());
    for (i, c) in chunks.iter().enumerate() {
        vecs.push(embed_with.embed(c.clone()).await.expect("embed chunk"));
        if i % 20 == 0 { eprintln!("  {i}/{}", chunks.len()); }
    }
    // Charged separately and honestly: this is a ONE-TIME cost per (corpus, model), paid here and
    // shipped as a file in the deployed case. Folding it into per-query cost would overstate the
    // candidate's price by the corpus size; hiding it entirely would understate the system's.
    let embed_secs = t_embed.elapsed().as_secs_f64();

    // Retrieval is resolved up front so the graded closure below does only generation, keeping the
    // two arms' timed work comparable. `retrieved` is kept for the audit line.
    let mut retrieved: Vec<usize> = Vec::with_capacity(qs.len());
    let mut margins: Vec<f32> = Vec::with_capacity(qs.len());
    for q in &qs {
        let qv = embed_with.embed(q.ask.clone()).await.expect("embed question");
        let mut scored: Vec<(usize, f32)> = vecs.iter().enumerate().map(|(i, v)| (i, cosine(&qv, v))).collect();
        scored.sort_by(|x, y| y.1.partial_cmp(&x.1).unwrap());
        retrieved.push(scored[0].0);
        margins.push(scored[0].1 - scored[1].1);
    }

    let t_cand = std::time::Instant::now();
    let mut cand_out: Vec<String> = Vec::with_capacity(qs.len());
    for (i, q) in qs.iter().enumerate() {
        let p = format!("{}\n\nQuestion: {}\nAnswer:", chunks[retrieved[i]], q.ask);
        cand_out.push(small.generate_plain(&p, n_gen).await.unwrap_or_default());
    }

    let cand_secs = t_cand.elapsed().as_secs_f64();

    // ---- CONTROL: the same small model, same prompt shape, NO passage ----
    // Without this the bench cannot support its own conclusion. A high retrieval score is equally
    // consistent with "the passage did the work" and with "this small model already knew these
    // facts", and those imply opposite things about whether knowledge can live outside the weights.
    // The control separates them, and it is the arm most likely to embarrass the thesis.
    let mut ctrl_out: Vec<String> = Vec::with_capacity(qs.len());
    for q in qs.iter() {
        let p = format!("Question: {}\nAnswer:", q.ask);
        ctrl_out.push(small.generate_plain(&p, n_gen).await.unwrap_or_default());
    }
    drop(retriever);
    drop(small);

    // ---- baseline arm: the larger model, from weights alone ----
    let big = ferric_web::FerricModel::load(std::fs::read(big_p).expect("read big")).await.expect("load big");
    let t_base = std::time::Instant::now();
    let mut base_out: Vec<String> = Vec::with_capacity(qs.len());
    for q in qs.iter() {
        let p = format!("Question: {}\nAnswer:", q.ask);
        base_out.push(big.generate_plain(&p, n_gen).await.unwrap_or_default());
    }
    let base_secs = t_base.elapsed().as_secs_f64();
    drop(big);

    // Grading goes through ferric-joule so the success counts are TALLIED from the graded outcomes
    // rather than asserted. No meter is available on AC power, so this path deliberately yields no
    // `Saving` at all — see the note where the energy verdict is printed.
    let idx: Vec<usize> = (0..qs.len()).collect();
    let (bok, cok, (_gb, _gc)) = grade_tasks(&idx,
        |i| graded(&base_out[*i], &qs[*i]),
        |i| graded(&cand_out[*i], &qs[*i]));
    let (ctrl, _, _) = grade_tasks(&idx, |i| graded(&ctrl_out[*i], &qs[*i]), |_| false);

    println!("{:<58} {:^7} {:^7} {:^7}", "question", "big", "small", "lookup");
    for (i, q) in qs.iter().enumerate() {
        let short: String = q.ask.chars().take(56).collect();
        let m = |b: bool| if b { "ok" } else { "-" };
        println!("{:<58} {:^7} {:^7} {:^7}  [c{} m{:.3}]",
                 short, m(bok[i]), m(ctrl[i]), m(cok[i]), retrieved[i], margins[i]);
    }

    let (nb, nc) = (bok.iter().filter(|x| **x).count(), cok.iter().filter(|x| **x).count());
    let nctl = ctrl.iter().filter(|x| **x).count();
    let n = qs.len();
    println!("\n  weights inside ({:.0} MB): {nb}/{n} correct  ({:.0}%)  in {base_secs:.1}s generating",
             bg_bytes as f64 / 1e6, nb as f64 / n as f64 * 100.0);
    println!("  CONTROL, small model closed book ({:.0} MB): {nctl}/{n} correct  ({:.0}%)",
             sm_bytes as f64 / 1e6, nctl as f64 / n as f64 * 100.0);
    println!("  weights outside ({:.0} MB + {} chunks): {nc}/{n} correct  ({:.0}%)  in {cand_secs:.1}s generating",
             sm_bytes as f64 / 1e6, chunks.len(), nc as f64 / n as f64 * 100.0);
    println!("  corpus embedding, one time per (corpus, model): {embed_secs:.1}s, amortised over every query after the first");

    // Where the candidate lost, say WHICH failure it was. Retrieval finding the wrong passage and the
    // model failing to read the right one are different defects with different fixes, and a single
    // accuracy number cannot tell them apart. This is the registered prediction's actual test.
    // Retrieval quality measured over ALL questions, not only the failures. The failure-only view is
    // the trap: a question can retrieve a useless passage and still be scored correct because the
    // small model knew the answer, which inflates apparent retrieval quality by exactly the amount
    // the control arm is there to detect.
    let hits_at_1 = (0..n).filter(|&i| qs[i].accept.iter().any(|acc| hit(&chunks[retrieved[i]], acc))).count();
    let mean_margin: f32 = margins.iter().sum::<f32>() / margins.len() as f32;
    let distinct: std::collections::BTreeSet<usize> = retrieved.iter().copied().collect();
    println!("  retrieval@1: {hits_at_1}/{n} questions retrieved a passage that CONTAINS the answer \
              (mean top1-top2 margin {mean_margin:.4}, {} distinct chunks chosen for {n} questions)",
             distinct.len());
    if distinct.len() * 2 < n {
        println!("    ⚠ the retriever is COLLAPSING: fewer than half as many distinct passages as \
                  questions means one chunk is winning for unrelated queries, and a mean margin near \
                  zero means top-1 and top-2 are effectively tied. Ranking is near-arbitrary.");
    }

    let mut retrieval_miss = 0usize;
    let mut extraction_miss = 0usize;
    for i in 0..n {
        if cok[i] { continue; }
        // The passage is the right one iff it contains the answer at all.
        if qs[i].accept.iter().any(|acc| hit(&chunks[retrieved[i]], acc)) { extraction_miss += 1; }
        else { retrieval_miss += 1; }
    }
    println!("  of {} lookup failures: {retrieval_miss} retrieved the wrong passage, {extraction_miss} were handed \
              the answer and did not use it", n - nc);

    // The only number that isolates what the CORPUS contributed, as opposed to what the small model
    // already carried. If this is not clearly positive, the passage is decoration and the thesis
    // fails on its own bench regardless of how the two headline arms compare.
    let gained: Vec<usize> = (0..n).filter(|&i| cok[i] && !ctrl[i]).collect();
    let lost: Vec<usize> = (0..n).filter(|&i| !cok[i] && ctrl[i]).collect();
    println!("  retrieval's own contribution: +{} answered only WITH the passage, -{} lost by having it \
              (net {:+})", gained.len(), lost.len(), gained.len() as i64 - lost.len() as i64);
    if !lost.is_empty() {
        println!("    a passage that made the model WORSE is the interesting failure; questions: {lost:?}");
    }

    // The energy verdict, and the reason there isn't one.
    match ferric_joule::MacBattery::new().filter(|m| m.available()) {
        Some(m) => println!("\n  a live sensor is present ({}); re-run under `compare_tasks` for a full Saving", m.source()),
        None => println!("\n  joules: UNAVAILABLE. This machine is on AC power, so no energy sensor is\n  \
                  readable and ferric-joule reports nothing rather than a nameplate estimate wearing a\n  \
                  measurement's clothes. The success rates above are the denominator any joules figure\n  \
                  gets divided by, and they are the half that is measurable on any machine."),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_grader_does_not_find_an_answer_inside_an_unrelated_number() {
        // This is the defect that would rig the whole bench in the thesis's favour, and it is one
        // character of code away: `text.contains("2")` is true for "1986", "212" and "2007", every
        // one of which is an answer to some OTHER question in this very key file. A substring grader
        // therefore scores "what is the smallest prime" correct whenever a model emits any year.
        assert!(!hit("The disaster occurred in 1986.", "2"));
        assert!(!hit("Water boils at 212 degrees.", "2"));
        assert!(!hit("The iPhone launched in 2007.", "2"));
        assert!(hit("The smallest prime number is 2.", "2"));
        assert!(hit("2 is prime", "2"), "an answer at the start of the text still counts");
        assert!(hit("the answer is 2", "2"), "an answer at the very end still counts");
    }

    #[test]
    fn the_grader_is_case_insensitive_but_not_word_blind() {
        assert!(hit("It is the Amazon.", "amazon"));
        assert!(hit("titan is the largest", "Titan"));
        // A longer word that merely CONTAINS the answer must not score: "gold" inside "golden" is the
        // same class of error as "2" inside "212", and it is the one that bites on short element symbols.
        assert!(!hit("a golden age of chemistry", "gold"));
        assert!(!hit("Winds swept the plain", "W"), "a bare symbol must not match inside a word");
        assert!(hit("The symbol is W.", "W"));
    }

    #[test]
    fn multi_word_answers_and_alternatives_both_work() {
        let q = Q { ask: "x".into(), accept: vec!["gold".into(), "Au".into()] };
        assert!(graded("the element is Au", &q), "either alternative counts");
        assert!(graded("it is gold, symbol Au", &q));
        assert!(!graded("it is silver", &q));
        let d = Q { ask: "x".into(), accept: vec!["Challenger Deep".into()] };
        assert!(graded("known as the Challenger Deep in the Mariana Trench", &d));
        assert!(!graded("the Challenger space shuttle", &d), "a partial phrase must not score");
    }
}
