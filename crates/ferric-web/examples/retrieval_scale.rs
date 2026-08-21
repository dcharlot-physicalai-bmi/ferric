//! **How far does "the corpus can be arbitrarily large" actually go?** — the decay curve, not two points.
//!
//! `lookup_vs_weights` answers the question end to end, which costs three generation passes per run
//! and gives ONE corpus size per ~10 minutes. That is enough to say retrieval@1 held at 66 and at
//! 1000, and not enough to say anything about the shape between or beyond them — two points fit any
//! curve, including one that falls off a cliff at 1100.
//!
//! So this embeds the corpus and the questions ONCE and then does pure arithmetic over nested
//! subsets. Every size shares the same 22 answer-bearing passages and the same embeddings, so the
//! only thing that changes across the sweep is how many distractors the right answer has to beat —
//! which is exactly the variable the claim is about, isolated from everything else.
//!
//! Two numbers per size, and they say different things:
//!   * **retrieval@1** is the pass/fail — did the top-ranked chunk contain the answer.
//!   * **margin** (top1 − top2) is the headroom. retrieval@1 can sit at 100% while the margin decays
//!     toward zero, and at that point the next doubling is a coin flip. A sweep that reported only
//!     the first would call a system healthy right up until it wasn't.
//!
//!   cargo run -p ferric-web --example retrieval_scale --release -- <retriever.gguf> <qa.tsv> <corpus.txt>
use std::collections::BTreeSet;

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let (mut d, mut na, mut nb) = (0.0f32, 0.0f32, 0.0f32);
    for i in 0..a.len() { d += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i]; }
    if na <= 0.0 || nb <= 0.0 { 0.0 } else { d / (na.sqrt() * nb.sqrt()) }
}

/// Word-boundary containment — the same rule `lookup_vs_weights` grades with, and for the same
/// reason: `contains("2")` is true of "1986", "212" and "2007", which are answers to other questions.
fn hit(text: &str, needle: &str) -> bool {
    let (t, n) = (text.to_lowercase(), needle.to_lowercase());
    let tb = t.as_bytes();
    let mut from = 0;
    while let Some(i) = t[from..].find(&n).map(|x| x + from) {
        let j = i + n.len();
        if (i == 0 || !(tb[i - 1] as char).is_alphanumeric())
            && (j >= tb.len() || !(tb[j] as char).is_alphanumeric()) { return true; }
        from = i + 1;
    }
    false
}

fn main() { pollster::block_on(run()); }

async fn run() {
    let a: Vec<String> = std::env::args().collect();
    let rp = a.get(1).expect("usage: retrieval_scale <retriever.gguf> <qa.tsv> <corpus.txt>");
    let qp = a.get(2).expect("qa.tsv");
    let cp = a.get(3).expect("corpus.txt");

    let qa_text = std::fs::read_to_string(qp).expect("read qa");
    let qs: Vec<(String, Vec<String>)> = qa_text.lines().filter(|l| !l.trim().is_empty()).map(|l| {
        let (ask, acc) = l.split_once('\t').expect("question<TAB>answer|answer");
        (ask.trim().to_string(), acc.split('|').map(|s| s.trim().to_string()).collect())
    }).collect();
    let corpus = std::fs::read_to_string(cp).expect("read corpus");
    let chunks: Vec<String> = corpus.split("\n\n").map(|c| c.trim().to_string()).filter(|c| !c.is_empty()).collect();

    // Partition ONCE, by whether a chunk contains any answer. Every subset below keeps all of the
    // answer-bearing chunks and varies only the distractor count, so the sweep isolates crowding.
    let bearing: Vec<usize> = (0..chunks.len())
        .filter(|&i| qs.iter().any(|(_, acc)| acc.iter().any(|x| hit(&chunks[i], x)))).collect();
    let filler: Vec<usize> = (0..chunks.len()).filter(|i| !bearing.contains(i)).collect();
    assert_eq!(bearing.len(), qs.len(),
               "{} questions but {} answer-bearing chunks — the corpus and key disagree, and every \
                number below would be measuring that instead of retrieval", qs.len(), bearing.len());

    let m = ferric_web::FerricModel::load(std::fs::read(rp).expect("read retriever")).await.expect("load");
    eprintln!("embedding {} chunks + {} questions once...", chunks.len(), qs.len());
    let mut cv: Vec<Vec<f32>> = Vec::with_capacity(chunks.len());
    for (i, c) in chunks.iter().enumerate() {
        cv.push(m.embed(c.clone()).await.expect("embed chunk"));
        if i % 100 == 0 { eprintln!("  {i}/{}", chunks.len()); }
    }
    let mut qv: Vec<Vec<f32>> = Vec::with_capacity(qs.len());
    for (ask, _) in &qs { qv.push(m.embed(ask.clone()).await.expect("embed question")); }

    println!("retrieval scaling — {} answer-bearing chunks among up to {} total", bearing.len(), chunks.len());
    println!("retriever {rp}\n");
    println!("{:>7}  {:>11}  {:>9}  {:>9}  {:>8}", "chunks", "retrieval@1", "margin", "top1", "distinct");

    let mut prev: Option<(usize, f32)> = None;
    for &n in &[bearing.len(), 50usize, 100, 200, 400, 700, 1000, usize::MAX] {
        let n = n.min(chunks.len());
        if n < bearing.len() { continue; }
        // Same answers every time; only the distractor count grows.
        let mut idx: Vec<usize> = bearing.clone();
        idx.extend(filler.iter().take(n - bearing.len()));
        if idx.len() < n { continue; }

        let (mut ok, mut msum, mut t1sum) = (0usize, 0.0f32, 0.0f32);
        let mut chosen: BTreeSet<usize> = BTreeSet::new();
        for (k, (_, acc)) in qs.iter().enumerate() {
            let mut best = (usize::MAX, f32::MIN);
            let mut second = f32::MIN;
            for &c in &idx {
                let s = cosine(&qv[k], &cv[c]);
                if s > best.1 { second = best.1; best = (c, s); } else if s > second { second = s; }
            }
            chosen.insert(best.0);
            msum += best.1 - second;
            t1sum += best.1;
            if acc.iter().any(|x| hit(&chunks[best.0], x)) { ok += 1; }
        }
        let (margin, top1) = (msum / qs.len() as f32, t1sum / qs.len() as f32);
        println!("{n:>7}  {:>7}/{:<3}  {margin:>9.4}  {top1:>9.4}  {:>8}", ok, qs.len(), chosen.len());
        // Report the decay PER DOUBLING, which is the number an extrapolation would need. Reporting
        // only the endpoints invites reading a 23% fall over 15x as if it were a 23% fall per step.
        if let Some((pn, pm)) = prev {
            let doublings = (n as f32 / pn as f32).log2();
            if doublings > 0.0 {
                println!("{:>7}  {:>11}  {:>+9.4}  per doubling ({doublings:.2} doublings since {pn})",
                         "", "", (margin - pm) / doublings);
            }
        }
        prev = Some((n, margin));
        if n == chunks.len() { break; }
    }

    println!("\n  retrieval@1 is the pass/fail; the margin is the headroom. A sweep that holds at 100%\n  \
              while the margin decays toward zero is one doubling away from being a coin flip, and\n  \
              only the second column can say which of those is happening.");
    println!("  NOTE: every size above shares ONE embedding pass and ONE set of answer-bearing\n  \
              chunks, so the sweep measures crowding alone. It does NOT re-measure extraction —\n  \
              whether the generator uses the passage it is handed is lookup_vs_weights' question.");
}
