//! Ingest a rotating-machinery fault corpus: 45 recordings, three torques by fifteen conditions.
//!
//!   cargo run -p ferric-signal --example rotating --release -- --data <dir of .mat> \
//!       [--windows N] [--window SAMPLES] [--patch N] [--split time|torque] [--tok-steps N]
//!
//! Vibration half of a public rotating-machinery dataset: four accelerometers at 25.6 kHz, one
//! recording per (torque, fault, severity). Not redistributed here; point `--data` at your own
//! copy, extracted from the archive.
//!
//! **The vibration half is the one with a usable design.** The same dataset's acoustic archive
//! holds five recordings — three fault classes, one torque — which cannot support a split that
//! holds out whole recordings, because two of those classes have exactly one. Only a
//! within-recording split would run there, and it would report a flattering number for a question
//! the data cannot answer. The current-and-temperature half is 45 recordings in TDMS, a format
//! this crate does not read.
//!
//! ## Why this corpus after the hydraulic one
//!
//! Its label space is the one the hydraulic corpus does not have: **perfectly balanced on torque**
//! — fifteen recordings at each of 0, 2 and 4 Nm — with five fault types and a physical severity
//! inside each. That makes three questions askable of the same tokens, and makes one of them
//! askable in a way the hydraulic corpus cannot support: whether a fault type survives a change of
//! operating point it was never trained at.
//!
//! ## Two splits, asking two different questions
//!
//! - `--split time` (default) holds out the LAST THIRD of every recording. Train and test windows
//!   come from the same recording, so anything recording-specific — sensor mounting, that day's
//!   noise floor — is available to the model. This is the protocol most of this literature uses
//!   and it flatters it.
//! - `--split torque` holds out every 4 Nm recording. Nothing about a held-out recording was seen.
//!   Every fault type and severity still appears on both sides, so the fault question is intact;
//!   the torque question is NOT, because training contains no 4 Nm at all, and it is reported as
//!   unanswerable rather than scored.
//!
//! **The gap between the two is the result**, more than either number alone.
//!
//! ## Two traps in this corpus, both silent
//!
//! **The 2 Nm unbalance files are spelled `Unbalalnce`.** A filename parser that trusts the corpus
//! gets SIX fault classes, one of which is all-2 Nm, and then "fault type" and "torque" are
//! partially the same question. Normalised here, and the normalisation is counted and printed.
//!
//! **Recordings are not the same length**: 60 s for the bearing faults, 120 s for misalignment and
//! unbalance, 300 s for normal. Taking every non-overlapping window would hand `Normal` five times
//! the examples of `BPFI` and make the class balance an artifact of recording length. A FIXED
//! number of windows per recording, spread across the whole of it, keeps the design balanced.

use ferric_signal::{
    build_words, chance, compact, majority, nb_probe, permutation_control, train_captions,
    EncoderConfig, EncoderWeights, Fsq, HybridVocab, MatFile, Patcher, RevIn, Sequencer,
};
use ferric_tensor::Tensor;
use std::collections::{BTreeMap, HashSet};
use std::sync::Arc;

const AXES: &[&str] = &["fault", "torque", "severity"];
const FAULTS: &[&str] = &["BPFI", "BPFO", "Misalign", "Normal", "Unbalance"];

fn flag(args: &[String], name: &str) -> Option<String> {
    args.iter().position(|a| a == name).and_then(|i| args.get(i + 1)).cloned()
}

/// What a filename says. `0Nm_BPFI_03` and `2Nm_Unbalalnce_1169mg` both parse here.
#[derive(Debug, Clone)]
struct Condition {
    torque: i32,
    fault: usize,
    /// Raw severity token, kept as written so the rank below can be checked against it.
    severity: String,
    misspelled: bool,
}

fn parse_name(stem: &str) -> Option<Condition> {
    let mut parts = stem.split('_');
    let torque: i32 = parts.next()?.trim_end_matches("Nm").parse().ok()?;
    let raw = parts.next()?;
    // The corpus spells the 2 Nm unbalance files `Unbalalnce`. Left alone it becomes a sixth fault
    // class containing exactly one torque.
    let misspelled = raw == "Unbalalnce";
    let name = if misspelled { "Unbalance" } else { raw };
    let fault = FAULTS.iter().position(|f| *f == name)?;
    let severity = parts.next().unwrap_or("none").to_string();
    Some(Condition { torque, fault, severity, misspelled })
}

/// Severity as a RANK WITHIN ITS FAULT TYPE, because the units are not comparable across them:
/// bearing and misalignment severities are millimetres, unbalance is milligrams, and `Normal` has
/// none. Comparing 03 mm against 0583 mg as numbers would be arithmetic on unlike quantities.
fn severity_ranks(conds: &[Condition]) -> Vec<i32> {
    let mut per_fault: Vec<Vec<String>> = vec![Vec::new(); FAULTS.len()];
    for c in conds {
        if !per_fault[c.fault].contains(&c.severity) {
            per_fault[c.fault].push(c.severity.clone());
        }
    }
    for v in per_fault.iter_mut() {
        // Numeric order where the token is numeric, lexical otherwise, so 03 < 10 < 30 rather than
        // the string order that puts 10 before 3.
        v.sort_by_key(|s| {
            let digits: String = s.chars().take_while(|c| c.is_ascii_digit()).collect();
            (digits.parse::<i64>().unwrap_or(i64::MAX), s.clone())
        });
    }
    conds
        .iter()
        .map(|c| per_fault[c.fault].iter().position(|s| *s == c.severity).unwrap() as i32)
        .collect()
}

/// The four accelerometer channels of one recording: the longest series in the file, by name.
fn recording(m: &MatFile) -> Vec<Vec<f64>> {
    let ch = m.channels();
    let longest = ch.values().map(|s| s.len()).max().unwrap_or(0);
    let mut named: Vec<(&String, &&[f64])> =
        ch.iter().filter(|(_, s)| s.len() == longest).collect();
    named.sort_by(|a, b| a.0.cmp(b.0));
    named.into_iter().map(|(_, s)| s.to_vec()).collect()
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let Some(dir) = flag(&args, "--data") else {
        eprintln!("usage: --data <dir of .mat> [--windows N] [--window SAMPLES] [--split time|torque]");
        std::process::exit(2);
    };
    let per_file: usize = flag(&args, "--windows").and_then(|v| v.parse().ok()).unwrap_or(20);
    let window: usize = flag(&args, "--window").and_then(|v| v.parse().ok()).unwrap_or(25_600);
    let patch: usize = flag(&args, "--patch").and_then(|v| v.parse().ok()).unwrap_or(256);
    // `torque` holds out an operating point; `part` holds out the physical condition itself.
    // See the CWRU example for why the second one had to exist: a split that changes only the
    // operating point leaves the same defective component on both sides of it.
    let split = flag(&args, "--split").unwrap_or_else(|| "time".to_string());
    if !matches!(split.as_str(), "time" | "torque" | "part") {
        eprintln!("error: --split must be one of time, torque, part");
        std::process::exit(2);
    }

    let mut files: Vec<std::path::PathBuf> = match std::fs::read_dir(&dir) {
        Ok(d) => d
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().is_some_and(|x| x == "mat"))
            .collect(),
        Err(e) => {
            eprintln!("error: {dir}: {e}");
            std::process::exit(1);
        }
    };
    files.sort();
    if files.is_empty() {
        eprintln!("error: no .mat files in {dir}");
        std::process::exit(1);
    }

    // ---- label space, from the filenames ----
    let mut conds = Vec::new();
    let mut kept = Vec::new();
    for p in &files {
        let stem = p.file_stem().unwrap().to_string_lossy().to_string();
        match parse_name(&stem) {
            Some(c) => {
                conds.push(c);
                kept.push(p.clone());
            }
            None => eprintln!("  skipping unparseable name: {stem}"),
        }
    }
    let misspelled = conds.iter().filter(|c| c.misspelled).count();
    let ranks = severity_ranks(&conds);

    println!("\nROTATING MACHINERY  {} recordings", kept.len());
    if misspelled > 0 {
        println!("  {misspelled} filenames spell `Unbalalnce`; normalised to `Unbalance`");
        println!("  (left alone that is a SIXTH fault class containing exactly one torque)");
    }
    let mut design: BTreeMap<(i32, usize), usize> = BTreeMap::new();
    for c in &conds {
        *design.entry((c.torque, c.fault)).or_insert(0) += 1;
    }
    print!("  {:<10}", "torque");
    for f in FAULTS {
        print!(" {f:>10}");
    }
    println!();
    let mut torques: Vec<i32> = conds.iter().map(|c| c.torque).collect();
    torques.sort_unstable();
    torques.dedup();
    for t in &torques {
        print!("  {:<10}", format!("{t} Nm"));
        for f in 0..FAULTS.len() {
            print!(" {:>10}", design.get(&(*t, f)).copied().unwrap_or(0));
        }
        println!();
    }

    let Ok(ctx) = pollster::block_on(ferric_core::Context::new()) else {
        eprintln!("no GPU context; this example needs one");
        std::process::exit(1);
    };
    let ctx = Arc::new(ctx);
    let q = Fsq::new(vec![8u32; 5]).unwrap();
    let cfg = EncoderConfig { patch_len: patch, d_model: 32, n_layers: 2, n_heads: 4, d_ff: 64, latent_dim: 5 };
    let enc = EncoderWeights::deterministic(&ctx, cfg, 7).unwrap();
    let patcher = Patcher::contiguous(patch).unwrap();

    println!("\n  {per_file} windows per recording, {window} samples each ({:.2} s at 25.6 kHz),",
             window as f64 / 25_600.0);
    println!("  {patch}-sample patches, {} patches per channel, codebook {} codes",
             window / patch, q.codebook_size());
    println!("  windows are SPREAD across each recording, not taken from its start\n");

    // ---- tokenize ----
    let mut docs: Vec<Vec<Vec<u32>>> = Vec::new();
    let mut doc_file: Vec<usize> = Vec::new();
    let mut doc_pos: Vec<usize> = Vec::new();
    let mut codes_seen: HashSet<u32> = HashSet::new();
    let mut n_chan = 0usize;

    for (fi, p) in kept.iter().enumerate() {
        let bytes = match std::fs::read(p) {
            Ok(b) => b,
            Err(e) => { eprintln!("error: {}: {e}", p.display()); std::process::exit(1); }
        };
        let m = match MatFile::parse(&bytes) {
            Ok(m) => m,
            Err(e) => { eprintln!("error: {}: {e}", p.display()); std::process::exit(1); }
        };
        let chans = recording(&m);
        if chans.is_empty() {
            eprintln!("error: {}: no recording channels", p.display());
            std::process::exit(1);
        }
        // A CONSTANT number of runs per document is not cosmetic: the probe's features are
        // (run index, token), so a file with a different channel count would put its channel 3
        // into the same feature space as another file's channel 3 only by luck.
        if n_chan == 0 {
            n_chan = chans.len();
        } else if chans.len() != n_chan {
            eprintln!("error: {} has {} channels, expected {n_chan}", p.display(), chans.len());
            std::process::exit(1);
        }
        let len = chans[0].len();
        if len < window {
            eprintln!("error: {} is shorter than one window", p.display());
            std::process::exit(1);
        }
        let stride = if per_file > 1 { (len - window) / (per_file - 1) } else { 0 };
        for w in 0..per_file {
            let start = w * stride;
            let mut runs = Vec::with_capacity(n_chan);
            for c in &chans {
                let raw: Vec<f32> = c[start..start + window].iter().map(|&v| v as f32).collect();
                let rev = RevIn::fit(&raw, 1).unwrap();
                let px = patcher.patchify(&rev.apply(&raw).unwrap()).unwrap();
                let t = px.len() / patch;
                let lat = pollster::block_on(
                    enc.forward(&ctx, &Tensor::from_vec(&ctx, &px, &[t, patch])).unwrap().to_vec(),
                );
                let codes: Vec<u32> = (0..t)
                    .map(|i| {
                        q.to_index(&q.quantize(&lat[i * cfg.latent_dim..(i + 1) * cfg.latent_dim]).unwrap())
                            .unwrap()
                    })
                    .collect();
                codes_seen.extend(&codes);
                runs.push(codes);
            }
            docs.push(runs);
            doc_file.push(fi);
            doc_pos.push(w);
        }
        if fi % 9 == 0 {
            println!("    tokenized {}/{} recordings", fi + 1, kept.len());
        }
    }

    println!("\n  {} windows, {} channels each, {} of {} codes visited ({:.1}%)",
             docs.len(), n_chan, codes_seen.len(), q.codebook_size(),
             codes_seen.len() as f64 / q.codebook_size() as f64 * 100.0);

    // ---- split ----
    let (train, held): (Vec<usize>, Vec<usize>) = match split.as_str() {
        "torque" => (0..docs.len()).partition(|&i| conds[doc_file[i]].torque != 4),
        // Rank 1 of every fault: BPFI_10, BPFO_10, Misalign_03, Unbalance_1169. The seeded
        // bearings behind BPFI and BPFO are then physically absent from training, and every torque
        // appears on both sides, so nothing about the operating point is being asked.
        "part" => (0..docs.len()).partition(|&i| ranks[doc_file[i]] != 1),
        _ => {
            // The last third of every recording, by window position within that recording.
            let cut = per_file * 2 / 3;
            (0..docs.len()).partition(|&i| doc_pos[i] < cut)
        }
    };
    println!("  split `{split}`: {} train / {} held out", train.len(), held.len());
    match split.as_str() {
        "torque" => {
            println!("  held-out recordings were never seen; training contains no 4 Nm at all");
            println!("  ⚠ the SAME PHYSICAL CONDITION is in both halves at the other two torques");
        }
        "part" => {
            println!("  held out is severity rank 1 of every fault, at all three torques");
            println!("  the BPFI and BPFO bearings held out are absent from training entirely");
            println!("  ⚠ misalignment and unbalance change a SETTING, not a part: for those two");
            println!("    classes this is an unseen severity rather than an unseen component");
        }
        _ => println!("  train and held-out windows come from the SAME recordings"),
    }

    // ---- the instruments ----
    let per_axis: Vec<Vec<i32>> = vec![
        docs.iter().map(|_| 0).enumerate().map(|(i, _)| conds[doc_file[i]].fault as i32).collect(),
        (0..docs.len()).map(|i| conds[doc_file[i]].torque).collect(),
        (0..docs.len()).map(|i| ranks[doc_file[i]]).collect(),
    ];

    println!("\n  TOKEN PROBE  naive Bayes over (channel, code) counts. No language model, no");
    println!("  capacity to speak — just whether the token stream separates the classes.\n");
    println!("  {:<10} {:>8} {:>10} {:>10} {:>12}", "axis", "classes", "chance", "majority", "probe");
    println!("  {:-<10} {:->8} {:->10} {:->10} {:->12}", "", "", "", "", "");
    let mut controls = Vec::new();
    for (a, name) in AXES.iter().enumerate() {
        let labels = &per_axis[a];
        let n_cls = labels.iter().collect::<HashSet<_>>().len();
        let train_cls: HashSet<i32> = train.iter().map(|&i| labels[i]).collect();
        let held_cls: HashSet<i32> = held.iter().map(|&i| labels[i]).collect();
        // An axis whose held-out classes are absent from training is not a hard question, it is an
        // unanswerable one, and scoring it would produce a number that looks like a failure.
        if !held_cls.is_subset(&train_cls) {
            println!("  {name:<10} {n_cls:>8} {:>10} {:>10} {:>12}", "-", "-", "not askable");
            controls.push(f64::NAN);
            continue;
        }
        let maj = majority(labels, &train, &held);
        let acc = nb_probe(&docs, labels, &train, &held);
        let ctl = permutation_control(&docs, labels, &train, &held, 20, 0xC0FF_EE);
        controls.push(ctl);
        // ⛔ THE STAR ONCE FIRED ON A BELOW-CHANCE NUMBER. `majority` is the accuracy of
        // predicting the training set's most common class, and when the training classes tie, the
        // class it picks can be one that does not occur in the held-out set at all — a 0.0%
        // baseline that anything clears. The 48 kHz part split hit exactly that and printed
        // `4.2%*` for a five-class problem whose chance rate is 16.7%. A result has to clear BOTH.
        let bar = maj.max(chance(labels));
        let mark = if acc > bar + 1e-9 { "*" } else { " " };
        println!("  {name:<10} {n_cls:>8} {:>9.1}% {maj:>9.1}% {acc:>11.1}%{mark}", chance(labels));
    }
    println!("\n  {:<10} {:>12}", "axis", "control");
    println!("  {:-<10} {:->12}", "", "");
    for (a, name) in AXES.iter().enumerate() {
        if controls[a].is_nan() {
            println!("  {name:<10} {:>12}", "-");
        } else {
            println!("  {name:<10} {:>+11.1}pt", controls[a]);
        }
    }
    println!("\n  `control` is the same probe on the same tokens with the window-to-label assignment");
    println!("  permuted, worst of twenty, each against its own majority. Read it before the probe:");
    println!("  an effect the size of its control is not an effect.");

    // Per-class recall on the held-out set. AN AGGREGATE ABOVE MAJORITY CAN BE ONE EFFECT OR
    // SEVERAL, and the aggregate cannot tell you which: a five-class figure well clear of majority
    // is equally consistent with separating every class and with separating two coarse ones while
    // guessing the rest. Those are different claims about what the tokens carry.
    if let Some((cls, m)) = ferric_signal::nb_confusion(&docs, &per_axis[0], &train, &held) {
        println!("\n  FAULT, CLASS BY CLASS  rows true, columns predicted, held-out windows\n");
        print!("  {:<10}", "");
        for c in &cls {
            print!(" {:>8}", FAULTS[*c as usize]);
        }
        println!("  {:>8}", "recall");
        for (t, row) in m.iter().enumerate() {
            let n: usize = row.iter().sum();
            if n == 0 {
                continue;
            }
            print!("  {:<10}", FAULTS[cls[t] as usize]);
            for v in row {
                print!(" {v:>8}");
            }
            println!("  {:>7.1}%", row[t] as f64 / n as f64 * 100.0);
        }
    }

    println!("\n  The tokenizer is UNTRAINED. RevIn normalises every channel of every window, so");
    println!("  absolute amplitude is gone by construction and what is read is shape.");

    if !args.iter().any(|a| a == "--train") {
        println!("\n  Pass --train to run signal-to-text on these conditions.\n");
        return;
    }

    // ---- signal to text: say the condition in words ----
    //
    // Every axis stays in the caption under BOTH splits, so the task is the same one in each and
    // the two columns stay comparable. An axis whose held-out classes are absent from training is
    // reported as unanswerable rather than scored, exactly as in the probe table above — the model
    // is still trained on it and still spends capacity there, which is the honest cost of asking.
    let rows: Vec<Vec<i32>> =
        (0..docs.len()).map(|i| (0..AXES.len()).map(|a| per_axis[a][i]).collect()).collect();
    let (words, caps) = build_words(AXES, &rows);
    let (remapped, size, unk_pct) = compact(&docs, &train, &held);
    let fq = Fsq::new(vec![size]).unwrap();
    let seq = Sequencer::new(HybridVocab::new(words.len() as u32, fq).unwrap());
    let steps: usize = flag(&args, "--steps").and_then(|v| v.parse().ok()).unwrap_or(250);
    let batch: usize = flag(&args, "--batch").and_then(|v| v.parse().ok()).unwrap_or(8);
    let seeds: usize = flag(&args, "--seeds").and_then(|v| v.parse().ok()).unwrap_or(3);
    let pool = args.iter().any(|a| a == "--pool");
    let lm_dim: usize = flag(&args, "--lm-dim").and_then(|v| v.parse().ok()).unwrap_or(64);
    let lm_layers: usize = flag(&args, "--lm-layers").and_then(|v| v.parse().ok()).unwrap_or(2);
    let lm_cfg = EncoderConfig {
        patch_len: 16, d_model: lm_dim, n_layers: lm_layers, n_heads: 4, d_ff: lm_dim * 2, latent_dim: 5,
    };

    println!("\n  SIGNAL -> TEXT  {} train / {} held out, {} caption words", train.len(), held.len(), words.len());
    println!("  vocabulary compacted to {size} signal rows from {}; {unk_pct:.2}% of held-out tokens",
             q.codebook_size());
    println!("  were unseen in training. {} embedding rows.", seq.embedding_rows());
    println!("  {steps} optimizer steps x batch {batch} = {} examples seen, {seeds} seeds, shuffled",
             steps * batch);
    // THE PREDICTION THIS FLAG EXISTS TO TEST. The probe pools a document into a count over every
    // observed (channel, code) pair — thousands of features. The decoder pools the same document
    // into `d_model` numbers. At 64 against a 6,600-code vocabulary that is a narrower summary by
    // two orders of magnitude, which is a candidate explanation for the axis the probe reads and
    // the decoder cannot: severity, at 51.1% against 33.7%, WITHIN recording.
    println!("  decoder: d_model {lm_dim}, {lm_layers} layers, {} signal rows to pool into {lm_dim} numbers\n",
             size - 1);

    let mut runs = Vec::new();
    for s in 0..seeds {
        let r = train_captions(
            &ctx, &seq, &remapped, &caps, &train, &held, steps, batch, lm_cfg, false, pool,
            AXES.len(), 3 + s as u64 * 17,
        )
        .unwrap_or_else(|e| { eprintln!("error: {e}"); std::process::exit(1) });
        println!("    seed {}: {}", 3 + s * 17,
                 r.acc.iter().enumerate().map(|(a, v)| format!("{}={v:.0}%", AXES[a]))
                     .collect::<Vec<_>>().join("  "));
        runs.push(r);
    }

    println!("\n  {:<10} {:>8} {:>7} {:>10} {:>8} {:>10} {:>8}",
             "axis", "mean", "sd", "majority", "chance", "said", "off-axis");
    println!("  {:-<10} {:->8} {:->7} {:->10} {:->8} {:->10} {:->8}", "", "", "", "", "", "", "");
    for (a, name) in AXES.iter().enumerate() {
        let labels = &per_axis[a];
        let train_cls: HashSet<i32> = train.iter().map(|&i| labels[i]).collect();
        let held_cls: HashSet<i32> = held.iter().map(|&i| labels[i]).collect();
        let v: Vec<f64> = runs.iter().map(|r| r.acc[a]).collect();
        let m = v.iter().sum::<f64>() / v.len() as f64;
        let sd = (v.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / v.len() as f64).sqrt();
        let said = runs.iter().map(|r| r.distinct[a] as f64).sum::<f64>() / runs.len() as f64;
        let off = runs.iter().map(|r| r.off_axis[a]).sum::<f64>() / runs.len() as f64;
        let n_cls = labels.iter().collect::<HashSet<_>>().len();
        if !held_cls.is_subset(&train_cls) {
            println!("  {name:<10} {m:>7.1}% {sd:>6.1} {:>10} {:>8} {said:>7.1} of {n_cls} {off:>7.0}%  <- not askable under this split",
                     "-", "-");
            continue;
        }
        let maj = majority(labels, &train, &held);
        let verdict = if said <= 1.0 {
            "  <- one word for every window"
        } else if m > maj + 1e-9 {
            ""
        } else {
            "  <- at or below majority"
        };
        println!("  {name:<10} {m:>7.1}% {sd:>6.1} {maj:>9.1}% {:>7.1}% {said:>7.1} of {n_cls} {off:>7.0}%{verdict}",
                 chance(labels));
    }
    println!("\n  `said` counts the distinct words the decoder actually emitted at that position");
    println!("  across the held-out set. 1.0 means it answered the same thing every time, which");
    println!("  scores that word's frequency and reads as a weak learner in an accuracy column.");
    println!("  `off-axis` is the share of predictions that were not a word for that axis at all —");
    println!("  a signal code, an end marker, another axis's word. An accuracy of exactly 0.0% is");
    println!("  its signature: a decoder settled on the WRONG caption word still scores that");
    println!("  word's frequency, so nothing at all means the argmax left the caption vocabulary.\n");
}
