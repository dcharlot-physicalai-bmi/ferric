//! Ingest the CWRU bearing corpus, whose labels live in the HTML pages beside the data.
//!
//!   cargo run -p ferric-signal --example cwru --release -- --data <dir of .mat> --pages <dir of .html>
//!       [--windows N] [--split time|load|bearing] [--tokenizer <file>] [--train]
//!
//! Not redistributed here. Point `--data` at your own copy of the `.mat` files and `--pages` at the
//! four Case Western pages they came from; this review located a licence for neither, and widely
//! mirrored is not the same as licensed.
//!
//! ## Why this corpus, after the other two
//!
//! Three registered predictions about the decoder were refuted in turn — width collapses it, budget
//! overfits it, and handing it a pooled summary moves nothing — and they converge on the corpus
//! rather than the design: a counting probe carries a 6,534-dimensional histogram that needs no
//! training, and no learned pooling of that width can be acquired from 360 examples.
//!
//! **The binding constraint is labelled RECORDINGS, not windows.** Windows from one recording are
//! not independent of each other; recordings are. The rotating corpus has 45. This one has 161,
//! and 138 carry both accelerometers — about three times as many, which is the only variable the
//! refutations left worth changing.
//!
//! ## The labels are in the pages, and they are complete
//!
//! Each table cell names a condition (`IR007_2`, `OR014@6_0`, `Normal_3`) and links to the file
//! number that holds it. Parsed from the pages rather than transcribed, so nothing is redistributed
//! and a mistranscription cannot happen. **All 161 files on disk resolve to a label, with no
//! label left over** — checked rather than assumed, and printed every run.
//!
//! Three axes come out of it: fault type (inner race, ball, outer race at three clock positions,
//! and normal), fault diameter, and motor load. **Load is balanced 41/40/40/40**, which makes it the
//! across-operating-point split — the same question the rotating corpus answers with 45 recordings.
//!
//! ## ⛔ The load split is not the generalization test it looks like
//!
//! `--split load` holds out every 3 HP recording, and that reads as an honest across-operating-point
//! test. Two published critiques say it is not, and they are right.
//!
//! Hendriks, Dumond & Knox (*Mechanical Systems and Signal Processing* **169**, 2022, "Towards
//! better benchmarking using the CWRU bearing fault dataset") identify the flaw directly: "the
//! accepted procedure of constructing training and testing datasets with different operating
//! conditions does not constitute a useful domain shift problem since the same physical bearings
//! exist in both training and testing sets". They find the usual framework "allows CNNs to learn
//! features related to specific bearings" and propose independent sets of bearings instead. Vieira,
//! Bauler, Rosa & Silva (arXiv:2509.22267) arrive from the leakage side: "segment-wise and
//! condition-wise splits introduce spurious correlations that inflate performance metrics", and
//! recommend bearing-wise partitioning.
//!
//! This example was already free of the segment-wise form — whole recordings are held out, never
//! windows from a recording that also trains — but a load split is the condition-wise form exactly.
//! Each seeded bearing was run at all four loads, so the part that produced the held-out 3 HP
//! recordings produced training recordings too.
//!
//! **The corpus contains its own remedy, and this example was throwing it away.** The condition
//! code `IR007` names a drive-end seeded bearing on one page and a *fan-end* seeded bearing on
//! another: two parts, two mountings, pooled into one label because only the fault name was kept.
//! At 12 kHz that is 60 drive-end recordings and 45 fan-end, and **both ends carry all five fault
//! classes**, so training on one end and testing on the other holds out the physical bearing and
//! its mounting location while leaving the axis askable. That is `--split bearing`, and the
//! difference between it and `--split load` is the size of the effect the critiques predict.
//!
//! ## Both accelerometers, or the recording is skipped
//!
//! Files vary in what they carry: 91 have drive-end, fan-end and base, 43 have drive-end and
//! fan-end, 8 have drive-end alone. The probe's features are (channel, code) pairs, so a document
//! with a different number of runs puts its channel 1 into another document's channel-1 feature
//! space by luck rather than by meaning. Recordings without both are skipped and counted.

use ferric_signal::{
    build_words, chance, compact, majority, nb_probe, permutation_control, train_captions,
    EncoderConfig, EncoderWeights, Fsq, HybridVocab, MatFile, Patcher, RevIn, Sequencer,
};
use ferric_tensor::Tensor;
use std::collections::{BTreeMap, HashSet};
use std::sync::Arc;

const AXES: &[&str] = &["fault", "diameter", "load"];
const FAULTS: &[&str] = &["Normal", "IR", "B", "OR@6", "OR@3", "OR@12"];

fn flag(args: &[String], name: &str) -> Option<String> {
    args.iter().position(|a| a == name).and_then(|i| args.get(i + 1)).cloned()
}

/// One recording's labels, decoded from the condition code the page gives it.
#[derive(Debug, Clone)]
struct Condition {
    /// Sampling rate of the source page, in kHz. **Not a nuisance field.**
    rate: u32,
    /// Which bearing carries the seeded defect: 0 drive end, 1 fan end, 2 neither (healthy).
    ///
    /// **This identifies the physical bearing, and until now it was thrown away.** `IR007` names a
    /// drive-end bearing on one page and a fan-end bearing on another: two seeded parts, in two
    /// mountings, pooled into one label. Keeping it is what makes a bearing-wise split possible.
    end: usize,
    fault: usize,
    /// Thousandths of an inch; 0 for a healthy bearing.
    diameter: i32,
    /// Motor load in horsepower.
    load: i32,
}

const ENDS: &[&str] = &["drive end", "fan end", "healthy"];

/// `IR007_2`, `OR014@6_0`, `B021_1`, `Normal_3`.
fn decode(code: &str, rate: u32, end: usize) -> Option<Condition> {
    let (head, load) = code.rsplit_once('_')?;
    let load: i32 = load.parse().ok()?;
    if head == "Normal" {
        return Some(Condition { rate, end: 2, fault: 0, diameter: 0, load });
    }
    // Outer-race codes carry the defect's clock position relative to the load zone, which is a
    // different condition rather than a detail: the same defect presents differently at 3, 6 and
    // 12 o'clock, and merging them would put three conditions in one class.
    let (head, pos) = match head.split_once('@') {
        Some((h, p)) => (h, Some(p)),
        None => (head, None),
    };
    let digits = head.trim_start_matches(|c: char| c.is_ascii_alphabetic());
    let kind = &head[..head.len() - digits.len()];
    let diameter: i32 = digits.parse().ok()?;
    let name = match (kind, pos) {
        ("IR", _) => "IR",
        ("B", _) => "B",
        ("OR", Some("6")) => "OR@6",
        ("OR", Some("3")) => "OR@3",
        ("OR", Some("12")) => "OR@12",
        _ => return None,
    };
    Some(Condition { rate, end, fault: FAULTS.iter().position(|f| *f == name)?, diameter, load })
}

/// Pull `(file number, condition code)` out of the corpus's own pages.
///
/// A cell is scanned for a `files/NNN.mat` link and its text taken as the condition. Tags are
/// stripped rather than parsed: the structure being relied on is one link and one code per cell,
/// which is far more stable than any particular markup.
fn labels_from_pages(dir: &str) -> BTreeMap<u32, (String, String)> {
    let mut out: BTreeMap<u32, (String, String)> = BTreeMap::new();
    let Ok(rd) = std::fs::read_dir(dir) else { return out };
    let mut files: Vec<_> = rd
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "html"))
        .collect();
    files.sort();
    for p in files {
        let Ok(html) = std::fs::read_to_string(&p) else { continue };
        for cell in html.split("<td").skip(1) {
            let cell = cell.split("</td>").next().unwrap_or("");
            // Skip the rest of the opening tag. Splitting on `<td` leaves its own attributes at the
            // head of the slice, and the tag-stripper below starts outside a tag, so it read
            // `bgcolor="#FFFFFF"IR007_0` as the condition and nothing decoded.
            let cell = &cell[cell.find('>').map(|i| i + 1).unwrap_or(0)..];
            let Some(i) = cell.find("files/") else { continue };
            let num: String =
                cell[i + 6..].chars().take_while(|c| c.is_ascii_digit()).collect();
            if num.is_empty() || !cell[i..].contains(".mat") {
                continue;
            }
            // Strip tags to get the visible text, which is the condition code.
            let mut text = String::new();
            let mut depth = 0;
            for c in cell.chars() {
                match c {
                    '<' => depth += 1,
                    '>' => depth = (depth - 1i32).max(0),
                    _ if depth == 0 => text.push(c),
                    _ => {}
                }
            }
            let code = text.replace('\u{a0}', " ").trim().to_string();
            if let Ok(n) = num.parse::<u32>() {
                let page = p.file_name().unwrap().to_string_lossy().to_string();
                out.entry(n).or_insert((code, page));
            }
        }
    }
    out
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let (Some(data), Some(pages)) = (flag(&args, "--data"), flag(&args, "--pages")) else {
        eprintln!("usage: --data <dir of .mat> --pages <dir of .html> [--windows N] [--split time|load|bearing]");
        std::process::exit(2);
    };
    let per_file: usize = flag(&args, "--windows").and_then(|v| v.parse().ok()).unwrap_or(12);
    let window: usize = flag(&args, "--window").and_then(|v| v.parse().ok()).unwrap_or(4096);
    let patch: usize = flag(&args, "--patch").and_then(|v| v.parse().ok()).unwrap_or(256);
    // `time` splits windows inside a recording, `load` holds out an operating point, `bearing`
    // holds out the physical part. See the module comment for why the third one had to exist.
    let split = flag(&args, "--split").unwrap_or_else(|| "time".to_string());
    if !matches!(split.as_str(), "time" | "load" | "bearing") {
        eprintln!("error: --split must be one of time, load, bearing");
        std::process::exit(2);
    }

    let rate_filter: Option<u32> = match flag(&args, "--rate").as_deref() {
        Some("all") => None,
        Some("48k") => Some(48),
        _ => Some(12),
    };
    let codes = labels_from_pages(&pages);
    println!("\nCWRU BEARINGS  {} labelled recordings in the pages", codes.len());

    // ---- match labels to the files actually present ----
    let mut records: Vec<(std::path::PathBuf, Condition)> = Vec::new();
    let mut undecoded = Vec::new();
    let mut unlabelled = Vec::new();
    let Ok(rd) = std::fs::read_dir(&data) else {
        eprintln!("error: cannot read {data}");
        std::process::exit(1);
    };
    let mut mats: Vec<_> = rd
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "mat"))
        .collect();
    mats.sort();
    for p in &mats {
        let stem = p.file_stem().unwrap().to_string_lossy().to_string();
        let Ok(n) = stem.parse::<u32>() else { continue };
        match codes.get(&n) {
            Some((code, page)) => {
                let rate = if page.contains("48k") || page.contains("normal") { 48 } else { 12 };
                let end = if page.contains("fan-end") { 1 } else { 0 };
                match decode(code, rate, end) {
                    Some(c) => records.push((p.clone(), c)),
                    None => undecoded.push((n, code.clone())),
                }
            }
            None => unlabelled.push(n),
        }
    }
    println!("  {} files on disk, {} matched to a label", mats.len(), records.len());

    // ⛔ SAMPLE RATE IS CONFOUNDED WITH THE LABEL IN THIS CORPUS, and a fixed-sample window turns
    // that into a leak. 105 recordings are at 12 kHz and 56 at 48 kHz, and **every healthy
    // recording is at 48 kHz** — zero at 12. A window of a fixed number of SAMPLES therefore spans
    // 2.1 s or 0.5 s depending on the file, so a model can separate healthy from faulty by
    // detecting the rate and never look at a bearing. The other classes split roughly 70/30 across
    // rates, so they leak partially too.
    //
    // The default is therefore ONE RATE, and what that costs is printed rather than hidden: at
    // 12 kHz the healthy class does not exist at all, so the fault axis is five classes of defect
    // with no negative case. `--rate all` restores the contaminated corpus for anyone who wants to
    // see the difference.
    let mut by_rate: BTreeMap<u32, usize> = BTreeMap::new();
    for (_, c) in &records {
        *by_rate.entry(c.rate).or_insert(0) += 1;
    }
    let healthy_by_rate: BTreeMap<u32, usize> =
        records.iter().filter(|(_, c)| c.fault == 0).fold(BTreeMap::new(), |mut m, (_, c)| {
            *m.entry(c.rate).or_insert(0) += 1;
            m
        });
    println!("  by sample rate: {by_rate:?}, of which healthy: {healthy_by_rate:?}");
    if let Some(r) = rate_filter {
        let before = records.len();
        records.retain(|(_, c)| c.rate == r);
        println!("  RATE-CONTROLLED to {r} kHz: {} of {before} recordings kept", records.len());
        if !records.iter().any(|(_, c)| c.fault == 0) {
            println!("  no healthy recording exists at {r} kHz, so `fault` is five kinds of defect");
        }
    } else {
        println!("  --rate all: MIXED RATES, and the healthy class is separable by rate alone");
    }
    // A file with no label, or a label that will not decode, is reported rather than dropped
    // quietly: a corpus silently smaller than it looks is the failure this crate keeps meeting.
    if !unlabelled.is_empty() {
        println!("  {} files have NO label in the pages: {:?}", unlabelled.len(),
                 &unlabelled[..unlabelled.len().min(8)]);
    }
    if !undecoded.is_empty() {
        println!("  {} labels did not decode: {:?}", undecoded.len(),
                 &undecoded[..undecoded.len().min(8)]);
    }

    let mut design: BTreeMap<(usize, i32), usize> = BTreeMap::new();
    for (_, c) in &records {
        *design.entry((c.fault, c.load)).or_insert(0) += 1;
    }
    print!("  {:<10}", "load");
    for f in FAULTS {
        print!(" {f:>7}");
    }
    println!();
    for load in 0..4 {
        print!("  {:<10}", format!("{load} HP"));
        for f in 0..FAULTS.len() {
            print!(" {:>7}", design.get(&(f, load)).copied().unwrap_or(0));
        }
        println!();
    }

    // ⛔ THE PHYSICAL BEARING IS A VARIABLE, AND IT WAS INVISIBLE HERE.
    //
    // Two published critiques land on exactly the split this example used to default to. Hendriks,
    // Dumond & Knox (Mechanical Systems and Signal Processing 169, 2022, "Towards better
    // benchmarking using the CWRU bearing fault dataset") find that "the accepted procedure of
    // constructing training and testing datasets with different operating conditions does not
    // constitute a useful domain shift problem since the same physical bearings exist in both
    // training and testing sets", and propose splitting by independent sets of bearings instead.
    // Vieira, Bauler, Rosa & Silva (arXiv:2509.22267) reach the same place from the leakage side:
    // "segment-wise and condition-wise splits introduce spurious correlations that inflate
    // performance metrics", and recommend bearing-wise partitioning.
    //
    // `--split load` is a condition-wise split. It holds out an operating point, and the bearing
    // that produced the held-out recordings also produced training recordings at the other three
    // loads. So this table, printed since the first version, was hiding the variable that matters.
    //
    // The remedy is in the corpus. `IR007` names a DRIVE-END seeded bearing on one page and a
    // FAN-END seeded bearing on another — that is why every cell below reads as two sets of four
    // loads — and both ends carry all five fault classes. Training on one end and testing on the
    // other holds out the physical part AND its mounting: `--split bearing`.
    let mut ends: BTreeMap<(usize, usize, i32), usize> = BTreeMap::new();
    for (_, c) in &records {
        *ends.entry((c.end, c.fault, c.diameter)).or_insert(0) += 1;
    }
    let diams: Vec<i32> = {
        let mut d: Vec<i32> = ends.keys().map(|k| k.2).collect::<HashSet<_>>().into_iter().collect();
        d.sort();
        d
    };
    for e in 0..ENDS.len() {
        let n: usize = ends.iter().filter(|(k, _)| k.0 == e).map(|(_, v)| v).sum();
        if n == 0 {
            continue;
        }
        println!("\n  {} — {n} recordings, by fault and defect diameter (thousandths of an inch)",
                 ENDS[e]);
        print!("  {:<10}", "");
        for d in &diams {
            print!(" {:>7}", format!("{d:03}\""));
        }
        println!();
        for f in 0..FAULTS.len() {
            let row: usize = diams.iter().map(|d| ends.get(&(e, f, *d)).copied().unwrap_or(0)).sum();
            if row == 0 {
                continue;
            }
            print!("  {:<10}", FAULTS[f]);
            for d in &diams {
                print!(" {:>7}", ends.get(&(e, f, *d)).copied().unwrap_or(0));
            }
            println!();
        }
    }

    let Ok(ctx) = pollster::block_on(ferric_core::Context::new()) else {
        eprintln!("no GPU context; this example needs one");
        std::process::exit(1);
    };
    let ctx = Arc::new(ctx);
    let q = Fsq::new(vec![8u32; 5]).unwrap();
    // DOES THE RELEASED TOKENIZER BEAT AN UNTRAINED ONE ON A DOWNSTREAM TASK?
    //
    // The four-corpus checkpoint is reported by reconstruction SNR, which is what it was trained
    // for. Whether its tokens SEPARATE LABELS better than an untrained projection is a different
    // question and the one a user actually has. `--tokenizer` loads it in place of the untrained
    // encoder; everything else — windows, split, probe, control — is unchanged, so the two arms
    // differ only in the weights.
    let (cfg, enc) = match flag(&args, "--tokenizer") {
        Some(path) => {
            let c = EncoderConfig {
                patch_len: patch, d_model: 256, n_layers: 5, n_heads: 4, d_ff: 896, latent_dim: 5,
            };
            let bytes = std::fs::read(&path)
                .unwrap_or_else(|e| { eprintln!("error: {path}: {e}"); std::process::exit(1) });
            let w = ferric_signal::Weights::from_bytes(&bytes)
                .unwrap_or_else(|e| { eprintln!("error: {path}: {e:?}"); std::process::exit(1) });
            let e = EncoderWeights::from_weights(&ctx, c, &w).unwrap_or_else(|e| {
                eprintln!("error: shapes do not match patch_len {patch}: {e:?}");
                eprintln!("the released checkpoint was trained at patch 128");
                std::process::exit(1)
            });
            println!("\n  TOKENIZER: {path}");
            println!("  digest {} — {} parameters", w.digest(), c.params().total());
            (c, e)
        }
        None => {
            let c = EncoderConfig { patch_len: patch, d_model: 32, n_layers: 2, n_heads: 4, d_ff: 64, latent_dim: 5 };
            println!("\n  TOKENIZER: untrained, {} parameters", c.params().total());
            (c, EncoderWeights::deterministic(&ctx, c, 7).unwrap())
        }
    };
    let patcher = Patcher::contiguous(patch).unwrap();

    println!("\n  {per_file} windows per recording, {window} samples each, {patch}-sample patches");
    println!("  windows are SPREAD across each recording, not taken from its start\n");

    // ---- tokenize: drive end and fan end, or skip the recording ----
    let mut docs: Vec<Vec<Vec<u32>>> = Vec::new();
    let mut doc_rec: Vec<usize> = Vec::new();
    let mut doc_pos: Vec<usize> = Vec::new();
    let mut codes_seen: HashSet<u32> = HashSet::new();
    let mut skipped_channels = 0usize;

    for (ri, (path, _)) in records.iter().enumerate() {
        let Ok(bytes) = std::fs::read(path) else { continue };
        let Ok(m) = MatFile::parse(&bytes) else { continue };
        let ch = m.channels();
        let pick = |suffix: &str| -> Option<Vec<f32>> {
            ch.iter()
                .find(|(n, _)| n.ends_with(suffix))
                .map(|(_, s)| s.iter().map(|&v| v as f32).collect())
        };
        // BOTH accelerometers or nothing. The probe's features are (channel, code) pairs, so a
        // document with fewer runs would land its channel 1 in another document's channel-1
        // feature space by luck rather than by meaning.
        let (Some(de), Some(fe)) = (pick("_DE_time"), pick("_FE_time")) else {
            skipped_channels += 1;
            continue;
        };
        let len = de.len().min(fe.len());
        if len < window {
            skipped_channels += 1;
            continue;
        }
        let stride = if per_file > 1 { (len - window) / (per_file - 1) } else { 0 };
        for w in 0..per_file {
            let start = w * stride;
            let mut runs = Vec::with_capacity(2);
            for c in [&de, &fe] {
                let raw = &c[start..start + window];
                let rev = RevIn::fit(raw, 1).unwrap();
                let px = patcher.patchify(&rev.apply(raw).unwrap()).unwrap();
                let t = px.len() / patch;
                let lat = pollster::block_on(
                    enc.forward(&ctx, &Tensor::from_vec(&ctx, &px, &[t, patch])).unwrap().to_vec(),
                );
                let seq: Vec<u32> = (0..t)
                    .map(|i| {
                        q.to_index(&q.quantize(&lat[i * cfg.latent_dim..(i + 1) * cfg.latent_dim]).unwrap())
                            .unwrap()
                    })
                    .collect();
                codes_seen.extend(&seq);
                runs.push(seq);
            }
            docs.push(runs);
            doc_rec.push(ri);
            doc_pos.push(w);
        }
        if ri % 25 == 0 {
            println!("    tokenized {}/{} recordings", ri + 1, records.len());
        }
    }

    let used: HashSet<usize> = doc_rec.iter().copied().collect();
    println!("\n  {} windows from {} recordings ({skipped_channels} skipped for missing a channel)",
             docs.len(), used.len());
    println!("  {} of {} codes visited ({:.1}%)", codes_seen.len(), q.codebook_size(),
             codes_seen.len() as f64 / q.codebook_size() as f64 * 100.0);

    // ---- split ----
    let (train, held): (Vec<usize>, Vec<usize>) = match split.as_str() {
        "load" => (0..docs.len()).partition(|&i| records[doc_rec[i]].1.load != 3),
        "bearing" => (0..docs.len()).partition(|&i| records[doc_rec[i]].1.end != 1),
        _ => {
            let cut = per_file * 2 / 3;
            (0..docs.len()).partition(|&i| doc_pos[i] < cut)
        }
    };
    println!("  split `{split}`: {} train / {} held out", train.len(), held.len());
    match split.as_str() {
        "load" => {
            println!("  held-out recordings were never seen; training contains no 3 HP at all");
            println!("  ⚠ the SAME PHYSICAL BEARINGS are in both halves — see the module comment");
        }
        "bearing" => {
            println!("  train is every drive-end defect, held out is every fan-end defect");
            println!("  NO physical bearing, and no mounting location, appears in both halves");
        }
        _ => println!("  train and held-out windows come from the SAME recordings"),
    }

    let per_axis: Vec<Vec<i32>> = vec![
        (0..docs.len()).map(|i| records[doc_rec[i]].1.fault as i32).collect(),
        (0..docs.len()).map(|i| records[doc_rec[i]].1.diameter).collect(),
        (0..docs.len()).map(|i| records[doc_rec[i]].1.load).collect(),
    ];

    println!("\n  TOKEN PROBE  naive Bayes over (channel, code) counts.\n");
    println!("  {:<10} {:>8} {:>10} {:>10} {:>12} {:>10}",
             "axis", "classes", "chance", "majority", "probe", "control");
    println!("  {:-<10} {:->8} {:->10} {:->10} {:->12} {:->10}", "", "", "", "", "", "");
    for (a, name) in AXES.iter().enumerate() {
        let labels = &per_axis[a];
        let n_cls = labels.iter().collect::<HashSet<_>>().len();
        let train_cls: HashSet<i32> = train.iter().map(|&i| labels[i]).collect();
        let held_cls: HashSet<i32> = held.iter().map(|&i| labels[i]).collect();
        if !held_cls.is_subset(&train_cls) {
            println!("  {name:<10} {n_cls:>8} {:>10} {:>10} {:>12} {:>10}",
                     "-", "-", "not askable", "-");
            continue;
        }
        let maj = majority(labels, &train, &held);
        let acc = nb_probe(&docs, labels, &train, &held);
        let ctl = permutation_control(&docs, labels, &train, &held, 20, 0xC0FF_EE);
        let mark = if acc > maj + 1e-9 { "*" } else { " " };
        println!("  {name:<10} {n_cls:>8} {:>9.1}% {maj:>9.1}% {acc:>11.1}%{mark} {ctl:>+9.1}pt",
                 chance(labels));
    }
    println!("\n  `control` is the same probe with the window-to-label assignment permuted, worst of");
    println!("  twenty, each against its own majority. Read it before the probe.");

    if !args.iter().any(|a| a == "--train") {
        println!("\n  Pass --train to run signal-to-text on these conditions.\n");
        return;
    }

    // ---- signal to text ----
    let rows: Vec<Vec<i32>> =
        (0..docs.len()).map(|i| (0..AXES.len()).map(|a| per_axis[a][i]).collect()).collect();
    let (words, caps) = build_words(AXES, &rows);
    let (remapped, size, unk) = compact(&docs, &train, &held);
    let fq = Fsq::new(vec![size]).unwrap();
    let seq = Sequencer::new(HybridVocab::new(words.len() as u32, fq).unwrap());
    let steps: usize = flag(&args, "--steps").and_then(|v| v.parse().ok()).unwrap_or(400);
    let batch: usize = flag(&args, "--batch").and_then(|v| v.parse().ok()).unwrap_or(8);
    let seeds: usize = flag(&args, "--seeds").and_then(|v| v.parse().ok()).unwrap_or(3);
    let lm_cfg = EncoderConfig { patch_len: 16, d_model: 64, n_layers: 2, n_heads: 4, d_ff: 128, latent_dim: 5 };

    println!("\n  SIGNAL -> TEXT  {} train / {} held out, {} caption words",
             train.len(), held.len(), words.len());
    println!("  vocabulary compacted to {size} signal rows; {unk:.2}% of held-out tokens unseen");
    println!("  {steps} steps x batch {batch} = {} examples, {seeds} seeds\n", steps * batch);

    let mut runs = Vec::new();
    for s in 0..seeds {
        let r = train_captions(&ctx, &seq, &remapped, &caps, &train, &held, steps, batch, lm_cfg,
                               false, false, AXES.len(), 3 + s as u64 * 17)
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
        let v: Vec<f64> = runs.iter().map(|r| r.acc[a]).collect();
        let m = v.iter().sum::<f64>() / v.len() as f64;
        let sd = (v.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / v.len() as f64).sqrt();
        let said = runs.iter().map(|r| r.distinct[a] as f64).sum::<f64>() / runs.len() as f64;
        let off = runs.iter().map(|r| r.off_axis[a]).sum::<f64>() / runs.len() as f64;
        let n_cls = labels.iter().collect::<HashSet<_>>().len();
        let train_cls: HashSet<i32> = train.iter().map(|&i| labels[i]).collect();
        let held_cls: HashSet<i32> = held.iter().map(|&i| labels[i]).collect();
        if !held_cls.is_subset(&train_cls) {
            println!("  {name:<10} {m:>7.1}% {sd:>6.1} {:>10} {:>8} {said:>7.1} of {n_cls} {off:>7.0}%  <- not askable",
                     "-", "-");
            continue;
        }
        let maj = majority(labels, &train, &held);
        let verdict = if said <= 1.0 { "  <- one word every window" }
            else if m > maj + 1e-9 { "" } else { "  <- at or below majority" };
        println!("  {name:<10} {m:>7.1}% {sd:>6.1} {maj:>9.1}% {:>7.1}% {said:>7.1} of {n_cls} {off:>7.0}%{verdict}",
                 chance(labels));
    }
    println!();
}
