//! Read MATLAB v5 corpora and report what is in them.
//!
//!   cargo run -p ferric-signal --example matdump --release -- <file-or-dir> [--check ref.tsv]
//!
//! Three of the four public sensor corpora this crate was pointed at ship as `.mat` and none of
//! them could be opened before [`ferric_signal::mat`] existed. This walks a file or a directory of
//! them and prints every variable, its class and shape, and — for anything that reads as a channel
//! — its length and first samples.
//!
//! `--check` compares against a reference table written by another implementation, one row per
//! `file<TAB>channel<TAB>len<TAB>first<TAB>last<TAB>sum`. A parser validated only against its own
//! output agrees with its own bugs; the point of the flag is that the numbers came from somewhere
//! else.

use ferric_signal::{MatError, MatFile, MatValue};
use std::collections::BTreeMap;

fn describe(v: &MatValue) -> String {
    let d = v.dims().iter().map(|x| x.to_string()).collect::<Vec<_>>().join("x");
    match v {
        MatValue::Numeric { class, .. } => format!("{class:?}[{d}]"),
        MatValue::Char { text, .. } => format!("char[{d}] {text:?}"),
        MatValue::Struct { fields, .. } => format!("struct[{d}] {{{}}}", fields.join(", ")),
        MatValue::Cell { .. } => format!("cell[{d}]"),
    }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let Some(target) = args.first() else {
        eprintln!("usage: matdump <file-or-dir> [--check ref.tsv]");
        std::process::exit(2);
    };
    let check = args.iter().position(|a| a == "--check").and_then(|i| args.get(i + 1)).cloned();

    let mut files: Vec<std::path::PathBuf> = Vec::new();
    let p = std::path::Path::new(target);
    if p.is_dir() {
        let mut e: Vec<_> = std::fs::read_dir(p)
            .unwrap_or_else(|e| { eprintln!("error: {target}: {e}"); std::process::exit(1) })
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().is_some_and(|x| x == "mat"))
            .collect();
        e.sort();
        files = e;
    } else {
        files.push(p.to_path_buf());
    }
    if files.is_empty() {
        eprintln!("error: no .mat files at {target}");
        std::process::exit(1);
    }

    // Reference rows, if a table was given.
    let mut want: BTreeMap<(String, String), (usize, f64, f64, f64)> = BTreeMap::new();
    if let Some(path) = &check {
        let text = std::fs::read_to_string(path)
            .unwrap_or_else(|e| { eprintln!("error: {path}: {e}"); std::process::exit(1) });
        for line in text.lines() {
            let f: Vec<&str> = line.split('\t').collect();
            if f.len() != 6 {
                continue;
            }
            want.insert(
                (f[0].to_string(), f[1].to_string()),
                (f[2].parse().unwrap(), f[3].parse().unwrap(), f[4].parse().unwrap(), f[5].parse().unwrap()),
            );
        }
        println!("\nreference table: {} rows from {path}", want.len());
    }

    let (mut ok, mut failed, mut chans, mut samples) = (0usize, 0usize, 0usize, 0usize);
    let mut errors: BTreeMap<String, usize> = BTreeMap::new();
    let (mut matched, mut mismatched) = (0usize, 0usize);

    for path in &files {
        let name = path.file_name().unwrap().to_string_lossy().to_string();
        let bytes = match std::fs::read(path) {
            Ok(b) => b,
            Err(e) => { eprintln!("  {name}: {e}"); failed += 1; continue; }
        };
        match MatFile::parse(&bytes) {
            Ok(m) => {
                ok += 1;
                let c = m.channels();
                chans += c.len();
                samples += c.values().map(|s| s.len()).sum::<usize>();
                if want.is_empty() && files.len() <= 4 {
                    println!("\n{name}  ({} bytes)\n  {}", bytes.len(), m.header);
                    for (n, v) in &m.vars {
                        println!("    {n:<28} {}", describe(v));
                    }
                    for (n, s) in &c {
                        let head: Vec<String> =
                            s.iter().take(3).map(|v| format!("{v:.6}")).collect();
                        println!("    channel {n:<20} {:>9} samples  [{}...]", s.len(), head.join(", "));
                    }
                }
                for (n, s) in &c {
                    if let Some(&(wl, wf, wlst, wsum)) = want.get(&(name.clone(), n.clone())) {
                        let sum: f64 = s.iter().sum();
                        let close = |a: f64, b: f64| (a - b).abs() <= 1e-9 * a.abs().max(b.abs()).max(1.0);
                        if s.len() == wl
                            && close(s[0], wf)
                            && close(s[s.len() - 1], wlst)
                            && close(sum, wsum)
                        {
                            matched += 1;
                        } else {
                            mismatched += 1;
                            println!("  MISMATCH {name} {n}: len {} vs {wl}, first {} vs {wf}, last {} vs {wlst}, sum {sum} vs {wsum}",
                                     s.len(), s[0], s[s.len() - 1]);
                        }
                    }
                }
            }
            Err(e) => {
                failed += 1;
                let key = match &e {
                    MatError::Compressed { .. } => "miCOMPRESSED (zlib)".to_string(),
                    MatError::UnsupportedClass { name, .. } => format!("class {name}"),
                    other => format!("{other}"),
                };
                *errors.entry(key).or_insert(0) += 1;
            }
        }
    }

    println!("\n  {} files: {ok} read, {failed} refused", files.len());
    println!("  {chans} channels, {samples} samples");
    for (e, n) in &errors {
        println!("    refused x{n}: {e}");
    }
    if !want.is_empty() {
        println!("\n  AGAINST THE REFERENCE TABLE: {matched} channels agree, {mismatched} differ");
        let unchecked = chans.saturating_sub(matched + mismatched);
        if unchecked > 0 {
            println!("  {unchecked} channels had no reference row and were NOT checked");
        }
        if mismatched > 0 {
            std::process::exit(1);
        }
    }
}
