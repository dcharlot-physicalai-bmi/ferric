//! `ferric pull owner/repo[:QUANT | :file.gguf]` — a GGUF from Hugging Face into the hub.
//!
//! Hugging Face is HTTPS and no HTTPS client is vendored, so every byte goes through the `curl`
//! binary — the same choice ferric-serve made for its downloader. `FERRIC_HF_ENDPOINT` (else
//! `HF_ENDPOINT`, huggingface_hub's variable) replaces `https://huggingface.co`, which is how a
//! mirror is used and how the tests point this at a local mock.
//!
//! Three things this does that ferric-serve's one-shot downloader does not, and why each matters:
//!
//! - **A quant tag selects the file** (`:Q4_K_M`, Ollama's `hf.co/owner/repo:Q4_K_M` syntax), matched
//!   as a whole token so `Q4_K` does not take `Q4_K_M` and `F16` does not take `BF16`.
//! - **Split GGUFs are fetched whole.** `-00001-of-00003.gguf` alone loads nothing: ferric-gguf's
//!   `GgufFile::open` follows `split.count` to siblings that must sit next to it.
//! - **A download is a `.partial` until its size is right.** ferric-serve treats any non-empty file
//!   as cached, so an interrupted download there is a truncated model that looks complete. Here the
//!   `.partial` is resumed with `curl -C -` and renamed only once it matches the size the API gave.
use crate::hub::{self, LocalModel};
use serde_json::Value;
use std::io::{IsTerminal, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

#[derive(Clone, Debug, PartialEq)]
pub struct HfRef { pub repo: String, pub tag: Option<String> }

impl HfRef {
    /// `owner/repo`, `owner/repo:Q4_K_M`, `owner/repo:file.gguf`, with an optional `hf.co/`,
    /// `huggingface.co/` or `https://huggingface.co/` prefix. `:latest` means no tag.
    pub fn parse(s: &str) -> Option<HfRef> {
        let mut s = s.trim();
        for p in ["https://huggingface.co/", "http://huggingface.co/", "huggingface.co/", "hf.co/"] {
            if let Some(r) = s.strip_prefix(p) { s = r; break; }
        }
        let (repo, tag) = match s.split_once(':') { Some((r, t)) => (r, Some(t)), None => (s, None) };
        let (owner, name) = repo.split_once('/')?;
        let ok = |x: &str| !x.is_empty() && x.chars().all(|c| c.is_ascii_alphanumeric() || "-_.".contains(c));
        if !ok(owner) || !ok(name) { return None; }
        let tag = tag.map(str::trim).filter(|t| !t.is_empty() && !t.eq_ignore_ascii_case("latest")).map(str::to_string);
        Some(HfRef { repo: format!("{owner}/{name}"), tag })
    }

    /// `owner_repo` — ferric-serve's directory naming, so both find the same files.
    pub fn dir_name(&self) -> String { self.repo.replace('/', "_") }
}

pub fn endpoint() -> String {
    ["FERRIC_HF_ENDPOINT", "HF_ENDPOINT"].iter().filter_map(|k| std::env::var(k).ok())
        .find(|v| !v.trim().is_empty()).unwrap_or_else(|| "https://huggingface.co".into())
        .trim_end_matches('/').to_string()
}

/// Does the file name carry this quant tag as a whole token? Case-insensitive; the character before
/// must not be alphanumeric (so `BF16` is not `F16`, `IQ4_XS` is not `Q4_XS`) and the one after must
/// not continue the name (so `Q4_K` is not `Q4_K_M`).
pub fn has_quant(file: &str, tag: &str) -> bool {
    let (f, t) = (file.to_ascii_uppercase(), tag.to_ascii_uppercase());
    let fb = f.as_bytes();
    let mut from = 0;
    while let Some(i) = f[from..].find(&t) {
        let (a, b) = (from + i, from + i + t.len());
        let before_ok = a == 0 || !fb[a - 1].is_ascii_alphanumeric();
        let after_ok = b == fb.len() || !(fb[b].is_ascii_alphanumeric() || fb[b] == b'_');
        if before_ok && after_ok { return true; }
        from = a + 1;
    }
    false
}

/// The quant token in a file name, for listing choices (`…-Q4_K_M.gguf` → `Q4_K_M`).
pub fn quant_of(file: &str) -> Option<String> {
    let base = file.rsplit('/').next().unwrap_or(file);
    let stem = base.strip_suffix(".gguf").or_else(|| base.strip_suffix(".GGUF")).unwrap_or(base);
    let stem = hub::split_parts(base).map(|(p, _, _)| p).unwrap_or(stem);
    stem.rsplit(['-', '.']).find(|tok| {
        let u = tok.to_ascii_uppercase();
        let body = u.strip_prefix("IQ").or_else(|| u.strip_prefix("BF")).or_else(|| u.strip_prefix("TQ"))
            .or_else(|| u.strip_prefix('Q')).or_else(|| u.strip_prefix('F'));
        body.is_some_and(|b| b.chars().next().is_some_and(|c| c.is_ascii_digit()))
    }).map(|t| t.to_ascii_uppercase())
}

/// One downloadable model: a single file, or every part of a split set (part order).
fn sets(files: &[String]) -> Vec<Vec<String>> {
    let mut out: Vec<Vec<String>> = Vec::new();
    let mut split: std::collections::BTreeMap<(String, u32), Vec<(u32, String)>> = Default::default();
    for f in files.iter().filter(|f| hub::is_gguf(f)) {
        let (dir, base) = match f.rsplit_once('/') { Some((d, b)) => (d.to_string(), b), None => (String::new(), f.as_str()) };
        match hub::split_parts(base) {
            Some((prefix, no, count)) => split.entry((format!("{dir}/{prefix}"), count)).or_default().push((no, f.clone())),
            None => out.push(vec![f.clone()]),
        }
    }
    for (_, mut parts) in split {
        parts.sort();
        out.push(parts.into_iter().map(|(_, f)| f).collect());
    }
    out
}

fn is_aux(f: &str) -> bool {
    let b = f.rsplit('/').next().unwrap_or(f).to_ascii_lowercase();
    b.starts_with("mmproj") || b.contains("imatrix")
}

/// Choose the model a tag names from a repo's file list. `Err` says what the choices were.
pub fn select(files: &[String], tag: Option<&str>) -> Result<Vec<String>, String> {
    let all = sets(files);
    if all.is_empty() { return Err("the repo holds no .gguf file".into()); }
    let base = |f: &str| f.rsplit('/').next().unwrap_or(f).to_string();
    let choices = |v: &[Vec<String>]| {
        let mut t: Vec<String> = v.iter().map(|s| quant_of(&s[0]).unwrap_or_else(|| base(&s[0]))).collect();
        t.sort(); t.dedup();
        t.join(", ")
    };
    // A vision projector or an importance matrix is never what "the model" means.
    let models: Vec<Vec<String>> = all.iter().filter(|s| !is_aux(&s[0])).cloned().collect();
    let models = if models.is_empty() { all.clone() } else { models };
    let hits: Vec<Vec<String>> = match tag {
        Some(t) if hub::is_gguf(t) => all.iter().filter(|s| s.iter().any(|f| f == t || base(f) == t)).cloned().collect(),
        Some(t) => models.iter().filter(|s| has_quant(&base(&s[0]), t)).cloned().collect(),
        None if models.len() == 1 => models.clone(),
        // Ollama's default for an hf.co reference.
        None => models.iter().filter(|s| has_quant(&base(&s[0]), "Q4_K_M")).cloned().collect(),
    };
    match hits.len() {
        1 => Ok(hits.into_iter().next().unwrap()),
        0 => Err(match tag {
            Some(t) => format!("no GGUF matches '{t}'; available: {}", choices(&models)),
            None => format!("{} GGUF models and no Q4_K_M default; name one with :TAG — available: {}", models.len(), choices(&models)),
        }),
        n => Err(format!("'{}' matches {n} files ({}); name the file with :file.gguf",
                         tag.unwrap_or("Q4_K_M"), hits.iter().map(|s| base(&s[0])).collect::<Vec<_>>().join(", "))),
    }
}

fn hf_token() -> Option<String> {
    if let Ok(t) = std::env::var("HF_TOKEN") { if !t.trim().is_empty() { return Some(t.trim().to_string()); } }
    let hf_home = std::env::var_os("HF_HOME").map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(std::env::var_os("HOME").unwrap_or_default()).join(".cache/huggingface"));
    std::fs::read_to_string(hf_home.join("token")).ok().map(|t| t.trim().to_string()).filter(|t| !t.is_empty())
}

/// A curl invocation. The token goes in on stdin as a curl config line (`-K -`), never in argv,
/// where every other user of the machine could read it from `ps`.
fn curl(args: &[&str], quiet_progress: bool) -> Result<(std::process::ExitStatus, Vec<u8>), String> {
    let mut c = Command::new("curl");
    c.args(["-L", "--noproxy", "localhost,127.0.0.1,::1", "-A", concat!("ferric/", env!("CARGO_PKG_VERSION"))]);
    if quiet_progress { c.args(["-sS"]); }
    let token = hf_token();
    if token.is_some() { c.args(["-K", "-"]); }
    c.args(args).stdin(if token.is_some() { Stdio::piped() } else { Stdio::null() }).stdout(Stdio::piped()).stderr(Stdio::inherit());
    let mut child = c.spawn().map_err(|e| format!("could not run curl ({e}); Hugging Face downloads go through the curl binary"))?;
    if let (Some(t), Some(mut stdin)) = (token, child.stdin.take()) {
        let _ = writeln!(stdin, "header = \"Authorization: Bearer {}\"", t.replace('"', ""));
    }
    let out = child.wait_with_output().map_err(|e| e.to_string())?;
    Ok((out.status, out.stdout))
}

/// Percent-encode a repo path for a URL, keeping `/`.
fn url_path(p: &str) -> String {
    p.bytes().map(|b| if b.is_ascii_alphanumeric() || b"-_.~/".contains(&b) { (b as char).to_string() } else { format!("%{b:02X}") }).collect()
}

/// The repo's files and sizes, and the commit to pin every part to (so a set is never mixed across
/// two commits of the repo).
fn listing(r: &HfRef) -> Result<(Vec<(String, Option<u64>)>, String), String> {
    let url = format!("{}/api/models/{}?blobs=true", endpoint(), r.repo);
    let (st, out) = curl(&["-w", "\n%{http_code}", &url], true)?;
    let text = String::from_utf8_lossy(&out);
    let (body, code) = text.rsplit_once('\n').unwrap_or(("", text.as_ref()));
    let code: u16 = code.trim().parse().unwrap_or(0);
    if !st.success() && code == 0 { return Err(format!("could not reach {url}")); }
    match code {
        200 => {}
        401 | 403 => return Err(format!("{} is gated or private (HTTP {code}): accept its license on huggingface.co and set HF_TOKEN", r.repo)),
        404 => return Err(format!("no model repo '{}' at {}", r.repo, endpoint())),
        c => return Err(format!("{url}: HTTP {c}")),
    }
    let v: Value = serde_json::from_str(body).map_err(|e| format!("{url}: not JSON ({e})"))?;
    let files = v["siblings"].as_array().map(|a| a.iter().filter_map(|s| {
        let name = s["rfilename"].as_str()?.to_string();
        let size = s["size"].as_u64().or_else(|| s["lfs"]["size"].as_u64());
        Some((name, size))
    }).collect()).unwrap_or_default();
    let rev = v["sha"].as_str().filter(|s| !s.is_empty()).unwrap_or("main").to_string();
    Ok((files, rev))
}

/// Download one file to `dest`, through `dest.partial`, resuming whatever is already there.
fn fetch(url: &str, dest: &Path, expect: Option<u64>, label: &str) -> Result<(), String> {
    let partial = PathBuf::from(format!("{}.partial", dest.display()));
    let len = |p: &Path| std::fs::metadata(p).map(|m| m.len()).ok();
    if let Some(have) = len(dest) {
        if expect.is_none_or(|e| e == have) && have > 0 { eprintln!("{label}: up to date ({})", hub::human_size(have)); return Ok(()); }
        // A final-named file of the wrong size is an interrupted download from elsewhere
        // (ferric-serve writes in place): resume it rather than trust it or throw it away.
        std::fs::rename(dest, &partial).map_err(|e| format!("{}: {e}", dest.display()))?;
    }
    if let (Some(have), Some(e)) = (len(&partial), expect) {
        if have > e { let _ = std::fs::remove_file(&partial); }
    }
    let have = len(&partial).unwrap_or(0);
    let size = expect.map(hub::human_size).unwrap_or_else(|| "size unknown".into());
    if have > 0 { eprintln!("{label}: resuming at {} of {size}", hub::human_size(have)); } else { eprintln!("{label}: {size}"); }
    if expect != Some(have) {
        let quiet = !std::io::stderr().is_terminal();
        let p = partial.to_string_lossy().to_string();
        let mut args = vec!["-f", "-C", "-", "-o", p.as_str()];
        if !quiet { args.push("--progress-bar"); }
        args.push(url);
        let (st, _) = curl(&args, quiet)?;
        if !st.success() {
            return Err(format!("download failed ({st}) — {url}; run the same pull again to resume from {}",
                               hub::human_size(len(&partial).unwrap_or(0))));
        }
    }
    let got = len(&partial).unwrap_or(0);
    if let Some(e) = expect {
        if got != e { return Err(format!("{}: {got} bytes, the repo says {e}; run the pull again to resume", partial.display())); }
    }
    std::fs::rename(&partial, dest).map_err(|e| format!("{}: {e}", dest.display()))?;
    Ok(())
}

pub fn pull(r: &HfRef) -> Result<LocalModel, String> {
    let (files, rev) = listing(r)?;
    let names: Vec<String> = files.iter().map(|(n, _)| n.clone()).collect();
    let set = select(&names, r.tag.as_deref()).map_err(|e| format!("{}: {e}", r.repo))?;
    let dir = hub::hub_dir().join(r.dir_name());
    std::fs::create_dir_all(&dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    let n = set.len();
    let mut dests = Vec::new();
    for (i, f) in set.iter().enumerate() {
        let base = f.rsplit('/').next().unwrap_or(f);
        let dest = dir.join(base);
        let expect = files.iter().find(|(x, _)| x == f).and_then(|(_, s)| *s);
        let url = format!("{}/{}/resolve/{rev}/{}", endpoint(), r.repo, url_path(f));
        let label = if n > 1 { format!("pulling {base} (part {} of {n})", i + 1) } else { format!("pulling {base}") };
        fetch(&url, &dest, expect, &label)?;
        dests.push(dest);
    }
    let m = LocalModel::from_file(&dests[0]);
    eprintln!("success: {} → {}", r.repo, m.path.display());
    Ok(m)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn refs() {
        assert_eq!(HfRef::parse("hf.co/bartowski/Qwen2.5-0.5B-Instruct-GGUF:Q4_K_M"),
                   Some(HfRef { repo: "bartowski/Qwen2.5-0.5B-Instruct-GGUF".into(), tag: Some("Q4_K_M".into()) }));
        assert_eq!(HfRef::parse("Qwen/Qwen2.5-0.5B-Instruct-GGUF").unwrap().tag, None);
        assert_eq!(HfRef::parse("a/b:latest").unwrap().tag, None);
        assert_eq!(HfRef::parse("a/b:sub/x.gguf").unwrap().tag.as_deref(), Some("sub/x.gguf"));
        assert!(HfRef::parse("llama3.2").is_none());
        assert!(HfRef::parse("./models/x.gguf").is_none());
        assert!(HfRef::parse("/abs/x.gguf").is_none());
    }

    #[test]
    fn quant_tokens_match_whole() {
        assert!(has_quant("qwen2.5-0.5b-instruct-q4_k_m.gguf", "Q4_K_M"));
        assert!(!has_quant("qwen2.5-0.5b-instruct-q4_k_m.gguf", "Q4_K"), "Q4_K must not take Q4_K_M");
        assert!(has_quant("tinyllama-1.1b-chat-v1.0.Q4_K_S.gguf", "q4_k_s"));
        assert!(!has_quant("m-BF16.gguf", "F16"), "F16 must not take BF16");
        assert!(!has_quant("m-IQ4_XS.gguf", "Q4_XS"));
        assert_eq!(quant_of("Qwen2.5-0.5B-Instruct-IQ4_XS.gguf").as_deref(), Some("IQ4_XS"));
        assert_eq!(quant_of("big-q4_k_m-00001-of-00002.gguf").as_deref(), Some("Q4_K_M"));
        assert_eq!(quant_of("tinyllama-1.1b-chat-v1.0.Q8_0.gguf").as_deref(), Some("Q8_0"));
    }

    #[test]
    fn selection() {
        let f = |v: &[&str]| v.iter().map(|s| s.to_string()).collect::<Vec<_>>();
        let repo = f(&["README.md", "m-Q8_0.gguf", "m-Q4_K_M.gguf", "m-Q4_K_S.gguf", "mmproj-m-F16.gguf",
                       "m-Q6_K-00002-of-00002.gguf", "m-Q6_K-00001-of-00002.gguf"]);
        assert_eq!(select(&repo, None).unwrap(), f(&["m-Q4_K_M.gguf"]), "Q4_K_M is the default");
        assert_eq!(select(&repo, Some("q8_0")).unwrap(), f(&["m-Q8_0.gguf"]));
        assert_eq!(select(&repo, Some("Q6_K")).unwrap(), f(&["m-Q6_K-00001-of-00002.gguf", "m-Q6_K-00002-of-00002.gguf"]));
        assert_eq!(select(&repo, Some("m-Q6_K-00002-of-00002.gguf")).unwrap().len(), 2, "any part names the set");
        assert!(select(&repo, Some("F16")).is_err(), "a projector is not the model");
        assert!(select(&f(&["a-Q8_0.gguf", "a-Q6_K.gguf"]), None).unwrap_err().contains("Q6_K, Q8_0"));
        assert_eq!(select(&f(&["only.gguf"]), None).unwrap(), f(&["only.gguf"]));
    }
}
