//! The local model store: `~/.cache/ferric/hub` (or `$FERRIC_HOME/hub`), the same directory
//! ferric-serve downloads into. Layout: `hub/<owner>_<repo>/<file>.gguf` for pulled models, plus any
//! loose `hub/<file>.gguf` someone put there by hand.
//!
//! A model is named by its file stem (`qwen2.5-0.5b-instruct-q8_0`), with a split set's
//! `-00001-of-00003` suffix removed so the three files are one model. Where two directories hold the
//! same stem the stem alone is ambiguous, and the hub-relative path (`<dir>/<stem>`) names it instead.
use ferric_gguf::{GgufFile, Meta};
use std::path::{Path, PathBuf};
use std::time::SystemTime;

pub fn ferric_home() -> PathBuf {
    match std::env::var_os("FERRIC_HOME") {
        Some(p) if !p.is_empty() => PathBuf::from(p),
        _ => PathBuf::from(std::env::var_os("HOME").unwrap_or_default()).join(".cache/ferric"),
    }
}

pub fn hub_dir() -> PathBuf { ferric_home().join("hub") }

/// `name-00002-of-00005.gguf` → `("name", 2, 5)` — llama-gguf-split's naming, which ferric-gguf's
/// `GgufFile::open` follows to find the siblings.
pub fn split_parts(file_name: &str) -> Option<(&str, u32, u32)> {
    let stem = file_name.strip_suffix(".gguf").or_else(|| file_name.strip_suffix(".GGUF"))?;
    let at = stem.rfind("-of-")?;
    let (no, count) = (&stem[..at], &stem[at + 4..]);
    if count.len() != 5 || !count.bytes().all(|b| b.is_ascii_digit()) || no.len() < 6 { return None; }
    let (prefix, no) = no.split_at(no.len() - 6);
    let no = no.strip_prefix('-')?;
    if !no.bytes().all(|b| b.is_ascii_digit()) { return None; }
    Some((prefix, no.parse().ok()?, count.parse().ok()?))
}

pub fn is_gguf(name: &str) -> bool { name.to_ascii_lowercase().ends_with(".gguf") }

#[derive(Clone, Debug)]
pub struct LocalModel {
    /// Stem, split suffix removed.
    pub name: String,
    /// The path to hand a loader — part 1 of a split set.
    pub path: PathBuf,
    /// Every file of the model, part order.
    pub parts: Vec<PathBuf>,
    /// Parts a split set declares but that are not on disk (an interrupted pull).
    pub missing: u32,
    pub size: u64,
    pub modified: SystemTime,
    /// Path relative to the hub, without `.gguf`, when the model lives in the hub.
    pub rel: Option<String>,
}

impl LocalModel {
    /// Build from any file of a model (for a split set, any part), locating its siblings. Paths are
    /// made canonical here, once, so the name sent to a server and the path compared against the hub
    /// are the same spelling (macOS's temp directory alone has two: /var and /private/var).
    pub fn from_file(path: &Path) -> LocalModel {
        let path = &absolute(path);
        let file = path.file_name().and_then(|s| s.to_str()).unwrap_or("").to_string();
        let dir = path.parent().map(Path::to_path_buf).unwrap_or_default();
        let (name, parts, missing) = match split_parts(&file) {
            Some((prefix, _, count)) => {
                let all: Vec<PathBuf> = (1..=count).map(|i| dir.join(format!("{prefix}-{i:05}-of-{count:05}.gguf"))).collect();
                let present: Vec<PathBuf> = all.iter().filter(|p| p.is_file()).cloned().collect();
                let missing = all.len() as u32 - present.len() as u32;
                (prefix.to_string(), all, missing)
            }
            None => (file.strip_suffix(".gguf").or_else(|| file.strip_suffix(".GGUF")).unwrap_or(&file).to_string(), vec![path.to_path_buf()], 0),
        };
        let mds: Vec<_> = parts.iter().filter_map(|p| std::fs::metadata(p).ok()).collect();
        LocalModel {
            name,
            path: parts[0].clone(),
            size: mds.iter().map(|m| m.len()).sum(),
            modified: mds.iter().filter_map(|m| m.modified().ok()).max().unwrap_or(SystemTime::UNIX_EPOCH),
            parts, missing, rel: None,
        }
    }
}

/// Every model in the hub: loose `.gguf` files and one directory level down. Split sets are one
/// entry; `.partial` downloads are not models.
pub fn scan(hub: &Path) -> Vec<LocalModel> {
    let hub = &absolute(hub);
    let mut files: Vec<PathBuf> = Vec::new();
    let Ok(rd) = std::fs::read_dir(hub) else { return Vec::new() };
    for e in rd.flatten() {
        let p = e.path();
        if p.is_dir() {
            if let Ok(sub) = std::fs::read_dir(&p) {
                files.extend(sub.flatten().map(|e| e.path()).filter(|p| p.is_file()));
            }
        } else if p.is_file() {
            files.push(p);
        }
    }
    files.retain(|p| p.file_name().and_then(|s| s.to_str()).is_some_and(is_gguf));
    files.sort();
    let mut seen = std::collections::HashSet::new();
    let mut out = Vec::new();
    for f in files {
        let m = LocalModel::from_file(&f);
        if !seen.insert(m.path.clone()) { continue; }
        out.push(m);
    }
    for m in out.iter_mut() {
        m.rel = m.path.parent().and_then(|d| d.strip_prefix(hub).ok()).map(|d| {
            let d = d.to_string_lossy();
            if d.is_empty() { m.name.clone() } else { format!("{d}/{}", m.name) }
        });
    }
    out
}

/// The name `list` prints and `run` accepts: the stem, or `<dir>/<stem>` where the stem is ambiguous.
pub fn display_names(models: &[LocalModel]) -> Vec<String> {
    models.iter().map(|m| {
        let dup = models.iter().filter(|o| o.name.eq_ignore_ascii_case(&m.name)).count() > 1;
        if dup { m.rel.clone().unwrap_or_else(|| m.name.clone()) } else { m.name.clone() }
    }).collect()
}

/// What a model reference names, before anything is downloaded.
#[derive(Debug)]
pub enum Target {
    Local(LocalModel),
    /// `owner/repo[:tag]` not (yet) in the hub.
    Hf(crate::pull::HfRef),
    /// Nothing local and not a Hugging Face reference: a name only the server can resolve (an Ollama
    /// model such as `llama3.2`, when `FERRIC_HOST` points at Ollama).
    Name(String),
}

/// Resolve a model reference: a path, a hub-relative path, a hub name, an HF reference (already
/// pulled → local), or a bare server-side name. `Err` only for an ambiguous hub name.
pub fn resolve(arg: &str) -> Result<Target, String> {
    let hub = hub_dir();
    let p = Path::new(arg);
    if p.is_file() { return Ok(Target::Local(LocalModel::from_file(&absolute(p)))); }
    for cand in [hub.join(arg), hub.join(format!("{arg}.gguf"))] {
        if cand.is_file() && is_gguf(&cand.to_string_lossy()) { return Ok(Target::Local(LocalModel::from_file(&cand))); }
    }
    let models = scan(&hub);
    let want = arg.strip_suffix(":latest").unwrap_or(arg);
    let exact: Vec<&LocalModel> = models.iter().filter(|m| m.name == want || m.rel.as_deref() == Some(want)).collect();
    let hits = if exact.is_empty() {
        models.iter().filter(|m| m.name.eq_ignore_ascii_case(want) || m.rel.as_deref().is_some_and(|r| r.eq_ignore_ascii_case(want))).collect()
    } else { exact };
    match hits.len() {
        1 => return Ok(Target::Local(hits[0].clone())),
        0 => {}
        _ => return Err(format!("'{arg}' names {} models in {}; use one of: {}", hits.len(), hub.display(),
                                hits.iter().filter_map(|m| m.rel.clone()).collect::<Vec<_>>().join(", "))),
    }
    if let Some(r) = crate::pull::HfRef::parse(arg) {
        // Already pulled? Then no network: pick the file the tag names from what is on disk.
        let dir = hub.join(r.dir_name());
        let names: Vec<String> = std::fs::read_dir(&dir).map(|rd| rd.flatten()
            .filter_map(|e| e.file_name().to_str().map(str::to_string)).filter(|n| is_gguf(n)).collect()).unwrap_or_default();
        if !names.is_empty() {
            if let Ok(set) = crate::pull::select(&names, r.tag.as_deref()) {
                let m = LocalModel::from_file(&dir.join(&set[0]));
                if m.missing == 0 { return Ok(Target::Local(m)); }
            }
        }
        return Ok(Target::Hf(r));
    }
    Ok(Target::Name(arg.to_string()))
}

pub fn absolute(p: &Path) -> PathBuf {
    std::fs::canonicalize(p).unwrap_or_else(|_| std::env::current_dir().map(|d| d.join(p)).unwrap_or_else(|_| p.to_path_buf()))
}

/// What the GGUF header says about a model — the fields `list` and `show` print.
#[derive(Debug, Default)]
pub struct Summary {
    pub arch: String,
    pub params: u64,
    /// The tensor type holding the most elements (ferric-serve's `/api/tags` rule).
    pub quant: String,
    pub context: Option<u64>,
    pub embedding: Option<u64>,
    pub layers: Option<u64>,
    pub template: Option<String>,
    pub license: Option<String>,
    pub general_name: Option<String>,
}

pub fn summarize(path: &Path) -> Result<Summary, String> {
    let g = GgufFile::open(path)?;
    let s = |k: &str| match g.metadata.get(k) { Some(Meta::Str(v)) => Some(v.clone()), _ => None };
    let u = |k: &str| match g.metadata.get(k) { Some(Meta::U(v)) => Some(*v), Some(Meta::I(v)) if *v >= 0 => Some(*v as u64), _ => None };
    let arch = s("general.architecture").unwrap_or_default();
    let mut by_type: std::collections::HashMap<u32, u64> = Default::default();
    let mut params = 0u64;
    for t in &g.tensors {
        let n: u64 = t.dims.iter().product();
        params += n;
        *by_type.entry(t.ggml_type).or_default() += n;
    }
    // Ties broken by type id so the answer does not depend on HashMap order.
    let quant = by_type.iter().max_by_key(|(t, n)| (**n, std::cmp::Reverse(**t)))
        .map(|(t, _)| ferric_gguf::type_name(*t).map(str::to_string).unwrap_or_else(|| format!("type{t}")))
        .unwrap_or_default();
    // A license is a string, or occasionally an array of them.
    let license = match g.metadata.get("general.license") {
        Some(Meta::Str(v)) => Some(v.clone()),
        Some(Meta::Arr(a)) => Some(a.iter().filter_map(|m| if let Meta::Str(s) = m { Some(s.clone()) } else { None }).collect::<Vec<_>>().join(", ")),
        _ => None,
    };
    Ok(Summary {
        context: u(&format!("{arch}.context_length")),
        embedding: u(&format!("{arch}.embedding_length")),
        layers: u(&format!("{arch}.block_count")),
        // Both spellings, as ferric-serve reads them: converters write the first, older files the second.
        template: s("tokenizer.chat_template").or_else(|| s("tokenizer.ggml.chat_template")),
        license,
        general_name: s("general.name"),
        arch, params, quant,
    })
}

// ---- formatting ----

/// Decimal units, as `ollama list` prints them.
pub fn human_size(b: u64) -> String {
    let b = b as f64;
    if b >= 1e9 { format!("{:.1} GB", b / 1e9) } else if b >= 1e6 { format!("{:.1} MB", b / 1e6) }
    else if b >= 1e3 { format!("{:.1} KB", b / 1e3) } else { format!("{b} B") }
}

pub fn human_params(p: u64) -> String {
    let p = p as f64;
    if p >= 1e9 { format!("{:.1}B", p / 1e9) } else if p >= 1e6 { format!("{:.1}M", p / 1e6) }
    else if p >= 1e3 { format!("{:.1}K", p / 1e3) } else { format!("{p}") }
}

pub fn ago(t: SystemTime) -> String {
    let Ok(d) = SystemTime::now().duration_since(t) else { return "just now".into() };
    let s = d.as_secs();
    let (n, unit) = match s {
        0..=59 => return "just now".into(),
        60..=3599 => (s / 60, "minute"),
        3600..=86_399 => (s / 3600, "hour"),
        86_400..=2_591_999 => (s / 86_400, "day"),
        2_592_000..=31_535_999 => (s / 2_592_000, "month"),
        _ => (s / 31_536_000, "year"),
    };
    format!("{n} {unit}{} ago", if n == 1 { "" } else { "s" })
}

/// Parse the RFC 3339 instants the servers emit (`2026-09-27T12:00:00.123Z`, `…+02:00`) to Unix
/// seconds. Hinnant's days-from-civil; no date crate.
pub fn parse_rfc3339(s: &str) -> Option<i64> {
    let b = s.as_bytes();
    if b.len() < 19 || b[4] != b'-' || b[7] != b'-' || (b[10] != b'T' && b[10] != b' ') { return None; }
    let n = |r: std::ops::Range<usize>| s.get(r)?.parse::<i64>().ok();
    let (y, mo, d, h, mi, se) = (n(0..4)?, n(5..7)?, n(8..10)?, n(11..13)?, n(14..16)?, n(17..19)?);
    let mut rest = &s[19..];
    if let Some(r) = rest.strip_prefix('.') { rest = r.trim_start_matches(|c: char| c.is_ascii_digit()); }
    let off = match rest {
        "" | "Z" | "z" => 0,
        r if r.len() == 6 && (r.starts_with('+') || r.starts_with('-')) => {
            let sign = if r.starts_with('-') { -1 } else { 1 };
            sign * (r[1..3].parse::<i64>().ok()? * 3600 + r[4..6].parse::<i64>().ok()? * 60)
        }
        _ => return None,
    };
    let y2 = if mo <= 2 { y - 1 } else { y };
    let era = y2.div_euclid(400);
    let yoe = y2 - era * 400;
    let mp = (mo + 9) % 12;
    let doy = (153 * mp + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    let days = era * 146_097 + doe - 719_468;
    Some(days * 86_400 + h * 3600 + mi * 60 + se - off)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn split_names() {
        assert_eq!(split_parts("qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf"), Some(("qwen2.5-7b-instruct-q4_k_m", 1, 2)));
        assert_eq!(split_parts("a-00003-of-00003.gguf"), Some(("a", 3, 3)));
        assert_eq!(split_parts("model-q4_k_m.gguf"), None);
        assert_eq!(split_parts("x-of-00002.gguf"), None);
        assert_eq!(split_parts("x-1-of-00002.gguf"), None);
    }

    #[test]
    fn rfc3339_round_trips_known_instants() {
        assert_eq!(parse_rfc3339("1970-01-01T00:00:00Z"), Some(0));
        assert_eq!(parse_rfc3339("2000-02-29T00:00:00.000000000Z"), Some(951_782_400));
        assert_eq!(parse_rfc3339("2026-09-27T12:00:00.5Z"), Some(1_790_510_400));
        assert_eq!(parse_rfc3339("2026-09-27T14:00:00+02:00"), Some(1_790_510_400));
        assert_eq!(parse_rfc3339("not a date"), None);
    }

    #[test]
    fn units() {
        assert_eq!(human_size(675_710_816), "675.7 MB");
        assert_eq!(human_size(4_683_073_632), "4.7 GB");
        assert_eq!(human_params(494_032_768), "494.0M");
        assert_eq!(human_params(7_615_616_512), "7.6B");
    }
}
