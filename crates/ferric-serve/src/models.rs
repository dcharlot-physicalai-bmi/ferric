//! **Which models a request may name**, and loading them when it does.
//!
//! Parity gap S24 (several models at once, load/unload on demand, keep-alive): Ollama, LM Studio,
//! LocalAI, llama-swap, Lemonade and Docker Model Runner all serve every model on disk from one endpoint
//! and load on first use; ferric-serve held exactly one. This is the other half of `batch::Pool`: the
//! pool decides WHEN to load and unload; this decides WHAT a name means and builds the `Engine`.
//!
//! A name resolves, in order, as
//! 1. a path to a `.gguf` file;
//! 2. a model in the model directory (`FERRIC_MODELS`, default `~/.cache/ferric/hub`, where
//!    `ferric-serve owner/repo` and `ferric pull` download to), by file stem — `qwen2.5-0.5b-instruct-q8_0`
//!    — with Ollama's `:latest` accepted and ignored; a stem two directories share is named `dir/stem`;
//! 3. a downloaded Hugging Face repository, `owner/repo[:tag]` (also `hf.co/owner/repo:tag`, Ollama's
//!    spelling): the tag picks the file whose name contains it (`:Q4_K_M`), and without one a repository
//!    holding several files resolves to its Q4_K_M or is refused with the list.
//!
//! Nothing is downloaded on a request: a name that is not on disk is a 404 that says how to fetch it,
//! the choice Ollama makes. A multi-gigabyte download started by a chat request would hold its client for
//! minutes with no progress to show.
//!
//! Only files a chat model can be built from are listed or loaded: a GGUF whose architecture the
//! registry refuses, a LoRA adapter, a vision projector or an encoder (served with `--embed` /
//! `--rerank` / `--asr`) is left out of `/api/tags` rather than listed and then refused.
use crate::{batch, ollama, Engine, Model, Shared};
use ferric_gguf::{GgufFile, Meta};
use serde_json::Value;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::SystemTime;

/// One file in the model directory.
#[derive(Clone)]
struct Entry { name: String, path: PathBuf, bytes: u64 }

pub(crate) struct Hub {
    shared: Shared,
    dirs: Vec<PathBuf>,
    /// Names given on the command line (`--name`), by key.
    names: HashMap<String, String>,
    /// What each file's header says, kept by path and invalidated by size and mtime: a header read
    /// costs up to 8 MB and `/api/tags` is polled by every front end.
    headers: HashMap<PathBuf, (u64, SystemTime, Option<ollama::Card>)>,
    /// (name, arch, context) of every model this source has loaded, by key — for `/metrics`.
    info: HashMap<String, (String, String, usize)>,
}

/// The split-part suffix `llama-gguf-split` writes: `-00001-of-00003`.
fn split_part(stem: &str) -> Option<(&str, usize)> {
    let at = stem.rfind("-of-")?;
    let (head, tail) = (&stem[..at], &stem[at + 4..]);
    if tail.len() != 5 || !tail.bytes().all(|b| b.is_ascii_digit()) || head.len() < 6 { return None; }
    let (base, no) = head.split_at(head.len() - 6);
    let no = no.strip_prefix('-')?;
    if no.len() != 5 || !no.bytes().all(|b| b.is_ascii_digit()) { return None; }
    Some((base, no.parse().ok()?))
}

/// Every part of a model's size: a split model is all its parts.
fn total_bytes(path: &Path) -> u64 {
    let one = |p: &Path| std::fs::metadata(p).map(|m| m.len()).unwrap_or(0);
    let Some(stem) = path.file_stem().and_then(|s| s.to_str()) else { return one(path) };
    let Some((base, _)) = split_part(stem) else { return one(path) };
    let dir = path.parent().unwrap_or(Path::new("."));
    std::fs::read_dir(dir).map(|rd| rd.flatten().map(|e| e.path())
        .filter(|p| p.file_stem().and_then(|s| s.to_str()).and_then(split_part).is_some_and(|(b, _)| b == base))
        .map(|p| one(&p)).sum()).unwrap_or(0)
}

fn base_name(s: &str) -> &str { s.strip_suffix(":latest").unwrap_or(s) }

impl Hub {
    pub(crate) fn new(shared: Shared) -> Hub {
        let dir = std::env::var("FERRIC_MODELS").ok().filter(|d| !d.is_empty()).map(PathBuf::from).unwrap_or_else(|| {
            PathBuf::from(std::env::var("HOME").unwrap_or_default()).join(".cache/ferric/hub")
        });
        Hub { shared, dirs: vec![dir], names: HashMap::new(), headers: HashMap::new(), info: HashMap::new() }
    }

    /// A name for the model at `key` other than its file stem (`--name`).
    pub(crate) fn name(&mut self, key: &str, name: String) { self.names.insert(key.to_string(), name); }

    /// The GGUFs in the model directory and one level below it, one entry per model (a split model's
    /// first part stands for it), named by stem — or `dir/stem` where two directories share a stem.
    fn scan(&self) -> Vec<Entry> {
        let mut files: Vec<PathBuf> = Vec::new();
        for d in &self.dirs {
            let Ok(rd) = std::fs::read_dir(d) else { continue };
            for e in rd.flatten() {
                let p = e.path();
                if p.is_dir() {
                    if let Ok(sub) = std::fs::read_dir(&p) { files.extend(sub.flatten().map(|e| e.path()).filter(|p| p.is_file())); }
                } else { files.push(p); }
            }
        }
        files.retain(|p| p.extension().is_some_and(|x| x == "gguf"));
        files.sort();
        let stem_of = |p: &Path| {
            let s = p.file_stem().and_then(|s| s.to_str()).unwrap_or("").to_string();
            match split_part(&s) { Some((b, _)) => b.to_string(), None => s }
        };
        files.retain(|p| p.file_stem().and_then(|s| s.to_str()).and_then(split_part).is_none_or(|(_, no)| no == 1));
        let mut count: HashMap<String, usize> = HashMap::new();
        for p in &files { *count.entry(stem_of(p).to_lowercase()).or_default() += 1; }
        files.into_iter().map(|p| {
            let stem = stem_of(&p);
            let name = if count[&stem.to_lowercase()] > 1 {
                let dir = p.parent().and_then(|d| d.file_name()).and_then(|s| s.to_str()).unwrap_or("");
                format!("{dir}/{stem}")
            } else { stem };
            Entry { bytes: total_bytes(&p), name, path: p }
        }).collect()
    }

    /// The card a file's header gives, or `None` when no chat model can be built from it. Cached.
    fn card(&mut self, e: &Entry) -> Option<ollama::Card> {
        let md = std::fs::metadata(&e.path).ok()?;
        let (len, mt) = (md.len(), md.modified().unwrap_or(SystemTime::UNIX_EPOCH));
        if let Some((l, m, c)) = self.headers.get(&e.path) {
            if *l == len && *m == mt { return c.clone().map(|mut c| { c.name = e.name.clone(); c }); }
        }
        let card = GgufFile::open(&e.path).ok().and_then(|g| {
            let s = |k: &str| match g.metadata.get(k) { Some(Meta::Str(v)) => Some(v.clone()), _ => None };
            let u = |k: &str| match g.metadata.get(k) { Some(Meta::U(v)) => Some(*v as usize), _ => None };
            if s("general.type").as_deref() == Some("adapter") { return None; }
            let arch = s("general.architecture")?;
            let entry = ferric_llama::arch::resolve(&arch).ok()?;
            Model::dispatchable(entry.runtime).ok()?;
            let template = s("tokenizer.chat_template").or_else(|| s("tokenizer.ggml.chat_template")).unwrap_or_default();
            Some(ollama::Card::from_gguf(&e.name, &e.path.to_string_lossy(), &g, false,
                u(&format!("{arch}.embedding_length")).unwrap_or(0), u(&format!("{arch}.context_length")).unwrap_or(4096), &template))
        });
        self.headers.insert(e.path.clone(), (len, mt, card.clone()));
        card
    }

    fn key_of(p: &Path) -> String { std::fs::canonicalize(p).unwrap_or_else(|_| p.to_path_buf()).to_string_lossy().into_owned() }
}

impl batch::Source for Hub {
    type M = Engine;

    fn resolve(&mut self, spec: &str) -> Result<(String, u64), String> {
        if spec.ends_with(".gguf") && Path::new(spec).is_file() {
            return Ok((Hub::key_of(Path::new(spec)), total_bytes(Path::new(spec))));
        }
        let want = base_name(spec.strip_prefix("hf.co/").or_else(|| spec.strip_prefix("huggingface.co/")).unwrap_or(spec));
        let entries = self.scan();
        if let Some(e) = entries.iter().find(|e| e.name.eq_ignore_ascii_case(want)) {
            return Ok((Hub::key_of(&e.path), e.bytes));
        }
        // owner/repo[:tag] — the directory `ferric-serve owner/repo` / `ferric pull` downloads into.
        if let Some((repo, tag)) = want.split_once(':').map(|(r, t)| (r, Some(t))).or(Some((want, None))).filter(|(r, _)| r.contains('/')) {
            let dir = repo.replace('/', "_");
            let mut hits: Vec<&Entry> = entries.iter().filter(|e| e.path.parent().and_then(|d| d.file_name()).and_then(|s| s.to_str())
                .is_some_and(|d| d.eq_ignore_ascii_case(&dir))).collect();
            if let Some(t) = tag {
                let t = t.to_lowercase();
                hits.retain(|e| e.path.file_name().and_then(|s| s.to_str()).is_some_and(|f| f.to_lowercase().contains(&t)));
            }
            if hits.len() > 1 && tag.is_none() {
                if let Some(q) = hits.iter().find(|e| e.name.to_lowercase().contains("q4_k_m")) { hits = vec![*q]; }
            }
            match hits.as_slice() {
                [e] => return Ok((Hub::key_of(&e.path), e.bytes)),
                [] => {}
                many => return Err(format!("{repo} has {} files here; name one as {repo}:<tag> — {}", many.len(),
                                           many.iter().map(|e| e.name.as_str()).collect::<Vec<_>>().join(", "))),
            }
        }
        let mut have: Vec<String> = entries.iter().filter_map(|e| self.card(e).map(|_| e.name.clone())).collect();
        have.truncate(24);
        Err(format!("model '{spec}' not found: it is not a .gguf path, a model in {}, or a downloaded owner/repo. \
                     Download it with `ferric pull owner/repo[:tag]`. On disk: {}",
                    self.dirs.iter().map(|d| d.display().to_string()).collect::<Vec<_>>().join(", "),
                    if have.is_empty() { "nothing".to_string() } else { have.join(", ") }))
    }

    fn load(&mut self, key: &str) -> Result<Engine, String> {
        let name = self.names.get(key).cloned().unwrap_or_else(|| {
            self.scan().into_iter().find(|e| Hub::key_of(&e.path) == key).map(|e| e.name)
                .unwrap_or_else(|| Path::new(key).file_stem().and_then(|s| s.to_str()).unwrap_or("model").to_string())
        });
        let t = std::time::Instant::now();
        let shared = self.shared.clone();
        // `Engine::load_in` refuses a bad file by panicking (fail-closed at startup is its contract); a
        // model loaded for a request must refuse THAT request, not take the server down.
        let eng = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| Engine::load_in(&shared, key, name.clone())))
            .map_err(|p| p.downcast_ref::<String>().cloned().or_else(|| p.downcast_ref::<&str>().map(|s| s.to_string()))
                .unwrap_or_else(|| "the loader panicked".to_string()))?;
        eprintln!("ferric-serve: loaded {name} in {:.1} s ({} layers, context {})", t.elapsed().as_secs_f64(), eng.model.n_layer(), eng.n_ctx);
        self.info.insert(key.to_string(), (name, eng.card.arch.clone(), eng.n_ctx));
        Ok(eng)
    }

    fn available(&mut self) -> Vec<(String, Value)> {
        let entries = self.scan();
        entries.iter().filter_map(|e| self.card(e).map(|c| (e.name.clone(), c.tag_entry()))).collect()
    }

    fn show(&mut self, key: &str) -> Option<Value> {
        let e = self.scan().into_iter().find(|e| Hub::key_of(&e.path) == key)?;
        self.card(&e).map(|c| ollama::show_body(&c))
    }

    fn metrics(&mut self, loaded: &[&str]) -> Option<String> {
        let models: Vec<(String, String, usize)> = loaded.iter().filter_map(|k| self.info.get(*k).cloned()).collect();
        Some(crate::metrics_body(&self.shared.metrics, self.shared.energy.available(), &models))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn split_parts_are_recognised_and_only_the_first_stands_for_the_model() {
        assert_eq!(split_part("Qwen3-235B-Q4_K_M-00001-of-00003"), Some(("Qwen3-235B-Q4_K_M", 1)));
        assert_eq!(split_part("m-00003-of-00003"), Some(("m", 3)));
        assert_eq!(split_part("qwen2.5-0.5b-instruct-q8_0"), None);
        assert_eq!(split_part("x-1-of-3"), None, "llama-gguf-split writes five digits");
    }
}
