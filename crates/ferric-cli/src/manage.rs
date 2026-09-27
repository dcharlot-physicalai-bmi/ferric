//! `list`, `show`, `ps`, `stop`, `rm` — the model store and the server's loaded set.
use crate::http::{self, Host};
use crate::hub::{self, LocalModel};
use crate::server::{self, Probe};
use serde_json::json;
use std::io::{BufRead, Write};

fn table(rows: &[Vec<String>]) -> String {
    let n = rows.iter().map(Vec::len).max().unwrap_or(0);
    let w: Vec<usize> = (0..n).map(|i| rows.iter().filter_map(|r| r.get(i)).map(|c| c.chars().count()).max().unwrap_or(0)).collect();
    let mut s = String::new();
    for r in rows {
        let line: Vec<String> = r.iter().enumerate().map(|(i, c)| if i + 1 == r.len() { c.clone() } else { format!("{c:<w$}", w = w[i]) }).collect();
        s.push_str(line.join("    ").trim_end());
        s.push('\n');
    }
    s
}

pub fn list() -> Result<(), String> {
    let hub = hub::hub_dir();
    let mut models = hub::scan(&hub);
    models.sort_by(|a, b| b.modified.cmp(&a.modified).then(a.name.cmp(&b.name)));
    let names = hub::display_names(&models);
    let mut rows = vec![["NAME", "ARCH", "PARAMS", "QUANT", "SIZE", "MODIFIED"].map(String::from).to_vec()];
    let mut notes = Vec::new();
    for (m, name) in models.iter().zip(&names) {
        let (arch, params, quant) = if m.missing > 0 {
            notes.push(format!("{name}: {} of {} parts missing — `ferric pull` it again to finish", m.missing, m.parts.len()));
            ("?".into(), "?".into(), "?".into())
        } else {
            match hub::summarize(&m.path) {
                Ok(s) => (s.arch, hub::human_params(s.params), s.quant),
                Err(e) => { notes.push(format!("{name}: {}", http::clip(&e, 160))); ("?".into(), "?".into(), "?".into()) }
            }
        };
        rows.push(vec![name.clone(), arch, params, quant, hub::human_size(m.size), hub::ago(m.modified)]);
    }
    print!("{}", table(&rows));
    for n in notes { eprintln!("note: {n}"); }
    if models.is_empty() { eprintln!("no models in {} — `ferric pull owner/repo:Q4_K_M` fetches one", hub.display()); }
    Ok(())
}

fn registry_lines(arch: &str) -> Vec<(String, String)> {
    use ferric_llama::arch;
    match arch::lookup(arch) {
        None => vec![("status".into(), format!("unsupported — ferric-serve refuses '{arch}' rather than load it down a similar path (a near-miss architecture generates fluent, wrong text)"))],
        Some(a) => {
            let verdict = match a.status {
                arch::Status::Verified => "verified — output compared against the reference implementation on real weights",
                arch::Status::Loads => "loads — generates, but not yet diffed against the reference implementation",
                arch::Status::Parts => "parts — components exist, no loader wires them: ferric-serve refuses it",
                arch::Status::Untried => "untried — runs only on a synthetic checkpoint: ferric-serve refuses it",
            };
            vec![("runtime".into(), a.runtime.label().into()), ("status".into(), verdict.into()), ("note".into(), a.note.split_whitespace().collect::<Vec<_>>().join(" "))]
        }
    }
}

/// Word-wrap `v` to lines of at most `w` columns (the registry notes run to a paragraph).
fn wrap(v: &str, w: usize) -> Vec<String> {
    let mut lines = vec![String::new()];
    for word in v.split_whitespace() {
        let cur = lines.last_mut().unwrap();
        if !cur.is_empty() && cur.chars().count() + 1 + word.chars().count() > w { lines.push(word.to_string()); }
        else { if !cur.is_empty() { cur.push(' '); } cur.push_str(word); }
    }
    lines
}

fn section(title: &str, rows: &[(String, String)]) {
    println!("  {title}");
    for (k, v) in rows {
        for (i, l) in wrap(v, 96).iter().enumerate() {
            if i == 0 { println!("    {k:<20}{l}"); } else { println!("    {:<20}{l}", ""); }
        }
    }
    println!();
}

/// `show` for a local file (read from its header), else ask the server (`/api/show`). `what` is
/// `info`, `template` or `license`.
pub fn show(host: &Host, arg: &str, local: Option<&LocalModel>, what: &str) -> Result<(), String> {
    let resolved;
    let mut pullable = None;
    let local = match local {
        Some(m) => Some(m),
        None => match hub::resolve(arg)? {
            hub::Target::Local(m) => { resolved = m; Some(&resolved) }
            hub::Target::Hf(r) => { pullable = Some(r); None }
            hub::Target::Name(_) => None,
        },
    };
    if let Some(m) = local {
        let s = hub::summarize(&m.path)?;
        match what {
            "template" => { println!("{}", s.template.as_deref().unwrap_or("(this GGUF carries no chat template)")); return Ok(()); }
            "license" => { println!("{}", s.license.as_deref().unwrap_or("(no general.license in this GGUF)")); return Ok(()); }
            _ => {}
        }
        let mut rows = vec![("architecture".to_string(), s.arch.clone()), ("parameters".into(), hub::human_params(s.params))];
        if let Some(n) = &s.general_name { rows.insert(0, ("name".into(), n.clone())); }
        if let Some(c) = s.context { rows.push(("context length".into(), c.to_string())); }
        if let Some(e) = s.embedding { rows.push(("embedding length".into(), e.to_string())); }
        if let Some(l) = s.layers { rows.push(("layers".into(), l.to_string())); }
        rows.push(("quantization".into(), s.quant.clone()));
        rows.push(("file".into(), m.path.display().to_string()));
        rows.push(("size".into(), format!("{}{}", hub::human_size(m.size), if m.parts.len() > 1 { format!(" in {} parts", m.parts.len()) } else { String::new() })));
        section("Model", &rows);
        section("Ferric", &registry_lines(&s.arch));
        let t = match &s.template { Some(t) => format!("present (Jinja, {} chars) — `ferric show {arg} --template` prints it", t.chars().count()), None => "absent — ferric-serve falls back to the vocabulary-family template".into() };
        section("Template", &[("chat template".into(), t)]);
        if let Some(l) = &s.license { section("License", &[("license".into(), l.lines().next().unwrap_or("").to_string())]); }
        return Ok(());
    }
    // Not a local file: the server may know it (an Ollama model, say).
    let kind = match server::probe(host) {
        Probe::Up(k) => k,
        _ if pullable.is_some() => return Err(format!("'{arg}' is not in the hub yet: `ferric pull {arg}`")),
        _ => return Err(format!("'{arg}' is not a local model and no server is answering at {} to ask", host.addr())),
    };
    if kind.is_ferric() && pullable.is_some() { return Err(format!("'{arg}' is not in the hub yet: `ferric pull {arg}`")); }
    let v = http::json(host, "POST", "/api/show", Some(&json!({"model": arg})))?;
    match what {
        "template" => { println!("{}", v["template"].as_str().unwrap_or("")); return Ok(()); }
        "license" => { println!("{}", v["license"].as_str().unwrap_or("(none reported)")); return Ok(()); }
        _ => {}
    }
    let d = &v["details"];
    let arch = v["model_info"]["general.architecture"].as_str().or_else(|| d["family"].as_str()).unwrap_or("").to_string();
    let mut rows = vec![("architecture".to_string(), arch.clone())];
    if let Some(p) = d["parameter_size"].as_str() { rows.push(("parameters".into(), p.into())); }
    if let Some(c) = v["model_info"][format!("{arch}.context_length")].as_u64() { rows.push(("context length".into(), c.to_string())); }
    if let Some(q) = d["quantization_level"].as_str() { rows.push(("quantization".into(), q.into())); }
    section("Model", &rows);
    if let Some(c) = v["capabilities"].as_array() {
        section("Capabilities", &c.iter().filter_map(|x| x.as_str()).map(|x| (x.to_string(), String::new())).collect::<Vec<_>>());
    }
    section("Ferric", &registry_lines(&arch));
    Ok(())
}

pub fn ps(host: &Host) -> Result<(), String> {
    let Probe::Up(_) = server::probe(host) else { return Err(format!("no server answering at {}", host.addr())) };
    let v = http::json(host, "GET", "/api/ps", None)?;
    let now = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs() as i64).unwrap_or(0);
    let mut rows = vec![["NAME", "SIZE", "CONTEXT", "UNTIL"].map(String::from).to_vec()];
    for m in v["models"].as_array().into_iter().flatten() {
        let name = m["name"].as_str().or_else(|| m["model"].as_str()).unwrap_or("?").to_string();
        let size = m["size"].as_u64().map(hub::human_size).unwrap_or_default();
        let ctx = m["context_length"].as_u64().map(|c| c.to_string()).unwrap_or_default();
        let until = match m["expires_at"].as_str().and_then(hub::parse_rfc3339) {
            // ferric-serve reports 2318-01-01 for "resident until the server stops"; Ollama uses a
            // far-future stamp for keep_alive -1. Either way it is not a countdown.
            Some(t) if t - now > 100 * 365 * 86_400 => "Forever".to_string(),
            Some(t) if t <= now => "Stopping...".to_string(),
            Some(t) => {
                let s = t - now;
                if s < 60 { format!("{s} seconds from now") } else if s < 3600 { format!("{} minutes from now", s / 60) } else { format!("{} hours from now", s / 3600) }
            }
            None => String::new(),
        };
        rows.push(vec![name, size, ctx, until]);
    }
    print!("{}", table(&rows));
    Ok(())
}

/// Ollama's documented unload: `/api/generate` with no prompt and `keep_alive: 0`. The answer says
/// whether it happened (`done_reason: "unload"`); a server that only loads says so, and this reports it
/// rather than claiming success.
pub fn stop(host: &Host, arg: &str) -> Result<(), String> {
    let kind = match server::probe(host) { Probe::Up(k) => k, _ => return Err(format!("no server answering at {}", host.addr())) };
    let model = if kind.is_ferric() {
        match hub::resolve(arg)? { hub::Target::Local(m) => crate::wire_name(&m), _ => arg.to_string() }
    } else { arg.to_string() };
    let v = http::json(host, "POST", "/api/generate", Some(&json!({"model": model, "keep_alive": 0})))?;
    match v["done_reason"].as_str() {
        Some("unload") => Ok(()),
        other => Err(format!("the server did not confirm the unload (done_reason: {}); ferric-serve before multi-model serving cannot unload — stop the process instead (pid in {})",
                             other.unwrap_or("none"), server::pid_path().display())),
    }
}

pub fn rm(host: &Host, args: &[String], yes: bool) -> Result<(), String> {
    let hub = hub::hub_dir();
    for arg in args {
        let m = match hub::resolve(arg)? {
            hub::Target::Local(m) => m,
            _ => return Err(format!("'{arg}' is not a model in {}", hub.display())),
        };
        if !m.path.starts_with(hub::absolute(&hub)) {
            return Err(format!("{} is outside the hub ({}); `ferric rm` only deletes what `ferric pull` put there", m.path.display(), hub.display()));
        }
        let files: Vec<_> = m.parts.iter().flat_map(|p| [p.clone(), std::path::PathBuf::from(format!("{}.partial", p.display()))])
            .filter(|p| p.exists()).collect();
        if !yes {
            eprint!("delete '{arg}' ({} file{}, {})? [y/N] ", files.len(), if files.len() == 1 { "" } else { "s" }, hub::human_size(m.size));
            let _ = std::io::stderr().flush();
            let mut a = String::new();
            std::io::stdin().lock().read_line(&mut a).map_err(|e| e.to_string())?;
            if !matches!(a.trim().to_ascii_lowercase().as_str(), "y" | "yes") { eprintln!("kept '{arg}'"); continue; }
        }
        // Unload first if a ferric server holds it, as `ollama rm` does; a failure here is not a reason to keep the files.
        if let Probe::Up(k) = server::probe(host) {
            if k.is_ferric() { let _ = http::post(host, "/api/generate", &json!({"model": crate::wire_name(&m), "keep_alive": 0})).map(|r| r.text()); }
        }
        for f in &files { std::fs::remove_file(f).map_err(|e| format!("{}: {e}", f.display()))?; }
        if let Some(dir) = m.path.parent() {
            if dir != hub::absolute(&hub) && std::fs::read_dir(dir).is_ok_and(|mut d| d.next().is_none()) { let _ = std::fs::remove_dir(dir); }
        }
        println!("deleted '{arg}'");
    }
    Ok(())
}
