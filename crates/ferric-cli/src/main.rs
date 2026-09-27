//! **`ferric`** — the command line for Ferric: run, chat with, pull, list, inspect and benchmark
//! local models, and see what each answer cost in joules.
//!
//! Built the way Ollama's CLI is built: the CLI is an HTTP client of a local server (ferric-serve,
//! speaking Ollama's `/api` dialect) and a process manager for it — `ferric run` starts a server in the
//! background when none is answering. Because it speaks that dialect and nothing private, it also
//! drives a real Ollama: `FERRIC_HOST=127.0.0.1:11434 ferric run llama3.2`.
mod bench;
mod chat;
mod http;
mod hub;
mod manage;
mod pull;
mod server;

use http::Host;
use hub::{LocalModel, Target};
use serde_json::{json, Value};
use server::{Kind, Probe};
use std::io::{IsTerminal, Read};

const USAGE: &str = "ferric — run local language models, and see what each answer costs in joules

Usage:
  ferric serve [model...] [ferric-serve flags]   run the server in the foreground
  ferric run <model> [prompt] [flags]            answer one prompt, or chat (no prompt)
  ferric pull <owner/repo[:QUANT|:file.gguf]>    download a GGUF from Hugging Face into the hub
  ferric list | ls                               models in the local hub
  ferric show <model> [--template|--license]     what a model is, and whether Ferric runs it
  ferric ps                                      models the server has loaded
  ferric stop <model>                            unload a model from the server
  ferric rm <model>... [-y]                      delete a model's files from the hub
  ferric bench <model> [flags]                   prefill and decode tok/s, and joules per token

A model is a path to a .gguf, a name from `ferric list`, or a Hugging Face reference
owner/repo[:tag] (also hf.co/owner/repo:Q4_K_M), pulled on first use.

run flags:
  --verbose            timings, tokens/s, and the joules and J/token of each answer
  --hidethinking       do not print a thinking model's reasoning
  --think[=false]      ask a thinking model to think, or not to
  --format json        constrain the answer to JSON
  --keepalive <dur>    how long the server keeps the model loaded (5m, 1h, 0)
  --option <k=v>       a generation parameter, as /set parameter (repeatable)
  --system <text>      a system message
  -i, --interactive    chat even when stdin/stdout are not a terminal (a scripted conversation)

bench flags:
  --reps <n> (3)   --prompt-tokens <n> (512 words)   --gen-tokens <n> (128)
  --gap-ms <ms> (3000)   idle time between runs, where the energy meter takes its idle baseline

Environment:
  FERRIC_HOST          server address (default 127.0.0.1:11435; Ollama's 11434 works too)
  FERRIC_HOME          state directory (default ~/.cache/ferric: hub/, serve.log, serve.pid)
  FERRIC_SERVE         the ferric-serve binary (default: beside `ferric`, then PATH)
  FERRIC_HF_ENDPOINT   Hugging Face endpoint (else HF_ENDPOINT, else https://huggingface.co)
  HF_TOKEN             token for gated repos (else ~/.cache/huggingface/token)
  FERRIC_START_TIMEOUT seconds to wait for a started server to answer (default 600)
";

fn die(msg: impl std::fmt::Display) -> ! {
    eprintln!("Error: {msg}");
    std::process::exit(1)
}

/// The name a request carries for a local model: its absolute path. A path is the one spelling every
/// server form resolves (ferric-serve takes it on its command line; the multi-model server resolves
/// paths, hub names and HF references alike), and it cannot be ambiguous the way a stem can.
pub fn wire_name(m: &LocalModel) -> String { m.path.to_string_lossy().to_string() }

/// Resolve a model reference and make sure a server is answering. For a foreign server (Ollama) the
/// name goes through untouched; for ferric (or none yet) it is resolved locally and pulled if it is an
/// uncached HF reference — then a stopped server is started WITH it.
pub fn open_model(host: &Host, arg: &str, allow_pull: bool) -> Result<(Kind, String, Option<LocalModel>), String> {
    let probed = server::probe(host);
    if let Probe::Up(k @ Kind::Foreign(_)) = probed { return Ok((k, arg.to_string(), None)); }
    let local = match hub::resolve(arg)? {
        Target::Local(m) => Some(m),
        Target::Hf(r) if allow_pull => Some(pull::pull(&r)?),
        Target::Hf(r) => return Err(format!("{} is not in the hub; `ferric pull {arg}` first", r.repo)),
        Target::Name(_) => None,
    };
    if let Some(m) = &local {
        if m.missing > 0 { return Err(format!("{}: {} of {} split parts are missing; pull it again to finish", m.name, m.missing, m.parts.len())); }
    }
    let kind = match probed {
        Probe::Up(k) => k,
        _ => match &local {
            Some(m) => server::ensure(host, Some(&m.path))?,
            None => return Err(format!("'{arg}' is not a file, a model in {}, or a Hugging Face owner/repo[:tag]", hub::hub_dir().display())),
        },
    };
    let wire = local.as_ref().map(wire_name).unwrap_or_else(|| arg.to_string());
    Ok((kind, wire, local))
}

/// Ask the server to load the model before the first message (Ollama's CLI does the same), so a
/// missing model is an error before the user types, and loading time is not charged to the answer.
pub fn preload(host: &Host, kind: &Kind, model: &str, keep_alive: Option<&Value>) -> Result<(), String> {
    let mut body = json!({"model": model});
    if let Some(k) = keep_alive { body["keep_alive"] = k.clone(); }
    let tty = std::io::stderr().is_terminal();
    if tty { eprint!("loading {} …", http::clip(model.rsplit('/').next().unwrap_or(model), 60)); }
    let r = http::json(host, "POST", "/api/generate", Some(&body));
    if tty { eprint!("\r\x1b[2K"); }
    r.map_err(|e| {
        if kind.is_ferric() { e } else { format!("{e} (on Ollama: `ollama pull {model}`)") }
    })?;
    if kind.is_ferric() { loaded_check(host, model); }
    Ok(())
}

/// ⚠ A ferric-serve from before multi-model serving answers EVERY request with the one model it was
/// started with, whatever `model` says. So after the load, look at what the server says it holds,
/// and say so when it is not what was asked for — the alternative is an answer from the wrong model
/// with nothing on screen to show it.
fn loaded_check(host: &Host, model: &str) {
    let Ok(v) = http::json(host, "GET", "/api/ps", None) else { return };
    let norm = |s: &str| {
        let s = s.strip_suffix(":latest").unwrap_or(s);
        let base = s.rsplit('/').next().unwrap_or(s);
        let base = base.strip_suffix(".gguf").unwrap_or(base);
        hub::split_parts(&format!("{base}.gguf")).map(|(p, _, _)| p.to_string()).unwrap_or_else(|| base.to_string()).to_ascii_lowercase()
    };
    let want = norm(model);
    let names: Vec<String> = v["models"].as_array().into_iter().flatten()
        .flat_map(|m| [m["name"].as_str(), m["model"].as_str()]).flatten().map(str::to_string).collect();
    if !names.is_empty() && !names.iter().any(|n| norm(n) == want) {
        eprintln!("warning: the server at {} lists {} as loaded, not '{want}' — a single-model ferric-serve answers with the model it was started with. Stop it (pid in {}) or set FERRIC_HOST to another port.",
                  host.addr(), names.first().map(|n| format!("'{n}'")).unwrap_or_default(), server::pid_path().display());
    }
}

fn keep_alive_value(s: &str) -> Value {
    s.parse::<i64>().map(|n| json!(n)).unwrap_or_else(|_| json!(s))
}

fn cmd_run(host: &Host, args: &[String]) -> Result<(), String> {
    let mut o = chat::ChatOpts::default();
    let (mut positional, mut interactive) = (Vec::new(), false);
    let mut i = 0;
    let val = |i: usize, f: &str| args.get(i + 1).cloned().ok_or_else(|| format!("{f} needs a value"));
    while i < args.len() {
        let a = args[i].as_str();
        match a {
            "--verbose" | "-v" => o.verbose = true,
            "--hidethinking" => o.hide_thinking = true,
            "--think" | "--think=true" => o.think = Some(true),
            "--think=false" | "--nothink" => o.think = Some(false),
            "-i" | "--interactive" => interactive = true,
            "--format" => { let f = val(i, a)?; o.format = Some(if f == "json" { json!("json") } else { serde_json::from_str(&f).map_err(|e| format!("--format: {e}"))? }); i += 1; }
            "--keepalive" => { o.keep_alive = Some(keep_alive_value(&val(i, a)?)); i += 1; }
            "--system" => { o.system = Some(val(i, a)?); i += 1; }
            "--option" | "-o" => {
                let kv = val(i, a)?;
                let (k, v) = kv.split_once('=').ok_or_else(|| format!("--option {kv}: expected name=value"))?;
                let vals: Vec<&str> = if k == "stop" { v.split(',').collect() } else { vec![v] };
                o.options.insert(k.to_string(), chat::parse_param(k, &vals)?);
                i += 1;
            }
            "-h" | "--help" => { print!("{USAGE}"); return Ok(()); }
            _ if a.starts_with("--") => return Err(format!("unknown flag {a} (see `ferric --help`)")),
            _ => positional.push(args[i].clone()),
        }
        i += 1;
    }
    let Some(arg) = positional.first().cloned() else { return Err("usage: ferric run <model> [prompt]".into()) };
    let mut prompt = positional[1..].join(" ");
    let stdin_tty = std::io::stdin().is_terminal();
    // Ollama's rule: a prompt on the command line, or input that is not a terminal, means one answer.
    // `cat notes.txt | ferric run m "summarise"` puts stdin first, then the words.
    if !interactive && !stdin_tty {
        let mut s = String::new();
        std::io::stdin().read_to_string(&mut s).map_err(|e| e.to_string())?;
        if !s.trim().is_empty() { prompt = if prompt.is_empty() { s } else { format!("{s} {prompt}") }; }
    }
    let (kind, model, local) = open_model(host, &arg, true)?;
    for k in o.options.keys() { if let Some(n) = chat::param_note(&kind, k) { eprintln!("{n}"); } }
    preload(host, &kind, &model, o.keep_alive.as_ref())?;
    let one_shot = !interactive && (!prompt.is_empty() || !stdin_tty || !std::io::stdout().is_terminal());
    if one_shot {
        if prompt.trim().is_empty() { return Err("no prompt: give one after the model name, or pipe it on stdin".into()); }
        let turn = chat::chat_turn(host, &model, &o, &[], &prompt, &mut std::io::stdout())?;
        if turn.cancelled { std::process::exit(130); }
        if o.verbose { eprint!("{}", chat::stats(&turn.done)); }
        return Ok(());
    }
    let mut s = chat::Session { host: host.clone(), kind, model, local, arg };
    chat::repl(&mut s, &mut o, &mut std::io::stdin().lock())
}

fn cmd_serve(host: &Host, args: &[String]) -> Result<(), String> {
    if args.iter().any(|a| a == "-h" || a == "--help") {
        println!("usage: ferric serve [model...] [ferric-serve flags]\n\nRuns ferric-serve in the foreground on FERRIC_HOST ({}) unless --host/--port are given.\nModels may be paths, hub names or owner/repo[:tag] (pulled first).", host.addr());
        return Ok(());
    }
    let bin = server::serve_binary()?;
    let mut out: Vec<String> = Vec::new();
    let mut i = 0;
    // Models lead; everything from the first flag on belongs to ferric-serve, verbatim.
    while i < args.len() && !args[i].starts_with('-') {
        let m = match hub::resolve(&args[i])? {
            Target::Local(m) => m.path.to_string_lossy().to_string(),
            Target::Hf(r) => pull::pull(&r)?.path.to_string_lossy().to_string(),
            Target::Name(n) => n,
        };
        out.push(m);
        i += 1;
    }
    let rest = &args[i..];
    out.extend(rest.iter().cloned());
    if !rest.iter().any(|a| a == "--host") { out.extend(["--host".into(), host.host.clone()]); }
    if !rest.iter().any(|a| a == "--port") { out.extend(["--port".into(), host.port.to_string()]); }
    let mut cmd = std::process::Command::new(&bin);
    cmd.args(&out);
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        let e = cmd.exec();
        Err(format!("could not run {}: {e}", bin.display()))
    }
    #[cfg(not(unix))]
    {
        let st = cmd.status().map_err(|e| format!("could not run {}: {e}", bin.display()))?;
        std::process::exit(st.code().unwrap_or(1));
    }
}

fn cmd_bench(host: &Host, args: &[String]) -> Result<(), String> {
    let mut o = bench::BenchOpts { reps: 3, prompt_tokens: 512, gen_tokens: 128, gap: std::time::Duration::from_millis(3000) };
    let mut model = None;
    let mut i = 0;
    let num = |i: usize, f: &str| args.get(i + 1).and_then(|s| s.parse::<usize>().ok()).filter(|&n| n > 0).ok_or_else(|| format!("{f} needs a positive number"));
    while i < args.len() {
        match args[i].as_str() {
            "--reps" => { o.reps = num(i, "--reps")?; i += 1; }
            "--prompt-tokens" => { o.prompt_tokens = num(i, "--prompt-tokens")?; i += 1; }
            "--gen-tokens" => { o.gen_tokens = num(i, "--gen-tokens")?; i += 1; }
            // The idle time between runs is where ferric-serve's meter takes its idle baseline: it skips
            // 0.3 s before and 1 s after every request, so a 1.5 s gap left 0.2 s (two 100 ms samples)
            // of baseline per gap. 3 s leaves 1.7 s.
            "--gap-ms" => { o.gap = std::time::Duration::from_millis(args.get(i + 1).and_then(|s| s.parse().ok()).ok_or("--gap-ms needs a number")?); i += 1; }
            a if a.starts_with('-') => return Err(format!("unknown flag {a}")),
            a => model = Some(a.to_string()),
        }
        i += 1;
    }
    let arg = model.ok_or("usage: ferric bench <model> [--reps N] [--prompt-tokens N] [--gen-tokens N]")?;
    let (kind, wire, local) = open_model(host, &arg, true)?;
    preload(host, &kind, &wire, None)?;
    let label = match local.as_ref().map(|m| (m, hub::summarize(&m.path))) {
        Some((m, Ok(s))) => format!("{} ({} {} {}, {})", m.name, s.arch, hub::human_params(s.params), s.quant, hub::human_size(m.size)),
        Some((m, Err(_))) => m.name.clone(),
        None => arg.clone(),
    };
    bench::bench(host, &wire, &label, &o)
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let Some(cmd) = args.first().map(String::as_str) else { print!("{USAGE}"); std::process::exit(2) };
    let rest = &args[1..];
    let host = Host::from_env().unwrap_or_else(|e| die(e));
    let need = |n: usize, u: &str| if rest.iter().filter(|a| !a.starts_with('-')).count() < n { die(format!("usage: {u}")) };
    let r = match cmd {
        "-h" | "--help" | "help" => { print!("{USAGE}"); Ok(()) }
        "-V" | "--version" | "version" => { println!("ferric {}", env!("CARGO_PKG_VERSION")); Ok(()) }
        "serve" => cmd_serve(&host, rest),
        "run" => cmd_run(&host, rest),
        "pull" => {
            need(1, "ferric pull <owner/repo[:QUANT|:file.gguf]>");
            let a = &rest[0];
            match pull::HfRef::parse(a) {
                Some(r) => pull::pull(&r).map(|_| ()),
                None => Err(format!("'{a}' is not a Hugging Face reference (owner/repo[:QUANT|:file.gguf])")),
            }
        }
        "list" | "ls" => manage::list(),
        "show" => {
            need(1, "ferric show <model> [--template|--license]");
            let what = if rest.iter().any(|a| a == "--template") { "template" } else if rest.iter().any(|a| a == "--license") { "license" } else { "info" };
            let m = rest.iter().find(|a| !a.starts_with('-')).unwrap();
            manage::show(&host, m, None, what)
        }
        "ps" => manage::ps(&host),
        "stop" => { need(1, "ferric stop <model>"); manage::stop(&host, &rest[0]) }
        "rm" => {
            need(1, "ferric rm <model>... [-y]");
            let yes = rest.iter().any(|a| a == "-y" || a == "--yes" || a == "-f" || a == "--force");
            let models: Vec<String> = rest.iter().filter(|a| !a.starts_with('-')).cloned().collect();
            manage::rm(&host, &models, yes)
        }
        "bench" => cmd_bench(&host, rest),
        other => Err(format!("unknown command '{other}' (see `ferric --help`)")),
    };
    if let Err(e) = r { die(e) }
}
