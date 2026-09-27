//! Finding, starting and waiting for the server — the process-manager half of the CLI.
//!
//! Ollama's shape: the CLI is a client, and when nothing answers on the configured address it starts
//! a server in the background and waits for it. The server is DETACHED (its own process group, stdin
//! closed, output to `~/.cache/ferric/serve.log`) so it outlives this command and a Ctrl-C typed at
//! the chat prompt reaches only the CLI.
use crate::http::{self, Host};
use crate::hub;
use std::io::{IsTerminal, Write};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

#[derive(Clone, Debug, PartialEq)]
pub enum Kind {
    /// ferric-serve; the version string it reports (`0.12.0-ferric`).
    Ferric(String),
    /// Anything else that speaks the Ollama API — a real Ollama. Model names are passed through
    /// untouched and nothing is pulled locally on its behalf.
    Foreign(String),
}

impl Kind {
    pub fn is_ferric(&self) -> bool { matches!(self, Kind::Ferric(_)) }
    pub fn version(&self) -> &str { match self { Kind::Ferric(v) | Kind::Foreign(v) => v } }
}

pub enum Probe { Up(Kind), Starting, Down }

/// Is a server answering? `/api/version` first — ferric-serve and Ollama both answer it, and its
/// value says which one this is — then `/health` for a ferric-serve without the Ollama routes.
pub fn probe(host: &Host) -> Probe {
    let t = Some(Duration::from_secs(5));
    match http::request(host, "GET", "/api/version", None, t) {
        Err(http::HttpError::Connect(_)) => Probe::Down,
        Err(_) => Probe::Starting,
        Ok(r) if r.status == 503 => Probe::Starting,
        Ok(r) if r.ok() => {
            let v = r.json().ok().and_then(|v| v["version"].as_str().map(str::to_string)).unwrap_or_default();
            if v.contains("ferric") { Probe::Up(Kind::Ferric(v)) } else { Probe::Up(Kind::Foreign(v)) }
        }
        Ok(_) => match http::request(host, "GET", "/health", None, t) {
            Ok(r) if r.ok() => Probe::Up(Kind::Ferric("unknown".into())),
            Ok(r) if r.status == 503 => Probe::Starting,
            _ => Probe::Up(Kind::Foreign("unknown".into())),
        },
    }
}

/// The ferric-serve binary: `$FERRIC_SERVE`, else next to this executable (a cargo build puts both in
/// `target/<profile>/`), else on `PATH`.
pub fn serve_binary() -> Result<PathBuf, String> {
    if let Some(p) = std::env::var_os("FERRIC_SERVE").filter(|p| !p.is_empty()) { return Ok(PathBuf::from(p)); }
    let exe = if cfg!(windows) { "ferric-serve.exe" } else { "ferric-serve" };
    if let Some(dir) = std::env::current_exe().ok().and_then(|e| e.parent().map(Path::to_path_buf)) {
        let p = dir.join(exe);
        if p.is_file() { return Ok(p); }
    }
    for dir in std::env::split_paths(&std::env::var_os("PATH").unwrap_or_default()) {
        let p = dir.join(exe);
        if p.is_file() { return Ok(p); }
    }
    Err("ferric-serve not found next to `ferric` or on PATH (build it: cargo build --release -p ferric-serve, or set FERRIC_SERVE)".into())
}

pub fn log_path() -> PathBuf { hub::ferric_home().join("serve.log") }
pub fn pid_path() -> PathBuf { hub::ferric_home().join("serve.pid") }

fn tail(p: &Path, n: usize) -> String {
    let t = std::fs::read_to_string(p).unwrap_or_default();
    let lines: Vec<&str> = t.lines().collect();
    lines[lines.len().saturating_sub(n)..].join("\n")
}

/// Answer the address, starting a server there if nothing is. `preload` is the model file to start it
/// with (today's ferric-serve serves exactly the model on its command line).
pub fn ensure(host: &Host, preload: Option<&Path>) -> Result<Kind, String> {
    match probe(host) {
        Probe::Up(k) => return Ok(k),
        Probe::Starting => return wait(host, None, None),
        Probe::Down => {}
    }
    if !host.is_local() {
        return Err(format!("no server answering at {} (it is not this machine, so `ferric` will not start one)", host.addr()));
    }
    let bin = serve_binary()?;
    let log = log_path();
    std::fs::create_dir_all(log.parent().unwrap()).map_err(|e| format!("{}: {e}", log.display()))?;
    let mut f = std::fs::OpenOptions::new().create(true).append(true).open(&log).map_err(|e| format!("{}: {e}", log.display()))?;
    let mut args: Vec<String> = preload.map(|p| p.to_string_lossy().to_string()).into_iter().collect();
    args.extend(["--host".to_string(), host.host.clone(), "--port".to_string(), host.port.to_string()]);
    let _ = writeln!(f, "\n==== ferric {}: {} {}", env!("CARGO_PKG_VERSION"), bin.display(), args.join(" "));
    let mut cmd = std::process::Command::new(&bin);
    cmd.args(&args).stdin(std::process::Stdio::null())
        .stdout(f.try_clone().map_err(|e| e.to_string())?).stderr(f);
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        cmd.process_group(0);
    }
    let child = cmd.spawn().map_err(|e| format!("could not start {}: {e}", bin.display()))?;
    let _ = std::fs::write(pid_path(), format!("{}\n", child.id()));
    eprintln!("ferric: started ferric-serve (pid {}) on {} — log: {}", child.id(), host.addr(), log.display());
    wait(host, Some(child), Some(&log))
}

/// Poll until the server answers, the child dies, or `FERRIC_START_TIMEOUT` (default 600 s) passes.
/// Loading a multi-GB model takes minutes on a cold disk, so the default is generous; a dead child is
/// reported at once with the end of its log rather than after the timeout.
fn wait(host: &Host, mut child: Option<std::process::Child>, log: Option<&Path>) -> Result<Kind, String> {
    let limit = std::env::var("FERRIC_START_TIMEOUT").ok().and_then(|s| s.parse::<f64>().ok()).unwrap_or(600.0);
    let t0 = Instant::now();
    let tty = std::io::stderr().is_terminal();
    let log_tail = |n| log.map(|l| format!("\nlast lines of {}:\n{}", l.display(), tail(l, n))).unwrap_or_default();
    loop {
        if let Probe::Up(k) = probe(host) {
            if tty { eprint!("\r\x1b[2K"); }
            return Ok(k);
        }
        if let Some(c) = child.as_mut() {
            if let Ok(Some(st)) = c.try_wait() {
                if tty { eprint!("\r\x1b[2K"); }
                return Err(format!("ferric-serve exited ({st}) before it answered on {}{}", host.addr(), log_tail(20)));
            }
        }
        let el = t0.elapsed().as_secs_f64();
        if el > limit {
            return Err(format!("no answer from {} after {el:.0} s (FERRIC_START_TIMEOUT){}", host.addr(), log_tail(20)));
        }
        if tty {
            let last = log.map(|l| tail(l, 1)).unwrap_or_default();
            eprint!("\r\x1b[2Kloading {el:.0}s  {}", http::clip(last.trim(), 70));
        }
        std::thread::sleep(Duration::from_millis(200));
    }
}
