//! `ferric run` — one answer, or a conversation, streamed from `/api/chat`.
//!
//! The conversation lives HERE, not in the server: every turn sends the whole history, exactly as
//! `ollama run` does, so a server that forgets (or is restarted) between turns loses nothing, and
//! ferric-serve's prompt cache makes the resend cheap (a follow-up turn costs 96% fewer joules, per
//! its own measurement in 0f0605d).
use crate::http::{self, Host};
use crate::server::Kind;
use serde_json::{json, Map, Value};
use std::io::{BufRead, BufReader, IsTerminal, Write};
use std::time::Duration;

/// Ctrl-C while an answer streams cancels THAT answer and returns to the prompt; at the prompt it
/// exits as usual. So the handler is installed only for the length of a stream.
mod sig {
    use std::sync::atomic::{AtomicBool, Ordering};
    static HIT: AtomicBool = AtomicBool::new(false);
    pub fn hit() -> bool { HIT.load(Ordering::SeqCst) }
    pub struct Guard(#[allow(dead_code)] usize);
    #[cfg(unix)]
    extern "C" fn on_int(_: libc::c_int) { HIT.store(true, Ordering::SeqCst); }
    impl Guard {
        pub fn install() -> Guard {
            HIT.store(false, Ordering::SeqCst);
            #[cfg(unix)]
            unsafe {
                let h: extern "C" fn(libc::c_int) = on_int;
                return Guard(libc::signal(libc::SIGINT, h as libc::sighandler_t));
            }
            #[allow(unreachable_code)]
            Guard(0)
        }
    }
    impl Drop for Guard {
        fn drop(&mut self) {
            #[cfg(unix)]
            unsafe { libc::signal(libc::SIGINT, self.0 as libc::sighandler_t); }
        }
    }
}

/// Options that decide what a request carries. `options` is Ollama's generation-parameter map.
#[derive(Clone, Debug, Default)]
pub struct ChatOpts {
    pub verbose: bool,
    pub hide_thinking: bool,
    pub think: Option<bool>,
    pub format: Option<Value>,
    pub keep_alive: Option<Value>,
    pub options: Map<String, Value>,
    pub system: Option<String>,
}

/// The parameters ferric-serve's Ollama routes read (`ollama::to_openai`). Anything else is still
/// sent — a real Ollama reads it — but the user is told ferric-serve will not.
const FERRIC_READS: &[&str] = &["num_predict", "temperature", "top_p", "top_k", "min_p", "seed", "stop",
                                "repeat_penalty", "repeat_last_n", "presence_penalty", "frequency_penalty"];

/// `/set parameter <name> <value...>` and `--option name=value`: Ollama's names and types.
pub fn parse_param(name: &str, vals: &[&str]) -> Result<Value, String> {
    const INTS: &[&str] = &["num_ctx", "num_predict", "num_keep", "seed", "top_k", "repeat_last_n", "num_batch",
                            "num_gpu", "main_gpu", "num_thread", "mirostat"];
    const FLOATS: &[&str] = &["temperature", "top_p", "min_p", "typical_p", "repeat_penalty", "presence_penalty",
                              "frequency_penalty", "mirostat_eta", "mirostat_tau"];
    const BOOLS: &[&str] = &["use_mmap", "penalize_newline"];
    if vals.is_empty() { return Err(format!("no value given for '{name}'")); }
    if name == "stop" { return Ok(json!(vals)); }
    let v = vals.join(" ");
    if INTS.contains(&name) {
        return v.parse::<i64>().map(|x| json!(x)).map_err(|_| format!("'{name}' takes a whole number, not '{v}'"));
    }
    if FLOATS.contains(&name) {
        return v.parse::<f64>().ok().filter(|x| x.is_finite()).map(|x| json!(x)).ok_or_else(|| format!("'{name}' takes a number, not '{v}'"));
    }
    if BOOLS.contains(&name) {
        return v.parse::<bool>().map(|x| json!(x)).map_err(|_| format!("'{name}' takes true or false, not '{v}'"));
    }
    Err(format!("unknown parameter '{name}' (known: stop, {}, {}, {})", INTS.join(", "), FLOATS.join(", "), BOOLS.join(", ")))
}

pub fn param_note(kind: &Kind, name: &str) -> Option<String> {
    (kind.is_ferric() && !FERRIC_READS.contains(&name))
        .then(|| format!("note: ferric-serve does not read '{name}'; it is sent for servers that do"))
}

/// Writes a streamed answer: reasoning (dimmed on a terminal, bracketed otherwise), then the answer.
struct Render { tty: bool, hide_thinking: bool, thinking_open: bool, any: bool, last_nl: bool }

impl Render {
    fn new(hide_thinking: bool) -> Render {
        Render { tty: std::io::stdout().is_terminal(), hide_thinking, thinking_open: false, any: false, last_nl: true }
    }
    fn put(&mut self, out: &mut impl Write, s: &str) {
        if s.is_empty() { return; }
        let _ = out.write_all(s.as_bytes());
        let _ = out.flush();
        self.any = true;
        self.last_nl = s.ends_with('\n');
    }
    fn thinking(&mut self, out: &mut impl Write, s: &str) {
        if self.hide_thinking || s.is_empty() { return; }
        if !self.thinking_open {
            self.thinking_open = true;
            self.put(out, if self.tty { "\x1b[2mThinking...\n" } else { "Thinking...\n" });
        }
        self.put(out, s);
    }
    fn close_thinking(&mut self, out: &mut impl Write) {
        if self.thinking_open {
            self.thinking_open = false;
            let nl = if self.last_nl { "" } else { "\n" };
            self.put(out, &format!("{nl}...done thinking.\n\n{}", if self.tty { "\x1b[0m" } else { "" }));
        }
    }
    fn content(&mut self, out: &mut impl Write, s: &str) {
        if s.is_empty() { return; }
        self.close_thinking(out);
        self.put(out, s);
    }
    fn finish(&mut self, out: &mut impl Write) {
        self.close_thinking(out);
        if self.any && !self.last_nl { self.put(out, "\n"); }
    }
}

pub struct Turn { pub content: String, pub done: Value, pub cancelled: bool }

/// Stream an NDJSON response line by line, on a reader thread so a Ctrl-C can cancel between lines
/// (the socket is shut down, which is how the server learns to stop generating). `Ok(true)` = cancelled.
pub fn stream_lines(resp: http::Response, mut on_line: impl FnMut(&Value) -> Result<(), String>) -> Result<bool, String> {
    let sock = resp.socket;
    let body = resp.body;
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        for l in BufReader::new(body).lines() {
            let stop = l.is_err();
            if tx.send(l).is_err() || stop { return; }
        }
    });
    loop {
        match rx.recv_timeout(Duration::from_millis(100)) {
            Ok(Ok(line)) => {
                if line.trim().is_empty() { continue; }
                let v: Value = serde_json::from_str(&line).map_err(|e| format!("server sent a line that is not JSON ({e}): {}", http::clip(&line, 200)))?;
                on_line(&v)?;
            }
            Ok(Err(e)) => return Err(format!("the stream broke: {e}")),
            Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
                if sig::hit() { let _ = sock.shutdown(std::net::Shutdown::Both); return Ok(true); }
            }
            Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => return Ok(false),
        }
    }
}

/// One `/api/chat` exchange: the whole history plus `user`, streamed to `out`.
pub fn chat_turn(host: &Host, model: &str, o: &ChatOpts, history: &[Value], user: &str, out: &mut impl Write) -> Result<Turn, String> {
    let mut messages = Vec::new();
    if let Some(s) = &o.system { messages.push(json!({"role": "system", "content": s})); }
    messages.extend(history.iter().cloned());
    messages.push(json!({"role": "user", "content": user}));
    let mut body = json!({"model": model, "messages": messages, "stream": true});
    if !o.options.is_empty() { body["options"] = Value::Object(o.options.clone()); }
    if let Some(t) = o.think { body["think"] = json!(t); }
    if let Some(f) = &o.format { body["format"] = f.clone(); }
    if let Some(k) = &o.keep_alive { body["keep_alive"] = k.clone(); }
    let _guard = sig::Guard::install();
    let resp = http::post(host, "/api/chat", &body).map_err(|e| format!("POST /api/chat on {}: {e}", host.addr()))?;
    if !resp.ok() { return Err(resp.error_text()); }
    let mut r = Render::new(o.hide_thinking);
    let (mut content, mut done) = (String::new(), Value::Null);
    let cancelled = stream_lines(resp, |v| {
        if let Some(e) = v.get("error") { return Err(e.as_str().map(str::to_string).unwrap_or_else(|| e.to_string())); }
        let m = &v["message"];
        if let Some(t) = m["thinking"].as_str() { r.thinking(out, t); }
        if let Some(c) = m["content"].as_str() { r.content(out, c); content.push_str(c); }
        if v["done"].as_bool() == Some(true) { done = v.clone(); }
        Ok(())
    });
    r.finish(out);
    let cancelled = cancelled?;
    if cancelled { let _ = writeln!(out); }
    // Both servers end a stream with a `done: true` line carrying the counts. A stream that stops
    // without it is a server that died mid-answer: the text above is cut short, and must not be kept
    // as the model's reply or reported with stats it never sent.
    if !cancelled && done.is_null() {
        return Err("the stream ended before the server's final message; the answer above is incomplete".into());
    }
    Ok(Turn { content, done, cancelled })
}

fn dur(ns: f64) -> String {
    let s = ns / 1e9;
    if s >= 1.0 { format!("{s:.3}s") } else if s >= 1e-3 { format!("{:.3}ms", s * 1e3) } else { format!("{:.1}µs", s * 1e6) }
}

/// Ollama's `--verbose` block, plus what Ferric adds: the joules this answer cost, and per token.
pub fn stats(done: &Value) -> String {
    let f = |k: &str| done[k].as_f64();
    let mut s = String::new();
    let mut line = |k: &str, v: String| s.push_str(&format!("{:<22}{v}\n", format!("{k}:")));
    if let Some(v) = f("total_duration") { line("total duration", dur(v)); }
    if let Some(v) = f("load_duration") { line("load duration", dur(v)); }
    if let Some(n) = f("prompt_eval_count") {
        line("prompt eval count", format!("{n} token(s)"));
        if let Some(d) = f("prompt_eval_duration") {
            line("prompt eval duration", dur(d));
            if d > 0.0 { line("prompt eval rate", format!("{:.2} tokens/s", n / (d / 1e9))); }
        }
    }
    if let Some(n) = f("eval_count") {
        line("eval count", format!("{n} token(s)"));
        if let Some(d) = f("eval_duration") {
            line("eval duration", dur(d));
            if d > 0.0 { line("eval rate", format!("{:.2} tokens/s", n / (d / 1e9))); }
        }
    }
    match done.get("energy") {
        None | Some(Value::Null) => line("energy", "not reported by this server".into()),
        Some(e) => match e["joules"].as_f64() {
            Some(j) => {
                let how = [e["boundary"].as_str(), e["class"].as_str()].into_iter().flatten().collect::<Vec<_>>().join(", ");
                line("energy", format!("{j:.3} J{}", if how.is_empty() { String::new() } else { format!(" ({how})") }));
                let n = f("eval_count").unwrap_or(0.0);
                let jpt = e["joules_per_token"].as_f64().or_else(|| (n > 0.0).then(|| j / n));
                if let Some(x) = jpt { line("energy per token", format!("{x:.4} J/token (the whole request over {n} generated)")); }
            }
            None => {
                let why = e["why"].as_str().unwrap_or("the server gave no reason");
                let gross = e["window_joules"].as_f64().map(|w| format!("; gross over the window {w:.3} J in {:.3} s", e["seconds"].as_f64().unwrap_or(0.0))).unwrap_or_default();
                line("energy", format!("unmeasured — {why}{gross}"));
            }
        },
    }
    s
}

/// Read one user message: a line, or a `"""` block spanning lines. `None` at end of input.
fn read_message(input: &mut impl BufRead, prompt: bool) -> Option<String> {
    let mut line = String::new();
    if prompt { print!(">>> "); let _ = std::io::stdout().flush(); }
    if input.read_line(&mut line).ok()? == 0 { return None; }
    let l = line.trim_end_matches(['\n', '\r']);
    let Some(first) = l.trim_start().strip_prefix("\"\"\"") else { return Some(l.to_string()) };
    if let Some(one) = first.strip_suffix("\"\"\"") { return Some(one.to_string()); }
    let mut parts = vec![first.to_string()];
    loop {
        if prompt { print!("... "); let _ = std::io::stdout().flush(); }
        let mut next = String::new();
        if input.read_line(&mut next).ok()? == 0 { break; }
        let n = next.trim_end_matches(['\n', '\r']);
        if let Some(end) = n.trim_end().strip_suffix("\"\"\"") { parts.push(end.to_string()); break; }
        parts.push(n.to_string());
    }
    // `"""` alone on its opening or closing line delimits; it does not add an empty line.
    if parts.first().is_some_and(|p| p.is_empty()) { parts.remove(0); }
    if parts.last().is_some_and(|p| p.is_empty()) { parts.pop(); }
    Some(parts.join("\n"))
}

const HELP: &str = "Available Commands:
  /set            Set session variables
  /show           Show model information
  /load <model>   Load a different model (the conversation is kept)
  /clear          Clear session context
  /bye            Exit
  /?, /help       Help for a command

Use \"\"\" to begin a multi-line message.
";

const HELP_SET: &str = "Available Commands:
  /set parameter <name> <value>  Set a parameter (temperature, top_p, top_k, min_p, num_predict, seed, stop, repeat_penalty, ...)
  /set system <string>           Set the system message
  /set verbose                   Show timings, tokens/s and joules after each answer
  /set quiet                     Hide them
  /set think | nothink           Ask a thinking model to think, or not
  /set format json | noformat    Constrain answers to JSON, or stop
";

const HELP_SHOW: &str = "Available Commands:
  /show info         Architecture, parameters, context, quantization and Ferric's verdict on it
  /show parameters   Parameters set in this session
  /show system       The system message
  /show template     The chat template
  /show license      The license
";

/// The model a session talks to: the name sent on the wire, and the local file when there is one.
pub struct Session { pub host: Host, pub kind: Kind, pub model: String, pub local: Option<crate::hub::LocalModel>, pub arg: String }

/// The interactive loop. Returns when the user says /bye or input ends.
pub fn repl(s: &mut Session, o: &mut ChatOpts, input: &mut impl BufRead) -> Result<(), String> {
    let prompt = std::io::stdin().is_terminal() && std::io::stdout().is_terminal();
    let mut history: Vec<Value> = Vec::new();
    let out = &mut std::io::stdout();
    if prompt { eprintln!("Send a message (/? for help)"); }
    while let Some(msg) = read_message(input, prompt) {
        let t = msg.trim();
        if t.is_empty() { continue; }
        if t.starts_with('/') {
            let words: Vec<&str> = t.split_whitespace().collect();
            match words.as_slice() {
                ["/bye"] | ["/exit"] => break,
                ["/clear"] => { history.clear(); println!("Cleared session context"); }
                ["/?"] | ["/help"] => print!("{HELP}"),
                ["/?", "set"] | ["/help", "set"] | ["/set"] => print!("{HELP_SET}"),
                ["/?", "show"] | ["/help", "show"] | ["/show"] => print!("{HELP_SHOW}"),
                ["/set", "parameter", name, vals @ ..] => match parse_param(name, vals) {
                    Ok(v) => {
                        if let Some(n) = param_note(&s.kind, name) { eprintln!("{n}"); }
                        o.options.insert(name.to_string(), v);
                        println!("Set parameter '{name}' to '{}'", vals.join(", "));
                    }
                    Err(e) => println!("Couldn't set parameter: {e}"),
                },
                ["/set", "system", ..] => {
                    let sys = t["/set".len()..].trim_start()["system".len()..].trim().trim_matches('"').to_string();
                    o.system = if sys.is_empty() { None } else { Some(sys) };
                    println!("Set system message.");
                }
                ["/set", "verbose"] => { o.verbose = true; println!("Set 'verbose' mode."); }
                ["/set", "quiet"] => { o.verbose = false; println!("Set 'quiet' mode."); }
                ["/set", "think"] => { o.think = Some(true); println!("Set 'think' mode."); }
                ["/set", "nothink"] => { o.think = Some(false); println!("Set 'nothink' mode."); }
                ["/set", "format", "json"] => { o.format = Some(json!("json")); println!("Set format to 'json' mode."); }
                ["/set", "noformat"] => { o.format = None; println!("Disabled format."); }
                ["/show", "parameters"] => {
                    if o.options.is_empty() { println!("No parameters set in this session."); }
                    for (k, v) in &o.options { println!("{k:<20}{v}"); }
                }
                ["/show", "system"] => println!("{}", o.system.as_deref().unwrap_or("No system message was specified.")),
                ["/show", what @ ("info" | "template" | "license")] => {
                    if let Err(e) = crate::manage::show(&s.host, &s.arg, s.local.as_ref(), what) { println!("Error: {e}"); }
                }
                ["/load", m] => match crate::open_model(&s.host, m, true) {
                    Ok((kind, model, local)) => {
                        s.kind = kind; s.model = model; s.local = local; s.arg = m.to_string();
                        crate::preload(&s.host, &s.kind, &s.model, o.keep_alive.as_ref()).unwrap_or_else(|e| println!("Error: {e}"));
                        println!("Loaded '{m}'");
                    }
                    Err(e) => println!("Error: {e}"),
                },
                _ => println!("Unknown command '{}'. Type /? for help", words[0]),
            }
            continue;
        }
        match chat_turn(&s.host, &s.model, o, &history, &msg, out) {
            Ok(turn) if turn.cancelled => {}
            Ok(turn) => {
                history.push(json!({"role": "user", "content": msg}));
                history.push(json!({"role": "assistant", "content": turn.content}));
                if prompt { println!(); }
                if o.verbose { eprint!("{}", stats(&turn.done)); }
            }
            Err(e) => eprintln!("Error: {e}"),
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn params_follow_ollama_types() {
        assert_eq!(parse_param("temperature", &["0.2"]).unwrap(), json!(0.2));
        assert_eq!(parse_param("num_predict", &["64"]).unwrap(), json!(64));
        assert_eq!(parse_param("stop", &["<|im_end|>", "User:"]).unwrap(), json!(["<|im_end|>", "User:"]));
        assert!(parse_param("temperature", &["hot"]).is_err());
        assert!(parse_param("top_k", &["1.5"]).is_err());
        assert!(parse_param("warp_factor", &["9"]).is_err());
    }

    #[test]
    fn multiline_blocks() {
        let mut i = std::io::Cursor::new("\"\"\"first\nsecond\nthird\"\"\"\nnext\n\"\"\"one line\"\"\"\n\"\"\"\nx\n\"\"\"\n");
        assert_eq!(read_message(&mut i, false).as_deref(), Some("first\nsecond\nthird"));
        assert_eq!(read_message(&mut i, false).as_deref(), Some("next"));
        assert_eq!(read_message(&mut i, false).as_deref(), Some("one line"));
        assert_eq!(read_message(&mut i, false).as_deref(), Some("x"));
        assert_eq!(read_message(&mut i, false), None);
    }

    #[test]
    fn stats_report_joules_or_say_why_not() {
        let d = json!({"total_duration": 2.0e9, "prompt_eval_count": 20, "prompt_eval_duration": 1.0e8,
                       "eval_count": 8, "eval_duration": 4.0e8,
                       "energy": {"joules": 2.0, "joules_per_token": 0.25, "boundary": "accelerator: GPU + DRAM rails", "class": "derived"}});
        let s = stats(&d);
        assert!(s.contains("eval rate:            20.00 tokens/s"), "{s}");
        assert!(s.contains("prompt eval rate:     200.00 tokens/s"), "{s}");
        assert!(s.contains("energy:               2.000 J (accelerator: GPU + DRAM rails, derived)"), "{s}");
        assert!(s.contains("energy per token:     0.2500 J/token"), "{s}");
        let n = stats(&json!({"eval_count": 3, "energy": {"joules": null, "why": "too short", "window_joules": 0.5, "seconds": 0.2}}));
        assert!(n.contains("unmeasured — too short; gross over the window 0.500 J in 0.200 s"), "{n}");
        assert!(stats(&json!({"eval_count": 3})).contains("not reported by this server"));
    }
}
