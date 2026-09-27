//! The real `ferric` binary against in-test servers: a mock that speaks Ollama's NDJSON API and records
//! every request it receives, and a mock Hugging Face that `curl` downloads from. The assertions are on
//! the exact requests sent (model, messages with history, options, `keep_alive: 0`) and on the rendered
//! output — the two things a user of a chat CLI actually depends on.
//!
//! Every run gets its own `FERRIC_HOME`/`HOME`, `FERRIC_SERVE` points at nothing unless a test says
//! otherwise, and no Hugging Face token leaks in from the machine, so nothing here touches the real
//! hub, the network, or a real server.
use serde_json::{json, Value};
use std::collections::HashMap;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::PathBuf;
use std::process::{Command, Output, Stdio};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

// ---------------------------------------------------------------------------------------------------
// scaffolding

struct Tmp(PathBuf);
impl Tmp {
    fn new(tag: &str) -> Tmp {
        static N: AtomicUsize = AtomicUsize::new(0);
        let p = std::env::temp_dir().join(format!("ferric-cli-{tag}-{}-{}", std::process::id(), N.fetch_add(1, Ordering::SeqCst)));
        let _ = std::fs::remove_dir_all(&p);
        std::fs::create_dir_all(p.join("hub")).unwrap();
        Tmp(p)
    }
    fn hub(&self) -> PathBuf { self.0.join("hub") }
}
impl Drop for Tmp { fn drop(&mut self) { let _ = std::fs::remove_dir_all(&self.0); } }

/// A qwen2 header with a template and a license; one F32 tensor (512 elements) and one Q8_0 tensor
/// (2048), so the majority type is Q8_0 and the parameter count 2560.
fn tiny_gguf() -> Vec<u8> {
    let mut w = ferric_gguf::write::GgufWriter::new("qwen2");
    w.kv_u32("qwen2.context_length", 4096).kv_u32("qwen2.embedding_length", 64).kv_u32("qwen2.block_count", 2)
        .kv_str("tokenizer.chat_template", "{% for m in messages %}{{ m.content }}{% endfor %}")
        .kv_str("general.license", "apache-2.0").kv_str("general.name", "Tiny Test");
    w.tensor_f32("token_embd.weight", &[64, 8], &[0.0; 512]);
    w.tensor("blk.0.ffn_up.weight", &[64, 32], 8, vec![0u8; 2048 / 32 * 34]);
    w.finish().unwrap()
}

/// A split set as llama-gguf-split writes it (split.no / split.count u16, split.tensors.count i32),
/// one 256-element F32 tensor per part; metadata in part 1.
fn split_gguf(prefix: &str, n: u16) -> Vec<(String, Vec<u8>)> {
    (0..n).map(|i| {
        let mut w = ferric_gguf::write::GgufWriter::new("llama");
        w.kv_u16("split.no", i).kv_u16("split.count", n).kv_i32("split.tensors.count", n as i32);
        if i == 0 { w.kv_u32("llama.context_length", 2048); }
        w.tensor_f32(&format!("blk.{i}.attn_q.weight"), &[64, 4], &[i as f32; 256]);
        (format!("{prefix}-{:05}-of-{n:05}.gguf", i + 1), w.finish().unwrap())
    }).collect()
}

fn free_port() -> u16 { TcpListener::bind("127.0.0.1:0").unwrap().local_addr().unwrap().port() }

struct Req { method: String, path: String, headers: HashMap<String, String>, body: Value }

fn read_req(s: &mut TcpStream) -> Option<Req> {
    let mut r = BufReader::new(s.try_clone().ok()?);
    let mut line = String::new();
    if r.read_line(&mut line).ok()? == 0 { return None; }
    let mut it = line.split_whitespace();
    let (method, path) = (it.next()?.to_string(), it.next()?.to_string());
    let mut headers = HashMap::new();
    loop {
        let mut h = String::new();
        if r.read_line(&mut h).ok()? == 0 || h.trim().is_empty() { break; }
        if let Some((k, v)) = h.split_once(':') { headers.insert(k.trim().to_ascii_lowercase(), v.trim().to_string()); }
    }
    let n: usize = headers.get("content-length").and_then(|v| v.parse().ok()).unwrap_or(0);
    let mut b = vec![0u8; n];
    r.read_exact(&mut b).ok()?;
    Some(Req { method, path, headers, body: serde_json::from_slice(&b).unwrap_or(Value::Null) })
}

fn write_json(s: &mut TcpStream, status: u16, v: &Value) {
    let b = v.to_string();
    let _ = write!(s, "HTTP/1.1 {status} X\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{b}", b.len());
}

// ---------------------------------------------------------------------------------------------------
// mock Ollama-API server

#[derive(Clone, Copy, PartialEq)]
enum Framing {
    /// ferric-serve: no length, the body ends when the socket closes.
    Close,
    /// Ollama: `Transfer-Encoding: chunked` — and here every line is split across TWO chunks, so a
    /// decoder that assumed one line per chunk would fail.
    Chunked,
}

#[derive(Clone)]
struct Reply { thinking: Vec<&'static str>, content: Vec<&'static str> }

struct MockCfg { version: &'static str, framing: Framing, replies: Vec<Reply>, unload_reason: &'static str, loaded: Vec<String>,
                 /// Close the chat stream without its final `done: true` line — a server dying mid-answer.
                 truncate: bool }

impl Default for MockCfg {
    fn default() -> Self {
        MockCfg { version: "0.12.0-ferric", framing: Framing::Close, replies: vec![], unload_reason: "unload", loaded: vec![], truncate: false }
    }
}

fn final_stats() -> Value {
    json!({"total_duration": 6.0e8, "load_duration": 0, "prompt_eval_count": 20, "prompt_eval_duration": 1.0e8,
           "eval_count": 8, "eval_duration": 4.0e8,
           "energy": {"joules": 1.5, "joules_per_token": 0.25, "boundary": "accelerator: GPU + DRAM rails", "class": "derived"}})
}

struct Mock { port: u16, reqs: Arc<Mutex<Vec<Req>>> }

impl Mock {
    fn start(cfg: MockCfg) -> Mock { Mock::start_on(TcpListener::bind("127.0.0.1:0").unwrap(), cfg) }

    fn start_on(l: TcpListener, cfg: MockCfg) -> Mock {
        let port = l.local_addr().unwrap().port();
        let reqs: Arc<Mutex<Vec<Req>>> = Arc::default();
        let (rq, cfg) = (reqs.clone(), Arc::new(cfg));
        let chats = Arc::new(AtomicUsize::new(0));
        std::thread::spawn(move || {
            for s in l.incoming() {
                let Ok(mut s) = s else { continue };
                let (rq, cfg, chats) = (rq.clone(), cfg.clone(), chats.clone());
                std::thread::spawn(move || {
                    let Some(req) = read_req(&mut s) else { return };
                    Mock::answer(&mut s, &req, &cfg, &chats);
                    rq.lock().unwrap().push(req);
                });
            }
        });
        Mock { port, reqs }
    }

    fn answer(s: &mut TcpStream, req: &Req, cfg: &MockCfg, chats: &AtomicUsize) {
        let model = req.body["model"].clone();
        match (req.method.as_str(), req.path.as_str()) {
            ("GET", "/api/version") => write_json(s, 200, &json!({"version": cfg.version})),
            ("GET", "/api/ps") => write_json(s, 200, &json!({"models": cfg.loaded.iter().map(|n| json!({
                "name": n, "model": n, "size": 675_710_816u64, "context_length": 4096, "expires_at": "2318-01-01T00:00:00Z"})).collect::<Vec<_>>()})),
            ("POST", "/api/generate") => {
                let prompt = req.body["prompt"].as_str().unwrap_or("");
                if req.body.get("keep_alive") == Some(&json!(0)) && prompt.is_empty() {
                    write_json(s, 200, &json!({"model": model, "response": "", "done": true, "done_reason": cfg.unload_reason}));
                } else if prompt.is_empty() {
                    write_json(s, 200, &json!({"model": model, "response": "", "done": true, "done_reason": "load"}));
                } else {
                    // bench: a prefill of 500 tokens in 0.25 s, or n generated tokens at 50 tok/s;
                    // 0.002 J per token either way.
                    let n = req.body["options"]["num_predict"].as_u64().unwrap_or(1);
                    let mut v = json!({"model": model, "response": "x", "done": true, "done_reason": "length",
                        "prompt_eval_count": 500, "prompt_eval_duration": 2.5e8, "eval_count": n, "eval_duration": n as f64 * 2.0e7});
                    v["energy"] = json!({"joules": 0.002 * if n == 1 { 500.0 } else { n as f64 }});
                    write_json(s, 200, &v);
                }
            }
            ("POST", "/api/chat") => {
                let k = chats.fetch_add(1, Ordering::SeqCst);
                let reply = cfg.replies.get(k).cloned().unwrap_or(Reply { thinking: vec![], content: vec!["?"] });
                let mut lines: Vec<Value> = Vec::new();
                for t in &reply.thinking { lines.push(json!({"model": model, "message": {"role": "assistant", "content": "", "thinking": t}, "done": false})); }
                for c in &reply.content { lines.push(json!({"model": model, "message": {"role": "assistant", "content": c}, "done": false})); }
                let mut last = json!({"model": model, "message": {"role": "assistant", "content": ""}, "done": true, "done_reason": "stop"});
                for (k, v) in final_stats().as_object().unwrap() {
                    // Only ferric-serve reports joules; a real Ollama's final line has no `energy`.
                    if k != "energy" || cfg.version.contains("ferric") { last[k] = v.clone(); }
                }
                if !cfg.truncate { lines.push(last); }
                match cfg.framing {
                    Framing::Close => {
                        let _ = write!(s, "HTTP/1.1 200 OK\r\nContent-Type: application/x-ndjson\r\nConnection: close\r\n\r\n");
                        for l in lines { let _ = write!(s, "{l}\n"); let _ = s.flush(); }
                    }
                    Framing::Chunked => {
                        let _ = write!(s, "HTTP/1.1 200 OK\r\nContent-Type: application/x-ndjson\r\nTransfer-Encoding: chunked\r\n\r\n");
                        for l in lines {
                            let t = format!("{l}\n");
                            let (a, b) = t.split_at(t.len() / 2);
                            for part in [a, b] { let _ = write!(s, "{:x}\r\n{part}\r\n", part.len()); let _ = s.flush(); }
                        }
                        let _ = write!(s, "0\r\n\r\n");
                    }
                }
            }
            ("POST", "/api/show") => write_json(s, 200, &json!({"details": {"family": "llama", "parameter_size": "3.2B", "quantization_level": "Q4_K_M"},
                "model_info": {"general.architecture": "llama", "llama.context_length": 131072}, "capabilities": ["completion", "tools"]})),
            _ => write_json(s, 404, &json!({"error": format!("no route {} {}", req.method, req.path)})),
        }
    }

    fn host(&self) -> String { format!("127.0.0.1:{}", self.port) }
    fn bodies(&self, method: &str, path: &str) -> Vec<Value> {
        self.reqs.lock().unwrap().iter().filter(|r| r.method == method && r.path == path).map(|r| r.body.clone()).collect()
    }
    fn paths(&self) -> Vec<String> { self.reqs.lock().unwrap().iter().map(|r| format!("{} {}", r.method, r.path)).collect() }
}

// ---------------------------------------------------------------------------------------------------
// mock Hugging Face (curl downloads from it)

struct Hf { port: u16, reqs: Arc<Mutex<Vec<Req>>> }

const SHA: &str = "0123456789abcdef0123456789abcdef01234567";

impl Hf {
    fn start(repo: &'static str, files: Vec<(String, Vec<u8>)>) -> Hf {
        let l = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = l.local_addr().unwrap().port();
        let reqs: Arc<Mutex<Vec<Req>>> = Arc::default();
        let rq = reqs.clone();
        let files: Arc<HashMap<String, Vec<u8>>> = Arc::new(files.into_iter().collect());
        std::thread::spawn(move || {
            for s in l.incoming() {
                let Ok(mut s) = s else { continue };
                let (rq, files) = (rq.clone(), files.clone());
                std::thread::spawn(move || {
                    let Some(req) = read_req(&mut s) else { return };
                    let api = format!("/api/models/{repo}?blobs=true");
                    let resolve = format!("/{repo}/resolve/{SHA}/");
                    if req.path == api {
                        let mut sib: Vec<Value> = vec![json!({"rfilename": "README.md", "size": 5})];
                        let mut names: Vec<&String> = files.keys().collect();
                        names.sort();
                        for n in names { sib.push(json!({"rfilename": n, "size": files[n].len(), "lfs": {"size": files[n].len()}})); }
                        write_json(&mut s, 200, &json!({"id": repo, "sha": SHA, "siblings": sib}));
                    } else if let Some(f) = req.path.strip_prefix(&resolve).and_then(|f| files.get(f)) {
                        let from: usize = req.headers.get("range").and_then(|r| r.strip_prefix("bytes=")).and_then(|r| r.trim_end_matches('-').parse().ok()).unwrap_or(0);
                        let body = &f[from.min(f.len())..];
                        let head = if from > 0 {
                            format!("HTTP/1.1 206 Partial Content\r\nContent-Range: bytes {from}-{}/{}\r\n", f.len() - 1, f.len())
                        } else { "HTTP/1.1 200 OK\r\n".to_string() };
                        let _ = write!(s, "{head}Content-Length: {}\r\nAccept-Ranges: bytes\r\nConnection: close\r\n\r\n", body.len());
                        let _ = s.write_all(body);
                    } else {
                        write_json(&mut s, 404, &json!({"error": "Repository not found"}));
                    }
                    rq.lock().unwrap().push(req);
                });
            }
        });
        Hf { port, reqs }
    }
    fn endpoint(&self) -> String { format!("http://127.0.0.1:{}", self.port) }
    fn gets(&self) -> Vec<(String, Option<String>, Option<String>)> {
        self.reqs.lock().unwrap().iter().map(|r| (r.path.clone(), r.headers.get("range").cloned(), r.headers.get("authorization").cloned())).collect()
    }
}

// ---------------------------------------------------------------------------------------------------
// running the binary

fn ferric(t: &Tmp, host: &str, args: &[&str], stdin: &str, env: &[(&str, &str)]) -> Output {
    let mut c = Command::new(env!("CARGO_BIN_EXE_ferric"));
    c.args(args).env("FERRIC_HOST", host).env("FERRIC_HOME", &t.0).env("HOME", &t.0)
        .env("FERRIC_SERVE", t.0.join("no-such-ferric-serve"))
        .env_remove("HF_TOKEN").env_remove("HF_HOME").env_remove("HF_ENDPOINT").env_remove("FERRIC_HF_ENDPOINT")
        .stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::piped());
    for (k, v) in env { c.env(k, v); }
    let mut child = c.spawn().unwrap();
    child.stdin.take().unwrap().write_all(stdin.as_bytes()).unwrap();
    let out = child.wait_with_output().unwrap();
    out
}

fn text(b: &[u8]) -> String { String::from_utf8_lossy(b).into_owned() }

fn with_tiny(t: &Tmp) -> String {
    let p = t.hub().join("tiny.gguf");
    std::fs::write(&p, tiny_gguf()).unwrap();
    std::fs::canonicalize(&p).unwrap().to_string_lossy().to_string()
}

// ---------------------------------------------------------------------------------------------------
// run / chat

#[test]
fn one_shot_streams_the_answer_and_reports_joules() {
    let t = Tmp::new("oneshot");
    let path = with_tiny(&t);
    let m = Mock::start(MockCfg { replies: vec![Reply { thinking: vec!["hidden"], content: vec!["Par", "is."] }], loaded: vec!["tiny:latest".into()], ..Default::default() });
    let o = ferric(&t, &m.host(), &["run", "tiny", "What is the capital of France?", "--verbose", "--hidethinking", "--option", "temperature=0"], "", &[]);
    let (out, err) = (text(&o.stdout), text(&o.stderr));
    assert!(o.status.success(), "{err}");
    assert_eq!(out, "Paris.\n", "the answer alone, thinking hidden");
    // The model is named by its absolute path: the one spelling every server form resolves.
    assert_eq!(m.bodies("POST", "/api/chat"), vec![json!({"model": path, "stream": true,
        "messages": [{"role": "user", "content": "What is the capital of France?"}], "options": {"temperature": 0.0}})]);
    // Loaded before the first message, as `ollama run` does.
    let p = m.paths();
    let (load, chat) = (p.iter().position(|x| x == "POST /api/generate").unwrap(), p.iter().position(|x| x == "POST /api/chat").unwrap());
    assert!(load < chat, "{p:?}");
    assert_eq!(m.bodies("POST", "/api/generate")[0], json!({"model": path}));
    for want in ["eval rate:            20.00 tokens/s", "prompt eval rate:     200.00 tokens/s",
                 "energy:               1.500 J (accelerator: GPU + DRAM rails, derived)", "energy per token:     0.2500 J/token"] {
        assert!(err.contains(want), "stderr lacks {want:?}:\n{err}");
    }
    assert!(!err.contains("warning"), "the loaded model matches: {err}");
}

#[test]
fn repl_sends_history_options_and_multiline_and_clear_resets_it() {
    let t = Tmp::new("repl");
    let path = with_tiny(&t);
    let m = Mock::start(MockCfg {
        framing: Framing::Chunked,
        replies: vec![
            Reply { thinking: vec!["Let me ", "recall."], content: vec!["Paris", "."] },
            Reply { thinking: vec![], content: vec!["PA", "RIS"] },
            Reply { thinking: vec![], content: vec!["Hello there."] },
        ],
        loaded: vec!["tiny".into()],
        ..Default::default()
    });
    let input = "/set parameter temperature 0.2\nWhat is the capital of France?\n\"\"\"Say it\nin capitals\"\"\"\n/clear\nHello\n/bye\nnever sent\n";
    let o = ferric(&t, &m.host(), &["run", "tiny", "-i"], input, &[]);
    assert!(o.status.success(), "{}", text(&o.stderr));
    assert_eq!(text(&o.stdout), "Set parameter 'temperature' to '0.2'\nThinking...\nLet me recall.\n...done thinking.\n\nParis.\nPARIS\nCleared session context\nHello there.\n");
    let u1 = json!({"role": "user", "content": "What is the capital of France?"});
    let opts = json!({"temperature": 0.2});
    assert_eq!(m.bodies("POST", "/api/chat"), vec![
        json!({"model": path, "stream": true, "options": opts, "messages": [u1]}),
        // Turn 2 carries turn 1 — the answer WITHOUT its reasoning — and the multi-line message joined by \n.
        json!({"model": path, "stream": true, "options": opts, "messages": [u1, {"role": "assistant", "content": "Paris."},
                                                                             {"role": "user", "content": "Say it\nin capitals"}]}),
        json!({"model": path, "stream": true, "options": opts, "messages": [{"role": "user", "content": "Hello"}]}),
    ]);
}

#[test]
fn a_stream_cut_short_is_an_error_not_an_answer() {
    let t = Tmp::new("cut");
    with_tiny(&t);
    let m = Mock::start(MockCfg { replies: vec![Reply { thinking: vec![], content: vec!["The capital of"] }], truncate: true, ..Default::default() });
    let o = ferric(&t, &m.host(), &["run", "tiny", "capital?", "--verbose"], "", &[]);
    assert!(!o.status.success());
    assert_eq!(text(&o.stdout), "The capital of\n");
    assert!(text(&o.stderr).contains("answer above is incomplete") && !text(&o.stderr).contains("eval rate"), "{}", text(&o.stderr));
}

#[test]
fn a_real_ollama_gets_the_name_as_typed_and_its_errors_are_shown() {
    let t = Tmp::new("foreign");
    let m = Mock::start(MockCfg { version: "0.12.3", replies: vec![Reply { thinking: vec![], content: vec!["Hi!"] }], ..Default::default() });
    let o = ferric(&t, &m.host(), &["run", "llama3.2", "hello"], "", &[]);
    assert!(o.status.success(), "{}", text(&o.stderr));
    assert_eq!(text(&o.stdout), "Hi!\n");
    assert_eq!(m.bodies("POST", "/api/chat")[0]["model"], "llama3.2");
    assert!(!m.paths().contains(&"GET /api/ps".to_string()), "no ferric-only checks against Ollama: {:?}", m.paths());
    let o = ferric(&t, &m.host(), &["run", "llama3.2", "--verbose", "hello"], "", &[]);
    assert!(text(&o.stderr).contains("energy:               not reported by this server"), "{}", text(&o.stderr));
}

#[test]
fn a_single_model_server_holding_another_model_is_warned_about() {
    let t = Tmp::new("mismatch");
    with_tiny(&t);
    let m = Mock::start(MockCfg { replies: vec![Reply { thinking: vec![], content: vec!["ok"] }], loaded: vec!["qwen2.5-0.5b-instruct-q8_0:latest".into()], ..Default::default() });
    let o = ferric(&t, &m.host(), &["run", "tiny", "hi"], "", &[]);
    assert!(o.status.success());
    assert!(text(&o.stderr).contains("lists 'qwen2.5-0.5b-instruct-q8_0:latest' as loaded, not 'tiny'"), "{}", text(&o.stderr));
}

// ---------------------------------------------------------------------------------------------------
// stop / ps / show / list / rm

#[test]
fn stop_sends_keep_alive_zero_and_believes_only_an_unload() {
    let t = Tmp::new("stop");
    let path = with_tiny(&t);
    let m = Mock::start(MockCfg::default());
    let o = ferric(&t, &m.host(), &["stop", "tiny"], "", &[]);
    assert!(o.status.success(), "{}", text(&o.stderr));
    assert_eq!(m.bodies("POST", "/api/generate"), vec![json!({"model": path, "keep_alive": 0})]);

    // A server that answers "load" did not unload; saying "stopped" would be a lie.
    let m = Mock::start(MockCfg { unload_reason: "load", ..Default::default() });
    let o = ferric(&t, &m.host(), &["stop", "tiny"], "", &[]);
    assert!(!o.status.success());
    assert!(text(&o.stderr).contains("did not confirm the unload (done_reason: load)"), "{}", text(&o.stderr));

    let o = ferric(&t, &format!("127.0.0.1:{}", free_port()), &["stop", "tiny"], "", &[]);
    assert!(!o.status.success() && text(&o.stderr).contains("no server answering"));
}

#[test]
fn ps_prints_what_the_server_holds() {
    let t = Tmp::new("ps");
    let m = Mock::start(MockCfg { loaded: vec!["qwen2.5-0.5b-instruct-q8_0:latest".into()], ..Default::default() });
    let o = ferric(&t, &m.host(), &["ps"], "", &[]);
    let out = text(&o.stdout);
    assert!(o.status.success(), "{}", text(&o.stderr));
    assert!(out.starts_with("NAME"), "{out}");
    assert!(out.contains("qwen2.5-0.5b-instruct-q8_0:latest    675.7 MB    4096       Forever"), "{out}");
}

#[test]
fn list_and_show_read_the_gguf_header_and_the_registry() {
    let t = Tmp::new("list");
    with_tiny(&t);
    let d = t.hub().join("acme_Big-GGUF");
    std::fs::create_dir_all(&d).unwrap();
    let parts = split_gguf("big-Q4_K_M", 3);
    for (n, b) in &parts { std::fs::write(d.join(n), b).unwrap(); }
    std::fs::write(d.join("junk.gguf.partial"), b"half").unwrap();
    let dead = format!("127.0.0.1:{}", free_port());
    let o = ferric(&t, &dead, &["list"], "", &[]);
    let out = text(&o.stdout);
    assert!(o.status.success(), "{}", text(&o.stderr));
    let size: usize = parts.iter().map(|(_, b)| b.len()).sum();
    let row = |name: &str| out.lines().find(|l| l.starts_with(name)).unwrap_or_else(|| panic!("no {name} in:\n{out}")).split_whitespace().collect::<Vec<_>>();
    assert_eq!(&row("tiny")[..4], ["tiny", "qwen2", "2.6K", "Q8_0"]);
    // One row for the three files: all parts' tensors, all parts' bytes.
    assert_eq!(&row("big-Q4_K_M")[..4], ["big-Q4_K_M", "llama", "768", "F32"]);
    assert_eq!(row("big-Q4_K_M")[4..6].join(" "), format!("{:.1} KB", size as f64 / 1e3));
    assert_eq!(out.lines().count(), 3, "header + 2 models, no .partial:\n{out}");

    let o = ferric(&t, &dead, &["show", "tiny"], "", &[]);
    let out = text(&o.stdout);
    for want in ["name                Tiny Test", "architecture        qwen2", "parameters          2.6K", "context length      4096",
                 "quantization        Q8_0", "runtime             dense", "status              verified", "present (Jinja, 50 chars)", "license             apache-2.0"] {
        assert!(out.contains(want), "show lacks {want:?}:\n{out}");
    }
    assert_eq!(text(&ferric(&t, &dead, &["show", "tiny", "--template"], "", &[]).stdout), "{% for m in messages %}{{ m.content }}{% endfor %}\n");
    // A model the hub does not have: ask the server, and still give Ferric's verdict on its architecture.
    let m = Mock::start(MockCfg { version: "0.12.3", ..Default::default() });
    let out = text(&ferric(&t, &m.host(), &["show", "llama3.2"], "", &[]).stdout);
    assert!(out.contains("quantization        Q4_K_M") && out.contains("context length      131072") && out.contains("status              verified"), "{out}");
}

#[test]
fn rm_asks_first_and_deletes_every_part() {
    let t = Tmp::new("rm");
    with_tiny(&t);
    let d = t.hub().join("acme_Big-GGUF");
    std::fs::create_dir_all(&d).unwrap();
    for (n, b) in split_gguf("big-Q4_K_M", 2) { std::fs::write(d.join(n), b).unwrap(); }
    let dead = format!("127.0.0.1:{}", free_port());
    let o = ferric(&t, &dead, &["rm", "tiny"], "n\n", &[]);
    assert!(o.status.success() && t.hub().join("tiny.gguf").exists(), "answered no: kept");
    let o = ferric(&t, &dead, &["rm", "tiny"], "y\n", &[]);
    assert!(o.status.success() && !t.hub().join("tiny.gguf").exists(), "{}", text(&o.stderr));
    let o = ferric(&t, &dead, &["rm", "-y", "big-Q4_K_M"], "", &[]);
    assert!(o.status.success(), "{}", text(&o.stderr));
    assert!(!d.exists(), "both parts and the emptied repo directory are gone");
    let outside = t.0.join("mine.gguf");
    std::fs::write(&outside, tiny_gguf()).unwrap();
    let o = ferric(&t, &dead, &["rm", "-y", outside.to_str().unwrap()], "", &[]);
    assert!(!o.status.success() && outside.exists(), "a file outside the hub is not ferric's to delete");
}

// ---------------------------------------------------------------------------------------------------
// pull

#[test]
fn pull_selects_the_file_a_quant_tag_names_and_sends_the_token() {
    let t = Tmp::new("pull-tag");
    let (q8, q4) = (tiny_gguf(), vec![7u8; 3000]);
    let hf = Hf::start("acme/Tiny-GGUF", vec![("tiny-Q8_0.gguf".into(), q8.clone()), ("tiny-Q4_K_M.gguf".into(), q4)]);
    let dead = format!("127.0.0.1:{}", free_port());
    let o = ferric(&t, &dead, &["pull", "hf.co/acme/Tiny-GGUF:q8_0"], "", &[("FERRIC_HF_ENDPOINT", &hf.endpoint()), ("HF_TOKEN", "hf_test")]);
    assert!(o.status.success(), "{}", text(&o.stderr));
    let d = t.hub().join("acme_Tiny-GGUF");
    assert_eq!(std::fs::read(d.join("tiny-Q8_0.gguf")).unwrap(), q8);
    assert!(!d.join("tiny-Q4_K_M.gguf").exists(), "only the named quant");
    let gets = hf.gets();
    assert_eq!(gets.iter().map(|g| g.0.as_str()).collect::<Vec<_>>(),
               ["/api/models/acme/Tiny-GGUF?blobs=true", &format!("/acme/Tiny-GGUF/resolve/{SHA}/tiny-Q8_0.gguf")]);
    assert!(gets.iter().all(|g| g.2.as_deref() == Some("Bearer hf_test")), "{gets:?}");
    // No tag: Ollama's default, Q4_K_M. A tag nothing matches: the choices are listed.
    let o = ferric(&t, &dead, &["pull", "acme/Tiny-GGUF:Q2_K"], "", &[("FERRIC_HF_ENDPOINT", &hf.endpoint())]);
    assert!(!o.status.success() && text(&o.stderr).contains("available: Q4_K_M, Q8_0"), "{}", text(&o.stderr));
}

#[test]
fn pull_fetches_every_split_part_and_list_shows_one_model() {
    let t = Tmp::new("pull-split");
    let parts = split_gguf("big-Q4_K_M", 3);
    let mut files = parts.clone();
    files.push(("big-Q8_0.gguf".into(), tiny_gguf()));
    let hf = Hf::start("acme/Big-GGUF", files);
    let dead = format!("127.0.0.1:{}", free_port());
    let o = ferric(&t, &dead, &["pull", "acme/Big-GGUF:Q4_K_M"], "", &[("FERRIC_HF_ENDPOINT", &hf.endpoint())]);
    assert!(o.status.success(), "{}", text(&o.stderr));
    let d = t.hub().join("acme_Big-GGUF");
    for (n, b) in &parts { assert_eq!(&std::fs::read(d.join(n)).unwrap(), b, "{n}"); }
    assert!(!d.join("big-Q8_0.gguf").exists());
    let out = text(&ferric(&t, &dead, &["list"], "", &[]).stdout);
    assert!(out.lines().any(|l| l.split_whitespace().take(4).collect::<Vec<_>>() == ["big-Q4_K_M", "llama", "768", "F32"]), "{out}");
}

#[test]
fn pull_resumes_a_partial_download_from_where_it_stopped() {
    let t = Tmp::new("pull-resume");
    let full = tiny_gguf();
    let hf = Hf::start("acme/Tiny-GGUF", vec![("tiny-Q8_0.gguf".into(), full.clone())]);
    let d = t.hub().join("acme_Tiny-GGUF");
    std::fs::create_dir_all(&d).unwrap();
    let cut = full.len() / 3;
    std::fs::write(d.join("tiny-Q8_0.gguf.partial"), &full[..cut]).unwrap();
    let dead = format!("127.0.0.1:{}", free_port());
    let o = ferric(&t, &dead, &["pull", "acme/Tiny-GGUF:Q8_0"], "", &[("FERRIC_HF_ENDPOINT", &hf.endpoint())]);
    assert!(o.status.success(), "{}", text(&o.stderr));
    assert_eq!(std::fs::read(d.join("tiny-Q8_0.gguf")).unwrap(), full, "resumed bytes appended, not restarted or doubled");
    assert!(!d.join("tiny-Q8_0.gguf.partial").exists());
    let ranged: Vec<_> = hf.gets().into_iter().filter_map(|g| g.1).collect();
    assert_eq!(ranged, vec![format!("bytes={cut}-")]);
    // Complete now: a second pull transfers nothing.
    let before = hf.gets().len();
    assert!(ferric(&t, &dead, &["pull", "acme/Tiny-GGUF:Q8_0"], "", &[("FERRIC_HF_ENDPOINT", &hf.endpoint())]).status.success());
    assert_eq!(hf.gets()[before..].iter().filter(|g| g.0.contains("/resolve/")).count(), 0);
}

#[test]
fn run_pulls_an_hf_reference_on_first_use() {
    let t = Tmp::new("run-pull");
    let hf = Hf::start("acme/Tiny-GGUF", vec![("tiny-Q8_0.gguf".into(), tiny_gguf())]);
    let m = Mock::start(MockCfg { replies: vec![Reply { thinking: vec![], content: vec!["Paris."] }], ..Default::default() });
    let o = ferric(&t, &m.host(), &["run", "acme/Tiny-GGUF", "capital of France?"], "", &[("FERRIC_HF_ENDPOINT", &hf.endpoint())]);
    assert!(o.status.success(), "{}", text(&o.stderr));
    assert_eq!(text(&o.stdout), "Paris.\n");
    let pulled = std::fs::canonicalize(t.hub().join("acme_Tiny-GGUF/tiny-Q8_0.gguf")).unwrap();
    assert_eq!(m.bodies("POST", "/api/chat")[0]["model"], json!(pulled.to_string_lossy()));
}

// ---------------------------------------------------------------------------------------------------
// the process manager

fn script(t: &Tmp, name: &str, body: &str) -> PathBuf {
    let p = t.0.join(name);
    std::fs::write(&p, format!("#!/bin/sh\n{body}\n")).unwrap();
    #[cfg(unix)]
    { use std::os::unix::fs::PermissionsExt; std::fs::set_permissions(&p, std::fs::Permissions::from_mode(0o755)).unwrap(); }
    p
}

/// Nothing answering → `ferric run` starts ferric-serve detached, with the model and FERRIC_HOST's
/// address, logs to FERRIC_HOME/serve.log, waits until it answers, then chats. The "server" is a
/// script that records its argv and stays alive; the test brings the mock up on the port only once
/// the script has run, so an answer before the spawn is impossible.
#[cfg(unix)]
#[test]
fn run_starts_a_detached_server_and_waits_for_it() {
    let t = Tmp::new("autostart");
    let path = with_tiny(&t);
    let argv = t.0.join("argv");
    let fake = script(&t, "fake-serve", &format!("echo \"$@\" > {0}.tmp; echo $$ > {0}.pid; mv {0}.tmp {0}; exec sleep 30", argv.display()));
    let port = free_port();
    let (a2, done) = (argv.clone(), Arc::new(Mutex::new(None::<Mock>)));
    let d2 = done.clone();
    std::thread::spawn(move || {
        let t0 = Instant::now();
        while !a2.exists() && t0.elapsed() < Duration::from_secs(20) { std::thread::sleep(Duration::from_millis(20)); }
        let l = TcpListener::bind(("127.0.0.1", port)).unwrap();
        *d2.lock().unwrap() = Some(Mock::start_on(l, MockCfg { replies: vec![Reply { thinking: vec![], content: vec!["Paris."] }], ..Default::default() }));
    });
    let o = ferric(&t, &format!("127.0.0.1:{port}"), &["run", "tiny", "capital?"], "", &[("FERRIC_SERVE", fake.to_str().unwrap()), ("FERRIC_START_TIMEOUT", "30")]);
    let pid = std::fs::read_to_string(format!("{}.pid", argv.display())).unwrap_or_default();
    let _ = Command::new("kill").arg(pid.trim()).status();
    assert!(o.status.success(), "{}", text(&o.stderr));
    assert_eq!(text(&o.stdout), "Paris.\n");
    assert_eq!(std::fs::read_to_string(&argv).unwrap().trim(), format!("{path} --host 127.0.0.1 --port {port}"));
    let log = std::fs::read_to_string(t.0.join("serve.log")).unwrap();
    assert!(log.contains("==== ferric") && log.contains("fake-serve"), "{log}");
    assert_eq!(std::fs::read_to_string(t.0.join("serve.pid")).unwrap().trim(), pid.trim());
    assert!(done.lock().unwrap().as_ref().unwrap().bodies("POST", "/api/chat").len() == 1);
}

/// A server that dies while loading is reported at once, with the end of its log.
#[cfg(unix)]
#[test]
fn a_server_that_dies_is_reported_with_its_log() {
    let t = Tmp::new("dies");
    with_tiny(&t);
    let fake = script(&t, "dying-serve", "echo 'load model: tensor blk.0 missing' >&2; exit 3");
    let o = ferric(&t, &format!("127.0.0.1:{}", free_port()), &["run", "tiny", "hi"], "", &[("FERRIC_SERVE", fake.to_str().unwrap())]);
    let err = text(&o.stderr);
    assert!(!o.status.success());
    assert!(err.contains("exited") && err.contains("load model: tensor blk.0 missing"), "{err}");
}

#[cfg(unix)]
#[test]
fn serve_execs_ferric_serve_with_resolved_models_and_the_flags_verbatim() {
    let t = Tmp::new("serve");
    let path = with_tiny(&t);
    let fake = script(&t, "echo-serve", "echo \"$@\"");
    let o = ferric(&t, "127.0.0.1:12345", &["serve", "tiny", "--max-batch", "4", "--no-batch"], "", &[("FERRIC_SERVE", fake.to_str().unwrap())]);
    assert!(o.status.success(), "{}", text(&o.stderr));
    assert_eq!(text(&o.stdout).trim(), format!("{path} --max-batch 4 --no-batch --host 127.0.0.1 --port 12345"));
    let o = ferric(&t, "127.0.0.1:12345", &["serve", "tiny", "--port", "9"], "", &[("FERRIC_SERVE", fake.to_str().unwrap())]);
    assert_eq!(text(&o.stdout).trim(), format!("{path} --port 9 --host 127.0.0.1"), "an explicit --port wins");
}

// ---------------------------------------------------------------------------------------------------
// bench

#[test]
fn bench_reports_ranges_and_joules_per_token_from_uncached_raw_prompts() {
    let t = Tmp::new("bench");
    let path = with_tiny(&t);
    let m = Mock::start(MockCfg::default());
    let o = ferric(&t, &m.host(), &["bench", "tiny", "--reps", "2", "--prompt-tokens", "32", "--gen-tokens", "8", "--gap-ms", "0"], "", &[]);
    let out = text(&o.stdout);
    assert!(o.status.success(), "{}", text(&o.stderr));
    let row = |p: &str| out.lines().find(|l| l.starts_with(p)).unwrap_or_else(|| panic!("no {p} row:\n{out}")).split_whitespace().collect::<Vec<_>>().join(" ");
    // Named by the server's own token count (the mock says 500), not the words requested.
    assert_eq!(row("pp500"), "pp500 500 2000.0 0.0020 1.000 2/2");
    assert_eq!(row("tg8"), "tg8 8 50.0 0.0020 0.016 2/2");
    let gens: Vec<Value> = m.bodies("POST", "/api/generate").into_iter().filter(|b| b["prompt"].as_str().is_some_and(|p| !p.is_empty())).collect();
    assert_eq!(gens.len(), 5, "warm-up + 2 x (pp, tg)");
    assert!(gens.iter().all(|g| g["raw"] == true && g["stream"] == false && g["model"] == json!(path)));
    assert_eq!(gens.iter().map(|g| g["options"]["num_predict"].as_u64().unwrap()).collect::<Vec<_>>(), [8, 1, 8, 1, 8]);
    let firsts: std::collections::HashSet<&str> = gens.iter().map(|g| g["prompt"].as_str().unwrap().split(' ').next().unwrap()).collect();
    assert_eq!(firsts.len(), 1, "one word 'Record' starts every prompt");
    let heads: std::collections::HashSet<String> = gens.iter().map(|g| g["prompt"].as_str().unwrap().chars().take(24).collect()).collect();
    assert_eq!(heads.len(), 5, "every prompt differs from its first tokens, so no prefix cache can serve it");
}

#[test]
fn help_and_bad_input() {
    let t = Tmp::new("help");
    let dead = format!("127.0.0.1:{}", free_port());
    assert!(text(&ferric(&t, &dead, &["--help"], "", &[]).stdout).contains("ferric run <model> [prompt]"));
    let o = ferric(&t, &dead, &["run", "no-such-model", "hi"], "", &[]);
    assert!(!o.status.success() && text(&o.stderr).contains("is not a file, a model in"), "{}", text(&o.stderr));
    let o = ferric(&t, "https://x:1", &["ps"], "", &[]);
    assert!(!o.status.success() && text(&o.stderr).contains("plain HTTP"));
}
