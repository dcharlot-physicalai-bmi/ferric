//! **The OpenAI Batch API** (parity gap S36): upload a JSONL file of requests, get a file of responses.
//!
//! `POST /v1/files` (multipart, `purpose=batch`), `GET /v1/files[/{id}[/content]]`, `DELETE /v1/files/{id}`,
//! `POST /v1/batches`, `GET /v1/batches[/{id}]`, `POST /v1/batches/{id}/cancel` — the shapes the OpenAI SDK
//! reads (`client.files.create`, `client.batches.create/retrieve/cancel/list`, `client.files.content`).
//!
//! Each line (`{"custom_id", "method": "POST", "url", "body"}`) is sent to this server's own HTTP endpoint,
//! so a batch line is answered by exactly the code that answers a live request — continuous batching, the
//! prompt cache, constrained decoding, energy attribution — with up to `max_batch` lines in flight at once.
//! Output lines keep the input order. A response that is not 2xx goes to the error file with its status
//! and body, as OpenAI's does. The batch object also carries what OpenAI's does not: the joules the batch
//! cost (`ferric_energy`, summed from each response's `energy`).
//!
//! Validation fails the whole batch before anything runs (OpenAI's behaviour): a line that is not JSON, a
//! missing or repeated `custom_id`, a method other than POST, a `url` other than the batch's endpoint, a
//! streamed request. Files and batches live in memory: a restart forgets them (said, not hidden: the
//! `ferric_note` on every batch object).
use crate::write_json;
use serde_json::{json, Value};
use std::io::{Read, Write};
use std::net::TcpStream;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

const ENDPOINTS: [&str; 3] = ["/v1/chat/completions", "/v1/completions", "/v1/embeddings"];

struct File { id: String, bytes: Arc<Vec<u8>>, filename: String, purpose: String, created_at: u64 }
struct Batch { v: Value, cancel: Arc<AtomicBool> }

#[derive(Default)]
struct Store { files: Vec<File>, batches: Vec<Batch> }

static STORE: OnceLock<Mutex<Store>> = OnceLock::new();
/// Where this server listens (for the lines it sends itself), its API key, and how many lines run at once.
static SELF: OnceLock<(String, Option<String>, usize)> = OnceLock::new();

fn store() -> std::sync::MutexGuard<'static, Store> { STORE.get_or_init(Default::default).lock().unwrap_or_else(|e| e.into_inner()) }

/// Called once at startup with the bound address.
pub(crate) fn init(host: &str, port: u16, api_key: Option<String>, workers: usize) {
    let h = if host == "0.0.0.0" || host == "::" || host == "[::]" { "127.0.0.1" } else { host };
    let _ = SELF.set((format!("{h}:{port}"), api_key, workers.max(1)));
}

fn now() -> u64 { std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0) }

fn new_id(prefix: &str) -> String {
    use std::hash::{BuildHasher, Hasher};
    static SEED: OnceLock<std::collections::hash_map::RandomState> = OnceLock::new();
    static N: AtomicU64 = AtomicU64::new(0);
    let mut h = SEED.get_or_init(Default::default).build_hasher();
    h.write_u64(N.fetch_add(1, Ordering::Relaxed));
    h.write_u128(std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_nanos()).unwrap_or(0));
    format!("{prefix}{:016x}", h.finish())
}

/// Whether this module answers `method path`.
pub(crate) fn handles(path: &str) -> bool {
    path == "/v1/files" || path.starts_with("/v1/files/") || path == "/v1/batches" || path.starts_with("/v1/batches/")
}

fn not_found(s: &mut TcpStream, what: &str) {
    write_json(s, 404, &json!({"error": {"message": format!("no such {what}"), "type": "invalid_request_error", "code": "not_found"}}));
}
fn bad(s: &mut TcpStream, m: &str) { write_json(s, 400, &json!({"error": {"message": m, "type": "invalid_request_error"}})); }

fn file_json(f: &File) -> Value {
    json!({"id": f.id, "object": "file", "bytes": f.bytes.len(), "created_at": f.created_at, "filename": f.filename,
           "purpose": f.purpose, "status": "processed", "status_details": Value::Null})
}

pub(crate) fn handle(method: &str, path: &str, headers: &[(String, String)], body: &[u8], s: &mut TcpStream) {
    let parts: Vec<&str> = path.trim_start_matches('/').split('/').collect(); // ["v1", "files"|"batches", id?, verb?]
    match (method, parts.get(1).copied(), parts.get(2).copied(), parts.get(3).copied()) {
        ("POST", Some("files"), None, None) => upload(headers, body, s),
        ("GET", Some("files"), None, None) => {
            let st = store();
            let data: Vec<Value> = st.files.iter().rev().map(file_json).collect();
            write_json(s, 200, &json!({"object": "list", "data": data, "has_more": false}));
        }
        ("GET", Some("files"), Some(id), None) => match store().files.iter().find(|f| f.id == id) {
            Some(f) => write_json(s, 200, &file_json(f)),
            None => not_found(s, "file"),
        },
        ("GET", Some("files"), Some(id), Some("content")) => {
            let bytes = store().files.iter().find(|f| f.id == id).map(|f| f.bytes.clone());
            match bytes {
                Some(b) => {
                    let _ = s.write_all(format!("HTTP/1.1 200 OK\r\nContent-Type: application/jsonl\r\nContent-Length: {}\r\nAccess-Control-Allow-Origin: *\r\nConnection: close\r\n\r\n", b.len()).as_bytes());
                    let _ = s.write_all(&b);
                    let _ = s.flush();
                }
                None => not_found(s, "file"),
            }
        }
        ("DELETE", Some("files"), Some(id), None) => {
            let mut st = store();
            let n = st.files.len();
            st.files.retain(|f| f.id != id);
            if st.files.len() < n { write_json(s, 200, &json!({"id": id, "object": "file", "deleted": true})) } else { not_found(s, "file") }
        }
        ("POST", Some("batches"), None, None) => create(body, s),
        ("GET", Some("batches"), None, None) => {
            let st = store();
            let data: Vec<Value> = st.batches.iter().rev().map(|b| b.v.clone()).collect();
            let (first, last) = (data.first().map(|b| b["id"].clone()), data.last().map(|b| b["id"].clone()));
            write_json(s, 200, &json!({"object": "list", "data": data, "first_id": first, "last_id": last, "has_more": false}));
        }
        ("GET", Some("batches"), Some(id), None) => match store().batches.iter().find(|b| b.v["id"] == id) {
            Some(b) => write_json(s, 200, &b.v),
            None => not_found(s, "batch"),
        },
        ("POST", Some("batches"), Some(id), Some("cancel")) => {
            let mut st = store();
            match st.batches.iter_mut().find(|b| b.v["id"] == id) {
                Some(b) => {
                    if matches!(b.v["status"].as_str(), Some("validating" | "in_progress")) {
                        b.cancel.store(true, Ordering::SeqCst);
                        b.v["status"] = json!("cancelling");
                        b.v["cancelling_at"] = json!(now());
                    }
                    write_json(s, 200, &b.v);
                }
                None => not_found(s, "batch"),
            }
        }
        _ => write_json(s, 405, &json!({"error": {"message": format!("{method} {path} is not a files or batches operation"), "type": "invalid_request_error"}})),
    }
}

fn upload(headers: &[(String, String)], body: &[u8], s: &mut TcpStream) {
    let ct = headers.iter().find(|(k, _)| k == "content-type").map(|(_, v)| v.as_str()).unwrap_or("");
    if !ct.starts_with("multipart/form-data") { return bad(s, "POST /v1/files takes multipart/form-data with `file` and `purpose`"); }
    let parts = match crate::audio::multipart(body, ct) { Ok(p) => p, Err(e) => return bad(s, &e) };
    let Some(file) = parts.iter().find(|p| p.name == "file") else { return bad(s, "no `file` part") };
    let purpose = parts.iter().find(|p| p.name == "purpose").map(|p| String::from_utf8_lossy(p.data).trim().to_string()).unwrap_or_default();
    if purpose != "batch" { return bad(s, &format!("purpose {purpose:?}: only \"batch\" files are kept here")); }
    let f = File { id: new_id("file-"), bytes: Arc::new(file.data.to_vec()), filename: file.filename.clone().unwrap_or_else(|| "upload.jsonl".into()),
                   purpose, created_at: now() };
    let v = file_json(&f);
    store().files.push(f);
    write_json(s, 200, &v);
}

/// One validated input line.
#[derive(Debug)]
struct Line { custom_id: String, url: String, body: Value }

/// Check every line before anything runs. Errors are OpenAI's shape: `{code, message, line}` (1-based).
fn validate(bytes: &[u8], endpoint: &str) -> Result<Vec<Line>, Vec<Value>> {
    let text = String::from_utf8_lossy(bytes);
    let (mut lines, mut errs, mut seen) = (Vec::new(), Vec::new(), std::collections::HashSet::new());
    for (i, l) in text.lines().enumerate().filter(|(_, l)| !l.trim().is_empty()) {
        let e = |code: &str, m: String| json!({"code": code, "message": m, "param": Value::Null, "line": i + 1});
        let v: Value = match serde_json::from_str(l) { Ok(v) => v, Err(err) => { errs.push(e("invalid_json_line", format!("not JSON: {err}"))); continue } };
        let Some(cid) = v["custom_id"].as_str() else { errs.push(e("missing_custom_id", "`custom_id` (a string) is required".into())); continue };
        if !seen.insert(cid.to_string()) { errs.push(e("duplicate_custom_id", format!("custom_id {cid:?} appears more than once"))); continue; }
        if v["method"].as_str() != Some("POST") { errs.push(e("invalid_method", "`method` must be \"POST\"".into())); continue; }
        let url = v["url"].as_str().unwrap_or("");
        if url != endpoint { errs.push(e("mismatched_url", format!("`url` {url:?} is not the batch's endpoint {endpoint:?}"))); continue; }
        if !v["body"].is_object() { errs.push(e("invalid_body", "`body` must be an object".into())); continue; }
        if v["body"]["stream"].as_bool() == Some(true) { errs.push(e("invalid_body", "a batch line cannot stream".into())); continue; }
        lines.push(Line { custom_id: cid.to_string(), url: url.to_string(), body: v["body"].clone() });
    }
    if lines.is_empty() && errs.is_empty() { errs.push(json!({"code": "empty_file", "message": "the input file has no requests", "param": Value::Null, "line": Value::Null})); }
    if errs.is_empty() { Ok(lines) } else { Err(errs) }
}

fn create(body: &[u8], s: &mut TcpStream) {
    let req: Value = match serde_json::from_slice(body) { Ok(v) => v, Err(e) => return bad(s, &format!("bad json: {e}")) };
    let endpoint = req["endpoint"].as_str().unwrap_or("").to_string();
    if !ENDPOINTS.contains(&endpoint.as_str()) { return bad(s, &format!("endpoint {endpoint:?}: {} are served", ENDPOINTS.join(", "))); }
    let Some(input) = req["input_file_id"].as_str() else { return bad(s, "`input_file_id` is required") };
    let bytes = match store().files.iter().find(|f| f.id == input) { Some(f) => f.bytes.clone(), None => return not_found(s, "input file") };
    let window = req["completion_window"].as_str().unwrap_or("24h").to_string();
    let id = new_id("batch_");
    let cancel = Arc::new(AtomicBool::new(false));
    let v = json!({
        "id": id, "object": "batch", "endpoint": endpoint, "errors": Value::Null, "input_file_id": input,
        "completion_window": window, "status": "validating", "output_file_id": Value::Null, "error_file_id": Value::Null,
        "created_at": now(), "in_progress_at": Value::Null, "expires_at": now() + 86400, "finalizing_at": Value::Null,
        "completed_at": Value::Null, "failed_at": Value::Null, "expired_at": Value::Null, "cancelling_at": Value::Null,
        "cancelled_at": Value::Null, "request_counts": {"total": 0, "completed": 0, "failed": 0},
        "metadata": if req["metadata"].is_object() { req["metadata"].clone() } else { Value::Null },
        "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
        "ferric_energy": {"joules": Value::Null, "responses_attributed": 0},
        "ferric_note": "files and batches are kept in memory: a server restart forgets them",
    });
    store().batches.push(Batch { v: v.clone(), cancel: cancel.clone() });
    write_json(s, 200, &v);
    std::thread::spawn(move || run(id, endpoint, bytes, cancel));
}

fn update(id: &str, f: impl FnOnce(&mut Value)) {
    if let Some(b) = store().batches.iter_mut().find(|b| b.v["id"] == id) { f(&mut b.v) }
}

/// POST one line to this server; (status, body).
fn send(line: &Line) -> Result<(u16, Value), String> {
    let (addr, key, _) = SELF.get().ok_or("the batch runner has no server address")?;
    let body = serde_json::to_vec(&line.body).map_err(|e| e.to_string())?;
    let mut s = TcpStream::connect(addr).map_err(|e| e.to_string())?;
    let _ = s.set_read_timeout(Some(std::time::Duration::from_secs(3600)));
    let auth = key.as_ref().map(|k| format!("Authorization: Bearer {k}\r\n")).unwrap_or_default();
    s.write_all(format!("POST {} HTTP/1.1\r\nHost: {addr}\r\nContent-Type: application/json\r\nContent-Length: {}\r\n{auth}Connection: close\r\n\r\n", line.url, body.len()).as_bytes())
        .and_then(|_| s.write_all(&body)).map_err(|e| e.to_string())?;
    let mut raw = Vec::new();
    s.read_to_end(&mut raw).map_err(|e| e.to_string())?;
    let split = raw.windows(4).position(|w| w == b"\r\n\r\n").ok_or("no HTTP header terminator")?;
    let head = String::from_utf8_lossy(&raw[..split]);
    let code = head.split_whitespace().nth(1).and_then(|c| c.parse::<u16>().ok()).ok_or("no HTTP status")?;
    let v = serde_json::from_slice(&raw[split + 4..]).unwrap_or_else(|_| json!(String::from_utf8_lossy(&raw[split + 4..])));
    Ok((code, v))
}

fn run(id: String, endpoint: String, bytes: Arc<Vec<u8>>, cancel: Arc<AtomicBool>) {
    let lines = match validate(&bytes, &endpoint) {
        Ok(l) => l,
        Err(errs) => {
            return update(&id, |v| { v["status"] = json!("failed"); v["failed_at"] = json!(now()); v["errors"] = json!({"object": "list", "data": errs}); });
        }
    };
    let n = lines.len();
    update(&id, |v| { v["status"] = json!("in_progress"); v["in_progress_at"] = json!(now()); v["request_counts"]["total"] = json!(n); });
    let workers = SELF.get().map(|s| s.2).unwrap_or(1).min(n);
    let lines = Arc::new(lines);
    let next = Arc::new(AtomicU64::new(0));
    let results: Arc<Mutex<Vec<Option<(bool, Value)>>>> = Arc::new(Mutex::new(vec![None; n]));
    let handles: Vec<_> = (0..workers).map(|_| {
        let (lines, next, results, cancel, id) = (lines.clone(), next.clone(), results.clone(), cancel.clone(), id.clone());
        std::thread::spawn(move || loop {
            if cancel.load(Ordering::SeqCst) { return; }
            let i = next.fetch_add(1, Ordering::SeqCst) as usize;
            if i >= lines.len() { return; }
            let l = &lines[i];
            let (ok, out) = match send(l) {
                Ok((code, body)) => ((200..300).contains(&code), json!({"id": new_id("batch_req_"), "custom_id": l.custom_id,
                    "response": {"status_code": code, "request_id": new_id("req_"), "body": body}, "error": Value::Null})),
                Err(e) => (false, json!({"id": new_id("batch_req_"), "custom_id": l.custom_id, "response": Value::Null,
                    "error": {"code": "server_error", "message": e}})),
            };
            let body = out["response"]["body"].clone();
            update(&id, |v| {
                let k = if ok { "completed" } else { "failed" };
                v["request_counts"][k] = json!(v["request_counts"][k].as_u64().unwrap_or(0) + 1);
                let u = &body["usage"];
                let (inp, outp) = (u["prompt_tokens"].as_u64().or(u["input_tokens"].as_u64()).unwrap_or(0),
                                   u["completion_tokens"].as_u64().or(u["output_tokens"].as_u64()).unwrap_or(0));
                for (key, add) in [("input_tokens", inp), ("output_tokens", outp), ("total_tokens", inp + outp)] {
                    v["usage"][key] = json!(v["usage"][key].as_u64().unwrap_or(0) + add);
                }
                if let Some(j) = body["energy"]["joules"].as_f64() {
                    v["ferric_energy"]["joules"] = json!(v["ferric_energy"]["joules"].as_f64().unwrap_or(0.0) + j);
                    v["ferric_energy"]["responses_attributed"] = json!(v["ferric_energy"]["responses_attributed"].as_u64().unwrap_or(0) + 1);
                }
            });
            results.lock().unwrap_or_else(|e| e.into_inner())[i] = Some((ok, out));
        })
    }).collect();
    for h in handles { let _ = h.join(); }
    update(&id, |v| { v["status"] = json!("finalizing"); v["finalizing_at"] = json!(now()); });
    let results = results.lock().unwrap_or_else(|e| e.into_inner());
    let (mut out, mut err) = (Vec::new(), Vec::new());
    for (ok, line) in results.iter().flatten() {
        let buf = if *ok { &mut out } else { &mut err };
        buf.extend_from_slice(&serde_json::to_vec(line).unwrap_or_default());
        buf.push(b'\n');
    }
    let mut keep = |bytes: Vec<u8>, name: &str| -> Value {
        if bytes.is_empty() { return Value::Null; }
        let f = File { id: new_id("file-"), bytes: Arc::new(bytes), filename: format!("{id}_{name}.jsonl"), purpose: "batch_output".into(), created_at: now() };
        let fid = f.id.clone();
        store().files.push(f);
        json!(fid)
    };
    let (o, e) = (keep(out, "output"), keep(err, "error"));
    let cancelled = cancel.load(Ordering::SeqCst);
    update(&id, |v| {
        v["output_file_id"] = o;
        v["error_file_id"] = e;
        if cancelled { v["status"] = json!("cancelled"); v["cancelled_at"] = json!(now()); }
        else { v["status"] = json!("completed"); v["completed_at"] = json!(now()); }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_batch_file_is_validated_whole_before_anything_runs() {
        let ok = br#"{"custom_id": "a", "method": "POST", "url": "/v1/chat/completions", "body": {"messages": []}}
{"custom_id": "b", "method": "POST", "url": "/v1/chat/completions", "body": {"messages": []}}
"#;
        let l = validate(ok, "/v1/chat/completions").unwrap();
        assert_eq!(l.iter().map(|l| l.custom_id.as_str()).collect::<Vec<_>>(), ["a", "b"]);
        let bad = br#"{"custom_id": "a", "method": "POST", "url": "/v1/chat/completions", "body": {}}
not json
{"custom_id": "a", "method": "POST", "url": "/v1/chat/completions", "body": {}}
{"custom_id": "c", "method": "GET", "url": "/v1/chat/completions", "body": {}}
{"custom_id": "d", "method": "POST", "url": "/v1/completions", "body": {}}
{"custom_id": "e", "method": "POST", "url": "/v1/chat/completions", "body": {"stream": true}}
{"method": "POST", "url": "/v1/chat/completions", "body": {}}
"#;
        let e = validate(bad, "/v1/chat/completions").unwrap_err();
        let codes: Vec<(&str, u64)> = e.iter().map(|x| (x["code"].as_str().unwrap(), x["line"].as_u64().unwrap())).collect();
        assert_eq!(codes, [("invalid_json_line", 2), ("duplicate_custom_id", 3), ("invalid_method", 4), ("mismatched_url", 5), ("invalid_body", 6), ("missing_custom_id", 7)]);
        assert_eq!(validate(b"\n\n", "/v1/completions").unwrap_err()[0]["code"], "empty_file");
    }
}
