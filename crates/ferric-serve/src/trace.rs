//! **OpenTelemetry traces** (parity gap O09): one span per request, exported over OTLP/HTTP to whatever
//! collector `OTEL_EXPORTER_OTLP_ENDPOINT` names — Jaeger, Tempo, Langfuse, Phoenix, an OTel Collector.
//!
//! Configured the way every OpenTelemetry SDK is, by the standard environment variables:
//! `OTEL_EXPORTER_OTLP_ENDPOINT` (the base; `/v1/traces` is appended) or `OTEL_EXPORTER_OTLP_TRACES_ENDPOINT`
//! (used as is), `OTEL_EXPORTER_OTLP_[TRACES_]PROTOCOL` (`http/json`, the default here, or
//! `http/protobuf`), `OTEL_EXPORTER_OTLP_[TRACES_]HEADERS` (`k=v,k2=v2`), `OTEL_SERVICE_NAME`,
//! `OTEL_RESOURCE_ATTRIBUTES`, `OTEL_SDK_DISABLED`. `--otlp <url>` on the command line sets the endpoint.
//! Refused at startup, never half-done: `grpc` (no gRPC stack here) and `https://` (no TLS is linked —
//! run a collector beside the server and point it at that).
//!
//! A request carrying a W3C `traceparent` header continues that trace: its span is the child of the
//! caller's. Span attributes follow the OpenTelemetry GenAI semantic conventions (`gen_ai.operation.name`,
//! `gen_ai.request.*`, `gen_ai.usage.input_tokens` / `output_tokens`, `gen_ai.response.finish_reasons`),
//! with vLLM's latency names (`gen_ai.latency.time_to_first_token`, `gen_ai.latency.e2e`) — and, what no
//! other server's spans carry, the joules the request cost (`ferric.energy.*`, the same numbers as the
//! response's `energy` field).
//!
//! A span lives with its request: it starts when the request is read, collects what the generation
//! reports, and is exported when it is dropped — so an error path that returns early still reports.
//! Export is a background thread batching up to 128 spans or 1 s; a collector that is down costs a
//! log line a minute, never a request.

use serde_json::{json, Value};
use std::io::{Read, Write};
use std::sync::mpsc::{Receiver, RecvTimeoutError, SyncSender};
use std::sync::OnceLock;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

/// An attribute value (the OTLP `AnyValue` shapes used here).
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum Attr { S(String), I(i64), F(f64), B(bool), Arr(Vec<Attr>) }

#[derive(Debug)]
pub(crate) struct SpanData {
    pub trace_id: [u8; 16],
    pub span_id: [u8; 8],
    pub parent: Option<[u8; 8]>,
    pub name: String,
    pub start: u64,
    pub end: u64,
    pub attrs: Vec<(String, Attr)>,
    pub events: Vec<(u64, String)>,
    /// None = unset; Some(true) = OK; Some(false) = ERROR.
    pub ok: Option<bool>,
}

/// A request's span. Dropping it exports it.
pub(crate) struct Span {
    d: SpanData,
    op: Option<&'static str>,
    model: Option<String>,
    first: Option<u64>,
    input: usize,
    output: usize,
    finishes: Vec<String>,
    joules: Option<f64>,
    energy: Vec<(String, Attr)>,
    status: Option<u16>,
}

fn now_ns() -> u64 { SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_nanos() as u64).unwrap_or(0) }

/// Random bytes without a random-number crate: std's per-process random hash keys, a counter and the clock.
fn random<const N: usize>() -> [u8; N] {
    use std::hash::{BuildHasher, Hasher};
    static SEED: OnceLock<std::collections::hash_map::RandomState> = OnceLock::new();
    static COUNT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let s = SEED.get_or_init(std::collections::hash_map::RandomState::new);
    let mut out = [0u8; N];
    for (i, c) in out.chunks_mut(8).enumerate() {
        let mut h = s.build_hasher();
        h.write_u64(COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed));
        h.write_u64(now_ns());
        h.write_usize(i);
        c.copy_from_slice(&h.finish().to_le_bytes()[..c.len()]);
    }
    if out.iter().all(|&b| b == 0) { out[0] = 1; } // all-zero ids are invalid
    out
}

fn hex(b: &[u8]) -> String { b.iter().map(|x| format!("{x:02x}")).collect() }

fn unhex<const N: usize>(s: &str) -> Option<[u8; N]> {
    if s.len() != 2 * N { return None; }
    let mut o = [0u8; N];
    for (i, x) in o.iter_mut().enumerate() { *x = u8::from_str_radix(s.get(2 * i..2 * i + 2)?, 16).ok()?; }
    Some(o).filter(|o| o.iter().any(|&b| b != 0))
}

/// W3C `traceparent`: `00-<32 hex trace id>-<16 hex parent id>-<flags>`. Anything else is ignored (a new
/// trace starts), as the W3C spec says to.
pub(crate) fn parse_traceparent(v: &str) -> Option<([u8; 16], [u8; 8])> {
    let p: Vec<&str> = v.trim().split('-').collect();
    if p.len() < 4 || p[0].len() != 2 || p[0] == "ff" || p[3].len() != 2 { return None; }
    if p[0] == "00" && p.len() != 4 { return None; }
    Some((unhex::<16>(p[1])?, unhex::<8>(p[2])?))
}

/// What a route is, in GenAI terms.
fn operation(path: &str) -> Option<&'static str> {
    Some(match path {
        "/v1/chat/completions" | "/api/chat" | "/v1/messages" | "/v1/responses" => "chat",
        "/v1/completions" | "/api/generate" => "text_completion",
        "/v1/embeddings" | "/api/embed" | "/api/embeddings" => "embeddings",
        _ => return None,
    })
}

impl Span {
    /// A span for a request just read, or None when tracing is off.
    pub fn begin(method: &str, path: &str, headers: &[(String, String)], body: &[u8]) -> Option<Span> {
        exporter()?;
        Some(Span::new(method, path, headers, body))
    }

    fn new(method: &str, path: &str, headers: &[(String, String)], body: &[u8]) -> Span {
        let (trace_id, parent) = headers.iter().find(|(k, _)| k == "traceparent")
            .and_then(|(_, v)| parse_traceparent(v)).map(|(t, p)| (t, Some(p))).unwrap_or_else(|| (random(), None));
        let op = operation(path).filter(|_| method == "POST");
        let mut attrs = vec![("http.request.method".to_string(), Attr::S(method.to_string())), ("url.path".into(), Attr::S(path.to_string()))];
        let mut model = None;
        if let Some(op) = op {
            let req: Value = serde_json::from_slice(body).unwrap_or(Value::Null);
            let o = &req["options"]; // Ollama's sampling fields
            let num = |a: &Value, b: &Value| a.as_f64().or_else(|| b.as_f64());
            let int = |a: &Value, b: &Value| a.as_i64().or_else(|| b.as_i64());
            attrs.push(("gen_ai.operation.name".into(), Attr::S(op.to_string())));
            model = req["model"].as_str().filter(|s| !s.is_empty()).map(String::from);
            if let Some(m) = &model { attrs.push(("gen_ai.request.model".into(), Attr::S(m.clone()))); }
            for (k, v) in [("gen_ai.request.temperature", num(&req["temperature"], &o["temperature"])),
                           ("gen_ai.request.top_p", num(&req["top_p"], &o["top_p"])),
                           ("gen_ai.request.frequency_penalty", num(&req["frequency_penalty"], &o["frequency_penalty"])),
                           ("gen_ai.request.presence_penalty", num(&req["presence_penalty"], &o["presence_penalty"]))] {
                if let Some(v) = v { attrs.push((k.into(), Attr::F(v))); }
            }
            let max = int(&req["max_completion_tokens"], &req["max_tokens"]).or_else(|| o["num_predict"].as_i64());
            for (k, v) in [("gen_ai.request.max_tokens", max), ("gen_ai.request.top_k", int(&req["top_k"], &o["top_k"])),
                           ("gen_ai.request.seed", int(&req["seed"], &o["seed"])), ("gen_ai.request.choice.count", req["n"].as_i64())] {
                if let Some(v) = v { attrs.push((k.into(), Attr::I(v))); }
            }
            if let Some(s) = req["stream"].as_bool() { attrs.push(("ferric.request.stream".into(), Attr::B(s))); }
        }
        Span {
            d: SpanData { trace_id, span_id: random(), parent, name: format!("{method} {path}"), start: now_ns(), end: 0, attrs, events: Vec::new(), ok: None },
            op, model, first: None, input: 0, output: 0, finishes: Vec::new(), joules: None, energy: Vec::new(), status: None,
        }
    }

    /// The first token reached the client.
    pub fn first_token(&mut self) {
        if self.first.is_none() { let t = now_ns(); self.first = Some(t); self.d.events.push((t, "gen_ai.first_token".into())); }
    }

    /// One generation finished (a request may run several: n > 1 choices, tool rounds).
    pub fn generation(&mut self, model: &str, input: usize, output: usize, finish: &str, energy: &Value) {
        self.model.get_or_insert_with(|| model.to_string());
        self.d.attrs.retain(|(k, _)| k != "gen_ai.response.model");
        self.d.attrs.push(("gen_ai.response.model".into(), Attr::S(model.to_string())));
        self.input = self.input.max(input);
        self.output += output;
        self.finishes.push(finish.to_string());
        if let Some(j) = energy["joules"].as_f64() { *self.joules.get_or_insert(0.0) += j; }
        self.energy.clear();
        if let Some(o) = energy.as_object() {
            for (k, v) in o {
                if k == "joules" { continue; }
                let a = match v { Value::String(s) => Attr::S(s.clone()), Value::Bool(b) => Attr::B(*b),
                    Value::Number(n) => n.as_i64().map(Attr::I).unwrap_or_else(|| Attr::F(n.as_f64().unwrap_or(0.0))), _ => continue };
                self.energy.push((format!("ferric.energy.{k}"), a));
            }
        }
    }

    pub fn status(&mut self, code: u16) { self.status = Some(code); }
    pub fn cancelled(&mut self) { self.d.attrs.push(("ferric.cancelled".into(), Attr::B(true))); }

    /// The finished span, as exported.
    fn finish(&mut self) -> SpanData {
        let empty = SpanData { trace_id: [0; 16], span_id: [0; 8], parent: None, name: String::new(), start: 0, end: 0, attrs: Vec::new(), events: Vec::new(), ok: None };
        let mut d = std::mem::replace(&mut self.d, empty);
        d.end = now_ns();
        if let Some(op) = self.op {
            d.name = match &self.model { Some(m) => format!("{op} {m}"), None => op.to_string() };
            if !self.finishes.is_empty() {
                d.attrs.push(("gen_ai.usage.input_tokens".into(), Attr::I(self.input as i64)));
                d.attrs.push(("gen_ai.usage.output_tokens".into(), Attr::I(self.output as i64)));
                d.attrs.push(("gen_ai.response.finish_reasons".into(), Attr::Arr(self.finishes.iter().map(|f| Attr::S(f.clone())).collect())));
            }
            if let Some(f) = self.first { d.attrs.push(("gen_ai.latency.time_to_first_token".into(), Attr::F(f.saturating_sub(d.start) as f64 * 1e-9))); }
            d.attrs.push(("gen_ai.latency.e2e".into(), Attr::F(d.end.saturating_sub(d.start) as f64 * 1e-9)));
            if let Some(j) = self.joules { d.attrs.push(("ferric.energy.joules".into(), Attr::F(j))); }
            d.attrs.append(&mut self.energy);
        }
        if let Some(c) = self.status {
            d.attrs.push(("http.response.status_code".into(), Attr::I(c as i64)));
            // Server spans: only a 5xx is an error (the OTel HTTP convention); a 4xx is the client's.
            d.ok = Some(c < 500);
        }
        d
    }
}

impl Drop for Span {
    fn drop(&mut self) {
        let d = self.finish();
        if let Some(tx) = exporter() { let _ = tx.try_send(d); }
    }
}

// ---------------------------------------------------------------------------------------------
// The serial path's current span: the handler runs synchronously on the engine thread, so the span of
// the request it is serving is a thread-local the generation code reports into.
// ---------------------------------------------------------------------------------------------

thread_local! { static CURRENT: std::cell::RefCell<Option<Span>> = const { std::cell::RefCell::new(None) }; }

pub(crate) fn enter(s: Option<Span>) { CURRENT.with(|c| *c.borrow_mut() = s); }
/// Ends the current span (exports it) when dropped — the end of routing one request, however it returns.
pub(crate) struct Current;
impl Drop for Current { fn drop(&mut self) { drop(leave()); } }
pub(crate) fn leave() -> Option<Span> { CURRENT.with(|c| c.borrow_mut().take()) }
/// Report into the current request's span, if it has one.
pub(crate) fn with(f: impl FnOnce(&mut Span)) { CURRENT.with(|c| if let Some(s) = c.borrow_mut().as_mut() { f(s) }); }

// ---------------------------------------------------------------------------------------------
// Encoding: OTLP/JSON and OTLP/protobuf, of the same ExportTraceServiceRequest.
// ---------------------------------------------------------------------------------------------

fn attr_json(a: &Attr) -> Value {
    match a {
        Attr::S(s) => json!({"stringValue": s}),
        Attr::I(i) => json!({"intValue": i.to_string()}),
        Attr::F(f) => json!({"doubleValue": f}),
        Attr::B(b) => json!({"boolValue": b}),
        Attr::Arr(v) => json!({"arrayValue": {"values": v.iter().map(attr_json).collect::<Vec<_>>()}}),
    }
}
fn kvs_json(a: &[(String, Attr)]) -> Value { Value::Array(a.iter().map(|(k, v)| json!({"key": k, "value": attr_json(v)})).collect()) }

/// OTLP/JSON: the protobuf JSON mapping, with trace and span ids as hex (the OTLP exception to base64).
pub(crate) fn encode_json(resource: &[(String, Attr)], spans: &[SpanData]) -> Vec<u8> {
    let spans: Vec<Value> = spans.iter().map(|s| {
        let mut v = json!({
            "traceId": hex(&s.trace_id), "spanId": hex(&s.span_id), "name": s.name, "kind": 2,
            "startTimeUnixNano": s.start.to_string(), "endTimeUnixNano": s.end.to_string(),
            "attributes": kvs_json(&s.attrs),
            "events": s.events.iter().map(|(t, n)| json!({"timeUnixNano": t.to_string(), "name": n})).collect::<Vec<_>>(),
        });
        if let Some(p) = s.parent { v["parentSpanId"] = json!(hex(&p)); }
        if let Some(ok) = s.ok { v["status"] = json!({"code": if ok { 1 } else { 2 }}); }
        v
    }).collect();
    serde_json::to_vec(&json!({"resourceSpans": [{
        "resource": {"attributes": kvs_json(resource)},
        "scopeSpans": [{"scope": {"name": "ferric-serve", "version": env!("CARGO_PKG_VERSION")}, "spans": spans}],
    }]})).unwrap_or_default()
}

/// Protobuf wire format, the few pieces OTLP needs.
struct Pb(Vec<u8>);
impl Pb {
    fn varint(&mut self, mut v: u64) { loop { let b = (v & 0x7f) as u8; v >>= 7; if v == 0 { self.0.push(b); break; } self.0.push(b | 0x80); } }
    fn key(&mut self, field: u32, wire: u8) { self.varint(((field as u64) << 3) | wire as u64); }
    fn bytes(&mut self, field: u32, b: &[u8]) { self.key(field, 2); self.varint(b.len() as u64); self.0.extend_from_slice(b); }
    fn msg(&mut self, field: u32, f: impl FnOnce(&mut Pb)) { let mut m = Pb(Vec::new()); f(&mut m); self.bytes(field, &m.0); }
    fn fixed64(&mut self, field: u32, v: u64) { self.key(field, 1); self.0.extend_from_slice(&v.to_le_bytes()); }
    fn int(&mut self, field: u32, v: u64) { self.key(field, 0); self.varint(v); }
}
fn any_pb(p: &mut Pb, a: &Attr) {
    match a {
        Attr::S(s) => p.bytes(1, s.as_bytes()),
        Attr::B(b) => p.int(2, *b as u64),
        Attr::I(i) => p.int(3, *i as u64),
        Attr::F(f) => p.fixed64(4, f.to_bits()),
        Attr::Arr(v) => p.msg(5, |m| for x in v { m.msg(1, |e| any_pb(e, x)) }),
    }
}
fn kv_pb(p: &mut Pb, field: u32, a: &[(String, Attr)]) {
    for (k, v) in a { p.msg(field, |m| { m.bytes(1, k.as_bytes()); m.msg(2, |x| any_pb(x, v)); }); }
}

/// OTLP/protobuf: `opentelemetry.proto.collector.trace.v1.ExportTraceServiceRequest`.
pub(crate) fn encode_protobuf(resource: &[(String, Attr)], spans: &[SpanData]) -> Vec<u8> {
    let mut p = Pb(Vec::new());
    p.msg(1, |rs| {                                   // ResourceSpans
        rs.msg(1, |r| kv_pb(r, 1, resource));         // Resource.attributes
        rs.msg(2, |ss| {                              // ScopeSpans
            ss.msg(1, |sc| { sc.bytes(1, b"ferric-serve"); sc.bytes(2, env!("CARGO_PKG_VERSION").as_bytes()); });
            for s in spans {
                ss.msg(2, |m| {                       // Span
                    m.bytes(1, &s.trace_id);
                    m.bytes(2, &s.span_id);
                    if let Some(pid) = s.parent { m.bytes(4, &pid); }
                    m.bytes(5, s.name.as_bytes());
                    m.int(6, 2);                      // SPAN_KIND_SERVER
                    m.fixed64(7, s.start);
                    m.fixed64(8, s.end);
                    kv_pb(m, 9, &s.attrs);
                    for (t, n) in &s.events { m.msg(11, |e| { e.fixed64(1, *t); e.bytes(2, n.as_bytes()); }); }
                    if let Some(ok) = s.ok { m.msg(15, |st| st.int(3, if ok { 1 } else { 2 })); }
                });
            }
        });
    });
    p.0
}

// ---------------------------------------------------------------------------------------------
// Configuration and the exporter thread.
// ---------------------------------------------------------------------------------------------

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Config { pub host: String, pub port: u16, pub path: String, pub protobuf: bool, pub headers: Vec<(String, String)>, pub resource: Vec<(String, Attr)> }

fn pct_decode(s: &str) -> String {
    let b = s.as_bytes();
    let mut o = Vec::with_capacity(b.len());
    let mut i = 0;
    while i < b.len() {
        if b[i] == b'%' && i + 3 <= b.len() {
            if let Some(v) = std::str::from_utf8(&b[i + 1..i + 3]).ok().and_then(|h| u8::from_str_radix(h, 16).ok()) { o.push(v); i += 3; continue; }
        }
        o.push(b[i]);
        i += 1;
    }
    String::from_utf8_lossy(&o).into_owned()
}

fn pairs(s: &str) -> Vec<(String, String)> {
    s.split(',').filter_map(|kv| kv.split_once('=')).map(|(k, v)| (pct_decode(k.trim()), pct_decode(v.trim()))).filter(|(k, _)| !k.is_empty()).collect()
}

/// The exporter configuration from the environment (`get` reads a variable) and the `--otlp` flag. None =
/// tracing off; Err = asked for something this build cannot do.
pub(crate) fn config(get: &dyn Fn(&str) -> Option<String>, flag: Option<&str>) -> Result<Option<Config>, String> {
    if get("OTEL_SDK_DISABLED").is_some_and(|v| v.trim().eq_ignore_ascii_case("true")) { return Ok(None); }
    let get = |k: &str| get(k).filter(|v| !v.trim().is_empty());
    let url = match (flag, get("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT"), get("OTEL_EXPORTER_OTLP_ENDPOINT")) {
        (Some(f), _, _) => format!("{}/v1/traces", f.trim_end_matches('/')),
        (None, Some(t), _) => t,
        (None, None, Some(b)) => format!("{}/v1/traces", b.trim_end_matches('/')),
        _ => return Ok(None),
    };
    let proto = get("OTEL_EXPORTER_OTLP_TRACES_PROTOCOL").or_else(|| get("OTEL_EXPORTER_OTLP_PROTOCOL")).unwrap_or_else(|| "http/json".into());
    let protobuf = match proto.trim() {
        "http/json" => false,
        "http/protobuf" => true,
        p => return Err(format!("OTLP protocol {p:?}: http/json and http/protobuf are served (no gRPC stack is linked)")),
    };
    let rest = url.trim().strip_prefix("http://").ok_or_else(|| format!("OTLP endpoint {url:?}: only http:// is served (no TLS is linked) — run a collector beside the server"))?;
    let (hostport, path) = match rest.find('/') { Some(i) => (&rest[..i], rest[i..].to_string()), None => (rest, "/".to_string()) };
    let (host, port) = if let Some(h6) = hostport.strip_prefix('[') {
        let (h, p) = h6.split_once(']').ok_or_else(|| format!("OTLP endpoint {url:?}: bad host"))?;
        (h.to_string(), match p.strip_prefix(':') { Some(p) => p.parse::<u16>().map_err(|_| format!("OTLP endpoint {url:?}: bad port"))?, None => 80 })
    } else {
        match hostport.rsplit_once(':') {
            Some((h, p)) => (h.to_string(), p.parse::<u16>().map_err(|_| format!("OTLP endpoint {url:?}: bad port"))?),
            None => (hostport.to_string(), 80),
        }
    };
    let headers = pairs(&get("OTEL_EXPORTER_OTLP_TRACES_HEADERS").or_else(|| get("OTEL_EXPORTER_OTLP_HEADERS")).unwrap_or_default());
    let mut resource: Vec<(String, Attr)> = pairs(&get("OTEL_RESOURCE_ATTRIBUTES").unwrap_or_default()).into_iter().map(|(k, v)| (k, Attr::S(v))).collect();
    let service = get("OTEL_SERVICE_NAME")
        .or_else(|| resource.iter().find(|(k, _)| k == "service.name").and_then(|(_, v)| if let Attr::S(s) = v { Some(s.clone()) } else { None }))
        .unwrap_or_else(|| "ferric-serve".into());
    resource.retain(|(k, _)| k != "service.name");
    resource.insert(0, ("service.name".into(), Attr::S(service)));
    resource.push(("telemetry.sdk.name".into(), Attr::S("ferric-serve".into())));
    resource.push(("telemetry.sdk.language".into(), Attr::S("rust".into())));
    resource.push(("telemetry.sdk.version".into(), Attr::S(env!("CARGO_PKG_VERSION").into())));
    Ok(Some(Config { host, port, path, protobuf, headers, resource }))
}

static EXPORTER: OnceLock<Option<SyncSender<SpanData>>> = OnceLock::new();

fn exporter() -> Option<&'static SyncSender<SpanData>> { EXPORTER.get().and_then(|x| x.as_ref()) }

/// Start exporting (once, at startup). Returns what is being done, for the startup log.
pub(crate) fn init(cfg: Option<Config>) -> Option<String> {
    let Some(cfg) = cfg else { let _ = EXPORTER.set(None); return None };
    let (tx, rx) = std::sync::mpsc::sync_channel::<SpanData>(4096);
    let what = format!("OTLP traces to http://{}:{}{} ({})", cfg.host, cfg.port, cfg.path, if cfg.protobuf { "http/protobuf" } else { "http/json" });
    std::thread::spawn(move || export_loop(cfg, rx));
    let _ = EXPORTER.set(Some(tx));
    Some(what)
}

fn export_loop(cfg: Config, rx: Receiver<SpanData>) {
    let mut last_warn: Option<std::time::Instant> = None;
    loop {
        let first = match rx.recv() { Ok(s) => s, Err(_) => return };
        let mut batch = vec![first];
        let deadline = std::time::Instant::now() + Duration::from_secs(1);
        while batch.len() < 128 {
            let left = deadline.saturating_duration_since(std::time::Instant::now());
            match rx.recv_timeout(left) {
                Ok(s) => batch.push(s),
                Err(RecvTimeoutError::Timeout) => break,
                Err(RecvTimeoutError::Disconnected) => { let _ = post(&cfg, &batch); return; }
            }
        }
        if let Err(e) = post(&cfg, &batch) {
            if last_warn.is_none_or(|t| t.elapsed() > Duration::from_secs(60)) {
                eprintln!("ferric-serve: OTLP export of {} span(s) failed: {e}", batch.len());
                last_warn = Some(std::time::Instant::now());
            }
        }
    }
}

fn post(cfg: &Config, spans: &[SpanData]) -> Result<(), String> {
    use std::net::ToSocketAddrs;
    let body = if cfg.protobuf { encode_protobuf(&cfg.resource, spans) } else { encode_json(&cfg.resource, spans) };
    let addr = (cfg.host.as_str(), cfg.port).to_socket_addrs().map_err(|e| e.to_string())?.next().ok_or("no address")?;
    let mut s = std::net::TcpStream::connect_timeout(&addr, Duration::from_secs(5)).map_err(|e| e.to_string())?;
    let _ = s.set_read_timeout(Some(Duration::from_secs(10)));
    let _ = s.set_write_timeout(Some(Duration::from_secs(10)));
    let host = if cfg.host.contains(':') { format!("[{}]", cfg.host) } else { cfg.host.clone() };
    let mut head = format!("POST {} HTTP/1.1\r\nHost: {host}:{}\r\nContent-Type: {}\r\nContent-Length: {}\r\nConnection: close\r\n",
        cfg.path, cfg.port, if cfg.protobuf { "application/x-protobuf" } else { "application/json" }, body.len());
    for (k, v) in &cfg.headers { head.push_str(&format!("{k}: {v}\r\n")); }
    head.push_str("\r\n");
    s.write_all(head.as_bytes()).and_then(|_| s.write_all(&body)).map_err(|e| e.to_string())?;
    let mut resp = [0u8; 64];
    let n = s.read(&mut resp).map_err(|e| e.to_string())?;
    let line = String::from_utf8_lossy(&resp[..n]);
    let code = line.split_whitespace().nth(1).and_then(|c| c.parse::<u16>().ok()).ok_or_else(|| format!("no HTTP status in {line:?}"))?;
    if (200..300).contains(&code) { Ok(()) } else { Err(format!("collector answered {code}")) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn traceparent_is_read_per_w3c() {
        let (t, p) = parse_traceparent("00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01").unwrap();
        assert_eq!(hex(&t), "4bf92f3577b34da6a3ce929d0e0e4736");
        assert_eq!(hex(&p), "00f067aa0ba902b7");
        for bad in ["00-00000000000000000000000000000000-00f067aa0ba902b7-01", "00-4bf92f3577b34da6a3ce929d0e0e4736-0000000000000000-01",
                    "ff-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01", "00-4bf92f3577b34da6a3ce929d0e0e473-00f067aa0ba902b7-01",
                    "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01-extra", "garbage"] {
            assert!(parse_traceparent(bad).is_none(), "{bad}");
        }
        // A later version may carry more fields; version 00 may not.
        assert!(parse_traceparent("01-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01-xyz").is_some());
    }

    #[test]
    fn configuration_follows_the_otel_environment_variables() {
        let env = |pairs: &'static [(&'static str, &'static str)]| move |k: &str| pairs.iter().find(|(a, _)| *a == k).map(|(_, v)| v.to_string());
        assert_eq!(config(&env(&[]), None).unwrap(), None);
        let c = config(&env(&[("OTEL_EXPORTER_OTLP_ENDPOINT", "http://collector:4318/")]), None).unwrap().unwrap();
        assert_eq!((c.host.as_str(), c.port, c.path.as_str(), c.protobuf), ("collector", 4318, "/v1/traces", false));
        let c = config(&env(&[("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", "http://h:9/custom"), ("OTEL_EXPORTER_OTLP_ENDPOINT", "http://x:1"),
                              ("OTEL_EXPORTER_OTLP_PROTOCOL", "http/protobuf"), ("OTEL_EXPORTER_OTLP_HEADERS", "authorization=Basic%20abc, x-team = ai"),
                              ("OTEL_SERVICE_NAME", "lab"), ("OTEL_RESOURCE_ATTRIBUTES", "deployment.environment=dev,service.name=ignored")]), None).unwrap().unwrap();
        assert_eq!((c.host.as_str(), c.port, c.path.as_str(), c.protobuf), ("h", 9, "/custom", true));
        assert_eq!(c.headers, vec![("authorization".into(), "Basic abc".into()), ("x-team".into(), "ai".into())]);
        assert_eq!(c.resource[0], ("service.name".into(), Attr::S("lab".into())));
        assert!(c.resource.contains(&("deployment.environment".into(), Attr::S("dev".into()))));
        let c = config(&env(&[("OTEL_EXPORTER_OTLP_ENDPOINT", "http://[::1]:4318")]), None).unwrap().unwrap();
        assert_eq!((c.host.as_str(), c.port), ("::1", 4318));
        let c = config(&env(&[("OTEL_EXPORTER_OTLP_ENDPOINT", "http://x:1")]), Some("http://127.0.0.1:4318")).unwrap().unwrap();
        assert_eq!((c.host.as_str(), c.port), ("127.0.0.1", 4318));
        assert_eq!(config(&env(&[("OTEL_EXPORTER_OTLP_ENDPOINT", "http://x:1"), ("OTEL_SDK_DISABLED", "true")]), None).unwrap(), None);
        assert!(config(&env(&[("OTEL_EXPORTER_OTLP_ENDPOINT", "https://x:1")]), None).unwrap_err().contains("TLS"));
        assert!(config(&env(&[("OTEL_EXPORTER_OTLP_ENDPOINT", "http://x:1"), ("OTEL_EXPORTER_OTLP_PROTOCOL", "grpc")]), None).unwrap_err().contains("gRPC"));
    }

    /// A span collects what its request reported: GenAI attributes from the body, a child of the caller's
    /// trace, usage summed over generations, joules summed, 4xx not an error.
    #[test]
    fn a_span_carries_the_request_and_what_it_cost() {
        let body = br#"{"model":"qwen","temperature":0.5,"max_tokens":9,"n":2,"stream":false}"#;
        let h = vec![("traceparent".to_string(), "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01".to_string())];
        let mut s = Span::new("POST", "/v1/chat/completions", &h, body);
        s.first_token();
        s.generation("qwen2.5-0.5b", 37, 5, "stop", &json!({"joules": 1.5, "joules_per_token": 0.3, "method": "macmon"}));
        s.generation("qwen2.5-0.5b", 37, 9, "length", &json!({"joules": 2.0, "joules_per_token": 0.22, "method": "macmon"}));
        s.status(200);
        let d = s.finish();
        std::mem::forget(s);
        let get = |k: &str| d.attrs.iter().find(|(a, _)| a == k).map(|(_, v)| v.clone());
        assert_eq!(d.name, "chat qwen");
        assert_eq!(hex(&d.trace_id), "4bf92f3577b34da6a3ce929d0e0e4736");
        assert_eq!(d.parent.map(|p| hex(&p)).as_deref(), Some("00f067aa0ba902b7"));
        assert_eq!(get("gen_ai.request.temperature"), Some(Attr::F(0.5)));
        assert_eq!(get("gen_ai.request.max_tokens"), Some(Attr::I(9)));
        assert_eq!(get("gen_ai.request.choice.count"), Some(Attr::I(2)));
        assert_eq!(get("gen_ai.usage.input_tokens"), Some(Attr::I(37)));
        assert_eq!(get("gen_ai.usage.output_tokens"), Some(Attr::I(14)));
        assert_eq!(get("gen_ai.response.finish_reasons"), Some(Attr::Arr(vec![Attr::S("stop".into()), Attr::S("length".into())])));
        assert_eq!(get("ferric.energy.joules"), Some(Attr::F(3.5)));
        assert_eq!(get("ferric.energy.method"), Some(Attr::S("macmon".into())));
        assert_eq!(d.ok, Some(true));
        assert!(d.events.len() == 1 && d.start <= d.events[0].0 && d.events[0].0 <= d.end);
        let mut e = Span::new("POST", "/v1/chat/completions", &[], b"{}");
        e.status(400);
        let d = e.finish();
        std::mem::forget(e);
        assert_eq!((d.ok, d.parent, d.name.as_str()), (Some(true), None, "chat"));
    }

    #[test]
    fn protobuf_encoding_has_the_otlp_field_numbers() {
        let s = SpanData { trace_id: [1; 16], span_id: [2; 8], parent: Some([3; 8]), name: "chat m".into(), start: 10, end: 20,
            attrs: vec![("a".into(), Attr::I(-1))], events: vec![(15, "e".into())], ok: Some(true) };
        let b = encode_protobuf(&[("service.name".into(), Attr::S("x".into()))], &[s]);
        assert_eq!(b[0], 0x0a); // field 1 (resource_spans), length-delimited
        let w = |needle: &[u8]| b.windows(needle.len()).any(|x| x == needle);
        // Span: trace id field 1, span id field 2, parent field 4 (bytes); kind field 6 = 2; start/end fixed64 7, 8.
        assert!(w(&[&[0x0a, 16][..], &[1; 16]].concat()) && w(&[&[0x12, 8][..], &[2; 8]].concat()) && w(&[&[0x22, 8][..], &[3; 8]].concat()));
        assert!(w(&[0x39, 10, 0, 0, 0, 0, 0, 0, 0]) && w(&[0x41, 20, 0, 0, 0, 0, 0, 0, 0]) && w(&[0x30, 2]));
        // int64 -1: a ten-byte varint in AnyValue field 3.
        assert!(w(&[0x18, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x01]));
        let j: Value = serde_json::from_slice(&encode_json(&[], &[SpanData { trace_id: [1; 16], span_id: [2; 8], parent: None, name: "n".into(), start: 1, end: 2, attrs: vec![], events: vec![], ok: None }])).unwrap();
        let sp = &j["resourceSpans"][0]["scopeSpans"][0]["spans"][0];
        assert_eq!(sp["traceId"], "01010101010101010101010101010101");
        assert_eq!(sp["startTimeUnixNano"], "1");
        assert!(sp.get("parentSpanId").is_none() && sp.get("status").is_none());
    }
}
