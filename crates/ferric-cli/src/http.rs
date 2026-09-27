//! A plain-HTTP/1.1 client for talking to a LOCAL model server — std `TcpStream`, nothing vendored.
//!
//! The server is loopback by default, so there is no TLS here; `https://` in `FERRIC_HOST` is refused
//! by name rather than sent in the clear. (Hugging Face is HTTPS and goes through `curl`, as
//! ferric-serve's own downloader does — see `pull`.)
//!
//! Three body framings are read, because the two servers this CLI is written for use different ones:
//! ferric-serve writes `Connection: close` and streams until it closes the socket (no length, no
//! chunks), while Ollama streams NDJSON with `Transfer-Encoding: chunked`. A client that only
//! understood one would print a hex chunk-size line into the middle of an answer on the other.
use serde_json::Value;
use std::io::{self, BufRead, BufReader, Read, Write};
use std::net::{TcpStream, ToSocketAddrs};
use std::time::Duration;

/// Ollama owns 11434. One above it, so both can run side by side and `FERRIC_HOST=127.0.0.1:11434`
/// points this CLI at a real Ollama without any other change.
pub const DEFAULT_PORT: u16 = 11435;

#[derive(Clone, Debug, PartialEq)]
pub struct Host { pub host: String, pub port: u16 }

impl Host {
    /// `FERRIC_HOST`, or 127.0.0.1:11435.
    pub fn from_env() -> Result<Host, String> {
        match std::env::var("FERRIC_HOST") {
            Ok(s) if !s.trim().is_empty() => Host::parse(&s),
            _ => Ok(Host { host: "127.0.0.1".into(), port: DEFAULT_PORT }),
        }
    }

    /// `host:port`, `http://host:port[/]`, `host` (default port), `:port` (loopback), `[::1]:port`.
    pub fn parse(s: &str) -> Result<Host, String> {
        let s = s.trim();
        if s.starts_with("https://") {
            return Err(format!("FERRIC_HOST={s}: this client speaks plain HTTP to a local server; use http://"));
        }
        let s = s.strip_prefix("http://").unwrap_or(s).trim_end_matches('/');
        let (h, p) = if let Some(rest) = s.strip_prefix('[') {
            let (h, after) = rest.split_once(']').ok_or_else(|| format!("FERRIC_HOST={s}: unclosed '['"))?;
            (h.to_string(), after.strip_prefix(':').map(str::to_string))
        } else {
            match s.rsplit_once(':') {
                Some((h, p)) => (h.to_string(), Some(p.to_string())),
                None => (s.to_string(), None),
            }
        };
        let port = match p {
            Some(p) => p.parse::<u16>().map_err(|_| format!("FERRIC_HOST={s}: port {p:?} is not a number"))?,
            None => DEFAULT_PORT,
        };
        let host = if h.is_empty() { "127.0.0.1".to_string() } else { h };
        Ok(Host { host, port })
    }

    pub fn addr(&self) -> String {
        if self.host.contains(':') { format!("[{}]:{}", self.host, self.port) } else { format!("{}:{}", self.host, self.port) }
    }

    /// Only a server on this machine may be started by `ferric run`; a remote one is the remote's business.
    pub fn is_local(&self) -> bool {
        matches!(self.host.as_str(), "localhost" | "::1" | "0.0.0.0" | "::") || self.host.starts_with("127.")
    }
}

#[derive(Debug)]
pub enum HttpError {
    /// Nothing accepted the connection — no server there.
    Connect(io::Error),
    Io(io::Error),
    Protocol(String),
}

impl std::fmt::Display for HttpError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            HttpError::Connect(e) => write!(f, "could not connect: {e}"),
            HttpError::Io(e) => write!(f, "{e}"),
            HttpError::Protocol(m) => write!(f, "{m}"),
        }
    }
}

impl From<io::Error> for HttpError { fn from(e: io::Error) -> Self { HttpError::Io(e) } }

pub struct Response {
    pub status: u16,
    pub body: Body,
    /// A second handle on the socket, so a caller streaming on another thread can `shutdown` it to
    /// cancel. Closing is how a client tells ferric-serve (and Ollama) to stop generating.
    pub socket: TcpStream,
}

impl Response {
    pub fn text(mut self) -> Result<String, HttpError> {
        let mut s = String::new();
        self.body.read_to_string(&mut s)?;
        Ok(s)
    }

    pub fn json(self) -> Result<Value, HttpError> {
        let t = self.text()?;
        serde_json::from_str(&t).map_err(|e| HttpError::Protocol(format!("server sent invalid JSON ({e}): {}", clip(&t, 200))))
    }

    pub fn ok(&self) -> bool { (200..300).contains(&self.status) }

    /// The `error` a server put in a non-2xx body (both ferric-serve and Ollama use `{"error": ...}`,
    /// ferric-serve's /v1 routes nest it as `{"error": {"message": ...}}`).
    pub fn error_text(self) -> String {
        let status = self.status;
        match self.text() {
            Ok(t) => match serde_json::from_str::<Value>(&t) {
                Ok(v) => v["error"]["message"].as_str().or_else(|| v["error"].as_str())
                    .map(str::to_string).unwrap_or_else(|| format!("HTTP {status}: {}", clip(&t, 300))),
                Err(_) => format!("HTTP {status}: {}", clip(t.trim(), 300)),
            },
            Err(e) => format!("HTTP {status} ({e})"),
        }
    }
}

pub fn clip(s: &str, n: usize) -> String {
    if s.chars().count() <= n { s.to_string() } else { format!("{}…", s.chars().take(n).collect::<String>()) }
}

/// The response body, whichever way the server framed it.
pub enum Body {
    Empty,
    Length(io::Take<BufReader<TcpStream>>),
    Chunked(Chunked<BufReader<TcpStream>>),
    Close(BufReader<TcpStream>),
}

impl Read for Body {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        match self {
            Body::Empty => Ok(0),
            Body::Length(r) => r.read(buf),
            Body::Chunked(r) => r.read(buf),
            Body::Close(r) => r.read(buf),
        }
    }
}

/// `Transfer-Encoding: chunked` — `<hex size>[;ext]\r\n<data>\r\n` … `0\r\n<trailers>\r\n`.
pub struct Chunked<R: BufRead> { r: R, left: u64, started: bool, done: bool }

impl<R: BufRead> Chunked<R> {
    pub fn new(r: R) -> Self { Chunked { r, left: 0, started: false, done: false } }
}

impl<R: BufRead> Read for Chunked<R> {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        if self.done || buf.is_empty() { return Ok(0); }
        if self.left == 0 {
            let mut line = String::new();
            if self.started {
                // The CRLF that closes the previous chunk's data.
                self.r.read_line(&mut line)?;
                if !line.trim().is_empty() {
                    return Err(io::Error::new(io::ErrorKind::InvalidData, format!("chunked body: expected CRLF after a chunk, got {line:?}")));
                }
                line.clear();
            }
            self.started = true;
            if self.r.read_line(&mut line)? == 0 {
                return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "chunked body ended without its 0-size chunk"));
            }
            let hex = line.trim().split(';').next().unwrap_or("").trim();
            let n = u64::from_str_radix(hex, 16)
                .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, format!("chunked body: bad chunk size {hex:?}")))?;
            if n == 0 {
                // Trailers, then the blank line that ends the message.
                loop {
                    let mut t = String::new();
                    if self.r.read_line(&mut t)? == 0 || t.trim().is_empty() { break; }
                }
                self.done = true;
                return Ok(0);
            }
            self.left = n;
        }
        let want = buf.len().min(self.left as usize);
        let n = self.r.read(&mut buf[..want])?;
        if n == 0 { return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "chunked body: connection closed inside a chunk")); }
        self.left -= n as u64;
        Ok(n)
    }
}

fn connect(host: &Host, timeout: Duration) -> Result<TcpStream, HttpError> {
    let addrs: Vec<_> = (host.host.as_str(), host.port).to_socket_addrs().map_err(HttpError::Connect)?.collect();
    let mut last = io::Error::new(io::ErrorKind::NotFound, format!("{} resolves to no address", host.host));
    for a in addrs {
        match TcpStream::connect_timeout(&a, timeout) {
            Ok(s) => return Ok(s),
            Err(e) => last = e,
        }
    }
    Err(HttpError::Connect(last))
}

/// One request, `Connection: close`. `read_timeout` bounds each read (a probe); `None` waits as long
/// as the server takes, which a first request that loads a multi-GB model needs.
pub fn request(host: &Host, method: &str, path: &str, body: Option<&Value>, read_timeout: Option<Duration>)
    -> Result<Response, HttpError>
{
    let mut s = connect(host, Duration::from_secs(3))?;
    s.set_read_timeout(read_timeout)?;
    let _ = s.set_nodelay(true);
    let payload = body.map(|b| serde_json::to_vec(b).unwrap_or_default());
    let mut head = format!("{method} {path} HTTP/1.1\r\nHost: {}\r\nUser-Agent: ferric/{}\r\nAccept: application/json, application/x-ndjson\r\nConnection: close\r\n",
                           host.addr(), env!("CARGO_PKG_VERSION"));
    if let Some(p) = &payload {
        head.push_str(&format!("Content-Type: application/json\r\nContent-Length: {}\r\n", p.len()));
    }
    head.push_str("\r\n");
    s.write_all(head.as_bytes())?;
    if let Some(p) = &payload { s.write_all(p)?; }
    s.flush()?;
    let socket = s.try_clone()?;
    let mut r = BufReader::new(s);
    let mut status_line = String::new();
    if r.read_line(&mut status_line)? == 0 {
        return Err(HttpError::Protocol(format!("{} closed the connection without answering {method} {path}", host.addr())));
    }
    let status = status_line.split_whitespace().nth(1).and_then(|c| c.parse::<u16>().ok())
        .ok_or_else(|| HttpError::Protocol(format!("not an HTTP response: {:?}", clip(status_line.trim(), 80))))?;
    let mut headers = Vec::new();
    loop {
        let mut h = String::new();
        if r.read_line(&mut h)? == 0 || h.trim().is_empty() { break; }
        if let Some((k, v)) = h.split_once(':') { headers.push((k.trim().to_ascii_lowercase(), v.trim().to_string())); }
    }
    let get = |k: &str| headers.iter().find(|(h, _)| h == k).map(|(_, v)| v.as_str());
    let body = if method == "HEAD" || status == 204 || status == 304 || (100..200).contains(&status) {
        Body::Empty
    } else if get("transfer-encoding").is_some_and(|v| v.to_ascii_lowercase().contains("chunked")) {
        Body::Chunked(Chunked::new(r))
    } else if let Some(n) = get("content-length").and_then(|v| v.parse::<u64>().ok()) {
        Body::Length(r.take(n))
    } else {
        Body::Close(r)
    };
    Ok(Response { status, body, socket })
}

pub fn post(host: &Host, path: &str, body: &Value) -> Result<Response, HttpError> { request(host, "POST", path, Some(body), None) }

/// GET/POST expecting a 2xx JSON answer; anything else becomes the server's own error text.
pub fn json(host: &Host, method: &str, path: &str, body: Option<&Value>) -> Result<Value, String> {
    let r = request(host, method, path, body, None).map_err(|e| format!("{method} {path} on {}: {e}", host.addr()))?;
    if !r.ok() { return Err(r.error_text()); }
    r.json().map_err(|e| e.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn host_forms() {
        assert_eq!(Host::parse("127.0.0.1:11434").unwrap(), Host { host: "127.0.0.1".into(), port: 11434 });
        assert_eq!(Host::parse("http://localhost:9000/").unwrap(), Host { host: "localhost".into(), port: 9000 });
        assert_eq!(Host::parse("myhost").unwrap().port, DEFAULT_PORT);
        assert_eq!(Host::parse(":8080").unwrap(), Host { host: "127.0.0.1".into(), port: 8080 });
        assert_eq!(Host::parse("[::1]:7").unwrap(), Host { host: "::1".into(), port: 7 });
        assert_eq!(Host::parse("[::1]:7").unwrap().addr(), "[::1]:7");
        assert!(Host::parse("https://x:1").is_err());
        assert!(Host::parse("x:notaport").is_err());
        assert!(Host::parse("127.0.0.1:1").unwrap().is_local() && !Host::parse("10.0.0.2:1").unwrap().is_local());
    }

    /// Chunk boundaries that split a line, a chunk extension, and trailers — the three places a
    /// hand-rolled decoder usually slips.
    #[test]
    fn chunked_decoding() {
        let wire = b"5\r\n{\"a\":\r\n7;ext=1\r\n1}\n{\"b\"\r\n3\r\n:2}\r\n1\r\n\n\r\n0\r\nX-Trailer: y\r\n\r\n";
        let mut out = String::new();
        Chunked::new(&wire[..]).read_to_string(&mut out).unwrap();
        assert_eq!(out, "{\"a\":1}\n{\"b\":2}\n");
        let mut trunc = String::new();
        assert!(Chunked::new(&b"5\r\nab"[..]).read_to_string(&mut trunc).is_err(), "a cut chunk is an error, not a short answer");
    }
}
