//! A tiny dependency-free WebSocket bridge so the scheduler can reach a **browser tab** as a device.
//! Browsers can't accept raw TCP, but a tab connects *out* via WebSocket — so we run a small HTTP
//! server that serves the worker page + wasm and upgrades `/ws` to a WebSocket. Op frames (the same
//! host-buffer format as Device::Remote) ride the socket to the tab, which computes on WebGPU and
//! sends the result back. RFC 6455 handshake (SHA-1 + base64) and frame codec implemented inline.

use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::{Arc, Mutex};

// ---- SHA-1 (for the WebSocket accept key) ----
fn sha1(data: &[u8]) -> [u8; 20] {
    let mut h: [u32; 5] = [0x67452301, 0xEFCDAB89, 0x98BADCFE, 0x10325476, 0xC3D2E1F0];
    let ml = (data.len() as u64) * 8;
    let mut msg = data.to_vec();
    msg.push(0x80);
    while msg.len() % 64 != 56 { msg.push(0); }
    msg.extend_from_slice(&ml.to_be_bytes());
    for chunk in msg.chunks(64) {
        let mut w = [0u32; 80];
        for i in 0..16 { w[i] = u32::from_be_bytes([chunk[i * 4], chunk[i * 4 + 1], chunk[i * 4 + 2], chunk[i * 4 + 3]]); }
        for i in 16..80 { w[i] = (w[i - 3] ^ w[i - 8] ^ w[i - 14] ^ w[i - 16]).rotate_left(1); }
        let (mut a, mut b, mut c, mut d, mut e) = (h[0], h[1], h[2], h[3], h[4]);
        for (i, &wi) in w.iter().enumerate() {
            let (f, k) = match i {
                0..=19 => ((b & c) | ((!b) & d), 0x5A827999u32),
                20..=39 => (b ^ c ^ d, 0x6ED9EBA1),
                40..=59 => ((b & c) | (b & d) | (c & d), 0x8F1BBCDC),
                _ => (b ^ c ^ d, 0xCA62C1D6),
            };
            let t = a.rotate_left(5).wrapping_add(f).wrapping_add(e).wrapping_add(k).wrapping_add(wi);
            e = d; d = c; c = b.rotate_left(30); b = a; a = t;
        }
        h[0] = h[0].wrapping_add(a); h[1] = h[1].wrapping_add(b); h[2] = h[2].wrapping_add(c);
        h[3] = h[3].wrapping_add(d); h[4] = h[4].wrapping_add(e);
    }
    let mut out = [0u8; 20];
    for i in 0..5 { out[i * 4..i * 4 + 4].copy_from_slice(&h[i].to_be_bytes()); }
    out
}
fn base64(data: &[u8]) -> String {
    const T: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut s = String::new();
    for c in data.chunks(3) {
        let b = [c[0], *c.get(1).unwrap_or(&0), *c.get(2).unwrap_or(&0)];
        let n = (b[0] as u32) << 16 | (b[1] as u32) << 8 | b[2] as u32;
        s.push(T[(n >> 18 & 63) as usize] as char);
        s.push(T[(n >> 12 & 63) as usize] as char);
        s.push(if c.len() > 1 { T[(n >> 6 & 63) as usize] as char } else { '=' });
        s.push(if c.len() > 2 { T[(n & 63) as usize] as char } else { '=' });
    }
    s
}

/// One WebSocket connection to a browser tab. Binary messages only.
pub struct WsConn(Mutex<TcpStream>);
impl WsConn {
    pub fn send(&self, payload: &[u8]) -> std::io::Result<()> {
        let mut s = self.0.lock().unwrap();
        let mut f = vec![0x82u8]; // FIN + binary
        let n = payload.len();
        if n < 126 { f.push(n as u8); }
        else if n < 65536 { f.push(126); f.extend_from_slice(&(n as u16).to_be_bytes()); }
        else { f.push(127); f.extend_from_slice(&(n as u64).to_be_bytes()); }
        f.extend_from_slice(payload);
        s.write_all(&f)
    }
    /// Blocking receive of one binary message (handles pings, ignores control frames).
    pub fn recv(&self) -> std::io::Result<Vec<u8>> {
        let mut s = self.0.lock().unwrap();
        loop {
            let mut hdr = [0u8; 2];
            s.read_exact(&mut hdr)?;
            let opcode = hdr[0] & 0x0f;
            let masked = hdr[1] & 0x80 != 0;
            let mut len = (hdr[1] & 0x7f) as usize;
            if len == 126 { let mut b = [0u8; 2]; s.read_exact(&mut b)?; len = u16::from_be_bytes(b) as usize; }
            else if len == 127 { let mut b = [0u8; 8]; s.read_exact(&mut b)?; len = u64::from_be_bytes(b) as usize; }
            let mut mask = [0u8; 4];
            if masked { s.read_exact(&mut mask)?; }
            let mut payload = vec![0u8; len];
            s.read_exact(&mut payload)?;
            if masked { for i in 0..len { payload[i] ^= mask[i % 4]; } }
            match opcode {
                0x2 | 0x1 => return Ok(payload),        // binary / text
                0x8 => return Err(std::io::Error::new(std::io::ErrorKind::ConnectionAborted, "close")),
                _ => continue,                          // ping/pong/continuation → ignore
            }
        }
    }
}

fn mime(path: &str) -> &'static str {
    if path.ends_with(".wasm") { "application/wasm" }
    else if path.ends_with(".js") { "text/javascript" }
    else if path.ends_with(".html") { "text/html" }
    else { "application/octet-stream" }
}

/// Serve `site_dir` over HTTP and hand back the first browser tab that connects to `/ws`.
pub struct Bridge {
    pub addr: String,
    worker: Arc<Mutex<Option<Arc<WsConn>>>>,
}
impl Bridge {
    pub fn url(&self) -> String { format!("http://{}/", self.addr) }
    pub fn take_worker(&self) -> Option<Arc<WsConn>> { self.worker.lock().unwrap().clone() }
}

/// Start the bridge on an ephemeral localhost port, serving files from `site_dir`.
pub fn start(site_dir: String) -> Bridge {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind bridge");
    let addr = listener.local_addr().unwrap().to_string();
    let worker: Arc<Mutex<Option<Arc<WsConn>>>> = Arc::new(Mutex::new(None));
    let w2 = worker.clone();
    std::thread::spawn(move || {
        for stream in listener.incoming().flatten() {
            let _ = handle_http(stream, &site_dir, &w2);
        }
    });
    Bridge { addr, worker }
}

fn handle_http(mut s: TcpStream, site_dir: &str, worker: &Arc<Mutex<Option<Arc<WsConn>>>>) -> std::io::Result<()> {
    // read request headers
    let mut buf = Vec::new();
    let mut byte = [0u8; 1];
    while !buf.ends_with(b"\r\n\r\n") {
        if s.read(&mut byte)? == 0 { return Ok(()); }
        buf.push(byte[0]);
    }
    let req = String::from_utf8_lossy(&buf);
    let path = req.lines().next().and_then(|l| l.split_whitespace().nth(1)).unwrap_or("/");
    if req.to_lowercase().contains("upgrade: websocket") {
        let key = req.lines().find(|l| l.to_lowercase().starts_with("sec-websocket-key:"))
            .map(|l| l.split(':').nth(1).unwrap().trim().to_string()).unwrap_or_default();
        let accept = base64(&sha1(format!("{key}258EAFA5-E914-47DA-95CA-C5AB0DC85B11").as_bytes()));
        let resp = format!("HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Accept: {accept}\r\n\r\n");
        s.write_all(resp.as_bytes())?;
        *worker.lock().unwrap() = Some(Arc::new(WsConn(Mutex::new(s))));
        return Ok(());
    }
    let rel = if path == "/" { "worker.html".to_string() } else { path.trim_start_matches('/').to_string() };
    match std::fs::read(format!("{site_dir}/{rel}")) {
        Ok(body) => {
            let hdr = format!("HTTP/1.1 200 OK\r\nContent-Type: {}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n", mime(&rel), body.len());
            s.write_all(hdr.as_bytes())?; s.write_all(&body)?;
        }
        Err(_) => { s.write_all(b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\n\r\n")?; }
    }
    Ok(())
}
