//! **`/v1/audio/transcriptions`** — OpenAI's speech-to-text endpoint over Ferric's verified recogniser.
//!
//! Parity gap S19 (9.5 of 14 serving peers). Ferric's speech recogniser (Parakeet, verified against NVIDIA
//! NeMo stage by stage) ran only from examples and the browser; `--asr <parakeet.gguf>` now serves it.
//! Multipart form: `file` (WAV), `model`, `language`, `response_format` (`json`, `text`, `verbose_json`).
//!
//! ⚠ WAV only (16-bit PCM or 32-bit float, any channel count — mixed to mono — at any rate — resampled to
//! the model's with the same windowed-sinc resampler the MiMo-ASR path verified). An MP3/M4A/WebM upload
//! is refused with the conversion to run; a decoder that guessed would transcribe noise with no error.
//! `srt` / `vtt` need timestamps this path does not produce yet, and are refused.
use crate::{write_json, Engine};
use serde_json::{json, Value};
use std::io::Write;
use std::net::TcpStream;

pub(crate) struct Part<'a> { pub name: String, pub filename: Option<String>, pub data: &'a [u8] }

/// `multipart/form-data` → its parts. The boundary comes from the Content-Type header.
pub(crate) fn multipart<'a>(body: &'a [u8], content_type: &str) -> Result<Vec<Part<'a>>, String> {
    let b = content_type.split(';').map(str::trim).find_map(|p| p.strip_prefix("boundary="))
        .ok_or("multipart/form-data without a boundary")?.trim_matches('"');
    let delim = format!("--{b}");
    let find = |hay: &[u8], needle: &[u8], from: usize| hay[from..].windows(needle.len()).position(|w| w == needle).map(|p| p + from);
    let mut parts = Vec::new();
    let mut i = find(body, delim.as_bytes(), 0).ok_or("no multipart boundary in the body")? + delim.len();
    loop {
        if body[i..].starts_with(b"--") { break; }
        let head_end = find(body, b"\r\n\r\n", i).ok_or("a multipart part has no header terminator")?;
        let head = std::str::from_utf8(&body[i..head_end]).map_err(|_| "multipart headers are not UTF-8")?;
        let next = find(body, format!("\r\n{delim}").as_bytes(), head_end + 4).ok_or("an unterminated multipart part")?;
        let (mut name, mut filename) = (String::new(), None);
        for line in head.lines() {
            if line.to_ascii_lowercase().starts_with("content-disposition:") {
                for kv in line.split(';').map(str::trim) {
                    if let Some(v) = kv.strip_prefix("name=") { name = v.trim_matches('"').to_string(); }
                    if let Some(v) = kv.strip_prefix("filename=") { filename = Some(v.trim_matches('"').to_string()); }
                }
            }
        }
        parts.push(Part { name, filename, data: &body[head_end + 4..next] });
        i = next + 2 + delim.len();
        if i >= body.len() { break; }
    }
    Ok(parts)
}

/// WAV → mono f32 and its rate: PCM 16-bit or IEEE float 32-bit (WAVE_FORMAT_EXTENSIBLE resolved), any
/// channel count averaged to mono. Anything else is refused by name.
pub(crate) fn wav(b: &[u8]) -> Result<(Vec<f32>, usize), String> {
    if b.len() < 12 || &b[0..4] != b"RIFF" || &b[8..12] != b"WAVE" {
        return Err("the file is not WAV; convert it first, e.g. `ffmpeg -i in.mp3 -ar 16000 -ac 1 out.wav`".into());
    }
    let (mut i, mut fmt, mut data) = (12usize, None, None);
    while i + 8 <= b.len() {
        let id = &b[i..i + 4];
        let sz = u32::from_le_bytes([b[i + 4], b[i + 5], b[i + 6], b[i + 7]]) as usize;
        let d = &b[i + 8..(i + 8 + sz).min(b.len())];
        if id == b"fmt " && d.len() >= 16 {
            let u16at = |k: usize| u16::from_le_bytes([d[k], d[k + 1]]);
            let tag = if u16at(0) == 0xFFFE && d.len() >= 26 { u16at(24) } else { u16at(0) };
            fmt = Some((tag, u16at(2) as usize, u32::from_le_bytes([d[4], d[5], d[6], d[7]]) as usize, u16at(14)));
        }
        if id == b"data" { data = Some(d); }
        i += 8 + sz + (sz & 1);
    }
    let ((tag, ch, rate, bits), d) = (fmt.ok_or("WAV without a fmt chunk")?, data.ok_or("WAV without a data chunk")?);
    if ch == 0 || rate == 0 { return Err("WAV declares 0 channels or 0 Hz".into()); }
    let frames: Vec<f32> = match (tag, bits) {
        (1, 16) => d.chunks_exact(2).map(|c| i16::from_le_bytes([c[0], c[1]]) as f32 / 32768.0).collect(),
        (3, 32) => d.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect(),
        _ => return Err(format!("WAV format tag {tag} at {bits} bits: only 16-bit PCM and 32-bit float are read")),
    };
    Ok((frames.chunks(ch).map(|f| f.iter().sum::<f32>() / ch as f32).collect(), rate))
}

pub(crate) fn transcriptions(eng: &Engine, body: &[u8], headers: &[(String, String)], stream: &mut TcpStream) {
    let bad = |s: &mut TcpStream, m: &str| write_json(s, 400, &json!({"error": {"message": m, "type": "invalid_request_error"}}));
    let Some((name, asr)) = eng.aux.asr.as_ref() else {
        return bad(stream, "no speech model loaded: start the server with --asr <parakeet.gguf> (or FERRIC_ASR_MODEL)");
    };
    let ct = headers.iter().find(|(k, _)| k == "content-type").map(|(_, v)| v.as_str()).unwrap_or("");
    if !ct.starts_with("multipart/form-data") { return bad(stream, "send the audio as multipart/form-data with a `file` field"); }
    let parts = match multipart(body, ct) { Ok(p) => p, Err(e) => return bad(stream, &e) };
    let field = |n: &str| parts.iter().find(|p| p.name == n).map(|p| String::from_utf8_lossy(p.data).trim().to_string());
    let Some(file) = parts.iter().find(|p| p.name == "file") else { return bad(stream, "no `file` field") };
    let format = field("response_format").unwrap_or_else(|| "json".into());
    if !matches!(format.as_str(), "json" | "text" | "verbose_json") {
        return bad(stream, &format!("response_format {format:?}: json, text and verbose_json are served; srt/vtt need timestamps this path does not produce yet"));
    }
    if let Some(l) = field("language").filter(|l| !l.is_empty() && !l.starts_with("en")) {
        return bad(stream, &format!("language {l:?}: the loaded speech model is English-only"));
    }
    let (pcm, rate) = match wav(file.data) { Ok(w) => w, Err(e) => return bad(stream, &format!("{}: {e}", file.filename.as_deref().unwrap_or("file"))) };
    let want = asr.cfg.sample_rate;
    let pcm = if rate == want { pcm } else { ferric_llama::mimo_asr::resample(&pcm, rate, want) };
    let secs = pcm.len() as f64 / want as f64;
    let ticket = eng.energy.begin();
    let text = match asr.transcribe(&pcm) { Ok(t) => t, Err(e) => return write_json(stream, 500, &json!({"error": {"message": e, "type": "server_error"}})) };
    let energy = eng.energy.end(ticket, text.split_whitespace().count());
    match format.as_str() {
        "text" => {
            let b = format!("{text}\n");
            let _ = stream.write_all(format!("HTTP/1.1 200 OK\r\nContent-Type: text/plain; charset=utf-8\r\nContent-Length: {}\r\nAccess-Control-Allow-Origin: *\r\nConnection: close\r\n\r\n", b.len()).as_bytes());
            let _ = stream.write_all(b.as_bytes());
            let _ = stream.flush();
        }
        "verbose_json" => write_json(stream, 200, &json!({"task": "transcribe", "language": "english", "duration": secs,
            "text": text, "segments": Value::Array(vec![]), "model": name, "energy": energy})),
        _ => write_json(stream, 200, &json!({"text": text, "model": name, "energy": energy})),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn multipart_parts_come_back_whole() {
        let body = b"--XyZ\r\nContent-Disposition: form-data; name=\"model\"\r\n\r\nparakeet\r\n--XyZ\r\nContent-Disposition: form-data; name=\"file\"; filename=\"a.wav\"\r\nContent-Type: audio/wav\r\n\r\nRIFF\x00\x01\r\n--XyZ--\r\n";
        let p = multipart(body, "multipart/form-data; boundary=XyZ").unwrap();
        assert_eq!(p.len(), 2);
        assert_eq!((p[0].name.as_str(), p[0].data), ("model", &b"parakeet"[..]));
        assert_eq!((p[1].name.as_str(), p[1].filename.as_deref(), p[1].data), ("file", Some("a.wav"), &b"RIFF\x00\x01"[..]));
    }

    fn wav_bytes(tag: u16, ch: u16, rate: u32, bits: u16, data: &[u8]) -> Vec<u8> {
        let mut b = b"RIFF\0\0\0\0WAVEfmt ".to_vec();
        b.extend(16u32.to_le_bytes()); b.extend(tag.to_le_bytes()); b.extend(ch.to_le_bytes()); b.extend(rate.to_le_bytes());
        b.extend((rate * ch as u32 * bits as u32 / 8).to_le_bytes()); b.extend((ch * bits / 8).to_le_bytes()); b.extend(bits.to_le_bytes());
        b.extend(b"data"); b.extend((data.len() as u32).to_le_bytes()); b.extend(data);
        b
    }

    #[test]
    fn wav_reads_pcm16_float32_and_stereo_and_refuses_the_rest() {
        let (p, r) = wav(&wav_bytes(1, 1, 16000, 16, &[0x00, 0x40, 0x00, 0xC0])).unwrap();
        assert_eq!((p, r), (vec![0.5, -0.5], 16000));
        let f: Vec<u8> = [0.25f32, 0.75].iter().flat_map(|x| x.to_le_bytes()).collect();
        let (p, _) = wav(&wav_bytes(3, 2, 48000, 32, &f)).unwrap();
        assert_eq!(p, vec![0.5], "stereo frames average to mono");
        assert!(wav(&wav_bytes(1, 1, 16000, 24, &[0; 6])).unwrap_err().contains("24 bits"));
        assert!(wav(b"ID3\x04 an mp3").unwrap_err().contains("ffmpeg"));
    }
}
