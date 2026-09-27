//! **Images from clients → the pixels the model's authors would have fed it.**
//!
//! A vision model is verified against its authors' processor, and every Hugging Face vision processor
//! reads an image through `transformers.image_utils.load_image`: `PIL.Image.open` → `ImageOps.exif_transpose`
//! → `.convert("RGB")`. This reproduces that chain, so the tower sees what it saw when it was verified:
//!
//! - **PNG** (`png` 0.18): palette, grey, grey+alpha, RGBA and tRNS expanded to 8-bit; alpha DROPPED, not
//!   composited — `convert("RGB")` drops it. Lossless, so the pixels must be equal, and are (fixtures).
//! - **JPEG** ([`jpeg`], a translation of the libjpeg-turbo 3.1.4.1 that Pillow 12.2.0 bundles): baseline,
//!   extended and progressive (successive approximation included), restart intervals, every sampling
//!   layout libjpeg-turbo upsamples (4:4:4, 4:2:2, 4:2:0, 4:4:0, 4:1:1, any integral ratio), greyscale,
//!   YCbCr and RGB colour spaces. A lossy decode is not unique, and "close" was not enough: zune-jpeg 0.5
//!   was within 3 levels of PIL (mean 0.38 at 4:2:0) and that moved MiMo-Embodied-7B's greedy answer at
//!   character 52. The pixels are now EQUAL to PIL's (max |Δ| 0) on 79 fixture JPEGs, the 1083x722 demo
//!   photo and three 12 MP files; what is not reproduced is refused by name (the module docs list it).
//! - **Orientation**, as `exif_transpose` finds it: EXIF IFD0 tag 0x0112 read with PIL's TIFF rules (type
//!   by type: a BYTE-typed value is `bytes` and does not rotate, a RATIONAL 12/2 rotates as 6), and —
//!   when EXIF has no such tag — the first `tiff:Orientation="N"` / `>N` in the XMP packet, which Pillow's
//!   `getexif` falls back to. A phone photo is usually stored sideways with tag 6; without this the model is shown the
//!   picture rotated 90°.
//! - **PPM** (P6), what the examples read.
//!
//! Anything else (GIF, WebP, HEIC, AVIF, BMP, TIFF) is refused by name, never guessed at.
//! ⚠ Not reproduced: 16-bit greyscale PNG, which PIL converts by CLIPPING values above 255 rather than
//! scaling (a PIL quirk); such a file is refused rather than decoded differently from the reference.
use ferric_tensor::image::Rgb8;

/// Decode an image a client sent, by its content (never its file name or declared type).
pub(crate) fn decode(b: &[u8]) -> Result<Rgb8, String> {
    if b.starts_with(b"\x89PNG\r\n\x1a\n") { return png(b); }
    if b.starts_with(&[0xFF, 0xD8, 0xFF]) { return jpeg(b); }
    if b.starts_with(b"P6") { return ferric_tensor::image::read_ppm(b); }
    let what = if b.starts_with(b"GIF8") { "GIF" } else if b.len() > 12 && &b[0..4] == b"RIFF" && &b[8..12] == b"WEBP" { "WebP" }
        else if b.len() > 12 && &b[4..8] == b"ftyp" { "HEIC/AVIF" } else if b.starts_with(b"BM") { "BMP" }
        else if b.starts_with(b"II*\0") || b.starts_with(b"MM\0*") { "TIFF" } else { "an unrecognised format" };
    Err(format!("the image is {what}; PNG and JPEG are read — convert it first (e.g. `sips -s format png in --out out.png`)"))
}

fn png(b: &[u8]) -> Result<Rgb8, String> {
    let mut d = ::png::Decoder::new(std::io::Cursor::new(b));
    d.set_transformations(::png::Transformations::EXPAND | ::png::Transformations::STRIP_16);
    let info = d.read_info().map_err(|e| format!("PNG: {e}"))?;
    let (ct, depth) = (info.info().color_type, info.info().bit_depth);
    if ct == ::png::ColorType::Grayscale && depth == ::png::BitDepth::Sixteen {
        return Err("16-bit greyscale PNG: the reference pipeline (PIL) clips it rather than scaling it; send 8-bit".into());
    }
    let mut r = info;
    let mut buf = vec![0u8; r.output_buffer_size().ok_or("PNG: image too large")?];
    let out = r.next_frame(&mut buf).map_err(|e| format!("PNG: {e}"))?;
    let (w, h) = (out.width as usize, out.height as usize);
    let src = &buf[..out.buffer_size()];
    let ch = match out.color_type {
        ::png::ColorType::Grayscale => 1, ::png::ColorType::GrayscaleAlpha => 2, ::png::ColorType::Rgb => 3,
        ::png::ColorType::Rgba => 4, ::png::ColorType::Indexed => return Err("PNG: palette was not expanded".into()),
    };
    let stride = out.line_size;
    let mut px = Vec::with_capacity(w * h * 3);
    for y in 0..h {
        let row = &src[y * stride..y * stride + w * ch];
        for p in row.chunks_exact(ch) {
            match ch { 1 | 2 => px.extend_from_slice(&[p[0], p[0], p[0]]), _ => px.extend_from_slice(&p[..3]) }
        }
    }
    Ok(Rgb8 { w, h, px })
}

mod jpeg;

fn jpeg(b: &[u8]) -> Result<Rgb8, String> {
    let d = jpeg::decode(b)?;
    let img = Rgb8 { w: d.w, h: d.h, px: d.px };
    Ok(match orientation(d.exif.as_deref(), d.xmp.as_deref())? { Some(o) => orient(img, o), None => img })
}

/// The transpose `exif_transpose` applies (2..=8), or None. `Image.getexif` reads PIL's `info["exif"]`; only
/// when its IFD0 holds no 0x0112 entry does it take the first `tiff:Orientation(="|>)([0-9])` of the XMP.
fn orientation(exif: Option<&[u8]>, xmp: Option<&[u8]>) -> Result<Option<u16>, String> {
    let value = match exif.map(exif_orientation).transpose()?.flatten() {
        Some(v) => v,
        None => match xmp.and_then(xmp_orientation) { Some(d) => Some(d), None => return Ok(None) },
    };
    // `{2: FLIP_LEFT_RIGHT, …, 8: ROTATE_90}.get(orientation)`: only a value EQUAL to one of those ints.
    Ok(value.filter(|v| (2..=8).contains(v)).map(|v| v as u16))
}

/// PIL's `Exif.load` + `ImageFileDirectory_v2.load` for tag 0x0112 of IFD0: `None` if the tag is absent;
/// `Some(v)` if present, with `v` the integer the stored value equals in Python (`None` for bytes/str, or a
/// non-integral number). Errors where PIL raises (a block that is not TIFF), so `load_image` would fail.
fn exif_orientation(e: &[u8]) -> Result<Option<Option<i64>>, String> {
    let mut t = e;
    while t.starts_with(b"Exif\0\0") { t = &t[6..]; }
    if t.is_empty() { return Ok(None); }
    let le = match t.get(0..4) {
        Some(b"II*\0") | Some(b"II\0*") => true,
        Some(b"MM\0*") | Some(b"MM*\0") => false,
        Some(b"II+\0") | Some(b"MM\0+") => return Err("JPEG: BigTIFF-structured EXIF is not read".into()),
        _ => return Err("JPEG: the EXIF block is not TIFF-structured (PIL raises on it)".into()),
    };
    let rd = |i: usize, n: usize| -> Option<u64> {
        let s = t.get(i..i + n)?;
        Some(if le { s.iter().rev().fold(0, |a, &b| a << 8 | b as u64) } else { s.iter().fold(0, |a, &b| a << 8 | b as u64) })
    };
    let ifd = rd(4, 4).ok_or("JPEG: truncated EXIF header (PIL raises on it)")? as usize;
    // Tags load in order until the first read past the end (an OSError PIL catches: tags so far are kept).
    let Some(count) = rd(ifd, 2) else { return Ok(None) };
    let mut found = None;
    for k in 0..count as usize {
        let at = ifd + 2 + 12 * k;
        if t.len() < at + 12 { break; } // `_ensure_read(fp, 12)` raised
        let (Some(tag), Some(typ), Some(n)) = (rd(at, 2), rd(at + 2, 2), rd(at + 4, 4)) else { break };
        let unit = match typ { 1 | 2 | 6 | 7 => 1, 3 | 8 => 2, 4 | 9 | 11 | 13 => 4, 5 | 10 | 12 | 16 => 8, _ => continue };
        let size = n as usize * unit;
        let data = if size > 4 {
            let off = rd(at + 8, 4).unwrap() as usize;
            match t.get(off..off + size) { Some(d) => d, None => break }
        } else {
            &t[at + 8..at + 8 + size]
        };
        if size == 0 || tag != 0x0112 { continue; }
        let v = |i: usize, n: usize| { let s = &data[i..i + n]; if le { s.iter().rev().fold(0u64, |a, &b| a << 8 | b as u64) } else { s.iter().fold(0u64, |a, &b| a << 8 | b as u64) } };
        found = Some(match typ {
            3 => Some(v(0, 2) as i64),
            8 => Some(v(0, 2) as u16 as i16 as i64),
            4 | 13 => Some(v(0, 4) as i64),
            9 => Some(v(0, 4) as u32 as i32 as i64),
            6 => Some(data[0] as i8 as i64),
            16 => i64::try_from(v(0, 8)).ok(),
            11 => { let f = f32::from_bits(v(0, 4) as u32) as f64; (f.fract() == 0.0).then_some(f as i64) }
            12 => { let f = f64::from_bits(v(0, 8)); (f.fract() == 0.0).then_some(f as i64) }
            5 => { let (a, b) = (v(0, 4) as i64, v(4, 4) as i64); (b != 0 && a % b == 0).then(|| a / b) }
            10 => { let (a, b) = (v(0, 4) as u32 as i32 as i64, v(4, 4) as u32 as i32 as i64); (b != 0 && a % b == 0).then(|| a / b) }
            _ => None, // BYTE and UNDEFINED load as bytes, ASCII as str: never equal to an int
        });
    }
    Ok(found)
}

/// `re.search(rb'tiff:Orientation(="|>)([0-9])', xmp)`.
fn xmp_orientation(x: &[u8]) -> Option<i64> {
    let key = b"tiff:Orientation";
    (0..x.len().saturating_sub(key.len())).filter(|&i| x[i..].starts_with(key)).find_map(|i| {
        let r = &x[i + key.len()..];
        let d = if r.starts_with(b"=\"") { r.get(2) } else if r.starts_with(b">") { r.get(1) } else { None }?;
        d.is_ascii_digit().then(|| (d - b'0') as i64)
    })
}

/// Apply an orientation the way `PIL.ImageOps.exif_transpose` does: 2 FLIP_LEFT_RIGHT, 3 ROTATE_180,
/// 4 FLIP_TOP_BOTTOM, 5 TRANSPOSE, 6 ROTATE_270, 7 TRANSVERSE, 8 ROTATE_90 (PIL's rotations are
/// counter-clockwise, so 6 turns the stored image 90° clockwise).
fn orient(img: Rgb8, o: u16) -> Rgb8 {
    let (w, h) = (img.w, img.h);
    let at = |r: usize, c: usize| &img.px[(r * w + c) * 3..(r * w + c) * 3 + 3];
    let (ow, oh) = if o >= 5 { (h, w) } else { (w, h) };
    let mut px = Vec::with_capacity(img.px.len());
    for r in 0..oh {
        for c in 0..ow {
            let (sr, sc) = match o {
                2 => (r, w - 1 - c), 3 => (h - 1 - r, w - 1 - c), 4 => (h - 1 - r, c),
                5 => (c, r), 6 => (h - 1 - c, r), 7 => (h - 1 - c, w - 1 - r), 8 => (c, w - 1 - r),
                _ => (r, c),
            };
            px.extend_from_slice(at(sr, sc));
        }
    }
    Rgb8 { w: ow, h: oh, px }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Read;

    fn b64(s: &str) -> Vec<u8> {
        let v = |c: u8| match c { b'A'..=b'Z' => c - b'A', b'a'..=b'z' => c - b'a' + 26, b'0'..=b'9' => c - b'0' + 52, b'+' => 62, _ => 63 };
        let s: Vec<u8> = s.bytes().filter(|&c| c != b'=' && !c.is_ascii_whitespace()).collect();
        s.chunks(4).flat_map(|q| {
            let n = q.iter().enumerate().fold(0u32, |a, (i, &c)| a | (v(c) as u32) << (18 - 6 * i));
            [(n >> 16) as u8, (n >> 8) as u8, n as u8].into_iter().take(q.len() - 1)
        }).collect()
    }

    fn crc32(b: &[u8]) -> u32 { let mut c = flate2::Crc::new(); c.update(b); c.sum() }

    /// Rows whose CRC32 differs from the reference's (the form make_images.py stores large images in).
    fn rows_differing(got: &Rgb8, want: &serde_json::Value) -> Vec<usize> {
        let rows = want.as_array().unwrap();
        assert_eq!(rows.len(), got.h);
        got.px.chunks_exact(got.w * 3).zip(rows).enumerate()
            .filter(|(_, (row, c))| crc32(row) as u64 != c.as_u64().unwrap()).map(|(y, _)| y).collect()
    }

    /// Every fixture decodes to EXACTLY the pixels `transformers.image_utils.load_image` gives (Pillow
    /// 12.2.0 + its libjpeg-turbo 3.1.4.1; make_images.py): 6 PNG and 79 JPEG — every sampling layout
    /// libjpeg-turbo upsamples, progressive with successive approximation, restart intervals, qualities
    /// 1..100, 16-bit tables, sizes 1x1 to 333x217, RGB and greyscale colour spaces, EXIF orientations 2–8
    /// with PIL's type rules, the XMP fallback, and a crop of a real photo. zune-jpeg 0.5 (the previous
    /// decoder) was up to 3 levels off with mean 0.38 at 4:2:0; the bound is now zero.
    #[test]
    fn images_decode_to_the_pixels_the_authors_pipeline_sees() {
        let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/images");
        let mut gz = Vec::new();
        std::fs::File::open(format!("{dir}/reference.json.gz")).unwrap().read_to_end(&mut gz).unwrap();
        let mut json = Vec::new();
        flate2::read::GzDecoder::new(&gz[..]).read_to_end(&mut json).unwrap();
        let r: serde_json::Value = serde_json::from_slice(&json).unwrap();
        let imgs = r["images"].as_object().unwrap();
        assert_eq!(imgs.len(), 85, "the fixture set changed size: regenerate reference.json.gz with make_images.py");
        let mut failed = Vec::new();
        for (name, want) in imgs {
            let got = decode(&std::fs::read(format!("{dir}/{name}")).unwrap()).unwrap_or_else(|e| panic!("{name}: {e}"));
            assert_eq!((got.h, got.w), (want["h"].as_u64().unwrap() as usize, want["w"].as_u64().unwrap() as usize), "{name}: shape (orientation?)");
            if let Some(rgb) = want["rgb"].as_str() {
                let w = b64(rgb);
                let d: Vec<i32> = got.px.iter().zip(&w).map(|(a, b)| (*a as i32 - *b as i32).abs()).collect();
                let (max, mean) = (*d.iter().max().unwrap(), d.iter().sum::<i32>() as f64 / d.len() as f64);
                eprintln!("{name:32} max|Δ| {max:3}  mean|Δ| {mean:.4}");
                if max != 0 { failed.push(format!("{name}: max|Δ| {max}, mean {mean:.4}")); }
            } else {
                let bad = rows_differing(&got, &want["rows_crc32"]);
                eprintln!("{name:32} rows differing {}/{}", bad.len(), got.h);
                if !bad.is_empty() { failed.push(format!("{name}: {} of {} rows differ (first {})", bad.len(), got.h, bad[0])); }
            }
        }
        assert!(failed.is_empty(), "not the reference's pixels:\n{}", failed.join("\n"));
    }

    /// The uncommitted full-size photo make_images.py recorded (MiMo-Embodied's demo.jpg, 1083x722, 4:2:0),
    /// checked when the Hugging Face cache holds that exact file; skipped (loudly) otherwise.
    #[test]
    fn a_real_photo_decodes_to_the_pixels_the_authors_pipeline_sees() {
        let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/images");
        let mut gz = Vec::new();
        std::fs::File::open(format!("{dir}/reference.json.gz")).unwrap().read_to_end(&mut gz).unwrap();
        let mut json = Vec::new();
        flate2::read::GzDecoder::new(&gz[..]).read_to_end(&mut json).unwrap();
        let r: serde_json::Value = serde_json::from_slice(&json).unwrap();
        let want = &r["external"]["demo.jpg"];
        let hub = std::env::var("HF_HUB_CACHE").map(std::path::PathBuf::from)
            .or_else(|_| std::env::var("HF_HOME").map(|h| std::path::Path::new(&h).join("hub")))
            .unwrap_or_else(|_| std::path::Path::new(&std::env::var("HOME").unwrap_or_default()).join(".cache/huggingface/hub"));
        let snaps = hub.join("models--XiaomiMiMo--MiMo-Embodied-7B/snapshots");
        let file = std::fs::read_dir(&snaps).into_iter().flatten().flatten()
            .map(|e| e.path().join("assets/demo.jpg")).filter_map(|p| std::fs::read(p).ok())
            .find(|b| crc32(b) as u64 == want["file_crc32"].as_u64().unwrap());
        let Some(b) = file else { eprintln!("SKIPPED: {} has no demo.jpg with the recorded CRC", snaps.display()); return };
        let got = decode(&b).unwrap();
        assert_eq!((got.h, got.w), (want["h"].as_u64().unwrap() as usize, want["w"].as_u64().unwrap() as usize));
        let bad = rows_differing(&got, &want["rows_crc32"]);
        assert!(bad.is_empty(), "demo.jpg: {} of {} rows differ from PIL's", bad.len(), got.h);
    }

    #[test]
    fn unsupported_formats_are_named() {
        assert!(decode(b"GIF89a....").unwrap_err().contains("GIF"));
        assert!(decode(b"RIFF\0\0\0\0WEBPVP8 ").unwrap_err().contains("WebP"));
        assert!(decode(b"\0\0\0\x18ftypheic....").unwrap_err().contains("HEIC"));
    }
}
