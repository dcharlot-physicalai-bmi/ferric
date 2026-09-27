//! **Images from clients → the pixels the model's authors would have fed it.**
//!
//! A vision model is verified against its authors' processor, and every Hugging Face vision processor
//! reads an image through `transformers.image_utils.load_image`: `PIL.Image.open` → `ImageOps.exif_transpose`
//! → `.convert("RGB")`. This reproduces that chain, so the tower sees what it saw when it was verified:
//!
//! - **PNG** (`png` 0.18): palette, grey, grey+alpha, RGBA and tRNS expanded to 8-bit; alpha DROPPED, not
//!   composited — `convert("RGB")` drops it. Lossless, so the pixels must be equal, and are (fixtures).
//! - **JPEG** (`zune-jpeg` 0.5): baseline and progressive, 4:2:0 / 4:2:2 / 4:4:4, greyscale. A lossy
//!   decode is not unique — PIL uses libjpeg-turbo's IDCT and upsampler — so the difference is MEASURED
//!   against PIL on the fixtures and bounded by the test, not assumed zero.
//! - **EXIF orientation** (JPEG APP1), applied as `exif_transpose` applies it. A phone photo is usually
//!   stored sideways with tag 6; without this the model is shown the picture rotated 90°.
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

fn jpeg(b: &[u8]) -> Result<Rgb8, String> {
    use zune_core::{bytestream::ZCursor, colorspace::ColorSpace, options::DecoderOptions};
    let mut d = zune_jpeg::JpegDecoder::new_with_options(ZCursor::new(b), DecoderOptions::default().jpeg_set_out_colorspace(ColorSpace::RGB));
    let px = d.decode().map_err(|e| format!("JPEG: {e:?}"))?;
    let info = d.info().ok_or("JPEG: no header")?;
    let img = Rgb8 { w: info.width as usize, h: info.height as usize, px };
    if img.px.len() != img.w * img.h * 3 { return Err(format!("JPEG: decoded {} bytes for {}x{} RGB", img.px.len(), img.w, img.h)); }
    Ok(match d.exif().and_then(|e| orientation(e)) { Some(o) => orient(img, o), None => img })
}

/// The EXIF orientation tag (0x0112) from a TIFF-structured EXIF block, if present and valid.
fn orientation(e: &[u8]) -> Option<u16> {
    let e = e.strip_prefix(b"Exif\0\0").unwrap_or(e);
    let le = match e.get(0..2)? { b"II" => true, b"MM" => false, _ => return None };
    let u16at = |i: usize| e.get(i..i + 2).map(|s| if le { u16::from_le_bytes([s[0], s[1]]) } else { u16::from_be_bytes([s[0], s[1]]) });
    let u32at = |i: usize| e.get(i..i + 4).map(|s| if le { u32::from_le_bytes([s[0], s[1], s[2], s[3]]) } else { u32::from_be_bytes([s[0], s[1], s[2], s[3]]) });
    let ifd = u32at(4)? as usize;
    for k in 0..u16at(ifd)? as usize {
        let at = ifd + 2 + 12 * k;
        if u16at(at)? == 0x0112 { return u16at(at + 8).filter(|o| (1..=8).contains(o)); }
    }
    None
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

    /// Every fixture decodes to the pixels `transformers.image_utils.load_image` gives (make_images.py):
    /// PNG exactly, JPEG within the bound measured here.
    #[test]
    fn images_decode_to_the_pixels_the_authors_pipeline_sees() {
        let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/images");
        let mut gz = Vec::new();
        std::fs::File::open(format!("{dir}/reference.json.gz")).unwrap().read_to_end(&mut gz).unwrap();
        let mut json = Vec::new();
        flate2::read::GzDecoder::new(&gz[..]).read_to_end(&mut json).unwrap();
        let r: serde_json::Value = serde_json::from_slice(&json).unwrap();
        let imgs = r["images"].as_object().unwrap();
        assert_eq!(imgs.len(), 13);
        for (name, want) in imgs {
            let got = decode(&std::fs::read(format!("{dir}/{name}")).unwrap()).unwrap_or_else(|e| panic!("{name}: {e}"));
            let w = b64(want["rgb"].as_str().unwrap());
            assert_eq!((got.h, got.w), (want["h"].as_u64().unwrap() as usize, want["w"].as_u64().unwrap() as usize), "{name}: shape (orientation?)");
            let d: Vec<i32> = got.px.iter().zip(&w).map(|(a, b)| (*a as i32 - *b as i32).abs()).collect();
            let (max, mean) = (*d.iter().max().unwrap(), d.iter().sum::<i32>() as f64 / d.len() as f64);
            eprintln!("{name:18} max|Δ| {max:3}  mean|Δ| {mean:.3}");
            if name.ends_with(".png") { assert_eq!(max, 0, "{name}: a lossless image must decode exactly"); }
            else { assert!(max <= JPEG_MAX && mean <= JPEG_MEAN, "{name}: max {max}, mean {mean:.3} vs PIL"); }
        }
    }

    /// Measured on the fixtures against PIL 12.2.0 (libjpeg-turbo): max 3 levels everywhere, mean 0.38 at
    /// 4:2:0, 0.18 at 4:2:2, 0.02 at 4:4:4 — the difference is chroma UPSAMPLING (libjpeg-turbo's "fancy"
    /// triangle filter vs zune's), since 4:4:4 has none and is nearly exact. The bound is the measurement.
    const JPEG_MAX: i32 = 3;
    const JPEG_MEAN: f64 = 0.4;

    #[test]
    fn unsupported_formats_are_named() {
        assert!(decode(b"GIF89a....").unwrap_err().contains("GIF"));
        assert!(decode(b"RIFF\0\0\0\0WEBPVP8 ").unwrap_err().contains("WebP"));
        assert!(decode(b"\0\0\0\x18ftypheic....").unwrap_err().contains("HEIC"));
    }
}
