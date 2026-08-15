//! **Pixels in, tensor out** — with no image-codec dependency.
//!
//! Ferric vendors every dependency in-repo and builds only from source it owns. Pulling a JPEG/PNG
//! decoder in to read a test image would be a poor trade: thousands of lines of parsing and entropy
//! decoding, vendored forever, to solve a problem the operating system already solves.
//!
//! **P6 PPM is the whole format**: an ASCII header, then raw interleaved RGB bytes. No compression, no
//! entropy coding, no colour management — about thirty lines to read. Anything else converts to it with
//! a tool that is already installed:
//!
//! ```text
//! ffmpeg -i in.jpg -pix_fmt rgb24 out.ppm                       # anything -> P6
//! sips -s format png in.heic --out /tmp/x.png &&                # HEIC needs the macOS decoder
//!   ffmpeg -i /tmp/x.png -pix_fmt rgb24 out.ppm                 # ...then ffmpeg
//! ```
//! (`sips` alone is not enough — it writes BMP and PNG but **not** PPM, so it always needs the second
//! step. Verified, because an untested command in a doc comment is a bug with a friendly face.)
//!
//! This also matches how the rest of the workspace already works: `pi0_vision` consumes a
//! pre-processed tensor from safetensors and validates against a reference, rather than decoding
//! anything itself. Preprocessing is a pipeline step, not a runtime dependency.

use crate::Tensor;
use ferric_core::Context;
use std::sync::Arc;

/// A decoded image: `[h, w, 3]` bytes, interleaved RGB.
#[derive(Debug)]
pub struct Rgb8 {
    pub w: usize,
    pub h: usize,
    pub px: Vec<u8>,
}

/// Read a binary PPM (**P6**). Returns an error rather than panicking, because the usual failure is a
/// P3 (ASCII) file or a PNG that was renamed, and saying which is more useful than a slice panic.
pub fn read_ppm(bytes: &[u8]) -> Result<Rgb8, String> {
    // Header fields are whitespace-separated ASCII, and `#` runs to end-of-line ANYWHERE in the
    // header — including between the dimensions, which is where a naive split breaks. Parsed with a
    // plain cursor rather than nested iterator borrows.
    let mut p = 0usize;
    let mut field = |p: &mut usize| -> Result<String, String> {
        let mut tok = String::new();
        while *p < bytes.len() {
            let ch = bytes[*p] as char;
            if ch == '#' {
                while *p < bytes.len() && bytes[*p] != b'\n' { *p += 1; }
                continue;
            }
            if ch.is_ascii_whitespace() {
                *p += 1;
                if !tok.is_empty() { return Ok(tok); }
                continue;
            }
            tok.push(ch);
            *p += 1;
        }
        if tok.is_empty() { Err("unexpected end of PPM header".into()) } else { Ok(tok) }
    };

    let magic = field(&mut p)?;
    if magic != "P6" {
        return Err(format!(
            "expected a binary PPM (P6), got {magic:?}. Convert first: \
             `ffmpeg -i in.jpg -pix_fmt rgb24 out.ppm`, or `sips` on macOS."
        ));
    }
    let w: usize = field(&mut p)?.parse().map_err(|_| "bad width")?;
    let h: usize = field(&mut p)?.parse().map_err(|_| "bad height")?;
    let maxval: usize = field(&mut p)?.parse().map_err(|_| "bad maxval")?;
    if maxval != 255 {
        return Err(format!("only 8-bit PPM is supported (maxval 255), got {maxval}"));
    }
    // `field` consumed the single whitespace byte that ends the header, so `p` is the first data byte.
    let start = p;
    let need = w * h * 3;
    let px = bytes.get(start..start + need)
        .ok_or_else(|| format!("PPM truncated: need {need} bytes of pixel data, have {}", bytes.len() - start))?
        .to_vec();
    Ok(Rgb8 { w, h, px })
}

/// Preprocess to a vision encoder's input: resize to `size × size`, scale to `[0,1]`, then normalise
/// per channel. Returns `[size, size, 3]`.
///
/// `mean`/`std` come from the checkpoint (`clip.vision.image_mean` / `image_std`) — Muse Glimmer uses
/// 0.5/0.5/0.5, which maps `[0,1]` to `[-1,1]`. Reading them from the file rather than hardcoding is
/// the point: a model trained with ImageNet statistics and fed 0.5s produces a plausible, wrong
/// embedding with nothing to flag it.
///
/// The resize runs on the GPU through `Tensor::resize_bilinear`, so it uses the *same* half-pixel
/// bilinear the position-embedding resize does — one implementation, one convention.
pub fn preprocess(ctx: &Arc<Context>, img: &Rgb8, size: usize, mean: [f32; 3], std: [f32; 3]) -> Tensor {
    let f: Vec<f32> = img.px.iter().map(|&b| b as f32 / 255.0).collect();
    let t = Tensor::from_vec(ctx, &f, &[img.h, img.w, 3]);
    let r = if img.h == size && img.w == size { t } else { t.resize_bilinear(size, size) };
    // (x - mean) / std, per channel, as a broadcast over the last dim.
    let m = Tensor::from_vec(ctx, &mean, &[3]);
    let s = Tensor::from_vec(ctx, &std, &[3]);
    r.sub(&m).div(&s)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ppm_header_survives_comments_and_odd_whitespace() {
        // A comment BETWEEN the dimensions, and a tab as separator — both legal, and both break a
        // parser that splits the first three lines.
        let mut b = b"P6\n# made by sips\n2\t#w\n2\n255\n".to_vec();
        let px: Vec<u8> = vec![1,2,3, 4,5,6, 7,8,9, 10,11,12];
        b.extend_from_slice(&px);
        let img = read_ppm(&b).expect("parse");
        assert_eq!((img.w, img.h), (2, 2));
        assert_eq!(img.px, px, "pixel payload must survive byte-for-byte");
    }

    #[test]
    fn rejects_ascii_ppm_with_a_useful_message() {
        let e = read_ppm(b"P3\n2 2\n255\n0 0 0").unwrap_err();
        assert!(e.contains("P6"), "error should name the expected format, got {e:?}");
        let e2 = read_ppm(b"P6\n4 4\n255\n\x01\x02").unwrap_err();
        assert!(e2.contains("truncated"), "short payload should say truncated, got {e2:?}");
    }

    #[test]
    fn preprocess_normalises_to_the_checkpoint_range() {
        let Ok(ctx) = pollster::block_on(ferric_core::Context::new()) else {
            eprintln!("SKIPPED preprocess: no GPU");
            return;
        };
        let ctx = Arc::new(ctx);
        // Solid mid-grey (128) with mean/std 0.5 must land near 0; pure black at -1, white at +1.
        for (val, want) in [(128u8, 0.00392f32), (0u8, -1.0), (255u8, 1.0)] {
            let img = Rgb8 { w: 3, h: 3, px: vec![val; 27] };
            let t = preprocess(&ctx, &img, 4, [0.5; 3], [0.5; 3]);
            let v = pollster::block_on(t.to_vec());
            assert_eq!(v.len(), 4 * 4 * 3, "output must be [size, size, 3]");
            let d = v.iter().fold(0f32, |a, &x| a.max((x - want).abs()));
            assert!(d < 1e-3, "value {val} -> expected ~{want}, worst deviation {d:.4}");
        }
        eprintln!("preprocess: black/grey/white map to -1/0/+1 under mean=std=0.5");
    }
}
