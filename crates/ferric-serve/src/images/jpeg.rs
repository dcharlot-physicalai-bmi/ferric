//! **JPEG → exactly the pixels Pillow produces.** A Rust translation of the parts of libjpeg-turbo 3.1.4.1
//! that `PIL.Image.open(f).convert("RGB")` runs in Pillow 12.2.0, whose wheels bundle that libjpeg-turbo
//! (`JPEGTURBO_VERSION=3.1.4.1` in Pillow's `.github/workflows/wheels-dependencies.sh`).
//!
//! **Why not a general JPEG decoder.** A lossy decode is not unique: the IDCT, the chroma upsampler and the
//! YCbCr table are each a choice. `zune-jpeg` 0.5 made different ones and landed within 3 levels of PIL
//! (mean 0.38 at 4:2:0), and that was enough to change MiMo-Embodied-7B's greedy answer at character 52 of
//! 232 (smallest top-1 margin 0.032). The vision tower was verified on PIL's pixels, so this module
//! reproduces PIL's pixels, bit for bit — measured on the fixtures in `tests/fixtures/images`.
//!
//! **What Pillow asks libjpeg-turbo for** (`src/libImaging/JpegDecode.c`, `src/decode.c`,
//! `JpegImagePlugin.py`, all at tag 12.2.0): `jpeg_read_header` defaults, `jpegmode` "" ("trust the decoder":
//! libjpeg's own colour-space guess stands), `out_color_space` JCS_EXT_RGBX for 3 components / GRAYSCALE for 1,
//! no `scale`, no `draft` — so `dct_method` = JDCT_ISLOW, `do_fancy_upsampling` = TRUE, `do_block_smoothing` =
//! TRUE, full size. Each stage below names the libjpeg-turbo function it translates:
//! - entropy decoding — `jdhuff.c` (sequential), `jdphuff.c` (progressive, incl. successive approximation),
//!   restart intervals (`process_restart`, `read_restart_marker`), the Annex K tables a Motion-JPEG frame
//!   omits (`jstdhuff.c`, installed for sequential files only, as `jinit_huff_decoder` does). Lossless coding:
//!   any correct decoder yields the same coefficients, so this is the part with no rounding choices.
//! - quantisation tables LATCHED at each component's first scan (`jdinput.c` `latch_quant_tables`): a DQT
//!   redefined between progressive scans does not apply to a component already seen. Multipliers are `short`
//!   (`MULTIPLIER` is `short` in a `WITH_SIMD` build, which Pillow's is).
//! - `jpeg_idct_islow` (`jidctint.c`): CONST_BITS 13, PASS1_BITS 2, the C's operation order and rounding.
//!   Computed on 8 lanes in wrapping 32-bit arithmetic, which is exact inside the guards below (every final
//!   sum stays under 1.6e9); `the_lane_idct_is_the_c_idct` checks it against a line-by-line translation
//!   (64-bit `JLONG`, the `& 1023` range-limit table of `prepare_range_limit_table`) on 60 000 blocks.
//! - upsampling (`jdsample.c`): per component by its ratio to the max sampling factors — fullsize,
//!   `h2v1_fancy_upsample`, `h1v2_fancy_upsample`, `h2v2_fancy_upsample` (the 9-3-3-1 triangle filter with the
//!   alternating +8/+7 bias), their box variants when the downsampled width is ≤ 2, and `int_upsample` for any
//!   other integral ratio (4:1:1 is a 4x box, not a filter). Context rows above the first / below the last
//!   real row replicate the edge row (`jdmainct.c` `make_funny_pointers`, `set_bottom_pointers`) — NOT the
//!   IDCT's padding rows. Merged upsampling (`jdmerge.c`) is not what runs: `use_merged_upsample` refuses when
//!   fancy upsampling is on.
//! - colour (`jdcolor.c`): `build_ycc_rgb_table` (SCALEBITS 16, FIX(1.40200) … ONE_HALF folded into Cb→G) and
//!   the colour-space guess of `default_decompress_parms` (`jdapimin.c`): JFIF APP0 → YCbCr, else Adobe APP14
//!   transform (0 → RGB, 1 → YCbCr, other → YCbCr), else component ids 'R','G','B' → RGB, else YCbCr.
//!
//! **The SIMD kernels Pillow actually runs.** On arm64 Pillow runs libjpeg-turbo's Neon kernels (x86:
//! SSE2/AVX2), not the C. From their source: the Neon YCbCr constants (2^-14/2^-15) give the same results as
//! the C tables for all 256 inputs (G's are exactly half; R and B round identically over −128..127), and
//! the Neon IDCT's `vaddhn` + `vqrshrn #2` equals C's DESCALE(x, 18). But the Neon IDCT dequantises and adds
//! pairs in 16-bit lanes, narrows pass 1 to 16 bits and SATURATES its output, where the C keeps int/64-bit
//! and WRAPS through `& 1023`. Measured with crafted one-block files, PIL on this machine gives 255 where the
//! C gives 0 (IDCT output 600), 0 where it gives 255 (−600), and differs again past 16-bit dequantisation.
//! There is no single reference there, so a block is REFUSED unless its dequantised values lie in
//! [−8192, 8191], its workspace in [−16384, 16383] and its outputs in [−512, 511] — the range where none of
//! those narrowings bites and every path computes the C value (the edges, 511 and −512, are fixtures). Real
//! data is far inside: |F(u,v)| ≤ 2048 for 8-bit samples, so a dequantised coefficient is at most 2048 + q/2.
//! (The x86 kernels were not examined; they were designed to the same bit-exactness on valid data.)
//!
//! **Refused by name, never guessed:** arithmetic coding (SOF9–11), lossless (SOF3), hierarchical (SOF5–7,
//! 13–15), 12-bit samples, CMYK/YCCK (4 components; PIL's CMYK;I→RGB path is not reproduced), 2 or >4
//! components; a progressive file whose coefficients 1–9 were left incomplete (libjpeg-turbo would apply
//! BLOCK SMOOTHING, not reproduced); blocks outside the range above; and damaged entropy data that
//! libjpeg-turbo would only warn about and patch (a bad Huffman code, a coefficient run past the band, a scan
//! cut short by a marker or the end of the file, a wrong restart marker, a bogus progression).
//!
//! **Measured** (Pillow 12.2.0 / libjpeg-turbo 3.1.4.1 on arm64; `images.rs` tests, `make_images.py`): 79
//! fixture JPEGs decode to PIL's exact pixels (max |Δ| 0) — every sampling layout above, baseline /
//! extended (16-bit DQT) / progressive with successive approximation, restart intervals interleaved and not,
//! qualities 1–100, sizes 1x1 to 333x217, RGB and greyscale colour spaces, a DQT redefined between scans, no
//! DHT — and so do MiMo-Embodied's 1083x722 demo photo and three 12 MP files (4:2:0, progressive, 4:4:4 q95:
//! 0 of 36 000 000 samples differ). Sixteen mutations (IDCT rounding, each fancy upsampler off, a bias, the
//! bottom context row, a colour-table entry, table latching, RST prediction reset, refinement sign, the
//! guard, …) each fail the tests. Decode time, 12 MP, release, single-threaded, this shared M5 Max at load
//! ~50 (3 rounds × 9 decodes, min–median ms): 4:2:0 102–256 vs zune-jpeg's 52–121, progressive 164–335 vs
//! 130–457, 4:4:4 q95 306–537 vs 155–358 — 1.0–2.3x zune-jpeg's time, against a 7B model's seconds per answer.
//!
//! ---
//! This module is derived from libjpeg-turbo, whose libjpeg API library is covered by the IJG License.
//! **This software is based in part on the work of the Independent JPEG Group.** The IJG README
//! (`README.ijg`, with its copyright and no-warranty notice unaltered) is included beside this file.
//! Changes: the C files named above were translated to Rust by the Ferric project (2026); buffering,
//! suspension, scaled/merged/quantised output, 12/16-bit and arithmetic/lossless paths were removed; libjpeg's
//! warnings on damaged data became errors. Original copyrights of the translated files:
//! Copyright (C) 1991-1998, Thomas G. Lane. Modified 2002-2018 by Guido Vollbeding. Lossless JPEG
//! Modifications: Copyright (C) 1999, Ken Murchison. libjpeg-turbo Modifications: Copyright (C) 2009-2026,
//! D. R. Commander; Copyright 2009 Pierre Ossman for Cendio AB; Copyright (C) 2013, Linaro Limited;
//! Copyright (C) 2014, MIPS Technologies, Inc.; Copyright (C) 2015, 2020, Google, Inc.; Copyright (C)
//! 2018, Matthias Räncker; Copyright (C) 2019-2020, Arm Limited.

/// `jpeg_natural_order` (jutils.c): zigzag index → position in the 8x8 block.
const NATURAL: [usize; 64] = [
    0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5, 12, 19, 26, 33, 40, 48, 41, 34, 27, 20, 13, 6, 7, 14, 21,
    28, 35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51, 58, 59, 52, 45, 38, 31, 39, 46, 53, 60, 61,
    54, 47, 55, 62, 63,
];

/// An image as PIL's JPEG plugin hands it on: RGB (a greyscale JPEG is replicated, as `convert("RGB")` does
/// with mode L), plus the metadata `exif_transpose` reads — PIL's `info["exif"]` and `info["xmp"]`, collected
/// by its rules (APP1 segments before the first SOS; later EXIF segments appended minus their "Exif\0\0").
pub(crate) struct Decoded {
    pub(crate) w: usize,
    pub(crate) h: usize,
    pub(crate) px: Vec<u8>,
    pub(crate) exif: Option<Vec<u8>>,
    pub(crate) xmp: Option<Vec<u8>>,
}

pub(crate) fn decode(d: &[u8]) -> Result<Decoded, String> {
    Dec::new(d).run().map_err(|e| format!("JPEG: {e}"))
}

// ─────────────────────────────── Huffman tables (jdhuff.c) ───────────────────────────────

#[derive(Clone)]
struct RawHuff {
    bits: [u8; 17],
    vals: [u8; 256],
}

/// Lookahead width. libjpeg-turbo uses 8 (HUFF_LOOKAHEAD); 9 finds the same codes (a canonical code of length
/// ≤ 9 is in the table, longer ones fall to the F.16 walk), one fewer slow step per 9-bit code.
const LOOK: u32 = 9;
/// Window of the combined code+value table (`Huff::fast`).
const FAST: u32 = 11;

struct Huff {
    /// (length << 8) | symbol for codes ≤ LOOK bits; 0 = longer (or no such code).
    look: Vec<u16>,
    /// AC tables: for a FAST-bit window holding a whole code AND its value bits, the decoded coefficient —
    /// value (low 16 bits), zero run (bits 16..20), bits consumed (bits 20..); 0 = take the general path.
    /// The same values the general path computes, in one step (cf. libjpeg-turbo's decode_mcu_fast).
    fast: Vec<u32>,
    maxcode: [i32; 18],
    valoff: [i32; 18],
    vals: [u8; 256],
}

impl Huff {
    /// `jpeg_make_d_derived_tbl`, with its validation (JERR_BAD_HUFF_TABLE).
    fn new(t: &RawHuff, dc: bool) -> Result<Huff, String> {
        let mut size = [0u8; 257];
        let mut p = 0usize;
        for l in 1..=16 {
            let i = t.bits[l] as usize;
            if p + i > 256 { return Err("bad Huffman table (more than 256 codes)".into()); }
            for _ in 0..i { size[p] = l as u8; p += 1; }
        }
        let n = p;
        let mut code_of = [0u32; 257];
        let (mut code, mut si) = (0u32, size[0] as u32);
        p = 0;
        while size[p] != 0 {
            while size[p] as u32 == si { code_of[p] = code; p += 1; code += 1; }
            if code as u64 >= 1u64 << si { return Err("bad Huffman table (code lengths overflow)".into()); }
            code <<= 1;
            si += 1;
        }
        let (mut maxcode, mut valoff) = ([0i32; 18], [0i32; 18]);
        p = 0;
        for l in 1..=16 {
            if t.bits[l] != 0 {
                valoff[l] = p as i32 - code_of[p] as i32;
                p += t.bits[l] as usize;
                maxcode[l] = code_of[p - 1] as i32;
            } else {
                maxcode[l] = -1;
            }
        }
        maxcode[17] = 0xFFFFF;
        let mut look = vec![0u16; 1 << LOOK];
        p = 0;
        for l in 1..=LOOK as usize {
            for _ in 0..t.bits[l] {
                let base = (code_of[p] << (LOOK as usize - l)) as usize;
                for k in 0..1usize << (LOOK as usize - l) { look[base + k] = ((l as u16) << 8) | t.vals[p] as u16; }
                p += 1;
            }
        }
        if dc && t.vals[..n].iter().any(|&s| s > 15) { return Err("bad Huffman table (DC symbol > 15)".into()); }
        let mut fast = Vec::new();
        if !dc {
            fast = vec![0u32; 1 << FAST];
            for (w, e) in fast.iter_mut().enumerate() {
                let l = look[w >> (FAST - LOOK)] as u32;
                let (len, rs) = (l >> 8, l & 0xFF);
                let s = rs & 15;
                if len == 0 || s == 0 || len + s > FAST { continue; }
                let v = (w as u32 >> (FAST - len - s)) & ((1 << s) - 1);
                *e = (extend(v, s) as u16 as u32) | (rs >> 4) << 16 | (len + s) << 20;
            }
        }
        Ok(Huff { look, fast, maxcode, valoff, vals: t.vals })
    }
}

/// `std_huff_tables` (jstdhuff.c): the Annex K tables, installed into EMPTY slots 0/1 for sequential files.
fn std_tables() -> [(bool, usize, RawHuff); 4] {
    fn t(bits: [u8; 17], v: &[u8]) -> RawHuff {
        let mut vals = [0u8; 256];
        vals[..v.len()].copy_from_slice(v);
        RawHuff { bits, vals }
    }
    let dc_vals: Vec<u8> = (0..12).collect();
    const AC_LUM: [u8; 162] = [
        0x01, 0x02, 0x03, 0x00, 0x04, 0x11, 0x05, 0x12, 0x21, 0x31, 0x41, 0x06, 0x13, 0x51, 0x61, 0x07, 0x22, 0x71, 0x14,
        0x32, 0x81, 0x91, 0xa1, 0x08, 0x23, 0x42, 0xb1, 0xc1, 0x15, 0x52, 0xd1, 0xf0, 0x24, 0x33, 0x62, 0x72, 0x82, 0x09,
        0x0a, 0x16, 0x17, 0x18, 0x19, 0x1a, 0x25, 0x26, 0x27, 0x28, 0x29, 0x2a, 0x34, 0x35, 0x36, 0x37, 0x38, 0x39, 0x3a,
        0x43, 0x44, 0x45, 0x46, 0x47, 0x48, 0x49, 0x4a, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58, 0x59, 0x5a, 0x63, 0x64, 0x65,
        0x66, 0x67, 0x68, 0x69, 0x6a, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78, 0x79, 0x7a, 0x83, 0x84, 0x85, 0x86, 0x87, 0x88,
        0x89, 0x8a, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97, 0x98, 0x99, 0x9a, 0xa2, 0xa3, 0xa4, 0xa5, 0xa6, 0xa7, 0xa8, 0xa9,
        0xaa, 0xb2, 0xb3, 0xb4, 0xb5, 0xb6, 0xb7, 0xb8, 0xb9, 0xba, 0xc2, 0xc3, 0xc4, 0xc5, 0xc6, 0xc7, 0xc8, 0xc9, 0xca,
        0xd2, 0xd3, 0xd4, 0xd5, 0xd6, 0xd7, 0xd8, 0xd9, 0xda, 0xe1, 0xe2, 0xe3, 0xe4, 0xe5, 0xe6, 0xe7, 0xe8, 0xe9, 0xea,
        0xf1, 0xf2, 0xf3, 0xf4, 0xf5, 0xf6, 0xf7, 0xf8, 0xf9, 0xfa,
    ];
    const AC_CHR: [u8; 162] = [
        0x00, 0x01, 0x02, 0x03, 0x11, 0x04, 0x05, 0x21, 0x31, 0x06, 0x12, 0x41, 0x51, 0x07, 0x61, 0x71, 0x13, 0x22, 0x32,
        0x81, 0x08, 0x14, 0x42, 0x91, 0xa1, 0xb1, 0xc1, 0x09, 0x23, 0x33, 0x52, 0xf0, 0x15, 0x62, 0x72, 0xd1, 0x0a, 0x16,
        0x24, 0x34, 0xe1, 0x25, 0xf1, 0x17, 0x18, 0x19, 0x1a, 0x26, 0x27, 0x28, 0x29, 0x2a, 0x35, 0x36, 0x37, 0x38, 0x39,
        0x3a, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48, 0x49, 0x4a, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58, 0x59, 0x5a, 0x63, 0x64,
        0x65, 0x66, 0x67, 0x68, 0x69, 0x6a, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78, 0x79, 0x7a, 0x82, 0x83, 0x84, 0x85, 0x86,
        0x87, 0x88, 0x89, 0x8a, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97, 0x98, 0x99, 0x9a, 0xa2, 0xa3, 0xa4, 0xa5, 0xa6, 0xa7,
        0xa8, 0xa9, 0xaa, 0xb2, 0xb3, 0xb4, 0xb5, 0xb6, 0xb7, 0xb8, 0xb9, 0xba, 0xc2, 0xc3, 0xc4, 0xc5, 0xc6, 0xc7, 0xc8,
        0xc9, 0xca, 0xd2, 0xd3, 0xd4, 0xd5, 0xd6, 0xd7, 0xd8, 0xd9, 0xda, 0xe2, 0xe3, 0xe4, 0xe5, 0xe6, 0xe7, 0xe8, 0xe9,
        0xea, 0xf2, 0xf3, 0xf4, 0xf5, 0xf6, 0xf7, 0xf8, 0xf9, 0xfa,
    ];
    [
        (true, 0, t([0, 0, 1, 5, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0], &dc_vals)),
        (false, 0, t([0, 0, 2, 1, 3, 3, 2, 4, 3, 5, 5, 4, 4, 0, 0, 1, 0x7d], &AC_LUM)),
        (true, 1, t([0, 0, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0], &dc_vals)),
        (false, 1, t([0, 0, 2, 1, 2, 4, 4, 3, 4, 7, 5, 4, 4, 0, 1, 2, 0x77], &AC_CHR)),
    ]
}

// ─────────────────────────────── bit reader (jdhuff.c jpeg_fill_bit_buffer) ───────────────────────────────

#[derive(Clone, Copy, PartialEq)]
enum Stop {
    /// A marker ended the entropy-coded data: its code and the offset just past it.
    Marker(u8, usize),
    Eof,
}

struct Bits<'a> {
    d: &'a [u8],
    pos: usize,
    /// Left-aligned: the next bit is bit 63.
    buf: u64,
    n: u32,
    /// Zero bits appended after the data ended (libjpeg's "fake zero bits"). Consuming any means the scan
    /// needed more data than it had: libjpeg-turbo would warn and zero-fill — refused here (`n < fake`).
    fake: u32,
    stop: Option<Stop>,
}

impl<'a> Bits<'a> {
    fn new(d: &'a [u8], pos: usize) -> Self { Bits { d, pos, buf: 0, n: 0, fake: 0, stop: None } }

    #[inline(always)]
    fn fill(&mut self) { if self.n <= 56 { self.refill(); } }

    fn refill(&mut self) {
        // Fast path: 8 bytes with no 0xFF among them.
        if self.stop.is_none() && self.pos + 8 <= self.d.len() {
            let w = u64::from_be_bytes(self.d[self.pos..self.pos + 8].try_into().unwrap());
            let x = !w;
            if x.wrapping_sub(0x0101_0101_0101_0101) & !x & 0x8080_8080_8080_8080 == 0 {
                let take = (64 - self.n) / 8;
                self.buf |= (w >> (64 - 8 * take)) << (64 - self.n - 8 * take);
                self.pos += take as usize;
                self.n += 8 * take;
                return;
            }
        }
        while self.n <= 56 { self.byte(); }
    }

    /// One byte, with libjpeg's rules: FF 00 is an FF data byte, and so is FF FF … FF 00 (padding FFs are
    /// swallowed); FF followed by anything else is a marker that ends the data.
    fn byte(&mut self) {
        if self.stop.is_some() { self.n += 8; self.fake += 8; return; }
        let Some(&c) = self.d.get(self.pos) else { self.stop = Some(Stop::Eof); return self.byte(); };
        let c = if c == 0xFF {
            let mut p = self.pos + 1;
            while p < self.d.len() && self.d[p] == 0xFF { p += 1; }
            match self.d.get(p) {
                Some(0) => { self.pos = p + 1; 0xFF }
                Some(&m) => { self.stop = Some(Stop::Marker(m, p + 1)); return self.byte(); }
                None => { self.stop = Some(Stop::Eof); return self.byte(); }
            }
        } else {
            self.pos += 1;
            c
        };
        self.buf |= (c as u64) << (56 - self.n);
        self.n += 8;
    }

    #[inline(always)]
    fn skip(&mut self, k: u32) { self.buf <<= k; self.n -= k; }

    /// `k` in 1..=16 raw bits.
    #[inline(always)]
    fn bits(&mut self, k: u32) -> u32 {
        self.fill();
        let v = (self.buf >> (64 - k)) as u32;
        self.skip(k);
        v
    }

    #[inline(always)]
    fn huff(&mut self, h: &Huff) -> Result<u8, String> {
        self.fill();
        let e = h.look[(self.buf >> (64 - LOOK)) as usize];
        if e != 0 {
            self.skip((e >> 8) as u32);
            return Ok(e as u8);
        }
        // Figure F.16 for codes longer than the lookahead (`jpeg_huff_decode`).
        let mut l = LOOK as usize + 1;
        while (self.buf >> (64 - l)) as i32 > h.maxcode[l] { l += 1; }
        if l > 16 { return Err("corrupt data: bad Huffman code (libjpeg-turbo would substitute 0)".into()); }
        let code = (self.buf >> (64 - l)) as i32;
        self.skip(l as u32);
        Ok(h.vals[(code + h.valoff[l]) as usize])
    }

    fn overran(&self) -> bool { self.n < self.fake }
}

/// HUFF_EXTEND.
#[inline(always)]
fn extend(v: u32, s: u32) -> i32 {
    if v < 1 << (s - 1) { v as i32 + ((-1i32) << s) + 1 } else { v as i32 }
}

/// `next_marker` (jdmarker.c): skip non-FF bytes, swallow FF padding, skip FF 00; return the marker code and
/// the offset after it.
fn next_marker(d: &[u8], mut p: usize) -> Option<(u8, usize)> {
    loop {
        while p < d.len() && d[p] != 0xFF { p += 1; }
        while p < d.len() && d[p] == 0xFF { p += 1; }
        let &c = d.get(p)?;
        p += 1;
        if c != 0 { return Some((c, p)); }
    }
}

// ─────────────────────────────── frame and decoder state ───────────────────────────────

struct Comp {
    id: u8,
    h: usize,
    v: usize,
    tq: usize,
    /// Latched quantisation table, natural order, as `short` multipliers.
    q: Option<[i16; 64]>,
    /// width_in_blocks / height_in_blocks: blocks that reach the output.
    bw: usize,
    bh: usize,
    /// downsampled_width / height: samples that reach the upsampler.
    dw: usize,
    dh: usize,
    /// Allocated block grid, padded to whole MCUs of an interleaved scan.
    sw: usize,
    coef: Vec<i16>,
    /// Progressive: per zigzag coefficient, -1 = never coded, else the Al of its last scan (jdphuff.c).
    coef_bits: [i32; 64],
}

struct Frame {
    w: usize,
    h: usize,
    prog: bool,
    comps: Vec<Comp>,
    hmax: usize,
    vmax: usize,
    mcux: usize,
    mcuy: usize,
}

struct Dec<'a> {
    d: &'a [u8],
    qt: [Option<[u16; 64]>; 4],
    dc: [Option<RawHuff>; 4],
    ac: [Option<RawHuff>; 4],
    restart: usize,
    frame: Option<Frame>,
    jfif: bool,
    adobe: Option<u8>,
    exif: Option<Vec<u8>>,
    xmp: Option<Vec<u8>>,
    scans: usize,
    multi: bool,
}

fn be16(d: &[u8], p: usize) -> Result<usize, String> {
    d.get(p..p + 2).map(|s| (s[0] as usize) << 8 | s[1] as usize).ok_or_else(|| "truncated (inside a marker segment)".to_string())
}

impl<'a> Dec<'a> {
    fn new(d: &'a [u8]) -> Self {
        Dec { d, qt: [None; 4], dc: Default::default(), ac: Default::default(), restart: 0, frame: None, jfif: false,
              adobe: None, exif: None, xmp: None, scans: 0, multi: false }
    }

    /// A marker segment's body and the offset after it.
    fn segment(&self, p: usize) -> Result<(&'a [u8], usize), String> {
        let n = be16(self.d, p)?;
        if n < 2 { return Err(format!("bad marker segment length {n}")); }
        let body = self.d.get(p + 2..p + n).ok_or("truncated (inside a marker segment)")?;
        Ok((body, p + n))
    }

    fn run(mut self) -> Result<Decoded, String> {
        if self.d.len() < 2 || self.d[0] != 0xFF || self.d[1] != 0xD8 { return Err("no SOI marker".into()); }
        let mut pos = 2;
        let mut pending: Option<(u8, usize)> = None;
        loop {
            let (m, p) = match pending.take().or_else(|| next_marker(self.d, pos)) {
                Some(x) => x,
                // A single-scan image is complete once its scan is decoded: Pillow tolerates a missing EOI
                // there (`jpeg_finish_decompress` suspends after every row is out). A multi-scan image is
                // decoded only at EOI, so there the file is truncated (PIL: "image file is truncated").
                None if self.scans > 0 && !self.multi => break,
                None => return Err(if self.scans == 0 { "no image data (no SOS before the end)" } else { "truncated (the file ends before EOI)" }.into()),
            };
            pos = p;
            match m {
                0xD8 => return Err("a second SOI marker".into()),
                0xC0..=0xC2 => { let (s, n) = self.segment(p)?; self.sof(s, m == 0xC2)?; pos = n; }
                0xC3 => return Err("lossless JPEG (SOF3) is not read; the reference decodes it, this does not — convert it".into()),
                0xC9..=0xCB => return Err(format!("arithmetic-coded JPEG (SOF{}) is not read — convert it", m - 0xC0)),
                0xC5..=0xC7 | 0xC8 | 0xCD..=0xCF => return Err(format!("hierarchical/differential JPEG (marker {m:#04X}) is not supported (nor by libjpeg-turbo)")),
                0xC4 => { let (s, n) = self.segment(p)?; self.dht(s)?; pos = n; }
                0xDB => { let (s, n) = self.segment(p)?; self.dqt(s)?; pos = n; }
                0xDD => {
                    let (s, n) = self.segment(p)?;
                    if s.len() != 2 { return Err("bad DRI length".into()); }
                    self.restart = be16(s, 0)?;
                    pos = n;
                }
                0xDA => {
                    let (s, n) = self.segment(p)?;
                    let (next, after) = self.sos(s, n)?;
                    pending = next;
                    pos = after;
                }
                0xD9 => {
                    if self.scans == 0 { return Err("no image data (EOI before any scan)".into()); }
                    break;
                }
                0xE0..=0xEF => { let (s, n) = self.segment(p)?; self.app(m, s); pos = n; }
                0xFE | 0xCC | 0xDC => { let (_, n) = self.segment(p)?; pos = n; } // COM, DAC, DNL (skipped, as libjpeg does)
                0xD0..=0xD7 | 0x01 => {}                                          // RSTn, TEM: parameterless, ignored
                _ => return Err(format!("unknown marker {m:#04X}")),
            }
        }
        let px = self.output()?;
        let f = self.frame.as_ref().unwrap();
        Ok(Decoded { w: f.w, h: f.h, px, exif: self.exif, xmp: self.xmp })
    }

    /// APPn: libjpeg's JFIF / Adobe detection (`examine_app0`, `examine_app14`: at least 14 / 12 bytes), and
    /// PIL's EXIF / XMP collection — both only before the first SOS (libjpeg guesses the colour space there;
    /// PIL's parser stops there).
    fn app(&mut self, m: u8, s: &[u8]) {
        if self.scans > 0 { return; }
        match m {
            0xE0 if s.len() >= 14 && s.starts_with(b"JFIF\0") => self.jfif = true,
            0xEE if s.len() >= 12 && s.starts_with(b"Adobe") => self.adobe = Some(s[11]),
            0xE1 if s.starts_with(b"Exif\0\0") => match &mut self.exif {
                Some(e) => e.extend_from_slice(&s[6..]),
                None => self.exif = Some(s.to_vec()),
            },
            0xE1 if s.starts_with(b"http://ns.adobe.com/xap/1.0/\0") => {
                let z = s.iter().position(|&b| b == 0).unwrap();
                self.xmp = Some(s[z + 1..].to_vec());
            }
            _ => {}
        }
    }

    /// `get_dqt`. Only full 64-entry tables (a shorter one is a jpeg-9 scaled-DCT table libjpeg pads with 1s).
    fn dqt(&mut self, mut s: &[u8]) -> Result<(), String> {
        while !s.is_empty() {
            let (prec, n) = ((s[0] >> 4) as usize, (s[0] & 15) as usize);
            if n >= 4 { return Err(format!("bad DQT table index {n}")); }
            let len = if prec != 0 { 128 } else { 64 };
            let body = s.get(1..1 + len).ok_or("DQT shorter than 64 entries (not supported)")?;
            let mut t = [0u16; 64];
            for i in 0..64 {
                t[NATURAL[i]] = if prec != 0 { (body[2 * i] as u16) << 8 | body[2 * i + 1] as u16 } else { body[i] as u16 };
            }
            self.qt[n] = Some(t);
            s = &s[1 + len..];
        }
        Ok(())
    }

    /// `get_dht`.
    fn dht(&mut self, mut s: &[u8]) -> Result<(), String> {
        while s.len() > 16 {
            let index = s[0] as usize;
            let mut bits = [0u8; 17];
            bits[1..].copy_from_slice(&s[1..17]);
            let count: usize = bits.iter().map(|&b| b as usize).sum();
            if count > 256 || count > s.len() - 17 { return Err("bad Huffman table".into()); }
            let mut vals = [0u8; 256];
            vals[..count].copy_from_slice(&s[17..17 + count]);
            let t = RawHuff { bits, vals };
            if index & 0x10 != 0 {
                let i = index - 0x10;
                if i >= 4 { return Err(format!("bad DHT index {index:#x}")); }
                self.ac[i] = Some(t);
            } else {
                if index >= 4 { return Err(format!("bad DHT index {index:#x}")); }
                self.dc[index] = Some(t);
            }
            s = &s[17 + count..];
        }
        if !s.is_empty() { return Err("bad DHT length".into()); }
        Ok(())
    }

    /// `get_sof` + `initial_setup` (jdinput.c): geometry and validation.
    fn sof(&mut self, s: &[u8], prog: bool) -> Result<(), String> {
        if self.frame.is_some() { return Err("a second SOF marker".into()); }
        if s.len() < 6 { return Err("bad SOF length".into()); }
        let prec = s[0];
        let (h, w, nc) = (be16(s, 1)?, be16(s, 3)?, s[5] as usize);
        if prec != 8 { return Err(format!("{prec}-bit samples are not read (PIL reads 8-bit JPEG only)")); }
        if h == 0 { return Err("height 0 (defined later by DNL) is not supported (nor by libjpeg-turbo)".into()); }
        if w == 0 || nc == 0 { return Err("empty image".into()); }
        if s.len() != 6 + 3 * nc { return Err("bad SOF length".into()); }
        match nc {
            1 | 3 => {}
            4 => return Err("CMYK/YCCK JPEG is not read (PIL's inverted-CMYK→RGB path is not reproduced) — convert it to RGB".into()),
            n => return Err(format!("{n}-component JPEG (PIL reads 1, 3 or 4)")),
        }
        if w > 65500 || h > 65500 { return Err("image larger than 65500 pixels (libjpeg-turbo's limit)".into()); }
        // `PIL.Image.open` raises DecompressionBombError past 2 × MAX_IMAGE_PIXELS (2 × 89 478 485), so the
        // reference pipeline never decodes such an image — and a 30-byte header must not buy gigabytes here.
        if w * h > 2 * 89_478_485 {
            return Err(format!("{w}x{h} exceeds 178 956 970 pixels, PIL's decompression-bomb limit (the reference refuses it)"));
        }
        let mut comps = Vec::with_capacity(nc);
        for i in 0..nc {
            let c = &s[6 + 3 * i..9 + 3 * i];
            let (hs, vs) = ((c[1] >> 4) as usize, (c[1] & 15) as usize);
            if !(1..=4).contains(&hs) || !(1..=4).contains(&vs) { return Err("bad sampling factors".into()); }
            comps.push(Comp { id: c[0], h: hs, v: vs, tq: c[2] as usize, q: None, bw: 0, bh: 0, dw: 0, dh: 0, sw: 0,
                              coef: Vec::new(), coef_bits: [-1; 64] });
        }
        let hmax = comps.iter().map(|c| c.h).max().unwrap();
        let vmax = comps.iter().map(|c| c.v).max().unwrap();
        let (mcux, mcuy) = (w.div_ceil(8 * hmax), h.div_ceil(8 * vmax));
        for c in &mut comps {
            c.bw = (w * c.h).div_ceil(8 * hmax);
            c.bh = (h * c.v).div_ceil(8 * vmax);
            c.dw = (w * c.h).div_ceil(hmax);
            c.dh = (h * c.v).div_ceil(vmax);
            c.sw = mcux * c.h;
            c.coef = vec![0i16; c.sw * mcuy * c.v * 64];
        }
        self.frame = Some(Frame { w, h, prog, comps, hmax, vmax, mcux, mcuy });
        Ok(())
    }

    /// `get_sos` + the scan: returns the marker that ended the entropy data (if the reader met one) and the
    /// offset to continue from.
    fn sos(&mut self, s: &[u8], data: usize) -> Result<(Option<(u8, usize)>, usize), String> {
        let f = self.frame.as_mut().ok_or("SOS before SOF")?;
        let n = *s.first().ok_or("bad SOS length")? as usize;
        if s.len() != 4 + 2 * n || !(1..=4).contains(&n) { return Err("bad SOS length".into()); }
        let mut sc: Vec<(usize, usize, usize)> = Vec::with_capacity(n);
        for i in 0..n {
            let (cc, t) = (s[1 + 2 * i], s[2 + 2 * i]);
            // libjpeg's lookup: the first frame component with this id whose index is not yet a filled scan
            // slot (`!cinfo->cur_comp_info[ci]`, so ci >= i), among the first 4.
            let ci = (0..f.comps.len().min(4)).find(|&ci| f.comps[ci].id == cc && ci >= i)
                .ok_or_else(|| format!("bad component id {cc} in SOS"))?;
            if sc.iter().any(|x| x.0 == ci) { return Err(format!("component id {cc} twice in one SOS")); }
            sc.push((ci, (t >> 4) as usize, (t & 15) as usize));
        }
        let (ss, se, ah, al) = (s[1 + 2 * n] as usize, s[2 + 2 * n] as usize, (s[3 + 2 * n] >> 4) as u32, (s[3 + 2 * n] & 15) as u32);
        if self.scans == 0 {
            self.multi = n < f.comps.len() || f.prog;
            if !f.prog {
                for (dc, slot, t) in std_tables() {
                    let tbl = if dc { &mut self.dc[slot] } else { &mut self.ac[slot] };
                    if tbl.is_none() { *tbl = Some(t); }
                }
            }
        } else if !self.multi {
            return Err("a second scan after a complete single-scan image (libjpeg-turbo: JERR_EOI_EXPECTED)".into());
        }
        self.scans += 1;
        // latch_quant_tables
        for &(ci, _, _) in &sc {
            let c = &mut f.comps[ci];
            if c.q.is_none() {
                let t = self.qt.get(c.tq).and_then(|t| t.as_ref()).ok_or_else(|| format!("no quantisation table {}", c.tq))?;
                let mut q = [0i16; 64];
                for i in 0..64 { q[i] = t[i] as i16; } // MULTIPLIER is `short` in a WITH_SIMD build
                c.q = Some(q);
            }
        }
        let tbl = |set: &[Option<RawHuff>; 4], i: usize, dc: bool| -> Result<Huff, String> {
            Huff::new(set.get(i).and_then(|t| t.as_ref()).ok_or_else(|| format!("no Huffman table {i}"))?, dc)
        };
        let mode = if f.prog {
            // start_pass_phuff_decoder: validation (JERR_BAD_PROGRESSION) and the progression bookkeeping;
            // its warnings (JWRN_BOGUS_PROGRESSION) are refusals here.
            let dc_band = ss == 0;
            let bad = if dc_band { se != 0 } else { ss > se || se >= 64 || n != 1 } || (ah != 0 && al + 1 != ah) || al > 13;
            if bad { return Err(format!("bad progression parameters Ss={ss} Se={se} Ah={ah} Al={al}")); }
            for &(ci, _, _) in &sc {
                let cb = &mut f.comps[ci].coef_bits;
                if !dc_band && cb[0] < 0 { return Err("corrupt progression: an AC scan before the DC scan".into()); }
                for k in ss..=se {
                    let expected = cb[k].max(0);
                    if ah as i32 != expected { return Err("corrupt progression: a refinement out of order".into()); }
                    cb[k] = al as i32;
                }
            }
            if dc_band && ah == 0 {
                Mode::DcFirst(sc.iter().map(|x| tbl(&self.dc, x.1, true)).collect::<Result<_, _>>()?)
            } else if dc_band {
                Mode::DcRefine
            } else if ah == 0 {
                Mode::AcFirst(tbl(&self.ac, sc[0].2, false)?)
            } else {
                Mode::AcRefine(tbl(&self.ac, sc[0].2, false)?)
            }
        } else {
            // jdhuff.c ignores Ss/Se/Ah/Al in a sequential scan (a warning: "some baseline files out there
            // have all zeroes in these bytes") and decodes all 64 coefficients.
            let dcs = sc.iter().map(|x| tbl(&self.dc, x.1, true)).collect::<Result<Vec<_>, _>>()?;
            let acs = sc.iter().map(|x| tbl(&self.ac, x.2, false)).collect::<Result<Vec<_>, _>>()?;
            Mode::Seq(dcs, acs)
        };
        let comps: Vec<usize> = sc.iter().map(|x| x.0).collect();
        let mut r = Bits::new(self.d, data);
        decode_scan(f, &comps, &mode, ss, se, al, self.restart, &mut r)?;
        if r.overran() || r.stop == Some(Stop::Eof) {
            return Err("truncated or corrupt: the scan's data ends early (libjpeg-turbo would fill with zeros)".into());
        }
        Ok(match r.stop {
            Some(Stop::Marker(m, after)) => (Some((m, after)), after),
            _ => (None, r.pos),
        })
    }

    /// Everything after the coefficients: `smoothing_ok`, IDCT, upsampling, colour conversion.
    fn output(&self) -> Result<Vec<u8>, String> {
        let f = self.frame.as_ref().unwrap();
        if f.prog && smoothing_ok(f) {
            return Err("progressive JPEG with incompletely coded low-frequency coefficients: libjpeg-turbo would \
                        apply block smoothing, which is not reproduced".into());
        }
        let planes: Vec<Vec<u8>> = f.comps.iter().map(idct_component).collect::<Result<_, _>>()?;
        let (w, h) = (f.w, f.h);
        let mut px = vec![0u8; w * h * 3];
        let ups: Vec<Up> = f.comps.iter().map(|c| upsampler(f, c)).collect::<Result<_, _>>()?;
        let color = match f.comps.len() {
            1 => Color::Gray,
            _ if self.jfif => Color::Ycc,
            _ => match self.adobe {
                Some(0) => Color::Rgb,
                Some(_) => Color::Ycc,
                None => {
                    let ids = (f.comps[0].id, f.comps[1].id, f.comps[2].id);
                    if ids == (82, 71, 66) { Color::Rgb } else { Color::Ycc }
                }
            },
        };
        let t = YccTables::new();
        let mut rows: Vec<Vec<u8>> = f.comps.iter().map(|c| vec![0u8; (w.max(2 * c.dw)) + 8]).collect();
        let mut sums: Vec<i32> = vec![0; f.comps.iter().map(|c| c.dw).max().unwrap() + 2];
        for y in 0..h {
            for (i, c) in f.comps.iter().enumerate() {
                upsample_row(&ups[i], c, &planes[i], y, &mut rows[i], &mut sums);
            }
            let out = &mut px[y * w * 3..(y + 1) * w * 3];
            match color {
                Color::Gray => for (o, &g) in out.chunks_exact_mut(3).zip(&rows[0][..w]) { o.fill(g); },
                Color::Rgb => for (x, o) in out.chunks_exact_mut(3).enumerate() {
                    o.copy_from_slice(&[rows[0][x], rows[1][x], rows[2][x]]);
                },
                Color::Ycc => t.convert(&rows[0][..w], &rows[1][..w], &rows[2][..w], out),
            }
        }
        Ok(px)
    }
}

enum Color { Gray, Rgb, Ycc }

// ─────────────────────────────── entropy decoding of a scan ───────────────────────────────

enum Mode {
    Seq(Vec<Huff>, Vec<Huff>),
    DcFirst(Vec<Huff>),
    DcRefine,
    AcFirst(Huff),
    AcRefine(Huff),
}

#[allow(clippy::too_many_arguments)]
fn decode_scan(f: &mut Frame, sc: &[usize], mode: &Mode, ss: usize, se: usize, al: u32, ri: usize, r: &mut Bits) -> Result<(), String> {
    let single = sc.len() == 1;
    let (mx, my) = if single { (f.comps[sc[0]].bw, f.comps[sc[0]].bh) } else { (f.mcux, f.mcuy) };
    // Blocks of each MCU in scan order: (component slot in scan, frame component, block x, block y offsets).
    let mut layout: Vec<(usize, usize, usize, usize)> = Vec::new();
    for (k, &ci) in sc.iter().enumerate() {
        let c = &f.comps[ci];
        let (h, v) = if single { (1, 1) } else { (c.h, c.v) };
        for by in 0..v { for bx in 0..h { layout.push((k, ci, bx, by)); } }
    }
    if layout.len() > 10 { return Err("more than 10 blocks in an MCU (libjpeg-turbo: JERR_BAD_MCU_SIZE)".into()); }
    let mut pred = [0i32; 4];
    let mut eobrun = 0u32;
    let (mut togo, mut next_rst) = (ri, 0u8);
    for mcu_y in 0..my {
        for mcu_x in 0..mx {
            if ri > 0 {
                if togo == 0 {
                    restart(r, &mut next_rst)?;
                    pred = [0; 4];
                    eobrun = 0;
                    togo = ri;
                }
                togo -= 1;
            }
            for &(k, ci, bx, by) in &layout {
                let c = &mut f.comps[ci];
                let (h, v) = if single { (1, 1) } else { (c.h, c.v) };
                let (x, y) = (mcu_x * h + bx, mcu_y * v + by);
                let at = (y * c.sw + x) * 64;
                let blk: &mut [i16] = &mut c.coef[at..at + 64];
                match mode {
                    Mode::Seq(dcs, acs) => seq_block(r, &dcs[k], &acs[k], &mut pred[k], blk)?,
                    Mode::DcFirst(dcs) => {
                        let s = r.huff(&dcs[k])? as u32;
                        let diff = if s != 0 { extend(r.bits(s), s) } else { 0 };
                        pred[k] = pred[k].checked_add(diff).ok_or("corrupt data: DC overflow (JERR_BAD_DCT_COEF)")?;
                        blk[0] = ((pred[k] as i64) << al) as i16;
                    }
                    Mode::DcRefine => { if r.bits(1) != 0 { blk[0] |= (1i32 << al) as i16; } }
                    Mode::AcFirst(t) => ac_first(r, t, ss, se, al, &mut eobrun, blk)?,
                    Mode::AcRefine(t) => ac_refine(r, t, ss, se, al, &mut eobrun, blk)?,
                }
            }
        }
    }
    Ok(())
}

/// `process_restart` + `read_restart_marker`: drop the bit buffer, expect RSTn (n counting mod 8).
fn restart(r: &mut Bits, next: &mut u8) -> Result<(), String> {
    if r.overran() { return Err("truncated or corrupt: a restart interval's data ends early".into()); }
    let (m, after) = match r.stop {
        Some(Stop::Marker(m, a)) => (m, a),
        Some(Stop::Eof) => return Err("truncated (inside the entropy-coded data)".into()),
        None => next_marker(r.d, r.pos).ok_or("truncated (inside the entropy-coded data)")?,
    };
    if m != 0xD0 + *next {
        return Err(format!("corrupt data: expected RST{} but found marker {m:#04X} (libjpeg-turbo would resynchronise)", *next));
    }
    *next = (*next + 1) & 7;
    *r = Bits::new(r.d, after);
    Ok(())
}

/// `decode_mcu_slow` for one block.
#[inline(always)]
fn seq_block(r: &mut Bits, dc: &Huff, ac: &Huff, pred: &mut i32, blk: &mut [i16]) -> Result<(), String> {
    let s = r.huff(dc)? as u32;
    let diff = if s != 0 { extend(r.bits(s), s) } else { 0 };
    *pred = pred.wrapping_add(diff);
    blk[0] = *pred as i16;
    let mut k = 1usize;
    while k < 64 {
        r.fill();
        let e = ac.fast[(r.buf >> (64 - FAST)) as usize];
        if e != 0 {
            k += (e >> 16 & 15) as usize;
            if k > 63 { return Err("corrupt data: a coefficient run past the block".into()); }
            blk[NATURAL[k]] = e as u16 as i16;
            r.skip(e >> 20);
            k += 1;
            continue;
        }
        let rs = r.huff(ac)?;
        let (run, s) = ((rs >> 4) as usize, (rs & 15) as u32);
        if s != 0 {
            k += run;
            if k > 63 { return Err("corrupt data: a coefficient run past the block".into()); }
            blk[NATURAL[k]] = extend(r.bits(s), s) as i16;
            k += 1;
        } else {
            if run != 15 { break; }
            k += 16;
        }
    }
    Ok(())
}

/// `decode_mcu_AC_first`.
fn ac_first(r: &mut Bits, t: &Huff, ss: usize, se: usize, al: u32, eobrun: &mut u32, blk: &mut [i16]) -> Result<(), String> {
    if *eobrun > 0 { *eobrun -= 1; return Ok(()); }
    let mut k = ss;
    while k <= se {
        let rs = r.huff(t)?;
        let (run, s) = ((rs >> 4) as usize, (rs & 15) as u32);
        if s != 0 {
            k += run;
            if k > se { return Err("corrupt data: a coefficient run past the band".into()); }
            blk[NATURAL[k]] = ((extend(r.bits(s), s) as i64) << al) as i16;
        } else if run == 15 {
            k += 15;
        } else {
            let mut e = 1u32 << run;
            if run != 0 { e += r.bits(run as u32); }
            *eobrun = e - 1;
            break;
        }
        k += 1;
    }
    Ok(())
}

/// `decode_mcu_AC_refine`.
fn ac_refine(r: &mut Bits, t: &Huff, ss: usize, se: usize, al: u32, eobrun: &mut u32, blk: &mut [i16]) -> Result<(), String> {
    let (p1, m1) = ((1i32 << al) as i16, ((-1i32) << al) as i16);
    let refine = |r: &mut Bits, c: &mut i16| {
        if r.bits(1) != 0 && (*c & p1) == 0 { *c = if *c >= 0 { c.wrapping_add(p1) } else { c.wrapping_add(m1) }; }
    };
    let mut k = ss;
    if *eobrun == 0 {
        while k <= se {
            let rs = r.huff(t)?;
            let (mut run, s0) = ((rs >> 4) as i32, rs & 15);
            let mut s = 0i16;
            if s0 != 0 {
                if s0 != 1 { return Err("corrupt data: a refinement coefficient of size > 1".into()); }
                s = if r.bits(1) != 0 { p1 } else { m1 };
            } else if run != 15 {
                *eobrun = 1 << run;
                if run != 0 { *eobrun += r.bits(run as u32); }
                break;
            }
            loop {
                let c = &mut blk[NATURAL[k]];
                if *c != 0 {
                    refine(r, c);
                } else {
                    run -= 1;
                    if run < 0 { break; }
                }
                k += 1;
                if k > se { break; }
            }
            if s != 0 {
                if k > se { return Err("corrupt data: a coefficient run past the band".into()); }
                blk[NATURAL[k]] = s;
            }
            k += 1;
        }
    }
    if *eobrun > 0 {
        while k <= se {
            let c = &mut blk[NATURAL[k]];
            if *c != 0 { refine(r, c); }
            k += 1;
        }
        *eobrun -= 1;
    }
    Ok(())
}

/// `smoothing_ok` (jdcoefct.c, SAVED_COEFS = 10): would libjpeg-turbo block-smooth this progressive image?
fn smoothing_ok(f: &Frame) -> bool {
    let mut useful = false;
    for c in &f.comps {
        let Some(q) = &c.q else { return false };
        // DC and the first 9 AC multipliers nonzero: natural positions 0,1,8,16,9,2,3,10,17,24.
        if [0, 1, 8, 16, 9, 2, 3, 10, 17, 24].iter().any(|&i| q[i] == 0) { return false; }
        if c.coef_bits[0] < 0 { return false; }
        if c.coef_bits[1..10].iter().any(|&b| b != 0) { useful = true; }
    }
    useful
}

// ─────────────────────────────── IDCT (jidctint.c jpeg_idct_islow) ───────────────────────────────

const CONST_BITS: u32 = 13;
const PASS1_BITS: u32 = 2;
const FIX_0_298631336: i32 = 2446;
const FIX_0_390180644: i32 = 3196;
const FIX_0_541196100: i32 = 4433;
const FIX_0_765366865: i32 = 6270;
const FIX_0_899976223: i32 = 7373;
const FIX_1_175875602: i32 = 9633;
const FIX_1_501321110: i32 = 12299;
const FIX_1_847759065: i32 = 15137;
const FIX_1_961570560: i32 = 16069;
const FIX_2_053119869: i32 = 16819;
const FIX_2_562915447: i32 = 20995;
const FIX_3_072711026: i32 = 25172;

fn idct_component(c: &Comp) -> Result<Vec<u8>, String> {
    let stride = c.bw * 8;
    let mut plane = vec![0u8; stride * c.bh * 8];
    // A component no scan ever coded has no latched table: jddctmgr leaves its multipliers zeroed.
    let q = c.q.unwrap_or([0; 64]);
    for by in 0..c.bh {
        for bx in 0..c.bw {
            let at = (by * c.sw + bx) * 64;
            if !idct_islow(&c.coef[at..at + 64], &q, &mut plane[by * 8 * stride + bx * 8..], stride) {
                return Err(format!(
                    "coefficient data outside what an encoder produces (block {bx},{by}: a dequantised value outside \
                     [-8192, 8191], an IDCT intermediate outside [-16384, 16383] or an output outside [-512, 511]); \
                     libjpeg-turbo's C and SIMD paths disagree there, so the reference's pixels depend on the machine — \
                     not decoded"));
            }
        }
    }
    Ok(plane)
}

/// One pass of `jpeg_idct_islow` on 8 independent lanes: `r[k]` holds frequency k of each lane; returns
/// spatial sample j of each lane, DESCALEd by `n`. The C code's per-column / per-row "AC terms all zero"
/// shortcuts are not needed: with the AC terms zero this computes exactly the value they shortcut to.
/// Wrapping 32-bit arithmetic is exact here: under the range guards of `idct_islow` every FINAL sum stays
/// below 1.6e9 in magnitude (bounded term by term), so arithmetic mod 2^32 equals C's 64-bit `JLONG` result
/// even where a partial sum in C's order would pass 2^31.
#[inline(always)]
fn idct_pass(r: &[[i32; 8]; 8], n: u32) -> [[i32; 8]; 8] {
    let mut o = [[0i32; 8]; 8];
    let round = 1i32 << (n - 1);
    for l in 0..8 {
        let (z2, z3) = (r[2][l], r[6][l]);
        let z1 = z2.wrapping_add(z3).wrapping_mul(FIX_0_541196100);
        let tmp2 = z1.wrapping_add(z3.wrapping_mul(-FIX_1_847759065));
        let tmp3 = z1.wrapping_add(z2.wrapping_mul(FIX_0_765366865));
        let tmp0 = r[0][l].wrapping_add(r[4][l]).wrapping_shl(CONST_BITS);
        let tmp1 = r[0][l].wrapping_sub(r[4][l]).wrapping_shl(CONST_BITS);
        let (tmp10, tmp13) = (tmp0.wrapping_add(tmp3), tmp0.wrapping_sub(tmp3));
        let (tmp11, tmp12) = (tmp1.wrapping_add(tmp2), tmp1.wrapping_sub(tmp2));
        let (t0, t1, t2, t3) = (r[7][l], r[5][l], r[3][l], r[1][l]);
        let (z1, z2, z3, z4) = (t0.wrapping_add(t3), t1.wrapping_add(t2), t0.wrapping_add(t2), t1.wrapping_add(t3));
        let z5 = z3.wrapping_add(z4).wrapping_mul(FIX_1_175875602);
        let (z1, z2) = (z1.wrapping_mul(-FIX_0_899976223), z2.wrapping_mul(-FIX_2_562915447));
        let z3 = z3.wrapping_mul(-FIX_1_961570560).wrapping_add(z5);
        let z4 = z4.wrapping_mul(-FIX_0_390180644).wrapping_add(z5);
        let t0 = t0.wrapping_mul(FIX_0_298631336).wrapping_add(z1).wrapping_add(z3);
        let t1 = t1.wrapping_mul(FIX_2_053119869).wrapping_add(z2).wrapping_add(z4);
        let t2 = t2.wrapping_mul(FIX_3_072711026).wrapping_add(z2).wrapping_add(z3);
        let t3 = t3.wrapping_mul(FIX_1_501321110).wrapping_add(z1).wrapping_add(z4);
        let d = |x: i32| x.wrapping_add(round) >> n;
        o[0][l] = d(tmp10.wrapping_add(t3));
        o[7][l] = d(tmp10.wrapping_sub(t3));
        o[1][l] = d(tmp11.wrapping_add(t2));
        o[6][l] = d(tmp11.wrapping_sub(t2));
        o[2][l] = d(tmp12.wrapping_add(t1));
        o[5][l] = d(tmp12.wrapping_sub(t1));
        o[3][l] = d(tmp13.wrapping_add(t0));
        o[4][l] = d(tmp13.wrapping_sub(t0));
    }
    o
}

/// `jpeg_idct_islow` (jidctint.c): CONST_BITS 13, PASS1_BITS 2, columns then rows, `range_limit` on the way
/// out. Returns false when the block leaves the range where libjpeg-turbo's C code and its SIMD kernels — the
/// ones Pillow's wheels run — provably agree. Measured on this machine with crafted one-block files: PIL
/// gives 255 where the C gives 0 for an output of 600 (the Neon kernel saturates, C's `& 1023` table wraps),
/// and differs again for dequantised values beyond 16 bits. The Neon kernel (`jidctint-neon.c`) dequantises
/// with 16-bit `vmul_s16`, shifts DC-only columns in 16 bits, adds pairs in 16 bits, narrows pass 1 to 16
/// bits (`vrshrn`) and saturates the output (`vqrshrn`); with dequantised values in [-8192, 8191], workspace
/// values in [-16384, 16383] (so a pair's sum or difference fits 16 bits) and outputs in [-512, 511], none of
/// those wraps or saturates, so every path computes the C value — and there C's range-limit table is a plain
/// clamp. Real data is far inside: |F(u,v)| ≤ 2048 for 8-bit samples, so a dequantised coefficient is at most
/// 2048 + q/2, and a coefficient is 0 once q > 4096.
fn idct_islow(coef: &[i16], q: &[i16; 64], out: &mut [u8], stride: usize) -> bool {
    // DEQUANTIZE: (ISLOW_MULT_TYPE)coef * quantval — two shorts promoted to int.
    let coef: &[i16; 64] = coef[..64].try_into().unwrap();
    let mut d = [[0i32; 8]; 8];
    let (mut lo, mut hi, mut ac) = (0i32, 0i32, 0i16);
    for k in 0..8 {
        for c in 0..8 {
            let v = coef[k * 8 + c] as i32 * q[k * 8 + c] as i32;
            (lo, hi) = (lo.min(v), hi.max(v));
            d[k][c] = v;
        }
    }
    if lo < -8192 || hi > 8191 { return false; }
    for &c in &coef[1..] { ac |= c; }
    if ac == 0 {
        // DC only: the workspace is 4·dc (one column), every sample range_limit[DESCALE(4·dc, 5)]; the same
        // guards apply as on the full path.
        let (w, x) = (4 * d[0][0], (4 * d[0][0] + 16) >> 5);
        if !(-16384..=16383).contains(&w) || !(-512..=511).contains(&x) { return false; }
        let v = (x + 128).clamp(0, 255) as u8;
        for row in 0..8 { out[row * stride..row * stride + 8].fill(v); }
        return true;
    }
    let ws = idct_pass(&d, CONST_BITS - PASS1_BITS); // ws[y][x]: C's workspace
    let mut t = [[0i32; 8]; 8];
    for y in 0..8 {
        for x in 0..8 {
            (lo, hi) = (lo.min(ws[y][x]), hi.max(ws[y][x]));
            t[x][y] = ws[y][x];
        }
    }
    if lo < -16384 || hi > 16383 { return false; }
    let p = idct_pass(&t, CONST_BITS + PASS1_BITS + 3); // p[x][y]
    let (mut lo, mut hi) = (0i32, 0i32);
    for y in 0..8 {
        let o = &mut out[y * stride..y * stride + 8];
        for x in 0..8 {
            let v = p[x][y];
            (lo, hi) = (lo.min(v), hi.max(v));
            o[x] = (v + 128).clamp(0, 255) as u8;
        }
    }
    lo >= -512 && hi <= 511
}

// ─────────────────────────────── upsampling (jdsample.c) ───────────────────────────────

enum Up {
    Full,
    H2V1Fancy,
    H2V1Box,
    H1V2Fancy,
    H2V2Fancy,
    H2V2Box,
    Int(usize, usize),
}

/// `jinit_upsampler`'s per-component choice (do_fancy is TRUE at full scale).
fn upsampler(f: &Frame, c: &Comp) -> Result<Up, String> {
    let (hi, vi, ho, vo) = (c.h, c.v, f.hmax, f.vmax);
    Ok(if hi == ho && vi == vo {
        Up::Full
    } else if hi * 2 == ho && vi == vo {
        if c.dw > 2 { Up::H2V1Fancy } else { Up::H2V1Box }
    } else if hi == ho && vi * 2 == vo {
        Up::H1V2Fancy
    } else if hi * 2 == ho && vi * 2 == vo {
        if c.dw > 2 { Up::H2V2Fancy } else { Up::H2V2Box }
    } else if ho % hi == 0 && vo % vi == 0 {
        Up::Int(ho / hi, vo / vi)
    } else {
        return Err("fractional sampling ratio (libjpeg-turbo: JERR_FRACT_SAMPLE_NOTIMPL)".into());
    })
}

/// Output row `y` of component `c`, at least `w` samples, into `out`. Rows outside [0, dh) are the edge row
/// (jdmainct.c's context pointers); columns past dw likewise (the first/last-column special cases).
fn upsample_row(u: &Up, c: &Comp, plane: &[u8], y: usize, out: &mut [u8], sums: &mut [i32]) {
    let stride = c.bw * 8;
    let row = |r: usize| &plane[r * stride..r * stride + c.dw];
    let dw = c.dw;
    match *u {
        Up::Full => out[..dw].copy_from_slice(row(y)),
        Up::H2V1Box => for (i, &v) in row(y).iter().enumerate() { out[2 * i] = v; out[2 * i + 1] = v; },
        Up::H2V2Box => for (i, &v) in row(y / 2).iter().enumerate() { out[2 * i] = v; out[2 * i + 1] = v; },
        Up::Int(hx, vy) => for (i, &v) in row(y / vy).iter().enumerate() { out[i * hx..(i + 1) * hx].fill(v); },
        Up::H2V1Fancy => {
            let r = row(y);
            // h2v1_fancy_upsample: 3/4 nearer + 1/4 further, biases 1 and 2.
            out[0] = r[0];
            out[1] = ((r[0] as u32 * 3 + r[1] as u32 + 2) >> 2) as u8;
            for i in 1..dw - 1 {
                let v = r[i] as u32 * 3;
                out[2 * i] = ((v + r[i - 1] as u32 + 1) >> 2) as u8;
                out[2 * i + 1] = ((v + r[i + 1] as u32 + 2) >> 2) as u8;
            }
            out[2 * dw - 2] = ((r[dw - 1] as u32 * 3 + r[dw - 2] as u32 + 1) >> 2) as u8;
            out[2 * dw - 1] = r[dw - 1];
        }
        Up::H1V2Fancy => {
            // h1v2_fancy_upsample: the upper output row of a pair leans on the row above (bias 1), the lower
            // on the row below (bias 2).
            let ri = y / 2;
            let (far, bias) = if y.is_multiple_of(2) { (ri.saturating_sub(1), 1) } else { ((ri + 1).min(c.dh - 1), 2) };
            let (a, b) = (row(ri), row(far));
            for i in 0..dw { out[i] = ((a[i] as u32 * 3 + b[i] as u32 + bias) >> 2) as u8; }
        }
        Up::H2V2Fancy => {
            let ri = y / 2;
            let far = if y.is_multiple_of(2) { ri.saturating_sub(1) } else { (ri + 1).min(c.dh - 1) };
            let (a, b) = (row(ri), row(far));
            for i in 0..dw { sums[i] = a[i] as i32 * 3 + b[i] as i32; }
            // h2v2_fancy_upsample: 9/16, 3/16, 3/16, 1/16 with biases 8 and 7.
            out[0] = ((sums[0] * 4 + 8) >> 4) as u8;
            out[1] = ((sums[0] * 3 + sums[1] + 7) >> 4) as u8;
            for i in 1..dw - 1 {
                out[2 * i] = ((sums[i] * 3 + sums[i - 1] + 8) >> 4) as u8;
                out[2 * i + 1] = ((sums[i] * 3 + sums[i + 1] + 7) >> 4) as u8;
            }
            out[2 * dw - 2] = ((sums[dw - 1] * 3 + sums[dw - 2] + 8) >> 4) as u8;
            out[2 * dw - 1] = ((sums[dw - 1] * 4 + 7) >> 4) as u8;
        }
    }
}

// ─────────────────────────────── colour (jdcolor.c) ───────────────────────────────

/// `JLONG` in the C; |Cb→G + Cr→G| < 1e7, so 32 bits hold them exactly.
struct YccTables {
    cr_r: [i32; 256],
    cb_b: [i32; 256],
    cr_g: [i32; 256],
    cb_g: [i32; 256],
}

impl YccTables {
    /// `build_ycc_rgb_table`: SCALEBITS 16, FIX(x) = (x * 65536 + 0.5) truncated, ONE_HALF folded into Cb→G.
    fn new() -> Self {
        const SCALEBITS: u32 = 16;
        const ONE_HALF: i64 = 1 << (SCALEBITS - 1);
        let fix = |x: f64| (x * 65536.0 + 0.5) as i64;
        let mut t = YccTables { cr_r: [0; 256], cb_b: [0; 256], cr_g: [0; 256], cb_g: [0; 256] };
        for i in 0..256 {
            let x = i as i64 - 128;
            t.cr_r[i] = ((fix(1.40200) * x + ONE_HALF) >> SCALEBITS) as i32;
            t.cb_b[i] = ((fix(1.77200) * x + ONE_HALF) >> SCALEBITS) as i32;
            t.cr_g[i] = (-fix(0.71414) * x) as i32;
            t.cb_g[i] = (-fix(0.34414) * x + ONE_HALF) as i32;
        }
        t
    }

    /// `ycc_rgb_convert`, with `sample_range_limit` as a clamp (its table spans [-256, 639]).
    fn convert(&self, y: &[u8], cb: &[u8], cr: &[u8], out: &mut [u8]) {
        let clamp = |v: i32| v.clamp(0, 255) as u8;
        for (((o, &y), &cb), &cr) in out.chunks_exact_mut(3).zip(y).zip(cb).zip(cr) {
            let (y, cb, cr) = (y as i32, cb as usize, cr as usize);
            o[0] = clamp(y + self.cr_r[cr]);
            o[1] = clamp(y + ((self.cb_g[cb] + self.cr_g[cr]) >> 16));
            o[2] = clamp(y + self.cb_b[cb]);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{decode, idct_islow, CONST_BITS, PASS1_BITS};

    // The literal translation of jpeg_idct_islow (64-bit JLONG, the C's shortcuts, the & 1023 range table),
    // kept to check the lane version against.
    const FIX_0_298631336: i64 = 2446;
    const FIX_0_390180644: i64 = 3196;
    const FIX_0_541196100: i64 = 4433;
    const FIX_0_765366865: i64 = 6270;
    const FIX_0_899976223: i64 = 7373;
    const FIX_1_175875602: i64 = 9633;
    const FIX_1_501321110: i64 = 12299;
    const FIX_1_847759065: i64 = 15137;
    const FIX_1_961570560: i64 = 16069;
    const FIX_2_053119869: i64 = 16819;
    const FIX_2_562915447: i64 = 20995;
    const FIX_3_072711026: i64 = 25172;

    #[inline(always)]
    fn descale(x: i64, n: u32) -> i64 { (x + (1i64 << (n - 1))) >> n }

    /// The post-IDCT half of `prepare_range_limit_table`, indexed by `x & 1023` (RANGE_MASK): 128+x for x in
    /// [-128, 127], 255 above, 0 below — for |x| ≤ 512; beyond that the mask wraps, as libjpeg's C does.
    fn idct_range_limit() -> [u8; 1024] {
        let mut t = [0u8; 1024];
        for (i, e) in t.iter_mut().enumerate() {
            *e = match i { 0..=127 => (i + 128) as u8, 128..=511 => 255, 512..=895 => 0, _ => (i - 896) as u8 };
        }
        t
    }

    /// jidctint.c line by line, with the same three range guards as `idct_islow`.
    fn idct_islow_c(coef: &[i16], q: &[i16; 64], out: &mut [u8], stride: usize, rl: &[u8; 1024]) -> bool {
        let mut ws = [0i32; 64];
        // DEQUANTIZE: (ISLOW_MULT_TYPE)coef * quantval — two shorts promoted to int.
        let dq = |i: usize| (coef[i] as i32 * q[i] as i32) as i64;
        if (0..64).any(|i| !(-8192..=8191).contains(&dq(i))) { return false; }
        for col in 0..8 {
            if (1..8).all(|r| coef[r * 8 + col] == 0) {
                // AC terms all zero: LEFT_SHIFT(DEQUANTIZE(dc), PASS1_BITS) into an int.
                let dcval = (dq(col) << PASS1_BITS) as i32;
                for r in 0..8 { ws[r * 8 + col] = dcval; }
                continue;
            }
            let (z2, z3) = (dq(16 + col), dq(48 + col));
            let z1 = (z2 + z3) * FIX_0_541196100;
            let tmp2 = z1 + z3 * -FIX_1_847759065;
            let tmp3 = z1 + z2 * FIX_0_765366865;
            let (z2, z3) = (dq(col), dq(32 + col));
            let tmp0 = (z2 + z3) << CONST_BITS;
            let tmp1 = (z2 - z3) << CONST_BITS;
            let (tmp10, tmp13, tmp11, tmp12) = (tmp0 + tmp3, tmp0 - tmp3, tmp1 + tmp2, tmp1 - tmp2);
            let (mut t0, mut t1, mut t2, mut t3) = (dq(56 + col), dq(40 + col), dq(24 + col), dq(8 + col));
            let (z1, z2, z3, z4) = (t0 + t3, t1 + t2, t0 + t2, t1 + t3);
            let z5 = (z3 + z4) * FIX_1_175875602;
            t0 *= FIX_0_298631336;
            t1 *= FIX_2_053119869;
            t2 *= FIX_3_072711026;
            t3 *= FIX_1_501321110;
            let (z1, z2) = (z1 * -FIX_0_899976223, z2 * -FIX_2_562915447);
            let (z3, z4) = (z3 * -FIX_1_961570560 + z5, z4 * -FIX_0_390180644 + z5);
            t0 += z1 + z3;
            t1 += z2 + z4;
            t2 += z2 + z3;
            t3 += z1 + z4;
            let n = CONST_BITS - PASS1_BITS;
            ws[col] = descale(tmp10 + t3, n) as i32;
            ws[56 + col] = descale(tmp10 - t3, n) as i32;
            ws[8 + col] = descale(tmp11 + t2, n) as i32;
            ws[48 + col] = descale(tmp11 - t2, n) as i32;
            ws[16 + col] = descale(tmp12 + t1, n) as i32;
            ws[40 + col] = descale(tmp12 - t1, n) as i32;
            ws[24 + col] = descale(tmp13 + t0, n) as i32;
            ws[32 + col] = descale(tmp13 - t0, n) as i32;
        }
        if ws.iter().any(|w| !(-16384..=16383).contains(w)) { return false; }
        let mut inside = true;
        let mut lim = |x: i64| { inside &= (-512..=511).contains(&x); rl[(x as i32 & 1023) as usize] };
        for row in 0..8 {
            let w = &ws[row * 8..row * 8 + 8];
            let o = &mut out[row * stride..row * stride + 8];
            if w[1..].iter().all(|&x| x == 0) {
                o.fill(lim(descale(w[0] as i64, PASS1_BITS + 3)));
                continue;
            }
            let (z2, z3) = (w[2] as i64, w[6] as i64);
            let z1 = (z2 + z3) * FIX_0_541196100;
            let tmp2 = z1 + z3 * -FIX_1_847759065;
            let tmp3 = z1 + z2 * FIX_0_765366865;
            let tmp0 = (w[0] as i64 + w[4] as i64) << CONST_BITS;
            let tmp1 = (w[0] as i64 - w[4] as i64) << CONST_BITS;
            let (tmp10, tmp13, tmp11, tmp12) = (tmp0 + tmp3, tmp0 - tmp3, tmp1 + tmp2, tmp1 - tmp2);
            let (mut t0, mut t1, mut t2, mut t3) = (w[7] as i64, w[5] as i64, w[3] as i64, w[1] as i64);
            let (z1, z2, z3, z4) = (t0 + t3, t1 + t2, t0 + t2, t1 + t3);
            let z5 = (z3 + z4) * FIX_1_175875602;
            t0 *= FIX_0_298631336;
            t1 *= FIX_2_053119869;
            t2 *= FIX_3_072711026;
            t3 *= FIX_1_501321110;
            let (z1, z2) = (z1 * -FIX_0_899976223, z2 * -FIX_2_562915447);
            let (z3, z4) = (z3 * -FIX_1_961570560 + z5, z4 * -FIX_0_390180644 + z5);
            t0 += z1 + z3;
            t1 += z2 + z4;
            t2 += z2 + z3;
            t3 += z1 + z4;
            let n = CONST_BITS + PASS1_BITS + 3;
            o[0] = lim(descale(tmp10 + t3, n));
            o[7] = lim(descale(tmp10 - t3, n));
            o[1] = lim(descale(tmp11 + t2, n));
            o[6] = lim(descale(tmp11 - t2, n));
            o[2] = lim(descale(tmp12 + t1, n));
            o[5] = lim(descale(tmp12 - t1, n));
            o[3] = lim(descale(tmp13 + t0, n));
            o[4] = lim(descale(tmp13 - t0, n));
        }
        inside
    }


    /// The lane IDCT equals the literal C translation — output AND accept/refuse — on 60 000 blocks (~15 600
    /// accepted, ~44 400 refused by both): random
    /// sparsity, 8- and 16-bit quantisers, magnitudes up to and past every guard.
    #[test]
    fn the_lane_idct_is_the_c_idct() {
        let mut s = 0x9E37_79B9_7F4A_7C15u64;
        let mut rnd = move || { s ^= s << 13; s ^= s >> 7; s ^= s << 17; s };
        let rl = idct_range_limit();
        let (mut accepted, mut refused) = (0, 0);
        for i in 0..60_000 {
            let (mut coef, mut q) = ([0i16; 64], [0i16; 64]);
            let density = rnd() % 65;
            // Target dequantised magnitude: realistic (≤ 2048), near the guards, and past them.
            let target = [64i64, 512, 2048, 2048, 4096, 8192, 12000][i % 7];
            for k in 0..64 {
                q[k] = if i % 11 == 0 { (rnd() % 65536) as u16 as i16 } else { (1 + rnd() % 255) as i16 };
                if i % 5 == 0 { q[k] = 1; } // magnitudes reach the guards through the coefficients alone
                let zz = super::NATURAL.iter().position(|&n| n == k).unwrap() as i64; // energy falls with frequency
                let m = (target / (1 + zz) / (q[k] as i64).abs().max(1)).max(1);
                if k == 0 || rnd() % 64 < density { coef[k] = (rnd() as i64 % (2 * m + 1) - m) as i16; }
            }
            if i % 13 == 0 { coef[1..].fill(0); } // DC-only blocks
            let (mut a, mut b) = ([0u8; 64], [0u8; 64]);
            let (ka, kb) = (idct_islow(&coef, &q, &mut a, 8), idct_islow_c(&coef, &q, &mut b, 8, &rl));
            assert_eq!(ka, kb, "accept/refuse differs: {coef:?} {q:?}");
            if ka { assert_eq!(a, b, "{coef:?} {q:?}"); accepted += 1; } else { refused += 1; }
        }
        eprintln!("{accepted} blocks equal, {refused} refused by both");
        assert!(accepted > 10_000 && refused > 3_000, "both outcomes must be exercised");
    }

    fn seg(m: u8, body: &[u8]) -> Vec<u8> {
        let mut v = vec![0xFF, m, ((body.len() + 2) >> 8) as u8, (body.len() + 2) as u8];
        v.extend_from_slice(body);
        v
    }

    /// One 8x8 grey block whose IDCT output is dc*q/8 everywhere (make_images.py's `one_block`), with an
    /// 8- or 16-bit quantiser for the DC.
    fn one_block(dc: i32, q: u16, q16: bool) -> Vec<u8> {
        let (mut bits, mut n, mut out) = (0u64, 0u32, Vec::new());
        let mut put = |v: u32, l: u32, out: &mut Vec<u8>| {
            bits = bits << l | (v & ((1 << l) - 1)) as u64;
            n += l;
            while n >= 8 { n -= 8; let b = (bits >> n) as u8; out.push(b); if b == 0xFF { out.push(0); } }
        };
        let s = 32 - dc.unsigned_abs().leading_zeros();
        put(s, 4, &mut out);
        if s > 0 { put(if dc > 0 { dc as u32 } else { (dc - 1) as u32 }, s, &mut out); }
        put(0, 1, &mut out);
        put(0x7F, 7, &mut out); // pad with 1s
        let mut f = vec![0xFF, 0xD8];
        let dqt: Vec<u8> = if q16 { [vec![0x10], q.to_be_bytes().to_vec(), [0, 1].repeat(63)].concat() } else { [vec![0, q as u8], vec![1; 63]].concat() };
        f.extend(seg(0xDB, &dqt));
        f.extend(seg(if q16 { 0xC1 } else { 0xC0 }, &[8, 0, 8, 0, 8, 1, 1, 0x11, 0]));
        let dht = [vec![0x00, 0, 0, 0, 15], vec![0; 12], (0..15).collect(), vec![0x10, 1], vec![0; 15], vec![0]].concat();
        f.extend(seg(0xC4, &dht));
        f.extend(seg(0xDA, &[1, 1, 0x00, 0, 63, 0]));
        f.extend(out);
        f.extend([0xFF, 0xD9]);
        f
    }

    /// Where libjpeg-turbo's C code (translated here) and the SIMD kernels Pillow runs disagree, there is no
    /// single reference, so those blocks are refused. Measured with PIL 12.2.0 on arm64 (Neon): an output of
    /// 600 gives 255 (C: 0), -600 gives 0 (C: 255), a dequantised 40000 gives 255 (C: 8), and a 16-bit
    /// quantiser of 40000 (C's `short` multiplier: -25536) gives 255 (C: 8). The edges that agree (511 → 255,
    /// -512 → 0) are fixtures with PIL's pixels.
    #[test]
    fn blocks_where_the_reference_depends_on_the_machine_are_refused() {
        assert!(decode(&one_block(2044, 2, false)).unwrap().px.iter().all(|&p| p == 255)); // output 511
        assert!(decode(&one_block(-2048, 2, false)).unwrap().px.iter().all(|&p| p == 0)); // output -512
        for (dc, q, q16, what) in [(2048, 2, false, "output 512"), (-2052, 2, false, "output -513"), (2400, 2, false, "output 600"),
                                   (40, 255, false, "dequantised 10200"), (2000, 20, true, "dequantised 40000"),
                                   (1, 40000, true, "16-bit quantiser 40000")] {
            let e = decode(&one_block(dc, q, q16)).err().unwrap_or_else(|| panic!("{what}: decoded"));
            assert!(e.contains("depend on the machine"), "{what}: {e}");
        }
    }

    fn fixture(name: &str) -> Vec<u8> {
        std::fs::read(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/images/").to_string() + name).unwrap()
    }

    /// Headers the reference decodes (or PIL refuses) but this does not reproduce: each named in the error.
    #[test]
    fn what_is_not_reproduced_is_refused_by_name() {
        let sof = |m: u8, prec: u8, nc: u8| {
            let mut body = vec![prec, 0, 8, 0, 8, nc];
            for i in 0..nc { body.extend([i + 1, 0x11, 0]); }
            [vec![0xFF, 0xD8], seg(m, &body), vec![0xFF, 0xD9]].concat()
        };
        for (bytes, want) in [(sof(0xC9, 8, 3), "arithmetic"), (sof(0xCA, 8, 3), "arithmetic"), (sof(0xC3, 8, 3), "lossless"),
                              (sof(0xC5, 8, 3), "hierarchical"), (sof(0xC0, 12, 3), "12-bit"), (sof(0xC0, 8, 4), "CMYK"),
                              (sof(0xC0, 8, 2), "2-component")] {
            let e = decode(&bytes).err().unwrap();
            assert!(e.contains(want), "want {want:?} in {e:?}");
        }
        // 13 400 x 13 400 = 179 560 000 pixels: past PIL's DecompressionBombError; 13 000 x 13 000 is not.
        let big = |w: u16, h: u16| [vec![0xFF, 0xD8], seg(0xC0, &[[8u8].as_slice(), &h.to_be_bytes(), &w.to_be_bytes(), &[1, 1, 0x11, 0]].concat())].concat();
        assert!(decode(&big(13_400, 13_400)).err().unwrap().contains("decompression-bomb"));
        assert!(!decode(&big(13_000, 13_000)).err().unwrap().contains("decompression-bomb"));
        assert!(decode(&sof(0xC0, 8, 1)).err().unwrap().contains("no image data")); // a header, then EOI
        // A progressive file cut after its first (DC) scan: libjpeg-turbo would BLOCK-SMOOTH the missing AC.
        let p = fixture("progressive.jpg");
        let sos: Vec<usize> = (0..p.len() - 1).filter(|&i| p[i] == 0xFF && p[i + 1] == 0xDA).collect();
        assert!(sos.len() > 2);
        let cut = [&p[..sos[1]], &[0xFF, 0xD9]].concat();
        assert!(decode(&cut).err().unwrap().contains("block smoothing"));
    }

    /// Decode time vs zune-jpeg 0.5 (the decoder this replaced), and — if `<file>.rgb` (PIL's pixels, raw RGB)
    /// sits beside the file — exactness at that size. Timing only; run in release:
    /// `FERRIC_JPEG_BENCH=big.jpg cargo test --release -p ferric-serve jpeg_decode_time -- --ignored --nocapture`
    #[test]
    #[ignore = "timing: needs FERRIC_JPEG_BENCH=<file.jpg> and --release"]
    fn jpeg_decode_time() {
        use std::time::Instant;
        let path = std::env::var("FERRIC_JPEG_BENCH").expect("FERRIC_JPEG_BENCH=<file.jpg>");
        let b = std::fs::read(&path).unwrap();
        let ours = decode(&b).unwrap();
        if let Ok(want) = std::fs::read(format!("{path}.rgb")) {
            assert_eq!(ours.px.len(), want.len());
            let bad = ours.px.iter().zip(&want).filter(|(a, b)| a != b).count();
            eprintln!("{path}: {}x{}, {bad} of {} samples differ from PIL", ours.w, ours.h, want.len());
            assert_eq!(bad, 0);
        }
        let zune = || {
            use zune_core::{bytestream::ZCursor, colorspace::ColorSpace, options::DecoderOptions};
            let o = DecoderOptions::default().jpeg_set_out_colorspace(ColorSpace::RGB).set_max_width(1 << 16).set_max_height(1 << 16);
            zune_jpeg::JpegDecoder::new_with_options(ZCursor::new(&b[..]), o).decode().unwrap()
        };
        let time = |f: &dyn Fn()| { let mut t: Vec<f64> = (0..9).map(|_| { let s = Instant::now(); f(); s.elapsed().as_secs_f64() * 1e3 }).collect(); t.sort_by(f64::total_cmp); t };
        let (a, z) = (time(&|| { decode(&b).unwrap(); }), time(&|| { zune(); }));
        eprintln!("{path}: {:.1} MP  this {:.0} ms (min) / {:.0} (median) / {:.0} (max)   zune-jpeg {:.0} / {:.0} / {:.0}",
                  (ours.w * ours.h) as f64 / 1e6, a[0], a[4], a[8], z[0], z[4], z[8]);
    }

    /// Damage libjpeg-turbo would only warn about (and patch with zeros or a resync) is refused.
    #[test]
    fn damaged_data_is_refused() {
        let b = fixture("baseline420.jpg");
        assert!(decode(&b[..b.len() - 300]).err().unwrap().contains("truncated"));
        let mut cut = b[..b.len() - 300].to_vec();
        cut.extend([0xFF, 0xD9]); // EOI where data should be: a scan cut short
        assert!(decode(&cut).is_err());
        let mut r = fixture("enc_restart3.jpg");
        let i = (0..r.len() - 1).find(|&i| r[i] == 0xFF && r[i + 1] == 0xD0).unwrap();
        r[i + 1] = 0xD1; // RST1 where RST0 belongs
        assert!(decode(&r).err().unwrap().contains("expected RST0"));
        assert!(decode(&b[..2]).is_err());
        assert!(decode(b"\xFF\xD8\xFF\xD9").err().unwrap().contains("no image data"));
    }
}
