//! DEFLATE decompression, enough of it to read a compressed MAT-file.
//!
//! ## Why this is written out rather than depended on
//!
//! The wind-turbine corpus compresses every element with zlib, and without inflate a third of the
//! `.mat` corpora stay closed. This crate carries two path dependencies and `pollster`, and wrote
//! its own SHA-256 for the same reason: a sensor tokenizer that has to be reachable from a browser
//! and a sensor node is easier to keep that way if the dependency list stays short enough to read.
//! Inflate is a bounded, forty-year-old format with a normative specification (RFC 1950/1951) and,
//! more usefully here, a corpus of real streams to check against.
//!
//! Decompression only. Nothing here writes a stream.
//!
//! ## What is verified
//!
//! Every stream in the tests is one this decoder did not produce: fixtures compressed by another
//! implementation, and the corpus's own 43 files. Round-tripping against a matching compressor
//! would prove the two agree, not that either is right.

/// Why a stream could not be decompressed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InflateError {
    /// The two-byte zlib header failed its own check, or asked for something not in the format.
    BadZlibHeader { cmf: u8, flg: u8 },
    /// A preset dictionary, which the format allows and MAT-files never use.
    PresetDictionary,
    /// Ran out of input mid-stream.
    OutOfInput { at: usize },
    /// A block type the format does not define.
    BadBlockType { code: u32 },
    /// A stored block whose length and its complement disagree, which is the format's own
    /// corruption check.
    StoredLengthMismatch { len: u16, nlen: u16 },
    /// A Huffman code that decodes to nothing, i.e. an incomplete or over-subscribed table.
    BadCode,
    /// A back-reference pointing before the start of the output.
    DistanceTooFar { distance: usize, have: usize },
    /// The Adler-32 trailer did not match what was decompressed. **This is the check that makes a
    /// silent partial decode impossible**, and it is verified rather than skipped.
    ChecksumMismatch { want: u32, got: u32 },
    /// The stream ended without a final block.
    Truncated,
}

impl std::fmt::Display for InflateError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::BadZlibHeader { cmf, flg } => write!(f, "bad zlib header {cmf:#04x} {flg:#04x}"),
            Self::PresetDictionary => write!(f, "stream needs a preset dictionary"),
            Self::OutOfInput { at } => write!(f, "input ended at byte {at}"),
            Self::BadBlockType { code } => write!(f, "undefined block type {code}"),
            Self::StoredLengthMismatch { len, nlen } => {
                write!(f, "stored block length {len} does not complement {nlen}")
            }
            Self::BadCode => write!(f, "invalid Huffman code"),
            Self::DistanceTooFar { distance, have } => {
                write!(f, "back-reference {distance} into {have} bytes of output")
            }
            Self::ChecksumMismatch { want, got } => {
                write!(f, "adler32 {got:#010x}, expected {want:#010x}")
            }
            Self::Truncated => write!(f, "stream ended with no final block"),
        }
    }
}

impl std::error::Error for InflateError {}

struct Bits<'a> {
    data: &'a [u8],
    pos: usize,
    acc: u32,
    have: u32,
}

impl<'a> Bits<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0, acc: 0, have: 0 }
    }
    /// DEFLATE packs bits least-significant first within each byte.
    fn take(&mut self, n: u32) -> Result<u32, InflateError> {
        while self.have < n {
            if self.pos >= self.data.len() {
                return Err(InflateError::OutOfInput { at: self.pos });
            }
            self.acc |= (self.data[self.pos] as u32) << self.have;
            self.pos += 1;
            self.have += 8;
        }
        let v = self.acc & ((1u32 << n) - 1);
        self.acc >>= n;
        self.have -= n;
        Ok(v)
    }
    fn align(&mut self) {
        let drop = self.have % 8;
        self.acc >>= drop;
        self.have -= drop;
    }
}

/// A canonical Huffman table, stored as counts-per-length plus symbols in code order — the
/// representation from the reference decoder, chosen because it needs no allocation per symbol and
/// decodes with a running comparison rather than a lookup that has to be sized in advance.
struct Huffman {
    counts: [u16; 16],
    symbols: Vec<u16>,
}

impl Huffman {
    fn new(lengths: &[u8]) -> Self {
        let mut counts = [0u16; 16];
        for &l in lengths {
            counts[l as usize] += 1;
        }
        counts[0] = 0;
        let mut offs = [0u16; 16];
        for i in 1..15 {
            offs[i + 1] = offs[i] + counts[i];
        }
        let mut symbols = vec![0u16; lengths.len()];
        for (sym, &l) in lengths.iter().enumerate() {
            if l != 0 {
                symbols[offs[l as usize] as usize] = sym as u16;
                offs[l as usize] += 1;
            }
        }
        Self { counts, symbols }
    }

    fn decode(&self, bits: &mut Bits) -> Result<u16, InflateError> {
        let (mut code, mut first, mut index) = (0i32, 0i32, 0i32);
        for len in 1..=15 {
            code |= bits.take(1)? as i32;
            let count = self.counts[len] as i32;
            if code - first < count {
                return Ok(self.symbols[(index + (code - first)) as usize]);
            }
            index += count;
            first = (first + count) << 1;
            code <<= 1;
        }
        Err(InflateError::BadCode)
    }
}

const LEN_BASE: [u16; 29] = [
    3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31, 35, 43, 51, 59, 67, 83, 99, 115, 131,
    163, 195, 227, 258,
];
const LEN_EXTRA: [u8; 29] = [
    0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0,
];
const DIST_BASE: [u16; 30] = [
    1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193, 257, 385, 513, 769, 1025, 1537,
    2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577,
];
const DIST_EXTRA: [u8; 30] = [
    0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13,
    13,
];
/// The order code lengths for the code-length alphabet are themselves written in.
const CLEN_ORDER: [usize; 19] =
    [16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15];

fn fixed_tables() -> (Huffman, Huffman) {
    let mut lit = [0u8; 288];
    for (i, l) in lit.iter_mut().enumerate() {
        *l = match i {
            0..=143 => 8,
            144..=255 => 9,
            256..=279 => 7,
            _ => 8,
        };
    }
    (Huffman::new(&lit), Huffman::new(&[5u8; 30]))
}

fn dynamic_tables(bits: &mut Bits) -> Result<(Huffman, Huffman), InflateError> {
    let hlit = bits.take(5)? as usize + 257;
    let hdist = bits.take(5)? as usize + 1;
    let hclen = bits.take(4)? as usize + 4;
    let mut clen = [0u8; 19];
    for &slot in CLEN_ORDER.iter().take(hclen) {
        clen[slot] = bits.take(3)? as u8;
    }
    let cl = Huffman::new(&clen);
    let mut lengths = vec![0u8; hlit + hdist];
    let mut i = 0;
    while i < lengths.len() {
        let sym = cl.decode(bits)?;
        match sym {
            0..=15 => {
                lengths[i] = sym as u8;
                i += 1;
            }
            16 => {
                // Repeat the PREVIOUS length; at position 0 there is none, and a stream that asks
                // is malformed rather than an implicit zero.
                if i == 0 {
                    return Err(InflateError::BadCode);
                }
                let prev = lengths[i - 1];
                let n = 3 + bits.take(2)? as usize;
                for _ in 0..n {
                    if i >= lengths.len() {
                        return Err(InflateError::BadCode);
                    }
                    lengths[i] = prev;
                    i += 1;
                }
            }
            17 | 18 => {
                let n = if sym == 17 { 3 + bits.take(3)? as usize } else { 11 + bits.take(7)? as usize };
                for _ in 0..n {
                    if i >= lengths.len() {
                        return Err(InflateError::BadCode);
                    }
                    lengths[i] = 0;
                    i += 1;
                }
            }
            _ => return Err(InflateError::BadCode),
        }
    }
    Ok((Huffman::new(&lengths[..hlit]), Huffman::new(&lengths[hlit..])))
}

/// Decompress a raw DEFLATE stream, with no zlib wrapper and no checksum.
pub fn inflate_raw(data: &[u8]) -> Result<Vec<u8>, InflateError> {
    let mut bits = Bits::new(data);
    let mut out: Vec<u8> = Vec::new();
    loop {
        let last = bits.take(1)?;
        let btype = bits.take(2)?;
        match btype {
            0 => {
                bits.align();
                let len = bits.take(16)? as u16;
                let nlen = bits.take(16)? as u16;
                if len != !nlen {
                    return Err(InflateError::StoredLengthMismatch { len, nlen });
                }
                for _ in 0..len {
                    out.push(bits.take(8)? as u8);
                }
            }
            1 | 2 => {
                let (lit, dist) = if btype == 1 { fixed_tables() } else { dynamic_tables(&mut bits)? };
                loop {
                    let sym = lit.decode(&mut bits)?;
                    match sym {
                        0..=255 => out.push(sym as u8),
                        256 => break,
                        257..=285 => {
                            let i = sym as usize - 257;
                            let len = LEN_BASE[i] as usize + bits.take(LEN_EXTRA[i] as u32)? as usize;
                            let dsym = dist.decode(&mut bits)? as usize;
                            if dsym >= DIST_BASE.len() {
                                return Err(InflateError::BadCode);
                            }
                            let d = DIST_BASE[dsym] as usize
                                + bits.take(DIST_EXTRA[dsym] as u32)? as usize;
                            if d > out.len() {
                                return Err(InflateError::DistanceTooFar {
                                    distance: d,
                                    have: out.len(),
                                });
                            }
                            // Copied one byte at a time on purpose: the format permits the
                            // reference to overlap what is being written, which is how it encodes
                            // a run, and a block copy would read the pre-copy bytes.
                            let start = out.len() - d;
                            for k in 0..len {
                                let b = out[start + k];
                                out.push(b);
                            }
                        }
                        _ => return Err(InflateError::BadCode),
                    }
                }
            }
            code => return Err(InflateError::BadBlockType { code }),
        }
        if last == 1 {
            return Ok(out);
        }
        if bits.pos >= data.len() && bits.have == 0 {
            return Err(InflateError::Truncated);
        }
    }
}

/// Adler-32, the checksum a zlib stream carries.
pub fn adler32(data: &[u8]) -> u32 {
    let (mut a, mut b) = (1u32, 0u32);
    // 5552 is the most bytes that can be summed before the 32-bit accumulator can overflow.
    for chunk in data.chunks(5552) {
        for &x in chunk {
            a += x as u32;
            b += a;
        }
        a %= 65521;
        b %= 65521;
    }
    (b << 16) | a
}

/// Decompress a zlib stream: two-byte header, DEFLATE data, Adler-32 trailer.
///
/// The trailer is CHECKED. A decoder that ignores it will happily return a truncated or subtly
/// wrong buffer, and a sensor channel that is wrong in its second half still tokenizes.
pub fn inflate_zlib(data: &[u8]) -> Result<Vec<u8>, InflateError> {
    if data.len() < 6 {
        return Err(InflateError::OutOfInput { at: data.len() });
    }
    let (cmf, flg) = (data[0], data[1]);
    // Compression method 8 is DEFLATE, and the two header bytes read as a big-endian 16-bit
    // number must be a multiple of 31.
    if cmf & 0x0f != 8 || ((cmf as u16) << 8 | flg as u16) % 31 != 0 {
        return Err(InflateError::BadZlibHeader { cmf, flg });
    }
    if flg & 0x20 != 0 {
        return Err(InflateError::PresetDictionary);
    }
    let out = inflate_raw(&data[2..])?;
    let n = data.len();
    let want = u32::from_be_bytes([data[n - 4], data[n - 3], data[n - 2], data[n - 1]]);
    let got = adler32(&out);
    if want != got {
        return Err(InflateError::ChecksumMismatch { want, got });
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- streams produced by ANOTHER implementation ----
    //
    // Every fixture here was compressed elsewhere and pasted in. Round-tripping against a matching
    // compressor written in this file would show the pair agrees with itself, which is not the
    // question. Where the expected OUTPUT is long it is regenerated from the rule that produced
    // it rather than pasted, so the comparison is against an independent description of the data
    // and not against a buffer this decoder once emitted.

    /// "hello hello hello\n" at level 1: fixed Huffman, one back-reference.
    const HELLO_FIXED: &[u8] = &[
        0x78, 0x01, 0xcb, 0x48, 0xcd, 0xc9, 0xc9, 0x57, 0xc8, 0x40, 0x90, 0x5c, 0x00, 0x40, 0xb5,
        0x06, 0x87,
    ];

    /// Ten `a`s: a literal followed by a length-9 distance-1 reference, which OVERLAPS the bytes
    /// it is writing.
    const RUN_OVERLAP: &[u8] = &[0x78, 0x9c, 0x4b, 0x4c, 0x84, 0x01, 0x00, 0x14, 0xe1, 0x03, 0xcb];

    /// 536 bytes at level 9, which forces a DYNAMIC Huffman table — the branch neither of the
    /// other fixtures reaches, and the one with the code-length alphabet and its repeat codes.
    const DYNAMIC: &[u8] = &[
        0x78, 0xda, 0xcd, 0xd0, 0x45, 0x56, 0x82, 0x01, 0x00, 0x45, 0xe1, 0x4d, 0xfc, 0x62, 0x77,
        0x62, 0x27, 0xb6, 0xd8, 0xdd, 0x98, 0x80, 0x01, 0xd8, 0xdd, 0x8a, 0x8a, 0x81, 0xdd, 0xad,
        0xa8, 0xd8, 0xdd, 0x8d, 0x8d, 0xdd, 0xdd, 0xb5, 0x19, 0xdf, 0x0e, 0x9c, 0x7a, 0xce, 0x1d,
        0x7f, 0x83, 0x4b, 0x70, 0x49, 0x45, 0x52, 0x3c, 0x99, 0x32, 0x79, 0xbe, 0x62, 0xa5, 0x72,
        0xb5, 0x5a, 0x9d, 0x46, 0x83, 0x56, 0x93, 0x76, 0xab, 0x5e, 0xbb, 0x41, 0xa7, 0x51, 0xb7,
        0x69, 0x2f, 0xa5, 0xdf, 0x62, 0xc0, 0x6a, 0xd8, 0x76, 0x94, 0x3a, 0xee, 0x30, 0xe9, 0x3c,
        0xe3, 0x3a, 0xe7, 0xbe, 0xe0, 0xb9, 0xec, 0xb3, 0xea, 0xb7, 0x1e, 0x20, 0x0a, 0xda, 0x0e,
        0xd9, 0x0d, 0xdb, 0xa7, 0x1f, 0x46, 0x1c, 0x47, 0x9d, 0xb2, 0xce, 0x63, 0xae, 0xe2, 0x6e,
        0x12, 0xee, 0x92, 0x1e, 0x53, 0x9f, 0xd3, 0x5f, 0x33, 0xdf, 0x73, 0xbe, 0xf2, 0x7e, 0x0a,
        0x24, 0x0a, 0xc1, 0x96, 0xca, 0x95, 0x2b, 0x80, 0x55, 0xad, 0x55, 0xaf, 0x07, 0xdb, 0xa2,
        0xdb, 0xa6, 0x0f, 0xd6, 0x44, 0x60, 0xd6, 0x07, 0x76, 0xc8, 0x66, 0xc4, 0x0e, 0xac, 0xd3,
        0xb4, 0xcb, 0x2c, 0xd8, 0x25, 0xef, 0x15, 0x5f, 0xb0, 0xb4, 0xad, 0xe0, 0x1d, 0xb0, 0x07,
        0xcc, 0xa3, 0x48, 0xb0, 0x9c, 0xcb, 0xd8, 0x6b, 0xb0, 0x0f, 0x29, 0x4f, 0x69, 0x60, 0xb3,
        0x3f, 0x73, 0xbf, 0xc1, 0x4a, 0x16, 0x4b, 0x97, 0x80, 0xad, 0x50, 0xaa, 0x52, 0x01, 0xab,
        0xd9, 0x48, 0x6e, 0x06, 0xdb, 0x61, 0xd8, 0x65, 0x0c, 0xd6, 0x5c, 0x68, 0x39, 0x08, 0x76,
        0xcc, 0x7e, 0xc2, 0x11, 0xac, 0xdb, 0xbc, 0xc7, 0x22, 0xd8, 0x35, 0xff, 0x8d, 0x40, 0xb0,
        0xa1, 0x7b, 0xe1, 0x62, 0xb0, 0x27, 0xd1, 0x67, 0x6c, 0xb0, 0xf1, 0xb7, 0x89, 0xf7, 0x60,
        0x5f, 0x32, 0xde, 0xb2, 0xc0, 0xe6, 0x13, 0x5c, 0x12, 0x58, 0x59, 0x6c, 0x00, 0x5b, 0x83,
        0x0d, 0x60, 0x75, 0xb0, 0x01, 0x6c, 0x0f, 0x36, 0x80, 0xb5, 0xc6, 0x06, 0xb0, 0x53, 0xd8,
        0x00, 0xd6, 0x0b, 0x1b, 0xc0, 0x6e, 0x62, 0x03, 0x58, 0x06, 0x36, 0x80, 0xbd, 0xc0, 0x06,
        0xb0, 0xc9, 0xd8, 0x00, 0xf6, 0x03, 0x1b, 0x88, 0x7f, 0x79, 0x97, 0xc5, 0xe6, 0xfc, 0xd9,
        0x2f, 0xc8, 0xf9, 0xf3, 0x4f,
    ];

    /// The rule DYNAMIC's input was generated from, so the expected bytes are a description rather
    /// than a recording.
    fn dynamic_expected() -> Vec<u8> {
        let mut v: Vec<u8> =
            (0..500usize).map(|i| (((i * 101 + i / 7) % 200) + 20) as u8).collect();
        for _ in 0..3 {
            v.extend_from_slice(b"abcabcabcabc");
        }
        v
    }

    #[test]
    fn a_fixed_huffman_stream_from_another_implementation_decompresses() {
        assert_eq!(inflate_zlib(HELLO_FIXED).unwrap(), b"hello hello hello\n");
    }

    /// The dynamic-table branch, which the other fixtures never enter: the code-length alphabet is
    /// itself Huffman-coded and carries repeat codes 16, 17 and 18. A decoder that mishandles them
    /// builds a wrong table and emits plausible bytes, so this asserts the full 536.
    #[test]
    fn a_dynamic_huffman_stream_decompresses_to_the_rule_that_generated_it() {
        let want = dynamic_expected();
        let got = inflate_zlib(DYNAMIC).unwrap();
        assert_eq!(got.len(), want.len(), "length");
        assert_eq!(got, want);
    }

    /// Back-references may overlap the bytes they are writing — that is how the format encodes a
    /// run — so the copy must go one byte at a time. A block copy reads the pre-copy bytes and
    /// produces a plausible, wrong result.
    #[test]
    fn an_overlapping_back_reference_produces_a_run() {
        assert_eq!(inflate_zlib(RUN_OVERLAP).unwrap(), b"aaaaaaaaaa");
    }

    #[test]
    fn adler32_matches_its_published_definition() {
        assert_eq!(adler32(b""), 1);
        assert_eq!(adler32(b"Wikipedia"), 0x11E6_0398);
        // And the trailer of a real stream is the checksum of what it decompresses to.
        assert_eq!(adler32(&dynamic_expected()), 0xc8f9_f34f);
    }

    /// THE CHECK THAT MAKES A PARTIAL DECODE IMPOSSIBLE. Without it a corrupted recording arrives
    /// looking like a recording, and a sensor channel that is wrong in its second half tokenizes
    /// perfectly well.
    #[test]
    fn a_corrupted_trailer_is_refused() {
        let mut bad = HELLO_FIXED.to_vec();
        let n = bad.len();
        bad[n - 1] ^= 0xff;
        match inflate_zlib(&bad) {
            Err(InflateError::ChecksumMismatch { .. }) => {}
            other => panic!("expected ChecksumMismatch, got {other:?}"),
        }
    }

    #[test]
    fn a_bad_header_is_refused_before_anything_is_decoded() {
        assert!(matches!(
            inflate_zlib(&[0x00, 0x00, 0, 0, 0, 0]),
            Err(InflateError::BadZlibHeader { .. })
        ));
        // Compression method 9 with an otherwise valid check value.
        assert!(matches!(
            inflate_zlib(&[0x79, 0x00, 0, 0, 0, 0]),
            Err(InflateError::BadZlibHeader { .. })
        ));
    }

    /// Every truncation point of a valid stream must error rather than return a short buffer.
    #[test]
    fn every_truncation_point_is_refused() {
        for cut in 0..DYNAMIC.len() {
            let r = inflate_zlib(&DYNAMIC[..cut]);
            assert!(r.is_err(), "truncating to {cut} of {} bytes returned Ok", DYNAMIC.len());
        }
        assert!(inflate_zlib(DYNAMIC).is_ok());
    }

    /// A stored (uncompressed) block, assembled by hand: BFINAL=1, BTYPE=00, LEN, ~LEN, bytes.
    #[test]
    fn a_stored_block_is_copied_verbatim_and_its_own_check_fires() {
        let payload = b"not compressible";
        let mut raw = vec![0x01u8];
        raw.extend((payload.len() as u16).to_le_bytes());
        raw.extend((!(payload.len() as u16)).to_le_bytes());
        raw.extend(payload);
        assert_eq!(inflate_raw(&raw).unwrap(), payload);
        let mut bad = raw.clone();
        bad[3] ^= 0xff;
        assert!(matches!(inflate_raw(&bad), Err(InflateError::StoredLengthMismatch { .. })));
    }

    /// A fixed-Huffman block whose first symbol is a LENGTH code, so it references output that
    /// does not exist yet. Hand-assembled bit by bit: BFINAL=1, BTYPE=01, symbol 257, distance
    /// code 0, end-of-block. Without the bounds check this indexes before the start of the buffer.
    #[test]
    fn a_reference_before_the_start_of_output_is_refused() {
        const BACKREF_BEFORE_START: &[u8] = &[0x03, 0x02, 0x00];
        match inflate_raw(BACKREF_BEFORE_START) {
            Err(InflateError::DistanceTooFar { distance, have }) => {
                assert_eq!((distance, have), (1, 0));
            }
            other => panic!("expected DistanceTooFar, got {other:?}"),
        }
    }

    #[test]
    fn an_undefined_block_type_is_refused() {
        // BFINAL=1, BTYPE=11, which the format reserves and never defines.
        assert!(matches!(
            inflate_raw(&[0x07u8, 0x00, 0x00, 0x00]),
            Err(InflateError::BadBlockType { code: 3 })
        ));
    }
}
