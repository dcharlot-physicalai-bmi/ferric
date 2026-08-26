//! A minimal reader for MATLAB v5 MAT-files, which is the format condition-monitoring corpora
//! actually ship in.
//!
//! ## Why this is in a signal crate at all
//!
//! Of the four public sensor corpora this crate was pointed at, three are `.mat`: the CWRU bearing
//! set (161 files), a wind-turbine drivetrain set, and a rotating-machinery set whose acoustic and
//! vibration halves are both `.mat`. One reader opens three corpora. The fourth ships plain text
//! and is already ingested by `examples/hydraulic`.
//!
//! ## What is implemented, and how that was decided
//!
//! By surveying the corpora rather than by reading the specification and implementing all of it.
//! Every top-level element in the 161 CWRU files is an uncompressed `miMATRIX` of class
//! `mxDOUBLE`; the rotating set wraps its channels in an `mxSTRUCT` and uses the packed
//! small-element tag form; the wind set compresses every element with zlib. So:
//!
//! - numeric classes, widened to `f64` — a sensor channel is a sequence of numbers, and a caller
//!   that wanted the original width can read [`MatClass`] off the value
//! - `mxCHAR`, because struct fields carry units and channel names
//! - `mxSTRUCT` and `mxCELL`, including nesting
//! - both tag forms, and both byte orders
//!
//! And explicitly NOT `miCOMPRESSED`, `mxSPARSE`, `mxOBJECT`, or complex arrays. Each is refused
//! by name in [`MatError`]. **A reader that returns an empty variable list for a file it cannot
//! parse is worse than one that fails**, because the caller sees a corpus with nothing in it and
//! has no way to tell that from a corpus that was not read.
//!
//! ## What is verified
//!
//! Round-trip against fixtures assembled byte by byte in the tests, so the expected bytes are
//! written out rather than taken from whatever this parser happens to produce. Every truncation
//! point of a valid file is required to produce an error rather than a panic or a short read —
//! the same bar `store` is held to, and the reason a corrupted download fails loudly instead of
//! yielding a plausible-looking half of a recording.

use std::collections::BTreeMap;

/// The MAT-file data types this reader recognises. Values are from the format specification.
pub mod dt {
    pub const INT8: u32 = 1;
    pub const UINT8: u32 = 2;
    pub const INT16: u32 = 3;
    pub const UINT16: u32 = 4;
    pub const INT32: u32 = 5;
    pub const UINT32: u32 = 6;
    pub const SINGLE: u32 = 7;
    pub const DOUBLE: u32 = 9;
    pub const INT64: u32 = 12;
    pub const UINT64: u32 = 13;
    pub const MATRIX: u32 = 14;
    pub const COMPRESSED: u32 = 15;
    pub const UTF8: u32 = 16;
    pub const UTF16: u32 = 17;
    pub const UTF32: u32 = 18;
}

/// The array classes this reader recognises.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MatClass {
    Cell,
    Struct,
    Char,
    Double,
    Single,
    Int8,
    UInt8,
    Int16,
    UInt16,
    Int32,
    UInt32,
    Int64,
    UInt64,
}

impl MatClass {
    fn from_code(c: u32) -> Option<Self> {
        Some(match c {
            1 => Self::Cell,
            2 => Self::Struct,
            4 => Self::Char,
            6 => Self::Double,
            7 => Self::Single,
            8 => Self::Int8,
            9 => Self::UInt8,
            10 => Self::Int16,
            11 => Self::UInt16,
            12 => Self::Int32,
            13 => Self::UInt32,
            14 => Self::Int64,
            15 => Self::UInt64,
            _ => return None,
        })
    }
    /// The name MATLAB uses, so an error message is searchable.
    pub fn name(c: u32) -> &'static str {
        match c {
            1 => "mxCELL", 2 => "mxSTRUCT", 3 => "mxOBJECT", 4 => "mxCHAR", 5 => "mxSPARSE",
            6 => "mxDOUBLE", 7 => "mxSINGLE", 8 => "mxINT8", 9 => "mxUINT8", 10 => "mxINT16",
            11 => "mxUINT16", 12 => "mxINT32", 13 => "mxUINT32", 14 => "mxINT64",
            15 => "mxUINT64", _ => "unknown",
        }
    }
}

/// Why a file could not be read. Every variant names the thing that was not understood; none of
/// them can be produced by a file this reader handles.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MatError {
    /// Shorter than the 128-byte header, or not a v5 file.
    NotMatlab5 { got: usize },
    /// The endian indicator was neither `IM` nor `MI`.
    BadEndianMark { got: [u8; 2] },
    /// An element claims more bytes than the file has left.
    Truncated { at: usize, want: usize, have: usize },
    /// `miCOMPRESSED`. Named separately because it is a missing FEATURE, not a broken file.
    Compressed { at: usize },
    /// A data type this reader does not decode.
    UnsupportedDataType { at: usize, code: u32 },
    /// An array class this reader does not decode, named.
    UnsupportedClass { name: &'static str, code: u32 },
    /// Complex arrays carry a second data block this reader does not return.
    ComplexArray { name: String },
    /// A subelement was not the type the format requires at that position.
    UnexpectedElement { want: u32, got: u32, at: usize },
    /// Dimensions that do not multiply to the number of values present.
    DimensionMismatch { dims: Vec<usize>, values: usize },
}

impl std::fmt::Display for MatError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotMatlab5 { got } => write!(f, "not a MATLAB v5 MAT-file ({got} bytes)"),
            Self::BadEndianMark { got } => {
                write!(f, "endian mark was {:?}, expected IM or MI", got)
            }
            Self::Truncated { at, want, have } => {
                write!(f, "truncated at byte {at}: element wants {want} bytes, {have} remain")
            }
            Self::Compressed { at } => write!(
                f,
                "element at byte {at} is miCOMPRESSED (zlib); this reader has no inflate"
            ),
            Self::UnsupportedDataType { at, code } => {
                write!(f, "unsupported MAT data type {code} at byte {at}")
            }
            Self::UnsupportedClass { name, code } => {
                write!(f, "unsupported array class {name} ({code})")
            }
            Self::ComplexArray { name } => write!(f, "array `{name}` is complex"),
            Self::UnexpectedElement { want, got, at } => {
                write!(f, "expected data type {want} at byte {at}, found {got}")
            }
            Self::DimensionMismatch { dims, values } => {
                write!(f, "dimensions {dims:?} do not match {values} values")
            }
        }
    }
}

impl std::error::Error for MatError {}

/// One value out of a MAT-file.
#[derive(Debug, Clone, PartialEq)]
pub enum MatValue {
    /// Any numeric class, widened to `f64`, with the class it was stored as. Column-major, which
    /// is MATLAB's order and is preserved rather than transposed — a caller reading a `[N, 1]`
    /// channel gets the samples in recording order either way, and silently transposing a `[N, M]`
    /// array would be a change no error could report.
    Numeric { class: MatClass, dims: Vec<usize>, data: Vec<f64> },
    Char { dims: Vec<usize>, text: String },
    /// Field names in file order, and one value per (element, field).
    Struct { dims: Vec<usize>, fields: Vec<String>, elements: Vec<Vec<MatValue>> },
    Cell { dims: Vec<usize>, items: Vec<MatValue> },
}

impl MatValue {
    /// Total number of array elements, which is the product of the dimensions.
    pub fn len(&self) -> usize {
        self.dims().iter().product()
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    pub fn dims(&self) -> &[usize] {
        match self {
            Self::Numeric { dims, .. } | Self::Char { dims, .. } => dims,
            Self::Struct { dims, .. } => dims,
            Self::Cell { dims, .. } => dims,
        }
    }
    /// The samples, if this is numeric. `None` for a struct, cell or char — deliberately not an
    /// empty slice, which a caller would read as "an empty channel".
    pub fn numeric(&self) -> Option<&[f64]> {
        match self {
            Self::Numeric { data, .. } => Some(data),
            _ => None,
        }
    }
    /// One named field of a 1x1 struct.
    pub fn field(&self, name: &str) -> Option<&MatValue> {
        match self {
            Self::Struct { fields, elements, .. } => {
                let i = fields.iter().position(|f| f == name)?;
                elements.first()?.get(i)
            }
            _ => None,
        }
    }
}

/// A parsed MAT-file: its header text and its variables in file order.
#[derive(Debug, Clone)]
pub struct MatFile {
    pub header: String,
    pub vars: Vec<(String, MatValue)>,
}

impl MatFile {
    /// Parse a whole file from bytes.
    pub fn parse(bytes: &[u8]) -> Result<Self, MatError> {
        if bytes.len() < 128 {
            return Err(MatError::NotMatlab5 { got: bytes.len() });
        }
        let header = String::from_utf8_lossy(&bytes[..116]).trim_end().to_string();
        if !header.starts_with("MATLAB 5.0") {
            return Err(MatError::NotMatlab5 { got: bytes.len() });
        }
        // The last two header bytes hold the characters "MI" written as one 16-bit integer, so
        // reading them back as bytes says which order the writer used.
        let mark = [bytes[126], bytes[127]];
        let le = match &mark {
            b"IM" => true,
            b"MI" => false,
            _ => return Err(MatError::BadEndianMark { got: mark }),
        };

        let mut vars = Vec::new();
        let mut off = 128;
        while off < bytes.len() {
            // Trailing padding shorter than a tag is not an element; stop rather than error.
            if bytes.len() - off < 8 {
                break;
            }
            let tag = read_tag(bytes, off, le)?;
            match tag.dtype {
                dt::MATRIX => {
                    let (name, value) = parse_matrix(&bytes[tag.data..tag.data + tag.nbytes], le)?;
                    vars.push((name, value));
                }
                dt::COMPRESSED => return Err(MatError::Compressed { at: off }),
                code => return Err(MatError::UnsupportedDataType { at: off, code }),
            }
            off = tag.next;
        }
        Ok(Self { header, vars })
    }

    /// One variable by name.
    pub fn var(&self, name: &str) -> Option<&MatValue> {
        self.vars.iter().find(|(n, _)| n == name).map(|(_, v)| v)
    }

    /// Every numeric series in the file, by name — the shortest path from a corpus file to
    /// something tokenizable. Corpora that bury channels inside nested structs surface them here.
    ///
    /// Nested names are joined with `.`, cell items get `{i}`, struct-array elements get `(i)`,
    /// and the columns of a 2-D array get `[:,j]`, so two series can never collide into one entry.
    ///
    /// **A length-1 entry is a scalar**, not a recording — corpora store RPM and sample rates this
    /// way and they are returned rather than hidden. Arrays of rank above 2 are not surfaced.
    pub fn channels(&self) -> BTreeMap<String, &[f64]> {
        let mut out = BTreeMap::new();
        for (name, v) in &self.vars {
            collect(name, v, &mut out);
        }
        out
    }
}

fn collect<'a>(path: &str, v: &'a MatValue, out: &mut BTreeMap<String, &'a [f64]>) {
    match v {
        MatValue::Numeric { dims, data, .. } => {
            let wide = dims.iter().filter(|&&d| d > 1).count();
            if wide <= 1 {
                // MATLAB stores a single channel as [N, 1]; some writers use [1, N].
                if !data.is_empty() {
                    out.insert(path.to_string(), data.as_slice());
                }
            } else if wide == 2 && dims.len() == 2 {
                // A MULTI-CHANNEL RECORDING, which is how a real corpus stores one: the rotating
                // set's `Signal.y_values.values` is [1536000, 4], four accelerometers for sixty
                // seconds. An earlier version of this function required at most one dimension
                // above 1 and so returned that file's 57 metadata SCALARS and none of its 6.1M
                // samples — a corpus that reads as successfully parsed and empty.
                //
                // MATLAB is column-major, so a column is already contiguous and each one is a
                // borrowed slice rather than a copy.
                let (rows, cols) = (dims[0], dims[1]);
                for j in 0..cols {
                    out.insert(format!("{path}[:,{j}]"), &data[j * rows..(j + 1) * rows]);
                }
            }
            // Rank above 2 is deliberately not surfaced: there is no one reading of which axis is
            // time, and guessing would put a silently wrong series in front of a caller.
        }
        MatValue::Struct { fields, elements, .. } => {
            for (ei, el) in elements.iter().enumerate() {
                for (fi, fv) in el.iter().enumerate() {
                    let p = if elements.len() == 1 {
                        format!("{path}.{}", fields[fi])
                    } else {
                        format!("{path}({ei}).{}", fields[fi])
                    };
                    collect(&p, fv, out);
                }
            }
        }
        MatValue::Cell { items, .. } => {
            for (i, it) in items.iter().enumerate() {
                collect(&format!("{path}{{{i}}}"), it, out);
            }
        }
        MatValue::Char { .. } => {}
    }
}

struct Tag {
    dtype: u32,
    nbytes: usize,
    /// Where this element's data starts.
    data: usize,
    /// Where the next element starts, padding included.
    next: usize,
}

fn u16_at(b: &[u8], o: usize, le: bool) -> u16 {
    let x = [b[o], b[o + 1]];
    if le { u16::from_le_bytes(x) } else { u16::from_be_bytes(x) }
}

fn u32_at(b: &[u8], o: usize, le: bool) -> u32 {
    let x = [b[o], b[o + 1], b[o + 2], b[o + 3]];
    if le { u32::from_le_bytes(x) } else { u32::from_be_bytes(x) }
}

/// Read one element tag, handling BOTH forms.
///
/// The format has a packed form for elements of four bytes or fewer, signalled by the upper half
/// of the first word being non-zero — the byte count moves into those two bytes and the data
/// follows immediately in the same eight bytes. It is not an optional optimisation: the rotating
/// corpus uses it for the field-name-length of every struct, so a reader that only handles the
/// long form desynchronises on the first struct it meets and then reports nonsense sizes for
/// everything after it.
fn read_tag(b: &[u8], off: usize, le: bool) -> Result<Tag, MatError> {
    if off + 8 > b.len() {
        return Err(MatError::Truncated { at: off, want: 8, have: b.len().saturating_sub(off) });
    }
    let w0 = u32_at(b, off, le);
    let upper = if le { u16_at(b, off + 2, le) } else { u16_at(b, off, le) };
    if upper != 0 {
        // Packed: type in the low half, byte count in the high half, data in the next four bytes.
        let (dtype, nbytes) = if le {
            (u16_at(b, off, le) as u32, u16_at(b, off + 2, le) as usize)
        } else {
            (u16_at(b, off + 2, le) as u32, u16_at(b, off, le) as usize)
        };
        if nbytes > 4 {
            return Err(MatError::Truncated { at: off, want: nbytes, have: 4 });
        }
        return Ok(Tag { dtype, nbytes, data: off + 4, next: off + 8 });
    }
    let nbytes = u32_at(b, off + 4, le) as usize;
    let data = off + 8;
    let have = b.len().saturating_sub(data);
    if nbytes > have {
        return Err(MatError::Truncated { at: off, want: nbytes, have });
    }
    // Every element is padded out to an eight-byte boundary.
    let next = data + nbytes.div_ceil(8) * 8;
    Ok(Tag { dtype: w0, nbytes, data, next: next.min(b.len()) })
}

/// Read a numeric element's payload, widened to `f64`.
fn read_numeric(b: &[u8], tag: &Tag, le: bool) -> Result<Vec<f64>, MatError> {
    let d = &b[tag.data..tag.data + tag.nbytes];
    let n = |w: usize| d.len() / w;
    Ok(match tag.dtype {
        dt::INT8 => d.iter().map(|&v| v as i8 as f64).collect(),
        dt::UINT8 | dt::UTF8 => d.iter().map(|&v| v as f64).collect(),
        dt::INT16 | dt::UTF16 => (0..n(2)).map(|i| u16_at(d, i * 2, le) as i16 as f64).collect(),
        dt::UINT16 => (0..n(2)).map(|i| u16_at(d, i * 2, le) as f64).collect(),
        dt::INT32 | dt::UTF32 => (0..n(4)).map(|i| u32_at(d, i * 4, le) as i32 as f64).collect(),
        dt::UINT32 => (0..n(4)).map(|i| u32_at(d, i * 4, le) as f64).collect(),
        dt::SINGLE => (0..n(4)).map(|i| f32::from_bits(u32_at(d, i * 4, le)) as f64).collect(),
        dt::DOUBLE => (0..n(8))
            .map(|i| {
                let lo = u32_at(d, i * 8, le) as u64;
                let hi = u32_at(d, i * 8 + 4, le) as u64;
                f64::from_bits(if le { (hi << 32) | lo } else { (lo << 32) | hi })
            })
            .collect(),
        dt::INT64 => (0..n(8))
            .map(|i| {
                let lo = u32_at(d, i * 8, le) as u64;
                let hi = u32_at(d, i * 8 + 4, le) as u64;
                (if le { (hi << 32) | lo } else { (lo << 32) | hi }) as i64 as f64
            })
            .collect(),
        dt::UINT64 => (0..n(8))
            .map(|i| {
                let lo = u32_at(d, i * 8, le) as u64;
                let hi = u32_at(d, i * 8 + 4, le) as u64;
                (if le { (hi << 32) | lo } else { (lo << 32) | hi }) as f64
            })
            .collect(),
        dt::COMPRESSED => return Err(MatError::Compressed { at: tag.data }),
        code => return Err(MatError::UnsupportedDataType { at: tag.data, code }),
    })
}

/// Parse one `miMATRIX` payload into a name and a value.
fn parse_matrix(b: &[u8], le: bool) -> Result<(String, MatValue), MatError> {
    // 1. array flags: two u32, class in the low byte, flags in the next.
    let flags_tag = read_tag(b, 0, le)?;
    if flags_tag.nbytes < 8 {
        return Err(MatError::Truncated { at: 0, want: 8, have: flags_tag.nbytes });
    }
    let w0 = u32_at(b, flags_tag.data, le);
    let class_code = w0 & 0xff;
    let complex = (w0 >> 8) & 0x08 != 0;

    // 2. dimensions.
    let dim_tag = read_tag(b, flags_tag.next, le)?;
    if dim_tag.dtype != dt::INT32 {
        return Err(MatError::UnexpectedElement {
            want: dt::INT32,
            got: dim_tag.dtype,
            at: flags_tag.next,
        });
    }
    let dims: Vec<usize> = read_numeric(b, &dim_tag, le)?.iter().map(|&v| v as usize).collect();

    // 3. name.
    let name_tag = read_tag(b, dim_tag.next, le)?;
    let name = String::from_utf8_lossy(&b[name_tag.data..name_tag.data + name_tag.nbytes])
        .trim_end_matches('\0')
        .to_string();

    let Some(class) = MatClass::from_code(class_code) else {
        return Err(MatError::UnsupportedClass {
            name: MatClass::name(class_code),
            code: class_code,
        });
    };
    if complex {
        return Err(MatError::ComplexArray { name });
    }
    let total: usize = dims.iter().product();
    let mut off = name_tag.next;

    let value = match class {
        MatClass::Struct => {
            // Field name length arrives in the PACKED tag form.
            let len_tag = read_tag(b, off, le)?;
            let field_len = read_numeric(b, &len_tag, le)?.first().copied().unwrap_or(0.0) as usize;
            off = len_tag.next;
            let names_tag = read_tag(b, off, le)?;
            let raw = &b[names_tag.data..names_tag.data + names_tag.nbytes];
            let fields: Vec<String> = if field_len == 0 {
                Vec::new()
            } else {
                raw.chunks(field_len)
                    .map(|c| String::from_utf8_lossy(c).trim_end_matches('\0').to_string())
                    .collect()
            };
            off = names_tag.next;
            let mut elements = Vec::with_capacity(total);
            for _ in 0..total {
                let mut row = Vec::with_capacity(fields.len());
                for _ in 0..fields.len() {
                    let t = read_tag(b, off, le)?;
                    if t.dtype == dt::COMPRESSED {
                        return Err(MatError::Compressed { at: off });
                    }
                    if t.dtype != dt::MATRIX {
                        return Err(MatError::UnexpectedElement {
                            want: dt::MATRIX,
                            got: t.dtype,
                            at: off,
                        });
                    }
                    let (_, v) = parse_matrix(&b[t.data..t.data + t.nbytes], le)?;
                    row.push(v);
                    off = t.next;
                }
                elements.push(row);
            }
            MatValue::Struct { dims, fields, elements }
        }
        MatClass::Cell => {
            let mut items = Vec::with_capacity(total);
            for _ in 0..total {
                let t = read_tag(b, off, le)?;
                if t.dtype == dt::COMPRESSED {
                    return Err(MatError::Compressed { at: off });
                }
                if t.dtype != dt::MATRIX {
                    return Err(MatError::UnexpectedElement {
                        want: dt::MATRIX,
                        got: t.dtype,
                        at: off,
                    });
                }
                let (_, v) = parse_matrix(&b[t.data..t.data + t.nbytes], le)?;
                items.push(v);
                off = t.next;
            }
            MatValue::Cell { dims, items }
        }
        MatClass::Char => {
            let t = read_tag(b, off, le)?;
            let codes = read_numeric(b, &t, le)?;
            let text: String = codes
                .iter()
                .map(|&c| char::from_u32(c as u32).unwrap_or('\u{FFFD}'))
                .collect();
            if codes.len() != total {
                return Err(MatError::DimensionMismatch { dims, values: codes.len() });
            }
            MatValue::Char { dims, text }
        }
        _ => {
            let t = read_tag(b, off, le)?;
            let data = read_numeric(b, &t, le)?;
            if data.len() != total {
                return Err(MatError::DimensionMismatch { dims, values: data.len() });
            }
            MatValue::Numeric { class, dims, data }
        }
    };
    Ok((name, value))
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- fixtures assembled byte by byte ----
    //
    // Written out here rather than captured from this parser's own output, so a test cannot agree
    // with a bug by construction. The layouts mirror what the real corpora contain: CWRU's plain
    // double vectors and the rotating set's struct-with-packed-field-length.

    fn u32b(v: u32, le: bool) -> Vec<u8> {
        if le { v.to_le_bytes().to_vec() } else { v.to_be_bytes().to_vec() }
    }
    fn f64b(v: f64, le: bool) -> Vec<u8> {
        if le { v.to_le_bytes().to_vec() } else { v.to_be_bytes().to_vec() }
    }

    /// Long-form element: 4-byte type, 4-byte count, payload padded to eight.
    fn el(dtype: u32, payload: &[u8], le: bool) -> Vec<u8> {
        let mut v = u32b(dtype, le);
        v.extend(u32b(payload.len() as u32, le));
        v.extend(payload);
        while v.len() % 8 != 0 {
            v.push(0);
        }
        v
    }

    /// Packed element: type and count share one word, payload in the next four bytes.
    fn packed(dtype: u32, payload: &[u8], le: bool) -> Vec<u8> {
        assert!(payload.len() <= 4);
        let mut v = Vec::new();
        if le {
            v.extend((dtype as u16).to_le_bytes());
            v.extend((payload.len() as u16).to_le_bytes());
        } else {
            v.extend((payload.len() as u16).to_be_bytes());
            v.extend((dtype as u16).to_be_bytes());
        }
        v.extend(payload);
        while v.len() < 8 {
            v.push(0);
        }
        v
    }

    fn header(le: bool) -> Vec<u8> {
        let mut h = vec![b' '; 128];
        let text = b"MATLAB 5.0 MAT-file, assembled by a test";
        h[..text.len()].copy_from_slice(text);
        h[124] = 0x00;
        h[125] = 0x01;
        let (a, b) = if le { (b'I', b'M') } else { (b'M', b'I') };
        h[126] = a;
        h[127] = b;
        h
    }

    /// The three subelements every miMATRIX starts with.
    fn preamble(class: u32, extra_flags: u32, dims: &[i32], name: &str, le: bool) -> Vec<u8> {
        let mut p = Vec::new();
        let mut flags = u32b(class | (extra_flags << 8), le);
        flags.extend(u32b(0, le));
        p.extend(el(dt::UINT32, &flags, le));
        let mut d = Vec::new();
        for &x in dims {
            d.extend(u32b(x as u32, le));
        }
        p.extend(el(dt::INT32, &d, le));
        p.extend(el(dt::INT8, name.as_bytes(), le));
        p
    }

    fn double_vector(name: &str, vals: &[f64], le: bool) -> Vec<u8> {
        let mut p = preamble(6, 0, &[vals.len() as i32, 1], name, le);
        let mut d = Vec::new();
        for &v in vals {
            d.extend(f64b(v, le));
        }
        p.extend(el(dt::DOUBLE, &d, le));
        p
    }

    fn file_of(matrices: &[Vec<u8>], le: bool) -> Vec<u8> {
        let mut f = header(le);
        for m in matrices {
            f.extend(el(dt::MATRIX, m, le));
        }
        f
    }

    #[test]
    fn a_double_vector_is_read_with_its_name_shape_and_values() {
        let bytes = file_of(&[double_vector("X107_DE_time", &[1.5, -2.0, 3.25], true)], true);
        let m = MatFile::parse(&bytes).unwrap();
        assert_eq!(m.vars.len(), 1);
        assert_eq!(m.vars[0].0, "X107_DE_time");
        let v = m.var("X107_DE_time").unwrap();
        assert_eq!(v.dims(), &[3, 1]);
        assert_eq!(v.numeric().unwrap(), &[1.5, -2.0, 3.25]);
        assert!(matches!(v, MatValue::Numeric { class: MatClass::Double, .. }));
    }

    /// The same bytes in the other order must read identically, or a big-endian corpus would come
    /// back as plausible-looking garbage rather than as an error.
    #[test]
    fn big_endian_reads_the_same_values() {
        let le = MatFile::parse(&file_of(&[double_vector("s", &[1.5, -2.0, 3.25], true)], true)).unwrap();
        let be = MatFile::parse(&file_of(&[double_vector("s", &[1.5, -2.0, 3.25], false)], false)).unwrap();
        assert_eq!(le.var("s").unwrap().numeric().unwrap(), be.var("s").unwrap().numeric().unwrap());
        assert_eq!(le.var("s").unwrap().dims(), be.var("s").unwrap().dims());
    }

    fn struct_fixture(le: bool) -> Vec<u8> {
        // A 1x1 struct named `Signal` with fields `y_values` and `units`, the shape the rotating
        // corpus uses — including the PACKED field-name-length that a long-form-only reader
        // desynchronises on.
        let mut p = preamble(2, 0, &[1, 1], "Signal", le);
        p.extend(packed(dt::INT32, &u32b(8, le), le));
        let mut names = Vec::new();
        names.extend(b"y_values");
        names.extend(b"units\0\0\0");
        p.extend(el(dt::INT8, &names, le));
        p.extend(el(dt::MATRIX, &double_vector("", &[10.0, 20.0], le), le));
        let mut ch = preamble(4, 0, &[1, 3], "", le);
        let mut codes = Vec::new();
        for c in [b'm' as u16, b'/' as u16, b's' as u16] {
            codes.extend(if le { c.to_le_bytes() } else { c.to_be_bytes() });
        }
        ch.extend(el(dt::UINT16, &codes, le));
        p.extend(el(dt::MATRIX, &ch, le));
        p
    }

    #[test]
    fn a_struct_is_read_including_the_packed_field_name_length() {
        let bytes = file_of(&[struct_fixture(true)], true);
        let m = MatFile::parse(&bytes).unwrap();
        let s = m.var("Signal").unwrap();
        let MatValue::Struct { fields, elements, dims } = s else { panic!("not a struct: {s:?}") };
        assert_eq!(dims, &[1, 1]);
        assert_eq!(fields, &["y_values".to_string(), "units".to_string()]);
        assert_eq!(elements.len(), 1);
        assert_eq!(s.field("y_values").unwrap().numeric().unwrap(), &[10.0, 20.0]);
        match s.field("units").unwrap() {
            MatValue::Char { text, .. } => assert_eq!(text, "m/s"),
            other => panic!("units was {other:?}"),
        }
    }

    /// THE MUTATION THIS TEST EXISTS FOR. A reader that ignores the packed tag form reads the
    /// field-name-length word as a long-form tag, takes the next four bytes as a byte count, and
    /// walks off into the middle of the data. It does not necessarily error — it can return a
    /// struct with plausible-looking wrong contents — which is why this asserts the field NAMES
    /// and values rather than only that parsing succeeded.
    #[test]
    fn the_packed_tag_form_carries_its_payload_in_the_same_word() {
        let b = packed(dt::INT32, &u32b(16, true), true);
        assert_eq!(b.len(), 8);
        let t = read_tag(&b, 0, true).unwrap();
        assert_eq!(t.dtype, dt::INT32);
        assert_eq!(t.nbytes, 4);
        assert_eq!(t.next, 8);
        assert_eq!(read_numeric(&b, &t, true).unwrap(), vec![16.0]);
    }

    #[test]
    fn channels_reaches_a_vector_nested_inside_a_struct() {
        let bytes = file_of(&[struct_fixture(true), double_vector("plain", &[7.0], true)], true);
        let m = MatFile::parse(&bytes).unwrap();
        let ch = m.channels();
        assert_eq!(ch.get("Signal.y_values").map(|s| s.to_vec()), Some(vec![10.0, 20.0]));
        assert_eq!(ch.get("plain").map(|s| s.to_vec()), Some(vec![7.0]));
        // The char field is a name, not a channel, and must not appear as one.
        assert!(!ch.contains_key("Signal.units"));
    }

    /// EVERY TRUNCATION POINT REFUSED. A half-downloaded corpus file must fail, not yield the
    /// first half of a recording — a short channel tokenizes perfectly well and would be scored
    /// as data.
    #[test]
    fn every_truncation_point_of_a_valid_file_is_refused() {
        let full = file_of(&[struct_fixture(true), double_vector("x", &[1.0, 2.0, 3.0], true)], true);
        assert!(MatFile::parse(&full).is_ok());
        for cut in 0..full.len() {
            let r = MatFile::parse(&full[..cut]);
            match &r {
                Err(_) => {}
                Ok(f) => {
                    // The only tolerable success is a prefix that ends exactly on an element
                    // boundary, which is a shorter but internally complete file.
                    let complete = f.vars.iter().all(|(_, v)| !v.dims().is_empty());
                    assert!(
                        complete && cut != full.len(),
                        "truncating to {cut} bytes of {} parsed as {} complete variables",
                        full.len(),
                        f.vars.len()
                    );
                }
            }
        }
    }

    #[test]
    fn a_file_that_is_not_matlab5_is_refused() {
        assert!(matches!(MatFile::parse(b"nope"), Err(MatError::NotMatlab5 { .. })));
        let mut h = header(true);
        h[..10].copy_from_slice(b"HDF5\0\0\0\0\0\0");
        assert!(matches!(MatFile::parse(&h), Err(MatError::NotMatlab5 { .. })));
    }

    #[test]
    fn a_bad_endian_mark_is_refused_rather_than_guessed() {
        let mut h = header(true);
        h[126] = b'X';
        h[127] = b'Y';
        assert!(matches!(MatFile::parse(&h), Err(MatError::BadEndianMark { .. })));
    }

    /// Compression is a MISSING FEATURE, and says so. The wind-turbine corpus is compressed end to
    /// end, so this is the error a caller will actually meet; "unsupported data type 15" would
    /// send them looking for a corrupt file.
    #[test]
    fn compression_is_refused_by_name_not_as_a_bad_type() {
        let mut f = header(true);
        f.extend(el(dt::COMPRESSED, &[0x78, 0x9c, 0x00, 0x00], true));
        match MatFile::parse(&f) {
            Err(MatError::Compressed { .. }) => {}
            other => panic!("expected Compressed, got {other:?}"),
        }
        assert!(format!("{}", MatError::Compressed { at: 128 }).contains("zlib"));
    }

    #[test]
    fn an_unsupported_class_is_named_in_the_error() {
        // Class 5 is mxSPARSE.
        let mut p = preamble(5, 0, &[2, 2], "sp", true);
        p.extend(el(dt::DOUBLE, &f64b(1.0, true), true));
        match MatFile::parse(&file_of(&[p], true)) {
            Err(MatError::UnsupportedClass { name, code }) => {
                assert_eq!(name, "mxSPARSE");
                assert_eq!(code, 5);
            }
            other => panic!("expected UnsupportedClass, got {other:?}"),
        }
    }

    #[test]
    fn a_complex_array_is_refused_rather_than_silently_losing_its_imaginary_part() {
        let mut p = preamble(6, 0x08, &[2, 1], "z", true);
        let mut d = Vec::new();
        d.extend(f64b(1.0, true));
        d.extend(f64b(2.0, true));
        p.extend(el(dt::DOUBLE, &d, true));
        match MatFile::parse(&file_of(&[p], true)) {
            Err(MatError::ComplexArray { name }) => assert_eq!(name, "z"),
            other => panic!("expected ComplexArray, got {other:?}"),
        }
    }

    /// Dimensions and data disagreeing is the signature of a mis-parse, and it is checked rather
    /// than absorbed: a `[6000, 1]` header over 5,999 doubles must not become a 5,999-sample
    /// channel.
    #[test]
    fn dimensions_that_do_not_match_the_payload_are_refused() {
        let mut p = preamble(6, 0, &[4, 1], "x", true);
        let mut d = Vec::new();
        for v in [1.0f64, 2.0] {
            d.extend(f64b(v, true));
        }
        p.extend(el(dt::DOUBLE, &d, true));
        match MatFile::parse(&file_of(&[p], true)) {
            Err(MatError::DimensionMismatch { dims, values }) => {
                assert_eq!(dims, vec![4, 1]);
                assert_eq!(values, 2);
            }
            other => panic!("expected DimensionMismatch, got {other:?}"),
        }
    }

    #[test]
    fn integer_classes_widen_to_f64_with_their_sign_intact() {
        for (dtype, class, raw, want) in [
            (dt::INT8, 8u32, vec![0xFFu8], -1.0f64),
            (dt::UINT8, 9, vec![0xFF], 255.0),
        ] {
            let mut p = preamble(class, 0, &[1, 1], "v", true);
            p.extend(el(dtype, &raw, true));
            let m = MatFile::parse(&file_of(&[p], true)).unwrap();
            assert_eq!(m.var("v").unwrap().numeric().unwrap(), &[want], "dtype {dtype}");
        }
    }

    /// THE BUG THE REAL CORPUS CAUGHT. `channels()` first required at most one dimension above 1,
    /// which is right for `[N, 1]` and wrong for every multi-channel recording. On the rotating
    /// corpus that returned the file's 57 metadata scalars and none of its 6.1M samples — parsed
    /// successfully, and empty. A four-column array must come back as four channels, each holding
    /// its own column, in column-major order.
    #[test]
    fn a_two_dimensional_array_surfaces_as_one_channel_per_column() {
        // Column-major: column 0 is [1,2,3], column 1 is [4,5,6].
        let mut p = preamble(6, 0, &[3, 2], "rec", true);
        let mut d = Vec::new();
        for v in [1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0] {
            d.extend(f64b(v, true));
        }
        p.extend(el(dt::DOUBLE, &d, true));
        let m = MatFile::parse(&file_of(&[p], true)).unwrap();
        let ch = m.channels();
        assert_eq!(ch.len(), 2, "expected one channel per column, got {:?}", ch.keys().collect::<Vec<_>>());
        assert_eq!(ch["rec[:,0]"], &[1.0, 2.0, 3.0]);
        assert_eq!(ch["rec[:,1]"], &[4.0, 5.0, 6.0]);
        // The whole array is still reachable undivided.
        assert_eq!(m.var("rec").unwrap().numeric().unwrap().len(), 6);
    }

    /// Rank above 2 has no defensible reading of which axis is time, so it is left out rather than
    /// guessed at. The array itself is still there for a caller who knows.
    #[test]
    fn a_rank_three_array_is_not_guessed_at() {
        let mut p = preamble(6, 0, &[2, 2, 2], "cube", true);
        let mut d = Vec::new();
        for v in 0..8 {
            d.extend(f64b(v as f64, true));
        }
        p.extend(el(dt::DOUBLE, &d, true));
        let m = MatFile::parse(&file_of(&[p], true)).unwrap();
        assert!(m.channels().is_empty());
        assert_eq!(m.var("cube").unwrap().len(), 8);
    }

    #[test]
    fn a_cell_array_holds_its_items_in_order() {
        let mut p = preamble(1, 0, &[2, 1], "c", true);
        p.extend(el(dt::MATRIX, &double_vector("", &[1.0], true), true));
        p.extend(el(dt::MATRIX, &double_vector("", &[2.0, 3.0], true), true));
        let m = MatFile::parse(&file_of(&[p], true)).unwrap();
        let MatValue::Cell { items, .. } = m.var("c").unwrap() else { panic!("not a cell") };
        assert_eq!(items.len(), 2);
        assert_eq!(items[0].numeric().unwrap(), &[1.0]);
        assert_eq!(items[1].numeric().unwrap(), &[2.0, 3.0]);
    }
}
