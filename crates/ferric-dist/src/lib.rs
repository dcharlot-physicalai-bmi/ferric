//! # ferric-dist — pipeline-parallel inference across machines
//!
//! Split a model's layers across N machines and stream activations down the chain. Machine 0 owns layers
//! `0..a`, machine 1 owns `a..b`, and so on; the last one produces logits.
//!
//! ## What this is honestly for
//!
//! **Capacity, not speed.** The measured record from the engines this was extracted from is sober:
//!
//! - ds4's headline "1.66×" is *pipeline* parallelism on **prefill**, and its baseline column compares a
//!   Q2 single-process run against a Q4 distributed one — which favours the baseline, as its own README
//!   concedes.
//! - colibri measures distributed **decode at 19.4% slower** than single-process.
//! - "120 t/s on 8×L40S" is aggregate over 16 concurrent sessions, i.e. ~7.5 t/s per stream.
//!
//! The reason is structural and worth stating up front: **prefill pipelines, decode does not.** Prefill
//! is one pass over many tokens, so chunks can be in flight at every stage at once. Decode is strictly
//! autoregressive — token *t+1* cannot start until token *t* has traversed the whole chain — so every
//! decode step pays the full round trip. Distributing buys you the ability to *run* a model that does not
//! fit on one machine. Expect prefill to speed up and decode to cost something.
//!
//! ## What this module actually provides
//!
//! The correctness layer, which is the part that is hard to get right and easy to get wrong silently:
//!
//! - a framed wire protocol that refuses malformed and truncated input rather than interpreting it;
//! - **prefix-hash guards** so a worker whose KV state does not match the request refuses it instead of
//!   answering confidently for the wrong position ([`Session`]);
//! - route construction that requires an **exact, gapless, non-overlapping** layer chain ([`route`]);
//! - a recovery model that distinguishes "wrong state, replay" from "connection died, re-route".
//!
//! Transport is a trait the caller implements, so all of it — including failure and recovery — is
//! testable in-process with no sockets and no second machine.

#![forbid(unsafe_code)]

pub mod route;
pub mod session;

pub use route::{plan_route, Registration, Route};
pub use session::{PrefixHash, Session, Verdict};

/// `"FDS1"` — Ferric distributed, v1.
pub const MAGIC: u32 = 0x4644_5331;

/// Frame kinds. Values are wire-stable; append only.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum Kind {
    Hello = 1,
    Error = 2,
    Work = 3,
    Result = 4,
    /// Non-final prefill chunk: the tail worker replies with a zero-length ack rather than shipping the
    /// hidden state back. Without this the coordinator pulls a full activation batch home per chunk —
    /// hundreds of megabytes it is going to discard.
    Ack = 5,
}

impl Kind {
    fn from_u32(v: u32) -> Option<Kind> {
        Some(match v {
            1 => Kind::Hello,
            2 => Kind::Error,
            3 => Kind::Work,
            4 => Kind::Result,
            5 => Kind::Ack,
            _ => return None,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DistError {
    /// Not our protocol, or a corrupted header.
    BadMagic(u32),
    UnknownKind(u32),
    /// The frame claims a length the buffer cannot satisfy. Kept distinct from `BadMagic` because it is
    /// the normal condition on a partially-arrived stream, not evidence of corruption.
    Truncated { want: usize, got: usize },
    /// A frame larger than the configured ceiling. Refusing beats allocating on a peer's say-so.
    TooLarge { want: usize, max: usize },
    /// The worker's KV state does not match what this request assumes.
    PrefixMismatch { expected: u64, got: u64 },
    /// No gapless chain of workers covers every layer exactly once.
    NoRoute,
    Transport(String),
}

impl core::fmt::Display for DistError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            DistError::BadMagic(m) => write!(f, "bad magic {m:#010x}"),
            DistError::UnknownKind(k) => write!(f, "unknown frame kind {k}"),
            DistError::Truncated { want, got } => write!(f, "truncated frame: want {want} bytes, got {got}"),
            DistError::TooLarge { want, max } => write!(f, "frame of {want} bytes exceeds the {max} limit"),
            DistError::PrefixMismatch { expected, got } => {
                write!(f, "prefix hash mismatch: worker holds {got:#018x}, request assumes {expected:#018x}")
            }
            DistError::NoRoute => write!(f, "no gapless worker chain covers every layer"),
            DistError::Transport(m) => write!(f, "transport: {m}"),
        }
    }
}

impl std::error::Error for DistError {}

/// Default ceiling on a single frame. A hidden-state batch is `chunk × n_hc × n_embd × 4` bytes, which is
/// hundreds of MB at realistic settings, so the cap is generous — but it exists, because the alternative
/// is allocating whatever a peer's length field says.
pub const MAX_FRAME: usize = 1 << 30;

/// One wire frame: `magic | kind | payload_len`, then the payload.
///
/// Header integers are **big-endian**. Not for performance — it is one field — but so a frame is
/// readable in a packet dump and so a byte-order mistake shows up as `BadMagic` on the first frame
/// instead of as a plausible-looking length.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Frame {
    pub kind: Kind,
    pub payload: Vec<u8>,
}

pub const HEADER_LEN: usize = 12;

impl Frame {
    pub fn new(kind: Kind, payload: Vec<u8>) -> Self { Self { kind, payload } }

    pub fn encode(&self) -> Vec<u8> {
        let mut o = Vec::with_capacity(HEADER_LEN + self.payload.len());
        o.extend_from_slice(&MAGIC.to_be_bytes());
        o.extend_from_slice(&(self.kind as u32).to_be_bytes());
        o.extend_from_slice(&(self.payload.len() as u32).to_be_bytes());
        o.extend_from_slice(&self.payload);
        o
    }

    /// Decode one frame, returning it and how many bytes it consumed.
    ///
    /// Validates in order: length, magic, kind, size ceiling, then availability. Checking magic before
    /// trusting the length is what stops a garbage or misaligned stream from being read as a request to
    /// allocate several gigabytes.
    pub fn decode(b: &[u8]) -> Result<(Frame, usize), DistError> {
        if b.len() < HEADER_LEN {
            return Err(DistError::Truncated { want: HEADER_LEN, got: b.len() });
        }
        let magic = u32::from_be_bytes(b[0..4].try_into().unwrap());
        if magic != MAGIC { return Err(DistError::BadMagic(magic)); }
        let kind_raw = u32::from_be_bytes(b[4..8].try_into().unwrap());
        let kind = Kind::from_u32(kind_raw).ok_or(DistError::UnknownKind(kind_raw))?;
        let len = u32::from_be_bytes(b[8..12].try_into().unwrap()) as usize;
        if len > MAX_FRAME { return Err(DistError::TooLarge { want: len, max: MAX_FRAME }); }
        let total = HEADER_LEN + len;
        if b.len() < total { return Err(DistError::Truncated { want: total, got: b.len() }); }
        Ok((Frame { kind, payload: b[HEADER_LEN..total].to_vec() }, total))
    }
}

/// Somewhere to send a frame and get one back. Implemented over TCP in production and in-process in
/// tests, which is what makes the failure paths exercisable.
pub trait Transport {
    fn round_trip(&mut self, f: &Frame) -> Result<Frame, DistError>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frames_round_trip() {
        for (k, p) in [
            (Kind::Hello, vec![]),
            (Kind::Work, vec![1, 2, 3, 4, 5]),
            (Kind::Ack, vec![]),
            (Kind::Result, vec![0xFF; 1000]),
        ] {
            let f = Frame::new(k, p.clone());
            let (back, n) = Frame::decode(&f.encode()).unwrap();
            assert_eq!(back, f);
            assert_eq!(n, HEADER_LEN + p.len());
        }
    }

    #[test]
    fn frames_decode_back_to_back_from_one_buffer() {
        let mut buf = Vec::new();
        buf.extend(Frame::new(Kind::Work, vec![7; 3]).encode());
        buf.extend(Frame::new(Kind::Ack, vec![]).encode());
        let (a, n) = Frame::decode(&buf).unwrap();
        assert_eq!(a.kind, Kind::Work);
        let (b, _) = Frame::decode(&buf[n..]).unwrap();
        assert_eq!(b.kind, Kind::Ack);
    }

    #[test]
    fn truncation_is_reported_as_truncation_not_corruption() {
        // A partly-arrived stream is normal; conflating it with corruption would make a caller drop a
        // healthy connection instead of waiting for the rest.
        let e = Frame::new(Kind::Work, vec![1; 100]).encode();
        for cut in [0usize, 5, HEADER_LEN, HEADER_LEN + 50] {
            assert!(
                matches!(Frame::decode(&e[..cut]), Err(DistError::Truncated { .. })),
                "cut at {cut} was not reported as truncation"
            );
        }
    }

    #[test]
    fn a_hostile_length_is_refused_before_it_is_allocated() {
        // The failure this prevents: a peer (or a misaligned stream) claiming a 4 GB payload and the
        // reader dutifully trying to reserve it.
        let mut b = Vec::new();
        b.extend_from_slice(&MAGIC.to_be_bytes());
        b.extend_from_slice(&(Kind::Work as u32).to_be_bytes());
        b.extend_from_slice(&u32::MAX.to_be_bytes());
        assert!(matches!(Frame::decode(&b), Err(DistError::TooLarge { .. })));
    }

    #[test]
    fn magic_is_checked_before_the_length_is_trusted() {
        let mut b = vec![0u8; HEADER_LEN];
        b[8..12].copy_from_slice(&u32::MAX.to_be_bytes());
        assert!(matches!(Frame::decode(&b), Err(DistError::BadMagic(_))), "trusted a length from a foreign frame");
    }

    #[test]
    fn a_byte_order_mistake_surfaces_immediately() {
        // Big-endian headers exist so this is the first thing that fails, loudly, rather than a
        // little-endian reader inventing a plausible length from a valid frame.
        let f = Frame::new(Kind::Work, vec![1, 2, 3]);
        let mut e = f.encode();
        e[0..4].copy_from_slice(&MAGIC.to_le_bytes());
        assert!(matches!(Frame::decode(&e), Err(DistError::BadMagic(_))));
    }

    #[test]
    fn unknown_kinds_are_refused_rather_than_ignored() {
        // A newer peer sending a frame we do not understand is an error, not something to skip: skipping
        // would silently drop work and look like a hang.
        let mut b = Vec::new();
        b.extend_from_slice(&MAGIC.to_be_bytes());
        b.extend_from_slice(&999u32.to_be_bytes());
        b.extend_from_slice(&0u32.to_be_bytes());
        assert!(matches!(Frame::decode(&b), Err(DistError::UnknownKind(999))));
    }
}
