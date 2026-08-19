//! One id space for words and measurements.
//!
//! A sensor-language decoder reads `"the vibration"`, then a run of signal codes, then
//! `"rose sharply"`, through a single embedding table. For that to work the two vocabularies have
//! to be laid out in one contiguous id range with an agreed boundary, and the mapping in both
//! directions has to be exact — an off-by-one here shifts every signal token onto a word and the
//! model still trains, just on nonsense.
//!
//! Layout: **text first, signals after.** Text occupies `0..text_len`, signal codes occupy
//! `text_len..text_len + codebook_size`, and the marker tokens sit at the very top. Text is placed
//! first on purpose so that an existing text tokenizer's ids are unchanged and a text-only
//! checkpoint can be extended into a sensor-language one without renumbering its embeddings.

use crate::fsq::Fsq;

/// What a token id turned out to be.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TokenKind {
    /// A text token, carrying its id in the original text vocabulary.
    Text(u32),
    /// A quantized signal patch, carrying its FSQ code index.
    Signal(u32),
    /// Opens a run of signal tokens.
    SignalBegin,
    /// Closes a run of signal tokens.
    SignalEnd,
    /// Separates channels within one run, so a multi-channel stream is one sequence.
    ChannelSep,
}

/// Text ids, signal codes and three markers in one contiguous id space.
#[derive(Debug, Clone)]
pub struct HybridVocab {
    text_len: u32,
    fsq: Fsq,
}

/// Why an id or a construction was rejected.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VocabError {
    /// The combined vocabulary would not fit in `u32`.
    TooLarge,
    /// An id past the end of the combined space.
    OutOfRange { id: u32, total: u32 },
    /// A text id at or past the text vocabulary length.
    TextOutOfRange { id: u32, text_len: u32 },
    /// A signal code at or past the codebook size.
    SignalOutOfRange { code: u32, codebook: u32 },
}

impl core::fmt::Display for VocabError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            VocabError::TooLarge => write!(f, "vocab: combined size overflows u32"),
            VocabError::OutOfRange { id, total } => write!(f, "vocab: id {id} >= total {total}"),
            VocabError::TextOutOfRange { id, text_len } => {
                write!(f, "vocab: text id {id} >= text vocabulary {text_len}")
            }
            VocabError::SignalOutOfRange { code, codebook } => {
                write!(f, "vocab: signal code {code} >= codebook {codebook}")
            }
        }
    }
}

/// Marker count. Kept as a named constant because every offset below depends on it.
const MARKERS: u32 = 3;

impl HybridVocab {
    pub fn new(text_len: u32, fsq: Fsq) -> Result<Self, VocabError> {
        let total = (text_len as u64) + (fsq.codebook_size() as u64) + (MARKERS as u64);
        if total > u32::MAX as u64 {
            return Err(VocabError::TooLarge);
        }
        Ok(Self { text_len, fsq })
    }

    #[inline]
    pub fn text_len(&self) -> u32 {
        self.text_len
    }

    #[inline]
    pub fn fsq(&self) -> &Fsq {
        &self.fsq
    }

    /// Total embedding rows a model needs.
    #[inline]
    pub fn total(&self) -> u32 {
        self.text_len + self.fsq.codebook_size() + MARKERS
    }

    #[inline]
    fn signal_base(&self) -> u32 {
        self.text_len
    }

    #[inline]
    fn marker_base(&self) -> u32 {
        self.text_len + self.fsq.codebook_size()
    }

    pub fn text(&self, id: u32) -> Result<u32, VocabError> {
        if id >= self.text_len {
            return Err(VocabError::TextOutOfRange { id, text_len: self.text_len });
        }
        Ok(id)
    }

    pub fn signal(&self, code: u32) -> Result<u32, VocabError> {
        let n = self.fsq.codebook_size();
        if code >= n {
            return Err(VocabError::SignalOutOfRange { code, codebook: n });
        }
        Ok(self.signal_base() + code)
    }

    #[inline]
    pub fn signal_begin(&self) -> u32 {
        self.marker_base()
    }
    #[inline]
    pub fn signal_end(&self) -> u32 {
        self.marker_base() + 1
    }
    #[inline]
    pub fn channel_sep(&self) -> u32 {
        self.marker_base() + 2
    }

    /// Resolve an id back to what it is. Exact inverse of the constructors above.
    pub fn kind(&self, id: u32) -> Result<TokenKind, VocabError> {
        let total = self.total();
        if id >= total {
            return Err(VocabError::OutOfRange { id, total });
        }
        if id < self.text_len {
            return Ok(TokenKind::Text(id));
        }
        let m = self.marker_base();
        if id < m {
            return Ok(TokenKind::Signal(id - self.signal_base()));
        }
        Ok(match id - m {
            0 => TokenKind::SignalBegin,
            1 => TokenKind::SignalEnd,
            _ => TokenKind::ChannelSep,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn v() -> HybridVocab {
        HybridVocab::new(32_000, Fsq::signal_15bit()).unwrap()
    }

    #[test]
    fn total_accounts_for_text_signals_and_markers() {
        assert_eq!(v().total(), 32_000 + 32_768 + 3);
    }

    /// EXHAUSTIVE over the whole combined space. Every id must resolve to exactly one kind, and
    /// every kind must map back to the same id. An off-by-one at either boundary silently aliases
    /// a signal code onto a word, and a model trained through that still converges — on nonsense.
    #[test]
    fn every_id_resolves_and_round_trips() {
        let v = v();
        let mut text = 0u32;
        let mut sig = 0u32;
        let mut mark = 0u32;
        for id in 0..v.total() {
            match v.kind(id).unwrap() {
                TokenKind::Text(t) => {
                    assert_eq!(v.text(t).unwrap(), id);
                    text += 1;
                }
                TokenKind::Signal(c) => {
                    assert_eq!(v.signal(c).unwrap(), id);
                    sig += 1;
                }
                TokenKind::SignalBegin => {
                    assert_eq!(v.signal_begin(), id);
                    mark += 1;
                }
                TokenKind::SignalEnd => {
                    assert_eq!(v.signal_end(), id);
                    mark += 1;
                }
                TokenKind::ChannelSep => {
                    assert_eq!(v.channel_sep(), id);
                    mark += 1;
                }
            }
        }
        assert_eq!((text, sig, mark), (32_000, 32_768, 3));
    }

    /// Text ids are unchanged by the extension, which is the point of putting text first: an
    /// existing checkpoint's embedding rows keep their meaning.
    #[test]
    fn text_ids_are_left_where_they_were() {
        let v = v();
        for id in [0u32, 1, 999, 31_999] {
            assert_eq!(v.text(id).unwrap(), id);
        }
    }

    #[test]
    fn the_three_boundaries_are_exactly_where_they_should_be() {
        let v = v();
        assert_eq!(v.kind(31_999).unwrap(), TokenKind::Text(31_999));
        assert_eq!(v.kind(32_000).unwrap(), TokenKind::Signal(0));
        assert_eq!(v.kind(32_000 + 32_767).unwrap(), TokenKind::Signal(32_767));
        assert_eq!(v.kind(32_000 + 32_768).unwrap(), TokenKind::SignalBegin);
        assert_eq!(v.kind(v.total() - 1).unwrap(), TokenKind::ChannelSep);
        assert!(v.kind(v.total()).is_err());
    }

    #[test]
    fn out_of_range_is_refused_in_both_directions() {
        let v = v();
        assert!(v.text(32_000).is_err());
        assert!(v.signal(32_768).is_err());
        assert!(v.kind(v.total()).is_err());
    }

    /// A quantized latent must survive the whole path: latent -> code -> index -> token id ->
    /// index -> code. This is the path a real signal takes, and it is where the two modules meet.
    #[test]
    fn a_latent_survives_the_full_path_to_a_token_and_back() {
        let v = v();
        let q = v.fsq().clone();
        for seed in 0..2_000u32 {
            // Deterministic spread of latents across the saturating range, no RNG.
            let z: Vec<f32> = (0..q.dim())
                .map(|d| {
                    let t = ((seed * 7 + d as u32 * 131) % 997) as f32 / 997.0;
                    (t - 0.5) * 24.0
                })
                .collect();
            let code = q.quantize(&z).unwrap();
            let idx = q.to_index(&code).unwrap();
            let id = v.signal(idx).unwrap();
            match v.kind(id).unwrap() {
                TokenKind::Signal(back) => {
                    assert_eq!(back, idx);
                    assert_eq!(q.from_index(back).unwrap(), code);
                }
                other => panic!("expected a signal token, got {other:?}"),
            }
        }
    }
}
