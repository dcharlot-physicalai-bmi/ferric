//! **Prefix-hash guards** — the reason distributed inference can be trusted.
//!
//! Every worker in a pipeline holds KV state for the tokens it has already processed. That state is
//! *implicit*: nothing in a request says "you must currently hold exactly these 8,000 tokens". So if a
//! worker restarts, or a request is retried against a replacement, or two sessions interleave, a worker
//! can be handed continuation work while holding the wrong state.
//!
//! It will not fail. It will compute attention against whatever KV it happens to have and return a
//! confident, wrong answer — and because the pipeline moves activations rather than text, nothing
//! downstream can tell.
//!
//! The fix is to make the state explicit: each request carries a **rolling hash of every token processed
//! so far**, and a worker refuses work whose hash does not match its own. A restarted worker at position
//! 0 cannot accept work for position 8,000, because `0 != h(8,000 tokens)`.
//!
//! ## Why a rolling hash rather than a token count
//!
//! A count answers "how many?" and the failure here is "*which* ones?". Two sessions of equal length
//! collide on a count and are then indistinguishable — which is precisely the interleaved-sessions case.

use crate::DistError;

/// FNV-1a-64 over the little-endian bytes of every token processed so far.
///
/// Chosen for being trivially reproducible across implementations and languages: a guard is only useful
/// if an independent worker computes the identical value, so an obscure or platform-dependent hash would
/// defeat the purpose. Not adversarial — this catches state mismatches, not attackers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PrefixHash(pub u64);

impl PrefixHash {
    pub const INIT: u64 = 0xcbf2_9ce4_8422_2325;
    const PRIME: u64 = 0x0000_0100_0000_01b3;

    pub fn new() -> Self { Self(Self::INIT) }

    /// Extend with one token. Order-dependent by construction — the same tokens in a different order give
    /// a different hash, which is exactly what is wanted.
    pub fn push(&mut self, token: u32) {
        for b in token.to_le_bytes() {
            self.0 ^= b as u64;
            self.0 = self.0.wrapping_mul(Self::PRIME);
        }
    }

    pub fn extend(&mut self, tokens: &[u32]) {
        for &t in tokens { self.push(t); }
    }

    pub fn of(tokens: &[u32]) -> Self {
        let mut h = Self::new();
        h.extend(tokens);
        h
    }
}

impl Default for PrefixHash {
    fn default() -> Self { Self::new() }
}

/// What a worker decided about an incoming request.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Verdict {
    /// State matches; the work was applied and the hash advanced.
    Accepted,
    /// State does not match. **Recoverable by replaying the transcript** — the coordinator keeps the
    /// route and re-sends from a point both sides agree on. Distinguished from a transport failure
    /// because the remedy differs: this worker is alive and correct to refuse.
    Refused { expected: u64, held: u64 },
}

/// One worker's view of a session: which tokens it believes it has processed.
#[derive(Debug, Clone)]
pub struct Session {
    pub id: u64,
    hash: PrefixHash,
    tokens: usize,
}

impl Session {
    pub fn new(id: u64) -> Self { Self { id, hash: PrefixHash::new(), tokens: 0 } }

    pub fn hash(&self) -> PrefixHash { self.hash }
    pub fn tokens(&self) -> usize { self.tokens }

    /// Apply work if and only if the request's assumed prefix matches this worker's state.
    ///
    /// On refusal the session is left **completely untouched** — not partially advanced. A guard that
    /// mutated state on the way to rejecting it would turn one recoverable mismatch into a permanently
    /// desynchronised worker, and the replay that was supposed to fix things would fail too.
    pub fn accept(&mut self, assumed_prefix: PrefixHash, tokens: &[u32]) -> Verdict {
        if assumed_prefix != self.hash {
            return Verdict::Refused { expected: assumed_prefix.0, held: self.hash.0 };
        }
        self.hash.extend(tokens);
        self.tokens += tokens.len();
        Verdict::Accepted
    }

    /// Roll back to a known-good point by replaying the agreed transcript from scratch.
    ///
    /// Deliberately a full recompute rather than an incremental rewind: the KV state behind this hash is
    /// not invertible, so "undo the last N tokens" is not an operation a worker can honestly offer.
    pub fn reset_to(&mut self, transcript: &[u32]) {
        self.hash = PrefixHash::of(transcript);
        self.tokens = transcript.len();
    }

    /// Convert a verdict into an error, for callers that would rather propagate than branch.
    pub fn check(&mut self, assumed: PrefixHash, tokens: &[u32]) -> Result<(), DistError> {
        match self.accept(assumed, tokens) {
            Verdict::Accepted => Ok(()),
            Verdict::Refused { expected, held } => {
                Err(DistError::PrefixMismatch { expected, got: held })
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_matching_worker_accepts_and_advances() {
        let mut s = Session::new(1);
        assert_eq!(s.accept(PrefixHash::new(), &[10, 11, 12]), Verdict::Accepted);
        assert_eq!(s.tokens(), 3);
        assert_eq!(s.hash(), PrefixHash::of(&[10, 11, 12]));
        assert_eq!(s.accept(PrefixHash::of(&[10, 11, 12]), &[13]), Verdict::Accepted);
        assert_eq!(s.hash(), PrefixHash::of(&[10, 11, 12, 13]));
    }

    #[test]
    fn a_restarted_worker_cannot_accept_continuation_work() {
        // THE failure this module exists for. Without the guard the fresh worker computes attention
        // against empty KV and returns a confident wrong answer, and because the pipeline carries
        // activations rather than text, nothing downstream can tell.
        let history: Vec<u32> = (0..8000).collect();
        let mut fresh = Session::new(1); // just restarted: holds nothing
        let v = fresh.accept(PrefixHash::of(&history), &[9001]);
        assert!(matches!(v, Verdict::Refused { .. }), "a fresh worker accepted work for position 8000");
        assert_eq!(fresh.tokens(), 0, "a refusal must not advance state");
    }

    #[test]
    fn two_sessions_of_equal_length_do_not_collide() {
        // Why a rolling hash and not a token count: a count answers "how many", and the failure mode is
        // "which ones". These two are indistinguishable by length.
        let a: Vec<u32> = vec![1, 2, 3, 4, 5];
        let b: Vec<u32> = vec![5, 4, 3, 2, 1];
        assert_ne!(PrefixHash::of(&a), PrefixHash::of(&b));
        let mut s = Session::new(1);
        s.accept(PrefixHash::new(), &a);
        assert!(matches!(s.accept(PrefixHash::of(&b), &[9]), Verdict::Refused { .. }));
    }

    #[test]
    fn a_refusal_leaves_the_session_byte_identical() {
        // A guard that mutated state on the way to rejecting would convert one recoverable mismatch into
        // a permanently desynchronised worker — and the replay meant to fix it would fail too.
        let mut s = Session::new(1);
        s.accept(PrefixHash::new(), &[1, 2, 3]);
        let (h, n) = (s.hash(), s.tokens());
        for bogus in [PrefixHash::new(), PrefixHash::of(&[9, 9]), PrefixHash(0)] {
            assert!(matches!(s.accept(bogus, &[42; 100]), Verdict::Refused { .. }));
            assert_eq!((s.hash(), s.tokens()), (h, n), "refusal mutated the session");
        }
        // Still usable afterwards.
        assert_eq!(s.accept(h, &[4]), Verdict::Accepted);
    }

    #[test]
    fn replay_recovers_a_refused_worker() {
        // The full recovery loop: mismatch -> refuse -> coordinator replays the agreed transcript ->
        // the worker accepts again. This is why a refusal keeps the route instead of dropping it.
        let transcript: Vec<u32> = (0..500).collect();
        let mut worker = Session::new(7);
        worker.accept(PrefixHash::new(), &[99, 98]); // diverged somehow
        assert!(matches!(worker.accept(PrefixHash::of(&transcript), &[1]), Verdict::Refused { .. }));
        worker.reset_to(&transcript);
        assert_eq!(worker.accept(PrefixHash::of(&transcript), &[1]), Verdict::Accepted);
        assert_eq!(worker.tokens(), 501);
    }

    #[test]
    fn incremental_hashing_equals_whole_sequence_hashing() {
        // The coordinator hashes as it goes; a worker may hash a whole transcript at once during replay.
        // If those disagreed, recovery would never converge.
        let all: Vec<u32> = (0..1000).map(|i| i * 7919).collect();
        let mut inc = PrefixHash::new();
        for c in all.chunks(37) { inc.extend(c); }
        assert_eq!(inc, PrefixHash::of(&all));
    }

    #[test]
    fn the_hash_is_a_fixed_wire_constant() {
        // Workers may be different builds. Pinning known values means a change to the hash shows up here
        // rather than as an entire fleet mysteriously refusing each other. (These were computed from the
        // spec, not transcribed from the implementation — a constant copied out of the code under test
        // pins nothing.)
        assert_eq!(PrefixHash::new().0, 0xcbf2_9ce4_8422_2325);
        assert_eq!(PrefixHash::of(&[0]).0, 0x4d25767f9dce13f5);
        assert_eq!(PrefixHash::of(&[1, 2, 3]).0, 0xfd1f0f4381eb0395);
    }

    #[test]
    fn check_reports_both_sides_of_the_mismatch() {
        // The error must name what the request assumed AND what the worker holds; with only one, an
        // operator cannot tell which side is stale.
        let mut s = Session::new(1);
        s.accept(PrefixHash::new(), &[1, 2, 3]);
        let e = s.check(PrefixHash::new(), &[4]).unwrap_err();
        match e {
            DistError::PrefixMismatch { expected, got } => {
                assert_eq!(expected, PrefixHash::INIT);
                assert_eq!(got, PrefixHash::of(&[1, 2, 3]).0);
            }
            other => panic!("wrong error: {other:?}"),
        }
    }
}
