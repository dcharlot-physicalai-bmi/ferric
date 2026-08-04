//! End-to-end pipeline behaviour, in-process: **a mid-session worker failure must be transparent.**
//!
//! The claim under test is not "the pipeline works when nothing goes wrong" — that is the easy half. It
//! is that a worker restarting mid-session produces *byte-identical output* to a run where nothing
//! failed, because the guard refuses the mismatched work and replay restores the state. If recovery were
//! merely "close", the failure mode would be a model that silently gets slightly worse under load, which
//! is close to undiagnosable in production.
//!
//! No sockets and no second machine: `Transport` is a trait, so the failure paths are ordinary code.

use ferric_dist::{plan_route, Frame, Kind, PrefixHash, Registration, Session, Verdict};

/// A stand-in worker. Its "computation" is a pure function of the token history it has processed, which
/// is what lets the test assert that failures do not perturb the result: the tail worker's output must
/// equal `PrefixHash::of(every token)` no matter what happened along the way.
struct Worker {
    id: String,
    session: Session,
    alive: bool,
    refusals: u32,
}

impl Worker {
    fn new(id: &str) -> Self {
        Self { id: id.into(), session: Session::new(1), alive: true, refusals: 0 }
    }
    /// Simulate a process restart: KV state is gone, the worker is up again and honest about it.
    fn restart(&mut self) {
        self.session = Session::new(1);
        self.alive = true;
    }
    fn handle(&mut self, assumed: PrefixHash, tokens: &[u32]) -> Result<PrefixHash, Verdict> {
        match self.session.accept(assumed, tokens) {
            Verdict::Accepted => Ok(self.session.hash()),
            v => { self.refusals += 1; Err(v) }
        }
    }
}

/// Run one chunk through the chain. Each hop must agree on the prefix before it will contribute.
///
/// Returns the tail worker's state hash, which stands in for logits.
fn push_chunk(chain: &mut [Worker], tokens: &[u32], agreed: &[u32]) -> Result<PrefixHash, String> {
    let assumed = PrefixHash::of(agreed);
    let mut last = None;
    for w in chain.iter_mut() {
        if !w.alive { return Err(format!("{} is down", w.id)); }
        match w.handle(assumed, tokens) {
            Ok(h) => last = Some(h),
            Err(Verdict::Refused { .. }) => return Err(format!("{} refused", w.id)),
            Err(_) => unreachable!(),
        }
    }
    last.ok_or_else(|| "empty chain".into())
}

/// The recovery loop a coordinator actually runs: on a refusal, keep the route and replay the agreed
/// transcript into every worker, then retry. Distinct from a transport failure, where the route is
/// dropped and a replacement is awaited — the worker that refused is alive and correct to do so.
fn push_with_recovery(
    chain: &mut [Worker],
    tokens: &[u32],
    agreed: &[u32],
    replays: &mut u32,
) -> PrefixHash {
    if let Ok(h) = push_chunk(chain, tokens, agreed) {
        return h;
    }
    *replays += 1;
    for w in chain.iter_mut() {
        w.restart();
        w.session.reset_to(agreed);
    }
    push_chunk(chain, tokens, agreed).expect("replay must converge")
}

fn chain3() -> Vec<Worker> {
    vec![Worker::new("head"), Worker::new("mid"), Worker::new("tail")]
}

#[test]
fn a_clean_run_produces_the_history_hash() {
    let mut c = chain3();
    let mut agreed: Vec<u32> = Vec::new();
    let mut out = PrefixHash::new();
    for chunk in [&[1u32, 2, 3][..], &[4, 5], &[6]] {
        out = push_chunk(&mut c, chunk, &agreed).unwrap();
        agreed.extend_from_slice(chunk);
    }
    assert_eq!(out, PrefixHash::of(&[1, 2, 3, 4, 5, 6]));
}

#[test]
fn a_mid_session_restart_is_transparent_after_replay() {
    // The property that matters: the final result is IDENTICAL to a run where nothing failed. Not close,
    // identical — otherwise a fleet under load degrades in a way nothing can attribute.
    let tokens: Vec<Vec<u32>> = (0..10).map(|i| vec![i * 3, i * 3 + 1, i * 3 + 2]).collect();

    let mut clean = chain3();
    let mut agreed: Vec<u32> = Vec::new();
    let mut clean_out = PrefixHash::new();
    for c in &tokens {
        clean_out = push_chunk(&mut clean, c, &agreed).unwrap();
        agreed.extend_from_slice(c);
    }

    let mut faulty = chain3();
    let mut agreed2: Vec<u32> = Vec::new();
    let mut faulty_out = PrefixHash::new();
    let mut replays = 0u32;
    for (i, c) in tokens.iter().enumerate() {
        if i == 5 { faulty[1].restart(); } // the middle worker loses everything mid-session
        faulty_out = push_with_recovery(&mut faulty, c, &agreed2, &mut replays);
        agreed2.extend_from_slice(c);
    }

    assert_eq!(faulty_out, clean_out, "a restart perturbed the result");
    assert_eq!(replays, 1, "expected exactly one replay, got {replays}");
    assert!(faulty[1].refusals > 0, "the restarted worker never refused — the guard did not engage");
}

#[test]
fn without_the_guard_a_restart_would_corrupt_the_result() {
    // Demonstrates what the guard buys, by doing the same run with the check bypassed. This is the
    // behaviour of a pipeline that trusts its workers, and the point is that it does not error — it just
    // returns a different answer.
    let tokens: Vec<Vec<u32>> = (0..10).map(|i| vec![i * 3, i * 3 + 1, i * 3 + 2]).collect();
    let mut unguarded = Session::new(1);
    let mut all: Vec<u32> = Vec::new();
    for (i, c) in tokens.iter().enumerate() {
        if i == 5 { unguarded = Session::new(1); } // restart, state silently gone
        let assumed = unguarded.hash(); // trust whatever the worker holds: no cross-check
        unguarded.accept(assumed, c);
        all.extend_from_slice(c);
    }
    assert_ne!(
        unguarded.hash(),
        PrefixHash::of(&all),
        "the unguarded path happened to agree; the test proves nothing"
    );
}

#[test]
fn a_transport_failure_is_not_treated_as_a_refusal() {
    // Different faults, different remedies. A refusal means the worker is alive and its state is wrong —
    // keep the route, replay. A dead worker means the route itself is gone — conflating them makes a
    // coordinator replay forever into a socket that will never answer.
    let mut c = chain3();
    c[1].alive = false;
    let err = push_chunk(&mut c, &[1, 2, 3], &[]).unwrap_err();
    assert!(err.contains("is down"), "got {err}");
    assert_eq!(c[1].refusals, 0, "a dead worker must not be recorded as having refused");
}

#[test]
fn routing_and_the_guard_compose_over_a_real_registration_set() {
    let regs = vec![
        Registration { id: "head".into(), layer_start: 0, layer_end: 12, has_output: false, model: 42 },
        Registration { id: "mid".into(), layer_start: 12, layer_end: 24, has_output: false, model: 42 },
        Registration { id: "tail".into(), layer_start: 24, layer_end: 36, has_output: true, model: 42 },
        // Decoys that must not be selected. "other" is the important one: it is a COMPLETE, valid,
        // single-hop chain — for the wrong checkpoint. An earlier plan_route inferred the model instead
        // of taking it, and happily returned this route; the caller now says which model it wants.
        Registration { id: "other".into(), layer_start: 0, layer_end: 36, has_output: true, model: 7 },
        Registration { id: "dup".into(), layer_start: 6, layer_end: 24, has_output: false, model: 42 },
    ];
    let route = plan_route(&regs, 36, 42).unwrap();
    assert_eq!(route.hops.iter().map(|h| h.id.as_str()).collect::<Vec<_>>(), ["head", "mid", "tail"]);
    assert_eq!(route.model, 42);

    let mut chain: Vec<Worker> = route.hops.iter().map(|h| Worker::new(&h.id)).collect();
    let out = push_chunk(&mut chain, &[9, 8, 7], &[]).unwrap();
    assert_eq!(out, PrefixHash::of(&[9, 8, 7]));
}

#[test]
fn frames_survive_a_byte_stream_that_arrives_in_pieces() {
    // A socket delivers arbitrary fragments. Decoding must report "not yet" rather than consuming a
    // partial frame — a reader that guesses here corrupts every subsequent frame in the stream.
    let msgs = [
        Frame::new(Kind::Work, vec![1; 40]),
        Frame::new(Kind::Ack, vec![]),
        Frame::new(Kind::Result, vec![2; 17]),
    ];
    let wire: Vec<u8> = msgs.iter().flat_map(|f| f.encode()).collect();

    let mut got = Vec::new();
    let mut buf: Vec<u8> = Vec::new();
    for byte in &wire {
        buf.push(*byte);
        while let Ok((f, n)) = Frame::decode(&buf) {
            got.push(f);
            buf.drain(..n);
        }
    }
    assert_eq!(got, msgs.to_vec(), "byte-at-a-time reassembly did not reproduce the messages");
    assert!(buf.is_empty(), "{} bytes left unconsumed", buf.len());
}
