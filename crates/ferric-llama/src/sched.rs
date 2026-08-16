//! **Continuous batching** — the scheduler that keeps a batch full instead of draining it.
//!
//! `ferric-serve` has run one request at a time since it was written, on the reasoning that "the GPU
//! serializes anyway". That is true of a *single* sequence and false of a server: decode is
//! memory-bandwidth-bound, so a batch of N sequences reads the weights **once** for all N instead of
//! once each. Serving 8 sessions one after another reads the weight set 8 times; serving them in a
//! batch reads it once. The throughput difference is close to N until the batch saturates compute.
//!
//! The batched forward already exists — [`crate::qwen3::Qwen3::forward_batch`] advances N independent
//! sequences by one token each and is asserted to produce logits identical to running each alone.
//! What was missing is the part above it: deciding *which* sequences are in the batch on any given
//! step, admitting new arrivals mid-flight, and retiring finished ones without stalling the rest.
//!
//! ## Why "continuous"
//!
//! Static batching waits for a batch to form, runs it to completion, and starts the next. A batch is
//! then only as fast as its slowest member, and every sequence that finishes early leaves a hole that
//! stays empty. Continuous batching admits a waiting request into that hole on the very next step.
//!
//! ```text
//!   static:      [A B C D] ....... A B C D all finish ....... [E F G H]
//!   continuous:  [A B C D] → B done → [A E C D] → C done → [A E F D] → ...
//! ```
//!
//! ## What this module is and is not
//!
//! It is the **policy**, kept deliberately free of the model, the GPU and the transport, so it can be
//! unit-tested exhaustively without any of them — which is the only reason the invariants below are
//! checkable at all. The engine loop calls [`Scheduler::step_batch`] to learn which sequences to run,
//! then feeds the sampled tokens back via [`Scheduler::record`].
//!
//! The invariants it exists to hold:
//!
//! - a sequence is in **at most one** state, and every transition is recorded;
//! - admission never exceeds `max_batch`, because exceeding it is an out-of-memory, not a slowdown;
//! - a finished sequence frees its slot **on the same step** it finishes;
//! - **arrival order is preserved among waiting requests** — without that, a busy server starves its
//!   oldest request indefinitely and the failure looks like a hang rather than a bug.

use std::collections::VecDeque;

/// Identifier for one in-flight sequence. Monotonic, never reused within a scheduler.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct SeqId(pub u64);

/// Why a sequence left the batch. Retained because "it stopped" is not one thing, and a server that
/// cannot distinguish these cannot report honestly to the caller.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Done {
    /// Hit a stop token.
    Eos,
    /// Hit its own `max_tokens` budget.
    Length,
    /// The caller went away.
    Cancelled,
}

/// One sequence's scheduling state. The model's KV cache lives outside this, keyed by `id`.
#[derive(Debug, Clone)]
pub struct Seq {
    pub id: SeqId,
    /// Tokens to feed on the next step. Multiple on the first (prefill) step, one thereafter.
    pub pending: Vec<u32>,
    /// Tokens generated so far.
    pub emitted: usize,
    pub max_tokens: usize,
}

impl Seq {
    /// Whether this sequence has spent its budget. Checked *before* admission as well as after each
    /// token, so a `max_tokens: 0` request is refused rather than admitted and immediately retired.
    pub fn exhausted(&self) -> bool { self.emitted >= self.max_tokens }
}

/// Continuous-batching policy over a fixed number of slots.
#[derive(Debug)]
pub struct Scheduler {
    max_batch: usize,
    /// How many sequences may be admitted on a single step. `None` = fill every free slot at once.
    ///
    /// Admission is not free: `forward_batch` is the DECODE step, so each new sequence prefills alone
    /// before joining. Filling five empty slots in one step therefore pays five sequential prefills
    /// before any batched decode happens, and that burst is measurable — it is most of why batching
    /// five short sequences measured 3.07x rather than the ~5x the weight-read argument suggests.
    ///
    /// Capping admission spreads that cost across steps. It is a genuine trade, not a free win: a low
    /// cap reaches a full batch more slowly, so a server that is mostly idle should leave it `None`
    /// and one under sustained load should cap it. Exposed rather than chosen because only the
    /// deployment knows which it is.
    max_admit_per_step: Option<usize>,
    waiting: VecDeque<Seq>,
    running: Vec<Seq>,
    next_id: u64,
    /// Cumulative counters, for the observability a server needs to explain itself.
    pub admitted: u64,
    pub finished: u64,
    /// Sequences retired this step, drained by the engine so it can free their caches.
    retired: Vec<(SeqId, Done)>,
}

impl Scheduler {
    pub fn new(max_batch: usize) -> Scheduler {
        assert!(max_batch > 0, "max_batch must be positive");
        Scheduler {
            max_batch, max_admit_per_step: None, waiting: VecDeque::new(), running: Vec::new(),
            next_id: 0, admitted: 0, finished: 0, retired: Vec::new(),
        }
    }

    /// Queue a request. Returns its id immediately — admission happens on a later step, so the caller
    /// gets a handle without the scheduler having to promise a slot it does not have.
    pub fn submit(&mut self, prompt: Vec<u32>, max_tokens: usize) -> SeqId {
        let id = SeqId(self.next_id);
        self.next_id += 1;
        self.waiting.push_back(Seq { id, pending: prompt, emitted: 0, max_tokens });
        id
    }

    /// Cap how many sequences may be admitted per step. See [`Scheduler::max_admit_per_step`].
    pub fn with_admit_cap(mut self, cap: usize) -> Scheduler {
        assert!(cap > 0, "an admission cap of 0 would never admit anything");
        self.max_admit_per_step = Some(cap);
        self
    }

    pub fn max_batch(&self) -> usize { self.max_batch }
    pub fn running(&self) -> &[Seq] { &self.running }
    pub fn waiting_len(&self) -> usize { self.waiting.len() }
    pub fn is_idle(&self) -> bool { self.running.is_empty() && self.waiting.is_empty() }

    /// Fill free slots from the waiting queue, **in arrival order**, and return the batch to run.
    ///
    /// This is where "continuous" happens: it is called every step, so a slot freed by a sequence
    /// finishing is refilled on the next step rather than at the end of the batch.
    pub fn step_batch(&mut self) -> Vec<SeqId> {
        let mut admitted_now = 0usize;
        while self.running.len() < self.max_batch {
            if self.max_admit_per_step.is_some_and(|c| admitted_now >= c) { break; }
            let Some(s) = self.waiting.pop_front() else { break };
            // A zero-budget request is retired without ever occupying a slot; admitting it would
            // consume a slot for one step to do nothing.
            if s.exhausted() {
                self.retired.push((s.id, Done::Length));
                self.finished += 1;
                continue;
            }
            self.admitted += 1;
            admitted_now += 1;
            self.running.push(s);
        }
        self.running.iter().map(|s| s.id).collect()
    }

    /// The tokens to feed for each running sequence, aligned with [`Scheduler::step_batch`]'s order.
    pub fn pending(&self) -> Vec<&[u32]> {
        self.running.iter().map(|s| s.pending.as_slice()).collect()
    }

    /// Record one sampled token for `id`, retiring the sequence if it is now complete.
    ///
    /// `stop` is the caller's decision that this token ends the sequence — the scheduler does not own
    /// the stop-token set, because that belongs to the tokenizer and varies per model.
    pub fn record(&mut self, id: SeqId, token: u32, stop: bool) {
        let Some(i) = self.running.iter().position(|s| s.id == id) else { return };
        let s = &mut self.running[i];
        s.emitted += 1;
        s.pending = vec![token];
        let done = if stop { Some(Done::Eos) } else if s.exhausted() { Some(Done::Length) } else { None };
        if let Some(d) = done {
            let s = self.running.remove(i);
            self.retired.push((s.id, d));
            self.finished += 1;
        }
    }

    /// Drop a sequence wherever it is — running, waiting, or already gone.
    pub fn cancel(&mut self, id: SeqId) {
        if let Some(i) = self.running.iter().position(|s| s.id == id) {
            self.running.remove(i);
            self.retired.push((id, Done::Cancelled));
            self.finished += 1;
        } else if let Some(i) = self.waiting.iter().position(|s| s.id == id) {
            self.waiting.remove(i);
            self.retired.push((id, Done::Cancelled));
            self.finished += 1;
        }
    }

    /// Take the sequences retired since the last call, so the engine can free their KV caches.
    ///
    /// Draining rather than peeking is deliberate: a cache freed twice is a bug the engine cannot
    /// detect, and one never freed is a leak that only shows up under load.
    pub fn take_retired(&mut self) -> Vec<(SeqId, Done)> { std::mem::take(&mut self.retired) }

    /// Fraction of slots occupied. The number that says whether batching is doing anything: a server
    /// sitting at 1/32 is paying batched-path complexity for serial throughput.
    pub fn occupancy(&self) -> f64 { self.running.len() as f64 / self.max_batch as f64 }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn drain_ids(s: &mut Scheduler) -> Vec<u64> { s.step_batch().iter().map(|i| i.0).collect() }

    #[test]
    fn a_freed_slot_is_refilled_on_the_very_next_step() {
        // THE property that makes it "continuous" rather than static. If a finished sequence's slot
        // stayed empty until the whole batch drained, throughput would collapse to the slowest member.
        let mut s = Scheduler::new(2);
        let a = s.submit(vec![1], 10);
        let _b = s.submit(vec![2], 10);
        let c = s.submit(vec![3], 10);
        assert_eq!(drain_ids(&mut s), [0, 1], "both slots fill first");
        assert_eq!(s.waiting_len(), 1, "c waits");

        s.record(a, 9, true);                       // a hits EOS
        assert_eq!(drain_ids(&mut s), [1, 2], "c took a's slot immediately");
        assert_eq!(s.take_retired(), [(a, Done::Eos)]);
        let _ = c;
    }

    #[test]
    fn admission_never_exceeds_max_batch() {
        // Overshooting is an out-of-memory, not a slowdown: each slot owns a KV cache.
        let mut s = Scheduler::new(3);
        for i in 0..10 { s.submit(vec![i as u32], 5); }
        assert_eq!(drain_ids(&mut s).len(), 3);
        assert_eq!(s.running().len(), 3);
        assert_eq!(s.waiting_len(), 7);
        // Stepping again must not admit more while the batch is full.
        assert_eq!(drain_ids(&mut s).len(), 3);
        assert_eq!(s.waiting_len(), 7);
    }

    #[test]
    fn waiting_requests_are_served_in_arrival_order() {
        // Without FIFO the oldest request can starve indefinitely on a busy server, and it presents
        // as a hang rather than as a bug.
        let mut s = Scheduler::new(1);
        let a = s.submit(vec![1], 1);
        let b = s.submit(vec![2], 1);
        let c = s.submit(vec![3], 1);
        assert_eq!(drain_ids(&mut s), [0]);
        s.record(a, 0, true);
        assert_eq!(drain_ids(&mut s), [1], "b before c");
        s.record(b, 0, true);
        assert_eq!(drain_ids(&mut s), [2]);
        let _ = c;
    }

    #[test]
    fn a_length_limited_sequence_retires_on_its_last_token() {
        let mut s = Scheduler::new(2);
        let a = s.submit(vec![1], 2);
        drain_ids(&mut s);
        s.record(a, 7, false);
        assert_eq!(s.running().len(), 1, "one token of two, still running");
        s.record(a, 8, false);
        assert!(s.running().is_empty(), "budget spent, retired without a stop token");
        assert_eq!(s.take_retired(), [(a, Done::Length)]);
    }

    #[test]
    fn a_zero_budget_request_never_occupies_a_slot() {
        // Admitting it would burn a slot for a step to accomplish nothing, and on a full server that
        // slot is the scarce resource.
        let mut s = Scheduler::new(1);
        let a = s.submit(vec![1], 0);
        assert!(drain_ids(&mut s).is_empty());
        assert_eq!(s.take_retired(), [(a, Done::Length)]);
        assert_eq!(s.admitted, 0, "it must not count as admitted");
    }

    #[test]
    fn cancel_works_from_both_running_and_waiting() {
        let mut s = Scheduler::new(1);
        let a = s.submit(vec![1], 9);
        let b = s.submit(vec![2], 9);
        drain_ids(&mut s);
        s.cancel(b);                                  // still waiting
        assert_eq!(s.waiting_len(), 0);
        s.cancel(a);                                  // running
        assert!(s.running().is_empty());
        let r = s.take_retired();
        assert!(r.contains(&(a, Done::Cancelled)) && r.contains(&(b, Done::Cancelled)));
        assert!(s.is_idle());
    }

    #[test]
    fn retired_is_drained_not_peeked() {
        // A cache freed twice is a bug the engine cannot detect; one never freed is a leak that only
        // appears under load. Draining makes both impossible by construction.
        let mut s = Scheduler::new(1);
        let a = s.submit(vec![1], 1);
        drain_ids(&mut s);
        s.record(a, 0, true);
        assert_eq!(s.take_retired().len(), 1);
        assert!(s.take_retired().is_empty(), "the same retirement must not be reported twice");
    }

    #[test]
    fn pending_is_the_prompt_first_then_one_token() {
        // Prefill feeds many tokens, decode feeds one. The engine relies on this to size its forward.
        let mut s = Scheduler::new(1);
        let a = s.submit(vec![7, 8, 9], 4);
        drain_ids(&mut s);
        assert_eq!(s.pending(), [&[7u32, 8, 9][..]], "first step is the whole prompt");
        s.record(a, 42, false);
        assert_eq!(s.pending(), [&[42u32][..]], "afterwards, one token");
    }

    #[test]
    fn an_admission_cap_bounds_the_prefill_burst() {
        // Every admitted sequence prefills ALONE before it can join a batched decode, so filling n
        // empty slots in one step costs n sequential prefills up front. That burst is most of why
        // batching five short sequences measured 3.07x rather than ~5x.
        let mut s = Scheduler::new(8).with_admit_cap(2);
        for i in 0..8 { s.submit(vec![i as u32], 9); }
        assert_eq!(drain_ids(&mut s).len(), 2, "at most 2 admitted on the first step");
        assert_eq!(drain_ids(&mut s).len(), 4, "2 more on the second");
        assert_eq!(drain_ids(&mut s).len(), 6);
        assert_eq!(drain_ids(&mut s).len(), 8, "reaches a full batch, just not in one step");
        assert_eq!(drain_ids(&mut s).len(), 8, "and stops at max_batch");
    }

    #[test]
    fn without_a_cap_every_free_slot_fills_at_once() {
        // The default must stay the eager one: a mostly-idle server should reach a full batch on the
        // first step, and capping admission there would only add latency.
        let mut s = Scheduler::new(8);
        for i in 0..8 { s.submit(vec![i as u32], 9); }
        assert_eq!(drain_ids(&mut s).len(), 8, "uncapped admission fills in one step");
    }

    #[test]
    fn a_cap_still_refills_a_slot_freed_mid_flight() {
        // The cap limits ADMISSION RATE, not refill. A freed slot must still be reusable, or the cap
        // would silently convert continuous batching back into static batching.
        let mut s = Scheduler::new(2).with_admit_cap(1);
        let _a = s.submit(vec![1], 9);
        let _b = s.submit(vec![2], 9);
        assert_eq!(drain_ids(&mut s).len(), 1);
        assert_eq!(drain_ids(&mut s).len(), 2, "second slot fills on the next step");
        let first = s.running()[0].id;
        s.record(first, 0, true);
        let c = s.submit(vec![3], 9);
        assert!(drain_ids(&mut s).contains(&c.0), "the freed slot took the new arrival");
    }

    #[test]
    fn occupancy_reports_whether_batching_is_doing_anything() {
        let mut s = Scheduler::new(4);
        assert_eq!(s.occupancy(), 0.0);
        s.submit(vec![1], 5);
        s.submit(vec![2], 5);
        drain_ids(&mut s);
        assert_eq!(s.occupancy(), 0.5);
    }

    #[test]
    fn recording_against_an_unknown_or_retired_sequence_is_a_no_op() {
        // The engine samples a whole batch and then reports; a sequence cancelled mid-batch must not
        // panic the loop that is still delivering the others.
        let mut s = Scheduler::new(2);
        let a = s.submit(vec![1], 5);
        drain_ids(&mut s);
        s.cancel(a);
        s.record(a, 1, false);            // must not panic
        s.record(SeqId(999), 1, false);   // never existed
        assert!(s.is_idle());
    }
}
