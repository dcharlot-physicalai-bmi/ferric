//! **Continuous batching** — let a request join the batch the step it is admitted, not the step the
//! previous batch drains.
//!
//! The difference from static batching is a scheduling one, and it is the whole reason modern servers
//! keep GPUs busy. Under static batching a batch is formed, run to completion, and only then is the next
//! formed: a 4-token request arriving behind a 500-token one waits for all 500. Under continuous
//! batching every step re-forms the batch from whatever is runnable, so the short request finishes in
//! roughly its own length regardless of what it queued behind.
//!
//! ## What this schedules, and what it does not
//!
//! Bookkeeping only, like the rest of this crate: which sequences step, how many tokens each contributes,
//! and who gets preempted when blocks run out. It never touches a tensor. That is what lets the
//! interesting failures — a preempted sequence losing tokens, an admission that overruns the block pool,
//! a finished sequence whose blocks are never returned — be tests that run in microseconds on any target,
//! including wasm.
//!
//! ## The three invariants
//!
//! 1. **A request is never lost.** Every admitted request either completes or returns to the queue with
//!    its tokens intact. A preemption that silently drops a sequence looks like a timeout to the caller.
//! 2. **The pool is never overdrawn.** Admission is refused before blocks are exhausted, not after — the
//!    alternative is a half-allocated sequence and a partially-extended block table.
//! 3. **Preemption is recoverable.** A preempted sequence releases its blocks and keeps its tokens, so
//!    restarting it recomputes rather than corrupts. Preempting by *truncating* KV instead would leave a
//!    sequence attending to a prefix of its own history and producing fluent, wrong output.

use crate::PagedKv;
use std::collections::VecDeque;

/// A request waiting for, or holding, capacity.
#[derive(Debug, Clone)]
pub struct Request {
    pub id: u64,
    /// Prompt plus everything generated so far. Kept whole so a preempted request can restart.
    pub tokens: Vec<u32>,
    /// Prompt length — tokens beyond this are generated.
    pub prompt_len: usize,
    /// Stop after this many generated tokens.
    pub max_new: usize,
}

impl Request {
    pub fn new(id: u64, prompt: Vec<u32>, max_new: usize) -> Self {
        let prompt_len = prompt.len();
        Self { id, tokens: prompt, prompt_len, max_new }
    }
    pub fn generated(&self) -> usize { self.tokens.len() - self.prompt_len }
    pub fn is_done(&self) -> bool { self.generated() >= self.max_new }
}

/// A sequence currently holding KV.
#[derive(Debug, Clone)]
pub struct Running {
    pub req: Request,
    pub seq: u64,
    /// Tokens whose KV is already computed. Below `req.tokens.len()` during a chunked prefill.
    pub filled: usize,
}

impl Running {
    /// Tokens still needing prefill before this sequence can decode.
    pub fn pending_prefill(&self) -> usize { self.req.tokens.len() - self.filled }
}

/// What one scheduler step wants the model to run.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Batch {
    /// `(seq, first_token_index, token_count)` — a prefill contributes many, a decode exactly one.
    pub work: Vec<(u64, usize, usize)>,
}

impl Batch {
    pub fn is_empty(&self) -> bool { self.work.is_empty() }
    pub fn sequences(&self) -> usize { self.work.len() }
    pub fn tokens(&self) -> usize { self.work.iter().map(|w| w.2).sum() }
}

/// Why a step did nothing, when it did nothing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Idle {
    /// Nothing queued and nothing running.
    Empty,
    /// Work is queued but no blocks are free and nothing could be preempted to make room. The caller
    /// must wait for a running sequence to finish; spinning here would livelock.
    Starved,
}

/// Continuous-batching scheduler over a paged KV pool.
#[derive(Debug)]
pub struct Scheduler {
    kv: PagedKv,
    waiting: VecDeque<Request>,
    running: Vec<Running>,
    max_seqs: usize,
    /// Cap on tokens per step, so one long prefill cannot stall every decode behind it. This is the
    /// chunked-prefill budget: a 4,000-token prompt is admitted immediately and prefilled across steps
    /// while other sequences keep decoding.
    token_budget: usize,
    pub admitted: u64,
    pub completed: u64,
    pub preempted: u64,
    pub steps: u64,
}

impl Scheduler {
    pub fn new(kv: PagedKv, max_seqs: usize, token_budget: usize) -> Self {
        assert!(max_seqs > 0 && token_budget > 0);
        Self {
            kv,
            waiting: VecDeque::new(),
            running: Vec::new(),
            max_seqs,
            token_budget,
            admitted: 0,
            completed: 0,
            preempted: 0,
            steps: 0,
        }
    }

    pub fn kv(&self) -> &PagedKv { &self.kv }
    pub fn waiting(&self) -> usize { self.waiting.len() }
    pub fn running(&self) -> &[Running] { &self.running }

    pub fn submit(&mut self, r: Request) { self.waiting.push_back(r); }

    /// Blocks needed to hold `tokens` tokens.
    fn blocks_for(&self, tokens: usize) -> usize { tokens.div_ceil(self.kv.block_tokens()) }

    /// Admit whatever fits, then return the work for this step.
    ///
    /// Admission is deliberately *before* the batch is formed, so a request queued this instant can run
    /// this step — that is the whole of "continuous". Returning it a step later would be static batching
    /// with extra bookkeeping.
    pub fn step(&mut self) -> Result<Batch, Idle> {
        self.steps += 1;
        self.admit();

        if self.running.is_empty() {
            return Err(if self.waiting.is_empty() { Idle::Empty } else { Idle::Starved });
        }

        let mut batch = Batch::default();
        let mut budget = self.token_budget;

        // Decodes first: they are one token each and are what latency is measured in. Letting a large
        // prefill consume the budget ahead of them is exactly the head-of-line blocking this exists to
        // remove.
        for r in &self.running {
            if r.pending_prefill() == 0 && budget > 0 {
                batch.work.push((r.seq, r.req.tokens.len(), 1));
                budget -= 1;
            }
        }
        for r in &self.running {
            let pend = r.pending_prefill();
            if pend > 0 && budget > 0 {
                let n = pend.min(budget);
                batch.work.push((r.seq, r.filled, n));
                budget -= n;
            }
        }

        if batch.is_empty() {
            return Err(Idle::Starved);
        }
        Ok(batch)
    }

    /// Move requests from the queue into the running set while blocks and slots allow.
    fn admit(&mut self) {
        while self.running.len() < self.max_seqs {
            let Some(front) = self.waiting.front() else { break };
            let need = self.blocks_for(front.tokens.len());
            if need > self.kv.pool().free_blocks() {
                // Refuse *before* allocating. Half-admitting and unwinding is the version of this that
                // leaves a partially-extended block table behind.
                break;
            }
            let req = self.waiting.pop_front().expect("front() just succeeded");
            let seq = self.kv.new_sequence();
            match self.kv.append(seq, req.tokens.len()) {
                Ok(_) => {
                    self.admitted += 1;
                    self.running.push(Running { req, seq, filled: 0 });
                }
                Err(_) => {
                    // Lost a race with our own accounting; put it back rather than drop it.
                    self.kv.free(seq);
                    self.waiting.push_front(req);
                    break;
                }
            }
        }
    }

    /// Report the result of a step: `(seq, new_token)` for each sequence that decoded.
    ///
    /// Sequences still prefilling advance `filled` instead. Finished sequences release their blocks here
    /// — holding them until a later sweep is how a pool starves while looking half-empty.
    pub fn complete(&mut self, batch: &Batch, produced: &[(u64, u32)]) -> Vec<Request> {
        let mut finished = Vec::new();

        for &(seq, start, n) in &batch.work {
            let Some(r) = self.running.iter_mut().find(|r| r.seq == seq) else { continue };
            if r.pending_prefill() > 0 {
                r.filled = (start + n).min(r.req.tokens.len());
            }
        }

        for &(seq, tok) in produced {
            if let Some(r) = self.running.iter_mut().find(|r| r.seq == seq) {
                r.req.tokens.push(tok);
                r.filled = r.req.tokens.len();
            }
        }

        // Grow the block tables of everything that gained a token, then retire what is done.
        let mut i = 0;
        while i < self.running.len() {
            let need = self.blocks_for(self.running[i].req.tokens.len());
            let have = self.kv.table(self.running[i].seq).map(|t| t.blocks().len()).unwrap_or(0);
            if need > have {
                let seq = self.running[i].seq;
                let extra = self.running[i].req.tokens.len();
                if self.kv.append(seq, extra - self.kv.table(seq).map(|t| t.len()).unwrap_or(0)).is_err() {
                    // Out of blocks mid-flight. Preempt the newest sequence — it has the least invested —
                    // and let this one continue.
                    self.preempt_newest();
                    continue;
                }
            }
            if self.running[i].req.is_done() {
                let r = self.running.remove(i);
                self.kv.free(r.seq);
                self.completed += 1;
                finished.push(r.req);
            } else {
                i += 1;
            }
        }
        finished
    }

    /// Return the most recently admitted sequence to the queue, releasing its blocks.
    ///
    /// Its tokens survive, so restarting recomputes rather than corrupts. Truncating its KV in place
    /// instead would leave it attending to a prefix of its own history — fluent, and wrong.
    fn preempt_newest(&mut self) -> bool {
        let Some(i) = (0..self.running.len()).next_back() else { return false };
        let r = self.running.remove(i);
        self.kv.free(r.seq);
        self.preempted += 1;
        // Front of the queue: it was admitted before everything still waiting, and sending it to the back
        // is how a request under memory pressure starves indefinitely.
        self.waiting.push_front(r.req);
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sched(blocks: usize, max_seqs: usize, budget: usize) -> Scheduler {
        Scheduler::new(PagedKv::new(blocks, 16), max_seqs, budget)
    }

    /// Drive to completion, returning each request's finish step.
    fn drive(s: &mut Scheduler, limit: u64) -> Vec<(u64, u64)> {
        let mut out = Vec::new();
        for step in 0..limit {
            let b = match s.step() {
                Ok(b) => b,
                Err(Idle::Empty) => break,
                Err(Idle::Starved) => panic!("starved at step {step} with {} waiting", s.waiting()),
            };
            // Every sequence that was decoding produces one token.
            let produced: Vec<(u64, u32)> = b
                .work
                .iter()
                .filter(|(seq, start, n)| {
                    *n == 1 && s.running().iter().any(|r| r.seq == *seq && r.filled == *start)
                })
                .map(|(seq, _, _)| (*seq, 7u32))
                .collect();
            for r in s.complete(&b, &produced) {
                out.push((r.id, step));
            }
        }
        out
    }

    #[test]
    fn a_short_request_behind_a_long_one_does_not_wait_for_it() {
        // THE reason continuous batching exists. Under static batching the short request finishes only
        // after the long one; here it should finish in roughly its own length.
        let mut s = sched(256, 8, 64);
        s.submit(Request::new(1, vec![1; 16], 200)); // long
        s.submit(Request::new(2, vec![2; 16], 4));   // short, queued behind it
        let done = drive(&mut s, 2000);

        let long_at = done.iter().find(|(id, _)| *id == 1).expect("long never finished").1;
        let short_at = done.iter().find(|(id, _)| *id == 2).expect("short never finished").1;
        assert!(
            short_at < long_at / 4,
            "short finished at step {short_at}, long at {long_at} — it waited for the long one, \
             which is static batching"
        );
        assert!(short_at < 12, "short request took {short_at} steps for 4 tokens");
    }

    #[test]
    fn a_request_submitted_mid_flight_joins_the_running_batch() {
        // Anti-vacuity for the test above: prove admission actually happens *while* others run, rather
        // than the queue draining before anything long gets going.
        let mut s = sched(256, 8, 64);
        s.submit(Request::new(1, vec![1; 16], 100));
        for _ in 0..5 {
            let b = s.step().unwrap();
            s.complete(&b, &b.work.iter().map(|(q, _, _)| (*q, 7u32)).collect::<Vec<_>>());
        }
        assert_eq!(s.running().len(), 1);

        s.submit(Request::new(2, vec![2; 16], 3));
        let b = s.step().unwrap();
        assert_eq!(b.sequences(), 2, "the new request did not join the in-flight batch");
        assert!(s.running().iter().any(|r| r.req.id == 2));
    }

    #[test]
    fn no_request_is_lost_under_memory_pressure() {
        // Invariant 1. A tiny pool forces preemption; every request must still complete.
        let mut s = sched(12, 8, 32);
        for i in 0..6u64 {
            s.submit(Request::new(i, vec![i as u32; 16], 8));
        }
        let done = drive(&mut s, 5000);
        let mut ids: Vec<u64> = done.iter().map(|(id, _)| *id).collect();
        ids.sort();
        assert_eq!(ids, (0..6).collect::<Vec<_>>(), "a request vanished under pressure");
        assert_eq!(s.waiting(), 0);
        assert!(s.running().is_empty());
    }

    #[test]
    fn the_block_pool_is_never_overdrawn() {
        // Invariant 2. Admission refuses before exhaustion, so the pool never goes negative and no
        // sequence is left half-allocated.
        let mut s = sched(10, 16, 64);
        for i in 0..8u64 {
            s.submit(Request::new(i, vec![i as u32; 32], 6));
        }
        for _ in 0..400 {
            let Ok(b) = s.step() else { break };
            assert!(
                s.kv().pool().used_blocks() <= s.kv().pool().capacity(),
                "pool overdrawn: {} of {}", s.kv().pool().used_blocks(), s.kv().pool().capacity()
            );
            let produced: Vec<(u64, u32)> = b.work.iter().filter(|(_, _, n)| *n == 1)
                .map(|(q, _, _)| (*q, 7u32)).collect();
            s.complete(&b, &produced);
        }
    }

    #[test]
    fn a_preempted_request_keeps_every_token_it_had() {
        // Invariant 3. Preemption must release blocks and retain tokens — a sequence that comes back
        // shorter than it left attends to a prefix of its own history and is fluently wrong.
        let mut s = sched(8, 8, 64);
        s.submit(Request::new(1, vec![1; 16], 40));
        s.submit(Request::new(2, vec![2; 16], 40));
        let mut lengths: std::collections::HashMap<u64, usize> = Default::default();
        for _ in 0..300 {
            let Ok(b) = s.step() else { break };
            for r in s.running() {
                let seen = lengths.entry(r.req.id).or_insert(0);
                assert!(
                    r.req.tokens.len() >= *seen,
                    "request {} came back with {} tokens, had {}", r.req.id, r.req.tokens.len(), *seen
                );
                *seen = r.req.tokens.len();
            }
            let produced: Vec<(u64, u32)> = b.work.iter().filter(|(_, _, n)| *n == 1)
                .map(|(q, _, _)| (*q, 7u32)).collect();
            s.complete(&b, &produced);
        }
    }

    #[test]
    fn finished_sequences_return_their_blocks_immediately() {
        // A pool that starves while looking half-empty is the symptom of retiring lazily.
        let mut s = sched(64, 8, 64);
        let before = s.kv().pool().free_blocks();
        for i in 0..4u64 {
            s.submit(Request::new(i, vec![i as u32; 16], 2));
        }
        drive(&mut s, 500);
        assert_eq!(s.kv().pool().free_blocks(), before, "blocks were not returned on completion");
        assert_eq!(s.completed, 4);
    }

    #[test]
    fn a_long_prefill_does_not_block_other_sequences_decoding() {
        // The token budget is what makes this true: without it one 4k prompt monopolises the step and
        // every other sequence's latency spikes — the exact head-of-line stall chunked prefill removes.
        let mut s = sched(2048, 8, 128);
        s.submit(Request::new(1, vec![1; 16], 50)); // gets going first
        for _ in 0..3 {
            let b = s.step().unwrap();
            s.complete(&b, &b.work.iter().map(|(q, _, _)| (*q, 7u32)).collect::<Vec<_>>());
        }
        s.submit(Request::new(2, vec![2; 4000], 5)); // a huge prompt arrives

        let mut decoded_while_prefilling = 0;
        for _ in 0..60 {
            // Everything may finish inside the window; Empty is the end, not a failure.
            let Ok(b) = s.step() else { break };
            let big_still_prefilling = s.running().iter().any(|r| r.req.id == 2 && r.pending_prefill() > 0);
            if big_still_prefilling && b.work.iter().any(|(q, _, n)| *n == 1 && *q == 1) {
                decoded_while_prefilling += 1;
            }
            let produced: Vec<(u64, u32)> = b.work.iter().filter(|(_, _, n)| *n == 1)
                .map(|(q, _, _)| (*q, 7u32)).collect();
            s.complete(&b, &produced);
        }
        assert!(
            decoded_while_prefilling > 5,
            "sequence 1 decoded on only {decoded_while_prefilling} steps while the 4k prefill ran — \
             the long prompt is blocking the batch"
        );
    }

    #[test]
    fn the_step_never_exceeds_its_token_budget() {
        let mut s = sched(2048, 16, 96);
        for i in 0..10u64 {
            s.submit(Request::new(i, vec![i as u32; 500], 5));
        }
        for _ in 0..200 {
            let Ok(b) = s.step() else { break };
            assert!(b.tokens() <= 96, "step ran {} tokens against a 96 budget", b.tokens());
            let produced: Vec<(u64, u32)> = b.work.iter().filter(|(_, _, n)| *n == 1)
                .map(|(q, _, _)| (*q, 7u32)).collect();
            s.complete(&b, &produced);
        }
    }

    #[test]
    fn an_empty_scheduler_reports_empty_not_starved() {
        // The caller distinguishes these: Empty means shut down, Starved means wait. Conflating them
        // turns an idle server into a spin.
        let mut s = sched(16, 4, 32);
        assert_eq!(s.step(), Err(Idle::Empty));
    }
}
