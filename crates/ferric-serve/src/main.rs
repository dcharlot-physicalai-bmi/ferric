//! Thin binary shim.
//!
//! Everything the server does lives in the library beside this file. The crate was binary-only until
//! 2026-08-16, which meant **none** of its logic — request parsing, the generate loop, guided
//! decoding, template detection, SSE framing — could be unit-tested; the only available check was
//! "curl it and look".
//!
//! That was worth fixing before the crate grew a scheduler. Continuous batching adds admission,
//! retirement, per-sequence streaming and cache lifetime, and `ferric_llama::sched` holds its ten
//! invariants under test precisely because it was written as a library. Wiring the tested half into
//! an untestable half would have discarded that.
fn main() { ferric_serve::run() }
