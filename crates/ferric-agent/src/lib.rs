//! **Ferric agent layer** — the model-agnostic smarts that sit above the tokenizer/runtime and below
//! any frontend (the native OpenAI server, or a browser/WASM tab). Two pieces, both pure Rust:
//!
//! - [`guide`] — guided decoding: a byte-level JSON acceptor + JSON-Schema compiler that the sampler
//!   masks logits against, so constrained output is **deterministic and identical across fabrics**.
//! - [`grammar`] — GBNF grammars, parsed and matched as llama.cpp does; [`json_schema`] converts a JSON
//!   Schema to GBNF as llama.cpp does, and [`regex`] a regex or choice list, so each runs on that matcher.
//! - [`tools`] — Hermes/qwen tool-call prompt injection + parsing to OpenAI-shaped `tool_calls`.
//!
//! Nothing here touches the GPU, the tokenizer, or the network, so it compiles to wasm32 unchanged —
//! the same guided decoding runs in a browser tab as on the server.
pub mod guide;
pub mod grammar;
pub mod json_schema;
pub mod regex;
pub mod tools;
