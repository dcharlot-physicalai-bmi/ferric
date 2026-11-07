//! Acceptance receipts for Ferric.
//!
//! Three kinds of statement exist in this repository, and they are not interchangeable:
//!
//! - a **measured** result, produced by running a named command on a named machine
//! - a **target**, the mission bar, which is not an observation
//! - a **historical author** claim, a sentence or an assert already in the tree, cited,
//!   and not re-run by the document that quotes it
//!
//! A receipt has exactly one of those classes. `outcome: pass` is legal only on a measured
//! receipt that carries the provenance the track requires. An empty measured set is an
//! honest state. A green test that invents a joule figure or a logit error is not.
//!
//! The rules live in [`check`]. They are the same rules named by
//! `docs/acceptance/schemas/`. This crate does not run a model, a browser, or a power meter.
//! [`smolvla`] is the Ferromotion action-chunk contract. It checks pins and bytes.
//! It does not execute SmolVLA.

#![forbid(unsafe_code)]

mod check;
mod sha256;
pub mod smolvla;

pub use check::{
    check_corpus, check_fabric_cell, check_issue, check_protocol, check_receipt, check_tolerance,
    repo_root, REQUIRED_FABRIC_CELLS, REQUIRED_PROTOCOL_HEADINGS,
};
pub use sha256::{hex, sha256};
