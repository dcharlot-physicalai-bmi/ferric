//! **What can Ferric prove about a model with no second runtime to check against?**
//!
//! `arch.rs` defines `Status::Verified` as "compared against the reference implementation and
//! matched". `laguna` is stuck one rung below it — not because anything is known to be wrong, but
//! because llama.cpp answers `unknown model architecture: laguna`, so there is nothing to diff.
//! An architecture's confidence ceiling should not be set by another runtime's coverage.
//!
//! This runs the invariants a correct forward pass must satisfy by construction: determinism,
//! prefill/decode equivalence, and prefix causality. All three need only Ferric.
//!
//!   cargo run -q -p ferric-serve --example self_check --release -- <model.gguf>
//!
//! The mutation control is unconditional — `control_ok` is a reported field and `passed()` is false
//! without it, so there is no env var to remember and no green result that skipped the control.
//! (An earlier revision of this header told the reader to set `FERRIC_SELFCHECK_MUTATE=1`; that
//! variable was deleted when the control stopped being optional, and the instruction outlived it —
//! following it would have produced an ordinary pass that read as a verified one.)
fn main() {
    let path = std::env::args().nth(1).expect("usage: self_check <model.gguf>");
    let r = ferric_serve::self_check(&path);
    println!("arch {:<12} tokens={}", r.arch, r.n_tok);
    println!("  deterministic          {}", yn(r.deterministic));
    println!("  prefill == decode      {}", yn(r.prefill_matches_decode));
    println!("  causal (prefix stable) {}", yn(r.causal));
    println!("  bit-identical          {}{}", yn(r.bitwise),
             if r.bitwise { "" } else { "   (expected off a pinned kernel — see below)" });
    println!("  max |Δlogit|           {:e}   over |logit| <= {:.3}   = {:e} relative",
             r.max_abs_diff, r.max_abs_logit, r.rel());
    println!("  mutation control       {}", if r.control_ok { "fired" } else { "DEAD" });
    // Two runs under different env settings that print the SAME fingerprint ran the same
    // arithmetic — whatever the setting claimed to do, it did nothing.
    println!("  logits_fnv             {:016x}", r.logits_fnv);
    println!("\n  {}", if r.passed() { "self-consistent" } else { "SELF-INCONSISTENT" });
    if !r.bitwise && r.passed() {
        // Separate prints, not one continued literal: a `\`-continued string in this tree comes
        // back from the formatter with the next line's indentation baked in as real spaces, which
        // lands in the output.
        println!();
        println!("  Not bit-identical, and that is not a defect: `q2_0_split_k` picks the split-K");
        println!("  kernel at rows<=2 and the flat one above, so a 1-row decode step and a {}-row", r.n_tok);
        println!("  prefill sum the same products in a different order. Pin either kernel and the");
        println!("  difference goes to exactly zero:");
        println!();
        println!("      FERRIC_Q2_0_KERNEL=splitk <self_check> {path}");
    }
    std::process::exit(if r.passed() { 0 } else { 1 });
}

fn yn(b: bool) -> &'static str { if b { "yes" } else { "NO" } }
