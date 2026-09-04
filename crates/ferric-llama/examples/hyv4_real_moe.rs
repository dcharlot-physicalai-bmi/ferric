//! **Tencent's real block 1 -- attention AND mixture-of-experts -- run at two quant levels and
//! checked against each other.**
//!
//! `hyv4_real_block` ran block 0's real attention. Block 0 is the model's one dense block, so that
//! run never touched an expert: the router, the stacked expert slabs, the shared expert and the
//! `2.827` routed scale were exercised only on synthetic weights. Those slabs are also where the
//! low-bit formats live -- IQ2_XXS and IQ3_XXS in the STQ1_0 build, Q4_K and Q6_K in the Q4_K_M
//! build -- so the decoders proved in `ferric-gguf` had never met Tencent's actual bytes.
//!
//! Block 1 is the first MoE block. Whole, it is 296.7 MB in one build and 288.9 MB in the other,
//! *if* the expert slabs are cut down: a slab is `[cols, rows, 256]` and expert `e` is a contiguous
//! run, so the first four experts are the first four runs. `hyv4.expert_count = 4` then makes a
//! model the loader accepts, with every weight in it real.
//!
//! ## Why two files, and not one
//!
//! There is still no reference implementation on this machine. But this model is published at two
//! quantisations, and that is a free oracle: the same trained tensor, rounded twice by two
//! different quantisers into two formats decoded by two different code paths. Neither arm can
//! borrow the other's bug. Agreement between them is evidence about the *weights and the wiring*,
//! which is what a scale check cannot reach.
//!
//! ⚠ The floor is measured, not assumed. Both arms share an attention stack and a shared expert, so
//! a control that garbles only the routed path cannot drive the cosine to chance -- it can only
//! reach whatever the output looks like with the routed path gone. That is exactly what the
//! `routed scale = 0` arm measures, and every expert-side control is read against *it*, not
//! against zero.
//!
//! ```text
//! # see plan.py / fetch.py in the scratchpad
//! cargo run --release -p ferric-llama --example hyv4_real_moe -- \
//!     blk1_stq.bin blk1_stq_plan.json blk1_q4k.bin blk1_q4k_plan.json
//! ```

use ferric_core::Context;
use ferric_gguf::write::GgufWriter;
use ferric_llama::hyv4::Hyv4;
use std::sync::Arc;

const D: usize = 6144;
const VOCAB: usize = 64;
const NE: usize = 4;   // experts sliced out of the published 256
const USED: usize = 2;

/// One real published build of the model: its bytes and the header plan that locates them.
struct Arm { raw: Vec<u8>, plan: serde_json::Value, name: &'static str }

impl Arm {
    fn open(bin: &str, plan: &str, name: &'static str) -> Arm {
        Arm {
            raw: std::fs::read(bin).unwrap_or_else(|e| panic!("{bin}: {e}")),
            plan: serde_json::from_str(&std::fs::read_to_string(plan).unwrap_or_else(|e| panic!("{plan}: {e}")))
                .expect("plan json"),
            name,
        }
    }
    /// `(dims, ggml type, bytes)` for one tensor, by its name in Tencent's file.
    fn get(&self, n: &str) -> (Vec<u64>, u32, &[u8]) {
        let e = &self.plan[n];
        let (s, len) = (e["local"].as_u64().expect("fetch did not record local offsets") as usize,
                        e["size"].as_u64().unwrap() as usize);
        let dims: Vec<u64> = e["dims"].as_array().unwrap().iter().map(|d| d.as_u64().unwrap()).collect();
        (dims, e["type"].as_u64().unwrap() as u32, &self.raw[s..s + len])
    }
}

/// What is done to one arm before it is run. `Real` is the published bytes, unaltered.
#[derive(Clone, Copy, PartialEq)]
enum Ctl {
    Real,
    /// Routed scale forced to zero: the shared expert and the attention stack, with the routed
    /// path contributing nothing. This is not a bug being injected -- it is the instrument that
    /// says how much of the output the routed path owns.
    NoRouted,
    /// Every expert slab rotated by half an expert. Same bytes, same count, same distribution,
    /// same block alignment -- each "expert" is now the back half of one and the front half of the
    /// next. The stacked-slab stride bug, in the form it would actually ship in.
    HalfExpertStride,
    /// Experts 0..4 paired with router rows 4..8. The router still picks two of four with real
    /// trained logits; it is picking them for the wrong experts.
    RouterShift,
}

fn build(a: &Arm, ctl: Ctl) -> Vec<u8> {
    let mut seed = 0x51ee_7a11u64;
    let mut rnd = |n: usize| -> Vec<f32> {
        (0..n).map(|_| {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (((seed >> 32) as f32 / (1u64 << 31) as f32) - 1.0) * 0.02
        }).collect()
    };
    let mut w = GgufWriter::new("hyv4");
    w.kv_u32("hyv4.block_count", 1).kv_u32("hyv4.embedding_length", D as u32)
        .kv_u32("hyv4.feed_forward_length", 18432)
        .kv_u32("hyv4.attention.head_count", 64).kv_u32("hyv4.attention.head_count_kv", 1)
        .kv_u32("hyv4.vocab_size", VOCAB as u32).kv_u32("hyv4.context_length", 1024)
        .kv_f32("hyv4.attention.layer_norm_rms_epsilon", 1e-5)
        .kv_u32("hyv4.attention.key_length_mla", 256).kv_u32("hyv4.attention.value_length_mla", 256)
        .kv_u32("hyv4.rope.dimension_count", 64).kv_f32("hyv4.rope.freq_base", 10_000_000.0)
        .kv_u32("hyv4.attention.q_lora_rank", 2048).kv_u32("hyv4.attention.kv_lora_rank", 512)
        // ⭐ zero, not one: block 0 of THIS model is Tencent's block 1, and it is a MoE block.
        .kv_u32("hyv4.leading_dense_block_count", 0)
        .kv_u32("hyv4.expert_count", NE as u32).kv_u32("hyv4.expert_used_count", USED as u32)
        .kv_u32("hyv4.expert_shared_count", 1).kv_u32("hyv4.expert_feed_forward_length", 2048)
        .kv_f32("hyv4.expert_weights_scale", if ctl == Ctl::NoRouted { 0.0 } else { 2.827 })
        .kv_bool("hyv4.expert_weights_norm", true)
        .kv_u32("hyv4.expert_gating_func", 2)
        .kv_arr_f32("hyv4.swiglu_clamp_exp", &[10.0])
        .kv_u32("hyv4.hyper_connection.count", 4)
        .kv_f32("hyv4.hyper_connection.epsilon", 1e-6).kv_f32("hyv4.hyper_connection.magnitude", 2.0)
        .kv_u32("hyv4.attention.indexer.head_count", 32)
        .kv_u32("hyv4.attention.indexer.key_length", 128)
        .kv_u32("hyv4.attention.indexer.top_k", 2048)
        // is_full[1] is 1 in the published schedule, so a single full layer is faithful to this block.
        .kv_arr_i32("hyv4.attention.indexer.is_full", &[1])
        .kv_str("tokenizer.ggml.model", "gpt2").kv_str("tokenizer.ggml.pre", "hyv4");

    // Real and unaltered in every arm: attention, hyper-connections, indexer, both norms, and the
    // shared expert. Passed through as the published bytes at the published type -- no dequantise
    // and requantise, so each arm runs its own build's actual numerics.
    for n in ["attn_norm.weight", "attn_q_a.weight", "attn_q_a_norm.weight", "attn_q_b.weight",
              "attn_kv_a_mqa.weight", "attn_kv_a_norm.weight", "attn_k_b.weight", "attn_v_b.weight",
              "attn_gate.weight", "attn_output.weight", "attn_sinks.weight",
              "hc_attn_fn.weight", "hc_attn_base.weight", "hc_attn_scale.weight",
              "hc_ffn_fn.weight", "hc_ffn_base.weight", "hc_ffn_scale.weight",
              "indexer.attn_q_b.weight", "indexer.attn_k.weight",
              "indexer.k_norm.weight", "indexer.k_norm.bias", "indexer.proj.weight",
              "ffn_norm.weight",
              "ffn_gate_shexp.weight", "ffn_up_shexp.weight", "ffn_down_shexp.weight"] {
        let (dims, ty, bytes) = a.get(&format!("blk.1.{n}"));
        w.tensor(&format!("blk.0.{n}"), &dims, ty, bytes.to_vec());
    }

    // The expert slabs. `HalfExpertStride` rotates each slab by half an expert -- a whole number of
    // rows, so every block stays aligned and the decoder parses exactly as before. Nothing about
    // the byte count, the type or the value distribution changes; only which bytes are expert `e`.
    for n in ["ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight"] {
        let (dims, ty, bytes) = a.get(&format!("blk.1.{n}"));
        let per = bytes.len() / NE;
        let data = if ctl == Ctl::HalfExpertStride {
            let half = per / 2;
            assert_eq!(per % 2, 0, "{n}: expert stride not even, cannot rotate by half");
            bytes.iter().cycle().skip(half).take(bytes.len()).copied().collect()
        } else {
            bytes.to_vec()
        };
        w.tensor(&format!("blk.0.{n}"), &dims, ty, data);
    }

    // The router. The published tensor is `[d, 256]`; expert e's row is a contiguous run of d.
    // `RouterShift` takes rows NE..2*NE, so the four experts are scored by four other experts'
    // trained rows. Written F32 in every arm so the slice is the only difference.
    let (rd, rty, rbytes) = a.get("blk.1.ffn_gate_inp.weight");
    let full = ferric_gguf::deq_raw(rbytes, rd.iter().product::<u64>() as usize, rty).expect("router dequant");
    let first = if ctl == Ctl::RouterShift { NE } else { 0 };
    let rows = &full[first * D..(first + NE) * D];
    w.tensor_f32("blk.0.ffn_gate_inp.weight", &[D as u64, NE as u64], rows);

    let (bd, bty, bbytes) = a.get("blk.1.exp_probs_b.bias");
    let bias = ferric_gguf::deq_raw(bbytes, bd.iter().product::<u64>() as usize, bty).expect("bias dequant");
    w.tensor_f32("blk.0.exp_probs_b.bias", &[NE as u64], &bias[first..first + NE]);

    // Synthetic, identical in every arm, and not under test: nothing here is downstream of a
    // difference between the arms except as a fixed linear read-out of the block's output.
    w.tensor_f32("token_embd.weight", &[D as u64, VOCAB as u64], &rnd(VOCAB * D))
        .tensor_f32("output.weight", &[D as u64, VOCAB as u64], &rnd(VOCAB * D))
        .tensor_f32("output_norm.weight", &[D as u64], &vec![1.0; D])
        .tensor_f32("output_hc_fn.weight", &[(4 * D) as u64, 4], &rnd(4 * 4 * D))
        .tensor_f32("output_hc_base.weight", &[4], &rnd(4))
        .tensor_f32("output_hc_scale.weight", &[1], &[0.8]);
    w.finish().expect("checkpoint")
}

fn run(ctx: &Arc<Context>, a: &Arm, ctl: Ctl) -> Vec<f32> {
    let g = ferric_gguf::parse(build(a, ctl)).expect("parse");
    let m = match Hyv4::load(ctx, &g) {
        Ok(m) => m,
        Err(e) => { eprintln!("LOAD FAILED ({}): {e}", a.name); std::process::exit(1) }
    };
    let toks: Vec<u32> = (0..8).collect();
    pollster::block_on(m.forward(&toks).to_vec())
}

fn cos(a: &[f32], b: &[f32]) -> f64 {
    let (mut d, mut na, mut nb) = (0.0f64, 0.0f64, 0.0f64);
    for (x, y) in a.iter().zip(b) { d += *x as f64 * *y as f64; na += (*x as f64).powi(2); nb += (*y as f64).powi(2) }
    if na == 0.0 || nb == 0.0 { return 0.0 }
    d / (na.sqrt() * nb.sqrt())
}
fn rms(v: &[f32]) -> f64 { (v.iter().map(|x| (*x as f64).powi(2)).sum::<f64>() / v.len() as f64).sqrt() }

fn main() {
    let mut ar = std::env::args().skip(1);
    let (sb, sp, qb, qp) = match (ar.next(), ar.next(), ar.next(), ar.next()) {
        (Some(a), Some(b), Some(c), Some(d)) => (a, b, c, d),
        _ => { eprintln!("usage: hyv4_real_moe <stq.bin> <stq_plan.json> <q4k.bin> <q4k_plan.json>"); return }
    };
    let stq = Arm::open(&sb, &sp, "STQ1_0 build");
    let q4k = Arm::open(&qb, &qp, "Q4_K_M build");

    println!("Tencent Hy4-preview block 1 -- the first MoE block -- from two published builds\n");
    // How wide the oracle actually is, read off the two headers rather than asserted: every tensor
    // whose ggml type differs between the builds is one the two arms decode with different code.
    let names: Vec<String> = stq.plan.as_object().unwrap().keys().cloned().collect();
    let mut differ: Vec<(String, u32, u32)> = Vec::new();
    let mut formats = std::collections::BTreeSet::new();
    for n in &names {
        let (_, st, _) = stq.get(n);
        let (_, qt, _) = q4k.get(n);
        formats.insert(st); formats.insert(qt);
        if st != qt { differ.push((n[6..].to_string(), st, qt)) }
    }
    differ.sort();
    println!("  {} of {} tensors are stored in DIFFERENT formats by the two builds, over ggml types {:?}:",
             differ.len(), names.len(), formats.iter().copied().collect::<Vec<u32>>());
    for (n, st, qt) in &differ {
        println!("    {n:<28} type {st:>2}  vs  type {qt:>2}");
    }
    let sliced = stq.plan["blk.1.ffn_gate_exps.weight"]["sliced_from"].as_u64().unwrap_or(0);
    println!("\n  experts: {NE} of the published {sliced}, top-{USED}; every other tensor whole and real\n");

    let Ok(ctx) = pollster::block_on(Context::new()) else { eprintln!("no GPU context"); return };
    let ctx = Arc::new(ctx);

    let ref_q4k = run(&ctx, &q4k, Ctl::Real);          // the oracle arm, never altered
    let real_stq = run(&ctx, &stq, Ctl::Real);
    if !real_stq.iter().chain(&ref_q4k).all(|x| x.is_finite()) {
        println!("  ⛔ a real arm produced non-finite output -- a defect, not a measurement");
        std::process::exit(1);
    }

    let no_routed = run(&ctx, &stq, Ctl::NoRouted);
    let half = run(&ctx, &stq, Ctl::HalfExpertStride);
    let shift = run(&ctx, &stq, Ctl::RouterShift);

    let c_real = cos(&real_stq, &ref_q4k);
    let c_none = cos(&no_routed, &ref_q4k);
    let c_half = cos(&half, &ref_q4k);
    let c_shift = cos(&shift, &ref_q4k);

    println!("  every cosine is against the UNALTERED Q4_K_M arm, which shares no bytes with any of these\n");
    println!("    {:<44} cos {:>8.5}   RMS {:.5}", "STQ1_0, published bytes", c_real, rms(&real_stq));
    println!("    {:<44} cos {:>8.5}   RMS {:.5}", "routed scale 0 (the floor: no routed path)", c_none, rms(&no_routed));
    println!("    {:<44} cos {:>8.5}   RMS {:.5}", "expert slabs rotated by half an expert", c_half, rms(&half));
    println!("    {:<44} cos {:>8.5}   RMS {:.5}", "router rows 4..8 on experts 0..4", c_shift, rms(&shift));

    // How much of the output the routed path owns. Everything an expert-side control can move lives
    // inside this; a control landing at the floor has destroyed all of it, and one landing above it
    // has destroyed some.
    println!("\n  The routed path owns {:.1}% of the distance from the floor to the real output.",
             100.0 * (c_real - c_none) / (1.0 - c_none).max(1e-12));

    let mut bad = Vec::new();
    // ⛔ The ORDERING of these arms is not enough on its own, and finding that out is what this
    // floor is for. Mutating the IQ2 and IQ3 shaders moves every STQ1_0 arm together, so
    // `real > controls` survives a broken decoder: three of four decoder mutations kept the
    // ordering intact while the real cosine fell to 0.83, 0.34 and 0.69. Only the fourth
    // (-0.09) inverted it. So the primary gate is an ABSOLUTE floor on the cross-quant
    // agreement, and it is placed from the measured mutation ladder, not chosen to pass:
    //
    //   pristine                                   0.96671
    //   IQ3 grid pair swapped                      0.68597
    //   IQ2 sub-scale from the index word          0.83199   <- the closest mutation
    //   IQ2 sign stride 7 -> 8 bits                0.34060
    //   IQ3 control words read interleaved        -0.08977
    //
    // CROSS_QUANT_FLOOR sits between the pristine value and the closest mutation, nearer the
    // mutation. It is a regression lock on a deterministic computation, in the same spirit as
    // `hyv4_synthetic`'s golden hash -- not a claim that 0.93 has any physical meaning.
    const CROSS_QUANT_FLOOR: f64 = 0.93;
    if c_real < CROSS_QUANT_FLOOR {
        bad.push("the two published builds no longer agree on this block: a format-specific defect \
                  (one that hits IQ2_XXS/IQ3_XXS but not Q4_K/Q6_K, or vice versa) is the class this catches")
    }
    if c_real <= c_none { bad.push("the published STQ1_0 experts agree with the Q4_K_M arm no better than NO experts at all") }
    if c_half >= c_real { bad.push("rotating every expert slab by half an expert did not hurt -- the slab stride is not being read") }
    if c_shift >= c_real { bad.push("pairing experts with the wrong router rows did not hurt -- routing is not selecting") }
    if bad.is_empty() {
        println!("\n  ✓ Both controls fall below the published bytes, and the published bytes beat the\n  \
                  floor. Two independently-quantised copies of Tencent's experts, decoded by two\n  \
                  different code paths (IQ2_XXS/IQ3_XXS against Q4_K/Q6_K), agree on the output of\n  \
                  this block -- and stop agreeing when the slab stride or the router pairing is wrong.");
        println!("\n  WHAT THIS IS NOT. It is not a correctness proof of hyv4's arithmetic. Both arms run\n  \
                  the SAME forward code, so a wrong-but-consistent formula agrees with itself. What\n  \
                  the two builds independently pin down is the part that differs between them: which\n  \
                  bytes are which weight. The formula is covered by the Kani harnesses and the\n  \
                  synthetic golden hash, not by this.");
        println!("  It is also {NE} experts of {sliced}. Nothing here exercises 256-way routing, an\n  \
                  expert past index {}, or any block but this one.", NE - 1);
        println!("  And a bug applied to BOTH arms is invisible here by construction: reversing the\n  \
                  expert order in the loader, or applying the routed scale before the renormalisation\n  \
                  instead of after, changes the two arms identically. Those belong to the synthetic\n  \
                  golden hash. What this reaches is what the two builds do NOT share: their formats.");
    } else {
        println!();
        for b in bad { println!("  ⛔ {b}") }
        std::process::exit(1);
    }
}
