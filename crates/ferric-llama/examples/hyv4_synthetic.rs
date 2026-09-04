//! **Run a hyv4 forward pass on a checkpoint this file writes.**
//!
//! Every component of `hyv4` is verified on its own — the hyper-connection closed form and both MLA
//! absorption folds exactly over GF(2⁶¹−1), the sink's softmax against its definition, the indexer's
//! sharing schedule by bounded model checking, the quant formats by Kani plus an interop check
//! against Tencent's published weights. None of that had ever been *composed*: no tensor had passed
//! through `Hyv4::forward`, because the smallest real checkpoint is 213.66 GiB.
//!
//! So write a small one. Two blocks, `d = 32`, four experts — the same architecture at a size that
//! fits, with the shapes and tensor names the real file uses.
//!
//! ⛔ **What this proves and what it cannot.** It proves the WIRING: that every tensor name
//! resolves, that the shapes agree end to end, that the block schedule composes, that a forward
//! terminates and produces finite logits of the right shape. It CANNOT prove fidelity to Tencent's
//! model, because the same conventions that write the file are used to read it back — a transposed
//! convention applied twice cancels. Fidelity needs the real weights, and that needs a bigger
//! machine. These are different claims and this file makes only the first.
//!
//! ```text
//! cargo run --release -p ferric-llama --example hyv4_synthetic
//! ```

use ferric_core::Context;
use ferric_gguf::write::GgufWriter;

use ferric_llama::hyv4::Hyv4;
use std::sync::Arc;

const HC: usize = 4;
const D: usize = 32;
const HEADS: usize = 2;
const QK_NOPE: usize = 8;
const ROPE: usize = 4;
const VH: usize = 8;
const KVL: usize = 12;
const QLORA: usize = 16;
const L: usize = 2;
const VOCAB: usize = 40;
const N_EXPERT: usize = 4;
const EXPERT_FF: usize = 16;
const FF: usize = 24;
const IDX_HEADS: usize = 2;
const IDX_DK: usize = 8;

/// Build a synthetic hyv4 checkpoint. `top_k` is a parameter so the DSA mask can be switched
/// between "selects everything" and "selects two positions" -- the only way to show, without the
/// real weights, that the indexer actually reaches the attention.
fn build(top_k: u32) -> Vec<u8> {
    let mut seed = 0xa5a5_1234u64;
    let mut rnd = |n: usize| -> Vec<f32> {
        (0..n).map(|_| {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            // Two-signed, and small: hyper-connections multiply four streams together and a wide
            // draw saturates the gates, which would test the plumbing on constants.
            (((seed >> 32) as f32 / (1u64 << 31) as f32) - 1.0) * 0.2
        }).collect()
    };

    let qk_head = QK_NOPE + ROPE;
    let mut w = GgufWriter::new("hyv4");
    w.kv_u32("hyv4.block_count", L as u32)
        .kv_u32("hyv4.embedding_length", D as u32)
        .kv_u32("hyv4.feed_forward_length", FF as u32)
        .kv_u32("hyv4.attention.head_count", HEADS as u32)
        .kv_u32("hyv4.attention.head_count_kv", 1)
        .kv_u32("hyv4.vocab_size", VOCAB as u32)
        .kv_u32("hyv4.context_length", 64)
        .kv_f32("hyv4.attention.layer_norm_rms_epsilon", 1e-5)
        .kv_u32("hyv4.attention.key_length_mla", qk_head as u32)
        .kv_u32("hyv4.attention.value_length_mla", VH as u32)
        .kv_u32("hyv4.rope.dimension_count", ROPE as u32)
        .kv_f32("hyv4.rope.freq_base", 10_000_000.0)
        .kv_u32("hyv4.attention.q_lora_rank", QLORA as u32)
        .kv_u32("hyv4.attention.kv_lora_rank", KVL as u32)
        .kv_u32("hyv4.leading_dense_block_count", 1)
        .kv_u32("hyv4.expert_count", N_EXPERT as u32)
        .kv_u32("hyv4.expert_used_count", 2)
        .kv_u32("hyv4.expert_shared_count", 1)
        .kv_u32("hyv4.expert_feed_forward_length", EXPERT_FF as u32)
        .kv_f32("hyv4.expert_weights_scale", 2.827)
        .kv_bool("hyv4.expert_weights_norm", true)
        .kv_u32("hyv4.expert_gating_func", 2)
        .kv_arr_f32("hyv4.swiglu_clamp_exp", &vec![10.0f32; L])
        .kv_u32("hyv4.hyper_connection.count", HC as u32)
        .kv_f32("hyv4.hyper_connection.epsilon", 1e-6)
        .kv_f32("hyv4.hyper_connection.magnitude", 2.0)
        .kv_u32("hyv4.attention.indexer.head_count", IDX_HEADS as u32)
        .kv_u32("hyv4.attention.indexer.key_length", IDX_DK as u32)
        .kv_u32("hyv4.attention.indexer.top_k", top_k)
        // Layer 0 full, layer 1 sharing — the smallest schedule that exercises BOTH branches of the
        // index reuse. A model where every layer were full would never test the sharing path.
        .kv_arr_i32("hyv4.attention.indexer.is_full", &[1, 0])   // I32, as Tencent's file stores it
        .kv_str("tokenizer.ggml.model", "gpt2")
        .kv_str("tokenizer.ggml.pre", "hyv4");

    w.tensor_f32("token_embd.weight", &[D as u64, VOCAB as u64], &rnd(VOCAB * D))
        .tensor_f32("output.weight", &[D as u64, VOCAB as u64], &rnd(VOCAB * D))
        .tensor_f32("output_norm.weight", &[D as u64], &vec![1.0; D])
        .tensor_f32("output_hc_fn.weight", &[(HC * D) as u64, HC as u64], &rnd(HC * HC * D))
        .tensor_f32("output_hc_base.weight", &[HC as u64], &rnd(HC))
        .tensor_f32("output_hc_scale.weight", &[1], &[0.8]);

    for il in 0..L {
        let b = |s: &str| format!("blk.{il}.{s}");
        // ⚠ Dims are in GGUF ne order: ne0 is the FASTEST axis, so a [out, in] matrix is written
        // [in, out]. Writing them the other way round is the transposition class of bug — right
        // element count, wrong arrangement — and it would survive a round trip through this file
        // because the loader reverses whatever is written.
        w.tensor_f32(&b("attn_norm.weight"), &[D as u64], &vec![1.0; D])
            .tensor_f32(&b("ffn_norm.weight"), &[D as u64], &vec![1.0; D])
            .tensor_f32(&b("attn_q_a.weight"), &[D as u64, QLORA as u64], &rnd(QLORA * D))
            .tensor_f32(&b("attn_q_a_norm.weight"), &[QLORA as u64], &vec![1.0; QLORA])
            .tensor_f32(&b("attn_q_b.weight"), &[QLORA as u64, (HEADS * qk_head) as u64], &rnd(HEADS * qk_head * QLORA))
            .tensor_f32(&b("attn_kv_a_mqa.weight"), &[D as u64, (KVL + ROPE) as u64], &rnd((KVL + ROPE) * D))
            .tensor_f32(&b("attn_kv_a_norm.weight"), &[KVL as u64], &vec![1.0; KVL])
            .tensor_f32(&b("attn_k_b.weight"), &[QK_NOPE as u64, KVL as u64, HEADS as u64], &rnd(HEADS * KVL * QK_NOPE))
            .tensor_f32(&b("attn_v_b.weight"), &[KVL as u64, VH as u64, HEADS as u64], &rnd(HEADS * VH * KVL))
            .tensor_f32(&b("attn_gate.weight"), &[D as u64, (HEADS * VH) as u64], &rnd(HEADS * VH * D))
            .tensor_f32(&b("attn_output.weight"), &[(HEADS * VH) as u64, D as u64], &rnd(D * HEADS * VH))
            .tensor_f32(&b("attn_sinks.weight"), &[HEADS as u64], &rnd(HEADS))
            .tensor_f32(&b("hc_attn_fn.weight"), &[(HC * D) as u64, (2 * HC) as u64], &rnd(2 * HC * HC * D))
            .tensor_f32(&b("hc_attn_base.weight"), &[(2 * HC) as u64], &rnd(2 * HC))
            .tensor_f32(&b("hc_attn_scale.weight"), &[2], &[0.7, 1.3])
            .tensor_f32(&b("hc_ffn_fn.weight"), &[(HC * D) as u64, (2 * HC) as u64], &rnd(2 * HC * HC * D))
            .tensor_f32(&b("hc_ffn_base.weight"), &[(2 * HC) as u64], &rnd(2 * HC))
            .tensor_f32(&b("hc_ffn_scale.weight"), &[2], &[0.7, 1.3]);

        if il == 0 {
            w.tensor_f32(&b("indexer.attn_q_b.weight"), &[QLORA as u64, (IDX_HEADS * IDX_DK) as u64], &rnd(IDX_HEADS * IDX_DK * QLORA))
                .tensor_f32(&b("indexer.attn_k.weight"), &[D as u64, IDX_DK as u64], &rnd(IDX_DK * D))
                .tensor_f32(&b("indexer.k_norm.weight"), &[IDX_DK as u64], &vec![1.0; IDX_DK])
                .tensor_f32(&b("indexer.k_norm.bias"), &[IDX_DK as u64], &rnd(IDX_DK))
                .tensor_f32(&b("indexer.proj.weight"), &[D as u64, IDX_HEADS as u64], &rnd(IDX_HEADS * D));
            // Block 0 is the dense one (leading_dense_block_count = 1).
            w.tensor_f32(&b("ffn_gate.weight"), &[D as u64, FF as u64], &rnd(FF * D))
                .tensor_f32(&b("ffn_up.weight"), &[D as u64, FF as u64], &rnd(FF * D))
                .tensor_f32(&b("ffn_down.weight"), &[FF as u64, D as u64], &rnd(D * FF));
        } else {
            w.tensor_f32(&b("ffn_gate_inp.weight"), &[D as u64, N_EXPERT as u64], &rnd(N_EXPERT * D))
                .tensor_f32(&b("exp_probs_b.bias"), &[N_EXPERT as u64], &rnd(N_EXPERT))
                .tensor_f32(&b("ffn_gate_exps.weight"), &[D as u64, EXPERT_FF as u64, N_EXPERT as u64], &rnd(N_EXPERT * EXPERT_FF * D))
                .tensor_f32(&b("ffn_up_exps.weight"), &[D as u64, EXPERT_FF as u64, N_EXPERT as u64], &rnd(N_EXPERT * EXPERT_FF * D))
                .tensor_f32(&b("ffn_down_exps.weight"), &[EXPERT_FF as u64, D as u64, N_EXPERT as u64], &rnd(N_EXPERT * D * EXPERT_FF))
                .tensor_f32(&b("ffn_gate_shexp.weight"), &[D as u64, EXPERT_FF as u64], &rnd(EXPERT_FF * D))
                .tensor_f32(&b("ffn_up_shexp.weight"), &[D as u64, EXPERT_FF as u64], &rnd(EXPERT_FF * D))
                .tensor_f32(&b("ffn_down_shexp.weight"), &[EXPERT_FF as u64, D as u64], &rnd(D * EXPERT_FF));
        }
    }

    w.finish().expect("the synthetic checkpoint must be writable")
}

fn main() {
    let Ok(ctx) = pollster::block_on(Context::new()) else { eprintln!("no GPU context"); return };
    let ctx = Arc::new(ctx);

    let run = |top_k: u32| -> (Vec<f32>, Vec<usize>) {
        let bytes = build(top_k);
        let g = ferric_gguf::parse(bytes).expect("Ferric must read back what it wrote");
        let model = match Hyv4::load(&ctx, &g) {
            Ok(m) => m,
            Err(e) => { eprintln!("LOAD FAILED: {e}"); std::process::exit(1) }
        };
        let tokens: Vec<u32> = vec![3, 11, 7, 29, 1];
        let logits = model.forward(&tokens);
        (pollster::block_on(logits.to_vec()), logits.shape.clone())
    };

    println!("wrote a synthetic hyv4 checkpoint: {} bytes, {L} blocks, d={D}, {N_EXPERT} experts",
             build(64).len());

    // Dense selection: top_k >= seq, so the indexer's mask admits everything and the attention is
    // ordinary causal attention.
    let (dense, shape) = run(64);
    assert_eq!(shape, vec![5, VOCAB], "logits must be [seq, vocab]");
    assert!(dense.iter().all(|x| x.is_finite()), "a forward producing NaN or inf has not run");

    // ── the check with independent meaning ────────────────────────────────────────────────────
    //
    // Sparse selection: top_k = 2, so each query keeps at most two positions and the mask must
    // change the answer for every token that has more than two visible. If the indexer's mask never
    // reached `Mla::forward_masked`, these would be identical -- and every other assertion in this
    // file would still pass, because shape, finiteness and row-distinctness all survive a mask
    // that goes nowhere.
    let (sparse, _) = run(2);
    let moved = dense.iter().zip(&sparse).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
    assert!(moved > 1e-4, "top_k=2 and top_k=64 give the same logits ({moved:.2e}); the DSA mask \
                           is not reaching the attention");
    // Token 0 sees only itself, so no selection can change it. If THAT moved, the mask is being
    // applied non-causally.
    let tok0 = dense[..VOCAB].iter().zip(&sparse[..VOCAB]).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
    assert!(tok0 < 1e-5, "the first token's logits moved under a tighter top_k ({tok0:.2e}); it has \
                          one visible position, so the mask is not respecting causality");

    // ── the regression lock ───────────────────────────────────────────────────────────────────
    //
    // ⚠ This hash is SELF-REFERENTIAL: it was generated from this code, so it cannot say the
    // forward is correct against Tencent's model. What it does is pin what this code computes, so
    // a later edit cannot silently change it. Without it the assertions above pass under a stale
    // residual, a router bias carried into the weight, or the stream collapse replaced by "take
    // stream 0" -- all four were mutation-tested and all four survived until this line existed.
    let h = fnv(&dense);
    const GOLDEN: u64 = 0x29142d075f1beadc;
    if GOLDEN == 0 {
        println!("  logits hash {h:#018x}  <- paste into GOLDEN to lock this in");
    } else {
        assert_eq!(h, GOLDEN, "the forward's output changed; if that was intended, update GOLDEN \
                               and say in the commit what moved and why");
    }

    println!("forward ran: logits {shape:?}, all finite");
    println!("  top_k 64 -> 2 moved logits by {moved:.4}, first token unchanged by {tok0:.2e}");
    for t in 0..5 {
        let r = &dense[t * VOCAB..(t + 1) * VOCAB];
        let arg = r.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).unwrap().0;
        println!("  pos {t} -> argmax {arg:>3}, range [{:+.4}, {:+.4}]",
                 r.iter().cloned().fold(f32::MAX, f32::min), r.iter().cloned().fold(f32::MIN, f32::max));
    }
    println!("\n  This proves the WIRING composes and that the sparse mask reaches the attention.\n  \
              It does not prove fidelity to Tencent's model: the conventions that wrote this file\n  \
              are the ones that read it back.");
}

/// FNV-1a over the logit bits. A hash, not a tolerance: the forward is deterministic on one
/// machine, and any change at all should be seen rather than absorbed.
fn fnv(v: &[f32]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for x in v {
        for b in x.to_bits().to_le_bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
    }
    h
}
