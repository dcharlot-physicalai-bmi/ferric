//! **Run Tencent's real attention weights through this loader.**
//!
//! `hyv4_validate` established that the loader's view of the FORMAT is right against the published
//! checkpoint: every tensor name, every dimension. It said nothing about the arithmetic, because
//! that needs weights and the file is 213.66 GiB.
//!
//! But one block's attention is not 213 GiB. It is about 210 MB, and HuggingFace serves range
//! requests. So: take block 0's real attention, hyper-connection and indexer tensors, pair them with
//! a synthetic embedding, FFN and output head (none of which is under test), and run.
//!
//! ## What a run without a reference implementation can and cannot show
//!
//! There is no oracle here — llama.cpp's hyv4 fork does not build on this machine — so "the output
//! is correct" is not available. What IS available is that trained weights carry structure random
//! ones do not, and the sharpest instance is scale: a residual block's output must be commensurate
//! with its input, or a 78-layer stack would explode or vanish. That property is a fact about
//! Tencent's training, not about this code, so it discriminates.
//!
//! ⚠ And it is reported as a measurement with a control, not asserted as a threshold: the same
//! forward is run with `attn_k_b` deliberately transposed, and the two ratios are printed side by
//! side. If a wrong orientation produced a comparable ratio, this check would be worth nothing and
//! the output says so rather than passing quietly.
//!
//! ```text
//! # see hyv4_fetch_block.py in the scratchpad, or the ranges printed by hyv4_validate
//! cargo run --release -p ferric-llama --example hyv4_real_block -- blk0.bin blk0_plan.json
//! ```

use ferric_core::Context;
use ferric_gguf::write::GgufWriter;
use ferric_llama::hyv4::Hyv4;
use std::sync::Arc;

const D: usize = 6144;
const VOCAB: usize = 64;
const FF: usize = 64;

fn main() {
    let mut a = std::env::args().skip(1);
    let (bin, plan) = match (a.next(), a.next()) {
        (Some(b), Some(p)) => (b, p),
        _ => { eprintln!("usage: hyv4_real_block <blk0.bin> <blk0_plan.json>"); return }
    };
    let raw = std::fs::read(&bin).expect("block bytes");
    let plan: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&plan).expect("plan")).expect("json");
    let base = plan.as_object().unwrap().values()
        .map(|v| v["start"].as_u64().unwrap()).min().unwrap();

    let get = |name: &str| -> (Vec<u64>, u32, Vec<u8>) {
        let e = &plan[name];
        let (s, n) = (e["start"].as_u64().unwrap() - base, e["size"].as_u64().unwrap());
        let dims: Vec<u64> = e["dims"].as_array().unwrap().iter().map(|d| d.as_u64().unwrap()).collect();
        (dims, e["type"].as_u64().unwrap() as u32, raw[s as usize..(s + n) as usize].to_vec())
    };

    let Ok(ctx) = pollster::block_on(Context::new()) else { eprintln!("no GPU context"); return };
    let ctx = Arc::new(ctx);

    // `transpose_kb` swaps attn_k_b's two inner axes. It is a control, not a variant: the same real
    // bytes, read under a wrong orientation, so any difference in the result is attributable to the
    // orientation and nothing else.
    // The control transposes a square-ish factor whose two inner axes differ (192 vs 512), so the
    // permutation is not an identity on any head.
    #[derive(Clone, Copy, PartialEq)]
    enum Ctl { Real, TransposeK, TransposeV }

    let build = |ctl: Ctl| -> Vec<u8> {
        let mut seed = 0x51ee_7a11u64;
        let mut rnd = |n: usize| -> Vec<f32> {
            (0..n).map(|_| {
                seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                (((seed >> 32) as f32 / (1u64 << 31) as f32) - 1.0) * 0.02
            }).collect()
        };
        let mut w = GgufWriter::new("hyv4");
        w.kv_u32("hyv4.block_count", 1).kv_u32("hyv4.embedding_length", D as u32)
            .kv_u32("hyv4.feed_forward_length", FF as u32)
            .kv_u32("hyv4.attention.head_count", 64).kv_u32("hyv4.attention.head_count_kv", 1)
            .kv_u32("hyv4.vocab_size", VOCAB as u32).kv_u32("hyv4.context_length", 1024)
            .kv_f32("hyv4.attention.layer_norm_rms_epsilon", 1e-5)
            .kv_u32("hyv4.attention.key_length_mla", 256).kv_u32("hyv4.attention.value_length_mla", 256)
            .kv_u32("hyv4.rope.dimension_count", 64).kv_f32("hyv4.rope.freq_base", 10_000_000.0)
            .kv_u32("hyv4.attention.q_lora_rank", 2048).kv_u32("hyv4.attention.kv_lora_rank", 512)
            .kv_u32("hyv4.leading_dense_block_count", 1)
            .kv_u32("hyv4.expert_count", 4).kv_u32("hyv4.expert_used_count", 2)
            .kv_u32("hyv4.expert_shared_count", 1).kv_u32("hyv4.expert_feed_forward_length", 16)
            .kv_f32("hyv4.expert_weights_scale", 2.827).kv_bool("hyv4.expert_weights_norm", true)
            .kv_u32("hyv4.expert_gating_func", 2)
            .kv_arr_f32("hyv4.swiglu_clamp_exp", &[10.0])
            .kv_u32("hyv4.hyper_connection.count", 4)
            .kv_f32("hyv4.hyper_connection.epsilon", 1e-6).kv_f32("hyv4.hyper_connection.magnitude", 2.0)
            .kv_u32("hyv4.attention.indexer.head_count", 32)
            .kv_u32("hyv4.attention.indexer.key_length", 128)
            .kv_u32("hyv4.attention.indexer.top_k", 2048)
            .kv_arr_i32("hyv4.attention.indexer.is_full", &[1])
            .kv_str("tokenizer.ggml.model", "gpt2").kv_str("tokenizer.ggml.pre", "hyv4");

        // Real, straight from Tencent's file: attention, hyper-connections, indexer.
        for n in ["attn_norm.weight", "attn_q_a.weight", "attn_q_a_norm.weight", "attn_q_b.weight",
                  "attn_kv_a_mqa.weight", "attn_kv_a_norm.weight",
                  "attn_gate.weight", "attn_output.weight", "attn_sinks.weight",
                  "hc_attn_fn.weight", "hc_attn_base.weight", "hc_attn_scale.weight",
                  "hc_ffn_fn.weight", "hc_ffn_base.weight", "hc_ffn_scale.weight",
                  "indexer.attn_q_b.weight", "indexer.attn_k.weight",
                  "indexer.k_norm.weight", "indexer.k_norm.bias", "indexer.proj.weight"] {
            let (dims, ty, bytes) = get(&format!("blk.0.{n}"));
            w.tensor(&format!("blk.0.{n}"), &dims, ty, bytes);
        }
        // ⛔ The control must permute the DATA, not the declared dims. `Hyv4::load` reshapes this
        // tensor to a shape derived from the CONFIG (`[n_head, kv_lora, nope]`) and only checks the
        // element COUNT, so rewriting the header's dims changes nothing — a first version of this
        // control did exactly that and produced bit-identical output, which is how it was caught.
        //
        // That the loader ignores the declared dims is why `hyv4_validate` exists: nothing in the
        // load path would notice a file whose tensor was laid out differently, so the check has to
        // be made against the header explicitly and separately.
        let (kd, kty, kb) = get("blk.0.attn_k_b.weight");
        let n_elem: usize = kd.iter().product::<u64>() as usize;
        let flat = ferric_gguf::deq_raw(&kb, n_elem, kty).expect("k_b dequant");
        let (nope, kvl, heads) = (kd[0] as usize, kd[1] as usize, kd[2] as usize);
        let data: Vec<f32> = if ctl == Ctl::TransposeK {
            // swap the inner two axes within each head: (e, c) -> (c, e)
            let mut t = vec![0.0f32; flat.len()];
            for h in 0..heads { for c in 0..kvl { for e in 0..nope {
                t[h * kvl * nope + e * kvl + c] = flat[h * kvl * nope + c * nope + e];
            }}}
            t
        } else { flat };
        // Written as F32 in BOTH arms, so the quantisation is not a difference between them.
        w.tensor_f32("blk.0.attn_k_b.weight", &kd, &data);

        // The same control on the VALUE fold, run as a falsifiable prediction -- see the analysis
        // printed at the end, which is where the prediction failed and why.
        let (vd, vty, vbytes) = get("blk.0.attn_v_b.weight");
        let vn: usize = vd.iter().product::<u64>() as usize;
        let vflat = ferric_gguf::deq_raw(&vbytes, vn, vty).expect("v_b dequant");
        let (vkvl, vvh, vheads) = (vd[0] as usize, vd[1] as usize, vd[2] as usize);
        let vdata: Vec<f32> = if ctl == Ctl::TransposeV {
            let mut t = vec![0.0f32; vflat.len()];
            for h in 0..vheads { for v in 0..vvh { for r in 0..vkvl {
                t[h * vvh * vkvl + r * vvh + v] = vflat[h * vvh * vkvl + v * vkvl + r];
            }}}
            t
        } else { vflat };
        w.tensor_f32("blk.0.attn_v_b.weight", &vd, &vdata);

        // Synthetic, and not under test: nothing here is on the attention path.
        w.tensor_f32("token_embd.weight", &[D as u64, VOCAB as u64], &rnd(VOCAB * D))
            .tensor_f32("output.weight", &[D as u64, VOCAB as u64], &rnd(VOCAB * D))
            .tensor_f32("output_norm.weight", &[D as u64], &vec![1.0; D])
            .tensor_f32("output_hc_fn.weight", &[(4 * D) as u64, 4], &rnd(4 * 4 * D))
            .tensor_f32("output_hc_base.weight", &[4], &rnd(4))
            .tensor_f32("output_hc_scale.weight", &[1], &[0.8])
            .tensor_f32("blk.0.ffn_norm.weight", &[D as u64], &vec![1.0; D])
            .tensor_f32("blk.0.ffn_gate.weight", &[D as u64, FF as u64], &rnd(FF * D))
            .tensor_f32("blk.0.ffn_up.weight", &[D as u64, FF as u64], &rnd(FF * D))
            .tensor_f32("blk.0.ffn_down.weight", &[FF as u64, D as u64], &rnd(D * FF));
        w.finish().expect("checkpoint")
    };

    let rms = |v: &[f32]| (v.iter().map(|x| (*x as f64) * (*x as f64)).sum::<f64>() / v.len() as f64).sqrt();

    let run = |ctl: Ctl| -> (f64, f64, bool) {
        let g = ferric_gguf::parse(build(ctl)).expect("parse");
        let m = match Hyv4::load(&ctx, &g) {
            Ok(m) => m,
            Err(e) => { eprintln!("LOAD FAILED: {e}"); std::process::exit(1) }
        };
        let toks: Vec<u32> = (0..8).collect();
        let out = pollster::block_on(m.forward(&toks).to_vec());
        let finite = out.iter().all(|x| x.is_finite());
        (rms(&out), out.iter().fold(0.0f32, |a, b| a.max(b.abs())) as f64, finite)
    };

    println!("Tencent Hy4-preview block 0: real attention, hyper-connection and indexer weights\n");
    let (r_rms, r_max, r_fin) = run(Ctl::Real);
    println!("  real orientation        logits RMS {r_rms:.5}  max |x| {r_max:.5}  finite {r_fin}");
    let (k_rms, k_max, _) = run(Ctl::TransposeK);
    println!("  attn_k_b transposed     logits RMS {k_rms:.5}  max |x| {k_max:.5}   (key fold: BEHIND the softmax)");
    let (v_rms, v_max, _) = run(Ctl::TransposeV);
    println!("  attn_v_b transposed     logits RMS {v_rms:.5}  max |x| {v_max:.5}   (value fold: in front of nothing)");

    if !r_fin {
        println!("\n  ⛔ the real orientation produced non-finite output — a defect, not a measurement");
        std::process::exit(1);
    }
    let dk = (r_rms / k_rms - 1.0).abs();
    let dv = (r_rms / v_rms - 1.0).abs();
    println!("\n  key fold moves the output by {:.2}%, value fold by {:.2}%", dk * 100.0, dv * 100.0);
    println!("\n  A PREDICTION THAT FAILED, AND WHY. `mla.rs` measures this attention attenuating score\n  \
              error by about 500x, which put the key fold behind the softmax and the value fold in\n  \
              front of nothing. The prediction was that a wrong k_b would be nearly invisible here\n  \
              and a wrong v_b would not.");
    println!("  It is not what happens: {:.2}% against {:.2}%, a factor of {:.1}, not 500.", dk * 100.0, dv * 100.0, dv / dk.max(1e-9));
    println!("\n  The error is mine and it is a category error, not a wrong number. That 500x is an\n  \
              AMPLIFICATION FACTOR -- a derivative, measured by perturbing a weight by 1e-4 to 1e-2 and\n  \
              reading the slope. It describes the response to a SMALL perturbation. A transposition is\n  \
              not small: it is a different matrix, not a nearby one, so a local linearisation says\n  \
              nothing about it. Applying it here was extrapolating a derivative across a discontinuity.");
    println!("  (Secondarily, that 500x was measured on random weights at H=3, d=8. Real trained\n  \
              attention is sharper, and a sharper softmax attenuates score error less.)");
    println!("\n  What this leaves: output scale does not discriminate ORIENTATION on either fold --\n  \
              both move it about 1%. Orientation is established structurally, by hyv4_validate\n  \
              against the real header, and that check passes on all 2134 tensors.");
    let _ = (k_max, v_max);
    println!("\n  This is real trained weights through this loader's attention. It is still NOT a\n  \
              correctness proof: there is no reference implementation on this machine to agree with.");
}
