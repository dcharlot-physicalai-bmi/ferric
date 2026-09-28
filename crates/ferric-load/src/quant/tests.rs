//! Unit tests over committed fixtures: ONE module of each real published checkpoint, sliced small
//! (`crates/ferric-llama/examples/refgen/quant_ref.py fixture`), beside what the DEFINING LIBRARY
//! dequantized that very mini-checkpoint to (`ref.bin`; the library and version are in `source.json`).
//! Bit equality, not a tolerance: the readers form the value the library forms at float32.
use super::*;
use std::path::PathBuf;

fn fixture_dir(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/quant").join(name)
}

fn read_ref(p: &Path) -> Vec<(String, Vec<f32>)> {
    let b = std::fs::read(p).unwrap_or_else(|e| panic!("{}: {e} — the fixture is missing; the test must not pass by \
                                                        finding nothing to compare", p.display()));
    let (mut i, mut out) = (0usize, Vec::new());
    while i < b.len() {
        let n = u32::from_le_bytes(b[i..i + 4].try_into().unwrap()) as usize; i += 4;
        let name = String::from_utf8(b[i..i + n].to_vec()).unwrap(); i += n;
        let k = u64::from_le_bytes(b[i..i + 8].try_into().unwrap()) as usize; i += 8;
        out.push((name, b[i..i + 4 * k].chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()));
        i += 4 * k;
    }
    out
}

/// Open the mini checkpoint, find its single quantized module, and compare bits.
fn check(name: &str, want_method: &str, want_act: bool) -> QModule {
    let dir = fixture_dir(name);
    let cfg: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(dir.join("config.json")).unwrap()).unwrap();
    let st = SafeTensors::open(&dir).unwrap();
    let q = Quant::detect(&dir, &cfg, &st).unwrap_or_else(|e| panic!("{name}: {e}")).expect("quantized");
    assert_eq!(q.method, want_method, "{name}");
    let mods: Vec<&QModule> = q.modules().collect();
    assert_eq!(mods.len(), 1, "{name}: fixture should hold one module");
    let m = mods[0].clone();
    assert_eq!(m.spec.act, want_act, "{name}: act-order detection");
    let refs = read_ref(&dir.join("ref.bin"));
    assert_eq!(refs.len(), 1, "{name}: ref.bin should hold one tensor");
    let want = &refs[0].1;
    let got = q.dequant(&st, &format!("{}.weight", m.prefix)).unwrap();
    assert_eq!(got.len(), want.len(), "{name}: {} values vs the library's {}", got.len(), want.len());
    let bad = (0..want.len()).filter(|&i| got[i].to_bits() != want[i].to_bits()).count();
    let first = (0..want.len()).find(|&i| got[i].to_bits() != want[i].to_bits());
    assert_eq!(bad, 0, "{name}: {bad} of {} values differ from the library; first at {:?}: ours {:?} vs {:?}",
               want.len(), first, first.map(|i| got[i]), first.map(|i| want[i]));
    // A reader that returned zeros would match a reference of zeros; these are real weights.
    let nz = want.iter().filter(|v| **v != 0.0).count();
    assert!(nz * 4 > want.len(), "{name}: only {nz} of {} reference values are non-zero", want.len());
    // The presented bytes must be exactly the GQ layout the kernel expects, at the claimed stride.
    let bytes = q.gq_bytes(&st, &format!("{}.weight", m.prefix)).unwrap();
    assert_eq!(bytes.len(), m.out * m.inp / m.spec.group as usize * m.spec.block_bytes(), "{name}: GQ stride");
    m
}

#[test] fn gptq_int4_is_bit_identical_to_gptqmodel() { check("gptq_int4_g128", "gptq", false); }
#[test] fn gptq_int8_is_bit_identical_to_gptqmodel() { check("gptq_int8_g128", "gptq", false); }
#[test] fn gptq_act_order_is_bit_identical_to_gptqmodel() { check("gptq_int4_actorder", "gptq", true); }
#[test] fn gptq_int3_straddling_words_is_bit_identical_to_gptqmodel() { check("gptq_int3", "gptq", false); }
#[test] fn gptq_int2_is_bit_identical_to_gptqmodel() { check("gptq_int2", "gptq", false); }
#[test] fn gptq_v2_zero_points_are_bit_identical_to_gptqmodel() { check("gptq_v2_int4", "gptq", false); }
#[test] fn awq_gemm_is_bit_identical_to_autoawq_dequantize_gemm() { check("awq_gemm", "awq", false); }
#[test] fn compressed_tensors_actorder_group_is_bit_identical_to_ct_0_11() { check("ct_w4a16_actorder_group", "compressed-tensors", true); }
#[test] fn compressed_tensors_asym_int4_is_bit_identical() { check("ct_w4a16_asym", "compressed-tensors", false); }
#[test] fn compressed_tensors_int8_channel_is_bit_identical() { check("ct_w8a16_channel", "compressed-tensors", false); }
#[test] fn compressed_tensors_fp8_channel_is_bit_identical() { check("ct_fp8_channel", "compressed-tensors", false); }
#[test] fn compressed_tensors_fp8_block_is_bit_identical() { check("ct_fp8_block", "compressed-tensors", false); }
#[test] fn compressed_tensors_nvfp4_is_bit_identical() { check("ct_nvfp4", "compressed-tensors", false); }
#[test] fn fp8_block_scaled_is_bit_identical_to_transformers() { check("fp8_block", "fp8", false); }
#[test] fn modelopt_nvfp4_is_bit_identical_to_modelopt() { check("modelopt_nvfp4", "modelopt", false); }

/// The AWQ nibble order is the whole difference between AutoAWQ's decode and an in-order unpack —
/// and GPTQModel 7.5's offline `convert_awq_file` does the in-order one. Checked here without the env
/// switch (tests run in parallel; an env var would leak between them): decode the fixture's first
/// output columns both ways and require them to differ on most values.
#[test]
fn awq_in_order_nibbles_would_be_a_different_weight() {
    let dir = fixture_dir("awq_gemm");
    let st = SafeTensors::open(&dir).unwrap();
    let p = st.names().find(|n| n.ends_with(".qweight")).unwrap().trim_end_matches(".qweight").to_string();
    let (qw, sh) = i32s(&st, &format!("{p}.qweight")).unwrap();
    let wpr = sh[1];
    let (mut same, mut n) = (0, 0);
    for i in 0..sh[0] { for o in 0..wpr * 8 {
        let w = &qw[i * wpr + o / 8..i * wpr + o / 8 + 1];
        same += (bit_code(w, awq::AWQ_REVERSE_ORDER[o % 8], 4) == bit_code(w, o % 8, 4)) as usize;
        n += 1;
    }}
    assert!(same * 3 < n, "in-order and AWQ-order decodes agree on {same}/{n} codes — the order would be invisible");
}

#[test]
fn act_order_maps_must_be_uniform() {
    // group 2 over 6 values: [0,1,2,0,1,2] is uniform; [0,0,0,1,2,2] is not.
    assert!(act_gidx(&[0, 1, 2, 0, 1, 2], 3, 2).is_ok());
    assert!(act_gidx(&[0, 0, 0, 1, 2, 2], 3, 2).unwrap_err().contains("non-uniform"));
    assert!(act_gidx(&[0, 3, 1, 1, 2, 2], 3, 2).unwrap_err().contains("outside"));
}

#[test]
fn bitstreams_cross_word_boundaries() {
    // 3-bit codes 0..31 packed densely; code 10 spans words 0/1 and code 21 spans 1/2.
    let mut w = [0u32; 4];
    for i in 0..32u32 { let bp = i as usize * 3; w[bp >> 5] |= (i & 7) << (bp & 31); if (bp & 31) > 29 { w[(bp >> 5) + 1] |= (i & 7) >> (32 - (bp & 31)); } }
    for i in 0..32usize { assert_eq!(bit_code(&w, i, 3), i as u32 & 7, "code {i}"); }
}
