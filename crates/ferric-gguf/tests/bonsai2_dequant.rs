//! **PrismML Bonsai 2 packings, decoded bit-for-bit like the authors' fork.**
//!
//! Fixture (`tests/fixtures/bonsai2/`): rows 100–101 of `blk.0.attn_qkv.weight` (2 × 5120 values)
//! from each of the three published files — `Ternary-Bonsai-2-27B-PQ2_0.gguf` (sha256 3907dc16…),
//! `-PTQ1_0.gguf` (53107f53…) and the dev repo's `-Q2_0-prism-fork-required.gguf` (4f99aed0…) —
//! as raw block bytes, plus ONE float32 reference: the fork's own `ggml_get_type_traits(t)->to_float`
//! (PrismML-Eng/llama.cpp @ adfffbe, `dumpdeq`), which produced byte-identical output for all three
//! files. Whole-file coverage (every quantized tensor of all three files, hashed) is in
//! `scripts/bonsai2_conformance.sh`'s record; this is the part that needs no 7 GB download.
use ferric_gguf::{deq_raw, prism};

const DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/bonsai2");
const N: usize = 2 * 5120;

fn read(name: &str) -> Vec<u8> { std::fs::read(format!("{DIR}/{name}")).unwrap_or_else(|e| panic!("{name}: {e}")) }
fn reference() -> Vec<f32> {
    read("fork_dequant.blk0_attn_qkv_r100x2.f32").chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect()
}
fn bits_differ(a: &[f32], b: &[f32]) -> usize { a.iter().zip(b).filter(|(x, y)| x.to_bits() != y.to_bits()).count() }

#[test]
fn every_packing_decodes_bit_identically_to_the_forks_dequantizer() {
    let want = reference();
    assert_eq!(want.len(), N);
    // PQ2_0 is decoded by the group-128 Q2_0 path (the parser rewrites file id 142 → 42).
    for (file, ty) in [("pq2_0", 42u32), ("pq2_0", prism::PQ2_0), ("ptq1_0", prism::PTQ1_0), ("q2_0", prism::Q2_0_G64)] {
        let got = deq_raw(&read(&format!("{file}.blk0_attn_qkv_r100x2.raw")), N, ty).unwrap();
        assert_eq!(bits_differ(&got, &want), 0, "{file} as type {ty}: values differ from the fork's dequantizer");
    }
    // Zero is a real value here (ternary), so a decoder returning zeros would not pass: count them.
    let nz = want.iter().filter(|v| **v != 0.0).count();
    assert!(nz > N / 4 && nz < N, "reference has {nz}/{N} non-zeros — not a plausible ternary row");
}

#[test]
fn the_gpu_carriers_are_lossless() {
    // What the GPU runs for PTQ1_0 and group-64 Q2_0: a transcode into the group-128 Q2_0 layout.
    let want = reference();
    for (file, ty) in [("ptq1_0", prism::PTQ1_0), ("q2_0", prism::Q2_0_G64)] {
        let g128 = prism::as_q2_0_g128(ty, &read(&format!("{file}.blk0_attn_qkv_r100x2.raw")), 5120).unwrap().unwrap();
        assert_eq!(g128.len(), N / 128 * 34);
        assert_eq!(bits_differ(&deq_raw(&g128, N, 42).unwrap(), &want), 0, "{file} transcode is not lossless");
    }
    assert!(prism::as_q2_0_g128(prism::PTQ1_0, &read("ptq1_0.blk0_attn_qkv_r100x2.raw"), 5120 + 64).is_err(),
            "a row width that is not whole 128-blocks must refuse");
}

#[test]
fn group_128_read_as_group_64_is_wrong_and_the_reverse_too() {
    // NEGATIVE CONTROL: the legacy-layout confusion PrismML's fork refuses by name. Read the PQ2_0
    // (group-128, 34 B) bytes as mainline group-64 (18 B) blocks — zero-padded to the length a g64
    // reader would walk — and the group-64 bytes as group-128. Both must be grossly wrong, so the
    // bit-exact match above is a statement about the layout, not an accident of it.
    let want = reference();
    let mut pq = read("pq2_0.blk0_attn_qkv_r100x2.raw");
    pq.resize(N / 64 * 18, 0);
    let a = bits_differ(&deq_raw(&pq, N, prism::Q2_0_G64).unwrap(), &want);
    let mut g = read("q2_0.blk0_attn_qkv_r100x2.raw");
    g.truncate(N / 128 * 34);
    let b = bits_differ(&deq_raw(&g, N, 42).unwrap(), &want);
    assert!(a > N / 2 && b > N / 2, "misread layouts should corrupt most values: {a}/{N} and {b}/{N}");
}

#[test]
fn a_file_with_a_hadamard_contract_refuses_rotated_reads_until_a_runtime_unlocks_it() {
    use ferric_gguf::{parse, write::GgufWriter, GgufSource};
    let raw = |f: &str| read(&format!("{f}.blk0_attn_qkv_r100x2.raw"));
    let mut w = GgufWriter::new("qwen35");
    w.kv_u32("prism.hadamard.version", 1)
        .kv_u32("prism.hadamard.block_size", 1024)
        .kv_str("prism.hadamard.transform", "normalized-sylvester-walsh-hadamard")
        .kv_str("prism.hadamard.axis", "input-last-dimension")
        .kv_str("prism.hadamard.sign_mode", "identity")
        .kv_arr_str("prism.hadamard.weight_names", &["blk.0.ffn_up.weight".to_string()])
        .tensor("blk.0.ffn_up.weight", &[5120, 2], prism::PQ2_0, raw("pq2_0"))
        .tensor("blk.0.ffn_gate.weight", &[5120, 2], prism::PTQ1_0, raw("ptq1_0"))
        .tensor_f32("blk.0.attn_norm.weight", &[4], &[1.0, 2.0, 3.0, 4.0]);
    let g = parse(w.finish().unwrap()).unwrap();
    let ty = |n: &str| g.tensors.iter().find(|t| t.name == n).unwrap().ggml_type;
    assert_eq!(ty("blk.0.ffn_up.weight"), 42, "PQ2_0 rides the group-128 Q2_0 id");
    assert_eq!(ty("blk.0.ffn_gate.weight"), prism::PTQ1_0);
    // (File id 42 at 18 B / 64 → Q2_0_G64 is asserted in the crate's own type-42 tests: this writer
    // spells 42 as the group-128 layout and cannot emit a group-64 tensor under it.)
    // The folded weight is locked; the others (not named by the contract) are not.
    let e = g.raw("blk.0.ffn_up.weight").unwrap_err();
    assert!(e.contains("Hadamard"), "refusal must say why: {e}");
    assert!(g.dequant("blk.0.ffn_up.weight").is_err());
    assert!(g.raw("blk.0.attn_norm.weight").is_ok() && g.dequant("blk.0.ffn_gate.weight").is_ok());
    assert!(g.unlock_prism_hadamard());
    assert_eq!(bits_differ(&g.dequant("blk.0.ffn_up.weight").unwrap(), &reference()), 0);
}
