use ferric_smolvla::{ensure_newline, tokenize};

#[test]
fn newline_task_matches_the_smolvlm_tokenizer() {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/tokenizer.json");
    let (ids, mask) = tokenize(path, &ensure_newline("pick the red cube")).unwrap();
    assert_eq!(&ids[..5], &[18188, 260, 2382, 20636, 198]);
    assert!(ids[5..].iter().all(|id| *id == 2));
    assert_eq!(&mask[..5], &[true, true, true, true, true]);
    assert!(mask[5..].iter().all(|m| !m));
}
