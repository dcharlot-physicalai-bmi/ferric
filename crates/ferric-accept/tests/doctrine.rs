//! The committed corpus is the foundation, not a result.
//!
//! These tests fail if a document starts claiming a pass, if a historical excerpt
//! is no longer in the file it cites, or if a required fabric cell disappears.

use ferric_accept::{check_corpus, check_issue, check_protocol, check_receipt, repo_root};
use serde_json::json;

#[test]
fn the_committed_corpus_obeys_the_doctrine() {
    let errs = check_corpus(&repo_root());
    assert!(errs.is_empty(), "{}", errs.join("\n"));
}

#[test]
fn a_target_cannot_carry_an_observation() {
    let v = json!({
        "schema_version": "1",
        "receipt_id": "target.sneak",
        "track": "C",
        "evidence_class": "target",
        "status": "registered",
        "outcome": "not_applicable",
        "protocol_id": "C.energy-leadership.v1",
        "claim": "leadership",
        "recorded_on": "2026-11-06",
        "not_a_measurement": true,
        "joules": 1.0
    });
    let e = check_receipt(&v);
    assert!(e.iter().any(|m| m.contains("joules")), "{e:?}");
}

#[test]
fn outcome_pass_on_a_target_is_rejected() {
    let v = json!({
        "schema_version": "1",
        "receipt_id": "target.pass",
        "track": "A",
        "evidence_class": "target",
        "status": "registered",
        "outcome": "pass",
        "protocol_id": "A.full-model-correctness.v1",
        "claim": "done",
        "recorded_on": "2026-11-06",
        "not_a_measurement": true
    });
    let e = check_receipt(&v);
    assert!(e.iter().any(|m| m.contains("only on evidence_class measured")), "{e:?}");
}

#[test]
fn a_measured_pass_without_provenance_is_rejected() {
    let v = json!({
        "schema_version": "1",
        "receipt_id": "measured.bare",
        "track": "A",
        "evidence_class": "measured",
        "status": "executed",
        "outcome": "pass",
        "protocol_id": "A.full-model-correctness.v1",
        "claim": "full model",
        "recorded_on": "2026-11-06",
        "tolerance": {"id": "A.full-model.v1", "preregistered": true, "predicate": "the mission bar"}
    });
    let e = check_receipt(&v);
    assert!(e.len() > 3, "{e:?}");
    assert!(e.iter().any(|m| m.contains("git")), "{e:?}");
}

#[test]
fn an_estimated_joule_figure_cannot_pass() {
    let mut v = measured_shell("C");
    v["energy"] = json!({
        "class": "estimated",
        "gross_joules": 1.0,
        "above_idle_joules": 1.0,
        "boundary_gross": "nameplate",
        "boundary_above_idle": "nameplate"
    });
    v["quality_gate_receipt_id"] = json!("q");
    let e = check_receipt(&v);
    assert!(e.iter().any(|m| m.contains("estimated")), "{e:?}");
}

#[test]
fn lavapipe_cannot_close_a_browser_cell() {
    let mut v = measured_shell("B");
    v["fabric"] = json!({
        "cell_id": "gpu.wgpu.webgpu.browser",
        "profile": "default",
        "kernel": "matmul",
        "dtype": "f32",
        "shape": "64x64x64",
        "driver": {"name": "lavapipe"},
        "satisfies": ["gpu.wgpu.webgpu.browser"]
    });
    v["browser"] = json!({"real": true, "name": "chrome"});
    v["machine"]["adapter"] = json!("llvmpipe");
    let e = check_receipt(&v);
    assert!(e.iter().any(|m| m.contains("software rasterizer")), "{e:?}");
}

#[test]
fn a_track_f_pass_with_an_unpinned_interface_is_rejected() {
    let mut v = measured_shell("F");
    v["task_id"] = json!("f.closed-loop.ferromotion-sin3x.v1");
    v["energy"] = json!({
        "class": "derived",
        "gross_joules": 3.0,
        "above_idle_joules": 2.0,
        "boundary_gross": "system",
        "boundary_above_idle": "accelerator rails minus idle",
        "window_covers_task": true,
        "ferrotherm_aba": {"a1_a2_agree": true}
    });
    v["quality_gate_receipt_id"] = json!("quality.1");
    v["interfaces"] = json!({
        "ferrotherm_metrology": {"revision": "unpinned"},
        "ferromorphic_event_nir": {"revision": "abc1234", "sensor": "event-nir"},
        "ferromotion_dynamics_control": {"revision": "abc1234"}
    });
    v["reference_agreement"] = json!({"met": true});
    let e = check_receipt(&v);
    assert!(e.iter().any(|m| m.contains("ferrotherm_metrology")), "{e:?}");
}

#[test]
fn a_synthetic_signal_does_not_count_as_event_nir() {
    let mut v = measured_shell("F");
    v["task_id"] = json!("f.closed-loop.ferromotion-sin3x.v1");
    v["energy"] = json!({
        "class": "measured",
        "gross_joules": 3.0,
        "above_idle_joules": 2.0,
        "boundary_gross": "system",
        "boundary_above_idle": "system minus idle",
        "window_covers_task": true,
        "ferrotherm_aba": {"a1_a2_agree": true}
    });
    v["quality_gate_receipt_id"] = json!("quality.1");
    v["interfaces"] = json!({
        "ferrotherm_metrology": {"revision": "ferrotherm@1"},
        "ferromorphic_event_nir": {"revision": "ferromorphic@1", "sensor": "synthetic"},
        "ferromotion_dynamics_control": {"revision": "ferromotion@1"}
    });
    v["reference_agreement"] = json!({"met": true});
    let e = check_receipt(&v);
    assert!(e.iter().any(|m| m.contains("synthetic")), "{e:?}");
}

#[test]
fn a_complete_track_a_fixture_is_accepted_and_is_not_a_corpus_result() {
    // This object is built inside the test. It is not written under docs/acceptance/corpus.
    // Accepting it proves the checker has a pass path. It is not a measurement of Ferric.
    let v = measured_shell("A");
    let e = check_receipt(&v);
    assert!(e.is_empty(), "{e:?}");
}

#[test]
fn openie_attribution_is_rejected() {
    let mut v = measured_shell("A");
    v["actor"] = json!({"email": "david@openie.dev", "account": "david"});
    let e = check_receipt(&v);
    assert!(e.iter().any(|m| m.contains("OpenIE")), "{e:?}");
}

#[test]
fn a_historical_claim_without_a_citation_is_rejected() {
    let v = json!({
        "schema_version": "1",
        "receipt_id": "hist.bare",
        "track": "A",
        "evidence_class": "historical_author",
        "status": "recorded",
        "outcome": "not_applicable",
        "protocol_id": "A.full-model-correctness.v1",
        "claim": "something",
        "recorded_on": "2026-11-06"
    });
    let e = check_receipt(&v);
    assert!(e.iter().any(|m| m.contains("not_a_new_measurement")), "{e:?}");
}

#[test]
fn a_protocol_missing_its_tolerance_section_is_rejected() {
    let text = "<!-- ferric-accept-protocol\nid: A.x.v1\ntrack: A\n-->\n## Mission\n\n## Independent reference\n\n## Procedure\ncargo test\n## Receipt\n\n## Withhold\n\n## Downgrade refusal\nnot loosened, looser is refused\n";
    let e = check_protocol(text);
    assert!(e.iter().any(|m| m.contains("Preregistered tolerances")), "{e:?}");
}

#[test]
fn an_issue_that_pretends_to_have_been_opened_is_rejected() {
    let text = r#"<!-- ferric-accept-issue
track: A
title: Full-model correctness against independent references
-->
## Mission
## Acceptance
## Evidence separation
## Dedup
gh issue list --repo dcharlot-physicalai-bmi/ferric --state all --limit 100 returned []
## Identity
cursor[bot] is not dcharlot-physicalai-bmi. openie is forbidden.
https://github.com/dcharlot-physicalai-bmi/ferric/issues/1
"#;
    let e = check_issue(text);
    assert!(e.iter().any(|m| m.contains("did not open")), "{e:?}");
}

fn measured_shell(track: &str) -> serde_json::Value {
    let tol = match track {
        "A" => "A.full-model.v1",
        "B" => "B.fabric-cell.v1",
        "C" => "C.matched-energy.v1",
        "D" => "D.hermetic.v1",
        "E" => "E.leadership-matrix.v1",
        "F" => "F.closed-loop.v1",
        _ => "A.full-model.v1",
    };
    let mut v = json!({
        "schema_version": "1",
        "receipt_id": "fixture.measured.not-a-result",
        "track": track,
        "evidence_class": "measured",
        "status": "executed",
        "outcome": "pass",
        "protocol_id": "A.full-model-correctness.v1",
        "claim": "fixture only",
        "recorded_on": "2026-11-06",
        "git": {
            "repository": "https://github.com/dcharlot-physicalai-bmi/ferric",
            "revision": "0123456789abcdef0123456789abcdef01234567",
            "dirty": false
        },
        "command": ["cargo", "test", "-p", "ferric-accept"],
        "machine": {"os": "linux", "adapter": "none"},
        "comparator": {"name": "independent-fixture", "independent": true},
        "tolerance": {"id": tol, "preregistered": true, "predicate": "mission bar for this track"},
        "artifacts": [{"name": "fixture.log", "sha256": "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"}]
    });
    if track == "A" {
        v["stages"] = json!([
            {"name": "tokenization", "result": "pass"},
            {"name": "import", "result": "pass"},
            {"name": "logits", "result": "pass"},
            {"name": "generation", "result": "pass"},
            {"name": "malformed", "result": "pass"},
            {"name": "resource_bounds", "result": "pass"}
        ]);
        v["quantized_file_used_as_math_proof"] = json!(false);
    }
    v
}
