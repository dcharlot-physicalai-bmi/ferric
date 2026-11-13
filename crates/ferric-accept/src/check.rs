//! The acceptance rules.
//!
//! JSON Schema files under `docs/acceptance/schemas/` name the same fields for anything
//! that is not Rust. The cross-field rules (a target cannot carry an observation, a pass
//! cannot use a looser tolerance id, a browser cell cannot be closed by lavapipe) are
//! enforced here, because a schema that only checks types will accept a fabricated pass.

use crate::sha256::{hex, sha256};
use serde_json::{Map, Value};
use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};

/// Every fabric cell a track-B pass has to be able to name.
///
/// The list is the `Device` enum in `crates/ferric-tensor/src/sched.rs` plus the wgpu
/// backends and the accelerator paths the tree already names (CUDA, ROCm, WebNN,
/// Tenstorrent). A cell whose code is absent stays on the list. Dropping it would be
/// a quieter goal, which this crate refuses.
pub const REQUIRED_FABRIC_CELLS: &[&str] = &[
    "gpu.wgpu.metal",
    "gpu.wgpu.vulkan.hardware",
    "gpu.wgpu.dx12",
    "gpu.wgpu.webgpu.browser",
    "gpu.wgpu.vulkan.lavapipe",
    "cpu.host",
    "remote.tcp",
    "browser.worker",
    "npu.execution-provider",
    "metal4.tensor-units",
    "cuda.native",
    "rocm.native",
    "webnn.browser",
    "tenstorrent.host",
];

pub const REQUIRED_PROTOCOL_HEADINGS: &[&str] = &[
    "Mission",
    "Independent reference",
    "Preregistered tolerances",
    "Procedure",
    "Receipt",
    "Withhold",
    "Downgrade refusal",
];

const REQUIRED_ISSUE_HEADINGS: &[&str] = &[
    "Mission",
    "Acceptance",
    "Evidence separation",
    "Dedup",
    "Identity",
];

/// Mission tolerance ids. A measured `pass` must name the one for its track.
/// Historical gates (`2e-3`, `1e-4`, the rounded-file logit band) are real and are
/// cited as historical records. They are not accepted as a substitute for these ids.
const MISSION_TOLERANCE: &[(&str, &str)] = &[
    ("A", "A.full-model.v1"),
    ("B", "B.fabric-cell.v1"),
    ("C", "C.matched-energy.v1"),
    ("D", "D.hermetic.v1"),
    ("E", "E.leadership-matrix.v1"),
    ("F", "F.closed-loop.v1"),
];

const TRACK_A_STAGES: &[&str] = &[
    "tokenization",
    "import",
    "logits",
    "generation",
    "malformed",
    "resource_bounds",
];

const REAL_BROWSER_CELLS: &[&str] = &[
    "gpu.wgpu.webgpu.browser",
    "browser.worker",
    "webnn.browser",
];

pub fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

pub fn check_receipt(v: &Value) -> Vec<String> {
    let mut e = Vec::new();
    let Some(obj) = v.as_object() else {
        return vec!["receipt is not an object".into()];
    };
    if obj.get("schema_version").and_then(|x| x.as_str()) != Some("1") {
        e.push("schema_version must be \"1\"".into());
    }
    require_id(obj, "receipt_id", &mut e);
    let track = require_enum(obj, "track", &["A", "B", "C", "D", "E", "F"], &mut e);
    let class = require_enum(
        obj,
        "evidence_class",
        &["measured", "target", "historical_author"],
        &mut e,
    );
    let status = require_enum(
        obj,
        "status",
        &["registered", "executed", "recorded", "withheld"],
        &mut e,
    );
    let outcome = require_enum(
        obj,
        "outcome",
        &["pass", "fail", "inconclusive", "not_applicable"],
        &mut e,
    );
    match obj.get("protocol_id").and_then(|v| v.as_str()) {
        Some(s) if is_protocol_id(s) => {}
        Some(s) => e.push(format!("`protocol_id` `{s}` must look like A.name.v1")),
        None => e.push("`protocol_id` must be a string".into()),
    }
    require_nonempty(obj, "claim", &mut e);
    require_date(obj, "recorded_on", &mut e);
    reject_openie(obj, &mut e);

    if outcome.as_deref() == Some("pass") && class.as_deref() != Some("measured") {
        e.push("outcome pass is legal only on evidence_class measured".into());
    }

    match class.as_deref() {
        Some("target") => check_target(obj, status.as_deref(), outcome.as_deref(), &mut e),
        Some("historical_author") => {
            check_historical(obj, status.as_deref(), outcome.as_deref(), &mut e)
        }
        Some("measured") => check_measured(
            obj,
            track.as_deref(),
            status.as_deref(),
            outcome.as_deref(),
            &mut e,
        ),
        _ => {}
    }
    e
}

fn check_target(obj: &Map<String, Value>, status: Option<&str>, outcome: Option<&str>, e: &mut Vec<String>) {
    if status != Some("registered") {
        e.push("a target receipt has status registered".into());
    }
    if outcome != Some("not_applicable") {
        e.push("a target receipt has outcome not_applicable".into());
    }
    for key in ["observed", "joules", "max_abs_error", "wer", "tokens_per_second"] {
        if obj.contains_key(key) {
            e.push(format!(
                "target receipt carries `{key}`, which is an observation field"
            ));
        }
    }
    if obj.get("not_a_measurement") != Some(&Value::Bool(true)) {
        e.push("a target receipt sets not_a_measurement true".into());
    }
}

fn check_historical(
    obj: &Map<String, Value>,
    status: Option<&str>,
    outcome: Option<&str>,
    e: &mut Vec<String>,
) {
    if status != Some("recorded") {
        e.push("a historical_author receipt has status recorded".into());
    }
    if outcome != Some("not_applicable") {
        e.push("a historical_author receipt has outcome not_applicable".into());
    }
    if obj.get("not_a_new_measurement") != Some(&Value::Bool(true)) {
        e.push("a historical_author receipt sets not_a_new_measurement true".into());
    }
    let Some(src) = obj.get("source").and_then(|s| s.as_object()) else {
        e.push("historical_author receipt needs a source object".into());
        return;
    };
    require_nonempty(src, "path", e);
    require_nonempty(src, "excerpt", e);
    require_enum(src, "kind", &["prose", "in_tree_gate", "audit"], e);
    require_nonempty(src, "git_commit", e);
    if let Some(c) = src.get("git_commit").and_then(|v| v.as_str())
        && !is_hex(c, 7, 40) {
            e.push("source.git_commit is 7 to 40 lowercase hex characters".into());
        }
    require_date(src, "date", e);
    require_nonempty(src, "author_name", e);
    let email = src.get("author_email").and_then(|v| v.as_str()).unwrap_or("");
    if email.is_empty() {
        e.push("source.author_email is required".into());
    } else if !author_is_physicalai(email) {
        e.push(format!(
            "source.author_email `{email}` is not the physicalai-bmi attribution recorded in git"
        ));
    }
}

fn check_measured(
    obj: &Map<String, Value>,
    track: Option<&str>,
    status: Option<&str>,
    outcome: Option<&str>,
    e: &mut Vec<String>,
) {
    if !matches!(status, Some("executed") | Some("withheld")) {
        e.push("a measured receipt has status executed or withheld".into());
    }
    let git = obj.get("git").and_then(|g| g.as_object());
    match git {
        None => e.push("measured receipt needs git".into()),
        Some(g) => {
            if g.get("repository").and_then(|v| v.as_str())
                != Some("https://github.com/dcharlot-physicalai-bmi/ferric")
            {
                e.push("git.repository must be the dcharlot-physicalai-bmi/ferric repository".into());
            }
            match g.get("revision").and_then(|v| v.as_str()) {
                Some(r) if is_hex(r, 40, 40) => {}
                _ => e.push("git.revision must be 40 lowercase hex characters".into()),
            }
            if g.get("dirty") != Some(&Value::Bool(false)) {
                e.push("git.dirty must be false on a measured receipt".into());
            }
        }
    }
    match obj.get("command").and_then(|c| c.as_array()) {
        Some(c) if !c.is_empty() && c.iter().all(|x| x.as_str().is_some_and(|s| !s.is_empty())) => {}
        _ => e.push("command must be a non-empty array of strings".into()),
    }
    require_nonempty_path(obj, &["machine", "os"], e);
    require_nonempty_path(obj, &["machine", "adapter"], e);
    require_nonempty_path(obj, &["comparator", "name"], e);
    if obj
        .get("comparator")
        .and_then(|c| c.get("independent"))
        != Some(&Value::Bool(true))
    {
        e.push("comparator.independent must be true".into());
    }
    let tol_id = obj
        .get("tolerance")
        .and_then(|t| t.get("id"))
        .and_then(|v| v.as_str())
        .unwrap_or("");
    if tol_id.is_empty() {
        e.push("tolerance.id is required".into());
    }
    if obj
        .get("tolerance")
        .and_then(|t| t.get("preregistered"))
        != Some(&Value::Bool(true))
    {
        e.push("tolerance.preregistered must be true".into());
    }
    require_nonempty_path(obj, &["tolerance", "predicate"], e);

    if outcome != Some("pass") {
        return;
    }
    if status != Some("executed") {
        e.push("outcome pass requires status executed".into());
    }
    let Some(track) = track else { return };
    let expected = MISSION_TOLERANCE
        .iter()
        .find(|(t, _)| *t == track)
        .map(|(_, id)| *id);
    if expected != Some(tol_id) {
        e.push(format!(
            "track {track} pass must use tolerance id {}; `{tol_id}` is not that bar",
            expected.unwrap_or("?")
        ));
    }
    match obj.get("artifacts").and_then(|a| a.as_array()) {
        Some(a) if !a.is_empty() => {
            for (i, art) in a.iter().enumerate() {
                let Some(o) = art.as_object() else {
                    e.push(format!("artifacts[{i}] is not an object"));
                    continue;
                };
                require_nonempty(o, "name", e);
                match o.get("sha256").and_then(|v| v.as_str()) {
                    Some(s) if is_hex(s, 64, 64) => {}
                    _ => e.push(format!("artifacts[{i}].sha256 must be 64 lowercase hex characters")),
                }
            }
        }
        _ => e.push("outcome pass requires at least one artifact with a sha256".into()),
    }
    match track {
        "A" => check_track_a_pass(obj, e),
        "B" => check_track_b_pass(obj, e),
        "C" => check_track_c_pass(obj, e),
        "D" => check_track_d_pass(obj, e),
        "E" => check_track_e_pass(obj, e),
        "F" => check_track_f_pass(obj, e),
        _ => {}
    }
}

fn check_track_a_pass(obj: &Map<String, Value>, e: &mut Vec<String>) {
    let Some(stages) = obj.get("stages").and_then(|s| s.as_array()) else {
        e.push("track A pass lists stages for tokenization, import, logits, generation, malformed, resource_bounds".into());
        return;
    };
    let mut seen = BTreeSet::new();
    for st in stages {
        let Some(o) = st.as_object() else {
            e.push("a stage is an object".into());
            continue;
        };
        let name = o.get("name").and_then(|v| v.as_str()).unwrap_or("");
        seen.insert(name.to_string());
        if o.get("result").and_then(|v| v.as_str()) != Some("pass") {
            e.push(format!("track A stage `{name}` must itself be pass"));
        }
    }
    for name in TRACK_A_STAGES {
        if !seen.contains(*name) {
            e.push(format!("track A pass is missing stage `{name}`"));
        }
    }
    if obj
        .get("quantized_file_used_as_math_proof")
        .and_then(|v| v.as_bool())
        == Some(true)
    {
        e.push("a quantized file cannot verify the math (lm_conformance.sh exits 2 on one)".into());
    }
}

fn check_track_b_pass(obj: &Map<String, Value>, e: &mut Vec<String>) {
    let Some(fab) = obj.get("fabric").and_then(|f| f.as_object()) else {
        e.push("track B pass needs a fabric object".into());
        return;
    };
    let cell = fab.get("cell_id").and_then(|v| v.as_str()).unwrap_or("");
    if !REQUIRED_FABRIC_CELLS.contains(&cell) {
        e.push(format!("fabric.cell_id `{cell}` is not a required cell"));
    }
    for key in ["profile", "kernel", "dtype", "shape"] {
        if fab.get(key).and_then(|v| v.as_str()).is_none_or(|s| s.is_empty()) {
            e.push(format!("fabric.{key} must be a non-empty string"));
        }
    }
    if fab
        .get("driver")
        .and_then(|d| d.get("name"))
        .and_then(|v| v.as_str())
        .is_none_or(|s| s.is_empty())
    {
        e.push("fabric.driver.name is required".into());
    }
    let satisfies = fab.get("satisfies").and_then(|v| v.as_array());
    match satisfies {
        Some(ids) if ids.len() == 1 && ids[0].as_str() == Some(cell) => {}
        _ => e.push("fabric.satisfies must be exactly [cell_id]; one receipt does not close another cell".into()),
    }
    if REAL_BROWSER_CELLS.contains(&cell) {
        let real = obj.get("browser").and_then(|b| b.get("real"));
        let name = obj.get("browser").and_then(|b| b.get("name")).and_then(|v| v.as_str()).unwrap_or("");
        if real != Some(&Value::Bool(true)) || name.is_empty() {
            e.push(format!("cell {cell} requires browser.real true and browser.name"));
        }
        let adapter = obj.get("machine").and_then(|m| m.get("adapter")).and_then(|v| v.as_str()).unwrap_or("");
        let driver = fab.get("driver").and_then(|d| d.get("name")).and_then(|v| v.as_str()).unwrap_or("");
        let blob = format!("{adapter} {driver}").to_ascii_lowercase();
        if blob.contains("lavapipe") || blob.contains("llvmpipe") || blob.contains("swiftshader") {
            e.push("a software rasterizer cannot close a real-browser cell".into());
        }
    }
    if cell == "gpu.wgpu.vulkan.lavapipe" {
        let blob = format!(
            "{} {}",
            obj.get("machine").and_then(|m| m.get("adapter")).and_then(|v| v.as_str()).unwrap_or(""),
            fab.get("driver").and_then(|d| d.get("name")).and_then(|v| v.as_str()).unwrap_or("")
        )
        .to_ascii_lowercase();
        if !blob.contains("lavapipe") {
            e.push("the lavapipe cell is only the software Vulkan stand-in; its adapter or driver must say lavapipe".into());
        }
    }
    if cell == "npu.execution-provider"
        && fab.get("compute_plan_receipt").and_then(|v| v.as_bool()) != Some(true)
    {
        e.push("an NPU pass requires fabric.compute_plan_receipt true; the device is not faked".into());
    }
    if cell == "metal4.tensor-units" && fab.get("dtype").and_then(|v| v.as_str()) == Some("f32") {
        e.push("metal4 tensor units take fp16 inputs by contract; dtype f32 would claim the WGSL metal cell".into());
    }
}

fn check_track_c_pass(obj: &Map<String, Value>, e: &mut Vec<String>) {
    check_energy_pass(obj, e, false);
    check_fabric_energy(obj, e);
}

/// Per-fabric energy methodology, adapted from Zeus v0.16.0 and the ML.ENERGY
/// Benchmark (NeurIPS 2025 Datasets and Benchmarks). A passing receipt is a
/// shape check. It does not assert that any fabric was run.
fn check_fabric_energy(obj: &Map<String, Value>, e: &mut Vec<String>) {
    let Some(fe) = obj.get("fabric_energy").and_then(|v| v.as_object()) else {
        e.push("track C pass needs fabric_energy; one cell, one workload, one instrument".into());
        return;
    };
    let cell = fe.get("cell_id").and_then(|v| v.as_str()).unwrap_or("");
    if !REQUIRED_FABRIC_CELLS.contains(&cell) {
        e.push(format!("fabric_energy.cell_id `{cell}` is not a required fabric cell"));
    }
    let workload = require_enum(fe, "workload", &["prefill", "decode", "training"], e);
    let instrument = fe.get("instrument").and_then(|v| v.as_str()).unwrap_or("");
    match instrument {
        "hardware_counter" | "sampled_rail" | "wall_meter" => {}
        "model_estimate" => e.push(
            "instrument model_estimate cannot pass; TDP arithmetic and Apple IOReport energy-model counters are not a claim (Zeus documents IOReport as model-based, 1 mJ)"
                .into(),
        ),
        _ => e.push(
            "fabric_energy.instrument must be hardware_counter, sampled_rail, or wall_meter".into(),
        ),
    }
    require_nonempty(fe, "counter_or_meter", e);
    if fe.get("copied_leaderboard_joules_as_own") != Some(&Value::Bool(false)) {
        e.push(
            "copied_leaderboard_joules_as_own must be false; an ML.ENERGY leaderboard row is not a Ferric measurement"
                .into(),
        );
    }
    if fe.get("zeus_is_production_dependency") != Some(&Value::Bool(false)) {
        e.push(
            "zeus_is_production_dependency must be false; Zeus is an external cross-check, not the Ferric meter"
                .into(),
        );
    }
    if fe.get("sync_before_read") != Some(&Value::Bool(true)) {
        e.push("sync_before_read must be true; the counter is read after the device finishes the window".into());
    }
    require_nonempty(fe, "sync_method", e);
    if fe.get("warmup_outside_window") != Some(&Value::Bool(true)) {
        e.push("warmup_outside_window must be true; ramp and warmup are not part of the measured window".into());
    }
    if fe.get("subtraction_recorded") != Some(&Value::Bool(true)) {
        e.push("subtraction_recorded must be true; above-idle is gross minus the same-domain idle, not a second name for gross".into());
    }
    let idle = fe.get("idle_joules").and_then(|v| v.as_f64());
    let gross = obj.get("energy").and_then(|v| v.get("gross_joules")).and_then(|v| v.as_f64());
    let above = obj.get("energy").and_then(|v| v.get("above_idle_joules")).and_then(|v| v.as_f64());
    match (gross, above, idle) {
        (Some(g), Some(a), Some(i)) if g.is_finite() && a.is_finite() && i.is_finite() => {
            if !joules_identity(g, a, i) {
                e.push(
                    "gross_joules must equal above_idle_joules + idle_joules within 0.001 J or 0.1%; the two boundaries are not interchangeable"
                        .into(),
                );
            }
        }
        _ => e.push("fabric_energy.idle_joules must be a finite number alongside gross and above-idle".into()),
    }
    let domain = fe.get("domain").and_then(|v| v.as_str()).unwrap_or("");
    match domain {
        "gpu_board" | "host_package" | "dram" | "soc_subsystem" | "host_rail" | "wall" => {}
        _ => e.push(
            "fabric_energy.domain must be gpu_board, host_package, dram, soc_subsystem, host_rail, or wall"
                .into(),
        ),
    }
    if instrument == "hardware_counter" {
        if fe.get("vendor_counter_is_not_wall") != Some(&Value::Bool(true)) {
            e.push("a hardware counter pass sets vendor_counter_is_not_wall true".into());
        }
        if domain == "wall" {
            e.push("a hardware counter domain cannot be wall; that is a plug or rail meter".into());
        }
        let boundary = obj
            .get("energy")
            .and_then(|v| v.get("boundary_gross"))
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_ascii_lowercase();
        if boundary.contains("wall") || boundary.contains("outlet") || boundary.contains("plug") {
            e.push("boundary_gross of a hardware counter must not name a wall outlet".into());
        }
        match fe.get("counter_delta_joules").and_then(|v| v.as_f64()) {
            Some(d) if d.is_finite() && gross.is_some_and(|g| joules_close(d, g)) => {}
            Some(_) => e.push(
                "counter_delta_joules must match gross_joules; a Zeus-style window delta includes static draw and is the gross of that domain"
                    .into(),
            ),
            None => e.push("hardware_counter requires counter_delta_joules".into()),
        }
    }
    if instrument == "wall_meter" {
        if domain != "wall" {
            e.push("wall_meter domain must be wall".into());
        }
        require_nonempty_path(fe, &["external_meter", "name"], e);
        if fe.get("cell_was_the_load") != Some(&Value::Bool(true)) {
            e.push("a wall-meter pass sets cell_was_the_load true; the plug does not know which device ran".into());
        }
    }
    if instrument == "sampled_rail" {
        match fe.get("sample_period_s").and_then(|v| v.as_f64()) {
            Some(p) if p.is_finite() && p > 0.0 => {}
            _ => e.push("sampled_rail requires sample_period_s > 0".into()),
        }
        match fe.get("sample_count").and_then(|v| v.as_u64()) {
            Some(n) if n >= 8 => {}
            _ => e.push(
                "sampled_rail requires sample_count >= 8; scripts/prefill_joules.sh refuses a window with fewer than 8 samples"
                    .into(),
            ),
        }
    }
    match cell {
        "gpu.wgpu.vulkan.lavapipe" => {
            if fe.get("claims_gpu_energy") != Some(&Value::Bool(false)) {
                e.push("lavapipe has no device energy counter; claims_gpu_energy must be false".into());
            }
            if instrument == "hardware_counter" && domain == "gpu_board" {
                e.push("lavapipe cannot report a GPU-board hardware counter".into());
            }
        }
        "gpu.wgpu.webgpu.browser" | "browser.worker" | "webnn.browser" => {
            if instrument == "hardware_counter" {
                e.push(format!(
                    "{cell} has no in-browser energy counter in the Zeus v0.16.0 device list; use a host rail or a wall meter and say the browser process is inside it"
                ));
            }
        }
        "rocm.native" if instrument == "hardware_counter" => {
            if fe.get("rocm_ge_6_3") != Some(&Value::Bool(true)) {
                e.push(
                    "an AMD energy-count pass requires rocm_ge_6_3 true; Zeus documents wrong AMDSMI values before ROCm 6.3"
                        .into(),
                );
            }
            require_nonempty(fe, "stack_version", e);
        }
        "cuda.native" if instrument == "hardware_counter" => {
            require_nonempty(fe, "stack_version", e);
            let api = fe.get("counter_or_meter").and_then(|v| v.as_str()).unwrap_or("");
            if !api.contains("nvmlDeviceGetTotalEnergyConsumption") && !api.contains("nvmlDeviceGetPowerUsage")
            {
                e.push(
                    "cuda.native hardware_counter names nvmlDeviceGetTotalEnergyConsumption (Volta or newer) or nvmlDeviceGetPowerUsage (polled)"
                        .into(),
                );
            }
        }
        "tenstorrent.host" => {
            if fe.get("zeus_has_no_device_binding") != Some(&Value::Bool(true)) {
                e.push("tenstorrent has no Zeus device binding; zeus_has_no_device_binding must be true".into());
            }
            if instrument == "hardware_counter" {
                e.push("tenstorrent has no Zeus hardware counter; a pass uses a rail or wall meter".into());
            }
        }
        _ => {}
    }
    let meter_name = fe.get("counter_or_meter").and_then(|v| v.as_str()).unwrap_or("").to_ascii_lowercase();
    if meter_name.contains("ioreport") || meter_name.contains("energy model") {
        e.push(
            "Apple IOReport Energy Model figures are model-based estimates in the Zeus docs; they cannot be the counter_or_meter of a pass"
                .into(),
        );
    }
    let Some(wp) = fe.get("workload_parity").and_then(|v| v.as_object()) else {
        e.push("workload_parity is required: checkpoint, prompts, batch, and token accounting".into());
        return;
    };
    match wp.get("checkpoint_sha256").and_then(|v| v.as_str()) {
        Some(s) if is_hex(s, 64, 64) => {}
        _ => e.push("workload_parity.checkpoint_sha256 must be 64 lowercase hex characters".into()),
    }
    require_nonempty(wp, "prompt_set_id", e);
    require_nonempty(wp, "batch_or_concurrency", e);
    require_enum(
        wp,
        "token_accounting",
        &["output_tokens", "prefill_tokens", "train_steps"],
        e,
    );
    let accounting = fe.get("accounting").and_then(|v| v.as_str()).unwrap_or("");
    match accounting {
        "steady_state_service" => check_steady_state(fe, e),
        "repeated_window" => check_repeated_window(fe, e),
        "whole_batch" => check_whole_batch(fe, e),
        _ => e.push(
            "fabric_energy.accounting must be steady_state_service, repeated_window, or whole_batch".into(),
        ),
    }
    let _ = workload;
    let closes = fe.get("closes_track_c") == Some(&Value::Bool(true));
    if closes {
        let loads = fe.get("workloads").and_then(|v| v.as_array());
        for name in ["prefill", "decode", "training"] {
            if loads.is_none_or(|a| !a.iter().any(|x| x.as_str() == Some(name))) {
                e.push(format!("closes_track_c requires workloads to include {name}"));
            }
        }
        let pareto = fe.get("pareto").and_then(|v| v.as_object());
        match pareto {
            Some(p) => {
                require_nonempty(p, "latency_target", e);
                if p.get("min_energy_meets_latency") != Some(&Value::Bool(true)) {
                    e.push("pareto.min_energy_meets_latency must be true to close track C".into());
                }
                if p.get("single_best_case_only") != Some(&Value::Bool(false)) {
                    e.push("a track C close is a latency-constrained minimum, not a single unlabeled best case".into());
                }
            }
            None => e.push("closes_track_c requires a pareto object".into()),
        }
        if obj
            .get("energy")
            .and_then(|v| v.get("ferrotherm_aba"))
            .and_then(|v| v.get("a1_a2_agree"))
            != Some(&Value::Bool(true))
        {
            e.push("closing track C still requires ferrotherm A/B/A agreement".into());
        }
    } else if fe.get("closes_track_c") != Some(&Value::Bool(false)) {
        e.push("closes_track_c must be true or false; a single cell does not close the track".into());
    }
}

fn check_steady_state(fe: &Map<String, Value>, e: &mut Vec<String>) {
    if fe.get("steady_state_excludes_ramp") != Some(&Value::Bool(true)) {
        e.push("steady_state_service excludes ramp-up and ramp-down (ML.ENERGY Benchmark §3.2)".into());
    }
    if fe.get("batch_saturated") != Some(&Value::Bool(true)) {
        e.push("steady_state_service requires batch_saturated true at the configured maximum batch".into());
    }
    let tokens = fe.get("output_tokens").and_then(|v| v.as_f64());
    let per = fe.get("energy_per_output_token_joules").and_then(|v| v.as_f64());
    let delta = fe
        .get("counter_delta_joules")
        .and_then(|v| v.as_f64())
        .or_else(|| fe.get("window_joules").and_then(|v| v.as_f64()));
    match (tokens, per, delta) {
        (Some(t), Some(p), Some(d)) if t > 0.0 && t.is_finite() && p.is_finite() && d.is_finite() => {
            if !joules_close(p * t, d) {
                e.push(
                    "energy_per_output_token_joules * output_tokens must match the steady-window joules"
                        .into(),
                );
            }
        }
        _ => e.push(
            "steady_state_service requires output_tokens > 0, energy_per_output_token_joules, and counter_delta_joules or window_joules"
                .into(),
        ),
    }
    check_stability(fe, e);
}

fn check_repeated_window(fe: &Map<String, Value>, e: &mut Vec<String>) {
    match fe.get("n_trials").and_then(|v| v.as_u64()) {
        Some(n) if n >= 10 => {}
        _ => e.push(
            "repeated_window requires n_trials >= 10, the Zeus profile default (zeus.profile, retrieved 2026-11-07)"
                .into(),
        ),
    }
    match fe.get("measurement_duration_s").and_then(|v| v.as_f64()) {
        Some(s) if s.is_finite() && s > 0.0 => {}
        _ => e.push("repeated_window requires the measurement_duration_s actually used".into()),
    }
    match fe.get("cooldown_duration_s").and_then(|v| v.as_f64()) {
        Some(s) if s.is_finite() && s >= 0.0 => {}
        _ => e.push("repeated_window requires the cooldown_duration_s actually used".into()),
    }
    check_stability(fe, e);
}

fn check_whole_batch(fe: &Map<String, Value>, e: &mut Vec<String>) {
    let batch = fe.get("batch_size").and_then(|v| v.as_u64());
    let per = fe.get("energy_per_request_joules").and_then(|v| v.as_f64());
    let window = fe.get("window_joules").and_then(|v| v.as_f64()).or_else(|| {
        fe.get("counter_delta_joules").and_then(|v| v.as_f64())
    });
    match (batch, per, window) {
        (Some(b), Some(p), Some(w)) if b >= 1 && p.is_finite() && w.is_finite() => {
            if !joules_close(p * b as f64, w) {
                e.push("whole_batch energy_per_request_joules * batch_size must match the window joules".into());
            }
        }
        _ => e.push("whole_batch requires batch_size >= 1, energy_per_request_joules, and window joules".into()),
    }
    check_stability(fe, e);
}

fn check_stability(fe: &Map<String, Value>, e: &mut Vec<String>) {
    if fe.get("is_stable") != Some(&Value::Bool(true)) {
        e.push("is_stable must be true on a pass; the v3 dataset keeps an unstable run as a row, not as a quoted result".into());
    }
    if fe.get("stability_rule").and_then(|v| v.as_str()).is_none_or(|s| s.is_empty()) {
        e.push("stability_rule must name the rule that was applied; the dataset card does not publish a numeric threshold to copy".into());
    }
}

fn joules_close(a: f64, b: f64) -> bool {
    let scale = a.abs().max(b.abs()).max(1.0);
    (a - b).abs() <= 1e-3 || (a - b).abs() <= 1e-3 * scale
}

fn joules_identity(gross: f64, above_idle: f64, idle: f64) -> bool {
    joules_close(gross, above_idle + idle)
}

fn check_energy_pass(obj: &Map<String, Value>, e: &mut Vec<String>, require_window: bool) {
    let Some(en) = obj.get("energy").and_then(|v| v.as_object()) else {
        e.push("energy object is required".into());
        return;
    };
    match en.get("class").and_then(|v| v.as_str()) {
        Some("measured") | Some("derived") => {}
        Some("estimated") => e.push("class estimated cannot back a pass (ferric-joule::Class::claimable)".into()),
        _ => e.push("energy.class must be measured or derived".into()),
    }
    require_finite(en, "gross_joules", e);
    require_finite(en, "above_idle_joules", e);
    if let Some(j) = en.get("above_idle_joules").and_then(|v| v.as_f64())
        && j < 0.0 {
            e.push("above_idle_joules below zero is an unresolved baseline, not a pass".into());
        }
    if en.get("boundary_gross").and_then(|v| v.as_str()).is_none_or(|s| s.is_empty())
        || en.get("boundary_above_idle").and_then(|v| v.as_str()).is_none_or(|s| s.is_empty())
    {
        e.push("energy.boundary_gross and energy.boundary_above_idle are both required".into());
    }
    if obj.get("quality_gate_receipt_id").and_then(|v| v.as_str()).is_none_or(|s| s.is_empty()) {
        e.push("quality_gate_receipt_id must name the matched-quality receipt for this same revision".into());
    }
    if require_window && en.get("window_covers_task") != Some(&Value::Bool(true)) {
        e.push("energy.window_covers_task must be true so the joules and the quality share one window".into());
    }
    if let Some(aba) = en.get("ferrotherm_aba").and_then(|v| v.as_object()) {
        if aba.get("a1_a2_agree") != Some(&Value::Bool(true)) {
            e.push("ferrotherm A/B/A pass requires a1_a2_agree true".into());
        }
    } else if require_window {
        e.push("ferrotherm_aba calibration object is required on a closed-loop energy pass".into());
    }
}

fn check_track_d_pass(obj: &Map<String, Value>, e: &mut Vec<String>) {
    let Some(h) = obj.get("hermetic").and_then(|v| v.as_object()) else {
        e.push("track D pass needs a hermetic object".into());
        return;
    };
    if h.get("offline").and_then(|v| v.as_bool()) != Some(true) {
        e.push("hermetic.offline must be true".into());
    }
    if h.get("lockfile_sha256").and_then(|v| v.as_str()).is_none_or(|s| !is_hex(s, 64, 64)) {
        e.push("hermetic.lockfile_sha256 must be 64 lowercase hex characters; a tree with no Cargo.lock cannot pass".into());
    }
    if h.get("rustc_commit").and_then(|v| v.as_str()).is_none_or(|s| s.len() < 7) {
        e.push("hermetic.rustc_commit must name the compiler used on both rebuild machines".into());
    }
    match h.get("machines").and_then(|v| v.as_array()) {
        Some(m) if m.len() >= 2 => {}
        _ => e.push("hermetic.machines must name at least two independent rebuilds".into()),
    }
    if h.get("artifact_sha256_agree") != Some(&Value::Bool(true)) {
        e.push("hermetic.artifact_sha256_agree must be true".into());
    }
    if h.get("bounds_refused") != Some(&Value::Bool(true)) || h.get("recovery_identical") != Some(&Value::Bool(true))
    {
        e.push("hermetic.bounds_refused and hermetic.recovery_identical must both be true".into());
    }
}

fn check_track_e_pass(obj: &Map<String, Value>, e: &mut Vec<String>) {
    let Some(m) = obj.get("matrix").and_then(|v| v.as_object()) else {
        e.push("track E pass needs a matrix object".into());
        return;
    };
    require_date(m, "date", e);
    let email = m.get("owner_email").and_then(|v| v.as_str()).unwrap_or("");
    if !author_is_physicalai(email) {
        e.push("matrix.owner_email must be the physicalai-bmi attribution".into());
    }
    if m.get("leadership_cells_open").and_then(|v| v.as_u64()) != Some(0) {
        e.push("a leadership pass has matrix.leadership_cells_open = 0, each cell backed by its own measured receipt".into());
    }
    require_nonempty(m, "peer_pins_sha256", e);
}

fn check_track_f_pass(obj: &Map<String, Value>, e: &mut Vec<String>) {
    if obj.get("task_id").and_then(|v| v.as_str()) != Some("f.closed-loop.ferromotion-sin3x.v1") {
        e.push("track F pass uses task_id f.closed-loop.ferromotion-sin3x.v1".into());
    }
    check_energy_pass(obj, e, true);
    let Some(ifs) = obj.get("interfaces").and_then(|v| v.as_object()) else {
        e.push("track F pass needs interfaces".into());
        return;
    };
    for name in ["ferrotherm_metrology", "ferromorphic_event_nir", "ferromotion_dynamics_control"] {
        let rev = ifs
            .get(name)
            .and_then(|v| v.get("revision"))
            .and_then(|v| v.as_str())
            .unwrap_or("");
        if rev.is_empty() || rev == "unpinned" || rev == "null" {
            e.push(format!("interfaces.{name}.revision must be pinned; unpinned is a withhold, not a pass"));
        }
        if name == "ferromorphic_event_nir"
            && ifs
                .get(name)
                .and_then(|v| v.get("sensor"))
                .and_then(|v| v.as_str())
                == Some("synthetic")
        {
            e.push("a synthetic ferric-signal fixture does not satisfy ferromorphic event-NIR".into());
        }
    }
    if obj
        .get("reference_agreement")
        .and_then(|v| v.get("met"))
        != Some(&Value::Bool(true))
    {
        e.push("reference_agreement.met must be true".into());
    }
}

pub fn check_protocol(text: &str) -> Vec<String> {
    let mut e = Vec::new();
    let Some(header) = header_block(text, "ferric-accept-protocol") else {
        e.push("missing <!-- ferric-accept-protocol ... --> header".into());
        return e;
    };
    let id = header_value(header, "id");
    let track = header_value(header, "track");
    if !id.starts_with(&(track.clone() + ".")) || track.len() != 1 || !"ABCDEF".contains(track.as_str()) {
        e.push(format!("protocol id `{id}` must start with its track `{track}.`"));
    }
    for h in REQUIRED_PROTOCOL_HEADINGS {
        if !text.lines().any(|l| l == format!("## {h}")) {
            e.push(format!("missing heading `## {h}`"));
        }
    }
    for (i, line) in text.lines().enumerate() {
        let t = line.trim();
        if t.starts_with("TODO") || t.starts_with("TBD") || t.starts_with("FIXME") || t.starts_with("XXX")
        {
            e.push(format!("line {}: placeholder marker `{t}`", i + 1));
        }
    }
    let lower = text.to_ascii_lowercase();
    if lower.contains("lorem ipsum") || lower.contains("placeholder result") {
        e.push("protocol contains placeholder prose".into());
    }
    if text.lines().any(|l| l.trim() == "outcome: pass") {
        e.push("a protocol must not declare outcome: pass".into());
    }
    let procedure = section(text, "Procedure");
    if !procedure.contains("cargo ") && !procedure.contains("scripts/") {
        e.push("Procedure must name a cargo command or a scripts/ entry point".into());
    }
    let tol = section(text, "Preregistered tolerances");
    if !tol.to_ascii_lowercase().contains("preregistered") {
        e.push("Preregistered tolerances section does not say what was preregistered".into());
    }
    let down = section(text, "Downgrade refusal").to_ascii_lowercase();
    if !down.contains("looser") && !down.contains("not loosened") {
        e.push("Downgrade refusal must state what is not loosened".into());
    }
    e
}

pub fn check_issue(text: &str) -> Vec<String> {
    let mut e = Vec::new();
    let Some(header) = header_block(text, "ferric-accept-issue") else {
        e.push("missing <!-- ferric-accept-issue ... --> header".into());
        return e;
    };
    let track = header_value(header, "track");
    let title = header_value(header, "title");
    if track.len() != 1 || !"ABCDEF".contains(track.as_str()) {
        e.push(format!("issue track `{track}` is not A-F"));
    }
    if title.chars().count() < 20 {
        e.push("issue title is shorter than 20 characters".into());
    }
    for h in REQUIRED_ISSUE_HEADINGS {
        if !text.lines().any(|l| l == format!("## {h}")) {
            e.push(format!("missing heading `## {h}`"));
        }
    }
    if !text.contains("dcharlot-physicalai-bmi") {
        e.push("issue body must name dcharlot-physicalai-bmi".into());
    }
    if !text.contains("cursor[bot]") {
        e.push("issue body must record that the available actor was cursor[bot]".into());
    }
    if !text.to_ascii_lowercase().contains("openie") {
        e.push("issue body must record that the OpenIE account is not an acceptable actor".into());
    }
    if !text.contains("returned []") {
        e.push("issue body must record the dedup result (issue list returned [])".into());
    }
    if text.contains("https://github.com/dcharlot-physicalai-bmi/ferric/issues/") {
        e.push("issue body links a GitHub issue; this workstream did not open one".into());
    }
    e
}

pub fn check_fabric_cell(v: &Value) -> Vec<String> {
    let mut e = Vec::new();
    let Some(obj) = v.as_object() else {
        return vec!["fabric cell is not an object".into()];
    };
    let id = obj.get("cell_id").and_then(|v| v.as_str()).unwrap_or("");
    if !REQUIRED_FABRIC_CELLS.contains(&id) {
        e.push(format!("unknown fabric cell `{id}`"));
    }
    if obj.get("evidence_class").and_then(|v| v.as_str()) != Some("target") {
        e.push(format!("{id}: evidence_class must be target"));
    }
    if obj.get("status").and_then(|v| v.as_str()) != Some("required") {
        e.push(format!("{id}: status must be required"));
    }
    if obj.get("outcome").and_then(|v| v.as_str()) != Some("not_applicable") {
        e.push(format!("{id}: outcome must be not_applicable"));
    }
    require_enum(obj, "implementation", &["present", "partial", "absent_in_tree"], &mut e);
    let browser = obj.get("real_browser_required").and_then(|v| v.as_bool());
    if REAL_BROWSER_CELLS.contains(&id) && browser != Some(true) {
        e.push(format!("{id}: real_browser_required must be true"));
    }
    if id == "gpu.wgpu.vulkan.lavapipe" {
        if browser == Some(true) {
            e.push("lavapipe is not a browser".into());
        }
        let dns = obj.get("does_not_satisfy").and_then(|v| v.as_array());
        let need = ["gpu.wgpu.vulkan.hardware", "gpu.wgpu.webgpu.browser"];
        for n in need {
            if dns.is_none_or(|a| !a.iter().any(|x| x.as_str() == Some(n))) {
                e.push(format!("lavapipe cell must list {n} in does_not_satisfy"));
            }
        }
    }
    e
}

pub fn check_tolerance(v: &Value) -> Vec<String> {
    let mut e = Vec::new();
    let Some(obj) = v.as_object() else {
        return vec!["tolerance is not an object".into()];
    };
    match obj.get("id").and_then(|v| v.as_str()) {
        Some(s) if is_protocol_id(s) => {}
        Some(s) => e.push(format!("tolerance id `{s}` must look like A.name.v1")),
        None => e.push("tolerance id is required".into()),
    }
    require_enum(obj, "track", &["A", "B", "C", "D", "E", "F"], &mut e);
    require_nonempty(obj, "comparison", &mut e);
    require_nonempty(obj, "predicate", &mut e);
    if obj.get("evidence_class").and_then(|v| v.as_str()) != Some("target") {
        e.push("a preregistered tolerance in the corpus is a target, not a measurement".into());
    }
    if obj.get("not_a_measurement") != Some(&Value::Bool(true)) {
        e.push("tolerance.not_a_measurement must be true".into());
    }
    let pred = obj.get("predicate").and_then(|v| v.as_str()).unwrap_or("").to_ascii_lowercase();
    for bad in ["tbd", "approximately", "good enough", "close enough", "placeholder"] {
        if pred.contains(bad) {
            e.push(format!("tolerance predicate contains `{bad}`"));
        }
    }
    e
}

pub fn check_corpus(root: &Path) -> Vec<String> {
    let mut e = Vec::new();
    let acc = root.join("docs/acceptance");
    let manifest = read_json(&acc.join("corpus/manifest.json"), &mut e);
    if let Some(m) = manifest.as_ref().and_then(|v| v.as_object()) {
        if m.get("evidence_class").and_then(|v| v.as_str()) != Some("target") {
            e.push("manifest.evidence_class must be target; hashes of documents are not model measurements".into());
        }
        if let Some(docs) = m.get("documents").and_then(|v| v.as_array()) {
            let mut listed = BTreeSet::new();
            for d in docs {
                let Some(o) = d.as_object() else { continue };
                let path = o.get("path").and_then(|v| v.as_str()).unwrap_or("");
                let want = o.get("sha256").and_then(|v| v.as_str()).unwrap_or("");
                listed.insert(path.to_string());
                let bytes = fs::read(root.join(path)).unwrap_or_default();
                if bytes.is_empty() {
                    e.push(format!("manifest path missing or empty: {path}"));
                    continue;
                }
                let got = hex(&sha256(&bytes));
                if got != want {
                    e.push(format!("sha256 mismatch for {path}: manifest {want} file {got}"));
                }
            }
            for dir in ["protocols", "issues"] {
                let rd = match fs::read_dir(acc.join(dir)) {
                    Ok(rd) => rd,
                    Err(_) => {
                        e.push(format!("missing docs/acceptance/{dir}"));
                        continue;
                    }
                };
                for ent in rd.flatten() {
                    let p = ent.path();
                    if p.extension().and_then(|s| s.to_str()) != Some("md") {
                        continue;
                    }
                    let rel = format!("docs/acceptance/{dir}/{}", ent.file_name().to_string_lossy());
                    if !listed.contains(&rel) {
                        e.push(format!("unlisted acceptance document {rel}"));
                    }
                }
            }
        } else {
            e.push("manifest.documents missing".into());
        }
    }

    if let Some(arr) = read_json(&acc.join("corpus/historical.json"), &mut e).and_then(|v| match v {
        Value::Array(a) => Some(a),
        _ => {
            e.push("historical.json must be an array".into());
            None
        }
    }) {
        if arr.is_empty() {
            e.push("historical ledger is empty".into());
        }
        for rec in &arr {
            e.extend(check_receipt(rec).into_iter().map(|m| format!("historical: {m}")));
            if let Some(src) = rec.get("source") {
                let path = src.get("path").and_then(|v| v.as_str()).unwrap_or("");
                let excerpt = src.get("excerpt").and_then(|v| v.as_str()).unwrap_or("");
                match fs::read_to_string(root.join(path)) {
                    Ok(body) => {
                        if !excerpt.is_empty() && !body.contains(excerpt) {
                            e.push(format!("excerpt not in {path}: {excerpt}"));
                        }
                    }
                    Err(_) => e.push(format!("historical source path missing: {path}")),
                }
            }
        }
    }

    if let Some(arr) = read_json(&acc.join("corpus/targets.json"), &mut e).and_then(|v| v.as_array().cloned()) {
        if arr.len() < 6 {
            e.push("targets.json needs one target receipt per track".into());
        }
        let mut tracks = BTreeSet::new();
        for rec in &arr {
            e.extend(check_receipt(rec).into_iter().map(|m| format!("target: {m}")));
            if let Some(t) = rec.get("track").and_then(|v| v.as_str()) {
                tracks.insert(t.to_string());
            }
        }
        for t in ["A", "B", "C", "D", "E", "F"] {
            if !tracks.contains(t) {
                e.push(format!("targets.json has no track {t}"));
            }
        }
    }

    if let Some(v) = read_json(&acc.join("corpus/measured.json"), &mut e) {
        let n = v.get("receipts").and_then(|r| r.as_array()).map(|a| a.len());
        if n != Some(0) {
            e.push(format!(
                "corpus/measured.json receipts must be empty in this foundation (found {n:?}); a pass belongs in a later measured receipt, not as a placeholder"
            ));
        }
        if v.get("evidence_class").and_then(|x| x.as_str()) != Some("measured") {
            e.push("measured.json evidence_class must be measured, stating that the set is empty".into());
        }
    }

    if let Some(arr) = read_json(&acc.join("corpus/fabric-cells.json"), &mut e)
        .and_then(|v| v.get("cells").and_then(|c| c.as_array()).cloned())
    {
        let mut ids = BTreeSet::new();
        for c in &arr {
            e.extend(check_fabric_cell(c));
            if let Some(id) = c.get("cell_id").and_then(|v| v.as_str()) {
                ids.insert(id.to_string());
            }
        }
        for id in REQUIRED_FABRIC_CELLS {
            if !ids.contains(*id) {
                e.push(format!("fabric matrix missing cell {id}"));
            }
        }
        if ids.len() != REQUIRED_FABRIC_CELLS.len() {
            e.push("fabric matrix has a cell that is not in REQUIRED_FABRIC_CELLS".into());
        }
    }

    if let Some(arr) = read_json(&acc.join("corpus/tolerances.json"), &mut e)
        .and_then(|v| v.get("tolerances").and_then(|t| t.as_array()).cloned())
    {
        let mut ids = BTreeSet::new();
        for t in &arr {
            e.extend(check_tolerance(t));
            if let Some(id) = t.get("id").and_then(|v| v.as_str()) {
                ids.insert(id.to_string());
            }
        }
        for (_, id) in MISSION_TOLERANCE {
            if !ids.contains(*id) {
                e.push(format!("tolerances.json missing mission id {id}"));
            }
        }
    }

    for name in [
        "receipt.schema.json",
        "fabric-cell.schema.json",
        "tolerance.schema.json",
        "fabric-energy.schema.json",
    ] {
        let v = read_json(&acc.join("schemas").join(name), &mut e);
        if let Some(obj) = v.as_ref().and_then(|v| v.as_object()) {
            if obj.get("$schema").and_then(|v| v.as_str()).is_none() {
                e.push(format!("{name} has no $schema"));
            }
            if name == "receipt.schema.json" {
                let req = obj.get("required").and_then(|v| v.as_array());
                for key in ["schema_version", "receipt_id", "track", "evidence_class", "status", "outcome", "protocol_id", "claim"] {
                    if req.is_none_or(|a| !a.iter().any(|x| x.as_str() == Some(key))) {
                        e.push(format!("receipt.schema.json required list dropped `{key}`"));
                    }
                }
            }
            if name == "fabric-energy.schema.json" {
                let req = obj.get("required").and_then(|v| v.as_array());
                for key in [
                    "cell_id",
                    "workload",
                    "instrument",
                    "domain",
                    "accounting",
                    "sync_before_read",
                    "warmup_outside_window",
                    "copied_leaderboard_joules_as_own",
                    "workload_parity",
                ] {
                    if req.is_none_or(|a| !a.iter().any(|x| x.as_str() == Some(key))) {
                        e.push(format!("fabric-energy.schema.json required list dropped `{key}`"));
                    }
                }
            }
        }
    }

    let mut tracks_p = BTreeSet::new();
    let mut tracks_i = BTreeSet::new();
    if let Ok(rd) = fs::read_dir(acc.join("protocols")) {
        for ent in rd.flatten() {
            let text = fs::read_to_string(ent.path()).unwrap_or_default();
            for msg in check_protocol(&text) {
                e.push(format!("{}: {msg}", ent.file_name().to_string_lossy()));
            }
            if let Some(h) = header_block(&text, "ferric-accept-protocol") {
                tracks_p.insert(header_value(h, "track"));
            }
            if ent.file_name().to_string_lossy().starts_with('C') {
                for needle in [
                    "https://github.com/ml-energy/zeus",
                    "https://ml.energy/leaderboard/",
                    "zeus-v0.16.0",
                    "hardware_counter",
                    "wall_meter",
                    "steady_state_service",
                    "nvmlDeviceGetTotalEnergyConsumption",
                ] {
                    if !text.contains(needle) {
                        e.push(format!("protocol C is missing methodology citation `{needle}`"));
                    }
                }
            }
            if ent.file_name().to_string_lossy().starts_with('B')
                && let Some(cells) = read_json(&acc.join("corpus/fabric-cells.json"), &mut Vec::new())
                    && let Some(arr) = cells.get("cells").and_then(|c| c.as_array()) {
                        for c in arr {
                            let id = c.get("cell_id").and_then(|v| v.as_str()).unwrap_or("");
                            if !text.contains(id) {
                                e.push(format!("protocol B does not name fabric cell {id}"));
                            }
                        }
                    }
        }
    }
    if let Ok(rd) = fs::read_dir(acc.join("issues")) {
        for ent in rd.flatten() {
            let text = fs::read_to_string(ent.path()).unwrap_or_default();
            for msg in check_issue(&text) {
                e.push(format!("{}: {msg}", ent.file_name().to_string_lossy()));
            }
            if let Some(h) = header_block(&text, "ferric-accept-issue") {
                tracks_i.insert(header_value(h, "track"));
            }
        }
    }
    for t in ["A", "B", "C", "D", "E", "F"] {
        if !tracks_p.contains(t) {
            e.push(format!("no protocol for track {t}"));
        }
        if !tracks_i.contains(t) {
            e.push(format!("no issue body for track {t}"));
        }
    }
    if tracks_p.len() != 6 {
        e.push(format!("expected 6 protocols, header tracks {:?}", tracks_p));
    }
    if tracks_i.len() != 6 {
        e.push(format!("expected 6 issue bodies, header tracks {:?}", tracks_i));
    }
    e
}

fn author_is_physicalai(email: &str) -> bool {
    let l = email.to_ascii_lowercase();
    !l.contains("openie")
        && (l.contains("physicalai-bmi") || l.contains("dcharlot-physicalai-bmi"))
}

fn reject_openie(obj: &Map<String, Value>, e: &mut Vec<String>) {
    for key in ["email", "account"] {
        if let Some(s) = obj.get("actor").and_then(|a| a.get(key)).and_then(|v| v.as_str())
            && s.to_ascii_lowercase().contains("openie") {
                e.push(format!("actor.{key} is an OpenIE identity; remote attribution must not use it"));
            }
    }
}

fn require_nonempty(obj: &Map<String, Value>, key: &str, e: &mut Vec<String>) {
    if obj.get(key).and_then(|v| v.as_str()).is_none_or(|s| s.is_empty()) {
        e.push(format!("`{key}` must be a non-empty string"));
    }
}

fn require_nonempty_path(obj: &Map<String, Value>, path: &[&str], e: &mut Vec<String>) {
    let mut cur = Some(obj);
    for (i, key) in path.iter().enumerate() {
        let Some(o) = cur else { break };
        if i + 1 == path.len() {
            if o.get(*key).and_then(|v| v.as_str()).is_none_or(|s| s.is_empty()) {
                e.push(format!("`{}` must be a non-empty string", path.join(".")));
            }
            return;
        }
        cur = o.get(*key).and_then(|v| v.as_object());
        if cur.is_none() {
            e.push(format!("`{}` is missing", path.join(".")));
            return;
        }
    }
}

fn require_enum(obj: &Map<String, Value>, key: &str, allowed: &[&str], e: &mut Vec<String>) -> Option<String> {
    match obj.get(key).and_then(|v| v.as_str()) {
        Some(s) if allowed.contains(&s) => Some(s.to_string()),
        Some(s) => {
            e.push(format!("`{key}` value `{s}` is not one of {}", allowed.join(", ")));
            None
        }
        None => {
            e.push(format!("`{key}` must be a string"));
            None
        }
    }
}

fn require_id(obj: &Map<String, Value>, key: &str, e: &mut Vec<String>) {
    match obj.get(key).and_then(|v| v.as_str()) {
        Some(s) if is_id(s) => {}
        Some(s) => e.push(format!("`{key}` `{s}` is not a lowercase id")),
        None => e.push(format!("`{key}` must be a string")),
    }
}

fn require_date(obj: &Map<String, Value>, key: &str, e: &mut Vec<String>) {
    match obj.get(key).and_then(|v| v.as_str()) {
        Some(s) if is_date(s) => {}
        _ => e.push(format!("`{key}` must be a YYYY-MM-DD date")),
    }
}

fn require_finite(obj: &Map<String, Value>, key: &str, e: &mut Vec<String>) {
    match obj.get(key).and_then(|v| v.as_f64()) {
        Some(n) if n.is_finite() => {}
        _ => e.push(format!("`{key}` must be a finite number")),
    }
}

fn is_protocol_id(s: &str) -> bool {
    let Some((head, rest)) = s.split_once('.') else {
        return false;
    };
    head.len() == 1 && "ABCDEF".contains(head) && is_id(rest)
}

fn is_id(s: &str) -> bool {
    let b = s.as_bytes();
    (3..=96).contains(&b.len())
        && b[0].is_ascii_lowercase()
        && b.iter().all(|c| {
            c.is_ascii_lowercase() || c.is_ascii_digit() || matches!(*c, b'.' | b'_' | b'-')
        })
}

fn is_hex(s: &str, min: usize, max: usize) -> bool {
    (min..=max).contains(&s.len()) && s.bytes().all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase())
}

fn is_date(s: &str) -> bool {
    let b = s.as_bytes();
    if b.len() != 10 || b[4] != b'-' || b[7] != b'-' {
        return false;
    }
    let y = s[0..4].parse::<u16>().unwrap_or(0);
    let m = s[5..7].parse::<u8>().unwrap_or(0);
    let d = s[8..10].parse::<u8>().unwrap_or(0);
    y >= 2000 && (1..=12).contains(&m) && (1..=31).contains(&d)
}

fn header_block<'a>(text: &'a str, name: &str) -> Option<&'a str> {
    let start = format!("<!-- {name}");
    let i = text.find(&start)?;
    let rest = &text[i + start.len()..];
    let j = rest.find("-->")?;
    Some(&rest[..j])
}

fn header_value(block: &str, key: &str) -> String {
    let prefix = format!("{key}:");
    block
        .lines()
        .find_map(|l| l.trim().strip_prefix(&prefix).map(|s| s.trim().to_string()))
        .unwrap_or_default()
}

fn section(text: &str, name: &str) -> String {
    let marker = format!("## {name}");
    let Some(i) = text.find(&marker) else { return String::new() };
    let rest = &text[i + marker.len()..];
    let end = rest.find("\n## ").unwrap_or(rest.len());
    rest[..end].to_string()
}

fn read_json(path: &Path, e: &mut Vec<String>) -> Option<Value> {
    match fs::read_to_string(path) {
        Ok(s) => match serde_json::from_str(&s) {
            Ok(v) => Some(v),
            Err(err) => {
                e.push(format!("{}: {err}", path.display()));
                None
            }
        },
        Err(err) => {
            e.push(format!("{}: {err}", path.display()));
            None
        }
    }
}
