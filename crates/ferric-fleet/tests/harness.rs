use ferric_fleet::{
    attributable, bind_clock, delta_uj, dram_joules, duration_is_explicit, evaluate_domains,
    gpu_receipt_allowed, require_drains, verify_bytes, ClockSnapshot, Domain, Drain, Phase, Sample,
    JOINT_STEP_EXECUTED, MODEL_RECEIPTS,
};

#[test]
fn model_and_joint_step_stay_unrun() {
    assert_eq!(MODEL_RECEIPTS, 0);
    assert!(!JOINT_STEP_EXECUTED);
}

#[test]
fn software_adapters_cannot_back_a_gpu_receipt() {
    assert!(gpu_receipt_allowed("llvmpipe").is_err());
    assert!(gpu_receipt_allowed("Lavapipe").is_err());
    assert!(gpu_receipt_allowed("Google SwiftShader").is_err());
    assert!(gpu_receipt_allowed("AMD Radeon 8060S").is_ok());
}

#[test]
fn drains_are_required_around_a_window() {
    let ok = Drain { submit_empty: true, poll_wait: true, readback: true };
    let no_read = Drain { readback: false, ..ok };
    assert!(require_drains(Phase::Load, ok, no_read).is_ok());
    assert!(require_drains(Phase::SteadyInfer, ok, no_read).is_err());
    assert!(require_drains(Phase::JointStep, ok, ok).is_ok());
    let open = Drain { submit_empty: false, poll_wait: false, readback: false };
    assert!(require_drains(Phase::Warmup, open, ok).is_err());
}

#[test]
fn five_seconds_is_not_a_default_and_a_busy_machine_is_not_attributable() {
    assert!(duration_is_explicit(None).is_err());
    assert!(attributable("busy", 4).is_err());
    assert!(attributable("quiet", 1).is_err());
    assert!(attributable("quiet", 2).is_ok());
}

#[test]
fn rapl_domains_wrap_and_do_not_sum() {
    let range = 65_532_610_987u64;
    let (delta, wrapped) = delta_uj(range - 10, 5, range).unwrap();
    assert!(wrapped);
    assert_eq!(delta, 15);
    let samples = vec![
        Sample { domain: Domain::RaplPackage, t_monotonic_ns: 0, counter_uj: Some(10), range_uj: range, watts: None },
        Sample { domain: Domain::RaplCore, t_monotonic_ns: 0, counter_uj: Some(4), range_uj: range, watts: None },
        Sample { domain: Domain::RocmPpt, t_monotonic_ns: 0, counter_uj: None, range_uj: 0, watts: Some(40.0) },
        Sample { domain: Domain::MacmonDram, t_monotonic_ns: 0, counter_uj: Some(0), range_uj: 0, watts: Some(0.0) },
    ];
    assert!(evaluate_domains(&samples, true, false, false).unwrap_err().contains("not additive"));
    assert!(evaluate_domains(&samples, false, true, false).unwrap_err().contains("GPU-only"));
    assert!(evaluate_domains(&samples, false, false, true).unwrap_err().contains("not a wall"));
    assert_eq!(dram_joules(&samples).unwrap(), None);
}

#[test]
fn clock_bind_matches_the_forward_product_every_call() {
    let period = 1000u64;
    let dt_bits = ((period as f64) * 1e-9).to_bits();
    assert_eq!(dt_bits, 0x3eb0c6f7a0b5ed8e);
    let mut s = ClockSnapshot {
        origin_ns: 0,
        period_ns: period,
        t0_ns: 3000,
        step_duration_ns: period,
        dt_bits,
    };
    assert_eq!(bind_clock(&s).unwrap(), 3);
    s.dt_bits ^= 1;
    assert!(bind_clock(&s).is_err());
    s.dt_bits = dt_bits;
    s.t0_ns = 2500;
    assert!(bind_clock(&s).is_err());
}

#[test]
fn real_state_proj_column_matches_the_weight_file() {
    let bytes = include_bytes!("fixtures/state_proj.safetensors");
    let mut x = [0f32; 32];
    x[0] = 1.0;
    let y = ferric_fleet::state_proj(bytes, &x).unwrap();
    assert_eq!(y.len(), 960);
    assert!((y[0] - (-0.13992482)).abs() < 1e-6, "{}", y[0]);
    assert!((y[1] - (-0.08349697)).abs() < 1e-6, "{}", y[1]);
    assert!((y[2] - (-0.13289170)).abs() < 1e-6, "{}", y[2]);
    assert!((y[3] - 0.04185394).abs() < 1e-6, "{}", y[3]);
    let full = std::path::Path::new("/tmp/smolvla/model.safetensors");
    if full.exists() {
        let file = std::fs::read(full).unwrap();
        assert!(ferric_fleet::verify_bytes("model.safetensors", &file).is_ok());
        let y2 = ferric_fleet::state_proj(&file, &x).unwrap();
        assert_eq!(y, y2);
    }
}

#[test]
fn pinned_small_artifacts_match_their_bytes() {
    let config = include_bytes!("fixtures/config.json");
    assert!(verify_bytes("config.json", config).is_ok());
    let stats = include_bytes!("fixtures/policy_preprocessor_step_5_normalizer_processor.safetensors");
    assert!(verify_bytes("policy_preprocessor_step_5_normalizer_processor.safetensors", stats).is_ok());
    assert!(verify_bytes("config.json", b"nope").is_err());
}
