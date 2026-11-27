//! The WGSL `det_tanh` and the GELU that calls it, against `cpu::det_tanh`.
//! Bit-identical on whatever GPU this process has. Lavapipe in the software-GPU
//! job and Metal in the macOS job both run this; a machine with no adapter
//! returns early, the same way `lavapipe_llvm21.rs` does.

use ferric_core::cpu::{det_tanh, gelu_tanh};

fn samples() -> Vec<f32> {
    let mut xs = vec![
        0.0,
        -0.0,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::NAN,
        f32::from_bits(1),
        f32::from_bits(0x8000_0001),
        f32::from_bits(0x007f_ffff),
        f32::from_bits(0x4110_2cb4),
        f32::from_bits(0x4110_2cb3),
        0.25,
        -0.25,
        1.0,
        -1.0,
        8.320618,
        -8.320618,
        9.0,
        -9.0,
    ];
    // Every 4096th positive finite encoding up to the saturation threshold,
    // both signs. That is a few hundred thousand points, dense next to the
    // width of one table slot.
    let sat = 0x4110_2cb4u32;
    let mut bits = 1u32;
    while bits < sat {
        xs.push(f32::from_bits(bits));
        xs.push(f32::from_bits(bits | 0x8000_0000));
        bits += 4096;
    }
    // Every f32 in two slots around the old 4-ulp site, and across saturation.
    for (start, end) in [(8.320f32, 8.322f32), (9.0, 9.02f32), (0.25, 0.2502)] {
        let mut b = start.to_bits();
        let stop = end.to_bits();
        while b < stop {
            xs.push(f32::from_bits(b));
            xs.push(-f32::from_bits(b));
            b += 1;
        }
    }
    xs
}

#[test]
fn the_shader_matches_the_cpu_on_a_dense_set() {
    let require = std::env::var("FERRIC_REQUIRE_ADAPTER").ok().as_deref() == Some("1");
    let ctx = match pollster::block_on(ferric_core::Context::new()) {
        Ok(ctx) => ctx,
        Err(e) => {
            if require {
                panic!("FERRIC_REQUIRE_ADAPTER=1 and no GPU adapter: {e}");
            }
            eprintln!("no GPU context — shader parity is checked on the CI adapters");
            return;
        }
    };
    let xs = samples();
    let gpu = pollster::block_on(ctx.det_tanh(&xs)).expect("det_tanh dispatch");
    assert_eq!(gpu.len(), xs.len());
    for (i, (&got, &x)) in gpu.iter().zip(&xs).enumerate() {
        let want = det_tanh(x);
        if x.is_nan() {
            assert!(got.is_nan(), "index {i} turned a NaN into {got}");
            continue;
        }
        assert_eq!(
            got.to_bits(),
            want.to_bits(),
            "det_tanh shader mismatch at x {x} (bits {:#x}) index {i}: gpu {:#x} cpu {:#x}",
            x.to_bits(),
            got.to_bits(),
            want.to_bits()
        );
    }
    let gelu_gpu = pollster::block_on(ctx.gelu_tanh(&xs)).expect("gelu_tanh dispatch");
    let gelu_cpu = gelu_tanh(&xs);
    for (i, (&got, &want)) in gelu_gpu.iter().zip(&gelu_cpu).enumerate() {
        if xs[i].is_nan() {
            assert!(got.is_nan(), "gelu index {i} turned a NaN into {got}");
            continue;
        }
        assert_eq!(
            got.to_bits(),
            want.to_bits(),
            "gelu_tanh shader mismatch at x {} index {i}",
            xs[i]
        );
    }
    println!(
        "det_tanh shader matched the CPU on {} values, adapter {}",
        xs.len(),
        ctx.adapter_name
    );
}
