//! **Real ANE execution provider** — the fabric's NPU, via CoreML, with the honesty gate
//! `ferric/docs/NPU.md` demands: the device is only reported dispatchable after **`MLComputePlan`
//! confirms the Neural Engine** is the preferred compute device for the model's matmul — we never
//! claim the ANE unless Apple's own scheduler receipt says so.
//!
//! CoreML runs *compiled* graphs, so like every real NPU deployment this EP executes a fixed
//! compiled program. Model selection was EMPIRICAL, receipts read from Rust: rank-2 matmuls and a
//! 3×3/64ch conv are scheduled to `MLCPUComputeDevice` even with `.all` — the ANE takes the
//! **attention-shaped rank-4 matmul** `(1,8,512,64)·(1,8,64,512)`, both inputs dynamic, every op
//! including the fp16 casts (`ios16.matmul→ANE`). That model (authored with coremltools MIL,
//! compiled with `coremlcompiler`, 4 files ≈ 16 KB) is embedded here. Arbitrary `bmm` shapes tile
//! onto it: 512×512 output tiles, K covered 8×64 per prediction (the batch lanes are K-slices,
//! host-summed) — correct everywhere, efficient near the compiled shape, the standard NPU trade
//! stated plainly. fp16 inputs by contract, like the Metal-4 device.
#![cfg(all(target_os = "macos", target_arch = "aarch64"))]
// MLMultiArray::dataPointer is deprecated in favor of block-based accessors; for synchronous
// fill/readback of freshly-created arrays the direct pointer is the right tool.
#![allow(deprecated)]

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2::AnyThread;
use objc2_core_ml::*;
use objc2_foundation::{NSArray, NSDictionary, NSNumber, NSString, NSURL};
use std::sync::mpsc;

const TILE: usize = 512; // output tile (m, n) per prediction
const KSLOT: usize = 64; // K per batch lane
const LANES: usize = 8; // batch lanes = K-slices per prediction (K covered: 512)
const F_COREMLDATA: &[u8] = include_bytes!("coreml_matmul4d/coremldata.bin");
const F_METADATA: &[u8] = include_bytes!("coreml_matmul4d/metadata.json");
const F_MIL: &[u8] = include_bytes!("coreml_matmul4d/model.mil");
const F_ANALYTICS: &[u8] = include_bytes!("coreml_matmul4d/analytics-coremldata.bin");

pub struct CoreMlNpu {
    model: Retained<MLModel>,
    /// Per-op device summary from `MLComputePlan` (the receipt).
    pub plan_report: String,
    /// True only when the plan's preferred device for the matmul is the Neural Engine.
    pub ane_confirmed: bool,
}

// SAFETY: MLModel prediction is documented thread-safe; we only share immutable handles.
unsafe impl Send for CoreMlNpu {}
unsafe impl Sync for CoreMlNpu {}

fn write_modelc() -> Option<std::path::PathBuf> {
    let dir = std::env::temp_dir().join("ferric_coreml_matmul4d.mlmodelc");
    std::fs::create_dir_all(dir.join("analytics")).ok()?;
    std::fs::write(dir.join("coremldata.bin"), F_COREMLDATA).ok()?;
    std::fs::write(dir.join("metadata.json"), F_METADATA).ok()?;
    std::fs::write(dir.join("model.mil"), F_MIL).ok()?;
    std::fs::write(dir.join("analytics/coremldata.bin"), F_ANALYTICS).ok()?;
    Some(dir)
}

struct SendPlan(Option<Retained<MLComputePlan>>);
// SAFETY: the completion handler hands us ownership; we only move it across the channel once.
unsafe impl Send for SendPlan {}

impl CoreMlNpu {
    /// Load the embedded compiled model with `computeUnits = .all`, query the compute plan, and
    /// build the EP. Returns `None` when CoreML can't load it (non-macOS callers never get here).
    pub fn new() -> Option<CoreMlNpu> {
        let dir = write_modelc()?;
        let url = NSURL::fileURLWithPath(&NSString::from_str(dir.to_str()?));
        let cfg = unsafe { MLModelConfiguration::new() };
        unsafe { cfg.setComputeUnits(MLComputeUnits::All) };
        let model = unsafe { MLModel::modelWithContentsOfURL_configuration_error(&url, &cfg) }.ok()?;

        let (report, ane) = plan_receipt(&url, &cfg);
        Some(CoreMlNpu { model, plan_report: report, ane_confirmed: ane })
    }
}

/// Load the `MLComputePlan` for a compiled model and summarize per-op preferred devices; `true`
/// only when the Neural Engine is preferred for the compute op (matmul/linear/conv).
pub fn plan_receipt(url: &NSURL, cfg: &MLModelConfiguration) -> (String, bool) {
    {
        // The receipt: MLComputePlan's per-operation preferred device.
        let (tx, rx) = mpsc::channel::<SendPlan>();
        let block = block2::RcBlock::new(move |plan: *mut MLComputePlan, _err: *mut objc2_foundation::NSError| {
            let owned = if plan.is_null() { None } else { Some(unsafe { Retained::retain(plan) }.unwrap()) };
            let _ = tx.send(SendPlan(owned));
        });
        unsafe { MLComputePlan::loadContentsOfURL_configuration_completionHandler(url, cfg, &block) };
        let plan = rx.recv_timeout(std::time::Duration::from_secs(20)).ok().and_then(|p| p.0);

        let (mut report, mut ane) = (String::new(), false);
        if let Some(plan) = &plan {
            let structure = unsafe { plan.modelStructure() };
            if let Some(prog) = unsafe { structure.program() } {
                let funcs = unsafe { prog.functions() };
                for fname in funcs.keys() {
                    let f = funcs.objectForKey(&fname).unwrap();
                    let ops = unsafe { f.block().operations() };
                    for op in ops.iter() {
                        let name = unsafe { op.operatorName() }.to_string();
                        if name == "const" {
                            continue;
                        }
                        let dev = unsafe { plan.computeDeviceUsageForMLProgramOperation(&op) }
                            .map(|u| unsafe { u.preferredComputeDevice() });
                        let dev_name = match &dev {
                            Some(d) => {
                                // class-name check keeps us off protocol-object downcast plumbing;
                                // MLNeuralEngineComputeDevice is the ANE's concrete class
                                let any = unsafe { &*(Retained::as_ptr(d) as *const objc2::runtime::AnyObject) };
                                let cls = format!("{:?}", any.class());
                                let is_ane = cls.contains("NeuralEngine");
                                if is_ane && (name.contains("matmul") || name.contains("linear") || name.contains("conv")) {
                                    ane = true;
                                }
                                if is_ane { "ANE".to_string() } else { cls }
                            }
                            None => "unknown".into(),
                        };
                        report.push_str(&format!("{name}→{dev_name} "));
                    }
                }
            }
        }
        (report.trim().to_string(), ane)
    }
}

impl CoreMlNpu {

    /// One compiled-shape prediction: x[1,8,512,64] · y[1,8,64,512] → out[1,8,512,512].
    fn predict_tile(&self, x: &[f32], y: &[f32]) -> Vec<f32> {
        let mk = |data: &[f32], d2: usize, d3: usize| -> Retained<MLMultiArray> {
            let shape = NSArray::from_retained_slice(&[
                NSNumber::new_usize(1),
                NSNumber::new_usize(LANES),
                NSNumber::new_usize(d2),
                NSNumber::new_usize(d3),
            ]);
            let arr = unsafe {
                MLMultiArray::initWithShape_dataType_error(MLMultiArray::alloc(), &shape, MLMultiArrayDataType::Float32)
            }
            .expect("MLMultiArray");
            unsafe {
                std::ptr::copy_nonoverlapping(data.as_ptr(), arr.dataPointer().as_ptr() as *mut f32, LANES * d2 * d3);
            }
            arr
        };
        let (ax, ay) = (mk(x, TILE, KSLOT), mk(y, KSLOT, TILE));
        let fx = unsafe { MLFeatureValue::featureValueWithMultiArray(&ax) };
        let fy = unsafe { MLFeatureValue::featureValueWithMultiArray(&ay) };
        let dict = NSDictionary::from_retained_objects(
            &[&*NSString::from_str("x"), &*NSString::from_str("y")],
            &[fx, fy],
        );
        // the initializer is typed over AnyObject values; MLFeatureValue IS-A AnyObject
        let dict_any = unsafe {
            &*(Retained::as_ptr(&dict) as *const NSDictionary<NSString, objc2::runtime::AnyObject>)
        };
        let provider = unsafe {
            MLDictionaryFeatureProvider::initWithDictionary_error(MLDictionaryFeatureProvider::alloc(), dict_any)
        }
        .expect("feature provider");
        let out = unsafe {
            self.model
                .predictionFromFeatures_error(ProtocolObject::from_ref(&*provider))
        }
        .expect("CoreML prediction");
        let fv = unsafe { out.featureValueForName(&NSString::from_str("out")) }.expect("out feature");
        let arr = unsafe { fv.multiArrayValue() }.expect("out array");
        let count = LANES * TILE * TILE;
        let mut res = vec![0.0f32; count];
        unsafe {
            match arr.dataType() {
                MLMultiArrayDataType::Float32 => {
                    std::ptr::copy_nonoverlapping(arr.dataPointer().as_ptr() as *const f32, res.as_mut_ptr(), count);
                }
                MLMultiArrayDataType::Float16 => {
                    let p = arr.dataPointer().as_ptr() as *const half::f16;
                    for (i, r) in res.iter_mut().enumerate() {
                        *r = (*p.add(i)).to_f32();
                    }
                }
                other => panic!("unexpected CoreML output dtype {other:?}"),
            }
        }
        res
    }
}

impl crate::sched::NpuBackend for CoreMlNpu {
    fn name(&self) -> String {
        "ANE (CoreML rank-4 matmul, fp16)".into()
    }

    /// General bmm tiled onto the compiled shape: 512×512 output tiles; K covered 8 lanes × 64 per
    /// prediction (lanes are K-slices, partial products host-summed). Zero-padded partials keep
    /// every shape correct; efficiency peaks near the compiled shape — the standard NPU trade.
    fn bmm(&self, a: &[f32], b: &[f32], batch: usize, m: usize, k: usize, n: usize) -> Vec<f32> {
        let kchunk = LANES * KSLOT; // K per prediction
        let mut out = vec![0.0f32; batch * m * n];
        let mut xa = vec![0.0f32; LANES * TILE * KSLOT];
        let mut xb = vec![0.0f32; LANES * KSLOT * TILE];
        for bt in 0..batch {
            for m0 in (0..m).step_by(TILE) {
                let mt = (m - m0).min(TILE);
                for n0 in (0..n).step_by(TILE) {
                    let nt = (n - n0).min(TILE);
                    let mut acc = vec![0.0f32; TILE * TILE];
                    for k0 in (0..k).step_by(kchunk) {
                        xa.fill(0.0);
                        xb.fill(0.0);
                        for lane in 0..LANES {
                            let kbase = k0 + lane * KSLOT;
                            if kbase >= k {
                                break;
                            }
                            let kt = (k - kbase).min(KSLOT);
                            for i in 0..mt {
                                let src = bt * m * k + (m0 + i) * k + kbase;
                                let dst = lane * TILE * KSLOT + i * KSLOT;
                                xa[dst..dst + kt].copy_from_slice(&a[src..src + kt]);
                            }
                            for i in 0..kt {
                                let src = (kbase + i) * n + n0;
                                let dst = lane * KSLOT * TILE + i * TILE;
                                xb[dst..dst + nt].copy_from_slice(&b[src..src + nt]);
                            }
                        }
                        let part = self.predict_tile(&xa, &xb);
                        for lane in 0..LANES {
                            if k0 + lane * KSLOT >= k {
                                break;
                            }
                            let po = lane * TILE * TILE;
                            for (o, p) in acc.iter_mut().zip(&part[po..po + TILE * TILE]) {
                                *o += p;
                            }
                        }
                    }
                    for i in 0..mt {
                        let dst = bt * m * n + (m0 + i) * n + n0;
                        out[dst..dst + nt].copy_from_slice(&acc[i * TILE..i * TILE + nt]);
                    }
                }
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sched::NpuBackend;

    /// The EP loads, the compute-plan receipt is read, and bmm matches the fp16 oracle — at the
    /// compiled tile, at a padded shape, and across K-tile accumulation.
    #[test]
    fn coreml_npu_loads_and_bmm_matches_the_fp16_oracle() {
        let Some(npu) = CoreMlNpu::new() else {
            eprintln!("CoreML EP failed to load — skipping");
            return;
        };
        eprintln!("plan: [{}]", npu.plan_report);
        eprintln!("ANE confirmed: {}", npu.ane_confirmed);

        let check = |m: usize, k: usize, n: usize| {
            let a: Vec<f32> = (0..m * k).map(|i| 0.02 * (((i + 1) % 13) as f32 - 6.0)).collect();
            let b: Vec<f32> = (0..k * n).map(|i| 0.02 * (((i + 7) % 11) as f32 - 5.0)).collect();
            let got = npu.bmm(&a, &b, 1, m, k, n);
            let q = |v: &[f32]| -> Vec<f32> { v.iter().map(|&x| half::f16::from_f32(x).to_f32()).collect() };
            let (af, bf) = (q(&a), q(&b));
            let mut err = 0.0f32;
            for i in 0..m {
                for j in 0..n {
                    let acc: f32 = (0..k).map(|l| af[i * k + l] * bf[l * n + j]).sum();
                    err = err.max((got[i * n + j] - acc).abs());
                }
            }
            assert!(err < 2e-2, "NPU bmm m={m} k={k} n={n}: err {err}");
            eprintln!("  bmm {m}x{k}x{n}: err {err:.3e}");
        };
        check(512, 512, 512); // the compiled tile exactly
        check(100, 300, 200); // padded partial tile
        check(64, 1024, 64); // K-tile accumulation (2 tiles)
    }
}

#[cfg(test)]
mod plan_experiments {
    use super::*;

    /// Which model patterns does CoreML actually schedule on the ANE? Point FERRIC_MLMODELC_DIR at
    /// a directory of compiled .mlmodelc bundles and read each receipt.
    #[test]
    fn plan_receipts_for_candidate_models() {
        let Ok(dir) = std::env::var("FERRIC_MLMODELC_DIR") else {
            eprintln!("FERRIC_MLMODELC_DIR unset — skipping");
            return;
        };
        let cfg = unsafe { MLModelConfiguration::new() };
        unsafe { cfg.setComputeUnits(MLComputeUnits::All) };
        for entry in std::fs::read_dir(&dir).unwrap().flatten() {
            let p = entry.path();
            if p.extension().is_none_or(|e| e != "mlmodelc") {
                continue;
            }
            let url = unsafe { NSURL::fileURLWithPath(&NSString::from_str(p.to_str().unwrap())) };
            let (report, ane) = plan_receipt(&url, &cfg);
            eprintln!("{}: ANE={} [{}]", p.file_name().unwrap().to_string_lossy(), ane, report);
        }
    }
}
