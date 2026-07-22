# NPU model generation

The CoreML models behind `ferric_tensor::npu_coreml` (and the ANE-eligibility experiments in
`docs/NPU.md`). Regenerate with:

```sh
pip3 install --user coremltools
python3 build_models.py         # rank-2 matmul + fixed-weight linear (CPU-scheduled — kept as evidence)
python3 build_conv_models.py    # conv3x3 (CPU), conv1x1 (ANE), rank-4 matmul (ANE — the embedded EP model)
for m in *.mlpackage; do xcrun coremlcompiler compile "$m" compiled/; done
```

The embedded EP model is `compiled/matmul4d.mlmodelc` → `crates/ferric-tensor/src/coreml_matmul4d/`
(4 files; `analytics/coremldata.bin` renamed `analytics-coremldata.bin`). Read any model's
per-op device receipt with:

```sh
FERRIC_MLMODELC_DIR=$PWD/compiled cargo test -p ferric-tensor --lib plan_receipts -- --nocapture
```
