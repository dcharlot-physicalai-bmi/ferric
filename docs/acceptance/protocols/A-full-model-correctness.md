<!-- ferric-accept-protocol
id: A.full-model-correctness.v1
track: A
-->

# Track A — full-model correctness

## Mission

A registered architecture is correct when an independent reference agrees with Ferric on the whole path: text to token ids, GGUF import, full logit rows, and generation. Malformed files and resource overruns are refused by name. The mission is that bar. A toy forward that matches its own CPU twin, and a greedy string that matches, are parts of the path. They are not the path.

Tolerance id: `A.full-model.v1` in `docs/acceptance/corpus/tolerances.json`.

## Independent reference

| Stage | Reference | What is not a reference |
|---|---|---|
| Tokenization | `llama-tokenize` from llama.cpp, invoked by `scripts/tokenizer_conformance.sh` | Ferric's own pretokenizer compared with itself |
| Import | The GGUF the authors' converter wrote from their snapshot (`convert_hf_to_gguf.py <snapshot> --outtype f32`, or the checkpoint's own F16/BF16 when `lm_conformance.sh` treats that file as EXACT) | A quantized file. The script exits 2 on one and says it cannot verify the math |
| Logits | The authors' implementation, eager attention, float32, recorded in `crates/ferric-llama/tests/fixtures/lm/<model>.json` by `crates/ferric-llama/examples/refgen/lm_logits_ref.py` | llama.cpp greedy argmax. `scripts/lm_conformance.sh` records that a small logit error almost never changes an argmax |
| Generation | The same independent implementation's greedy token ids, stored as a separate stage | The logit stage. Generation does not stand in for it |
| KV cache, where the contract is exactness | Full recompute of the same prefix inside the same example | The in-tree assert `worst < 2e-3` in `crates/ferric-core/examples/generate.rs` |
| Malformed | The reader's named error | A repaired parse that then emits logits |
| Resource bounds | `TierError::BudgetTooSmall` and `Engine::budget`'s named context error | A silent truncation or a buffer of zeros |

`scripts/validate_vs_llamacpp.sh` compares greedy text to `llama-simple`. Its header says matching argmax means the logits are right. `scripts/lm_conformance.sh` retracts that. This protocol follows the retraction.

## Preregistered tolerances

These bars are preregistered with the protocol. They are not the error a run happened to print.

- Token ids: byte-identical to `llama-tokenize` on the corpus. A SKIP because llama.cpp cannot load a Ferric-only format is not a pass.
- Logits, EXACT class (`file_type` F32, or F16/BF16 when `checkpoint_dtype` is that width): max absolute logit difference ≤ 1e-3 and the script's sum-of-squares tolerance ≤ 1e-4. Source of the numbers: `LOGIT_TOL, SSQ_TOL = (1e-3, 1e-4) if EXACT else (5e-2, 5e-3)` in `scripts/lm_conformance.sh`.
- The else branch, 5e-2 and 5e-3, applies only to a file that rounded the authors' weights. A receipt that uses it as the math proof is not a track A pass.
- KV-cache versus full recompute: max abs = 0 on the f32 logits the example compares. The example's assert is `worst < 2e-3`. That assert may keep guarding the example. It does not meet this protocol.
- Generation: token-id identity with the independent reference, reported as the generation stage only.
- Malformed: `ferric_gguf::parse` returns `not a GGUF file` for a bad magic, `GGUF header truncated or malformed` for a short header, and an error containing `truncated` when a declared tensor extent passes EOF (`crates/ferric-gguf/src/lib.rs`).
- Context: `Engine::budget` returns `the prompt is {prompt_len} tokens and {whose} is {ctx} — shorten the conversation`.
- Tier: `LayerCache::with_budget` returns `BudgetTooSmall` when the budget cannot hold one layer (`budget_that_cannot_hold_one_layer_is_refused_up_front`).

## Procedure

Run from a clean work tree of `https://github.com/dcharlot-physicalai-bmi/ferric`. Record `git rev-parse HEAD` and `git status --porcelain` (empty). Do not point any command at a developer machine checkout outside this clone.

1. Build the tools the scripts already build: `cargo build -p ferric-llama --release --example lm_logits` and `cargo build -p ferric-llama --release --example pretokenizer_conformance`.
2. Tokenization: `scripts/tokenizer_conformance.sh`. `llama-tokenize` must be on `PATH`. Exit 2 because it is missing is a withhold, not a pass.
3. Logits: for every fixture `crates/ferric-llama/tests/fixtures/lm/*.json`, `scripts/lm_conformance.sh <exact-class.gguf> <fixture.json>`. The GGUF is the authors' weights at the dtype the fixture records.
4. Generation: the independent reference's greedy ids for the same prompt, same `n`, same weights, compared as ids. Keep the logit fixture's top-k and the sum of squares in the logit stage.
5. Malformed, in a temporary directory, three byte strings fed to `ferric_gguf::parse`: four bytes that are not `GGUF`; a `GGUF` header cut short; a header whose tensor offset plus extent passes the buffer. Expect the three named errors above. A parse that returns `Ok` fails the stage.
6. Resource bounds: `cargo test -p ferric-tier --lib budget_that_cannot_hold_one_layer_is_refused_up_front` and a server request whose prompt length is greater than or equal to that model's context. The response names both numbers.
7. KV exactness, on one architecture the runtime caches: the cached-decode logits and the full recompute, max abs = 0. Record the observed max abs even when it is not zero. A non-zero observation fails this stage and stays in the receipt.

Write one measured receipt with `tolerance.id` = `A.full-model.v1` and a `stages` array of six names: `tokenization`, `import`, `logits`, `generation`, `malformed`, `resource_bounds`. Each stage's `result` is `pass` only when that stage met its bar. `quantized_file_used_as_math_proof` is false.

## Receipt

Schema: `docs/acceptance/schemas/receipt.schema.json`. Checker: `cargo test -p ferric-accept` and `cargo run -p ferric-accept --example check`.

A pass requires `evidence_class` `measured`, `git.dirty` false, a 40-character revision, the command array actually run, `comparator.independent` true, and at least one artifact sha256 of the raw log. The six stages are all present and `pass`.

## Withhold

Withhold, do not pass, when `llama-tokenize` is absent, the authors' snapshot is absent, the meter or GPU is the wrong question for this track, or any stage was not run. Withhold is `status: withheld` and `outcome: inconclusive`. It is not a zero error and not an empty green test.

## Downgrade refusal

The EXACT logit band is not loosened to the rounded-file band to admit a quantized checkpoint. The KV mission bar is not loosened from 0 to the example's 2e-3. Greedy identity is not accepted in place of the logit row. A toy `tiny_llm` assert of 2e-3 is not a registered-architecture result. A later edit that widens a number in `scripts/lm_conformance.sh` and attaches a pass in the same change is a looser bar, and this protocol refuses it.
