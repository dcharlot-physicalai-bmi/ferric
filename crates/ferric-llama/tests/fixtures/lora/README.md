# LoRA conformance fixtures

Checked by `scripts/lora_conformance.sh` against **Hugging Face PEFT** (the library that defines the adapter
format and its math) over the model authors' own base weights. Generated 2026-09-27 with PEFT 0.21.0,
transformers 5.17.0, torch 2.14.0 (CPU), by `crates/ferric-llama/examples/refgen/lora_ref.py`.

| file | what it is |
|---|---|
| `syn_a/` | PEFT adapter made **by PEFT** on `Qwen/Qwen2.5-0.5B-Instruct`: seed 1, r=2, alpha=4 (scaling 2), all seven projections, layers 0 and 23, `init_lora_weights=False` with lora_B x0.1 (nonzero B — PEFT's default zero B is an adapter every loader reproduces), stored float16 |
| `syn_a.gguf` | `syn_a` through llama.cpp's own `convert_lora_to_gguf.py --outtype f16` (llama.cpp `0cea36222`) — the peer format |
| `syn_b/` | seed 2, **rslora** (alpha/sqrt(r) = 6/2 = 3), r=4, q/v/o/down on layers 5 and 17, float16 |
| `ref.json.gz` | PEFT's logits for a 139-token text with `syn_a` on and off, float32 and float64; and PEFT's mixed batch `forward(..., adapter_names=["a","b","__base__"])` over three 40-token sequences |
| `llama_qk/`, `llama_qk.gguf`, `llama_qk.ref.json.gz` | the same for `unsloth/Llama-3.2-1B-Instruct` (seed 3, r=2, alpha=4, q/k/v/o, layers 0 and 15) — a base whose GGUF q/k rows are **permuted** by the converter, so a PEFT adapter's q/k `lora_B` must be too |
| `taronklm-lora-chatbot.ref.json.gz` | PEFT's reference for a PUBLISHED adapter, `taronklm/Qwen2.5-0.5B-Instruct-lora-chatbot` (Apache-2.0; r=4, alpha=32, all seven projections, float32). The adapter itself is not redistributed here |

Regenerate (a venv with torch, transformers, peft; `HF_HUB_OFFLINE=1` once the models are cached):

    R=crates/ferric-llama/examples/refgen/lora_ref.py; FX=crates/ferric-llama/tests/fixtures/lora
    python $R make Qwen/Qwen2.5-0.5B-Instruct $FX/syn_a --seed 1 --r 2 --alpha 4 --layers 0,23 --b-scale 0.1 --f16
    python $R make Qwen/Qwen2.5-0.5B-Instruct $FX/syn_b --seed 2 --r 4 --alpha 6 --rslora \
        --targets q_proj,v_proj,o_proj,down_proj --layers 5,17 --b-scale 0.1 --f16
    python <llama.cpp>/convert_lora_to_gguf.py --base <Qwen2.5-0.5B-Instruct snapshot> --outtype f16 \
        --outfile $FX/syn_a.gguf $FX/syn_a
    python $R ref Qwen/Qwen2.5-0.5B-Instruct $FX/ref.json.gz $FX/syn_a $FX/syn_b
    python $R make unsloth/Llama-3.2-1B-Instruct $FX/llama_qk --seed 3 --r 2 --alpha 4 \
        --targets q_proj,k_proj,v_proj,o_proj --layers 0,15 --b-scale 0.1 --f16
    python <llama.cpp>/convert_lora_to_gguf.py --base <Llama-3.2-1B-Instruct snapshot> --outtype f16 \
        --outfile $FX/llama_qk.gguf $FX/llama_qk
    python $R ref unsloth/Llama-3.2-1B-Instruct $FX/llama_qk.ref.json.gz $FX/llama_qk
    python $R ref Qwen/Qwen2.5-0.5B-Instruct $FX/taronklm-lora-chatbot.ref.json.gz <taronklm snapshot>

(`make` also writes PEFT's auto-generated model card `README.md` into the adapter directory; it is not kept.)

Run:

    scripts/lora_conformance.sh ~/.cache/ferric/hub/qwen2.5-0.5b-F32-authors.gguf
    scripts/lora_conformance.sh ~/.cache/ferric/hub/llama-3.2-1b-F32-authors.gguf $FX/llama_qk $FX/llama_qk.ref.json.gz
    scripts/lora_conformance.sh ~/.cache/ferric/hub/qwen2.5-0.5b-F32-authors.gguf \
        ~/.cache/huggingface/hub/models--taronklm--Qwen2.5-0.5B-Instruct-lora-chatbot/snapshots/<rev> \
        $FX/taronklm-lora-chatbot.ref.json.gz

⛔ `llama.cpp`'s GGUF adapter carries no rslora flag: its converter writes `lora_alpha` as `adapter.lora.alpha`
and its loader scales by `alpha / rank`, so `syn_b` converted by that script says alpha 6 → 6/4 = 1.5 where PEFT
runs it at 3. Measured 2026-09-27 through Ferric's GGUF reader (which applies llama.cpp's `get_scale` formula; llama.cpp
itself was not run): 1.03 logits from PEFT, 5,744x the floor. Ferric's own export (`examples/lora_convert.rs`) writes
alpha 12 for the same adapter and lands at 0.84x the floor, as does the PEFT directory itself.
