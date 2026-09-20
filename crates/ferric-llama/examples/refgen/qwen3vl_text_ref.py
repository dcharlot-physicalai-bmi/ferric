#!/usr/bin/env python3
"""Emit the last-token logit vector for `examples/qwen3vl_hf.rs` (the TEXT path).

⛔ IN THE REPO BECAUSE THIS FILE WAS LOST ONCE. The original generator and its `qwen3vl_ref.bin`
lived in a scratch directory that was wiped between sessions, leaving a shipped check that could not
be re-run. Regenerating it needs a machine with torch + transformers and the checkpoint.

⚠ The DEFAULT prompt cannot see what the qwen3vl port is about. The llama.cpp/HF divergence at
rotary sectors 61/62 scales with position (~1.6e-6 rad at position 4, ~4.2e-4 at 1024), so a
5-token prompt puts it two orders of magnitude under f32 noise. Pass a long id list to reach the
operating point where the difference is observable.

    python3 qwen3vl_text_ref.py <out.bin> [ids.csv]
"""
import struct, sys, torch
from transformers import AutoModel

out = sys.argv[1]
ids = ([int(x) for x in open(sys.argv[2]).read().strip().split(",")]
       if len(sys.argv) > 2 else [785, 6722, 315, 9625, 374])   # "The capital of France is"

m = AutoModel.from_pretrained("Qwen/Qwen3-VL-Embedding-2B", local_files_only=True,
                              dtype=torch.float32).eval()
base = m.model if hasattr(m, "model") else m
lm = base.language_model
with torch.no_grad():
    h = lm(input_ids=torch.tensor([ids], dtype=torch.long)).last_hidden_state
    # the head is tied to the embedding table on this checkpoint
    w = base.get_input_embeddings().weight
    logits = (h[0, -1].float() @ w.float().T)

v = logits.contiguous()
with open(out, "wb") as f:
    f.write(struct.pack("<I", v.numel()))
    f.write(v.numpy().astype("<f4").tobytes())
print(f"wrote {out}: {v.numel()} logits for {len(ids)} tokens (last position {len(ids)-1})")
print("argmax:", int(v.argmax()), "| top5:", [int(i) for i in v.topk(5).indices])
