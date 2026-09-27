"""`flash_attn.flash_attn_varlen_func` in plain torch — a shim so the authors' MiMo-Audio-Tokenizer runs
without CUDA.

WHY. Xiaomi's `modeling_audio_tokenizer.py` imports `flash_attn_varlen_func` unconditionally and calls it
for every attention in the encoder. That is a KERNEL, not a model choice: packed variable-length attention
where `cu_seqlens` marks independent sequences in one row block. The arithmetic it computes is ordinary
scaled dot-product attention inside each sequence, which is what this does, one sequence at a time: scores
and softmax in float32 and P@V at the inputs' dtype, as the kernel does.

⚠ The kernel only accepts fp16/bf16, so at float64 there is no "authors' behaviour" to copy: a float64
noise-floor run gets float32 attention scores from this shim. That is the KERNEL's arithmetic carried over,
not a cast in the authors' Python; it bounds how far below float32 that floor can reach in the encoder.

⛔ Only the configuration the encoder uses is implemented: no causal mask, no window (`window_size ==
(-1, -1)`), no dropout, the default scale 1/sqrt(head_dim), equal q/k boundaries. Anything else REFUSES —
a shim that quietly ignored `causal=True` or a window would compute a different model and nothing would
say so. The reference generator records that this shim ran.
"""
import math

import torch


def flash_attn_varlen_func(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
                           dropout_p=0.0, softmax_scale=None, causal=False, window_size=(-1, -1), **kw):
    if causal:
        raise NotImplementedError("flash_attn shim: causal=True is not implemented — refusing")
    if tuple(window_size) != (-1, -1):
        raise NotImplementedError(f"flash_attn shim: window_size={window_size} is not implemented — refusing")
    if dropout_p:
        raise NotImplementedError("flash_attn shim: dropout is not implemented — refusing")
    if kw:
        raise NotImplementedError(f"flash_attn shim: unexpected arguments {sorted(kw)} — refusing")
    if not torch.equal(cu_seqlens_q, cu_seqlens_k):
        raise NotImplementedError("flash_attn shim: different q/k boundaries — refusing")
    scale = softmax_scale if softmax_scale is not None else 1.0 / math.sqrt(q.shape[-1])
    out = torch.empty_like(q)
    b = cu_seqlens_q.tolist()
    for s, e in zip(b[:-1], b[1:]):
        qs, ks, vs = (t[s:e].transpose(0, 1) for t in (q, k, v))          # [heads, n, d]
        # As the kernel does: scores and softmax in float32 whatever the input dtype, P back to it for P@V.
        att = torch.softmax((qs.float() @ ks.float().transpose(1, 2)) * scale, dim=-1).to(vs.dtype)
        out[s:e] = (att @ vs).transpose(0, 1)
    return out
