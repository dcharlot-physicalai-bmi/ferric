"""mamba_ssm's gated RMSNorm, as the KERNEL AUTHORS' OWN PURE-TORCH REFERENCE — a shim, not the package.

WHY. NVIDIA's Nemotron-H repo ships its own `modeling_nemotron_h.py`, the authors' implementation in the
most literal sense. It imports `rmsnorm_fn` from mamba_ssm's Triton kernels unconditionally, even on its
pure-torch path, so it cannot run without CUDA. `rms_norm_ref` below is copied VERBATIM from the same
upstream file, where it is the ground truth the Triton kernel is tested against. `rmsnorm_fn` calls it
with the kernel's own convention (upcast=True). Nothing else in mamba_ssm is provided: with this
directory on sys.path, `import mamba_ssm` resolves here, the package has no distribution metadata,
so transformers reports the fast path unavailable and the model takes its torch path.

Source: https://github.com/state-spaces/mamba/blob/e9594ce1c732d97440f0332fdc43170a2294dbfa/mamba_ssm/ops/triton/layernorm_gated.py
Copyright (c) 2024, Tri Dao. Licensed under the Apache License, Version 2.0.
"""
import torch
import torch.nn.functional as F


def rearrange(t, pattern, d=None):
    """The two einops patterns rms_norm_ref uses, and nothing else, so it runs without einops."""
    if pattern == "... (g d) -> ... g d":
        return t.reshape(*t.shape[:-1], t.shape[-1] // d, d)
    if pattern == "... g d -> ... (g d)":
        return t.reshape(*t.shape[:-2], t.shape[-2] * t.shape[-1])
    raise NotImplementedError(pattern)


def rms_norm_ref(x, weight, bias, z=None, eps=1e-6, group_size=None, norm_before_gate=True, upcast=True):
    dtype = x.dtype
    N = x.shape[-1]
    weight = weight.float()
    bias = bias.float() if bias is not None else None
    if upcast:
        x = x.float()
        z = z.float() if z is not None else z
    if z is not None and not norm_before_gate:
        x = x * F.silu(z)
    if group_size is None:
        rstd = 1 / torch.sqrt((x.square()).mean(dim=-1, keepdim=True) + eps)
        out = (x * rstd * weight) + bias if bias is not None else (x * rstd * weight)
    else:
        x_group = rearrange(x, "... (g d) -> ... g d", d=group_size)
        rstd = 1 / torch.sqrt((x_group.square()).mean(dim=-1, keepdim=True) + eps)
        out = rearrange(x_group * rstd, "... g d -> ... (g d)") * weight
        if bias is not None:
            out = out + bias
    if z is not None and norm_before_gate:
        out *= F.silu(z)
    return out.to(dtype)


def rmsnorm_fn(x, weight, bias, z=None, eps=1e-6, group_size=None, norm_before_gate=True):
    # The Triton LayerNormFn is called with is_rms_norm=True and computes in float32 — upcast=True.
    return rms_norm_ref(x, weight, bias, z=z, eps=eps, group_size=group_size,
                        norm_before_gate=norm_before_gate, upcast=True)
