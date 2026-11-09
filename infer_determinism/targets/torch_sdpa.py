"""Forced PyTorch SDPA backend probes."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from ..core import BatchSpec, Case, seeded
from ..runner import SkipCase
from ._attention import AttentionConfig, GQA_HEADS, attention_call, require_keywords

BACKENDS = ("FLASH_ATTENTION", "EFFICIENT_ATTENTION", "CUDNN_ATTENTION", "MATH")


def _run(spec: BatchSpec, dtype: torch.dtype, backend_name: str, config: AttentionConfig = AttentionConfig()) -> torch.Tensor:
    if not torch.cuda.is_available():
        raise SkipCase("CUDA is required for torch_sdpa")
    from torch.nn.attention import SDPBackend, sdpa_kernel
    backend = getattr(SDPBackend, backend_name, None)
    if backend is None:
        raise SkipCase(f"SDPA backend {backend_name} is unavailable")
    gqa = config.num_q_heads != config.num_kv_heads
    if gqa:
        require_keywords(F.scaled_dot_product_attention, ("enable_gqa",), backend_name)
    ql = 32 if backend_name in ("FLASH_ATTENTION", "CUDNN_ATTENTION") else 8  # flash rejects is_causal with Q != K
    q = torch.stack([seeded((config.num_q_heads, ql, config.head_dim), spec, i, dtype, salt=31) for i in range(spec.batch_size)])
    k = torch.stack([seeded((config.num_kv_heads, 32, config.head_dim), spec, i, dtype, salt=37) for i in range(spec.batch_size)])
    v = torch.stack([seeded((config.num_kv_heads, 32, config.head_dim), spec, i, dtype, salt=41) for i in range(spec.batch_size)])
    with sdpa_kernel(backend):
        return attention_call(F.scaled_dot_product_attention, q, k, v,
                              is_causal=True, **({"enable_gqa": True} if gqa else {}),
                              context=f"SDPA {backend_name}/{config.label}")[spec.probe_index].clone()


def cases() -> list[Case]:
    if not torch.cuda.is_available():
        raise SkipCase("CUDA is required for torch_sdpa")
    out = [Case("torch_sdpa", f"{str(dtype).split('.')[-1]}/{backend}", str(dtype), {"heads": 4, "Q": 32 if backend in ("FLASH_ATTENTION", "CUDNN_ATTENTION") else 8, "K": 32, "D": 64}, lambda spec, d=dtype, b=backend: _run(spec, d, b)) for dtype in (torch.bfloat16, torch.float16, torch.float32) for backend in BACKENDS]
    for hq, hkv in GQA_HEADS:
        config = AttentionConfig(hq, hkv, 128)
        for backend in BACKENDS:
            out.append(Case("torch_sdpa", f"bfloat16/{backend}/{config.label}", "bfloat16",
                            {**config.shapes(), "Q": 32 if backend in ("FLASH_ATTENTION", "CUDNN_ATTENTION") else 8,
                             "K": 32, "full_only": hq != 32},
                            lambda spec, b=backend, c=config: _run(spec, torch.bfloat16, b, c)))
    return out
