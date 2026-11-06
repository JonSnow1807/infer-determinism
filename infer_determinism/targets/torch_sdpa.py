"""Forced PyTorch SDPA backend probes."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from ..core import BatchSpec, Case, seeded
from ..runner import SkipCase

BACKENDS = ("FLASH_ATTENTION", "EFFICIENT_ATTENTION", "CUDNN_ATTENTION", "MATH")


def _run(spec: BatchSpec, dtype: torch.dtype, backend_name: str) -> torch.Tensor:
    if not torch.cuda.is_available():
        raise SkipCase("CUDA is required for torch_sdpa")
    from torch.nn.attention import SDPBackend, sdpa_kernel
    backend = getattr(SDPBackend, backend_name, None)
    if backend is None:
        raise SkipCase(f"SDPA backend {backend_name} is unavailable")
    ql = 32 if backend_name in ("FLASH_ATTENTION", "CUDNN_ATTENTION") else 8  # flash rejects is_causal with Q != K
    q = torch.stack([seeded((4, ql, 64), spec, i, dtype, salt=31) for i in range(spec.batch_size)])
    k = torch.stack([seeded((4, 32, 64), spec, i, dtype, salt=37) for i in range(spec.batch_size)])
    v = torch.stack([seeded((4, 32, 64), spec, i, dtype, salt=41) for i in range(spec.batch_size)])
    try:
        with sdpa_kernel(backend):
            return F.scaled_dot_product_attention(q, k, v, is_causal=True)[spec.probe_index].clone()
    except RuntimeError as exc:
        if "No available kernel" in str(exc) or "not supported" in str(exc):
            raise SkipCase(f"{backend_name}: {exc}") from exc
        raise


REPRO = '''
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
def run(spec):
    dtype_name, backend_name = CASE_NAME.split("/")
    dtype = getattr(torch, dtype_name)
    def seeded(shape, row, salt):
        g = torch.Generator(device="cpu").manual_seed(spec["seed"] + salt + (0 if row == spec["probe_index"] else 10000 + row))
        return torch.randn(shape, generator=g).to(device="cuda", dtype=dtype)
    ql = 32 if backend_name in ("FLASH_ATTENTION", "CUDNN_ATTENTION") else 8
    q = torch.stack([seeded((4,ql,64), i, 31) for i in range(spec["batch_size"])])
    k = torch.stack([seeded((4,32,64), i, 37) for i in range(spec["batch_size"])])
    v = torch.stack([seeded((4,32,64), i, 41) for i in range(spec["batch_size"])])
    with sdpa_kernel(getattr(SDPBackend, backend_name)):
        return F.scaled_dot_product_attention(q,k,v,is_causal=True)[spec["probe_index"]].clone()
a,b=run(REFERENCE),run(CANDIDATE)
bits=torch.int16 if a.element_size()==2 else torch.int32
print("bitwise equal:",torch.equal(a.contiguous().view(bits),b.contiguous().view(bits)))
print("max abs diff:",(a.float()-b.float()).abs().max().item())
'''


def cases() -> list[Case]:
    if not torch.cuda.is_available():
        raise SkipCase("CUDA is required for torch_sdpa")
    return [Case("torch_sdpa", f"{str(dtype).split('.')[-1]}/{backend}", str(dtype), {"heads": 4, "Q": 32 if backend in ("FLASH_ATTENTION", "CUDNN_ATTENTION") else 8, "K": 32, "D": 64}, lambda spec, d=dtype, b=backend: _run(spec, d, b), repro_source=REPRO) for dtype in (torch.bfloat16, torch.float16, torch.float32) for backend in BACKENDS]
