"""Plain torch ops after vLLM's init_batch_invariance() (what a vLLM worker really runs).

On Hopper vLLM does not override aten::mm; it relies on cuBLAS workspace settings.
Hence this target calls torch.mm / F.linear / torch.softmax ... after
init_batch_invariance() instead of calling the Triton kernels directly.
"""
from __future__ import annotations

import os

import torch
import torch.nn.functional as F

from ..core import BatchSpec, Case
from ..runner import SkipCase

_READY = False


def _init() -> None:
    global _READY
    if not torch.cuda.is_available():
        raise SkipCase("CUDA is required")
    if os.environ.get("VLLM_BATCH_INVARIANT") != "1":
        raise SkipCase("set VLLM_BATCH_INVARIANT=1 before importing vLLM")
    if _READY:
        return
    try:
        from vllm.model_executor.determinism.batch_invariant import init_batch_invariance
    except ImportError as exc:
        raise SkipCase(f"vLLM init_batch_invariance unavailable: {exc}") from exc
    init_batch_invariance()
    _READY = True


_CACHE: dict = {}


def _gpu(shape, spec: BatchSpec, row: int, dtype: torch.dtype, salt: int) -> torch.Tensor:
    """Like core.seeded but generated on the GPU (fast for large shapes)."""
    g = torch.Generator(device="cuda").manual_seed(spec.seed + salt + (0 if row == spec.probe_index else 10000 + row))
    return torch.randn(shape, generator=g, device="cuda", dtype=torch.float32).to(dtype)


def _weight(shape, spec: BatchSpec, dtype: torch.dtype, salt: int) -> torch.Tensor:
    key = (shape, spec.seed, spec.probe_index, dtype, salt)
    if key not in _CACHE:
        _CACHE.clear()
        _CACHE[key] = _gpu(shape, spec, spec.probe_index, dtype, salt)
    return _CACHE[key]


def _run(spec: BatchSpec, op: str, dtype: torch.dtype, k: int, n: int) -> torch.Tensor:
    _init()
    M = spec.batch_size
    if op in ("softmax", "log_softmax", "mean", "rms_norm"):
        x = torch.stack([_gpu((k,), spec, i, dtype, 101) for i in range(M)])
        if op == "softmax":
            y = torch.softmax(x, dim=-1)
        elif op == "log_softmax":
            y = torch.log_softmax(x, dim=-1)
        elif op == "mean":
            y = torch.mean(x, dim=-1)
        else:
            from vllm.model_executor.determinism.batch_invariant import rms_norm_batch_invariant
            w = _weight((k,), spec, dtype, 103)
            y = rms_norm_batch_invariant(x, w, 1e-6)
        return y[spec.probe_index].clone()
    x = torch.stack([_gpu((k,), spec, i, dtype, 107) for i in range(M)])
    w = _weight((n, k), spec, dtype, 109) * 0.02
    if op == "mm":
        y = torch.mm(x, w.t())
    elif op == "linear":
        y = F.linear(x, w)
    elif op == "linear_bias":
        y = F.linear(x, w, _weight((n,), spec, dtype, 113))
    elif op == "matmul3d":
        y = torch.matmul(x.unsqueeze(0), w.t())[0]
    else:
        raise ValueError(op)
    return y[spec.probe_index].clone()


def cases() -> list[Case]:
    _init()
    out: list[Case] = []
    bf, f16, f32 = torch.bfloat16, torch.float16, torch.float32
    gemm = [(4096, 4096), (4096, 14336), (14336, 4096), (5120, 5120), (4096, 1024), (2048, 128256 // 8)]
    for op in ("mm", "linear", "linear_bias", "matmul3d"):
        for dtype in (bf, f16, f32) if op == "mm" else (bf,):
            for k, n in gemm if dtype is bf else gemm[:2]:
                out.append(Case("vllm_invariant_mode", f"{op}/{str(dtype).split('.')[-1]}/K{k}xN{n}", str(dtype), {"K": k, "N": n},
                                lambda spec, o=op, d=dtype, k=k, n=n: _run(spec, o, d, k, n),
                                ("repeat", "batch", "position", "tile_m", "big_m")))
    for op, sizes in (("softmax", (4096, 32000, 128256)), ("log_softmax", (4096, 32000, 128256)), ("mean", (4096, 5120, 11008)), ("rms_norm", (4096, 5120, 8192))):
        for dtype in (bf, f32):
            for k in sizes:
                out.append(Case("vllm_invariant_mode", f"{op}/{str(dtype).split('.')[-1]}/D{k}", str(dtype), {"D": k},
                                lambda spec, o=op, d=dtype, k=k: _run(spec, o, d, k, 0),
                                ("repeat", "batch", "position", "tile_m") + (("big_m",) if k < 100000 else ())))
    return out
