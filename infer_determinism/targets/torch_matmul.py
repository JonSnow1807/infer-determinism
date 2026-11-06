"""PyTorch GEMM and linear probes around M tile boundaries."""
from __future__ import annotations

from dataclasses import replace

import torch
import torch.nn.functional as F

from ..core import BatchSpec, Case, seeded
from ..runner import SkipCase


def _run(spec: BatchSpec, dtype: torch.dtype, op: str, compiled: bool) -> torch.Tensor:
    if not torch.cuda.is_available():
        raise SkipCase("CUDA is required for torch_matmul")
    if compiled and not hasattr(torch, "compile"):
        raise SkipCase("torch.compile is unavailable")
    x = torch.stack([seeded((128,), spec, i, dtype, salt=11) for i in range(spec.batch_size)])
    weight = seeded((128, 128), spec, spec.probe_index, dtype, salt=29)
    fn = (lambda a, b: torch.mm(a, b)) if op == "mm" else (lambda a, b: F.linear(a, b))
    if compiled:
        fn = torch.compile(fn)
    previous = torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = spec.tf32
        result = fn(x, weight)
        return result[spec.probe_index].clone()
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous


REPRO = '''
import torch.nn.functional as F
def run(spec):
    dtype_name, op, compiled = CASE_NAME.split("/")
    dtype = getattr(torch, dtype_name)
    def seeded(shape, row, salt):
        g = torch.Generator(device="cpu").manual_seed(spec["seed"] + salt + (0 if row == spec["probe_index"] else 10000 + row))
        return torch.randn(shape, generator=g).to(device="cuda", dtype=dtype)
    x = torch.stack([seeded((128,), i, 11) for i in range(spec["batch_size"])])
    w = seeded((128, 128), spec["probe_index"], 29)
    fn = (lambda a, b: torch.mm(a, b)) if op == "mm" else (lambda a, b: F.linear(a, b))
    if compiled == "compile": fn = torch.compile(fn)
    torch.backends.cuda.matmul.allow_tf32 = spec["tf32"]
    return fn(x, w)[spec["probe_index"]].clone()
a, b = run(REFERENCE), run(CANDIDATE)
print("bitwise equal:", torch.equal(a.contiguous().view(torch.int16 if a.element_size()==2 else torch.int32), b.contiguous().view(torch.int16 if b.element_size()==2 else torch.int32)))
print("max abs diff:", (a.float()-b.float()).abs().max().item())
'''


def cases() -> list[Case]:
    if not torch.cuda.is_available():
        raise SkipCase("CUDA is required for torch_matmul")
    output = []
    for dtype in (torch.bfloat16, torch.float16, torch.float32):
        for op in ("mm", "linear"):
            for compiled in (False, True):
                name = f"{str(dtype).split('.')[-1]}/{op}/{'compile' if compiled else 'eager'}"
                output.append(Case("torch_matmul", name, str(dtype), {"K": 128, "N": 128}, lambda spec, d=dtype, o=op, c=compiled: _run(spec, d, o, c), ("repeat", "batch", "position", "tile_m", "tf32") if dtype == torch.float32 else ("repeat", "batch", "position", "tile_m"), REPRO))
    return output
