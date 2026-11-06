# Generated standalone reproducer. Install torch and the target library.
# Recorded environment: {'type': 'metadata', 'schema_version': 1, 'timestamp_utc': '2026-11-06T05:08:00.740575+00:00', 'versions': {'torch': '2.14.0', 'vllm': '0.33.0', 'flashinfer-python': '0.7.2.post1', 'vllm-flash-attn': None}, 'gpu': 'NVIDIA H100 NVL', 'driver': '595.99.02', 'cuda_runtime': '13.0', 'env': {'VLLM_BATCH_INVARIANT': '1', 'CUBLAS_WORKSPACE_CONFIG': None, 'CUDA_VISIBLE_DEVICES': None}, 'torch_flags': {'matmul_allow_tf32': False, 'cudnn_allow_tf32': True, 'matmul_fp32_precision': 'none', 'cudnn_deterministic': False, 'cudnn_benchmark': False, 'float32_matmul_precision': 'highest'}}
import os
os.environ.update({'VLLM_BATCH_INVARIANT': '1'})
import torch
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.allow_tf32 = False
REFERENCE = {'batch_size': 1, 'probe_index': 0, 'seed': 1729, 'other_kv_length': 32, 'probe_kv_length': 32, 'chunk_size': 0, 'block_size': 16, 'num_splits': 0, 'fixed_split_size': 0, 'disable_split_kv': False, 'tf32': False, 'graph_capture_kv_length': 0}
CANDIDATE = {'batch_size': 2, 'probe_index': 0, 'seed': 1729, 'other_kv_length': 32, 'probe_kv_length': 32, 'chunk_size': 0, 'block_size': 16, 'num_splits': 0, 'fixed_split_size': 0, 'disable_split_kv': False, 'tf32': False, 'graph_capture_kv_length': 0}
CASE_NAME = 'mm/float32/K4096xN4096'
"""Small, import-safe target API and bitwise comparison machinery."""

from dataclasses import asdict, dataclass, field, replace
from typing import Callable, Mapping

import torch


@dataclass(frozen=True)
class BatchSpec:
    batch_size: int = 1
    probe_index: int = 0
    seed: int = 1729
    other_kv_length: int = 32
    probe_kv_length: int = 32
    chunk_size: int = 0
    block_size: int = 16
    num_splits: int = 0
    fixed_split_size: int = 0
    disable_split_kv: bool = False
    tf32: bool = False
    graph_capture_kv_length: int = 0

    def __post_init__(self) -> None:
        if self.batch_size < 1 or not 0 <= self.probe_index < self.batch_size:
            raise ValueError("probe_index must be inside a nonempty batch")

    def json(self) -> dict[str, int | bool]:
        return asdict(self)


@dataclass
class Case:
    target: str
    name: str
    dtype: str
    shapes: Mapping[str, object]
    run: Callable[[BatchSpec], torch.Tensor]
    perturbations: tuple[str, ...] = ("repeat", "batch", "position")
    repro_source: str = ""
    baseline: BatchSpec = field(default_factory=BatchSpec)
    allowed_block_sizes: tuple[int, ...] = (8, 16, 32, 64)


@dataclass(frozen=True)
class Difference:
    equal: bool
    differing_elements: int
    max_abs_diff: float | str
    max_ulp_distance: int


def _bits(t: torch.Tensor) -> torch.Tensor:
    """Same-width integer view; compares NaN payloads and signed zero too."""
    if t.dtype in (torch.float16, torch.bfloat16, torch.int16):
        return t.contiguous().view(torch.int16)
    if t.dtype in (torch.float32, torch.int32):
        return t.contiguous().view(torch.int32)
    if t.dtype in (torch.float64, torch.int64):
        return t.contiguous().view(torch.int64)
    if t.dtype in (torch.int8, torch.uint8, torch.bool):
        return t.contiguous().view(torch.uint8)
    raise TypeError(f"unsupported comparison dtype: {t.dtype}")


def compare(reference: torch.Tensor, candidate: torch.Tensor) -> Difference:
    if reference.shape != candidate.shape or reference.dtype != candidate.dtype:
        raise ValueError(f"probe output shape/dtype changed: {reference.shape}/{reference.dtype} vs {candidate.shape}/{candidate.dtype}")
    a, b = reference.detach().cpu(), candidate.detach().cpu()
    ab, bb = _bits(a), _bits(b)
    diff = ab != bb
    count = int(diff.sum().item())
    if not count:
        return Difference(True, 0, 0.0, 0)
    absdiff = (a.double() - b.double()).abs()
    # NaN payload differences have no meaningful finite absolute error.
    finite = absdiff[torch.isfinite(absdiff)]
    max_abs: float | str = ("inf" if torch.isinf(absdiff).any() else
                            float(finite.max().item()) if finite.numel() else "nan")
    width = ab.element_size() * 8
    mask = (1 << width) - 1
    sign = 1 << (width - 1)
    av = [int(x) & mask for x in ab.reshape(-1).tolist()]
    bv = [int(x) & mask for x in bb.reshape(-1).tolist()]
    def ordered(v: int) -> int:
        return ((~v) & mask) if v & sign else (v | sign)
    ulp = max(abs(ordered(x) - ordered(y)) for x, y in zip(av, bv))
    return Difference(False, count, max_abs, ulp)


def seeded(shape: tuple[int, ...], spec: BatchSpec, row: int, dtype: torch.dtype, device: str = "cuda", salt: int = 0) -> torch.Tensor:
    """Probe data is stable across batch positions; other rows vary by slot."""
    g = torch.Generator(device="cpu").manual_seed(spec.seed + salt + (0 if row == spec.probe_index else 10000 + row))
    return torch.randn(shape, generator=g, dtype=torch.float32).to(device=device, dtype=dtype)


def probe_specs(case: Case, quick: bool = False) -> list[tuple[str, BatchSpec]]:
    base = case.baseline
    sizes = (1, 2, 3, 8, 16) if quick else (1, 2, 3, 7, 8, 15, 16, 31, 64, 128, 257)
    specs: list[tuple[str, BatchSpec]] = []
    if "repeat" in case.perturbations:
        specs += [("repeat", base) for _ in range(2 if quick else 5)]
    if "batch" in case.perturbations:
        specs += [("batch", replace(base, batch_size=n, probe_index=0)) for n in sizes if n != base.batch_size]
    if "tile_m" in case.perturbations:
        ms = (31, 32, 33, 64, 65) if quick else (31, 32, 33, 63, 64, 65, 127, 128, 129, 255, 256, 257)
        specs += [("tile_m", replace(base, batch_size=n, probe_index=0)) for n in ms]
    if "big_m" in case.perturbations:
        specs += [("big_m", replace(base, batch_size=n, probe_index=0)) for n in ((512, 1024) if quick else (500, 512, 1000, 1024, 2048, 4096))]
    if "position" in case.perturbations:
        specs += [("position", replace(base, batch_size=n, probe_index=p)) for n in sizes if n > 1 for p in sorted({n // 2, n - 1})]
    if "other_kv_length" in case.perturbations:
        specs += [("other_kv_length", replace(base, batch_size=3, probe_index=1, other_kv_length=k)) for k in ((1, 65) if quick else (1, 7, 31, 33, 65, 129, 257, 1025, 3000))]
    if "chunk_size" in case.perturbations:
        specs += [("chunk_size", replace(base, chunk_size=k)) for k in ((1,) if quick else (1, 2, 7))]
    if "block_size" in case.perturbations:
        specs += [("block_size", replace(base, block_size=k)) for k in ((8, 32) if quick else (8, 16, 32, 64)) if k != base.block_size and k in case.allowed_block_sizes]
    if "num_splits" in case.perturbations:
        specs += [("num_splits", replace(base, num_splits=k)) for k in ((1, 2) if quick else (1, 2, 4, 8)) if k != base.num_splits]
    if "fixed_split_size" in case.perturbations:
        specs += [("fixed_split_size", replace(base, fixed_split_size=k)) for k in ((32, 128) if quick else (16, 32, 128, 256, 512)) if k != base.fixed_split_size]
    if "tf32" in case.perturbations:
        specs.append(("tf32", replace(base, tf32=not base.tf32)))
    if "graph_replay" in case.perturbations:
        specs.append(("graph_replay", replace(base, batch_size=3, probe_index=0,
                                              graph_capture_kv_length=max(1, base.probe_kv_length // 2))))
    return specs
class SkipCase(Exception): pass
"""Plain torch ops after vLLM's init_batch_invariance() (what a vLLM worker really runs).

On Hopper vLLM does not override aten::mm; it relies on cuBLAS workspace settings.
Hence this target calls torch.mm / F.linear / torch.softmax ... after
init_batch_invariance() instead of calling the Triton kernels directly.
"""

import os

import torch
import torch.nn.functional as F


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
case = next(c for c in cases() if c.name == CASE_NAME)
a = case.run(BatchSpec(**REFERENCE)); b = case.run(BatchSpec(**CANDIDATE))
print('bitwise equal:', compare(a, b).equal)
print('max abs diff:', compare(a, b).max_abs_diff)
