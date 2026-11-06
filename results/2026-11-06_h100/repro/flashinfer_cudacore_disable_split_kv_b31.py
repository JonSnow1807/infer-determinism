# Generated standalone reproducer. Install torch and the target library.
# Recorded environment: {'type': 'metadata', 'schema_version': 1, 'timestamp_utc': '2026-11-06T05:13:55.968844+00:00', 'versions': {'torch': '2.14.0', 'vllm': '0.33.0', 'flashinfer-python': '0.7.2.post1', 'vllm-flash-attn': None}, 'gpu': 'NVIDIA H100 NVL', 'driver': '595.99.02', 'cuda_runtime': '13.0', 'env': {'VLLM_BATCH_INVARIANT': '1', 'CUBLAS_WORKSPACE_CONFIG': ':4096:8', 'CUDA_VISIBLE_DEVICES': None}, 'torch_flags': {'matmul_allow_tf32': False, 'cudnn_allow_tf32': True, 'matmul_fp32_precision': 'none', 'cudnn_deterministic': False, 'cudnn_benchmark': False, 'float32_matmul_precision': 'highest'}}
import os
os.environ.update({'VLLM_BATCH_INVARIANT': '1', 'CUBLAS_WORKSPACE_CONFIG': ':4096:8'})
import torch
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.allow_tf32 = True
REFERENCE = {'batch_size': 1, 'probe_index': 0, 'seed': 1729, 'other_kv_length': 1024, 'probe_kv_length': 1024, 'chunk_size': 0, 'block_size': 16, 'num_splits': 0, 'fixed_split_size': 0, 'disable_split_kv': True, 'tf32': False, 'graph_capture_kv_length': 0}
CANDIDATE = {'batch_size': 31, 'probe_index': 0, 'seed': 1729, 'other_kv_length': 1024, 'probe_kv_length': 1024, 'chunk_size': 0, 'block_size': 16, 'num_splits': 0, 'fixed_split_size': 0, 'disable_split_kv': True, 'tf32': False, 'graph_capture_kv_length': 0}
CASE_NAME = 'fa2_disable_split_kv_cudacore/batch_decode_long'
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
"""FlashInfer paged decode/prefill and single-request attention."""

import importlib
import math

import torch



def _library():
    if not torch.cuda.is_available():
        raise SkipCase("CUDA is required for FlashInfer")
    try:
        return importlib.import_module("flashinfer")
    except ImportError as exc:
        raise SkipCase(f"flashinfer-python is missing: {exc}") from exc


def _inputs(spec: BatchSpec, q_len: int):
    queries, keys, values = [], [], []
    for row in range(spec.batch_size):
        length = spec.probe_kv_length if row == spec.probe_index else spec.other_kv_length
        queries.append(seeded((q_len, 4, 64), spec, row, torch.bfloat16, salt=71))
        keys.append(seeded((length, 4, 64), spec, row, torch.bfloat16, salt=73))
        values.append(seeded((length, 4, 64), spec, row, torch.bfloat16, salt=79))
    return queries, keys, values


def _paged(keys, values, page_size: int):
    pages, indptr, last = [], [0], []
    for k, v in zip(keys, values):
        length = k.shape[0]
        n = math.ceil(length / page_size)
        for page in range(n):
            end = min((page + 1) * page_size, length)
            kp = torch.zeros((page_size, 4, 64), device="cuda", dtype=k.dtype)
            vp = torch.zeros_like(kp)
            kp[:end-page*page_size] = k[page*page_size:end]
            vp[:end-page*page_size] = v[page*page_size:end]
            pages.append(torch.stack((kp, vp)))
        indptr.append(indptr[-1] + n)
        last.append(length - (n-1)*page_size)
    cache = torch.stack(pages)
    ptr = torch.tensor(indptr, device="cuda", dtype=torch.int32)
    indices = torch.arange(len(pages), device="cuda", dtype=torch.int32)
    tail = torch.tensor(last, device="cuda", dtype=torch.int32)
    return cache, ptr, indices, tail


def _split_kwargs(spec: BatchSpec) -> dict:
    """fixed_split_size is in pages in FlashInfer; BatchSpec.fixed_split_size is in tokens."""
    kwargs: dict = {}
    pages = spec.fixed_split_size // spec.block_size
    if pages > 0:
        kwargs["fixed_split_size"] = pages
    if spec.disable_split_kv:
        kwargs["disable_split_kv"] = True
    return kwargs


def _run(spec: BatchSpec, kind: str, force_fa2: bool = False, tensor_cores: bool = True) -> torch.Tensor:
    fi = _library()
    q_len = 1 if "decode" in kind else 8
    queries, keys, values = _inputs(spec, q_len)
    if kind.startswith("single"):
        if spec.batch_size != 1:
            raise SkipCase("single-request kernel has no batch dimension")
        if kind == "single_decode":
            return fi.single_decode_with_kv_cache(queries[0][0], keys[0], values[0]).clone()
        if not spec.chunk_size:
            return fi.single_prefill_with_kv_cache(queries[0], keys[0], values[0], causal=True).clone()
        outputs = []
        prefix = spec.probe_kv_length - q_len
        for start in range(0, q_len, spec.chunk_size):
            end = min(start + spec.chunk_size, q_len)
            outputs.append(fi.single_prefill_with_kv_cache(queries[0][start:end], keys[0][:prefix+end], values[0][:prefix+end], causal=True))
        return torch.cat(outputs).clone()
    workspace = torch.zeros(1024 * 1024 * 1024, device="cuda", dtype=torch.uint8)
    cache, ptr, indices, tail = _paged(keys, values, spec.block_size)
    if kind == "batch_decode":
        kwargs = {"backend": "fa2", "use_tensor_cores": tensor_cores} if force_fa2 else {}
        wrapper = fi.BatchDecodeWithPagedKVCacheWrapper(workspace, "NHD", **kwargs)
        plan_kwargs = _split_kwargs(spec) if force_fa2 else {}
        wrapper.plan(ptr, indices, tail, 4, 4, 64, spec.block_size, q_data_type=torch.bfloat16, kv_data_type=torch.bfloat16, **plan_kwargs)
        return wrapper.run(torch.stack([q[0] for q in queries]), cache)[spec.probe_index].clone()
    wrapper = fi.BatchPrefillWithPagedKVCacheWrapper(workspace, "NHD", **({"backend": "fa2"} if force_fa2 else {}))
    plan_kwargs = _split_kwargs(spec) if force_fa2 else {}
    if spec.chunk_size:
        # Only the probe is chunked; compare its full sequence of output rows.
        outputs = []
        prefix = spec.probe_kv_length - q_len
        for start in range(0, q_len, spec.chunk_size):
            end = min(start + spec.chunk_size, q_len)
            current = [q if i != spec.probe_index else q[start:end] for i, q in enumerate(queries)]
            lengths = [len(k) if i != spec.probe_index else prefix+end for i, k in enumerate(keys)]
            # Rebuild page metadata so the probe's last page reflects the visible prefix.
            use_k = [k[:length] for k, length in zip(keys, lengths)]
            use_v = [v[:length] for v, length in zip(values, lengths)]
            c, p, idx, t = _paged(use_k, use_v, spec.block_size)
            starts = [0]
            for q in current: starts.append(starts[-1] + len(q))
            qptr = torch.tensor(starts, device="cuda", dtype=torch.int32)
            wrapper.plan(qptr, p, idx, t, 4, 4, 64, spec.block_size, causal=True,
                         q_data_type=torch.bfloat16, kv_data_type=torch.bfloat16, **plan_kwargs)
            out = wrapper.run(torch.cat(current), c)
            outputs.append(out[starts[spec.probe_index]:starts[spec.probe_index+1]].clone())
        return torch.cat(outputs)
    starts = [0]
    for q in queries: starts.append(starts[-1] + len(q))
    qptr = torch.tensor(starts, device="cuda", dtype=torch.int32)
    wrapper.plan(qptr, ptr, indices, tail, 4, 4, 64, spec.block_size, causal=True,
                 q_data_type=torch.bfloat16, kv_data_type=torch.bfloat16, **plan_kwargs)
    out = wrapper.run(torch.cat(queries), cache)
    return out[starts[spec.probe_index]:starts[spec.probe_index+1]].clone()


def cases() -> list[Case]:
    _library()
    common = ("repeat", "batch", "position", "other_kv_length", "block_size")
    long_kv = dict(probe_kv_length=1024, other_kv_length=1024)
    fixed = BatchSpec(fixed_split_size=64, **long_kv)
    nosplit = BatchSpec(disable_split_kv=True, **long_kv)
    longdef = BatchSpec(**long_kv)
    out = [
        Case("flashinfer", "batch_decode", "bfloat16", {"H": 4, "D": 64, "K": 32, "page": 16}, lambda spec: _run(spec, "batch_decode"), common),
        Case("flashinfer", "batch_prefill", "bfloat16", {"H": 4, "D": 64, "Q": 8, "K": 32, "page": 16}, lambda spec: _run(spec, "batch_prefill"), common + ("chunk_size",)),
        Case("flashinfer", "single_decode", "bfloat16", {"H": 4, "D": 64, "K": 32}, lambda spec: _run(spec, "single_decode"), ("repeat",)),
        Case("flashinfer", "single_prefill", "bfloat16", {"H": 4, "D": 64, "Q": 8, "K": 32}, lambda spec: _run(spec, "single_prefill"), ("repeat", "chunk_size")),
        # Context only: default planner picks the split from batch composition (no invariance claim).
        Case("flashinfer", "ctx_default_planner/batch_decode_long", "bfloat16", {"H": 4, "D": 64, "K": 1024}, lambda spec: _run(spec, "batch_decode"), common, baseline=longdef),
        Case("flashinfer", "ctx_default_planner/batch_prefill_long", "bfloat16", {"H": 4, "D": 64, "Q": 8, "K": 1024}, lambda spec: _run(spec, "batch_prefill"), common + ("chunk_size",), baseline=longdef),
        # Claimed: fixed_split_size (batch-invariance control) keeps the reduction split independent of batch.
        Case("flashinfer", "fa2_fixed_split64tok/batch_decode_long", "bfloat16", {"H": 4, "D": 64, "K": 1024, "fixed_split_tokens": 64}, lambda spec: _run(spec, "batch_decode", True), common + ("fixed_split_size",), baseline=fixed),
        Case("flashinfer", "fa2_fixed_split64tok/batch_prefill_long", "bfloat16", {"H": 4, "D": 64, "Q": 8, "K": 1024, "fixed_split_tokens": 64}, lambda spec: _run(spec, "batch_prefill", True), common + ("chunk_size", "fixed_split_size"), baseline=fixed),
        Case("flashinfer", "fa2_disable_split_kv/batch_decode_long", "bfloat16", {"H": 4, "D": 64, "K": 1024}, lambda spec: _run(spec, "batch_decode", True), common, baseline=nosplit),
        # CUDA-core (non tensor-core) decode with disable_split_kv: see flashinfer issue 7981.
        Case("flashinfer", "fa2_disable_split_kv_cudacore/batch_decode_long", "bfloat16", {"H": 4, "D": 64, "K": 1024}, lambda spec: _run(spec, "batch_decode", True, False), common, baseline=nosplit),
        Case("flashinfer", "fa2_disable_split_kv/batch_prefill_long", "bfloat16", {"H": 4, "D": 64, "Q": 8, "K": 1024}, lambda spec: _run(spec, "batch_prefill", True), common + ("chunk_size",), baseline=nosplit),
    ]
    return out
case = next(c for c in cases() if c.name == CASE_NAME)
a = case.run(BatchSpec(**REFERENCE)); b = case.run(BatchSpec(**CANDIDATE))
print('bitwise equal:', compare(a, b).equal)
print('max abs diff:', compare(a, b).max_abs_diff)
