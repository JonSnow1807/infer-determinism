"""FlashInfer paged decode/prefill and single-request attention."""
from __future__ import annotations

import importlib
import math

import torch

from ..core import BatchSpec, Case, seeded
from ..runner import SkipCase


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
