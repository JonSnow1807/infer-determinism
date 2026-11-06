"""vLLM FlashAttention FA2/FA3 varlen paged-KV probes."""
from __future__ import annotations

import importlib
import math

import torch

from ..core import BatchSpec, Case, seeded
from ..runner import SkipCase


def _api():
    if not torch.cuda.is_available():
        raise SkipCase("CUDA is required for vLLM FlashAttention")
    for path in ("vllm.vllm_flash_attn.flash_attn_interface", "vllm.vllm_flash_attn"):
        try:
            module = importlib.import_module(path)
            if hasattr(module, "flash_attn_varlen_func"):
                return module
        except ImportError:
            continue
    raise SkipCase("vllm.vllm_flash_attn.flash_attn_varlen_func is unavailable")


def _run(spec: BatchSpec, version: int) -> torch.Tensor:
    api = _api()
    supported = getattr(api, "is_fa_version_supported", None)
    if supported is not None and not supported(version):
        raise SkipCase(f"FA{version} is unsupported on this installation/GPU")
    q = torch.stack([seeded((1, 4, 64), spec, i, torch.bfloat16, salt=83) for i in range(spec.batch_size)]).reshape(spec.batch_size, 4, 64)
    lengths = [spec.probe_kv_length if i == spec.probe_index else spec.other_kv_length for i in range(spec.batch_size)]
    block = spec.block_size
    max_blocks = max(math.ceil(length / block) for length in lengths)
    tables = torch.zeros((spec.batch_size, max_blocks), device="cuda", dtype=torch.int32)
    kp, vp = [], []
    for row, length in enumerate(lengths):
        k = seeded((length, 4, 64), spec, row, torch.bfloat16, salt=89)
        v = seeded((length, 4, 64), spec, row, torch.bfloat16, salt=97)
        for page in range(math.ceil(length / block)):
            index = len(kp)
            tables[row, page] = index
            a, b = torch.zeros((block, 4, 64), device="cuda", dtype=q.dtype), torch.zeros((block, 4, 64), device="cuda", dtype=q.dtype)
            take = min(block, length-page*block)
            a[:take], b[:take] = k[page*block:page*block+take], v[page*block:page*block+take]
            kp.append(a); vp.append(b)
    k_cache, v_cache = torch.stack(kp), torch.stack(vp)
    seqused_k = torch.tensor(lengths, device="cuda", dtype=torch.int32)
    cu_q = torch.arange(spec.batch_size+1, device="cuda", dtype=torch.int32)
    output = torch.empty_like(q)

    def launch(metadata=None):
        kwargs = dict(q=q, k=k_cache, v=v_cache, cu_seqlens_q=cu_q,
                      max_seqlen_q=1, max_seqlen_k=max(lengths),
                      seqused_k=seqused_k, block_table=tables,
                      num_splits=spec.num_splits, fa_version=version, out=output)
        if metadata is not None:
            kwargs["scheduler_metadata"] = metadata
        result = api.flash_attn_varlen_func(**kwargs)
        return result[0] if isinstance(result, tuple) else result

    metadata = None
    if version == 3 and hasattr(api, "get_scheduler_metadata"):
        def schedule():
            return api.get_scheduler_metadata(spec.batch_size, 1, max(lengths), 4, 4, 64,
                                              seqused_k, qkv_dtype=q.dtype, cu_seqlens_q=cu_q,
                                              page_size=block, num_splits=spec.num_splits)
        metadata = schedule()
    if spec.graph_capture_kv_length:
        # Fixed physical capacity, but capture one active request and replay
        # three. KV length metadata also grows after capture.
        original = seqused_k.clone()
        live_cu_q = cu_q.clone()
        cu_q.copy_(torch.tensor([0] + [1] * spec.batch_size, device="cuda", dtype=torch.int32))
        seqused_k.fill_(spec.graph_capture_kv_length)
        # Let FA3 schedule from live device metadata in the captured call.
        # A precomputed capture-time scheduler would ignore newly active rows.
        metadata = None
        launch()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            launch()
        cu_q.copy_(live_cu_q)
        seqused_k.copy_(original)
        graph.replay()
        torch.cuda.synchronize()
        return output[spec.probe_index].clone()
    return launch(metadata)[spec.probe_index].clone()


def cases() -> list[Case]:
    api = _api()
    supported = getattr(api, "is_fa_version_supported", None)
    versions = [v for v in (2, 3) if supported is None or supported(v)]
    perts = ("repeat", "batch", "position", "other_kv_length", "block_size", "graph_replay")
    long_default = BatchSpec(probe_kv_length=2048, other_kv_length=2048)
    # vLLM's FlashAttention backend forces num_splits=1 when VLLM_BATCH_INVARIANT=1.
    long_inv = BatchSpec(probe_kv_length=2048, other_kv_length=2048, num_splits=1)
    out = []
    for version in versions:
        run = lambda spec, v=version: _run(spec, v)
        sp = (("num_splits",) if version == 3 else ())
        out.append(Case("vllm_flash_attn", f"FA{version}/paged_varlen", "bfloat16", {"H": 4, "D": 64, "Q": 1, "K": 32}, run, perts + sp, allowed_block_sizes=(16, 32, 64)))
        out.append(Case("vllm_flash_attn", f"FA{version}/paged_varlen_long_num_splits1", "bfloat16", {"H": 4, "D": 64, "Q": 1, "K": 2048, "num_splits": 1}, run, perts, baseline=long_inv, allowed_block_sizes=(16, 32, 64)))
        # Context only: default num_splits heuristic (no invariance claim).
        out.append(Case("vllm_flash_attn", f"FA{version}/ctx_paged_varlen_long_default_splits", "bfloat16", {"H": 4, "D": 64, "Q": 1, "K": 2048}, run, perts, baseline=long_default, allowed_block_sizes=(16, 32, 64)))
    return out
