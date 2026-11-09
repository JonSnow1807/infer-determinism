"""vLLM FlashAttention FA2/FA3 varlen paged-KV probes."""
from __future__ import annotations

import importlib
import math

import torch

from ..core import BatchSpec, Case, seeded
from ..runner import SkipCase
from ._attention import AttentionConfig, Q_SCALE, K_SCALE, V_SCALE, attention_configs, attention_call, fp8_dtype, quantize_fp8, replay_graph, require_keywords


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


def _graph_query_activation(version: int, config: AttentionConfig) -> bool:
    # FA2's single-query GQA transpose optimization discards cu_seqlens_q
    # without a window. That path can test KV growth, but not query activation.
    return not (version == 2 and not config.window and config.num_q_heads != config.num_kv_heads)


def _run(spec: BatchSpec, version: int, config: AttentionConfig = AttentionConfig()) -> torch.Tensor:
    api = _api()
    supported = getattr(api, "is_fa_version_supported", None)
    if supported is not None and not supported(version):
        reason = getattr(api, "fa_version_unsupported_reason", lambda v: "unavailable build/device")(version)
        raise SkipCase(f"FA{version} is unsupported on this installation/GPU: {reason}")
    context = f"FA{version}/{config.label}"
    feature_kwargs = {}
    if config.window:
        feature_kwargs.update(window_size=(config.window_left, 0), causal=True)
    if config.fp8:
        if version != 3:
            raise SkipCase(f"{context}: FP8 Q/K/V descales require FA3")
        # FA3 accepts same-dtype FP8 Q/K/V, with BF16 output. Per-tensor
        # constants are expanded to its [batch, KV-heads] descale contract.
        feature_kwargs.update({name: torch.full((spec.batch_size, config.num_kv_heads), scale,
                                               device="cuda", dtype=torch.float32)
                               for name, scale in (("q_descale", Q_SCALE),
                                                   ("k_descale", K_SCALE), ("v_descale", V_SCALE))})
    require_keywords(api.flash_attn_varlen_func, feature_kwargs, context)
    q = torch.cat([seeded((1, config.num_q_heads, config.head_dim), spec, i,
                         torch.bfloat16, salt=83) for i in range(spec.batch_size)])
    if config.fp8:
        q = quantize_fp8(q, Q_SCALE)
    lengths = [spec.probe_kv_length if i == spec.probe_index else spec.other_kv_length for i in range(spec.batch_size)]
    block = spec.block_size
    counts = [math.ceil(length / block) for length in lengths]
    table_rows, offset = [], 0
    # FA3 FP8 output must not inherit Q's FP8 dtype.
    output = torch.empty(q.shape, device="cuda", dtype=torch.bfloat16)
    k_cache = torch.zeros((sum(counts), block, config.num_kv_heads, config.head_dim),
                          device="cuda", dtype=fp8_dtype() if config.fp8 else q.dtype)
    v_cache = torch.zeros_like(k_cache)
    for row, (length, count) in enumerate(zip(lengths, counts)):
        k = seeded((length, config.num_kv_heads, config.head_dim), spec, row, torch.bfloat16, salt=89)
        v = seeded((length, config.num_kv_heads, config.head_dim), spec, row, torch.bfloat16, salt=97)
        if config.fp8:
            k, v = quantize_fp8(k, K_SCALE), quantize_fp8(v, V_SCALE)
        k_cache[offset:offset+count].view(-1, config.num_kv_heads, config.head_dim)[:length] = k
        v_cache[offset:offset+count].view(-1, config.num_kv_heads, config.head_dim)[:length] = v
        table_rows.append(list(range(offset, offset + count)) + [0] * (max(counts) - count))
        offset += count
    tables = torch.tensor(table_rows, device="cuda", dtype=torch.int32)
    seqused_k = torch.tensor(lengths, device="cuda", dtype=torch.int32)
    cu_q = torch.arange(spec.batch_size+1, device="cuda", dtype=torch.int32)

    def launch(metadata=None):
        kwargs = dict(q=q, k=k_cache, v=v_cache, cu_seqlens_q=cu_q,
                      max_seqlen_q=1, max_seqlen_k=max(lengths),
                      seqused_k=seqused_k, block_table=tables,
                      num_splits=spec.num_splits, fa_version=version, out=output,
                      **feature_kwargs)
        if metadata is not None:
            kwargs["scheduler_metadata"] = metadata
        result = attention_call(api.flash_attn_varlen_func, context=context, **kwargs)
        return result[0] if isinstance(result, tuple) else result

    def schedule():
        if version != 3 or not hasattr(api, "get_scheduler_metadata"):
            return None
        kwargs = dict(qkv_dtype=q.dtype, cu_seqlens_q=cu_q,
                      page_size=block, num_splits=spec.num_splits)
        if config.window:
            kwargs.update(window_size=(config.window_left, 0), causal=True)
        require_keywords(api.get_scheduler_metadata, kwargs, context + " scheduler")
        return attention_call(api.get_scheduler_metadata, spec.batch_size, 1, max(lengths),
                              config.num_q_heads, config.num_kv_heads, config.head_dim,
                              seqused_k, context=context + " scheduler", **kwargs)

    if spec.graph_capture_kv_length:
        # Fixed capacity; grow visible KV and, when prefix sums are honored,
        # activate the other query rows. Never reuse a capture-time scheduler.
        live_cu_q = cu_q.clone()
        original = seqused_k.clone()
        launch()
        if _graph_query_activation(version, config):
            cu_q.copy_(torch.tensor([0] + [1] * spec.batch_size, device="cuda", dtype=torch.int32))
        seqused_k.copy_(torch.clamp(original, max=spec.graph_capture_kv_length))
        launch()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            launch()
        cu_q.copy_(live_cu_q)
        seqused_k.copy_(original)
        replay_graph(graph, output)
        return output[spec.probe_index].clone()
    return launch(schedule())[spec.probe_index].clone()


def cases() -> list[Case]:
    api = _api()
    versions = (2, 3)  # Keep unsupported versions visible as case-level skips.
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
    for config, full_only in attention_configs():
        length = config.window + 128 if config.window else 257
        for version in ((3,) if config.fp8 else versions):
            out.append(Case("vllm_flash_attn", f"run2/FA{version}/num_splits1/{config.label}", "bfloat16",
                            {**config.shapes(), "Q": 1, "K": length, "num_splits": 1,
                             "graph_query_activation": _graph_query_activation(version, config),
                             "q_dtype": "float8_e4m3fn" if config.fp8 else "bfloat16",
                             **({"q_scale": Q_SCALE} if config.fp8 else {}), "full_only": full_only},
                            lambda spec, v=version, c=config: _run(spec, v, c),
                            perts, baseline=BatchSpec(probe_kv_length=length, num_splits=1),
                            allowed_block_sizes=(16, 32, 64)))
    return out
