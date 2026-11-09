"""FlashInfer paged decode/prefill and single-request attention."""
from __future__ import annotations

import importlib
import math

import torch

from ..core import BatchSpec, Case, seeded
from ..runner import SkipCase
from ._attention import AttentionConfig, K_SCALE, V_SCALE, attention_configs, attention_call, fp8_dtype, quantize_fp8, replay_graph, require_keywords


def _library():
    if not torch.cuda.is_available():
        raise SkipCase("CUDA is required for FlashInfer")
    try:
        return importlib.import_module("flashinfer")
    except ImportError as exc:
        raise SkipCase(f"flashinfer-python is missing: {exc}") from exc


def _inputs(spec: BatchSpec, q_len: int, config: AttentionConfig = AttentionConfig()):
    queries, keys, values = [], [], []
    for row in range(spec.batch_size):
        length = spec.probe_kv_length if row == spec.probe_index else spec.other_kv_length
        queries.append(seeded((q_len, config.num_q_heads, config.head_dim), spec, row, torch.bfloat16, salt=71))
        keys.append(seeded((length, config.num_kv_heads, config.head_dim), spec, row, torch.bfloat16, salt=73))
        values.append(seeded((length, config.num_kv_heads, config.head_dim), spec, row, torch.bfloat16, salt=79))
    if config.fp8:
        keys = [quantize_fp8(k, K_SCALE) for k in keys]
        values = [quantize_fp8(v, V_SCALE) for v in values]
    return queries, keys, values


def _paged(keys, values, page_size: int):
    # Allocate once; copying an entire request avoids thousands of tiny GPU ops.
    counts = [math.ceil(len(k) / page_size) for k in keys]
    cache = keys[0].new_zeros((sum(counts), 2, page_size, *keys[0].shape[1:]))
    indptr, last = [0], []
    for k, v, n in zip(keys, values, counts):
        start = indptr[-1]
        # The K/V slices have a gap between pages, so use padded temporaries.
        for side, data in enumerate((k, v)):
            padded = data.new_zeros((n * page_size, *data.shape[1:]))
            padded[:len(data)] = data
            cache[start:start+n, side] = padded.reshape(n, page_size, *data.shape[1:])
        indptr.append(start + n)
        last.append(len(k) - (n-1)*page_size)
    device = cache.device
    return (cache, torch.tensor(indptr, device=device, dtype=torch.int32),
            torch.arange(sum(counts), device=device, dtype=torch.int32),
            torch.tensor(last, device=device, dtype=torch.int32))


def _short_metadata(ptr, indices, lengths, page_size, capture_length):
    """Shorter visible prefixes, retaining each request's physical page mapping."""
    short_ptr, short_indices, tails = [0], [], []
    for row, length in enumerate(lengths):
        length = min(length, capture_length)
        pages = math.ceil(length / page_size)
        start = int(ptr[row].item())
        short_indices.append(indices[start:start+pages])
        short_ptr.append(short_ptr[-1] + pages)
        tails.append(length - (pages-1)*page_size)
    return (torch.tensor(short_ptr, device=ptr.device, dtype=torch.int32),
            torch.cat(short_indices), torch.tensor(tails, device=ptr.device, dtype=torch.int32))


def _split_kwargs(spec: BatchSpec) -> dict:
    """fixed_split_size is in pages in FlashInfer; BatchSpec.fixed_split_size is in tokens."""
    kwargs: dict = {}
    pages = spec.fixed_split_size // spec.block_size
    if pages > 0:
        kwargs["fixed_split_size"] = pages
    if spec.disable_split_kv:
        kwargs["disable_split_kv"] = True
    return kwargs


def _run(spec: BatchSpec, kind: str, force_fa2: bool = False, tensor_cores: bool = True,
         config: AttentionConfig = AttentionConfig()) -> torch.Tensor:
    fi = _library()
    q_len = 1 if "decode" in kind else 8
    queries, keys, values = _inputs(spec, q_len, config)
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

    context = f"FlashInfer {kind}/{config.label}"
    decode = kind == "batch_decode"
    cls_name = "BatchDecodeWithPagedKVCacheWrapper" if decode else "BatchPrefillWithPagedKVCacheWrapper"
    cls = getattr(fi, cls_name, None)
    if cls is None:
        raise SkipCase(f"{context}: installed API lacks {cls_name}")
    # Retain the original workspace budget; paged storage scales with KV heads.
    workspace = torch.zeros(1024 * 1024 * 1024, device="cuda", dtype=torch.uint8)
    cache, ptr, indices, tail = _paged(keys, values, spec.block_size)
    qptr = torch.arange(spec.batch_size + 1, device="cuda", dtype=torch.int32) * q_len
    kwargs = {"backend": "fa2"} if force_fa2 else {}
    if force_fa2 and decode:
        kwargs["use_tensor_cores"] = tensor_cores
    if spec.graph_capture_kv_length:
        if not spec.disable_split_kv:
            raise SkipCase(f"{context}: graph length changes require disable_split_kv=True (fixed split can change CTA count)")
        suffix = "buffer" if decode else "buf"
        kwargs.update({"use_cuda_graph": True,
                       f"paged_kv_indptr_{suffix}": ptr.clone(),
                       f"paged_kv_indices_{suffix}": indices.clone(),
                       f"paged_kv_last_page_len_{suffix}": tail.clone()})
        if not decode:
            kwargs["qo_indptr_buf"] = qptr.clone()
    require_keywords(cls, kwargs, context)
    wrapper = attention_call(cls, workspace, "NHD", context=context, **kwargs)
    plan_kwargs = _split_kwargs(spec) if force_fa2 else {}
    plan_kwargs.update(q_data_type=torch.bfloat16,
                       kv_data_type=fp8_dtype() if config.fp8 else torch.bfloat16)
    if config.window:
        plan_kwargs["window_left"] = config.window_left
    if not decode:
        plan_kwargs["causal"] = True
    require_keywords(wrapper.plan, plan_kwargs, context + ".plan")
    run_kwargs = {"k_scale": K_SCALE, "v_scale": V_SCALE} if config.fp8 else {}
    require_keywords(wrapper.run, run_kwargs, context + ".run")

    def plan(p, idx, t, qp=qptr):
        args = (p, idx, t) if decode else (qp, p, idx, t)
        attention_call(wrapper.plan, *args, config.num_q_heads, config.num_kv_heads,
                       config.head_dim, spec.block_size, context=context + ".plan", **plan_kwargs)

    def run(q, c, **extra):
        return attention_call(wrapper.run, q, c, context=context + ".run", **run_kwargs, **extra)

    if spec.chunk_size and not decode:
        outputs = []
        prefix = spec.probe_kv_length - q_len
        for start in range(0, q_len, spec.chunk_size):
            end = min(start + spec.chunk_size, q_len)
            current = [q if i != spec.probe_index else q[start:end] for i, q in enumerate(queries)]
            lengths = [len(k) if i != spec.probe_index else prefix+end for i, k in enumerate(keys)]
            c, p, idx, t = _paged([k[:n] for k, n in zip(keys, lengths)],
                                   [v[:n] for v, n in zip(values, lengths)], spec.block_size)
            starts = [0]
            for q in current:
                starts.append(starts[-1] + len(q))
            qp = torch.tensor(starts, device="cuda", dtype=torch.int32)
            plan(p, idx, t, qp)
            out = run(torch.cat(current), c)
            outputs.append(out[starts[spec.probe_index]:starts[spec.probe_index+1]].clone())
        return torch.cat(outputs)

    q = torch.cat(queries)
    plan(ptr, indices, tail)
    if spec.graph_capture_kv_length:
        require_keywords(wrapper.run, ("out",), context + ".run CUDA graph")
        output = torch.empty_like(q)
        # Warm the replay capacity first. All plans stay outside capture.
        run(q, cache, out=output)
        short = _short_metadata(ptr, indices, [len(k) for k in keys], spec.block_size,
                                max(q_len, spec.graph_capture_kv_length))
        plan(*short)
        run(q, cache, out=output)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run(q, cache, out=output)
        plan(ptr, indices, tail)
        replay_graph(graph, output)
        out = output
    else:
        out = run(q, cache)
    start = spec.probe_index * q_len
    return out[start].clone() if decode else out[start:start+q_len].clone()


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
    for config, full_only in attention_configs():
        length = config.window + 128 if config.window else 257
        baseline = BatchSpec(probe_kv_length=length, other_kv_length=32, disable_split_kv=True)
        for kind in ("batch_decode", "batch_prefill"):
            perts = common + (("graph_replay",) if config.window else ())
            out.append(Case("flashinfer", f"run2/fa2_nosplit/{kind}/{config.label}", "bfloat16",
                            {**config.shapes(), "Q": 1 if kind == "batch_decode" else 8,
                             "K": length, "full_only": full_only, "backend": "fa2", "q_dtype": "bfloat16",
                             "use_tensor_cores": True, "disable_split_kv": True},
                            lambda spec, k=kind, c=config: _run(spec, k, True, True, c),
                            perts, baseline=baseline))
    return out
