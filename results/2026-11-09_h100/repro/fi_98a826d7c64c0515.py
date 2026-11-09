# Generated standalone reproducer. Install torch and the target library.
# Recorded environment: {'type': 'metadata', 'schema_version': 1, 'timestamp_utc': '2026-11-09T02:56:46.638627+00:00', 'versions': {'torch': '2.14.0', 'vllm': '0.33.0', 'flashinfer-python': '0.7.2.post1', 'vllm-flash-attn': None}, 'gpu': 'NVIDIA H100 NVL', 'driver': '580.178.04', 'cuda_runtime': '13.0', 'env': {'VLLM_BATCH_INVARIANT': '1', 'CUBLAS_WORKSPACE_CONFIG': ':4096:8', 'CUDA_VISIBLE_DEVICES': None}, 'torch_flags': {'matmul_allow_tf32': False, 'cudnn_allow_tf32': True, 'matmul_fp32_precision': 'none', 'cudnn_deterministic': False, 'cudnn_benchmark': False, 'float32_matmul_precision': 'highest'}}
import os
os.environ.update({'VLLM_BATCH_INVARIANT': '1', 'CUBLAS_WORKSPACE_CONFIG': ':4096:8'})
import torch
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.allow_tf32 = True
REFERENCE = {'batch_size': 3, 'probe_index': 0, 'seed': 1729, 'other_kv_length': 32, 'probe_kv_length': 4224, 'chunk_size': 0, 'block_size': 16, 'num_splits': 0, 'fixed_split_size': 0, 'disable_split_kv': True, 'tf32': False, 'graph_capture_kv_length': 0}
CANDIDATE = {'batch_size': 3, 'probe_index': 0, 'seed': 1729, 'other_kv_length': 32, 'probe_kv_length': 4224, 'chunk_size': 0, 'block_size': 16, 'num_splits': 0, 'fixed_split_size': 0, 'disable_split_kv': True, 'tf32': False, 'graph_capture_kv_length': 2112}
CASE_NAME = 'run2/fa2_nosplit/batch_prefill/q32_kv8_d128/window4096'
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
    if "big_m" in case.perturbations and not quick:
        specs += [("big_m", replace(base, batch_size=n, probe_index=0)) for n in (500, 512, 1000, 1024, 2048, 4096)]
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
        specs += [("graph_replay", replace(base, batch_size=3, probe_index=p,
                                           graph_capture_kv_length=max(1, base.probe_kv_length // 2)))
                  for p in (0, 2)]
    return specs
class SkipCase(Exception): pass
"""CPU-safe attention case descriptions and capability checks.

Also embedded in standalone reproducers by runner._script.
"""

import inspect
import re
from dataclasses import dataclass

import torch



@dataclass(frozen=True)
class AttentionConfig:
    num_q_heads: int = 4
    num_kv_heads: int = 4
    head_dim: int = 64
    window: int = 0  # Total visible tokens, including the current token.
    fp8: bool = False

    @property
    def window_left(self) -> int:
        return self.window - 1 if self.window else -1

    @property
    def label(self) -> str:
        return (f"q{self.num_q_heads}_kv{self.num_kv_heads}_d{self.head_dim}"
                + (f"/window{self.window}" if self.window else "")
                + ("/fp8_e4m3" if self.fp8 else ""))

    def shapes(self) -> dict[str, object]:
        return {"num_q_heads": self.num_q_heads, "num_kv_heads": self.num_kv_heads,
                "D": self.head_dim, "window": self.window,
                "kv_dtype": "float8_e4m3fn" if self.fp8 else "bfloat16",
                **({"k_scale": K_SCALE, "v_scale": V_SCALE} if self.fp8 else {})}


# Fixed per-tensor dequantization scales; never computed over a batch.
Q_SCALE, K_SCALE, V_SCALE = 0.5, 0.25, 0.125
GQA_HEADS = ((32, 8), (64, 8), (8, 1))


def attention_configs():
    """(config, full_only) matrix shared by FlashInfer and vLLM FA."""
    for dim in (128, 256):
        for hq, hkv in GQA_HEADS:
            yield AttentionConfig(hq, hkv, dim), (dim != 128 or hq != 32)
    for window in (1024, 4096):
        for dim in (128, 256):
            yield AttentionConfig(32, 8, dim, window), (window != 1024 or dim != 128)
    for hq, hkv in GQA_HEADS:
        yield AttentionConfig(hq, hkv, 128, fp8=True), hq != 32
    # Interaction probes; no claim that every build supports these kernels.
    yield AttentionConfig(32, 8, 256, fp8=True), True
    yield AttentionConfig(32, 8, 128, 4096, fp8=True), True


def require_keywords(fn, names, context: str) -> None:
    """Do not silently drop a requested feature on an older API."""
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return  # C bindings may not expose a signature; validate at call time.
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return
    missing = sorted(set(names) - set(params))
    if missing:
        raise SkipCase(f"{context}: installed API lacks keyword(s) {', '.join(missing)}")


def attention_call(fn, *args, context: str, **kwargs):
    """Classify explicit capability rejections, never arbitrary CUDA failures."""
    try:
        return fn(*args, **kwargs)
    except (TypeError, RuntimeError, NotImplementedError, ValueError) as exc:
        msg = str(exc).lower()
        # Recognized backend/build rejections. Shape/stride/scale assertions,
        # JIT failures and arbitrary errors mentioning FP8 remain errors.
        unsupported = (
            isinstance(exc, TypeError) and "unexpected keyword argument" in msg
            or "no available kernel" in msg
            or msg.startswith("this flash attention build does not support ")
            or msg.startswith("flashattention forward only supports head dimension at most ")
            or re.match(r"flashattention (?:on ampere/ada cards )?only supports? fp16", msg)
            or re.fullmatch(r"unsupported head_dim[ :]+\d+\.?", msg)
        )
        if unsupported:
            raise SkipCase(f"{context}: {type(exc).__name__}: {exc}") from exc
        raise


def fp8_dtype():
    dtype = getattr(torch, "float8_e4m3fn", None)
    if dtype is None:
        raise SkipCase("FP8 KV requires torch.float8_e4m3fn")
    return dtype


def quantize_fp8(tensor: torch.Tensor, scale: float) -> torch.Tensor:
    # Scale convention: real value ~= stored FP8 value * scale.
    limit = torch.finfo(fp8_dtype()).max
    return (tensor.float() / scale).clamp(-limit, limit).to(fp8_dtype())


def replay_graph(graph, output: torch.Tensor) -> None:
    """Poison all active outputs so a missing graph write cannot look equal."""
    output.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()
    if not torch.isfinite(output).all().item():
        raise RuntimeError("CUDA graph replay produced nonfinite or unwritten active output")
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
case = next(c for c in cases() if c.name == CASE_NAME)
a = case.run(BatchSpec(**REFERENCE)); b = case.run(BatchSpec(**CANDIDATE))
print('bitwise equal:', compare(a, b).equal)
print('max abs diff:', compare(a, b).max_abs_diff)
