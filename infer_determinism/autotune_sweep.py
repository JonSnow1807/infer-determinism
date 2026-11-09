"""Autotune-config sweep: replay every config of every captured Triton autotuner launch.

A scenario calls a library entry point once (for example vLLM's vendored
``chunk_gated_delta_rule``). While it runs, ``Autotuner.run`` is intercepted:
each launch's tensor arguments are snapshotted at the storage level before and
after the call, so aliasing and views survive a replay. Each captured launch is
then replayed once per config of its autotune list, on identical inputs:

* ``repeats`` replays from the same input bytes must be bitwise identical;
* the output must agree with a float32 reference when the scenario registers one
  for that kernel, and with the elementwise median of the other configs;
* two extra replays with differently poisoned outputs count elements the kernel
  never writes (informational: a kernel may leave padding or reuse a buffer).

The capture call itself pins one config per kernel (the first that compiles,
like vLLM's ``VLLM_TRITON_FORCE_FIRST_CONFIG``) so downstream kernels see inputs
produced by a fixed config, not by a timing-dependent autotune choice. Import
this module only where Triton is installed; the engine takes the autotuner
class as a parameter so CPU tests can use a stand-in.
"""
from __future__ import annotations

import contextlib
import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping

import torch

from .core import Difference, _bits

INVALID_CONFIG_ERRORS: tuple[type[BaseException], ...] = ()


def _invalid_config_errors() -> tuple[type[BaseException], ...]:
    """Triton compile/resource rejections that mean 'config not runnable here'."""
    global INVALID_CONFIG_ERRORS
    if INVALID_CONFIG_ERRORS:
        return INVALID_CONFIG_ERRORS
    errors: list[type[BaseException]] = []
    for module, name in (("triton.runtime.errors", "OutOfResources"),
                         ("triton.runtime.errors", "PTXASError"),
                         ("triton.compiler.errors", "CompileTimeAssertionFailure"),
                         ("triton.compiler.errors", "CompilationError")):
        try:
            errors.append(getattr(__import__(module, fromlist=[name]), name))
        except (ImportError, AttributeError):
            continue
    INVALID_CONFIG_ERRORS = tuple(errors) or (RuntimeError,)
    return INVALID_CONFIG_ERRORS


def default_autotuner_class():
    from triton.runtime.autotuner import Autotuner
    return Autotuner


def kernel_name(tuner: Any) -> str:
    fn = getattr(tuner, "base_fn", None) or tuner
    for _ in range(6):
        name = getattr(fn, "__name__", None)
        if name:
            return name
        fn = getattr(fn, "fn", None)
        if fn is None:
            break
    return type(tuner).__name__


def config_json(config: Any) -> dict[str, object]:
    out: dict[str, object] = {"kwargs": dict(getattr(config, "kwargs", {}) or {})}
    for attr in ("num_warps", "num_stages", "num_ctas", "maxnreg"):
        value = getattr(config, attr, None)
        if value is not None:
            out[attr] = value
    return out


def config_label(config: Any) -> str:
    j = config_json(config)
    parts = [f"{k}={v}" for k, v in sorted(j["kwargs"].items())]  # type: ignore[union-attr]
    parts += [f"{k}={j[k]}" for k in ("num_warps", "num_stages", "num_ctas", "maxnreg") if k in j and not (k == "num_ctas" and j[k] == 1)]
    return " ".join(parts)


def fast_compare(reference: torch.Tensor, candidate: torch.Tensor) -> Difference:
    """Vectorized, on-device version of core.compare (same raw-bit semantics).

    core.compare walks every element in Python to get the ULP distance, which is
    too slow for the multi-million-element outputs of the swept kernels.
    """
    if reference.shape != candidate.shape or reference.dtype != candidate.dtype:
        raise ValueError(f"output shape/dtype changed: {tuple(reference.shape)}/{reference.dtype} vs {tuple(candidate.shape)}/{candidate.dtype}")
    ab, bb = _bits(reference.detach()), _bits(candidate.detach())
    diff = ab != bb
    count = int(diff.sum().item())
    if not count:
        return Difference(True, 0, 0.0, 0)
    a, b = reference.detach()[diff].double(), candidate.detach()[diff].double()
    absdiff = (a - b).abs()
    finite = absdiff[torch.isfinite(absdiff)]
    max_abs: float | str = ("inf" if torch.isinf(absdiff).any().item() else
                            float(finite.max().item()) if finite.numel() else "nan")
    width = ab.element_size() * 8
    if width > 32:
        return Difference(False, count, max_abs, -1)
    mask, sign = (1 << width) - 1, 1 << (width - 1)
    av, bv = ab[diff].long() & mask, bb[diff].long() & mask
    ordered = lambda v: torch.where((v & sign) != 0, (~v) & mask, v | sign)
    ulp = int((ordered(av) - ordered(bv)).abs().max().item())
    return Difference(False, count, max_abs, ulp)


# ---------------------------------------------------------------------------
# Storage-level snapshots


@dataclass(frozen=True)
class TensorRef:
    storage: int  # index into the launch's storage table
    offset: int
    size: tuple[int, ...]
    stride: tuple[int, ...]
    dtype: torch.dtype


def _storage_bytes(t: torch.Tensor) -> torch.Tensor:
    """uint8 view of the tensor's whole untyped storage."""
    return torch.empty(0, dtype=torch.uint8, device=t.device).set_(t.untyped_storage())


def _view(base: torch.Tensor, ref: TensorRef) -> torch.Tensor:
    return torch.empty(0, dtype=ref.dtype, device=base.device).set_(base.untyped_storage(), ref.offset, ref.size, ref.stride)


@dataclass
class Launch:
    scenario: str
    kernel: str
    index: int
    tuner: Any
    configs: list[Any]
    args: list[Any]  # tensors replaced by TensorRef
    kwargs: dict[str, Any]
    pre: list[torch.Tensor]  # uint8 storage clones before the call
    post: list[torch.Tensor]  # after the capture call
    outputs: dict[str, TensorRef] = field(default_factory=dict)
    capture_config: int = -1
    signature: tuple = ()

    def named_refs(self) -> dict[str, TensorRef]:
        names = list(getattr(self.tuner, "arg_names", []) or [])
        refs: dict[str, TensorRef] = {}
        for i, a in enumerate(self.args):
            if isinstance(a, TensorRef):
                refs[names[i] if i < len(names) else f"arg{i}"] = a
        for k, a in self.kwargs.items():
            if isinstance(a, TensorRef):
                refs[k] = a
        return refs

    def materialize(self, source: list[torch.Tensor] | None = None) -> tuple[list[Any], dict[str, Any], dict[str, torch.Tensor]]:
        """Fresh storages copied from ``source`` (default: pre-call bytes)."""
        bases = [s.clone() for s in (source if source is not None else self.pre)]
        conv = lambda a: _view(bases[a.storage], a) if isinstance(a, TensorRef) else a
        args = [conv(a) for a in self.args]
        kwargs = {k: conv(v) for k, v in self.kwargs.items()}
        views = {name: _view(bases[ref.storage], ref) for name, ref in self.named_refs().items()}
        return args, kwargs, views

    def inputs(self, which: str = "pre") -> dict[str, Any]:
        """Named arguments (tensors as views of the snapshot) for references."""
        source = self.pre if which == "pre" else self.post
        names = list(getattr(self.tuner, "arg_names", []) or [])
        named: dict[str, Any] = {}
        for i, a in enumerate(self.args):
            name = names[i] if i < len(names) else f"arg{i}"
            named[name] = _view(source[a.storage], a) if isinstance(a, TensorRef) else a
        for k, a in self.kwargs.items():
            named[k] = _view(source[a.storage], a) if isinstance(a, TensorRef) else a
        return named


def _snapshot(args: Iterable[Any], kwargs: Mapping[str, Any]):
    storages: dict[tuple[str, int], int] = {}
    pre: list[torch.Tensor] = []
    sources: list[torch.Tensor] = []

    def ref(a: Any) -> Any:
        if not isinstance(a, torch.Tensor):
            return a
        key = (str(a.device), a.untyped_storage().data_ptr())
        if key not in storages:
            storages[key] = len(pre)
            raw = _storage_bytes(a)
            pre.append(raw.clone())
            sources.append(raw)
        return TensorRef(storages[key], a.storage_offset(), tuple(a.shape), tuple(a.stride()), a.dtype)

    refs = [ref(a) for a in args]
    krefs = {k: ref(v) for k, v in kwargs.items()}
    return refs, krefs, pre, sources


def _signature(tuner: Any, args: Iterable[Any], kwargs: Mapping[str, Any]) -> tuple:
    def sig(a: Any) -> object:
        if isinstance(a, torch.Tensor):
            return ("T", tuple(a.shape), str(a.dtype))
        if a is None or isinstance(a, (bool, int, float, str)):
            return a
        if callable(a):
            return "callable"
        return type(a).__name__
    return (id(tuner), tuple(sig(a) for a in args), tuple(sorted((k, sig(v)) for k, v in kwargs.items() if k not in ("grid", "warmup"))))


@contextlib.contextmanager
def pinned_config(tuner: Any, config: Any):
    configs, cache = tuner.configs, dict(getattr(tuner, "cache", {}) or {})
    tuner.configs = [config]
    tuner.cache.clear()
    try:
        yield
    finally:
        tuner.configs = configs
        tuner.cache.clear()
        tuner.cache.update(cache)


@dataclass
class CaptureResult:
    launches: list[Launch]
    output: Any
    picked: dict[str, list[str]]  # kernel -> config labels used per launch


def capture(run: Callable[[], Any], scenario: str = "", *, mode: str = "first", autotuner_cls: Any = None,
            synchronize: Callable[[], None] | None = None, dedupe: bool = True) -> CaptureResult:
    """Call ``run`` and record every autotuned launch.

    mode="first": each launch runs with the first config that compiles (stable
    inputs for downstream kernels). mode="autotune": normal autotuning (what
    production does); the chosen config is recorded.
    """
    cls = autotuner_cls or default_autotuner_class()
    sync = synchronize or (torch.cuda.synchronize if torch.cuda.is_available() else (lambda: None))
    original = cls.run
    launches: list[Launch] = []
    seen: set[tuple] = set()
    picked: dict[str, list[str]] = {}
    invalid = _invalid_config_errors()

    def patched(self, *args, **kwargs):
        sig = _signature(self, args, kwargs)
        record = not (dedupe and sig in seen)
        if record:
            seen.add(sig)
            sync()
            refs, krefs, pre, sources = _snapshot(args, kwargs)
        configs = list(self.configs)
        chosen = -1
        if mode == "first" and len(configs) > 1:
            last: BaseException | None = None
            result = None
            for i, config in enumerate(configs):
                try:
                    with pinned_config(self, config):
                        result = original(self, *args, **kwargs)
                    chosen = i
                    break
                except invalid as exc:  # compile/resource rejection: try the next config
                    last = exc
            if chosen < 0:
                raise RuntimeError(f"no config of {kernel_name(self)} compiled") from last
        else:
            result = original(self, *args, **kwargs)
            best = getattr(self, "best_config", None)
            chosen = next((i for i, c in enumerate(configs) if c is best), -1)
        name = kernel_name(self)
        picked.setdefault(name, []).append(config_label(configs[chosen]) if chosen >= 0 else "?")
        if record:
            sync()
            post = [s.clone() for s in sources]
            launch = Launch(scenario, name, len(launches), self, configs, refs, krefs, pre, post,
                            capture_config=chosen, signature=sig)
            for slot, tref in launch.named_refs().items():
                before = _view(pre[tref.storage], tref)
                after = _view(post[tref.storage], tref)
                if not fast_compare(before, after).equal:
                    launch.outputs[slot] = tref
            launches.append(launch)
        return result

    cls.run = patched
    try:
        output = run()
        sync()
    finally:
        cls.run = original
    return CaptureResult(launches, output, picked)


# ---------------------------------------------------------------------------
# Replay and metrics


def rel_err(out: torch.Tensor, ref: torch.Tensor) -> float:
    """||out - ref|| / ||ref|| over all elements (float64); inf if out has new nonfinites."""
    a, b = out.detach().double(), ref.detach().double()
    if a.shape != b.shape:
        raise ValueError(f"reference shape {tuple(b.shape)} != output {tuple(a.shape)}")
    bad = ~torch.isfinite(a) & torch.isfinite(b)
    if bad.any():
        return math.inf
    both = torch.isfinite(a) & torch.isfinite(b)
    denom = torch.linalg.vector_norm(b[both]).item()
    num = torch.linalg.vector_norm((a - b)[both]).item()
    return num / denom if denom else (0.0 if num == 0 else math.inf)


def _nonfinite(t: torch.Tensor) -> int:
    return int((~torch.isfinite(t)).sum().item()) if t.is_floating_point() else 0


def _poison(t: torch.Tensor, pattern: int) -> None:
    if t.is_floating_point():
        t.fill_(float("nan") if pattern == 0 else -7777.0)
    elif t.dtype == torch.bool:
        t.fill_(pattern == 0)
    else:
        t.fill_(-1 if pattern == 0 else 0x5A)


def run_config(launch: Launch, config: Any, *, poison: int | None = None,
               synchronize: Callable[[], None] | None = None) -> dict[str, torch.Tensor]:
    sync = synchronize or (torch.cuda.synchronize if torch.cuda.is_available() else (lambda: None))
    args, kwargs, views = launch.materialize()
    if poison is not None:
        for slot in launch.outputs:
            _poison(views[slot], poison)
    with pinned_config(launch.tuner, config):
        launch.tuner.run(*args, **kwargs)
    sync()
    result = {slot: views[slot].clone() for slot in launch.outputs}
    # Arguments the capture call left unchanged but this config wrote to.
    stray = {}
    for slot, ref in launch.named_refs().items():
        if slot not in launch.outputs:
            n = fast_compare(_view(launch.pre[ref.storage], ref), views[slot]).differing_elements
            if n:
                stray[slot] = n
    if stray:
        result["__stray_writes__"] = stray  # type: ignore[assignment]
    return result


Reference = Callable[[Launch], Mapping[str, torch.Tensor]]


def sweep_launch(launch: Launch, *, repeats: int = 3, reference: Reference | None = None, tol: float = 1e-2,
                 poison_check: bool = True, synchronize: Callable[[], None] | None = None,
                 only: Iterable[int] | None = None, invalid_errors: tuple[type[BaseException], ...] | None = None) -> list[dict[str, object]]:
    """One ``sweep_config`` record per config plus a closing ``sweep_summary``."""
    invalid = invalid_errors or _invalid_config_errors()
    base = {"scenario": launch.scenario, "kernel": launch.kernel, "launch": launch.index}
    refs: dict[str, torch.Tensor] = {}
    ref_error = None
    if reference is not None:
        try:
            named = launch.named_refs()
            refs = {k: v for k, v in reference(launch).items() if k in named}
            for slot in refs:  # validate declared outputs even if the capture call left them unchanged
                launch.outputs.setdefault(slot, named[slot])
        except Exception as exc:  # a broken reference must not hide the sweep
            ref_error = f"{type(exc).__name__}: {exc}"
    records: list[dict[str, object]] = []
    outputs: dict[int, dict[str, torch.Tensor]] = {}
    selected = set(only) if only is not None else None
    for idx, config in enumerate(launch.configs):
        if selected is not None and idx not in selected:
            continue
        rec: dict[str, object] = {"type": "sweep_config", **base, "config_index": idx,
                                  "config": config_json(config), "label": config_label(config)}
        start = time.perf_counter()
        try:
            runs = [run_config(launch, config, synchronize=synchronize) for _ in range(max(1, repeats))]
        except invalid as exc:
            rec.update(status="invalid", error=f"{type(exc).__name__}: {str(exc)[:300]}")
            records.append(rec)
            continue
        except Exception as exc:
            rec.update(status="error", error=f"{type(exc).__name__}: {str(exc)[:500]}")
            records.append(rec)
            continue
        stray = [r.pop("__stray_writes__") for r in runs if "__stray_writes__" in r]
        if stray:
            rec["stray_writes"] = stray[0]
        first = runs[0]
        diff_elems, max_ulp, diff_runs = 0, 0, 0
        max_abs: float | str = 0.0
        for other in runs[1:]:
            run_diff = False
            for slot in launch.outputs:
                d = fast_compare(first[slot], other[slot])
                if d.equal:
                    continue
                run_diff = True
                diff_elems = max(diff_elems, d.differing_elements)
                max_ulp = max(max_ulp, d.max_ulp_distance)
                if isinstance(d.max_abs_diff, str) or isinstance(max_abs, str):
                    max_abs = d.max_abs_diff if isinstance(d.max_abs_diff, str) else max_abs
                else:
                    max_abs = max(max_abs, d.max_abs_diff)
            diff_runs += run_diff
        det = diff_runs == 0
        rec.update(status="ok", deterministic=det, repeats=len(runs), differing_repeats=diff_runs,
                   repeat_differing_elements=diff_elems, repeat_max_ulp=max_ulp, repeat_max_abs=max_abs,
                   nonfinite={s: _nonfinite(t) for s, t in first.items()})
        if refs:
            rec["rel_err_ref"] = {s: rel_err(first[s], r) for s, r in refs.items()}
            rec["rel_err_ref_worst_repeat"] = {s: max(rel_err(r_[s], r) for r_ in runs) for s, r in refs.items()}
        if poison_check:
            try:
                p0 = run_config(launch, config, poison=0, synchronize=synchronize)
                p1 = run_config(launch, config, poison=1, synchronize=synchronize)
                p0.pop("__stray_writes__", None)
                p1.pop("__stray_writes__", None)
                rec["poison_differing"] = {s: fast_compare(p0[s], p1[s]).differing_elements for s in launch.outputs}
                # Poison 1 is finite, so a nonfinite value here was written by the kernel.
                rec["nonfinite_written"] = {s: _nonfinite(t) for s, t in p1.items()}
            except Exception as exc:
                rec["poison_error"] = f"{type(exc).__name__}: {exc}"
        rec["seconds"] = round(time.perf_counter() - start, 3)
        outputs[idx] = first
        records.append(rec)
    _consensus(records, outputs, launch)
    for rec in records:
        if rec.get("status") != "ok":
            continue
        flags = []
        if not rec["deterministic"]:
            flags.append("nondeterministic")
        if any(v > tol for v in (rec.get("rel_err_ref") or {}).values()) or any(
                v > tol for v in (rec.get("rel_err_ref_worst_repeat") or {}).values()):
            flags.append("wrong_vs_ref")
        if any(v > tol for v in (rec.get("rel_err_consensus") or {}).values()):
            flags.append("outlier_vs_consensus")
        ref_nonfinite = rec.get("consensus_nonfinite") or {}
        if any(n > ref_nonfinite.get(s, 0) for s, n in rec["nonfinite"].items()) or any(  # type: ignore[union-attr]
                (rec.get("nonfinite_written") or {}).values()):
            flags.append("nonfinite")
        if rec.get("stray_writes"):
            flags.append("stray_writes")
        rec["flags"] = flags
    ok = [r for r in records if r.get("status") == "ok"]
    summary = {"type": "sweep_summary", **base, "configs": len(launch.configs), "swept": len(records),
               "invalid": sum(r.get("status") == "invalid" for r in records),
               "errors": sum(r.get("status") == "error" for r in records),
               "nondeterministic": [r["config_index"] for r in ok if not r["deterministic"]],
               "flagged": [r["config_index"] for r in ok if r["flags"]],
               "flagged_labels": [r["label"] for r in ok if r["flags"]],
               "capture_config": launch.capture_config, "tolerance": tol,
               "reference": bool(refs), "reference_error": ref_error,
               "outputs": {s: {"shape": list(t.size), "dtype": str(t.dtype)} for s, t in launch.outputs.items()}}
    if refs and ok:
        summary["best_rel_err_ref"] = {s: min(r["rel_err_ref"][s] for r in ok) for s in refs}  # type: ignore[index]
    records.append(summary)
    return records


def _consensus(records: list[dict[str, object]], outputs: dict[int, dict[str, torch.Tensor]], launch: Launch) -> None:
    """Elementwise median over deterministic configs, per output slot."""
    good = [r["config_index"] for r in records if r.get("status") == "ok" and r.get("deterministic")]
    pool = [i for i in good if i in outputs] or list(outputs)
    if len(pool) < 3:
        return
    for slot in launch.outputs:
        stack = torch.stack([outputs[i][slot].float() for i in pool])
        median = stack.nanmedian(dim=0).values if stack.is_floating_point() else stack.median(dim=0).values
        del stack
        nonfinite = _nonfinite(median)
        for rec in records:
            idx = rec["config_index"]
            if idx in outputs:
                rec.setdefault("rel_err_consensus", {})[slot] = rel_err(outputs[idx][slot].float(), median)  # type: ignore[index]
                rec.setdefault("consensus_nonfinite", {})[slot] = nonfinite  # type: ignore[index]
        del median
    for idx in list(outputs):
        outputs.pop(idx)


def launch_record(launch: Launch) -> dict[str, object]:
    return {"type": "sweep_launch", "scenario": launch.scenario, "kernel": launch.kernel, "launch": launch.index,
            "configs": len(launch.configs), "capture_config": launch.capture_config,
            "capture_label": config_label(launch.configs[launch.capture_config]) if launch.capture_config >= 0 else None,
            "outputs": {s: {"shape": list(t.size), "dtype": str(t.dtype)} for s, t in launch.outputs.items()},
            "scalars": {k: v for k, v in launch.inputs().items() if isinstance(v, (bool, int, float, str)) or v is None}}
