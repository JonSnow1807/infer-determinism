"""Execute cases and write reproducible JSONL results."""
from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import inspect
import json
import os
import subprocess
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import torch

from .core import Case, compare, probe_specs

TARGETS = ("torch_matmul", "torch_sdpa", "vllm_batch_invariant", "vllm_invariant_mode", "flashinfer", "vllm_flash_attn", "moe")


class SkipCase(Exception):
    """An expected unsupported dependency, device, or backend."""


def _flag(getter, default=None):
    try:
        return getter()
    except Exception:  # mixed legacy/new TF32 APIs make the legacy getters raise
        return default


def environment() -> dict[str, object]:
    versions: dict[str, str | None] = {}
    for name in ("torch", "vllm", "flashinfer-python", "vllm-flash-attn"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    try:
        driver = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], capture_output=True, text=True, timeout=5).stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        driver = None
    return {
        "schema_version": 1,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "versions": versions,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "driver": driver,
        "cuda_runtime": torch.version.cuda,
        "env": {k: os.environ.get(k) for k in ("VLLM_BATCH_INVARIANT", "CUBLAS_WORKSPACE_CONFIG", "CUDA_VISIBLE_DEVICES")},
        "torch_flags": {
            "matmul_allow_tf32": _flag(lambda: torch.backends.cuda.matmul.allow_tf32, "unreadable"),
            "cudnn_allow_tf32": _flag(lambda: torch.backends.cudnn.allow_tf32, "unreadable"),
            "matmul_fp32_precision": _flag(lambda: torch.backends.cuda.matmul.fp32_precision, None),
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
        },
    }


def load_cases(target: str) -> list[Case]:
    if target not in TARGETS and target != "fake":
        raise ValueError(f"unknown target: {target}")
    module = importlib.import_module(f"infer_determinism.targets.{target}")
    return module.cases()


def _script(case: Case, reference: dict[str, object], candidate: dict[str, object], run_environment: dict[str, object] | None = None) -> str:
    run_environment = run_environment or environment()
    recorded_env = run_environment.get("env", {})
    assert isinstance(recorded_env, dict)
    env = {k: v for k, v in recorded_env.items() if k in ("VLLM_BATCH_INVARIANT", "CUBLAS_WORKSPACE_CONFIG") and v is not None}
    prefix = (
        "# Generated standalone reproducer. Install torch and the target library.\n"
        f"# Recorded environment: {run_environment!r}\n"
        "import os\n"
        f"os.environ.update({env!r})\n"
        "import torch\n"
        f"torch.backends.cudnn.deterministic = {torch.backends.cudnn.deterministic!r}\n"
        f"torch.backends.cudnn.benchmark = {torch.backends.cudnn.benchmark!r}\n"
        f"torch.backends.cudnn.allow_tf32 = {_flag(lambda: torch.backends.cudnn.allow_tf32, False)!r}\n"
        f"REFERENCE = {reference!r}\nCANDIDATE = {candidate!r}\nCASE_NAME = {case.name!r}\n"
    )
    if not case.repro_source:
        # Embed only this adapter and the small core so the script has no
        # infer_determinism installation dependency.
        from . import core
        module = importlib.import_module(f"infer_determinism.targets.{case.target}")
        clean = lambda src: "\n".join(line for line in src.splitlines() if not line.startswith(("from __future__", "from ..core import", "from ..runner import")))
        return (
            prefix + clean(inspect.getsource(core)) + "\nclass SkipCase(Exception): pass\n"
            + clean(inspect.getsource(module))
            + "\ncase = next(c for c in cases() if c.name == CASE_NAME)\n"
            + "a = case.run(BatchSpec(**REFERENCE)); b = case.run(BatchSpec(**CANDIDATE))\n"
            + "print('bitwise equal:', compare(a, b).equal)\n"
            + "print('max abs diff:', compare(a, b).max_abs_diff)\n"
        )
    return (
        prefix + case.repro_source
    )


def run_case(case: Case, quick: bool = False, run_environment: dict[str, object] | None = None) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    run_environment = run_environment or environment()
    base = case.baseline
    try:
        reference = case.run(base).detach().clone()
        for perturbation, spec in probe_specs(case, quick):
            try:
                candidate = case.run(spec)
                delta = compare(reference, candidate)
            except SkipCase as exc:
                records.append({"type": "skip", "target": case.target, "case": case.name, "perturbation": perturbation, "reason": str(exc)})
                continue
            except Exception as exc:
                records.append({"type": "error", "target": case.target, "case": case.name, "perturbation": perturbation, "config": spec.json(), "error": f"{type(exc).__name__}: {exc}"})
                continue
            if not delta.equal:
                payload: dict[str, object] = {
                    "type": "divergence", "target": case.target, "case": case.name,
                    "dtype": case.dtype, "shapes": dict(case.shapes), "perturbation": perturbation,
                    "reference": base.json(), "candidate": spec.json(),
                    "environment": run_environment,
                    **asdict(delta),
                }
                payload["id"] = hashlib.sha256(json.dumps(payload, sort_keys=True, allow_nan=False).encode()).hexdigest()[:16]
                payload["repro_script"] = _script(case, base.json(), spec.json(), run_environment)
                records.append(payload)
            else:
                records.append({"type": "check", "target": case.target, "case": case.name,
                                "perturbation": perturbation, "config": spec.json(), "equal": True})
        if all(r["type"] == "check" for r in records):
            records.append({"type": "pass", "target": case.target, "case": case.name, "checks": len(probe_specs(case, quick))})
    except SkipCase as exc:
        records.append({"type": "skip", "target": case.target, "case": case.name, "reason": str(exc)})
    except Exception as exc:
        records.append({"type": "error", "target": case.target, "case": case.name, "error": f"{type(exc).__name__}: {exc}"})
    return records


def run_targets(targets: Iterable[str], out: Path, quick: bool = False) -> list[dict[str, object]]:
    records: list[dict[str, object]] = [{"type": "metadata", **environment()}]
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as file:
        def append(new: list[dict[str, object]]) -> None:
            for record in new:
                records.append(record)
                file.write(json.dumps(record, allow_nan=False) + "\n")
            file.flush()

        file.write(json.dumps(records[0], allow_nan=False) + "\n")
        file.flush()
        for target in targets:
            try:
                cases = load_cases(target)
                if not cases:
                    append([{"type": "skip", "target": target, "reason": "no importable cases in installed library"}])
                for case in cases:
                    append(run_case(case, quick, records[0]))
            except (ImportError, SkipCase) as exc:
                append([{"type": "skip", "target": target, "reason": str(exc)}])
            except Exception as exc:
                append([{"type": "error", "target": target, "error": f"{type(exc).__name__}: {exc}"}])
    return records
