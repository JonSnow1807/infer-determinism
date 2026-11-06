from __future__ import annotations

import subprocess
import sys

import torch

from infer_determinism.core import BatchSpec, Case, compare, probe_specs
from infer_determinism.report import markdown, read_records
from infer_determinism.runner import TARGETS, run_case, run_targets
from infer_determinism.targets.fake import cases


def test_fake_detects_batch_dependence() -> None:
    rows = run_case(cases()[1], quick=True)
    divergences = [r for r in rows if r["type"] == "divergence"]
    assert divergences
    assert all(r["perturbation"] == "batch" or r["perturbation"] == "position" for r in divergences)
    assert divergences[0]["differing_elements"] == 1
    assert divergences[0]["reference"]["batch_size"] == 1


def test_invariant_fake_has_no_false_positive() -> None:
    rows = run_case(cases()[0], quick=True)
    assert rows[-1]["type"] == "pass"
    assert all(r["type"] in ("check", "pass") for r in rows)


def test_raw_bits_distinguish_signed_zero() -> None:
    difference = compare(torch.tensor([-0.0]), torch.tensor([0.0]))
    assert not difference.equal
    assert difference.differing_elements == 1


def test_raw_bits_distinguish_nan_payloads() -> None:
    a = torch.tensor([0x7FC00001], dtype=torch.int32).view(torch.float32)
    b = torch.tensor([0x7FC00002], dtype=torch.int32).view(torch.float32)
    difference = compare(a, b)
    assert not difference.equal
    assert difference.max_abs_diff == "nan"


def test_cpu_cli_report_and_standalone_repro(tmp_path) -> None:
    path = tmp_path / "results.jsonl"
    records = run_targets(["fake"], path, quick=True)
    assert records[0]["type"] == "metadata"
    divergent = next(r for r in read_records(path) if r["type"] == "divergence")
    assert "fake" in markdown(path)
    script = tmp_path / "repro.py"
    script.write_text(divergent["repro_script"])
    result = subprocess.run([sys.executable, str(script)], cwd=tmp_path, capture_output=True, text=True, check=True)
    assert "bitwise equal: False" in result.stdout


def test_gpu_targets_skip_cleanly_without_cuda(tmp_path) -> None:
    if torch.cuda.is_available():
        return
    records = run_targets(TARGETS, tmp_path / "skips.jsonl", quick=True)
    assert len([r for r in records if r["type"] == "skip"]) == len(TARGETS)
    assert not any(r["type"] == "error" for r in records)


def test_backend_block_size_filter() -> None:
    case = Case("fixture", "paged", "bfloat16", {}, lambda spec: torch.zeros(1),
                ("block_size",), allowed_block_sizes=(16, 32, 64))
    assert [spec.block_size for _, spec in probe_specs(case, quick=True)] == [32]
