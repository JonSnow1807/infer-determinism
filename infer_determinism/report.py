"""Compact Markdown report for a JSONL result file."""
from __future__ import annotations

import json
from pathlib import Path


def read_records(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def markdown(path: Path) -> str:
    records = read_records(path)
    lines = ["# Inference determinism report", "", f"Source: `{path}`", "", "| Target | Case | Perturbation | First divergent config | Max abs diff | Differing elements | Max ULP | Record ID |", "|---|---|---|---|---:|---:|---:|---|"]
    seen: set[tuple[object, object, object]] = set()
    for row in records:
        if row.get("type") == "divergence":
            key = (row["target"], row["case"], row["perturbation"])
            if key in seen:
                continue
            seen.add(key)
            config = json.dumps(row["candidate"], sort_keys=True).replace("|", "\\|")
            lines.append(f"| {row['target']} | {row['case']} | {row['perturbation']} | `{config}` | {row['max_abs_diff']} | {row['differing_elements']} | {row['max_ulp_distance']} | `{row['id']}` |")
    if len(lines) == 6:
        lines.append("| — | No divergences found | — | — | — | — | — | — |")
    lines += ["", "## Run summary", ""]
    for kind in ("check", "pass", "divergence", "skip", "error"):
        matching = [r for r in records if r.get("type") == kind]
        lines.append(f"- {kind}: {len(matching)}")
    lines += ["", "## Skips and errors", ""]
    for row in records:
        if row.get("type") in ("skip", "error"):
            lines.append(f"- **{row.get('target')}/{row.get('case', '*')}**: {row.get('reason', row.get('error'))}")
    return "\n".join(lines) + "\n"


def _fmt(x: object) -> str:
    if isinstance(x, float):
        return f"{x:.2e}"
    if isinstance(x, dict):
        return ", ".join(f"{k}={_fmt(v)}" for k, v in x.items())
    return str(x)


def sweep_markdown(path: Path) -> str:
    """Autotune-config sweep report: one row per captured kernel launch."""
    records = read_records(path)
    meta = next((r for r in records if r.get("type") == "sweep_metadata"), {})
    versions = meta.get("versions", {}) if isinstance(meta, dict) else {}
    lines = ["# Autotune-config sweep report", "", f"Source: `{path}`", "",
             f"GPU: {meta.get('gpu')} (sm{''.join(map(str, meta.get('capability', [])))}), driver {meta.get('driver')}, "
             f"torch {versions.get('torch')}, triton {versions.get('triton_imported')}, vllm {versions.get('vllm')}, "
             f"fla {versions.get('fla-core') or versions.get('flash-linear-attention')}",
             "", f"ptxas used for sm_90: {versions.get('ptxas_sm90')}", ""]
    lines += ["## End-to-end outputs vs float32 reference", "",
              "| Scenario | Mode | Rel. error | Deterministic over repeats | Flags | Configs picked |", "|---|---|---|---|---|---|"]
    for r in records:
        if r.get("type") == "sweep_e2e":
            picked = "; ".join(f"{k}: {', '.join(sorted(set(v)))}" for k, v in r["picked"].items())
            lines.append(f"| {r['scenario']} | {r['mode']} | {_fmt(r['rel_err'])} | {r['deterministic']} ({r['repeats']} runs) | "
                         f"{', '.join(r['flags']) or '-'} | {picked} |")
    lines += ["", "## Per-kernel config sweep", "",
              "| Scenario | Kernel | Launch | Configs | Invalid | Errors | Nondeterministic | Flagged | Best rel. err vs ref | Flagged configs |",
              "|---|---|---:|---:|---:|---:|---:|---:|---|---|"]
    for r in records:
        if r.get("type") == "sweep_summary":
            labels = "; ".join(r["flagged_labels"]) or "-"
            lines.append(f"| {r['scenario']} | {r['kernel']} | {r['launch']} | {r['configs']} | {r['invalid']} | {r['errors']} | "
                         f"{len(r['nondeterministic'])} | {len(r['flagged'])} | {_fmt(r.get('best_rel_err_ref', '-'))} | {labels} |")
    flagged = [r for r in records if r.get("type") == "sweep_config" and r.get("flags")]
    lines += ["", "## Flagged configs", ""]
    if not flagged:
        lines.append("None.")
    else:
        lines += ["| Scenario | Kernel | Config | Flags | Differing elements (repeats) | Max ULP | Rel. err vs ref | Rel. err vs consensus |",
                  "|---|---|---|---|---:|---:|---|---|"]
        for r in flagged:
            lines.append(f"| {r['scenario']} | {r['kernel']} | {r['label']} | {', '.join(r['flags'])} | "
                         f"{r['repeat_differing_elements']} | {r['repeat_max_ulp']} | {_fmt(r.get('rel_err_ref', '-'))} | "
                         f"{_fmt(r.get('rel_err_consensus', '-'))} |")
    other = [r for r in records if r.get("type") in ("sweep_skip", "sweep_error")]
    errs = [r for r in records if r.get("type") == "sweep_config" and r.get("status") == "error"]
    lines += ["", "## Skips and errors", ""]
    for r in other:
        lines.append(f"- {r['type']} **{r['scenario']}**: {r.get('reason') or r.get('error')}")
    for r in errs:
        lines.append(f"- config error **{r['scenario']}/{r['kernel']}** `{r['label']}`: {r['error']}")
    if not other and not errs:
        lines.append("None.")
    return "\n".join(lines) + "\n"
