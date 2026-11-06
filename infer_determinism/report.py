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
