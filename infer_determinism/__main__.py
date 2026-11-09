"""Command-line interface."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .report import markdown, read_records, sweep_markdown
from .runner import TARGETS, run_targets


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="infer_determinism")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="run determinism probes")
    run.add_argument("--targets", nargs="+", default=list(TARGETS))
    run.add_argument("--quick", action="store_true")
    run.add_argument("--include", default=None, help="regex; only run cases whose name matches")
    run.add_argument("--out", type=Path, default=Path("results.jsonl"))
    report = sub.add_parser("report", help="print Markdown report")
    report.add_argument("path", type=Path)
    sweep = sub.add_parser("sweep", help="run every autotune config of the Triton kernels under a scenario")
    sweep.add_argument("--scenarios", nargs="+", default=None, help="default: all; see --list")
    sweep.add_argument("--list", action="store_true")
    sweep.add_argument("--out", type=Path, default=Path("sweep.jsonl"))
    sweep.add_argument("--repeats", type=int, default=3)
    sweep.add_argument("--kernels", default=None, help="regex on kernel names")
    sweep.add_argument("--no-e2e", action="store_true", help="skip the end-to-end autotune check")
    sweep.add_argument("--e2e-only", action="store_true")
    sweep.add_argument("--e2e-repeats", type=int, default=3)
    sweep.add_argument("--no-poison", action="store_true")
    sweep.add_argument("--append", action="store_true")
    sweep_report = sub.add_parser("sweep-report", help="print Markdown report for sweep JSONL")
    sweep_report.add_argument("path", type=Path)
    repro = sub.add_parser("repro", help="print standalone reproducer")
    repro.add_argument("record_id")
    repro.add_argument("--file", type=Path, default=Path("results.jsonl"))
    args = parser.parse_args(argv)
    if args.command == "run":
        records = run_targets(args.targets, args.out, args.quick, args.include)
        for kind in ("pass", "divergence", "skip", "error"):
            print(f"{kind}: {sum(r.get('type') == kind for r in records)}")
        print(f"Wrote {args.out}")
        return 1 if any(r.get("type") == "error" for r in records) else 0
    if args.command == "sweep":
        from .sweep_scenarios import SCENARIOS, run_sweep
        if args.list:
            for name, scenario in SCENARIOS.items():
                print(f"{name}: {scenario.description}")
            return 0
        names = args.scenarios or list(SCENARIOS)
        unknown = [n for n in names if n not in SCENARIOS]
        if unknown:
            parser.error(f"unknown scenario(s): {', '.join(unknown)}")
        records = run_sweep(names, args.out, repeats=args.repeats, kernels=args.kernels, e2e=not args.no_e2e,
                            sweep=not args.e2e_only, e2e_repeats=args.e2e_repeats, poison=not args.no_poison,
                            append=args.append)
        summaries = [r for r in records if r.get("type") == "sweep_summary"]
        flagged = sum(len(r["flagged"]) for r in summaries)
        e2e_flags = sum(bool(r.get("flags")) for r in records if r.get("type") == "sweep_e2e")
        config_errors = sum(r.get("type") == "sweep_config" and r.get("status") == "error" for r in records)
        print(f"config errors: {config_errors}")
        print(f"launches swept: {len(summaries)}, flagged configs: {flagged}, flagged e2e checks: {e2e_flags}, "
              f"skips: {sum(r.get('type') == 'sweep_skip' for r in records)}, errors: {sum(r.get('type') == 'sweep_error' for r in records)}")
        print(f"Wrote {args.out}")
        return 1 if config_errors or any(r.get("type") == "sweep_error" for r in records) else 0
    if args.command == "sweep-report":
        print(sweep_markdown(args.path), end="")
        return 0
    if args.command == "report":
        print(markdown(args.path), end="")
        return 0
    for record in read_records(args.file):
        if record.get("id") == args.record_id:
            print(record["repro_script"], end="")
            return 0
    parser.error(f"record {args.record_id!r} not found in {args.file}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
