"""Command-line interface."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .report import markdown, read_records
from .runner import TARGETS, run_targets


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="infer_determinism")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="run determinism probes")
    run.add_argument("--targets", nargs="+", default=list(TARGETS))
    run.add_argument("--quick", action="store_true")
    run.add_argument("--out", type=Path, default=Path("results.jsonl"))
    report = sub.add_parser("report", help="print Markdown report")
    report.add_argument("path", type=Path)
    repro = sub.add_parser("repro", help="print standalone reproducer")
    repro.add_argument("record_id")
    repro.add_argument("--file", type=Path, default=Path("results.jsonl"))
    args = parser.parse_args(argv)
    if args.command == "run":
        records = run_targets(args.targets, args.out, args.quick)
        for kind in ("pass", "divergence", "skip", "error"):
            print(f"{kind}: {sum(r.get('type') == kind for r in records)}")
        print(f"Wrote {args.out}")
        return 1 if any(r.get("type") == "error" for r in records) else 0
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
