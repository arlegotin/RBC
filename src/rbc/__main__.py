"""Command-line entry point for local RBC runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m rbc")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("doctor")
    run = sub.add_parser("run")
    run.add_argument("--through", required=True, choices=("g0", "g1", "g2", "g3"))
    run.add_argument("--config", required=True, type=Path)
    verify = sub.add_parser("verify")
    verify.add_argument("--run", required=True, type=Path)
    report = sub.add_parser("report")
    report.add_argument("--run", required=True, type=Path)
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        if exc.code == 0:
            return 0
        raise
    if args.command == "doctor":
        from .experiment import doctor

        print(json.dumps(doctor(), indent=2, sort_keys=True))
        return 0
    if args.command == "report":
        from .report import verify_run, write_report

        print(write_report(args.run))
        return 0 if verify_run(args.run).ok else 1
    if args.command == "verify":
        from .report import verify_run

        audit = verify_run(args.run)
        print(json.dumps({"ok": audit.ok, "errors": audit.errors, "notes": audit.notes}, indent=2, sort_keys=True))
        return 0 if audit.ok else 1
    if args.command == "run":
        from .experiment import load_config, open_run, run_g0, run_g1

        ctx = open_run(load_config(args.config))
        try:
            stage = run_g0(ctx) if args.through == "g0" else run_g1(ctx)
        finally:
            ctx.close()
        if args.through in {"g2", "g3"}:
            if stage["status"] == "no_go":
                print(json.dumps({"status": "blocked_by_g1", "requested": args.through, "reason": stage["reason"]}, indent=2, sort_keys=True))
            else:
                print(json.dumps({"status": "not_implemented", "requested": args.through, "g1_status": stage["status"]}, indent=2, sort_keys=True))
            return 2
        print(json.dumps(stage, indent=2, sort_keys=True))
        return 0 if stage["status"] in {"passed", "no_go"} else 1
    raise AssertionError("argparse allowed an unknown command")


if __name__ == "__main__":
    raise SystemExit(main())
