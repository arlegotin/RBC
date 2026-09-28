"""Human-readable reports from recorded local experiments."""

from __future__ import annotations

import json
from pathlib import Path


def write_report(run_dir: Path) -> Path:
    record = json.loads((run_dir / "run.json").read_text())
    stages = record.get("stages", {})
    body = ["# RBC research report", "", "Outcome: no executed research gate yet.", "", "## Execution", ""]
    if stages:
        body += [f"- {name}: {stage['status']}" for name, stage in sorted(stages.items())]
    else:
        body.append("No gates executed.")
    body += ["", "Numerical results: unavailable.", ""]
    target = run_dir / "REPORT.md"
    target.write_text("\n".join(body))
    return target
