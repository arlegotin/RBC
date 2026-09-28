"""Offline artifact replay, audit, and deterministic Markdown reports."""

from __future__ import annotations

import json
import hashlib
from pathlib import Path
from typing import Any

from .calibration import PredictionRow, risk_upper
from .data import AuditResult
from .logic import decide, enumerate_worlds, evaluate_scalar, load_bundle, validate_ast
from .types import Accept, Ambiguous, BundleProvenance, OutOfScope, read_private_jsonl, read_public_jsonl


def _rows(run_dir: Path) -> list[PredictionRow]:
    return [PredictionRow(**json.loads(line)) for line in (run_dir / "predictions.jsonl").read_text().splitlines() if line]


def recompute_metrics(run_dir: Path) -> dict[str, Any]:
    private = {case.case_id: case for case in read_private_jsonl(run_dir / "data/private.jsonl")}
    public = {case.case_id: case for case in read_public_jsonl(run_dir / "data/public.jsonl")}
    rows = _rows(run_dir)
    primary: dict[str, list[PredictionRow]] = {}
    for row in rows:
        if row.primary:
            primary.setdefault(row.model_id, []).append(row)
    result: dict[str, Any] = {"g0": {}}
    M = len(primary)
    for model_id, model_rows in sorted(primary.items()):
        accepted = errors = 0
        for row in model_rows:
            if row.status == "accept":
                accepted += 1
                truth = private[row.case_id]
                policy = validate_ast(row.policy_ast, public[row.case_id].schema)
                errors += bool(row.action != evaluate_scalar(policy, public[row.case_id].schema, truth.true_world))
        N = len(model_rows)
        result["g0"][model_id] = {
            "planned": N, "accepted": accepted, "accepted_errors": errors,
            "coverage": accepted / N if N else None,
            "risk": errors / accepted if accepted else None,
            "risk_upper": risk_upper(errors, accepted, 0.05 / M) if M else 1.0,
        }
    return result


def verify_run(run_dir: Path) -> AuditResult:
    errors: list[str] = []
    notes: list[str] = []
    try:
        record = json.loads((run_dir / "run.json").read_text())
        if record.get("stages", {}).get("g0", {}).get("status") != "passed":
            errors.append("G0 was not recorded as passed")
        encoded_config = json.dumps(record["config"], sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        if record.get("config_hash") != hashlib.sha256(encoded_config).hexdigest():
            errors.append("configuration hash mismatch")
        for relpath, expected in record.get("artifact_hashes", {}).items():
            target = (run_dir / relpath).resolve()
            if not target.is_relative_to(run_dir.resolve()) or not target.is_file() or hashlib.sha256(target.read_bytes()).hexdigest() != expected:
                errors.append(f"missing or changed artifact: {relpath}")
        if not record.get("artifact_hashes"):
            errors.append("artifact manifest missing")
        public = {case.case_id: case for case in read_public_jsonl(run_dir / "data/public.jsonl")}
        private = {case.case_id: case for case in read_private_jsonl(run_dir / "data/private.jsonl")}
        rows = _rows(run_dir)
        if not rows or set(public) != set(private):
            errors.append("public/private cases or predictions are incomplete")
        primary_seen: dict[tuple[str, str], set[str]] = {}
        for row in rows:
            case = public.get(row.case_id)
            label = private.get(row.case_id)
            if case is None or label is None:
                errors.append(f"prediction case missing: {row.case_id}")
                continue
            if row.group_id != label.source_group:
                errors.append(f"source group mismatch: {row.case_id}")
            relpath = row.bundle_relpath
            if not isinstance(relpath, str):
                errors.append(f"bundle reference missing: {row.model_id}")
                continue
            bundle_path = (run_dir / relpath).resolve()
            if not bundle_path.is_relative_to(run_dir.resolve()):
                errors.append(f"bundle reference escapes run: {relpath}")
                continue
            metadata = json.loads((bundle_path / "metadata.json").read_text())
            bundle = load_bundle(bundle_path, BundleProvenance(**metadata["provenance"]))
            if isinstance(bundle, OutOfScope):
                errors.append(f"invalid bundle: {relpath}: {bundle.reason}")
                continue
            policy = validate_ast(row.policy_ast, case.schema)
            result = decide(bundle, policy)
            if isinstance(result, Accept):
                if row.status != "accept" or row.action is not result.action:
                    errors.append(f"accepted action changed: {row.case_id}/{row.model_id}")
                worlds = enumerate_worlds(bundle.schema)
                for index, keep in zip(bundle.world_indices, bundle.retained):
                    if keep and evaluate_scalar(policy, bundle.schema, tuple(worlds[index])) is not row.action:
                        errors.append(f"scalar accepted action mismatch: {row.case_id}/{row.model_id}")
                        break
            elif isinstance(result, Ambiguous):
                if row.status != "ambiguous" or tuple(row.witnesses or ()) != (result.world_a, result.world_b):
                    errors.append(f"ambiguity witnesses changed: {row.case_id}/{row.model_id}")
            elif row.status != "out_of_scope":
                errors.append(f"out-of-scope prediction changed: {row.case_id}/{row.model_id}")
            if row.primary:
                primary_seen.setdefault((row.panel_id, row.group_id), set()).add(row.model_id)
                planned = record.get("planned_primary", {}).get(row.panel_id, {})
                if row.policy_key_hex != planned.get("policy_key_hex") or row.group_id != planned.get("group_id"):
                    errors.append(f"primary policy changed: {row.panel_id}")
        for panel_id, planned in record.get("planned_primary", {}).items():
            if primary_seen.get((panel_id, planned["group_id"])) != set(planned["models"]):
                errors.append(f"planned panel incomplete: {panel_id}")
        expected_metrics = recompute_metrics(run_dir)
        if json.loads((run_dir / "metrics.json").read_text()) != expected_metrics:
            errors.append("stored metrics differ from replay")
        notes.append("G0 predictions and scalar decisions replayed; no learned or local-LLM model was used")
    except (OSError, ValueError, KeyError, TypeError, IndexError, AttributeError, json.JSONDecodeError) as exc:
        errors.append(f"audit could not replay run: {exc}")
    return AuditResult(not errors, errors, notes)

def write_report(run_dir: Path) -> Path:
    record = json.loads((run_dir / "run.json").read_text())
    audit = verify_run(run_dir)
    stage = record.get("stages", {}).get("g0", {})
    if stage.get("status") != "passed":
        outcome = "G0 was not completed."
    elif not audit.ok:
        outcome = "G0 artifacts failed verification."
    else:
        outcome = "G0 passed: exact semantics and the numeric XOR fixture replayed."
    body = ["# RBC research report", "", f"Outcome: {outcome}", "", "## What ran", "", f"- G0 status: {stage.get('status', 'not_attempted')}", f"- Deterministic checks: {stage.get('test_output', 'unavailable')}", f"- Generated truth-in-support checks: {stage.get('generated_truth_checks', 'unavailable')}", f"- Audit: {'passed' if audit.ok else 'failed'}", ""]
    if stage.get("status") == "passed":
        metrics = recompute_metrics(run_dir)
        body += ["## Numeric representation fixture", "", "The oracle and exact XOR constraints have the same support. Each individual damage fact is unresolved; the exactly-one policy is invariant. The oracle marginal product adds worlds absent from the evidence.", "", "| Method | Primary accepted | Primary errors |", "| --- | ---: | ---: |"]
        for method, numbers in sorted(metrics["g0"].items()):
            body.append(f"| {method} | {numbers['accepted']} / {numbers['planned']} | {numbers['accepted_errors']} |")
        body += ["", "These are mathematical fixtures, not language-learning or accepted-risk evidence.", ""]
    if audit.errors:
        body += ["## Audit failures", "", *[f"- {error}" for error in audit.errors], ""]
    body += ["## Hypotheses", "", "- Representation: numeric example only; generated-workload comparison not attempted.", "- Learning: not attempted.", "- Auxiliary policy supervision: not attempted.", "", "## Reproduction", "", "```bash", "python -m pytest -q", f"python -m rbc verify --run {run_dir}", f"python -m rbc report --run {run_dir}", "```", ""]
    target = run_dir / "REPORT.md"
    target.write_text("\n".join(body))
    return target
