"""Offline artifact replay, audit, and deterministic Markdown reports."""

from __future__ import annotations

import json
import hashlib
from pathlib import Path
from typing import Any

import numpy as np

from .calibration import PredictionRow, risk_upper
from .data import AuditResult
from .logic import decide, enumerate_worlds, evaluate_scalar, load_bundle, validate_ast
from .types import Accept, Ambiguous, BundleProvenance, OutOfScope, read_private_jsonl, read_public_jsonl


def _rows(run_dir: Path) -> list[PredictionRow]:
    return [PredictionRow(**json.loads(line)) for line in (run_dir / "predictions.jsonl").read_text().splitlines() if line]


def recompute_metrics(run_dir: Path) -> dict[str, Any]:
    private = {case.case_id: case for case in read_private_jsonl(run_dir / "data/private.jsonl")}
    public = {case.case_id: case for case in read_public_jsonl(run_dir / "data/public.jsonl")}
    if (run_dir / "data/g1_public.jsonl").exists():
        public.update({case.case_id: case for case in read_public_jsonl(run_dir / "data/g1_public.jsonl")})
        private.update({case.case_id: case for case in read_private_jsonl(run_dir / "data/g1_private.jsonl")})
    rows = _rows(run_dir)
    primary: dict[str, dict[str, list[PredictionRow]]] = {}
    for row in rows:
        if row.primary:
            primary.setdefault(row.panel_id, {}).setdefault(row.model_id, []).append(row)
    result: dict[str, Any] = {}
    for panel_id, models in primary.items():
        result["g0" if panel_id == "g0_fixture" else "g1"] = panel = {}
        M = len(models)
        for model_id, model_rows in sorted(models.items()):
            accepted = errors = 0
            for row in model_rows:
                if row.status == "accept":
                    accepted += 1
                    truth = private[row.case_id]
                    policy = validate_ast(row.policy_ast, public[row.case_id].schema)
                    errors += bool(row.action != evaluate_scalar(policy, public[row.case_id].schema, truth.true_world))
            N = len(model_rows)
            panel[model_id] = {
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
        g1_cases = []
        g1_labels = []
        g1_masks = None
        if record.get("stages", {}).get("g1", {}).get("status") in {"passed", "no_go"}:
            from .baselines import compile_oracle, compile_parser, marginal_product
            from .data import ManifestView, SplitManifest, audit_dataset, build_policy_pools, returns_schema
            from .logic import make_bundle
            from .types import SupportBelief

            g1_cases = read_public_jsonl(run_dir / "data/g1_public.jsonl")
            g1_labels = read_private_jsonl(run_dir / "data/g1_private.jsonl")
            public.update({case.case_id: case for case in g1_cases})
            private.update({case.case_id: case for case in g1_labels})
            manifest_raw = json.loads((run_dir / "data/g1_manifest.json").read_text())
            manifest = SplitManifest(manifest_raw["group_splits"], tuple(ManifestView(**view) for view in manifest_raw["views"]), manifest_raw["settings"])
            data_audit = audit_dataset(run_dir / "data/g1_public.jsonl", run_dir / "data/g1_private.jsonl", manifest, build_policy_pools(returns_schema(), record["config"]["data"]["policy_seed"]))
            errors.extend(data_audit.errors)
            with np.load(run_dir / "data/g1_masks.npz", allow_pickle=False) as arrays:
                g1_masks = {name: arrays[name].copy() for name in arrays.files}
            expected_models = {"oracle_joint", "conventional_parser", "oracle_marginal_product"}
            if set(g1_masks) != expected_models or any(mask.shape != (len(g1_cases), 64) or mask.dtype != np.bool_ for mask in g1_masks.values()):
                errors.append("G1 masks have incompatible shape or models")
            if len(g1_cases) != record["stages"]["g1"]["n_cases"] or len(g1_cases) != len(g1_labels):
                errors.append("G1 planned case count differs from data")
        rows = _rows(run_dir)
        if not rows or set(public) != set(private):
            errors.append("public/private cases or predictions are incomplete")
        g1_index = {case.case_id: index for index, case in enumerate(g1_cases)}
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
            if row.panel_id == "g1_dev":
                if row.case_id not in g1_index or relpath != "data/g1_masks.npz" or g1_masks is None:
                    errors.append(f"invalid G1 mask reference: {row.case_id}")
                    continue
                index = g1_index[row.case_id]
                oracle_bundle = compile_oracle(case, label)
                if row.model_id == "oracle_joint":
                    bundle = oracle_bundle
                elif row.model_id == "conventional_parser":
                    bundle = compile_parser(case)
                elif row.model_id == "oracle_marginal_product":
                    product = marginal_product(np.asarray(label.oracle_posterior), case.schema)
                    bundle = make_bundle(case.schema, SupportBelief(product > 0, product), product > 0, oracle_bundle.provenance)
                else:
                    errors.append(f"unknown G1 model: {row.model_id}")
                    continue
                expected_mask = np.zeros(64, dtype=np.bool_) if isinstance(bundle, OutOfScope) else bundle.retained
                if not np.array_equal(g1_masks[row.model_id][index], expected_mask):
                    errors.append(f"G1 support mismatch: {row.case_id}/{row.model_id}")
            else:
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
            result = bundle if isinstance(bundle, OutOfScope) else decide(bundle, policy)
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
                if row.panel_id == "g1_dev":
                    planned = record.get("g1_planned_primary", {}).get(row.group_id, {})
                    if row.policy_key_hex != planned.get("policy_key_hex") or row.policy_ast is None:
                        errors.append(f"primary G1 policy changed: {row.group_id}")
                else:
                    planned = record.get("planned_primary", {}).get(row.panel_id, {})
                    if row.policy_key_hex != planned.get("policy_key_hex") or row.group_id != planned.get("group_id"):
                        errors.append(f"primary policy changed: {row.panel_id}")
        for panel_id, planned in record.get("planned_primary", {}).items():
            if primary_seen.get((panel_id, planned["group_id"])) != set(planned["models"]):
                errors.append(f"planned panel incomplete: {panel_id}")
        for group_id, planned in record.get("g1_planned_primary", {}).items():
            if primary_seen.get(("g1_dev", group_id)) != set(planned["models"]):
                errors.append(f"planned G1 case incomplete: {group_id}")
        expected_metrics = recompute_metrics(run_dir)
        if json.loads((run_dir / "metrics.json").read_text()) != expected_metrics:
            errors.append("stored metrics differ from replay")
        stress_stage = record.get("stages", {}).get("g4_stress", {})
        if stress_stage.get("status") == "measured":
            from .stress import evaluate_stress

            expected_stress = evaluate_stress(record["config"], tuple(stress_stage["methods"]))
            if json.loads((run_dir / "stress.json").read_text()) != expected_stress:
                errors.append("stress results differ from deterministic replay")
        notes.append("Saved predictions and scalar decisions replayed; no learned or local-LLM model was used")
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
    g1 = record.get("stages", {}).get("g1")
    if g1 and g1.get("status") in {"passed", "no_go"}:
        g1_metrics = recompute_metrics(run_dir)["g1"]
        diagnostics = json.loads((run_dir / "data/g1_diagnostics.json").read_text())
        body[2] = f"Outcome: G1 {g1['status'].upper()} — {g1['reason']}. Conventional parsing is the strongest deployable baseline in the completed screen."
        body += ["## G1 generated-text development screen", "", f"{g1['n_cases']} independent generated development cases; no final calibration or test cases were evaluated. Wording is agent-authored generated text. Gate reason: `{g1['reason']}`. The optional local LLM was {g1['local_llm'].replace('_', ' ')}.", "", "| Method | Accepted / cases | Accepted errors | Coverage | One-sided risk upper bound (descriptive) |", "| --- | ---: | ---: | ---: | ---: |"]
        for method, numbers in sorted(g1_metrics.items()):
            body.append(f"| {method} | {numbers['accepted']} / {numbers['planned']} | {numbers['accepted_errors']} | {numbers['coverage']:.3f} | {numbers['risk_upper']:.3f} |")
        joint_gap = g1_metrics["oracle_joint"]["coverage"] - g1_metrics["oracle_marginal_product"]["coverage"]
        parser_gap = g1_metrics["oracle_joint"]["coverage"] - g1_metrics["conventional_parser"]["coverage"]
        body += ["", "The oracle uses private observation labels and is an information ceiling. The parser reads only the public case and schema. The marginal product uses oracle marginals and is an information-loss diagnostic. These are development results, not a 2% accepted-risk claim.", "", f"Joint-minus-marginal coverage: {joint_gap:.3f}; oracle-minus-parser coverage: {parser_gap:.3f}. The fixed majority-action baseline made {diagnostics['constant_baseline']['check_errors']} errors on {diagnostics['constant_baseline']['check_groups']} held-out development cases. Always-abstain coverage is zero and its conditional risk is undefined.", f"Renderer audit: {g1.get('rendering_review', 'pending')}; the examples were agent-authored and reviewed by the coding agent, not independent humans.", "", "## Hypotheses", "", f"- Representation: {('joint support has a measured development headroom gap over marginal products' if joint_gap >= 0.10 else 'the measured development headroom did not meet the 10-point gate')}; no learned representation claim.", "- Learning: not attempted because G1 stopped the learned route.", "- Auxiliary policy supervision: not attempted.", "", "## Limits and next step", "", "This controlled generator and its renderer grammar do not validate transfer to independently written cases. Retain the exact parser as the reference for this workload; only reopen learning if a separately designed workload shows credible headroom over it.", ""]
    if record.get("stages", {}).get("g4_stress", {}).get("status") == "measured":
        stress = json.loads((run_dir / "stress.json").read_text())
        old = stress["shift"]["old_calibration"]
        fresh = stress["shift"]["recalibrated"]
        body += ["## Stress and shift boundaries", "", f"The parser matched the declared behavior on {stress['targeted']['cases']} agent-authored targeted cases; semantic fixture mismatches: {stress['targeted']['semantic_failures']}. These repeated fixtures are not independent risk samples.", f"In a separate numeric counterexample, a frozen anti-correlation score retained the wrong XOR action on {old['accepted']} / {stress['shift']['test_groups']} shifted cases ({old['accepted_errors']} errors). Recalibration on separate shifted cases widened the retained set and accepted {fresh['accepted']} actions. The two populations have identical one-variable marginals; this is not a trained-model result.", ""]
    if audit.errors:
        body += ["## Audit failures", "", *[f"- {error}" for error in audit.errors], ""]
    if not g1 or g1.get("status") not in {"passed", "no_go"}:
        body += ["## Hypotheses", "", "- Representation: numeric example only; generated-workload comparison not attempted.", "- Learning: not attempted.", "- Auxiliary policy supervision: not attempted.", ""]
    body += ["## Reproduction", "", "```bash", "python -m pytest -q", f"python -m rbc verify --run {run_dir}", f"python -m rbc report --run {run_dir}", "```", ""]
    target = run_dir / "REPORT.md"
    target.write_text("\n".join(body))
    return target
