"""Offline artifact replay, audit, and deterministic Markdown reports."""

from __future__ import annotations

import json
import hashlib
import math
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
        if panel_id not in {"g0_fixture", "g1_dev"}:
            raise ValueError(f"unknown primary panel: {panel_id}")
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
    if "g1_dev" in primary:
        from .baselines import marginal_product, policy_probability

        g1_rows = primary["g1_dev"]
        labels = {case.case_id: case for case in read_private_jsonl(run_dir / "data/g1_private.jsonl")}
        ids = [case.case_id for case in read_public_jsonl(run_dir / "data/g1_public.jsonl")]
        with np.load(run_dir / "data/g1_masks.npz", allow_pickle=False) as arrays:
            masks = {name: arrays[name].copy() for name in arrays.files}
        by_model_case = {model: {row.case_id: row for row in model_rows} for model, model_rows in g1_rows.items()}
        result["g1_diagnostics"] = {}
        for model, keyed in by_model_case.items():
            retained = []
            state_hits = []
            nll = []
            brier = []
            for index, case_id in enumerate(ids):
                label = labels[case_id]
                case = public[case_id]
                mask = masks[model][index]
                retained.append(int(mask.sum()))
                state_hits.append(bool(mask[label.true_index]))
                posterior = np.asarray(label.oracle_posterior, dtype=np.float64)
                if model == "oracle_joint":
                    probabilities = posterior
                elif model == "oracle_marginal_product":
                    probabilities = marginal_product(posterior, case.schema)
                elif mask.any():
                    probabilities = mask.astype(np.float64) / mask.sum()
                else:
                    continue
                actual = bool(evaluate_scalar(validate_ast(keyed[case_id].policy_ast, case.schema), case.schema, label.true_world))
                brier.append((policy_probability(probabilities, validate_ast(keyed[case_id].policy_ast, case.schema), case.schema) - float(actual)) ** 2)
                nll.append(float(-np.log(probabilities[label.true_index])) if probabilities[label.true_index] > 0 else None)
            result["g1_diagnostics"][model] = {
                "state_set_coverage": sum(state_hits) / len(state_hits),
                "mean_retained_worlds": float(np.mean(retained)),
                "median_retained_worlds": float(np.median(retained)),
                "joint_nll": float(np.mean(nll)) if nll and all(value is not None for value in nll) else None,
                "policy_brier": float(np.mean(brier)) if brier else None,
                "probability_defined_cases": len(brier),
            }
        slices: dict[str, dict[str, dict[str, Any]]] = {}
        oracle_rows = by_model_case["oracle_joint"]
        for case_id in ids:
            label = labels[case_id]
            observation = label.observation["plan"]
            relation_order = max((len(probe["ids"]) for probe in observation["probes"] if probe["kind"] != "bit"), default=0)
            truth = bool(evaluate_scalar(validate_ast(oracle_rows[case_id].policy_ast, public[case_id].schema), public[case_id].schema, label.true_world))
            tags = {"component": observation["component"], "renderer": label.renderer, "workflow": observation["workflow"], "relation_order": str(relation_order), "oracle_ambiguity": oracle_rows[case_id].status, "action_label": str(truth).lower()}
            for attribute, value in tags.items():
                cell = slices.setdefault(attribute, {}).setdefault(value, {})
                for model, keyed in by_model_case.items():
                    row = keyed[case_id]
                    counts = cell.setdefault(model, {"cases": 0, "accepted": 0, "accepted_errors": 0})
                    counts["cases"] += 1
                    counts["accepted"] += int(row.status == "accept")
                    counts["accepted_errors"] += int(row.status == "accept" and row.action != truth)
        for values in slices.values():
            for models in values.values():
                for counts in models.values():
                    counts["coverage"] = counts["accepted"] / counts["cases"]
        result["g1_slices"] = slices
    return result


def verify_run(run_dir: Path, *, require_report: bool = True) -> AuditResult:
    errors: list[str] = []
    notes: list[str] = []
    try:
        record = json.loads((run_dir / "run.json").read_text())
        if record.get("stages", {}).get("g0", {}).get("status") != "passed":
            errors.append("G0 was not recorded as passed")
        encoded_config = json.dumps(record["config"], sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        if record.get("config_hash") != hashlib.sha256(encoded_config).hexdigest():
            errors.append("configuration hash mismatch")
        from .experiment import _replay_source_fingerprint

        if record.get("replay_source_fingerprint") != _replay_source_fingerprint():
            errors.append("replay source code differs from the sealed run version")
        for relpath, expected in record.get("artifact_hashes", {}).items():
            target = (run_dir / relpath).resolve()
            if not target.is_relative_to(run_dir.resolve()) or not target.is_file() or hashlib.sha256(target.read_bytes()).hexdigest() != expected:
                errors.append(f"missing or changed artifact: {relpath}")
        if not record.get("artifact_hashes"):
            errors.append("artifact manifest missing")
        snapshots = record.get("source_snapshots", {})
        source_fingerprints = record.get("g0_source_fingerprint", {})
        if not snapshots or len(snapshots) != len(source_fingerprints):
            errors.append("G0 source snapshots are missing or incomplete")
        for relpath, expected in snapshots.items():
            target = (run_dir / relpath).resolve()
            original = relpath.removeprefix("source/")
            if not relpath.startswith("source/src/rbc/") or source_fingerprints.get(original) != expected or not target.is_relative_to(run_dir.resolve()) or not target.is_file() or hashlib.sha256(target.read_bytes()).hexdigest() != expected:
                errors.append(f"invalid source snapshot: {relpath}")
        public = {case.case_id: case for case in read_public_jsonl(run_dir / "data/public.jsonl")}
        private = {case.case_id: case for case in read_private_jsonl(run_dir / "data/private.jsonl")}
        g1_cases = []
        g1_labels = []
        g1_masks = None
        if record.get("stages", {}).get("g1", {}).get("status") in {"passed", "no_go"}:
            from .baselines import compile_oracle, compile_parser, marginal_product
            from dataclasses import replace
            from .data import ManifestView, SplitManifest, audit_dataset, build_policy_pools, generate_source, policy_key, returns_schema, select_primary_policy
            from .logic import make_bundle
            from .types import SupportBelief, private_case_to_dict

            parser_snapshot = record.get("g1_parser_source_snapshot", {})
            parser_relpath = parser_snapshot.get("path")
            if parser_relpath != "source/src/rbc/baselines.py":
                errors.append("G1 parser source snapshot is missing")
            else:
                parser_target = (run_dir / parser_relpath).resolve()
                if not parser_target.is_relative_to(run_dir.resolve()) or not parser_target.is_file() or hashlib.sha256(parser_target.read_bytes()).hexdigest() != parser_snapshot.get("sha256"):
                    errors.append("G1 parser source snapshot is invalid")

            g1_cases = read_public_jsonl(run_dir / "data/g1_public.jsonl")
            g1_labels = read_private_jsonl(run_dir / "data/g1_private.jsonl")
            public.update({case.case_id: case for case in g1_cases})
            private.update({case.case_id: case for case in g1_labels})
            manifest_raw = json.loads((run_dir / "data/g1_manifest.json").read_text())
            manifest = SplitManifest(manifest_raw["group_splits"], tuple(ManifestView(**view) for view in manifest_raw["views"]), manifest_raw["settings"])
            pools = build_policy_pools(returns_schema(), record["config"]["data"]["policy_seed"])
            data_audit = audit_dataset(run_dir / "data/g1_public.jsonl", run_dir / "data/g1_private.jsonl", manifest, pools)
            errors.extend(data_audit.errors)
            with np.load(run_dir / "data/g1_masks.npz", allow_pickle=False) as arrays:
                g1_masks = {name: arrays[name].copy() for name in arrays.files}
            expected_models = {"oracle_joint", "conventional_parser", "oracle_marginal_product"}
            if set(g1_masks) != expected_models or any(mask.shape != (len(g1_cases), 64) or mask.dtype != np.bool_ for mask in g1_masks.values()):
                errors.append("G1 masks have incompatible shape or models")
            if len(g1_cases) != record["stages"]["g1"]["n_cases"] or len(g1_cases) != len(g1_labels):
                errors.append("G1 planned case count differs from data")
            for case, label in zip(g1_cases, g1_labels, strict=True):
                generated_cfg = {**record["config"], "data": {**record["config"]["data"], "renderer_family": label.renderer}}
                regenerated = generate_source(generated_cfg, case.case_id, label.generation_seed)
                generated_private = replace(regenerated.private, split="dev")
                if regenerated.public != case or json.dumps(private_case_to_dict(generated_private), sort_keys=True) != json.dumps(private_case_to_dict(label), sort_keys=True):
                    errors.append(f"G1 case differs from frozen generator seed: {case.case_id}")
                policy_rng = np.random.default_rng(record["config"]["data"]["policy_seed"] + int(case.case_id.split("-")[-1]))
                expected_policy = select_primary_policy(case, {"workflow": regenerated.workflow, "queried_item": regenerated.queried_item}, pools.dev[regenerated.workflow], policy_rng)
                expected_key = policy_key(expected_policy, case.schema).hex()
                if record.get("g1_planned_primary", {}).get(case.case_id, {}).get("policy_key_hex") != expected_key:
                    errors.append(f"G1 primary policy selection changed: {case.case_id}")
        rows = _rows(run_dir)
        if not rows or set(public) != set(private):
            errors.append("public/private cases or predictions are incomplete")
        g1_index = {case.case_id: index for index, case in enumerate(g1_cases)}
        primary_seen: dict[tuple[str, str], set[str]] = {}
        primary_counts: dict[tuple[str, str, str], int] = {}
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
            from .data import policy_key
            if row.policy_key_hex != policy_key(policy, case.schema).hex():
                errors.append(f"policy truth table differs from saved key: {row.case_id}/{row.model_id}")
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
                primary_key = (row.panel_id, row.group_id, row.model_id)
                primary_counts[primary_key] = primary_counts.get(primary_key, 0) + 1
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
        if any(count != 1 for count in primary_counts.values()):
            errors.append("duplicate primary prediction rows")
        expected_primary_groups = {(panel, planned["group_id"]) for panel, planned in record.get("planned_primary", {}).items()}
        expected_primary_groups.update(("g1_dev", group) for group in record.get("g1_planned_primary", {}))
        if set(primary_seen) != expected_primary_groups:
            errors.append("unplanned or missing primary source groups")
        expected_metrics = recompute_metrics(run_dir)
        if json.loads((run_dir / "metrics.json").read_text()) != expected_metrics:
            errors.append("stored metrics differ from replay")
        g1_stage = record.get("stages", {}).get("g1", {})
        if g1_stage.get("status") in {"passed", "no_go"}:
            from .calibration import paired_coverage_interval
            from .experiment import evaluate_g1

            g1_metrics = expected_metrics["g1"]
            oracle = g1_metrics["oracle_joint"]
            parser = g1_metrics["conventional_parser"]
            marginal = g1_metrics["oracle_marginal_product"]
            g1_rows = [row for row in rows if row.panel_id == "g1_dev"]
            _, lower, _ = paired_coverage_interval(
                [row for row in g1_rows if row.model_id == "oracle_joint"],
                [row for row in g1_rows if row.model_id == "conventional_parser"],
                record["config"]["statistics"]["bootstrap_resamples"],
                record["config"]["statistics"]["bootstrap_seed"],
            )
            decision = evaluate_g1({"oracle_coverage": oracle["coverage"], "marginal_coverage": marginal["coverage"], "parser_coverage": parser["coverage"], "parser_errors": parser["accepted_errors"], "parser_acceptances": parser["accepted"], "paired_lower_vs_parser": lower, "n_cases": oracle["planned"], "scope": "mixture"}, record["config"])
            if any(g1_stage.get(key) != decision[key] for key in ("status", "reason", "claim_scope")):
                errors.append("G1 decision differs from replayed development metrics")
            claims = record.get("claims", {})
            if decision["status"] == "no_go" and (decision["reason"] not in claims.get("H1", "") or claims.get("H2") != "NOT_ATTEMPTED_G1_NO_GO" or claims.get("H3") != "NOT_ATTEMPTED_G1_NO_GO"):
                errors.append("G1 decision claims differ from stopped-route status")
        stress_stage = record.get("stages", {}).get("g4_stress", {})
        if stress_stage.get("status") == "measured":
            from .stress import evaluate_stress

            expected_stress = evaluate_stress(record["config"], tuple(stress_stage["methods"]))
            if json.loads((run_dir / "stress.json").read_text()) != expected_stress:
                errors.append("stress results differ from deterministic replay")
        timing_stage = record.get("stages", {}).get("g4_timing", {})
        if timing_stage.get("status") == "measured":
            from .timing import economic_sensitivity

            timing = json.loads((run_dir / "timing.json").read_text())
            measured = timing["measured"]
            needed = {"compile_ms", "one_policy_ms"} | {f"{prefix}_{count}_ms" for prefix in ("simultaneous", "future_only", "compile_plus_future") for count in (1, 4, 16)}
            if measured["method"] != "conventional_parser" or measured["samples"] < 100 or measured["warmups"] < 10 or measured["batch_size"] != 1 or set(measured["latency_ms"]) != needed:
                errors.append("timing stage is missing required full-pipeline measurements")
            if set(measured["case_ids"]) - {case.case_id for case in g1_cases} or len(set(measured["case_ids"])) != measured["samples"]:
                errors.append("timing case IDs are not distinct completed G1 cases")
            if not math.isfinite(measured["cold_process_import_compile_ms"]) or measured["cold_process_import_compile_ms"] < 0:
                errors.append("invalid cold-load timing")
            for name, values in measured["latency_ms"].items():
                if not all(math.isfinite(values[key]) and values[key] >= 0 for key in ("p50", "p95", "mean")) or values["p95"] < values["p50"]:
                    errors.append(f"invalid timing summary: {name}")
            input_metrics = expected_metrics["g1"]["conventional_parser"]
            expected_economics = economic_sensitivity({"compile_plus_future_16_ms": measured["latency_ms"]["compile_plus_future_16_ms"]["mean"], "coverage": input_metrics["coverage"]}, timing["economic_assumptions"])
            if timing["economic_sensitivity"] != expected_economics:
                errors.append("economic sensitivity does not match measured inputs and assumptions")
        notes.append("Saved predictions and scalar decisions replayed; no learned or local-LLM model was used")
        if require_report and not errors:
            report_path = run_dir / "REPORT.md"
            if not report_path.is_file():
                errors.append("report artifact is missing")
            elif report_path.read_text() != _render_report(run_dir, AuditResult(True, [], notes)):
                errors.append("report artifact differs from deterministic rendering")
    except (OSError, ValueError, KeyError, TypeError, IndexError, AttributeError, json.JSONDecodeError) as exc:
        errors.append(f"audit could not replay run: {exc}")
    return AuditResult(not errors, errors, notes)

def _render_report(run_dir: Path, audit: AuditResult) -> str:
    record = json.loads((run_dir / "run.json").read_text())
    stage = record.get("stages", {}).get("g0", {})
    if not audit.ok:
        body = ["# Outcome: verification failed", "", "## Audit failures", "", *[f"- {error}" for error in audit.errors], "", "The saved result is not eligible for a scientific conclusion until these artifact failures are resolved.", ""]
        return "\n".join(body)
    if stage.get("status") != "passed":
        outcome = "G0 was not completed."
    elif not audit.ok:
        outcome = "G0 artifacts failed verification."
    else:
        outcome = "G0 passed: exact semantics and the numeric XOR fixture replayed."
    body = [f"# Outcome: {outcome}", "", "## What ran", "", f"- G0 status: {stage.get('status', 'not_attempted')}", f"- Deterministic checks: {stage.get('test_output', 'unavailable').replace(chr(10), ' ')}", f"- Generated truth-in-support checks: {stage.get('generated_truth_checks', 'unavailable')}", f"- Audit: {'passed' if audit.ok else 'failed'}", ""]
    if stage.get("status") == "passed":
        metrics = recompute_metrics(run_dir)
        body += ["## Numeric representation fixture", "", "The oracle and exact XOR constraints have the same support. Each individual damage fact is unresolved; the exactly-one policy is invariant. The oracle marginal product adds worlds absent from the evidence.", "", "| Method | Primary accepted | Primary errors |", "| --- | ---: | ---: |"]
        for method, numbers in sorted(metrics["g0"].items()):
            body.append(f"| {method} | {numbers['accepted']} / {numbers['planned']} | {numbers['accepted_errors']} |")
        body += ["", "These are mathematical fixtures, not language-learning or accepted-risk evidence.", ""]
    g1 = record.get("stages", {}).get("g1")
    if g1 and g1.get("status") in {"passed", "no_go"}:
        all_metrics = recompute_metrics(run_dir)
        g1_metrics = all_metrics["g1"]
        diagnostics = json.loads((run_dir / "data/g1_diagnostics.json").read_text())
        body[0] = f"# Outcome: G1 {g1['status'].upper()} — {g1['reason']}. Conventional parsing is the strongest deployable baseline in the completed screen."
        body += ["## G1 generated-text development screen", "", f"{g1['n_cases']} independent generated development cases; no final calibration or test cases were evaluated. Wording is agent-authored generated text. Gate reason: `{g1['reason']}`. The optional local LLM was {g1['local_llm'].replace('_', ' ')}.", "", "| Method | Accepted / cases | Accepted errors | Coverage | One-sided risk upper bound (descriptive) |", "| --- | ---: | ---: | ---: | ---: |"]
        for method, numbers in sorted(g1_metrics.items()):
            body.append(f"| {method} | {numbers['accepted']} / {numbers['planned']} | {numbers['accepted_errors']} | {numbers['coverage']:.3f} | {numbers['risk_upper']:.3f} |")
        body += ["", "| Method | True state retained | Mean / median retained worlds | Joint NLL | Policy Brier |", "| --- | ---: | ---: | ---: | ---: |"]
        for method, values in sorted(all_metrics["g1_diagnostics"].items()):
            nll = "undefined" if values["joint_nll"] is None else f"{values['joint_nll']:.3f}"
            brier = "undefined" if values["policy_brier"] is None else f"{values['policy_brier']:.3f}"
            body.append(f"| {method} | {values['state_set_coverage']:.3f} | {values['mean_retained_worlds']:.1f} / {values['median_retained_worlds']:.1f} | {nll} | {brier} |")
        body += ["", "### Generated case slices", "", "The following rows are descriptive development slices. The full metrics file also groups by workflow, relation order, oracle ambiguity, and true action label.", "", "| Slice | Cases | Oracle accepted | Parser accepted | Marginal accepted |", "| --- | ---: | ---: | ---: | ---: |"]
        for attribute in ("component", "renderer"):
            for value, methods in sorted(all_metrics["g1_slices"][attribute].items()):
                body.append(f"| {attribute}={value} | {methods['oracle_joint']['cases']} | {methods['oracle_joint']['accepted']} | {methods['conventional_parser']['accepted']} | {methods['oracle_marginal_product']['accepted']} |")
        joint_gap = g1_metrics["oracle_joint"]["coverage"] - g1_metrics["oracle_marginal_product"]["coverage"]
        parser_gap = g1_metrics["oracle_joint"]["coverage"] - g1_metrics["conventional_parser"]["coverage"]
        body += ["", "The oracle uses private observation labels and is an information ceiling. The parser reads only the public case and schema. The marginal product uses oracle marginals and is an information-loss diagnostic. These are development results, not a 2% accepted-risk claim.", "", f"Joint-minus-marginal coverage: {joint_gap:.3f}; oracle-minus-parser coverage: {parser_gap:.3f}. The fixed majority-action baseline accepted all {diagnostics['constant_baseline']['check_groups']} held-out development cases and made {diagnostics['constant_baseline']['check_errors']} errors. Always-abstain coverage is zero and its conditional risk is undefined.", f"Renderer audit: {g1.get('rendering_review', 'pending')}; the examples were agent-authored and reviewed by the coding agent, not independent humans.", "", "## Hypotheses", "", f"- Representation: {('joint support has a measured development headroom gap over marginal products' if joint_gap >= 0.10 else 'the measured development headroom did not meet the 10-point gate')}; no learned representation claim.", "- Learning: not attempted because G1 stopped the learned route.", "- Auxiliary policy supervision: not attempted.", "", "## Limits and next step", "", "This controlled generator and its renderer grammar do not validate transfer to independently written cases. Retain the exact parser as the reference for this workload; only reopen learning if a separately designed workload shows credible headroom over it.", ""]
        direct_rows = ["### Oracle direct-policy readout", "", "This uses private oracle probabilities and is only an information ceiling. Thresholds were not chosen from these development outcomes for a final claim.", "", "| Probability threshold | Accepted / 500 | Errors |", "| ---: | ---: | ---: |"]
        for point in diagnostics["oracle_direct_curve"]:
            direct_rows.append(f"| {point['threshold']:.3f} | {point['accepted']} | {point['errors']} |")
        direct_rows.append("")
        insertion = body.index("## Hypotheses")
        body[insertion:insertion] = direct_rows
    if record.get("stages", {}).get("g4_stress", {}).get("status") == "measured":
        stress = json.loads((run_dir / "stress.json").read_text())
        old = stress["shift"]["old_calibration"]
        fresh = stress["shift"]["recalibrated"]
        body += ["## Stress and shift boundaries", "", f"The parser matched the declared behavior on {stress['targeted']['cases']} agent-authored targeted cases; semantic fixture mismatches: {stress['targeted']['semantic_failures']}. These repeated fixtures are not independent risk samples.", f"In a separate numeric counterexample, a frozen anti-correlation score retained the wrong XOR action on {old['accepted']} / {stress['shift']['test_groups']} shifted cases ({old['accepted_errors']} errors). Recalibration on separate shifted cases widened the retained set and accepted {fresh['accepted']} actions. The two populations have identical one-variable marginals; this is not a trained-model result.", ""]
    if record.get("stages", {}).get("g4_timing", {}).get("status") == "measured":
        timing = json.loads((run_dir / "timing.json").read_text())
        measured = timing["measured"]
        body += ["## Measured local work", "", f"On the recorded {measured['hardware'].get('chip') or measured['hardware']['architecture']} machine ({measured['hardware']['physical_memory_bytes'] / 1e9:.1f} GB physical memory), the public parser ran {measured['warmups']} warmups and {measured['samples']} timed cases at batch size one. Cold process import plus first compile: {measured['cold_process_import_compile_ms']:.3f} ms. Peak process RSS: {measured['peak_process_rss_bytes'] / 1e6:.1f} MB. No encoder or LLM timing was measured.", "", "| Pipeline stage | p50 ms | p95 ms |", "| --- | ---: | ---: |"]
        for name in ("compile_ms", "one_policy_ms", "simultaneous_1_ms", "simultaneous_4_ms", "simultaneous_16_ms", "future_only_1_ms", "future_only_4_ms", "future_only_16_ms", "compile_plus_future_16_ms"):
            values = measured["latency_ms"][name]
            body.append(f"| {name} | {values['p50']:.4f} | {values['p95']:.4f} |")
        body += ["", "The parser was compiled once per case in every 1/4/16-policy scenario. These timings do not establish an RBC speed advantage because no learned route was built.", "", "## Hypothetical cost sensitivity", "", "Money inputs are hypothetical; review fraction comes from the generated G1 parser coverage. End-to-end success is unknown, so cost per successful case is unavailable.", "", "| Review cost | Parser vs all-review cost/case | Zero review advantage cost/case | Hypothetical tenth machine-cost alternative |", "| ---: | ---: | ---: | ---: |"]
        by_cost = {}
        for row in timing["economic_sensitivity"]:
            by_cost.setdefault(row["review_cost"], {})[row["scenario"]] = row["cost_per_case"]
        for cost, scenarios in sorted(by_cost.items()):
            body.append(f"| ${cost:.2f} | ${scenarios['parser_vs_always_review']:.4f} | ${scenarios['zero_review_advantage']:.4f} | ${scenarios['hypothetical_llm_tenth_machine_cost']:.4f} |")
        body += [""]
    if g1 and g1.get("status") in {"passed", "no_go"}:
        body += ["## Explicit result statuses", "", "- `INSUFFICIENT_JOINT_MARGINAL_HEADROOM`: the measured joint support gain was below ten points.", "- `NO_HEADROOM_OVER_SIMPLE_BASELINE`: the competent public parser tied oracle support on all 500 development cases.", "- `EXTERNAL_VALIDATION_NOT_PERFORMED`: no independently written cases were available.", "- Learned heads, auxiliary-loss ablation, and local-LLM routes: not attempted after the G1 no-go; no result is imputed to them.", ""]
    environment = record.get("environment", {})
    body += ["## Environment and budget", "", f"Effective configuration: `runs/{record['config']['run_id']}/run.json`; tested Python {environment.get('python', 'unrecorded')}, {environment.get('architecture', 'unrecorded')}, macOS {environment.get('macos', 'unrecorded')}. Core package versions: {json.dumps({name: environment.get('packages', {}).get(name) for name in ('numpy', 'scipy', 'PyYAML', 'pytest')}, sort_keys=True)}.", f"Measured experiment compute charged: {sum(record.get('budget_charges', {}).values()):.2f} s of {record.get('config', {}).get('budget', {}).get('total_seconds', 'unknown')} s ceiling. Downloaded learned and local-LLM model files: none.", ""]
    if not g1 or g1.get("status") not in {"passed", "no_go"}:
        body += ["## Hypotheses", "", "- Representation: numeric example only; generated-workload comparison not attempted.", "- Learning: not attempted.", "- Auxiliary policy supervision: not attempted.", ""]
    run_ref = f"runs/{record['config']['run_id']}"
    body += ["## Reproduction", "", "```bash", "python -m pytest -q", "python -m rbc run --through g0 --config configs/poc.yaml", "python -m rbc run --through g1 --config configs/poc.yaml", f"python -m rbc verify --run {run_ref}", f"python -m rbc report --run {run_ref}", "```", ""]
    return "\n".join(body)


def write_report(run_dir: Path) -> Path:
    audit = verify_run(run_dir, require_report=False)
    target = run_dir / "REPORT.md"
    target.write_text(_render_report(run_dir, audit))
    return target
