"""Local run configuration, resource accounting, and stage records."""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterator

import yaml
import numpy as np


_KEYS: dict[str, set[str]] = {
    "": {"format_version", "run_id", "run_root", "budget", "downloads", "memory", "schema", "data", "encoder", "training", "statistics", "local_llm", "timing", "numerics"},
    "budget": {"total_seconds", "llm_seconds", "repair_seconds"},
    "downloads": {"core_bytes", "llm_bytes"},
    "memory": {"encoder_gib", "llm_gib", "reserve_fraction"},
    "schema": {"kernel_max", "learned_slots", "ast_depth", "ast_nodes"},
    "data": {"seed", "split_seed", "policy_seed", "mixture", "stress_groups", "g1_dev", "pilot_train", "pilot_dev", "train", "dev", "calibration", "test", "train_renderer_weights", "dev_renderer_weights", "target_renderer_weights"},
    "encoder": {"repo", "batch", "max_tokens", "num_workers"},
    "training": {"seeds", "lr", "batch", "epochs", "lambdas", "betas", "eps", "weight_decay", "queries_per_update", "h3_gate", "curve_sizes", "repair_attempts"},
    "statistics": {"risk_target", "family_delta", "bootstrap_resamples", "bootstrap_seed", "alpha_grid", "confidence_grid"},
    "local_llm": {"enabled", "repo", "probe", "dev", "calibration", "test", "prompts_per_route", "penalties", "syntax_repairs"},
    "timing": {"warmups", "cheap_cases", "policy_counts"},
    "numerics": {"atol", "rtol"},
}


def load_config(path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, dict):
        raise ValueError("config must be a mapping")
    for section, allowed in _KEYS.items():
        node = raw if not section else raw.get(section, {})
        if not isinstance(node, dict):
            raise ValueError(f"config {section} must be a mapping")
        unknown = set(node) - allowed
        if unknown:
            raise ValueError(f"unknown config keys in {section or 'root'}: {sorted(unknown)}")
    if raw.get("format_version") != 1:
        raise ValueError("unsupported config format_version")
    _validate_run_id(raw.get("run_id"))
    for name in ("total_seconds", "llm_seconds", "repair_seconds"):
        if not isinstance(raw.get("budget", {}).get(name), int) or raw["budget"][name] <= 0:
            raise ValueError(f"invalid budget {name}")
    return raw


def _validate_run_id(run_id: Any) -> None:
    if not isinstance(run_id, str) or run_id in {".", ".."} or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", run_id) is None:
        raise ValueError("invalid run_id")


@dataclass
class BudgetLedger:
    total_seconds: float
    llm_seconds: float
    repair_seconds: float
    charges: dict[str, float] | None = None
    clock: Callable[[], float] = time.perf_counter

    def __post_init__(self) -> None:
        self.charges = dict(self.charges or {})
        for amount in self.charges.values():
            if not math.isfinite(amount) or amount < 0:
                raise ValueError("invalid recorded charge")

    def remaining(self, category: str) -> float:
        if category == "total":
            return self.total_seconds - sum(self.charges.values())
        if category == "llm":
            return min(self.llm_seconds - self.charges.get("llm", 0), self.remaining("total"))
        if category == "repair":
            return min(self.repair_seconds - self.charges.get("repair", 0), self.remaining("total"))
        if category == "core":
            return self.remaining("total")
        raise ValueError(f"unknown budget category: {category}")

    def can_start(self, category: str, forecast_seconds: float) -> bool:
        if not math.isfinite(forecast_seconds) or forecast_seconds < 0:
            raise ValueError("invalid forecast")
        return forecast_seconds <= self.remaining(category)

    def charge(self, category: str, seconds: float) -> None:
        if category not in {"core", "llm", "repair"}:
            raise ValueError("invalid charge category")
        if not math.isfinite(seconds) or seconds < 0:
            raise ValueError("invalid charge")
        self.charges[category] = self.charges.get(category, 0.0) + seconds

    @contextmanager
    def measure(self, category: str) -> Iterator[None]:
        started = self.clock()
        try:
            yield
        finally:
            self.charge(category, max(0.0, self.clock() - started))


@dataclass
class RunContext:
    path: Path
    config: dict[str, Any]
    record: dict[str, Any]
    budget: BudgetLedger
    lock_path: Path

    def close(self) -> None:
        self.lock_path.unlink(missing_ok=True)


def _hash_json(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def open_run(config: dict[str, Any]) -> RunContext:
    _validate_run_id(config.get("run_id"))
    run_root = Path(config.get("run_root", "runs")).resolve()
    path = (run_root / config["run_id"]).resolve()
    if path.parent != run_root:
        raise ValueError("run path escapes root")
    path.mkdir(parents=True, exist_ok=True)
    lock_path = path / ".run.lock"
    try:
        fd = os.open(lock_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as exc:
        try:
            old_pid = int(lock_path.read_text().strip())
            os.kill(old_pid, 0)
        except ProcessLookupError:
            lock_path.unlink()
            fd = os.open(lock_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except (ValueError, PermissionError, OSError):
            raise RuntimeError("run is locked") from exc
        else:
            raise RuntimeError("run is locked") from exc
    with os.fdopen(fd, "w") as lock:
        lock.write(str(os.getpid()))
    try:
        content_hash = _hash_json(config)
        record_path = path / "run.json"
        if record_path.exists():
            record = json.loads(record_path.read_text())
            if record.get("config_hash") != content_hash:
                raise ValueError("incompatible run configuration")
        else:
            record = {"format_version": 1, "config": config, "config_hash": content_hash, "stages": {}, "budget_charges": {}, "claims": {"H1": "not_attempted", "H2": "not_attempted", "H3": "not_attempted"}}
        b = config["budget"]
        budget = BudgetLedger(b["total_seconds"], b["llm_seconds"], b["repair_seconds"], record.get("budget_charges", {}))
        return RunContext(path, config, record, budget, lock_path)
    except Exception:
        lock_path.unlink(missing_ok=True)
        raise


def write_run(ctx: RunContext) -> None:
    ctx.record["budget_charges"] = dict(ctx.budget.charges)
    ctx.record["config_hash"] = _hash_json(ctx.config)
    encoded = json.dumps(ctx.record, sort_keys=True, indent=2, allow_nan=False).encode() + b"\n"
    with tempfile.NamedTemporaryFile(dir=ctx.path, prefix=".run-", suffix=".tmp", delete=False) as tmp:
        tmp_path = Path(tmp.name)
        tmp.write(encoded)
        tmp.flush()
        os.fsync(tmp.fileno())
    tmp_path.replace(ctx.path / "run.json")


def doctor() -> dict[str, Any]:
    memory = 0
    source = "unavailable"
    if sys.platform == "darwin":
        try:
            result = subprocess.run(["system_profiler", "SPHardwareDataType"], check=True, capture_output=True, text=True, timeout=15)
            match = re.search(r"^\s*Memory:\s*([\d.]+)\s*(GB|TB)\s*$", result.stdout, re.MULTILINE)
            if match:
                memory = int(float(match.group(1)) * (1000**3 if match.group(2) == "GB" else 1000**4))
                source = "system_profiler SPHardwareDataType"
        except (OSError, subprocess.SubprocessError):
            pass
    if memory == 0:
        try:
            memory = os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
            source = "os.sysconf"
        except (AttributeError, ValueError, OSError):
            pass
    return {
        "architecture": platform.machine(),
        "macos": platform.mac_ver()[0] or None,
        "python": platform.python_version(),
        "physical_memory_bytes": memory,
        "physical_memory_source": source,
        "disk_free_bytes": shutil.disk_usage(Path.cwd()).free,
        "disk_source": "shutil.disk_usage(cwd)",
        "optional_backends": {name: __import__("importlib.util", fromlist=["find_spec"]).find_spec(name) is not None for name in ("torch", "transformers", "mlx_lm")},
    }


def _source_fingerprint() -> dict[str, str]:
    root = Path(__file__).resolve().parents[2]
    return {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest() for path in (root / "src/rbc/types.py", root / "src/rbc/logic.py", root / "src/rbc/data.py", root / "src/rbc/calibration.py")}


def run_g0(ctx: RunContext) -> dict[str, Any]:
    from .calibration import PredictionRow
    from .data import generate_source, policy_key
    from .logic import ast_to_dict, decide, make_bundle, save_bundle, validate_ast
    from .report import recompute_metrics, verify_run, write_report
    from .types import Accept, Ambiguous, BundleProvenance, SupportBelief, private_case_to_dict, public_case_to_dict, read_private_jsonl, read_public_jsonl

    old = ctx.record["stages"].get("g0")
    if old and old.get("status") == "passed":
        if ctx.record.get("g0_source_fingerprint") != _source_fingerprint():
            raise ValueError("G0 source changed; use a new run ID")
        audit = verify_run(ctx.path)
        if not audit.ok:
            raise ValueError(f"completed G0 artifacts failed audit: {audit.errors}")
        return old
    if not ctx.budget.can_start("core", 30.0):
        ctx.record["stages"]["g0"] = {"status": "budget_stopped", "reason": "insufficient remaining time for deterministic checks"}
        write_run(ctx)
        return ctx.record["stages"]["g0"]
    ctx.record["stages"]["g0"] = {"status": "running"}
    write_run(ctx)
    with ctx.budget.measure("core"):
        root = Path(__file__).resolve().parents[2]
        command = [sys.executable, "-m", "pytest", "-q", "tests/test_logic.py", "tests/test_public_boundary.py", "tests/test_decisions.py", "tests/test_bundle_io.py", "tests/test_generator.py", "tests/test_renderers.py", "tests/test_splits.py", "tests/test_policy_pools.py", "tests/test_calibration.py", "tests/test_statistics.py"]
        completed = subprocess.run(command, cwd=root, capture_output=True, text=True, timeout=120)
        if completed.returncode:
            ctx.record["stages"]["g0"] = {"status": "failed_correctness", "test_command": " ".join(command), "test_output": completed.stdout[-5000:] + completed.stderr[-2000:]}
        else:
            audited = [generate_source(ctx.config, f"g0-sample-{index:03d}", ctx.config["data"]["seed"] + index) for index in range(16)]
            if any(sample.private.oracle_posterior[sample.private.true_index] <= 0 for sample in audited):
                ctx.record["stages"]["g0"] = {"status": "failed_correctness", "reason": "generated true world missing from oracle posterior"}
            else:
                data_dir = ctx.path / "data"
                data_dir.mkdir(exist_ok=True)
                fixture_public = read_public_jsonl(root / "tests/fixtures/public_cases.jsonl")[0]
                fixture_private = read_private_jsonl(root / "tests/fixtures/private_cases.jsonl")[0]
                (data_dir / "public.jsonl").write_text(json.dumps(public_case_to_dict(fixture_public), sort_keys=True) + "\n")
                (data_dir / "private.jsonl").write_text(json.dumps(private_case_to_dict(fixture_private), sort_keys=True) + "\n")
                schema = fixture_public.schema
                support = np.array([False, True, True, False], dtype=np.bool_)
                model_supports = {
                    "oracle_joint": support,
                    "exact_xor_constraints": support,
                    "oracle_marginal_product": np.ones(4, dtype=np.bool_),
                }
                raw_policies = [
                    {"op": "count_eq", "ids": ["A_damaged", "B_damaged"], "k": 1},
                    {"op": "and", "args": [{"op": "var", "id": "A_damaged"}, {"op": "var", "id": "B_damaged"}]},
                    {"op": "var", "id": "A_damaged"},
                ]
                bundle_root = ctx.path / "bundles"
                bundle_root.mkdir(exist_ok=True)
                rows: list[PredictionRow] = []
                for model_id, retained in model_supports.items():
                    provenance = BundleProvenance(
                        _hash_json(fixture_public.text), _hash_json(dict(fixture_public.records)),
                        _hash_json({"schema_id": fixture_public.schema_id, "schema": [asdict(item) for item in schema]}),
                        "none", "none", _hash_json(model_id + ":v1"), "exact-support-v1",
                    )
                    probabilities = np.array([0.0, 0.5, 0.5, 0.0]) if model_id != "oracle_marginal_product" else np.full(4, 0.25)
                    bundle = make_bundle(schema, SupportBelief(retained, probabilities), retained, provenance)
                    relpath = f"bundles/{model_id}"
                    save_bundle(bundle, ctx.path / relpath)
                    for policy_index, raw in enumerate(raw_policies):
                        ast = validate_ast(raw, schema)
                        result = decide(bundle, ast)
                        status = "accept" if isinstance(result, Accept) else ("ambiguous" if isinstance(result, Ambiguous) else "out_of_scope")
                        rows.append(PredictionRow(
                            "g0_fixture", fixture_private.source_group, fixture_public.case_id, model_id,
                            policy_key(ast, schema).hex(), policy_index == 0, None, status,
                            result.action if isinstance(result, Accept) else None, None,
                            ast_to_dict(ast),
                            (result.world_a, result.world_b) if isinstance(result, Ambiguous) else None,
                            relpath,
                        ))
                (ctx.path / "predictions.jsonl").write_text("".join(json.dumps(asdict(row), sort_keys=True, allow_nan=False) + "\n" for row in rows))
                ctx.record["planned_primary"] = {"g0_fixture": {"group_id": fixture_private.source_group, "policy_key_hex": rows[0].policy_key_hex, "models": sorted(model_supports)}}
                ctx.record["g0_source_fingerprint"] = _source_fingerprint()
                ctx.record["stages"]["g0"] = {"status": "passed", "test_command": " ".join(command), "test_output": completed.stdout.strip(), "generated_truth_checks": len(audited), "scope": "numeric_oracle_and_kernel_only"}
                (ctx.path / "metrics.json").write_text(json.dumps(recompute_metrics(ctx.path), sort_keys=True, indent=2, allow_nan=False) + "\n")
                file_paths = [path for path in ctx.path.rglob("*") if path.is_file() and path.name not in {"run.json", "REPORT.md", ".run.lock"} and not path.name.startswith(".")]
                ctx.record["artifact_hashes"] = {str(path.relative_to(ctx.path)): hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(file_paths)}
    write_run(ctx)
    write_report(ctx.path)
    return ctx.record["stages"]["g0"]
