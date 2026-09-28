import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

from rbc.report import recompute_metrics, verify_run, write_report


SOURCE = Path(__file__).resolve().parents[1] / "runs/poc-20260928-01"


def _copy_run(tmp_path):
    target = tmp_path / "run"
    shutil.copytree(SOURCE, target)
    return target


def _rehash(run, relative):
    record = json.loads((run / "run.json").read_text())
    record["artifact_hashes"][relative] = hashlib.sha256((run / relative).read_bytes()).hexdigest()
    (run / "run.json").write_text(json.dumps(record, sort_keys=True, indent=2) + "\n")


def test_saved_run_replays_and_report_is_deterministic(tmp_path):
    run = _copy_run(tmp_path)
    assert verify_run(run).ok
    first = write_report(run).read_bytes()
    second = write_report(run).read_bytes()
    assert first == second
    assert first.startswith(b"# Outcome:")
    record = json.loads((run / "run.json").read_text())
    assert record["source_snapshots"]
    assert record["g1_parser_source_snapshot"]
    assert record["replay_source_fingerprint"]
    assert "EXTERNAL_VALIDATION_NOT_PERFORMED" in first.decode()


def test_changed_primary_policy_fails_even_after_file_hash_is_updated(tmp_path):
    run = _copy_run(tmp_path)
    path = run / "predictions.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    row = next(row for row in rows if row["panel_id"] == "g1_dev")
    row["policy_ast"] = {"op": "var", "id": "v0"}
    path.write_text("".join(json.dumps(item, sort_keys=True) + "\n" for item in rows))
    _rehash(run, "predictions.jsonl")
    assert not verify_run(run).ok


def test_duplicate_primary_rows_fail_even_if_metrics_and_hashes_are_rewritten(tmp_path):
    run = _copy_run(tmp_path)
    path = run / "predictions.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows.append(next(row for row in rows if row["panel_id"] == "g1_dev"))
    path.write_text("".join(json.dumps(item, sort_keys=True) + "\n" for item in rows))
    _rehash(run, "predictions.jsonl")
    (run / "metrics.json").write_text(json.dumps(recompute_metrics(run), sort_keys=True, indent=2) + "\n")
    _rehash(run, "metrics.json")
    assert not verify_run(run).ok


def test_source_snapshot_is_required_for_replay_provenance(tmp_path):
    run = _copy_run(tmp_path)
    record = json.loads((run / "run.json").read_text())
    relpath = next(iter(record["source_snapshots"]))
    (run / relpath).unlink()
    assert not verify_run(run).ok


def test_parser_source_snapshot_is_required_for_g1_replay(tmp_path):
    run = _copy_run(tmp_path)
    record = json.loads((run / "run.json").read_text())
    relpath = record["g1_parser_source_snapshot"]["path"]
    (run / relpath).unlink()
    assert not verify_run(run).ok


def test_replay_code_hash_mismatch_fails_audit(tmp_path):
    run = _copy_run(tmp_path)
    path = run / "run.json"
    record = json.loads(path.read_text())
    record["replay_source_fingerprint"]["src/rbc/report.py"] = "0" * 64
    path.write_text(json.dumps(record, sort_keys=True, indent=2) + "\n")
    audit = verify_run(run)
    assert not audit.ok
    assert any("replay source" in error for error in audit.errors)


def test_private_label_drift_is_detected_even_with_updated_file_hash(tmp_path):
    run = _copy_run(tmp_path)
    path = run / "data/g1_private.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    row = next(row for row in rows if sum(value > 0 for value in row["oracle_posterior"]) > 1)
    old_index = sum(1 << bit for bit, value in enumerate(row["true_world"]) if value)
    replacement = next(index for index, value in enumerate(row["oracle_posterior"]) if value > 0 and index != old_index)
    row["true_world"] = [bool((replacement >> bit) & 1) for bit in range(6)]
    path.write_text("".join(json.dumps(item, sort_keys=True) + "\n" for item in rows))
    _rehash(run, "data/g1_private.jsonl")
    (run / "metrics.json").write_text(json.dumps(recompute_metrics(run), sort_keys=True, indent=2) + "\n")
    _rehash(run, "metrics.json")
    assert not verify_run(run).ok


def test_extra_primary_case_is_rejected_even_with_updated_file_hash(tmp_path):
    run = _copy_run(tmp_path)
    path = run / "predictions.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    extra = dict(next(row for row in rows if row["panel_id"] == "g0_fixture"))
    extra["panel_id"] = "unplanned_panel"
    rows.append(extra)
    path.write_text("".join(json.dumps(item, sort_keys=True) + "\n" for item in rows))
    _rehash(run, "predictions.jsonl")
    assert not verify_run(run).ok


def test_saved_replay_needs_no_optional_backend_or_network():
    code = r'''
import builtins, socket, sys
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split('.')[0] in {'torch', 'transformers', 'mlx_lm', 'huggingface_hub'}:
        raise RuntimeError('optional backend imported: ' + name)
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
socket.socket.connect = lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError('network used'))
from pathlib import Path
from rbc.report import verify_run, write_report
run = Path(sys.argv[1])
assert verify_run(run).ok
write_report(run)
'''
    result = subprocess.run([sys.executable, "-c", code, str(SOURCE)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_g1_go_decision_cannot_be_forged_by_editing_run_record(tmp_path):
    run = _copy_run(tmp_path)
    path = run / "run.json"
    record = json.loads(path.read_text())
    record["stages"]["g1"]["status"] = "passed"
    record["stages"]["g1"]["reason"] = "DEVELOPMENT_HEADROOM"
    record["claims"] = {"H1": "supported", "H2": "supported", "H3": "supported"}
    path.write_text(json.dumps(record, sort_keys=True, indent=2) + "\n")
    audit = verify_run(run)
    assert not audit.ok
    assert any("G1 decision" in error for error in audit.errors)


def test_report_headline_and_exit_reflect_failed_verification(tmp_path):
    run = _copy_run(tmp_path)
    (run / "metrics.json").write_text("{}\n")
    result = subprocess.run([sys.executable, "-m", "rbc", "report", "--run", str(run)], capture_output=True, text=True)
    assert result.returncode != 0
    assert "verification failed" in (run / "REPORT.md").read_text().splitlines()[0].lower()


def test_report_text_is_part_of_offline_artifact_audit(tmp_path):
    run = _copy_run(tmp_path)
    path = run / "REPORT.md"
    path.write_text(path.read_text().replace("G1 NO_GO", "G1 PASSED", 1))
    audit = verify_run(run)
    assert not audit.ok
    assert any("report" in error.lower() for error in audit.errors)
