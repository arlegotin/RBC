import json

import yaml

from rbc.__main__ import main
from rbc.experiment import load_config


def test_g0_cli_writes_replayable_artifacts_without_models(tmp_path):
    config = load_config(__import__("pathlib").Path("configs/poc.yaml"))
    config["run_id"] = "g0-fixture"
    config["run_root"] = str(tmp_path)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))
    assert main(["run", "--through", "g0", "--config", str(config_path)]) == 0
    run_dir = tmp_path / "g0-fixture"
    record = json.loads((run_dir / "run.json").read_text())
    assert record["stages"]["g0"]["status"] == "passed"
    assert record["claims"]["H2"] == "not_attempted"
    assert (run_dir / "predictions.jsonl").exists()
    assert (run_dir / "metrics.json").exists()
    assert main(["verify", "--run", str(run_dir)]) == 0
    assert main(["report", "--run", str(run_dir)]) == 0
    assert "G0" in (run_dir / "REPORT.md").read_text()


def test_g0_replay_detects_tampered_metrics(tmp_path):
    config = load_config(__import__("pathlib").Path("configs/poc.yaml"))
    config["run_id"] = "g0-tamper"
    config["run_root"] = str(tmp_path)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))
    assert main(["run", "--through", "g0", "--config", str(config_path)]) == 0
    run_dir = tmp_path / "g0-tamper"
    metrics = json.loads((run_dir / "metrics.json").read_text())
    metrics["g0"]["oracle_joint"]["accepted"] = 999
    (run_dir / "metrics.json").write_text(json.dumps(metrics))
    assert main(["verify", "--run", str(run_dir)]) != 0
