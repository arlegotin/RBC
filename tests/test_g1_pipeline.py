import json
from pathlib import Path

from rbc.experiment import load_config, open_run, run_g1, run_stress
from rbc.report import verify_run


def test_g1_run_records_replayable_development_screen(tmp_path):
    config = load_config(Path("configs/poc.yaml"))
    config["run_root"] = str(tmp_path)
    config["run_id"] = "g1-smoke"
    config["data"].update(train=0, dev=24, calibration=0, test=0, g1_dev=24)
    ctx = open_run(config)
    try:
        result = run_g1(ctx)
        stress = run_stress(ctx, ("conventional_parser",))
    finally:
        ctx.close()
    assert result["status"] in {"passed", "no_go"}
    assert result["n_cases"] == 24
    assert stress["status"] == "measured"
    rows = [json.loads(line) for line in (tmp_path / "g1-smoke" / "predictions.jsonl").read_text().splitlines()]
    assert len([r for r in rows if r["panel_id"] == "g1_dev" and r["model_id"] == "conventional_parser"]) == 24
    assert verify_run(tmp_path / "g1-smoke").ok
