from pathlib import Path

from rbc.report import recompute_metrics, write_report


def test_negative_report_names_three_distinct_hypotheses_and_unattempted_methods():
    path = Path(__file__).resolve().parents[1] / "runs/poc-20260928-01"
    metrics = recompute_metrics(path)
    assert "g1_slices" in metrics
    assert "component" in metrics["g1_slices"]
    assert metrics["g1_diagnostics"]["conventional_parser"]["state_set_coverage"] == 1
    report = write_report(path).read_text()
    assert "Representation:" in report
    assert "Learning:" in report
    assert "Auxiliary policy supervision:" in report
    assert "local LLM" in report
    assert "NO_HEADROOM_OVER_SIMPLE_BASELINE" in report
    assert "EXTERNAL_VALIDATION_NOT_PERFORMED" in report
    assert "Generated case slices" in report
