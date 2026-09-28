import json

import pytest

from rbc.experiment import BudgetLedger, open_run, write_run


def test_category_charge_counts_against_total_and_survives_record_reload(tmp_path):
    ledger = BudgetLedger(total_seconds=10800, llm_seconds=3600, repair_seconds=1800)
    assert ledger.remaining("total") == 10800
    ledger.charge("llm", 30)
    assert ledger.remaining("llm") == 3570
    assert ledger.remaining("total") == 10770
    assert not ledger.can_start("llm", 3571)
    with pytest.raises(ValueError):
        ledger.charge("repair", -1)
    config = {"run_id": "test-1", "run_root": str(tmp_path), "budget": {"total_seconds": 10800, "llm_seconds": 3600, "repair_seconds": 1800}}
    ctx = open_run(config)
    ctx.budget.charge("llm", 30)
    write_run(ctx)
    ctx.close()
    resumed = open_run(config)
    assert resumed.budget.remaining("llm") == 3570
    assert resumed.budget.remaining("total") == 10770
    resumed.close()


def test_run_id_cannot_escape_root_or_overwrite_active_writer(tmp_path):
    for invalid in ("../escape", "a/b", "", "."):
        with pytest.raises(ValueError):
            open_run({"run_id": invalid, "run_root": str(tmp_path), "budget": {"total_seconds": 10800, "llm_seconds": 3600, "repair_seconds": 1800}})
    config = {"run_id": "test-1", "run_root": str(tmp_path), "budget": {"total_seconds": 10800, "llm_seconds": 3600, "repair_seconds": 1800}}
    ctx = open_run(config)
    try:
        with pytest.raises(RuntimeError, match="locked"):
            open_run(config)
    finally:
        ctx.close()


def test_incomplete_stage_is_not_resumed_as_complete(tmp_path):
    config = {"run_id": "test-1", "run_root": str(tmp_path), "budget": {"total_seconds": 10800, "llm_seconds": 3600, "repair_seconds": 1800}}
    ctx = open_run(config)
    ctx.record["stages"]["g0"] = {"status": "running"}
    write_run(ctx)
    ctx.close()
    resumed = open_run(config)
    assert resumed.record["stages"]["g0"]["status"] == "running"
    resumed.close()


def test_failed_work_is_charged_by_injected_clock():
    ticks = iter((100.0, 107.5))
    ledger = BudgetLedger(10800, 3600, 1800, clock=lambda: next(ticks))
    with pytest.raises(RuntimeError, match="failed"):
        with ledger.measure("core"):
            raise RuntimeError("failed")
    assert ledger.remaining("total") == 10792.5
