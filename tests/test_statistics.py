import math

import numpy as np
import pytest
from scipy.stats import beta

from rbc.calibration import GateChoice, EvaluatedRow, PredictionRow, min_zero_error_acceptances, paired_coverage_interval, passing_error_count, risk_power, risk_upper, select_gate
from rbc.types import ValidationError


def test_exact_accepted_risk_and_power_boundaries():
    delta = 0.05 / 3
    assert min_zero_error_acceptances(0.02, delta) == 203
    assert min_zero_error_acceptances(0.005, delta) == 817
    assert risk_upper(0, 202, delta) > 0.02
    assert risk_upper(0, 203, delta) <= 0.02
    assert risk_upper(0, 0, delta) == 1.0
    assert risk_upper(3, 3, delta) == 1.0
    assert risk_upper(1, 300, delta) == pytest.approx(beta.ppf(1 - delta, 2, 299))
    assert passing_error_count(300, 0.02, delta) == 1
    assert 0.55 < risk_power(300, 0.005, 0.02, delta) < 0.56
    assert 0.19 < risk_power(300, 0.01, 0.02, delta) < 0.20
    with pytest.raises(ValidationError):
        risk_upper(2, 1, delta)


def _row(i, score=0.9, correct=True, status="scored"):
    prediction = PredictionRow("dev", f"group-{i}", f"case-{i}", "direct", "policy", True, 17, status, True, score)
    return EvaluatedRow(prediction, 0, correct, None)


def test_gate_needs_fifty_accepted_cases_and_one_percent_error():
    config = {"statistics": {"confidence_grid": [0.7, 0.8, 0.9, 0.95, 1.0], "alpha_grid": [0.005, 0.01, 0.02, 0.05]}}
    fit = [_row(i) for i in range(60)]
    chosen = select_gate(fit, [_row(i + 100) for i in range(50)], config)
    assert chosen.feasible and chosen.kind == "confidence" and chosen.value == 0.9
    assert not select_gate(fit, [_row(i + 100) for i in range(49)], config).feasible
    error_check = [_row(i + 100, correct=(i != 0)) for i in range(50)]
    assert not select_gate(fit, error_check, config).feasible
    rounded = [_row(i + 100, score=1.0) for i in range(50)]
    strict = {"statistics": {"confidence_grid": [1.0], "alpha_grid": []}}
    assert not select_gate(fit, rounded, strict).feasible


def test_paired_bootstrap_uses_matching_source_groups_once():
    a = tuple(_row(i).prediction for i in range(10))
    assert paired_coverage_interval(a, a, 2000, 1729) == (0.0, 0.0, 0.0)
    b = list(a)
    b[0] = PredictionRow("other", "group-0", "case-0", "direct", "policy", True, 17, "accept", True, 0.9)
    with pytest.raises(ValidationError):
        paired_coverage_interval(a, b, 2000, 1729)
