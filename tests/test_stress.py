import numpy as np

from rbc.baselines import compile_parser
from rbc.data import returns_schema
from rbc.logic import decide, enumerate_worlds, evaluate_worlds, validate_ast
from rbc.stress import evaluate_stress, make_correlation_shift, make_stress_cases
from rbc.types import Accept, Ambiguous, OutOfScope


def test_shift_preserves_all_single_variable_marginals_but_reverses_xor():
    fixture = make_correlation_shift(returns_schema(), 53, {"a_cal": 64, "b_cal": 64, "b_test": 64})
    a = fixture["posterior_a"]
    b = fixture["posterior_b"]
    worlds = enumerate_worlds(returns_schema())
    np.testing.assert_allclose(a @ worlds, b @ worlds)
    xor = validate_ast({"op": "count_eq", "ids": ["v0", "v2"], "k": 1}, returns_schema())
    assert np.isclose(a @ evaluate_worlds(xor, returns_schema(), worlds), 1)
    assert np.isclose(b @ evaluate_worlds(xor, returns_schema(), worlds), 0)
    assert all(world[0] != world[2] for world in fixture["worlds_a_cal"])
    assert all(world[0] == world[2] for world in fixture["worlds_b_test"])
    assert fixture["public"].text == "No item condition was established."


def test_targeted_stress_cases_keep_failures_and_uncertainty_visible():
    cases = make_stress_cases({"data": {"stress_groups": 256}})
    assert len(cases) == 256
    counts = {}
    for case in cases:
        counts[case.category] = counts.get(case.category, 0) + 1
        bundle = compile_parser(case.public)
        decision = bundle if isinstance(bundle, OutOfScope) else decide(bundle, case.policy)
        assert type(decision).__name__.lower() == case.expected_status
        if case.expected_support is not None:
            np.testing.assert_array_equal(bundle.retained, case.expected_support)
    assert set(counts) == {"no_evidence", "exactly_one", "at_least_one", "fully_observed", "contradiction", "unknown_prose", "quoted_instruction", "long_input"}
    assert all(count == 32 for count in counts.values())


def test_frozen_numeric_shift_breaks_old_gate_and_recalibration_loses_coverage():
    result = evaluate_stress({"data": {"stress_groups": 256}, "statistics": {"alpha_grid": [0.02]}}, ("conventional_parser",))
    assert result["targeted"]["semantic_failures"] == 0
    assert result["targeted"]["cases"] == 256
    assert result["shift"]["old_calibration"]["accepted_errors"] == result["shift"]["test_groups"]
    assert result["shift"]["recalibrated"]["accepted"] == 0
    assert result["shift"]["old_calibration"]["threshold"] == 0
    assert result["shift"]["recalibrated"]["threshold"] == 10
