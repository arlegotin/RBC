from rbc.experiment import evaluate_g1


def _metrics(oracle=0.70, marginal=0.60, parser=0.60, parser_errors=0, parser_acceptances=300, lower=0.02):
    return {"oracle_coverage": oracle, "marginal_coverage": marginal, "parser_coverage": parser, "parser_errors": parser_errors, "parser_acceptances": parser_acceptances, "paired_lower_vs_parser": lower, "n_cases": 500, "scope": "mixture"}


def test_g1_stops_when_parser_saturates_even_with_joint_headroom():
    result = evaluate_g1(_metrics(parser=0.68), {})
    assert result["status"] == "no_go"
    assert result["reason"] == "NO_HEADROOM_OVER_SIMPLE_BASELINE"


def test_g1_rejects_insufficient_joint_headroom():
    result = evaluate_g1(_metrics(oracle=0.699, marginal=0.60), {})
    assert result["status"] == "no_go"
    assert result["reason"] == "INSUFFICIENT_JOINT_MARGINAL_HEADROOM"


def test_g1_can_continue_only_with_credible_low_error_gap():
    assert evaluate_g1(_metrics(), {})["status"] == "passed"
    assert evaluate_g1(_metrics(lower=0.0), {})["status"] == "no_go"
    assert evaluate_g1(_metrics(parser_errors=4), {})["status"] == "no_go"
    assert evaluate_g1(_metrics(parser_acceptances=49), {})["status"] == "no_go"
    slice_result = evaluate_g1({**_metrics(), "scope": "relational_slice"}, {})
    assert slice_result["status"] == "passed" and slice_result["claim_scope"] == "relational_slice"
