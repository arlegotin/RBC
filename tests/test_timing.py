from rbc.timing import measure_pipeline


def test_pipeline_timing_uses_one_compile_for_sixteen_future_policies():
    calls = {"compile": 0, "decide": 0}
    def compile_case(case):
        calls["compile"] += 1
        return case
    def decide(bundle, policy):
        calls["decide"] += 1
        return bool(bundle and policy)
    ticks = iter(range(100000))
    result = measure_pipeline(compile_case, decide, list(range(12)), [True] * 16, warmups=10, clock=lambda: next(ticks))
    assert result["samples"] == 12
    assert result["warmups"] == 10
    assert result["compile_calls_per_case"]["future_16"] == 1
    assert result["decision_calls_per_case"]["future_16"] == 16
    assert "compile_plus_future_16_ms" in result["latency_ms"]
    assert calls["compile"] > 12
    assert calls["decide"] > 12 * 16


def test_pipeline_requires_enough_timed_cases_and_warmups():
    try:
        measure_pipeline(lambda case: case, lambda bundle, policy: True, [1] * 9, [True] * 16, warmups=9)
    except ValueError as exc:
        assert "warmups" in str(exc)
    else:
        assert False
