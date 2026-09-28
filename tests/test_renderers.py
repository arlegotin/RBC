import numpy as np

from rbc.data import ObservationPlan, Probe, observe, render_observation, returns_schema


def test_rendering_depends_on_observations_not_unobserved_truth():
    schema = returns_schema()
    plan = ObservationPlan(tuple(v.id for v in schema), (Probe("count", ("v0", "v2")),), "relational", "aggregate", "red")
    a = observe(plan, (True, False, False, False, False, False))
    b = observe(plan, (False, True, True, True, False, True))
    assert a.results == b.results == (1,)
    for family in ("plain", "alternative"):
        text_a = render_observation(schema, a, family, np.random.default_rng(73))
        text_b = render_observation(schema, b, family, np.random.default_rng(73))
        assert text_a == text_b


def test_renderer_families_express_negation_counts_and_equality():
    schema = returns_schema()
    ids = tuple(v.id for v in schema)
    plan = ObservationPlan(ids, (Probe("bit", ("v0",)), Probe("count", ("v2", "v4")), Probe("same", ("v1", "v3"))), "relational", "item", "blue")
    observation = observe(plan, (False, True, True, True, False, False))
    plain = render_observation(schema, observation, "plain", np.random.default_rng(2))
    alternate = render_observation(schema, observation, "alternative", np.random.default_rng(2))
    assert "not damaged" in plain
    assert "one" in plain
    assert "same" in plain
    assert "not damaged" in alternate or "undamaged" in alternate
    assert "one" in alternate
    assert "same" in alternate or "matches" in alternate


def test_no_evidence_is_explicitly_noncommittal():
    schema = returns_schema()
    plan = ObservationPlan(tuple(v.id for v in schema), (), "missing", "aggregate", "red")
    text = render_observation(schema, observe(plan, (True,) * 6), "plain", np.random.default_rng(2))
    assert text == "No item condition was established."
