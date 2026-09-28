import numpy as np
import pytest

from rbc.data import ObservationPlan, Probe, generate_source, observe, oracle_posterior, returns_schema
from rbc.logic import enumerate_worlds


def test_observation_posterior_is_same_program_on_every_world():
    schema = returns_schema()
    worlds = enumerate_worlds(schema)
    ids = tuple(variable.id for variable in schema)
    probes = [Probe("bit", ("v0",)), Probe("count", ("v0", "v2", "v4")), Probe("same", ("v1", "v3"))]
    for probe in probes:
        plan = ObservationPlan(ids, (probe,), "relational", "aggregate", "red")
        for index, world in enumerate(worlds):
            observation = observe(plan, tuple(bool(v) for v in world))
            posterior = oracle_posterior(schema, observation, np.full(64, 1 / 64))
            assert np.isclose(posterior.sum(), 1.0)
            assert posterior[index] > 0
            for other_index, other_world in enumerate(worlds):
                same_result = observe(plan, tuple(bool(v) for v in other_world)).results == observation.results
                assert (posterior[other_index] > 0) == same_result


def test_missing_full_and_relational_evidence_have_exact_posteriors():
    schema = returns_schema()
    ids = tuple(v.id for v in schema)
    prior = np.arange(1, 65, dtype=float)
    prior /= prior.sum()
    blank = ObservationPlan(ids, (), "missing", "aggregate", "red")
    assert np.array_equal(oracle_posterior(schema, observe(blank, (False,) * 6), prior), prior)
    full = ObservationPlan(ids, tuple(Probe("bit", (id,)) for id in ids), "explicit", "item", "red")
    posterior = oracle_posterior(schema, observe(full, (True, False, True, False, True, False)), prior)
    assert np.flatnonzero(posterior).tolist() == [21]
    assert posterior[21] == 1.0
    relation = ObservationPlan(ids, (Probe("count", ("v0", "v2")),), "relational", "aggregate", "red")
    observed = observe(relation, (True, False, False, False, False, False))
    posterior = oracle_posterior(schema, observed, np.full(64, 1 / 64))
    assert np.flatnonzero(posterior).size == 32
    assert np.all(posterior[[1, 4]] == 1 / 32)
    unequal_prior = oracle_posterior(schema, observed, prior)
    assert unequal_prior[1] != unequal_prior[4]


def test_generated_case_labels_remain_in_oracle_support():
    config = {"data": {"mixture": [0.30, 0.50, 0.20]}}
    for seed in range(100):
        case = generate_source(config, f"case-{seed:06d}", seed)
        assert len(case.public.schema) == 6
        assert case.private.oracle_posterior[case.private.true_index] > 0
        assert case.public.records == {}
        assert "true_world" not in case.public.text
        assert "observation" not in case.public.text
