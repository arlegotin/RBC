import json
from pathlib import Path

import numpy as np
import pytest

from conftest import all_of, count_eq, neg, schema_n, var
from rbc.logic import decide, evaluate_scalar, make_bundle, validate_ast
from rbc.types import Accept, Ambiguous, BundleProvenance, DenseBelief, OutOfScope, SupportBelief, ValidationError, Variable


P = BundleProvenance("e", "r", "s", "t", "encoder", "head", "calibration")


def test_mandatory_xor_decisions():
    fixture = json.loads((Path(__file__).parent / "fixtures/mandatory_xor.json").read_text())
    schema = tuple(Variable(**entry) for entry in fixture["schema"])
    support = np.array(fixture["support"], dtype=bool)
    bundle = make_bundle(schema, SupportBelief(support, np.array(fixture["probabilities"])), support, P)
    aggregate = decide(bundle, count_eq(["A_damaged", "B_damaged"], 1))
    conjunction = decide(bundle, all_of(var("A_damaged"), var("B_damaged")))
    individual = decide(bundle, var("A_damaged"))
    assert isinstance(aggregate, Accept) and aggregate.action is True
    assert aggregate.certificate["retained_count"] == 2
    assert isinstance(conjunction, Accept) and conjunction.action is False
    assert isinstance(individual, Ambiguous) and (individual.world_a, individual.world_b) == (1, 2)
    assert evaluate_scalar(validate_ast(count_eq(["A_damaged", "B_damaged"], 1), schema), schema, (True, False))
    assert evaluate_scalar(validate_ast(count_eq(["A_damaged", "B_damaged"], 1), schema), schema, (False, True))


def _table_ast(n, truth):
    terms = []
    for index in range(1 << n):
        if truth & (1 << index):
            terms.append(all_of(*(var(f"v{j}") if index & (1 << j) else neg(var(f"v{j}")) for j in range(n))))
    if not terms:
        return all_of(var("v0"), neg(var("v0")))
    return {"op": "or", "args": terms}


def test_all_small_retained_sets():
    for n in (1, 2, 3):
        schema = schema_n(n)
        world_count = 1 << n
        formulas = [_table_ast(n, bits) for bits in range(1 << world_count)]
        belief = DenseBelief(np.zeros(world_count, dtype=float))
        for subset in range(1, 1 << world_count):
            kept = np.array([(subset >> i) & 1 for i in range(world_count)], dtype=bool)
            bundle = make_bundle(schema, belief, kept, P)
            for truth, formula in enumerate(formulas):
                values = {bool((truth >> i) & 1) for i in range(world_count) if kept[i]}
                result = decide(bundle, formula)
                assert isinstance(result, Accept) == (len(values) == 1)
                if isinstance(result, Accept):
                    assert result.action is next(iter(values))
                else:
                    assert isinstance(result, Ambiguous)
        empty = make_bundle(schema, belief, np.zeros(world_count, dtype=bool), P)
        assert isinstance(decide(empty, var("v0")), OutOfScope)


def test_expanding_retained_set_cannot_restore_disagreed_action():
    schema = schema_n(2)
    belief = DenseBelief(np.zeros(4))
    smaller = make_bundle(schema, belief, np.array([False, True, True, False]), P)
    larger = make_bundle(schema, belief, np.array([True, True, True, True]), P)
    assert isinstance(decide(smaller, var("v0")), Ambiguous)
    assert isinstance(decide(larger, var("v0")), Ambiguous)


def test_reordered_world_array_is_remapped_and_bad_inputs_fail_closed():
    schema = schema_n(2)
    original = np.array([False, True, True, False])
    permutation = np.array([2, 0, 3, 1])
    bundle = make_bundle(schema, SupportBelief(original[permutation]), original[permutation], P, world_indices=permutation)
    assert isinstance(decide(bundle, count_eq(["v0", "v1"], 1)), Accept)
    ambiguous = decide(bundle, var("v0"))
    assert isinstance(ambiguous, Ambiguous) and (ambiguous.world_a, ambiguous.world_b) == (1, 2)
    assert isinstance(decide(bundle, var("unknown")), OutOfScope)
    with pytest.raises(ValidationError):
        make_bundle(schema, DenseBelief(np.array([0.0, float("nan"), 0.0, 0.0])), original, P)
    with pytest.raises(ValidationError):
        make_bundle(schema, DenseBelief(np.array([0.0, float("inf"), 0.0, 0.0])), original, P)
    with pytest.raises(ValidationError):
        make_bundle(schema, SupportBelief(original), original, P, world_indices=np.array([0, 0, 2, 3]))
