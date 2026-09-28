import random

import numpy as np
import pytest

from conftest import all_of, any_of, count_eq, count_ge, neg, schema_n, var, xor
from rbc.logic import canonical_schema, enumerate_worlds, evaluate_scalar, evaluate_worlds, validate_ast
from rbc.types import ValidationError, Variable


def test_bit_order_and_canonical_schema():
    schema = schema_n(2)
    assert enumerate_worlds(schema).tolist() == [[False, False], [True, False], [False, True], [True, True]]
    assert canonical_schema(schema[::-1]) == schema
    with pytest.raises(ValidationError):
        canonical_schema(())
    with pytest.raises(ValidationError):
        canonical_schema(schema_n(13))
    with pytest.raises(ValidationError):
        canonical_schema((schema[0], schema[0]))


@pytest.mark.parametrize("raw,expected", [
    (var("v0"), [False, True, False, True]),
    (neg(var("v0")), [True, False, True, False]),
    (all_of(var("v0"), var("v1")), [False, False, False, True]),
    (any_of(var("v0"), var("v1")), [False, True, True, True]),
    (xor(var("v0"), var("v1")), [False, True, True, False]),
    (count_eq(["v0", "v1"], 0), [True, False, False, False]),
    (count_ge(["v0", "v1"], 2), [False, False, False, True]),
    (count_ge(["v0"], 0), [True, True, True, True]),
])
def test_scalar_vector_agree_on_hand_checked_tables(raw, expected):
    schema = schema_n(2)
    ast = validate_ast(raw, schema)
    worlds = enumerate_worlds(schema)
    assert evaluate_worlds(ast, schema, worlds).tolist() == expected
    assert [evaluate_scalar(ast, schema, tuple(row)) for row in worlds] == expected


def test_ast_limits_and_invalid_values():
    schema = schema_n(2)
    cases = [
        {"op": "and", "args": []},
        {"op": "var", "id": "v0", "extra": 1},
        {"op": "var", "id": "missing"},
        {"op": "xor", "args": [var("v0")]},
        count_eq(["v0", "v0"], 1),
        count_eq(["v0"], True),
        count_ge(["v0"], 2),
        count_eq([], 0),
        {"op": "eval", "code": "True"},
    ]
    deep = var("v0")
    for _ in range(6):
        deep = neg(deep)
    cases.append(deep)
    cases.append(any_of(*([var("v0")] * 64)))
    cycle = {"op": "not"}
    cycle["arg"] = cycle
    cases.append(cycle)
    for raw in cases:
        with pytest.raises(ValidationError):
            validate_ast(raw, schema)
    with pytest.raises(ValidationError):
        evaluate_scalar(validate_ast(var("v0"), schema), schema, (1, False))
    with pytest.raises(ValidationError):
        evaluate_worlds(validate_ast(var("v0"), schema), schema, np.zeros((4, 2), dtype=int))


def test_demorgan_and_complements_to_twelve_variables():
    rng = random.Random(91)
    for n in (1, 3, 6, 12):
        schema = schema_n(n)
        worlds = enumerate_worlds(schema)
        for _ in range(10):
            a, b = rng.randrange(n), rng.randrange(n)
            left = validate_ast(neg(all_of(var(f"v{a}"), var(f"v{b}"))), schema)
            right = validate_ast(any_of(neg(var(f"v{a}")), neg(var(f"v{b}"))), schema)
            assert np.array_equal(evaluate_worlds(left, schema, worlds), evaluate_worlds(right, schema, worlds))
            assert np.array_equal(evaluate_worlds(left, schema, worlds), ~evaluate_worlds(validate_ast(all_of(var(f"v{a}"), var(f"v{b}")), schema), schema, worlds))
