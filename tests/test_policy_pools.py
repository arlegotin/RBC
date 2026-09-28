import numpy as np

from conftest import all_of, neg, var, xor
from rbc.data import build_policy_pools, policy_key, returns_schema, select_primary_policy
from rbc.logic import ast_to_dict, validate_ast
from rbc.types import PublicCase, Variable


def test_truth_table_key_recognizes_equivalent_formulas_and_remapped_ids():
    schema = returns_schema()
    left = validate_ast(xor(var("v0"), var("v2")), schema)
    right = validate_ast({"op": "or", "args": [all_of(var("v0"), neg(var("v2"))), all_of(neg(var("v0")), var("v2"))]}, schema)
    assert policy_key(left, schema) == policy_key(right, schema)
    remapped_schema = (Variable("A", "red item", "red item is damaged"), Variable("B", "blue item", "blue item is damaged"))
    assert policy_key(validate_ast(xor(var("A"), var("B")), remapped_schema), remapped_schema) == policy_key(validate_ast(xor(var("v0"), var("v1")), returns_schema()[:2]), returns_schema()[:2])


def test_policy_pools_are_truth_disjoint_and_nontrivial():
    schema = returns_schema()
    pools = build_policy_pools(schema, 31415)
    keysets = []
    for pool in (pools.train, pools.dev, pools.confirmation):
        assert set(pool) == {"aggregate", "item", "combination"}
        keys = set()
        for workflow, formulas in pool.items():
            assert formulas, workflow
            for ast in formulas:
                key = policy_key(ast, schema)
                assert 8 <= int.from_bytes(key, "little").bit_count() <= 56
                keys.add(key)
        keysets.append(keys)
    assert keysets[0].isdisjoint(keysets[1])
    assert keysets[0].isdisjoint(keysets[2])
    assert keysets[1].isdisjoint(keysets[2])
    assert all(any(child in str(ast_to_dict(ast)) for child in ("and", "or", "not", "count")) for ast in pools.confirmation["item"])


def test_primary_policy_choice_uses_public_workflow_only():
    schema = returns_schema()
    pools = build_policy_pools(schema, 31415)
    case = PublicCase("case-1", "returns-six-v1", schema, {}, "No item condition was established.")
    context = {"workflow": "item", "queried_item": "red"}
    choice1 = select_primary_policy(case, context, pools.train["item"], np.random.default_rng(8))
    choice2 = select_primary_policy(case, context, pools.train["item"], np.random.default_rng(8))
    assert ast_to_dict(choice1) == ast_to_dict(choice2)
    assert "v0" in str(ast_to_dict(choice1))
