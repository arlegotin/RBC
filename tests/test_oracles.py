import numpy as np

from conftest import count_eq, var
from rbc.baselines import compile_oracle, compile_parser, marginal_product, policy_probability
from rbc.data import generate_source, returns_schema
from rbc.logic import decide, validate_ast
from rbc.types import Accept, Ambiguous


def test_oracle_joint_and_parser_tie_on_same_generated_information():
    config = {"data": {"mixture": [0.30, 0.50, 0.20]}}
    source = generate_source(config, "case-1", 1)
    oracle = compile_oracle(source.public, source.private)
    parser = compile_parser(source.public)
    assert np.array_equal(oracle.retained, parser.retained)
    assert policy_probability(np.array(source.private.oracle_posterior), validate_ast(var("v0"), source.public.schema), source.public.schema) in {0.0, 0.5, 1.0}


def test_exactly_one_marginal_product_introduces_extra_worlds():
    schema = returns_schema()
    joint = np.zeros(64)
    joint[1] = joint[4] = 0.5
    product = marginal_product(joint, schema)
    assert np.isclose(product.sum(), 1)
    assert np.isclose(product[0], 0.25)
    assert np.isclose(product[5], 0.25)
    assert policy_probability(joint, validate_ast(count_eq(["v0", "v2"], 1), schema), schema) == 1.0
    assert policy_probability(product, validate_ast(count_eq(["v0", "v2"], 1), schema), schema) == 0.5
