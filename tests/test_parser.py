import numpy as np
import pytest

from conftest import all_of, count_eq, var
from rbc.baselines import compile_parser, parse_constraints
from rbc.data import generate_source, returns_schema
from rbc.logic import decide, evaluate_worlds, enumerate_worlds, validate_ast
from rbc.types import Accept, Ambiguous, OutOfScope, PublicCase


def _case(text):
    return PublicCase("plain-1", "returns-six-v1", returns_schema(), {}, text)


@pytest.mark.parametrize("text,policy,want", [
    ("The red item is damaged.", var("v0"), True),
    ("The red item is not damaged.", var("v0"), False),
    ("The red item is undamaged.", var("v0"), False),
    ("The blue item has been used.", var("v3"), False),
    ("The red item is damaged and the blue item is unused.", all_of(var("v0"), var("v3")), True),
    ("At least one of the red and blue items is damaged.", {"op": "count_ge", "ids": ["v0", "v2"], "k": 1}, True),
    ("Exactly one of the red and blue items is damaged.", count_eq(["v0", "v2"], 1), True),
    ("The number of damaged items among the red and blue items is one.", count_eq(["v0", "v2"], 1), True),
    ("The damage status of the red item matches the damage status of the blue item.", {"op": "not", "arg": {"op": "xor", "args": [var("v0"), var("v2")]}}, True),
    ("The unused status of the blue item differs from the damage status of the red item.", {"op": "xor", "args": [var("v3"), var("v0")]}, True),
])
def test_parser_understands_ordinary_relational_language(text, policy, want):
    bundle = compile_parser(_case(text))
    assert not isinstance(bundle, OutOfScope), bundle
    result = decide(bundle, policy)
    assert isinstance(result, Accept) and result.action is want


def test_parser_keeps_unknowns_open_and_detects_contradictions():
    blank = compile_parser(_case("No item condition was established."))
    assert blank.retained.sum() == 64
    assert isinstance(decide(blank, var("v0")), Ambiguous)
    assert isinstance(compile_parser(_case("The red item is damaged. The red item is not damaged.")), OutOfScope)
    assert isinstance(compile_parser(_case("The red item seems damaged.")), OutOfScope)


def test_parser_recovers_executable_generator_observations_without_private_ast():
    config = {"data": {"mixture": [0.30, 0.50, 0.20]}}
    for seed in range(160):
        source = generate_source(config, f"case-{seed}", seed)
        parser = compile_parser(source.public)
        assert not isinstance(parser, OutOfScope), (seed, source.public.text, parser)
        expected = np.array(source.private.oracle_posterior) > 0
        assert np.array_equal(parser.retained, expected), (seed, source.public.text)


def test_parser_does_not_read_evaluator_private_file(monkeypatch):
    import pathlib

    original = pathlib.Path.read_text

    def deny_private(path, *args, **kwargs):
        if "private" in str(path):
            raise AssertionError("private file read on parser path")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(pathlib.Path, "read_text", deny_private)
    result = compile_parser(_case("The red item is damaged."))
    assert isinstance(decide(result, var("v0")), Accept)
