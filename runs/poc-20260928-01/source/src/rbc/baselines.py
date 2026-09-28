"""Conventional public-text parsing and evaluator-only oracle controls."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict
from typing import Any

import numpy as np

from .logic import AST, canonical_schema, enumerate_worlds, evaluate_worlds, make_bundle, validate_ast
from .types import BeliefBundle, BundleProvenance, OutOfScope, PrivateCase, PublicCase, Schema, SupportBelief, ValidationError


_NUMBER = {word: index for index, word in enumerate(("zero", "one", "two", "three"))}


def _schema_slots(case: PublicCase) -> tuple[dict[tuple[str, str], str], dict[str, str]]:
    slots: dict[tuple[str, str], str] = {}
    entities: dict[str, str] = {}
    for variable in canonical_schema(case.schema):
        match = re.search(r"\bis (damaged|unused)$", variable.meaning, flags=re.IGNORECASE)
        if match is None or not variable.entity.endswith(" item"):
            raise ValidationError("parser cannot bind the declared schema")
        attribute = match.group(1).lower()
        name = variable.entity[:-5].lower()
        slots[(name, attribute)] = variable.id
        entities[name] = variable.entity
    if len(slots) != len(case.schema):
        raise ValidationError("ambiguous schema bindings")
    return slots, entities


def _group_ids(phrase: str, attribute: str, slots: dict[tuple[str, str], str]) -> list[str]:
    if not phrase.startswith("the ") or not phrase.endswith(" items"):
        raise ValidationError("unsupported count entity list")
    names = phrase[4:-6].replace(", and ", ", ").replace(" and ", ", ").split(", ")
    if len(names) not in (2, 3) or len(set(names)) != len(names):
        raise ValidationError("count requires distinct named items")
    try:
        return [slots[(name, attribute)] for name in names]
    except KeyError as exc:
        raise ValidationError("count references an unknown entity or attribute") from exc


def _parse_clause(clause: str, slots: dict[tuple[str, str], str]) -> dict[str, Any]:
    # Individual facts, including a deliberately small set of ordinary synonyms.
    match = re.fullmatch(r"the ([a-z]+) item is (not )?(damaged|unused)", clause, flags=re.IGNORECASE)
    if match:
        key = (match.group(1).lower(), match.group(3).lower())
        if key not in slots:
            raise ValidationError("unknown fact binding")
        fact = {"op": "var", "id": slots[key]}
        return {"op": "not", "arg": fact} if match.group(2) else fact
    match = re.fullmatch(r"the ([a-z]+) item is undamaged", clause, flags=re.IGNORECASE)
    if match:
        key = (match.group(1).lower(), "damaged")
        if key not in slots:
            raise ValidationError("unknown damage binding")
        return {"op": "not", "arg": {"op": "var", "id": slots[key]}}
    match = re.fullmatch(r"the ([a-z]+) item has been used", clause, flags=re.IGNORECASE)
    if match:
        key = (match.group(1).lower(), "unused")
        if key not in slots:
            raise ValidationError("unknown unused binding")
        return {"op": "not", "arg": {"op": "var", "id": slots[key]}}

    # Counts over named items; these forms cover all frozen renderers and a basic at-least phrase.
    match = re.fullmatch(r"(exactly|at least) (zero|one|two|three) of (the .+ items) (?:is|are) (damaged|unused)", clause, flags=re.IGNORECASE)
    if match:
        ids = _group_ids(match.group(3).lower(), match.group(4).lower(), slots)
        return {"op": "count_eq" if match.group(1).lower() == "exactly" else "count_ge", "ids": ids, "k": _NUMBER[match.group(2).lower()]}
    match = re.fullmatch(r"the number of (damaged|unused) items among (the .+ items) is (zero|one|two|three)", clause, flags=re.IGNORECASE)
    if match:
        ids = _group_ids(match.group(2).lower(), match.group(1).lower(), slots)
        return {"op": "count_eq", "ids": ids, "k": _NUMBER[match.group(3).lower()]}

    # Equality or difference between two named attributes.
    match = re.fullmatch(r"the (damage|unused) status of the ([a-z]+) item (is the same as|differs from|matches|does not match) the (damage|unused) status of the ([a-z]+) item", clause, flags=re.IGNORECASE)
    if match:
        first_attribute = "damaged" if match.group(1).lower() == "damage" else "unused"
        second_attribute = "damaged" if match.group(4).lower() == "damage" else "unused"
        left = slots.get((match.group(2).lower(), first_attribute))
        right = slots.get((match.group(5).lower(), second_attribute))
        if left is None or right is None:
            raise ValidationError("unknown relation binding")
        relation = {"op": "xor", "args": [{"op": "var", "id": left}, {"op": "var", "id": right}]}
        return {"op": "not", "arg": relation} if match.group(3).lower() in {"is the same as", "matches"} else relation
    raise ValidationError(f"unsupported meaningful clause: {clause}")


def parse_constraints(public_case: PublicCase) -> tuple[AST, ...] | OutOfScope:
    try:
        slots, _ = _schema_slots(public_case)
        text = public_case.text.strip()
        if not text:
            return ()
        if not text.endswith("."):
            raise ValidationError("case sentences must end with a period")
        clauses = [piece.strip() for piece in re.split(r"\.\s+", text[:-1])]
        if any(not piece for piece in clauses):
            raise ValidationError("empty case sentence")
        if len(clauses) == 1 and clauses[0].lower() == "no item condition was established":
            return ()
        if any(clause.lower() == "no item condition was established" for clause in clauses):
            raise ValidationError("no-information claim conflicts with other evidence")
        raw: list[dict[str, Any]] = []
        for clause in clauses:
            # Split only a conjunction of complete atomic facts; count lists also contain 'and'.
            atoms = re.split(r"\s+and\s+(?=the [a-z]+ item is )", clause, flags=re.IGNORECASE)
            raw.extend(_parse_clause(atom, slots) for atom in atoms)
        if raw:
            validate_ast({"op": "and", "args": raw}, public_case.schema)
        return tuple(validate_ast(item, public_case.schema) for item in raw)
    except (ValidationError, KeyError, TypeError) as exc:
        return OutOfScope(str(exc))


def _hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def _provenance(case: PublicCase, method: str) -> BundleProvenance:
    return BundleProvenance(
        _hash(case.text), _hash(dict(case.records)),
        _hash({"schema_id": case.schema_id, "schema": [asdict(item) for item in canonical_schema(case.schema)]}),
        "none", "none", _hash(method + ":v1"), "exact-support-v1",
    )


def compile_parser(public_case: PublicCase) -> BeliefBundle | OutOfScope:
    constraints = parse_constraints(public_case)
    if isinstance(constraints, OutOfScope):
        return constraints
    worlds = enumerate_worlds(public_case.schema)
    support = np.ones(len(worlds), dtype=np.bool_)
    for ast in constraints:
        support &= evaluate_worlds(ast, public_case.schema, worlds)
    if not np.any(support):
        return OutOfScope("unsatisfiable_constraints")
    probabilities = support.astype(np.float64) / np.sum(support)
    return make_bundle(public_case.schema, SupportBelief(support, probabilities), support, _provenance(public_case, "conventional_parser"))


def compile_oracle(public_case: PublicCase, private_case: PrivateCase) -> BeliefBundle:
    if private_case.case_id != public_case.case_id:
        raise ValidationError("public/private case IDs do not match")
    probabilities = np.asarray(private_case.oracle_posterior, dtype=np.float64)
    support = probabilities > 0
    return make_bundle(public_case.schema, SupportBelief(support, probabilities), support, _provenance(public_case, "oracle_joint"))


def marginal_product(probabilities: np.ndarray, schema: Schema) -> np.ndarray:
    ordered = canonical_schema(schema)
    worlds = enumerate_worlds(ordered)
    p = np.asarray(probabilities, dtype=np.float64)
    if p.shape != (len(worlds),) or not np.isfinite(p).all() or np.any(p < 0) or not np.isclose(p.sum(), 1.0):
        raise ValidationError("invalid joint probabilities")
    marginals = p @ worlds.astype(np.float64)
    product = np.prod(np.where(worlds, marginals[None, :], 1 - marginals[None, :]), axis=1)
    return product / product.sum()


def policy_probability(probabilities: np.ndarray, ast: AST, schema: Schema) -> float:
    worlds = enumerate_worlds(schema)
    p = np.asarray(probabilities, dtype=np.float64)
    if p.shape != (len(worlds),) or not np.isfinite(p).all() or np.any(p < 0) or not np.isclose(p.sum(), 1.0):
        raise ValidationError("invalid policy probability distribution")
    return float(p @ evaluate_worlds(ast, schema, worlds).astype(np.float64))
