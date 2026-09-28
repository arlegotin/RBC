"""Exact, allowlisted Boolean policies over complete small worlds."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from .types import Schema, ValidationError, Variable


def canonical_schema(entries: Sequence[Variable]) -> Schema:
    if not isinstance(entries, (tuple, list)) or not 1 <= len(entries) <= 12:
        raise ValidationError("schema must contain 1–12 variables")
    if any(not isinstance(item, Variable) for item in entries):
        raise ValidationError("schema entries must be Variables")
    ordered = tuple(sorted(entries, key=lambda item: item.id))
    if len({item.id for item in ordered}) != len(ordered):
        raise ValidationError("duplicate variable ID")
    return ordered


def enumerate_worlds(schema: Schema) -> np.ndarray:
    n = len(canonical_schema(schema))
    indexes = np.arange(1 << n, dtype=np.uint16)[:, None]
    shifts = np.arange(n, dtype=np.uint16)[None, :]
    return ((indexes >> shifts) & 1).astype(np.bool_)


@dataclass(frozen=True)
class AST:
    op: str
    id: str | None = None
    arg: AST | None = None
    args: tuple[AST, ...] = ()
    ids: tuple[str, ...] = ()
    k: int | None = None


def ast_to_dict(ast: AST) -> dict[str, Any]:
    if ast.op == "var":
        return {"op": "var", "id": ast.id}
    if ast.op == "not":
        return {"op": "not", "arg": ast_to_dict(ast.arg)}
    if ast.op in {"and", "or", "xor"}:
        return {"op": ast.op, "args": [ast_to_dict(item) for item in ast.args]}
    return {"op": ast.op, "ids": list(ast.ids), "k": ast.k}


def validate_ast(raw: Mapping[str, Any], schema: Schema) -> AST:
    ids = {item.id for item in canonical_schema(schema)}
    active: set[int] = set()
    nodes = 0

    def walk(node: Any, depth: int) -> AST:
        nonlocal nodes
        if not isinstance(node, Mapping):
            raise ValidationError("policy node must be a mapping")
        if id(node) in active:
            raise ValidationError("cyclic policy AST")
        nodes += 1
        if nodes > 64 or depth > 6:
            raise ValidationError("policy AST exceeds node or depth limit")
        op = node.get("op")
        expected = {
            "var": {"op", "id"},
            "not": {"op", "arg"},
            "and": {"op", "args"},
            "or": {"op", "args"},
            "xor": {"op", "args"},
            "count_eq": {"op", "ids", "k"},
            "count_ge": {"op", "ids", "k"},
        }.get(op)
        if expected is None or set(node) != expected:
            raise ValidationError("unknown operator or invalid policy node fields")
        active.add(id(node))
        try:
            if op == "var":
                if not isinstance(node["id"], str) or node["id"] not in ids:
                    raise ValidationError("unknown policy variable")
                return AST(op, id=node["id"])
            if op == "not":
                return AST(op, arg=walk(node["arg"], depth + 1))
            if op in {"and", "or", "xor"}:
                values = node["args"]
                if not isinstance(values, list) or not values or (op == "xor" and len(values) != 2):
                    raise ValidationError("invalid policy operands")
                return AST(op, args=tuple(walk(item, depth + 1) for item in values))
            names = node["ids"]
            threshold = node["k"]
            if not isinstance(names, list) or not names or any(not isinstance(name, str) or name not in ids for name in names) or len(set(names)) != len(names):
                raise ValidationError("invalid count variable list")
            if type(threshold) is not int or not 0 <= threshold <= len(names):
                raise ValidationError("invalid count threshold")
            return AST(op, ids=tuple(names), k=threshold)
        finally:
            active.remove(id(node))

    return walk(raw, 1)


def evaluate_scalar(ast: AST, schema: Schema, world: tuple[bool, ...]) -> bool:
    ordered = canonical_schema(schema)
    if len(world) != len(ordered) or any(not isinstance(bit, (bool, np.bool_)) for bit in world):
        raise ValidationError("world must contain one Boolean per variable")
    by_id = {variable.id: bool(world[j]) for j, variable in enumerate(ordered)}

    def ev(node: AST) -> bool:
        if node.op == "var":
            return by_id[node.id]
        if node.op == "not":
            return not ev(node.arg)
        if node.op == "and":
            return all(ev(item) for item in node.args)
        if node.op == "or":
            return any(ev(item) for item in node.args)
        if node.op == "xor":
            return ev(node.args[0]) != ev(node.args[1])
        total = sum(by_id[name] for name in node.ids)
        if node.op == "count_eq":
            return total == node.k
        if node.op == "count_ge":
            return total >= node.k
        raise ValidationError("invalid validated policy")

    return bool(ev(ast))


def evaluate_worlds(ast: AST, schema: Schema, worlds: np.ndarray) -> np.ndarray:
    ordered = canonical_schema(schema)
    if not isinstance(worlds, np.ndarray) or worlds.ndim != 2 or worlds.shape[1] != len(ordered) or worlds.dtype != np.bool_:
        raise ValidationError("world array must be Boolean and match schema")
    cols = {variable.id: j for j, variable in enumerate(ordered)}

    def ev(node: AST) -> np.ndarray:
        if node.op == "var":
            return worlds[:, cols[node.id]]
        if node.op == "not":
            return np.logical_not(ev(node.arg))
        if node.op == "and":
            return np.logical_and.reduce([ev(item) for item in node.args])
        if node.op == "or":
            return np.logical_or.reduce([ev(item) for item in node.args])
        if node.op == "xor":
            return np.logical_xor(ev(node.args[0]), ev(node.args[1]))
        counts = np.sum(worlds[:, [cols[name] for name in node.ids]], axis=1)
        if node.op == "count_eq":
            return counts == node.k
        if node.op == "count_ge":
            return counts >= node.k
        raise ValidationError("invalid validated policy")

    result = ev(ast)
    if result.shape != (len(worlds),) or result.dtype != np.bool_:
        raise ValidationError("policy did not produce a Boolean vector")
    return result
