"""Exact, allowlisted Boolean policies over complete small worlds."""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import asdict
import hashlib
import io
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .types import Accept, Ambiguous, BeliefBundle, BundleProvenance, DenseBelief, Decision, OutOfScope, Schema, SupportBelief, ValidationError, Variable


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


def make_bundle(
    schema: Schema,
    belief: DenseBelief | SupportBelief,
    retained: np.ndarray,
    provenance: BundleProvenance,
    *,
    world_indices: np.ndarray | None = None,
) -> BeliefBundle:
    ordered = canonical_schema(schema)
    size = 1 << len(ordered)
    indices = np.arange(size, dtype=np.int64) if world_indices is None else np.asarray(world_indices)
    if indices.shape != (size,) or indices.dtype.kind not in "iu" or set(indices.tolist()) != set(range(size)):
        raise ValidationError("bundle world indices must be a bijection")
    if not isinstance(retained, np.ndarray) or retained.shape != (size,) or retained.dtype != np.bool_:
        raise ValidationError("retained set must be a complete Boolean mask")
    if not isinstance(provenance, BundleProvenance) or any(not isinstance(value, str) or not value for value in asdict(provenance).values()):
        raise ValidationError("bundle provenance is incomplete")
    if isinstance(belief, DenseBelief):
        logits = np.asarray(belief.logits)
        if logits.shape != (size,) or logits.dtype.kind != "f" or not np.isfinite(logits).all():
            raise ValidationError("dense logits must be finite for every world")
        belief = DenseBelief(logits.astype(np.float64, copy=True))
    elif isinstance(belief, SupportBelief):
        support = np.asarray(belief.support)
        if support.shape != (size,) or support.dtype != np.bool_:
            raise ValidationError("support must be a complete Boolean mask")
        if np.any(retained & ~support):
            raise ValidationError("retained worlds cannot exceed structural support")
        probabilities = None
        if belief.probabilities is not None:
            probabilities = np.asarray(belief.probabilities, dtype=np.float64)
            if probabilities.shape != (size,) or not np.isfinite(probabilities).all() or np.any(probabilities < 0) or np.any(probabilities[~support] != 0) or not np.isclose(probabilities.sum(), 1.0, atol=1e-10, rtol=1e-10):
                raise ValidationError("invalid support probabilities")
            probabilities = probabilities.copy()
        belief = SupportBelief(support.copy(), probabilities)
    else:
        raise ValidationError("unknown belief kind")
    return BeliefBundle(ordered, indices.astype(np.int64, copy=True), belief, retained.copy(), provenance)


def decide(bundle: BeliefBundle, policy_ast: Mapping[str, Any] | AST) -> Decision:
    try:
        bundle = make_bundle(bundle.schema, bundle.belief, bundle.retained, bundle.provenance, world_indices=bundle.world_indices)
        raw = ast_to_dict(policy_ast) if isinstance(policy_ast, AST) else policy_ast
        ast = validate_ast(raw, bundle.schema)
        if not np.any(bundle.retained):
            return OutOfScope("empty_retained_set")
        worlds = enumerate_worlds(bundle.schema)[bundle.world_indices]
        values = evaluate_worlds(ast, bundle.schema, worlds)
    except (ValidationError, TypeError, AttributeError, KeyError, ValueError) as exc:
        return OutOfScope(f"invalid_bundle_or_policy: {exc}")
    kept = bundle.retained
    if np.all(values[kept] == values[kept][0]):
        policy_hash = hashlib.sha256(json.dumps(ast_to_dict(ast), sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        certificate = {
            "retained_count": int(np.sum(kept)),
            "policy_hash": policy_hash,
            "provenance": asdict(bundle.provenance),
            "statement": "the action agrees in every retained interpretation",
        }
        return Accept(bool(values[kept][0]), certificate)
    false_indices = bundle.world_indices[kept & ~values]
    true_indices = bundle.world_indices[kept & values]
    a, b = sorted((int(false_indices.min()), int(true_indices.min())))
    return Ambiguous(a, b)


def save_bundle(bundle: BeliefBundle, path: Path) -> None:
    checked = make_bundle(bundle.schema, bundle.belief, bundle.retained, bundle.provenance, world_indices=bundle.world_indices)
    path.mkdir(parents=True, exist_ok=True)
    if isinstance(checked.belief, DenseBelief):
        kind = "dense"
        scores = checked.belief.logits
        support = np.empty(0, dtype=np.bool_)
        probabilities = np.empty(0, dtype=np.float64)
    else:
        kind = "support"
        scores = np.empty(0, dtype=np.float64)
        support = checked.belief.support
        probabilities = checked.belief.probabilities if checked.belief.probabilities is not None else np.empty(0, dtype=np.float64)
    buffer = io.BytesIO()
    np.savez_compressed(buffer, world_indices=checked.world_indices, retained=checked.retained, logits=scores, support=support, probabilities=probabilities)
    array_bytes = buffer.getvalue()
    digest = hashlib.sha256(array_bytes).hexdigest()
    array_name = f"arrays-{digest}.npz"
    array_path = path / array_name
    if not array_path.exists():
        with tempfile.NamedTemporaryFile(dir=path, prefix=".arrays-", suffix=".tmp", delete=False) as temp:
            temp_path = Path(temp.name)
            temp.write(array_bytes)
            temp.flush()
            os.fsync(temp.fileno())
        temp_path.replace(array_path)
    metadata = {
        "format_version": 1,
        "schema": [asdict(item) for item in checked.schema],
        "provenance": asdict(checked.provenance),
        "belief_kind": kind,
        "arrays_file": array_name,
        "arrays_sha256": digest,
    }
    encoded = json.dumps(metadata, sort_keys=True, separators=(",", ":"), allow_nan=False).encode() + b"\n"
    with tempfile.NamedTemporaryFile(dir=path, prefix=".metadata-", suffix=".tmp", delete=False) as temp:
        temp_path = Path(temp.name)
        temp.write(encoded)
        temp.flush()
        os.fsync(temp.fileno())
    temp_path.replace(path / "metadata.json")


def load_bundle(path: Path, expected_provenance: BundleProvenance) -> BeliefBundle | OutOfScope:
    try:
        metadata = json.loads((path / "metadata.json").read_text())
        if metadata.get("format_version") != 1 or metadata.get("provenance") != asdict(expected_provenance):
            return OutOfScope("incompatible_bundle_provenance")
        array_name = metadata["arrays_file"]
        if not isinstance(array_name, str) or re.fullmatch(r"arrays-[0-9a-f]{64}\.npz", array_name) is None:
            return OutOfScope("invalid_bundle_array_reference")
        array_path = path / array_name
        if array_path.stat().st_size > 5_000_000:
            return OutOfScope("oversized_bundle_array")
        array_bytes = array_path.read_bytes()
        digest = hashlib.sha256(array_bytes).hexdigest()
        if digest != metadata["arrays_sha256"] or array_name != f"arrays-{digest}.npz":
            return OutOfScope("bundle_array_hash_mismatch")
        with np.load(io.BytesIO(array_bytes), allow_pickle=False) as arrays:
            if set(arrays.files) != {"world_indices", "retained", "logits", "support", "probabilities"}:
                return OutOfScope("invalid_bundle_arrays")
            world_indices = arrays["world_indices"].copy()
            retained = arrays["retained"].copy()
            if metadata["belief_kind"] == "dense":
                belief = DenseBelief(arrays["logits"].copy())
            elif metadata["belief_kind"] == "support":
                probabilities = arrays["probabilities"].copy()
                belief = SupportBelief(arrays["support"].copy(), None if probabilities.size == 0 else probabilities)
            else:
                return OutOfScope("invalid_belief_kind")
        schema = tuple(Variable(**entry) for entry in metadata["schema"])
        return make_bundle(schema, belief, retained, expected_provenance, world_indices=world_indices)
    except (OSError, ValueError, KeyError, TypeError, ValidationError, json.JSONDecodeError) as exc:
        return OutOfScope(f"invalid_bundle: {exc}")
