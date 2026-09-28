"""Typed public inputs and evaluator-only private labels."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


class ValidationError(ValueError):
    """An input violates the declared RBC contract."""


@dataclass(frozen=True)
class Variable:
    id: str
    entity: str
    meaning: str

    def __post_init__(self) -> None:
        for name in ("id", "entity", "meaning"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValidationError(f"variable {name} must be nonempty text")


Schema = tuple[Variable, ...]


@dataclass(frozen=True)
class PublicCase:
    case_id: str
    schema_id: str
    schema: Schema
    records: Mapping[str, Any]
    text: str

    def __post_init__(self) -> None:
        if not isinstance(self.case_id, str) or not self.case_id:
            raise ValidationError("case_id must be nonempty text")
        if not isinstance(self.schema_id, str) or not self.schema_id:
            raise ValidationError("schema_id must be nonempty text")
        if not isinstance(self.text, str):
            raise ValidationError("text must be a string")
        if not isinstance(self.records, Mapping) or self.records:
            raise ValidationError("structured records are not supported in the first workload")
        from .logic import canonical_schema

        canonical_schema(self.schema)


@dataclass(frozen=True)
class PrivateCase:
    case_id: str
    true_world: tuple[bool, ...]
    observation: Mapping[str, Any]
    oracle_posterior: tuple[float, ...]
    source_group: str
    split: str
    renderer: str
    generation_seed: int
    relation_tags: tuple[str, ...]

    def __post_init__(self) -> None:
        if not 1 <= len(self.true_world) <= 12 or any(type(bit) is not bool for bit in self.true_world):
            raise ValidationError("true_world must contain 1–12 actual booleans")
        if len(self.oracle_posterior) != (1 << len(self.true_world)):
            raise ValidationError("oracle posterior has incompatible universe")
        if any(not math.isfinite(float(p)) or p < 0 for p in self.oracle_posterior):
            raise ValidationError("oracle posterior must be finite and nonnegative")
        if not math.isclose(sum(self.oracle_posterior), 1.0, rel_tol=1e-9, abs_tol=1e-9):
            raise ValidationError("oracle posterior must sum to one")
        if type(self.generation_seed) is not int:
            raise ValidationError("generation seed must be an integer")

    @property
    def true_index(self) -> int:
        return sum((1 << j) for j, bit in enumerate(self.true_world) if bit)


def public_case_from_dict(raw: Mapping[str, Any]) -> PublicCase:
    expected = {"case_id", "schema_id", "schema", "records", "text"}
    if not isinstance(raw, Mapping) or set(raw) != expected:
        raise ValidationError("public case fields do not match the declared contract")
    entries = raw["schema"]
    if not isinstance(entries, list):
        raise ValidationError("schema must be a list")
    variables: list[Variable] = []
    for item in entries:
        if not isinstance(item, Mapping) or set(item) != {"id", "entity", "meaning"}:
            raise ValidationError("invalid schema entry")
        variables.append(Variable(item["id"], item["entity"], item["meaning"]))
    return PublicCase(raw["case_id"], raw["schema_id"], tuple(variables), raw["records"], raw["text"])


def read_public_jsonl(path: Path) -> list[PublicCase]:
    cases: list[PublicCase] = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            raise ValidationError(f"blank public case at line {line_number}")
        try:
            cases.append(public_case_from_dict(json.loads(line)))
        except (json.JSONDecodeError, TypeError, KeyError) as exc:
            raise ValidationError(f"invalid public case at line {line_number}") from exc
    return cases


def public_model_input(case: PublicCase) -> str:
    from .logic import canonical_schema

    schema = canonical_schema(case.schema)
    payload = {
        "schema_id": case.schema_id,
        "schema": [{"id": v.id, "entity": v.entity, "meaning": v.meaning} for v in schema],
        "records": dict(case.records),
        "text": case.text,
    }
    return json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
