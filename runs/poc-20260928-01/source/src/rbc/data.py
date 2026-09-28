"""Executable observations and controlled, agent-authored case language."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import re
from pathlib import Path
from typing import Any
from typing import Mapping, Sequence

import numpy as np

from .logic import AST, ast_to_dict, canonical_schema, enumerate_worlds, evaluate_worlds, validate_ast
from .types import PrivateCase, PublicCase, Schema, ValidationError, Variable, public_model_input, read_private_jsonl, read_public_jsonl


ITEMS = ("red", "blue", "green")
ATTRIBUTES = ("damaged", "unused")
NUMBER_WORDS = ("zero", "one", "two", "three")


@dataclass(frozen=True)
class Probe:
    kind: str
    ids: tuple[str, ...]


@dataclass(frozen=True)
class ObservationPlan:
    schema_ids: tuple[str, ...]
    probes: tuple[Probe, ...]
    component: str
    workflow: str
    queried_item: str


@dataclass(frozen=True)
class Observation:
    plan: ObservationPlan
    results: tuple[bool | int, ...]


@dataclass(frozen=True)
class GeneratedSource:
    public: PublicCase
    private: PrivateCase
    workflow: str
    queried_item: str


def returns_schema() -> Schema:
    return tuple(
        Variable(f"v{2 * item_index + attribute_index}", f"{item} item", f"the {item} item is {attribute}")
        for item_index, item in enumerate(ITEMS)
        for attribute_index, attribute in enumerate(ATTRIBUTES)
    )


def sample_observation_plan(config: dict[str, Any], rng: np.random.Generator) -> ObservationPlan:
    data = config.get("data", {})
    mixture = data.get("mixture", [0.30, 0.50, 0.20])
    if len(mixture) != 3 or not np.isclose(sum(mixture), 1) or any(p < 0 for p in mixture):
        raise ValidationError("invalid observation mixture")
    component = str(rng.choice(("explicit", "relational", "missing"), p=mixture))
    ids = tuple(v.id for v in returns_schema())
    workflow = str(rng.choice(("aggregate", "item", "combination")))
    queried_item = str(rng.choice(ITEMS))
    probes: list[Probe] = []
    if component == "explicit":
        count = int(rng.choice((4, 5, 6)))
        probes.extend(Probe("bit", (str(id),)) for id in rng.choice(ids, size=count, replace=False))
    elif component == "relational":
        for _ in range(int(rng.choice((1, 2)))):
            if bool(rng.integers(0, 2)):
                attribute_index = int(rng.integers(0, 2))
                group_size = int(rng.choice((2, 3)))
                group = rng.choice((0, 1, 2), size=group_size, replace=False)
                probes.append(Probe("count", tuple(f"v{2 * int(i) + attribute_index}" for i in group)))
            else:
                pair = rng.choice(ids, size=2, replace=False)
                probes.append(Probe("same", tuple(str(id) for id in pair)))
        if bool(rng.integers(0, 2)):
            probes.append(Probe("bit", (str(rng.choice(ids)),)))
    else:
        if bool(rng.integers(0, 2)):
            probes.append(Probe("bit", (str(rng.choice(ids)),)))
    return ObservationPlan(ids, tuple(probes), component, workflow, queried_item)


def observe(plan: ObservationPlan, world: tuple[bool, ...]) -> Observation:
    if len(world) != len(plan.schema_ids) or any(type(bit) is not bool for bit in world):
        raise ValidationError("latent world must contain actual booleans")
    if len(set(plan.schema_ids)) != len(plan.schema_ids):
        raise ValidationError("duplicate observation variable")
    by_id = dict(zip(plan.schema_ids, world))
    results: list[bool | int] = []
    for probe in plan.probes:
        if len(set(probe.ids)) != len(probe.ids) or any(id not in by_id for id in probe.ids):
            raise ValidationError("invalid probe variable")
        if probe.kind == "bit" and len(probe.ids) == 1:
            results.append(by_id[probe.ids[0]])
        elif probe.kind == "count" and 2 <= len(probe.ids) <= 3:
            results.append(sum(by_id[id] for id in probe.ids))
        elif probe.kind == "same" and len(probe.ids) == 2:
            results.append(by_id[probe.ids[0]] == by_id[probe.ids[1]])
        else:
            raise ValidationError("unsupported observation probe")
    return Observation(plan, tuple(results))


def oracle_posterior(schema: Schema, observation: Observation, prior: np.ndarray) -> np.ndarray:
    ordered = canonical_schema(schema)
    if tuple(v.id for v in ordered) != observation.plan.schema_ids:
        raise ValidationError("observation and schema disagree")
    worlds = enumerate_worlds(ordered)
    probabilities = np.asarray(prior, dtype=np.float64)
    if probabilities.shape != (len(worlds),) or not np.isfinite(probabilities).all() or np.any(probabilities < 0) or not np.isclose(probabilities.sum(), 1):
        raise ValidationError("invalid declared prior")
    compatible = np.array([observe(observation.plan, tuple(bool(bit) for bit in world)).results == observation.results for world in worlds])
    posterior = np.where(compatible, probabilities, 0.0)
    mass = posterior.sum()
    if mass <= 0:
        raise ValidationError("observation has zero likelihood under prior")
    return posterior / mass


def _descriptor(variable: Variable) -> tuple[str, str]:
    meaning = variable.meaning
    if meaning.endswith(" damaged"):
        return variable.entity, "damaged"
    if meaning.endswith(" unused"):
        return variable.entity, "unused"
    raise ValidationError("renderer does not recognize variable meaning")


def _item_list(entities: list[str]) -> str:
    names = [entity.removesuffix(" item") for entity in entities]
    if len(names) == 2:
        return f"the {names[0]} and {names[1]} items"
    if len(names) == 3:
        return f"the {names[0]}, {names[1]}, and {names[2]} items"
    raise ValidationError("count renderer requires two or three items")


def _render_probe(lookup: dict[str, Variable], probe: Probe, result: bool | int, renderer_id: str) -> str:
    if probe.kind == "bit":
        entity, attribute = _descriptor(lookup[probe.ids[0]])
        if renderer_id == "plain":
            return f"The {entity} is {'not ' if not result else ''}{attribute}."
        if result:
            return f"The {entity} is {attribute}."
        if attribute == "damaged":
            return f"The {entity} is undamaged."
        return f"The {entity} has been used."
    if probe.kind == "count":
        descriptors = [_descriptor(lookup[id]) for id in probe.ids]
        attributes = {attribute for _, attribute in descriptors}
        if len(attributes) != 1:
            raise ValidationError("count probe must use one attribute")
        attribute = attributes.pop()
        item_list = _item_list([entity for entity, _ in descriptors])
        word = NUMBER_WORDS[int(result)]
        if renderer_id == "plain":
            verb = "is" if result == 1 else "are"
            return f"Exactly {word} of {item_list} {verb} {attribute}."
        return f"The number of {attribute} items among {item_list} is {word}."
    first_entity, first_attribute = _descriptor(lookup[probe.ids[0]])
    second_entity, second_attribute = _descriptor(lookup[probe.ids[1]])
    first_status = "damage" if first_attribute == "damaged" else "unused"
    second_status = "damage" if second_attribute == "damaged" else "unused"
    left = f"the {first_status} status of the {first_entity}"
    right = f"the {second_status} status of the {second_entity}"
    if renderer_id == "plain":
        relationship = "is the same as" if result else "differs from"
    else:
        relationship = "matches" if result else "does not match"
    return f"{left.capitalize()} {relationship} {right}."


def render_observation(schema: Schema, observation: Observation, renderer_id: str, rng: np.random.Generator) -> str:
    if renderer_id not in {"plain", "alternative"}:
        raise ValidationError("unknown renderer family")
    lookup = {item.id: item for item in canonical_schema(schema)}
    if tuple(lookup) != observation.plan.schema_ids:
        raise ValidationError("renderer schema does not match observation")
    clauses = [_render_probe(lookup, probe, result, renderer_id) for probe, result in zip(observation.plan.probes, observation.results, strict=True)]
    if not clauses:
        return "No item condition was established."
    order = rng.permutation(len(clauses))
    return " ".join(clauses[int(index)] for index in order)


def generate_source(config: dict[str, Any], source_id: str, seed: int) -> GeneratedSource:
    if not isinstance(source_id, str) or not source_id:
        raise ValidationError("source_id must be a nonempty opaque string")
    state_seed, plan_seed, render_seed = np.random.SeedSequence(seed).spawn(3)
    state_rng = np.random.default_rng(state_seed)
    plan_rng = np.random.default_rng(plan_seed)
    render_rng = np.random.default_rng(render_seed)
    schema = returns_schema()
    latent = tuple(bool(bit) for bit in state_rng.integers(0, 2, size=6))
    plan = sample_observation_plan(config, plan_rng)
    observed = observe(plan, latent)
    posterior = oracle_posterior(schema, observed, np.full(64, 1 / 64))
    renderer_id = config.get("data", {}).get("renderer_family") or str(render_rng.choice(("plain", "alternative")))
    public = PublicCase(source_id, "returns-six-v1", schema, {}, render_observation(schema, observed, renderer_id, render_rng))
    private = PrivateCase(
        source_id,
        latent,
        {"plan": asdict(plan), "results": list(observed.results)},
        tuple(float(value) for value in posterior),
        source_id,
        "unassigned",
        renderer_id,
        seed,
        (plan.component, *(probe.kind for probe in plan.probes)),
    )
    return GeneratedSource(public, private, plan.workflow, plan.queried_item)


@dataclass(frozen=True)
class PolicyPools:
    train: Mapping[str, tuple[AST, ...]]
    dev: Mapping[str, tuple[AST, ...]]
    confirmation: Mapping[str, tuple[AST, ...]]


@dataclass(frozen=True)
class ManifestView:
    case_id: str
    source_group: str
    split: str
    parent_id: str | None
    renderer: str
    primary: bool
    public_hash: str | None = None
    normalized_text_hash: str | None = None
    policy_key_hex: str | None = None


@dataclass(frozen=True)
class SplitManifest:
    group_splits: Mapping[str, str]
    views: tuple[ManifestView, ...] = ()
    settings: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AuditResult:
    ok: bool
    errors: list[str]
    notes: list[str]


def policy_key(ast: AST, schema: Schema) -> bytes:
    ordered = canonical_schema(schema)
    checked = validate_ast(ast_to_dict(ast), ordered)
    values = evaluate_worlds(checked, ordered, enumerate_worlds(ordered))
    return np.packbits(values, bitorder="little").tobytes()


def build_policy_pools(schema: Schema, seed: int) -> PolicyPools:
    ordered = canonical_schema(schema)
    if tuple(v.id for v in ordered) != tuple(f"v{i}" for i in range(6)):
        raise ValidationError("policy pools require the fixed six-slot returns schema")
    damage = ("v0", "v2", "v4")
    unused = ("v1", "v3", "v5")

    def v(id: str) -> dict[str, Any]:
        return {"op": "var", "id": id}

    def neg(expr: dict[str, Any]) -> dict[str, Any]:
        return {"op": "not", "arg": expr}

    def conj(*exprs: dict[str, Any]) -> dict[str, Any]:
        return {"op": "and", "args": list(exprs)}

    def disj(*exprs: dict[str, Any]) -> dict[str, Any]:
        return {"op": "or", "args": list(exprs)}

    def count(op: str, ids: Sequence[str], k: int) -> dict[str, Any]:
        return {"op": op, "ids": list(ids), "k": k}

    candidates: tuple[dict[str, list[dict[str, Any]]], ...] = (
        {
            "aggregate": [count("count_ge", damage, 2), count("count_eq", damage, 1), count("count_ge", damage[:2], 1)],
            "item": [v(id) for id in damage],
            "combination": [conj(v(d), v(u)) for d, u in zip(damage, unused)],
        },
        {
            "aggregate": [conj(count("count_ge", damage, 1), neg(count("count_eq", damage[:2], 2)))],
            "item": [conj(v(d), neg(v(u))) for d, u in zip(damage, unused)],
            "combination": [disj(conj(v(d), v(u)), v(damage[(i + 1) % 3])) for i, (d, u) in enumerate(zip(damage, unused))],
        },
        {
            "aggregate": [conj(count("count_eq", damage, 1), disj(*(v(u) for u in unused)))],
            "item": [disj(conj(v(d), v(u)), conj(v(damage[(i + 1) % 3]), neg(v(unused[(i + 1) % 3])))) for i, (d, u) in enumerate(zip(damage, unused))],
            "combination": [conj(disj(v(d), v(damage[(i + 1) % 3])), count("count_ge", (u, unused[(i + 1) % 3]), 1)) for i, (d, u) in enumerate(zip(damage, unused))],
        },
    )
    pools: list[dict[str, tuple[AST, ...]]] = []
    across_pools: set[bytes] = set()
    for candidate_pool in candidates:
        resolved: dict[str, tuple[AST, ...]] = {}
        local_keys: set[bytes] = set()
        for workflow, formulas in candidate_pool.items():
            unique: dict[bytes, AST] = {}
            for raw in formulas:
                ast = validate_ast(raw, ordered)
                key = policy_key(ast, ordered)
                truth_count = int.from_bytes(key, "little").bit_count()
                if not 8 <= truth_count <= 56:
                    raise ValidationError("primary policy has a skewed or constant action")
                if key in across_pools:
                    raise ValidationError("policy truth table leaked across pools")
                unique[key] = ast
                local_keys.add(key)
            if not unique:
                raise ValidationError(f"empty policy workflow: {workflow}")
            resolved[workflow] = tuple(unique.values())
        across_pools.update(local_keys)
        pools.append(resolved)
    return PolicyPools(*pools)


def assign_splits(source_ids: Sequence[str], config: dict[str, Any]) -> SplitManifest:
    data = config.get("data", {})
    counts = {name: data.get(name) for name in ("train", "dev", "calibration", "test")}
    if any(type(value) is not int or value < 0 for value in counts.values()) or sum(counts.values()) != len(source_ids):
        raise ValidationError("split sizes must equal independent source group count")
    if len(set(source_ids)) != len(source_ids) or any(not isinstance(id, str) or not id for id in source_ids):
        raise ValidationError("source IDs must be unique opaque strings")
    rng = np.random.default_rng(int(data["split_seed"]))
    randomized = [source_ids[int(index)] for index in rng.permutation(len(source_ids))]
    groups: dict[str, str] = {}
    start = 0
    for split, count in counts.items():
        groups.update((id, split) for id in randomized[start : start + count])
        start += count
    return SplitManifest(groups, settings={"split_seed": int(data["split_seed"]), "counts": counts})


def renderer_for_split(config: dict[str, Any], split: str, rng: np.random.Generator) -> str:
    key = {"train": "train_renderer_weights", "dev": "dev_renderer_weights", "calibration": "target_renderer_weights", "test": "target_renderer_weights"}.get(split)
    if key is None:
        raise ValidationError("unknown split for renderer sampling")
    weights = config.get("data", {}).get(key)
    if not isinstance(weights, Mapping) or set(weights) != {"plain", "alternative"}:
        raise ValidationError("renderer weights are missing")
    values = [weights["plain"], weights["alternative"]]
    if any(not isinstance(value, (int, float)) or not np.isfinite(value) or value < 0 for value in values) or not np.isclose(sum(values), 1.0):
        raise ValidationError("renderer weights must be nonnegative and sum to one")
    return str(rng.choice(("plain", "alternative"), p=values))


def _normalized_text(text: str) -> str:
    return " ".join(text.casefold().split())


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def audit_dataset(public_path: Path, private_path: Path, manifest: SplitManifest, pools: PolicyPools) -> AuditResult:
    errors: list[str] = []
    notes: list[str] = []
    try:
        public = read_public_jsonl(public_path)
        private = read_private_jsonl(private_path)
    except (ValidationError, OSError) as exc:
        return AuditResult(False, [f"unreadable public/private case files: {exc}"], notes)
    by_public = {case.case_id: case for case in public}
    by_private = {case.case_id: case for case in private}
    if len(by_public) != len(public) or len(by_private) != len(private) or set(by_public) != set(by_private):
        errors.append("public/private case IDs are duplicate or unmatched")
    views = {view.case_id: view for view in manifest.views}
    if manifest.views and (len(views) != len(manifest.views) or set(views) != set(by_public)):
        errors.append("manifest views are duplicate or unmatched")
    grouped_primary: dict[str, int] = {}
    parent_groups = {view.case_id: view.source_group for view in manifest.views}
    parent_splits = {view.case_id: view.split for view in manifest.views}
    for case_id, case in by_public.items():
        label = by_private.get(case_id)
        view = views.get(case_id)
        if label is None:
            continue
        if manifest.group_splits.get(label.source_group) != label.split:
            errors.append(f"private group/split mismatch: {case_id}")
        if view is not None:
            if view.source_group != label.source_group or view.split != label.split or view.renderer != label.renderer:
                errors.append(f"manifest group/split/renderer mismatch: {case_id}")
            grouped_primary[view.source_group] = grouped_primary.get(view.source_group, 0) + int(view.primary)
            if view.parent_id is not None:
                if view.parent_id not in parent_groups or parent_groups[view.parent_id] != view.source_group or parent_splits[view.parent_id] != view.split:
                    errors.append(f"copied parent crosses split or group: {case_id}")
            if view.public_hash is not None and view.public_hash != _sha(public_model_input(case)):
                errors.append(f"public hash mismatch: {case_id}")
            if view.normalized_text_hash is not None and view.normalized_text_hash != _sha(_normalized_text(case.text)):
                errors.append(f"normalized text hash mismatch: {case_id}")
        else:
            grouped_primary[label.source_group] = grouped_primary.get(label.source_group, 0) + 1
    for group, count in grouped_primary.items():
        if count != 1:
            errors.append(f"source group has {count} primary views: {group}")
    texts: dict[str, list[str]] = {}
    for case in public:
        texts.setdefault(_normalized_text(case.text), []).append(case.case_id)
    recurrence_count = sum(len(ids) - 1 for ids in texts.values())
    notes.append(f"independently sampled text recurrence count: {recurrence_count}; copied views require shared provenance")
    keys_by_pool: list[set[bytes]] = []
    for pool in (pools.train, pools.dev, pools.confirmation):
        keys_by_pool.append({policy_key(ast, returns_schema()) for formulas in pool.values() for ast in formulas})
    if not keys_by_pool[0].isdisjoint(keys_by_pool[2]) or not keys_by_pool[0].isdisjoint(keys_by_pool[1]) or not keys_by_pool[1].isdisjoint(keys_by_pool[2]):
        errors.append("policy truth table overlap across pools")
    return AuditResult(not errors, errors, notes)


def select_primary_policy(public_case: PublicCase, public_workflow: Mapping[str, str], pool: Sequence[AST], rng: np.random.Generator) -> AST:
    if tuple(v.id for v in canonical_schema(public_case.schema)) != tuple(f"v{i}" for i in range(6)):
        raise ValidationError("primary policy selection requires fixed returns schema")
    workflow = public_workflow.get("workflow")
    item = public_workflow.get("queried_item")
    if workflow not in {"aggregate", "item", "combination"} or item not in ITEMS:
        raise ValidationError("unknown public workflow or queried item")
    if workflow == "aggregate":
        eligible = list(pool)
    else:
        target = f"v{2 * ITEMS.index(item)}"
        def references(node: AST) -> set[str]:
            if node.op == "var":
                return {node.id}
            if node.op in {"count_eq", "count_ge"}:
                return set(node.ids)
            if node.op == "not":
                return references(node.arg)
            return set().union(*(references(child) for child in node.args))

        eligible = [ast for ast in pool if target in references(ast)]
    if not eligible:
        raise ValidationError("no eligible policy for public workflow")
    return eligible[int(rng.integers(0, len(eligible)))]
