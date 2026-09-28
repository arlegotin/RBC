"""Executable observations and controlled, agent-authored case language."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from .logic import canonical_schema, enumerate_worlds
from .types import PrivateCase, PublicCase, Schema, ValidationError, Variable


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
