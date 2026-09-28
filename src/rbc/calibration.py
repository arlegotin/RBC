"""State-set calibration, development gates, and exact accepted-risk statistics."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
from scipy.stats import beta, binom

from .types import ValidationError


@dataclass(frozen=True)
class PredictionRow:
    panel_id: str
    group_id: str
    case_id: str
    model_id: str
    policy_key_hex: str
    primary: bool
    seed: int | None
    status: str
    action: bool | None
    score: float | None
    policy_ast: dict[str, Any] | None = None
    witnesses: tuple[int, int] | None = None
    bundle_relpath: str | None = None
    duration_ms: float | None = None
    structural_certainty: bool = False


@dataclass(frozen=True)
class EvaluatedRow:
    prediction: PredictionRow
    true_index: int
    correct: bool | None
    true_score: float | None
    world_scores: tuple[float, ...] | None = None
    policy_values: tuple[bool, ...] | None = None


@dataclass(frozen=True)
class GateChoice:
    kind: str
    value: float | None
    feasible: bool
    provenance: dict[str, Any] = field(default_factory=dict)


def _dense_1d(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values)
    if arr.ndim != 1 or arr.size == 0 or arr.dtype.kind != "f" or not np.isfinite(arr).all():
        raise ValidationError("dense scores must be a nonempty finite float vector")
    return arr.astype(np.float64)


def state_scores(logits: np.ndarray) -> np.ndarray:
    arr = _dense_1d(logits)
    return np.max(arr) - arr


def fit_threshold(true_scores: np.ndarray, alpha: float) -> float:
    if not isinstance(alpha, (float, int)) or not math.isfinite(alpha) or not 0 < alpha < 1:
        raise ValidationError("alpha must lie strictly between zero and one")
    scores = np.asarray(true_scores)
    if scores.ndim != 1 or scores.dtype.kind != "f" or not np.isfinite(scores).all() or np.any(scores < 0):
        raise ValidationError("calibration scores must be finite nonnegative floats")
    m = len(scores)
    rank = math.ceil((m + 1) * (1 - alpha))
    if rank > m:
        return math.inf
    return float(np.partition(scores.astype(np.float64), rank - 1)[rank - 1])


def retained_mask(logits: np.ndarray, threshold: float) -> np.ndarray:
    if not (math.isfinite(threshold) or threshold == math.inf):
        raise ValidationError("invalid retained-set threshold")
    return state_scores(logits) <= threshold


def risk_upper(errors: int, accepted: int, delta: float) -> float:
    if type(errors) is not int or type(accepted) is not int or not 0 <= errors <= accepted or not isinstance(delta, (float, int)) or not 0 < delta < 1:
        raise ValidationError("invalid accepted-risk counts or delta")
    if accepted == 0 or errors == accepted:
        return 1.0
    return float(beta.ppf(1 - delta, errors + 1, accepted - errors))


def min_zero_error_acceptances(target: float, delta: float) -> int:
    if not 0 < target < 1 or not 0 < delta < 1:
        raise ValidationError("invalid power target or delta")
    estimate = max(1, math.ceil(math.log(delta) / math.log1p(-target)))
    while risk_upper(0, estimate, delta) > target:
        estimate += 1
    while estimate > 1 and risk_upper(0, estimate - 1, delta) <= target:
        estimate -= 1
    return estimate


def passing_error_count(accepted: int, target: float, delta: float) -> int | None:
    if type(accepted) is not int or accepted < 0 or not 0 < target < 1 or not 0 < delta < 1:
        raise ValidationError("invalid power inputs")
    if risk_upper(0, accepted, delta) > target:
        return None
    low, high = 0, accepted
    while low + 1 < high:
        middle = (low + high) // 2
        if risk_upper(middle, accepted, delta) <= target:
            low = middle
        else:
            high = middle
    return low


def risk_power(accepted: int, true_risk: float, target: float, delta: float) -> float:
    if not 0 <= true_risk <= 1:
        raise ValidationError("invalid hypothetical true risk")
    maximum = passing_error_count(accepted, target, delta)
    return 0.0 if maximum is None else float(binom.cdf(maximum, accepted, true_risk))


def _primary_unique(rows: Sequence[EvaluatedRow]) -> list[EvaluatedRow]:
    primary = [row for row in rows if row.prediction.primary]
    groups = [row.prediction.group_id for row in primary]
    if len(groups) != len(set(groups)):
        raise ValidationError("duplicate primary source group")
    return primary


def select_gate(fit_rows: Sequence[EvaluatedRow], check_rows: Sequence[EvaluatedRow], config: dict[str, Any]) -> GateChoice:
    fit = _primary_unique(fit_rows)
    check = _primary_unique(check_rows)
    if not fit or not check or set(row.prediction.group_id for row in fit) & set(row.prediction.group_id for row in check):
        raise ValidationError("gate fit/check groups must be nonempty and disjoint")
    settings = config["statistics"]
    candidates: list[tuple[float, float, float, GateChoice]] = []
    for threshold in settings.get("confidence_grid", ()):
        chosen = [row for row in check if row.prediction.score is not None and row.prediction.action is not None and row.prediction.score >= threshold and (threshold < 1.0 or row.prediction.structural_certainty)]
        accepted = len(chosen)
        errors = sum(row.correct is False for row in chosen)
        if accepted >= 50 and errors / accepted <= 0.01:
            coverage = accepted / len(check)
            candidates.append((coverage, -(errors / accepted), threshold, GateChoice("confidence", float(threshold), True, {"fit_groups": len(fit), "check_groups": len(check), "check_accepted": accepted, "check_errors": errors})))
    if all(row.true_score is not None for row in fit) and all(row.world_scores is not None and row.policy_values is not None for row in check):
        for alpha in settings.get("alpha_grid", ()):
            threshold = fit_threshold(np.array([row.true_score for row in fit], dtype=np.float64), alpha)
            accepted = errors = 0
            for row in check:
                scores = np.asarray(row.world_scores, dtype=np.float64)
                values = np.asarray(row.policy_values, dtype=np.bool_)
                if scores.shape != values.shape or not np.isfinite(scores).all():
                    raise ValidationError("incompatible state gate rows")
                kept = scores <= threshold
                if np.any(kept) and np.all(values[kept] == values[kept][0]):
                    accepted += 1
                    errors += bool(values[kept][0] != values[row.true_index])
            if accepted >= 50 and errors / accepted <= 0.01:
                coverage = accepted / len(check)
                candidates.append((coverage, -(errors / accepted), -alpha, GateChoice("state_set", float(alpha), True, {"provisional_threshold": None if math.isinf(threshold) else threshold, "provisional_infinite": math.isinf(threshold), "fit_groups": len(fit), "check_groups": len(check), "check_accepted": accepted, "check_errors": errors})))
    if not candidates:
        return GateChoice("abstain_all", None, False, {"fit_groups": len(fit), "check_groups": len(check)})
    return max(candidates, key=lambda item: (item[0], item[1], item[2]))[3]


def paired_coverage_interval(rows_a: Sequence[PredictionRow], rows_b: Sequence[PredictionRow], resamples: int, seed: int) -> tuple[float, float, float]:
    def keyed(rows: Sequence[PredictionRow]) -> dict[tuple[str, str, str], PredictionRow]:
        chosen = [row for row in rows if row.primary]
        keys = [(row.panel_id, row.group_id, row.policy_key_hex) for row in chosen]
        if len(keys) != len(set(keys)):
            raise ValidationError("duplicate primary case in paired comparison")
        return dict(zip(keys, chosen))

    a = keyed(rows_a)
    b = keyed(rows_b)
    if not a or set(a) != set(b) or type(resamples) is not int or resamples <= 0:
        raise ValidationError("paired comparisons need the same nonempty source cases")
    keys = sorted(a)
    differences = np.array([int(a[key].status == "accept") - int(b[key].status == "accept") for key in keys], dtype=float)
    estimate = float(differences.mean())
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(keys), size=(resamples, len(keys)))
    boot = differences[draws].mean(axis=1)
    return estimate, float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))
