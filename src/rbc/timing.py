"""Measured local pipeline timings and explicitly hypothetical cost sensitivity."""

from __future__ import annotations

import math
import time
from typing import Any, Callable, Sequence

import numpy as np


def measure_pipeline(
    compile_case: Callable[[Any], Any],
    decide: Callable[[Any, Any], Any],
    cases: Sequence[Any],
    policies: Sequence[Any],
    *,
    warmups: int,
    clock: Callable[[], int] = time.perf_counter_ns,
) -> dict[str, Any]:
    if type(warmups) is not int or warmups < 10 or not cases or len(policies) < 16:
        raise ValueError("pipeline timing needs at least ten warmups, cases, and 16 policies")
    counts = (1, 4, 16)
    labels = ["compile_ms", "one_policy_ms"] + [f"simultaneous_{n}_ms" for n in counts] + [f"future_only_{n}_ms" for n in counts] + [f"compile_plus_future_{n}_ms" for n in counts]
    samples: dict[str, list[float]] = {name: [] for name in labels}
    for item_index in range(warmups + len(cases)):
        case = cases[item_index % len(cases)] if item_index < warmups else cases[item_index - warmups]
        record = item_index >= warmups

        def timed(fn: Callable[[], Any]) -> tuple[Any, int]:
            start = clock()
            value = fn()
            end = clock()
            if end < start:
                raise ValueError("nonmonotonic timing clock")
            return value, end - start

        bundle, compile_ns = timed(lambda: compile_case(case))
        _, policy_ns = timed(lambda: decide(bundle, policies[0]))
        if record:
            samples["compile_ms"].append(compile_ns / 1e6)
            samples["one_policy_ms"].append(policy_ns / 1e6)
        for count in counts:
            def simultaneous() -> None:
                current = compile_case(case)
                for policy in policies[:count]:
                    decide(current, policy)

            _, simultaneous_ns = timed(simultaneous)
            future_bundle, fresh_compile_ns = timed(lambda: compile_case(case))

            def future() -> None:
                for policy in policies[:count]:
                    decide(future_bundle, policy)

            _, future_ns = timed(future)
            if record:
                samples[f"simultaneous_{count}_ms"].append(simultaneous_ns / 1e6)
                samples[f"future_only_{count}_ms"].append(future_ns / 1e6)
                samples[f"compile_plus_future_{count}_ms"].append((fresh_compile_ns + future_ns) / 1e6)
    return {
        "samples": len(cases), "warmups": warmups, "batch_size": 1,
        "compile_calls_per_case": {f"simultaneous_{n}": 1 for n in counts} | {f"future_{n}": 1 for n in counts},
        "decision_calls_per_case": {f"simultaneous_{n}": n for n in counts} | {f"future_{n}": n for n in counts},
        "latency_ms": {name: {"p50": float(np.percentile(values, 50)), "p95": float(np.percentile(values, 95)), "mean": float(np.mean(values)), "throughput_cases_per_second_batch1": 1000 / float(np.mean(values)) if np.mean(values) > 0 else None} for name, values in samples.items()},
    }


def economic_sensitivity(measured: dict[str, float], assumptions: dict[str, Any]) -> list[dict[str, Any]]:
    """Cost per case; successful-outcome cost is absent without an observed success rate."""
    latency_ms = float(measured["compile_plus_future_16_ms"])
    coverage = float(measured["coverage"])
    volume = assumptions["volume"]
    rate = assumptions["machine_cost_per_second"]
    fixed = assumptions["fixed_cost"]
    retry = assumptions["audit_retry_cost_per_case"]
    success = assumptions.get("end_to_end_success_rate")
    costs = assumptions["review_costs"]
    if not math.isfinite(latency_ms) or latency_ms < 0 or not 0 <= coverage <= 1 or type(volume) is not int or volume <= 0 or any(not math.isfinite(float(value)) or value < 0 for value in (rate, fixed, retry, *costs)) or (success is not None and not 0 < success <= 1):
        raise ValueError("invalid economic inputs")
    machine_cost = latency_ms / 1000 * rate
    rows = []
    for review_cost in costs:
        for scenario, review_fraction, machine in (
            ("parser_vs_always_review", 1 - coverage, machine_cost),
            ("zero_review_advantage", 1.0, machine_cost),
            ("hypothetical_llm_tenth_machine_cost", 1.0, machine_cost / 10),
        ):
            per_case = fixed / volume + machine + review_fraction * review_cost + retry
            rows.append({
                "scenario": scenario, "review_cost": review_cost, "review_fraction": review_fraction,
                "review_savings_per_case": (1 - review_fraction) * review_cost,
                "machine_cost_per_case": machine, "fixed_cost_per_case": fixed / volume,
                "audit_retry_cost_per_case": retry, "cost_per_case": per_case,
                "cost_per_successful_case": None if success is None else per_case / success,
                "monetary_inputs": "hypothetical", "latency_input": "measured" if scenario != "hypothetical_llm_tenth_machine_cost" else "hypothetical_tenth_of_measured",
            })
    return rows
