from rbc.timing import economic_sensitivity


def test_unknown_success_rate_stays_unknown_and_zero_review_advantage_is_zero():
    rows = economic_sensitivity({"compile_plus_future_16_ms": 10.0, "coverage": 0.336},
                                {"volume": 10000, "machine_cost_per_second": 0.001, "fixed_cost": 0,
                                 "audit_retry_cost_per_case": 0, "review_costs": [0.25, 1.5, 5.0],
                                 "end_to_end_success_rate": None})
    assert len(rows) == 9
    assert all(row["cost_per_successful_case"] is None for row in rows)
    equal = [row for row in rows if row["scenario"] == "zero_review_advantage"]
    assert all(row["review_savings_per_case"] == 0 for row in equal)
    assert all(row["review_fraction"] == 1 for row in equal)
    cheap = [row for row in rows if row["scenario"] == "hypothetical_llm_tenth_machine_cost"]
    assert all(row["machine_cost_per_case"] < rows[0]["machine_cost_per_case"] for row in cheap)
