# Outcome: G1 NO_GO — INSUFFICIENT_JOINT_MARGINAL_HEADROOM. Conventional parsing is the strongest deployable baseline in the completed screen.

## What ran

- G0 status: passed
- Deterministic checks: ........................................                                 [100%] 40 passed in 7.07s
- Generated truth-in-support checks: 16
- Audit: passed

## Numeric representation fixture

The oracle and exact XOR constraints have the same support. Each individual damage fact is unresolved; the exactly-one policy is invariant. The oracle marginal product adds worlds absent from the evidence.

| Method | Primary accepted | Primary errors |
| --- | ---: | ---: |
| exact_xor_constraints | 1 / 1 | 0 |
| oracle_joint | 1 / 1 | 0 |
| oracle_marginal_product | 0 / 1 | 0 |

These are mathematical fixtures, not language-learning or accepted-risk evidence.

## G1 generated-text development screen

500 independent generated development cases; no final calibration or test cases were evaluated. Wording is agent-authored generated text. Gate reason: `INSUFFICIENT_JOINT_MARGINAL_HEADROOM`. The optional local LLM was not attempted early screen.

| Method | Accepted / cases | Accepted errors | Coverage | One-sided risk upper bound (descriptive) |
| --- | ---: | ---: | ---: | ---: |
| conventional_parser | 168 / 500 | 0 | 0.336 | 0.024 |
| oracle_joint | 168 / 500 | 0 | 0.336 | 0.024 |
| oracle_marginal_product | 158 / 500 | 0 | 0.316 | 0.026 |

| Method | True state retained | Mean / median retained worlds | Joint NLL | Policy Brier |
| --- | ---: | ---: | ---: | ---: |
| conventional_parser | 1.000 | 17.4 / 12.0 | 2.256 | 0.146 |
| oracle_joint | 1.000 | 17.4 / 12.0 | 2.256 | 0.146 |
| oracle_marginal_product | 1.000 | 27.1 / 16.0 | 2.581 | 0.153 |

### Generated case slices

The following rows are descriptive development slices. The full metrics file also groups by workflow, relation order, oracle ambiguity, and true action label.

| Slice | Cases | Oracle accepted | Parser accepted | Marginal accepted |
| --- | ---: | ---: | ---: | ---: |
| component=explicit | 136 | 102 | 102 | 102 |
| component=missing | 98 | 5 | 5 | 5 |
| component=relational | 266 | 61 | 61 | 51 |
| renderer=alternative | 375 | 133 | 133 | 127 |
| renderer=plain | 125 | 35 | 35 | 31 |

The oracle uses private observation labels and is an information ceiling. The parser reads only the public case and schema. The marginal product uses oracle marginals and is an information-loss diagnostic. These are development results, not a 2% accepted-risk claim.

Joint-minus-marginal coverage: 0.020; oracle-minus-parser coverage: 0.000. The fixed majority-action baseline accepted all 250 held-out development cases and made 117 errors. Always-abstain coverage is zero and its conditional risk is undefined.
Renderer audit: 20/20 inspected; no apparent mismatch; the examples were agent-authored and reviewed by the coding agent, not independent humans.

### Oracle direct-policy readout

This uses private oracle probabilities and is only an information ceiling. Thresholds were not chosen from these development outcomes for a final claim.

| Probability threshold | Accepted / 500 | Errors |
| ---: | ---: | ---: |
| 0.700 | 279 | 27 |
| 0.800 | 179 | 2 |
| 0.900 | 168 | 0 |
| 0.950 | 168 | 0 |
| 0.980 | 168 | 0 |
| 0.990 | 168 | 0 |
| 0.995 | 168 | 0 |
| 1.000 | 168 | 0 |

## Hypotheses

- Representation: the measured development headroom did not meet the 10-point gate; no learned representation claim.
- Learning: not attempted because G1 stopped the learned route.
- Auxiliary policy supervision: not attempted.

## Limits and next step

This controlled generator and its renderer grammar do not validate transfer to independently written cases. Retain the exact parser as the reference for this workload; only reopen learning if a separately designed workload shows credible headroom over it.

## Stress and shift boundaries

The parser matched the declared behavior on 256 agent-authored targeted cases; semantic fixture mismatches: 0. These repeated fixtures are not independent risk samples.
In a separate numeric counterexample, a frozen anti-correlation score retained the wrong XOR action on 128 / 128 shifted cases (128 errors). Recalibration on separate shifted cases widened the retained set and accepted 0 actions. The two populations have identical one-variable marginals; this is not a trained-model result.

## Measured local work

On the recorded Apple M3 Max machine (36.0 GB physical memory), the public parser ran 10 warmups and 100 timed cases at batch size one. Cold process import plus first compile: 103.553 ms. Peak process RSS: 120.0 MB. No encoder or LLM timing was measured.

| Pipeline stage | p50 ms | p95 ms |
| --- | ---: | ---: |
| compile_ms | 0.1307 | 0.2132 |
| one_policy_ms | 0.0584 | 0.0856 |
| simultaneous_1_ms | 0.1806 | 0.2581 |
| simultaneous_4_ms | 0.3359 | 0.4627 |
| simultaneous_16_ms | 1.0129 | 1.8210 |
| future_only_1_ms | 0.0561 | 0.1133 |
| future_only_4_ms | 0.2104 | 0.3795 |
| future_only_16_ms | 0.8866 | 1.4463 |
| compile_plus_future_16_ms | 1.0276 | 1.7445 |

The parser was compiled once per case in every 1/4/16-policy scenario. These timings do not establish an RBC speed advantage because no learned route was built.

## Hypothetical cost sensitivity

Money inputs are hypothetical; review fraction comes from the generated G1 parser coverage. End-to-end success is unknown, so cost per successful case is unavailable.

| Review cost | Parser vs all-review cost/case | Zero review advantage cost/case | Hypothetical tenth machine-cost alternative |
| ---: | ---: | ---: | ---: |
| $0.25 | $0.1660 | $0.2500 | $0.2500 |
| $1.50 | $0.9960 | $1.5000 | $1.5000 |
| $5.00 | $3.3200 | $5.0000 | $5.0000 |

## Explicit result statuses

- `INSUFFICIENT_JOINT_MARGINAL_HEADROOM`: the measured joint support gain was below ten points.
- `NO_HEADROOM_OVER_SIMPLE_BASELINE`: the competent public parser tied oracle support on all 500 development cases.
- `EXTERNAL_VALIDATION_NOT_PERFORMED`: no independently written cases were available.
- Learned heads, auxiliary-loss ablation, and local-LLM routes: not attempted after the G1 no-go; no result is imputed to them.

## Environment and budget

Effective configuration: `runs/poc-20260928-01/run.json`; tested Python 3.12.9, arm64, macOS 26.6.2. Core package versions: {"PyYAML": "6.0.3", "numpy": "2.5.3", "pytest": "9.1.1", "scipy": "1.18.1"}.
Measured experiment compute charged: 8.86 s of 10800 s ceiling. Downloaded learned and local-LLM model files: none.

## Reproduction

```bash
python -m pytest -q
python -m rbc run --through g0 --config configs/poc.yaml
python -m rbc run --through g1 --config configs/poc.yaml
python -m rbc verify --run runs/poc-20260928-01
python -m rbc report --run runs/poc-20260928-01
```
