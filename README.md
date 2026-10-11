# RBC — Relational Belief Compilation

Two returned items are equally likely to be damaged. But you also know **exactly one** is damaged. Store only the two 50% probabilities and you lose that relationship: an independent model invents worlds where both items are damaged or neither is.

RBC tests whether keeping complete, joint interpretations of uncertain facts lets us reuse the same evidence across different policies—and abstain when those interpretations disagree.

**The exact representation works. On 500 generated returns cases, a conventional text parser matched the oracle, and keeping relationships added only 2 percentage points over independent marginals.** That was below the declared continuation threshold, so the experiment stopped before training a learned compiler.

The result gives a working relational decision kernel, a replayable benchmark, and a concrete reason to avoid a model-training project on this workload.

## One case makes the difference

| Retained interpretation | Red item damaged | Blue item damaged |
|---|:---:|:---:|
| World 1 | Yes | No |
| World 2 | No | Yes |

Ask whether exactly one item is damaged: both worlds say yes. Ask whether both are damaged: both say no. Ask whether the red item is damaged: they disagree, so the executor abstains and returns the two worlds as witnesses.

The proposed compiler would turn case text and a declared schema into a distribution over complete interpretations, then reuse it for later policies. This prototype evaluates policies exactly over the retained worlds. It recommends an action only when every retained world agrees. That certificate establishes agreement within the set; it doesn't prove the true world was retained.

The experiment uses six Boolean facts—damaged or unused for each of three items—giving 64 possible worlds. Individual facts, relational observations, and missing information narrow the retained set.

## What the screen found

Each of the 500 generated development cases had one preselected policy. The oracle saw the generator's private observation program. The parser saw only public case text and the schema. The marginal-product control used oracle marginals, deliberately discarding their relationships.

| Method | Accepted decisions / 500 | Accepted mistakes |
|---|---:|---:|
| Exact joint oracle | 168 | 0 |
| Conventional text parser | 168 | 0 |
| Product of oracle marginals | 158 | 0 |

Keeping relationships gained **10 decisions: 2 percentage points**. The predeclared gate required at least **10 points**. Even within the 266 relational cases, the gain was only about 3.8 points.

The parser reproduced the oracle's retained worlds on all 500 cases. Under the rule requiring agreement in every possible world, it reached the information ceiling on this generated language. Many remaining cases simply lacked the facts needed for a certain answer. A probabilistic rule could accept more decisions with some risk; that would change the operating contract.

Zero observed accepted mistakes also didn't establish the project's 2% risk target: the parser's panel-adjusted one-sided upper bound was about **2.41%**. These were development cases from agent-authored templates, with no independent human-written validation. Learned heads, policy-supervision ablations, final held-out evaluation, and the local-LLM baseline were **not run**.

[Full report](runs/poc-20260928-01/REPORT.md) · [Stop decision](docs/results/2026-09-28-g1.md) · [Research specification](docs/superpowers/specs/2026-09-28-rbc-design.md)

## What remains useful

The [logic kernel](src/rbc/logic.py) preserves relationships, executes policies, and supplies disagreement witnesses. The [parser and controls](src/rbc/baselines.py) show what ordinary relational software can already capture. The [case generator](src/rbc/data.py) and [verification/reporting code](src/rbc/report.py) make the negative result inspectable.

A separate synthetic shift example also shows a failure worth understanding: two populations can have identical individual marginals and different relationships. A frozen old calibration accepted 128 wrong XOR actions out of 128 shifted cases; recalibration on separate shifted cases abstained on all of them. This is a numeric counterexample, not a trained-model result or 128 independent language evaluations.

The next useful test would use independently written cases and competent parser/LLM baselines before reopening training. Changing the generator merely to break the parser wouldn't establish practical value.

## Replay without a model

The saved run is about 2 MB. It contains the [effective config](runs/poc-20260928-01/run.json), [per-case predictions](runs/poc-20260928-01/predictions.jsonl), and report. Verification needs no model weights or network access after installation.

Use Python 3.12 for the pinned dependencies below; the recorded run used native ARM Python 3.12.9 on an Apple M3 Max with 36 GB RAM.

```bash
git clone https://github.com/arlegotin/RBC.git
cd RBC
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.lock.txt
python -m pip install -e . --no-deps
python -m pytest -q
python -m rbc verify --run runs/poc-20260928-01
```

To regenerate the report or resume the completed gates:

```bash
python -m rbc report --run runs/poc-20260928-01
python -m rbc doctor
python -m rbc run --through g0 --config configs/poc.yaml
python -m rbc run --through g1 --config configs/poc.yaml
```

Compatible stages reuse saved artifacts. Requests for `g2` or `g3` report `blocked_by_g1`; no learned or local-LLM snapshot was downloaded. The existing failure remains the result of this workload.

## Credit and reuse

By **Artem Legotin**. Original code is licensed under [AGPL-3.0-only](LICENSE); original prose and reports under [CC BY 4.0](LICENSES/CC-BY-4.0.txt). Preserve author credit when sharing covered material; see [licensing scope](LICENSING.md).

If you build on the decision kernel, benchmark, findings, or ideas, please cite [CITATION.cff](CITATION.cff), link this repository, and identify the commit you used. Citation of independently reused ideas is a scholarly request, separate from the license obligations.
