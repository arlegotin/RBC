# Relational Belief Compilation

**Research status: stopped after the first development experiment.** The representation works in a small, exact example, but its advantage was too small on our generated returns cases, and an ordinary text parser matched the best possible certainty decisions. That is a useful negative result: the planned learned model was not trained.

RBC asks a simple question: **can we keep relationships between uncertain facts, then reuse them to answer different policies?** This repository is a local, replayable experiment that tried to reject that idea cheaply before spending time on model training.

## The idea in one case

Imagine two returned items. Exactly one is damaged, but we do not know which.

| Possible interpretation | Red damaged | Blue damaged |
| --- | :---: | :---: |
| World 1 | Yes | No |
| World 2 | No | Yes |

Each item is damaged with probability 50%. If we keep only those two percentages and treat them as independent, we accidentally introduce two impossible worlds: both damaged and neither damaged. Keeping the **joint** possibilities preserves “exactly one.”

That matters when a policy asks a question:

| Policy question | Answer from these two worlds |
| --- | --- |
| Is exactly one item damaged? | **Yes** in both worlds: accept. |
| Are both items damaged? | **No** in both worlds: accept. |
| Is the red item damaged? | The worlds disagree: abstain. |

An RBC compiler is meant to turn case text and a declared schema into a distribution over complete interpretations, then reuse that result for later policies. This prototype's exact executor evaluates a policy in every retained world. It recommends an action only when those worlds agree; otherwise it returns two disagreeing worlds as witnesses. The agreement certificate describes the retained set. It does **not** prove that the true world was retained.

```text
case text + six named facts
          ↓
  belief over 64 worlds
          ↓
  executable policy
          ↓
  agreed answer, or abstention with witnesses
```

## What happened in the experiment

We generated 500 short returns cases, with one preselected policy per case. The six Boolean facts describe whether each of three items is damaged or unused. The experiment deliberately mixes individual facts, relational evidence, and missing information. The **oracle** knows the generator's true observation program; the **conventional parser** reads only the public case text and schema.

| Method | Decisions accepted out of 500 | Accepted mistakes | Role |
| --- | ---: | ---: | --- |
| Exact joint oracle | 168 | 0 | Information ceiling; sees private generator labels |
| Conventional text parser | 168 | 0 | Public-text baseline with exact policy execution |
| Product of oracle marginals | 158 | 0 | Measures information lost by discarding relationships |

The joint representation gained **10 decisions in 500 cases: 2 percentage points** over the marginal product. The predeclared continuation gate required at least **10 points**. Even on the 266 relational cases, the gain was 10 decisions, about **3.8 points**.

The parser reproduced the joint oracle's retained worlds on all 500 cases. For decisions requiring agreement across *every* possible world, it reached the information ceiling on this workload. The remaining cases often lack enough evidence for a certain answer; better text processing cannot reveal an item identity the case never supplied. A probabilistic action rule could accept more decisions with some risk, but that is a different operating choice.

These were **development cases**, not a final test set. Zero observed accepted mistakes did not establish the project's 2% accepted-risk target: the panel-adjusted one-sided upper bound for the parser was about **2.41%**. The language was generated from agent-authored templates, with no independent human-written validation. [Read the full report](runs/poc-20260928-01/REPORT.md) for slices, the oracle probability readout, stress tests, timing, and exact bounds.

## What this result says

| Research question | Result here |
| --- | --- |
| Does preserving relationships ever help? | **Yes in the exact example.** The workload-average gain was too small to pass the project gate. |
| Can a small model learn a useful language-to-joint-belief compiler? | **Untested.** The early gate stopped training before any encoder or learned head was downloaded. |
| Does policy-outcome supervision improve a joint model? | **Untested.** There was no learned-model comparison. |

For this generated workload, **a learned RBC compiler is not a good next investment**. The negative finding is specific: the relationship advantage was small under the chosen policies, and simple relational software already captured it. It does not establish how well the parser, a learned compiler, or a local LLM would handle independently written cases. The local-LLM baseline was not run after the early stop.

The smallest worthwhile follow-up is a separately written case set, evaluated against a competent parser and a local relational LLM compiler *before* reopening model training. Changing the generator just to make the parser fail would not answer the practical question.

## Explore and reproduce

The repository includes the complete [effective config](runs/poc-20260928-01/run.json), [per-case predictions](runs/poc-20260928-01/predictions.jsonl), and [replayable report](runs/poc-20260928-01/REPORT.md). The archived run is about 2 MB and needs no model weights or network access to verify. It was produced on an Apple M3 Max with 36 GB of physical memory, using native arm64 Python 3.12.9.

Create a local environment with Python 3.11 or 3.12; replace `python3.12` below with `python3.11` if that is your installed version. These commands use the tested core dependency versions and leave system Python untouched:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.lock.txt
python -m pip install -e . --no-deps
python -m pytest -q
```

Replay the saved decisions and regenerate the report:

```bash
python -m rbc verify --run runs/poc-20260928-01
python -m rbc report --run runs/poc-20260928-01
```

The following commands resume the completed correctness and development gates; compatible stages reuse their saved artifacts:

```bash
python -m rbc doctor
python -m rbc run --through g0 --config configs/poc.yaml
python -m rbc run --through g1 --config configs/poc.yaml
```

Requests for `g2` or `g3` on this run report `blocked_by_g1`. No learned or local-LLM model snapshot was fetched or pinned, and no result for those methods is implied.

## Where to look

- [Research specification](docs/superpowers/specs/2026-09-28-rbc-design.md) and [implementation plan](docs/superpowers/plans/2026-09-28-rbc.md): the predeclared gates and limits.
- [Exact logic kernel](src/rbc/logic.py) and [case generator](src/rbc/data.py): worlds, safe policies, observations, and oracle labels.
- [Conventional parser and controls](src/rbc/baselines.py): the baseline that tied the oracle on generated text.
- [Verification and reporting](src/rbc/report.py): offline replay from saved per-case records.
- [G1 decision note](docs/results/2026-09-28-g1.md): the concise research conclusion.
