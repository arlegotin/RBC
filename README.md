# Relational Belief Compilation

A local research PoC for testing whether preserving joint uncertainty improves reusable policy decisions, and whether auxiliary policy supervision helps beyond ordinary joint likelihood training. A well-supported negative result counts as success.

The repository contains an executable exact kernel and a development-only G1 screen. The frozen 500-case screen stopped the learned route: exact joint support accepted 168 cases, oracle marginal products accepted 158, and the conventional public-text parser tied the joint oracle at 168 with no errors. The two-point joint–marginal gap missed the predeclared ten-point continuation gate. No learned or local-LLM comparison was run.

- [Research specification](docs/superpowers/specs/2026-09-28-rbc-design.md): semantics, data boundaries, models, baselines, resource ceilings, statistical rules, and decision gates.
- [Implementation plan](docs/superpowers/plans/2026-09-28-rbc.md): concrete interfaces, tests, commands, conditional tasks, and commit boundaries.

The [G1 outcome record](docs/results/2026-09-28-g1.md) gives the gate decision and its limits. The [replayable run](runs/poc-20260928-01/REPORT.md) includes public/private generated cases, source snapshots, per-case predictions, masks, metrics, stress results, timing, and the effective config in `run.json`. The run is about 2 MB. The plan targets an Apple Silicon M3 Max and caps experiment compute at 180 minutes; this run used the machine's measured 36 GB memory configuration.

Bootstrap in the project directory with native arm64 Python 3.11 or 3.12:

```bash
/opt/homebrew/bin/python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test]'
python -m rbc doctor
python -m pytest -q
```

The tested core versions are in `requirements.lock.txt`. For a pinned reinstall in a fresh venv, use:

```bash
python -m pip install -r requirements.lock.txt
python -m pip install -e . --no-deps
```

Replay and resume the completed gates offline, without model downloads:

```bash
python -m rbc run --through g0 --config configs/poc.yaml
python -m rbc run --through g1 --config configs/poc.yaml
python -m rbc verify --run runs/poc-20260928-01
python -m rbc report --run runs/poc-20260928-01
```

The G1 no-go stops the learned and local-LLM routes, so this run has no encoder or Qwen snapshot revision to fetch and no model weights to redistribute. `run --through g2` and `run --through g3` report `blocked_by_g1` and return nonzero on this run; those stages were not executed. The 256 targeted stress cases and 128-case numeric shift counterexample are in `stress.json`; parser timings from 100 local cases are in `timing.json`. The latter do not establish an RBC speed advantage.
