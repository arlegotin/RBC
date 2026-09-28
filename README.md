# Relational Belief Compilation

A local research PoC for testing whether preserving joint uncertainty improves reusable policy decisions, and whether auxiliary policy supervision helps beyond ordinary joint likelihood training. A well-supported negative result counts as success.

The repository contains an executable exact kernel and a development-only G1 screen. The frozen 500-case screen stopped the learned route: exact joint support accepted 168 cases, oracle marginal products accepted 158, and the conventional public-text parser tied the joint oracle at 168 with no errors. The two-point joint–marginal gap missed the predeclared ten-point continuation gate. No learned or local-LLM comparison was run.

- [Research specification](docs/superpowers/specs/2026-09-28-rbc-design.md): semantics, data boundaries, models, baselines, resource ceilings, statistical rules, and decision gates.
- [Implementation plan](docs/superpowers/plans/2026-09-28-rbc.md): concrete interfaces, tests, commands, conditional tasks, and commit boundaries.

The [G1 outcome record](docs/results/2026-09-28-g1.md) gives the gate decision and its limits. The plan targets an Apple Silicon M3 Max, with memory and throughput discovered during implementation, and caps experiment compute at 180 minutes.

Bootstrap in the project directory with native arm64 Python 3.11 or 3.12:

```bash
/opt/homebrew/bin/python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test]'
python -m rbc doctor
python -m pytest -q
```

The tested core versions are in `requirements.lock.txt`. For a pinned reinstall in a fresh venv, install that file and then `python -m pip install -e . --no-deps`. Run the exact kernel without model downloads:

```bash
python -m rbc run --through g0 --config configs/poc.yaml
python -m rbc run --through g1 --config configs/poc.yaml
python -m rbc verify --run runs/poc-20260928-01
python -m rbc report --run runs/poc-20260928-01
```
