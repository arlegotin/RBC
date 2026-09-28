# Relational Belief Compilation

A local research PoC for testing whether preserving joint uncertainty improves reusable policy decisions, and whether auxiliary policy supervision helps beyond ordinary joint likelihood training. A well-supported negative result counts as success.

The repository contains the specification, implementation plan, and the first local package shell. Only `doctor` and a status-only `report` command are currently implemented; no research gate has been run yet.

- [Research specification](docs/superpowers/specs/2026-09-28-rbc-design.md): semantics, data boundaries, models, baselines, resource ceilings, statistical rules, and decision gates.
- [Implementation plan](docs/superpowers/plans/2026-09-28-rbc.md): concrete interfaces, tests, commands, conditional tasks, and commit boundaries.

Start with the exact kernel and conventional parser. Build the six-variable learned prototype only if those baselines leave headroom. Keep representation, learning, and auxiliary-training conclusions separate. The plan targets an Apple Silicon M3 Max, with memory and throughput discovered during implementation, and caps experiment compute at 180 minutes.

Bootstrap in the project directory with native arm64 Python 3.11 or 3.12:

```bash
/opt/homebrew/bin/python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test]'
python -m rbc doctor
python -m pytest -q
```

The package lock will record the tested installation. Later gate commands are described in the implementation plan and will be documented here when they work.
