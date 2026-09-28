# Relational Belief Compilation

A local research PoC for testing whether preserving joint uncertainty improves reusable policy decisions, and whether auxiliary policy supervision helps beyond ordinary joint likelihood training. A well-supported negative result counts as success.

The repository currently contains the specification and implementation plan. The experiment, package, and CLI have not been implemented or run.

- [Research specification](docs/superpowers/specs/2026-09-28-rbc-design.md): semantics, data boundaries, models, baselines, resource ceilings, statistical rules, and decision gates.
- [Implementation plan](docs/superpowers/plans/2026-09-28-rbc.md): concrete interfaces, tests, commands, conditional tasks, and commit boundaries.

Start with the exact kernel and conventional parser. Build the six-variable learned prototype only if those baselines leave headroom. Keep representation, learning, and auxiliary-training conclusions separate. The plan targets an Apple Silicon M3 Max, with memory and throughput discovered during implementation, and caps experiment compute at 180 minutes.
