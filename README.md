# floating-router-study

This study owns the floating-island router implementation, adapter, plots and Q1–Q3 experiments. It depends on the reusable remote-circuit engine.

## Development

Keep a sibling remote-circuit checkout at version 0.11.0 and run:

```bash
uv sync
uv run remote-circuit check experiment.toml
uv run remote-circuit run experiment.toml
uv run pytest
uv run ruff check src tests scripts examples
uv run mypy
uv build
```

The distribution pins remote-circuit==0.11.0; the uv source override uses the sibling
editable checkout during development. Published installations do not use that override.
The package registers model ID `floating_island_router` through `remote_circuit.models`.
Result HDF5 files are immutable. Keep large generated results and figures out of Git.

## Research checks

Short solver and notebook checks run in normal CI. Long convergence/reproduction
checks are opt-in:

```bash
CIRCUIT_STUDY_RUN_EXPENSIVE=1 FLOATING_ROUTER_RUN_EXPENSIVE=1 uv run pytest -m 'converged or expensive'
```

The default experiment is the two-point Q1–Q3 smoke. It preserves Q1 at 21.584 nH
and Q3 at 20.796 nH. Use `scripts/run_q1_q3.py baseline` for the 101-point baseline
and `scripts/run_q1_q3.py convergence` for the eight convergence runs.
`notebooks/q1-q3-analysis.ipynb` uses public experiment/result APIs. Exact matching
is pointwise dressed-proposer Gale–Shapley over the full retained basis.

## CI core dependency

CI checks out the reviewed core ref from the repository variable
REMOTE_CIRCUIT_REF (default main). A private core repository requires the
REMOTE_CIRCUIT_READ_TOKEN secret. Pin that ref to the reviewed 0.11 release before
publishing study results. Repository creation, secret configuration and publication
are not performed by this migration.
