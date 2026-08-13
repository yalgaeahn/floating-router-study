# Circuit research project

This repository depends on a reviewed `remote-circuit` Git tag. It contains only the
experiment-specific circuit, configuration, notebooks, and tests.

```bash
uv sync
uv run remote-circuit check experiment.toml
uv run python scripts/run.py
uv run pytest
```

Start with `notebooks/quickstart.ipynb`. Keep generated HDF5 and figures out of Git;
commit the circuit, experiment configuration, and any small publication summary CSV.
Use `smoke` while editing and verify `standard`/`converged` stability before publication.
