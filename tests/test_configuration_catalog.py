from pathlib import Path
import pytest
from remote_circuit import load_experiment

ROOT = Path(__file__).parents[1]
CONFIGS = sorted(path for path in (ROOT / "configs").rglob("*.toml") if "gate_simulation" not in path.parts)

@pytest.mark.parametrize("path", CONFIGS, ids=lambda path: str(path.relative_to(ROOT)))
def test_configuration_catalog_preflight(path):
    spec = load_experiment(path)
    assert spec.model_type == "floating_island_router"
    assert spec.circuit.path.is_file()
