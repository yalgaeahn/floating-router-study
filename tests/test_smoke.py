from pathlib import Path

from remote_circuit import load_experiment


def test_experiment_is_valid():
    root = Path(__file__).parents[1]
    spec = load_experiment(root / "experiment.toml")
    assert spec.name == "my-remote-dtc-study"
    assert spec.circuit.path.is_file()
