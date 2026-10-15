"""Run the declarative floating-island 2D smoke experiment."""

from pathlib import Path

from remote_circuit import load_experiment, plot_result, run_experiment


ROOT = Path(__file__).resolve().parents[1]


if __name__ == "__main__":
    spec = load_experiment(ROOT / "configs" / "smoke" / "floating_island_router.toml")
    result = run_experiment(spec)
    destination = result.save(ROOT / "results" / "floating-island-router-example.h5")
    for figure in plot_result(result, ROOT / "results" / "floating-island-router-example"):
        print(figure)
    print(destination)
