from __future__ import annotations

import argparse
from pathlib import Path

from remote_circuit import load_experiment, run_experiment, validate_result

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "experiments"
GROUPS = {
    "smoke": (EXPERIMENTS / "q1-q3-smoke.toml",),
    "baseline": (EXPERIMENTS / "q1-q3-baseline.toml",),
    "convergence": tuple(sorted(EXPERIMENTS.glob("q1-q3-convergence-*.toml"))),
}


def run_config(path: Path) -> Path:
    spec = load_experiment(path)
    result = run_experiment(spec)
    report = validate_result(result)
    if not report["ok"]:
        raise RuntimeError("; ".join(report["errors"]))
    basename = spec.output.basename or f"{spec.name}.h5"
    if not basename.endswith(".h5"):
        basename += ".h5"
    destination = result.save(spec.output.directory / basename)
    print(f"{spec.name}: {destination}")
    for warning in report["warnings"]:
        print(f"  warning: {warning}")
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run declarative Q1-Q3 floating-island experiments."
    )
    parser.add_argument(
        "group",
        choices=(*GROUPS, "all"),
        help="smoke, 101-point baseline, eight convergence runs, or all",
    )
    args = parser.parse_args()
    paths = (
        tuple(path for group in GROUPS.values() for path in group)
        if args.group == "all"
        else GROUPS[args.group]
    )
    for path in paths:
        run_config(path)


if __name__ == "__main__":
    main()
