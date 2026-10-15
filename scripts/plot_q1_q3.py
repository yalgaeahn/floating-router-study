from __future__ import annotations

import argparse
from pathlib import Path

from remote_circuit import plot_result, read_result

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot stored Q1-Q3 baseline results.")
    parser.add_argument(
        "result",
        nargs="?",
        type=Path,
        default=ROOT / "results" / "q1-q3-baseline-v2.h5",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results" / "q1-q3-baseline-v2-plots",
    )
    args = parser.parse_args()
    for path in plot_result(read_result(args.result), args.output_dir):
        print(path)


if __name__ == "__main__":
    main()
