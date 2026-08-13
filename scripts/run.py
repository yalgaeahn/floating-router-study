from pathlib import Path

from remote_circuit import load_experiment, run_experiment, validate_result


ROOT = Path(__file__).resolve().parents[1]
spec = load_experiment(ROOT / "experiment.toml")
result = run_experiment(spec)
report = validate_result(result)
if not report["ok"]:
    raise SystemExit(report["errors"])
print(result.save(ROOT / "results" / "smoke.h5"))
