from pathlib import Path
import json
import numpy as np

from remote_circuit import load_experiment, run_experiment

ROOT = Path(__file__).parents[1]

def test_solver_smoke_regression():
    baseline = json.loads((ROOT / "tests/fixtures/smoke_baseline.json").read_text())
    result = run_experiment(load_experiment(ROOT / "configs" / baseline["config"]))
    assert result.metadata["circuit_fingerprint"] == baseline["circuit_fingerprint"]
    rtol, atol = baseline["rtol"], baseline["atol"]
    exact = result.methods["exact"]
    sw = result.methods["schrieffer_wolff"]
    assert np.isclose(np.real(exact["exact_j_ghz"][0, 0]), baseline["exact_j_ghz"], rtol=rtol, atol=atol)
    assert np.isclose(exact["exact_zz_ghz"][0, 0], baseline["exact_zz_ghz"], rtol=rtol, atol=atol)
    assert np.isclose(sw["sw_zz_cumulative_by_order_ghz"][0, 0, -1], baseline["sw_zz_ghz"], rtol=rtol, atol=atol)
    assert np.all(result.diagnostics["validity"]["exact"])
