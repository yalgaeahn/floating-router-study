import os
from pathlib import Path

import numpy as np
import pytest
from remote_circuit import load_experiment, run_experiment

ROOT = Path(__file__).parents[1]


def test_experiment_is_valid():
    spec = load_experiment(ROOT / "experiment.toml")
    assert spec.name == "floating-island-q1-q3-smoke"
    assert spec.model_type == "floating_island_router"
    assert spec.schema_version == 2
    assert spec.circuit.path.is_file()


def test_q1_q3_baseline_contract():
    spec = load_experiment(ROOT / "experiments" / "q1-q3-baseline.toml")
    assert spec.model_type == "floating_island_router"
    assert spec.sweep.paths == ("common",)
    assert spec.sweep.axes[0].points == 101
    assert spec.circuit.topology.parameters["computational_qubits"] == ["Q1", "Q3"]
    assert spec.numerics.components == {
        "Q1": {"truncated_dim": 4, "ncut": 7},
        "Q3": {"truncated_dim": 4, "ncut": 7},
        "R": {"truncated_dim": 16, "ncut": 11},
    }
    assert spec.numerics.backends["exact"] == {
        "reliable_overlap_floor": 0.75,
        "ambiguity_overlap_floor": 0.5,
    }
    assert spec.numerics.backends["schrieffer_wolff"]["maximum_order"] == 5
    circuit = spec.circuit.path.read_text(encoding="utf-8")
    assert "EJqL=21.584nH" in circuit
    assert "EJqR=20.796nH" in circuit


@pytest.mark.parametrize("kind, values", [("ncut", (7, 9, 11, 13)), ("dim", (12, 16, 20, 24))])
def test_q1_q3_convergence_contract(kind: str, values: tuple[int, ...]):
    for value in values:
        spec = load_experiment(
            ROOT / "experiments" / f"q1-q3-convergence-{kind}-{value}.toml"
        )
        assert spec.sweep.axes[0].points == 21
        assert spec.circuit.topology.parameters["computational_qubits"] == ["Q1", "Q3"]
        options = spec.numerics.components["R"]
        if kind == "ncut":
            assert options == {"truncated_dim": 20, "ncut": value}
        else:
            assert options == {"truncated_dim": value, "ncut": 11}
        assert spec.numerics.backends["schrieffer_wolff"]["maximum_order"] == 5


@pytest.mark.expensive
def test_expensive_q1_q3_baseline_result_contract():
    if os.environ.get("FLOATING_ROUTER_RUN_EXPENSIVE") != "1":
        pytest.skip("set FLOATING_ROUTER_RUN_EXPENSIVE=1")
    result = run_experiment(
        load_experiment(ROOT / "experiments" / "q1-q3-baseline.toml")
    )
    assert result.metadata["circuit_fingerprint"] == (
        "152cdff49d99b2295741849e0e609d5be11815aed5e621530bb7575d89d76a0c"
    )
    assert result.metadata["exact_reference_assignment"] == (
        "pointwise dressed-proposer Gale-Shapley"
    )
    assert result.metadata["assignment_scope"] == "full retained basis"
    assert result.metadata["assignment_algorithm"] == "gale-shapley"
    assert result.metadata["assignment_proposer"] == "dressed"
    frame = result.observables["common"]
    assert np.isclose(frame["qubit_1_transition_ghz"][0], 4.470408823292331)
    assert np.isclose(frame["qubit_2_transition_ghz"][0], 4.543847060423367)
    assignment = result.methods["exact"]["common"]["assignment_indices"]
    eigenvalues = result.methods["exact"]["common"]["eigenvalues_from_ground_ghz"]
    assert assignment.shape == (101, 256)
    assert eigenvalues.shape == (101, 256)
    assert np.asarray(result.metadata["qubit_labels"]).tolist() == ["Q1", "Q3"]
    np.testing.assert_allclose(
        frame["router_transition_1_ghz"][[0, 50, 100]],
        [5.464838559528701, 4.573601719531709, 0.5218828820304772],
    )
    np.testing.assert_allclose(
        frame["router_transition_2_ghz"][[0, 50, 100]],
        [5.554608049093787, 4.6487277386220285, 0.5218828820304969],
    )
    np.testing.assert_allclose(
        frame["sw_zz_order_5_khz"][[0, 50, 100]],
        [3.5942018109392, -468793.1238828548, 0.002224317603190347],
    )
    np.testing.assert_allclose(
        frame["sw_j_mhz"][[0, 50, 100]],
        [-0.4837911361365631, -40.50369376808398, -0.0099099048553366],
    )
    np.testing.assert_array_equal(
        result.methods["exact"]["common"]["selected_indices"][[52, 53]],
        [[0, 1, 2, 9], [0, 3, 4, 11]],
    )


@pytest.mark.expensive
@pytest.mark.parametrize(
    "path",
    sorted((ROOT / "experiments").glob("q1-q3-convergence-*.toml")),
)
def test_expensive_q1_q3_convergence_runs(path: Path):
    if os.environ.get("FLOATING_ROUTER_RUN_EXPENSIVE") != "1":
        pytest.skip("set FLOATING_ROUTER_RUN_EXPENSIVE=1")
    result = run_experiment(load_experiment(path))
    assert len(result.coordinates["common"]) == 21
    assert result.metadata["assignment_algorithm"] == "gale-shapley"
    assert result.metadata["assignment_scope"] == "full retained basis"
