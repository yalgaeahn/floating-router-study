from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from remote_circuit import load_experiment, validate_experiment
from remote_circuit.experiment import ExperimentConfigError, NumericsSpec


ROOT = Path(__file__).parents[1]


def test_unsupported_method_is_rejected():
    spec = load_experiment(
        ROOT / "configs" / "smoke" / "floating_island_router.toml"
    )
    from floating_router_study.plugin import MODEL
    from remote_circuit import ModelRegistry

    registry = ModelRegistry([replace(MODEL, supported_methods=frozenset({"exact"}))])
    invalid = replace(spec, methods=replace(spec.methods, schrieffer_wolff=True))
    with pytest.raises(ExperimentConfigError, match="does not support"):
        validate_experiment(invalid, registry=registry)


def test_model_specific_numerics_are_checked_before_a_run():
    spec = load_experiment(
        ROOT / "configs" / "smoke" / "floating_island_router.toml"
    )
    invalid = replace(
        spec,
        numerics=NumericsSpec("smoke", {}, {"exact": {"unknown_option": 1}}),
    )
    with pytest.raises(ValueError, match="unsupported option"):
        validate_experiment(invalid)


def test_floating_qubit_display_labels_are_validated_before_run():
    spec = load_experiment(
        ROOT / "configs" / "smoke" / "floating_island_router.toml"
    )
    valid = replace(
        spec,
        circuit=replace(
            spec.circuit,
            topology=replace(
                spec.circuit.topology,
                parameters={**spec.circuit.topology.parameters, "computational_qubits": ["Q1", "Q2"]},
            ),
        ),
    )
    validate_experiment(valid)
    invalid = replace(
        spec,
        circuit=replace(
            spec.circuit,
            topology=replace(
                spec.circuit.topology,
                parameters={**spec.circuit.topology.parameters, "computational_qubits": ["Q1", ""]},
            ),
        ),
    )
    with pytest.raises(ValueError, match="exactly two names"):
        validate_experiment(invalid)


