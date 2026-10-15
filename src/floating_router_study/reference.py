"""Reference-only builders for the historical flagship floating-island circuit."""

from __future__ import annotations

import numpy as np

from floating_router_study.physical import (
    CoordinateReductionSpec,
    PhysicalCircuitSpec,
    SQUIDSpec,
    SubsystemSpec,
    SweepSpec,
)


def flagship_floating_island_reduction_spec() -> CoordinateReductionSpec:
    transform = np.asarray([
        [0, 0, 1, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 1, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 1, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 1, 0],
        [1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1],
    ], dtype=float)
    return CoordinateReductionSpec(
        coordinate_names=("lambda_L", "lambda_R", "u1", "u2", "q1", "q2", "s1", "s2", "m"),
        transformation_matrix=transform,
        active_coordinates=("q1", "q2", "s1", "s2"),
        free_coordinates=("lambda_L", "lambda_R", "u1", "u2", "m"),
    )


def build_flagship_floating_island_spec(
    circuit_yaml: str,
    *,
    qubit_truncated_dim: int = 4,
    router_truncated_dim: int = 16,
    qubit_ncut: int = 7,
    router_ncut: int = 7,
) -> PhysicalCircuitSpec:
    return PhysicalCircuitSpec(
        circuit_yaml=str(circuit_yaml),
        subsystems=(
            SubsystemSpec(
                "Q1", "floating_transmon", (1, 7), role="computational_qubit",
                truncated_dim=int(qubit_truncated_dim), ncut=int(qubit_ncut),
                external_flux_parameters=("Phi1",),
                squids=(SQUIDSpec(1, ("EJqL", "EJqL"), "Phi1", other_node=7),),
            ),
            SubsystemSpec(
                "Q2", "floating_transmon", (2, 8), role="computational_qubit",
                truncated_dim=int(qubit_truncated_dim), ncut=int(qubit_ncut),
                external_flux_parameters=("Phi2",),
                squids=(SQUIDSpec(2, ("EJqR", "EJqR"), "Phi2", other_node=8),),
            ),
            SubsystemSpec(
                "R", "floating_island_router", (3, 5, 6, 4), role="coupler",
                truncated_dim=int(router_truncated_dim), ncut=int(router_ncut),
                external_flux_parameters=("Phi3", "Phi4"),
                squids=(
                    SQUIDSpec(3, ("EJs1", "EJs1"), "Phi3", other_node=5),
                    SQUIDSpec(6, ("EJs2", "EJs2"), "Phi4", other_node=4),
                ),
            ),
        ),
        sweep=SweepSpec("phi_over_pi", 0.0, 2.0, 101),
        computational_qubits=("Q1", "Q2"),
        tunable_subsystem="R",
    )


__all__ = [
    "build_flagship_floating_island_spec",
    "flagship_floating_island_reduction_spec",
]
