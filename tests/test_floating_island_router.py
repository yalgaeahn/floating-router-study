from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import scipy.linalg

from floating_router_study.floating_island_router import (
    FloatingIslandAnalysisConfig,
    FloatingIslandMapConfig,
    FloatingIslandRouterModel,
    FloatingIslandScqubitsReference,
    _fors_pointwise_exact_point,
    _run_exact_pointwise,
    plot_floating_island_router_map,
    run_floating_island_router_map,
    run_floating_island_router_methods,
)
from floating_router_study.reference import (
    build_flagship_floating_island_spec,
    flagship_floating_island_reduction_spec,
)
from floating_router_study.physical import reduce_capacitance_to_active_coordinates
from remote_circuit.circuits import load_circuit_topology
from remote_circuit.matching import gale_shapley_assignment
from remote_circuit.methods.exact import lowdin_effective_hamiltonian


EXAMPLE = (
    Path(__file__).parents[1]
    / "examples"
    / "circuits"
    / "flagship_floating_island_router.yaml"
)


def _yaml() -> str:
    return load_circuit_topology(EXAMPLE).scqubits_yaml()


def _fast_model() -> FloatingIslandRouterModel:
    spec = build_flagship_floating_island_spec(
        _yaml(), qubit_truncated_dim=3, router_truncated_dim=6,
        qubit_ncut=4, router_ncut=4,
    )
    return FloatingIslandRouterModel(spec, flagship_floating_island_reduction_spec())


def _grouped_reference_point(model: FloatingIslandRouterModel, x: float, y: float):
    point = model.build_point(x, y)
    eigenvalues, eigenvectors = scipy.linalg.eigh(
        point.full_hamiltonian_ghz,
        check_finite=False,
    )
    targets = np.asarray([
        np.ravel_multi_index(state, point.dimensions)
        for state in ((0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0))
    ])
    assignment = gale_shapley_assignment(np.abs(eigenvectors))
    selected = assignment[targets]
    coefficient = eigenvectors[np.ix_(targets, selected)]
    energies = eigenvalues[selected] - eigenvalues[selected[0]]
    h_eff, _ = lowdin_effective_hamiltonian(coefficient, energies)
    return energies, complex(h_eff[1, 2])


def test_physical_node9_and_coordinate_schur_complement():
    spec = build_flagship_floating_island_spec(_yaml())
    parsed = spec.parse()
    reduced = reduce_capacitance_to_active_coordinates(
        parsed, flagship_floating_island_reduction_spec()
    )
    physical = parsed.capacitance_matrix_ff
    assert np.isclose(physical[4, 8], -106.56, rtol=1e-12, atol=1e-10)
    assert np.isclose(physical[5, 8], -107.12, rtol=1e-12, atol=1e-10)
    assert np.isclose(physical[8, 8], 106.56 + 107.12 + 844.97, rtol=1e-12, atol=1e-10)
    expected_coordinate = (
        reduced.transformation_matrix.T @ physical @ reduced.transformation_matrix
    )
    assert np.allclose(
        reduced.coordinate_capacitance_matrix_ff,
        expected_coordinate,
        rtol=1e-12,
        atol=1e-12,
    )
    active_indices = np.asarray([4, 5, 6, 7])
    inverse_block = np.linalg.inv(expected_coordinate)[
        np.ix_(active_indices, active_indices)
    ]
    assert np.allclose(
        np.linalg.inv(reduced.active_capacitance_matrix_ff),
        inverse_block,
        rtol=1e-12,
        atol=1e-12,
    )
    assert np.allclose(reduced.active_w_matrix_ghz, reduced.active_w_matrix_ghz.T)
    assert np.min(np.linalg.eigvalsh(reduced.active_capacitance_matrix_ff)) > 0.0


def test_floating_island_grounding_controls_series_mediated_coupling():
    reduction = flagship_floating_island_reduction_spec()
    no_ground = "\n".join(line for line in _yaml().splitlines() if "ECg9=" not in line)
    large_ground = _yaml().replace("ECg9=844.97fF", "ECg9=1000000fF")
    reduced_zero = reduce_capacitance_to_active_coordinates(
        build_flagship_floating_island_spec(no_ground), reduction
    )
    reduced_large = reduce_capacitance_to_active_coordinates(
        build_flagship_floating_island_spec(large_ground), reduction
    )
    assert reduced_zero.active_capacitance_matrix_ff[2, 3] < -10.0
    assert abs(reduced_large.active_w_matrix_ghz[2, 3]) < (
        1.0e-3 * abs(reduced_zero.active_w_matrix_ghz[2, 3])
    )


def test_grouped_partition_does_not_double_count_switch_interaction():
    model = _fast_model()
    point = model.build_point(0.2, 0.4)
    assert np.allclose(
        point.full_hamiltonian_ghz,
        np.diag(point.bare_energies_ghz) + point.interaction_ghz,
    )
    assert np.allclose(np.diag(point.interaction_ghz), 0.0, atol=1e-13)
    # s1-s2 is internal to H_R; V only contains terms across the three groups.
    assert point.dimensions == (3, 3, 6)


@pytest.mark.converged
def test_full_scqubits_reference_matches_converged_reduced_point():
    spec = build_flagship_floating_island_spec(_yaml())
    reduction = flagship_floating_island_reduction_spec()
    model = FloatingIslandRouterModel(spec, reduction)
    reference = FloatingIslandScqubitsReference(
        spec, reduction, local_cutoff=7, subsystem_trunc_dims=(4, 4, 6, 6)
    )
    grouped_energies, grouped_j = _grouped_reference_point(model, 0.0, 0.0)
    full = reference.evaluate_point(0.0, 0.0, evals_count=48)
    assert np.allclose(
        grouped_energies, full.branch_energies_ghz, rtol=0.0, atol=1.0e-5
    )
    grouped_zz = grouped_energies[3] - grouped_energies[1] - grouped_energies[2]
    assert abs(grouped_zz - full.zz_ghz) <= max(1.0e-6, 0.05 * abs(full.zz_ghz))
    assert abs(abs(grouped_j) - abs(full.j_ghz)) <= 1.0e-5


def test_one_dimensional_paths_and_five_by_five_map_are_consistent():
    model = _fast_model()
    one_d = run_floating_island_router_methods(
        model,
        FloatingIslandAnalysisConfig(
            coordinate_min=0.0,
            coordinate_max=0.4,
            points=5,
            paths=("common", "single_s1", "single_s2"),
            exact_evals_count=24,
            spectral_boundary_margin=1,
        ),
    )
    assert np.allclose(one_d["single_s1"].frame["phi4_over_pi"], 0.0)
    assert np.allclose(one_d["single_s2"].frame["phi3_over_pi"], 0.0)
    assert all(
        result.metadata["circuit_fingerprint"] == model.circuit_fingerprint
        for result in one_d.values()
    )
    map_result = run_floating_island_router_map(
        model,
        FloatingIslandMapConfig(
            phi3_min_over_pi=0.0,
            phi3_max_over_pi=0.4,
            phi4_min_over_pi=0.0,
            phi4_max_over_pi=0.4,
            phi3_points=5,
            phi4_points=5,
            exact_evals_count=24,
            spectral_boundary_margin=1,
        ),
    )
    assert np.allclose(
        np.diag(map_result.data["exact_zz_ghz"]) * 1.0e6,
        one_d["common"].frame["exact_zz_khz"],
        rtol=0.0,
        atol=1.0e-6,
    )
    assert np.allclose(
        np.diag(map_result.data["exact_j_ghz"]) * 1.0e3,
        one_d["common"].frame["exact_j_mhz"],
        rtol=0.0,
        atol=1.0e-9,
    )
    assert np.array_equal(
        np.diagonal(map_result.data["exact_selected_indices"], axis1=0, axis2=1).T,
        one_d["common"].exact["selected_indices"],
    )
    assert one_d["common"].metadata["exact_reference_assignment"] == (
        "pointwise dressed-proposer Gale-Shapley"
    )
    assert one_d["common"].metadata["assignment_scope"] == "full retained basis"
    assert map_result.data["sw_zz_cumulative_by_order_ghz"].shape == (5, 5, 6)
    figure, axes = plot_floating_island_router_map(map_result)
    assert axes.shape == (2, 3)
    figure.clear()


def test_weak_qubit_router_coupling_converges_sw5_and_resonant_map_stays_finite():
    weak_yaml = (
        _yaml()
        .replace("EC13=15.52fF", "EC13=0.1fF")
        .replace("EC24=15.55fF", "EC24=0.1fF")
    )
    weak_spec = build_flagship_floating_island_spec(
        weak_yaml, qubit_truncated_dim=3, router_truncated_dim=8,
        qubit_ncut=5, router_ncut=5,
    )
    weak = FloatingIslandRouterModel(weak_spec, flagship_floating_island_reduction_spec())
    weak_frame = run_floating_island_router_methods(
        weak,
        FloatingIslandAnalysisConfig(
            coordinate_min=0.0, coordinate_max=0.02, points=2,
            paths=("common",), exact_evals_count=48, spectral_boundary_margin=1,
        ),
    )["common"].frame
    error_2 = np.abs(weak_frame["sw_zz_order_2_khz"] - weak_frame["exact_zz_khz"])
    error_5 = np.abs(weak_frame["sw_zz_order_5_khz"] - weak_frame["exact_zz_khz"])
    assert np.all(weak_frame["sw_valid"])
    assert np.all(error_5 <= np.maximum(1.0, 0.1 * np.abs(weak_frame["exact_zz_khz"])))
    assert np.mean(error_5) < np.mean(error_2)

    resonant = run_floating_island_router_methods(
        _fast_model(),
        FloatingIslandAnalysisConfig(
            coordinate_min=0.0, coordinate_max=2.0, points=5,
            paths=("common",), exact_evals_count=54,
            exact_overlap_floor=0.1, spectral_boundary_margin=0,
        ),
    )["common"].frame
    assert np.all(np.isfinite(resonant["exact_zz_khz"]))
    assert set(resonant["exact_status"]) <= {"reliable", "borderline", "ambiguous"}


def test_fors_pointwise_assignment_is_order_independent_and_matches_direct_point():
    model = _fast_model()
    coordinates = ((0.0, 0.0), (0.2, 0.2), (0.4, 0.4))
    points = [model.build_point(left, right) for left, right in coordinates]
    config = FloatingIslandAnalysisConfig(
        coordinate_min=0.0,
        coordinate_max=0.4,
        points=3,
        paths=("common",),
    )
    forward = _run_exact_pointwise(points, config)
    for permutation in (np.asarray((2, 1, 0)), np.asarray((2, 0, 1))):
        shuffled = _run_exact_pointwise([points[index] for index in permutation], config)
        inverse = np.argsort(permutation)
        for key in (
            "assignment_indices",
            "selected_indices",
            "eigenvalues_from_ground_ghz",
            "zz_ghz",
            "j_ghz",
        ):
            assert np.allclose(
                forward[key], shuffled[key][inverse], rtol=0.0, atol=1.0e-12
            )

    repeated = _run_exact_pointwise([points[1], points[1]], config)
    for key in (
        "assignment_indices",
        "selected_indices",
        "eigenvalues_from_ground_ghz",
        "zz_ghz",
        "j_ghz",
    ):
        assert np.allclose(repeated[key][0], repeated[key][1], rtol=0.0, atol=1.0e-12)

    direct = _fors_pointwise_exact_point(
        points[1], reliable_floor=0.75, ambiguity_floor=0.5
    )
    assert np.array_equal(direct["assignment_indices"], forward["assignment_indices"][1])
    assert np.isclose(direct["zz_ghz"], forward["zz_ghz"][1], rtol=0.0, atol=1.0e-12)
    assert np.isclose(direct["j_ghz"], forward["j_ghz"][1], rtol=0.0, atol=1.0e-12)
    eigenvalues = scipy.linalg.eigvalsh(points[1].full_hamiltonian_ghz, check_finite=False)
    assert np.allclose(
        direct["eigenvalues_from_ground_ghz"],
        eigenvalues - eigenvalues[0],
        rtol=0.0,
        atol=1.0e-12,
    )
    assert np.asarray(direct["one_excitation_bare_states"]).shape == (4, 3)
    assert np.asarray(direct["two_excitation_bare_states"]).shape == (10, 3)


def test_fors_well_separated_point_uses_shared_gs_computational_branches():
    point = _fast_model().build_point(0.0, 0.0)
    _, eigenvectors = scipy.linalg.eigh(point.full_hamiltonian_ghz, check_finite=False)
    similarity = np.abs(eigenvectors)
    assignment = gale_shapley_assignment(similarity)
    direct = _fors_pointwise_exact_point(
        point, reliable_floor=0.75, ambiguity_floor=0.5
    )
    targets = np.asarray([
        np.ravel_multi_index(state, point.dimensions)
        for state in ((0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0))
    ])
    assert np.array_equal(direct["selected_indices"], assignment[targets])
