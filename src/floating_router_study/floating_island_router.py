"""Reduced and full-reference analysis for the 9-node floating-island router."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap, TwoSlopeNorm
import numpy as np
import pandas as pd
import scipy.linalg
import scqubits as scq

from floating_router_study.physical import (
    CoordinateReductionSpec,
    PhysicalCircuitSpec,
    SQUIDSpec,
    reduce_capacitance_to_active_coordinates,
)
from remote_circuit.floating_transmon import (
    diagonalize_floating_transmon,
    effective_squid_ej_ghz,
)
from remote_circuit.matching import gale_shapley_match
from remote_circuit.methods.exact import lowdin_effective_hamiltonian
from remote_circuit.schrieffer_wolff import (
    compute_schrieffer_wolff_energy_series,
    compute_schrieffer_wolff_exchange_second_order,
)


FLOATING_ROUTER_PATHS = ("common", "single_s1", "single_s2")
FORS_EXACT_ASSIGNMENT = "pointwise dressed-proposer Gale-Shapley"
FLOATING_ONE_EXCITATION_STATES = (
    (1, 0, 0),
    (0, 1, 0),
    (0, 0, 1),
    (0, 0, 2),
)
FLOATING_TWO_EXCITATION_STATES = (
    (2, 0, 0),
    (1, 1, 0),
    (0, 2, 0),
    (1, 0, 1),
    (1, 0, 2),
    (0, 1, 1),
    (0, 1, 2),
    (0, 0, 3),
    (0, 0, 4),
    (0, 0, 5),
)


@dataclass(frozen=True)
class FloatingIslandPathResult:
    """Typed result for one floating-island flux path."""

    frame: pd.DataFrame
    exact: Mapping[str, np.ndarray]
    schrieffer_wolff: Mapping[str, np.ndarray]
    metadata: Mapping[str, object]


@dataclass(frozen=True)
class _FloatingIslandPointHamiltonian:
    effective_ej_ghz: tuple[float, float]
    dimensions: tuple[int, int, int]
    bare_energies_ghz: np.ndarray
    interaction_ghz: np.ndarray
    full_hamiltonian_ghz: np.ndarray
    router_energies_ghz: np.ndarray


@dataclass(frozen=True)
class FloatingIslandAnalysisConfig:
    """Numerical controls for the three one-dimensional switch-flux paths."""

    coordinate_min: float = 0.0
    coordinate_max: float = 2.0
    points: int = 101
    paths: tuple[str, ...] = FLOATING_ROUTER_PATHS
    idle_phi3_over_pi: float = 0.0
    idle_phi4_over_pi: float = 0.0
    maximum_order: int = 5
    coupling_ratio_max: float = 0.2
    coupling_ratio_reliable_max: float = 0.1
    denominator_floor_ghz: float | None = None
    exact_evals_count: int = 64
    exact_overlap_floor: float = 0.75
    exact_ambiguity_floor: float = 0.5
    spectral_boundary_margin: int = 2

    def __post_init__(self) -> None:
        if self.coordinate_max <= self.coordinate_min or int(self.points) < 2:
            raise ValueError("1D sweep requires an increasing range and at least two points")
        if not self.paths or len(set(self.paths)) != len(self.paths):
            raise ValueError("paths must be nonempty and unique")
        unknown = set(self.paths) - set(FLOATING_ROUTER_PATHS)
        if unknown:
            raise ValueError(f"unsupported floating-island paths: {sorted(unknown)}")
        _validate_numerics(self)


@dataclass(frozen=True)
class FloatingIslandMapConfig:
    """Numerical controls for an independent two-switch-flux map."""

    phi3_min_over_pi: float = 0.0
    phi3_max_over_pi: float = 2.0
    phi4_min_over_pi: float = 0.0
    phi4_max_over_pi: float = 2.0
    phi3_points: int = 31
    phi4_points: int = 31
    idle_phi3_over_pi: float = 0.0
    idle_phi4_over_pi: float = 0.0
    maximum_order: int = 5
    coupling_ratio_max: float = 0.2
    coupling_ratio_reliable_max: float = 0.1
    denominator_floor_ghz: float | None = None
    exact_evals_count: int = 64
    exact_overlap_floor: float = 0.75
    exact_ambiguity_floor: float = 0.5
    spectral_boundary_margin: int = 2
    mirror_reconstruction: bool = False

    def __post_init__(self) -> None:
        if self.phi3_max_over_pi <= self.phi3_min_over_pi or int(self.phi3_points) < 2:
            raise ValueError("phi3 map axis must increase and contain at least two points")
        if self.phi4_max_over_pi <= self.phi4_min_over_pi or int(self.phi4_points) < 2:
            raise ValueError("phi4 map axis must increase and contain at least two points")
        if self.mirror_reconstruction:
            raise NotImplementedError(
                "mirror reconstruction is intentionally disabled in v1; request the full grid"
            )
        _validate_numerics(self)


@dataclass(frozen=True)
class FloatingIslandMapResult:
    phi3_over_pi: np.ndarray
    phi4_over_pi: np.ndarray
    data: Mapping[str, np.ndarray]
    metadata: Mapping[str, object]


@dataclass(frozen=True)
class FloatingIslandReferencePoint:
    phi3_over_pi: float
    phi4_over_pi: float
    eigenvalues_ghz: np.ndarray
    selected_indices: np.ndarray
    branch_energies_ghz: np.ndarray
    computational_overlaps: np.ndarray
    zz_ghz: float
    j_ghz: complex
    lowdin_error: float


@dataclass(frozen=True)
class _LocalEigensystem:
    energies_ghz: np.ndarray
    charge_operator: np.ndarray
    native_vectors: np.ndarray


class FloatingIslandRouterModel:
    """Grouped q1-q2-(s1,s2) model obtained from the physical 9-node network."""

    def __init__(
        self,
        spec: PhysicalCircuitSpec,
        reduction_spec: CoordinateReductionSpec,
    ) -> None:
        self.spec = spec
        self.parsed = spec.parse()
        self.reduction_spec = reduction_spec
        self.reduced = reduce_capacitance_to_active_coordinates(self.parsed, reduction_spec)
        if tuple(self.reduced.active_coordinates) != ("q1", "q2", "s1", "s2"):
            raise ValueError("active coordinate order must be exactly (q1, q2, s1, s2)")
        by_name = {item.name: item for item in spec.subsystems}
        self.q1 = by_name[spec.computational_qubits[0]]
        self.q2 = by_name[spec.computational_qubits[1]]
        self.router = by_name[spec.tunable_subsystem]
        self._validate_topology()
        self.qubit_squids = (self.q1.squids[0], self.q2.squids[0])
        self.switch_squids = tuple(self.router.squids)
        self._qubit_local = tuple(
            diagonalize_floating_transmon(
                ej_ghz=self.effective_ej_ghz(squid, 0.0),
                ec_ghz=float(self.active_w_matrix_ghz[index, index]),
                ncut=subsystem.ncut,
                truncated_dim=subsystem.truncated_dim,
                offset_charge=subsystem.offset_charge,
            )
            for index, (squid, subsystem) in enumerate(
                zip(self.qubit_squids, (self.q1, self.q2), strict=True)
            )
        )

    @property
    def active_w_matrix_ghz(self) -> np.ndarray:
        return self.reduced.active_w_matrix_ghz

    @property
    def active_capacitance_matrix_ff(self) -> np.ndarray:
        return self.reduced.active_capacitance_matrix_ff

    @property
    def circuit_fingerprint(self) -> str:
        return self.reduced.fingerprint

    @property
    def qubit_transition_frequencies_ghz(self) -> tuple[float, float]:
        return tuple(float(local.energies_ghz[1]) for local in self._qubit_local)

    def effective_ej_ghz(self, squid: SQUIDSpec, flux_phi0: float) -> float:
        left, right = (
            float(self.parsed.symbol_values_ghz[symbol])
            for symbol in squid.josephson_symbols
        )
        return effective_squid_ej_ghz(
            left,
            right,
            flux_phi0,
            flux_offset_phi0=squid.flux_offset_phi0,
        )

    def build_point(
        self, phi3_over_pi: float, phi4_over_pi: float
    ) -> _FloatingIslandPointHamiltonian:
        """Build the grouped Hamiltonian at CZ coordinates x=phi3/pi, y=phi4/pi."""

        fluxes_phi0 = (0.5 * float(phi3_over_pi), 0.5 * float(phi4_over_pi))
        effective_ej = tuple(
            self.effective_ej_ghz(squid, flux)
            for squid, flux in zip(self.switch_squids, fluxes_phi0, strict=True)
        )
        router_local = self._build_router(effective_ej)
        local = (*self._qubit_local, router_local)
        dimensions = tuple(len(item.energies_ghz) for item in local)
        bare = (
            local[0].energies_ghz[:, None, None]
            + local[1].energies_ghz[None, :, None]
            + local[2].energies_ghz[None, None, :]
        ).reshape(-1)
        interaction = np.zeros((len(bare), len(bare)), dtype=complex)
        charge_by_coordinate = {
            0: (0, local[0].charge_operator),
            1: (1, local[1].charge_operator),
            2: (2, local[2].charge_operator[0]),
            3: (2, local[2].charge_operator[1]),
        }
        for left in range(4):
            for right in range(left + 1, 4):
                left_group, left_operator = charge_by_coordinate[left]
                right_group, right_operator = charge_by_coordinate[right]
                if left_group == right_group:
                    continue
                coefficient = float(8.0 * self.active_w_matrix_ghz[left, right])
                if coefficient != 0.0:
                    interaction += coefficient * _tensor_product_operator(
                        dimensions,
                        {left_group: left_operator, right_group: right_operator},
                    )
        interaction = 0.5 * (interaction + interaction.conj().T)
        interaction[np.diag_indices_from(interaction)] = 0.0
        full = np.diag(bare) + interaction
        return _FloatingIslandPointHamiltonian(
            effective_ej_ghz=(float(effective_ej[0]), float(effective_ej[1])),
            dimensions=dimensions,
            bare_energies_ghz=np.asarray(bare, dtype=float),
            interaction_ghz=interaction,
            full_hamiltonian_ghz=full,
            router_energies_ghz=router_local.energies_ghz,
        )

    def _validate_topology(self) -> None:
        if len(self.spec.subsystems) != 3:
            raise ValueError("floating-island backend requires Q1, Q2, and one router")
        if len(self.q1.squids) != 1 or len(self.q2.squids) != 1:
            raise ValueError("each floating qubit requires one two-junction SQUID specification")
        if len(self.router.squids) != 2:
            raise ValueError("the router requires two switch SQUID specifications")
        if min(self.q1.truncated_dim, self.q2.truncated_dim, self.router.truncated_dim) < 3:
            raise ValueError("all local truncated dimensions must be at least three")
        if any(abs(item.offset_charge) > 0.0 for item in (self.q1, self.q2, self.router)):
            raise ValueError("floating-island v1 requires the zero fixed-charge sector")

    @staticmethod
    def _charge_operators(ncut: int) -> tuple[np.ndarray, np.ndarray]:
        if int(ncut) < 2:
            raise ValueError("periodic charge cutoff must be at least two")
        charges = np.arange(-int(ncut), int(ncut) + 1, dtype=float)
        number = np.diag(charges)
        cosine = np.zeros((len(charges), len(charges)), dtype=float)
        cosine += 0.5 * np.diag(np.ones(len(charges) - 1), 1)
        cosine += 0.5 * np.diag(np.ones(len(charges) - 1), -1)
        return number, cosine

    def _build_router(self, effective_ej_ghz: tuple[float, float]) -> _LocalEigensystem:
        number, cosine = self._charge_operators(self.router.ncut)
        identity = np.eye(len(number))
        n1 = np.kron(number, identity)
        n2 = np.kron(identity, number)
        cos1 = np.kron(cosine, identity)
        cos2 = np.kron(identity, cosine)
        w = self.active_w_matrix_ghz[2:4, 2:4]
        hamiltonian = (
            4.0 * w[0, 0] * (n1 @ n1)
            + 4.0 * w[1, 1] * (n2 @ n2)
            + 8.0 * w[0, 1] * (n1 @ n2)
            - effective_ej_ghz[0] * cos1
            - effective_ej_ghz[1] * cos2
        )
        count = min(int(self.router.truncated_dim), len(hamiltonian))
        energies, vectors = scipy.linalg.eigh(
            hamiltonian, subset_by_index=(0, count - 1), check_finite=False
        )
        operators = []
        for native in (n1, n2):
            projected = vectors.conj().T @ native @ vectors
            projected[np.abs(projected) < 1.0e-14] = 0.0
            operators.append(projected)
        return _LocalEigensystem(
            energies_ghz=np.asarray(energies - energies[0], dtype=float),
            charge_operator=np.asarray(operators, dtype=complex),
            native_vectors=np.asarray(vectors, dtype=complex),
        )


class FloatingIslandScqubitsReference:
    """Full four-periodic-coordinate scqubits validation backend."""

    def __init__(
        self,
        spec: PhysicalCircuitSpec,
        reduction_spec: CoordinateReductionSpec,
        *,
        local_cutoff: int | None = None,
        subsystem_trunc_dims: tuple[int, int, int, int] | None = None,
    ) -> None:
        self.spec = spec
        self.parsed = spec.parse()
        self.reduction_spec = reduction_spec
        self.reduced = reduce_capacitance_to_active_coordinates(self.parsed, reduction_spec)
        by_name = {item.name: item for item in spec.subsystems}
        q1 = by_name[spec.computational_qubits[0]]
        q2 = by_name[spec.computational_qubits[1]]
        router = by_name[spec.tunable_subsystem]
        squids = (q1.squids[0], q2.squids[0], *router.squids)
        self.circuit = scq.Circuit(
            spec.circuit_yaml, from_file=False, ext_basis="harmonic", truncated_dim=6
        )
        closures = [self._second_junction(squid) for squid in squids]
        self.circuit.configure(
            closure_branches=closures,
            transformation_matrix=np.asarray(reduction_spec.transformation_matrix, dtype=float),
        )
        cutoff = int(local_cutoff or max(q1.ncut, q2.ncut, router.ncut))
        for name in self.circuit.cutoff_names:
            setattr(self.circuit, name, cutoff)
        periodic = sorted(self.circuit.var_categories.get("periodic", []))
        if len(periodic) != 4:
            raise ValueError(f"expected four periodic variables, found {periodic}")
        dims = subsystem_trunc_dims or (
            int(q1.truncated_dim), int(q2.truncated_dim),
            min(6, int(router.truncated_dim)), min(6, int(router.truncated_dim)),
        )
        self.subsystem_trunc_dims = tuple(int(value) for value in dims)
        self.circuit.configure(
            system_hierarchy=[[variable] for variable in periodic],
            subsystem_trunc_dims=list(self.subsystem_trunc_dims),
        )
        self.flux_names = sorted(
            [str(symbol) for symbol in self.circuit.external_fluxes],
            key=lambda name: int("".join(character for character in name if character.isdigit()) or 0),
        )
        if len(self.flux_names) != 4:
            raise ValueError(f"expected four external fluxes, found {self.flux_names}")

    def evaluate_point(
        self, phi3_over_pi: float, phi4_over_pi: float, *, evals_count: int = 64
    ) -> FloatingIslandReferencePoint:
        params = dict(zip(
            self.flux_names,
            (0.0, 0.0, 0.5 * float(phi3_over_pi), 0.5 * float(phi4_over_pi)),
            strict=True,
        ))
        self.circuit.set_params(**params)
        dimension = int(np.prod(self.subsystem_trunc_dims))
        full_hamiltonian = self.circuit.hamiltonian()
        if hasattr(full_hamiltonian, "full"):
            full_hamiltonian = full_hamiltonian.full()
        elif hasattr(full_hamiltonian, "toarray"):
            full_hamiltonian = full_hamiltonian.toarray()
        full_hamiltonian = np.asarray(full_hamiltonian, dtype=complex)
        eigenvalues, vectors = scipy.linalg.eigh(
            full_hamiltonian, check_finite=False, driver="evd"
        )
        if vectors.shape != (dimension, dimension):
            raise RuntimeError("full scqubits reference requires the retained eigensystem")
        matching = gale_shapley_match(np.abs(vectors))
        targets = np.asarray([
            np.ravel_multi_index(state, self.subsystem_trunc_dims)
            for state in ((0, 0, 0, 0), (1, 0, 0, 0), (0, 1, 0, 0), (1, 1, 0, 0))
        ])
        selected = matching.receiver_to_proposer[targets]
        energies = np.asarray(eigenvalues[selected], dtype=float)
        energies -= energies[0]
        overlap = vectors[np.ix_(targets, selected)]
        try:
            h_eff, error = lowdin_effective_hamiltonian(overlap, energies)
            exchange = complex(h_eff[1, 2])
        except ValueError:
            error = np.nan
            exchange = complex(np.nan, np.nan)
        return FloatingIslandReferencePoint(
            phi3_over_pi=float(phi3_over_pi),
            phi4_over_pi=float(phi4_over_pi),
            eigenvalues_ghz=np.asarray(eigenvalues, dtype=float),
            selected_indices=selected,
            branch_energies_ghz=energies,
            computational_overlaps=np.abs(np.diag(overlap)) ** 2,
            zz_ghz=float(energies[3] + energies[0] - energies[1] - energies[2]),
            j_ghz=exchange,
            lowdin_error=float(error),
        )

    def _second_junction(self, squid: SQUIDSpec):
        matches = []
        endpoints = {int(squid.node), int(squid.other_node)}
        for branch in self.circuit.branches:
            node_ids = branch.node_ids() if callable(branch.node_ids) else branch.node_ids
            if branch.type == "JJ" and set(node_ids) == endpoints:
                matches.append(branch)
        matches.sort(key=lambda branch: branch.index)
        if len(matches) != 2:
            raise ValueError(f"expected two junctions on SQUID branch {sorted(endpoints)}")
        return matches[1]


def _fors_pointwise_exact_point(
    point: _FloatingIslandPointHamiltonian,
    *,
    reliable_floor: float,
    ambiguity_floor: float,
) -> dict[str, object]:
    """Assign every retained bare state using the Fors pointwise prescription."""

    dimension = len(point.bare_energies_ghz)
    eigenvalues, eigenvectors = scipy.linalg.eigh(
        point.full_hamiltonian_ghz,
        check_finite=False,
        driver="evd",
    )
    if eigenvectors.shape != (dimension, dimension):
        raise RuntimeError("Fors assignment requires the full retained eigensystem")

    similarity = np.abs(eigenvectors)
    matching = gale_shapley_match(similarity)
    assignment = matching.receiver_to_proposer
    bare_indices = np.arange(dimension)
    if not matching.bijective:
        raise RuntimeError("Fors pointwise assignment is not bijective")

    assigned_probability = matching.assigned_probability
    assigned_overlap = similarity[bare_indices, assignment]

    targets = _computational_target_indices(point.dimensions)
    selected = assignment[targets]
    branch_energies = np.asarray(eigenvalues[selected] - eigenvalues[selected[0]], dtype=float)
    coefficient = eigenvectors[np.ix_(targets, selected)]
    branch_overlaps = assigned_probability[targets]
    one_overlap = coefficient[np.ix_((1, 2), (1, 2))]
    singular_values = np.linalg.svd(one_overlap, compute_uv=False)
    one_subspace_overlap = float(np.min(singular_values) ** 2)
    principal_angles = np.arccos(np.clip(singular_values, 0.0, 1.0))
    zz_ghz = float(
        branch_energies[3]
        + branch_energies[0]
        - branch_energies[1]
        - branch_energies[2]
    )
    try:
        h_eff, lowdin_error = lowdin_effective_hamiltonian(
            coefficient, branch_energies
        )
        exchange_ghz = complex(h_eff[1, 2])
    except ValueError:
        lowdin_error = np.nan
        exchange_ghz = complex(np.nan, np.nan)

    minimum_overlap = float(np.min(branch_overlaps))
    finite = bool(
        np.isfinite(zz_ghz)
        and np.isfinite(exchange_ghz)
        and np.isfinite(lowdin_error)
        and np.all(np.isfinite(branch_overlaps))
        and np.isfinite(one_subspace_overlap)
    )
    ambiguous = bool(
        not finite
        or minimum_overlap < float(ambiguity_floor)
        or one_subspace_overlap < float(ambiguity_floor)
    )
    reliable = bool(
        not ambiguous
        and minimum_overlap >= float(reliable_floor)
        and one_subspace_overlap >= float(reliable_floor)
    )
    status = "ambiguous" if ambiguous else "reliable" if reliable else "borderline"
    one_states = np.asarray(FLOATING_ONE_EXCITATION_STATES, dtype=int)
    two_states = np.asarray(FLOATING_TWO_EXCITATION_STATES, dtype=int)
    if np.any(one_states >= np.asarray(point.dimensions)) or np.any(
        two_states >= np.asarray(point.dimensions)
    ):
        raise ValueError(
            "Fors dressed-manifold output requires qubit dimensions >= 3 "
            "and router dimension >= 6"
        )
    return {
        "eigenvalues_from_ground_ghz": np.asarray(
            eigenvalues - eigenvalues[0], dtype=float
        ),
        "retained_dimensions": np.asarray(point.dimensions, dtype=int),
        "one_excitation_bare_states": one_states,
        "two_excitation_bare_states": two_states,
        "one_excitation_bare_indices": np.ravel_multi_index(
            one_states.T, point.dimensions
        ),
        "two_excitation_bare_indices": np.ravel_multi_index(
            two_states.T, point.dimensions
        ),
        "assignment_indices": assignment,
        "bare_overlap": assigned_overlap,
        "bare_overlap_probability": assigned_probability,
        "assignment_row_margin": matching.row_margin,
        "assignment_column_margin": matching.column_margin,
        "assignment_confidence_margin": matching.confidence_margin,
        "assignment_bijective": matching.bijective,
        "branch_energies_ghz": branch_energies,
        "selected_indices": selected,
        "branch_overlaps": branch_overlaps,
        "computational_overlap": float(np.sum(np.abs(coefficient) ** 2) / 4.0),
        "one_excitation_subspace_overlap": one_subspace_overlap,
        "one_excitation_principal_angles_rad": principal_angles,
        "spectral_boundary_margin": int(dimension - 1 - np.max(selected)),
        "lowdin_error": float(lowdin_error),
        "zz_ghz": zz_ghz,
        "j_ghz": exchange_ghz,
        "minimum_overlap": minimum_overlap,
        "reliable": reliable,
        "borderline": bool(not ambiguous and not reliable),
        "ambiguous": ambiguous,
        "status": status,
        "valid": bool(not ambiguous),
    }


def _run_exact_pointwise(
    points: Sequence[_FloatingIslandPointHamiltonian],
    config: FloatingIslandAnalysisConfig | FloatingIslandMapConfig,
) -> dict[str, np.ndarray]:
    """Run full-basis dressed-proposer Gale-Shapley at independent points."""

    if not points:
        raise ValueError("at least one floating-island point is required")
    ambiguity_floor = min(
        float(config.exact_ambiguity_floor),
        float(config.exact_overlap_floor),
    )
    rows = [
        _fors_pointwise_exact_point(
            point,
            reliable_floor=float(config.exact_overlap_floor),
            ambiguity_floor=ambiguity_floor,
        )
        for point in points
    ]
    result = {
        key: np.asarray([row[key] for row in rows])
        for key in rows[0]
    }
    result["branch_labels"] = np.asarray(("00", "10", "01", "11"), dtype="S")
    return result


def run_floating_island_router_methods(
    model: FloatingIslandRouterModel,
    config: FloatingIslandAnalysisConfig | None = None,
) -> dict[str, FloatingIslandPathResult]:
    """Run grouped Exact/SW calculations on common and single-switch paths."""

    config = config or FloatingIslandAnalysisConfig()
    coordinate = np.linspace(config.coordinate_min, config.coordinate_max, int(config.points))
    results: dict[str, FloatingIslandPathResult] = {}
    for path in config.paths:
        phi3, phi4 = _floating_path_fluxes(path, coordinate, config)
        points = [
            model.build_point(left, right)
            for left, right in zip(phi3, phi4, strict=True)
        ]
        exact = _run_exact_pointwise(points, config)
        sw = _run_floating_schrieffer_wolff(points, config)
        frame = _floating_frame(model, coordinate, phi3, phi4, points, exact, sw, config)
        results[path] = FloatingIslandPathResult(
            frame=frame,
            exact=exact,
            schrieffer_wolff=sw,
            metadata={
                "path": path,
                "exact_reference_assignment": FORS_EXACT_ASSIGNMENT,
                "assignment_scope": "full retained basis",
                "circuit_fingerprint": model.circuit_fingerprint,
                "physical_fingerprint": model.reduced.physical_fingerprint,
                "active_coordinates": model.reduced.active_coordinates,
                "active_capacitance_matrix_ff": np.array(
                    model.active_capacitance_matrix_ff, copy=True
                ),
                "active_w_matrix_ghz": np.array(model.active_w_matrix_ghz, copy=True),
                "flux_convention": "Phi_scqubits=(phi_ext/pi)/2",
                "config": asdict(config),
            },
        )
    return results


def run_floating_island_router_map(
    model: FloatingIslandRouterModel,
    config: FloatingIslandMapConfig | None = None,
) -> FloatingIslandMapResult:
    """Compute a full independent-switch map with pointwise GS assignment."""

    config = config or FloatingIslandMapConfig()
    x = np.linspace(config.phi3_min_over_pi, config.phi3_max_over_pi, int(config.phi3_points))
    y = np.linspace(config.phi4_min_over_pi, config.phi4_max_over_pi, int(config.phi4_points))
    points_grid = [[model.build_point(x_value, y_value) for x_value in x] for y_value in y]
    flat_points = [point for row in points_grid for point in row]
    pointwise = _run_exact_pointwise(flat_points, config)
    shape = (len(y), len(x))
    exact = {}
    for key, value in pointwise.items():
        array = np.asarray(value)
        if key == "branch_labels":
            exact["exact_branch_labels"] = array
        else:
            exact[f"exact_{key}"] = array.reshape(
                (shape[0], shape[1], *array.shape[1:])
            )
    sw_flat = _run_floating_schrieffer_wolff(flat_points, config)
    shape = (len(y), len(x))
    sw = {key: np.asarray(value).reshape((shape[0], shape[1], *np.asarray(value).shape[1:]))
          for key, value in sw_flat.items()}
    cumulative = np.asarray(sw["zz_cumulative_by_order"])
    order = int(config.maximum_order)
    data: dict[str, np.ndarray] = {
        **exact,
        "router_transition_1_ghz": np.asarray([
            [point.router_energies_ghz[1] for point in row] for row in points_grid
        ]),
        "router_transition_2_ghz": np.asarray([
            [point.router_energies_ghz[2] for point in row] for row in points_grid
        ]),
        "sw_zz_ghz": cumulative[..., order],
        "sw_j_ghz": np.asarray(sw["exchange_j_through_second_order"]),
        "sw_valid": np.asarray(sw["valid"], dtype=bool),
        "sw_reliable": np.asarray(sw["reliable"], dtype=bool),
        "sw_borderline": np.asarray(sw["borderline"], dtype=bool),
        "sw_status": np.asarray(sw["status"], dtype=str),
        "sw_minimum_denominator_ghz": np.asarray(sw["minimum_denominator"]),
        "sw_maximum_coupling_ratio": np.asarray(sw["maximum_coupling_ratio"]),
        "sw_denominator_singular": np.asarray(sw["denominator_singular"], dtype=bool),
        "sw_global_minimum_denominator_ghz": np.asarray(
            sw["global_minimum_denominator"]
        ),
        "sw_global_maximum_coupling_ratio": np.asarray(
            sw["global_maximum_coupling_ratio"]
        ),
        "sw_global_denominator_singular": np.asarray(
            sw["global_denominator_singular"], dtype=bool
        ),
        "sw_zz_cumulative_by_order_ghz": cumulative,
    }
    data["zz_error_ghz"] = data["sw_zz_ghz"] - data["exact_zz_ghz"]
    data["j_error_ghz"] = data["sw_j_ghz"] - data["exact_j_ghz"]
    return FloatingIslandMapResult(
        phi3_over_pi=_readonly(x),
        phi4_over_pi=_readonly(y),
        data={key: _readonly(value) for key, value in data.items()},
        metadata={
            "exact_reference_assignment": FORS_EXACT_ASSIGNMENT,
            "assignment_scope": "full retained basis",
            "circuit_fingerprint": model.circuit_fingerprint,
            "physical_fingerprint": model.reduced.physical_fingerprint,
            "active_coordinates": model.reduced.active_coordinates,
            "active_capacitance_matrix_ff": np.array(
                model.active_capacitance_matrix_ff, copy=True
            ),
            "active_w_matrix_ghz": np.array(model.active_w_matrix_ghz, copy=True),
            "flux_convention": "Phi_scqubits=(phi_ext/pi)/2",
            "config": asdict(config),
        },
    )


def plot_floating_island_router_map(
    result: FloatingIslandMapResult,
) -> tuple[plt.Figure, np.ndarray]:
    """Plot router spectrum, Exact/SW interactions, errors, and validity."""

    x, y = np.meshgrid(result.phi3_over_pi, result.phi4_over_pi, indexing="xy")
    data = result.data
    figure, axes = plt.subplots(2, 3, figsize=(15, 8.5), constrained_layout=True)
    panels = (
        ("router_transition_1_ghz", "Router 0→1 (GHz)", None, 1.0, None),
        ("exact_zz_ghz", "Exact ZZ (MHz)", "diverging", 1.0e3, "exact"),
        ("sw_zz_ghz", "SW5 ZZ (MHz)", "diverging", 1.0e3, "sw"),
        ("zz_error_ghz", "SW5 − Exact ZZ (kHz)", "diverging", 1.0e6, "both"),
        ("exact_j_ghz", "Exact |J| (MHz)", None, 1.0e3, "exact"),
    )
    exact_ok = np.asarray(data["exact_valid"], dtype=bool)
    sw_ok = np.asarray(data["sw_valid"], dtype=bool)
    masks = {"exact": exact_ok, "sw": sw_ok, "both": exact_ok & sw_ok}
    for axis, (key, title, scale_kind, scale, validity) in zip(
        axes.flat[:5], panels, strict=True
    ):
        values = np.asarray(data[key])
        if np.iscomplexobj(values):
            values = np.abs(values)
        values = values * scale
        if validity is not None:
            values = np.ma.masked_where(~masks[validity], values)
        norm = None
        if scale_kind == "diverging":
            finite = np.abs(np.asarray(values.compressed() if np.ma.isMaskedArray(values) else values[np.isfinite(values)]))
            limit = float(np.max(finite, initial=1.0))
            norm = TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)
        cmap = plt.get_cmap("coolwarm" if norm else "viridis").copy()
        cmap.set_bad("0.85")
        mesh = axis.pcolormesh(x, y, values, shading="auto", cmap=cmap, norm=norm)
        figure.colorbar(mesh, ax=axis)
        axis.set_title(title)
    status = exact_ok.astype(int) + 2 * sw_ok.astype(int)
    status_cmap = ListedColormap(("#bdbdbd", "#2c7fb8", "#fdae61", "#31a354"))
    status_norm = BoundaryNorm((-0.5, 0.5, 1.5, 2.5, 3.5), status_cmap.N)
    mesh = axes[1, 2].pcolormesh(
        x, y, status, shading="auto", cmap=status_cmap, norm=status_norm
    )
    colorbar = figure.colorbar(mesh, ax=axes[1, 2], ticks=(0, 1, 2, 3))
    colorbar.ax.set_yticklabels(("neither", "Exact", "SW", "both"))
    axes[1, 2].set_title("Validity domains")
    for axis in axes.flat:
        axis.set_xlabel(r"$\phi_3/\pi$")
        axis.set_ylabel(r"$\phi_4/\pi$")
    return figure, axes


def _run_floating_schrieffer_wolff(
    points: Sequence[_FloatingIslandPointHamiltonian],
    config: FloatingIslandAnalysisConfig | FloatingIslandMapConfig,
) -> dict[str, np.ndarray]:
    rows: list[dict[str, object]] = []
    for point in points:
        targets = _computational_target_indices(point.dimensions)
        series = compute_schrieffer_wolff_energy_series(
            point.bare_energies_ghz,
            point.interaction_ghz,
            targets,
            maximum_order=int(config.maximum_order),
            coupling_ratio_max=float(config.coupling_ratio_max),
            denominator_floor=config.denominator_floor_ghz,
        )
        exchange = compute_schrieffer_wolff_exchange_second_order(
            point.bare_energies_ghz,
            point.interaction_ghz,
            int(targets[1]),
            int(targets[2]),
            denominator_floor=config.denominator_floor_ghz,
        )
        global_ratio, global_minimum, global_singular = _global_sw_diagnostics(
            point.bare_energies_ghz,
            point.interaction_ghz,
            config.denominator_floor_ghz,
        )
        valid = bool(series["valid"] and exchange["valid"])
        reliable = bool(
            valid
            and float(series["maximum_coupling_ratio"])
            < float(config.coupling_ratio_reliable_max)
        )
        rows.append({
            **series,
            **{f"exchange_{key}": value for key, value in exchange.items()},
            "global_maximum_coupling_ratio": global_ratio,
            "global_minimum_denominator": global_minimum,
            "global_denominator_singular": global_singular,
            "valid": valid,
            "reliable": reliable,
            "borderline": bool(valid and not reliable),
            "status": "reliable" if reliable else "borderline" if valid else "invalid",
        })
    keys = rows[0].keys()
    return {key: np.asarray([row[key] for row in rows]) for key in keys}


def _computational_target_indices(dimensions: tuple[int, int, int]) -> np.ndarray:
    states = ((0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0))
    return np.asarray(
        [np.ravel_multi_index(state, dimensions) for state in states], dtype=int
    )


def _tensor_product_operator(
    dimensions: tuple[int, ...], operators: Mapping[int, np.ndarray]
) -> np.ndarray:
    result = np.asarray([[1.0 + 0.0j]])
    for index, dimension in enumerate(dimensions):
        operator = operators.get(index, np.eye(dimension, dtype=complex))
        result = np.kron(result, np.asarray(operator, dtype=complex))
    return result


def _global_sw_diagnostics(
    energies: np.ndarray,
    interaction: np.ndarray,
    denominator_floor: float | None,
) -> tuple[float, float, bool]:
    scale = max(float(np.max(np.abs(energies))), 1.0)
    floor = 1.0e-12 * scale if denominator_floor is None else float(denominator_floor)
    rows, columns = np.triu_indices(len(energies), k=1)
    amplitudes = np.abs(interaction[rows, columns])
    active = amplitudes > 1.0e-11
    if not np.any(active):
        return 0.0, np.inf, False
    denominators = np.abs(energies[rows[active]] - energies[columns[active]])
    singular = bool(np.any(denominators <= floor))
    allowed = denominators > floor
    maximum = (
        float(np.max(amplitudes[active][allowed] / denominators[allowed]))
        if np.any(allowed)
        else np.inf
    )
    return maximum, float(np.min(denominators)), singular


def _floating_frame(
    model: FloatingIslandRouterModel,
    coordinate: np.ndarray,
    phi3: np.ndarray,
    phi4: np.ndarray,
    points: Sequence[_FloatingIslandPointHamiltonian],
    exact: Mapping[str, np.ndarray],
    sw: Mapping[str, np.ndarray],
    config: FloatingIslandAnalysisConfig,
) -> pd.DataFrame:
    frame = pd.DataFrame({
        "coordinate_phi_over_pi": coordinate,
        "phi3_over_pi": phi3,
        "phi4_over_pi": phi4,
        "flux_phi3_scqubits": 0.5 * phi3,
        "flux_phi4_scqubits": 0.5 * phi4,
        "effective_ej_s1_ghz": [point.effective_ej_ghz[0] for point in points],
        "effective_ej_s2_ghz": [point.effective_ej_ghz[1] for point in points],
        "qubit_1_transition_ghz": model.qubit_transition_frequencies_ghz[0],
        "qubit_2_transition_ghz": model.qubit_transition_frequencies_ghz[1],
        "router_transition_1_ghz": [point.router_energies_ghz[1] for point in points],
        "router_transition_2_ghz": [point.router_energies_ghz[2] for point in points],
        "exact_zz_khz": np.asarray(exact["zz_ghz"]) * 1.0e6,
        "exact_j_mhz": np.asarray(exact["j_ghz"]) * 1.0e3,
        "exact_abs_j_mhz": np.abs(np.asarray(exact["j_ghz"])) * 1.0e3,
        "exact_valid": np.asarray(exact["valid"], dtype=bool),
        "exact_reliable": np.asarray(
            exact.get("reliable", exact["valid"]), dtype=bool
        ),
        "exact_borderline": np.asarray(
            exact.get("borderline", np.zeros(len(coordinate), dtype=bool)),
            dtype=bool,
        ),
        "exact_ambiguous": np.asarray(
            exact.get("ambiguous", ~np.asarray(exact["valid"], dtype=bool)),
            dtype=bool,
        ),
        "exact_status": np.asarray(
            exact.get(
                "status",
                np.where(np.asarray(exact["valid"], dtype=bool), "reliable", "ambiguous"),
            ),
            dtype=str,
        ),
        "exact_minimum_overlap": np.asarray(
            exact.get(
                "minimum_overlap",
                np.min(np.asarray(exact["branch_overlaps"]), axis=1),
            )
        ),
        "exact_one_excitation_subspace_overlap": np.asarray(
            exact["one_excitation_subspace_overlap"]
        ),
        "exact_lowdin_error": np.asarray(exact["lowdin_error"]),
        "sw_zz_total_khz": np.asarray(sw["zz_total"]) * 1.0e6,
        "sw_j_mhz": np.asarray(sw["exchange_j_through_second_order"]) * 1.0e3,
        "sw_abs_j_mhz": np.abs(np.asarray(sw["exchange_j_through_second_order"])) * 1.0e3,
        "sw_maximum_coupling_ratio": np.asarray(sw["maximum_coupling_ratio"]),
        "sw_minimum_denominator_ghz": np.asarray(sw["minimum_denominator"]),
        "sw_denominator_singular": np.asarray(sw["denominator_singular"], dtype=bool),
        "sw_global_maximum_coupling_ratio": np.asarray(
            sw["global_maximum_coupling_ratio"]
        ),
        "sw_global_minimum_denominator_ghz": np.asarray(
            sw["global_minimum_denominator"]
        ),
        "sw_global_denominator_singular": np.asarray(
            sw["global_denominator_singular"], dtype=bool
        ),
        "sw_valid": np.asarray(sw["valid"], dtype=bool),
        "sw_reliable": np.asarray(sw["reliable"], dtype=bool),
        "sw_borderline": np.asarray(sw["borderline"], dtype=bool),
        "sw_status": np.asarray(sw["status"], dtype=str),
    })
    cumulative = np.asarray(sw["zz_cumulative_by_order"])
    for order in range(2, int(config.maximum_order) + 1):
        frame[f"sw_zz_order_{order}_khz"] = cumulative[:, order] * 1.0e6
    return frame


def _floating_path_fluxes(
    path: str,
    coordinate: np.ndarray,
    config: FloatingIslandAnalysisConfig,
) -> tuple[np.ndarray, np.ndarray]:
    if path == "common":
        return coordinate.copy(), coordinate.copy()
    if path == "single_s1":
        return coordinate.copy(), np.full_like(coordinate, config.idle_phi4_over_pi)
    if path == "single_s2":
        return np.full_like(coordinate, config.idle_phi3_over_pi), coordinate.copy()
    raise ValueError(f"unsupported floating-island path: {path}")


def _validate_numerics(config) -> None:
    if not 2 <= int(config.maximum_order) <= 8:
        raise ValueError("maximum_order must be between 2 and 8")
    if config.coupling_ratio_max <= 0.0:
        raise ValueError("coupling_ratio_max must be positive")
    if not 0.0 < config.coupling_ratio_reliable_max <= config.coupling_ratio_max:
        raise ValueError(
            "coupling_ratio_reliable_max must lie in (0, coupling_ratio_max]"
        )
    if config.denominator_floor_ghz is not None and config.denominator_floor_ghz <= 0.0:
        raise ValueError("denominator_floor_ghz must be positive")
    if not 0.0 < config.exact_overlap_floor <= 1.0:
        raise ValueError("exact_overlap_floor must lie in (0, 1]")
    if not 0.0 < config.exact_ambiguity_floor <= 1.0:
        raise ValueError("exact_ambiguity_floor must lie in (0, 1]")
    if int(config.spectral_boundary_margin) < 0:
        raise ValueError("spectral_boundary_margin must be nonnegative")


def _readonly(value) -> np.ndarray:
    array = np.array(value, copy=True)
    array.setflags(write=False)
    return array
