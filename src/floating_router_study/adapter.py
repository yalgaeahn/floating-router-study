"""Static experiment adapter for the floating-island router model."""

from __future__ import annotations

from dataclasses import fields
from typing import Any, Mapping

import numpy as np

from remote_circuit.experiment import ExperimentSpec
from remote_circuit.workflows.contracts import StaticModelAdapter, StaticRunPayload
from remote_circuit.options import _backend_options, _selected


def _subsystem(raw: Mapping[str, Any], defaults: Mapping[str, Any] | None = None):
    from floating_router_study.physical import SQUIDSpec, SubsystemSpec

    allowed = {item.name for item in fields(SubsystemSpec)}
    unknown = set(raw) - allowed
    if unknown:
        raise ValueError(f"circuit.subsystems: unknown field(s): {sorted(unknown)}")
    value = {**dict(defaults or {}), **dict(raw)}
    for key in ("nodes", "josephson_symbols", "external_flux_parameters"):
        if key in value:
            value[key] = tuple(value[key])
    squids = []
    for squid in value.pop("squids", ()):
        squid_value = dict(squid)
        squid_value["josephson_symbols"] = tuple(squid_value["josephson_symbols"])
        squids.append(SQUIDSpec(**squid_value))
    value["squids"] = tuple(squids)
    return SubsystemSpec(**value)


def _subsystem_defaults(
    spec: ExperimentSpec,
    raw: Mapping[str, Any],
    presets: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    kind = str(raw.get("kind", ""))
    name = str(raw.get("name", ""))
    return {
        **dict(presets.get(kind, {})),
        **dict(spec.numerics.components.get(kind, {})),
        **dict(spec.numerics.components.get(name, {})),
    }


def _floating_model(spec: ExperimentSpec):
    from floating_router_study.physical import (
        CoordinateReductionSpec,
        PhysicalCircuitSpec,
        SweepSpec as PhysicalSweepSpec,
    )
    from floating_router_study.floating_island_router import FloatingIslandRouterModel

    defaults = {
        "smoke": {
            "floating_transmon": {"truncated_dim": 3, "ncut": 5},
            "floating_island_router": {"truncated_dim": 8, "ncut": 5},
        },
        "standard": {
            "floating_transmon": {"truncated_dim": 4, "ncut": 6},
            "floating_island_router": {"truncated_dim": 12, "ncut": 6},
        },
        "converged": {
            "floating_transmon": {"truncated_dim": 4, "ncut": 7},
            "floating_island_router": {"truncated_dim": 16, "ncut": 7},
        },
    }[spec.numerics.preset]
    subsystems = tuple(
        _subsystem(item, _subsystem_defaults(spec, item, defaults))
        for item in spec.circuit.topology.subsystems
    )
    parameters = spec.circuit.topology.parameters
    computational = tuple(parameters.get("computational_qubits", ()))
    if len(computational) != 2 or any(not str(value).strip() for value in computational):
        raise ValueError("circuit.parameters.computational_qubits: exactly two names are required")
    axis = spec.sweep.axes[0]
    physical = PhysicalCircuitSpec(
        circuit_yaml=spec.circuit.topology.scqubits_yaml(),
        subsystems=subsystems,
        sweep=PhysicalSweepSpec(axis.name, axis.minimum, axis.maximum, axis.points),
        computational_qubits=(str(computational[0]), str(computational[1])),
        tunable_subsystem=str(parameters.get("tunable_subsystem", "")),
    )
    reduction_raw = spec.circuit.topology.coordinate_reduction
    if reduction_raw is None:
        raise ValueError("circuit.coordinate_reduction: required for floating_island_router")
    reduction = CoordinateReductionSpec(
        coordinate_names=tuple(reduction_raw.get("coordinate_names", ())),
        transformation_matrix=np.asarray(
            reduction_raw.get("transformation_matrix", ()), dtype=float
        ),
        active_coordinates=tuple(reduction_raw.get("active_coordinates", ())),
        free_coordinates=tuple(reduction_raw.get("free_coordinates", ())),
    )
    return FloatingIslandRouterModel(physical, reduction)


def _run_floating(spec: ExperimentSpec) -> StaticRunPayload:
    from floating_router_study.floating_island_router import (
        FloatingIslandAnalysisConfig,
        FloatingIslandMapConfig,
        run_floating_island_router_map,
        run_floating_island_router_methods,
    )

    model = _floating_model(spec)
    exact = _backend_options(spec, "exact")
    options = {
        "exact_overlap_floor": exact.pop("reliable_overlap_floor", 0.75),
        "exact_ambiguity_floor": exact.pop("ambiguity_overlap_floor", 0.5),
        **_backend_options(spec, "schrieffer_wolff"),
    }
    if exact:
        raise ValueError(f"numerics.backends.exact: unsupported option(s): {sorted(exact)}")
    if len(spec.sweep.axes) == 2:
        x, y = spec.sweep.axes
        map_config = FloatingIslandMapConfig(
            phi3_min_over_pi=x.minimum,
            phi3_max_over_pi=x.maximum,
            phi3_points=x.points,
            phi4_min_over_pi=y.minimum,
            phi4_max_over_pi=y.maximum,
            phi4_points=y.points,
            **_selected(FloatingIslandMapConfig, options),
        )
        result = run_floating_island_router_map(model, map_config)
        data = dict(result.data)
        exact_map = {key: value for key, value in data.items() if key.startswith("exact_")}
        sw_map = {key: value for key, value in data.items() if key.startswith("sw_")}
        observables = {
            key: value
            for key, value in data.items()
            if key not in exact_map and key not in sw_map
        }
        methods: dict[str, Any] = {}
        if spec.methods.exact:
            methods["exact"] = exact_map
        if spec.methods.schrieffer_wolff:
            methods["schrieffer_wolff"] = sw_map
        diagnostics = {
            "validity": {
                "exact": np.asarray(data["exact_valid"], dtype=bool),
                "schrieffer_wolff": np.asarray(data["sw_valid"], dtype=bool),
            }
        }
        return StaticRunPayload(
            coordinates={x.name: result.phi3_over_pi, y.name: result.phi4_over_pi},
            observables=observables,
            methods=methods,
            diagnostics=diagnostics,
            circuit_fingerprint=model.circuit_fingerprint,
        )
    axis = spec.sweep.axes[0]
    path_config = FloatingIslandAnalysisConfig(
        coordinate_min=axis.minimum,
        coordinate_max=axis.maximum,
        points=axis.points,
        paths=spec.sweep.paths,
        idle_phi3_over_pi=float(spec.sweep.idle.get("phi3_over_pi", 0.0)),
        idle_phi4_over_pi=float(spec.sweep.idle.get("phi4_over_pi", 0.0)),
        **_selected(FloatingIslandAnalysisConfig, options),
    )
    results = run_floating_island_router_methods(model, path_config)
    coordinates = {
        path: result.frame["coordinate_phi_over_pi"].to_numpy()
        for path, result in results.items()
    }
    path_observables = {
        path: {column: result.frame[column].to_numpy() for column in result.frame}
        for path, result in results.items()
    }
    path_exact: dict[str, Any] = (
        {path: result.exact for path, result in results.items()}
        if spec.methods.exact
        else {}
    )
    path_sw: dict[str, Any] = (
        {path: result.schrieffer_wolff for path, result in results.items()}
        if spec.methods.schrieffer_wolff
        else {}
    )
    path_methods: dict[str, Any] = {
        name: value
        for name, value in (("exact", path_exact), ("schrieffer_wolff", path_sw))
        if value
    }
    validity = {
        f"{name}.{path}": np.asarray(payload["valid"], dtype=bool)
        for name, paths in path_methods.items()
        for path, payload in paths.items()
    }
    return StaticRunPayload(
        coordinates=coordinates,
        observables=path_observables,
        methods=path_methods,
        diagnostics={"validity": validity},
        circuit_fingerprint=model.circuit_fingerprint,
    )


def _check_floating(spec: ExperimentSpec) -> None:
    from floating_router_study.floating_island_router import FloatingIslandAnalysisConfig, FloatingIslandMapConfig

    _floating_model(spec)
    exact = _backend_options(spec, "exact")
    options = {
        "exact_overlap_floor": exact.pop("reliable_overlap_floor", 0.75),
        "exact_ambiguity_floor": exact.pop("ambiguity_overlap_floor", 0.5),
        **_backend_options(spec, "schrieffer_wolff"),
    }
    if exact:
        raise ValueError(f"numerics.backends.exact: unsupported option(s): {sorted(exact)}")
    if len(spec.sweep.axes) == 2:
        x, y = spec.sweep.axes
        FloatingIslandMapConfig(
            phi3_min_over_pi=x.minimum,
            phi3_max_over_pi=x.maximum,
            phi3_points=x.points,
            phi4_min_over_pi=y.minimum,
            phi4_max_over_pi=y.maximum,
            phi4_points=y.points,
            **_selected(FloatingIslandMapConfig, options),
        )
    else:
        axis = spec.sweep.axes[0]
        FloatingIslandAnalysisConfig(
            coordinate_min=axis.minimum,
            coordinate_max=axis.maximum,
            points=axis.points,
            paths=spec.sweep.paths,
            idle_phi3_over_pi=float(spec.sweep.idle.get("phi3_over_pi", 0.0)),
            idle_phi4_over_pi=float(spec.sweep.idle.get("phi4_over_pi", 0.0)),
            **_selected(FloatingIslandAnalysisConfig, options),
        )


class _FloatingIslandRouterAdapter:
    def check(self, spec: ExperimentSpec) -> None:
        _check_floating(spec)

    def run(self, spec: ExperimentSpec) -> StaticRunPayload:
        return _run_floating(spec)


ADAPTER: StaticModelAdapter = _FloatingIslandRouterAdapter()
