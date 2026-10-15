from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest

from floating_router_study.floating_island_plotting import (
    _assignment_segments,
    plot_floating_island_diagnostics,
    plot_floating_island_dressed_manifolds,
    plot_floating_island_interactions,
    write_floating_island_router_plots,
)
from floating_router_study.floating_island_router import (
    FloatingIslandAnalysisConfig,
    FloatingIslandRouterModel,
    run_floating_island_router_methods,
)
from floating_router_study.reference import (
    build_flagship_floating_island_spec,
    flagship_floating_island_reduction_spec,
)
from remote_circuit.result_tools import plot_result
from remote_circuit.results import ExperimentResult
from remote_circuit.circuits import load_circuit_topology


ROOT = Path(__file__).parents[1]
CIRCUIT = ROOT / "examples" / "circuits" / "flagship_floating_island_router.yaml"


def _result() -> ExperimentResult:
    spec = build_flagship_floating_island_spec(
        load_circuit_topology(CIRCUIT).scqubits_yaml(),
        qubit_truncated_dim=3,
        router_truncated_dim=6,
        qubit_ncut=4,
        router_ncut=4,
    )
    model = FloatingIslandRouterModel(spec, flagship_floating_island_reduction_spec())
    path = run_floating_island_router_methods(
        model,
        FloatingIslandAnalysisConfig(
            coordinate_min=0.0,
            coordinate_max=0.1,
            points=3,
            paths=("common",),
        ),
    )["common"]
    return ExperimentResult(
        coordinates={"common": path.frame["coordinate_phi_over_pi"].to_numpy()},
        observables={
            "common": {column: path.frame[column].to_numpy() for column in path.frame}
        },
        methods={
            "exact": {"common": path.exact},
            "schrieffer_wolff": {"common": path.schrieffer_wolff},
        },
        diagnostics={"validity": {}},
        metadata={
            "model_type": "floating_island_router",
            "uses_adjacent_flux": False,
            "qubit_labels": ["Q1", "Q3"],
        },
        config={
            "model": {"options": {"qubit_labels": ["Q1", "Q3"]}},
            "numerics": {
                "options": {
                    "exact_overlap_floor": 0.75,
                    "exact_ambiguity_floor": 0.5,
                    "coupling_ratio_reliable_max": 0.1,
                    "coupling_ratio_max": 0.2,
                }
            },
        },
    )


def test_assignment_segments_never_cross_an_index_change():
    assignment = np.asarray((0, 0, 2, 2, 2, 4))
    segments = _assignment_segments(assignment, np.ones(6, dtype=bool))
    assert [segment.tolist() for segment in segments] == [[0, 1], [2, 3, 4], [5]]
    assert all(np.all(assignment[segment] == assignment[segment[0]]) for segment in segments)


def test_model_specific_plotters_create_expected_manifold_and_diagnostics():
    result = _result()
    manifold, manifold_axes = plot_floating_island_dressed_manifolds(result)
    interactions, interaction_axes = plot_floating_island_interactions(result)
    diagnostics, diagnostic_axes = plot_floating_island_diagnostics(result)
    assert manifold_axes.shape == (2,)
    assert "Q1–Q3" in manifold_axes[0].get_title()
    assert len(manifold_axes[0].get_legend().get_texts()) == 4
    assert len(manifold_axes[1].get_legend().get_texts()) == 10
    assert interaction_axes[0].get_yscale() == "symlog"
    assert interaction_axes[1].get_yscale() == "log"
    assert diagnostic_axes.shape == (2,)
    for figure in (manifold, interactions, diagnostics):
        plt.close(figure)


def test_ambiguous_exact_interactions_remain_visible_as_markers():
    result = _result()
    frame = {
        key: np.array(value, copy=True)
        for key, value in result.observables["common"].items()
    }
    frame["exact_status"][1] = "ambiguous"
    frame["exact_zz_khz"][1] = 12345.0
    frame["exact_abs_j_mhz"][1] = 7.89
    frame["sw_valid"][1] = False
    marked = ExperimentResult(
        **{**result.__dict__, "observables": {"common": frame}}
    )

    figure, axes = plot_floating_island_interactions(marked)
    for axis, expected_color in zip(axes, ("#2CA02C", "#D62728"), strict=True):
        legend_labels = [text.get_text() for text in axis.get_legend().get_texts()]
        assert "ambiguous Fors Exact" not in legend_labels
        assert axis.lines[0].get_color() == expected_color
        assert axis.lines[1].get_color() == expected_color
        assert axis.lines[1].get_linestyle() == "--"
        assert axis.lines[2].get_color() == expected_color
        assert axis.lines[2].get_linestyle() == ":"
        assert np.array_equal(axis.collections[0].get_sizes(), np.array([28]))
    coordinate = float(marked.coordinates["common"][1])
    top_offsets = [
        np.asarray(collection.get_offsets())
        for collection in axes[0].collections
        if hasattr(collection, "get_offsets")
    ]
    bottom_offsets = [
        np.asarray(collection.get_offsets())
        for collection in axes[1].collections
        if hasattr(collection, "get_offsets")
    ]
    assert any(np.any(np.all(offsets == [coordinate, 12.345], axis=1)) for offsets in top_offsets)
    assert any(np.any(np.all(offsets == [coordinate, 7.89], axis=1)) for offsets in bottom_offsets)
    plt.close(figure)


def test_plot_bundle_and_generic_cli_dispatch_write_six_files(tmp_path: Path):
    result = _result()
    direct = write_floating_island_router_plots(result, tmp_path / "direct")
    dispatched = plot_result(result, tmp_path / "dispatch")
    assert len(direct) == 6
    assert len(dispatched) == 6
    assert all(path.is_file() for path in (*direct, *dispatched))
    assert {path.name for path in dispatched} == {
        "common_dressed_manifolds.png",
        "common_dressed_manifolds.pdf",
        "common_interactions.png",
        "common_interactions.pdf",
        "common_diagnostics.png",
        "common_diagnostics.pdf",
    }


def test_old_result_is_rejected_with_actionable_error():
    result = _result()
    old_exact = dict(result.methods["exact"]["common"])
    old_exact.pop("eigenvalues_from_ground_ghz")
    old = ExperimentResult(
        **{**result.__dict__, "methods": {**result.methods, "exact": {"common": old_exact}}}
    )
    with pytest.raises(ValueError, match="rerun the experiment"):
        plot_floating_island_dressed_manifolds(old)
