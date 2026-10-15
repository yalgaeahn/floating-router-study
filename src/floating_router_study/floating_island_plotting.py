"""Publication plots for one-dimensional floating-island router results."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from remote_circuit.results import ExperimentResult


_EXACT_MANIFOLD_FIELDS = {
    "assignment_indices",
    "bare_overlap_probability",
    "eigenvalues_from_ground_ghz",
    "one_excitation_bare_indices",
    "one_excitation_bare_states",
    "two_excitation_bare_indices",
    "two_excitation_bare_states",
}

def _nested(mapping: Mapping[str, Any], *keys: str) -> Mapping[str, Any]:
    current: Any = mapping
    for key in keys:
        if not isinstance(current, Mapping):
            return {}
        current = current.get(key, {})
    return current if isinstance(current, Mapping) else {}


def _path_payload(
    result: ExperimentResult, path: str
) -> tuple[np.ndarray, Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]]:
    if result.metadata.get("model_type") != "floating_island_router":
        raise ValueError("floating-island plotting requires a floating_island_router result")
    if path not in result.coordinates or path not in result.observables:
        raise ValueError(f"floating-island result does not contain path {path!r}")
    exact_paths = _nested(result.methods, "exact")
    sw_paths = _nested(result.methods, "schrieffer_wolff")
    if path not in exact_paths or path not in sw_paths:
        raise ValueError("floating-island plotting requires both Exact and SW method results")
    exact = exact_paths[path]
    sw = sw_paths[path]
    if not isinstance(exact, Mapping) or not isinstance(sw, Mapping):
        raise ValueError("floating-island method payload is malformed")
    missing = _EXACT_MANIFOLD_FIELDS - set(exact)
    if missing:
        raise ValueError(
            "floating-island result predates dressed-manifold storage; rerun the "
            f"experiment with the current engine (missing {sorted(missing)})"
        )
    return (
        np.asarray(result.coordinates[path], dtype=float),
        result.observables[path],
        exact,
        sw,
    )


def _qubit_labels(result: ExperimentResult) -> tuple[str, str]:
    options = _nested(result.config, "model", "options")
    labels = options.get("qubit_labels", result.metadata.get("qubit_labels", ("Q1", "Q2")))
    if (
        not isinstance(labels, (list, tuple, np.ndarray))
        or len(labels) != 2
        or any(not str(label).strip() for label in labels)
    ):
        raise ValueError("floating-island result must contain two qubit display labels")
    return str(labels[0]), str(labels[1])


def _thresholds(result: ExperimentResult) -> tuple[float, float, float, float]:
    options = _nested(result.config, "numerics", "options")
    return (
        float(options.get("exact_overlap_floor", 0.75)),
        float(options.get("exact_ambiguity_floor", 0.5)),
        float(options.get("coupling_ratio_reliable_max", 0.1)),
        float(options.get("coupling_ratio_max", 0.2)),
    )


def _assignment_segments(
    assignment: np.ndarray, mask: np.ndarray
) -> list[np.ndarray]:
    """Return contiguous point indices without crossing assignment changes."""

    assignment = np.asarray(assignment, dtype=int)
    mask = np.asarray(mask, dtype=bool)
    if assignment.ndim != 1 or mask.shape != assignment.shape:
        raise ValueError("assignment and mask must be equally sized one-dimensional arrays")
    segments: list[np.ndarray] = []
    start: int | None = None
    for index in range(len(assignment)):
        if not mask[index]:
            if start is not None:
                segments.append(np.arange(start, index))
                start = None
            continue
        if start is None:
            start = index
        elif assignment[index] != assignment[index - 1]:
            segments.append(np.arange(start, index))
            start = index
    if start is not None:
        segments.append(np.arange(start, len(assignment)))
    return [segment for segment in segments if len(segment)]


def _state_label(state: np.ndarray) -> str:
    q_left, q_right, router = (int(value) for value in state)
    if router == 0:
        return rf"$|{q_left}{q_right},g_R\rangle$"
    router_label = {3: "2a", 4: "2b", 5: "2c"}.get(router, str(router))
    return rf"$|{q_left}{q_right},R_{{{router_label}}}\rangle$"


def _plot_assigned_curve(
    axis: plt.Axes,
    x: np.ndarray,
    energies: np.ndarray,
    assigned_indices: np.ndarray,
    overlap: np.ndarray,
    *,
    label: str,
    color: Any,
    reliable_floor: float,
    ambiguity_floor: float,
) -> None:
    reliable = overlap >= reliable_floor
    borderline = (overlap >= ambiguity_floor) & ~reliable
    labeled = False
    for mask, style in ((reliable, "-"), (borderline, "--")):
        for segment in _assignment_segments(assigned_indices, mask):
            if len(segment) == 1:
                axis.scatter(
                    x[segment], energies[segment], color=color, s=16,
                    label=label if not labeled else None,
                )
            else:
                axis.plot(
                    x[segment], energies[segment], color=color, linestyle=style,
                    linewidth=2.0, label=label if not labeled else None,
                )
            labeled = True
    ambiguous = overlap < ambiguity_floor
    if np.any(ambiguous):
        axis.scatter(
            x[ambiguous], energies[ambiguous], color=color, marker="x", s=34,
            linewidths=1.5, label=label if not labeled else None,
        )


def _manifold_arrays(
    exact: Mapping[str, Any], prefix: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    states = np.asarray(exact[f"{prefix}_bare_states"], dtype=int)
    bare_indices = np.asarray(exact[f"{prefix}_bare_indices"], dtype=int)
    if states.ndim == 3:
        states = states[0]
    if bare_indices.ndim == 2:
        bare_indices = bare_indices[0]
    assignment = np.asarray(exact["assignment_indices"], dtype=int)
    eigenvalues = np.asarray(exact["eigenvalues_from_ground_ghz"], dtype=float)
    overlap = np.asarray(exact["bare_overlap_probability"], dtype=float)
    rows = np.arange(len(assignment))[:, None]
    assigned = assignment[:, bare_indices]
    energies = eigenvalues[rows, assigned]
    weights = overlap[:, bare_indices]
    return states, assigned, energies, weights


def plot_floating_island_dressed_manifolds(
    result: ExperimentResult, path: str = "common"
) -> tuple[plt.Figure, np.ndarray]:
    """Plot one- and two-excitation states labeled by pointwise Fors assignment."""

    x, _, exact, _ = _path_payload(result, path)
    reliable_floor, ambiguity_floor, _, _ = _thresholds(result)
    left_label, right_label = _qubit_labels(result)
    figure, axes = plt.subplots(1, 2, figsize=(17, 6.5), constrained_layout=True)
    for axis, (prefix, title) in zip(
        axes,
        (("one_excitation", "One-excitation"), ("two_excitation", "Two-excitation")),
        strict=True,
    ):
        states, assigned, energies, weights = _manifold_arrays(exact, prefix)
        colors = [plt.get_cmap("tab10")(index) for index in range(10)]
        for state_index, state in enumerate(states):
            _plot_assigned_curve(
                axis,
                x,
                energies[:, state_index],
                assigned[:, state_index],
                weights[:, state_index],
                label=_state_label(state),
                color=colors[state_index % len(colors)],
                reliable_floor=reliable_floor,
                ambiguity_floor=ambiguity_floor,
            )
        axis.set_xlabel(r"Common flux $\phi_3/\pi=\phi_4/\pi$")
        axis.set_ylabel(r"Dressed energy from ground / $2\pi$ (GHz)")
        axis.set_title(f"{left_label}–{right_label} {title} dressed manifold")
        axis.grid(alpha=0.18, linewidth=0.9)
        axis.legend(frameon=False, ncol=2, fontsize=10.5)
    return figure, axes


def _shade_regions(
    axis: plt.Axes, coordinate: np.ndarray, mask: np.ndarray, *, label: str | None = None
) -> None:
    coordinate = np.asarray(coordinate, dtype=float)
    mask = np.asarray(mask, dtype=bool)
    if not np.any(mask):
        return
    edges = np.empty(len(coordinate) + 1, dtype=float)
    edges[1:-1] = 0.5 * (coordinate[:-1] + coordinate[1:])
    edges[0] = coordinate[0] - 0.5 * (coordinate[1] - coordinate[0])
    edges[-1] = coordinate[-1] + 0.5 * (coordinate[-1] - coordinate[-2])
    transitions = np.flatnonzero(np.diff(np.r_[False, mask, False])).reshape(-1, 2)
    for region_index, (start, stop) in enumerate(transitions):
        axis.axvspan(
            edges[start], edges[stop], color="0.5", alpha=0.16, zorder=0,
            label=label if region_index == 0 else None,
        )


def _plot_sw_curve(
    axis: plt.Axes,
    coordinate: np.ndarray,
    values: np.ndarray,
    invalid: np.ndarray,
    *,
    color: str,
) -> None:
    """Plot valid SW data dashed and invalid regions as subdued dotted segments."""

    valid_values = np.ma.masked_where(invalid, values)
    axis.plot(
        coordinate, valid_values, color=color, linewidth=2.1,
        linestyle="--", label="SW",
    )
    transitions = np.flatnonzero(
        np.diff(np.r_[False, np.asarray(invalid, dtype=bool), False])
    ).reshape(-1, 2)
    for start, stop in transitions:
        segment = np.arange(max(0, start - 1), min(len(coordinate), stop + 1))
        axis.plot(
            coordinate[segment], values[segment], color=color,
            linewidth=1.7, linestyle=":", alpha=0.5,
        )


def plot_floating_island_interactions(
    result: ExperimentResult, path: str = "common"
) -> tuple[plt.Figure, np.ndarray]:
    """Plot Fors Exact and SW ZZ/exchange using the legacy publication scales."""

    x, frame, _, _ = _path_payload(result, path)
    exact_status = np.asarray(frame["exact_status"]).astype(str)
    exact_visible = exact_status != "ambiguous"
    exact_ambiguous = ~exact_visible
    sw_invalid = ~np.asarray(frame["sw_valid"], dtype=bool)
    exact_zz_raw = np.asarray(frame["exact_zz_khz"]) / 1.0e3
    exact_zz = np.where(exact_visible, exact_zz_raw, np.nan)
    sw_zz_raw = np.asarray(frame["sw_zz_total_khz"]) / 1.0e3
    exact_j_raw = np.asarray(frame["exact_abs_j_mhz"])
    exact_j = np.where(exact_visible, exact_j_raw, np.nan)
    sw_j_raw = np.asarray(frame["sw_abs_j_mhz"])

    figure, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=True, constrained_layout=True)
    axes[0].plot(
        x, exact_zz, color="#2CA02C", linewidth=2.5,
        marker="s", markersize=4.8, markerfacecolor="white", label="Fors Exact",
    )
    axes[0].scatter(
        x[exact_ambiguous], exact_zz_raw[exact_ambiguous],
        facecolors="white", edgecolors="#2CA02C",
        marker="s", s=28, linewidths=1.2, zorder=5,
    )
    _plot_sw_curve(axes[0], x, sw_zz_raw, sw_invalid, color="#2CA02C")
    axes[0].axhline(0.0, color="0.4", linewidth=0.9)
    axes[0].set_yscale("symlog", linthresh=1.0e-3)
    axes[0].set_ylabel(r"$\zeta/2\pi$ (MHz)")

    axes[1].plot(
        x, exact_j, color="#D62728", linewidth=2.5,
        marker="o", markersize=4.8, markerfacecolor="white", label="Fors Exact",
    )
    axes[1].scatter(
        x[exact_ambiguous], exact_j_raw[exact_ambiguous],
        facecolors="white", edgecolors="#D62728",
        marker="o", s=28, linewidths=1.2, zorder=5,
    )
    _plot_sw_curve(axes[1], x, sw_j_raw, sw_invalid, color="#D62728")
    axes[1].set_yscale("log")
    axes[1].set_ylabel(r"$|g_{\mathrm{eff}}|/2\pi$ (MHz)")
    axes[1].set_xlabel(r"Common flux $\phi_3/\pi=\phi_4/\pi$")
    for axis in axes:
        _shade_regions(axis, x, exact_ambiguous)
        axis.grid(alpha=0.18, linewidth=0.9)
        axis.legend(frameon=False)
    return figure, axes


def plot_floating_island_diagnostics(
    result: ExperimentResult, path: str = "common"
) -> tuple[plt.Figure, np.ndarray]:
    """Plot pointwise assignment weights and SW validity diagnostics."""

    x, frame, exact, _ = _path_payload(result, path)
    reliable_floor, ambiguity_floor, sw_reliable, sw_invalid = _thresholds(result)
    branch_weights = np.asarray(exact["branch_overlaps"], dtype=float)
    subspace = np.asarray(exact["one_excitation_subspace_overlap"], dtype=float)
    figure, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True, constrained_layout=True)
    for index, label in enumerate(("00", "10", "01", "11")):
        axes[0].plot(x, branch_weights[:, index], label=rf"$|{label},g_R\rangle$")
    axes[0].plot(x, subspace, color="black", linestyle="-.", label="1-excitation subspace")
    axes[0].axhline(reliable_floor, color="C2", linestyle=":", label="reliable threshold")
    axes[0].axhline(ambiguity_floor, color="C3", linestyle="--", label="ambiguous threshold")
    axes[0].set(ylabel="Fors assignment diagnostic", ylim=(0.0, 1.02))
    axes[0].legend(ncol=3, fontsize="small")

    axes[1].semilogy(x, frame["sw_maximum_coupling_ratio"], label=r"target max $|V/\Delta|$")
    axes[1].semilogy(
        x, frame["sw_global_maximum_coupling_ratio"], alpha=0.65,
        label=r"global max $|V/\Delta|$",
    )
    axes[1].axhline(sw_reliable, color="C2", linestyle="--", label="reliable threshold")
    axes[1].axhline(sw_invalid, color="C3", linestyle="--", label="invalid threshold")
    axes[1].set(
        xlabel=r"Common flux $\phi_3/\pi=\phi_4/\pi$",
        ylabel="SW coupling ratio",
    )
    axes[1].legend(ncol=2, fontsize="small")
    for axis in axes:
        axis.grid(alpha=0.2)
    return figure, axes


def write_floating_island_router_plots(
    result: ExperimentResult, output: str | Path
) -> list[Path]:
    """Write the complete floating-island 1D plot bundle as PNG and PDF files."""

    output_dir = Path(output).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    plotters = (
        ("dressed_manifolds", plot_floating_island_dressed_manifolds),
        ("interactions", plot_floating_island_interactions),
        ("diagnostics", plot_floating_island_diagnostics),
    )
    for path_name in result.coordinates:
        for suffix, plotter in plotters:
            figure, _ = plotter(result, str(path_name))
            for extension in ("png", "pdf"):
                destination = output_dir / f"{path_name}_{suffix}.{extension}"
                figure.savefig(
                    destination,
                    dpi=300 if extension == "png" else None,
                    bbox_inches="tight",
                )
                paths.append(destination)
            plt.close(figure)
    return paths
