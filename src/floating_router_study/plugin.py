"""External model registration and model-owned validation."""
from dataclasses import replace
from remote_circuit import ModelDefinition
from remote_circuit.experiment import ExperimentConfigError
from .adapter import ADAPTER

def validate_model_spec(spec):
    if spec.model_type != "floating_island_router":
        raise ExperimentConfigError("circuit.model: expected floating_island_router")
    if spec.numerics.preset not in {"smoke", "standard", "converged"}:
        raise ExperimentConfigError("numerics.preset: expected smoke, standard, or converged")
    dimensions = len(spec.sweep.axes)
    if dimensions == 1:
        if not spec.sweep.paths:
            raise ExperimentConfigError("sweep.paths: at least one path is required")
        unknown = set(spec.sweep.paths) - set(spec.circuit.topology.controls)
        if unknown:
            raise ExperimentConfigError(f"sweep.paths: undefined by circuit controls: {sorted(unknown)}")
    elif spec.sweep.paths:
        raise ExperimentConfigError("sweep.paths: omit paths for a 2D sweep")
    labels = spec.circuit.topology.parameters.get("computational_qubits")
    if labels is not None:
        names = {subsystem["name"] for subsystem in spec.circuit.topology.subsystems}
        if not isinstance(labels, (list, tuple)) or len(labels) != 2 or len(set(labels)) != 2 or not set(labels) <= names:
            raise ExperimentConfigError("circuit.parameters.computational_qubits: exactly two names referring to distinct declared subsystems")
    if spec.output.storage != "full":
        raise ExperimentConfigError("output.storage: this model supports only full storage")
    expected = "over_pi"
    idle_keys = {"phi3_over_pi", "phi4_over_pi"}

    if any(axis.unit != expected for axis in spec.sweep.axes):
        raise ExperimentConfigError(f"sweep.axes.unit: expects {expected!r}")
    if set(spec.sweep.idle) - idle_keys:
        raise ExperimentConfigError("sweep.idle: unsupported control")


class StudyAdapter:
    def check(self, spec):
        validate_model_spec(spec)
        ADAPTER.check(spec)

    def run(self, spec):
        payload = ADAPTER.run(spec)
        metadata = dict(payload.metadata)
        if spec.methods.exact:
            metadata.update({
                "assignment_algorithm": "gale-shapley",
                "exact_reference_assignment": "pointwise dressed-proposer Gale-Shapley",
                "assignment_scope": "full retained basis",
                "assignment_similarity": "absolute-bare-dressed-overlap",
                "assignment_proposer": "dressed",
                "qubit_labels": list(spec.circuit.topology.parameters.get("computational_qubits", ("Q1", "Q2"))),
            })
        return replace(payload, metadata=metadata)

def plot(result, output):
    from .floating_island_plotting import write_floating_island_router_plots
    if all(key in result.observables and isinstance(result.observables[key], dict) for key in result.coordinates):
        return write_floating_island_router_plots(result, output)
    from remote_circuit import ModelRegistry, plot_result
    return plot_result(result, output, registry=ModelRegistry())

MODEL = ModelDefinition(
    model_id="floating_island_router",
    supported_methods=frozenset({"exact", "schrieffer_wolff"}),
    adapter=StudyAdapter(),
    plot=plot,
)
