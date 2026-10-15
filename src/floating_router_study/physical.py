from __future__ import annotations
from dataclasses import dataclass, field
from typing import Mapping
from remote_circuit.circuits.physical import PhysicalCircuitSpec as BasePhysicalCircuitSpec
from remote_circuit.circuits.physical import (
    CoordinateReductionSpec as CoordinateReductionSpec,
    SQUIDSpec as SQUIDSpec,
    SubsystemSpec as SubsystemSpec,
    ParsedBranch as ParsedBranch,
    ParsedPhysicalCircuit as ParsedPhysicalCircuit,
    ReducedPhysicalCircuit as ReducedPhysicalCircuit,
    reduce_capacitance_to_active_coordinates as reduce_capacitance_to_active_coordinates,
    ChargingInteraction as ChargingInteraction,
    UnreachableTargetFrequency as UnreachableTargetFrequency,
    subsystem_capacitance_matrix_ff as subsystem_capacitance_matrix_ff,
    inter_subsystem_charging_terms as inter_subsystem_charging_terms,
    parse_scqubits_circuit_yaml as parse_scqubits_circuit_yaml,
)

@dataclass(frozen=True)
class SweepSpec:
    parameter: str
    minimum: float
    maximum: float
    points: int
    assignments: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True, kw_only=True)
class PhysicalCircuitSpec(BasePhysicalCircuitSpec):
    sweep: SweepSpec
    computational_qubits: tuple[str, str]
    tunable_subsystem: str

    def parse(self):
        parsed = super().parse()
        missing = set(self.computational_qubits + (self.tunable_subsystem,)) - {s.name for s in self.subsystems}
        if missing:
            raise ValueError(f"metadata references unknown subsystems: {sorted(missing)}")
        if self.sweep.points < 2 or self.sweep.maximum <= self.sweep.minimum:
            raise ValueError("sweep requires an increasing domain and at least two points")
        return parsed
