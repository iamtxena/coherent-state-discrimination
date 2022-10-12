""" Multi Layer Circuit Structure """

import copy
from dataclasses import InitVar, dataclass, field

import strawberryfields as sf
from csd.circuit import Circuit


@dataclass
class FirstLayerCircuit:
    """First Layer Circuit"""

    circuit: Circuit


@dataclass
class SecondLayerCircuit:
    """Second Layer Circuit"""

    circuit: InitVar[Circuit]
    circuit_zero_on_first_layer_mode: Circuit = field(init=False)
    circuit_one_on_first_layer_mode: Circuit = field(init=False)

    def __post_init__(self, circuit: Circuit):
        self.circuit_zero_on_first_layer_mode = copy.deepcopy(circuit)
        self.circuit_one_on_first_layer_mode = copy.deepcopy(circuit)


@dataclass
class MultiLayerCircuit:
    """Multi Layer Circuit"""

    number_layers: int
    base_circuit: InitVar[Circuit]
    first_layer: FirstLayerCircuit = field(init=False)
    second_layer: SecondLayerCircuit | None = field(init=False)

    def __post_init__(self, base_circuit: Circuit):
        if self.number_layers < 1 or self.number_layers > 2:
            raise ValueError("Number of layers must be at least 1 or no more than 2.")

        self.first_layer = FirstLayerCircuit(circuit=base_circuit)
        if self.number_layers == 1:
            self.second_layer = None
        if self.number_layers == 2:
            self.second_layer = SecondLayerCircuit(circuit=base_circuit)

    @property
    def circuit(self) -> sf.Program:
        """Circuit Program representation"""
        return self.first_layer.circuit.circuit

    @property
    def free_parameters(self) -> int:
        """Number of free parameters to optimize

        Returns:
            int: Number of free parameters to optimize
        """
        if self.number_layers == 1:
            return self.first_layer.circuit.free_parameters
        return (
            self.first_layer.circuit.free_parameters
            + self.second_layer.circuit_one_on_first_layer_mode.free_parameters
            + self.second_layer.circuit_zero_on_first_layer_mode.free_parameters
        )

    @property
    def parameters(self) -> dict:
        """circuit parameters dictionary"""
        if self.number_layers == 1:
            return self.first_layer.circuit.parameters
        return (
            self.first_layer.circuit.parameters
            | self.second_layer.circuit_one_on_first_layer_mode.parameters
            | self.second_layer.circuit_zero_on_first_layer_mode.parameters
        )

    @property
    def number_modes(self) -> int:
        """circuit number_modes"""
        return self.first_layer.circuit.number_modes

    @property
    def number_ancillas(self) -> int:
        """circuit number_ancillas"""
        return self.first_layer.circuit.number_ancillas

    @property
    def number_input_modes(self) -> int:
        """circuit number_input_modes"""
        return self.first_layer.circuit.number_input_modes
