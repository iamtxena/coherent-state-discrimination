# circuit.py
from abc import ABC
from csd.typings import Architecture, MeasuringTypes
import strawberryfields as sf
from strawberryfields.parameters import FreeParameter
from typeguard import typechecked


class Circuit(ABC):
    """ Photonic Circuit class

    """

    @typechecked
    def __init__(self, architecture: Architecture, measurement_type: MeasuringTypes) -> None:

        self._prog = sf.Program(architecture['number_qumodes'])
        alpha = self._prog.params("alpha")
        if 'displacement' in architecture:
            beta = self._prog.params("beta")
        if 'squeezing' in architecture:
            gamma = self._prog.params("gamma")

        with self._prog.context as q:
            sf.ops.Dgate(alpha, 0.0) | q[0]
            if 'displacement' in architecture:
                sf.ops.Dgate(beta, 0.0) | q[0]
            if 'squeezing' in architecture:
                sf.ops.Sgate(gamma, 0.0) | q[0]
            if measurement_type is MeasuringTypes.SAMPLING:
                sf.ops.MeasureFock() | q[0]

    @property
    def circuit(self) -> sf.Program:
        return self._prog

    @property
    def parameters(self) -> list[FreeParameter]:
        return self._prog.params()
