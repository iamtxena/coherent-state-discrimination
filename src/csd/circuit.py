# circuit.py
from abc import ABC
from csd.typings import Architecture, MeasuringTypes
import strawberryfields as sf
from typeguard import typechecked


class Circuit(ABC):
    """ Photonic Circuit class

    """

    @typechecked
    def __init__(self, architecture: Architecture, measuring_type: MeasuringTypes) -> None:

        self._prog = sf.Program(architecture['number_qumodes'])
        alpha = self._prog.params("alpha")
        if architecture['displacement']:
            beta = self._prog.params("beta")
        if architecture['squeezing']:
            r = self._prog.params("r")
            phi_r = self._prog.params("phi_r")

        with self._prog.context as q:
            sf.ops.Dgate(alpha, 0.0) | q[0]
            if architecture['displacement']:
                sf.ops.Dgate(beta, 0.0) | q[0]
            if architecture['squeezing']:
                sf.ops.Sgate(r, phi_r) | q[0]
            if measuring_type is MeasuringTypes.SAMPLING:
                sf.ops.MeasureFock() | q[0]

    @property
    def circuit(self) -> sf.Program:
        return self._prog

    @property
    def parameters(self) -> dict:
        return self._prog.free_params
