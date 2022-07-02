# universal_multimode.py
from abc import ABC
from typing import List, Union
from strawberryfields.parameters import FreeParameter
from typeguard import typechecked
import strawberryfields as sf
from tensorflow.python.framework.ops import EagerTensor
from .interferometer import Interferometer


class UniversalMultimode(ABC):
    """Creates a set of gates that represents universal gates to
    a multi mode circuit.
    Based on pennylane CVNeuralNetLayers:
    https://pennylane.readthedocs.io/en/stable/code/api/pennylane.templates.layers.CVNeuralNetLayers.html

    A sequence of layers of a continuous-variable quantum
    neural network, as specified in arXiv:1806.06871.
    The layer consists of interferometers, displacement and
    squeezing gates mimicking the linear transformation of a
    neural network in the x-basis of the quantum system,
    in this implementation, without the Kerr gate.

    This implementation applys to only one layer with M modes,
    and include interferometers of K=M(M−1)/2 beamsplitters.

    """

    @typechecked
    def __init__(
        self,
        theta_1: List[Union[float, EagerTensor, FreeParameter]],
        phi_1: List[Union[float, EagerTensor, FreeParameter]],
        varphi_1: List[Union[float, EagerTensor, FreeParameter]],
        r: List[Union[float, EagerTensor, FreeParameter]],
        phi_r: List[Union[float, EagerTensor, FreeParameter]],
        theta_2: List[Union[float, EagerTensor, FreeParameter]],
        phi_2: List[Union[float, EagerTensor, FreeParameter]],
        varphi_2: List[Union[float, EagerTensor, FreeParameter]],
        a: List[Union[float, EagerTensor, FreeParameter]],
        number_modes: int,
        context,
        squeezing: bool = True,
    ) -> None:
        """Creates an Universal Multimode gate to the specified circuit (program context)

        Args:
            theta_1 (List[Union[float, EagerTensor]]): shape (1,K) tensor of transmittivity
                                                        angles for first interferometer
            phi_1 (List[Union[float, EagerTensor]]): shape (1,K) tensor of phase angles for first interferometer
            varphi_1 (List[Union[float, EagerTensor]]): shape (1,M) tensor of rotation angles
                                                        to apply after first interferometer
            r (List[Union[float, shape (1,M) tensor of squeezing amounts for Squeezing operations
            phi_r (List[Union[float, EagerTensor]]): [shape (1,M) tensor of squeezing angles for Squeezing operations
            theta_2 (List[Union[float, EagerTensor]]): shape (1,K) tensor of transmittivity
                                                        angles for second interferometer
            phi_2 (List[Union[float, EagerTensor]]): shape (1,K) tensor of phase angles for second interferometer
            varphi_2 (List[Union[float, EagerTensor]]): shape (1,M) tensor of rotation angles
                                                        to apply after second interferometer
            a (List[Union[float, EagerTensor]]): shape (1,M) tensor of displacement magnitudes
                                                        for Displacement operations
            number_modes (int): wires that the interferometer acts on.
            context ([type]): circuit context
            squeezing (bool): boolean indicating if squeezing gates layer needs to be added to circuit
        """
        modes = list(range(number_modes))

        Interferometer(theta=theta_1, phi=phi_1, varphi=varphi_1, number_modes=number_modes, context=context)

        if squeezing:
            self._apply_squeezing_layer_to_all_modes(r, phi_r, context, modes)
            Interferometer(theta=theta_2, phi=phi_2, varphi=varphi_2, number_modes=number_modes, context=context)

        self._apply_displacement_layer_to_all_modes(a, context, modes)

    def _apply_displacement_layer_to_all_modes(self, a, context, modes):
        for mode in modes:
            sf.ops.Dgate(a[mode], 0.0) | context[mode]

    def _apply_squeezing_layer_to_all_modes(self, r, phi_r, context, modes):
        for mode in modes:
            sf.ops.Sgate(r[mode], phi_r[mode]) | context[mode]
