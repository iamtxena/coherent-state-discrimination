# interferometer.py

from abc import ABC
from typing import List, Union
from typeguard import typechecked
import strawberryfields as sf
from tensorflow.python.framework.ops import EagerTensor


class Interferometer(ABC):
    """ General linear interferometer, an array of beamsplitters and phase shifters.

        For M wires, the general interferometer is specified by providing
        M(M−1)/2 transmittivity angles θ and the same number of phase angles ϕ,
        as well as M−1 additional rotation parameters φ.

        The default scheme is 'rectangular': uses the scheme described in Clements et al.,
        resulting in a rectangular array of M(M−1)/2 beamsplitters arranged in M slices
        and ordered from left to right and top to bottom in each slice.
        The first beamsplitter acts on wires 0 and 1.

        The implementation is based on pennylane, but using strawberry library:
        https://pennylane.readthedocs.io/en/stable/code/api/pennylane.templates.subroutines.Interferometer.html
    """

    @typechecked
    def __init__(self,
                 theta: List[Union[float, EagerTensor]],
                 phi: List[Union[float, EagerTensor]],
                 varphi: List[Union[float, EagerTensor]],
                 number_modes: int,
                 context) -> None:
        """ Creates an Interferometer to the specified circuit (program context)

        Args:
            theta (tensor_like): size :math:`(M(M-1)/2,)` tensor of transmittivity angles :math:`\theta`
            phi (tensor_like): size :math:`(M(M-1)/2,)` tensor of phase angles :math:`\phi`
            varphi (tensor_like): size :math:`(M,)` tensor of rotation angles :math:`\varphi`
            number_modes (int): wires that the interferometer acts on
            context: (sf.Program): circuit context
        """
        modes = list(range(number_modes))

        if number_modes > 1:
            self._apply_clements_beamsplitter_array(theta, phi, context, modes)

        self._apply_final_local_phase_shits_to_all_modes(varphi, context, modes)

    def _apply_clements_beamsplitter_array(self, theta, phi, context, modes):
        free_parameters = 0  # keep track of free parameters
        for mode in modes:
            for k, (w1, w2) in enumerate(zip(modes[:-1], modes[1:])):
                # skip even or odd pairs depending on layer
                if (mode + k) % 2 != 1:
                    sf.ops.BSgate(theta[free_parameters], phi[free_parameters]) | (context[w1], context[w2])
                    free_parameters += 1

    def _apply_final_local_phase_shits_to_all_modes(self, varphi, context, modes):
        for mode in modes:
            sf.ops.Rgate(varphi[mode]) | context[mode]
