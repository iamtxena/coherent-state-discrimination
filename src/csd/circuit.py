# circuit.py
from abc import ABC
from typing import List, Union

from strawberryfields.parameters import FreeParameter
from tensorflow.python.framework.ops import EagerTensor
from csd.operations.universal_multimode import UniversalMultimode
from csd.typings import Architecture, MeasuringTypes
import strawberryfields as sf
from typeguard import typechecked


class Circuit(ABC):
    """ Photonic Circuit class

    """

    @typechecked
    def __init__(self, architecture: Architecture, measuring_type: MeasuringTypes) -> None:

        if not architecture or 'number_modes' not in architecture:
            raise ValueError('No architecture or number_modes specified')

        M = architecture['number_modes']
        K = int(M * (M - 1) / 2)

        self._prog = sf.Program(M)
        theta_1 = self._create_free_parameter_list(base_name='theta_1', number_elems=K, circuit=self._prog)
        phi_1 = self._create_free_parameter_list(base_name='phi_1', number_elems=K, circuit=self._prog)
        varphi_1 = self._create_free_parameter_list(base_name='varphi_1', number_elems=M, circuit=self._prog)
        r = self._create_free_parameter_list(base_name='r', number_elems=M, circuit=self._prog)
        phi_r = self._create_free_parameter_list(base_name='phi_r', number_elems=M, circuit=self._prog)
        theta_2 = self._create_free_parameter_list(base_name='theta_2', number_elems=K, circuit=self._prog)
        phi_2 = self._create_free_parameter_list(base_name='phi_2', number_elems=K, circuit=self._prog)
        varphi_2 = self._create_free_parameter_list(base_name='varphi_2', number_elems=M, circuit=self._prog)
        a = self._create_free_parameter_list(base_name='a', number_elems=M, circuit=self._prog)

        self._free_parameters = self._count_free_parameters(
            theta_1, phi_1, varphi_1, r, phi_r, theta_2, phi_2, varphi_2, a, architecture['squeezing'])

        with self._prog.context as q:
            UniversalMultimode(theta_1=theta_1,
                               phi_1=phi_1,
                               varphi_1=varphi_1,
                               r=r,
                               phi_r=phi_r,
                               theta_2=theta_2,
                               phi_2=phi_2,
                               varphi_2=varphi_2,
                               a=a,
                               number_modes=M,
                               context=q,
                               squeezing=architecture['squeezing'])
            if measuring_type is MeasuringTypes.SAMPLING:
                sf.ops.MeasureFock() | q

    def _create_free_parameter_list(self,
                                    base_name: str,
                                    number_elems: int,
                                    circuit: sf.Program) -> List[FreeParameter]:
        return [circuit.params(f'{base_name}_{str(elem)}') for elem in range(number_elems)]

    def _count_free_parameters(self,
                               theta_1: List[Union[float, EagerTensor, FreeParameter]],
                               phi_1: List[Union[float, EagerTensor, FreeParameter]],
                               varphi_1: List[Union[float, EagerTensor, FreeParameter]],
                               r: List[Union[float, EagerTensor, FreeParameter]],
                               phi_r: List[Union[float, EagerTensor, FreeParameter]],
                               theta_2: List[Union[float, EagerTensor, FreeParameter]],
                               phi_2: List[Union[float, EagerTensor, FreeParameter]],
                               varphi_2: List[Union[float, EagerTensor, FreeParameter]],
                               a: List[Union[float, EagerTensor, FreeParameter]],
                               squeezing: bool = True) -> None:
        if squeezing:
            return (len(theta_1) + len(phi_1) + len(varphi_1) + len(r) +
                    len(phi_r) + len(theta_2) + len(phi_2) + len(varphi_2) + len(a))
        return (len(theta_1) + len(phi_1) + len(varphi_1) +
                len(theta_2) + len(phi_2) + len(varphi_2) + len(a))

    @property
    def circuit(self) -> sf.Program:
        return self._prog

    @property
    def free_parameters(self) -> int:
        """Number of free parameters to optimize

        Returns:
            int: Number of free parameters to optimize
        """
        return self._free_parameters
