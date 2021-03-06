# circuit.py
from abc import ABC
from typing import List, Optional, Union

from strawberryfields.parameters import FreeParameter
from tensorflow.python.framework.ops import EagerTensor
from csd.operations.universal_multimode import UniversalMultimode
from csd.typings.typing import Architecture, MeasuringTypes, RunningTypes

# from csd.config import logger
import strawberryfields as sf
from typeguard import typechecked


class Circuit(ABC):
    """Photonic Circuit class"""

    @typechecked
    def __init__(self, architecture: Architecture, measuring_type: MeasuringTypes, running_type: RunningTypes) -> None:

        if not architecture or "number_modes" not in architecture:
            raise ValueError("No architecture or number_modes specified")

        self._number_input_modes = architecture["number_modes"]
        self._number_ancillas = architecture["number_ancillas"] if "number_ancillas" in architecture else 0
        self._number_modes = self._number_input_modes + self._number_ancillas

        self._prog = sf.Program(self.number_modes)

        if self.number_modes == 1:
            self._create_circuit_for_one_mode(
                squeezing=architecture["squeezing"], measuring_type=measuring_type, running_type=running_type
            )
            return

        self._create_multimode_circuit(
            squeezing=architecture["squeezing"],
            number_input_modes=self.number_input_modes,
            number_ancillas=self.number_ancillas,
            measuring_type=measuring_type,
            running_type=running_type,
        )

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

    @property
    def parameters(self) -> dict:
        return self._prog.free_params

    @property
    def number_modes(self) -> int:
        return self._number_modes

    @property
    def number_ancillas(self) -> int:
        return self._number_ancillas

    @property
    def number_input_modes(self) -> int:
        return self._number_input_modes

    def _create_multimode_circuit(
        self,
        squeezing: bool,
        number_input_modes: int,
        number_ancillas: int,
        measuring_type: MeasuringTypes,
        running_type: RunningTypes,
    ) -> None:
        M = number_input_modes + number_ancillas
        K = int(M * (M - 1) / 2)

        alpha = self._create_free_parameter_list(base_name="alpha", number_elems=number_input_modes, circuit=self._prog)
        theta_1 = self._create_free_parameter_list(base_name="theta_1", number_elems=K, circuit=self._prog)
        phi_1 = self._create_free_parameter_list(base_name="phi_1", number_elems=K, circuit=self._prog)
        varphi_1 = self._create_free_parameter_list(base_name="varphi_1", number_elems=M, circuit=self._prog)
        if squeezing:
            r = self._create_free_parameter_list(base_name="r", number_elems=M, circuit=self._prog)
            phi_r = self._create_free_parameter_list(base_name="phi_r", number_elems=M, circuit=self._prog)
            theta_2 = self._create_free_parameter_list(base_name="theta_2", number_elems=K, circuit=self._prog)
            phi_2 = self._create_free_parameter_list(base_name="phi_2", number_elems=K, circuit=self._prog)
            varphi_2 = self._create_free_parameter_list(base_name="varphi_2", number_elems=M, circuit=self._prog)
        a = self._create_free_parameter_list(base_name="a", number_elems=M, circuit=self._prog)

        self._free_parameters = self._count_free_parameters(
            theta_1=theta_1,
            phi_1=phi_1,
            varphi_1=varphi_1,
            theta_2=theta_2 if squeezing else [],
            phi_2=phi_2 if squeezing else [],
            varphi_2=varphi_2 if squeezing else [],
            a=a,
            r=r if squeezing else [],
            phi_r=phi_r if squeezing else [],
        )
        # logger.debug(f'registered parameters: {self.parameters}')
        with self._prog.context as q:
            self._apply_displacement_layer_to_only_input_modes(
                alpha, context=q, number_input_modes=number_input_modes, number_ancillas=number_ancillas
            )
            # self._apply_thermal_loss_layer_to_all_modes(context=q, number_modes=M)
            UniversalMultimode(
                theta_1=theta_1,
                phi_1=phi_1,
                varphi_1=varphi_1,
                r=r if squeezing else [],
                phi_r=phi_r if squeezing else [],
                theta_2=theta_2 if squeezing else [],
                phi_2=phi_2 if squeezing else [],
                varphi_2=varphi_2 if squeezing else [],
                a=a,
                number_modes=M,
                context=q,
                squeezing=squeezing,
            )
            if measuring_type is MeasuringTypes.SAMPLING:
                sf.ops.MeasureFock() | q

    def _create_circuit_for_one_mode(
        self, squeezing: bool, measuring_type: MeasuringTypes, running_type: RunningTypes
    ) -> None:
        number_modes = 1
        alpha = self._create_free_parameter_list(base_name="alpha", number_elems=number_modes, circuit=self._prog)
        if squeezing:
            r = self._create_free_parameter_list(base_name="r", number_elems=number_modes, circuit=self._prog)
            phi_r = self._create_free_parameter_list(base_name="phi_r", number_elems=number_modes, circuit=self._prog)
            self._free_parameters = 3
        if not squeezing:
            self._free_parameters = 1
        a = self._create_free_parameter_list(base_name="a", number_elems=number_modes, circuit=self._prog)

        with self._prog.context as q:
            self._apply_displacement_layer_to_all_modes(alpha, context=q, number_modes=number_modes)
            self._apply_displacement_layer_to_all_modes(a, context=q, number_modes=number_modes)
            if squeezing:
                self._apply_squeezing_layer_to_all_modes(r, phi_r, context=q, number_modes=number_modes)
            if measuring_type is MeasuringTypes.SAMPLING:
                sf.ops.MeasureFock() | q

    def _apply_displacement_layer_to_only_input_modes(
        self, alpha: List[FreeParameter], context: sf.Program, number_input_modes: int, number_ancillas: int
    ):
        modes = list(range(number_input_modes))
        for mode in modes:
            sf.ops.Dgate(alpha[mode], 0.0) | context[mode + number_ancillas]

    def _apply_thermal_loss_layer_to_all_modes(self, context: sf.Program, number_modes: int):
        modes = list(range(number_modes))
        for mode in modes:
            sf.ops.ThermalLossChannel(0.5, 0.25) | context[mode]

    def _apply_displacement_layer_to_all_modes(
        self, alpha: List[FreeParameter], context: sf.Program, number_modes: int
    ):
        self._apply_displacement_layer_to_only_input_modes(
            alpha=alpha, context=context, number_input_modes=number_modes, number_ancillas=0
        )

    def _apply_squeezing_layer_to_all_modes(self, r, phi_r, context, number_modes):
        modes = list(range(number_modes))
        for mode in modes:
            sf.ops.Sgate(r[mode], phi_r[mode]) | context[mode]

    def _create_free_parameter_list(
        self, base_name: str, number_elems: int, circuit: sf.Program
    ) -> List[FreeParameter]:
        return [circuit.params(f"{base_name}_{str(elem)}") for elem in range(number_elems)]

    def _count_free_parameters(
        self,
        theta_1: List[Union[float, EagerTensor, FreeParameter]],
        phi_1: List[Union[float, EagerTensor, FreeParameter]],
        varphi_1: List[Union[float, EagerTensor, FreeParameter]],
        theta_2: List[Union[float, EagerTensor, FreeParameter]],
        phi_2: List[Union[float, EagerTensor, FreeParameter]],
        varphi_2: List[Union[float, EagerTensor, FreeParameter]],
        a: List[Union[float, EagerTensor, FreeParameter]],
        r: Optional[List[Union[float, EagerTensor, FreeParameter]]] = [],
        phi_r: Optional[List[Union[float, EagerTensor, FreeParameter]]] = [],
    ) -> int:

        if r is None:
            r = []
        if phi_r is None:
            phi_r = []

        return (
            len(theta_1)
            + len(phi_1)
            + len(varphi_1)
            + len(r)
            + len(phi_r)
            + len(theta_2)
            + len(phi_2)
            + len(varphi_2)
            + len(a)
        )
