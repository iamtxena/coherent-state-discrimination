# engine.py
from abc import ABC
import strawberryfields as sf
from strawberryfields.api import Result
from tensorflow.python.framework.ops import EagerTensor
from typeguard import typechecked
from csd.typings import Backends, BackendOptions, EngineRunOptions, MeasuringTypes
from typing import Optional, Union
from .circuit import Circuit
from nptyping import NDArray
import numpy as np


class Engine(ABC):
    """ Photonic Backend Engine class

    """

    @typechecked
    def __init__(self, backend: Backends, options: Optional[BackendOptions] = None) -> None:
        self._backend = backend
        self._engine = sf.Engine(backend=self._backend.value,
                                 backend_options=options)

    @property
    def backend_name(self) -> str:
        return self._engine.backend_name

    def run_circuit_checking_measuring_type(
            self,
            circuit: Circuit,
            options: EngineRunOptions) -> Union[list[float], EagerTensor, NDArray[np.float]]:
        if options['measuring_type'] is MeasuringTypes.SAMPLING:
            return self._run_circuit_sampling(circuit=circuit, options=options)
        return self._run_circuit_probabilities(circuit=circuit, options=options)

    def _run_circuit_sampling(self,
                              circuit: Circuit,
                              options: EngineRunOptions) -> Union[list[float], EagerTensor, NDArray[np.float]]:
        """Run a circuit experiment doing MeasureFock and performing samplint with nshots

        Returns:
            float: probability of getting |0> state
        """
        if self._engine.backend_name == Backends.GAUSSIAN.value:
            return sum([1 for read_value in self._run_circuit(circuit=circuit, options=options).samples
                        if read_value[0] == 0]) / options['shots']

        return sum([1 for read_value in [self._run_circuit(circuit=circuit, options=options).samples[0][0]
                                         for _ in range(options['shots'])] if read_value == 0]) / options['shots']

    def _run_circuit_probabilities(self,
                                   circuit: Circuit,
                                   options: EngineRunOptions) -> Union[list[float], EagerTensor, NDArray[np.float]]:
        """Run a circuit experiment computing the fock probability

        Returns:
            float: probability of getting |0> state
        """
        options['shots'] = 0
        return self._run_circuit(circuit=circuit, options=options).state.fock_prob([0])

    def _run_circuit(self,
                     circuit: Circuit,
                     options: EngineRunOptions) -> Result:
        """ Run an experiment using the engine with the passed options
        """
        # reset the engine if it has already been executed
        if self._engine.run_progs:
            self._engine.reset()

        return self._engine.run(program=circuit.circuit,
                                args=self._parse_circuit_parameters(
                                    circuit=circuit,
                                    options=options),
                                shots=options['shots'])

    def _parse_circuit_parameters(self,
                                  circuit: Circuit,
                                  options: EngineRunOptions) -> dict:
        all_values = []
        all_values.append(options['batch_or_codeword'].to_list())
        [all_values.append(param) for param in options['params']]

        return {name: value for (name, value) in zip(circuit.parameters.keys(), all_values)}
