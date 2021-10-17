# engine.py
from abc import ABC
import strawberryfields as sf
from strawberryfields.api import Result
from typeguard import typechecked
from csd.typings import Backends, BackendOptions, EngineRunOptions
from typing import Optional, cast
from .circuit import Circuit


class Engine(ABC):
    """ Photonic Backend Engine class

    """

    @typechecked
    def __init__(self, backend: Backends, options: Optional[BackendOptions] = None) -> None:
        self._backend = backend
        self._engine = sf.Engine(backend=self._backend.value,
                                 backend_options=options)

    @typechecked
    def run(self,
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
        alpha_list = cast(list, (options['sample_or_batch']
                                 if type(options['sample_or_batch']) is list
                                 else [options['sample_or_batch']]))
        all_values = alpha_list + options['params']

        return {name: value for (name, value) in zip(circuit.parameters, all_values)}
