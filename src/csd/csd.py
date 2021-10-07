from abc import ABC
from csd.typings import (CSDConfiguration, Backends, RunConfiguration, MeasuringTypes)
from typeguard import typechecked
import strawberryfields as sf
from strawberryfields.api import Result
from strawberryfields.backends import BaseState
from typing import Union, cast
import numpy as np
from csd.config import logger
from tensorflow.python.framework.ops import EagerTensor


class CSD(ABC):
    NUM_SHOTS = 100

    @typechecked
    def __init__(self, csd_config: Union[CSDConfiguration, None] = None):
        if csd_config is not None:
            self._displacement_magnitude = csd_config.get('displacement_magnitude')
            self._steps = csd_config.get('steps')
            self._learning_rate = csd_config.get('learning_rate')
            self._batch_size = csd_config.get('batch_size')
            self._threshold = csd_config.get('threshold')
        self._result = None
        self._run_configuration: Union[RunConfiguration, None] = None

    def single_layer(self, backend: Backends = Backends.FOCK) -> Result:
        """ Creates a single mode quantum "program".
            https://strawberryfields.readthedocs.io/en/stable/code/api/strawberryfields.Program.html
        """

        prog = sf.Program(1)

        # Instantiate the Gaussian backend.
        # https://strawberryfields.readthedocs.io/en/stable/introduction/circuits.html
        eng = sf.Engine(backend=backend.value, backend_options={"cutoff_dim": 5})

        with prog.context as q:
            # Phase space squeezing gate.
            # https://strawberryfields.readthedocs.io/en/stable/code/api/strawberryfields.ops.Sgate.html
            sf.ops.Dgate(self._displacement_magnitude) | q[0]

            # Measures whether a mode contains zero or nonzero photons.
            # https://strawberryfields.readthedocs.io/en/stable/code/api/strawberryfields.ops.MeasureThreshold.html
            sf.ops.MeasureFock() | q[0]

        self._result = eng.run(prog)
        return self._result

    def _run_one_layer_probabilities(self, engine: sf.Engine) -> Union[float, EagerTensor]:
        """Run a one layer experiment

        Args:
            engine (sf.Engine): Strawberry fields already instantiated engine

        Returns:
            float: probability of getting |0> state
        """
        if self._run_configuration is None:
            raise ValueError("Run configuration not specified")
        if self._circuit is None:
            raise ValueError("Circuit MUST be created first")

        self._result = self._run_engine(engine=engine)
        return cast(Result, self._result).state.fock_prob([0])

    def _run_engine_and_compute_probability_0_photons(self, engine: sf.Engine, shots: int) -> Union[float, EagerTensor]:
        return sum([1 for read_value in [self._run_engine(engine=engine).samples[0][0]
                                         for _ in range(0, shots)] if read_value == 0]) / shots

    def _run_engine(self, engine: sf.Engine) -> Result:
        if self._run_configuration is None:
            raise ValueError("Run configuration not specified")

        self._result = engine.run(self._circuit, args={
            "displacement_magnitude": self._run_configuration['displacement_magnitude'],
            "alpha": self._run_configuration['alpha']
        })
        return self._result

    def _run_one_layer_sampling(self, engine: sf.Engine) -> Union[float, EagerTensor]:
        """Run a one layer experiment doing MeasureFock and performing samplint with nshots

        Args:
            engine (sf.Engine): Strawberry fields already instantiated engine

        Returns:
            float: probability of getting |0> state
        """
        if self._run_configuration is None:
            raise ValueError("Run configuration not specified")
        if self._circuit is None:
            raise ValueError("Circuit MUST be created first")
        if 'shots' not in self._run_configuration:
            logger.debug('Using default number of shots: 1000')
            self._run_configuration['shots'] = 1000

        return self._run_engine_and_compute_probability_0_photons(
            engine=engine,
            shots=self._run_configuration['shots'])

    def _create_circuit(self) -> sf.Program:
        """Creates a circuit to run an experiment based on configuration parameters

        Returns:
            sf.Program: the created circuit
        """
        if self._run_configuration is None:
            raise ValueError("Run configuration not specified")

        prog = sf.Program(self._run_configuration['number_qumodes'])
        alpha = prog.params("alpha")
        displacement_magnitude = prog.params("displacement_magnitude")

        with prog.context as q:
            sf.ops.Dgate(alpha, 0.0) | q[0]
            sf.ops.Dgate(displacement_magnitude, 0.0) | q[0]
            if self._run_configuration['measuring_type'] is MeasuringTypes.SAMPLING:
                sf.ops.MeasureFock() | q[0]

        return prog

    def _run_circuit_and_get_result_probabilities(self) -> Union[float, EagerTensor]:
        """Runs an experiment with an already created circuit and
            returns the post-processed probability of getting |0> state

        Returns:
            float: probability of getting |0> state
        """
        if self._run_configuration is None:
            raise ValueError("Run configuration not specified")
        if self._circuit is None:
            raise ValueError("Circuit MUST be created first")

        logger.debug(f"Executing One Layer circuit with Backend: {self._run_configuration['backend'].value}, "
                     f"alpha: {self._run_configuration['alpha']} "
                     f"beta: {self._run_configuration['displacement_magnitude']}"
                     " with measuring_type: "
                     f"{cast(MeasuringTypes, self._run_configuration['measuring_type']).value}")

        engine = sf.Engine(backend=self._run_configuration['backend'].value,
                           backend_options={"cutoff_dim": 7})

        if self._run_configuration['measuring_type'] is MeasuringTypes.SAMPLING:
            return self._run_one_layer_sampling(engine=engine)
        return self._run_one_layer_probabilities(engine=engine)

    @typechecked
    def execute(self, configuration: RunConfiguration) -> Union[float, EagerTensor]:
        """Run an experiment with the given configuration

        Args:
            configuration (RunConfiguration): Specific experiment configuration

        Returns:
            float: probability of getting |0> state
        """
        if configuration['number_layers'] != 1 or configuration['number_qumodes'] != 1:
            raise ValueError('Experiment only available for ONE qumode and ONE layer')

        self._run_configuration = configuration
        self._circuit = self._create_circuit()
        return self._run_circuit_and_get_result_probabilities()

    def show_result(self) -> dict:
        if self._result is None:
            raise ValueError("Circuit not executed yet.")
        sf_result = cast(Result, self._result)
        sf_state = cast(BaseState, sf_result.state)

        return {
            'result': str(sf_result),
            'state': str(sf_state),
            'trace': sf_state.trace(),
            'density_matrix': sf_state.dm(),
            'dm_shape': cast(np.ndarray, sf_state.dm()).shape,
            'samples': sf_result.samples,
            'first_sample': sf_result.samples[0],
            'fock_probability': sf_state.fock_prob([0])
        }
