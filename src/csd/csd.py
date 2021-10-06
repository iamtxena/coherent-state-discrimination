from abc import ABC
from csd.typings import CSDConfiguration, Backends, RunConfiguration
from typeguard import typechecked
import strawberryfields as sf
from strawberryfields.api import Result
from strawberryfields.backends import BaseState
from typing import Union, cast
import numpy as np
from csd.config import logger


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

    @typechecked
    def _run_one_layer(self) -> Result:
        """Run a one layer experiment

        Args:
            configuration (RunConfiguration): Specific experiment configuration

        Returns:
            Result: Result of a computation. More info at:
            https://strawberryfields.readthedocs.io/en/stable/code/api/strawberryfields.api.Result.html
        """
        if self._run_configuration is None:
            raise ValueError("Run configuration not specified")

        logger.debug(f"Executing One Layer circuit with Backend: {self._run_configuration['backend'].value}, "
                     f"alpha: {self._run_configuration['alpha']} "
                     f"and beta: {self._run_configuration['displacement_magnitude']}")

        prog = sf.Program(self._run_configuration['number_qumodes'])
        alpha = prog.params("alpha")
        displacement_magnitude = prog.params("displacement_magnitude")

        eng = sf.Engine(backend=self._run_configuration['backend'].value,
                        backend_options={"cutoff_dim": 7})

        with prog.context as q:
            sf.ops.Dgate(alpha, 0.0) | q[0]
            sf.ops.Dgate(displacement_magnitude, 0.0) | q[0]
            sf.ops.MeasureFock() | q[0]

        return eng.run(prog, args={
            "displacement_magnitude": self._run_configuration['displacement_magnitude'],
            "alpha": self._run_configuration['alpha']
        })

    @typechecked
    def execute(self, configuration: RunConfiguration) -> Result:
        """Run an experiment with the given configuration

        Args:
            configuration (RunConfiguration): Specific experiment configuration

        Returns:
            Result: Result of a computation. More info at:
            https://strawberryfields.readthedocs.io/en/stable/code/api/strawberryfields.api.Result.html
        """
        if configuration['number_layers'] != 1 or configuration['number_qumodes'] != 1:
            raise ValueError('Experiment only available for ONE qumode and ONE layer')

        self._run_configuration = configuration
        self._result = self._run_one_layer()
        return self._result

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
