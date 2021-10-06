from abc import ABC
from csd.typings import CSDConfiguration, Backends
from typeguard import typechecked
import strawberryfields as sf
from strawberryfields.api import Result
from strawberryfields.backends import BaseState
from typing import cast
import numpy as np


class CSD(ABC):
    NUM_SHOTS = 100

    @typechecked
    def __init__(self, csd_config: CSDConfiguration):
        self._displacement_magnitude = csd_config.get('displacement_magnitude')
        self._steps = csd_config.get('steps')
        self._learning_rate = csd_config.get('learning_rate')
        self._batch_size = csd_config.get('batch_size')
        self._threshold = csd_config.get('threshold')
        self._result = None

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
