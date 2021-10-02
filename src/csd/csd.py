from abc import ABC
from csd.typings import CSDConfiguration
from typeguard import typechecked
import strawberryfields as sf


class CSD(ABC):
    NUM_SHOTS = 100

    @typechecked
    def __init__(self, csd_config: CSDConfiguration):
        self._displacement_magnitude = csd_config.get('displacement_magnitude')
        self._steps = csd_config.get('steps')
        self._learning_rate = csd_config.get('learning_rate')
        self._batch_size = csd_config.get('batch_size')
        self._threshold = csd_config.get('threshold')

    def single_layer(self):
        """ Creates a single mode quantum "program".
            https://strawberryfields.readthedocs.io/en/stable/code/api/strawberryfields.Program.html
        """

        prog = sf.Program(1)

        # Instantiate the Gaussian backend.
        # https://strawberryfields.readthedocs.io/en/stable/introduction/circuits.html
        eng = sf.Engine("tf", backend_options={"cutoff_dim": 5})

        with prog.context as q:
            # Phase space squeezing gate.
            # https://strawberryfields.readthedocs.io/en/stable/code/api/strawberryfields.ops.Sgate.html
            sf.ops.Dgate(self._displacement_magnitude) | q[0]

            # Measures whether a mode contains zero or nonzero photons.
            # https://strawberryfields.readthedocs.io/en/stable/code/api/strawberryfields.ops.MeasureThreshold.html
            sf.ops.MeasureFock() | q[0]

        return eng.run(prog)
