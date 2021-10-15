from csd import CSD
from csd.typings import MeasuringTypes, RunConfiguration, CSDConfiguration, Backends
from csd.config import logger
import numpy as np


def test_tf() -> None:
    alphas = list(np.arange(0.05, 2.1, 0.05))
    # logger.info(alphas)
    # alphas = [0.05, 0.7]

    csd_configuration = CSDConfiguration({
        'cutoff_dim': 10,
        'steps': 10
    })
    csd = CSD(csd_config=csd_configuration)
    run_configuration = RunConfiguration({
        'alphas': alphas,
        'backend': Backends.TENSORFLOW,
        'number_qumodes': 1,
        'number_layers': 1,
        'measuring_type': MeasuringTypes.PROBABILITIES,
        'codeword_size': 10,
        'cutoff_dim': 10,
        'steps': 10
    })
    result = csd.execute(configuration=run_configuration)
    logger.info(result)


if __name__ == '__main__':
    test_tf()
