from csd import CSD
from csd.typings import MeasuringTypes, RunConfiguration, CSDConfiguration, Backends
from csd.config import logger
# import numpy as np
import json


def test_tf() -> None:
    # alphas = list(np.arange(0.05, 2.1, 0.05))
    alphas = [0.7]

    csd_configuration = CSDConfiguration({
        'cutoff_dim': 10,
        'steps': 500
    })
    csd = CSD(csd_config=csd_configuration)
    run_configuration = RunConfiguration({
        'alphas': alphas,
        'backend': Backends.TENSORFLOW,
        'number_qumodes': 1,
        'number_layers': 1,
        'measuring_type': MeasuringTypes.PROBABILITIES,
        'codeword_size': 10,
    })
    result = csd.execute(configuration=run_configuration)
    logger.info(result)


def test_gaus_sampling() -> None:
    # alphas = list(np.arange(0.05, 2.1, 0.05))
    alphas = [0.7]

    csd_configuration = CSDConfiguration({
        'batch_size': 2,
        'shots': 100
    })
    csd = CSD(csd_config=csd_configuration)
    run_configuration = RunConfiguration({
        'alphas': alphas,
        'backend': Backends.GAUSSIAN,
        'number_qumodes': 1,
        'number_layers': 1,
        'measuring_type': MeasuringTypes.SAMPLING,
    })
    result = csd.execute(configuration=run_configuration)
    logger.info(result)


def test_sampling() -> None:
    # alphas = list(np.arange(0.05, 2.1, 0.05))
    alphas = [0.7]

    csd_configuration = CSDConfiguration({
        'batch_size': 10,
        'shots': 100,
        'cutoff_dim': 10
    })
    csd = CSD(csd_config=csd_configuration)
    run_configuration = RunConfiguration({
        'alphas': alphas,
        'backend': Backends.FOCK,
        'number_qumodes': 1,
        'number_layers': 1,
        'measuring_type': MeasuringTypes.SAMPLING,
    })
    result = csd.execute(configuration=run_configuration)
    logger.info(json.dumps(result, indent=2))


def test_tf_2() -> None:
    # alphas = list(np.arange(0.05, 1.1, 0.05))
    alphas = [0.7]

    csd = CSD(csd_config=CSDConfiguration({
        'steps': 500,
        'cutoff_dim': 10
    }))

    csd.plot_success_probabilities(alphas=alphas)
    csd.execute_all_backends_and_measuring_types(
        alphas=alphas,
        measuring_types=[MeasuringTypes.PROBABILITIES]
    )
    csd.plot_success_probabilities(measuring_types=[MeasuringTypes.PROBABILITIES])


if __name__ == '__main__':
    # test_tf()
    # test_gaus_sampling()
    test_sampling()
    # test_tf_2()
