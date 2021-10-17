from typing import List
from csd import CSD
from csd.typings import MeasuringTypes, CSDConfiguration, Backends
import numpy as np
from csd.util import timing


@timing
def execute_probabilities_fock_backend(alphas: List[float], csd: CSD) -> None:
    csd.execute_all_backends_and_measuring_types(
        alphas=alphas,
        backends=[Backends.FOCK],
        measuring_types=[MeasuringTypes.PROBABILITIES]
    )


@timing
def execute_probabilities_gaussian_backend(alphas: List[float], csd: CSD) -> None:
    csd.execute_all_backends_and_measuring_types(
        alphas=alphas,
        backends=[Backends.GAUSSIAN],
        measuring_types=[MeasuringTypes.PROBABILITIES]
    )


@timing
def execute_probabilities_tf_backend(alphas: List[float], csd: CSD) -> None:
    csd.execute_all_backends_and_measuring_types(
        alphas=alphas,
        backends=[Backends.TENSORFLOW],
        measuring_types=[MeasuringTypes.PROBABILITIES]
    )


@timing
def execute_sampling_fock_backend(alphas: List[float], csd: CSD) -> None:
    csd.execute_all_backends_and_measuring_types(
        alphas=alphas,
        backends=[Backends.FOCK],
        measuring_types=[MeasuringTypes.SAMPLING]
    )


@timing
def execute_sampling_gaussian_backend(alphas: List[float], csd: CSD) -> None:
    csd.execute_all_backends_and_measuring_types(
        alphas=alphas,
        backends=[Backends.GAUSSIAN],
        measuring_types=[MeasuringTypes.SAMPLING]
    )


@timing
def execute_sampling_tf_backend(alphas: List[float], csd: CSD) -> None:
    csd.execute_all_backends_and_measuring_types(
        alphas=alphas,
        backends=[Backends.TENSORFLOW],
        measuring_types=[MeasuringTypes.SAMPLING]
    )


if __name__ == '__main__':
    alphas = list(np.arange(0.05, 1.55, 0.05))
    csd = CSD(csd_config=CSDConfiguration({
        'steps': 500,
        'cutoff_dim': 10,
        'batch_size': 1000,
        'architecture': {
            'displacement': True,
            'squeezing': False,
        },
        'save_results': True,
        'save_plots': True
    }))
    execute_probabilities_fock_backend(alphas=alphas, csd=csd)
    execute_probabilities_gaussian_backend(alphas=alphas, csd=csd)
    # execute_probabilities_tf_backend(alphas=alphas, csd=csd)
    execute_sampling_fock_backend(alphas=alphas, csd=csd)
    execute_sampling_gaussian_backend(alphas=alphas, csd=csd)
    # execute_sampling_tf_backend(alphas=alphas, csd=csd)
