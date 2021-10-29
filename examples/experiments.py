from csd import CSD
from csd.typings.typing import MeasuringTypes, CSDConfiguration, Backends
import numpy as np
from csd.util import timing
import json


@timing
def execute_probabilities_fock_backend(csd: CSD) -> None:
    results = csd.execute_all_backends_and_measuring_types(
        backends=[Backends.FOCK],
        measuring_types=[MeasuringTypes.PROBABILITIES]
    )
    for result in results:
        # print(json.dumps(result, indent=2))
        print(result)


@timing
def execute_probabilities_gaussian_backend(csd: CSD) -> None:
    csd.execute_all_backends_and_measuring_types(
        backends=[Backends.GAUSSIAN],
        measuring_types=[MeasuringTypes.PROBABILITIES]
    )


@timing
def execute_probabilities_tf_backend(csd: CSD) -> None:
    csd.execute_all_backends_and_measuring_types(
        backends=[Backends.TENSORFLOW],
        measuring_types=[MeasuringTypes.PROBABILITIES]
    )


@timing
def execute_sampling_fock_backend(csd: CSD) -> None:
    csd.execute_all_backends_and_measuring_types(
        backends=[Backends.FOCK],
        measuring_types=[MeasuringTypes.SAMPLING]
    )


@timing
def execute_sampling_gaussian_backend(csd: CSD) -> None:
    csd.execute_all_backends_and_measuring_types(
        backends=[Backends.GAUSSIAN],
        measuring_types=[MeasuringTypes.SAMPLING]
    )


@timing
def execute_sampling_tf_backend(csd: CSD) -> None:
    csd.execute_all_backends_and_measuring_types(
        backends=[Backends.TENSORFLOW],
        measuring_types=[MeasuringTypes.SAMPLING]
    )


if __name__ == '__main__':
    # alphas = list(np.arange(0.05, 1.05, 0.05))
    alphas = [0.7]
    csd = CSD(csd_config=CSDConfiguration({
        'alphas': alphas,
        'steps': 500,
        'learning_rate': 0.1,
        'batch_size': 10,
        'shots': 100,
        'plays': 1,
        'cutoff_dim': 10,
        'architecture': {
            'number_modes': 1,
            'number_layers': 1,
            'squeezing': False,
        },
        'save_results': False,
        'save_plots': False
    }))
    # execute_probabilities_fock_backend(csd=csd)
    # execute_probabilities_gaussian_backend(csd=csd)
    execute_probabilities_tf_backend(csd=csd)
    # execute_sampling_fock_backend(csd=csd)
    # execute_sampling_gaussian_backend(csd=csd)
    # execute_sampling_tf_backend(csd=csd)
