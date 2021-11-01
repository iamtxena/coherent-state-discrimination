import multiprocessing
from csd import CSD
from csd.typings.typing import MeasuringTypes, CSDConfiguration, Backends
import numpy as np
from csd.util import timing
import os


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
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
    multiprocessing.set_start_method('spawn', force=True)

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
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

    csd.execute_all_backends_and_measuring_types(
        backends=[Backends.TENSORFLOW],
        measuring_types=[MeasuringTypes.SAMPLING]
    )


if __name__ == '__main__':
    alpha_init = 0.05
    alpha_end = 1.05
    number_points_to_plot = 10
    alpha_step = (alpha_end - alpha_init) / number_points_to_plot
    # alphas = list(np.arange(0.05, 1.05, 0.05))
    alphas = list(np.arange(alpha_init, alpha_end, alpha_step))

    csd = CSD(csd_config=CSDConfiguration({
        'alphas': alphas,
        'steps': 300,
        'learning_rate': 0.1,
        'batch_size': 100,
        'shots': 100,
        'plays': 1,
        'cutoff_dim': 10,
        'architecture': {
            'number_modes': 1,
            'number_layers': 1,
            'squeezing': False,
        },
        'save_results': False,
        'save_plots': True,
        'parallel_optimization': True
    }))
    # execute_probabilities_fock_backend(csd=csd)
    # execute_probabilities_gaussian_backend(csd=csd)
    execute_probabilities_tf_backend(csd=csd)
    # execute_sampling_fock_backend(csd=csd)
    # execute_sampling_gaussian_backend(csd=csd)
    # execute_sampling_tf_backend(csd=csd)
