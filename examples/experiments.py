import multiprocessing
from csd import CSD
from csd.typings.typing import CutOffDimensions, LearningRate, LearningSteps, MeasuringTypes, CSDConfiguration, Backends
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
    alpha_init = 0.1
    alpha_end = 1.4
    number_points_to_plot = 16
    alpha_step = (alpha_end - alpha_init) / number_points_to_plot
    alphas = list(np.arange(alpha_init, alpha_end, alpha_step))
    # alphas = [0.1]
    # alphas = alphas[:13]

    learning_steps = LearningSteps(default=60,
                                   high=100,
                                   extreme=1000)
    learning_rate = LearningRate(default=0.1,
                                 high=0.1,
                                 extreme=0.1)
    cutoff_dim = CutOffDimensions(default=7,
                                  high=10,
                                  extreme=30)

    number_input_modes = 1
    number_ancillas = 0
    squeezing = False

    batch_size = 2**number_input_modes
    shots = 10
    plays = 1
    number_layers = 1

    number_alphas = len(alphas)

    print(f'number alphas: {number_alphas}')

    csd = CSD(csd_config=CSDConfiguration({
        'alphas': alphas,
        'learning_steps': learning_steps,
        'learning_rate': learning_rate,
        'batch_size': batch_size,
        'shots': shots,
        'plays': plays,
        'cutoff_dim': cutoff_dim,
        'architecture': {
            'number_modes': number_input_modes,
            'number_ancillas': number_ancillas,
            'number_layers': number_layers,
            'squeezing': squeezing,
        },
        'save_results': False,
        'save_plots': True,
        'parallel_optimization': False
    }))
    # execute_probabilities_fock_backend(csd=csd)
    # execute_probabilities_gaussian_backend(csd=csd)
    # execute_probabilities_tf_backend(csd=csd)
    # execute_sampling_fock_backend(csd=csd)
    # execute_sampling_gaussian_backend(csd=csd)
    execute_sampling_tf_backend(csd=csd)
