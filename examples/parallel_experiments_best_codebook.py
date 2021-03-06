import itertools
from multiprocessing import Pool  # , cpu_count
from time import time  # , sleep
from typing import Iterator, List, NamedTuple
from csd import CSD
from csd.global_result_manager import GlobalResultManager
from csd.plot import Plot
from csd.typings.typing import (
    CutOffDimensions,
    LearningRate,
    LearningSteps,
    MeasuringTypes,
    CSDConfiguration,
    Backends,
    OneProcessResultExecution,
    OptimizationBackends,
    ResultExecution,
    RunConfiguration,
)
import numpy as np
from csd.utils.util import timing
import os

# from csd.config import logger

TESTING = True
PATH_RESULTS = GlobalResultManager(testing=TESTING)._base_dir_path


class MultiProcessConfiguration(NamedTuple):
    alphas: List[float]
    learning_steps: List[LearningSteps]
    learning_rate: List[LearningRate]
    batch_size: List[int]
    shots: List[int]
    plays: List[int]
    cutoff_dim: List[CutOffDimensions]
    number_input_modes: List[int]
    number_layers: List[int]
    squeezing: List[bool]
    number_ancillas: List[int]
    max_combinations: List[int]
    binary_codebook: List[List[List[int]]]


class LaunchExecutionConfiguration(NamedTuple):
    launch_backend: Backends
    measuring_type: MeasuringTypes
    learning_steps: LearningSteps
    learning_rate: LearningRate
    batch_size: int
    shots: int
    plays: int
    cutoff_dim: CutOffDimensions
    number_input_modes: int
    number_layers: int
    squeezing: bool
    number_ancillas: int
    alpha: float
    max_combinations: int
    binary_codebook: List[int]


def _set_plot_label(plot_label_backend: Backends, measuring_type: MeasuringTypes) -> str:
    """Set the label for the success probability plot

    Args:
        backend (Backends): Current experiment backend
        measuring_type (MeasuringTypes): Current experiment measuring type

    Returns:
        str: the determined label
    """
    if plot_label_backend is Backends.FOCK and measuring_type is MeasuringTypes.PROBABILITIES:
        return "pFockProb(a)"
    if plot_label_backend is Backends.GAUSSIAN and measuring_type is MeasuringTypes.PROBABILITIES:
        return "pGausProb(a)"
    if plot_label_backend is Backends.TENSORFLOW and measuring_type is MeasuringTypes.PROBABILITIES:
        return "pTFProb(a)"
    if plot_label_backend is Backends.FOCK and measuring_type is MeasuringTypes.SAMPLING:
        return "pFockSampl(a)"
    if plot_label_backend is Backends.TENSORFLOW and measuring_type is MeasuringTypes.SAMPLING:
        return "pTFSampl(a)"
    if plot_label_backend is Backends.GAUSSIAN and measuring_type is MeasuringTypes.SAMPLING:
        return "pGausSampl(a)"
    raise ValueError(
        f"Values not supported. backend: {plot_label_backend.value} and measuring_type: {measuring_type.value}"
    )


def _set_plot_title(
    plot_title_backend: Backends,
    measuring_type: MeasuringTypes,
    batch_size: int,
    plays: int,
    modes: int,
    layers: int,
    squeezing: bool,
    ancillas: int,
    learning_rate: LearningRate,
    learning_steps: LearningSteps,
    cutoff_dim: CutOffDimensions,
) -> str:

    return (
        f"backend:{plot_title_backend.value}, "
        f"measuring:{measuring_type.value}, \n"
        f"batch size:{batch_size}, plays:{plays}, modes:{modes}, ancillas:{ancillas}, \n"
        f"steps: {learning_steps}, l_rate: {learning_rate}, cutoff_dim: {cutoff_dim}, \n"
        f"layers,{layers}, squeezing:{squeezing}"
    )


@timing
def launch_execution(configuration: LaunchExecutionConfiguration) -> ResultExecution:
    csd = CSD(
        csd_config=CSDConfiguration(
            {
                "alphas": [configuration.alpha],
                "learning_steps": configuration.learning_steps,
                "learning_rate": configuration.learning_rate,
                "batch_size": configuration.batch_size,
                "shots": configuration.shots,
                "plays": configuration.plays,
                "cutoff_dim": configuration.cutoff_dim,
                "architecture": {
                    "number_modes": configuration.number_input_modes,
                    "number_ancillas": configuration.number_ancillas,
                    "number_layers": configuration.number_layers,
                    "squeezing": configuration.squeezing,
                },
                "save_results": False,
                "save_plots": False,
                "parallel_optimization": True,
                "max_combinations": configuration.max_combinations,
            }
        )
    )
    return csd.execute(
        configuration=RunConfiguration(
            {
                "run_backend": configuration.launch_backend,
                "optimization_backend": OptimizationBackends.TENSORFLOW,
                "measuring_type": configuration.measuring_type,
                "binary_codebook": configuration.binary_codebook,
            }
        )
    )


def uncurry_launch_execution(t) -> ResultExecution:
    one_execution_configuration = LaunchExecutionConfiguration(
        launch_backend=t[0],
        measuring_type=t[1],
        learning_steps=t[2],
        learning_rate=t[3],
        batch_size=t[4],
        shots=t[5],
        plays=t[6],
        cutoff_dim=t[7],
        number_input_modes=t[8],
        number_layers=t[9],
        squeezing=t[10],
        number_ancillas=t[11],
        alpha=t[12],
        max_combinations=t[13],
        binary_codebook=t[14],
    )
    return launch_execution(configuration=one_execution_configuration)


def update_execution_result(
    acumulated_one_process_result: OneProcessResultExecution,
    input_result: ResultExecution,
) -> OneProcessResultExecution:
    new_one_process_result = acumulated_one_process_result.copy()

    for opt_param, p_err, p_succ, p_hels, p_homo in zip(
        input_result["opt_params"],
        input_result["p_err"],
        input_result["p_succ"],
        input_result["p_helstrom"],
        input_result["p_homodyne"],
    ):
        new_one_process_result["opt_params"].append(opt_param)
        new_one_process_result["p_err"].append(p_err)
        new_one_process_result["p_succ"].append(p_succ)
        new_one_process_result["p_helstrom"].append(p_hels)
        new_one_process_result["p_homodyne"].append(p_homo)

    return new_one_process_result


def create_full_execution_result(
    full_backend: Backends,
    measuring_type: MeasuringTypes,
    multiprocess_configuration: MultiProcessConfiguration,
    results: List[ResultExecution],
) -> ResultExecution:
    acumulated_one_process_result = OneProcessResultExecution(
        {
            "opt_params": [],
            "p_err": [],
            "p_succ": [],
            "p_helstrom": [],
            "p_homodyne": [],
        }
    )
    for result in results:
        acumulated_one_process_result = update_execution_result(
            acumulated_one_process_result=acumulated_one_process_result,
            input_result=result,
        )

    return ResultExecution(
        {
            "alphas": multiprocess_configuration.alphas,
            "batches": [],
            "opt_params": acumulated_one_process_result["opt_params"],
            "p_err": acumulated_one_process_result["p_err"],
            "p_succ": acumulated_one_process_result["p_succ"],
            "result_backend": full_backend.value,
            "measuring_type": measuring_type.value,
            "plot_label": _set_plot_label(plot_label_backend=full_backend, measuring_type=measuring_type),
            "plot_title": _set_plot_title(
                plot_title_backend=full_backend,
                measuring_type=measuring_type,
                batch_size=multiprocess_configuration.batch_size[0],
                plays=multiprocess_configuration.plays[0],
                modes=multiprocess_configuration.number_input_modes[0],
                layers=multiprocess_configuration.number_layers[0],
                squeezing=multiprocess_configuration.squeezing[0],
                ancillas=multiprocess_configuration.number_ancillas[0],
                learning_rate=multiprocess_configuration.learning_rate[0],
                learning_steps=multiprocess_configuration.learning_steps[0],
                cutoff_dim=multiprocess_configuration.cutoff_dim[0],
            ),
            "total_time": 0.0,
            "p_helstrom": acumulated_one_process_result["p_helstrom"],
            "p_homodyne": acumulated_one_process_result["p_homodyne"],
            "number_modes": multiprocess_configuration.number_input_modes,
        }
    )


def plot_results(alphas: List[float], execution_result: ResultExecution, number_input_modes: int) -> None:
    plot = Plot(alphas=alphas, number_modes=number_input_modes, path=PATH_RESULTS)
    plot.plot_success_probabilities(executions=[execution_result], save_plot=True)


@timing
def _general_execution(
    multiprocess_configuration: MultiProcessConfiguration,
    backend: Backends,
    measuring_type: MeasuringTypes,
):
    start_time = time()
    pool = Pool(3)
    # pool = Pool(number_points_to_plot if number_points_to_plot <= cpu_count() else cpu_count())
    execution_results = pool.map_async(
        func=uncurry_launch_execution,
        iterable=_build_iterator(multiprocess_configuration, backend, measuring_type),
    ).get()

    result = create_full_execution_result(
        full_backend=backend,
        measuring_type=measuring_type,
        multiprocess_configuration=multiprocess_configuration,
        results=execution_results,
    )
    pool.close()
    pool.join()

    _update_result_with_total_time(result=result, start_time=start_time)
    plot_results(
        alphas=multiprocess_configuration.alphas,
        execution_result=result,
        number_input_modes=multiprocess_configuration.number_input_modes[0],
    )


def _update_result_with_total_time(result: ResultExecution, start_time: float) -> None:
    end_time = time()
    result["total_time"] = end_time - start_time


def _build_iterator(
    multiprocess_configuration: MultiProcessConfiguration,
    backend: Backends,
    measuring_type: MeasuringTypes,
) -> Iterator:
    return zip(
        [backend] * number_alphas,
        [measuring_type] * number_alphas,
        multiprocess_configuration.learning_steps,
        multiprocess_configuration.learning_rate,
        multiprocess_configuration.batch_size,
        multiprocess_configuration.shots,
        multiprocess_configuration.plays,
        multiprocess_configuration.cutoff_dim,
        multiprocess_configuration.number_input_modes,
        multiprocess_configuration.number_layers,
        multiprocess_configuration.squeezing,
        multiprocess_configuration.number_ancillas,
        multiprocess_configuration.alphas,
        multiprocess_configuration.max_combinations,
        multiprocess_configuration.binary_codebook,
    )


def multi_fock_backend(multiprocess_configuration: MultiProcessConfiguration) -> None:

    backend = Backends.FOCK
    measuring_type = MeasuringTypes.PROBABILITIES

    _general_execution(
        multiprocess_configuration=multiprocess_configuration,
        backend=backend,
        measuring_type=measuring_type,
    )


def multi_tf_backend(multiprocess_configuration: MultiProcessConfiguration) -> None:
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    backend = Backends.TENSORFLOW
    measuring_type = MeasuringTypes.PROBABILITIES

    _general_execution(
        multiprocess_configuration=multiprocess_configuration,
        backend=backend,
        measuring_type=measuring_type,
    )


if __name__ == "__main__":
    alpha_init = 0.1
    alpha_end = 1.4
    number_points_to_plot = 16
    alpha_step = (alpha_end - alpha_init) / number_points_to_plot
    alphas = list(np.arange(alpha_init, alpha_end, alpha_step))
    # alphas.pop(5)
    # one_alpha = alphas[5]
    # alphas = [alphas[8]]
    # alphas = alphas[:-3]
    # alphas = [alphas[3], alphas[4], alphas[5]]
    # alphas = [alphas[4]]
    alphas = [alphas[4], alphas[5]]

    # list_number_input_modes = list(range(6, 11))

    list_number_input_modes = [4]
    # list_number_input_modes = [4]
    list_squeezing = [False]
    list_number_ancillas = [3]
    shots = 1
    plays = 1
    number_layers = 1
    max_combinations = 200000
    # binary_codebook = [[0, 0, 0], [1, 1, 1]] # modes=3
    # binary_codebook = [[0, 0, 0, 0], [1, 1, 1, 1]]  # modes=4 ancilla=0,1 alphas[3]
    binary_codebook = [[0, 0, 0, 0], [0, 1, 1, 1], [1, 0, 1, 1], [1, 1, 0, 0]]  # modes=4 ancilla=0,1,2,3 alphas[4]
    # binary_codebook = [[0, 0, 0, 0], [0, 1, 1, 1], [1, 0, 1, 1], [1, 1, 0, 0]] # modes=4 ancilla=0,1,2,3 alphas[5]

    for number_input_modes, squeezing_option, number_ancillas in itertools.product(
        list_number_input_modes, list_squeezing, list_number_ancillas
    ):
        learning_steps = LearningSteps(default=100, high=150, extreme=300)
        learning_rate = LearningRate(default=0.1, high=0.01, extreme=0.01)
        cutoff_dim = CutOffDimensions(default=7, high=7, extreme=7)

        batch_size = 2**number_input_modes
        number_alphas = len(alphas)

        print(f"number alphas: {number_alphas}")
        print(f"number_input_modes: {number_input_modes}")
        print(f"squeezing_option: {squeezing_option}")
        # seconds_to_sleep = 2 * 60 * 60
        # print(f'going to sleep for {seconds_to_sleep}')
        # sleep(seconds_to_sleep)

        multiprocess_configuration = MultiProcessConfiguration(
            alphas=alphas,
            learning_steps=[learning_steps] * number_alphas,
            learning_rate=[learning_rate] * number_alphas,
            batch_size=[batch_size] * number_alphas,
            shots=[shots] * number_alphas,
            plays=[plays] * number_alphas,
            cutoff_dim=[cutoff_dim] * number_alphas,
            number_input_modes=[number_input_modes] * number_alphas,
            number_layers=[number_layers] * number_alphas,
            squeezing=[squeezing_option] * number_alphas,
            number_ancillas=[number_ancillas] * number_alphas,
            max_combinations=[max_combinations] * number_alphas,
            binary_codebook=[binary_codebook] * number_alphas,
        )

        multi_tf_backend(multiprocess_configuration=multiprocess_configuration)
        # multi_fock_backend(multiprocess_configuration=multiprocess_configuration)
