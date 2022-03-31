from functools import wraps
from time import time
from typing import Dict, List, NamedTuple, Optional, Tuple, Union

from tensorflow.python.framework.ops import EagerTensor
from csd.codeword import CodeWord
from csd.config import logger
import numpy as np
import pickle
from datetime import datetime
import re
import itertools

SECONDS_PER_MINUTE = 60
MINUTES_PER_HOUR = 60
HOURS_PER_DAY = 24


class CodeBookLogInformation(NamedTuple):
    alpha_value: float
    alpha_init_time: float
    codebooks_size: int
    codebook_iteration: int
    codebook_current_iteration_init_time: float


class CodebookLearningStepsLogInformation(NamedTuple):
    total_steps: int
    step_iteration: int
    learning_init_time: float
    step_current_iteration_init_time: float


def set_friendly_time(time_interval: float) -> str:
    days = 0
    hours = 0
    minutes = int(np.floor(time_interval / SECONDS_PER_MINUTE))
    seconds = int(np.floor(time_interval % SECONDS_PER_MINUTE))

    if minutes > MINUTES_PER_HOUR:
        hours = int(np.floor(minutes / MINUTES_PER_HOUR))
        minutes = int(np.floor(minutes % MINUTES_PER_HOUR))
    if hours > HOURS_PER_DAY:
        days = int(np.floor(hours / HOURS_PER_DAY))
        hours = int(np.floor(hours % HOURS_PER_DAY))
    friendly_time = ""

    if days == 1:
        friendly_time += f"{days} day, "
    if days > 1:
        friendly_time += f"{days} days, "
    if hours == 1:
        friendly_time += f"{hours} hour, "
    if hours > 1:
        friendly_time += f"{hours} hours, "
    if minutes == 1:
        friendly_time += f"{minutes} minute and "
    if minutes > 1:
        friendly_time += f"{minutes} minutes and "
    if seconds == 1:
        friendly_time += f"{seconds} second"
    if seconds > 1:
        friendly_time += f"{seconds} seconds"
    if seconds < 1:
        friendly_time += f"{seconds} seconds"
    return friendly_time


def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        # logger.debug('func:%r args:[%r, %r] took: %2.4f sec' % (f.__name__, args, kw, te - ts))
        # logger.debug('%r took: %2.4f' % (f.__name__, te - ts))
        logger.debug(f"{f.__name__}, took: {set_friendly_time(time_interval=te-ts)}")
        return result

    return wrap


def save_object_to_disk(obj: object, name: Optional[str] = None, path: Optional[str] = "") -> None:
    """save result to a file"""
    object_name = name if name is not None else f"{type(obj).__name__}_{set_current_time()}"
    fixed_path = _fix_path(path)

    with open(f"{fixed_path}{object_name}.pkl", "wb") as file:
        pickle.dump(obj, file, pickle.HIGHEST_PROTOCOL)


def _fix_path(path: str = None) -> str:
    if path is None or path == "":
        return "./"
    fixed_path = path
    if not fixed_path.endswith("/"):
        fixed_path += "/"
    if re.match("^[a-zA-Z]", fixed_path):
        fixed_path = f"./{fixed_path}"
    return fixed_path


def load_object_from_file(name: str, path: Optional[str] = ""):
    """load object from a file"""
    fixed_path = _fix_path(path)
    with open(f"{fixed_path}{name}.pkl", "rb") as file:
        return pickle.load(file)


def set_current_time() -> str:
    """Return the current time as string
    with the following format 'YYYYMMDD_HHMMSS'
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def generate_all_codewords_from_codeword(codeword: CodeWord) -> List[CodeWord]:
    letters = [codeword.alpha, -codeword.alpha]
    return [
        CodeWord(size=len(word), alpha_value=codeword.alpha, word=list(word))
        for word in itertools.product(letters, repeat=codeword.size)
    ]


def generate_all_codewords(word_size: int, alpha_value: float) -> List[CodeWord]:
    letters = [alpha_value, -alpha_value]
    return [
        CodeWord(size=len(word), alpha_value=alpha_value, word=list(word))
        for word in itertools.product(letters, repeat=word_size)
    ]


def estimate_remaining_time(
    total_iterations: int,
    current_iteration: int,
    init_time: float,
    current_iteration_init_time: float,
    concept: str = "",
) -> str:
    now = time()
    current_computation_time = now - init_time
    time_interval = (total_iterations - current_iteration) * current_computation_time / current_iteration
    current_iteration_time_interval = set_friendly_time(time_interval=current_computation_time)
    return (
        "\n\n---------------------------------------------------------------------\n"
        f"{concept}[{current_iteration}/{total_iterations}]: in {current_iteration_time_interval}\n"
        f", current iteration in: {set_friendly_time(time_interval=now - current_iteration_init_time)}"
        f", estimated remaining time: {set_friendly_time(time_interval=time_interval)}"
        "\n---------------------------------------------------------------------\n\n"
    )


def print_codebook_log(log_info: CodeBookLogInformation) -> None:
    now = time()
    total_computation_time = now - log_info.alpha_init_time
    current_iteration_time = now - log_info.codebook_current_iteration_init_time
    remaining_time = (
        (log_info.codebooks_size - (log_info.codebook_iteration + 1))
        * total_computation_time
        / (log_info.codebook_iteration + 1)
    )
    text = (
        "\n\n---------------------------------------------------------------------\n"
        f"  ALPHA:{log_info.alpha_value} -> "
        f"CODEBOOK[{log_info.codebook_iteration+1}/{log_info.codebooks_size}]: \n\n"
        f"  TOTAL computation time: {set_friendly_time(time_interval=total_computation_time)}\n"
        f"  Current codebook iteration time: {set_friendly_time(time_interval=current_iteration_time)}\n"
        f"  Estimated codebook REMAINING time: {set_friendly_time(time_interval=remaining_time)}"
        "\n---------------------------------------------------------------------\n\n"
    )
    logger.info(text)


def print_codebook_learning_steps_log(
    codebook_log_info: CodeBookLogInformation, learning_log_info: CodebookLearningStepsLogInformation
) -> None:
    now = time()
    total_computation_time = now - codebook_log_info.alpha_init_time
    current_iteration_time = now - codebook_log_info.codebook_current_iteration_init_time
    remaining_time = (
        (codebook_log_info.codebooks_size - (codebook_log_info.codebook_iteration + 1))
        * total_computation_time
        / (codebook_log_info.codebook_iteration + 1)
    )
    total_learning_computation_time = now - learning_log_info.learning_init_time
    current_learning_iteration_time = set_friendly_time(
        time_interval=(now - learning_log_info.step_current_iteration_init_time)
    )
    learning_remaining_time = (
        (learning_log_info.total_steps - (learning_log_info.step_iteration + 1))
        * total_learning_computation_time
        / (learning_log_info.step_iteration + 1)
    )
    text = (
        "\n\n---------------------------------------------------------------------\n"
        f"  ALPHA:{codebook_log_info.alpha_value} -> "
        f"CODEBOOK[{codebook_log_info.codebook_iteration+1}/{codebook_log_info.codebooks_size}] -> "
        f"LEARNING_STEPS[{learning_log_info.step_iteration+1}/{learning_log_info.total_steps}]: \n\n"
        f"  TOTAL computation time: {set_friendly_time(time_interval=total_computation_time)}\n"
        f"  Current codebook iteration time: {set_friendly_time(time_interval=current_iteration_time)}\n"
        f"  Estimated codebook REMAINING time: {set_friendly_time(time_interval=remaining_time)}\n\n"
        f"  TOTAL learning computation time: {set_friendly_time(time_interval=total_learning_computation_time)}\n"
        f"  Current learning block step iteration time: {current_learning_iteration_time}\n"
        f"  Estimated learning REMAINING time: {set_friendly_time(time_interval=learning_remaining_time)}"
        "\n---------------------------------------------------------------------\n\n"
    )
    logger.info(text)


def strtobool(val: str) -> bool:
    val = val.lower()
    if val in ("y", "yes", "t", "true", "on", "1"):
        return True
    elif val in ("n", "no", "f", "false", "off", "0"):
        return False
    else:
        raise ValueError("invalid truth value %r" % (val,))


def get_index_shape_from_outcome_one_mode(outcome_one_mode: int, cutoff_dim: int) -> Tuple[int, int]:
    if outcome_one_mode != 0 and outcome_one_mode != 1:
        raise ValueError("Outcome mode must be either 0 or 1")
    if outcome_one_mode == 0:
        return (0, 1)
    return (1, cutoff_dim)


def get_index_shape_from_outcome(outcome: Tuple[int, ...], cutoff_dim: int) -> List[int]:
    result: List[int] = []
    for outcome_one_mode in outcome:
        result += get_index_shape_from_outcome_one_mode(outcome_one_mode=outcome_one_mode, cutoff_dim=cutoff_dim)
    return result


def generate_measurement_matrix_one_outcome(
    outcome: Tuple[int, ...], cutoff_dim: int, zeros_matrix: np.ndarray
) -> np.ndarray:

    ones_matrix = np.ones((1, 1), dtype=np.int32)
    indices = get_index_shape_from_outcome(outcome=outcome, cutoff_dim=cutoff_dim)

    final_matrix = zeros_matrix.copy()
    if len(indices) < 2 or len(indices) > 16:
        raise ValueError("modes not supported. Only from 1 to 8 modes supported.")
    if len(indices) == 2:
        final_matrix[indices[0] : indices[1]] = ones_matrix
    if len(indices) == 4:
        final_matrix[indices[0] : indices[1], indices[2] : indices[3]] = ones_matrix
    if len(indices) == 6:
        final_matrix[indices[0] : indices[1], indices[2] : indices[3], indices[4] : indices[5]] = ones_matrix
    if len(indices) == 8:
        final_matrix[
            indices[0] : indices[1],
            indices[2] : indices[3],
            indices[4] : indices[5],
            indices[6] : indices[7],
        ] = ones_matrix
    if len(indices) == 10:
        final_matrix[
            indices[0] : indices[1],
            indices[2] : indices[3],
            indices[4] : indices[5],
            indices[6] : indices[7],
            indices[8] : indices[9],
        ] = ones_matrix
    if len(indices) == 12:
        final_matrix[
            indices[0] : indices[1],
            indices[2] : indices[3],
            indices[4] : indices[5],
            indices[6] : indices[7],
            indices[8] : indices[9],
            indices[10] : indices[11],
        ] = ones_matrix
    if len(indices) == 14:
        final_matrix[
            indices[0] : indices[1],
            indices[2] : indices[3],
            indices[4] : indices[5],
            indices[6] : indices[7],
            indices[8] : indices[9],
            indices[10] : indices[11],
            indices[12] : indices[13],
        ] = ones_matrix
    if len(indices) == 16:
        final_matrix[
            indices[0] : indices[1],
            indices[2] : indices[3],
            indices[4] : indices[5],
            indices[6] : indices[7],
            indices[8] : indices[9],
            indices[10] : indices[11],
            indices[12] : indices[13],
            indices[14] : indices[15],
        ] = ones_matrix

    return final_matrix


def generate_all_outcomes(modes=int) -> List[Tuple[int, ...]]:
    options = [0, 1]
    return [*itertools.product(options, repeat=modes)]


def generate_measurement_matrices(num_modes: int, cutoff_dim: int) -> List[np.ndarray]:
    matrix_shape = [cutoff_dim] * num_modes
    zeros_matrix = np.zeros(matrix_shape, dtype=np.float32)
    outcomes = generate_all_outcomes(modes=num_modes)
    return [
        generate_measurement_matrix_one_outcome(outcome=outcome, cutoff_dim=cutoff_dim, zeros_matrix=zeros_matrix)
        for outcome in outcomes
    ]


def extend_input_codebook_to_output_codebook(input_codebook: List[CodeWord], output_modes: int) -> List[CodeWord]:
    if input_codebook[0].size > output_modes:
        raise ValueError("input modes are greater than output modes.")
    if input_codebook[0].size == output_modes:
        return input_codebook
    ancilla_modes = output_modes - input_codebook[0].size
    ancilla_codewords = generate_all_codewords(word_size=ancilla_modes, alpha_value=input_codebook[0].alpha)
    return [
        CodeWord(word=codeword.word + ancilla_codeword.word)
        for codeword in input_codebook
        for ancilla_codeword in ancilla_codewords
    ]


def filter_outcomes_only_in_codebook(
    outcomes: List[Tuple[int, ...]], codebook: List[CodeWord]
) -> List[Tuple[int, ...]]:
    output_codebook = extend_input_codebook_to_output_codebook(input_codebook=codebook, output_modes=len(outcomes[0]))
    all_codewords = generate_all_codewords_from_codeword(codeword=output_codebook[0])
    if len(all_codewords) != len(outcomes):
        raise ValueError("outcomes and all_codewords size mismatch.")

    return [outcomes[all_codewords.index(codeword)] for codeword in output_codebook]


def generate_measurement_matrices_only_in_codebook(
    num_modes: int, cutoff_dim: int, codebook: List[CodeWord]
) -> List[np.ndarray]:
    matrix_shape = [cutoff_dim] * num_modes
    zeros_matrix = np.zeros(matrix_shape, dtype=np.float32)
    outcomes = generate_all_outcomes(modes=num_modes)
    outcomes_in_codebook = filter_outcomes_only_in_codebook(outcomes, codebook)
    return [
        generate_measurement_matrix_one_outcome(outcome=outcome, cutoff_dim=cutoff_dim, zeros_matrix=zeros_matrix)
        for outcome in outcomes_in_codebook
    ]


def filter_number_modes_from_codebook(codebook: List[CodeWord]) -> int:
    number_codebooks = len(codebook)
    if number_codebooks <= 0:
        return 0
    orig_number_modes = codebook[0].size
    if number_codebooks == 1:
        return orig_number_modes
    binary_codebook = [codeword.binary_code for codeword in codebook]
    summed_codes = list(map(sum, zip(*binary_codebook)))
    zero_constant_modes = summed_codes.count(0)
    one_constant_modes = summed_codes.count(number_codebooks)

    return orig_number_modes - zero_constant_modes - one_constant_modes


def create_optimized_parameters_to_print(
    modes: int, optimized_parameters: List[float], circuit_parameters: List[str]
) -> Dict[str, Union[List[float], EagerTensor]]:

    return dict(zip(circuit_parameters[modes:], optimized_parameters))
