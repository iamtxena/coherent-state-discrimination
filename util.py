"""
Home to various utility functions.
"""

from itertools import product
import numpy as np

def get_index_shape_from_outcome_single_mode(outcome_single_mode, cutoff_dimension):
    if outcome_single_mode != 0 and outcome_single_mode != 1:
        raise ValueError("Outcome mode must be either 0 or 1")
    if outcome_single_mode == 0:
        return (0, 1)
    return (1, cutoff_dimension)

def get_index_shape_from_outcome(outcome, cutoff_dimension):
    result = []
    for outcome_single_mode in outcome:
        result += get_index_shape_from_outcome_single_mode(
            outcome_single_mode=outcome_single_mode,
            cutoff_dimension=cutoff_dimension
        )
    return result

def generate_measurement_matrix_one_outcome(outcome, cutoff_dimension, zeros_matrix):
    ones_matrix = np.ones((1, 1), dtype=np.int32)
    indices = get_index_shape_from_outcome(outcome=outcome, cutoff_dimension=cutoff_dimension)

    final_matrix = zeros_matrix.copy()
    if len(indices) < 2 or len(indices) > 14:
        raise ValueError("modes not supported. Only from 1 to 7 modes supported.")
    if len(indices) == 2:
        final_matrix[indices[0]:indices[1]] = ones_matrix
    if len(indices) == 4:
        final_matrix[indices[0]:indices[1], indices[2]:indices[3]] = ones_matrix
    if len(indices) == 6:
        final_matrix[indices[0]:indices[1], indices[2]:indices[3], indices[4]:indices[5]] = ones_matrix
    if len(indices) == 8:
        final_matrix[indices[0]:indices[1], indices[2]:indices[3],
                     indices[4]:indices[5], indices[6]:indices[7]] = ones_matrix
    if len(indices) == 10:
        final_matrix[indices[0]:indices[1], indices[2]:indices[3],
                     indices[4]:indices[5], indices[6]:indices[7], indices[8]:indices[9]] = ones_matrix
    if len(indices) == 12:
        final_matrix[indices[0]:indices[1], indices[2]:indices[3],
                     indices[4]:indices[5], indices[6]:indices[7],
                     indices[8]:indices[9], indices[10]:indices[11]] = ones_matrix
    if len(indices) == 14:
        final_matrix[indices[0]:indices[1], indices[2]:indices[3],
                     indices[4]:indices[5], indices[6]:indices[7],
                     indices[8]:indices[9], indices[10]:indices[11], indices[12]:indices[13]] = ones_matrix

    return final_matrix

def generate_measurement_matrices(num_modes: int, cutoff_dimension: int):
    assert 1 <= num_modes <= 2, "`num_modes` must be between 1 and 2."
    assert cutoff_dimension >= 1, "`cutoff_dimension` must be greater than 1."

    # https://strawberryfields.readthedocs.io/en/stable/code/api/strawberryfields.backends.BaseFockState.html?highlight=all_fock#strawberryfields.backends.BaseFockState.all_fock_probs
    matrix_shape = [cutoff_dimension] * num_modes
    zeros_matrix = np.zeros(matrix_shape)
    possible_outcomes = list(product([0, 1], repeat=num_modes))

    return [generate_measurement_matrix_one_outcome(
                outcome=outcome,
                cutoff_dimension=cutoff_dimension,
                zeros_matrix=zeros_matrix)
            for outcome in possible_outcomes]


if __name__ == "__main__":
    from pprint import pprint
    m = generate_measurement_matrices(2, 3)
    pprint(m)