from typing import List
from csd.batch import Batch
# from csd.codeword import CodeWord
from csd.codebooks import CodeBooks
from csd.codeword import CodeWord

import numpy as np


def _get_alphas() -> List[float]:
    alpha_init = 0.1
    alpha_end = 1.4
    number_points_to_plot = 16
    alpha_step = (alpha_end - alpha_init) / number_points_to_plot
    return list(np.arange(alpha_init, alpha_end, alpha_step))


def test_linear_codes(word_size: int = 3, alpha_value: float = 0.6) -> CodeBooks:
    batch = Batch(size=0, word_size=word_size, alpha_value=alpha_value, random_words=False)
    cb = CodeBooks(batch=batch)
    return cb

# def test_ideal_probabilities():
#    alphas = _get_alphas()


def filter_number_modes_from_codebook(codebook: List[CodeWord]) -> int:
    number_codebooks = len(codebook)
    if number_codebooks <= 0:
        print(f'WARNING: codebook len: {number_codebooks}')
        return 0
    orig_number_modes = codebook[0].size
    if number_codebooks == 1:
        print(f'INFO: codebook len: {number_codebooks} & number modes: {orig_number_modes}')
        return orig_number_modes
    binary_codebook = [codeword.binary_code for codeword in codebook]
    # print(f'binary_codebook: {binary_codebook}')
    summed_codes = list(map(sum, zip(*binary_codebook)))
    # print(f'summed_codes: {summed_codes}')
    zero_constant_modes = summed_codes.count(0)
    # print(f'zero_constant_modes: {zero_constant_modes}')
    one_constant_modes = summed_codes.count(number_codebooks)
    if one_constant_modes > 0:
        print(f'one_constant_modes: {one_constant_modes}')
        print(f'summed_codes: {summed_codes}')
    return orig_number_modes - zero_constant_modes - one_constant_modes


def test_filter_modes():
    alphas = _get_alphas()
    for size in range(2, 8):
        for alpha in alphas:
            cb = test_linear_codes(word_size=size, alpha_value=alpha)
            final_number_modes = [filter_number_modes_from_codebook(codebook=codebook) for codebook in cb.codebooks]
            for codebook, final_number_mode in zip(cb.codebooks, final_number_modes):
                # print(f'codebook: {codebook}\nmodes: {codebook[0].size} and '
                #       f'final_number_mode: {final_number_mode}')
                different_modes = '*' * (codebook[0].size - final_number_mode)
                print(f'ALPHA: {np.round(alpha, 2)} modes: {codebook[0].size} and '
                      f'final_number_mode: {final_number_mode} {different_modes}')


if __name__ == '__main__':
    # test_linear_codes()
    test_filter_modes()
