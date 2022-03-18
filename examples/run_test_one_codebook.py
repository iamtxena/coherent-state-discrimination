import argparse
import itertools
from typing import List
from csd import CSD
from csd.batch import Batch
from csd.codebooks import CodeBooks
from csd.codeword import CodeWord
from csd.typings.typing import CSDConfiguration, Backends
from csd.config import logger
import numpy as np


def create_alphas() -> List[float]:
    alpha_init = 0.1
    alpha_end = 1.4
    number_points_to_plot = 16
    alpha_step = (alpha_end - alpha_init) / number_points_to_plot
    return list(np.arange(alpha_init, alpha_end, alpha_step))


def create_codeword_from_binary(binary_codeword: List[int], one_alpha: float) -> CodeWord:
    word = []
    for bit in binary_codeword:
        if bit != 0 and bit != 1:
            raise ValueError("codeword can only accept 0 or 1 values")
        if bit == 0:
            word.append(one_alpha)
        if bit == 1:
            word.append(-one_alpha)
    return CodeWord(alpha_value=one_alpha, word=word)


def generate_all_outcomes(modes=int):
    options = [0, 1]
    return [*itertools.product(options, repeat=modes)]


def create_all_codewords(alpha_value: float, number_modes: int) -> List[CodeWord]:
    all_binary_codewords = generate_all_outcomes(modes=number_modes)

    return [create_codeword_from_binary(binary_codeword=one_binary_codeword, one_alpha=alpha_value)
            for one_binary_codeword in all_binary_codewords]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Runs a circuit")
    parser.add_argument(
        "-m",
        "--modes",
        type=int,
        required=False,
        default=2,
        metavar="MODES",
        choices=[2, 3, 4, 5, 6, 7],
        help="Input modes from 2 to 7. Default is 2",
    )
    parser.add_argument(
        "-a",
        "--alpha-index",
        type=int,
        required=False,
        default=7,
        metavar="ALPHA-INDEX",
        choices=list(range(16)),
        help="Alpha index from 0 to 15. Default is 7.",
    )

    args = parser.parse_args()
    alphas = create_alphas()
    one_alpha = alphas[args.alpha_index]
    input_modes = args.modes
    # one_binary_codeword = [0, 0]
    all_codewords = create_all_codewords(alpha_value=one_alpha, number_modes=input_modes)
    codebooks = CodeBooks.from_codewords_list(codewords_list=[all_codewords])
    input_batch = Batch(size=0, word_size=input_modes, all_words=False, input_batch=all_codewords)

    # run_one_codebook()
    print(all_codewords)
    print(codebooks.codebooks)
    print(input_batch.codewords)
