from typing import List
from csd.codebooks import CodeBooks
from csd.codeword import CodeWord


def _create_codeword_from_binary(binary_codeword: List[int], one_alpha: float) -> CodeWord:
    word = []
    for bit in binary_codeword:
        if bit not in [0, 1]:
            raise ValueError("codeword can only accept 0 or 1 values")
        if bit == 0:
            word.append(one_alpha)
        else:
            word.append(-one_alpha)
    return CodeWord(alpha_value=one_alpha, word=word)


def create_codebook_from_binary(binary_codebook: List[List[int]], one_alpha: float) -> CodeBooks:
    codewords = [
        _create_codeword_from_binary(binary_codeword=one_binary_codeword, one_alpha=one_alpha)
        for one_binary_codeword in binary_codebook
    ]
    return CodeBooks.from_codewords_list([codewords])
