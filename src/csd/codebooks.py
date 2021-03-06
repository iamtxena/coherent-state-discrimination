# codebooks.py

from dataclasses import dataclass
import math
import itertools
import random
from typing import List, Union
import numpy as np

from csd.codeword import CodeWord

from .batch import Batch
from csd.config import logger


@dataclass
class CodeBooks:
    """Class for managing the codebooks from a given batch."""

    def __init__(
        self,
        batch: Batch,
        max_combinations: int = 0,
        codewords_list: Union[List[List[CodeWord]], None] = None,
    ):
        # to use fully random codebooks, max_combinations must be set
        use_linear_codes = max_combinations == 0
        self._max_combinations = max_combinations
        self._codebook_maximum_size = self._compute_codebook_maximum_size(batch=batch)

        if codewords_list is None and use_linear_codes:
            self._information_bits = self._compute_maximum_information_bits(batch=batch)
            self._codebooks = self._generate_all_codebooks_with_linear_codes(
                batch=batch, information_bits=self._information_bits
            )
        if codewords_list is None and not use_linear_codes:
            self._codebooks = list(
                self._generate_all_random_codebooks_with_specific_size(
                    batch=batch, codebook_maximum_size=self._codebook_maximum_size
                )
            )
            self._alpha_value = batch.alpha
        if codewords_list is not None:
            self._codebooks = codewords_list
            self._alpha_value = codewords_list[0][0].alpha

    @staticmethod
    def from_codewords_list(codewords_list: List[List[CodeWord]]):
        return CodeBooks(
            batch=Batch(size=len(codewords_list[0]), word_size=codewords_list[0][0].size), codewords_list=codewords_list
        )

    @property
    def codebooks(self) -> List[List[CodeWord]]:
        return self._codebooks

    @property
    def size(self) -> int:
        return len(self._codebooks)

    @property
    def alpha(self) -> float:
        return self._alpha_value

    @property
    def binary_codes(self) -> List[List[int]]:
        return [codeword.binary_code for codebook in self._codebooks for codeword in codebook]

    def _compute_codebook_maximum_size(self, batch: Batch) -> int:
        channel_max_communication_rate = self._compute_channel_max_communication_rate(alpha=batch.alpha)
        return math.floor(math.pow(2, (batch.one_codeword.size * channel_max_communication_rate)))

    def _compute_channel_max_communication_rate(self, alpha: float) -> float:
        x_value = (1 + math.exp(-2 * alpha**2)) / 2
        return self._quantum_shannon_entropy(x=x_value)

    def _quantum_shannon_entropy(self, x: float) -> float:
        return -x * math.log2(x) - (1 - x) * math.log2(1 - x)

    def _compute_maximum_information_bits(self, batch: Batch) -> int:
        channel_max_communication_rate = self._compute_channel_max_communication_rate(alpha=batch.alpha)
        return math.floor(batch.one_codeword.size * channel_max_communication_rate)

    def _generate_all_random_codebooks_with_specific_size(
        self, batch: Batch, codebook_maximum_size: int
    ) -> List[List[CodeWord]]:
        if codebook_maximum_size > batch.size:
            raise ValueError("codebook size must be smaller than batch size")
        if codebook_maximum_size == batch.size:
            return [batch.codewords]

        total_combinations = self._compute_all_combinations(batch=batch, codebook_maximum_size=codebook_maximum_size)

        logger.debug(f"Total combinations: {total_combinations} and max combinations: {self._max_combinations}")
        if total_combinations <= self._max_combinations:
            all_combinations = itertools.combinations(batch.codewords, codebook_maximum_size)
            return [list(codewords) for codewords in all_combinations]

        return [random.choices(batch.codewords, k=codebook_maximum_size) for _ in range(self._max_combinations)]

    def _compute_all_combinations(self, batch: Batch, codebook_maximum_size: int) -> int:
        return int(
            math.factorial(batch.size)
            / (math.factorial(batch.size - codebook_maximum_size) * math.factorial(codebook_maximum_size))
        )

    def _generate_all_codebooks_with_linear_codes(self, batch: Batch, information_bits: int) -> List[List[CodeWord]]:

        rows = k = information_bits
        n = batch.one_codeword.size
        columns = n - k
        if not information_bits > 0:
            logger.warning(f"information bits are less than 1. value={information_bits}")
            return []
        options = [0, 1]

        k_identity = np.identity(k)
        all_columns = list([*itertools.product(options, repeat=columns)])
        all_As = list(itertools.product(all_columns, repeat=rows))
        all_generators = [np.hstack((k_identity, one_A)) for one_A in all_As]
        input_codes = list(itertools.product([0, 1], repeat=information_bits))

        all_codebooks = []
        for generator in all_generators:
            codebooks = []
            for code in input_codes:
                binary_code = np.mod(np.matmul(code, generator), 2)
                codeword = CodeWord(word=[batch.alpha if bit == 0 else -batch.alpha for bit in binary_code])
                codebooks.append(codeword)
            all_codebooks.append(codebooks)

        return all_codebooks
