# codebooks.py

from dataclasses import dataclass
import math
import itertools
import random
from typing import List

from csd.codeword import CodeWord

from .batch import Batch
from csd.config import logger

DEFAULT_MAX_COMBINATIONS = 120


@dataclass
class CodeBooks():
    """Class for managing the codebooks from a given batch."""

    def __init__(self,
                 batch: Batch,
                 max_combinations: int = DEFAULT_MAX_COMBINATIONS):
        self._max_combinations = max_combinations
        self._codebook_maximum_size = self._compute_codebook_maximum_size(batch=batch)
        self._codebooks: List[List[CodeWord]] = list(self._generate_all_random_codebooks_with_specific_size(
            batch=batch, codebook_maximum_size=self._codebook_maximum_size))
        self._alpha_value = batch.alpha

    @property
    def codebooks(self) -> List[List[CodeWord]]:
        return self._codebooks

    @property
    def size(self) -> int:
        return len(self._codebooks)

    @property
    def alpha(self) -> float:
        return self._alpha_value

    def _compute_codebook_maximum_size(self, batch: Batch) -> int:
        channel_max_communication_rate = self._compute_channel_max_communication_rate(alpha=batch.alpha)
        return math.floor(math.pow(2, (batch.one_codeword.size * channel_max_communication_rate)))

    def _compute_channel_max_communication_rate(self, alpha: float) -> float:
        x_value = (1 + math.exp(-2 * alpha**2)) / 2
        return self._quantum_shannon_entropy(x=x_value)

    def _quantum_shannon_entropy(self, x: float) -> float:
        return -x * math.log2(x) - (1 - x) * math.log2(1 - x)

    def _generate_all_random_codebooks_with_specific_size(self,
                                                          batch: Batch,
                                                          codebook_maximum_size: int) -> List[List[CodeWord]]:
        if codebook_maximum_size > batch.size:
            raise ValueError("codebook size must be smaller than batch size")
        if codebook_maximum_size == batch.size:
            return [batch.codewords]

        total_combinations = self._compute_all_combinations(batch=batch, codebook_maximum_size=codebook_maximum_size)

        logger.debug(f'Total combinations: {total_combinations} and max combinations: {self._max_combinations}')
        if total_combinations <= self._max_combinations:
            all_combinations = itertools.combinations(batch.codewords, codebook_maximum_size)
            return [list(codewords) for codewords in all_combinations]

        return [random.choices(batch.codewords, k=codebook_maximum_size) for _ in range(0, self._max_combinations)]

    def _compute_all_combinations(self, batch: Batch, codebook_maximum_size: int) -> int:
        return int(math.factorial(batch.size) /
                   (math.factorial(batch.size - codebook_maximum_size) * math.factorial(codebook_maximum_size)))
