# codebooks.py

from dataclasses import dataclass
import math
import random
from typing import List

from csd.codeword import CodeWord

from .batch import Batch


@dataclass
class CodeBooks():
    """Class for managing the codebooks from a given batch."""

    def __init__(self,
                 batch: Batch):
        self._codebook_maximum_size = self._compute_codebook_maximum_size(batch=batch)
        self._codebooks = self._generate_all_random_codebooks_with_specific_size(
            batch=batch, codebook_maximum_size=self._codebook_maximum_size)

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

        remaining_codewords = batch.codewords.copy()
        codebooks = []

        while len(remaining_codewords) >= codebook_maximum_size:
            selected_codebook = random.sample(remaining_codewords, codebook_maximum_size)
            for codeword in selected_codebook:
                index = remaining_codewords.index(codeword)
                remaining_codewords.pop(index)
            codebooks.append(selected_codebook)

        if len(remaining_codewords) > 0:
            codebooks.append(remaining_codewords.copy())

        return codebooks
