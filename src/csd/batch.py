# batch.py

from dataclasses import dataclass
from typing import List, Optional
from .codeword import CodeWord

DEFAULT_ALPHA_VALUE = 0.7


@dataclass
class Batch():
    """Class for keeping track of an input batch for an experiment."""

    def __init__(self, size: int, word_size: int, alpha_value: Optional[float] = DEFAULT_ALPHA_VALUE):
        self._alpha_value = alpha_value if alpha_value is not None else DEFAULT_ALPHA_VALUE
        self._batch: List[CodeWord] = self._create_batch_with_random_word(batch_size=size,
                                                                          word_size=word_size,
                                                                          alpha_value=self._alpha_value)

    def _create_batch_with_random_word(self, batch_size: int, word_size: int, alpha_value: float) -> List[CodeWord]:
        return [CodeWord(size=word_size, alpha_value=alpha_value) for _ in range(batch_size)]

    @property
    def batch(self) -> List[List[float]]:
        return [codeword.word for codeword in self._batch]

    @property
    def alpha(self) -> float:
        return self._alpha_value
