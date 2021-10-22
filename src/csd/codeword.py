# codeword.py

from dataclasses import dataclass
from typing import List, Optional
import random
import numpy as np

A = 1
MINUS_A = -1
DEFAULT_ALPHA_VALUE = 0.7
DEFAULT_WORD_SIZE = 10


@dataclass
class CodeWord():
    """Class for keeping track of an input word for an experiment."""

    def __init__(self, size: int, alpha_value: Optional[float] = DEFAULT_ALPHA_VALUE):
        self._alpha_value = alpha_value if alpha_value is not None else DEFAULT_ALPHA_VALUE
        self._word = self._create_random_word(word_size=size,
                                              alpha_value=self._alpha_value)

    def _create_word(self, samples: List[float], word_size=10) -> List[float]:
        return [random.choice(samples) for _ in range(word_size)]

    def _create_input_word(self, word: List[float], alpha_value: float) -> List[float]:
        return list(alpha_value * np.array(word))

    def _create_random_word(self,
                            word_size=DEFAULT_WORD_SIZE,
                            alpha_value: float = DEFAULT_ALPHA_VALUE) -> List[float]:
        base_word = self._create_word(samples=[A, MINUS_A], word_size=word_size)
        return self._create_input_word(word=base_word, alpha_value=alpha_value)

    @property
    def word(self) -> List[float]:
        return self._word

    @property
    def alpha(self) -> float:
        return self._alpha_value

    @property
    def size(self) -> int:
        return len(self._word)

    @property
    def number_alphas(self) -> int:
        return self._word.count(self.alpha)

    def to_list(self) -> List[float]:
        return self.word

    @property
    def zero_list(self) -> List[float]:
        """ Returns a list of 0 where there is alpha_value
            leaving the rest the same

        Returns:
            List[float]: the generated list
        """
        return [0 if letter == self.alpha else letter for letter in self.word]

    @property
    def minus_indices(self) -> List[int]:
        """ Returns a list of indices where there is -alpha_values
        """
        return np.where(np.array(self.word) != self.alpha)[0]
