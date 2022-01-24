# codeword.py

from dataclasses import dataclass
from typing import List, Optional
import random
import numpy as np
# import json

A = 1
MINUS_A = -1
DEFAULT_ALPHA_VALUE = 0.7
DEFAULT_WORD_SIZE = 10


@dataclass
class CodeWord():
    """Class for keeping track of an input word for an experiment."""

    def __init__(self,
                 size: Optional[int] = 0,
                 alpha_value: Optional[float] = DEFAULT_ALPHA_VALUE,
                 word: Optional[List[float]] = None):
        self._alpha_value = self._create_alpha_value_from_input_parameters(alpha_value, word)
        self._size = self._create_size_from_input_parameters(size, word)

        self._word = self._create_random_word(word_size=self._size,
                                              alpha_value=self._alpha_value) if word is None else word

    def _create_size_from_input_parameters(self,
                                           size: Optional[int] = 0,
                                           word: Optional[List[float]] = None) -> int:
        if size == 0 and word is None:
            raise ValueError('size and word not defined! One of them MUST be defined.')
        if word is not None:
            return len(word)
        if size is None:
            raise ValueError('size not defined')
        return size

    def _create_alpha_value_from_input_parameters(self,
                                                  alpha_value: Optional[float] = DEFAULT_ALPHA_VALUE,
                                                  word: Optional[List[float]] = None) -> float:
        if alpha_value is not None:
            return alpha_value
        if word is not None:
            return np.abs(word[0])
        return DEFAULT_ALPHA_VALUE

    def _create_word(self, samples: List[float], word_size=10) -> List[float]:
        return [random.choice(samples) for _ in range(word_size)]

    def _create_input_word(self, word: List[float], alpha_value: float) -> List[float]:
        return list(alpha_value * np.array(word))

    def _create_random_word(self,
                            word_size=DEFAULT_WORD_SIZE,
                            alpha_value: float = DEFAULT_ALPHA_VALUE) -> List[float]:
        base_word = self._create_word(samples=[A, MINUS_A], word_size=word_size)
        return self._create_input_word(word=base_word, alpha_value=alpha_value)

    def __str__(self) -> str:
        # return json.dumps(self.word)
        return str(self.word)

    def __repr__(self) -> str:
        return self.__str__()

    @property
    def word(self) -> List[float]:
        return self._word

    @property
    def alpha(self) -> float:
        return self._alpha_value

    @property
    def size(self) -> int:
        return self._size

    @property
    def number_alphas(self) -> int:
        return self._word.count(self.alpha)

    @property
    def number_minus_alphas(self) -> int:
        return self.size - self._word.count(self.alpha)

    def to_list(self) -> List[float]:
        return self.word

    @property
    def zero_list(self) -> List[int]:
        """ Returns a list of 0 where there is alpha_value
            leaving the rest the same

        Returns:
            List[int]: the generated list
        """
        return [0 if letter == self.alpha else -1 for letter in self.word]

    @property
    def minus_indices(self) -> List[int]:
        """ Returns a list of indices where there is -alpha_values
        """
        return np.where(np.array(self.word) != self.alpha)[0]

    def __eq__(self, other) -> bool:
        if other is None:
            return False
        if not hasattr(other, 'word'):
            return False
        return self.word == other.word
