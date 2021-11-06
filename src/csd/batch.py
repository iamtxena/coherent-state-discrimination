# batch.py

from dataclasses import dataclass
from typing import List, Optional

from csd.util import generate_all_codewords
from .codeword import CodeWord

DEFAULT_ALPHA_VALUE = 0.7


@dataclass
class Batch():
    """Class for keeping track of an input batch for an experiment."""

    def __init__(self,
                 word_size: int,
                 alpha_value: Optional[float] = DEFAULT_ALPHA_VALUE,
                 random_words: Optional[bool] = True,
                 size: Optional[int] = 1):
        self._alpha_value = alpha_value if alpha_value is not None else DEFAULT_ALPHA_VALUE
        if random_words is None or random_words:
            if size is None:
                size = 1
            self._batch = self._create_batch_with_random_word(batch_size=size,
                                                              word_size=word_size,
                                                              alpha_value=self._alpha_value)
        if random_words is not None and not random_words:
            self._batch = generate_all_codewords(
                word_size=word_size, alpha_value=self._alpha_value)
        self._batch_length = len(self._batch)

    def _create_batch_with_random_word(self, batch_size: int, word_size: int, alpha_value: float) -> List[CodeWord]:
        return [CodeWord(size=word_size, alpha_value=alpha_value) for _ in range(batch_size)]

    @property
    def batch(self) -> List[List[float]]:
        return [codeword.word for codeword in self._batch]

    @property
    def codewords(self) -> List[CodeWord]:
        return self._batch

    @property
    def one_codeword(self) -> CodeWord:
        return self._batch[0]

    @property
    def alpha(self) -> float:
        return self._alpha_value

    @property
    def size(self) -> int:
        return self._batch_length

    def to_list(self) -> List[List[float]]:
        return self.batch

    @property
    def letters(self) -> List[List[float]]:
        word_size = self.one_codeword.size
        letters: List[List[float]] = [[] for _ in range(word_size)]

        for codeword in self._batch:
            for index, letter in enumerate(codeword.word):
                letters[index].append(letter)
        return letters
