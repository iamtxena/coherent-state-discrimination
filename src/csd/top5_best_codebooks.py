# top5_best_codebooks.py

from dataclasses import dataclass, field
from typing import List
from csd.best_codebook import BestCodeBook


@dataclass
class Top5_BestCodeBooks():
    """Class for managing the best five codebooks."""
    top5: List[BestCodeBook] = field(default_factory=list)

    @property
    def size(self) -> int:
        return len(self.top5)

    @property
    def modes(self) -> int:
        if self.size <= 0:
            raise ValueError("empty top5 list")
        return self.top5[0].modes

    @property
    def alpha(self) -> float:
        if self.size <= 0:
            raise ValueError("empty top5 list")
        return self.top5[0].alpha

    @property
    def first(self) -> BestCodeBook:
        if self.size <= 0:
            raise ValueError("empty top5 list")
        return self.top5[0]

    @first.setter
    def first(self, value: BestCodeBook) -> None:
        if self.size <= 0:
            self.top5.append(value)
            return
        self.top5[0] = value

    @property
    def second(self) -> BestCodeBook:
        if self.size <= 1:
            raise ValueError("empty top5 list")
        return self.top5[1]

    @second.setter
    def second(self, value: BestCodeBook) -> None:
        if self.size < 1:
            raise ValueError("top5 with no elements")
        if self.size == 1:
            self.top5.append(value)
            return
        self.top5[1] = value

    @property
    def third(self) -> BestCodeBook:
        if self.size <= 2:
            raise ValueError("empty top5 list")
        return self.top5[2]

    @third.setter
    def third(self, value: BestCodeBook) -> None:
        if self.size < 2:
            raise ValueError("top5 with just one element")
        if self.size == 2:
            self.top5.append(value)
            return
        self.top5[2] = value

    @property
    def fourth(self) -> BestCodeBook:
        if self.size <= 3:
            raise ValueError("empty top5 list")
        return self.top5[3]

    @fourth.setter
    def fourth(self, value: BestCodeBook) -> None:
        if self.size < 3:
            raise ValueError("top5 with just two elements")
        if self.size == 3:
            self.top5.append(value)
            return
        self.top5[3] = value

    @property
    def fifth(self) -> BestCodeBook:
        if self.size <= 4:
            raise ValueError("empty top5 list")
        return self.top5[4]

    @fifth.setter
    def fifth(self, value: BestCodeBook) -> None:
        if self.size < 4:
            raise ValueError("top5 with just three elements")
        if self.size == 4:
            self.top5.append(value)
            return
        self.top5[4] = value

    def _same_codebook(self, other: BestCodeBook) -> bool:
        if self.size == 0:
            return False
        return (self.first.alpha == other.alpha and self.first.modes == other.modes)

    def add(self, potential_best_codebook: BestCodeBook) -> None:
        """ Adds a new best codebook to the top5 if it is one of the top5 best values.

        Args:
            potential_best_codebook (BestCodeBook): A potential best Codebook
        """
        if self.size == 0:
            self.top5.append(potential_best_codebook)
            return
        if not self._same_codebook(other=potential_best_codebook):
            raise ValueError("Trying to add a new codebook with different modes or alpha")
        codebook_to_compare = potential_best_codebook
        if codebook_to_compare.success_probability > self.first.success_probability:
            tmp_codebook = self.first
            self.first = codebook_to_compare
            codebook_to_compare = tmp_codebook
        if self.size < 2:
            self.second = codebook_to_compare
            return
        if codebook_to_compare.success_probability > self.second.success_probability:
            tmp_codebook = self.second
            self.second = codebook_to_compare
            codebook_to_compare = tmp_codebook
        if self.size < 3:
            self.third = codebook_to_compare
            return
        if codebook_to_compare.success_probability > self.third.success_probability:
            tmp_codebook = self.third
            self.third = codebook_to_compare
            codebook_to_compare = tmp_codebook
        if self.size < 4:
            self.fourth = codebook_to_compare
            return
        if codebook_to_compare.success_probability > self.fourth.success_probability:
            tmp_codebook = self.fourth
            self.fourth = codebook_to_compare
            codebook_to_compare = tmp_codebook
        if self.size < 5:
            self.fifth = codebook_to_compare
            return
        if codebook_to_compare.success_probability > self.fifth.success_probability:
            self.fifth = codebook_to_compare
