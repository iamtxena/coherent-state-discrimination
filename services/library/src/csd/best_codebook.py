# best_codebook.py

from dataclasses import dataclass
from typing import List

from csd.codeword import CodeWord


@dataclass
class BestCodeBook():
    """Class for managing the best codebook."""
    codebook: List[CodeWord]
    measurements: List[List[int]]
    success_probability: float
    helstrom_probability: float
    homodyne_probability: float

    @property
    def modes(self) -> int:
        if len(self.codebook) <= 0:
            return 0
        return self.codebook[0].size

    @property
    def alpha(self) -> float:
        if len(self.codebook) <= 0:
            raise ValueError("empty codebook")
        return self.codebook[0].alpha
