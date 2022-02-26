# best_codebook.py

from dataclasses import dataclass
from typing import List, Union
from tensorflow.python.framework.ops import EagerTensor
from csd.codeword import CodeWord
import json


@dataclass
class BestCodeBook():
    """Class for managing the best codebook."""
    codebook: List[CodeWord]
    measurements: List[CodeWord]
    success_probability: float
    helstrom_probability: float
    homodyne_probability: float
    optimized_parameters: List[Union[List[float], EagerTensor]]

    @property
    def parsed_optimized_parameters(self) -> List[List[float]]:
        return [[param if isinstance(param, float) else float(param.numpy())
                for param in params] for params in self.optimized_parameters]

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

    @property
    def binary_codebook(self) -> List[List[int]]:
        return [codeword.binary_code for codeword in self.codebook]

    @property
    def binary_measurements(self) -> List[List[int]]:
        return [codeword.binary_code for codeword in self.measurements]

    def to_dict(self) -> dict:
        return {
            'alpha': float(self.alpha),
            'modes': self.modes,
            'best_success_probability': float(self.success_probability),
            'helstrom_probability': self.helstrom_probability,
            'homodyne_probability': self.homodyne_probability,
            'best_codebook': [codeword.binary_code for codeword in self.codebook],
            'measurements': [codeword.binary_code for codeword in self.measurements],
            'optimized_parameters': self.parsed_optimized_parameters
        }

    def __str__(self) -> str:
        return json.dumps(self.to_dict())

    def __repr__(self) -> str:
        return self.__str__()
