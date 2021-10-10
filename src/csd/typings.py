from typing import TypedDict, List, Union
import enum

from tensorflow.python.framework.ops import EagerTensor


class CodewordProbabilities(TypedDict):
    prob_a: float
    prob_minus_a: float


class PhotodetectorProbabilities(TypedDict):
    prob_click: List[Union[float, EagerTensor]]
    prob_no_click: List[Union[float, EagerTensor]]


class CSDConfiguration(TypedDict):
    displacement_magnitude: float
    steps: int
    learning_rate: float
    batch_size: int
    threshold: float


class Backends(enum.Enum):
    FOCK = 'fock'
    GAUSSIAN = 'gaussian'
    BOSONIC = 'bosonic'
    TENSORFLOW = 'tf'


class MeasuringTypes(enum.Enum):
    SAMPLING = 'sampling'
    PROBABILITIES = 'probabilities'


class RunConfiguration(TypedDict, total=False):
    alpha: float
    displacement_magnitude: float
    backend: Backends
    number_qumodes: int
    number_layers: int
    measuring_type: MeasuringTypes
    shots: int
    codeword_size: int
    cutoff_dim: int
