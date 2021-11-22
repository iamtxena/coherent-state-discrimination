from typing import NamedTuple, TypedDict, Union, List
import enum
import numpy as np
from nptyping import NDArray
import json
from dataclasses import dataclass

from tensorflow.python.framework.ops import EagerTensor
from tensorflow import Variable
from csd.batch import Batch

from csd.codeword import CodeWord


class LearningSteps(NamedTuple):
    default: int
    high: int
    extreme: int

    def __str__(self) -> str:
        return f'[{self.default},{self.high},{self.extreme}]'

    def __repr__(self) -> str:
        return self.__str__()


class LearningRate(NamedTuple):
    default: float
    high: float
    extreme: float

    def __str__(self) -> str:
        return f'[{self.default},{self.high},{self.extreme}]'

    def __repr__(self) -> str:
        return self.__str__()


class CutOffDimensions(NamedTuple):
    default: int
    high: int
    extreme: int

    def __str__(self) -> str:
        return f'[{self.default},{self.high},{self.extreme}]'

    def __repr__(self) -> str:
        return self.__str__()


class BackendOptions(TypedDict, total=False):
    cutoff_dim: int
    batch_size: Union[int, None]


class Architecture(TypedDict, total=False):
    number_modes: int
    number_ancillas: int
    number_layers: int
    squeezing: bool


class CSDConfiguration(TypedDict, total=False):
    alphas: List[float]
    learning_steps: LearningSteps
    learning_rate: LearningRate
    batch_size: int
    shots: int
    plays: int
    cutoff_dim: CutOffDimensions
    save_results: bool
    save_plots: bool
    architecture: Architecture
    parallel_optimization: bool


class Backends(enum.Enum):
    FOCK = 'fock'
    GAUSSIAN = 'gaussian'
    BOSONIC = 'bosonic'
    TENSORFLOW = 'tf'


class MeasuringTypes(enum.Enum):
    SAMPLING = 'sampling'
    PROBABILITIES = 'probabilities'


class RunConfiguration(TypedDict, total=False):
    run_backend: Backends
    measuring_type: MeasuringTypes


class OptimizationResult(NamedTuple):
    optimized_parameters: List[float]
    error_probability: float


class OneProcessResultExecution(TypedDict):
    opt_params: List[Union[List[float], EagerTensor]]
    p_err: List[Union[float, EagerTensor]]
    p_succ: List[Union[float, EagerTensor]]


class ResultExecution(TypedDict):
    alphas: List[float]
    batches: List[List[CodeWord]]
    opt_params: List[Union[List[float], EagerTensor]]
    p_err: List[Union[float, EagerTensor]]
    p_succ: List[Union[float, EagerTensor]]
    result_backend: str
    measuring_type: str
    plot_label: str
    plot_title: str
    total_time: float


class EngineRunOptions(TypedDict):
    params: Union[List[Union[float, EagerTensor]],
                  NDArray[np.float]]
    input_codeword: CodeWord
    output_codeword: CodeWord
    shots: int
    measuring_type: MeasuringTypes


class TFEngineRunOptions(TypedDict):
    params: List[EagerTensor]
    input_batch: Batch
    output_batch: Batch
    shots: int
    all_counts: List[Variable]
    measuring_type: MeasuringTypes


@dataclass
class CodeWordSuccessProbability():
    guessed_codeword: CodeWord
    output_codeword: CodeWord
    success_probability: Union[float, EagerTensor]
    counts: EagerTensor = Variable(0.0, trainable=False)

    def __str__(self) -> str:
        return json.dumps({
            "input_codeword": self.guessed_codeword.word if self.guessed_codeword is not None else None,
            "output_codeword": self.output_codeword.word,
            "psucc": (self.success_probability
                      if isinstance(self.success_probability, float)
                      else float(self.success_probability.numpy())),
            "counts": (self.counts
                       if isinstance(self.counts, float)
                       else float(self.counts.numpy()))
        })

    def __repr__(self) -> str:
        return self.__str__()


class CodeWordIndices(NamedTuple):
    codeword: CodeWord
    indices: List[List[int]]

    def __str__(self) -> str:
        return json.dumps({
            "codeword": self.codeword.word,
            "indices": self.indices
        })

    def __repr__(self) -> str:
        return self.__str__()


@dataclass
class BatchSuccessProbability():
    codeword_indices: CodeWordIndices
    success_probability: List[EagerTensor]

    def __str__(self) -> str:
        return json.dumps({
            "codeword_indices": self.codeword_indices.__str__(),
            "psucc": self.success_probability
        })

    def __repr__(self) -> str:
        return self.__str__()
