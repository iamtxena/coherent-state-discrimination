import enum
import json
from dataclasses import dataclass
from typing import Dict, List, NamedTuple, TypedDict, Union

import numpy as np
from csd.batch import Batch
from csd.codeword import CodeWord
from tensorflow import Variable
from tensorflow.python.framework.ops import EagerTensor  # pylint: disable=no-name-in-module


class LearningSteps(NamedTuple):
    default: int
    high: int
    extreme: int

    def __str__(self) -> str:
        return f"[{self.default},{self.high},{self.extreme}]"

    def __repr__(self) -> str:
        return self.__str__()


class LearningRate(NamedTuple):
    default: float
    high: float
    extreme: float

    def __str__(self) -> str:
        return f"[{self.default},{self.high},{self.extreme}]"

    def __repr__(self) -> str:
        return self.__str__()


class CutOffDimensions(NamedTuple):
    default: int
    high: int
    extreme: int

    def __str__(self) -> str:
        return f"[{self.default},{self.high},{self.extreme}]"

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
    max_combinations: int


class Backends(enum.Enum):
    FOCK = "fock"
    GAUSSIAN = "gaussian"
    BOSONIC = "bosonic"
    TENSORFLOW = "tf"


class MeasuringTypes(enum.Enum):
    SAMPLING = "sampling"
    PROBABILITIES = "probabilities"


class RunningTypes(enum.Enum):
    TRAINING = "training"
    TESTING = "testing"


class OptimizationBackends(enum.Enum):
    SCIPY = "scipy"
    TENSORFLOW = "tf"


class MetricTypes(enum.Enum):
    SUCCESS_PROBABILITY = "success_probability"
    MUTUAL_INFORMATION = "mutual_information"


class RunConfiguration(TypedDict, total=False):
    run_backend: Backends
    optimization_backend: OptimizationBackends
    measuring_type: MeasuringTypes
    running_type: RunningTypes
    binary_codebook: List[List[int]]
    metric_type: MetricTypes


class OneProcessResultExecution(TypedDict):
    opt_params: List[Dict[str, Union[List[float], EagerTensor]]]
    p_err: List[Union[float, EagerTensor]]
    p_succ: List[Union[float, EagerTensor]]
    p_helstrom: List[float]
    p_homodyne: List[float]


class ResultExecution(TypedDict):
    alphas: List[float]
    batches: List[List[CodeWord]]
    opt_params: List[Dict[str, Union[List[float], EagerTensor]]]
    p_err: List[Union[float, EagerTensor]]
    p_succ: List[Union[float, EagerTensor]]
    result_backend: str
    measuring_type: str
    plot_label: str
    plot_title: str
    total_time: float
    p_helstrom: List[float]
    p_homodyne: List[float]
    number_modes: List[int]


class EngineRunOptions(TypedDict):
    params: Union[List[Union[float, EagerTensor]], List[np.float32]]
    input_codeword: CodeWord
    output_codeword: CodeWord
    shots: int
    measuring_type: MeasuringTypes


class TFEngineRunOptions(TypedDict):
    params: List[EagerTensor]
    input_batch: Batch
    output_batch: Batch
    shots: int
    measuring_type: Union[MeasuringTypes, None]
    running_type: Union[RunningTypes, None]
    metric_type: MetricTypes


@dataclass
class CodeWordSuccessProbability:
    input_codeword: CodeWord
    guessed_codeword: CodeWord
    output_codeword: CodeWord
    success_probability: Union[float, EagerTensor]
    mutual_information: Union[float, EagerTensor]
    counts: Variable

    def __str__(self) -> str:
        return json.dumps(
            {
                "input_codeword": self.input_codeword.word,
                "guessed_codeword": self.guessed_codeword.word if self.guessed_codeword is not None else None,
                "output_codeword": self.output_codeword.word,
                "psucc": (
                    float(self.success_probability)
                    if isinstance(self.success_probability, (float, np.float32))
                    else 0 if self.success_probability is None else float(self.success_probability.numpy())
                ),
                "mutual_information": (
                    float(self.mutual_information)
                    if isinstance(self.mutual_information, (float, np.float32))
                    else 0 if self.mutual_information is None else float(self.mutual_information.numpy())
                ),
            }
        )

    def __repr__(self) -> str:
        return self.__str__()

    @property
    def binary_code(self) -> dict:
        return {
            "input_codeword": self.input_codeword.binary_code,
            "guessed_codeword": self.guessed_codeword.binary_code if self.guessed_codeword is not None else None,
            "output_codeword": self.output_codeword.binary_code,
            "psucc": (
                float(self.success_probability)
                if isinstance(self.success_probability, (float, np.float32))
                else 0 if self.success_probability is None else float(self.success_probability.numpy())
            ),
            "mutual_information": (
                float(self.mutual_information)
                if isinstance(self.mutual_information, (float, np.float32))
                else 0 if self.mutual_information is None else float(self.mutual_information.numpy())
            ),
        }


class CodeWordIndices(NamedTuple):
    codeword: CodeWord
    indices: List[List[int]]

    def __str__(self) -> str:
        return json.dumps({"codeword": self.codeword.word, "indices": self.indices})

    def __repr__(self) -> str:
        return self.__str__()


@dataclass
class BatchSuccessProbability:
    codeword_indices: CodeWordIndices
    success_probability: List[EagerTensor]

    def __str__(self) -> str:
        return json.dumps({"codeword_indices": self.codeword_indices.__str__(), "psucc": self.success_probability})

    def __repr__(self) -> str:
        return self.__str__()


class OptimizationResult(NamedTuple):
    optimized_parameters: List[float]
    error_probability: float
    measurements: List[CodeWordSuccessProbability]
