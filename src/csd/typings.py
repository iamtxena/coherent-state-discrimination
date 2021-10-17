from typing import TypedDict, Union, List
import enum
import numpy as np
from nptyping import NDArray

from tensorflow.python.framework.ops import EagerTensor


class BackendOptions(TypedDict, total=False):
    cutoff_dim: int
    batch_size: Union[int, None]


class PhotodetectorProbabilities(TypedDict):
    prob_click: List[Union[float, EagerTensor]]
    prob_no_click: List[Union[float, EagerTensor]]


class Architecture(TypedDict, total=False):
    number_qumodes: int
    number_layers: int
    displacement: bool
    squeezing: bool
    interferometer: bool


class CSDConfiguration(TypedDict, total=False):
    steps: int
    learning_rate: float
    batch_size: int
    shots: int
    cutoff_dim: int
    batch: List[float]
    save_results: bool
    save_plots: bool
    architecture: Architecture


class Backends(enum.Enum):
    FOCK = 'fock'
    GAUSSIAN = 'gaussian'
    BOSONIC = 'bosonic'
    TENSORFLOW = 'tf'


class MeasuringTypes(enum.Enum):
    SAMPLING = 'sampling'
    PROBABILITIES = 'probabilities'


class RunConfiguration(TypedDict, total=False):
    alphas: List[float]
    backend: Backends
    measuring_type: MeasuringTypes
    shots: int
    batch_size: int
    cutoff_dim: int
    steps: int
    batch: List[float]


class OptimizationResult(TypedDict):
    optimized_parameters: List[Union[float, EagerTensor]]
    current_p_err: float


class ResultExecution(TypedDict):
    alphas: List[float]
    batches: List[List[float]]
    opt_params: List[Union[List[float], EagerTensor]]
    p_err: List[Union[float, EagerTensor]]
    p_succ: List[Union[float, EagerTensor]]
    backend: str
    measuring_type: str
    plot_label: str


class EngineRunOptions(TypedDict):
    params: Union[List[Union[float, EagerTensor]],
                  NDArray[np.float]]
    sample_or_batch: Union[Union[float, EagerTensor],
                           Union[List[float], List[EagerTensor]]]
    shots: int
    measuring_type: MeasuringTypes
