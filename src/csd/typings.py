from typing import TypedDict
import enum


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
