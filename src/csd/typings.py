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
