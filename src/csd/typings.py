from typing import TypedDict


class CSDConfiguration(TypedDict):
    displacement_magnitude: float
    steps: int
    learning_rate: float
    batch_size: int
    threshold: float
