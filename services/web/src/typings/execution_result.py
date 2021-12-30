from typing import List, NamedTuple


class ExecutionResultInput(NamedTuple):
    alpha: float
    success_probability: float
    number_modes: int
    squeezing: bool
    number_ancillas: int
    learning_rate: float
    learning_steps: int
    cutoff_dimensions: int
    full_batch_used: bool
    distance_to_homodyne_probability: float
    bit_error_rate: float
    time_in_seconds: float


class TrainingExecutionResultInput(ExecutionResultInput):
    optimized_parameters: List[float]


class ExecutionResult(ExecutionResultInput):
    id: int


class TrainingExecutionResult(TrainingExecutionResultInput):
    id: int


class TestingExecutionResultInput(ExecutionResultInput):
    pass


class TestingExecutionResult(ExecutionResult):
    pass
