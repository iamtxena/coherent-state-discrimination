from typing import List, NamedTuple


class ExecutionResultInput(NamedTuple):
    alpha: float
    success_probability: float
    number_modes: int
    squeezing: bool
    number_ancillas: int
    number_layers: int
    distance_to_homodyne_probability: float
    bit_error_rate: float
    time_in_seconds: float


class TrainingExecutionResultInput(ExecutionResultInput):
    learning_rate: float
    learning_steps: int
    cutoff_dimensions: int
    full_batch_used: bool
    optimized_parameters: List[float]

    def __new__(cls,
                learning_rate: float,
                learning_steps: int,
                cutoff_dimensions: int,
                full_batch_used: bool,
                optimized_parameters: List[float],
                **kwargs):
        self = super(ExecutionResultInput, cls).__new__(cls, **kwargs)
        self.learning_rate = learning_rate
        self.learning_steps = learning_steps
        self.cutoff_dimensions = cutoff_dimensions
        self.full_batch_used = full_batch_used
        self.optimized_parameters = optimized_parameters
        return self


class ExecutionResult(ExecutionResultInput):
    id: int

    def __new__(cls, id: int, **kwargs):
        self = super(ExecutionResultInput, cls).__new__(cls, **kwargs)
        self.id = id
        return self


class TrainingExecutionResult(TrainingExecutionResultInput):
    id: int

    def __new__(cls, id: int, **kwargs):
        self = super(TrainingExecutionResultInput, cls).__new__(cls, **kwargs)
        self.id = id
        return self


class TestingExecutionResultInput(ExecutionResultInput):
    pass


class TestingExecutionResult(ExecutionResult):
    pass
