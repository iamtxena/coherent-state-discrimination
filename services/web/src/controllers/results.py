from abc import ABC
from typing import Any
from sqlalchemy.orm.exc import NoResultFound
from src.typings.execution_result import (TrainingExecutionResultInput,
                                          TrainingExecutionResult,
                                          TestingExecutionResultInput,
                                          TestingExecutionResult)
from src.models import (Alpha,
                        Architecture,
                        TrainingOptions,
                        TrainingResult,
                        TestingResult,
                        AdditionalResult,
                        OptimizedParameters)


class ResultsController(ABC):
    def __init__(self, database):
        self._database = database

    def add_training_execution_result(
            self,
            training_execution_result_input: TrainingExecutionResultInput) -> TrainingExecutionResult:
        alpha_model: Alpha = self._database.add_instance_if_not_exist(model=Alpha,
                                                                      alpha=training_execution_result_input.alpha)
        arquitecture: Architecture = self._database.add_instance_if_not_exist(model=Architecture)
        return None

    def add_testing_execution_result(self,
                                     testing_execution_result_input: TestingExecutionResultInput) -> TestingExecutionResult:
        return None

    def get_training_execution_result_one_alpha(self, alpha: float) -> TrainingExecutionResult:
        return None

    def get_testing_execution_result_one_alpha(self, alpha: float) -> TestingExecutionResult:
        return None
