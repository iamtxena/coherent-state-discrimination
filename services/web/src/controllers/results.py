from abc import ABC
from typing import List, Tuple, Union
import json
from flask_sqlalchemy import Model
from src.typings.execution_result import (ExecutionResultInput,
                                          TrainingExecutionResultInput,
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
        alpha, architecture, additional_result = self._add_execution_result(
            execution_result_input=training_execution_result_input)
        training_options: TrainingOptions = self._database.add_instance_if_not_exist(
            model=TrainingOptions,
            learning_rate=training_execution_result_input.learning_rate,
            learning_steps=training_execution_result_input.learning_steps,
            cutoff_dimensions=training_execution_result_input.cutoff_dimensions,
            full_batch_used=training_execution_result_input.full_batch_used
        )
        optimized_parameters: OptimizedParameters = self._database.add_instance_if_not_exist(
            model=OptimizedParameters,
            parameters=json.dumps(training_execution_result_input.optimized_parameters)
        )
        training_result: TrainingResult = self._database.add_instance(
            model=TrainingResult,
            success_probability=training_execution_result_input.success_probability,
            alpha_id=alpha.id,
            architecture_id=architecture.id,
            training_options_id=training_options.id,
            additional_result_id=additional_result.id,
            optimized_parameters_id=optimized_parameters.id,
        )

        return TrainingExecutionResult(
            id=training_result.id,
            **training_execution_result_input.__dict__
        )

    def add_testing_execution_result(
            self,
            testing_execution_result_input: TestingExecutionResultInput) -> TestingExecutionResult:
        alpha, architecture, additional_result = self._add_execution_result(
            execution_result_input=testing_execution_result_input)

        testing_result: TestingResult = self._database.add_instance(
            model=TestingResult,
            success_probability=testing_execution_result_input.success_probability,
            alpha_id=alpha.id,
            architecture_id=architecture.id,
            additional_result_id=additional_result.id,
        )

        return TestingExecutionResult(
            id=testing_result.id,
            **testing_execution_result_input.__dict__
        )

    def _add_execution_result(
        self,
        execution_result_input: ExecutionResultInput) -> Tuple[Alpha,
                                                               Architecture,
                                                               AdditionalResult]:
        alpha: Alpha = self._database.add_instance_if_not_exist(
            model=Alpha,
            alpha=execution_result_input.alpha
        )
        architecture: Architecture = self._database.add_instance_if_not_exist(
            model=Architecture,
            number_modes=execution_result_input.number_modes,
            squeezing=execution_result_input.squeezing,
            number_modes=execution_result_input.number_ancillas,
            number_layers=execution_result_input.number_layers
        )
        additional_result: AdditionalResult = self._database.add_instance_if_not_exist(
            model=AdditionalResult,
            distance_to_homodyne_probability=execution_result_input.distance_to_homodyne_probability,
            bit_error_rate=execution_result_input.bit_error_rate,
            time_in_seconds=execution_result_input.time_in_seconds
        )
        return alpha, architecture, additional_result

    def get_training_execution_result(self, id: int) -> TrainingExecutionResult:
        training_result: TrainingResult = self._database.get_instance(
            model=TrainingResult,
            id=id)
        alpha, architecture, additional_result = self._get_execution_result(
            execution_result=training_result
        )
        training_options: TrainingOptions = self._database.get_instance(
            model=TrainingOptions,
            id=training_result.training_options_id
        )
        optimized_parameters: OptimizedParameters = self._database.get_instance(
            model=OptimizedParameters,
            id=training_result.optimized_parameters_id
        )
        return TrainingExecutionResult(
            id=training_result.id,
            alpha=alpha.alpha,
            success_probability=training_result.success_probability,
            number_modes=architecture.number_modes,
            squeezing=architecture.squeezing,
            number_ancillas=architecture.number_ancillas,
            number_layers=architecture.number_layers,
            learning_rate=training_options.learning_rate,
            learning_steps=training_options.learning_steps,
            cutoff_dimensions=training_options.cutoff_dimensions,
            full_batch_used=training_options.full_batch_used,
            distance_to_homodyne_probability=additional_result.distance_to_homodyne_probability,
            bit_error_rate=additional_result.bit_error_rate,
            time_in_seconds=additional_result.time_in_seconds,
            optimized_parameters=json.loads(optimized_parameters.parameters)
        )

    def get_testing_execution_result(self, id: int) -> TestingExecutionResult:
        testing_result: TestingResult = self._database.get_instance(
            model=TestingResult,
            id=id)
        alpha, architecture, additional_result = self._get_execution_result(
            execution_result=testing_result
        )

        return TestingExecutionResult(
            id=testing_result.id,
            alpha=alpha.alpha,
            success_probability=testing_result.success_probability,
            number_modes=architecture.number_modes,
            squeezing=architecture.squeezing,
            number_ancillas=architecture.number_ancillas,
            number_layers=architecture.number_layers,
            distance_to_homodyne_probability=additional_result.distance_to_homodyne_probability,
            bit_error_rate=additional_result.bit_error_rate,
            time_in_seconds=additional_result.time_in_seconds
        )

    def _get_execution_result(
        self,
        execution_result: Union[TrainingResult, TestingResult]) -> Tuple[Alpha,
                                                                         Architecture,
                                                                         AdditionalResult]:
        alpha: Alpha = self._database.get_instance(
            model=Alpha,
            id=execution_result.alpha_id
        )
        architecture: Architecture = self._database.get_instance(
            model=Architecture,
            id=execution_result.architecture_id
        )
        additional_result: AdditionalResult = self._database.get_instance(
            model=AdditionalResult,
            id=execution_result.additional_result_id
        )
        return alpha, architecture, additional_result

    def get_best_training_execution_results_for_all_alphas(self) -> List[TrainingExecutionResult]:
        alpha_values = self._get_all_alpha_values()
        return [self._get_best_execution_result_for_one_alpha(model=TrainingResult, alpha=alpha)
                for alpha in alpha_values]

    def get_best_testing_execution_results_for_all_alphas(self) -> List[TestingExecutionResult]:
        alpha_values = self._get_all_alpha_values()
        return [self._get_best_execution_result_for_one_alpha(model=TestingResult, alpha=alpha)
                for alpha in alpha_values]

    def get_best_training_execution_result_for_one_alpha(self, alpha: float) -> TrainingExecutionResult:
        return self._get_best_execution_result_for_one_alpha(model=TrainingResult, alpha=alpha)

    def get_best_testing_execution_result_for_one_alpha(self, alpha: float) -> TestingExecutionResult:
        return self._get_best_execution_result_for_one_alpha(model=TestingResult, alpha=alpha)

    def _get_best_execution_result_for_one_alpha(self,
                                                 model: Model,
                                                 alpha: float) -> Union[TrainingResult, TestingResult]:
        all_matching_alphas: List[Alpha] = Alpha.query.filter(Alpha.alpha == alpha).all()
        return self._get_best_result_from_list_of_same_alphas(alphas=all_matching_alphas, model=model)

    def _get_all_alpha_values(self) -> List[float]:
        alphas: List[Alpha] = Alpha.query(Alpha.alpha).group_by(Alpha.alpha).all()
        return [alpha.alpha for alpha in alphas]

    def _get_distance_to_homodyne_probability_from_alpha_id(
            self,
            model: Union[TrainingResult, TestingResult],
            alpha_id: int) -> Tuple[float, Union[TrainingResult, TestingResult]]:
        result: Union[TrainingResult, TestingResult] = model.query.filter(model.alpha_id == alpha_id).all()[0]
        additional_result: AdditionalResult = self._database.get_instance(
            AdditionalResult, id=result.additional_result_id)
        return additional_result.distance_to_homodyne_probability, result

    def _get_best_result_from_list_of_same_alphas(
            self,
            alphas: List[Alpha],
            model: Model) -> Union[TrainingResult, TestingResult]:
        alpha_value = alphas[0].alpha
        distance_to_homodyne_probability: Union[float, None] = None
        best_result: Union[TrainingResult, TestingResult, None] = None
        for alpha in alphas:
            if alpha.alpha != alpha_value:
                raise ValueError(f"alpha value {alpha.alpha} MUST be the same as {alpha_value}")
            current_distance_to_homodyne_probability, result = self._get_distance_to_homodyne_probability_from_alpha_id(
                model=model,
                alpha_id=alpha.id)
            if (distance_to_homodyne_probability is None or
                    current_distance_to_homodyne_probability < distance_to_homodyne_probability):
                distance_to_homodyne_probability = current_distance_to_homodyne_probability
                best_result = result
        if best_result is None:
            raise ValueError(f"No result found for this alpha: {alpha_value}")
        return best_result
