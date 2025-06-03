# cost_function.py
from abc import ABC
from typing import List, Tuple, Union

import tensorflow as tf
from csd.batch import Batch
from csd.config import logger
from csd.tf_engine import TFEngine
from csd.typings.optimization_testing import OptimizationTestingOptions
from csd.typings.typing import Backends, CodeWordSuccessProbability, MetricTypes, RunningTypes, TFEngineRunOptions
from tensorflow.python.framework.ops import EagerTensor


class OptimizationTesting(ABC):
    """Class to execute the experiment with the optimized parameters for one specific alpha"""

    def __init__(self, batch: Batch, params: List[float], options: OptimizationTestingOptions) -> None:
        self._params = params
        self._options = options
        self._input_batch = batch
        self._output_batch = Batch(
            size=0,
            word_size=self._options.circuit.number_modes,
            alpha_value=self._input_batch.alpha,
            all_words=True,
            random_words=False,
        )
        self._codeword_guesses: Union[None, List[CodeWordSuccessProbability]] = None

    @property
    def measurements(self) -> List[CodeWordSuccessProbability]:
        if self._codeword_guesses is None:
            raise ValueError("codeword guesses still not computed")
        return self._codeword_guesses

    def run_and_compute_average_batch_metric(self) -> Tuple[EagerTensor, List[CodeWordSuccessProbability]]:
        """
        Computes and returns the average batch metric based on the specified metric type.

        Raises:
            ValueError: If an unsupported metric type is specified.

        Returns:
            Tuple[EagerTensor, List[CodeWordSuccessProbability]]: The computed metric value and the list of codeword success probabilities.
        """
        metric_type = self._options.metric_type

        if metric_type == MetricTypes.SUCCESS_PROBABILITY:
            return self._compute_success_probability()
        elif metric_type == MetricTypes.MUTUAL_INFORMATION:
            return self._compute_mutual_information()
        else:
            raise ValueError(f"Unsupported metric type: {metric_type}")

    def _compute_mutual_information(self) -> Tuple[EagerTensor, List[CodeWordSuccessProbability]]:
        """
        Compute the mutual information of the optimized parameters.

        Returns:
            Tuple[EagerTensor, List[CodeWordSuccessProbability]]: The mutual information and the list of codeword success probabilities.
        """
        options = TFEngineRunOptions(
            params=self._params,
            input_batch=self._input_batch,
            output_batch=self._output_batch,
            shots=self._options.shots,
            measuring_type=self._options.measuring_type,
            running_type=RunningTypes.TRAINING,
            metric_type=MetricTypes.MUTUAL_INFORMATION,
        )
        mutual_information, codeword_guesses = self._options.engine.run_mutual_information(
            self._options.circuit, options
        )
        logger.debug(f"TESTING Mutual information from trained parameters: {mutual_information}")
        return mutual_information, codeword_guesses

    def _run_and_get_codeword_guesses(self) -> List[CodeWordSuccessProbability]:
        if self._options.backend_name != Backends.TENSORFLOW.value:
            raise ValueError("TF Backend is the only supported.")
        if not isinstance(self._options.engine, TFEngine):
            raise ValueError("TF Backend can only run on TFEngine.")
        return self._options.engine.run_tf_circuit_checking_measuring_type(
            circuit=self._options.circuit,
            options=TFEngineRunOptions(
                params=[tf.Variable(param) for param in self._params],
                input_batch=self._input_batch,
                output_batch=self._output_batch,
                shots=self._options.shots,
                running_type=RunningTypes.TRAINING,
                measuring_type=self._options.measuring_type,
            ),
        )

    def _compute_one_play_average_batch_success_probability(
        self, codeword_guesses: List[CodeWordSuccessProbability]
    ) -> EagerTensor:

        self._codeword_guesses = codeword_guesses
        if len(codeword_guesses) != self._output_batch.size:
            raise ValueError(
                f"Codeword guesses length: {len(codeword_guesses)}"
                f" MUST be equal to output batch size: {self._output_batch.size}"
            )

        success_probability_from_guesses = [
            codeword_success_prob.success_probability for codeword_success_prob in codeword_guesses
        ]
        return sum(success_probability_from_guesses) / self._input_batch.size

    def _compute_success_probability(self) -> Tuple[EagerTensor, List[CodeWordSuccessProbability]]:
        """
        Compute the success probability of the optimized parameters.

        Returns:
            Tuple[EagerTensor, List[CodeWordSuccessProbability]]: The success probability and the list of codeword success probabilities.
        """

        batch_success_probability = self._compute_one_play_average_batch_success_probability(
            codeword_guesses=self._run_and_get_codeword_guesses()
        )

        logger.debug(f"TESTING Success probability from trained parameters: {batch_success_probability}")
        return batch_success_probability, self.measurements
