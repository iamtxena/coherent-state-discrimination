# cost_function.py
"""
Cost function for the CSD algorithm.
"""
from abc import ABC
from typing import List, Tuple, Union

from csd.batch import Batch
from csd.codeword import CodeWord
from csd.config import logger
from csd.tf_engine import TFEngine
from csd.typings.cost_function import CostFunctionOptions
from csd.typings.typing import (
    Backends,
    CodeWordSuccessProbability,
    EngineRunOptions,
    MetricTypes,
    RunningTypes,
    TFEngineRunOptions,
)

# import tensorflow as tf
from tensorflow.python.framework.ops import EagerTensor

# import numpy as np


class CostFunction(ABC):
    """
    This class is the base class for all cost functions.
    """

    def __init__(self, batch: Batch, params: List[float], options: CostFunctionOptions) -> None:
        """
        This method initializes the cost function.
        """
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
        """
        Provides the computed codeword success probabilities.

        Raises:
            ValueError: If codeword guesses have not been computed yet.

        Returns:
            List[CodeWordSuccessProbability]: List of codeword success probabilities.
        """
        if self._codeword_guesses is None:
            raise ValueError("Codeword guesses not computed")
        return self._codeword_guesses

    def _run_and_get_codeword_guesses(self) -> List[CodeWordSuccessProbability]:
        """
        Executes the quantum circuit and retrieves codeword guesses based on the configured backend.

        Raises:
            ValueError: If the backend is not TensorFlow or the engine is not an instance of TFEngine.
            NotImplementedError: If a backend other than TensorFlow is specified.

        Returns:
            List[CodeWordSuccessProbability]: List of codeword success probabilities.
        """
        if self._options.backend_name == Backends.TENSORFLOW.value:
            if not isinstance(self._options.engine, TFEngine):
                raise ValueError("TF Backend can only run on TFEngine.")
            options = TFEngineRunOptions(
                params=self._params,
                input_batch=self._input_batch,
                output_batch=self._output_batch,
                shots=self._options.shots,
                measuring_type=self._options.measuring_type,
                running_type=RunningTypes.TRAINING,
                metric_type=MetricTypes.SUCCESS_PROBABILITY,  # Explicitly for success probability
            )
            return self._options.engine.run_tf_circuit_checking_measuring_type(
                circuit=self._options.circuit, options=options
            )
        else:
            raise NotImplementedError("Other backends not implemented")

    def _compute_success_probability(self, codeword_guesses):
        """
        Computes the success probability from the codeword guesses.

        Args:
            codeword_guesses (List[CodeWordSuccessProbability]): List of codeword success probabilities.

        Raises:
            ValueError: If the number of codeword guesses does not match the output batch size.

        Returns:
            float: The average success probability.
        """
        self._codeword_guesses = codeword_guesses
        if len(codeword_guesses) != self._output_batch.size:
            raise ValueError("Mismatch between codeword guesses and output batch size")
        success_probability = sum(codeword.success_probability for codeword in codeword_guesses) / len(codeword_guesses)
        return success_probability

    def _compute_mutual_information(self):
        """
        Computes the mutual information for the batch using the configured quantum circuit and engine.

        Returns:
            Union[float, EagerTensor]: The computed mutual information.
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
        mutual_information = self._options.engine.run_tf_mutual_information(self._options.circuit, options)
        logger.info("Computed mutual information: %s", mutual_information)
        return mutual_information

    def run_and_compute_average_batch_metric(self) -> Tuple[float, List[CodeWordSuccessProbability]]:
        """
        Computes and minimizes the average metric of the batch based on the specified metric type.
        Raises:
            ValueError: If an unsupported metric type is specified.
        Returns:
            Tuple[float, List[CodeWordSuccessProbability]]: The minimized metric value and
                the list of codeword success probabilities.
        """
        if self._options.metric_type == MetricTypes.SUCCESS_PROBABILITY.value:
            codeword_guesses = self._run_and_get_codeword_guesses()
            metric = self._compute_success_probability(codeword_guesses)
        elif self._options.metric_type == MetricTypes.MUTUAL_INFORMATION.value:
            metric = self._compute_mutual_information()
        else:
            raise ValueError("Unsupported metric type")
        return 1 - metric, self.measurements
