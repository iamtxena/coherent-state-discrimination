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
            return self._options.engine.run_tf_circuit_checking_measuring_type(
                circuit=self._options.circuit,
                options=TFEngineRunOptions(
                    params=self._params,
                    input_batch=self._input_batch,
                    output_batch=self._output_batch,
                    shots=self._options.shots,
                    measuring_type=self._options.measuring_type,
                    metric_type=MetricTypes.SUCCESS_PROBABILITY,
                    running_type=RunningTypes.TRAINING,
                ),
            )
        return [
            self._options.engine.run_circuit_checking_measuring_type(
                circuit=self._options.circuit,
                options=EngineRunOptions(
                    params=self._params,
                    input_codeword=codeword,
                    output_codeword=CodeWord(size=self._options.circuit.number_modes, alpha_value=codeword.alpha),
                    shots=self._options.shots,
                    measuring_type=self._options.measuring_type,
                ),
            )
            for codeword in self._input_batch.codewords
        ]

    def _compute_one_play_average_batch_success_probability(
        self, codeword_guesses: List[CodeWordSuccessProbability]
    ) -> Union[float, EagerTensor]:
        """
        Computes the one play average batch success probability.

        Args:
            codeword_guesses (List[CodeWordSuccessProbability]): The list of codeword success probabilities.

        Raises:
            ValueError: If the length of codeword_guesses is not equal to the size of the output batch.

        Returns:
            Union[float, EagerTensor]: The one play average batch success probability.
        """
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

    def _compute_mutual_information(self) -> Tuple[float, List[CodeWordSuccessProbability]]:
        """
        Computes the mutual information for the batch using the configured quantum circuit and engine.

        Returns:
            Tuple[float, List[CodeWordSuccessProbability]]: The computed mutual information and the list of codeword success probabilities.
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
        return mutual_information, codeword_guesses

    def run_and_compute_average_batch_metric(self) -> Tuple[float, List[CodeWordSuccessProbability]]:
        """
        Computes and minimizes the average metric of the batch based on the specified metric type.
        Raises:
            ValueError: If an unsupported metric type is specified.
        Returns:
            Tuple[float, List[CodeWordSuccessProbability]]: The minimized metric value and
                the list of codeword success probabilities.
        """
        metric_type = self._options.metric_type
        if isinstance(metric_type, MetricTypes):
            metric_type = metric_type.value

        if metric_type == MetricTypes.SUCCESS_PROBABILITY.value:
            codeword_guesses = self._run_and_get_codeword_guesses()
            metric = self._compute_one_play_average_batch_success_probability(codeword_guesses=codeword_guesses)
        elif metric_type == MetricTypes.MUTUAL_INFORMATION.value:
            metric, self._codeword_guesses = self._compute_mutual_information()
        else:
            raise ValueError("Unsupported metric type")
        return 1 - metric, self.measurements
