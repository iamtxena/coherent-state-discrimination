# cost_function.py
from abc import ABC
from typing import List

from tensorflow.python.framework.ops import EagerTensor
from csd.batch import Batch
from csd.tf_engine import TFEngine
from csd.typings.optimization_testing import OptimizationTestingOptions
from csd.typings.typing import (Backends, CodeWordSuccessProbability, RunningTypes, TFEngineRunOptions)
from csd.config import logger


class OptimizationTesting(ABC):
    """Class to execute the experiment with the optimized parameters for one specific alpha
    """

    def __init__(self, batch: Batch, params: List[float], options: OptimizationTestingOptions) -> None:
        self._params = params
        self._options = options
        self._input_batch = batch
        self._output_batch = Batch(size=self._input_batch.size,
                                   word_size=self._options.circuit.number_modes,
                                   alpha_value=self._input_batch.alpha,
                                   random_words=False)

    def _run_and_get_codeword_guesses(self) -> List[CodeWordSuccessProbability]:
        if self._options.backend_name != Backends.TENSORFLOW.value:
            raise ValueError("TF Backend is the only supported.")
        if not isinstance(self._options.engine, TFEngine):
            raise ValueError("TF Backend can only run on TFEngine.")
        return self._options.engine.run_tf_circuit_training(
            circuit=self._options.circuit,
            options=TFEngineRunOptions(
                params=self._params,
                input_batch=self._input_batch,
                output_batch=self._output_batch,
                shots=self._options.shots,
                running_type=RunningTypes.TRAINING,
                measuring_type=None))

    def _compute_one_play_average_batch_success_probability(
            self,
            codeword_guesses: List[CodeWordSuccessProbability]) -> EagerTensor:

        success_probability_from_guesses = [
            codeword_success_prob.success_probability
            if batch_codeword == codeword_success_prob.guessed_codeword
            else 1 - codeword_success_prob.success_probability
            for batch_codeword, codeword_success_prob in zip(self._input_batch.codewords, codeword_guesses)]
        return sum(success_probability_from_guesses) / self._input_batch.size

    def run_and_compute_average_batch_success_probability(self) -> EagerTensor:

        batch_success_probability = sum([self._compute_one_play_average_batch_success_probability(
            codeword_guesses=self._run_and_get_codeword_guesses())
            for _ in range(self._options.plays)]
        ) / self._options.plays

        logger.debug(f'batch_success_probability: {batch_success_probability}')
        return batch_success_probability
