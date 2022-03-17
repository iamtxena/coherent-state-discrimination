# cost_function.py
from abc import ABC
from typing import List, Tuple, Union
# import numpy as np

# import tensorflow as tf
from tensorflow.python.framework.ops import EagerTensor
from csd.batch import Batch
from csd.codeword import CodeWord
from csd.tf_engine import TFEngine
from csd.typings.typing import (Backends, CodeWordSuccessProbability,
                                EngineRunOptions, RunningTypes, TFEngineRunOptions)
from csd.typings.cost_function import CostFunctionOptions
# from csd.config import logger


class CostFunction(ABC):
    """Class to encapsulate cost function logic with the probability computation
    """

    def __init__(self, batch: Batch, params: List[float], options: CostFunctionOptions) -> None:
        self._params = params
        self._options = options
        self._input_batch = batch
        self._output_batch = Batch(size=0,
                                   word_size=self._options.circuit.number_modes,
                                   alpha_value=self._input_batch.alpha,
                                   all_words=True,
                                   random_words=False)
        self._codeword_guesses: Union[None, List[CodeWordSuccessProbability]] = None

    @property
    def measurements(self) -> List[CodeWordSuccessProbability]:
        if self._codeword_guesses is None:
            raise ValueError("codeword guesses still not computed")
        return self._codeword_guesses

    def _run_and_get_codeword_guesses(self) -> List[CodeWordSuccessProbability]:
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
                    running_type=RunningTypes.TRAINING))
        return [self._options.engine.run_circuit_checking_measuring_type(
            circuit=self._options.circuit,
            options=EngineRunOptions(
                params=self._params,
                input_codeword=codeword,
                output_codeword=CodeWord(size=self._options.circuit.number_modes,
                                         alpha_value=codeword.alpha),
                shots=self._options.shots,
                measuring_type=self._options.measuring_type))
                for codeword in self._input_batch.codewords]

    def _compute_one_play_average_batch_success_probability(
            self,
            codeword_guesses: List[CodeWordSuccessProbability]) -> Union[float, EagerTensor]:

        self._codeword_guesses = codeword_guesses
        if len(codeword_guesses) != self._output_batch.size:
            raise ValueError(f'Codeword guesses length: {len(codeword_guesses)}'
                             f' MUST be equal to output batch size: {self._output_batch.size}')

        # max_success_probability = tf.Variable(0.0)
        # for codeword_success_prob in codeword_guesses:
        #     if max_success_probability < codeword_success_prob.success_probability:
        #         max_success_probability = codeword_success_prob.success_probability
        # return max_success_probability

        # success_probability_from_guesses = [
        #     codeword_success_prob.success_probability
        #     if batch_codeword == codeword_success_prob.guessed_codeword
        #     else 1 - codeword_success_prob.success_probability
        #     for batch_codeword, codeword_success_prob in zip(self._input_batch.codewords, codeword_guesses)]
        # return sum(success_probability_from_guesses) / self._input_batch.size
        success_probability_from_guesses = [
            codeword_success_prob.success_probability
            for codeword_success_prob in codeword_guesses]
        return sum(success_probability_from_guesses) / self._input_batch.size

    def run_and_compute_average_batch_error_probability(self) -> Tuple[Union[float, EagerTensor],
                                                                       List[CodeWordSuccessProbability]]:
        # loss = 1 - sum([self._compute_one_play_average_batch_success_probability(
        #     codeword_guesses=self._run_and_get_codeword_guesses())
        #     for _ in range(self._options.plays)]
        # ) / self._options.plays
        loss = 1 - self._compute_one_play_average_batch_success_probability(
            codeword_guesses=self._run_and_get_codeword_guesses())
        return loss, self.measurements
