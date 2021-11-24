# cost_function.py
from abc import ABC
from typing import List, Union
# import numpy as np

from tensorflow.python.framework.ops import EagerTensor
from csd.batch import Batch
from csd.codeword import CodeWord
from csd.tf_engine import TFEngine
from csd.typings.typing import (Backends, CodeWordSuccessProbability, EngineRunOptions, TFEngineRunOptions)
from csd.typings.cost_function import CostFunctionOptions
from csd.config import logger
# import tensorflow as tf


class CostFunction(ABC):
    """Class to encapsulate cost function logic with the probability computation
    """

    def __init__(self, batch: Batch, params: List[float], options: CostFunctionOptions) -> None:
        self._params = params
        self._options = options
        self._input_batch = batch
        self._output_batch = Batch(size=self._input_batch.size,
                                   word_size=self._options.circuit.number_modes,
                                   alpha_value=self._input_batch.alpha,
                                   random_words=False)

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
                    measuring_type=self._options.measuring_type))
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

        # success_probability_from_guesses = []
        # for input_codeword in self._input_batch.codewords:
        #     found = False
        #     for result_codeword_success_probability in codeword_guesses:
        #         if not found and input_codeword == result_codeword_success_probability.input_codeword:
        #             found = True
        #             (success_probability_from_guesses.append(result_codeword_success_probability.success_probability)
        #              if input_codeword == result_codeword_success_probability.guessed_codeword
        #              else success_probability_from_guesses.append(
        #                  tf.subtract(tf.constant(1.0),
        #                              result_codeword_success_probability.success_probability)))
        #     if not found:
        #         raise ValueError(f"input codeword: {input_codeword} not found as result")

        # logger.debug(f'success_probability_from_guesses: {success_probability_from_guesses}')
        # avg_succ = tf.divide(tf.math.add_n(success_probability_from_guesses),
        #                      tf.constant(self._input_batch.size, dtype=tf.float32))
        # logger.debug(f'average success: {avg_succ}')
        # return avg_succ
        success_probability_from_guesses = [
            codeword_success_prob.success_probability
            if batch_codeword == codeword_success_prob.guessed_codeword
            else 1 - codeword_success_prob.success_probability
            for batch_codeword, codeword_success_prob in zip(self._input_batch.codewords, codeword_guesses)]
        return sum(success_probability_from_guesses) / self._input_batch.size

    def run_and_compute_average_batch_error_probability(self) -> Union[float, EagerTensor]:
        # probs = self._options.engine.run_tf_circuit_sampling(
        #     circuit=self._options.circuit,
        #     options=TFEngineRunOptions(
        #         params=self._params,
        #         input_batch=self._input_batch,
        #         output_batch=self._output_batch,
        #         shots=self._options.shots,
        #         measuring_type=self._options.measuring_type))
        # logger.debug(f"probs: {probs}")
        # loss = 1 - probs
        loss = 1 - sum([self._compute_one_play_average_batch_success_probability(
            codeword_guesses=self._run_and_get_codeword_guesses())
            for _ in range(self._options.plays)]
        ) / self._options.plays
        # loss = (tf.subtract(
        #     tf.constant(1.0),
        #     self._compute_one_play_average_batch_success_probability(
        #         codeword_guesses=self._run_and_get_codeword_guesses())))
        logger.debug(f'loss: {loss}')
        return loss
