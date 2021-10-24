# cost_function.py
from abc import ABC
from typing import List
# import numpy as np

# from tensorflow.python.framework.ops import EagerTensor

from csd.batch import Batch
from csd.codeword import CodeWord
from csd.typings.typing import CodeWordSuccessProbability, EngineRunOptions
from csd.typings.cost_function import CostFunctionOptions
# from csd.config import logger


class CostFunction(ABC):
    """Class to encapsulate cost function logic with the probability computation
    """

    def __init__(self, batch: Batch, params: List[float], options: CostFunctionOptions) -> None:
        self._batch = batch
        self._params = params
        self._options = options

    def _run_and_compute_codewords_success_probabilities(self) -> List[List[CodeWordSuccessProbability]]:
        # if self._options.backend_name == Backends.TENSORFLOW.value:
        #     return self._options.engine.run_circuit_checking_measuring_type(
        #         circuit=self._options.circuit,
        #         options=EngineRunOptions(
        #             params=self._params,
        #             batch_or_codeword=self._batch,
        #             shots=self._options.shots,
        #             measuring_type=self._options.measuring_type))
        return [self._options.engine.run_circuit_checking_measuring_type(
            circuit=self._options.circuit,
            options=EngineRunOptions(
                params=self._params,
                codeword=codeword,
                shots=self._options.shots,
                measuring_type=self._options.measuring_type))
                for codeword in self._batch.codewords]

    def _find_codeword_success_probability(self,
                                           codeword_success_probabilities: List[CodeWordSuccessProbability],
                                           batch_codeword: CodeWord) -> float:
        for codeword_success_probability in codeword_success_probabilities:
            if codeword_success_probability.codeword == batch_codeword:
                return codeword_success_probability.success_probability
        raise ValueError(f'batch_codeword: {batch_codeword.to_list()} not found in codewords_success_probabilities')

    def _compute_batch_average_success_probabilities(
            self,
            codewords_success_probabilities: List[List[CodeWordSuccessProbability]]) -> float:

        # return sum([self._find_codeword_success_probability(
        #               codeword_success_probabilities = codeword_success_probabilities,
        #               batch_codeword = batch_codeword)
        #             for batch_codeword, codeword_success_probabilities in zip(self._batch.codewords,
        #                                                                       codewords_success_probabilities)]
        #           ) / self._batch.size
        #
        probs = []
        for batch_codeword, codeword_success_probabilities in zip(self._batch.codewords,
                                                                  codewords_success_probabilities):
            probs.append(self._find_codeword_success_probability(
                codeword_success_probabilities=codeword_success_probabilities,
                batch_codeword=batch_codeword))
        average = sum(probs) / self._batch.size
        # logger.debug(f'probs: {probs}, batch_size: {self._batch.size} and average: {average}')
        return average

    def run_and_compute_average_batch_error_probability(self) -> float:
        codewords_success_probabilities = self._run_and_compute_codewords_success_probabilities()
        # logger.debug(f'codewords sucess probability: {codewords_success_probabilities}')
        average_batch_success_probability = self._compute_batch_average_success_probabilities(
            codewords_success_probabilities=codewords_success_probabilities)
        return 1 - average_batch_success_probability
