# engine.py
from abc import ABC
import strawberryfields as sf
from strawberryfields.api import Result
from strawberryfields.backends import BaseState
# from tensorflow.python.framework.ops import EagerTensor
from typeguard import typechecked
from csd.codeword import CodeWord
from csd.typings.typing import (Backends, BackendOptions, CodeWordIndices,
                                CodeWordSuccessProbability, EngineRunOptions, MeasuringTypes)
from csd.circuit import Circuit
from typing import List, Optional
import itertools
# from csd.config import logger


class Engine(ABC):
    """ Engine class

    """

    DEFAULT_CUTOFF_DIMENSION = 10

    @typechecked
    def __init__(self, backend: Backends, options: Optional[BackendOptions] = None) -> None:
        self._backend = backend
        self._cutoff_dim = options['cutoff_dim'] if options is not None else self.DEFAULT_CUTOFF_DIMENSION
        self._engine = sf.Engine(backend=self._backend.value,
                                 backend_options=options)

    @property
    def backend_name(self) -> str:
        return self._engine.backend_name

    def _max_probability_codeword(
            self,
            codewords_sucess_probabilities: List[CodeWordSuccessProbability]) -> CodeWordSuccessProbability:
        max_codeword_success_probability = codewords_sucess_probabilities[0]

        for codeword_success_probability in codewords_sucess_probabilities:
            if codeword_success_probability.success_probability > max_codeword_success_probability.success_probability:
                max_codeword_success_probability = codeword_success_probability
        # logger.debug(f"Max probability: {max_codeword_success_probability}")
        return max_codeword_success_probability

    def run_circuit_checking_measuring_type(
            self,
            circuit: Circuit,
            options: EngineRunOptions) -> CodeWordSuccessProbability:

        if options['measuring_type'] is MeasuringTypes.SAMPLING:
            codewords_sucess_probabilities = self._run_circuit_sampling(circuit=circuit, options=options)
        else:
            codewords_sucess_probabilities = self._run_circuit_probabilities(circuit=circuit, options=options)
        # logger.debug(codewords_sucess_probabilities)
        return self._max_probability_codeword(codewords_sucess_probabilities=codewords_sucess_probabilities)

    def _run_circuit_sampling(self,
                              circuit: Circuit,
                              options: EngineRunOptions) -> List[CodeWordSuccessProbability]:
        """Run a circuit experiment doing MeasureFock and performing samplint with nshots
        """
        # if self._engine.backend_name == Backends.GAUSSIAN.value:
        #     return sum([1 for read_value in self._run_circuit(circuit=circuit, options=options).samples
        #                 if read_value[0] == 0]) / options['shots']
        shots = options['shots']
        options['shots'] = 1
        zero_prob = sum([1 for read_value in [self._run_circuit(circuit=circuit, options=options).samples[0][0]
                                              for _ in range(shots)] if read_value == 0]) / shots
        codewords = self._generate_all_codewords(codeword=options['codeword'])
        return [CodeWordSuccessProbability(codeword=codewords[0], success_probability=zero_prob),
                CodeWordSuccessProbability(codeword=codewords[1], success_probability=1 - zero_prob)]

    def _run_circuit_probabilities(self,
                                   circuit: Circuit,
                                   options: EngineRunOptions) -> List[CodeWordSuccessProbability]:
        """Run a circuit experiment computing the fock probability
        """
        options['shots'] = 0
        result = self._run_circuit(circuit=circuit, options=options)
        return self._compute_fock_probabilities_for_all_codewords(state=result.state,
                                                                  codeword=options['codeword'],
                                                                  cutoff_dim=self._cutoff_dim)

    def _get_fock_prob_indices_from_modes(self, codeword: CodeWord, cutoff_dimension: int) -> List[CodeWordIndices]:
        if codeword.size > cutoff_dimension:
            raise ValueError("cutoff dimension MUST be equal or greater than modes")
        codewords = self._generate_all_codewords(codeword)

        return [CodeWordIndices(codeword=codeword,
                                indices=self._convert_word_to_fock_prob_indices(
                                    codeword=codeword,
                                    cutoff_dim=cutoff_dimension))
                for codeword in codewords]

    def _generate_all_codewords(self, codeword: CodeWord) -> List[CodeWord]:
        letters = [codeword.alpha, -codeword.alpha]
        codewords = [CodeWord(size=len(word), alpha_value=codeword.alpha, word=list(word))
                     for word in itertools.product(letters, repeat=codeword.size)]
        return codewords

    def _compute_fock_probabilities_for_all_codewords(self,
                                                      state: BaseState,
                                                      codeword: CodeWord,
                                                      cutoff_dim: int) -> List[CodeWordSuccessProbability]:
        all_codewords_indices = self._get_fock_prob_indices_from_modes(
            codeword=codeword, cutoff_dimension=cutoff_dim)
        # logger.debug(f"codewords indices: {all_codewords_indices}")
        codewords_success_probabilities = [
            CodeWordSuccessProbability(codeword=codeword_indices.codeword,
                                       success_probability=self._compute_fock_prob_one_word(
                                           state=state,
                                           fock_prob_indices_one_word=codeword_indices.indices))
            for codeword_indices in all_codewords_indices]
        # logger.debug(f"Success probabilities: {codewords_success_probabilities}")
        # only_probs = [codeword_succes_prob.success_probability
        # for codeword_succes_prob in codewords_success_probabilities]
        # logger.debug(f"sum of all probs: {sum(only_probs)}")
        return codewords_success_probabilities

    def _run_circuit(self,
                     circuit: Circuit,
                     options: EngineRunOptions) -> Result:
        """ Run an experiment using the engine with the passed options
        """
        # reset the engine if it has already been executed
        if self._engine.run_progs:
            self._engine.reset()

        return self._engine.run(program=circuit.circuit,
                                args=self._parse_circuit_parameters(
                                    circuit=circuit,
                                    options=options),
                                shots=options['shots'])

    def _parse_circuit_parameters(self,
                                  circuit: Circuit,
                                  options: EngineRunOptions) -> dict:
        all_values = [elem for elem in options['codeword'].to_list()]
        for param in options['params']:
            all_values.append(param)

        return {name: value for (name, value) in zip(circuit.parameters.keys(), all_values)}

    def _compute_fock_prob_one_word(self, state: BaseState, fock_prob_indices_one_word: List[List[int]]) -> float:
        return sum([state.fock_prob(fock_prob_indices) for fock_prob_indices in fock_prob_indices_one_word])

    def _convert_word_to_fock_prob_indices(self, codeword: CodeWord, cutoff_dim: int) -> List[List[int]]:
        if codeword.number_minus_alphas == 0:
            return [[0] * codeword.size]

        prob_indices: List[List[int]] = []
        dimensions_more_than_0_photons = [i for i in range(cutoff_dim) if i > 0]
        zero_list = codeword.zero_list.copy()
        minus_indices = codeword.minus_indices
        minus_groups = [p for p in itertools.product(
            dimensions_more_than_0_photons, repeat=codeword.number_minus_alphas)]

        for minus_group in minus_groups:
            for dimension, index in zip(minus_group, minus_indices):
                zero_list[index] = dimension
            prob_indices.append(zero_list.copy())
        return prob_indices
