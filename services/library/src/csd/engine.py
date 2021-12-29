# engine.py
from abc import ABC
import strawberryfields as sf
from strawberryfields.result import Result
from strawberryfields.backends import BaseState
from typeguard import typechecked
from csd.codeword import CodeWord
from csd.typings.typing import (Backends, BackendOptions, CodeWordIndices,
                                CodeWordSuccessProbability, EngineRunOptions, MeasuringTypes)
from csd.circuit import Circuit
from typing import List, Optional, Union
import itertools
from tensorflow.python.framework.ops import EagerTensor
import tensorflow as tf

from csd.util import generate_all_codewords_from_codeword, generate_measurement_matrices
# from csd.config import logger


class Engine(ABC):
    """ Engine class

    """

    DEFAULT_CUTOFF_DIMENSION = 10

    @typechecked
    def __init__(self,
                 number_modes: int,
                 engine_backend: Optional[Backends] = Backends.FOCK,
                 options: Optional[BackendOptions] = None) -> None:
        self._backend: Backends = engine_backend if engine_backend is not None else Backends.FOCK
        self._cutoff_dim = options['cutoff_dim'] if options is not None else self.DEFAULT_CUTOFF_DIMENSION
        self._engine = sf.Engine(backend=self._backend.value,
                                 backend_options=options)
        self._measurement_matrices = generate_measurement_matrices(num_modes=number_modes, cutoff_dim=self._cutoff_dim)

    @property
    def backend_name(self) -> str:
        return self._engine.backend_name

    def _apply_guess_strategy_move_noise_to_ancillas(
            self,
            max_success_probability_codeword_selected: CodeWordSuccessProbability,
            codewords_success_probabilities: List[CodeWordSuccessProbability]) -> CodeWordSuccessProbability:
        """ Create a codeword with the same size as the codeword to guess (input codeword)
            using the last modes from the output codeword -> meaning, ignoring the ancilla modes.

            Doing that, we have two chances for each ancilla to detect one input codeword
                (when ancilla mode measurement is 0 or 1).
            What it means then, is that the optimization is going to move all possible
                communication "noise" to the ancillas.
            The more ancillas you add, the better to detect the appropiate input codeword.
            Beware of the computation time, though.
        """
        self._error_when_codeword_to_guess_is_larger_than_output_codeword(
            max_success_probability_codeword_selected.output_codeword,
            max_success_probability_codeword_selected.guessed_codeword)

        guessed_codeword = self._create_codeword_from_last_output_modes(
            max_success_probability_codeword_selected.output_codeword,
            max_success_probability_codeword_selected.guessed_codeword)
        success_probability = self._compute_guessed_codeword_probability(guessed_codeword,
                                                                         codewords_success_probabilities)

        return CodeWordSuccessProbability(input_codeword=max_success_probability_codeword_selected.input_codeword,
                                          guessed_codeword=guessed_codeword,
                                          output_codeword=max_success_probability_codeword_selected.output_codeword,
                                          success_probability=success_probability,
                                          counts=max_success_probability_codeword_selected.counts)

    def _create_codeword_from_last_output_modes(self,
                                                output_codeword: CodeWord,
                                                codeword_to_guess: CodeWord) -> CodeWord:
        return CodeWord(word=output_codeword.word[-codeword_to_guess.size:])

    def _compute_guessed_codeword_probability(
            self,
            guessed_codeword: CodeWord,
            codewords_success_probabilities: List[CodeWordSuccessProbability]) -> Union[float, EagerTensor]:
        """ Sum each probability from an output codeword (with ancillas) that matches
            the guessed codeword (input codeword size).

            Example:
                output codewords: [0,0], [0,1], [1,0], [1,1]
                guessed codeword: [0]
                Result: sum probabilites from [0,0] and [1,0] because those codewords contain
                        the guessed codeword [0] as the last digit.

        Args:
            guessed_codeword (CodeWord): [description]
            codewords_success_probabilities (List[CodeWordSuccessProbability]): [description]

        Returns:
            Union[float, EagerTensor]: [description]
        """
        return tf.math.add_n([codeword_success_probabilities.success_probability
                              for codeword_success_probabilities in codewords_success_probabilities
                              if self._create_codeword_from_last_output_modes(
                                  output_codeword=codeword_success_probabilities.output_codeword,
                                  codeword_to_guess=guessed_codeword) == guessed_codeword])

    def _error_when_codeword_to_guess_is_larger_than_output_codeword(self,
                                                                     output_codeword: CodeWord,
                                                                     codeword_to_guess: CodeWord):
        if output_codeword.size < codeword_to_guess.size:
            raise ValueError(f"Output codeword size: {output_codeword.size} MUST NOT be"
                             f"larger than codeword to guess size: {codeword_to_guess.size}")

    def _max_probability_codeword(
            self,
            codewords_success_probabilities: List[CodeWordSuccessProbability]) -> CodeWordSuccessProbability:
        max_codeword_success_probability = codewords_success_probabilities[0]

        for codeword_success_probability in codewords_success_probabilities:
            if codeword_success_probability.success_probability > max_codeword_success_probability.success_probability:
                max_codeword_success_probability = codeword_success_probability

        return self._apply_guess_strategy_move_noise_to_ancillas(max_codeword_success_probability,
                                                                 codewords_success_probabilities)

    def run_circuit_checking_measuring_type(
            self,
            circuit: Circuit,
            options: EngineRunOptions) -> CodeWordSuccessProbability:

        if options['measuring_type'] is MeasuringTypes.SAMPLING:
            codewords_success_probabilities = self._run_circuit_sampling(circuit=circuit, options=options)
        else:
            codewords_success_probabilities = self._run_circuit_probabilities(circuit=circuit, options=options)

        return self._max_probability_codeword(codewords_success_probabilities=codewords_success_probabilities)

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
        output_codewords = generate_all_codewords_from_codeword(codeword=options['output_codeword'])
        return [CodeWordSuccessProbability(input_codeword=options['input_codeword'],
                                           guessed_codeword=CodeWord(size=options['input_codeword'].size,
                                                                     alpha_value=options['input_codeword'].alpha),
                                           output_codeword=output_codewords[0],
                                           success_probability=zero_prob,
                                           counts=0),
                CodeWordSuccessProbability(input_codeword=options['input_codeword'],
                                           guessed_codeword=CodeWord(size=options['input_codeword'].size,
                                                                     alpha_value=options['input_codeword'].alpha),
                                           output_codeword=output_codewords[1],
                                           success_probability=1 - zero_prob,
                                           counts=0)]

    def _run_circuit_probabilities(self,
                                   circuit: Circuit,
                                   options: EngineRunOptions) -> List[CodeWordSuccessProbability]:
        """Run a circuit experiment computing the fock probability
        """
        options['shots'] = 0
        result = self._run_circuit(circuit=circuit, options=options)
        return self._compute_fock_probabilities_for_all_codewords(state=result.state,
                                                                  input_codeword=options['input_codeword'],
                                                                  output_codeword=options['output_codeword'],
                                                                  cutoff_dim=self._cutoff_dim)

    def _get_fock_prob_indices_from_modes(self,
                                          output_codeword: CodeWord,
                                          cutoff_dimension: int) -> List[CodeWordIndices]:
        if output_codeword.size > cutoff_dimension:
            raise ValueError("cutoff dimension MUST be equal or greater than modes")
        output_codewords = generate_all_codewords_from_codeword(output_codeword)

        return [CodeWordIndices(codeword=output_codeword,
                                indices=self._convert_word_to_fock_prob_indices(
                                    codeword=output_codeword,
                                    cutoff_dim=cutoff_dimension))
                for output_codeword in output_codewords]

    def _compute_fock_probabilities_for_all_codewords(self,
                                                      state: BaseState,
                                                      input_codeword: CodeWord,
                                                      output_codeword: CodeWord,
                                                      cutoff_dim: int) -> List[CodeWordSuccessProbability]:
        all_codewords_indices = self._get_fock_prob_indices_from_modes(
            output_codeword=output_codeword, cutoff_dimension=cutoff_dim)
        return [CodeWordSuccessProbability(
            input_codeword=input_codeword,
            guessed_codeword=CodeWord(size=input_codeword.size,
                                      alpha_value=input_codeword.alpha),
            output_codeword=codeword_indices.codeword,
            success_probability=self._compute_fock_prob_one_word(
                state=state,
                fock_prob_indices_one_word=codeword_indices.indices),
            counts=0)
            for codeword_indices in all_codewords_indices]

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
        all_values = [elem for elem in options['input_codeword'].to_list()]
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
