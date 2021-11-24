# engine.py
from typing import List
from csd.batch import Batch
from csd.codeword import CodeWord
from .engine import Engine
from strawberryfields.api import Result
from strawberryfields.backends.tfbackend.states import FockStateTF
import tensorflow as tf
from tensorflow.python.framework.ops import EagerTensor
from csd.typings.typing import (BatchSuccessProbability, CodeWordSuccessProbability, MeasuringTypes, TFEngineRunOptions)
from csd.circuit import Circuit
from csd.config import logger
import numpy as np


class TFEngine(Engine):
    """ Tensor Flow Engine class

    """

    #
    #   TRAINING: TF CIRCUIT SAMPLING
    #

    def _compute_error(self, tf_state: FockStateTF, mode: int, alpha_value: float) -> EagerTensor:
        one_mode_mean_photons = self._get_mean_photon_one_mode(tf_state, mode=mode)
        exact_mean_photons = np.abs(alpha_value)**2
        return tf.math.abs(one_mode_mean_photons - exact_mean_photons)

    def _get_mean_photon_one_mode(self, tf_state: FockStateTF, mode: int) -> EagerTensor:
        # it is a tuple of mean, variance. We only want the mean
        return tf_state.mean_photon(mode)[0]

    def run_tf_circuit_sampling(
            self,
            circuit: Circuit,
            options: TFEngineRunOptions) -> EagerTensor:

        options['shots'] = 1
        tf_state: FockStateTF = self._run_tf_circuit(circuit=circuit, options=options).state

        error_all_modes = [self._compute_error(tf_state,
                                               mode=mode_i,
                                               alpha_value=options['input_batch'].alpha)
                           for mode_i in range(tf_state.num_modes)]
        # logger.debug(f'error_all_modes: {error_all_modes}')

        batch_error = tf.reduce_mean(error_all_modes)
        # logger.debug(f'batch error: {batch_error}')
        return batch_error

    #
    #   TRAINING: TF CIRCUIT PROBABILITIES
    #

    def run_tf_circuit_checking_measuring_type(
            self,
            circuit: Circuit,
            options: TFEngineRunOptions) -> List[CodeWordSuccessProbability]:

        if options['measuring_type'] is not MeasuringTypes.PROBABILITIES:
            raise ValueError("Run TF Circuit only accepts Probabilities MeasuringTypes")

        batch_success_probabilities = self._run_tf_circuit_probabilities(circuit=circuit, options=options)

        return self._compute_max_probability_for_all_codewords(batch_success_probabilities)

    def _compute_max_probability_for_all_codewords(
            self,
            batch_success_probabilities: List[List[CodeWordSuccessProbability]]) -> List[CodeWordSuccessProbability]:
        max_probs = [self._max_probability_codeword(codewords_success_probabilities=codewords_success_probabilities)
                     for codewords_success_probabilities in batch_success_probabilities]
        logger.debug(f'max_probs: {max_probs}')
        return max_probs

    def _run_tf_circuit_probabilities(self,
                                      circuit: Circuit,
                                      options: TFEngineRunOptions) -> List[List[CodeWordSuccessProbability]]:
        """Run a circuit experiment computing the fock probability
        """

        options['shots'] = 0
        result = self._run_tf_circuit(circuit=circuit, options=options)
        return self._compute_tf_fock_probabilities_for_all_codewords(state=result.state,
                                                                     input_codeword=options['input_batch'].one_codeword,
                                                                     output_batch=options['output_batch'],
                                                                     cutoff_dim=self._cutoff_dim)

    def _compute_tf_fock_probabilities_for_all_codewords(self,
                                                         state: FockStateTF,
                                                         input_codeword: CodeWord,
                                                         output_batch: Batch,
                                                         cutoff_dim: int) -> List[List[CodeWordSuccessProbability]]:
        all_codewords_indices = self._get_fock_prob_indices_from_modes(
            output_codeword=output_batch.one_codeword, cutoff_dimension=cutoff_dim)

        success_probabilities_batches = [
            BatchSuccessProbability(codeword_indices=codeword_indices,
                                    success_probability=self._compute_tf_fock_prob_one_codeword_indices(
                                        state=state,
                                        fock_prob_indices_one_word=codeword_indices.indices))
            for codeword_indices in all_codewords_indices]

        return self._compute_codewords_success_probabilities(
            input_codeword=input_codeword,
            batch=output_batch,
            success_probabilities_batches=success_probabilities_batches)

    def _compute_codewords_success_probabilities(
            self,
            input_codeword: CodeWord,
            batch: Batch,
            success_probabilities_batches: List[BatchSuccessProbability]) -> List[List[CodeWordSuccessProbability]]:
        return [self._compute_one_batch_codewords_success_probabilities(
            input_codeword=input_codeword,
            index_codeword=index_codeword,
            success_probabilities_batches=success_probabilities_batches)
            for index_codeword in range(batch.size)]

    def _compute_one_batch_codewords_success_probabilities(
            self,
            input_codeword: CodeWord,
            index_codeword: int,
            success_probabilities_batches: List[BatchSuccessProbability]) -> List[CodeWordSuccessProbability]:
        return [CodeWordSuccessProbability(
            input_codeword=input_codeword,
            guessed_codeword=CodeWord(size=input_codeword.size,
                                      alpha_value=input_codeword.alpha),
            output_codeword=batch_success_probabilities.codeword_indices.codeword,
            success_probability=self._compute_one_codeword_success_probability(
                index_codeword=index_codeword,
                success_probabilities_indices=batch_success_probabilities.success_probability),
            counts=tf.Variable(0))
            for batch_success_probabilities in success_probabilities_batches]

    def _compute_one_codeword_success_probability(
            self,
            index_codeword: int,
            success_probabilities_indices: List[EagerTensor]) -> EagerTensor:
        return sum([tf.gather(success_probability, indices=[index_codeword])
                    for success_probability in success_probabilities_indices])

    def _compute_tf_fock_prob_one_codeword_indices(self,
                                                   state: FockStateTF,
                                                   fock_prob_indices_one_word: List[List[int]]) -> List[EagerTensor]:
        return [state.fock_prob(fock_prob_indices) for fock_prob_indices in fock_prob_indices_one_word]

    #
    #   TESTING: TF
    #

    def run_tf_circuit_training(self,
                                circuit: Circuit,
                                options: TFEngineRunOptions) -> List[CodeWordSuccessProbability]:
        """Run a circuit experiment doing MeasureFock and performing sampling with nshots
        """
        shots = options['shots']
        options['shots'] = 1
        alpha_value = options['input_batch'].one_codeword.alpha
        batch_success_probabilities = self._init_batch_success_probabilities(input_batch=options['input_batch'],
                                                                             output_batch=options['output_batch'])
        # logger.debug(f'INIT batch success probability: {batch_success_probabilities}')
        for _ in range(shots):
            output_codewords = self._convert_batch_sampling_output_to_codeword_list(
                alpha_value=alpha_value,
                batch_samples=self._run_tf_circuit(circuit=circuit, options=options).samples)
            self._assign_counts_to_each_actual_codeword_result(result_codewords=output_codewords,
                                                               batch_success_probabilities=batch_success_probabilities)

        batch_success_probabilities = self._compute_average_batch_success_probabilities(
            batch_success_probabilities=batch_success_probabilities,
            shots=shots)
        return self._compute_max_probability_for_all_codewords(batch_success_probabilities)

    def _compute_average_batch_success_probabilities(
            self,
            batch_success_probabilities: List[List[CodeWordSuccessProbability]],
            shots: int) -> List[List[CodeWordSuccessProbability]]:
        if shots <= 0:
            raise ValueError(f"shots MUST be greater than zero. Current value: {shots}")
        # logger.debug(f'batch success probability BEFORE shots: {batch_success_probabilities}')
        for input_codeword_success_probabilities in batch_success_probabilities:
            for codeword_success_probability in input_codeword_success_probabilities:
                codeword_success_probability.success_probability = codeword_success_probability.counts / shots

        return batch_success_probabilities

    def _assign_counts_to_each_actual_codeword_result(
            self,
            result_codewords: List[CodeWord],
            batch_success_probabilities: List[List[CodeWordSuccessProbability]]) -> None:
        if len(result_codewords) != len(batch_success_probabilities):
            raise ValueError(
                f'result_codewords size: {len(result_codewords)} and '
                f'batch_success_probabilities length {len(batch_success_probabilities)} differs!')

        # logger.debug(f'Result output codewords: {result_codewords}')
        for result_codeword, input_codeword_success_probabilities in zip(result_codewords, batch_success_probabilities):
            found = False
            for codeword_success_probability in input_codeword_success_probabilities:
                if not found and codeword_success_probability.output_codeword == result_codeword:
                    found = True
                    codeword_success_probability.counts.assign_add(1)

    def _convert_sampling_output_to_codeword(self, alpha_value: float, one_codeword_output: EagerTensor) -> CodeWord:
        ON = -1
        OFF = 1
        word = [alpha_value * (ON if one_mode_output != 0 else OFF) for one_mode_output in one_codeword_output[0]]
        return CodeWord(word=tf.constant(word))

    def _convert_batch_sampling_output_to_codeword_list(self,
                                                        alpha_value: float,
                                                        batch_samples: List[EagerTensor]) -> List[CodeWord]:
        return [self._convert_sampling_output_to_codeword(alpha_value=alpha_value,
                                                          one_codeword_output=one_codeword_output)
                for one_codeword_output in batch_samples]

    def _init_one_input_codeword_success_probabilities(self,
                                                       input_codeword: CodeWord,
                                                       output_batch: Batch) -> List[CodeWordSuccessProbability]:
        return [CodeWordSuccessProbability(input_codeword=input_codeword,
                                           guessed_codeword=input_codeword,
                                           output_codeword=output_codeword,
                                           success_probability=tf.Variable(0.0),
                                           counts=tf.Variable(0))
                for output_codeword in output_batch.codewords]

    def _init_batch_success_probabilities(self,
                                          input_batch: Batch,
                                          output_batch: Batch) -> List[List[CodeWordSuccessProbability]]:

        return [self._init_one_input_codeword_success_probabilities(input_codeword=input_codeword,
                                                                    output_batch=output_batch)
                for input_codeword in input_batch.codewords]

    #
    #   TRAINING AND TESTING: TF
    #

    def _run_tf_circuit(self,
                        circuit: Circuit,
                        options: TFEngineRunOptions) -> Result:
        """ Run an experiment using the engine with the passed options
        """
        # reset the engine if it has already been executed
        if self._engine.run_progs:
            self._engine.reset()

        return self._engine.run(program=circuit.circuit,
                                args=self._parse_tf_circuit_parameters(
                                    circuit=circuit,
                                    options=options),
                                shots=options['shots'])

    def _parse_tf_circuit_parameters(self,
                                     circuit: Circuit,
                                     options: TFEngineRunOptions) -> dict:
        all_values = options['input_batch'].letters
        for param in options['params']:
            all_values.append(param)

        return {name: value for (name, value) in zip(circuit.parameters.keys(), all_values)}
