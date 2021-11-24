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
        tf_result: Result = self._run_tf_circuit(circuit=circuit, options=options)
        # logger.debug(f'tf_result: {tf_result}')
        tf_state: FockStateTF = tf_result.state
        # tf_samples = tf_result.samples
        # tf_all_probs = tf_state.all_fock_probs
        # fock_prob = tf_state.fock_prob([0])
        # mean, var = tf_state.mean_photon(0)
        # logger.debug(f'mean: {mean} and var: {var}')
        # round_mean = tf.round(mean)
        # logger.debug(f'round_mean: {round_mean}')

        # logger.debug(f'tf_state: {tf_state}, num_modes: {tf_state.num_modes}')
        # logger.debug(f'tf_samples: {tf_samples}')
        # logger.debug(f'tf_all_probs: {tf_all_probs}')
        # logger.debug(f'fock_prob: {fock_prob}')
        # means =
        # for mode_i in range(tf_state.num_modes):
        #      logger.debug(f'mean_photon({mode_i}): {tf_state.mean_photon(mode_i)}')

        # means = tf.map_fn(self._compute_prob, mean)
        error_all_modes = [self._compute_error(tf_state,
                                               mode=mode_i,
                                               alpha_value=options['input_batch'].alpha)
                           for mode_i in range(tf_state.num_modes)]
        logger.debug(f'error_all_modes: {error_all_modes}')
        # probs_mean = tf.reduce_mean(floor_photons)
        # logger.debug(f'probs_mean: {probs_mean}')
        # probs = sum(floor_photons) / shots
        batch_error = tf.reduce_mean(error_all_modes)
        logger.debug(f'batch error: {batch_error}')
        return batch_error

    def run_tf_circuit_checking_measuring_type(
            self,
            circuit: Circuit,
            options: TFEngineRunOptions) -> List[CodeWordSuccessProbability]:

        batch_success_probabilities = (self._run_tf_circuit_probabilities(circuit=circuit, options=options)
                                       if options['measuring_type'] is MeasuringTypes.PROBABILITIES
                                       else self._run_tf_circuit_sampling(circuit=circuit, options=options))

        logger.debug(f'batch success probability: {batch_success_probabilities}')

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

    def _run_tf_circuit_sampling(self,
                                 circuit: Circuit,
                                 options: TFEngineRunOptions) -> List[List[CodeWordSuccessProbability]]:
        """Run a circuit experiment doing MeasureFock and performing sampling with nshots
        """
        options['shots'] = 1

        tf_state: FockStateTF = self._run_tf_circuit(circuit=circuit, options=options).state
        return self._init_batch_success_probabilities(input_batch=options['input_batch'],
                                                      output_batch=options['output_batch'],
                                                      tf_state=tf_state)

    def _compute_success_prob(self,
                              output_codeword: CodeWord,
                              tf_state: FockStateTF,
                              index_input_codeword: int) -> EagerTensor:

        mean, _ = tf_state.mean_photon(0)
        # logger.debug(f'mean: {mean} and var: {var}')

        # mean_photons_prob = self._compute_prob(mean[index_input_codeword])
        # is_alpha = output_codeword.word[0] > 0

        # succ_prob = 1 - mean_photons_prob if is_alpha else mean_photons_prob
        mean_photons_prob = mean[index_input_codeword]
        logger.debug(f'mean_photons_prob: {mean_photons_prob}')
        return mean_photons_prob

    def _init_one_input_codeword_success_probabilities(self,
                                                       input_codeword: CodeWord,
                                                       output_batch: Batch,
                                                       tf_state: FockStateTF,
                                                       index_input_codeword: int) -> List[CodeWordSuccessProbability]:

        return [CodeWordSuccessProbability(input_codeword=input_codeword,
                                           guessed_codeword=input_codeword,
                                           output_codeword=output_codeword,
                                           success_probability=self._compute_success_prob(
                                               output_codeword=output_codeword,
                                               tf_state=tf_state,
                                               index_input_codeword=index_input_codeword))
                for output_codeword in output_batch.codewords]

    def _init_batch_success_probabilities(self,
                                          input_batch: Batch,
                                          output_batch: Batch,
                                          tf_state: FockStateTF) -> List[List[CodeWordSuccessProbability]]:

        return [self._init_one_input_codeword_success_probabilities(input_codeword=input_codeword,
                                                                    output_batch=output_batch,
                                                                    tf_state=tf_state,
                                                                    index_input_codeword=index)
                for index, input_codeword in enumerate(input_batch.codewords)]

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
                success_probabilities_indices=batch_success_probabilities.success_probability))
                for batch_success_probabilities in success_probabilities_batches]

    def _compute_one_codeword_success_probability(
            self,
            index_codeword: int,
            success_probabilities_indices: List[EagerTensor]) -> EagerTensor:
        return sum([tf.gather(success_probability, indices=[index_codeword])
                    for success_probability in success_probabilities_indices])

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

    def _compute_tf_fock_prob_one_codeword_indices(self,
                                                   state: FockStateTF,
                                                   fock_prob_indices_one_word: List[List[int]]) -> List[EagerTensor]:
        return [state.fock_prob(fock_prob_indices) for fock_prob_indices in fock_prob_indices_one_word]
