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


class TFEngine(Engine):
    """ Tensor Flow Engine class

    """

    def _compute_prob(self, value: EagerTensor) -> EagerTensor:
        if value <= 1:
            return value
        return tf.divide(value, value)

    def run_tf_circuit_sampling(
            self,
            circuit: Circuit,
            options: TFEngineRunOptions) -> EagerTensor:

        options['shots'] = 1
        tf_result: Result = self._run_tf_circuit(circuit=circuit, options=options)
        logger.debug(f'tf_result: {tf_result}')
        tf_state: FockStateTF = tf_result.state
        # tf_samples = tf_result.samples
        # tf_all_probs = tf_state.all_fock_probs
        # fock_prob = tf_state.fock_prob([0])
        mean, var = tf_state.mean_photon(0)
        logger.debug(f'mean: {mean} and var: {var}')
        # round_mean = tf.round(mean)
        # logger.debug(f'round_mean: {round_mean}')

        # logger.debug(f'tf_state: {tf_state}, num_modes: {tf_state.num_modes}')
        # logger.debug(f'tf_samples: {tf_samples}')
        # logger.debug(f'tf_all_probs: {tf_all_probs}')
        # logger.debug(f'fock_prob: {fock_prob}')
        # for mode_i in range(tf_state.num_modes):
        #     logger.debug(f'mean_photon({mode_i}): {tf_state.mean_photon(mode_i)}')

        # probs = tf.map_fn(self._compute_prob, mean)
        probs = self._compute_prob(mean[0])
        # probs_mean = tf.reduce_mean(floor_photons)
        # logger.debug(f'probs_mean: {probs_mean}')
        # probs = sum(floor_photons) / shots
        logger.debug(f'probs: {probs}')
        return probs

    def run_tf_circuit_checking_measuring_type(
            self,
            circuit: Circuit,
            options: TFEngineRunOptions) -> List[CodeWordSuccessProbability]:

        batch_success_probabilities = (self._run_tf_circuit_probabilities(circuit=circuit, options=options)
                                       if options['measuring_type'] is MeasuringTypes.PROBABILITIES
                                       else self._run_tf_circuit_sampling(circuit=circuit, options=options))

        logger.debug(f'batch success probability: {batch_success_probabilities}')
        # max_probs = []
        # for codewords_success_probabilities in batch_success_probabilities:
        #     one_max = self._max_probability_codeword(codewords_success_probabilities=codewords_success_probabilities)
        #     logger.debug(f'codewords_success_probabilities: {codewords_success_probabilities}')
        #     logger.debug(f'one_max: {one_max}')
        #     max_probs.append(one_max)

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

    def _init_one_input_codeword_success_probabilities(self,
                                                       input_codeword: CodeWord,
                                                       output_batch: Batch,
                                                       tf_state: FockStateTF) -> List[CodeWordSuccessProbability]:
        mean, var = tf_state.mean_photon(0)
        logger.debug(f'mean: {mean} and var: {var}')

        return [CodeWordSuccessProbability(input_codeword=input_codeword,
                                           guessed_codeword=input_codeword,
                                           output_codeword=output_codeword,
                                           success_probability=self._compute_prob(mean[index_output]))
                for index_output, output_codeword in enumerate(output_batch.codewords)]

    def _init_batch_success_probabilities(self,
                                          input_batch: Batch,
                                          output_batch: Batch,
                                          tf_state: FockStateTF) -> List[List[CodeWordSuccessProbability]]:

        return [self._init_one_input_codeword_success_probabilities(input_codeword=input_codeword,
                                                                    output_batch=output_batch,
                                                                    tf_state=tf_state)
                for input_codeword in input_batch.codewords]

    def _convert_sampling_output_to_codeword(self, alpha_value: float, one_codeword_output: EagerTensor) -> CodeWord:
        ON = -1
        OFF = 1
        word = [alpha_value * (ON if one_mode_output != 0 else OFF) for one_mode_output in one_codeword_output[0]]
        # logger.debug(f'one_codeword_output: {one_codeword_output}')
        # logger.debug(f'one_codeword_output[0]: {one_codeword_output[0]}')
        # word = [alpha_value * tf.cast(ON * (1 + one_mode_output - one_mode_output) if one_mode_output != 0
        #                               else one_mode_output + OFF, dtype=tf.float64)
        #         for one_mode_output in one_codeword_output[0]]
        # tf_word = tf.reshape(tf.concat(word, axis=0), [len(word), ])
        # logger.debug(f'tf_word: {tf_word}, type: {type(tf_word)}')

        # logger.debug(f'tf_word: {tf_word}, type: {type(tf_word)}')
        cw = CodeWord(word=word)
        logger.debug(f'codeword: {cw}')
        return cw

    def _convert_batch_sampling_output_to_codeword_list(self,
                                                        alpha_value: float,
                                                        batch_samples: List[EagerTensor]) -> List[CodeWord]:
        # logger.debug(f'result samples: {batch_samples}')
        return [self._convert_sampling_output_to_codeword(alpha_value=alpha_value,
                                                          one_codeword_output=one_codeword_output)
                for one_codeword_output in batch_samples]

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
