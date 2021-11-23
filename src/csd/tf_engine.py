# engine.py
from csd.batch import Batch
from csd.codeword import CodeWord
from .engine import Engine
from strawberryfields.api import Result
from strawberryfields.backends.tfbackend.states import FockStateTF
import tensorflow as tf
from tensorflow.python.framework.ops import EagerTensor
from csd.typings.typing import (BatchSuccessProbability, CodeWordSuccessProbability, MeasuringTypes, TFEngineRunOptions)
from csd.circuit import Circuit
from typing import List
from csd.config import logger


class TFEngine(Engine):
    """ Tensor Flow Engine class

    """

    def run_tf_circuit_checking_measuring_type(
            self,
            circuit: Circuit,
            options: TFEngineRunOptions) -> List[CodeWordSuccessProbability]:

        batch_success_probabilities = (self._run_tf_circuit_probabilities(circuit=circuit, options=options)
                                       if options['measuring_type'] is MeasuringTypes.PROBABILITIES
                                       else self._run_tf_circuit_sampling(circuit=circuit, options=options))

        logger.debug(f'batch success probability: {batch_success_probabilities}')
        max_probs = []
        for codewords_success_probabilities in batch_success_probabilities:
            one_max = self._max_probability_codeword(codewords_success_probabilities=codewords_success_probabilities)
            logger.debug(f'codewords_success_probabilities: {codewords_success_probabilities}')
            logger.debug(f'one_max: {one_max}')
            max_probs.append(one_max)

        # max_probs = [self._max_probability_codeword(codewords_success_probabilities=codewords_success_probabilities)
        #              for codewords_success_probabilities in batch_success_probabilities]
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
        shots = options['shots']
        options['shots'] = 1
        alpha_value = options['input_batch'].one_codeword.alpha
        batch_success_probabilities = self._init_batch_success_probabilities(input_batch=options['input_batch'],
                                                                             output_batch=options['output_batch'],
                                                                             all_counts=options['all_counts'])
        # logger.debug(f'INIT batch success probability: {batch_success_probabilities}')
        for _ in range(shots):
            output_codewords = self._convert_batch_sampling_output_to_codeword_list(
                alpha_value=alpha_value,
                batch_samples=self._run_tf_circuit(circuit=circuit, options=options).samples)
            self._assign_counts_to_each_actual_codeword_result(result_codewords=output_codewords,
                                                               batch_success_probabilities=batch_success_probabilities,
                                                               all_counts=options['all_counts'])

        return self._compute_average_batch_success_probabilities(
            batch_success_probabilities=batch_success_probabilities,
            shots=shots)

    def _compute_average_batch_success_probabilities(
            self,
            batch_success_probabilities: List[List[CodeWordSuccessProbability]],
            shots: int) -> List[List[CodeWordSuccessProbability]]:
        if shots <= 0:
            raise ValueError(f"shots MUST be greater than zero. Current value: {shots}")
        tf_shots = tf.constant(shots, dtype=tf.float32)
        # logger.debug(f'batch success probability BEFORE shots: {batch_success_probabilities}')
        for input_codeword_success_probabilities in batch_success_probabilities:
            for codeword_success_probability in input_codeword_success_probabilities:
                codeword_success_probability.success_probability = tf.divide(
                    codeword_success_probability.counts, tf_shots)

        return batch_success_probabilities

    def _assign_counts_to_each_actual_codeword_result(
            self,
            result_codewords: List[CodeWord],
            batch_success_probabilities: List[List[CodeWordSuccessProbability]],
            all_counts: List[tf.Variable]) -> None:
        if len(result_codewords) != len(batch_success_probabilities):
            raise ValueError(
                f'result_codewords size: {len(result_codewords)} and '
                f'batch_success_probabilities length {len(batch_success_probabilities)} differs!')

        # logger.debug(f'Result output codewords: {result_codewords}')
        index_count = 0
        for result_codeword, input_codeword_success_probabilities in zip(result_codewords, batch_success_probabilities):
            found = False
            for codeword_success_probability in input_codeword_success_probabilities:
                if not found and codeword_success_probability.output_codeword == result_codeword:
                    found = True
                    all_counts[index_count].assign_add(1.0)
                    codeword_success_probability.counts = all_counts[index_count]
                index_count += 1

    def _init_one_input_codeword_success_probabilities(self,
                                                       input_codeword: CodeWord,
                                                       output_batch: Batch) -> List[CodeWordSuccessProbability]:
        return [CodeWordSuccessProbability(input_codeword=input_codeword,
                                           guessed_codeword=input_codeword,
                                           output_codeword=output_codeword,
                                           success_probability=tf.Variable(0.0),
                                           counts=tf.Variable(0.))
                for output_codeword in output_batch.codewords]

    def _init_batch_success_probabilities(self,
                                          input_batch: Batch,
                                          output_batch: Batch,
                                          all_counts: List[tf.Variable]) -> List[List[CodeWordSuccessProbability]]:

        for counts in all_counts:
            counts.assign(0.)

        return [self._init_one_input_codeword_success_probabilities(input_codeword=input_codeword,
                                                                    output_batch=output_batch)
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
