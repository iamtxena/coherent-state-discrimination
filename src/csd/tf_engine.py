# engine.py
from csd.batch import Batch
from csd.codeword import CodeWord
from .engine import Engine
from strawberryfields.api import Result
from strawberryfields.backends.tfbackend.states import FockStateTF
import tensorflow as tf
from tensorflow.python.framework.ops import EagerTensor
from csd.typings.typing import (BatchSuccessProbability, CodeWordSuccessProbability, TFEngineRunOptions)
from csd.circuit import Circuit
from typing import List
# from csd.config import logger


class TFEngine(Engine):
    """ Tensor Flow Engine class

    """

    def run_tf_circuit_checking_measuring_type(
            self,
            circuit: Circuit,
            options: TFEngineRunOptions) -> List[CodeWordSuccessProbability]:

        # if options['measuring_type'] is MeasuringTypes.SAMPLING:
        #     codewords_success_probabilities = self._run_circuit_sampling(circuit=circuit, options=options)
        # else:
        batch_sucess_probabilities = self._run_tf_circuit_probabilities(circuit=circuit, options=options)

        return [self._max_probability_codeword(codewords_success_probabilities=codewords_success_probabilities)
                for codewords_success_probabilities in batch_sucess_probabilities]

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

    # def _run_tf_circuit_sampling(self,
    #                              circuit: Circuit,
    #                              options: EngineRunOptions) -> List[CodeWordSuccessProbability]:
    #     """Run a circuit experiment doing MeasureFock and performing samplint with nshots
    #     """
    #     # if self._engine.backend_name == Backends.GAUSSIAN.value:
    #     #     return sum([1 for read_value in self._run_circuit(circuit=circuit, options=options).samples
    #     #                 if read_value[0] == 0]) / options['shots']
    #     shots = options['shots']
    #     options['shots'] = 1
    #     zero_prob = sum([1 for read_value in [self._run_circuit(circuit=circuit, options=options).samples[0][0]
    #                                           for _ in range(shots)] if read_value == 0]) / shots
    #     codewords = self._generate_all_codewords(codeword=options['codeword'])
    #     return [CodeWordSuccessProbability(codeword=codewords[0], success_probability=zero_prob),
    #             CodeWordSuccessProbability(codeword=codewords[1], success_probability=1 - zero_prob)]

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
            guessed_codeword=CodeWord(size=input_codeword.size,
                                      alpha_value=input_codeword.alpha),
            output_codeword=batch_sucess_probabilities.codeword_indices.codeword,
            success_probability=self._compute_one_codeword_success_probability(
                index_codeword=index_codeword,
                success_probabilities_indices=batch_sucess_probabilities.success_probability))
                for batch_sucess_probabilities in success_probabilities_batches]

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
