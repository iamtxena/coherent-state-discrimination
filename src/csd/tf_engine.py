# engine.py
from typing import List, Union

import tensorflow as tf
from strawberryfields.result import Result
from tensorflow.python.framework.ops import EagerTensor

from csd.batch import Batch
from csd.circuit import Circuit
from csd.codeword import CodeWord
from csd.config import logger
from csd.typings.multi_layer_circuit import MultiLayerCircuit
from csd.typings.typing import CodeWordSuccessProbability, MeasuringTypes, TFEngineRunOptions
from csd.utils.probability import compute_maximum_likelihood, compute_multilayer_maximum_likelihood

from .engine import Engine


class TFEngine(Engine):
    """EagerTensor Flow Engine class"""

    def _run_internal_circuit_checking_measuring_type(
        self, circuit: Circuit, options: TFEngineRunOptions
    ) -> List[List[CodeWordSuccessProbability]]:
        """run internal circuit checking measuring type

        Args:
            circuit (Circuit): _description_
            options (TFEngineRunOptions): _description_

        Returns:
            _type_: _description_
        """
        return (
            self._run_tf_circuit_probabilities(circuit=circuit, options=options)
            if options["measuring_type"] is MeasuringTypes.PROBABILITIES
            else self._run_tf_sampling(circuit=circuit, options=options)
        )

    def _first_mode_result_is_zero(self, batch_success_probabilities: List[List[CodeWordSuccessProbability]]) -> bool:
        """returns if the first mode result is zero from the given probabilities"""
        # TODO: check the first mode result if it is zero or not
        logger.info(batch_success_probabilities)
        return True

    def run_tf_multi_layer_circuit_checking_measuring_type(
        self, multi_layer_circuit: MultiLayerCircuit, options: TFEngineRunOptions
    ) -> List[CodeWordSuccessProbability]:
        """Run Tensorflow circuit checking for measuring type

        Args:
            circuit (MultiLayerCircuit): _description_
            options (TFEngineRunOptions): _description_

        Returns:
            List[CodeWordSuccessProbability]: _description_
        """
        if multi_layer_circuit.number_layers == 1:
            return self.run_tf_circuit_checking_measuring_type(
                circuit=multi_layer_circuit.first_layer.circuit, options=options
            )

        batch_success_probabilities_first_layer = self._run_internal_circuit_checking_measuring_type(
            circuit=multi_layer_circuit.first_layer.circuit, options=options
        )
        if multi_layer_circuit.second_layer is None:
            raise ValueError("Second layer MUST be defined when number_layers == 2")

        # take all codeword_success_probabilities that has a 0 in the first mode output -> and adapt the batch from the options
        # do the same for the ones that has a 1 (the rest)
        # run both sets
        # join both set results and check that all possibilites are satisfied -> put them in the same order as the first list

        second_layer_circuit_to_run = (
            multi_layer_circuit.second_layer.circuit_zero_on_first_layer_mode
            if self._first_mode_result_is_zero(batch_success_probabilities=batch_success_probabilities_first_layer)
            else multi_layer_circuit.second_layer.circuit_one_on_first_layer_mode
        )
        batch_success_probabilities_second_layer = self._run_internal_circuit_checking_measuring_type(
            circuit=second_layer_circuit_to_run, options=options
        )
        return compute_multilayer_maximum_likelihood(
            batch_success_probabilities_first_layer=batch_success_probabilities_first_layer,
            batch_success_probabilities_second_layer=batch_success_probabilities_second_layer,
            output_batch=options["output_batch"],
        )

    def run_tf_circuit_checking_measuring_type(
        self, circuit: Circuit, options: TFEngineRunOptions
    ) -> List[CodeWordSuccessProbability]:
        """Run Tensorflow circuit checking for measuring type

        Args:
            circuit (Circuit): _description_
            options (TFEngineRunOptions): _description_

        Returns:
            List[CodeWordSuccessProbability]: _description_
        """

        batch_success_probabilities = self._run_internal_circuit_checking_measuring_type(
            circuit=circuit, options=options
        )

        return compute_maximum_likelihood(
            batch_success_probabilities=batch_success_probabilities, output_batch=options["output_batch"]
        )

    def _run_tf_circuit_probabilities(
        self, circuit: Circuit, options: TFEngineRunOptions
    ) -> List[List[CodeWordSuccessProbability]]:
        """Run a circuit experiment computing the fock probability"""

        options["shots"] = 0
        result = self._run_tf_circuit(circuit=circuit, options=options)
        self._all_fock_probs = (
            [result.state.all_fock_probs()]
            if self.only_one_codeword(input_batch=options["input_batch"])
            else result.state.all_fock_probs()
        )
        # for i in range(circuit.number_input_modes):
        #     e, v = result.state.quad_expectation(mode=i)
        #     print(f'Mode: {i}, Expectation: {e.numpy()}, Variance: {v.numpy()}')

        # logger.debug(f'all_fock_probs: {self._all_fock_probs}')
        # logger.debug(f'len all_fock_probs: {len(self._all_fock_probs)}')

        return self._compute_tf_fock_probabilities_for_all_codewords(
            input_batch=options["input_batch"], output_batch=options["output_batch"]
        )

    def _compute_tf_fock_probabilities_for_all_codewords(
        self, input_batch: Batch, output_batch: Batch
    ) -> List[List[CodeWordSuccessProbability]]:
        return [
            self._compute_one_batch_codewords_success_probabilities(
                input_codeword=input_codeword, index_input_batch=index_input_batch, output_batch=output_batch
            )
            for index_input_batch, input_codeword in enumerate(input_batch.codewords)
        ]

    def _compute_one_batch_codewords_success_probabilities(
        self, input_codeword: CodeWord, index_input_batch: int, output_batch: Batch
    ) -> List[CodeWordSuccessProbability]:

        success_probabilities_all_outcomes = self._compute_success_probabilities_all_outcomes(
            index_input_batch=index_input_batch
        )
        if len(success_probabilities_all_outcomes) != output_batch.size:
            logger.debug(
                f"len success_probabilities_all_outcomes: {len(success_probabilities_all_outcomes)} and, "
                f"output_batch.size: {output_batch.size}"
            )
            raise ValueError("success probability outcomes and output batch sizes differs.")

        return [
            CodeWordSuccessProbability(
                input_codeword=input_codeword,
                guessed_codeword=CodeWord(size=input_codeword.size, alpha_value=input_codeword.alpha),
                output_codeword=output_codeword,
                success_probability=success_probabilities_one_outcome,
                counts=tf.Variable(0),
            )
            for success_probabilities_one_outcome, output_codeword in zip(
                success_probabilities_all_outcomes, output_batch.codewords
            )
        ]

    def _compute_success_probabilities_all_outcomes(self, index_input_batch: int) -> List[EagerTensor]:
        return [
            tf.reduce_sum(tf.math.multiply(measurement_matrix, self._all_fock_probs[index_input_batch]))
            for measurement_matrix in self._measurement_matrices
        ]

    #
    #   TESTING: TF
    #

    def run_tf_circuit_training(
        self, circuit: Circuit, options: TFEngineRunOptions
    ) -> List[CodeWordSuccessProbability]:
        """Run a circuit experiment doing MeasureFock and performing sampling with nshots"""
        batch_success_probabilities = self._run_tf_sampling(circuit, options)
        return compute_maximum_likelihood(
            batch_success_probabilities=batch_success_probabilities, output_batch=options["output_batch"]
        )

    def _run_tf_sampling(self, circuit: Circuit, options: TFEngineRunOptions) -> List[List[CodeWordSuccessProbability]]:
        shots = options["shots"]
        options["shots"] = 1
        alpha_value = options["input_batch"].one_codeword.alpha
        batch_success_probabilities = self._init_batch_success_probabilities(
            input_batch=options["input_batch"], output_batch=options["output_batch"]
        )
        for _ in range(shots):
            output_codewords = self._convert_batch_sampling_output_to_codeword_list(
                alpha_value=alpha_value, batch_samples=self._run_tf_circuit(circuit=circuit, options=options).samples
            )

            self._assign_counts_to_each_actual_codeword_result(
                result_codewords=output_codewords, batch_success_probabilities=batch_success_probabilities
            )

        batch_success_probabilities = self._compute_average_batch_success_probabilities(
            batch_success_probabilities=batch_success_probabilities, shots=shots
        )

        return batch_success_probabilities

    def _compute_average_batch_success_probabilities(
        self, batch_success_probabilities: List[List[CodeWordSuccessProbability]], shots: int
    ) -> List[List[CodeWordSuccessProbability]]:
        if shots <= 0:
            raise ValueError(f"shots MUST be greater than zero. Current value: {shots}")

        for input_codeword_success_probabilities in batch_success_probabilities:
            for codeword_success_probability in input_codeword_success_probabilities:
                codeword_success_probability.success_probability = codeword_success_probability.counts / shots

        return batch_success_probabilities

    def _assign_counts_to_each_actual_codeword_result(
        self, result_codewords: List[CodeWord], batch_success_probabilities: List[List[CodeWordSuccessProbability]]
    ) -> None:
        if len(result_codewords) != len(batch_success_probabilities):
            raise ValueError(
                f"result_codewords size: {len(result_codewords)} and "
                f"batch_success_probabilities length {len(batch_success_probabilities)} differs!"
            )

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

    def _convert_batch_sampling_output_to_codeword_list(
        self, alpha_value: float, batch_samples: List[EagerTensor]
    ) -> List[CodeWord]:
        return [
            self._convert_sampling_output_to_codeword(alpha_value=alpha_value, one_codeword_output=one_codeword_output)
            for one_codeword_output in batch_samples
        ]

    def _init_one_input_codeword_success_probabilities(
        self, input_codeword: CodeWord, output_batch: Batch
    ) -> List[CodeWordSuccessProbability]:
        return [
            CodeWordSuccessProbability(
                input_codeword=input_codeword,
                guessed_codeword=input_codeword,
                output_codeword=output_codeword,
                success_probability=tf.Variable(0.0),
                counts=tf.Variable(0),
            )
            for output_codeword in output_batch.codewords
        ]

    def _init_batch_success_probabilities(
        self, input_batch: Batch, output_batch: Batch
    ) -> List[List[CodeWordSuccessProbability]]:

        return [
            self._init_one_input_codeword_success_probabilities(
                input_codeword=input_codeword, output_batch=output_batch
            )
            for input_codeword in input_batch.codewords
        ]

    #
    #   TRAINING AND TESTING: TF
    #

    def _run_tf_circuit(self, circuit: Circuit, options: TFEngineRunOptions) -> Result:
        """Run an experiment using the engine with the passed options"""
        # reset the engine if it has already been executed
        if self._engine.run_progs:
            self._engine.reset()

        args = self._parse_tf_circuit_parameters(circuit=circuit, options=options)

        return self._engine.run(program=circuit.circuit, args=args, shots=options["shots"])

    def _parse_tf_circuit_parameters(self, circuit: Circuit, options: TFEngineRunOptions) -> dict:
        all_values: Union[None, List[float], List[List[float]]] = None
        if self.only_one_codeword(input_batch=options["input_batch"]):
            all_values = [
                one_letter for one_letter_list in options["input_batch"].letters for one_letter in one_letter_list
            ]
        if not self.only_one_codeword(input_batch=options["input_batch"]):
            all_values = options["input_batch"].letters

        if all_values is None:
            raise ValueError("all_values is None")
        for param in options["params"]:
            all_values.append(param)

        if len(circuit.parameters.keys()) != len(all_values):
            logger.error(f"all_values: {all_values}")
            logger.error(f"circuit.parameters.keys(): {circuit.parameters.keys()}")
            raise ValueError("length parameters.keys() differes from all_values")

        return dict(zip(circuit.parameters.keys(), all_values))

    def only_one_codeword(self, input_batch: Batch) -> bool:
        return len(input_batch.codewords) == 1
