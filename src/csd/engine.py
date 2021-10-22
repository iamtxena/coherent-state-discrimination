# engine.py
from abc import ABC
import strawberryfields as sf
from strawberryfields.api import Result
from tensorflow.python.framework.ops import EagerTensor
from typeguard import typechecked
from csd.batch import Batch
from csd.codeword import CodeWord
from csd.typings.typing import Backends, BackendOptions, EngineRunOptions, MeasuringTypes
from typing import List, Optional, Union
from .circuit import Circuit
from nptyping import NDArray
import numpy as np
import itertools


class Engine(ABC):
    """ Engine class

    """

    @typechecked
    def __init__(self, backend: Backends, options: Optional[BackendOptions] = None) -> None:
        self._backend = backend
        self._cutoff_dim = options['cutoff_dim']
        self._engine = sf.Engine(backend=self._backend.value,
                                 backend_options=options)

    @property
    def backend_name(self) -> str:
        return self._engine.backend_name

    def run_circuit_checking_measuring_type(
        self,
        circuit: Circuit,
        options: EngineRunOptions) -> Union[
            Union[float, EagerTensor, np.float],
            Union[list[float], EagerTensor, NDArray[np.float]]]:
        if options['measuring_type'] is MeasuringTypes.SAMPLING:
            return self._run_circuit_sampling(circuit=circuit, options=options)
        return self._run_circuit_probabilities(circuit=circuit, options=options)

    def _run_circuit_sampling(self,
                              circuit: Circuit,
                              options: EngineRunOptions) -> Union[
            Union[float, EagerTensor, np.float],
            Union[list[float], EagerTensor, NDArray[np.float]]]:
        """Run a circuit experiment doing MeasureFock and performing samplint with nshots

        Returns:
            float: probability of getting |0> state
        """
        if self._engine.backend_name == Backends.GAUSSIAN.value:
            return sum([1 for read_value in self._run_circuit(circuit=circuit, options=options).samples
                        if read_value[0] == 0]) / options['shots']

        return sum([1 for read_value in [self._run_circuit(circuit=circuit, options=options).samples[0][0]
                                         for _ in range(options['shots'])] if read_value == 0]) / options['shots']

    def _run_circuit_probabilities(self,
                                   circuit: Circuit,
                                   options: EngineRunOptions) -> Union[
                                       Union[float, EagerTensor, np.float],
                                       Union[list[float], EagerTensor, NDArray[np.float]]]:
        """Run a circuit experiment computing the fock probability

        Returns:
            float: probability of getting |0> state
        """
        options['shots'] = 0
        result = self._run_circuit(circuit=circuit, options=options)
        return self._compute_fock_probabilities_for_batch_or_codeword(result=result,
                                                                      batch_or_codeword=options['batch_or_codeword'])

    def _compute_fock_probabilities_for_codeword(self, result: Result, codeword: CodeWord, cutoff_dim: int) -> float:
        fock_prob_indices_one_word = self._convert_word_to_fock_prob_indices(codeword=codeword, cutoff_dim=cutoff_dim)
        return self._compute_fock_prob_one_word(result=result, fock_prob_indices_one_word=fock_prob_indices_one_word)

    def _compute_fock_probabilities_for_batch_or_codeword(
            self,
            result: Result,
            batch_or_codeword: Union[Batch, CodeWord]) -> Union[
            Union[float, EagerTensor, np.float],
            Union[list[float], EagerTensor, NDArray[np.float]]]:
        # !!! TODO: it will be different on TF, because Result already is a List
        if isinstance(batch_or_codeword, Batch):
            raise ValueError(f'Not implemented for Batch. printing the results: {result}')
        return self._compute_fock_probabilities_for_codeword(result=result,
                                                             codeword=batch_or_codeword,
                                                             cutoff_dim=self._cutoff_dim)

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
        all_values = []
        all_values.append(options['batch_or_codeword'].to_list())
        [all_values.append(param) for param in options['params']]

        return {name: value for (name, value) in zip(circuit.parameters.keys(), all_values)}

    def _compute_fock_prob_one_word(result: Result, fock_prob_indices_one_word: List[List[int]]) -> float:
        return sum([result.state.fock_prob(fock_prob_indices) for fock_prob_indices in fock_prob_indices_one_word])

    def _convert_word_to_fock_prob_indices(self, codeword: CodeWord, cutoff_dim: int) -> List[List[int]]:
        number_modes = len(codeword.size)

        if codeword.number_alphas == 0:
            return [[0] * number_modes]

        prob_indices = []
        dimensions_more_than_0_photons = [i for i in range(cutoff_dim) if i > 0]
        zero_list = codeword.zero_list.copy()
        minus_indices = codeword.minus_indices
        minus_groups = [p for p in itertools.product(dimensions_more_than_0_photons, repeat=codeword.number_alphas)]

        for minus_group in minus_groups:
            for dimension, index in zip(minus_group, minus_indices):
                zero_list[index] = dimension
            prob_indices.append(zero_list.copy())
        return prob_indices
