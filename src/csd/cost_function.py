# cost_function.py
from abc import ABC
from typing import List, Union, cast

from tensorflow.python.framework.ops import EagerTensor

from csd.batch import Batch
from csd.typings.typing import Backends, EngineRunOptions
from csd.typings.cost_function import CostFunctionOptions


class CostFunction(ABC):
    """Class to encapsulate cost function logic with the probability computation
    """

    def __init__(self, batch: Batch, params: List[float], options: CostFunctionOptions) -> None:
        self._batch = batch
        self._params = params
        self._options = options

    def _run_and_compute_batch_probabilities(self) -> Union[List[float], EagerTensor]:
        if self._options.backend_name == Backends.TENSORFLOW.value:
            return self._options.engine.run_circuit_checking_measuring_type(
                circuit=self._options.circuit,
                options=EngineRunOptions(
                    params=self._params,
                    batch_or_codeword=self._batch,
                    shots=self._options.shots,
                    measuring_type=self._options.measuring_type))

        return [cast(float, self._options.engine.run_circuit_checking_measuring_type(
            circuit=self._options.circuit,
            options=EngineRunOptions(
                params=self._params,
                batch_or_codeword=codeword,
                shots=self._options.shots,
                measuring_type=self._options.measuring_type)))
                for codeword in self._batch.codewords]

    def run_and_compute_batch_error_probabilities(self) -> float:
        return 1 - sum(self._run_and_compute_batch_probabilities()) / self._batch.size
