from abc import ABC
from typing import Callable, List, Union, Optional
from csd.typings import Backends
from .optimizers.tf import TFOptimizer
from .optimizers.scipy import ScipyOptimizer


class Optimize(ABC):

    def __init__(self, backend: Optional[Union[Backends, None]] = None, nparams: int = 1):
        self._backends: Union[Backends, None] = backend
        self._optimizer: Union[TFOptimizer, ScipyOptimizer, None] = None

        if self._backends is Backends.TENSORFLOW:
            self._optimizer = TFOptimizer(nparams=nparams)
        if self._backends is Backends.FOCK or self._backends is Backends.GAUSSIAN:
            self._optimizer = ScipyOptimizer()

    def optimize(self, cost_function: Callable) -> List[float]:
        if self._optimizer is None:
            raise ValueError("optimizer not initilized")
        return self._optimizer.optimize(cost_function=cost_function)
