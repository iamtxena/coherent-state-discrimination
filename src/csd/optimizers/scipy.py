from abc import ABC
from typing import Callable, List
from scipy.optimize import minimize


class ScipyOptimizer(ABC):

    def __init__(self, nparams: int = 1):
        self._method = 'BFGS'
        self._params = [0.0 for _ in range(nparams)]

    def reset(self):
        raise NotImplementedError()

    def optimize(self, cost_function: Callable) -> List[float]:
        return minimize(cost_function,
                        self._params,
                        method=self._method,
                        tol=1e-6).x
