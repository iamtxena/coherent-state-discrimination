from abc import ABC
from typing import Callable, List
from scipy.optimize import minimize


class ScipyOptimizer(ABC):

    def __init__(self):
        self._method = 'BFGS'

    def reset(self):
        raise NotImplementedError()

    def optimize(self, cost_function: Callable) -> List[float]:
        return minimize(cost_function,
                        [0.0],
                        method=self._method,
                        tol=1e-6).x
