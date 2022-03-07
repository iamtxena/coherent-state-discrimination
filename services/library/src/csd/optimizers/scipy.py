from abc import ABC
from typing import Callable, Optional, Union
from csd.utils.util import CodeBookLogInformation
from scipy.optimize import minimize, OptimizeResult
from csd.config import logger
from csd.typings.typing import OptimizationResult


class ScipyOptimizer(ABC):

    def __init__(self, nparams: int = 1):
        self._method = 'BFGS'
        self._params = [0.1 for _ in range(nparams)]

    def reset(self):
        raise NotImplementedError()

    def optimize(self, cost_function: Callable,
                 current_alpha: Optional[float] = 0.0,
                 codebook_log_info: Union[CodeBookLogInformation, None] = None) -> OptimizationResult:
        logger.debug(f"Launching basic scipy optimization for alpha={current_alpha}")
        result: OptimizeResult = minimize(cost_function,
                                          self._params,
                                          options={
                                              'disp': True,
                                              'gtol': 1e-3,
                                              'eps': 1e-2,
                                          },
                                          method=self._method)
        logger.debug(f"Optimization result for alpha={current_alpha} :\n{result}")

        return OptimizationResult(optimized_parameters=result.x,
                                  error_probability=result.fun)
