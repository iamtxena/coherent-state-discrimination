from abc import ABC
from typing import Callable, Optional
from optimparallel import minimize_parallel
from scipy.optimize import OptimizeResult
# from multiprocessing import cpu_count
from csd.config import logger
from csd.typings.typing import OptimizationResult


class ParallelOptimizer(ABC):

    def __init__(self, nparams: int = 1):
        self._params = [0.0 for _ in range(nparams)]

    def optimize(self,
                 cost_function: Callable,
                 current_alpha: Optional[float] = 0.0) -> OptimizationResult:
        if self._params is None:
            raise ValueError("params not initialized")
        # max_workers = cpu_count()
        # logger.debug(f"Launching minimize_parallel optimization with {max_workers} workers")
        logger.debug(f"Launching minimize_parallel optimization for alpha={current_alpha}")
        # result: OptimizeResult = minimize_parallel(fun=cost_function,
        #                                            x0=self._params,
        #                                            parallel={'max_workers': max_workers})
        result: OptimizeResult = minimize_parallel(fun=cost_function,
                                                   x0=self._params,
                                                   options={
                                                       'disp': True,
                                                       'gtol': 1e-3,
                                                       'eps': 1e-2,
                                                   })
        logger.debug(f"Optimization result for alpha={current_alpha} :\n{result}")

        return OptimizationResult(optimized_parameters=result.x,
                                  error_probability=result.fun)
