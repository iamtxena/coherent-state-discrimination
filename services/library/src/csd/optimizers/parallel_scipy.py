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
                 current_alpha: Optional[float] = 1.0,
                 codebooks_info: dict = {}) -> OptimizationResult:
        if self._params is None:
            raise ValueError("params not initialized")
        # max_workers = cpu_count()
        max_workers = 4
        logger.debug(f"Launching minimize_parallel optimization with {max_workers} workers")
        # logger.debug(f"Launching minimize_parallel optimization for alpha={current_alpha}")
        if current_alpha is None:
            current_alpha = 1.0
        bounds = [(-2 * current_alpha, 2 * current_alpha) for _ in range(len(self._params))]
        logger.debug(f"bounds: {bounds}")
        result: OptimizeResult = minimize_parallel(fun=cost_function,
                                                   x0=self._params,
                                                   bounds=bounds,
                                                   parallel={'max_workers': max_workers},
                                                   options={
                                                       'disp': True,
                                                       'eps': 0.1,
                                                   })
        # result: OptimizeResult = minimize_parallel(fun=cost_function,
        #                                            x0=self._params,
        #                                            options={
        #                                                'disp': True,
        #                                                'eps': 0.01,
        #                                            })
        logger.debug(f"Optimization result for alpha={current_alpha} :\n{result}")

        return OptimizationResult(optimized_parameters=result.x,
                                  error_probability=result.fun)
