from abc import ABC
from typing import Callable, Optional
import tensorflow as tf
import os
import concurrent.futures
import itertools
import functools

from csd.typings.typing import OptimizationResult
from csd.config import logger


class ParallelTFOptimizer(ABC):

    def __init__(self, nparams: int = 1):
        os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
        self._opt = tf.keras.optimizers.Adam(learning_rate=0.01)
        self._params = [tf.Variable(0.1) for _ in range(nparams)]

    def optimize(self, cost_function: Callable,
                 current_alpha: Optional[float] = 0.0) -> OptimizationResult:
        if self._opt is None:
            raise ValueError("opt not initialized")
        if self._params is None:
            raise ValueError("params not initialized")

        max_workers = 4
        executor = concurrent.futures.ProcessPoolExecutor(max_workers=max_workers)
        ftmp = functools.partial(cost_function, self._params)
        result = executor.map(cost_function, itertools.repeat(self._params))
        # self._opt.apply_gradients(zip(gradients, self._params))
        result = list(result)
        logger.debug(result)
        return OptimizationResult(optimized_parameters=[param.numpy() for param in self._params],
                                  error_probability=loss.numpy())
