from abc import ABC
import time
from typing import Callable, List, Optional
import tensorflow as tf
import numpy as np
import os

from csd.typings.typing import OptimizationResult
from csd.util import set_friendly_time
from csd.config import logger


class TFOptimizer(ABC):

    def __init__(self, nparams: int = 1):
        os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
        self._opt = tf.keras.optimizers.Adam(learning_rate=0.01)
        self._params = [tf.Variable(0.1) for _ in range(nparams)]
        self._learning_steps = 300
        self._current_alpha = 0.0

    def optimize(self, cost_function: Callable,
                 current_alpha: Optional[float] = 0.0) -> OptimizationResult:
        if self._opt is None:
            raise ValueError("opt not initialized")
        if self._params is None:
            raise ValueError("params not initialized")
        self._current_alpha = current_alpha if current_alpha is not None else 0.0

        init_time = time.time()
        loss = tf.Variable(0.0)

        for step in range(self._learning_steps):
            one_loop_time = time.time()

            loss = self._tf_optimize(cost_function)

            self._print_time_when_necessary(self._learning_steps, init_time, step, one_loop_time, self._params)

        return OptimizationResult(optimized_parameters=[param.numpy() for param in self._params],
                                  error_probability=loss.numpy())

    def _tf_optimize(self, cost_function):
        with tf.GradientTape() as tape:
            loss = cost_function(self._params)

        gradients = tape.gradient(loss, self._params)
        self._opt.apply_gradients(zip(gradients, self._params))
        return loss

    def _print_time_when_necessary(self,
                                   learning_steps: int,
                                   init_time: float,
                                   step: int,
                                   one_loop_time: float,
                                   optimized_parameters: List[float]) -> None:
        self._print_one_loop_time(step=step, total_steps=learning_steps, one_loop_time=one_loop_time)
        reset = self._print_optimized_parameters_for_tf_backend_only(step, optimized_parameters)
        init_time = time.time() if reset else init_time

    def _print_optimized_parameters_for_tf_backend_only(self,
                                                        step: int,
                                                        optimized_parameters: List[float]) -> bool:
        print_every_n_times = 50
        if (step + 1) % print_every_n_times == 0:
            logger.debug("Learned parameters value at step {}: {}".format(
                step + 1, optimized_parameters))
            return True
        return False

    def _print_one_loop_time(self, step: int, total_steps: int, one_loop_time: float) -> None:
        now = time.time()
        logger.debug(f'Optimized for alpha: {np.round(self._current_alpha, 2)} took:'
                     f"{set_friendly_time(time_interval=now-one_loop_time)}"
                     f" steps: [{step+1}/{total_steps}]")
