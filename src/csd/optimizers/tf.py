from abc import ABC
import time
from typing import Callable, List, Optional, Tuple
import tensorflow as tf
import numpy as np
import os

from tensorflow.python.framework.ops import EagerTensor
from tensorflow.python.ops.variables import Variable

from csd.typings.typing import OptimizationResult
from csd.util import set_friendly_time
from csd.config import logger


class TFOptimizer(ABC):

    def __init__(self, nparams: int = 1, learning_steps: int = 300):
        os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
        self._learning_steps = learning_steps
        self._current_alpha = 0.0
        self._number_parameters = nparams

    def optimize(self, cost_function: Callable,
                 current_alpha: Optional[float] = 0.0) -> OptimizationResult:

        self._current_alpha = current_alpha if current_alpha is not None else 0.0
        init_time = time.time()
        loss = tf.Variable(0.0)
        params = [tf.Variable(0.1) for _ in range(self._number_parameters)]

        for step in range(self._learning_steps):
            loss, params = self._one_step_optimization(cost_function=cost_function,
                                                       parameters=params,
                                                       init_time=init_time,
                                                       step=step,
                                                       total_steps=self._learning_steps)

        return OptimizationResult(optimized_parameters=[param.numpy() for param in params],
                                  error_probability=loss.numpy())

    def _one_step_optimization(self,
                               cost_function: Callable,
                               parameters: List[Variable],
                               init_time: float,
                               step: int,
                               total_steps: int) -> Tuple[EagerTensor, List[Variable]]:
        one_loop_time = time.time()

        loss, parameters = self._tf_optimize(cost_function=cost_function,
                                             parameters=parameters)

        self._print_time_when_necessary(learning_steps=total_steps,
                                        init_time=init_time,
                                        step=step,
                                        one_loop_time=one_loop_time,
                                        optimized_parameters=parameters)
        return loss, parameters

    def _tf_optimize(self,
                     cost_function: Callable,
                     parameters: List[Variable]) -> Tuple[EagerTensor, List[Variable]]:

        opt = tf.keras.optimizers.Adam(learning_rate=0.01)
        with tf.GradientTape() as tape:
            loss = cost_function(parameters)

        gradients = tape.gradient(loss, parameters)
        opt.apply_gradients(zip(gradients, parameters))
        return loss, parameters

    def _print_time_when_necessary(self,
                                   learning_steps: int,
                                   init_time: float,
                                   step: int,
                                   one_loop_time: float,
                                   optimized_parameters: List[float]) -> None:
        # self._print_one_loop_time(step=step, total_steps=learning_steps, one_loop_time=one_loop_time)
        reset = self._print_optimized_parameters_for_tf_backend_only(step, optimized_parameters)
        if reset:
            self._print_one_loop_time(step=step, total_steps=learning_steps, one_loop_time=init_time)
        init_time = time.time() if reset else init_time

    def _print_optimized_parameters_for_tf_backend_only(self,
                                                        step: int,
                                                        optimized_parameters: List[EagerTensor]) -> bool:
        print_every_n_times = 50
        if (step + 1) % print_every_n_times == 0:
            logger.debug("Learned parameters value at step {}: {}".format(
                step + 1, [param.numpy() for param in optimized_parameters]))
            return True
        return False

    def _print_one_loop_time(self, step: int, total_steps: int, one_loop_time: float) -> None:
        now = time.time()
        logger.debug(f'Optimized for alpha: {np.round(self._current_alpha, 2)} took:'
                     f"{set_friendly_time(time_interval=now-one_loop_time)}"
                     f" steps: [{step+1}/{total_steps}]")
