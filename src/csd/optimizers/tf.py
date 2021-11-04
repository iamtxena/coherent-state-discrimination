from abc import ABC
import time
from typing import Callable, List, Optional, Tuple
import tensorflow as tf
import numpy as np
import os
from datetime import datetime

from tensorflow.python.framework.ops import EagerTensor
from tensorflow.python.ops.variables import Variable

from csd.typings.typing import OptimizationResult
from csd.util import set_friendly_time
from csd.config import logger


class TFOptimizer(ABC):

    def __init__(self,
                 nparams: int = 1,
                 learning_steps: int = 300,
                 learning_rate: float = 0.1,
                 modes: int = 1):
        os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
        self._learning_steps = learning_steps
        self._learning_rate = learning_rate
        self._current_alpha = 0.0
        self._number_parameters = nparams
        self._number_modes = modes

    def optimize(self, cost_function: Callable,
                 current_alpha: Optional[float] = 0.0) -> OptimizationResult:

        self._current_alpha = current_alpha if current_alpha is not None else 0.0
        if self._current_alpha < 0.2 and self._number_modes > 1:
            self._learning_rate /= 100
            self._learning_steps *= 10

        self._opt = tf.keras.optimizers.Adam(learning_rate=self._learning_rate)
        init_time = time.time()
        loss = tf.Variable(0.0)
        params = [tf.Variable(0.1) for _ in range(self._number_parameters)]

        current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
        train_log_dir = (f'logs/gradient_tape/modes/{self._number_modes}/{current_time}'
                         f'_alpha_{np.round(current_alpha, 2)}')
        self._train_summary_writer = tf.summary.create_file_writer(train_log_dir)

        # Define our metrics
        self._train_loss = tf.keras.metrics.Mean('train_loss')
        self._train_params = [tf.keras.metrics.Mean(f'train_param_{index}',
                                                    dtype=tf.float32)
                              for index, _ in enumerate(params)]

        for step in range(self._learning_steps):
            loss, params = self._one_step_optimization(cost_function=cost_function,
                                                       parameters=params,
                                                       init_time=init_time,
                                                       step=step,
                                                       total_steps=self._learning_steps,
                                                       learning_rate=self._learning_rate)
            with self._train_summary_writer.as_default():
                tf.summary.scalar('loss', self._train_loss.result(), step=step)
                for index, train_param in enumerate(self._train_params):
                    tf.summary.scalar(f'param_{index}',
                                      train_param.result(), step=step)

            # template = 'Step {}, Loss: {}'
            # logger.info(template.format(step + 1, self._train_loss.result()))

            # Reset metrics every epoch
            self._train_loss.reset_states()
            for train_param in self._train_params:
                train_param.reset_states()

        return OptimizationResult(optimized_parameters=[param.numpy() for param in params],
                                  error_probability=loss.numpy())

    def _one_step_optimization(self,
                               cost_function: Callable,
                               parameters: List[Variable],
                               init_time: float,
                               step: int,
                               total_steps: int,
                               learning_rate: float) -> Tuple[EagerTensor, List[Variable]]:
        one_loop_time = time.time()

        loss, parameters = self._tf_optimize(cost_function=cost_function,
                                             parameters=parameters,
                                             learning_rate=learning_rate)

        self._print_time_when_necessary(learning_steps=total_steps,
                                        init_time=init_time,
                                        step=step,
                                        one_loop_time=one_loop_time,
                                        optimized_parameters=parameters)
        return loss, parameters

    def _tf_optimize(self,
                     cost_function: Callable,
                     parameters: List[Variable],
                     learning_rate: float) -> Tuple[EagerTensor, List[Variable]]:

        with tf.GradientTape() as tape:
            loss = cost_function(parameters)

        gradients = tape.gradient(loss, parameters)
        self._opt.apply_gradients(zip(gradients, parameters))

        self._train_loss(loss)
        for train_param, parameter in zip(self._train_params, parameters):
            train_param(parameter)

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
