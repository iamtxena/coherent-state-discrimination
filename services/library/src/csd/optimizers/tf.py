from abc import ABC
import time
from typing import Callable, List, Optional, Tuple, Union
from csd.codeword import CodeWord
import tensorflow as tf
import numpy as np
import os
from datetime import datetime

from tensorflow.python.framework.ops import EagerTensor
from tensorflow.python.ops.variables import Variable
from tensorflow.keras import optimizers as tfOptimizers
from tensorflow.keras import metrics as tfMetrics

from csd.typings.typing import LearningRate, LearningSteps, OptimizationResult
from csd.utils.util import (CodeBookLogInformation, CodebookLearningStepsLogInformation,
                            print_codebook_learning_steps_log)
from csd.config import logger


class TFOptimizer(ABC):

    def __init__(self,
                 nparams: int = 1,
                 learning_steps: LearningSteps = LearningSteps(default=300, high=500, extreme=2000),
                 learning_rate: LearningRate = LearningRate(default=0.01, high=0.001, extreme=0.001),
                 modes: int = 1,
                 params_name: List[str] = []):
        os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
        self._learning_steps = learning_steps
        self._learning_rate = learning_rate
        self._current_alpha = 0.0
        self._number_parameters = nparams
        self._number_modes = modes
        if len(params_name) - modes != nparams:
            raise ValueError(f"params names length: {params_name.__len__()} does not match nparams: {nparams}")
        self._params_name = params_name[modes:]

    def optimize(self,
                 cost_function: Callable,
                 current_alpha: Optional[float] = 0.0,
                 codebook_log_info: Union[CodeBookLogInformation, None] = None) -> OptimizationResult:

        self._current_alpha = current_alpha if current_alpha is not None else 0.0

        self._opt = tfOptimizers.Adam(learning_rate=self._learning_rate.default)
        init_time = time.time()
        loss = tf.Variable(0.0, name='loss')
        parameters = [tf.Variable(0.1, name=param_name) for param_name in self._params_name]

        self._prepare_tf_board(self._current_alpha)
        current_learning_steps = self._set_learning_values_by_alpha(self._current_alpha)
        logger.info(f"number of parameters: {len(parameters)}")
        for step in range(current_learning_steps):
            step_init_time = time.time()
            loss, parameters, measurements = self._tf_optimize(cost_function=cost_function,
                                                               parameters=parameters)

            reset = self._print_time_when_necessary(learning_steps=current_learning_steps,
                                                    init_time=init_time,
                                                    step=step,
                                                    step_init_time=step_init_time,
                                                    optimized_parameters=parameters,
                                                    codebook_log_info=codebook_log_info)
            init_time = time.time() if reset else init_time
            self._update_tf_board_metrics(step)

        return OptimizationResult(optimized_parameters=[param.numpy() for param in parameters],
                                  error_probability=loss.numpy(),
                                  measurements=measurements)

    def _update_tf_board_metrics(self, step: int) -> None:
        with self._train_summary_writer.as_default():
            tf.summary.scalar('loss', self._train_loss.result(), step=step)
            tf.summary.scalar('avg.succ', self._train_succ.result(), step=step)
            for train_param, param_name in zip(self._train_params, self._params_name):
                tf.summary.scalar(param_name, train_param.result(), step=step)

            # Reset metrics every step
        self._train_loss.reset_states()
        for train_param in self._train_params:
            train_param.reset_states()

    def _set_learning_values_by_alpha(self, alpha: float) -> int:
        current_learning_steps = self._learning_steps.default

        if alpha > 1.2:
            current_learning_steps = self._learning_steps.high

        if alpha <= 0.1 and self._number_modes >= 3:
            self._opt = tfOptimizers.Adam(learning_rate=self._learning_rate.high)
            current_learning_steps = self._learning_steps.high
        return current_learning_steps

    def _prepare_tf_board(self, current_alpha: float) -> None:
        current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
        train_log_dir = (f'logs/gradient_tape/modes/{self._number_modes}/{current_time}'
                         f'_alpha_{np.round(current_alpha, 2)}')
        self._train_summary_writer = tf.summary.create_file_writer(train_log_dir)

        # Define our metrics
        self._train_loss = tfMetrics.Mean('train_loss')
        self._train_succ = tfMetrics.Mean('train_succ')
        self._train_params = [tfMetrics.Mean(f'train_param_{param_name}',
                                             dtype=tf.float32)
                              for param_name in self._params_name]

    def _tf_optimize(self,
                     cost_function: Callable,
                     parameters: List[Variable]) -> Tuple[EagerTensor,
                                                          List[Variable],
                                                          List[CodeWord]]:

        with tf.GradientTape() as tape:
            loss, measurements = cost_function(params=parameters)

        gradients = tape.gradient(loss, parameters)
        self._opt.apply_gradients(zip(gradients, parameters))

        self._train_loss(loss)
        self._train_succ(1 - loss)
        # logger.debug(f'parameters: {parameters}')
        for train_param, parameter in zip(self._train_params, parameters):
            train_param(parameter)

        return loss, parameters, measurements

    def _print_time_when_necessary(self,
                                   learning_steps: int,
                                   init_time: float,
                                   step: int,
                                   step_init_time: float,
                                   optimized_parameters: List[float],
                                   codebook_log_info: Union[CodeBookLogInformation, None] = None) -> bool:

        reset = self._print_optimized_parameters_for_tf_backend_only(step, optimized_parameters)
        if reset and codebook_log_info is not None:
            learning_log_info = CodebookLearningStepsLogInformation(
                total_steps=learning_steps,
                step_iteration=step,
                learning_init_time=init_time,
                step_current_iteration_init_time=step_init_time
            )
            print_codebook_learning_steps_log(codebook_log_info=codebook_log_info,
                                              learning_log_info=learning_log_info)
        return reset

    def _print_optimized_parameters_for_tf_backend_only(self,
                                                        step: int,
                                                        optimized_parameters: List[EagerTensor]) -> bool:
        print_every_n_times = 50
        if (step + 1) % print_every_n_times == 0:
            logger.debug("Learned parameters value at step {}: {}".format(
                step + 1, [param.numpy() for param in optimized_parameters]))
            return True
        return False
