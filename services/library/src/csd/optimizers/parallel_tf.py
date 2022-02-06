import time

from csd.util import CodeBookLogInformation

from .tf import TFOptimizer
from typing import Callable, Optional, Union
import tensorflow as tf
from multiprocessing import Pool

from csd.typings.typing import OptimizationResult
# from csd.config import logger


class ParallelTFOptimizer(TFOptimizer):

    # @staticmethod
    # def _eval_args(fun, cost_fun, params, init_time, step, total_steps):
    #     return fun(cost_fun, params, init_time, step, total_steps)

    # def optimize(self, cost_function: Callable,
    #              current_alpha: Optional[float] = 0.0) -> OptimizationResult:

    #     self._current_alpha = current_alpha if current_alpha is not None else 0.0
    #     # opt = tf.keras.optimizers.Adam(learning_rate=0.01)

    #     init_time = time.time()
    #     loss = tf.Variable(0.0)
    #     params = [tf.Variable(0.1) for _ in range(self._number_parameters)]
    #     learning_steps = range(self._learning_steps)

    #     ftmp = self._eval_args
    #     executor = concurrent.futures.ProcessPoolExecutor()
    #     result = executor.map(ftmp,
    #                           itertools.repeat(self._one_step_optimization),
    #                           itertools.repeat(cost_function),
    #                           itertools.repeat(params),
    #                           itertools.repeat(init_time),
    #                           learning_steps,
    #                           itertools.repeat(self._learning_steps))
    #     logger.debug(f"result: {list(result)}")
    #     ret = np.array(list(result))
    #     logger.debug(f"ret: {result}")
    #     loss = ret[0]
    #     params = ret[1]

    #     return OptimizationResult(optimized_parameters=[param.numpy() for param in params],
    #                               error_probability=loss.numpy())

    @staticmethod
    def uncurry_launch_execution(t):
        fun = t[0]
        return fun(t[1], t[2], t[3], t[4], t[5])

    def optimize(self, cost_function: Callable,
                 current_alpha: Optional[float] = 0.0,
                 codebook_log_info: Union[CodeBookLogInformation, None] = None) -> OptimizationResult:

        self._current_alpha = current_alpha if current_alpha is not None else 0.0

        init_time = time.time()
        min_loss = tf.Variable(1.0)
        params = [tf.Variable(0.1) for _ in range(self._number_parameters)]
        min_params = params.copy()
        learning_steps = list(range(self._learning_steps))
        number_steps = self._learning_steps

        iterator = zip([self._one_step_optimization] * number_steps,
                       [cost_function] * number_steps,
                       [params] * number_steps,
                       [init_time] * number_steps,
                       learning_steps,
                       [number_steps] * number_steps)

        # multiprocessing.set_start_method('spawn', force=True)

        pool = Pool(8)
        result = pool.map_async(func=self.uncurry_launch_execution,
                                iterable=iterator).get()

        pool.close()
        pool.join()

        for one_result in result:
            if one_result[0] < min_loss:
                min_loss = one_result[0]
                min_params = one_result[1]

        # print(f"min loss: {loss.numpy()}")

        return OptimizationResult(optimized_parameters=[param.numpy() for param in min_params],
                                  error_probability=min_loss.numpy())
