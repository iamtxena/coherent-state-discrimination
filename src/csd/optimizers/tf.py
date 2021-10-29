from abc import ABC
from typing import Callable, List
import tensorflow as tf
from csd.config import logger


class TFOptimizer(ABC):

    def __init__(self, nparams: int = 1):
        self._opt = tf.keras.optimizers.Adam(learning_rate=0.01)
        self._params = [tf.Variable(0.1) for _ in range(nparams)]
        logger.debug(f'params: {self._params}, {type(self._params)}, len: {len(self._params)}')

    def optimize(self, cost_function: Callable) -> List[float]:
        if self._opt is None:
            raise ValueError("opt not initialized")
        if self._params is None:
            raise ValueError("params not initialized")

        with tf.GradientTape() as tape:
            loss = cost_function(self._params)

        logger.debug(f'loss: {loss}, {type(loss)}')
        logger.debug(f'params: {self._params}, {type(self._params)}')
        gradients = tape.gradient(loss, self._params)
        self._opt.apply_gradients(zip(gradients, self._params))
        return [param.numpy() for param in self._params]
