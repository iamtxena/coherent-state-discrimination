from abc import ABC
from typing import Callable, List, Union, Optional
# from scipy.optimize import minimize
import tensorflow as tf
from csd.typings import Backends
# import strawberryfields as sf


class Optimize(ABC):

    def __init__(self, backend: Optional[Union[Backends, None]] = None):
        self._backends: Union[Backends, None] = backend
        self._opt = None
        if self._backends is Backends.TENSORFLOW:
            self._opt = tf.keras.optimizers.Adam(learning_rate=0.01)
            self._beta = tf.Variable(0.1)

    def optimize(self, cost_function: Callable) -> List[float]:
        if self._backends is Backends.TENSORFLOW:
            return self._optimize_tensorflow(cost_function=cost_function)
        # return minimize(cost_function,
        #                 [0],
        #                 method='BFGS',
        #                 tol=1e-6).x
        raise NotImplementedError()

    def _optimize_tensorflow(self, cost_function: Callable) -> List[float]:
        if self._opt is None:
            raise ValueError("opt not initialized")
        if self._beta is None:
            raise ValueError("beta not initialized")

        with tf.GradientTape() as tape:
            loss = cost_function([self._beta])

        gradients = tape.gradient(loss, [self._beta])
        self._opt.apply_gradients(zip(gradients, [self._beta]))
        return [self._beta.numpy()]
