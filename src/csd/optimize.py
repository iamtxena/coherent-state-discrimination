from abc import ABC
from typing import Callable, List, Union, Optional
from scipy.optimize import minimize
import tensorflow as tf
from csd.typings import Backends


class Optimize(ABC):

    def __init__(self, backend: Optional[Union[Backends, None]] = None):
        self._backends: Union[Backends, None] = backend
        self._opt = None
        if self._backends is Backends.TENSORFLOW:
            self._opt = tf.keras.optimizers.Adam(learning_rate=0.01)

    def optimize(self, alpha: float, cost_function: Callable) -> List[float]:
        if self._backends is Backends.TENSORFLOW:
            return self._optimize_tensorflow(alpha=alpha, cost_function=cost_function)
        return minimize(cost_function,
                        [0],
                        args=(alpha, ),
                        method='BFGS',
                        tol=1e-6).x

    def _optimize_tensorflow(self, alpha: float, cost_function: Callable) -> List[float]:
        if self._opt is None:
            raise ValueError("opt not initialized")
        beta = tf.Variable(0.1)

        with tf.GradientTape() as tape:
            loss = cost_function([beta], alpha)
        gradients = tape.gradient(loss, [beta])
        self._opt.apply_gradients(zip(gradients, [beta]))
        return [float(beta.numpy())]
