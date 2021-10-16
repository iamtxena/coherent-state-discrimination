from abc import ABC
from typing import Callable, List
import tensorflow as tf


class TFOptimizer(ABC):

    def __init__(self):
        self._opt = tf.keras.optimizers.Adam(learning_rate=0.01)
        self._beta = tf.Variable(0.1)

    def optimize(self, cost_function: Callable) -> List[float]:
        if self._opt is None:
            raise ValueError("opt not initialized")
        if self._beta is None:
            raise ValueError("beta not initialized")

        with tf.GradientTape() as tape:
            loss = cost_function([self._beta])

        gradients = tape.gradient(loss, [self._beta])
        self._opt.apply_gradients(zip(gradients, [self._beta]))
        return [self._beta.numpy()]
