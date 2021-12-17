"""
Wrapper for scikit-learn models to make them compatible for scipy's optimizers.
"""

from abc import ABC
import numpy as np
from loguru import logger
from sklearn.linear_model import LinearRegression


class _ModelWrapper(ABC):
    pass

class LinearRegressionWrapper(_ModelWrapper):
    def __init__(self, input_size, output_size, model=LinearRegression()):
        self.model = model
        self.input_size = input_size
        self.output_size = output_size

        # This call to fit is required to initialize the weights of the model.
        # We optimize these weights later using scipy.
        dummy_X = np.zeros((1, self.input_size))
        dummy_y = np.zeros((1, self.output_size))

        # logger.debug(f"{dummy_X.shape = }")
        # logger.debug(f"{dummy_y.shape = }")

        self.model.fit(dummy_X, dummy_y)

        self.serialized_length = self.input_size * self.output_size + self.output_size

    def get_learnable_parameters_as_flattened_list(self):
        learnable_weights = np.array([])

        # logger.debug(f"{self.model.coef_.shape = }")
        learnable_weights = np.append(learnable_weights, self.model.coef_.reshape(self.input_size * self.output_size))

        # logger.debug(f"{self.model.intercept_.shape = }")
        learnable_weights = np.append(learnable_weights, self.model.intercept_)

        return learnable_weights

    def set_learnable_parameteres_from_flattended_list(self, list_of_weights):
        assert len(list_of_weights) == self.serialized_length

        coef = list_of_weights[:self.input_size * self.output_size].reshape(
            (self.output_size, self.input_size)
        )
        intercept = list_of_weights[self.input_size * self.output_size: ]

        self.model.coef_ = coef
        assert np.allclose(self.model.coef_, coef)

        self.model.intercept_ = intercept
        assert np.allclose(self.model.intercept_, intercept)

        # logger.debug(f"{self.model.coef_.shape = }")
        # logger.debug(f"{self.model.intercept_.shape = }")

    def __call__(self, input_vector):
        # logger.debug(f"{input_vector.shape = }")

        return np.squeeze(
            self.model.predict(input_vector),
            axis=0)
