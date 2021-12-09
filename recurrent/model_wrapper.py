"""
Wrapper for scikit-learn models to make them compatible for scipy's optimizers.
"""

from abc import ABC
import numpy as np


class _ModelWrapper(ABC):
    pass

class LinearRegressionWrapper(_ModelWrapper):
    def __init__(self, model, input_size, output_size):
        self.model = model
        self.input_size = input_size
        self.output_size = output_size

        # This call to fit is required to initialize the weights of the model.
        # We optimize these weights later using scipy.
        self.model.fit(np.zeros((1, self.input_size)), np.zeros(1, self.output_size))

        self.serialized_length = self.input_size * self.output_size + self.output_size

    def get_learnable_parameters_as_flattened_list(self):
        learnable_weights = np.array([])

        learnable_weights.append(self.model.coef_.reshape(self.input_size * self.output_size))
        learnable_weights.append(self.model.intercept_)

        return learnable_weights

    def set_learnable_parameteres_from_flattended_list(self, list_of_weights):
        assert len(list_of_weights) == self.serialized_length

        self.model.coef_ = list_of_weights[:self.input_size * self.output_size].reshape(
            (self.input_size, self.output_size)
        )

        self.model.intercept_ = list_of_weights[self.input_size * self.output_size: ]
