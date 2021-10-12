from abc import ABC
from typing import Callable, List, Union
from scipy.optimize import minimize
from tensorflow.python.framework.ops import EagerTensor


class Optimize(ABC):

    def optimize(self, alpha: float, cost_function: Callable) -> List[Union[float, EagerTensor]]:
        return minimize(cost_function,
                        [0],
                        args=(alpha, ),
                        method='BFGS',
                        tol=1e-6).x
