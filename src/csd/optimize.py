from abc import ABC
from typing import Callable, List, Union, Optional, cast
from csd.optimizers.parallel_tf import ParallelTFOptimizer
from csd.typings.typing import Backends, LearningRate, LearningSteps, OptimizationResult
from .optimizers.tf import TFOptimizer
from .optimizers.scipy import ScipyOptimizer
from .optimizers.parallel_scipy import ParallelOptimizer
from tensorflow import Variable


class Optimize(ABC):

    def __init__(self,
                 opt_backend: Optional[Backends] = Backends.FOCK,
                 nparams: int = 1,
                 parallel_optimization: bool = False,
                 learning_steps: LearningSteps = LearningSteps(default=300, high=500, extreme=2000),
                 learning_rate: LearningRate = LearningRate(default=0.01, high=0.001, extreme=0.001),
                 modes: int = 1,
                 params_name: List[str] = []):

        self._opt_backend: Backends = opt_backend if opt_backend is not None else Backends.FOCK
        self._optimizer: Union[TFOptimizer, ScipyOptimizer, ParallelOptimizer, ParallelTFOptimizer, None] = None

        if self._opt_backend is Backends.TENSORFLOW:
            self._optimizer = TFOptimizer(nparams=nparams,
                                          learning_steps=learning_steps,
                                          learning_rate=learning_rate,
                                          modes=modes,
                                          params_name=params_name)
        # if self._opt_backend is Backends.TENSORFLOW and parallel_optimization is True:
        #     self._optimizer = ParallelTFOptimizer(nparams=nparams)
        if ((self._opt_backend is Backends.FOCK or self._opt_backend is Backends.GAUSSIAN) and
                parallel_optimization is False):
            self._optimizer = ScipyOptimizer(nparams=nparams)
        if ((self._opt_backend is Backends.FOCK or self._opt_backend is Backends.GAUSSIAN) and
                parallel_optimization is True):
            self._optimizer = ParallelOptimizer(nparams=nparams)

    @property
    def parameters(self) -> List[Variable]:
        if self._opt_backend is not Backends.TENSORFLOW:
            raise ValueError("backend is NOT Tensorflow")

        return cast(TFOptimizer, self._optimizer).parameters

    @property
    def all_counts(self) -> List[Variable]:
        if self._opt_backend is not Backends.TENSORFLOW:
            raise ValueError("backend is NOT Tensorflow")

        return cast(TFOptimizer, self._optimizer).all_counts

    def optimize(self,
                 cost_function: Callable,
                 current_alpha: Optional[float] = 0.0) -> OptimizationResult:
        if self._optimizer is None:
            raise ValueError("optimizer not initilized")
        return self._optimizer.optimize(cost_function=cost_function, current_alpha=current_alpha)
