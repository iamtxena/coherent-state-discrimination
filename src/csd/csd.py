from abc import ABC
from csd.circuit import Circuit
from csd.engine import Engine
from csd.tf_engine import TFEngine
from csd.typings.typing import (Backends,
                                CSDConfiguration,
                                RunConfiguration,
                                MeasuringTypes,
                                ResultExecution,
                                Architecture)
from typing import Optional, Union, cast, List
import numpy as np
from csd.config import logger
from tensorflow.python.framework.ops import EagerTensor
from csd.optimize import Optimize
from csd.plot import Plot
from csd.typings.cost_function import CostFunctionOptions
from csd.util import timing, save_object_to_disk
from csd.batch import Batch
from .cost_function import CostFunction


class CSD(ABC):

    DEFAULT_NUM_SHOTS = 1000
    DEFAULT_CUTOFF_DIMENSION = 10
    DEFAULT_ALPHA = 0.7
    DEFAULT_BATCH_SIZE = 1
    DEFAULT_WORD_SIZE = 10
    DEFAULT_STEPS = 500
    DEFAULT_LEARNING_RATE: float = 0.1
    DEFAULT_PLAYS = 100

    def __init__(self, csd_config: Union[CSDConfiguration, None] = None):
        self._set_default_values()

        if csd_config is not None:
            self._set_values_from_config(csd_config)

        self._set_default_values_after_config()

    def _set_default_values_after_config(self):
        self._architecture = self._set_architecture(self._architecture).copy()

    def _set_values_from_config(self, csd_config):
        self._alphas = csd_config.get('alphas', [self.DEFAULT_ALPHA])
        self._steps = csd_config.get('steps', self.DEFAULT_STEPS)
        self._learning_rate = csd_config.get('learning_rate', self.DEFAULT_LEARNING_RATE)
        self._batch_size = csd_config.get('batch_size', self.DEFAULT_BATCH_SIZE)
        self._shots = csd_config.get('shots', self.DEFAULT_NUM_SHOTS)
        self._plays = csd_config.get('plays', self.DEFAULT_PLAYS)
        self._cutoff_dim = csd_config.get('cutoff_dim', self.DEFAULT_CUTOFF_DIMENSION)
        self._save_results = csd_config.get('save_results', False)
        self._save_plots = csd_config.get('save_plots', False)
        self._architecture = self._set_architecture(csd_config.get('architecture')).copy()

    def _set_default_values(self):
        self._alphas: List[float] = []
        self._steps = self.DEFAULT_STEPS
        self._learning_rate = self.DEFAULT_LEARNING_RATE
        self._batch_size = self.DEFAULT_BATCH_SIZE
        self._shots = self.DEFAULT_NUM_SHOTS
        self._plays = self.DEFAULT_PLAYS
        self._cutoff_dim = self.DEFAULT_CUTOFF_DIMENSION
        self._save_results = False
        self._save_plots = False

        self._current_batch: Union[Batch, None] = None
        self._result = None
        self._probability_results: List[ResultExecution] = []
        self._sampling_results: List[ResultExecution] = []
        self._plot: Union[Plot, None] = None
        self._circuit: Union[Circuit, None] = None
        self._run_configuration: Union[RunConfiguration, None] = None

    def _set_architecture(self, architecture: Optional[Architecture] = None) -> Architecture:
        tmp_architecture = self._default_architecture()
        if architecture is None:
            return tmp_architecture
        if 'number_modes' in architecture:
            tmp_architecture['number_modes'] = architecture['number_modes']
        if 'number_layers' in architecture:
            tmp_architecture['number_layers'] = architecture['number_layers']
        if 'squeezing' in architecture:
            tmp_architecture['squeezing'] = architecture['squeezing']

        return tmp_architecture

    def _default_architecture(self) -> Architecture:
        return {
            'number_modes': 1,
            'number_layers': 1,
            'squeezing': False,
        }

    def _create_batch_for_alpha(self, alpha_value: float) -> Batch:
        return Batch(size=self._batch_size, word_size=self._architecture['number_modes'], alpha_value=alpha_value)

    def _create_circuit(self) -> Circuit:
        """Creates a circuit to run an experiment based on configuration parameters

        Returns:
            sf.Program: the created circuit
        """
        if self._run_configuration is None:
            raise ValueError("Run configuration not specified")

        return Circuit(architecture=self._architecture,
                       measuring_type=self._run_configuration['measuring_type'])

    def _cost_function(self, params: List[float]) -> Union[float, EagerTensor]:
        if self._circuit is None:
            raise ValueError("Circuit must be initialized")
        if self._run_configuration is None:
            raise ValueError("Run configuration not specified")
        if self._current_batch is None:
            raise ValueError("Current Batch must be initialized")

        cost_function = CostFunction(batch=self._current_batch,
                                     params=params,
                                     options=CostFunctionOptions(
                                         engine=self._engine,
                                         circuit=self._circuit,
                                         backend_name=self._engine.backend_name,
                                         measuring_type=self._run_configuration['measuring_type'],
                                         shots=self._shots,
                                         plays=self._plays))

        loss = cost_function.run_and_compute_average_batch_error_probability()
        self._current_p_err = self._save_current_p_error(p_err=loss)
        # logger.debug(f"average error: {self._current_p_err} for params: {params}")
        return loss

    def _save_current_p_error(self, p_err: Union[float, EagerTensor]) -> float:
        if isinstance(p_err, EagerTensor):
            return float(p_err.numpy())
        return p_err

    @timing
    def execute(self, configuration: RunConfiguration) -> ResultExecution:
        """Run an experiment for the same batch with the given configuration

        Args:
            configuration (RunConfiguration): Specific experiment configuration

        Returns:
            float: probability of getting |0> state
        """
        if not configuration:
            raise ValueError('No configuration specified')

        self._run_configuration = configuration.copy()
        self._circuit = self._create_circuit()
        self._engine = self._create_engine()
        optimization = self._create_optimization()
        result = self._init_result()

        logger.debug(f"Executing One Layer circuit with Backend: {self._run_configuration['backend'].value}, "
                     " with measuring_type: "
                     f"{cast(MeasuringTypes, self._run_configuration['measuring_type']).value}")

        return self._single_process_optimization(optimization, result)

    def _single_process_optimization(self, optimization, result):
        for sample_alpha in self._alphas:
            optimized_parameters = self._execute_for_one_alpha(optimization=optimization,
                                                               sample_alpha=sample_alpha)
            self._update_result(result=result, optimized_parameters=optimized_parameters)
        self._save_results_to_file(result)
        self._save_plot_to_file(result)

        return result

    def _create_optimization(self) -> Optimize:
        if self._circuit is None:
            raise ValueError("Circuit must be initialized")
        if self._run_configuration is None:
            raise ValueError("Run configuration not specified")
        return Optimize(backend=self._run_configuration['backend'],
                        nparams=self._circuit.free_parameters)

    def _save_plot_to_file(self, result):
        if self._save_plots:
            if self._plot is None:
                self._plot = Plot(alphas=self._alphas)
            logger.info("Save plot to file")
            self._plot.plot_success_probabilities(executions=[result], save_plot=True)

    def _save_results_to_file(self, result):
        if self._save_results:
            logger.info("Save results to file")
            save_object_to_disk(obj=result, path='results')

    @timing
    def _execute_for_one_alpha(self, optimization: Optimize, sample_alpha: float) -> List[float]:
        self._alpha_value = sample_alpha
        self._current_batch = self._create_batch_for_alpha(alpha_value=self._alpha_value)

        logger.debug(f'Optimizing for alpha: {np.round(self._alpha_value, 2)}')

        learning_steps = self._set_learning_steps()

        for step in range(learning_steps):
            optimized_parameters = optimization.optimize(cost_function=self._cost_function)
            self._print_optimized_parameters_for_tf_backend_only(step, optimized_parameters)
        return optimized_parameters

    def _print_optimized_parameters_for_tf_backend_only(self, step, optimized_parameters):
        if self._backend_is_tf() and (step + 1) % 100 == 0:
            logger.debug("Learned parameters value at step {}: {}".format(
                step + 1, optimized_parameters))

    def _update_result(self, result: ResultExecution, optimized_parameters: List[float]):
        logger.debug(f'Optimized for alpha: {np.round(self._alpha_value, 2)}'
                     f' parameters: {optimized_parameters}'
                     f' p_succ: {1 - self._current_p_err}')
        result['alphas'].append(np.round(self._alpha_value, 2))
        # result['batches'].append([])  # self._current_batch.codewords if self._current_batch is not None else [])
        result['opt_params'].append(list(optimized_parameters))
        result['p_err'].append(self._current_p_err)
        result['p_succ'].append(1 - self._current_p_err)

    def _init_result(self):
        result: ResultExecution = {
            'alphas': [],
            'batches': [],
            'opt_params': [],
            'p_err': [],
            'p_succ': [],
            'backend': self._run_configuration['backend'].value,
            'measuring_type': self._run_configuration['measuring_type'].value,
            'plot_label': self._set_plot_label(backend=self._run_configuration['backend'],
                                               measuring_type=self._run_configuration['measuring_type']),
            'plot_title': self._set_plot_title(batch_size=self._batch_size,
                                               plays=self._plays,
                                               modes=self._architecture['number_modes'],
                                               layers=self._architecture['number_layers'],
                                               squeezing=self._architecture['squeezing'])
        }

        return result

    def _create_engine(self) -> Union[Engine, TFEngine]:
        if self._run_configuration is None:
            raise ValueError("Run configuration not specified")

        if self._backend_is_tf():
            return TFEngine(backend=self._run_configuration['backend'], options={
                "cutoff_dim": self._cutoff_dim,
                "batch_size": self._batch_size
            })
        return Engine(backend=self._run_configuration['backend'], options={
            "cutoff_dim": self._cutoff_dim
        })

    def _backend_is_tf(self):
        return self._run_configuration['backend'] == Backends.TENSORFLOW

    def _set_learning_steps(self) -> int:
        if self._backend_is_tf():
            return self._steps
        return 1

    def _set_plot_label(self, backend: Backends, measuring_type: MeasuringTypes) -> str:
        """Set the label for the success probability plot

        Args:
            backend (Backends): Current experiment backend
            measuring_type (MeasuringTypes): Current experiment measuring type

        Returns:
            str: the determined label
        """
        if backend is Backends.FOCK and measuring_type is MeasuringTypes.PROBABILITIES:
            return "pFockProb(a)"
        if backend is Backends.GAUSSIAN and measuring_type is MeasuringTypes.PROBABILITIES:
            return "pGausProb(a)"
        if backend is Backends.TENSORFLOW and measuring_type is MeasuringTypes.PROBABILITIES:
            return "pTFProb(a)"
        if backend is Backends.FOCK and measuring_type is MeasuringTypes.SAMPLING:
            return "pFockSampl(a)"
        if backend is Backends.TENSORFLOW and measuring_type is MeasuringTypes.SAMPLING:
            return "pTFSampl(a)"
        if backend is Backends.GAUSSIAN and measuring_type is MeasuringTypes.SAMPLING:
            return "pGausSampl(a)"
        raise ValueError(f"Values not supported. backend: {backend.value} and measuring_type: {measuring_type.value}")

    def _set_plot_title(self,
                        batch_size: int,
                        plays: int,
                        modes: int,
                        layers: int,
                        squeezing: bool) -> str:
        if self._run_configuration is None:
            raise ValueError("Run configuration not specified")

        return (f"backend:{self._run_configuration['backend'].value}, "
                f"measuring:{self._run_configuration['measuring_type'].value}, \n"
                f"batch size:{batch_size}, plays:{plays}, modes:{modes}, layers,{layers}, squeezing:{squeezing}")

    @timing
    def execute_all_backends_and_measuring_types(
            self,
            backends: List[Backends] = [Backends.FOCK,
                                        Backends.GAUSSIAN,
                                        Backends.TENSORFLOW],
            measuring_types: Optional[List[MeasuringTypes]] = [MeasuringTypes.PROBABILITIES]) -> List[ResultExecution]:
        """Execute the experiment for all backends and measuring types

        Args:
            alphas (List[float]): List of input alpha values to generate batches and run the experiments

        Returns:
            List[ResultExecution]: List of the Result Executions
        """
        if measuring_types is None:
            measuring_types = [MeasuringTypes.PROBABILITIES]

        required_probability_execution = measuring_types.count(MeasuringTypes.PROBABILITIES) > 0
        required_sampling_execution = measuring_types.count(MeasuringTypes.SAMPLING) > 0

        if required_probability_execution:
            self._probability_results = self._execute_with_given_backends(backends=backends,
                                                                          measuring_type=MeasuringTypes.PROBABILITIES)

        if required_sampling_execution:
            self._sampling_results += self._execute_with_given_backends(backends=backends,
                                                                        measuring_type=MeasuringTypes.SAMPLING)
        return self._probability_results + self._sampling_results

    def _execute_with_given_backends(self,
                                     backends: List[Backends],
                                     measuring_type: MeasuringTypes) -> List[ResultExecution]:

        return [self.execute(configuration=RunConfiguration({
            'backend': backend,
            'measuring_type': measuring_type,
        })) for backend in backends]

    def plot_success_probabilities(self,
                                   alphas: Optional[Union[List[float], None]] = None,
                                   measuring_types: Optional[Union[List[MeasuringTypes], None]] = None) -> None:
        if self._alphas is None and alphas is None:
            raise ValueError("alphas not set. You must execute the optimization first.")
        if self._alphas is None:
            self._alphas = alphas
        if self._plot is None:
            self._plot = Plot(alphas=self._alphas)
        if measuring_types is None:
            self._plot.plot_success_probabilities()
            return None

        if self._probability_results == [] and self._sampling_results == []:
            raise ValueError("results not set.")

        required_probability_execution = measuring_types.count(MeasuringTypes.PROBABILITIES) > 0
        required_sampling_execution = measuring_types.count(MeasuringTypes.SAMPLING) > 0

        if required_probability_execution and required_sampling_execution:
            self._plot.plot_success_probabilities(executions=self._probability_results + self._sampling_results)
            return None
        if required_probability_execution:
            self._plot.plot_success_probabilities(executions=self._probability_results)
            return None
        if required_sampling_execution:
            self._plot.plot_success_probabilities(executions=self._sampling_results)
            return None
        raise ValueError('Value not expected')
