from abc import ABC
from csd.circuit import Circuit
from csd.engine import Engine
from csd.global_result_manager import GlobalResultManager
# from csd.optimization_testing import OptimizationTesting
from csd.tf_engine import TFEngine
from csd.typings.global_result import GlobalResult
# from csd.typings.optimization_testing import OptimizationTestingOptions
from csd.typings.typing import (Backends,
                                CSDConfiguration,
                                CutOffDimensions,
                                LearningRate,
                                LearningSteps,
                                OptimizationBackends,
                                OptimizationResult,
                                RunConfiguration,
                                MeasuringTypes,
                                ResultExecution,
                                Architecture, RunningTypes)
from typing import Optional, Union, cast, List
import numpy as np
from time import time
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
    DEFAULT_CUTOFF_DIMENSION = CutOffDimensions(default=7, high=14, extreme=30)
    DEFAULT_ALPHA = 0.7
    DEFAULT_BATCH_SIZE = 1
    DEFAULT_WORD_SIZE = 10
    DEFAULT_LEARNING_STEPS = LearningSteps(default=300, high=500, extreme=2000)
    DEFAULT_LEARNING_RATE = LearningRate(default=0.01, high=0.001, extreme=0.001)
    DEFAULT_PLAYS = 1

    def __init__(self, csd_config: Union[CSDConfiguration, None] = None):
        self._set_default_values()

        if csd_config is not None:
            self._set_values_from_config(csd_config)

        self._set_default_values_after_config()

    def _set_default_values_after_config(self):
        self._architecture = self._set_architecture(self._architecture).copy()

    def _set_values_from_config(self, csd_config):
        self._alphas = csd_config.get('alphas', [self.DEFAULT_ALPHA])
        self._learning_steps = csd_config.get('learning_steps', self.DEFAULT_LEARNING_STEPS)
        self._learning_rate = csd_config.get('learning_rate', self.DEFAULT_LEARNING_RATE)
        self._batch_size = csd_config.get('batch_size', self.DEFAULT_BATCH_SIZE)
        self._shots = csd_config.get('shots', self.DEFAULT_NUM_SHOTS)
        self._plays = csd_config.get('plays', self.DEFAULT_PLAYS)
        self._cutoff_dim = csd_config.get('cutoff_dim', self.DEFAULT_CUTOFF_DIMENSION)
        self._save_results = csd_config.get('save_results', False)
        self._save_plots = csd_config.get('save_plots', False)
        self._architecture = self._set_architecture(csd_config.get('architecture')).copy()
        self._parallel_optimization = csd_config.get('parallel_optimization', False)

    def _set_default_values(self):
        self._alphas: List[float] = []
        self._learning_steps = self.DEFAULT_LEARNING_STEPS
        self._learning_rate = self.DEFAULT_LEARNING_RATE
        self._batch_size = self.DEFAULT_BATCH_SIZE
        self._shots = self.DEFAULT_NUM_SHOTS
        self._plays = self.DEFAULT_PLAYS
        self._cutoff_dim = self.DEFAULT_CUTOFF_DIMENSION
        self._save_results = False
        self._save_plots = False
        self._parallel_optimization = False

        self._current_batch: Union[Batch, None] = None
        self._result = None
        self._probability_results: List[ResultExecution] = []
        self._sampling_results: List[ResultExecution] = []
        self._plot: Union[Plot, None] = None
        self._training_circuit: Union[Circuit, None] = None
        self._run_configuration: Union[RunConfiguration, None] = None

    def _set_architecture(self, architecture: Optional[Architecture] = None) -> Architecture:
        tmp_architecture = self._default_architecture()
        if architecture is None:
            return tmp_architecture
        if 'number_modes' in architecture:
            tmp_architecture['number_modes'] = architecture['number_modes']
        if 'number_ancillas' in architecture:
            tmp_architecture['number_ancillas'] = architecture['number_ancillas']
        if 'number_layers' in architecture:
            tmp_architecture['number_layers'] = architecture['number_layers']
        if 'squeezing' in architecture:
            tmp_architecture['squeezing'] = architecture['squeezing']

        return tmp_architecture

    def _default_architecture(self) -> Architecture:
        return {
            'number_modes': 1,
            'number_ancillas': 0,
            'number_layers': 1,
            'squeezing': False,
        }

    def _create_batch_for_alpha(self, alpha_value: float, random_words: bool) -> Batch:
        if self._training_circuit is None:
            raise ValueError("Circuit must be initialized")

        return Batch(size=self._batch_size,
                     word_size=self._training_circuit.number_input_modes,
                     alpha_value=alpha_value,
                     random_words=random_words)

    def _create_circuit(self, running_type: RunningTypes = RunningTypes.TRAINING) -> Circuit:
        """Creates a circuit to run an experiment based on configuration parameters

        Returns:
            sf.Program: the created circuit
        """
        if self._run_configuration is None:
            raise ValueError("Run configuration not specified")

        return Circuit(architecture=self._architecture,
                       measuring_type=self._run_configuration['measuring_type'],
                       running_type=running_type)

    def _cost_function(self, params: List[float]) -> Union[float, EagerTensor]:
        if self._training_circuit is None:
            raise ValueError("Circuit must be initialized")
        if self._run_configuration is None:
            raise ValueError("Run configuration not specified")
        if self._current_batch is None:
            raise ValueError("Current Batch must be initialized")
        if self._engine is None:
            raise ValueError("Engine must be initialized")

        return CostFunction(batch=self._current_batch,
                            params=params,
                            options=CostFunctionOptions(
                                engine=self._engine,
                                circuit=self._training_circuit,
                                backend_name=self._engine.backend_name,
                                measuring_type=self._run_configuration['measuring_type'],
                                shots=self._shots,
                                plays=self._plays)).run_and_compute_average_batch_error_probability()

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
        self._training_circuit = self._create_circuit(running_type=RunningTypes.TRAINING)
        # self._testing_circuit = self._create_circuit(running_type=RunningTypes.TESTING)
        result = self._init_result()

        logger.debug(f"Executing One Layer circuit with Backend: {self._run_configuration['run_backend'].value}, "
                     " with measuring_type: "
                     f"{cast(MeasuringTypes, self._run_configuration['measuring_type']).value} \n"
                     f"batch_size:{self._batch_size} plays:{self._plays}"
                     f" modes:{self._training_circuit.number_input_modes}"
                     f" ancillas: {self._training_circuit.number_ancillas} \n"
                     f"steps: {self._learning_steps}, l_rate: {self._learning_rate}, cutoff_dim: {self._cutoff_dim} \n"
                     f"layers:{self._architecture['number_layers']} squeezing: {self._architecture['squeezing']}")

        return self._train_and_test(result=result,
                                    random_words=(not self._backend_is_tf()))

    def _get_succ_prob(self,
                       one_alpha_success_probability: Union[EagerTensor, None],
                       one_alpha_optimization_result: OptimizationResult) -> float:
        return (one_alpha_success_probability.numpy() if one_alpha_success_probability is not None
                else 1 - one_alpha_optimization_result.error_probability)

    def _train_and_test(self,
                        result: ResultExecution,
                        random_words: bool):
        if self._training_circuit is None:
            raise ValueError("Training circuit must be initialized")
        # if self._testing_circuit is None:
        #     raise ValueError("Testing circuit must be initialized")

        start_time = time()
        for sample_alpha in self._alphas:
            one_alpha_start_time = time()
            self._alpha_value = sample_alpha
            self._current_batch = self._create_batch_for_alpha(alpha_value=self._alpha_value, random_words=random_words)
            self._engine = self._create_engine()
            one_alpha_optimization_result = self._train_for_one_alpha()
            one_alpha_success_probability = None
            # one_alpha_success_probability = self._test_for_one_alpha(
            #     optimized_parameters=one_alpha_optimization_result.optimized_parameters)

            self._update_result(result=result,
                                one_alpha_optimization_result=one_alpha_optimization_result,
                                one_alpha_success_probability=one_alpha_success_probability)
            self._write_result(alpha=sample_alpha,
                               one_alpha_start_time=one_alpha_start_time,
                               success_probability=self._get_succ_prob(
                                   one_alpha_success_probability, one_alpha_optimization_result))

            logger.debug(f'Optimized and trained for alpha: {np.round(self._alpha_value, 2)} '
                         f'pSucc: {self._get_succ_prob(one_alpha_success_probability, one_alpha_optimization_result)} '
                         f"batch_size:{self._batch_size} plays:{self._plays}"
                         f" modes:{self._training_circuit.number_input_modes}"
                         f" ancillas: {self._training_circuit.number_ancillas} steps: {self._learning_steps}, "
                         f"l_rate: {self._learning_rate}, cutoff_dim: {self._cutoff_dim}"
                         f" layers:{self._architecture['number_layers']} squeezing: {self._architecture['squeezing']}")

        self._update_result_with_total_time(result=result, start_time=start_time)
        self._save_results_to_file(result=result)
        self._save_plot_to_file(result=result)

        return result

    # def _test_for_one_alpha(self, optimized_parameters: List[float]) -> EagerTensor:
    #     if self._testing_circuit is None:
    #         raise ValueError("Circuit must be initialized")
    #     if self._run_configuration is None:
    #         raise ValueError("Run configuration not specified")
    #     if self._current_batch is None:
    #         raise ValueError("Current Batch must be initialized")
    #     # list_optimized_parameters = [one_parameter.numpy() for one_parameter in optimized_parameters]
    #     # optimized_parameters = [-self._alpha_value]
    #     logger.debug(
    #         f'Going to train with the optimized parameters: {optimized_parameters}')

    #     return OptimizationTesting(batch=self._current_batch,
    #                                params=optimized_parameters,
    #                                options=OptimizationTestingOptions(
    #                                    engine=self._engine,
    #                                    circuit=self._testing_circuit,
    #                                    backend_name=self._engine.backend_name,
    #                                    shots=self._shots,
    #                                    plays=self._plays)).run_and_compute_average_batch_success_probability()

    def _write_result(self,
                      alpha: float,
                      one_alpha_start_time: float,
                      success_probability: float):
        if self._training_circuit is None:
            raise ValueError("Circuit must be initialized")

        GlobalResultManager().write_result(GlobalResult(alpha=alpha,
                                                        success_probability=success_probability,
                                                        number_modes=self._training_circuit.number_input_modes,
                                                        time_in_seconds=time() - one_alpha_start_time,
                                                        squeezing=self._architecture['squeezing'],
                                                        number_ancillas=self._training_circuit.number_ancillas))

    def _update_result_with_total_time(self, result: ResultExecution, start_time: float) -> None:
        end_time = time()
        result['total_time'] = end_time - start_time

    def _create_optimization(self) -> Optimize:
        if self._training_circuit is None:
            raise ValueError("Circuit must be initialized")
        if self._run_configuration is None:
            raise ValueError("Run configuration not specified")
        return Optimize(opt_backend=self._run_configuration['optimization_backend'],
                        nparams=self._training_circuit.free_parameters,
                        parallel_optimization=self._parallel_optimization,
                        learning_steps=self._learning_steps,
                        learning_rate=self._learning_rate,
                        modes=self._training_circuit.number_input_modes,
                        params_name=[*self._training_circuit.parameters])

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
    def _train_for_one_alpha(self) -> OptimizationResult:
        if self._training_circuit is None:
            raise ValueError("Circuit must be initialized")

        logger.debug(f'Optimizing for alpha: {np.round(self._alpha_value, 2)} \n'
                     f"batch_size:{self._batch_size} plays:{self._plays}"
                     f" modes:{self._training_circuit.number_input_modes}"
                     f" ancillas: {self._training_circuit.number_ancillas} \n "
                     f"steps: {self._learning_steps}, l_rate: {self._learning_rate}, cutoff_dim: {self._cutoff_dim} \n"
                     f"layers:{self._architecture['number_layers']} squeezing: {self._architecture['squeezing']}")

        self._optimization = self._create_optimization()
        optimization_result = self._optimization.optimize(
            cost_function=self._cost_function, current_alpha=self._alpha_value)

        logger.debug(f'Trained for alpha: {np.round(self._alpha_value, 2)}'
                     f' parameters: {optimization_result.optimized_parameters}'
                     f' p_err: {optimization_result.error_probability}')

        return optimization_result

    def _update_result(self,
                       result: ResultExecution,
                       one_alpha_optimization_result: OptimizationResult,
                       one_alpha_success_probability: EagerTensor):
        # logger.debug(f'Tested for alpha: {np.round(self._alpha_value, 2)}'
        #              f' parameters: {one_alpha_optimization_result.optimized_parameters}'
        #              f' p_succ: {one_alpha_success_probability.numpy()}')
        result['alphas'].append(np.round(self._alpha_value, 2))
        # result['batches'].append([])  # self._current_batch.codewords if self._current_batch is not None else [])
        result['opt_params'].append(list(one_alpha_optimization_result.optimized_parameters))
        result['p_err'].append(one_alpha_optimization_result.error_probability)
        result['p_succ'].append(self._get_succ_prob(one_alpha_success_probability, one_alpha_optimization_result))

    def _init_result(self):
        if self._training_circuit is None:
            raise ValueError("Circuit must be initialized")

        result: ResultExecution = {
            'alphas': [],
            'batches': [],
            'opt_params': [],
            'p_err': [],
            'p_succ': [],
            'result_backend': self._run_configuration['run_backend'].value,
            'measuring_type': self._run_configuration['measuring_type'].value,
            'plot_label': self._set_plot_label(plot_label_backend=self._run_configuration['run_backend'],
                                               measuring_type=self._run_configuration['measuring_type']),
            'plot_title': self._set_plot_title(batch_size=self._batch_size,
                                               plays=self._plays,
                                               modes=self._training_circuit.number_input_modes,
                                               layers=self._architecture['number_layers'],
                                               squeezing=self._architecture['squeezing'],
                                               ancillas=self._training_circuit.number_ancillas,),
            'total_time': 0.0
        }

        return result

    def _create_engine(self) -> Union[Engine, TFEngine]:
        if self._run_configuration is None:
            raise ValueError("Run configuration not specified")
        if self._training_circuit is None:
            raise ValueError("Circuit must be initialized")

        current_cutoff = self._cutoff_dim.default

        if self._alpha_value > 0.6:
            current_cutoff = self._cutoff_dim.high
        if self._alpha_value > 1.2:
            current_cutoff = self._cutoff_dim.extreme

        if self._backend_is_tf():
            return TFEngine(
                number_modes=self._training_circuit.number_modes,
                engine_backend=Backends.TENSORFLOW, options={
                    "cutoff_dim": current_cutoff,
                    "batch_size": self._batch_size
                })
        return Engine(
            number_modes=self._training_circuit.number_modes,
            engine_backend=self._run_configuration['run_backend'], options={
                "cutoff_dim": current_cutoff
            })

    def _backend_is_tf(self):
        return self._run_configuration['run_backend'] == Backends.TENSORFLOW

    def _set_plot_label(self, plot_label_backend: Backends, measuring_type: MeasuringTypes) -> str:
        """Set the label for the success probability plot

        Args:
            backend (Backends): Current experiment backend
            measuring_type (MeasuringTypes): Current experiment measuring type

        Returns:
            str: the determined label
        """
        if plot_label_backend is Backends.FOCK and measuring_type is MeasuringTypes.PROBABILITIES:
            return "pFockProb(a)"
        if plot_label_backend is Backends.GAUSSIAN and measuring_type is MeasuringTypes.PROBABILITIES:
            return "pGausProb(a)"
        if plot_label_backend is Backends.TENSORFLOW and measuring_type is MeasuringTypes.PROBABILITIES:
            return "pTFProb(a)"
        if plot_label_backend is Backends.FOCK and measuring_type is MeasuringTypes.SAMPLING:
            return "pFockSampl(a)"
        if plot_label_backend is Backends.TENSORFLOW and measuring_type is MeasuringTypes.SAMPLING:
            return "pTFSampl(a)"
        if plot_label_backend is Backends.GAUSSIAN and measuring_type is MeasuringTypes.SAMPLING:
            return "pGausSampl(a)"
        raise ValueError(
            f"Values not supported. backend: {plot_label_backend.value} and measuring_type: {measuring_type.value}")

    def _set_plot_title(self,
                        batch_size: int,
                        plays: int,
                        modes: int,
                        layers: int,
                        squeezing: bool,
                        ancillas: int) -> str:
        if self._run_configuration is None:
            raise ValueError("Run configuration not specified")

        return (f"backend:{self._run_configuration['run_backend'].value}, "
                f"measuring:{self._run_configuration['measuring_type'].value}, \n"
                f"batch size:{batch_size}, plays:{plays}, modes:{modes}, ancillas:{ancillas}, \n"
                f"steps: {self._learning_steps}, l_rate: {self._learning_rate}, cutoff_dim: {self._cutoff_dim}, \n"
                f"layers,{layers}, squeezing:{squeezing}")

    @timing
    def execute_all_backends_and_measuring_types(
            self,
            backends: List[Backends] = [Backends.FOCK,
                                        Backends.GAUSSIAN,
                                        Backends.TENSORFLOW],
            measuring_types: Optional[List[MeasuringTypes]] = [MeasuringTypes.PROBABILITIES],
            optimization_backend: OptimizationBackends = OptimizationBackends.SCIPY) -> List[ResultExecution]:
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
                                                                          measuring_type=MeasuringTypes.PROBABILITIES,
                                                                          optimization_backend=optimization_backend)

        if required_sampling_execution:
            self._sampling_results += self._execute_with_given_backends(backends=backends,
                                                                        measuring_type=MeasuringTypes.SAMPLING,
                                                                        optimization_backend=optimization_backend)
        return self._probability_results + self._sampling_results

    def _execute_with_given_backends(self,
                                     backends: List[Backends],
                                     measuring_type: MeasuringTypes,
                                     optimization_backend: OptimizationBackends) -> List[ResultExecution]:

        return [self.execute(configuration=RunConfiguration({
            'run_backend': backend,
            'measuring_type': measuring_type,
            'optimization_backend': optimization_backend,
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
