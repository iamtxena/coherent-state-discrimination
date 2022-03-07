from abc import ABC
from csd.best_codebook import BestCodeBook
from csd.circuit import Circuit
from csd.codebooks import CodeBooks
from csd.codeword import CodeWord
from csd.engine import Engine
from csd.global_result_manager import GlobalResultManager
from csd.ideal_probabilities import IdealLinearCodesHelstromProbability, IdealLinearCodesHomodyneProbability
from csd.optimization_testing import OptimizationTesting
# from csd.optimization_testing import OptimizationTesting
from csd.tf_engine import TFEngine
from csd.top5_best_codebooks import Top5_BestCodeBooks
from csd.typings.global_result import GlobalResult
from csd.typings.optimization_testing import OptimizationTestingOptions
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
from typing import Optional, Tuple, Union, cast, List
import numpy as np
from time import time
from csd.config import logger
from tensorflow.python.framework.ops import EagerTensor
from csd.optimize import Optimize
from csd.plot import Plot
from csd.typings.cost_function import CostFunctionOptions
from csd.utils.util import CodeBookLogInformation, print_codebook_log, timing, save_object_to_disk
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
    DEFAULT_MAX_COMBINATIONS = 120

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
        self._max_combinations = csd_config.get('max_combinations', self.DEFAULT_MAX_COMBINATIONS)

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
        self._max_combinations = self.DEFAULT_MAX_COMBINATIONS

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
        if self._current_codebook is None:
            raise ValueError("Current Codebook must be initialized")
        if self._engine is None:
            raise ValueError("Engine must be initialized")

        return CostFunction(batch=Batch(size=0,
                                        word_size=0,
                                        alpha_value=self._alpha_value,
                                        all_words=False,
                                        input_batch=self._current_codebook),
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
        self._testing_circuit = self._create_circuit(running_type=RunningTypes.TRAINING)
        training_result = self._init_result()
        testing_result = self._init_result()

        logger.debug(f"Executing One Layer circuit with Backend: {self._run_configuration['run_backend'].value}, "
                     " with measuring_type: "
                     f"{cast(MeasuringTypes, self._run_configuration['measuring_type']).value} \n"
                     f"batch_size:{self._batch_size} plays:{self._plays}"
                     f" modes:{self._training_circuit.number_input_modes}"
                     f" ancillas: {self._training_circuit.number_ancillas} \n"
                     f"steps: {self._learning_steps}, l_rate: {self._learning_rate}, cutoff_dim: {self._cutoff_dim} \n"
                     f"layers:{self._architecture['number_layers']} squeezing: {self._architecture['squeezing']}")

        return self._train_and_test(training_result=training_result,
                                    testing_result=testing_result,
                                    random_words=(not self._backend_is_tf()))

    def _get_succ_prob(self,
                       one_alpha_success_probability: Union[EagerTensor, None],
                       one_alpha_optimization_result: OptimizationResult) -> float:
        return (one_alpha_success_probability.numpy() if one_alpha_success_probability is not None
                else 1 - one_alpha_optimization_result.error_probability)

    def _train_and_test(self,
                        training_result: ResultExecution,
                        testing_result: ResultExecution,
                        random_words: bool):
        if self._training_circuit is None:
            raise ValueError("Training circuit must be initialized")
        if self._testing_circuit is None:
            raise ValueError("Testing circuit must be initialized")

        # change it when you are not testing the optimized parameters
        self._training_path_results = GlobalResultManager(testing=False)._base_dir_path
        self._testing_path_results = GlobalResultManager(testing=True)._base_dir_path

        start_time = time()
        for sample_alpha in self._alphas:
            one_alpha_start_time = time()
            self._alpha_value = sample_alpha
            self._current_batch = self._create_batch_for_alpha(alpha_value=self._alpha_value, random_words=random_words)

            training_average_error_probability_all_codebooks = 0.0
            training_average_ideal_homodyne_probability_all_codebooks = 0.0
            training_average_ideal_helstrom_probability_all_codebooks = 0.0
            testing_average_error_probability_all_codebooks = 0.0
            testing_average_ideal_homodyne_probability_all_codebooks = 0.0
            testing_average_ideal_helstrom_probability_all_codebooks = 0.0
            codebooks = CodeBooks(batch=self._current_batch, max_combinations=self._max_combinations)
            self._all_codebooks_size = codebooks.size
            logger.debug(
                f'Optimizing for alpha: {np.round(self._alpha_value, 2)} '
                f'with {self._all_codebooks_size} codebooks.')
            top5_training_codebooks = Top5_BestCodeBooks()
            top5_testing_codebooks = Top5_BestCodeBooks()

            for index, codebook in enumerate(codebooks.codebooks):
                one_codebook_start_time = time()
                self._current_codebook_index = index
                self._current_codebook = codebook
                logger.debug(f"current codebook: {self._current_codebook}")
                self._codebook_size = len(codebook)
                self._engine = self._create_engine()
                self._current_codebook_log_info = CodeBookLogInformation(
                    alpha_value=np.round(self._alpha_value, 2),
                    alpha_init_time=one_alpha_start_time,
                    codebooks_size=codebooks.size,
                    codebook_iteration=index,
                    codebook_current_iteration_init_time=one_codebook_start_time
                )
                # TRAINING RESULTS
                one_codebook_optimization_result = self._train_for_one_alpha_one_codebook()
                training_average_error_probability_all_codebooks += one_codebook_optimization_result.error_probability
                training_homodyne_probability = IdealLinearCodesHomodyneProbability(
                    codebook=self._current_codebook).homodyne_probability
                training_average_ideal_homodyne_probability_all_codebooks += training_homodyne_probability
                training_helstrom_probability = IdealLinearCodesHelstromProbability(
                    codebook=self._current_codebook).helstrom_probability
                training_average_ideal_helstrom_probability_all_codebooks += training_helstrom_probability
                top5_training_codebooks.add(potential_best_codebook=BestCodeBook(
                    codebook=self._current_codebook,
                    measurements=one_codebook_optimization_result.measurements,
                    success_probability=self._get_succ_prob(None, one_codebook_optimization_result),
                    helstrom_probability=training_helstrom_probability,
                    homodyne_probability=training_homodyne_probability,
                    optimized_parameters=one_codebook_optimization_result.optimized_parameters,
                    program=self._training_circuit.circuit))

                # TESTING RESULTS
                one_alpha_success_probability, measurements = self._test_for_one_alpha_one_codebook(
                    optimized_parameters=one_codebook_optimization_result.optimized_parameters)

                testing_average_error_probability_all_codebooks += one_codebook_optimization_result.error_probability
                testing_homodyne_probability = IdealLinearCodesHomodyneProbability(
                    codebook=self._current_codebook).homodyne_probability
                testing_average_ideal_homodyne_probability_all_codebooks += testing_homodyne_probability
                testing_helstrom_probability = IdealLinearCodesHelstromProbability(
                    codebook=self._current_codebook).helstrom_probability
                testing_average_ideal_helstrom_probability_all_codebooks += testing_helstrom_probability

                top5_testing_codebooks.add(potential_best_codebook=BestCodeBook(
                    codebook=self._current_codebook,
                    measurements=measurements,
                    success_probability=one_alpha_success_probability.numpy(),
                    helstrom_probability=testing_helstrom_probability,
                    homodyne_probability=testing_homodyne_probability,
                    optimized_parameters=one_codebook_optimization_result.optimized_parameters,
                    program=self._testing_circuit.circuit))
                self._current_codebook_log_info = CodeBookLogInformation(
                    alpha_value=np.round(self._alpha_value, 2),
                    alpha_init_time=one_alpha_start_time,
                    codebooks_size=codebooks.size,
                    codebook_iteration=index,
                    codebook_current_iteration_init_time=one_codebook_start_time
                )
                print_codebook_log(log_info=self._current_codebook_log_info)
                logger.debug(
                    f'pSucc: {self._get_succ_prob(one_alpha_success_probability, one_codebook_optimization_result)} '
                    f"codebook_size:{self._codebook_size}")

            if not codebooks.size > 0:
                logger.warning('codebooks size is 0. Going to next iteration.')
                continue

            training_average_error_probability_all_codebooks /= codebooks.size
            training_average_ideal_homodyne_probability_all_codebooks /= codebooks.size
            training_average_ideal_helstrom_probability_all_codebooks /= codebooks.size
            testing_average_error_probability_all_codebooks /= codebooks.size
            testing_average_ideal_homodyne_probability_all_codebooks /= codebooks.size
            testing_average_ideal_helstrom_probability_all_codebooks /= codebooks.size

            # TRAINED RESULTS
            training_one_alpha_optimization_result = OptimizationResult(
                optimized_parameters=one_codebook_optimization_result.optimized_parameters,
                error_probability=training_average_error_probability_all_codebooks,
                measurements=one_codebook_optimization_result.measurements)
            self._update_result(result=training_result,
                                one_alpha_optimization_result=training_one_alpha_optimization_result,
                                one_alpha_success_probability=self._get_succ_prob(
                                    None, training_one_alpha_optimization_result),
                                helstrom_probability=training_average_ideal_helstrom_probability_all_codebooks,
                                homodyne_probability=training_average_ideal_homodyne_probability_all_codebooks,
                                number_mode=self._training_circuit.number_input_modes)
            self._write_result(alpha=sample_alpha,
                               one_alpha_start_time=one_alpha_start_time,
                               success_probability=self._get_succ_prob(
                                   None, training_one_alpha_optimization_result),
                               helstrom_probability=training_average_ideal_helstrom_probability_all_codebooks,
                               homodyne_probability=training_average_ideal_homodyne_probability_all_codebooks,
                               best_codebook=top5_training_codebooks.first,
                               testing=False)
            # TESTED RESULTS
            testing_one_alpha_optimization_result = OptimizationResult(
                optimized_parameters=one_codebook_optimization_result.optimized_parameters,
                error_probability=testing_average_error_probability_all_codebooks,
                measurements=measurements)
            self._update_result(result=testing_result,
                                one_alpha_optimization_result=testing_one_alpha_optimization_result,
                                one_alpha_success_probability=one_alpha_success_probability.numpy(),
                                helstrom_probability=testing_average_ideal_helstrom_probability_all_codebooks,
                                homodyne_probability=testing_average_ideal_homodyne_probability_all_codebooks,
                                number_mode=self._testing_circuit.number_input_modes)
            self._write_result(alpha=sample_alpha,
                               one_alpha_start_time=one_alpha_start_time,
                               success_probability=one_alpha_success_probability.numpy(),
                               helstrom_probability=testing_average_ideal_helstrom_probability_all_codebooks,
                               homodyne_probability=testing_average_ideal_homodyne_probability_all_codebooks,
                               best_codebook=top5_testing_codebooks.first,
                               testing=True)

            logger.debug(
                f'Optimized and trained for alpha: {np.round(self._alpha_value, 2)} '
                f'pSucc: {one_alpha_success_probability.numpy()} '
                f"batch_size:{self._batch_size} plays:{self._plays}"
                f" modes:{self._training_circuit.number_input_modes}"
                f" ancillas: {self._training_circuit.number_ancillas} steps: {self._learning_steps}, "
                f"l_rate: {self._learning_rate}, cutoff_dim: {self._cutoff_dim}"
                f" layers:{self._architecture['number_layers']} squeezing: {self._architecture['squeezing']}")
            logger.debug(
                f'alpha: {np.round(self._alpha_value, 2)}\n'
                f'TRAINING IDEAL HELSTROM: {training_average_ideal_homodyne_probability_all_codebooks}\n'
                f'TRAINING IDEAL HOMODYNE: {training_average_ideal_helstrom_probability_all_codebooks}\n'
                f'TESTING IDEAL HELSTROM: {testing_average_ideal_homodyne_probability_all_codebooks}\n'
                f'TESTING IDEAL HOMODYNE: {testing_average_ideal_helstrom_probability_all_codebooks}\n'
                f'BEST TRAINING success probability: {top5_training_codebooks.first.success_probability}\n'
                f'BEST TESTING success probability: {top5_testing_codebooks.first.success_probability}\n'
                f'Best TRAINING codebook: {top5_training_codebooks.first}\n'
                f'Best TESTING codebook: {top5_testing_codebooks.first}\n\n')
            self._print_top5_codebooks(top5_codebooks=top5_training_codebooks, testing=False)
            self._print_top5_codebooks(top5_codebooks=top5_testing_codebooks, testing=True)

        self._update_result_with_total_time(result=training_result, start_time=start_time)
        self._save_results_to_file(result=training_result)
        self._save_plot_to_file(result=training_result)
        self._update_result_with_total_time(result=testing_result, start_time=start_time)
        self._save_results_to_file(result=testing_result)
        self._save_plot_to_file(result=testing_result)

        return testing_result

    def _print_top5_codebooks(self, top5_codebooks: Top5_BestCodeBooks, testing: bool) -> None:
        first = f'FIRST: {top5_codebooks.first}\n\n' if top5_codebooks.size >= 1 else ''
        second = f'SECOND: {top5_codebooks.second}\n\n' if top5_codebooks.size >= 2 else ''
        third = f'THIRD: {top5_codebooks.third}\n\n' if top5_codebooks.size >= 3 else ''
        fourth = f'FOURTH: {top5_codebooks.fourth}\n\n' if top5_codebooks.size >= 4 else ''
        fifth = f'FIFTH: {top5_codebooks.fifth}\n\n' if top5_codebooks.size >= 5 else ''

        codebook_type = 'TESTING' if testing else 'TRAINING'
        logger.debug('\n\n************************************\n'
                     f'** TOP{top5_codebooks.size} {codebook_type} codebooks: \n'
                     '************************************\n\n'
                     f'{first}'
                     f'{second}'
                     f'{third}'
                     f'{fourth}'
                     f'{fifth}'
                     '************************************\n')

    def _update_best_and_worst_success_probability(self,
                                                   current_success_probability: float,
                                                   best_success_probability: float,
                                                   worst_success_probability: float,
                                                   current_codebook: List[CodeWord],
                                                   best_codebook: List[CodeWord]) -> Tuple[
                                                       float, float, List[CodeWord]]:
        updated_best_success_probability = best_success_probability
        updated_worst_success_probability = worst_success_probability
        new_best_codebook = best_codebook.copy()
        if current_success_probability > best_success_probability:
            updated_best_success_probability = current_success_probability
            new_best_codebook = current_codebook.copy()
        if current_success_probability < worst_success_probability:
            updated_worst_success_probability = current_success_probability
        return updated_best_success_probability, updated_worst_success_probability, new_best_codebook

    def _test_for_one_alpha_one_codebook(self, optimized_parameters: List[float]) -> Tuple[EagerTensor, List[CodeWord]]:
        if self._testing_circuit is None:
            raise ValueError("Circuit must be initialized")
        if self._run_configuration is None:
            raise ValueError("Run configuration not specified")
        if self._current_batch is None:
            raise ValueError("Current Batch must be initialized")
        if self._current_codebook is None:
            raise ValueError("Current codebook must be initialized")
        # list_optimized_parameters = [one_parameter.numpy() for one_parameter in optimized_parameters]
        # optimized_parameters = [-self._alpha_value]
        logger.debug(
            f'Going to test with the trained optimized parameters: {optimized_parameters}')

        return OptimizationTesting(batch=Batch(size=0,
                                               word_size=0,
                                               alpha_value=self._alpha_value,
                                               all_words=False,
                                               input_batch=self._current_codebook),
                                   params=optimized_parameters,
                                   options=OptimizationTestingOptions(
                                       engine=self._engine,
                                       circuit=self._testing_circuit,
                                       backend_name=self._engine.backend_name,
                                       measuring_type=self._run_configuration['measuring_type'],
                                       shots=self._shots,
                                       plays=self._plays)).run_and_compute_average_batch_success_probability()

    def _write_result(self,
                      alpha: float,
                      one_alpha_start_time: float,
                      success_probability: float,
                      helstrom_probability: float,
                      homodyne_probability: float,
                      best_codebook: BestCodeBook,
                      testing: bool):
        if self._training_circuit is None:
            raise ValueError("Circuit must be initialized")

        GlobalResultManager(testing=testing).write_result(
            GlobalResult(alpha=alpha,
                         success_probability=success_probability,
                         number_modes=self._training_circuit.number_input_modes,
                         time_in_seconds=time() - one_alpha_start_time,
                         squeezing=self._architecture['squeezing'],
                         number_ancillas=self._training_circuit.number_ancillas,
                         helstrom_probability=helstrom_probability,
                         homodyne_probability=homodyne_probability,
                         best_success_probability=best_codebook.success_probability,
                         best_helstrom_probability=best_codebook.helstrom_probability,
                         best_homodyne_probability=best_codebook.homodyne_probability,
                         best_codebook=best_codebook.binary_codebook,
                         best_measurements=best_codebook.binary_measurements,
                         best_optimized_parameters=best_codebook.parsed_optimized_parameters))
        if testing:
            directory_name = (f"./circuit_tex/alpha_{np.round(alpha, 2)}"
                              f"_modes_{self._training_circuit.number_input_modes}"
                              f"_squeezing_{self._architecture['squeezing']}"
                              f"_ancillas_{self._training_circuit.number_ancillas}/")

            print("\n\n****************************************\n\n"
                  "CIRCUIT for "
                  f'alpha: {np.round(self._alpha_value, 2)}'
                  f" codebook_size:{self._codebook_size}"
                  f" modes:{self._training_circuit.number_input_modes}"
                  f" ancillas: {self._training_circuit.number_ancillas} \n "
                  f" cutoff_dim: {self._cutoff_dim}"
                  f" squeezing: {self._architecture['squeezing']}: \n")
            best_codebook.program.print()
            print("\n\n****************************************\n\n")
            best_codebook.program.draw_circuit(tex_dir=directory_name, write_to_file=True)

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
                self._plot = Plot(alphas=self._alphas, path=self._path_results)
            logger.info("Save plot to file")
            self._plot.plot_success_probabilities(executions=[result], save_plot=True)

    def _save_results_to_file(self, result):
        if self._save_results:
            logger.info("Save results to file")
            save_object_to_disk(obj=result, path=self._path_results)

    @timing
    def _train_for_one_alpha_one_codebook(self) -> OptimizationResult:
        if self._training_circuit is None:
            raise ValueError("Circuit must be initialized")

        logger.debug(f'Optimizing for alpha: {np.round(self._alpha_value, 2)} \n'
                     f"codebook_size:{self._codebook_size} plays:{self._plays}"
                     f" modes:{self._training_circuit.number_input_modes}"
                     f" ancillas: {self._training_circuit.number_ancillas} \n "
                     f"steps: {self._learning_steps}, l_rate: {self._learning_rate}, cutoff_dim: {self._cutoff_dim} \n"
                     f"layers:{self._architecture['number_layers']} squeezing: {self._architecture['squeezing']}")

        self._optimization = self._create_optimization()
        optimization_result = self._optimization.optimize(
            cost_function=self._cost_function,
            current_alpha=self._alpha_value,
            codebook_log_info=self._current_codebook_log_info)

        logger.debug(f'Trained for alpha: {np.round(self._alpha_value, 2)}'
                     f' parameters: {optimization_result.optimized_parameters}'
                     f' p_err: {optimization_result.error_probability}')

        return optimization_result

    def _update_result(self,
                       result: ResultExecution,
                       one_alpha_optimization_result: OptimizationResult,
                       one_alpha_success_probability: float,
                       helstrom_probability: float,
                       homodyne_probability: float,
                       number_mode: int):
        # logger.debug(f'Tested for alpha: {np.round(self._alpha_value, 2)}'
        #              f' parameters: {one_alpha_optimization_result.optimized_parameters}'
        #              f' p_succ: {one_alpha_success_probability.numpy()}')
        result['alphas'].append(np.round(self._alpha_value, 2))
        # result['batches'].append([])  # self._current_batch.codewords if self._current_batch is not None else [])
        result['opt_params'].append(list(one_alpha_optimization_result.optimized_parameters))
        result['p_err'].append(one_alpha_optimization_result.error_probability)
        result['p_succ'].append(one_alpha_success_probability)
        result['p_helstrom'].append(helstrom_probability)
        result['p_homodyne'].append(homodyne_probability)
        result['number_modes'].append(number_mode)

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
            'total_time': 0.0,
            'p_helstrom': [],
            'p_homodyne': [],
            'number_modes': []
        }

        return result

    def _create_engine(self) -> Union[Engine, TFEngine]:
        if self._run_configuration is None:
            raise ValueError("Run configuration not specified")
        if self._training_circuit is None:
            raise ValueError("Circuit must be initialized")
        if self._current_codebook is None:
            raise ValueError("_current_codebook must be initialized")
        if self._codebook_size is None:
            raise ValueError("_codebook_size must be initialized")

        self._current_cutoff = self._cutoff_dim.default

        if self._alpha_value > 0.6:
            self._current_cutoff = self._cutoff_dim.high
        if self._alpha_value > 1.2:
            self._current_cutoff = self._cutoff_dim.extreme

        if self._backend_is_tf():
            return TFEngine(
                number_modes=self._training_circuit.number_modes,
                engine_backend=Backends.TENSORFLOW, options={
                    "cutoff_dim": self._current_cutoff,
                    "batch_size": self._codebook_size if self._codebook_size > 1 else None
                })
        return Engine(
            number_modes=self._training_circuit.number_modes,
            engine_backend=self._run_configuration['run_backend'], options={
                "cutoff_dim": self._current_cutoff
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
            path_results = GlobalResultManager(testing=True)._base_dir_path
            self._plot = Plot(alphas=self._alphas, path=path_results)
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
