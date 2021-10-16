from abc import ABC
from csd.typings import (Backends, CSDConfiguration,
                         RunConfiguration,
                         MeasuringTypes,
                         PhotodetectorProbabilities,
                         ResultExecution)
# from typeguard import typechecked
import strawberryfields as sf
from strawberryfields.api import Result
from strawberryfields.backends import BaseState
from typing import Optional, Union, cast, List
import numpy as np
import random
from csd.config import logger
from tensorflow.python.framework.ops import EagerTensor
from csd.optimize import Optimize
from csd.plot import Plot
from csd.util import timing


class CSD(ABC):
    A = 1
    MINUS_A = -1
    DEFAULT_NUM_SHOTS = 100
    DEFAULT_CUTOFF_DIMENSION = 2
    DEFAULT_ALPHA = 0.7
    DEFAULT_BATCH_SIZE = 10
    DEFAULT_STEPS: int = 100
    DEFAULT_LEARNING_RATE: float = 0.1

    def __init__(self, csd_config: Union[CSDConfiguration, None] = None):
        self._shots = self.DEFAULT_NUM_SHOTS
        self._batch_size = self.DEFAULT_BATCH_SIZE
        self._cutoff_dim = self.DEFAULT_CUTOFF_DIMENSION
        self._steps = self.DEFAULT_STEPS
        self._learning_rate = self.DEFAULT_LEARNING_RATE
        self._batch: List[float] = []

        if csd_config is not None:
            self._beta = csd_config.get('beta', 0.1)
            self._steps = csd_config.get('steps', self.DEFAULT_STEPS)
            self._learning_rate = csd_config.get('learning_rate', self.DEFAULT_LEARNING_RATE)
            self._batch_size = csd_config.get('batch_size', self.DEFAULT_BATCH_SIZE)
            self._shots = csd_config.get('shots', self.DEFAULT_NUM_SHOTS)
            self._cutoff_dim = csd_config.get('cutoff_dim', self.DEFAULT_CUTOFF_DIMENSION)
            self._batch = csd_config.get('batch', [])
        self._result = None
        self._probability_results: List[ResultExecution] = []
        self._sampling_results: List[ResultExecution] = []
        self._plot: Union[Plot, None] = None
        self._circuit = None
        self._run_configuration: Union[RunConfiguration, None] = None
        self._alphas: Union[List[float], None] = None

    def _create_batch(self, samples: List[float], batch_size=10) -> List[float]:
        return [random.choice(samples) for _ in range(batch_size)]

    def _create_input_batch(self, batch: List[float], alpha_value: float) -> List[float]:
        return list(alpha_value * np.array(batch))

    def _create_random_batch(self, batch_size=10, alpha_value: float = 0.7) -> List[float]:
        base_batch = self._create_batch(samples=[self.A, self.MINUS_A], batch_size=batch_size)
        return self._create_input_batch(batch=base_batch, alpha_value=alpha_value)

    def _compute_photodetector_probabilities(self,
                                             engine: sf.Engine,
                                             batch: List[float],
                                             params: List[float]) -> PhotodetectorProbabilities:
        no_click_probabilities = self._compute_no_click_probabilities(engine, batch, params)
        if isinstance(no_click_probabilities, EagerTensor):
            return {
                'prob_click': 1 - no_click_probabilities,
                'prob_no_click': no_click_probabilities
            }
        return {
            'prob_click': list(1 - np.array(no_click_probabilities)),
            'prob_no_click': list(no_click_probabilities)
        }

    def _compute_no_click_probabilities(self,
                                        engine: sf.Engine,
                                        batch: List[float],
                                        params: List[float]) -> Union[List[float], EagerTensor]:
        if engine.backend_name == Backends.TENSORFLOW.value:
            return self._run_one_layer_checking_measuring_type(engine=engine,
                                                               sample_or_batch=batch,
                                                               params=params)

        return [cast(float, self._run_one_layer_checking_measuring_type(engine=engine,
                                                                        sample_or_batch=sample,
                                                                        params=params))
                for sample in batch]

    def _compute_average_error_probability(self,
                                           batch: List[float],
                                           photodetector_prob: PhotodetectorProbabilities) -> float:

        p_errors = [photodetector_prob['prob_click'][batch_index] if sample == self._alpha_value
                    else photodetector_prob['prob_no_click'][batch_index]
                    for batch_index, sample in enumerate(batch)]

        return sum(p_errors) / len(batch)

    def _run_one_layer_probabilities(self, engine: sf.Engine,
                                     sample_or_batch: Union[float, List[float]],
                                     params: List[float]) -> Union[Union[float, EagerTensor],
                                                                   List[Union[float, EagerTensor]]]:
        """Run a one layer experiment

        Args:
            engine (sf.Engine): Strawberry fields already instantiated engine

        Returns:
            float: probability of getting |0> state
        """
        if self._run_configuration is None:
            raise ValueError("Run configuration not specified")
        if self._circuit is None:
            raise ValueError("Circuit MUST be created first")

        self._result = self._run_engine(engine=engine, sample_or_batch=sample_or_batch, params=params)
        return cast(Result, self._result).state.fock_prob([0])

    def _run_engine_and_compute_probability_0_photons(self,
                                                      engine: sf.Engine,
                                                      shots: int,
                                                      sample_or_batch: Union[float, List[float]],
                                                      params: List[float]) -> Union[Union[float, EagerTensor],
                                                                                    List[Union[float, EagerTensor]]]:
        if engine.backend_name == Backends.GAUSSIAN.value:
            return sum([1 for read_value in self._run_engine(engine=engine,
                                                             sample_or_batch=sample_or_batch,
                                                             params=params,
                                                             shots=shots).samples
                        if read_value[0] == 0]) / shots

        return sum([1 for read_value in [self._run_engine(engine=engine,
                                                          sample_or_batch=sample_or_batch,
                                                          params=params).samples[0][0]
                                         for _ in range(0, shots)] if read_value == 0]) / shots

    def _run_engine(self,
                    engine: sf.Engine,
                    sample_or_batch: Union[float, List[float]],
                    params: List[float],
                    shots: int = 1) -> Result:
        if self._run_configuration is None:
            raise ValueError("Run configuration not specified")

        # reset the engine if it has already been executed
        if engine.run_progs:
            engine.reset()

        self._result = engine.run(self._circuit,
                                  args={
                                      # current alpha value from generated batch or
                                      # the batch when using tf with batches
                                      "alpha": sample_or_batch,
                                      "beta": params[0],  # beta
                                  },
                                  shots=shots)
        return self._result

    def _run_one_layer_sampling(self,
                                engine: sf.Engine,
                                sample_or_batch: Union[float, List[float]],
                                params: List[float]) -> Union[Union[float, EagerTensor],
                                                              List[Union[float, EagerTensor]]]:
        """Run a one layer experiment doing MeasureFock and performing samplint with nshots

        Args:
            engine (sf.Engine): Strawberry fields already instantiated engine

        Returns:
            float: probability of getting |0> state
        """
        if self._run_configuration is None:
            raise ValueError("Run configuration not specified")
        if self._circuit is None:
            raise ValueError("Circuit MUST be created first")
        if 'shots' in self._run_configuration:
            self._shots = self._run_configuration['shots']

        return self._run_engine_and_compute_probability_0_photons(
            engine=engine,
            shots=self._shots,
            sample_or_batch=sample_or_batch,
            params=params)

    def _create_circuit(self) -> sf.Program:
        """Creates a circuit to run an experiment based on configuration parameters

        Returns:
            sf.Program: the created circuit
        """
        if self._run_configuration is None:
            raise ValueError("Run configuration not specified")

        prog = sf.Program(self._run_configuration['number_qumodes'])
        alpha = prog.params("alpha")
        beta = prog.params("beta")

        with prog.context as q:
            sf.ops.Dgate(alpha, 0.0) | q[0]
            sf.ops.Dgate(beta, 0.0) | q[0]
            if self._run_configuration['measuring_type'] is MeasuringTypes.SAMPLING:
                sf.ops.MeasureFock() | q[0]

        return prog

    def _run_one_layer_checking_measuring_type(self,
                                               engine: sf.Engine,
                                               sample_or_batch: Union[float, List[float]],
                                               params: List[float]) -> Union[Union[float, EagerTensor],
                                                                             List[Union[float, EagerTensor]]]:
        if self._run_configuration is None:
            raise ValueError("Run configuration not specified")

        if self._run_configuration['measuring_type'] is MeasuringTypes.SAMPLING:
            return self._run_one_layer_sampling(engine=engine, sample_or_batch=sample_or_batch, params=params)
        return self._run_one_layer_probabilities(engine=engine, sample_or_batch=sample_or_batch, params=params)

    def _cost_function(self, params: List[float]) -> Union[float, EagerTensor]:
        photodetector_probabilities = self._compute_photodetector_probabilities(engine=self._engine,
                                                                                batch=self._batch,
                                                                                params=params)

        p_err = self._compute_average_error_probability(batch=self._batch,
                                                        photodetector_prob=photodetector_probabilities)
        self._current_p_err = self._save_current_p_error(p_err)
        return p_err

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
        if configuration['number_layers'] != 1 or configuration['number_qumodes'] != 1:
            raise ValueError('Experiment only available for ONE qumode and ONE layer')
        self._run_configuration = configuration.copy()

        backend_is_tf = self._run_configuration['backend'] == Backends.TENSORFLOW
        self._circuit = self._create_circuit()

        self._batch_size = (configuration['batch_size'] if 'batch_size' in configuration
                            else self._batch_size)
        alphas = configuration['alphas'] if 'alphas' in configuration else [self.DEFAULT_ALPHA]
        self._alphas = alphas

        cutoff_dim = (self._run_configuration['cutoff_dim']
                      if 'cutoff_dim' in self._run_configuration
                      else self._cutoff_dim)

        self._engine = sf.Engine(backend=self._run_configuration['backend'].value,
                                 backend_options={
                                     "cutoff_dim": cutoff_dim,
                                     "batch_size": self._batch_size if backend_is_tf else None
        })

        result: ResultExecution = {
            'alphas': [],
            'batches': [],
            'opt_betas': [],
            'p_err': [],
            'p_succ': [],
            'backend': self._run_configuration['backend'].value,
            'measuring_type': self._run_configuration['measuring_type'].value,
            'plot_label': self._set_plot_label(backend=self._run_configuration['backend'],
                                               measuring_type=self._run_configuration['measuring_type'])
        }

        logger.debug(f"Executing One Layer circuit with Backend: {self._run_configuration['backend'].value}, "
                     " with measuring_type: "
                     f"{cast(MeasuringTypes, self._run_configuration['measuring_type']).value}"
                     f" and cutoff_dim: {self._cutoff_dim}")

        for alpha in alphas:
            self._alpha_value = alpha
            logger.debug(f'Optimizing for alpha: {np.round(self._alpha_value, 2)}')

            optimization = Optimize(backend=self._run_configuration['backend'])
            learning_steps = self._set_learning_steps(backend_is_tf)

            for step in range(0, learning_steps):
                optimized_parameters = self._execute_for_one_alpha(batch_size=self._batch_size,
                                                                   optimization=optimization)

                if backend_is_tf and (step + 1) % 100 == 0:
                    logger.debug("Learned displacement value at step {}: {}".format(
                        step + 1, optimized_parameters[0]))

            logger.debug(f'Optimized for alpha: {np.round(self._alpha_value, 2)}'
                         f' beta: {optimized_parameters[0]}'
                         f' p_succ: {1 - self._current_p_err}')
            result['alphas'].append(np.round(self._alpha_value, 2))
            result['batches'].append(self._batch)
            result['opt_betas'].append(optimized_parameters[0])
            result['p_err'].append(self._current_p_err)
            result['p_succ'].append(1 - self._current_p_err)

        return result

    def _set_learning_steps(self, backend_is_tf) -> int:
        if self._run_configuration is None:
            raise ValueError("Run configuration not specified")

        if backend_is_tf:
            steps = (self._steps if self._steps is not None and 'steps' not in self._run_configuration
                     else self._steps)
            return self._run_configuration['steps'] if 'steps' in self._run_configuration else steps
        return 1

    def _execute_for_one_alpha(self,
                               batch_size: int,
                               optimization: Optimize) -> Union[EagerTensor, List[float]]:

        self._batch = self._set_batch(batch_size)
        return optimization.optimize(cost_function=self._cost_function)

    def _set_batch(self, batch_size: int) -> List[float]:
        if self._run_configuration is None:
            raise ValueError("Run configuration not specified")
        if self._batch != [] and 'batch' not in self._run_configuration:
            return self._batch
        if 'batch' in self._run_configuration:
            return self._run_configuration['batch']
        return self._create_random_batch(batch_size=batch_size, alpha_value=self._alpha_value)

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

    @timing
    def execute_all_backends_and_measuring_types(
            self,
            alphas: List[float],
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
            self._probability_results = self._execute_with_given_backends(alphas=alphas,
                                                                          backends=[
                                                                              # Backends.FOCK,
                                                                              # Backends.GAUSSIAN,
                                                                              Backends.TENSORFLOW,
                                                                          ],
                                                                          measuring_type=MeasuringTypes.PROBABILITIES)

        if required_sampling_execution:
            self._sampling_results += self._execute_with_given_backends(alphas=alphas,
                                                                        backends=[
                                                                            Backends.FOCK,
                                                                            Backends.GAUSSIAN
                                                                            # Backends.TENSORFLOW,
                                                                        ],
                                                                        measuring_type=MeasuringTypes.SAMPLING)
        return self._probability_results + self._sampling_results

    def _execute_with_given_backends(self,
                                     alphas: List[float],
                                     backends: List[Backends],
                                     measuring_type: MeasuringTypes) -> List[ResultExecution]:

        return [self.execute(configuration=RunConfiguration({
            'alphas': alphas,
            'backend': backend,
            'number_qumodes': 1,
            'number_layers': 1,
            'measuring_type': measuring_type,
            'shots': self._shots,
            'batch_size': self._batch_size,
            'cutoff_dim': self._cutoff_dim,
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

    def show_result(self) -> dict:
        if self._result is None:
            raise ValueError("Circuit not executed yet.")
        sf_result = cast(Result, self._result)
        sf_state = cast(BaseState, sf_result.state)

        return {
            'result': str(sf_result),
            'state': str(sf_state),
            'trace': sf_state.trace(),
            'density_matrix': sf_state.dm(),
            'dm_shape': cast(np.ndarray, sf_state.dm()).shape,
            'samples': sf_result.samples,
            'first_sample': sf_result.samples[0],
            'fock_probability': sf_state.fock_prob([0])
        }
