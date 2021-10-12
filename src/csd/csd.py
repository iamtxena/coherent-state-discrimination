from abc import ABC
from csd.typings import (CSDConfiguration,
                         RunConfiguration,
                         MeasuringTypes,
                         CodewordProbabilities,
                         PhotodetectorProbabilities,
                         ResultExecution)
from typeguard import typechecked
import strawberryfields as sf
from strawberryfields.api import Result
from strawberryfields.backends import BaseState
from typing import Union, cast, List
import numpy as np
import random
from csd.config import logger
from tensorflow.python.framework.ops import EagerTensor
from csd.optimize import Optimize


class CSD(ABC):
    NUM_SHOTS = 100
    A = 1
    MINUS_A = -1
    DEFAULT_CUTOFF_DIMENSION = 5
    DEFAULT_ALPHA = 0.7

    @typechecked
    def __init__(self, csd_config: Union[CSDConfiguration, None] = None):
        if csd_config is not None:
            self._displacement_magnitude = csd_config.get('displacement_magnitude')
            self._steps = csd_config.get('steps')
            self._learning_rate = csd_config.get('learning_rate')
            self._batch_size = csd_config.get('batch_size')
            self._threshold = csd_config.get('threshold')
        self._result = None
        self._circuit = None
        self._run_configuration: Union[RunConfiguration, None] = None

    def _create_codeword(self, letters: List[float], codeword_size=10) -> List[float]:
        return [random.choice(letters) for _ in range(codeword_size)]

    def _create_input_codeword(self, codeword: List[float], alpha_value: float) -> List[float]:
        return list(alpha_value * np.array(codeword))

    def _create_random_codeword(self, codeword_size=10, alpha_value: float = 0.7) -> List[float]:
        base_codeword = self._create_codeword(letters=[self.A, self.MINUS_A], codeword_size=codeword_size)
        return self._create_input_codeword(codeword=base_codeword, alpha_value=alpha_value)

    def _compute_a_minus_a_probabilities(self, codeword: List[float], alpha_value: float) -> CodewordProbabilities:
        # prob_a should be 0.5 with larger codewords
        prob_a = codeword.count(alpha_value) / len(codeword)
        return {
            'prob_a': prob_a,
            'prob_minus_a': 1 - prob_a
        }

    def _compute_photodetector_probabilities(self,
                                             engine: sf.Engine,
                                             codeword: List[float],
                                             params: List[float]) -> PhotodetectorProbabilities:
        no_click_probabilities = [self._run_one_layer_checking_measuring_type(engine=engine,
                                                                              letter=letter,
                                                                              params=params)
                                  for letter in codeword]
        return {
            'prob_click': list(1 - np.array(no_click_probabilities)),
            'prob_no_click': no_click_probabilities
        }

    def _compute_average_error_probability(self,
                                           codeword: List[float],
                                           alpha: float,
                                           codeword_prob: CodewordProbabilities,
                                           photodetector_prob: PhotodetectorProbabilities) -> float:

        p_errors = [(codeword_prob['prob_a'] * photodetector_prob['prob_click'][codeword_index]) if letter == alpha
                    else (codeword_prob['prob_minus_a'] * photodetector_prob['prob_no_click'][codeword_index])
                    for codeword_index, letter in enumerate(codeword)]

        return sum(p_errors) / len(codeword)

    def _run_one_layer_probabilities(self, engine: sf.Engine,
                                     letter: float,
                                     params: List[float]) -> Union[float, EagerTensor]:
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

        self._result = self._run_engine(engine=engine, letter=letter, params=params)
        return cast(Result, self._result).state.fock_prob([0])

    def _run_engine_and_compute_probability_0_photons(self,
                                                      engine: sf.Engine,
                                                      shots: int,
                                                      letter: float,
                                                      params: List[float]) -> Union[float, EagerTensor]:
        return sum([1 for read_value in [self._run_engine(engine=engine,
                                                          letter=letter,
                                                          params=params).samples[0][0]
                                         for _ in range(0, shots)] if read_value == 0]) / shots

    def _run_engine(self, engine: sf.Engine,
                    letter: float,
                    params: List[float]) -> Result:
        if self._run_configuration is None:
            raise ValueError("Run configuration not specified")

        # reset the engine if it has already been executed
        if engine.run_progs:
            engine.reset()

        self._result = engine.run(self._circuit, args={
            "alpha": letter,  # current alpha value from generated codeword
            "beta": params[0],  # beta
        })
        return self._result

    def _run_one_layer_sampling(self,
                                engine: sf.Engine,
                                letter: float,
                                params: List[float]) -> Union[float, EagerTensor]:
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
        if 'shots' not in self._run_configuration:
            logger.debug(f'Using default number of shots: {self.NUM_SHOTS}')
            self._run_configuration['shots'] = self.NUM_SHOTS

        return self._run_engine_and_compute_probability_0_photons(
            engine=engine,
            shots=self._run_configuration['shots'],
            letter=letter,
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
                                               letter: float,
                                               params: List[float]) -> Union[float, EagerTensor]:
        if self._run_configuration is None:
            raise ValueError("Run configuration not specified")

        if self._run_configuration['measuring_type'] is MeasuringTypes.SAMPLING:
            return self._run_one_layer_sampling(engine=engine, letter=letter, params=params)
        return self._run_one_layer_probabilities(engine=engine, letter=letter, params=params)

    def _cost_function(self, params: List[float], alpha: float) -> Union[float, EagerTensor]:
        photodetector_probabilities = self._compute_photodetector_probabilities(engine=self._engine,
                                                                                codeword=self._codeword,
                                                                                params=params)

        self._current_p_err = self._compute_average_error_probability(codeword=self._codeword,
                                                                      alpha=alpha,
                                                                      codeword_prob=self._codeword_probabilities,
                                                                      photodetector_prob=photodetector_probabilities)
        return float(self._current_p_err)

    @typechecked
    def execute(self, configuration: RunConfiguration) -> ResultExecution:
        """Run an experiment for the same codeword with the given configuration

        Args:
            configuration (RunConfiguration): Specific experiment configuration

        Returns:
            float: probability of getting |0> state
        """
        if configuration['number_layers'] != 1 or configuration['number_qumodes'] != 1:
            raise ValueError('Experiment only available for ONE qumode and ONE layer')
        self._run_configuration = configuration
        self._circuit = self._create_circuit()

        cutoff_dim = (self._run_configuration['cutoff_dim']
                      if 'cutoff_dim' in self._run_configuration
                      else self.DEFAULT_CUTOFF_DIMENSION)

        self._engine = sf.Engine(backend=self._run_configuration['backend'].value,
                                 backend_options={"cutoff_dim": cutoff_dim})

        codeword_size = configuration['codeword_size'] if 'codeword_size' in configuration else None
        alphas = configuration['alphas'] if 'alphas' in configuration else [self.DEFAULT_ALPHA]

        result: ResultExecution = {
            'alphas': [],
            'opt_betas': [],
            'p_err': [],
            'p_succ': [],
        }

        logger.debug(f"Executing One Layer circuit with Backend: {self._run_configuration['backend'].value}, "
                     " with measuring_type: "
                     f"{cast(MeasuringTypes, self._run_configuration['measuring_type']).value}")

        optimization = Optimize()
        for alpha in alphas:
            self._codeword = self._create_random_codeword(codeword_size, alpha)
            self._codeword_probabilities = self._compute_a_minus_a_probabilities(self._codeword, alpha)
            logger.debug(f'Optimizing for alpha: {alpha}')
            optimized_parameters = optimization.optimize(alpha=alpha, cost_function=self._cost_function)

            result['alphas'].append(np.round(alpha, 2))
            result['opt_betas'].append(optimized_parameters[0])
            result['p_err'].append(self._current_p_err)
            result['p_succ'].append(1 - self._current_p_err)

        return result

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
