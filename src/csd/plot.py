# plot.py

from abc import ABC
from typing import List, Optional, Tuple
import matplotlib.pyplot as plt
# from typeguard import typechecked
from csd.ideal_probabilities import IdealProbabilities
from csd.typings.global_result import GlobalResult
from csd.typings.typing import ResultExecution
from csd.util import set_current_time, _fix_path, set_friendly_time


class Plot(ABC):
    """ Class for plotting the results

    """

    # @typechecked
    def __init__(self, alphas: List[float] = None, number_modes: int = 1):
        if alphas is None:
            raise ValueError("alphas not set.")
        self._ideal_probabilities = IdealProbabilities(alphas=alphas, number_modes=number_modes)
        self._alphas = alphas

    def success_probabilities(self,
                              number_modes: List[int],
                              global_results: List[GlobalResult],
                              save_plot: Optional[bool] = False) -> None:
        fig, axes = plt.subplots(figsize=[10, 8])
        probs_labels = [self._ideal_probabilities.p_hels]

        for number_mode in number_modes:
            probs = [global_result.success_probability
                     for global_result in global_results
                     if global_result.number_modes == number_mode]
            one_prob_label = (probs, f"mode_{number_mode}")
            probs_labels.append(one_prob_label)
            probs_labels.append(IdealProbabilities(alphas=self._alphas, number_modes=number_mode).p_homos)

        self._plot_computed_variables(save_plot,
                                      fig,
                                      axes,
                                      probs_labels,
                                      "Average Success Probability Results",
                                      'Average Success Probabilities',
                                      "_probs")

    def _plot_computed_variables(self, save_plot, fig, axes, probs_labels, title, ylabel, suffix=None):
        plt.title(title, fontsize=24)
        self._plot_probs_and_label_into_axis(axes=axes,
                                             probs_labels=probs_labels)
        plt.legend()
        plt.xlabel('alpha values')
        plt.ylabel(ylabel)
        self._show_or_save_plot(save_plot, fig, suffix)

    def distances(self,
                  number_modes: List[int],
                  global_results: List[GlobalResult],
                  save_plot: Optional[bool] = False) -> None:
        fig, axes = plt.subplots(figsize=[10, 8])
        distances_labels = []

        for number_mode in number_modes:
            distances = [global_result.distance_to_homodyne_probability
                         for global_result in global_results
                         if global_result.number_modes == number_mode]
            one_distance_label = (distances, f"mode_{number_mode}")
            distances_labels.append(one_distance_label)

        self._plot_computed_variables(save_plot,
                                      fig,
                                      axes,
                                      distances_labels,
                                      "Distance to Homodyne Probability Results",
                                      'Distance to Homodyne Probability',
                                      "_dist")

    def bit_error_rates(self,
                        number_modes: List[int],
                        global_results: List[GlobalResult],
                        save_plot: Optional[bool] = False) -> None:
        fig, axes = plt.subplots(figsize=[10, 8])
        bit_error_labels = []

        for number_mode in number_modes:
            bit_error_rates = [global_result.bit_error_rate
                               for global_result in global_results
                               if global_result.number_modes == number_mode]
            one_bit_errorlabel = (bit_error_rates, f"mode_{number_mode}")
            bit_error_labels.append(one_bit_errorlabel)

        self._plot_computed_variables(save_plot,
                                      fig,
                                      axes,
                                      bit_error_labels,
                                      "Bit Error Rates Results",
                                      'Bit Error Rates',
                                      "_bits")

    def times(self,
              number_modes: List[int],
              global_results: List[GlobalResult],
              save_plot: Optional[bool] = False) -> None:
        fig, axes = plt.subplots(figsize=[10, 8])
        times_labels = []

        for number_mode in number_modes:
            times = [global_result.time_in_seconds
                     for global_result in global_results
                     if global_result.number_modes == number_mode]
            one_time_label = (times, f"mode_{number_mode}")
            times_labels.append(one_time_label)

        self._plot_computed_variables(save_plot,
                                      fig,
                                      axes,
                                      times_labels,
                                      "Computation Time Results",
                                      'Computation Time (seconds)',
                                      "_times")

    def plot_success_probabilities(self,
                                   executions: Optional[List[ResultExecution]] = None,
                                   save_plot: Optional[bool] = False) -> None:
        fig, axes = plt.subplots(figsize=[10, 8])

        executions_probs_labels = [self._ideal_probabilities.p_homos,
                                   self._ideal_probabilities.p_ken_op,
                                   self._ideal_probabilities.p_hels]

        if executions is not None:
            executions_probs_labels += [(execution['p_succ'], execution['plot_label'])
                                        for execution in executions]
            self._plot_title(execution=executions[0])

        self._plot_probs_and_label_into_axis(axes=axes,
                                             probs_labels=executions_probs_labels)
        plt.legend()
        # plt.suptitle('Simulation results', fontsize=24, y=1)
        plt.xlabel('alpha values')
        plt.ylabel('Average Success Probabilities')

        self._show_or_save_plot(save_plot, fig)

    def _show_or_save_plot(self, save_plot, fig, suffix=None):
        if not save_plot:
            plt.show()

        # save plot to file
        if save_plot:
            fixed_path = _fix_path('results')
            fig.savefig(f'{fixed_path}plot_{set_current_time()}{suffix}.png')

    def _plot_title(self, execution: ResultExecution) -> None:
        total_time = f"\n Total time: {set_friendly_time(execution['total_time']) if 'total_time' in execution else ''}"
        alpha_time = set_friendly_time(execution['total_time'] /
                                       len(execution['alphas'])) if 'total_time' in execution else ''
        total_time_per_alpha = f"\n Average one alpha computation time: {alpha_time}"
        plt.title(f"{execution['plot_title']}{total_time}{total_time_per_alpha}")

    def _plot_probs_and_label_into_axis(self,
                                        axes: plt.Axes,
                                        probs_labels: List[Tuple[List[float], str]]):
        for prob, label in probs_labels:
            axes.plot(self._alphas, prob, label=label)
