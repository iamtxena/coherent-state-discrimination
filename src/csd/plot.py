# plot.py

from abc import ABC
from typing import List, Optional, Tuple
import matplotlib.pyplot as plt
# from typeguard import typechecked
from csd.ideal_probabilities import IdealHomodyneProbability, IdealProbabilities
from csd.typings.global_result import GlobalResult
from csd.typings.typing import ResultExecution
from csd.util import set_current_time, _fix_path, set_friendly_time
import numpy as np
# from csd.config import logger


class Plot(ABC):
    """ Class for plotting the results

    """

    # @typechecked
    def __init__(self, alphas: List[float] = None, number_modes: int = 1):
        if alphas is None:
            raise ValueError("alphas not set.")
        self._ideal_probabilities = IdealProbabilities(alphas=alphas, number_modes=number_modes)
        self._alphas = alphas

    def success_probabilities_all_alphas(self,
                                         number_modes: List[int],
                                         number_ancillas: List[int],
                                         global_results: List[GlobalResult],
                                         save_plot: Optional[bool] = False) -> None:
        fig = plt.figure(figsize=(25, 20))
        fig.suptitle("Average Success Probability", fontsize=20)

        for idx, one_alpha in enumerate(self._alphas):
            # if idx == 9:
            #     break
            homodyne_probabilities = []
            squeezed_probabilities = []
            non_squeezed_probabilities = []

            probs_labels = []
            for number_mode in number_modes:
                homodyne_probabilities.append(IdealHomodyneProbability(
                    alpha=one_alpha, number_modes=number_mode).homodyne_probability)
                squeezed_probabilities_mode_i = []
                non_squeezed_probabilities_mode_i = []
                for ancilla_i in number_ancillas:
                    squeezed_probability_ancilla_i = [global_result.success_probability
                                                      for global_result in global_results
                                                      if (global_result.number_modes == number_mode and
                                                          global_result.number_ancillas == ancilla_i and
                                                          global_result.squeezing and
                                                          global_result.alpha == one_alpha)]
                    non_squeezed_probability_ancilla_i = [global_result.success_probability
                                                          for global_result in global_results
                                                          if (global_result.number_modes == number_mode and
                                                              global_result.number_ancillas == ancilla_i and
                                                              not global_result.squeezing and
                                                              global_result.alpha == one_alpha)]
                    if len(squeezed_probability_ancilla_i) > 1:
                        raise ValueError("more than one squeezed_probability found!")
                    if len(non_squeezed_probability_ancilla_i) > 1:
                        raise ValueError("more than one non_squeezed_probability found!")
                    if len(squeezed_probability_ancilla_i) == 0:
                        squeezed_probability_ancilla_i.append(0.0)
                    squeezed_probabilities_mode_i.append(squeezed_probability_ancilla_i.pop(0))
                    if len(non_squeezed_probability_ancilla_i) == 0:
                        non_squeezed_probability_ancilla_i.append(0.0)
                    non_squeezed_probabilities_mode_i.append(non_squeezed_probability_ancilla_i.pop(0))
                squeezed_probabilities.append(squeezed_probabilities_mode_i)
                non_squeezed_probabilities.append(non_squeezed_probabilities_mode_i)

            for ancilla_i in number_ancillas:
                sq_prob_ancilla_i = [sq_prob.pop(0) for sq_prob in squeezed_probabilities]
                probs_labels.append((sq_prob_ancilla_i, f"pSucc Squeez anc:{ancilla_i}"))
                non_sq_prob_ancilla_i = [non_sq_prob.pop(0) for non_sq_prob in non_squeezed_probabilities]
                probs_labels.append((non_sq_prob_ancilla_i, f"pSucc No Squeez anc:{ancilla_i}"))
            probs_labels.append((homodyne_probabilities, "pSucc Homodyne"))

            ax = fig.add_subplot(4, 4, idx + 1 % 4)
            ax.set_ylim([0, 1])
            ax.set_title(f"$\\alpha$={np.round(one_alpha, 2)}", fontsize=14)

            for prob, label in probs_labels:
                ax.plot(number_modes, prob, label=label)

            ax.set_xticks(number_modes)
            ax.legend()
            ax.set_xlabel('number modes')
            ax.set_ylabel('Average Success Probabilities')
        plt.subplots_adjust(hspace=0.4)
        self._show_or_save_plot(save_plot, fig, "_probs_all")

    def success_probabilities_one_alpha(self,
                                        one_alpha: float,
                                        number_modes: List[int],
                                        number_ancillas: List[int],
                                        global_results: List[GlobalResult],
                                        save_plot: Optional[bool] = False) -> None:
        fig, axes = plt.subplots(figsize=[10, 8])
        homodyne_probabilities = []
        squeezed_probabilities = []
        non_squeezed_probabilities = []

        for number_mode in number_modes:
            homodyne_prob = IdealHomodyneProbability(alpha=one_alpha, number_modes=number_mode).homodyne_probability
            squeezed_probability = [global_result.success_probability
                                    for global_result in global_results
                                    if (global_result.number_modes == number_mode and
                                        global_result.squeezing and
                                        global_result.alpha == one_alpha)]
            non_squeezed_probability = [global_result.success_probability
                                        for global_result in global_results
                                        if (global_result.number_modes == number_mode and
                                            not global_result.squeezing and
                                            global_result.alpha == one_alpha)]
            if len(squeezed_probability) > 1:
                raise ValueError("more than one squeezed_probability found!")
            if len(non_squeezed_probability) > 1:
                raise ValueError("more than one non_squeezed_probability found!")
            homodyne_probabilities.append(homodyne_prob)
            if len(squeezed_probability) == 0:
                squeezed_probability.append(0.0)
            squeezed_probabilities.append(squeezed_probability.pop(0))
            if len(non_squeezed_probability) == 0:
                non_squeezed_probability.append(0.0)
            non_squeezed_probabilities.append(non_squeezed_probability.pop(0))

        probs_labels = ((squeezed_probabilities, "pSucc Squeezing"),
                        (non_squeezed_probabilities, "pSucc No Squeezing"),
                        (homodyne_probabilities, "pSucc Homodyne"))

        plt.title(f"Average Success Probability for alpha={one_alpha}", fontsize=24)

        for prob, label in probs_labels:
            axes.plot(number_modes, prob, label=label)

        axes.set_xticks(number_modes)
        plt.legend()
        plt.xlabel('number modes')
        plt.ylabel('Average Success Probabilities')
        self._show_or_save_plot(save_plot, fig, f"_probs_{str(np.round(one_alpha, 2))}")

    def success_probabilities(self,
                              number_modes: List[int],
                              number_ancillas: List[int],
                              global_results: List[GlobalResult],
                              save_plot: Optional[bool] = False) -> None:
        fig, axes = plt.subplots(figsize=[10, 8])
        probs_labels = [self._ideal_probabilities.p_hels]
        squeezing_options = [False, True]
        for number_mode in number_modes:
            for squeezing_option in squeezing_options:
                for number_ancilla in number_ancillas:
                    probs = [global_result.success_probability
                             for global_result in global_results
                             if (global_result.number_modes == number_mode and
                                 global_result.number_ancillas == number_ancilla and
                                 global_result.squeezing == squeezing_option)]

                    probs.extend([0.0] * (len(self._alphas) - len(probs)))
                    if len(probs) > len(self._alphas):
                        raise ValueError(f"len(probs): {len(probs)}")
                    one_prob_label = (probs, f"mode_{number_mode} squeez:{squeezing_option} anc:{number_ancilla}")
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
                  number_ancillas: List[int],
                  global_results: List[GlobalResult],
                  save_plot: Optional[bool] = False) -> None:
        fig, axes = plt.subplots(figsize=[10, 8])
        distances_labels = []
        squeezing_options = [False, True]

        for number_mode in number_modes:
            for squeezing_option in squeezing_options:
                for number_ancilla in number_ancillas:
                    distances = [global_result.distance_to_homodyne_probability
                                 for global_result in global_results
                                 if (global_result.number_modes == number_mode and
                                     global_result.number_ancillas == number_ancilla and
                                     global_result.squeezing == squeezing_option)]
                    one_distance_label = (
                        distances, f"mode_{number_mode} squeez:{squeezing_option} anc:{number_ancilla}")
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
                        number_ancillas: List[int],
                        global_results: List[GlobalResult],
                        save_plot: Optional[bool] = False) -> None:
        fig, axes = plt.subplots(figsize=[10, 8])
        bit_error_labels = []
        squeezing_options = [False, True]

        for number_mode in number_modes:
            for squeezing_option in squeezing_options:
                for number_ancilla in number_ancillas:
                    bit_error_rates = [global_result.bit_error_rate
                                       for global_result in global_results
                                       if (global_result.number_modes == number_mode and
                                           global_result.number_ancillas == number_ancilla and
                                           global_result.squeezing == squeezing_option)]
                    one_bit_errorlabel = (
                        bit_error_rates, f"mode_{number_mode} squeez:{squeezing_option} anc:{number_ancilla}")
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
              number_ancillas: List[int],
              global_results: List[GlobalResult],
              save_plot: Optional[bool] = False) -> None:
        fig, axes = plt.subplots(figsize=[10, 8])
        times_labels = []
        squeezing_options = [False, True]

        for number_mode in number_modes:
            for squeezing_option in squeezing_options:
                for number_ancilla in number_ancillas:
                    times = [global_result.time_in_seconds
                             for global_result in global_results
                             if (global_result.number_modes == number_mode and
                                 global_result.number_ancillas == number_ancilla and
                                 global_result.squeezing == squeezing_option)]
                    one_time_label = (times, f"mode_{number_mode} squeez:{squeezing_option} anc:{number_ancilla}")
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

    def _show_or_save_plot(self, save_plot, fig, suffix=''):
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
        plt.title(f"{execution['plot_title']}{total_time}{total_time_per_alpha}", fontsize=8)

    def _plot_probs_and_label_into_axis(self,
                                        axes: plt.Axes,
                                        probs_labels: List[Tuple[List[float], str]]):
        for prob, label in probs_labels:
            prob.extend([0.0] * (len(self._alphas) - len(prob)))
            axes.plot(self._alphas, prob, label=label)
