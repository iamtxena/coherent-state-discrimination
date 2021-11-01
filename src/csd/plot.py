# plot.py

from abc import ABC
from typing import List, Optional, Tuple
import matplotlib.pyplot as plt
# from typeguard import typechecked
from csd.ideal_probabilities import IdealProbabilities
from csd.typings.typing import ResultExecution
from csd.util import set_current_time, _fix_path, set_friendly_time


class Plot(ABC):
    """ Class for plotting the results

    """

    # @typechecked
    def __init__(self, alphas: List[float] = None):
        if alphas is None:
            raise ValueError("alphas not set.")
        self._ideal_probabilities = IdealProbabilities(alphas=alphas)
        self._alphas = alphas

    def plot_success_probabilities(self,
                                   executions: Optional[List[ResultExecution]] = None,
                                   save_plot: Optional[bool] = False) -> None:
        fig, axes = plt.subplots(figsize=[10, 7])

        executions_probs_labels = [self._ideal_probabilities.p_homos,
                                   self._ideal_probabilities.p_ken_op,
                                   self._ideal_probabilities.p_hels]

        if executions is not None:
            executions_probs_labels += [(execution['p_succ'], execution['plot_label'])
                                        for execution in executions]
            self._plot_title(execution=executions[0])

        self._plot_probs_and_label_into_axis(axes=axes,
                                             probs_labels=executions_probs_labels)
        plt.legend(bbox_to_anchor=(1, 1), loc="upper left")
        # plt.suptitle('Simulation results', fontsize=24, y=1)
        plt.xlabel('alpha values')
        plt.ylabel('Average Success Probabilities')

        if not save_plot:
            plt.show()

        # save plot to file
        if save_plot:
            fixed_path = _fix_path('results')
            fig.savefig(f'{fixed_path}plot_{set_current_time()}.png')

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
