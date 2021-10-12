# plot.py

from abc import ABC
from typing import List, Optional, Tuple
import matplotlib.pyplot as plt
from typeguard import typechecked
from csd.ideal_probabilities import IdealProbabilities
from csd.typings import ResultExecution


class Plot(ABC):
    """ Class for plotting the results

    """

    @typechecked
    def __init__(self, alphas: List[float] = None):
        if alphas is None:
            raise ValueError("alphas not set.")
        self._ideal_probabilities = IdealProbabilities(alphas=alphas)
        self._alphas = alphas

    def plot_success_probabilities(self,
                                   executions: Optional[List[ResultExecution]] = None) -> None:
        _, axes = plt.subplots(figsize=[10, 7])

        executions_probs_labels = [self._ideal_probabilities.p_homos,
                                   self._ideal_probabilities.p_ken_op,
                                   self._ideal_probabilities.p_hels]

        if executions is not None:
            executions_probs_labels += [(execution['p_succ'], execution['plot_label'])
                                        for execution in executions]

        self._plot_probs_and_label_into_axis(axes=axes,
                                             probs_labels=executions_probs_labels)
        plt.legend(bbox_to_anchor=(1, 1), loc="upper left")
        plt.show()

    def _plot_probs_and_label_into_axis(self,
                                        axes: plt.Axes,
                                        probs_labels: List[Tuple[List[float], str]]):
        for prob, label in probs_labels:
            axes.plot(self._alphas, prob, label=label)
