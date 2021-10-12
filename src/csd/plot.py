# plot.py

from abc import ABC
from typing import List, Tuple
import matplotlib.pyplot as plt
from typeguard import typechecked
from csd.ideal_probabilities import IdealProbabilities
from csd.typings import ResultExecution


class Plot(ABC):
    """ Class for plotting the results

    """

    @typechecked
    def __init__(self, alphas: List[float], executions: List[ResultExecution]):
        if alphas is None:
            raise ValueError("alphas not set.")
        if executions is None:
            raise ValueError("executions not set.")
        self._ideal_probabilities = IdealProbabilities(alphas=alphas)
        self._alphas = alphas
        self._executions = executions

    def plot_success_probabilities(self) -> None:
        _, axes = plt.subplots(figsize=[10, 7])

        executions_probs_labels = [self._ideal_probabilities.p_homos,
                                   self._ideal_probabilities.p_ken_op,
                                   self._ideal_probabilities.p_hels]

        executions_probs_labels += [(execution['p_succ'], execution['plot_label']) for execution in self._executions]

        self._plot_probs_and_label_into_axis(axes=axes,
                                             probs_labels=executions_probs_labels)
        plt.legend(bbox_to_anchor=(1, 1), loc="upper left")
        plt.show()

    def _plot_probs_and_label_into_axis(self,
                                        axes: plt.Axes,
                                        probs_labels: List[Tuple[List[float], str]]):
        for prob, label in probs_labels:
            axes.plot(self._alphas, prob, label=label)
