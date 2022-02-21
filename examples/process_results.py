from typing import Optional, Union
from csd.global_result_manager import GlobalResultManager
import numpy as np


def _consolidate_results(testing: bool) -> None:
    GlobalResultManager(testing=testing)._consolidate_results()


def load_results(consolidate_results: bool, testing: bool) -> None:
    grm = GlobalResultManager(testing=testing)
    grm.load_results(consolidate_results=consolidate_results)
    # print(grm._global_results)


def plot_probabilities(testing: bool) -> None:
    GlobalResultManager(testing=testing).plot_success_probabilities(save_plot=True)


def plot_distances(testing: bool) -> None:
    GlobalResultManager(testing=testing).plot_distances(save_plot=True)


def plot_bit_error_rates(testing: bool) -> None:
    GlobalResultManager(testing=testing).plot_bit_error_rates(save_plot=True)


def plot_computation_times(testing: bool) -> None:
    GlobalResultManager(testing=testing).plot_computation_times(save_plot=True)


def plot_modes_probs(one_alpha: Optional[Union[float, None]] = None, apply_log=False, testing: bool = True) -> None:
    GlobalResultManager(testing=testing).plot_modes_probs(one_alpha=one_alpha, save_plot=True,
                                                          apply_log=apply_log)


if __name__ == '__main__':
    alpha_init = 0.1
    alpha_end = 1.4
    number_points_to_plot = 16
    alpha_step = (alpha_end - alpha_init) / number_points_to_plot
    alphas = list(np.arange(alpha_init, alpha_end, alpha_step))

    consolidate_results = True
    testing = False
    # # _consolidate_results(testing=testing)
    load_results(consolidate_results=consolidate_results, testing=testing)
    plot_probabilities(testing=testing)
    plot_distances(testing=testing)
    plot_bit_error_rates(testing=testing)
    plot_computation_times(testing=testing)
    # # plot_modes_probs(one_alpha=alphas[5], testing=testing)
    # # plot_modes_probs(one_alpha=alphas[5], apply_log=True, testing=testing)
    plot_modes_probs(testing=testing)
    plot_modes_probs(apply_log=True, testing=testing)
