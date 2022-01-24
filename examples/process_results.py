from typing import Optional, Union
from csd.global_result_manager import GlobalResultManager
import numpy as np


def _consolidate_results() -> None:
    GlobalResultManager()._consolidate_results()


def load_results(consolidate_results: bool) -> None:
    grm = GlobalResultManager()
    grm.load_results(consolidate_results=consolidate_results)
    # print(grm._global_results)


def plot_probabilities() -> None:
    GlobalResultManager().plot_success_probabilities(save_plot=True)


def plot_distances() -> None:
    GlobalResultManager().plot_distances(save_plot=True)


def plot_bit_error_rates() -> None:
    GlobalResultManager().plot_bit_error_rates(save_plot=True)


def plot_computation_times() -> None:
    GlobalResultManager().plot_computation_times(save_plot=True)


def plot_modes_probs(one_alpha: Optional[Union[float, None]] = None, apply_log=False) -> None:
    GlobalResultManager().plot_modes_probs(one_alpha=one_alpha, save_plot=True,
                                           apply_log=apply_log)


if __name__ == '__main__':
    alpha_init = 0.1
    alpha_end = 1.4
    number_points_to_plot = 16
    alpha_step = (alpha_end - alpha_init) / number_points_to_plot
    alphas = list(np.arange(alpha_init, alpha_end, alpha_step))

    consolidate_results = True
    # # _consolidate_results()
    load_results(consolidate_results=consolidate_results)
    # plot_probabilities()
    # plot_distances()
    # plot_bit_error_rates()
    # plot_computation_times()
    # plot_modes_probs(one_alpha=alphas[5])
    # plot_modes_probs(one_alpha=alphas[5], apply_log=True)
    plot_modes_probs()
    plot_modes_probs(apply_log=True)
