from typing import Optional, Union
from csd.global_result_manager import GlobalResultManager


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


def plot_modes_probs(one_alpha: Optional[Union[float, None]] = None) -> None:
    GlobalResultManager().plot_modes_probs(one_alpha=one_alpha, save_plot=True)


if __name__ == '__main__':
    # consolidate_results = True
    # # _consolidate_results()
    # load_results(consolidate_results=consolidate_results)
    # plot_probabilities()
    # plot_distances()
    # plot_bit_error_rates()
    # plot_computation_times()
    # # plot_modes_probs(one_alpha=0.05)
    plot_modes_probs()
