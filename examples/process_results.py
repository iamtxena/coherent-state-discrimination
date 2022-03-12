from typing import Optional, Union
from csd.global_result_manager import GlobalResultManager
import numpy as np


def _consolidate_results(testing: bool) -> None:
    GlobalResultManager(testing=testing)._consolidate_results()


def load_results(consolidate_results: bool, testing: bool) -> None:
    grm = GlobalResultManager(testing=testing)
    grm.load_results(consolidate_results=consolidate_results)
    # print(grm._global_results)


def plot_probabilities(testing: bool, best_codebook: bool) -> None:
    GlobalResultManager(testing=testing).plot_success_probabilities(save_plot=True, best_codebook=best_codebook)


def plot_distances(testing: bool) -> None:
    GlobalResultManager(testing=testing).plot_distances(save_plot=True)


def plot_bit_error_rates(testing: bool) -> None:
    GlobalResultManager(testing=testing).plot_bit_error_rates(save_plot=True)


def plot_computation_times(testing: bool) -> None:
    GlobalResultManager(testing=testing).plot_computation_times(save_plot=True)


def plot_modes_probs(best_codebook: bool,
                     squeezing: bool = True,
                     non_squeezing: bool = False,
                     one_alpha: Optional[Union[float, None]] = None,
                     apply_log=False, testing: bool = True) -> None:
    GlobalResultManager(testing=testing).plot_modes_probs(one_alpha=one_alpha, save_plot=True,
                                                          apply_log=apply_log, best_codebook=best_codebook,
                                                          squeezing=squeezing, non_squeezing=non_squeezing)


if __name__ == '__main__':
    alpha_init = 0.1
    alpha_end = 1.4
    number_points_to_plot = 16
    alpha_step = (alpha_end - alpha_init) / number_points_to_plot
    alphas = list(np.arange(alpha_init, alpha_end, alpha_step))

    consolidate_results = True
    testing_options = [False, True]
    best_codebook_options = [False, True]
    apply_log_options = [False, True]
    squeezing_options = [False, True]
    # # _consolidate_results(testing=testing)
    for testing in testing_options:
        load_results(consolidate_results=consolidate_results, testing=testing)

        plot_distances(testing=testing)
        plot_bit_error_rates(testing=testing)
        plot_computation_times(testing=testing)

        for best_codebook in best_codebook_options:
            plot_probabilities(testing=testing, best_codebook=best_codebook)

            for apply_log in apply_log_options:
                for squeezing_option in squeezing_options:
                    # # plot_modes_probs(one_alpha=alphas[5], testing=testing)
                    # # plot_modes_probs(one_alpha=alphas[5], apply_log=True, testing=testing)
                    plot_modes_probs(apply_log=apply_log,
                                     testing=testing,
                                     best_codebook=best_codebook,
                                     squeezing=squeezing_option,
                                     non_squeezing=not squeezing_option)
