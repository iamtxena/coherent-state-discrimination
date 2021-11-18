
from abc import ABC
import os
from pathlib import Path
import csv
from typing import List, Optional, Union
import numpy as np
import glob
from csd.plot import Plot
from csd.util import strtobool

from csd.typings.global_result import GlobalResult
# from csd.config import logger

RESULTS_PATH = "results/globals/"
ALPHAS_PATH = "alphas/"
RESULTS_FILENAME = "global_results"


class GlobalResultManager(ABC):

    def __init__(self):
        self._global_results_file = self._check_if_file_exists()

    def _check_if_file_exists(self, global_result: Union[GlobalResult, None] = None) -> str:
        global_results_path = f'{RESULTS_PATH}'
        results_file = f'{RESULTS_PATH}'

        if global_result is None:
            results_file += RESULTS_FILENAME
        if global_result is not None:
            global_results_path += ALPHAS_PATH
            results_file += f'{ALPHAS_PATH}{RESULTS_FILENAME}{str(np.round(global_result.alpha, 2))}'

        results_file += '.csv'

        if not os.path.exists(results_file):
            Path(global_results_path).mkdir(parents=True, exist_ok=True)
            with open(results_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(GlobalResult(alpha=0.0,
                                             success_probability=0.0,
                                             number_modes=1,
                                             time_in_seconds=1,
                                             squeezing=False,
                                             number_ancillas=0).header())
        return results_file

    def write_result(self, global_result: GlobalResult) -> None:
        results_file = self._check_if_file_exists(global_result=global_result)

        with open(results_file, 'a+', newline='') as write_obj:
            # Create a writer object from csv module
            csv_writer = csv.writer(write_obj)
            # Add contents of list as last row in the csv file
            csv_writer.writerow(global_result.values())

    def _consolidate_results(self) -> None:
        # global_results_file = self._reset_global_results_file()
        global_results_file = self._check_if_file_exists()

        for alpha_file in glob.glob(f"{RESULTS_PATH}{ALPHAS_PATH}*.csv"):
            self._transfer_alpha_results_to_global_file(global_results_file, alpha_file)

    def _reset_global_results_file(self):
        global_results_file = self._check_if_file_exists()
        with open(global_results_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(GlobalResult(alpha=0.0,
                                         success_probability=0.0,
                                         number_modes=1,
                                         time_in_seconds=1,
                                         squeezing=False,
                                         number_ancillas=0).header())
        return global_results_file

    def _transfer_alpha_results_to_global_file(self, global_results_file: str, alpha_file: str) -> None:
        with open(alpha_file, 'r') as f:
            reader = csv.reader(f)
            next(reader)

            alpha_results = [GlobalResult(alpha=float(row[0]),
                                          success_probability=float(row[1]),
                                          number_modes=int(row[2]),
                                          time_in_seconds=float(row[3]),
                                          squeezing=strtobool(row[6] if len(row) >= 7 else 'False'),
                                          number_ancillas=int(row[7] if len(row) >= 8 else 0)) for row in reader]
            new_results = self._filter_only_new_results(loaded_results=alpha_results)
            with open(global_results_file, 'a+', newline='') as write_obj:
                writer = csv.writer(write_obj)
                [writer.writerow(new_result.values()) for new_result in new_results]

    def _filter_only_new_results(self, loaded_results: List[GlobalResult]) -> List[GlobalResult]:
        return [loaded_result for loaded_result in loaded_results if self._global_results.count(loaded_result) == 0]

    def load_results(self, consolidate_results: bool = False) -> None:
        self._load_all_results_from_global_file()
        if consolidate_results:
            self._consolidate_results()

        self._load_all_results_from_global_file()
        self._create_unique_alphas()
        self._create_unique_modes()
        self._create_unique_ancillas()
        self._select_global_results()

    def _load_all_results_from_global_file(self):
        with open(self._global_results_file, 'r') as f:
            reader = csv.reader(f)
            next(reader)

            self._global_results = [GlobalResult(alpha=float(row[0]),
                                                 success_probability=float(row[1]),
                                                 number_modes=int(row[2]),
                                                 time_in_seconds=float(row[3]),
                                                 squeezing=strtobool(row[6]),
                                                 number_ancillas=int(row[7])) for row in reader]

    def _create_unique_alphas(self):
        self._alphas = [result.alpha for result in self._global_results]
        self._alphas = list(set(self._alphas))
        self._alphas.sort()

    def _create_unique_modes(self):
        self._number_modes = [result.number_modes for result in self._global_results]
        self._number_modes = list(set(self._number_modes))
        self._number_modes.sort()

    def _create_unique_ancillas(self):
        self._number_ancillas = [result.number_ancillas for result in self._global_results]
        self._number_ancillas = list(set(self._number_ancillas))
        self._number_ancillas.sort()

    def plot_success_probabilities(self,
                                   save_plot: bool = False,
                                   interactive_plot: bool = False) -> None:
        if (not hasattr(self, '_selected_global_results') or
            not hasattr(self, '_number_modes') or
                not hasattr(self, '_alphas')):
            self.load_results()
        return Plot(alphas=self._alphas).success_probabilities(
            number_modes=self._number_modes,
            number_ancillas=self._number_ancillas,
            global_results=self._selected_global_results,
            save_plot=save_plot,
            interactive_plot=interactive_plot)

    def plot_distances(self, save_plot=False,
                       interactive_plot: bool = False) -> None:
        if (not hasattr(self, '_selected_global_results') or
            not hasattr(self, '_number_modes') or
                not hasattr(self, '_alphas')):
            self.load_results()
        Plot(alphas=self._alphas).distances(
            number_modes=self._number_modes,
            number_ancillas=self._number_ancillas,
            global_results=self._selected_global_results,
            save_plot=save_plot,
            interactive_plot=interactive_plot)

    def plot_bit_error_rates(self, save_plot=False,
                             interactive_plot: bool = False) -> None:
        if (not hasattr(self, '_selected_global_results') or
            not hasattr(self, '_number_modes') or
                not hasattr(self, '_alphas')):
            self.load_results()
        Plot(alphas=self._alphas).bit_error_rates(
            number_modes=self._number_modes,
            number_ancillas=self._number_ancillas,
            global_results=self._selected_global_results,
            save_plot=save_plot,
            interactive_plot=interactive_plot)

    def plot_computation_times(self, save_plot=False,
                               interactive_plot: bool = False) -> None:
        if (not hasattr(self, '_selected_global_results') or
            not hasattr(self, '_number_modes') or
                not hasattr(self, '_alphas')):
            self.load_results()
        Plot(alphas=self._alphas).times(
            number_modes=self._number_modes,
            number_ancillas=self._number_ancillas,
            global_results=self._selected_global_results,
            save_plot=save_plot,
            interactive_plot=interactive_plot)

    def plot_modes_probs(self,
                         one_alpha: Optional[Union[float, None]] = None,
                         save_plot=False,
                         apply_log=False,
                         interactive_plot: bool = False) -> None:
        if (not hasattr(self, '_selected_global_results') or
            not hasattr(self, '_number_modes') or
                not hasattr(self, '_alphas')):
            self.load_results()

        if one_alpha is not None:
            Plot(alphas=[one_alpha]).success_probabilities_one_alpha(
                one_alpha=one_alpha,
                number_modes=self._number_modes,
                number_ancillas=self._number_ancillas,
                global_results=(self._selected_global_results if not apply_log else self._selected_log_global_results),
                save_plot=save_plot,
                apply_log=apply_log,
                interactive_plot=interactive_plot)
            return

        Plot(alphas=self._alphas).success_probabilities_all_alphas(
            number_modes=self._number_modes,
            number_ancillas=self._number_ancillas,
            global_results=(self._selected_global_results if not apply_log else self._selected_log_global_results),
            save_plot=save_plot,
            apply_log=apply_log,
            interactive_plot=interactive_plot)

    def _select_global_results(self) -> None:
        self._selected_global_results = []
        self._selected_log_global_results = []
        squeezing_options = [False, True]
        for alpha in self._alphas:
            for number_mode in self._number_modes:
                for number_ancilla in self._number_ancillas:
                    for squeezing_option in squeezing_options:
                        min_distance = 1.0
                        min_result = None
                        for result in self._global_results:
                            if (result.alpha == alpha and
                                result.number_modes == number_mode and
                                result.number_ancillas == number_ancilla and
                                result.squeezing == squeezing_option and
                                    result.distance_to_homodyne_probability < min_distance):
                                min_result = result
                        if min_result is not None:
                            self._selected_global_results.append(min_result)
                            self._selected_log_global_results.append(
                                GlobalResult(alpha=min_result.alpha,
                                             success_probability=np.log(min_result.success_probability),
                                             number_modes=min_result.number_modes,
                                             time_in_seconds=min_result.time_in_seconds,
                                             squeezing=min_result.squeezing,
                                             number_ancillas=min_result.number_ancillas))
