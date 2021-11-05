
from abc import ABC
import os
from pathlib import Path
import csv
from typing import Optional, Union
import numpy as np
import glob
from csd.plot import Plot

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
                                             squeezing=False).header())
        return results_file

    def write_result(self, global_result: GlobalResult) -> None:
        results_file = self._check_if_file_exists(global_result=global_result)

        with open(results_file, 'a+', newline='') as write_obj:
            # Create a writer object from csv module
            csv_writer = csv.writer(write_obj)
            # Add contents of list as last row in the csv file
            csv_writer.writerow(global_result.values())

    def _consolidate_results(self) -> None:
        global_results_file = self._reset_global_results_file()

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
                                         squeezing=False).header())
        return global_results_file

    def _transfer_alpha_results_to_global_file(self, global_results_file: str, alpha_file: str) -> None:
        with open(alpha_file, 'r') as f:
            reader = csv.reader(f)
            next(reader)

            alpha_results = [GlobalResult(alpha=float(row[0]),
                                          success_probability=float(row[1]),
                                          number_modes=int(row[2]),
                                          time_in_seconds=float(row[3]),
                                          squeezing=bool(row[6])) for row in reader]

            with open(global_results_file, 'a+', newline='') as write_obj:
                writer = csv.writer(write_obj)
                [writer.writerow(alpha_result.values()) for alpha_result in alpha_results]

    def load_results(self, consolidate_results: bool = False) -> None:
        if consolidate_results:
            self._consolidate_results()

        with open(self._global_results_file, 'r') as f:
            reader = csv.reader(f)
            next(reader)

            self._global_results = [GlobalResult(alpha=float(row[0]),
                                                 success_probability=float(row[1]),
                                                 number_modes=int(row[2]),
                                                 time_in_seconds=float(row[3]),
                                                 squeezing=bool(row[6])) for row in reader]
        self._create_unique_alphas()
        self._create_unique_modes()
        self._select_global_results()

    def _create_unique_alphas(self):
        self._alphas = [result.alpha for result in self._global_results]
        self._alphas = list(set(self._alphas))
        self._alphas.sort()

    def _create_unique_modes(self):
        self._number_modes = [result.number_modes for result in self._global_results]
        self._number_modes = list(set(self._number_modes))
        self._number_modes.sort()

    def plot_success_probabilities(self, save_plot=False) -> None:
        if (not hasattr(self, '_selected_global_results') or
            not hasattr(self, '_number_modes') or
                not hasattr(self, '_alphas')):
            self.load_results()
        Plot(alphas=self._alphas).success_probabilities(
            number_modes=self._number_modes,
            global_results=self._selected_global_results,
            save_plot=save_plot)

    def plot_distances(self, save_plot=False) -> None:
        if (not hasattr(self, '_selected_global_results') or
            not hasattr(self, '_number_modes') or
                not hasattr(self, '_alphas')):
            self.load_results()
        Plot(alphas=self._alphas).distances(
            number_modes=self._number_modes,
            global_results=self._selected_global_results,
            save_plot=save_plot)

    def plot_bit_error_rates(self, save_plot=False) -> None:
        if (not hasattr(self, '_selected_global_results') or
            not hasattr(self, '_number_modes') or
                not hasattr(self, '_alphas')):
            self.load_results()
        Plot(alphas=self._alphas).bit_error_rates(
            number_modes=self._number_modes,
            global_results=self._selected_global_results,
            save_plot=save_plot)

    def plot_computation_times(self, save_plot=False) -> None:
        if (not hasattr(self, '_selected_global_results') or
            not hasattr(self, '_number_modes') or
                not hasattr(self, '_alphas')):
            self.load_results()
        Plot(alphas=self._alphas).times(
            number_modes=self._number_modes,
            global_results=self._selected_global_results,
            save_plot=save_plot)

    def plot_modes_probs(self, one_alpha: Optional[Union[float, None]] = None, save_plot=False) -> None:
        if (not hasattr(self, '_selected_global_results') or
            not hasattr(self, '_number_modes') or
                not hasattr(self, '_alphas')):
            self.load_results()

        if one_alpha is not None:
            Plot(alphas=[one_alpha]).success_probabilities_one_alpha(
                one_alpha=one_alpha,
                number_modes=self._number_modes,
                global_results=self._selected_global_results,
                save_plot=save_plot)
            return

        Plot(alphas=self._alphas).success_probabilities_all_alphas(
            number_modes=self._number_modes,
            global_results=self._selected_global_results,
            save_plot=save_plot)

    def _select_global_results(self) -> None:
        self._selected_global_results = []
        for alpha in self._alphas:
            for number_mode in self._number_modes:
                max_duration = 0.0
                max_result = None
                for result in self._global_results:
                    if (result.alpha == alpha and
                        result.number_modes == number_mode and
                            result.time_in_seconds > max_duration):
                        max_result = result
                if max_result is None:
                    raise ValueError(
                        f"Max result not found for this alpha: {alpha} and mode: {number_mode}")
                self._selected_global_results.append(max_result)
