
from abc import ABC
import json
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

TRAINING_RESULTS = "training_results"
TESTING_RESULTS = "testing_results"
TRAINING_RESULTS_PATH = f'{TRAINING_RESULTS}/globals/'
TESTING_RESULTS_PATH = f'{TESTING_RESULTS}/globals/'
# RESULTS_PATH = "results/globals/"
ALPHAS_PATH = "alphas/"
RESULTS_FILENAME = "global_results"


class GlobalResultManager(ABC):

    def __init__(self, testing: bool = True):
        self._global_results_path = f'{TESTING_RESULTS_PATH}' if testing else f'{TRAINING_RESULTS_PATH}'
        self._base_dir_path = f'{TESTING_RESULTS}' if testing else f'{TRAINING_RESULTS}'
        self._global_results_file = self._check_if_file_exists()

    def _check_if_file_exists(self, global_result: Union[GlobalResult, None] = None) -> str:
        global_results_path = self._global_results_path
        results_file = self._global_results_path

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
                                             number_ancillas=0,
                                             helstrom_probability=0.0,
                                             homodyne_probability=0.0,
                                             best_success_probability=0.0,
                                             best_helstrom_probability=0.0,
                                             best_homodyne_probability=0.0,
                                             best_codebook=[],
                                             best_measurements=[],
                                             best_optimized_parameters=[]).header())
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
        results_path = self._global_results_path

        for alpha_file in glob.glob(f"{results_path}{ALPHAS_PATH}*.csv"):
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
                                         number_ancillas=0,
                                         helstrom_probability=0.0,
                                         homodyne_probability=0.0,
                                         best_success_probability=0.0,
                                         best_helstrom_probability=0.0,
                                         best_homodyne_probability=0.0,
                                         best_codebook=[],
                                         best_measurements=[],
                                         best_optimized_parameters=[]).header())
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
                                          number_ancillas=int(row[7] if len(row) >= 8 else 0),
                                          helstrom_probability=float(row[8] if len(row) >= 9 else 0.0),
                                          homodyne_probability=float(row[9] if len(row) >= 10 else 0.0),
                                          best_success_probability=float(row[10] if len(row) >= 11 else 0.0),
                                          best_helstrom_probability=float(row[11] if len(row) >= 12 else 0.0),
                                          best_homodyne_probability=float(row[12] if len(row) >= 13 else 0.0),
                                          best_codebook=json.loads(row[13]) if len(row) >= 14 else [],
                                          best_measurements=json.loads(row[14]) if len(row) >= 15 else [],
                                          best_optimized_parameters=json.loads(row[15]) if len(row) >= 16 else [])
                             for row in reader]
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

            self._global_results = [GlobalResult(
                alpha=float(row[0]),
                success_probability=float(row[1]),
                number_modes=int(row[2]),
                time_in_seconds=float(row[3]),
                squeezing=strtobool(row[6]),
                number_ancillas=int(row[7]),
                helstrom_probability=float(row[8]),
                homodyne_probability=float(row[9]),
                best_success_probability=float(row[10] if len(row) >= 11 else 0.0),
                best_helstrom_probability=float(row[11] if len(row) >= 12 else 0.0),
                best_homodyne_probability=float(row[12] if len(row) >= 13 else 0.0),
                best_codebook=json.loads(row[13]) if len(row) >= 14 else [],
                best_measurements=json.loads(row[14]) if len(row) >= 15 else [],
                best_optimized_parameters=json.loads(row[15]) if len(row) >= 16 else [])
                for row in reader]

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
                                   interactive_plot: bool = False,
                                   best_codebook: Optional[bool] = False) -> None:
        if (not hasattr(self, '_selected_global_results') or
            not hasattr(self, '_number_modes') or
                not hasattr(self, '_alphas')):
            self.load_results()
        return Plot(alphas=self._alphas, path=self._base_dir_path).success_probabilities(
            number_modes=self._number_modes,
            number_ancillas=self._number_ancillas,
            global_results=self._selected_global_results,
            save_plot=save_plot,
            interactive_plot=interactive_plot,
            best_codebook=best_codebook)

    def plot_distances(self, save_plot=False,
                       interactive_plot: bool = False) -> None:
        if (not hasattr(self, '_selected_global_results') or
            not hasattr(self, '_number_modes') or
                not hasattr(self, '_alphas')):
            self.load_results()
        Plot(alphas=self._alphas, path=self._base_dir_path).distances(
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
        Plot(alphas=self._alphas, path=self._base_dir_path).bit_error_rates(
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
        Plot(alphas=self._alphas, path=self._base_dir_path).times(
            number_modes=self._number_modes,
            number_ancillas=self._number_ancillas,
            global_results=self._selected_global_results,
            save_plot=save_plot,
            interactive_plot=interactive_plot)

    def plot_modes_probs(self,
                         one_alpha: Optional[Union[float, None]] = None,
                         save_plot=False,
                         apply_log=False,
                         interactive_plot: bool = False,
                         squeezing: bool = True,
                         non_squeezing: bool = False,
                         best_codebook: Optional[bool] = False) -> None:
        if (not hasattr(self, '_selected_global_results') or
            not hasattr(self, '_number_modes') or
                not hasattr(self, '_alphas')):
            self.load_results()

        if one_alpha is not None:
            Plot(alphas=[one_alpha], path=self._base_dir_path).success_probabilities_one_alpha(
                one_alpha=one_alpha,
                number_modes=self._number_modes,
                number_ancillas=self._number_ancillas,
                global_results=(self._selected_global_results if not apply_log else self._selected_log_global_results),
                save_plot=save_plot,
                apply_log=apply_log,
                interactive_plot=interactive_plot,
                best_codebook=best_codebook)
            return

        Plot(alphas=self._alphas, path=self._base_dir_path).success_probabilities_all_alphas(
            number_modes=self._number_modes,
            number_ancillas=self._number_ancillas,
            global_results=(self._selected_global_results if not apply_log else self._selected_log_global_results),
            save_plot=save_plot,
            apply_log=apply_log,
            interactive_plot=interactive_plot,
            squeezing=squeezing,
            non_squeezing=non_squeezing,
            best_codebook=best_codebook)

    def _select_global_results(self) -> None:
        self._selected_global_results = []
        self._selected_log_global_results = []
        squeezing_options = [False, True]
        for alpha in self._alphas:
            for number_mode in self._number_modes:
                for number_ancilla in self._number_ancillas:
                    for squeezing_option in squeezing_options:
                        max_distance = -1.0
                        max_result = None
                        for result in self._global_results:
                            if (result.alpha == alpha and
                                result.number_modes == number_mode and
                                result.number_ancillas == number_ancilla and
                                result.squeezing == squeezing_option and
                                    result.distance_to_helstrom_probability > max_distance):
                                max_result = result
                        if max_result is not None:
                            self._selected_global_results.append(max_result)
                            self._selected_log_global_results.append(
                                GlobalResult(alpha=max_result.alpha,
                                             success_probability=np.log(max_result.success_probability),
                                             number_modes=max_result.number_modes,
                                             time_in_seconds=max_result.time_in_seconds,
                                             squeezing=max_result.squeezing,
                                             number_ancillas=max_result.number_ancillas,
                                             helstrom_probability=max_result.helstrom_probability,
                                             homodyne_probability=max_result.homodyne_probability,
                                             best_success_probability=max_result.best_success_probability,
                                             best_helstrom_probability=max_result.best_helstrom_probability,
                                             best_homodyne_probability=max_result.best_homodyne_probability,
                                             best_codebook=max_result.best_codebook,
                                             best_measurements=max_result.best_measurements,
                                             best_optimized_parameters=max_result.best_optimized_parameters))
