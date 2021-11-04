
from abc import ABC
import os
from pathlib import Path
import csv
from typing import Union
import numpy as np

from csd.typings.global_result import GlobalResult

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
                                             time_in_seconds=1).header())
        return results_file

    def write_result(self, global_result: GlobalResult) -> None:
        results_file = self._check_if_file_exists(global_result=global_result)

        with open(results_file, 'a+', newline='') as write_obj:
            # Create a writer object from csv module
            csv_writer = csv.writer(write_obj)
            # Add contents of list as last row in the csv file
            csv_writer.writerow(global_result.values())
