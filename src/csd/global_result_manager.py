
from abc import ABC
import os
import csv
from typing import Union
import numpy as np

from csd.typings.global_result import GlobalResult

RESULTS_PATH = "results/globals/"
RESULTS_FILE = "global_results.csv"


class GlobalResultManager(ABC):

    def __init__(self):
        self._results_file = self._check_if_file_exists()

    def _check_if_file_exists(self, global_result: Union[GlobalResult, None] = None) -> str:
        results_file = f'{RESULTS_PATH}{RESULTS_FILE}'
        if global_result is not None:
            results_file += str(np.round(global_result.alpha, 2))

        if not os.path.exists(results_file):
            with open('countries.csv', 'w', encoding='UTF8') as f:
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
