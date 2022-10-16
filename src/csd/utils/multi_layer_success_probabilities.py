""" Multi Layer Circuit Success Probabilities util functions """

from dataclasses import dataclass

from csd.batch import Batch
from csd.typings.typing import CodeWordSuccessProbability


@dataclass
class SecondLayerSplitProbabilities:
    """Second Layer Split Probabilities"""

    first_mode_zero_probabilities: list[CodeWordSuccessProbability]
    first_mode_zero_input_batch: Batch
    first_mode_one_probabilities: list[CodeWordSuccessProbability]
    first_mode_one_input_batch: Batch


def split_success_probabilities_based_on_first_mode_result(
    batch_success_probabilities: list[list[CodeWordSuccessProbability]],
) -> SecondLayerSplitProbabilities:
    """Split Success Probabilities based on first mode result

    Args:
        batch_success_probabilities (list[list[CodeWordSuccessProbability]]): _description_

    Returns:
        SecondLayerSplitProbabilities: _description_
    """
    raise NotImplementedError
