from typing import List, Tuple, Union

from csd.batch import Batch
from csd.codeword import CodeWord
from csd.typings.typing import CodeWordSuccessProbability


def _find_success_probability_and_input_codeword_for_one_output_codeword(
    codewords_success_probabilities: List[CodeWordSuccessProbability], output_codeword: CodeWord
) -> Tuple[float, CodeWord]:
    findings = [
        (codeword.success_probability, codeword.input_codeword)
        for codeword in codewords_success_probabilities
        if codeword.output_codeword == output_codeword
    ]
    if len(findings) <= 0:
        raise ValueError("No output codeword found to get its success_probability")
    if len(findings) > 1:
        raise ValueError("More than one codewords found to get its success_probability")
    if len(findings[0]) != 2:
        raise ValueError("The selected result has more than two elements")
    return findings[0][0], findings[0][1]


def _find_max_success_probability_of_all_input_codewords_for_one_output_codeword(
    batch_success_probabilities: List[List[CodeWordSuccessProbability]], output_codeword: CodeWord
) -> Tuple[float, CodeWord]:
    if len(batch_success_probabilities) <= 0:
        raise ValueError("empty batch_success_probabilities")

    max_success_probability = 0.0
    associated_input_codeword: Union[None, CodeWord] = None

    for codewords_success_probabilities in batch_success_probabilities:
        success_probability, input_codeword = _find_success_probability_and_input_codeword_for_one_output_codeword(
            codewords_success_probabilities=codewords_success_probabilities, output_codeword=output_codeword
        )
        if success_probability > max_success_probability:
            max_success_probability = success_probability
            associated_input_codeword = input_codeword

    if associated_input_codeword is None:
        raise ValueError(f"No input codeword found for the max success probability: {max_success_probability}")
    return max_success_probability, associated_input_codeword


def compute_maximum_likelihood(
    batch_success_probabilities: List[List[CodeWordSuccessProbability]], output_batch: Batch
) -> List[CodeWordSuccessProbability]:
    """Compute the maximum likelihood of a given batch probabilities defined by
        a list of input codewords that for each one has a list of all possible output codewords
        with its success probability

    Args:
        batch_success_probabilities (List[List[CodeWordSuccessProbability]]): a list of input codewords
        that for each one has a list of all possible output codewords with its success probability
        output_batch (Batch): the output batch to know all possible output codewords

    Returns:
        List[CodeWordSuccessProbability]: A list of all possible output codewords and with each the
        associated input codeword with its max success probability
    """
    maximum_likelihood: List[CodeWordSuccessProbability] = []

    for output_codeword in output_batch.codewords:
        (max_success_probability, associated_input_codeword) = (
            _find_max_success_probability_of_all_input_codewords_for_one_output_codeword(
                batch_success_probabilities=batch_success_probabilities, output_codeword=output_codeword
            )
        )

        codeword_success_probability = CodeWordSuccessProbability(
            input_codeword=associated_input_codeword,
            guessed_codeword=output_codeword,
            output_codeword=output_codeword,
            success_probability=max_success_probability,
            mutual_information=None,
            counts=0,
        )

        maximum_likelihood.append(codeword_success_probability)
    return maximum_likelihood
