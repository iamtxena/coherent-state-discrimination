import random
from typing import List, Union
from csd.batch import Batch
from csd.best_codebook import BestCodeBook
from csd.codebooks import CodeBooks
import numpy as np
import pytest
from csd.top5_best_codebooks import Top5_BestCodeBooks


def _get_alphas() -> List[float]:
    alpha_init = 0.1
    alpha_end = 1.4
    number_points_to_plot = 16
    alpha_step = (alpha_end - alpha_init) / number_points_to_plot
    return list(np.arange(alpha_init, alpha_end, alpha_step))


def _generate_codebooks(word_size: int = 3, alpha_value: float = 0.6) -> CodeBooks:
    batch = Batch(size=0, word_size=word_size, alpha_value=alpha_value, random_words=False)
    return CodeBooks(batch=batch)


def _generate_best_codebook(success_probability: float,
                            modes: int = 3,
                            alpha_value: float = 0.6,
                            helstrom_probability: float = 0.87,
                            homodyne_probability: float = 0.85
                            ) -> Union[None, BestCodeBook]:
    cbs = _generate_codebooks(word_size=modes, alpha_value=alpha_value)
    if cbs.size <= 0:
        return None
    cb = random.choice(cbs.codebooks)
    random_measurement = [[random.choice([0, 1]) for _ in range(codeword.size)] for codeword in cb]
    return BestCodeBook(codebook=cb,
                        measurement=random_measurement,
                        success_probability=success_probability,
                        helstrom_probability=helstrom_probability,
                        homodyne_probability=homodyne_probability)


alphas = _get_alphas()
modes = list(range(1, 8))


def test_top5_constructor():
    top5_cbs = Top5_BestCodeBooks()
    assert isinstance(top5_cbs, Top5_BestCodeBooks)
    assert top5_cbs.top5 == []
    assert top5_cbs.size == 0


def test_add_one_element():
    top5_cbs = Top5_BestCodeBooks()
    one_best_codebook = _generate_best_codebook(success_probability=0.4)
    top5_cbs.add(potential_best_codebook=one_best_codebook)
    assert top5_cbs.size == 1
    assert top5_cbs.first == one_best_codebook


def test_add_two_elements():
    top5_cbs = Top5_BestCodeBooks()
    one_best_codebook = _generate_best_codebook(success_probability=0.6)
    two_best_codebook = _generate_best_codebook(success_probability=0.4)
    top5_cbs.add(potential_best_codebook=one_best_codebook)
    top5_cbs.add(potential_best_codebook=two_best_codebook)
    assert top5_cbs.size == 2
    assert top5_cbs.first == one_best_codebook
    assert top5_cbs.second == two_best_codebook


def test_add_two_elements_different_order():
    top5_cbs = Top5_BestCodeBooks()
    one_best_codebook = _generate_best_codebook(success_probability=0.6)
    two_best_codebook = _generate_best_codebook(success_probability=0.4)
    top5_cbs.add(potential_best_codebook=two_best_codebook)
    top5_cbs.add(potential_best_codebook=one_best_codebook)
    assert top5_cbs.size == 2
    assert top5_cbs.first == one_best_codebook
    assert top5_cbs.second == two_best_codebook


def test_add_three_elements():
    top5_cbs = Top5_BestCodeBooks()
    one_best_codebook = _generate_best_codebook(success_probability=0.6)
    two_best_codebook = _generate_best_codebook(success_probability=0.4)
    three_best_codebook = _generate_best_codebook(success_probability=0.3)
    top5_cbs.add(potential_best_codebook=one_best_codebook)
    top5_cbs.add(potential_best_codebook=two_best_codebook)
    top5_cbs.add(potential_best_codebook=three_best_codebook)
    assert top5_cbs.size == 3
    assert top5_cbs.first == one_best_codebook
    assert top5_cbs.second == two_best_codebook
    assert top5_cbs.third == three_best_codebook


def test_add_three_elements_different_order():
    top5_cbs = Top5_BestCodeBooks()
    one_best_codebook = _generate_best_codebook(success_probability=0.6)
    two_best_codebook = _generate_best_codebook(success_probability=0.4)
    three_best_codebook = _generate_best_codebook(success_probability=0.3)
    top5_cbs.add(potential_best_codebook=three_best_codebook)
    top5_cbs.add(potential_best_codebook=two_best_codebook)
    top5_cbs.add(potential_best_codebook=one_best_codebook)
    assert top5_cbs.size == 3
    assert top5_cbs.first == one_best_codebook
    assert top5_cbs.second == two_best_codebook
    assert top5_cbs.third == three_best_codebook


def test_add_four_elements():
    top5_cbs = Top5_BestCodeBooks()
    one_best_codebook = _generate_best_codebook(success_probability=0.8)
    two_best_codebook = _generate_best_codebook(success_probability=0.7)
    three_best_codebook = _generate_best_codebook(success_probability=0.5)
    four_best_codebook = _generate_best_codebook(success_probability=0.4)
    top5_cbs.add(potential_best_codebook=one_best_codebook)
    top5_cbs.add(potential_best_codebook=two_best_codebook)
    top5_cbs.add(potential_best_codebook=three_best_codebook)
    top5_cbs.add(potential_best_codebook=four_best_codebook)
    assert top5_cbs.size == 4
    assert top5_cbs.first == one_best_codebook
    assert top5_cbs.second == two_best_codebook
    assert top5_cbs.third == three_best_codebook
    assert top5_cbs.fourth == four_best_codebook


def test_add_four_elements_different_order():
    top5_cbs = Top5_BestCodeBooks()
    one_best_codebook = _generate_best_codebook(success_probability=0.8)
    two_best_codebook = _generate_best_codebook(success_probability=0.7)
    three_best_codebook = _generate_best_codebook(success_probability=0.5)
    four_best_codebook = _generate_best_codebook(success_probability=0.4)
    top5_cbs.add(potential_best_codebook=four_best_codebook)
    top5_cbs.add(potential_best_codebook=three_best_codebook)
    top5_cbs.add(potential_best_codebook=two_best_codebook)
    top5_cbs.add(potential_best_codebook=one_best_codebook)
    assert top5_cbs.size == 4
    assert top5_cbs.first == one_best_codebook
    assert top5_cbs.second == two_best_codebook
    assert top5_cbs.third == three_best_codebook
    assert top5_cbs.fourth == four_best_codebook


def test_add_five_elements():
    top5_cbs = Top5_BestCodeBooks()
    one_best_codebook = _generate_best_codebook(success_probability=0.8)
    two_best_codebook = _generate_best_codebook(success_probability=0.7)
    three_best_codebook = _generate_best_codebook(success_probability=0.5)
    four_best_codebook = _generate_best_codebook(success_probability=0.4)
    five_best_codebook = _generate_best_codebook(success_probability=0.3)
    top5_cbs.add(potential_best_codebook=one_best_codebook)
    top5_cbs.add(potential_best_codebook=two_best_codebook)
    top5_cbs.add(potential_best_codebook=three_best_codebook)
    top5_cbs.add(potential_best_codebook=four_best_codebook)
    top5_cbs.add(potential_best_codebook=five_best_codebook)
    assert top5_cbs.size == 5
    assert top5_cbs.first == one_best_codebook
    assert top5_cbs.second == two_best_codebook
    assert top5_cbs.third == three_best_codebook
    assert top5_cbs.fourth == four_best_codebook
    assert top5_cbs.fifth == five_best_codebook


def test_add_five_elements_different_order():
    top5_cbs = Top5_BestCodeBooks()
    one_best_codebook = _generate_best_codebook(success_probability=0.8)
    two_best_codebook = _generate_best_codebook(success_probability=0.7)
    three_best_codebook = _generate_best_codebook(success_probability=0.5)
    four_best_codebook = _generate_best_codebook(success_probability=0.4)
    five_best_codebook = _generate_best_codebook(success_probability=0.3)
    top5_cbs.add(potential_best_codebook=five_best_codebook)
    top5_cbs.add(potential_best_codebook=four_best_codebook)
    top5_cbs.add(potential_best_codebook=three_best_codebook)
    top5_cbs.add(potential_best_codebook=two_best_codebook)
    top5_cbs.add(potential_best_codebook=one_best_codebook)
    assert top5_cbs.size == 5
    assert top5_cbs.first == one_best_codebook
    assert top5_cbs.second == two_best_codebook
    assert top5_cbs.third == three_best_codebook
    assert top5_cbs.fourth == four_best_codebook
    assert top5_cbs.fifth == five_best_codebook


def test_add_six_elements():
    top5_cbs = Top5_BestCodeBooks()
    one_best_codebook = _generate_best_codebook(success_probability=0.8)
    two_best_codebook = _generate_best_codebook(success_probability=0.7)
    three_best_codebook = _generate_best_codebook(success_probability=0.5)
    four_best_codebook = _generate_best_codebook(success_probability=0.4)
    five_best_codebook = _generate_best_codebook(success_probability=0.3)
    six_best_codebook = _generate_best_codebook(success_probability=0.9)
    top5_cbs.add(potential_best_codebook=one_best_codebook)
    top5_cbs.add(potential_best_codebook=two_best_codebook)
    top5_cbs.add(potential_best_codebook=three_best_codebook)
    top5_cbs.add(potential_best_codebook=four_best_codebook)
    top5_cbs.add(potential_best_codebook=five_best_codebook)
    top5_cbs.add(potential_best_codebook=six_best_codebook)
    assert top5_cbs.size == 5
    assert top5_cbs.first == six_best_codebook
    assert top5_cbs.second == one_best_codebook
    assert top5_cbs.third == two_best_codebook
    assert top5_cbs.fourth == three_best_codebook
    assert top5_cbs.fifth == four_best_codebook


def test_add_10_best_codebooks_one_alpha_one_mode():
    alpha = 0.75
    mode = 3
    number_codebooks = 10
    top5_cbs = Top5_BestCodeBooks()
    for _ in range(number_codebooks):
        top5_cbs.add(_generate_best_codebook(success_probability=random.uniform(0, 1),
                                             modes=mode,
                                             alpha_value=alpha))

    assert top5_cbs.size == 5
    assert top5_cbs.first.success_probability > top5_cbs.second.success_probability
    assert top5_cbs.second.success_probability > top5_cbs.third.success_probability
    assert top5_cbs.third.success_probability > top5_cbs.fourth.success_probability
    assert top5_cbs.fourth.success_probability > top5_cbs.fifth.success_probability


def test_add_10_best_codebooks_alpha_01_mode_1():
    alpha = 0.1
    mode = 1
    number_codebooks = 10
    top5_cbs = Top5_BestCodeBooks()
    generated_codebooks = 0
    for _ in range(number_codebooks):
        cb = _generate_best_codebook(success_probability=random.uniform(0, 1),
                                     modes=mode,
                                     alpha_value=alpha)
        if cb is None:
            continue
        top5_cbs.add(potential_best_codebook=cb)
        generated_codebooks += 1

    assert generated_codebooks == 0
    assert top5_cbs.top5 == []
    assert top5_cbs.size == 0
    if generated_codebooks >= 5:
        assert top5_cbs.size == 5
    if generated_codebooks < 5:
        assert top5_cbs.size == generated_codebooks


@pytest.mark.parametrize("alpha", alphas)
@pytest.mark.parametrize("mode", modes)
def test_add_10_best_codebooks(alpha: float, mode: int):
    number_codebooks = 10
    top5_cbs = Top5_BestCodeBooks()
    generated_codebooks = 0
    for _ in range(number_codebooks):
        cb = _generate_best_codebook(success_probability=random.uniform(0, 1),
                                     modes=mode,
                                     alpha_value=alpha)
        if cb is None:
            continue
        top5_cbs.add(potential_best_codebook=cb)
        generated_codebooks += 1

    if generated_codebooks == 0:
        assert top5_cbs.top5 == []
        assert top5_cbs.size == 0
    if generated_codebooks >= 5:
        assert top5_cbs.size == 5
    if generated_codebooks < 5:
        assert top5_cbs.size == generated_codebooks
    if top5_cbs.size > 0:
        assert top5_cbs.first.success_probability > top5_cbs.second.success_probability
    if top5_cbs.size > 1:
        assert top5_cbs.second.success_probability > top5_cbs.third.success_probability
    if top5_cbs.size > 2:
        assert top5_cbs.third.success_probability > top5_cbs.fourth.success_probability
    if top5_cbs.size > 3:
        assert top5_cbs.fourth.success_probability > top5_cbs.fifth.success_probability
