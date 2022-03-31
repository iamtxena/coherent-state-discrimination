from csd.codeword import CodeWord
from csd.batch import Batch
import numpy as np


def test_codewords(size: int, alpha_value: float = None):
    if alpha_value is None:
        codeword = CodeWord(size)
    codeword = CodeWord(size, alpha_value)
    print(codeword.word)
    print(codeword.alpha)


def test_batches(size: int, word_size: int, alpha_value: float = None):
    if alpha_value is None:
        a_batch = Batch(size, word_size)
    a_batch = Batch(size, word_size, alpha_value)
    print(a_batch.batch)
    print(a_batch.alpha)


if __name__ == '__main__':
    alphas = list(np.arange(0.05, 1.55, 0.05))
    test_codewords(size=10, alpha_value=0.99)
    test_codewords(size=3)
    [test_codewords(size=5, alpha_value=alpha) for alpha in alphas]

    test_batches(size=10, word_size=1, alpha_value=0.99)
    test_batches(size=10, word_size=4)
    [test_batches(size=10, word_size=4, alpha_value=alpha) for alpha in alphas]
