from csd.batch import Batch
# from csd.codeword import CodeWord
from csd.codebooks import CodeBooks


def test_linear_codes():
    batch = Batch(size=0, word_size=1, alpha_value=0.6, random_words=False)
    cb = CodeBooks(batch=batch)
    print(cb.codebooks)


if __name__ == '__main__':
    test_linear_codes()
