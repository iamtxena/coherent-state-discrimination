from csd.batch import Batch
# from csd.codeword import CodeWord
from csd.codebooks import CodeBooks
import random


def test_linear_codes():
    batch = Batch(size=0, word_size=3, alpha_value=0.6, random_words=False)
    cb = CodeBooks(batch=batch)
    print(cb.codebooks)
    print(cb.binary_codes)
    codeword = random.choice(random.choice(cb.codebooks))
    print(f'ONE CODEWORD: {codeword}')
    print(f'BINARY CODEWORD: {codeword.binary_code}')


if __name__ == '__main__':
    test_linear_codes()
