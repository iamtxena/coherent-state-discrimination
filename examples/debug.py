from csd.utils.util import load_object_from_file
from csd.utils.probability import compute_maximum_likelihood
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

batch_success_prob = load_object_from_file('batch_success_prob')
output_batch = load_object_from_file('output_batch')
guessed_codewords_probabilities = compute_maximum_likelihood(batch_success_probabilities=batch_success_prob,
                                                             output_batch=output_batch)
input_batch = load_object_from_file('input_batch')

print(guessed_codewords_probabilities)

success_probability_from_guesses = [
    codeword_success_prob.success_probability for codeword_success_prob in guessed_codewords_probabilities]
avg_succ_prob = sum(success_probability_from_guesses) / input_batch.size

print(avg_succ_prob.numpy())
