import time
from itertools import product

import numpy as np
from loguru import logger
from scipy.optimize import minimize
from tqdm import tqdm
import wandb

from model_wrapper import LinearRegressionWrapper
from quantum_circuit import QuantumBox
from util import generate_measurement_matrices


def generate_random_codeword(NUM_MODES):
    """
    Generates a random codeword for `NUM_MODES` modes.
    """
    return np.random.choice([-1, +1], size=NUM_MODES)

def generate_training_batch(NUM_MODES):
    """
    Generates a batch of training data containing all possible codewords.
    """
    batch = np.array(list(product([-1, +1], repeat=NUM_MODES)))
    np.random.shuffle(batch)

    return  batch

def loss_metric(prediction, target, NUM_MODES):
    """
    Computes the numerical loss incurred on generating `prediction` instead of
    `target`.
    Both `prediction` and `target` are tensors.
    """
    ones = np.ones((NUM_MODES))
    zeros = np.zeros((NUM_MODES))

    indices_where_input_codeword_was_minus = np.where(target == -1, ones, zeros)
    indices_where_no_photon_is_observed = np.where(prediction == 0, ones, zeros)

    indices_where_input_codeword_was_plus = np.where(target == +1, ones, zeros)
    indices_where_at_least_one_photon_is_observed = np.where(prediction > 0, ones, zeros)

    input_minus_and_output_minus = np.logical_and(
        indices_where_input_codeword_was_minus,
        indices_where_no_photon_is_observed
    )

    input_plus_and_output_plus = np.logical_and(
        indices_where_input_codeword_was_plus,
        indices_where_at_least_one_photon_is_observed
    )

    any_error = np.logical_or(input_minus_and_output_minus, input_plus_and_output_plus)
    sum_of_combined_indices = np.sum(any_error)

    return sum_of_combined_indices / NUM_MODES

def training_error(weights, target, input_vector, layer_number, model, q_box, NUM_MODES):
    model.set_learnable_parameteres_from_flattended_list(weights)
    predicted_displacements = model(np.expand_dims(input_vector, axis=0))

    measurement_of_nth_layer = q_box(
        layer_number,
        target,
        2 * q_box.SIGNAL_AMPLITUDE * predicted_displacements)

    prediction = measurement_of_nth_layer.samples[0]
    error = loss_metric(prediction, target, NUM_MODES)

    # logger.debug(f"q::{prediction = }, c::{target = }, {error = }")

    return error

def batched_training_error(weights, targets, input_vectors, layer_number, model, q_box, NUM_MODES):
    global global_accumulated_training_error

    accumulated_error = 0.0
    batch_size = len(targets)

    for i in range(batch_size):
        accumulated_error += training_error(weights, targets[i], input_vectors[i], layer_number, model, q_box, NUM_MODES)

    batch_error = accumulated_error / batch_size

    wandb.log({"batch_error": batch_error})

    global_accumulated_training_error += batch_error

    return batch_error

def train(model, q_box, config):
    """
    Runs a single step of optimization for a single value of alpha across all
    layers of the Dolinar receiver.
    """
    input_codewords_batch = generate_training_batch(config.NUM_MODES)
    batch_size = len(input_codewords_batch)

    # logger.debug(f"before iteration:: {model.get_learnable_parameters_as_flattened_list() = }")

    for _ in range(config.NUM_REPEAT):
        previous_prediction = np.random.normal(size=(batch_size, config.NUM_MODES))

        for nth_layer in range(config.NUM_LAYERS):
            # logger.debug(f"Optimising for layer {nth_layer + 1} of {NUM_LAYERS}")

            one_hot_layer_vector = np.zeros(config.NUM_LAYERS)
            one_hot_layer_vector[nth_layer] = 1

            one_hot_layer_vectors = np.repeat([one_hot_layer_vector], batch_size, axis=0)
            input_vectors = np.concatenate([previous_prediction, one_hot_layer_vectors], axis=1)

            modes = config.NUM_MODES

            result = minimize(
                fun=batched_training_error,
                x0=model.get_learnable_parameters_as_flattened_list(),
                args=(
                    input_codewords_batch,
                    input_vectors,
                    nth_layer,
                    model,
                    q_box,
                    modes
                )
            )

            wandb.log({"accumulated_error": global_accumulated_training_error})

            # Update parameters so that previous parameters are not overwritten.
            prev_params = model.get_learnable_parameters_as_flattened_list()
            current_params = result.x
            new_params = prev_params + config.STEP_SIZE * (current_params - prev_params)

            model.set_learnable_parameteres_from_flattended_list(new_params)

            predictions = []

            for i in range(batch_size):
                input_vector = input_vectors[i]
                predicted_displacements = model(np.expand_dims(input_vector, axis=0))

                measurement_of_nth_layer = q_box(
                    nth_layer,
                    input_codewords_batch[i],
                    2 * q_box.SIGNAL_AMPLITUDE * predicted_displacements)

                prediction = measurement_of_nth_layer.samples[0]
                predictions.append(prediction)

            previous_prediction = np.array(predictions)

    # logger.debug(f"after iteration:: {model.get_learnable_parameters_as_flattened_list() = }")


def evaluate(step, model, q_box):
    codewords = list(product([-1, +1], repeat=config.NUM_MODES))
    # logger.debug(codewords)

    p_correct = 0.0
    total = 0

    for codeword in codewords:
        # logger.debug(f"{codeword = }")
        stack = [(0, np.random.normal(size=config.NUM_MODES), 1.0)]

        while stack:
            # logger.debug(f"{stack = }")
            layer_number, previous_predictions, probability_of_prediction = stack.pop()

            one_hot_layer_vector = np.zeros(config.NUM_LAYERS)
            one_hot_layer_vector[layer_number] = 1

            input_vector = np.concatenate([previous_predictions, one_hot_layer_vector])
            input_vector = np.expand_dims(input_vector, 0)

            predicted_displacements = model(input_vector)

            # logger.debug(f"{predicted_displacements = }")

            q_result = q_box(
                layer_number,
                codeword,
                2 * config.SIGNAL_AMPLITUDE * predicted_displacements)

            all_fock_probs = q_result.state.all_fock_probs()
            measurement_matrices = generate_measurement_matrices(config.NUM_MODES, config.CUTOFF_DIM)

            success_probs = [np.sum(np.multiply(mm, all_fock_probs)) for mm in measurement_matrices]
            # logger.debug(f"{success_probs = }")

            for ip, p in enumerate(success_probs):
                if layer_number < config.NUM_LAYERS - 1:
                    stack.append((layer_number + 1, codewords[ip], p * probability_of_prediction))

                if layer_number == config.NUM_LAYERS - 1:
                    if np.sum(np.array(codewords[ip]) + np.array(codeword)) == 0:
                        p_correct += probability_of_prediction

                    total += 1


    wandb.log({"average_accuracy": p_correct / total, "eval_step": step})
    logger.info(f"Accuracy: {p_correct/total:.4f}.")



if __name__ == '__main__':
    # Number of layers of the Dolinar receiver. Default is 2.
    NUM_LAYERS = 2

    # Number of quantum modes. Default is 2.
    NUM_MODES = 2

    # Number of variables being optimized per mode. Default is 1.
    NUM_VARIABLES = 1

    # Signal amplitude. Default is 1.0.
    SIGNAL_AMPLITUDE = 0.0

    # Initialize wandb logging.
    wandb.init(
        project="dolinar-receiver",
        config={
            "CUTOFF_DIM": 8,

            "STEP_SIZE": 0.95,

            "NUM_MODES": NUM_MODES,
            "NUM_LAYERS": NUM_LAYERS,
            "NUM_VARIABLES": NUM_VARIABLES,
            "SIGNAL_AMPLITUDE": SIGNAL_AMPLITUDE,

            "INPUT_VECTOR_SIZE": NUM_MODES * NUM_VARIABLES + NUM_LAYERS,
            "OUTPUT_VECTOR_SIZE": NUM_MODES * NUM_VARIABLES,

            "NUM_REPEAT": 2,
            "NUM_TRAINING_ITERATIONS": 10,

            "VERSION": "v5"
        }
    )
    wandb.run.name = f"l{NUM_LAYERS}_m{NUM_MODES}_a{SIGNAL_AMPLITUDE}"

    config = wandb.config

    # ML model to predict the displacement magnitude for each of the layers of
    # the Dolinar receiver.
    logger.info("Building model.")
    model = LinearRegressionWrapper(
        input_size=config.INPUT_VECTOR_SIZE,
        output_size=config.OUTPUT_VECTOR_SIZE
    )
    logger.info("Done.")

    # Layers of the Dolinar receiver.
    logger.info("Building quantum circuits.")
    q_box = QuantumBox(config)
    logger.info("Done.")

    wandb.log({"average_accuracy": 0.0, "eval_step": 0})

    # Training loop (with evaluation).
    logger.info("Begin training.")
    start = time.time()

    global_accumulated_training_error = 0.0

    for step in tqdm(range(config.NUM_TRAINING_ITERATIONS)):
        train(model, q_box, config)

        # Evaluate.
        evaluate(step + 1, model, q_box)

    end = time.time()
    elapsed = (end - start)
    print(f"Training took {elapsed:.2f} seconds.")

    wandb.finish()
