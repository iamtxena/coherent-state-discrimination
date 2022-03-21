import time

import numpy as np
from loguru import logger
from scipy.optimize import minimize
# from optimparallel import minimize_parallel as minimize
from tqdm import tqdm
import wandb

from model_wrapper import LinearRegressionWrapper
from quantum_circuit import QuantumBox


def generate_random_codeword(NUM_MODES):
    """
    Generates a random codeword for `NUM_MODES` modes.
    """
    return np.random.choice([-1, +1], size=NUM_MODES)


def loss_metric(prediction, target, NUM_MODES):
    """
    Computes the numerical loss incurred on generating `prediction` instead of
    `target`.
    Both `prediction` and `target` are tensors.
    """
    true_values = np.ones((NUM_MODES))
    false_values = np.zeros((NUM_MODES))

    indices_where_input_codeword_was_minus = np.where(target == -1, true_values, false_values)
    indices_where_no_photon_is_observed = np.where(prediction == 0, true_values, false_values)

    indices_where_input_codeword_was_plus = np.where(target == +1, true_values, false_values)
    indices_where_at_least_one_photon_is_observed = np.where(prediction > 0, true_values, false_values)

    combined_indices_1 = np.logical_and(
        indices_where_input_codeword_was_minus,
        indices_where_no_photon_is_observed
    )

    combined_indices_2 = np.logical_and(
        indices_where_input_codeword_was_plus,
        indices_where_at_least_one_photon_is_observed
    )

    combined_indices = np.logical_or(combined_indices_1, combined_indices_2)
    sum_of_combined_indices = np.sum(np.sum(combined_indices))

    return sum_of_combined_indices / NUM_MODES

def training_error(weights, target, input_vector, layer_number, model, q_box, NUM_MODES):
    model.set_learnable_parameteres_from_flattended_list(weights)
    predicted_displacements = model(input_vector)

    measurement_of_nth_layer = q_box(
        layer_number,
        target,
        2 * q_box.SIGNAL_AMPLITUDE * predicted_displacements)

    prediction = measurement_of_nth_layer.samples[0]
    # logger.debug(f"{prediction = }, {target = }")

    error = loss_metric(prediction, target, NUM_MODES)

    wandb.log({"error": error})

    return error

def train(model, q_box):
    """
    Runs a single step of optimization for a single value of alpha across all
    layers of the Dolinar receiver.
    """
    input_codeword = generate_random_codeword(config.NUM_MODES)
    # logger.debug(f"{input_codeword = }")

    for _ in range(config.NUM_REPEAT):
        previous_prediction = np.random.normal(size=config.NUM_MODES * config.NUM_VARIABLES)

        for nth_layer in range(config.NUM_LAYERS):
            # logger.debug(f"Optimising for layer {nth_layer + 1} of {NUM_LAYERS}")

            one_hot_layer_vector = np.zeros(config.NUM_LAYERS)
            one_hot_layer_vector[nth_layer] = 1

            input_vector = np.concatenate([previous_prediction, one_hot_layer_vector])
            input_vector = np.expand_dims(input_vector, 0)

            # TODO: Use weights to compute previous_prediction for layer 1...n-1 for layer n.

            modes = config.NUM_MODES

            result = minimize(
                fun=training_error,
                x0=model.get_learnable_parameters_as_flattened_list(),
                args=(
                    input_codeword,
                    input_vector,
                    nth_layer,
                    model,
                    q_box,
                    modes
                )
            )

            model.set_learnable_parameteres_from_flattended_list(result.x)

            predicted_displacements = model(input_vector)

            measurement_of_nth_layer = q_box(
                nth_layer,
                input_codeword,
                2 * config.SIGNAL_AMPLITUDE * predicted_displacements)

            previous_prediction = measurement_of_nth_layer.samples[0]

def evaluate(step, model, q_box):
    codewords = []
    for i in range(2 ** config.NUM_MODES):
        string_in_binary = bin(i)[2:]
        pad_length = config.NUM_MODES - len(string_in_binary)

        padded_string_in_binary = "0" * pad_length + string_in_binary
        codeword = np.array(list(map(int, padded_string_in_binary)))
        codeword = codeword * 2 - 1

        codewords.append(codeword)
    # logger.debug(codewords)

    n_correct = 0
    total = 0

    for _ in range(config.NUM_REPEAT):
        for codeword in codewords:
            t_previous_predictions = np.random.normal(size=config.NUM_MODES * config.NUM_VARIABLES)

            for nth_layer in range(config.NUM_LAYERS):
                one_hot_layer_vector = np.zeros(config.NUM_LAYERS)
                one_hot_layer_vector[nth_layer] = 1

                input_vector = np.concatenate([t_previous_predictions, one_hot_layer_vector])
                input_vector = np.expand_dims(input_vector, 0)

                predicted_displacements = model(input_vector)

                measurement_of_nth_layer = q_box(
                    nth_layer,
                    codeword,
                    2 * config.SIGNAL_AMPLITUDE * predicted_displacements)

                prediction = measurement_of_nth_layer.samples[0]
                t_previous_predictions = prediction

            prediction_of_final_layer = t_previous_predictions

            all_modes_correct = True
            for i in range(codeword.shape[0]):
                if codeword[i] * prediction_of_final_layer[i] > 0:
                    all_modes_correct = False

            n_correct = n_correct + 1 if all_modes_correct else 0
            total += 1

    wandb.log({"average_accuracy": n_correct / total, "eval_step": step})
    logger.info(f"Accuracy: {n_correct/total:.4f}.")



if __name__ == '__main__':
    # Number of layers of the Dolinar receiver. Selecting 4 as the most basic,
    # non-trivial case.
    NUM_LAYERS = 2 # 1,2,3

    # Number of quantum modes. Basic 2-mode case.
    NUM_MODES = 2

    # Number of variables being optimized per mode.
    NUM_VARIABLES = 1

    # Initialize wandb logging.
    wandb.init(
        project="dolinar-receiver",
        config={
            "CUTOFF_DIM": 8,

            "NUM_MODES": NUM_MODES,
            "NUM_LAYERS": NUM_LAYERS,
            "NUM_VARIABLES": NUM_VARIABLES,
            "SIGNAL_AMPLITUDE": 0.2, # Signal amplitude. Default is 1.0.

            "INPUT_VECTOR_SIZE": NUM_MODES * NUM_VARIABLES + NUM_LAYERS,
            "OUTPUT_VECTOR_SIZE": NUM_MODES * NUM_VARIABLES,

            "NUM_REPEAT": 25,
            "NUM_TRAINING_ITERATIONS": 50,
            "MAX_ITERATIONS": 100,
        }
    )

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

    wandb.log({"average_accuracy": 0.0})

    # Training loop (with evaluation).
    logger.info("Begin training.")
    start = time.time()

    for step in tqdm(range(config.NUM_TRAINING_ITERATIONS)):
        train(model, q_box)

        # Evaluate.
        evaluate(step, model, q_box)

    end = time.time()
    elapsed = (end - start)
    print(f"Training took {elapsed:.2f} seconds.")

    wandb.finish()