import strawberryfields as sf
import numpy as np
import tensorflow as tf
import os
from loguru import logger

# Hides info messages from TensorFlow.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Number of layers of the Dolinar receiver. Selecting 4 as the most basic,
# non-trivial case.
NUM_LAYERS = 4

# Number of quantum modes. Basic 2-mode case.
NUM_MODES = 2

# Number of variables being optimized per mode.
NUM_VARIABLES = 1

# Signal amplitude. Default is 1.0.
SIGNAL_AMPLITUDE = 1.0

# Fock backend.
ENGINE = sf.Engine("fock", backend_options={"cutoff_dim": 6})


def generate_nth_layer(layer_number, engine):
    """Generates the nth layer of the Dolinar receiver.
    Given the `layer_number` and `engine` as input, it returns a
    function that generates the necessary quantum circuit for the n-th layer of
    the Dolinar receiver.
    """

    # Need k values for the splits of the coherent state.
    amplitudes =  np.ones(NUM_LAYERS) * (SIGNAL_AMPLITUDE / NUM_LAYERS)

    def quantum_layer(input_codeword, displacement_magnitudes_for_each_mode):
        program = sf.Program(NUM_MODES)

        with program.context as q:
            # Prepare the coherent states for the layer. Appropriately scales
            # the amplitudes for each of the layers.
            for m in range(NUM_MODES):
                sf.ops.Coherent(amplitudes[layer_number] * input_codeword[m]) | q[m]

            # Displace each of the modes by using the displacement magnitudes
            # generated by the ML backend.
            for m in range(NUM_MODES):
                sf.Dgate(displacement_magnitudes_for_each_mode[m]) | q[m]

            # Perform measurements.
            sf.ops.MeasureFock() | q

        return engine.run(program)

    return quantum_layer


def build_model(name="predictor"):
    model = tf.keras.Sequential([
        tf.keras.Input(shape=(NUM_MODES * NUM_VARIABLES, )),
        tf.keras.layers.Dense(8, activation="relu", name="layer-1"),
        tf.keras.layers.Dense(16, activation="relu", name="layer-2"),
        tf.keras.layers.Dense(16, activation="relu", name="layer-3"),
    ], name=name)

    return model


if __name__ == '__main__':
    # ML model to predict the displacement magnitude for each of the layers of
    # the Dolinar receiver.
    logger.info("Building model.")
    model = build_model(f"predictor-l-{NUM_LAYERS}-alpha-{SIGNAL_AMPLITUDE}-modes-{NUM_MODES}")
    logger.info("Done.")

    # Layers of the Dolinar receiver.
    logger.info("Building quantum circuits.")
    layers = [generate_nth_layer(n, ENGINE) for n in range(NUM_LAYERS)]
    logger.info("Done.")

    # TODO: Add training loop.
    # TODO: Add previous measurement to current layer and use it while calling the
    # predictor to obtain the new displacement values for the layer.
