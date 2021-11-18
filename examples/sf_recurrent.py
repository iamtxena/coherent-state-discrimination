import strawberryfields as sf
from strawberryfields import ops
import numpy as np

# Number of layers of the Dolinar receiver. Selecting 2 as the most basic,
# non-trivial case.
num_layers = 2

# Need k values for the splits of the coherent state. They need to sum one.
# TODO: We need to decide whether this needs to be some specific value or should
# I stick to the uniform split that I am doing presently.
energy_splits = np.ones(num_layers)/num_layers

def generate_nth_layer(layer_number, previous_measurement=None):
    """Generates the nth layer of the Dolinar receiver.
    Given the `layer_number` and `previous_measurement` as input, it returns a
    function that generates the necessary quantum circuit for the layer.
    `previous_measurement` is a free parameter for the quantum circuit.

    This function should also query the ML backend for the appropriate
    displacement.
    """
    pass

if __name__ == '__main__':
    # Basic 2-mode case.
    program = sf.Program(2)
