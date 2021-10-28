import pennylane as qml
import numpy as np
import tensorflow as tf


N_ITER = 10
N_EPOCHS = 50
BATCH_SIZE = 1_000

tf.keras.backend.set_floatx('float64')

cutoff = 5
n_modes = 2
n_layers = 1
k = int(n_modes * (n_modes - 1) / 2)

weight_shapes = {
    "theta_1" : (n_layers, k),
    "phi_1": (n_layers, k),
    "varphi_1": (n_layers, n_modes),
    "r": (n_layers, n_modes),
    "phi_r": (n_layers, n_modes),
    "theta_2" : (n_layers, k),
    "phi_2": (n_layers, k),
    "varphi_2": (n_layers, n_modes),
    "a": (n_layers, n_modes),
    "phi_a": (n_layers, n_modes),
}

dev = qml.device("strawberryfields.fock", cutoff_dim=cutoff, wires=n_modes)

@qml.qnode(dev)
def node(inputs, theta_1, phi_1, varphi_1, r, phi_r, theta_2, phi_2, varphi_2, a, phi_a):
    """
    inputs (tensor like): shape (N <= M) where M is the number of modes. Used for displacement of each of the M modes.

    For the parameters below, L is the number of layers, K = (M * (M-1))/2 is the number of beamsplitters and M is the
    number of modes.

    theta_1 (tensor_like): shape (L, K) tensor of transmittivity angles for first interferometer.
    phi_1 (tensor_like): shape (L, K) tensor of phase angles for first interferometer.
    varphi_1 (tensor_like): shape (L, M) tensor of rotation angles to apply after first interferometer.
    r (tensor_like): shape (L, M) tensor of squeezing amounts for pennylane.ops.Squeezing operations.
    phi_r (tensor_like): shape (L, M) tensor of squeezing angles for pennylane.ops.Squeezing operations.
    theta_2 (tensor_like): shape (L, K) tensor of transmittivity angles for second interferometer.
    phi_2 (tensor_like): shape (L, K) tensor of phase angles for second interferometer.
    varphi_2 (tensor_like): shape (L, M) tensor of rotation angles to apply after second interferometer.
    a (tensor_like): shape (L, M) tensor of displacement magnitudes for pennylane.ops.Displacement operations.
    phi_a (tensor_like): shape (L, M) tensor of displacement angles for pennylane.ops.Displacement operations.
    k (tensor_like): shape (L, M) tensor of kerr parameters for pennylane.ops.Kerr operations. <= should be 0.
    wires (Iterable): wires that the template acts on.
    """
    k = np.zeros((n_layers, n_modes)) # Kerr parameters

    qml.templates.DisplacementEmbedding(inputs, wires=range(n_modes), method='amplitude', c=0.0)
    qml.templates.layers.CVNeuralNetLayers(theta_1, phi_1, varphi_1, r, phi_r, theta_2, phi_2, varphi_2, a, phi_a, k, wires=range(n_modes))

    return [qml.expval(qml.FockStateProjector(np.array([0]), wires=w)) for w in range(n_modes)]


def generate_training_data(n_datapoints):
    x_data, y_data = [], []

    for _ in range(n_datapoints):
        t_data = np.random.choice([0, 1], size=n_modes)
        index = int("".join(map(str, t_data)), base=2)
        onehot = np.zeros(2 ** n_modes)
        onehot[index] = 1

        x_data.append(t_data * 2 - 1)
        y_data.append(onehot)

    return np.array(x_data), np.array(y_data)


if __name__ == "__main__":
    q_layer = qml.qnn.KerasLayer(node, weight_shapes, output_dim=int(n_modes))
    c_layer_out = tf.keras.layers.Dense(int(2 ** n_modes), activation=tf.keras.activations.softmax)

    model = tf.keras.models.Sequential([q_layer, c_layer_out])

    opt = tf.keras.optimizers.Adam(learning_rate=0.2)
    model.compile(opt, loss="mae", metrics=["accuracy"])

    for iteration in range(N_ITER):
        print(f"Iteration {iteration + 1} of {N_ITER}.")
        X_batch, y_batch = generate_training_data(BATCH_SIZE)
        model.fit(X_batch, y_batch, epochs=N_EPOCHS, batch_size=BATCH_SIZE)

    print(model.summary())