import pennylane as qml
import numpy as np
import tensorflow as tf


cutoff = 5
n_modes = 1

dev = qml.device("strawberryfields.fock", cutoff_dim=cutoff, wires=n_modes)

@qml.qnode(dev)
def node(inputs, theta_1, phi_1, varphi_1, r, phi_r, theta_2, phi_2, varphi_2, a, phi_a, k):
    """
    inputs (tensor like): shape (N <= M) where M is the number of modes.

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
    k (tensor_like): shape (L, M) tensor of kerr parameters for pennylane.ops.Kerr operations.
    wires (Iterable): wires that the template acts on.
    """
    qml.templates.DisplacementEmbedding(inputs, wires=range(n_modes), method='amplitude', c=0.0)
    qml.templates.layers.CVNeuralNetLayers(theta_1, phi_1, varphi_1, r, phi_r, theta_2, phi_2, varphi_2, a, phi_a, k, wires=range(n_modes))

    return [qml.expval(qml.X(wires = w)) for w in range(n_modes)]
