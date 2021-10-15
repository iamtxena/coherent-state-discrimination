import pennylane as qml

n_modes = 1
cutoff = 5

dev = qml.device("strawberryfields.fock", wires=n_modes, cutoff_dim=cutoff, analytic=False)


@qml.qnode(dev)
def qnode_layer_1():
    pass