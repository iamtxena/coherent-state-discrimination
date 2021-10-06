
# data.py
""" Data to used alongside the test suite """

import numpy as np
from csd.typings import CSDConfiguration, Backends, RunConfiguration

csd_test_configurations = [
    {
        'displacement_magnitude': 0.1,
        'steps': 500,
        'learning_rate': 0.01,
        'batch_size': 10,
        'threshold': 0.5
    }
]

csd_configurations = [CSDConfiguration({
    'displacement_magnitude': config['displacement_magnitude'],
    'steps': int(config['steps']),
    'learning_rate': config['learning_rate'],
    'batch_size': int(config['batch_size']),
    'threshold': config['threshold']
})
    for config in csd_test_configurations]

backends = [
    Backends.FOCK,
    Backends.GAUSSIAN,
    Backends.BOSONIC,
    Backends.TENSORFLOW,
]

valid_backends = [
    Backends.FOCK,
    Backends.GAUSSIAN,
    Backends.TENSORFLOW,
]

alphas = np.arange(0.4, 1.4, 0.1)
betas = np.arange(0.0, 1.0, 0.1)

csd_run_configurations = [
    RunConfiguration({
        'alpha': 0.5,
        'displacement_magnitude': 0.1,
        'backend': Backends.FOCK,
        'number_qumodes': 1,
        'number_layers': 1
    })
]
