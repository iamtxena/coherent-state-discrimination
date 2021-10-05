
# data.py
""" Data to used alongside the test suite """

from csd.typings import CSDConfiguration, Backends

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
