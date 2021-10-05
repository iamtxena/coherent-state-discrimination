""" Test methods for csd """

import pytest
from csd import CSD
from csd.typings import CSDConfiguration, Backends
from csd.config import logger
from .data import csd_test_configurations, csd_configurations, backends


@pytest.mark.parametrize("csd_test_configuration", csd_test_configurations)
def test_csd_construction(csd_test_configuration: dict):
    csd_configuration = CSDConfiguration({
        'displacement_magnitude': csd_test_configuration['displacement_magnitude'],
        'steps': csd_test_configuration['steps'],
        'learning_rate': csd_test_configuration['learning_rate'],
        'batch_size': csd_test_configuration['batch_size'],
        'threshold': csd_test_configuration['threshold']
    })
    csd = CSD(csd_configuration)
    assert isinstance(csd, CSD)


@pytest.mark.parametrize("csd_configuration", csd_configurations)
@pytest.mark.parametrize("backend", backends)
def test_csd_single_layer(csd_configuration: CSDConfiguration, backend: Backends):
    csd = CSD(csd_config=csd_configuration)
    logger.info(f'using backend: {backend}')
    if backend is Backends.BOSONIC:
        with pytest.raises(Exception) as execinfo:
            csd.single_layer(backend)
        assert str(execinfo.value) == "The operation MeasureFock cannot be used with the compiler 'bosonic'."
    else:
        result = csd.single_layer(backend)

        assert result is not None
        assert result.state is not None
        logger.info(f'Fock Probability: {result.state.fock_prob([0])}')
        assert result.state.fock_prob([0]) > 0.99
