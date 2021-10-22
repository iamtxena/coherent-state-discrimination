from csd import CSD  # noqa: E402
from csd.typings.typing import CSDConfiguration, Backends
from csd.config import logger


def example_csd() -> None:
    initial_configuration = CSDConfiguration({
        'displacement_magnitude': 0.1,
        'steps': 500,
        'learning_rate': 0.01,
        'batch_size': 10,
        'threshold': 0.5})

    csd = CSD(csd_config=initial_configuration)
    result = csd.single_layer(backend=Backends.FOCK)
    logger.info(result.state.fock_prob([0]))
    logger.info(result.state.fock_prob([0]) > 0.99)
    logger.info(csd.show_result())


if __name__ == '__main__':
    example_csd()
