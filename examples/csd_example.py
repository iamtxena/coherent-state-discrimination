from csd import CSD  # noqa: E402
from csd.typings import CSDConfiguration


def example_csd() -> None:
    CSD(CSDConfiguration({
        'displacement_magnitude': 0.1,
        'steps': 500,
        'learning_rate': 0.01,
        'batch_size': 10,
        'threshold': 0.5
    }))


if __name__ == '__main__':
    example_csd()
