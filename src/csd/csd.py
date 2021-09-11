from csd.config import logger
from abc import ABC


class CSD(ABC):
    def __init__(self):
        logger.info("Just a test that logger is working. Happy coding!")
