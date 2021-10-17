from functools import wraps
from time import time
from typing import Optional
from csd.config import logger
import numpy as np
import pickle

SECONDS_PER_MINUTE = 60
MINUTES_PER_HOUR = 60


def set_friendly_time(time_interval: float) -> str:
    hours = 0
    minutes = int(np.floor(time_interval / SECONDS_PER_MINUTE))
    seconds = int(np.floor(time_interval % SECONDS_PER_MINUTE))

    if minutes > MINUTES_PER_HOUR:
        hours = int(np.floor(minutes / MINUTES_PER_HOUR))
        minutes = int(np.floor(minutes % MINUTES_PER_HOUR))

    friendly_time = ''

    if hours:
        friendly_time += f'{hours} hours, '
    if minutes == 1:
        friendly_time += f'{minutes} minute and '
    if minutes > 1:
        friendly_time += f'{minutes} minutes and '
    friendly_time += f'{seconds} seconds.'
    return friendly_time


def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        # logger.debug('func:%r args:[%r, %r] took: %2.4f sec' % (f.__name__, args, kw, te - ts))
        # logger.debug('%r took: %2.4f' % (f.__name__, te - ts))
        logger.debug(f'{f.__name__}, took: {set_friendly_time(time_interval=te-ts)}')
        return result
    return wrap


def save_object_to_disk(obj, name: str, path: Optional[str] = "") -> None:
    """ save result to a file """
    with open(f'./{path}{name}.pkl', 'wb') as file:
        pickle.dump(obj, file, pickle.HIGHEST_PROTOCOL)


def load_object_from_file(name: str, path: Optional[str] = ""):
    """ load object from a file """
    with open(f'./{path}{name}.pkl', 'rb') as file:
        return pickle.load(file)
