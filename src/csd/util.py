from functools import wraps
from time import time
from csd.config import logger


def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        # logger.debug('func:%r args:[%r, %r] took: %2.4f sec' % (f.__name__, args, kw, te - ts))
        logger.debug('%r took: %2.4f' % (f.__name__, te - ts))
        return result
    return wrap
