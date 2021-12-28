

import os
import logging

# log level settings
log_level = os.environ.get("PUBLIC_LOG_LEVEL", "DEBUG")

# create logger
logger = logging.getLogger('API')
logger.setLevel(log_level)

# create console handler and set level to debug
ch = logging.StreamHandler()
ch.setLevel(log_level)

# create formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# add formatter to ch
ch.setFormatter(formatter)

# add ch to logger
logger.addHandler(ch)
