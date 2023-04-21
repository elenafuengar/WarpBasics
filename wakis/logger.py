'''
Logger module to manage console outputs
with a custom coloured scheme

@date: Created on 20.10.2022
@author: Elena de la Fuente
'''

import logging
import sys

class Logger(logging.Formatter):

    grey = "\x1b[1;37m"
    green = "\x1b[0;32m"
    blue = "\x1b[36m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    #format = "%(asctime)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)"
    format = "- %(levelname)s - %(message)s (%(filename)s:%(lineno)d)"

    FORMATS = {
        logging.DEBUG: "["+ blue + "%(levelname)s" + reset + "] %(message)s (" + blue +"%(filename)s:%(lineno)d" + reset + ")",
        logging.INFO: "["+ green + "%(levelname)s" + reset + "] %(message)s",
        logging.WARNING: "["+ yellow + "%(levelname)s" + reset + "] %(message)s (" + yellow +"%(filename)s:%(lineno)d" + reset + ")",
        logging.ERROR: red + format + reset,
        logging.CRITICAL: bold_red + format + reset
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)

def get_logger(Logger=Logger, level=2):
    log = logging.getLogger(__name__)

    #set level
    if level == 1:
        log.setLevel(logging.DEBUG)
    elif level == 2:
        log.setLevel(logging.INFO)
    elif level == 3:
        log.setLevel(logging.WARNING)
    elif level == 4:
        log.setLevel(logging.ERROR)
    elif level == 5:
        log.setLevel(logging.CRITICAL)
    else:
        #default
        log.setLevel(logging.INFO)

    #handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(Logger())
    log.addHandler(ch)

    return log
