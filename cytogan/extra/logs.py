import logging
import sys


def get_root_logger(filename=None):
    handlers = [logging.StreamHandler()]
    if filename:
        handlers.append(logging.FileHandler(filename))

    logger = logging.getLogger('cytogan')
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        '%(asctime)s [%(levelname)s] %(filename)s:%(lineno)d | %(message)s')
    for handler in handlers:
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


def get_raw_logger(name):
    '''
    Returns a logger with the same handlers as the root logger but no
    formatting at all.
    '''
    logger = logging.getLogger('{0}-raw'.format(name))
    logger.setLevel(logger.parent.level)
    logger.propagate = False
    formatter = logging.Formatter('%(message)s')
    for parent_handler in logger.parent.handlers:
        if isinstance(parent_handler, logging.FileHandler):
            handler = logging.FileHandler(parent_handler.baseFilename)
        else:
            handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        handler.terminator = ''
        logger.addHandler(handler)

    return logger


def get_logger(name):
    return logging.getLogger(name)


class LogFile(object):
    '''Class that acts like a file but actually logs.'''
    def __init__(self, logger):
        self.logger = logger

    def write(self, message):
        self.logger.info(message)

    def flush(self):
        pass
