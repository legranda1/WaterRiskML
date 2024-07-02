import logging
from config import *


def start_logging():
    """
    To set up logging formats and log file names for different scenarios
    :return: None, but three logger objects will be created
    """
    loggers_config = {
        "info_logger": {"level": logging.INFO, "filename": "info.log",
                        "format": "[%(asctime)s] %(message)s"},
        "warning_logger": {
            "level": logging.WARNING, "filename": "warning.log",
            "format": "[%(asctime)s %(levelname)s: %(message)s"
        },
        "error_logger": {"level": logging.ERROR, "filename": "error.log",
                         "format": "[%(asctime)s %(levelname)s: %(message)s"}
    }
    for logger_key, logger_value in loggers_config.items():
        logger = logging.getLogger(logger_key)
        logger.setLevel(logger_value["level"])
        # file handler for log file
        file_handler = logging.FileHandler(logger_value["filename"], mode="w")
        file_handler.setFormatter(logging.Formatter(logger_value["format"]))
        logger.addHandler(file_handler)
        # stream handler for terminal output
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter(logger_value["format"]))
        logger.addHandler(stream_handler)


def log_actions(fun):
    """
    'log_actions' decorator
    :param fun: a function
    :return: result of applying the wrapper function to the
    corresponding "fun"
    """
    def wrapper(*args, **kwargs):
        """
        Wrapper function in order to log script execution messages
        :param args: optional arguments
        :param kwargs: optional keyword arguments
        :return: None
        """
        start_logging()
        fun(*args, **kwargs)
        logging.shutdown()
    return wrapper
