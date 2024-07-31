import os
import logging


def split_data(data, split_ratio=0.7):
    """
    Split data into training and testing sets based on a split ratio.
    :param data: NP.ARRAY or LIST containing the data to be split.
    :param split_ratio: FLOAT indicating the ratio of data to be used for training.
    :return: TUPLE with training data and testing data.
    """
    # Calculate split point
    split_point = int(len(data) * split_ratio)

    # Split data
    train_data = data[:split_point]
    test_data = data[split_point:]

    return train_data, test_data


def create_directory(directory):
    """
    Create a directory if it does not already exist
    :param directory: STR with the path of the directory to be created
    :return: None
    """
    # Check if the directory does not exist
    if not os.path.exists(directory):
        # Print a message indicating the directory is being created
        print(f"Creation of {directory}")
        # Create the directory (and any intermediate directories if necessary)
        os.makedirs(directory)


def start_logging(dir="../log_actions", nick_name="w_outliers", code_name="WV14"):
    """
    To set up logging formats and log file names for different scenarios
    :param dir: SRT with the directory where log files will be stored.
    :param nick_name: STR indicating whether they contain outliers
    :param code_name: STR with the file code name.
    :return: None, but three logger objects will be created
    """
    create_directory(dir)

    loggers_config = {
        "info_logger": {
            "level": logging.INFO,
            "filename": os.path.join(dir, f"info_gpr_{nick_name}_found_in_{code_name}.log"),
            "format": "[%(asctime)s] %(message)s"
        },
        "warning_logger": {
            "level": logging.WARNING,
            "filename": os.path.join(dir, f"warning_gpr_{nick_name}_found_in_{code_name}.log"),
            "format": "[%(asctime)s %(levelname)s: %(message)s"
        }
    }
    for logger_key, logger_value in loggers_config.items():
        logger = logging.getLogger(logger_key)
        logger.setLevel(logger_value["level"])
        # Remove existing handlers to avoid duplicate logs
        if logger.hasHandlers():
            logger.handlers.clear()
        # file handler for log file
        file_handler = logging.FileHandler(logger_value["filename"], mode="a")
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
