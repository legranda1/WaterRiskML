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


def split_data_by_year(data, year_column, train_year_range, test_year_range):
    """
    Split data into training and testing sets based on year ranges.
    :param data: List, NP.Array, or Pandas DataFrame containing the
    data to be split.
    :param year_column: STR or INT indicating the column that
    contains the year information.
    :param train_year_range: TUPLE (start_year, end_year) indicating
    the range of years for the training data.
    :param test_year_range: TUPLE (start_year, end_year) indicating the
    range of years for the testing data.
    :return: TUPLE with training data and testing data.
    """
    # Split data based on the provided year ranges
    train_data = data[(data[year_column] >= train_year_range[0])
                      & (data[year_column] <= train_year_range[1])]
    test_data = data[(data[year_column] >= test_year_range[0])
                     & (data[year_column] <= test_year_range[1])]

    return train_data, test_data


def is_highly_correlated(feature, selected_feats,
                         corr_matrix):
    """
    Checks if a given feature is highly correlated with any of the
    already selected features
    :param feature: STR with the name of the feature to be checked for
    high correlation
    :param selected_feats: LIST of strings containing the already
    selected features
    :param corr_matrix: PD.DATAFRAME containing correlation
    coefficients between features
    :return: BOOL which returns True if the feature is highly
    correlated with any of the selected features, otherwise False
    """
    # Iterate over each selected feature
    for selected_feature in selected_feats:
        # Check if the absolute correlation between the current feature
        # and any selected feature exceeds the threshold
        if abs(corr_matrix.loc[feature, selected_feature]) > 0.7:
            return True
    return False


def is_lowly_correlated(feature, selected_feats, corr_matrix):
    """
    Checks if a given feature is lowly correlated with any of the
    already selected features
    :param feature: STR with the name of the feature to be checked for
    low correlation
    :param selected_feats: LIST of strings containing the already
    selected features
    :param corr_matrix: PD.DATAFRAME containing correlation coefficients
    between features
    :return: BOOL which returns True if the feature is lowly correlated
    with all selected features, otherwise False
    """
    # If no features are selected yet, consider the feature as not
    # correlated
    if not selected_feats:
        return True

    # Iterate over each selected feature
    for selected_feature in selected_feats:
        # Check if the absolute correlation between the current feature and any selected feature
        # is non-zero (i.e., there is some correlation, even if small)
        if abs(corr_matrix.loc[feature, selected_feature]) > 0.3:
            return False
    return True


def selected_features(data, feat1=None, feat2=None, prioritize_feature=None):
    """
    Selects features based on their correlation with a target feature
    and ensures a specific feature is always listed first.
    :param data: PD.DATAFRAME with the dataset to plot
    :param feat1: STR of the target variable (output)
    :param feat2: LIST of strings of feature names (inputs)
    :param threshold: FLOAT above which two features are considered
    highly correlated
    :param prioritize_feature: STR with the feature to prioritize
    :return: LIST of strings with the selected_features
    """
    # Compute the correlation matrix
    corr_matrix = data[[feat1] + feat2].corr()
    # Sort features by their correlation with the target feature
    target_corr = corr_matrix[feat1].drop(feat1).abs().sort_values(ascending=False)

    # Initialize the list of selected features
    selected_feats = []
    # Add the prioritized feature if it's in the list
    if prioritize_feature and prioritize_feature in target_corr.index:
        selected_feats.append(prioritize_feature)
        # Remove the prioritized feature from the target_corr index to avoid re-adding
        target_corr = target_corr.drop(prioritize_feature)

    # Iterate through the sorted features
    for feature in target_corr.index:
        # If the feature is not highly correlated with any of the
        # selected features
        if is_lowly_correlated(feature, selected_feats, corr_matrix):
            selected_feats.append(feature)

    return selected_feats


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


def logging_decorator(dir="../log_actions", nick_name="w_outliers", code_name="WV14"):
    """
    A decorator factory that allows setting parameters for logging before wrapping the function.
    :param dir: SRT with the directory where log files will be stored.
    :param nick_name: STR indicating whether they contain outliers
    :param code_name: STR with the file code name.
    :return: The actual decorator function that applies logging to the wrapped function
    """
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
            start_logging(dir, nick_name, code_name)
            fun(*args, **kwargs)
            logging.shutdown()
        return wrapper
    return log_actions

