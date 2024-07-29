import os


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
