# Import the os module for interacting with the operating system
import os
import time
import datetime
# Importing pickle for serializing (converting Python objects to a byte
# stream)  and deserializing (reconstructing Python objects from a byte
# stream). This is useful for saving Python objects to a file or
# transferring them over a network.
import pickle
# Import the Pool class from the multiprocessing module to handle
# parallel processing
from multiprocessing import Pool
import numpy as np
from config import *
from data import DataManager
from fun import *
from plot import PlotGPR
from gpr import GPR
from sklearn import preprocessing
# Importing combinations from itertools for generating
# combinations of elements
from itertools import combinations
from sklearn.gaussian_process.kernels import (ConstantKernel, Matern,
                                              WhiteKernel)
from sklearn.metrics import root_mean_squared_error, mean_absolute_error


def col_combos(cols, min_len=1):
    """
    Generate combinations of input columns with varying lengths.
    :param cols: LIST of input column indices
    :param min_len: INT representing the minimum number of columns
    to combine (Default is 1)
    :return: A LIST of numpy arrays, each containing a combination
    of column indices
    """
    # Determine the maximum number of columns that can be combined
    max_len = len(cols)
    # Initialize an empty list to hold the combinations of columns
    col_feats = []
    # Generate combinations for each length from min_len to max_len
    for n in range(min_len, max_len + 1):
        # Generate combinations of the current length and convert them
        # to numpy arrays of type int8
        for combo in combinations(cols, n):
            col_feats.append(np.array(combo, dtype=np.int8))
    # Return the list of column combinations
    return col_feats


def fit_and_test(iter_params):
    """
    Fits a machine learning pipeline and tests its performance on
    given data.
    :param iter_params: A TUPLE containing parameters (object),
    features (np.array),
    targets (np.array), training and testing indices (np.array)
    :return: The parameter object with updated performance metrics
    """
    # Unpack the tuple
    cnt_i = iter_params[0].idx  # Iteration counter
    # Parameters (kernel, scaler, features, indexes, pipe, etc.)
    params = iter_params[0]
    features = iter_params[1]
    targets = iter_params[2]
    train_idx = iter_params[3]
    test_idx = iter_params[4]

    # Record the start time of the iteration
    start_time = time.time()
    print(f"Iteration = {cnt_i + 1}")
    print(f"{params}\n")

    # Get the pipeline from the parameters
    pipeline = params.pipe

    # Fit the pipeline to the training data
    pipeline.fit(features[train_idx], targets[train_idx])

    # Get the GaussianProcessRegressor from the pipeline
    gp = pipeline[1]

    # Predict the target values for the testing data
    predictions = pipeline.predict(features[test_idx])

    # Calculate performance metrics and update the parameter object
    params.r2 = pipeline.score(features[test_idx], targets[test_idx])
    params.rmse = root_mean_squared_error(targets[test_idx], predictions)
    params.mae = mean_absolute_error(targets[test_idx], predictions)
    params.marg_lh = gp.log_marginal_likelihood_value_

    # Record the end time of the approach
    end_time = time.time()

    # Print the learned kernel and iteration details
    print(f"Kernel learned: {gp.kernel_}")
    print(params)
    print(f"Iteration time: {end_time - start_time:.2f} seconds")
    print("\n")

    # Return the updated parameter object
    return params


if __name__ == "__main__":
    # Record the initial time for tracking script duration
    init_time = time.time()

    # Choose the file name of the Excel data you want to work with
    FNAME = "Auswertung WV14 Unteres Elsenztal.xlsx"
    # FNAME = "Auswertung WV25 SW Füssen.xlsx"
    # FNAME = "Auswertung WV69 SW Landshut.xlsx"

    # Flags
    SHOW_PLOTS = False
    SAVE_PLOTS = False
    SAVE_WORKSPACE = False

    # Directories to create
    DIR_PLOTS = "../plots"
    DIR_OUT_DATA = "../output_data"

    # System Configuration: CPU Allocation and Data Chunking
    # Number of CPU cores used, impacting the speed and efficiency
    # of parallel processing.
    NUMBER_CPUS = 8
    # Controls the size of data units processed at a time (per CPU),
    # affecting load balancing and processing efficiency in parallel tasks
    CHUNK_SIZE = None
    # CHUNK_SIZE = 100

    # Load and filter the data
    data = DataManager(xlsx_file_name=FNAME).filter_data()
    # data = DataManager(xlsx_file_name=FNAME).iterative_cleaning(COL_ALL)

    # Extraction of all input and output data
    x_all = np.array(data[COL_FEAT])
    y_all = np.array(data[COL_TAR])

    # Extraction of important data from the x-axis
    x_indexes = np.arange(x_all.shape[0])  # X-axis indexes
    x_indexes_train, x_indexes_test = split_data(x_indexes, 0.7)

    # Splitting training and test data
    x_train, x_test = split_data(x_all, 0.7)
    y_train, y_test = split_data(y_all, 0.7)

    number_cols = len(COL_FEAT)  # Number of feature columns
    indexes_cols = np.arange(number_cols)  # Array of column indices
    # Generate combinations of input columns with varying lengths.
    comb_feats = col_combos(indexes_cols)

    # List of the most popular scalers for preprocessing
    pop_scalers = [preprocessing.StandardScaler(),
                   preprocessing.QuantileTransformer(random_state=0),
                   preprocessing.MinMaxScaler(feature_range=(0, 1)),
                   preprocessing.RobustScaler(),
                   preprocessing.PowerTransformer(method='yeo-johnson',
                                                  standardize=True),
                   preprocessing.Normalizer(norm='l1'),
                   preprocessing.Normalizer(norm='l2'),
                   preprocessing.Normalizer(norm='max')]
    scalers = [pop_scalers[0], pop_scalers[1]]
    # scalers = sc_all

    # List to toggle noise addition
    noise_s = ["Yes", "No"]

    # nu recommended values for the Matern kernel
    # For nu=inf, the kernel becomes equivalent to the RBF kernel
    nu_s = [0.5, 1.5, 2.5, np.inf]

    # Initialize an empty list for storing results of features
    results_feats = []
    # Initialize the iteration counter
    counter = 0
    # Iterate over each combination of feature columns
    for comb_feat in comb_feats:
        # Iterate over each scaler within the scalers list
        for scaler in scalers:
            # Iterate over the noise switch (Yes/No)
            for noise in noise_s:
                for nu in nu_s:  # Iterate over each nu value
                    if noise == "Yes":  # If noise is to be added
                        kernel = (ConstantKernel(constant_value=1.0,
                                                 constant_value_bounds=(0.1, 10.0))
                                  * Matern(nu=nu, length_scale=1.0,
                                           length_scale_bounds=(1e-3, 1e3))
                                  + WhiteKernel(noise_level=1e-5,
                                                noise_level_bounds=(1e-10, 1e1)))
                    else:  # If noise is not to be added
                        kernel = (ConstantKernel(constant_value=1.0,
                                                 constant_value_bounds=(0.1, 10.0))
                                  * Matern(nu=nu, length_scale=1.0,
                                           length_scale_bounds=(1e-3, 1e3)))

                    results_feats.append((GPR(kernel=kernel, scaler=scaler,
                                              feats=comb_feat, idx=counter),
                                          x_all[:, comb_feat],
                                          y_all, x_indexes_train,
                                          x_indexes_test))

                    counter += 1

    results_iter = results_feats

    # Use multiprocessing Pool to distribute tasks across NUMBER_CPUS
    with Pool(NUMBER_CPUS) as pool:
        # Apply fit_and_test function to each tuple
        # in results_iter asynchronously
        result = pool.map_async(fit_and_test, results_iter,
                                chunksize=CHUNK_SIZE)
        result.wait()
        # Check if all processes were successful
        if result.successful():
            # Retrieve results from asynchronous processing
            # (i.e., objects of GPRPars())
            final_results = result.get()
            # Extract R2 scores from each parameter set
            r2_scores = np.array([par_set.r2 for par_set in final_results])
        else:
            # Handle case where GPR approach failed
            print("\n\nGPR approach went wrong!")

    # Record the end time of the approach
    end_time = time.time()

    # Get the best
    best_index = r2_scores.argmax()  # Find the index of the best R2 score
    # Get the parameter set corresponding to the best score
    par_set = final_results[best_index]
    # Print the index of the best iteration
    print(f"\nBest iteration: {best_index}")
    # Print the best R2 score
    print(f"Best R2 score: {r2_scores[best_index]}")
    print(par_set)  # Print the best parameter set
    total_time = end_time - init_time
    # Print the running time in seconds and as a timedelta
    print(f"Running time: \n{total_time:5.3f} seconds /"
          f" {datetime.timedelta(seconds=total_time)}")
