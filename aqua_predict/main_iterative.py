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

import matplotlib.pyplot as plt
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
from sklearn.gaussian_process import GaussianProcessRegressor
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
    params.marg_lh = gp.log_marginal_likelihood_value_
    params.r2_test = pipeline.score(features[test_idx], targets[test_idx])
    params.rmse_test = root_mean_squared_error(targets[test_idx], predictions)
    params.mae_test = mean_absolute_error(targets[test_idx], predictions)

    # Record the end time of the approach
    conclude_time = time.time()

    # Print the learned kernel and iteration details
    print(f"Kernel learned: {gp.kernel_}")
    print(params)
    print(f"Iteration time: {conclude_time - start_time:.2f} seconds")
    print("\n")

    # Return the updated parameter object
    return params


if __name__ == "__main__":
    # Record the initial time for tracking script duration
    init_time = time.time()

    # Choose the file name of the Excel data you want to work with
    FNAME = "Auswertung WV14 Unteres Elsenztal"
    # FNAME = "Auswertung WV25 SW Füssen"
    # FNAME = "Auswertung WV69 SW Landshut"

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
    data = DataManager(xlsx_file_name=f"{FNAME}.xlsx").filter_data()
    # data = DataManager(xlsx_file_name=f"{FNAME}.xlsx").iterative_cleaning(COL_ALL)

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

    # Initialize an empty list for storing tuples of parameters
    all_par_sets = []
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

                    all_par_sets.append((GPR(kernel=kernel, scaler=scaler,
                                             feats=comb_feat, idx=counter),
                                         x_all[:, comb_feat],
                                         y_all, x_indexes_train,
                                         x_indexes_test))

                    counter += 1

    results_iter = all_par_sets

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
            all_par_sets_updated = result.get()
            # Extract R2 scores from each parameter set
            r2_test_scores = np.array([par_set.r2_test for par_set in all_par_sets_updated])
        else:
            # Handle case where GPR approach failed
            print("\n\nGPR approach went wrong!")

    # Get the best
    best_index = r2_test_scores.argmax()  # Find the index of the best R2 score
    # Get the parameter set corresponding to the best score
    best_par_set = all_par_sets_updated[best_index]
    # Print the index of the best iteration
    print(f"\nBest iteration: {best_index}")
    # Print the best R2 score
    print(f"Best R2 score: {r2_test_scores[best_index]}")
    print(best_par_set)  # Print the best parameter set

    # Extract and check results by re-computing without pipe
    best_scaler = best_par_set.scaler
    x_all_scaled = best_scaler.fit_transform(x_all[:, best_par_set.feats])
    x_train_scaled = best_scaler.fit_transform(x_train[:, best_par_set.feats])
    x_test_scaled = best_scaler.fit_transform(x_test[:, best_par_set.feats])
    best_gp = GaussianProcessRegressor(kernel=best_par_set.kernel,
                                       n_restarts_optimizer=50,
                                       random_state=42,
                                       normalize_y=True,
                                       alpha=1e-10)
    # If necessary, change x_train_scaled to x_all_scaled to fit the GPR to all data.
    best_gp.fit(x_train_scaled, y_train)
    marg_lh = best_gp.log_marginal_likelihood_value_
    y_pred_test = best_gp.predict(x_test_scaled)
    r2_test = best_gp.score(x_test_scaled, y_test)
    rmse_test = root_mean_squared_error(y_test, y_pred_test)
    mae_test = mean_absolute_error(y_test, y_pred_test)

    # Check if the re-computed results differ significantly from the one with pipe
    eps = 1.0e-10  # Tolerance for detecting significant differences
    check_error = (np.abs(best_par_set.r2_test - r2_test) > eps or
                   np.abs(best_par_set.rmse_test - rmse_test) > eps or
                   np.abs(best_par_set.mae_test - mae_test) > eps)

    if check_error:
        print(f"New fit, result change, difference > {eps}!")
        print(f"log marginal likelihood (LML): {marg_lh}",
              f"R2_test: {r2_test}", f"RMSE_test: {rmse_test}",
              f"MAE_test: {mae_test}", sep="\n")

    # Record the end time of the approach
    end_time = time.time()
    total_time = end_time - init_time
    # Print the running time in seconds and as a timedelta
    print(f"Running time: \n{total_time:5.3f} seconds /"
          f" {datetime.timedelta(seconds=total_time)}")

    if SAVE_PLOTS or SHOW_PLOTS:
        y_mean, y_cov = best_gp.predict(x_all_scaled, return_cov=True)
        # Create plotter instance and plot
        plotter = PlotGPR(data, f"GPR with {best_gp.kernel_}",
                          "Time [Month/Year]",
                          "Monthly per capita water consumption [L/(C*d)]",
                          1.96,
                          fig_size=(12, 6))
        if SAVE_PLOTS:
            if not os.path.exists(DIR_PLOTS):
                print(f"Creation of {DIR_PLOTS}")
                os.makedirs(DIR_PLOTS)
            PATH = f"{DIR_PLOTS}/best_gpr_found_in_{FNAME}.png"
            plt.savefig(PATH, dpi=200,
                        bbox_inches="tight", pad_inches=0.25)
        if SHOW_PLOTS:
            plotter.plot(y_train, y_test, y_mean, y_cov)

    if SAVE_WORKSPACE and result.successful():
        workspace = {"best_par_set": best_par_set,
                     "best_gp": best_gp, "all features": x_all,
                     "target": y_all, "best_scaler": best_scaler,
                     "all_par_sets_updated": all_par_sets_updated,
                     "r2_test_scores": r2_test_scores,
                     "x_indexes_train": x_indexes_train,
                     "x_indexes_test": x_indexes_test,
                     "time": total_time}
        if not os.path.exists(DIR_OUT_DATA):
            print(f"Creation of {DIR_OUT_DATA}")
            os.makedirs(DIR_OUT_DATA)
        PATH = f"{DIR_OUT_DATA}/gpr_workspace_of_{FNAME}.p"
        print(f"Writing output to {PATH}")
        with open(PATH, "wb") as out_file:  # Open the file for writing in binary mode
            pickle.dump(workspace, out_file)  # Save the workspace to the .p file
