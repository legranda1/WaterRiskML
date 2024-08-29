# Import the os module for interacting with the operating system
import time
import datetime
import re
# Importing pickle for serializing (converting Python objects to a byte
# stream)  and deserializing (reconstructing Python objects from a byte
# stream). This is useful for saving Python objects to a file or
# transferring them over a network.
import pickle
import warnings
# Import the Pool class from the multiprocessing module to handle
# parallel processing
from multiprocessing import Pool
import numpy as np
import pandas as pd
from config import *
from data import DataManager
from fun import *
from plot import PlotGPR
from gpr import GPR
from sklearn import preprocessing
# Importing combinations from itertools for generating
# combinations of elements
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (ConstantKernel, Matern,
                                              WhiteKernel)
from sklearn.metrics import root_mean_squared_error, mean_absolute_error
from sklearn.exceptions import ConvergenceWarning

logging.captureWarnings(True)

# Choose the file name of the Excel data you want to work with
FNAME = FNAMES[0]
CODE_NAME = re.search(r"WV\d+", FNAME).group(0) \
    if re.search(r"WV\d+", FNAME) else None

# Flags
# Put anything except True or False to have the target wo outliers
YEAR_TEST = True
OUTLIERS = True
BEST_R2 = True
SHOW_PLOTS = True
SAVE_PLOTS = False
SAVE_WORKSPACE = False

# Directories to create
DIR_PLOTS = "../plots/gpr/group_feature_analysis/all_feats/"
DIR_GPR_OUT_DATA = "../gpr_output_data/group_feature_analysis/all_feats/"
DIR_LOG_ACTIONS = "../log_actions/group_feature_analysis/all_feats/"
DIR_RESULTS = "../results/group_feature_analysis/all_feats/"

# System Configuration: CPU Allocation and Data Chunking
# Number of CPU cores used, impacting the speed and efficiency
# of parallel processing.
NUMBER_CPUS = 8
# Controls the size of data units processed at a time (per CPU),
# affecting load balancing and processing efficiency in parallel tasks
CHUNK_SIZE = None
# CHUNK_SIZE = 100

# Load and filter the data
if OUTLIERS is True:
    NICK_NAME = "w_outliers"
    data = DataManager(xlsx_file_name=FNAME).filter_data().reset_index(drop=True)
elif OUTLIERS is False:
    NICK_NAME = "wo_outliers"
    data = DataManager(xlsx_file_name=FNAME).iterative_cleaning(COL_ALL).reset_index(drop=True)
else:
    NICK_NAME = "tar_wo_outliers"
    data = DataManager(xlsx_file_name=FNAME).iterative_cleaning("Gesamt/Kopf").reset_index(drop=True)

SEL_FEATS = [
    "NS Monat",         # Monthly precipitation
    "T Monat Mittel",   # Average temperature of the month
    "T Max Monat",      # Maximum temperature of the month
    "pot Evap",         # Potential evaporation
    "klimat. WB",       # Climatic water balance
    "pos. klimat. WB",  # Positive climatic water balance
    "Heiße Tage",       # Number of hot days (peak temp. greater than
                        # or equal to 30 °C)
    "Sommertage",       # Number of summer days (peak temp. greater
                        # than or equal to 25 °C)
    "Eistage",          # Number of ice days
    "T Min Monat"       # Minimum temperature of the month
]


def fit_and_test(iter_params):
    """
    Fits a machine learning pipeline and tests its performance on
    given data.
    :param iter_params: A TUPLE containing parameters (object),
    features (np.array), targets (np.array), training and
    testing indices (np.array)
    :return: The parameter object with updated performance metrics
    """

    start_logging(dir=DIR_LOG_ACTIONS,
                  nick_name=f"of_all_feats_{NICK_NAME}_w_noise",
                  code_name=CODE_NAME)
    try:
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
        info_logger = logging.getLogger("info_logger")
        info_logger.info(f"Iteration = {cnt_i + 1}")
        info_logger.info(f"{params}\n")

        # Get the pipeline from the parameters
        pipeline = params.pipe

        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always")
            pipeline.fit(features[train_idx], targets[train_idx])

            for warning in caught_warnings:
                if issubclass(warning.category, ConvergenceWarning):
                    warning_logger = logging.getLogger("warning_logger")
                    warning_logger.warning(f"ConvergenceWarning in iteration"
                                           f" {cnt_i + 1}: {warning.message}")

        # Get the GaussianProcessRegressor from the pipeline
        gp = pipeline[1]

        # Predict the target values for the testing data
        predictions = pipeline.predict(features[test_idx])

        y_mean_test = np.mean(targets[test_idx])
        rmse = root_mean_squared_error(targets[test_idx], predictions)
        mae = mean_absolute_error(targets[test_idx], predictions)

        # Calculate performance metrics and update the parameter object
        params.kernel_learned = gp.kernel_
        params.marg_lh = gp.log_marginal_likelihood_value_
        params.r2_test = pipeline.score(features[test_idx], targets[test_idx])
        params.rmse_test = rmse
        params.nrmse_test = (rmse / y_mean_test) * 100
        params.mae_test = mae
        params.nmae_test = (mae / y_mean_test) * 100

        # Record the end time of the approach
        conclude_time = time.time()

        # Print more parameters and iteration details
        info_logger.info(params)
        info_logger.info(f"Iteration time:"
                         f" {conclude_time - start_time:.2f} seconds\n")

        # Return the updated parameter object
        return params

    except Exception as e:
        warning_logger = logging.getLogger("warning_logger")
        warning_logger.warning(f"An error occurred: {str(e)}")
        return None


@logging_decorator(dir=DIR_LOG_ACTIONS,
                   nick_name=f"of_all_feats_{NICK_NAME}_w_noise",
                   code_name=CODE_NAME)
def main():
    """
    Main functionality of the script
    :return: None or -1, but generates three log files
    """
    # Record the initial time for tracking script duration
    init_time = time.time()

    if YEAR_TEST:
        # Define the testing year range or individual testing years
        # Example: testing from 2015 to 2017 or non-contiguous years like [2013, 2017, 2019]
        test_years = [2015, 2016, 2017]  # Can be a range or specific years

        # Select the testing data based on the given test years
        test_df = data[data["Jahr"].isin(test_years)]

        # Define the training data by excluding the testing years
        train_df = data[~data["Jahr"].isin(test_years)]

        # Extract all features and target arrays for training and testing
        x_train = np.array(train_df[SEL_FEATS])
        y_train = np.array(train_df[COL_TAR])

        x_test = np.array(test_df[SEL_FEATS])
        y_test = np.array(test_df[COL_TAR])
        y_mean_test = np.mean(y_test)

        # Extract all features and target arrays
        x_all = np.array(data[SEL_FEATS])
        y_all = np.array(data[COL_TAR])

        # Find the exact positions (indexes) of the training and testing data in the original dataset
        x_indexes_train = train_df.index.values  # Exact positions of the training years
        x_indexes_test = test_df.index.values  # Exact positions of the testing years
        combined_index = np.arange(x_all.shape[0])
    else:
        # Extraction of all input and output data
        x_all = np.array(data[SEL_FEATS])
        y_all = np.array(data[COL_TAR])

        # Extraction of important data from the x-axis
        x_indexes = np.arange(x_all.shape[0])  # X-axis indexes
        x_indexes_train, x_indexes_test = split_data(x_indexes, 0.7)
        combined_index = np.arange(x_all.shape[0])

        # Splitting training and test data
        x_train, x_test = split_data(x_all, 0.7)
        y_train, y_test = split_data(y_all, 0.7)
        y_mean_test = np.mean(y_test)

    # List of the most popular scalers for preprocessing
    pop_scalers = [preprocessing.StandardScaler(),
                   preprocessing.QuantileTransformer(
                       n_quantiles=len(x_train), random_state=0
                   ),
                   preprocessing.MinMaxScaler(feature_range=(0, 1)),
                   preprocessing.RobustScaler(),
                   preprocessing.PowerTransformer(method='yeo-johnson',
                                                  standardize=True),
                   preprocessing.Normalizer(norm='l1'),
                   preprocessing.Normalizer(norm='l2'),
                   preprocessing.Normalizer(norm='max')]
    scalers = [pop_scalers[0]]
    # scalers = pop_scalers

    # nu recommended values for the Matern kernel
    # For nu=inf, the kernel becomes equivalent to the RBF kernel
    nu_s = [0.5, 1.5, 2.5, np.inf]

    # Initialize an empty list for storing tuples of parameters
    all_par_sets = []
    # Initialize the iteration counter
    counter = 0
    # Iterate over each scaler within the scalers list
    for scaler in scalers:
        for nu in nu_s:  # Iterate over each nu value
            kernel = (ConstantKernel(constant_value=1.0,
                                     constant_value_bounds=(0.1, 10.0))
                      * Matern(nu=nu, length_scale=1.0,
                               length_scale_bounds=(1e-3, 1e3))
                      + WhiteKernel(noise_level=1e-5,
                                    noise_level_bounds=(1e-10, 1e1)))

            all_par_sets.append((GPR(kernel=kernel, scaler=scaler,
                                     feats=SEL_FEATS,
                                     idx=counter),
                                 x_all,
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
            # Extract LML scores from each parameter set
            lml_scores = np.array([par_set.marg_lh
                                   for par_set in all_par_sets_updated])
            # Extract R2 scores from each parameter set
            r2_test_scores = np.array([par_set.r2_test
                                       for par_set in all_par_sets_updated])
            # Extract RMSE scores from each parameter set
            rmse_test_scores = np.array([par_set.rmse_test
                                         for par_set in all_par_sets_updated])
            # Extract NRMSE scores from each parameter set
            nrmse_test_scores = np.array([par_set.nrmse_test
                                         for par_set in all_par_sets_updated])
            # Extract MAE scores from each parameter set
            mae_test_scores = np.array([par_set.mae_test
                                        for par_set in all_par_sets_updated])
            # Extract NMAE scores from each parameter set
            nmae_test_scores = np.array([par_set.nmae_test
                                        for par_set in all_par_sets_updated])

            # Create a DataFrame with the results
            result_df = pd.DataFrame({
                "lh": lml_scores,
                "r2": r2_test_scores,
                "rmse": rmse_test_scores,
                "nrmse": nrmse_test_scores,
                "mae": mae_test_scores,
                "nmae": nmae_test_scores,
            })

            # Round numerical values to 2 decimal places
            result_df = result_df.round(2)

            # Save DataFrame to a CSV file
            create_directory(DIR_RESULTS)
            result_df.to_csv(f"{DIR_RESULTS}/results_of_all_feats_{NICK_NAME}_"
                             f"w_noise_in_{CODE_NAME}.csv", index=False)

        else:
            # Handle case where GPR approach failed
            warning_logger = logging.getLogger("warning_logger")
            warning_logger.warning("\n\nGPR approach went wrong!")

    # Get the best
    if BEST_R2:
        best_index = r2_test_scores.argmax()  # Find the index of the best R2 score
    # Get the parameter set corresponding to the best score
    best_par_set = all_par_sets_updated[best_index]
    # Log the total number of iterations
    info_logger = logging.getLogger("info_logger")
    info_logger.info(f"\nTotal number of iterations: {counter}")
    # Log the index of the best iteration
    info_logger.info(f"Best iteration: {best_index + 1}")
    if BEST_R2:
        # Log the best R2 score
        info_logger.info(f"Best R2 score: {r2_test_scores[best_index]}")
        info_logger.info(best_par_set)  # Log the best parameter set

    # Extract and check results by re-computing without pipe
    best_scaler = best_par_set.scaler
    best_feats = best_par_set.feats
    feats_indexes = [SEL_FEATS.index(feature) for feature in best_feats]
    x_train_scaled = best_scaler.fit_transform(x_train[:, feats_indexes])
    x_test_scaled = best_scaler.transform(x_test[:, feats_indexes])
    best_gp = GaussianProcessRegressor(kernel=best_par_set.kernel,
                                       n_restarts_optimizer=50,
                                       random_state=42,
                                       normalize_y=True,
                                       alpha=1e-10)
    # If necessary, change x_train_scaled to x_all_scaled to fit
    # the GPR to all data.
    best_gp.fit(x_train_scaled, y_train)
    marg_lh = best_gp.log_marginal_likelihood_value_
    y_pred_test = best_gp.predict(x_test_scaled)
    r2_test = best_gp.score(x_test_scaled, y_test)
    rmse_test = root_mean_squared_error(y_test, y_pred_test)
    nrmse_test = (rmse_test / y_mean_test) * 100
    mae_test = mean_absolute_error(y_test, y_pred_test)
    nmae_test = (mae_test / y_mean_test) * 100

    # Check if the re-computed results differ significantly
    # from the one with pipe
    eps = 1.0e-10  # Tolerance for detecting significant differences
    check_error = (np.abs(best_par_set.marg_lh - marg_lh) > eps or
                   np.abs(best_par_set.r2_test - r2_test) > eps or
                   np.abs(best_par_set.rmse_test - rmse_test) > eps or
                   np.abs(best_par_set.nrmse_test - nrmse_test) > eps or
                   np.abs(best_par_set.mae_test - mae_test) > eps or
                   np.abs(best_par_set.nmae_test - nmae_test) > eps)

    if check_error:
        info_logger = logging.getLogger("info_logger")
        info_logger.info("\nNew fit, result change, "
                         "difference > {eps}!")
        info_logger.info(f"log marginal likelihood (LML): {marg_lh}")
        info_logger.info(f"R2_test: {r2_test}")
        info_logger.info(f"RMSE_test: {rmse_test}")
        info_logger.info(f"NRMSE_test: {nrmse_test} %")
        info_logger.info(f"MAE_test: {mae_test}")
        info_logger.info(f"NMAE_test: {nmae_test} %")

    # Record the end time of the approach
    end_time = time.time()
    total_time = end_time - init_time
    # Print the running time in seconds and as a timedelta
    info_logger.info(f"\nRunning time: \n{total_time:5.3f} seconds "
                     f"/ {datetime.timedelta(seconds=total_time)}")

    if SAVE_PLOTS or SHOW_PLOTS:
        x_all_scaled = best_scaler.transform(x_all[:, feats_indexes])
        y_mean, y_cov = best_gp.predict(x_all_scaled, return_cov=True)
        # Create plotter instance and plot
        plotter = PlotGPR(data, f"GPR with {best_gp.kernel_}",
                          "Time [Month/Year]",
                          "Monthly per capita water consumption [L/(C*d)]",
                          1.96,
                          fig_size=(12, 6), dpi=150)
        if SHOW_PLOTS:
            plotter.plot(y_train, y_test, y_mean, y_cov,
                         x_indexes_train, x_indexes_test, combined_index,
                         r2=r2_test)

        if SAVE_PLOTS:
            create_directory(DIR_PLOTS)
            path = (f"{DIR_PLOTS}best_gpr_of_all_feats_{NICK_NAME}_"
                    f"w_noise_found_in_{CODE_NAME}.png")
            plotter.plot(y_train, y_test, y_mean, y_cov,
                         x_indexes_train, x_indexes_test, combined_index,
                         r2=r2_test, file_name=path)

    if SAVE_WORKSPACE and result.successful():
        workspace = {"best_par_set": best_par_set,
                     "best_gp": best_gp, "all_features": x_all,
                     "target": y_all, "best_scaler": best_scaler,
                     "all_par_sets_updated": all_par_sets_updated,
                     "scores": r2_test_scores,
                     "x_indexes_train": x_indexes_train,
                     "x_indexes_test": x_indexes_test,
                     "time": total_time}
        create_directory(DIR_GPR_OUT_DATA)
        # .pkl for Pickle files
        path = (f"{DIR_GPR_OUT_DATA}gpr_workspace_of_all_feats_{NICK_NAME}_"
                f"w_noise_in_{CODE_NAME}.pkl")
        info_logger.info(f"Writing output to {path}")
        # Open the file for writing in binary mode
        with open(path, "wb") as out_file:
            # Save the workspace to the .pkl file
            pickle.dump(workspace, out_file)

if __name__ == "__main__":

    main()
