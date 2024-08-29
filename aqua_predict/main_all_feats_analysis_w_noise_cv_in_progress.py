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
from sklearn.model_selection import cross_validate, cross_val_score, KFold, StratifiedKFold
from sklearn.metrics import make_scorer, mean_squared_error, mean_absolute_error, r2_score
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


def fit_and_test_cv(params, X, y):
    """
    Applies cross-validation to evaluate the model.
    :param params: The parameter object (kernel, scaler, etc.)
    :param X: Feature matrix
    :param y: Target vector
    :return: The parameter object with updated performance metrics
    """
    start_logging(dir=DIR_LOG_ACTIONS, nick_name=f"of_all_feats_{NICK_NAME}_w_noise", code_name=CODE_NAME)

    try:
        # Record the start time
        start_time = time.time()
        info_logger = logging.getLogger("info_logger")
        info_logger.info(f"{params}\n")

        # Define a custom scoring function for cross-validation
        scoring = {
            "r2": make_scorer(r2_score),
            "rmse": make_scorer(mean_squared_error, squared=False),
            "mae": make_scorer(mean_absolute_error)
        }

        # Perform cross-validation
        cv_results = cross_validate(params.pipe, X, y,
                                    cv=KFold(n_splits=5, shuffle=True, random_state=42),
                                    scoring=scoring, return_train_score=False)

        # Store the mean of cross-validation metrics in the params object
        params.r2_test = np.mean(cv_results["test_r2"])
        params.rmse_test = np.mean(cv_results["test_rmse"])
        params.mae_test = np.mean(cv_results["test_mae"])

        # Record the end time of the iteration
        conclude_time = time.time()

        # Log the parameters and metrics
        info_logger.info(params)
        info_logger.info(f"Iteration time: {conclude_time - start_time:.2f} seconds\n")

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

    # Extract all features and target arrays
    x_all = np.array(data[SEL_FEATS])
    y_all = np.array(data[COL_TAR])

    # List of the most popular scalers for preprocessing
    pop_scalers = [preprocessing.StandardScaler(),
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
                                     idx=counter)))
            counter += 1

    # Use multiprocessing Pool to distribute tasks across NUMBER_CPUS
    with Pool(NUMBER_CPUS) as pool:
        # Apply fit_and_test function to each tuple
        # in results_iter asynchronously
        result = pool.map_async(lambda p: fit_and_test_cv(p, x_all, y_all), all_par_sets, chunksize=CHUNK_SIZE)
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
        info_logger.info(f"Best R2 score: {r2_test_scores[best_index]}")
        info_logger.info(best_par_set)  # Log the best parameter set

    # Record the end time of the approach
    end_time = time.time()
    total_time = end_time - init_time
    # Print the running time in seconds and as a timedelta
    info_logger.info(f"\nRunning time: \n{total_time:5.3f} seconds "
                     f"/ {datetime.timedelta(seconds=total_time)}")

    if SAVE_WORKSPACE and result.successful():
        workspace = {"best_par_set": best_par_set,
                     "all_features": x_all,
                     "target": y_all,
                     "best_scaler": best_par_set.scaler,
                     "all_par_sets_updated": all_par_sets_updated,
                     "scores": r2_test_scores,
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
