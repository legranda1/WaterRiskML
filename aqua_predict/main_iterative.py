import numpy as np
from config import *
from data import *
from fun import *
from plot import *
from gpr import GPR
from sklearn import preprocessing
from itertools import combinations
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.pipeline import Pipeline
from sklearn.gaussian_process.kernels import (ConstantKernel, Matern,
                                              WhiteKernel)
from sklearn.metrics import root_mean_squared_error, mean_absolute_error


def col_combos(cols, min_len=1):
    """
    Generate combinations of input columns with varying lengths.
    :param cols: LIST of input column indices.
    :param min_len: INT representing the minimum number of columns to combine. Default is 1.
    :return: A LIST of numpy arrays, each containing a combination of column indices.
    """
    # Determine the maximum number of columns that can be combined
    max_len = len(cols)
    # Initialize an empty list to hold the combinations of columns
    col_feats = []
    # Generate combinations for each length from min_len to max_len
    for n in range(min_len, max_len + 1):
        # Generate combinations of the current length and convert them to numpy arrays of type int8
        col_feats.extend(np.array(combo, dtype=np.int8)
                         for combo in combinations(cols, n))
    # Return the list of column combinations
    return col_feats

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
    N_CPUS = 8   # Number of CPU cores used, impacting the speed and efficiency of parallel processing.
    CHUNK_SIZE = None  # Controls the size of data units processed at a time (per CPU),
    # affecting load balancing and processing efficiency in parallel tasks
    # CHUNK_SIZE = 100

    # Load and filter the data
    data = DataManager(xlsx_file_name=FNAME).filter_data()
    # data = DataManager(xlsx_file_name=FNAME).iterative_cleaning(COL_ALL)

    # Extraction of all input and output data
    x_all = np.array(data[COL_FEAT])
    y_all = np.array(data[COL_TAR])

    # Splitting training and test data
    x_train, x_test = split_data(x_all, 0.7)
    y_train, y_test = split_data(y_all, 0.7)

    number_cols = len(COL_FEAT)  # Number of feature columns
    indexes_cols = np.arange(number_cols)  # Array of column indices
    comb_feats = col_combos(indexes_cols)  # Generate combinations of input columns with varying lengths.

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
    nu_s = [0.5, 1.5, 2.5, np.inf]  # For nu=inf, the kernel becomes equivalent to the RBF kernel

    results_feats = []  # Initialize an empty list for storing results of features
    counter = 0  # Initialize the iteration counter
    for comb_feat in comb_feats:  # Iterate over each combination of feature columns
        for scaler in scalers:  # Iterate over each scaler within the scalers list
            for noise in noise_s:  # Iterate over the noise switch (Yes/No)
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

                    results_feats.extend([GPR(kernel=kernel, scaler=scaler,
                                              feats=comb_feat, idx=counter),
                                          x_all[:, comb_feats],
                                          y_all, x_train, x_test])  # Ojo con los dos ultimos 

                    counter += 1

    results_iter =  results_feats



                    """
                    gp = GaussianProcessRegressor(kernel=kernel,
                                                  n_restarts_optimizer=50,
                                                  random_state=42,
                                                  normalize_y=True,
                                                  alpha=1e-10)

                    pipe = Pipeline([("scaler", scaler), ("gp", gp)])
                    pipe.fit(x_train, y_train)
                    print(f"initial kernel: {pipe[1].kernel}")
                    print(f"kernel learned: {pipe[1].kernel_}")
                    print(f"scaler: {scaler}")
                    print(f"marginal log likelihood:"
                          f" {pipe[1].log_marginal_likelihood_value_}")
                    print(f"R2 (coefficient of determination):"
                          f" {pipe.score(x_test, y_test)}")
                    y_pred_test = pipe.predict(x_test)
                    print(f"RMSE: {root_mean_squared_error(y_test, y_pred_test)}")
                    print(f"MAE: {mean_absolute_error(y_test, y_pred_test)}\n")
                    y_mean, y_cov = pipe.predict(x_all, return_cov=True)

                    # Create plotter instance and plot
                    plotter = PlotGPR(data, f"GPR with {pipe[1].kernel_}",
                                      "Time [Month/Year]",
                                      "Monthly per capita water consumption [L/(C*d)]",
                                      1.96,
                                      fig_size=(12, 6))
                    plotter.plot(y_train, y_test, y_mean, y_cov)
                    """

