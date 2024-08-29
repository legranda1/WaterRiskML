import re
from data import DataManager
from config import *
from plot import *
from sklearn import preprocessing
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.pipeline import Pipeline
from sklearn.gaussian_process.kernels import (ConstantKernel, Matern,
                                              WhiteKernel)
from sklearn.metrics import root_mean_squared_error, mean_absolute_error
from sklearn.model_selection import KFold
import numpy as np

if __name__ == "__main__":
    # Flags
    # Put anything except True or False to have the target wo outliers
    OUTLIERS = True
    SHOW_PLOTS = True
    SAVE_PLOTS = False

    # Directories to create
    DIR_PLOTS = "../plots/gpr/testing_folder"

    # Choose the file name of the Excel data you want to work with
    FNAME = FNAMES[0]
    CODE_NAME = re.search(r"WV\d+", FNAME).group(0) \
        if re.search(r"WV\d+", FNAME) else None

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

    # Extract all features and target arrays
    x_all = np.array(data[SEL_FEATS])
    y_all = np.array(data[COL_TAR])

    # Initialize KFold for cross-validation
    kf = KFold(n_splits=4)  # Use shuffle if you want randomized folds

    # Generating the GPR without any class (from scratch)
    nu_s = [0.5, 1.5, 2.5, np.inf]
    noise_s = ["Yes", "No"]

    for noise in noise_s:
        for nu in nu_s:
            if noise == "Yes":
                label = "w_noise"
                kernel = (ConstantKernel(constant_value=1.0,
                                         constant_value_bounds=(0.1, 10.0))
                          * Matern(nu=nu, length_scale=1.0,
                                   length_scale_bounds=(1e-3, 1e3))
                          + WhiteKernel(noise_level=1e-5,
                                        noise_level_bounds=(1e-10, 1e1)))
            else:
                label = "wo_noise"
                kernel = (ConstantKernel(constant_value=1.0,
                                         constant_value_bounds=(0.1, 10.0))
                          * Matern(nu=nu, length_scale=1.0,
                                   length_scale_bounds=(1e-3, 1e3)))

            gp = GaussianProcessRegressor(kernel=kernel,
                                          n_restarts_optimizer=50,
                                          random_state=42,
                                          normalize_y=True,
                                          alpha=1e-10)

            scaler = preprocessing.StandardScaler()
            pipe = Pipeline([("scaler", scaler), ("gp", gp)])

            for fold, (train_index, test_index) in enumerate(kf.split(x_all)):
                x_train, x_test = x_all[train_index], x_all[test_index]
                y_train, y_test = y_all[train_index], y_all[test_index]
                y_mean_test = np.mean(y_test)

                pipe.fit(x_train, y_train)
                print(f"Fold {fold + 1}")
                print(f"initial kernel: {pipe[1].kernel}")
                print(f"kernel learned: {pipe[1].kernel_}")
                print(f"scaler: {scaler}")
                print(f"log marginal likelihood (LML):"
                      f" {pipe[1].log_marginal_likelihood_value_}")
                r2_test = pipe.score(x_test, y_test)
                print(f"R2_test: {r2_test}")
                y_pred_test = pipe.predict(x_test)

                # Calculate MAE
                mae_test = mean_absolute_error(y_test, y_pred_test)
                # Calculate NRMSE as a percentage
                nmae_test = (mae_test / y_mean_test) * 100
                # Calculate RMSE
                rmse_test = root_mean_squared_error(y_test, y_pred_test)
                # Calculate NRMSE as a percentage
                nrmse_test = (rmse_test / y_mean_test) * 100
                print(f"RMSE_test: {rmse_test}")
                print(f"NRMSE_test: {nrmse_test} %")
                print(f"MAE_test: {mae_test}")
                print(f"NMAE_test: {nmae_test} %\n")
                combined_index = np.arange(x_all.shape[0])
                y_mean, y_cov = pipe.predict(x_all[combined_index], return_cov=True)

                # Create plotter instance and plot
                plotter = PlotGPR(data, f"GPR with {pipe[1].kernel_}",
                                  "Time [Month/Year]",
                                  "Monthly per capita water consumption [L/(C*d)]",
                                  1.96,
                                  fig_size=(12, 6), dpi=150)
                if SHOW_PLOTS:
                    plotter.plot(y_train, y_test, y_mean, y_cov,
                                 train_index, test_index, combined_index,
                                 r2=r2_test)
                if SAVE_PLOTS:
                    create_directory(DIR_PLOTS)
                    path = (f"{DIR_PLOTS}/gpr_{NICK_NAME}_{label}_nu_of_{nu}_"
                            f"fold_{fold + 1}_found_in_{CODE_NAME}.png")
                    plotter.plot(y_train, y_test, y_mean, y_cov,
                                 train_index, test_index, combined_index,
                                 r2=r2_test, file_name=path)
