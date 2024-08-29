from gpr import *
from data import *
from fun import *
from plot import *
from config import *
from sklearn import preprocessing
from sklearn.gaussian_process.kernels import (ConstantKernel, Matern,
                                              WhiteKernel)
from sklearn.metrics import root_mean_squared_error, mean_absolute_error

# Define the column name for the target variable (output)
COL_TAR = "Gesamt/Kopf"  # Monthly water demand

# Define a list of feature column names (inputs)
COL_FEAT = [
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

if __name__ == "__main__":
    # Choose the file name of the Excel data you want to work with
    FNAME = FNAMES[0]

    # Load and filter the data
    data = DataManager(xlsx_file_name=FNAME).filter_data().reset_index(drop=True)
    # data = DataManager(xlsx_file_name=FNAME).iterative_cleaning(COL_ALL).reset_index(drop=True)

    # Define the testing year range or individual testing years
    # Example: testing from 2015 to 2017 or non-contiguous years like [2013, 2017, 2019]
    test_years = [2015, 2016, 2017]  # Can be a range or specific years

    # Select the testing data based on the given test years
    test_df = data[data["Jahr"].isin(test_years)]

    # Define the training data by excluding the testing years
    train_df = data[~data["Jahr"].isin(test_years)]

    # Extract all features and target arrays for training and testing
    x_train = np.array(train_df[COL_FEAT])
    y_train = np.array(train_df[COL_TAR])

    x_test = np.array(test_df[COL_FEAT])
    y_test = np.array(test_df[COL_TAR])
    y_mean_test = np.mean(y_test)

    # Extract all features and target arrays
    x_all = np.array(data[COL_FEAT])
    y_all = np.array(data[COL_TAR])

    # These indexes aren't used in this script, but the fit_and_test function does use them
    # and loses some decimals. That's why it's recalculated at the end to ensure the difference is no greater than 1e-10
    x_indexes_train = train_df.index.values  # Exact positions of the training years
    x_indexes_test = test_df.index.values  # Exact positions of the testing years
    combined_index = np.arange(x_all.shape[0])

    # Generating the GPR without any class (from scratch)
    nu_s = [0.5, 1.5, 2.5, np.inf]
    # For nu=inf, the kernel becomes equivalent to the RBF kernel
    nu = nu_s[0]

    kernel = (ConstantKernel(constant_value=1.0,
                             constant_value_bounds=(0.1, 10.0))
              * Matern(nu=nu, length_scale=1.0,
                       length_scale_bounds=(1e-3, 1e3))
              + WhiteKernel(noise_level=1e-5,
                            noise_level_bounds=(1e-10, 1e1)))

    scaler = preprocessing.StandardScaler()

    gpr = GPR(kernel=kernel, scaler=scaler)

    gpr.pipe.fit(x_train, y_train)
    print(f"initial kernel: {gpr.pipe[1].kernel}")
    print(f"kernel learned: {gpr.pipe[1].kernel_}")
    print(f"scaler: {scaler}")
    print(f"log marginal likelihood (LML):"
          f" {gpr.pipe[1].log_marginal_likelihood_value_}")
    r2_test = gpr.pipe.score(x_test, y_test)
    print(f"R2_test: {r2_test}")
    y_pred_test = gpr.pipe.predict(x_test)

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
    y_mean, y_cov = gpr.pipe.predict(x_all, return_cov=True)

    # Create plotter instance and plot
    plotter = PlotGPR(data,f"GPR with {gpr.pipe[1].kernel_}",
                      "Time [Month/Year]",
                      "Monthly per capita water consumption [L/(C*d)]",
                      1.96,
                      fig_size=(12, 6), dpi=200)
    plotter.plot(y_train, y_test, y_mean, y_cov,
                 x_indexes_train, x_indexes_test, combined_index,
                 r2=r2_test)
