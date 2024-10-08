from data import *
from fun import *
from plot import *
from config import *
from sklearn import preprocessing
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (ConstantKernel, Matern,
                                              WhiteKernel)
from sklearn.metrics import root_mean_squared_error, mean_absolute_error

if __name__ == "__main__":
    # Choose the file name of the Excel data you want to work with
    FNAME = FNAMES[0]

    # Load and filter the data
    data = DataManager(xlsx_file_name=FNAME).filter_data().reset_index(drop=True)
    # data = DataManager(xlsx_file_name=FNAME).iterative_cleaning(COL_ALL).reset_index(drop=True)

    # Define the testing year range or individual testing years
    # Example: testing from 2015 to 2017 or non-contiguous years like [2015, 2017, 2019]
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

    # Scaling all input data, training, and testing data
    scaler = preprocessing.StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    # Generating the GPR without pipe
    nu_s = [0.5, 1.5, 2.5, np.inf]
    # For nu=inf, the kernel becomes equivalent to the RBF kernel
    nu = nu_s[0]

    kernel = (ConstantKernel(constant_value=1.0,
                             constant_value_bounds=(0.1, 10.0))
              * Matern(nu=nu, length_scale=1.0,
                       length_scale_bounds=(1e-3, 1e3))
              + WhiteKernel(noise_level=1e-5,
                            noise_level_bounds=(1e-10, 1e1)))

    gp = GaussianProcessRegressor(kernel=kernel,
                                  n_restarts_optimizer=50,
                                  random_state=42,
                                  normalize_y=True,
                                  alpha=1e-10)

    gp.fit(x_train_scaled, y_train)
    print(f"initial kernel: {gp.kernel}")
    print(f"kernel learned: {gp.kernel_}")
    print(f"scaler: {scaler}")
    print(f"log marginal likelihood (LML):"
          f" {gp.log_marginal_likelihood_value_}")
    r2_test = gp.score(x_test_scaled, y_test)
    print(f"R2_test: {r2_test}")
    y_pred_test = gp.predict(x_test_scaled)

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
    x_all_scaled = scaler.transform(x_all)
    y_mean, y_cov = gp.predict(x_all_scaled, return_cov=True)

    # Create plotter instance and plot
    plotter = PlotGPR(data, f"GPR with {gp.kernel_}",
                      "Time [Month/Year]",
                      "Monthly per capita water consumption [L/(C*d)]",
                      1.96,
                      fig_size=(12, 6), dpi=200)
    plotter.plot(y_train, y_test, y_mean, y_cov,
                 x_indexes_train, x_indexes_test, combined_index,
                 r2=r2_test)
