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
    FNAME = "Auswertung WV14 Unteres Elsenztal.xlsx"
    # FNAME = "Auswertung WV25 SW Füssen.xlsx"
    # FNAME = "Auswertung WV69 SW Landshut.xlsx"

    # Load and filter the data
    data = DataManager(xlsx_file_name=FNAME).filter_data()
    # data = DataManager(xlsx_file_name=FNAME).iterative_cleaning(COL_ALL)

    # Extraction of all input and output data
    x_all = np.array(data[COL_FEAT])
    y_all = np.array(data[COL_TAR])

    # Scaling all input data
    scaler = preprocessing.QuantileTransformer(
                n_quantiles=92, random_state=0
            )
    x_all_scaled = scaler.fit_transform(x_all)

    # Splitting training and test data
    x_train_scaled, x_test_scaled = split_data(x_all_scaled, 0.7)
    y_train, y_test = split_data(y_all, 0.7)

    # Generating the GPR without pipe
    nu_s = [0.5, 1.5, 2.5, np.inf]
    # For nu=inf, the kernel becomes equivalent to the RBF kernel
    nu = nu_s[-1]

    kernel = (ConstantKernel(constant_value=1.0,
                             constant_value_bounds=(0.1, 10.0))
              * Matern(nu=nu, length_scale=1.0,
                       length_scale_bounds=(1e-2, 1e3))
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
    print(f"RMSE_test: {root_mean_squared_error(y_test, y_pred_test)}")
    print(f"MAE_test: {mean_absolute_error(y_test, y_pred_test)}")
    y_mean, y_cov = gp.predict(x_all_scaled, return_cov=True)

    # Create plotter instance and plot
    plotter = PlotGPR(data, f"GPR with {gp.kernel_}",
                      "Time [Month/Year]",
                      "Monthly per capita water consumption [L/(C*d)]",
                      1.96,
                      fig_size=(12, 6), dpi=200)
    plotter.plot(y_train, y_test, y_mean, y_cov, r2=r2_test)
