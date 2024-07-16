from gpr import *
from data import *
from fun import *
from plot import *
from config import *
from sklearn import preprocessing
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

    # Extraction of important data from the x-axis for plotting
    x_labels = np.array(data["Monat/Jahr"])    # All X-axis time labels
    x_indexes = np.arange(x_labels.shape[0])   # X-axis indexes
    x_indexes_train, x_indexes_test = split_data(x_indexes, 0.7)
    x_ticks = np.arange(0, len(x_indexes), 6)  # X ticks for plotting
    x_labs_plot = x_labels[x_ticks]            # X labels for plotting

    # Extraction of all input and output data
    x_all = np.array(data[COL_FEAT])
    y_all = np.array(data[COL_TAR])

    # Splitting training and test data
    x_train, x_test = split_data(x_all, 0.7)
    y_train, y_test = split_data(y_all, 0.7)

    # Generating the GPR without any class (from scratch)
    nu_s = [0.5, 1.5, 2.5, np.inf]
    # For nu=inf, the kernel becomes equivalent to the RBF kernel
    nu = nu_s[-1]

    kernel = (ConstantKernel(constant_value=1.0,
                             constant_value_bounds=(0.1, 10.0))
              * Matern(nu=nu, length_scale=1.0,
                       length_scale_bounds=(1e-2, 1e3))
              + WhiteKernel(noise_level=1e-5,
                            noise_level_bounds=(1e-10, 1e1)))

    scaler = preprocessing.QuantileTransformer(
        n_quantiles=len(x_indexes_train), random_state=0
    )

    gpr = GPR(kernel=kernel, scaler=scaler)

    gpr.pipe.fit(x_train, y_train)
    print(f"initial kernel: {gpr.pipe[1].kernel}")
    print(f"kernel learned: {gpr.pipe[1].kernel_}")
    print(f"scaler: {scaler}")
    print(f"marginal log likelihood:"
          f" {gpr.pipe[1].log_marginal_likelihood_value_}")
    print(f"R2 (coefficient of determination):"
          f" {gpr.pipe[1].score(x_test, y_test)}")
    y_pred_test = gpr.pipe.predict(x_test)
    print(f"RMSE: {root_mean_squared_error(y_test, y_pred_test)}")
    print(f"MAE: {mean_absolute_error(y_test, y_pred_test)}")
    y_mean, y_cov = gpr.pipe.predict(x_all, return_cov=True)

    # Create plotter instance and plot
    plotter = PlotGPR(f"GPR with {gpr.pipe[1].kernel_}",
                      "Time [Month/Year]",
                      "Monthly per capita water consumption [L/(C*d)]",
                      1.96,
                      fig_size=(12, 6))
    plotter.plot(x_indexes_train, y_train, x_indexes_test, y_test,
                 x_indexes, y_mean, y_cov, x_ticks, x_labs_plot)
