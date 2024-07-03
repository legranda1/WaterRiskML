import matplotlib.pyplot as plt
import numpy as np


class PlotGPR:
    def __init__(self, title="GPR", xlabel="Time",
                 ylabel="Water Demand", std=1.96):
        self.title = title
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.std = std  # Standard deviation
        self.figure, self.ax = plt.subplots()

    def set_xticks(self, ticks, labels):
        """
        Sets the x-axis ticks and labels
        :param ticks: NP.ARRAY with the positions of the ticks on the
        x-axis
        :param labels: NP.ARRAY with labels corresponding to the tick
        positions
        :return: None
        """
        self.ax.set_xticks(ticks)
        self.ax.set_xticklabels(labels)

    def plot(self, x_indexes_train, y_train, x_indexes_test, y_test,
             x_indexes, y_mean, y_cov, x_ticks, x_labs_plot):
        """
        Plots the knowing data points, GPR predictions, and uncertainty
        intervals.
        :param x_indexes_train: NP.ARRAY with indices for the training
        data points on the x-axis
        :param y_train: NP.ARRAY with target values corresponding to
        the training data points
        :param x_indexes_test: NP.ARRAY with indices for the test data
        points on the x-axis
        :param y_test: NP.ARRAY with target values corresponding to the
        test data points
        :param x_indexes: NP.ARRAY with indexes for all data points on
        the x-axis
        :param y_mean: NP.ARRAY with predicted mean values for all data
        points
        :param y_cov: 2D NP.ARRAY containing the covariance matrix for
        the predicted values
        :param x_ticks: NP.ARRAY with positions of x-axis ticks
        :param x_labs_plot: NP.ARRAY with labels corresponding to the
        x-axis ticks
        :return: None
        """
        # Set the title and labels
        self.ax.set_title(self.title)
        self.ax.set_xlabel(self.xlabel)
        self.ax.set_ylabel(self.ylabel)

        # Plot knowing data points
        self.ax.scatter(x_indexes_train, y_train, c="k", s=15,
                        zorder=10, marker="x")
        self.ax.scatter(x_indexes_test, y_test, c="r", s=15,
                        zorder=10, edgecolors=(0, 0, 0))

        # Plot predictions and uncertainty
        self.ax.plot(x_indexes, y_mean, "C0", lw=1.5, zorder=9)
        self.ax.fill_between(x_indexes,
                             y_mean - self.std * np.sqrt(np.diag(y_cov)),
                             y_mean + self.std * np.sqrt(np.diag(y_cov)),
                             color="C0", alpha=0.2)

        # Customize the plot
        self.ax.grid(True)
        self.ax.tick_params(axis="x", rotation=70)
        self.ax.legend(["Training Data", "Test Data",
                        "GP Mean",
                        f"GP conf interval ({int(self.std)} std)"])
        self.set_xticks(x_ticks, x_labs_plot)

        # Show the plot
        plt.show()
