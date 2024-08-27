import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd
import seaborn as sns
from fun import *


class PlotBp:
    def __init__(self, df=None, title="Boxplot",
                 ylabel="Ranges", fig_size=(14, 6), dpi=150):
        """
        Initialize the PlotBp class.
        :param df: PD.DATAFRAME containing the data
        :param title: STR with the title of the boxplot
        :param ylabel: STR with the Y-axis label
        :param fig_size: TUPLE with the figure size for the plot
        :param dpi: INT with the expected dots per inch of the figure
        """
        self.df = df
        self.title = title
        self.ylabel = ylabel
        self.fig_size = fig_size
        self.dpi = dpi

    def plotdf(self, features, path=None):
        """
        Generate and display a boxplot of a dataframe for the
        specified features
        :param features: LIST of strings with the specified features
        :param path: STR with the name of the path to save the plot
        :return: None
        """
        if not isinstance(self.df, pd.DataFrame):
            raise ValueError("df must be a pandas DataFrame")

        plt.figure(figsize=self.fig_size)  # Create a new figure
        axes = plt.gca()  # Get current axes

        # Set the title and labels
        axes.set_title(self.title)
        axes.set_ylabel(self.ylabel)

        # Plot boxplot
        self.df.boxplot(column=features)

        if path:
            plt.savefig(path, dpi=self.dpi,
                        bbox_inches="tight", pad_inches=0.25)
            plt.close()
        else:
            plt.tight_layout()
            plt.show()

    def plot(self, features, units, path=None):
        """
        Generate and display a boxplot for the specified features
        :param features: LIST of strings with the specified features
        :param units: LIST of strings with the units corresponding to
        each feature
        :param path: STR with the name of the path to save the plot
        :return: None
        """
        plt.figure(figsize=self.fig_size)  # Create a new figure
        axes = plt.gca()  # Get current axes

        # Set the title and labels
        axes.set_title(self.title, fontsize=12)
        axes.set_ylabel(self.ylabel, fontsize=11)

        # Prepare data for boxplot
        if isinstance(features, str):
            features = [features]
        if isinstance(units, str):
            units = [units]
        # Ensure features and units lists are the same length
        assert len(features) == len(units), ("Features and units must have"
                                             " the same length")

        # .values on a pandas Series returns a NP.ARRAY containing
        # the underlying data of that Series
        data = [self.df[feature].values for feature in features]

        # Customize the plot
        axes.grid(True)
        flierprops = dict(marker=".", markersize=6, markeredgewidth=0.5,
                          markeredgecolor="red", markerfacecolor="red")
        # Generate random colors for each boxplot
        colors = [cm.tab10(i) for i in np.linspace(0, 1, len(features))]

        # Plot boxplot
        bp = axes.boxplot(data, flierprops=flierprops,
                          labels=features, patch_artist=True)

        # Change the color of the median line
        for median in bp["medians"]:
            median.set(color="black", linewidth=1)

        # Assign random colors to each box
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)

        # Add unit labels below the feature labels
        for i, (feature, unit) in enumerate(zip(features, units), start=1):
            axes.annotate(unit,
                          xy=(i, -0.05),
                          xycoords=("data", "axes fraction"),
                          xytext=(0, -15), textcoords="offset points",
                          ha="center", va="bottom",
                          fontsize=11, color="black")

        # Increase font sizes of tick labels
        axes.tick_params(axis='x', labelsize=11)
        axes.tick_params(axis='y', labelsize=11)

        if path:
            plt.savefig(path, dpi=self.dpi,
                        bbox_inches="tight", pad_inches=0.25)
            plt.close()
        else:
            plt.show()


class PlotCorr:
    def __init__(self, df=None, title="Correlatiom",
                 fig_size=(8, 6), dpi=150):
        """
        Initialize the PlotCorr class.
        :param df: PD.DATAFRAME containing the filtered data
        :param title: STR with the title of the heatmap
        """
        self.df = df
        self.title = title
        self.fig_size = fig_size
        self.dpi = dpi

    def plot_hm(self, path=None):
        """
        Generate and display a correlation coefficient
        heatmap for the specified features
        :param path: STR with the name of the path to save the plot
        :return: None
        """
        plt.figure(figsize=self.fig_size)
        # Plot heat map
        sns.heatmap(self.df, annot=True, fmt=".2f", cmap="coolwarm",
                    cbar=True, square=True, linewidths=0.5)
        plt.suptitle(self.title)

        if path:
            plt.savefig(path, dpi=self.dpi,
                        bbox_inches="tight", pad_inches=0.25)
            plt.close()
        else:
            # Show the plot
            plt.tight_layout()
            plt.show()

    def plot_pp(self, path=None):
        """
        Generate and display a correlation pairplot for the
        specified features
        :param path: STR with the name of the path to save the plot
        :return: None
        """
        # Plot pairplot
        pp = sns.pairplot(self.df, markers=".", plot_kws={"linewidth": 0.5})

        # Delete the for loop to have vertical axis labels again
        # Iterate through each axes object and set the y-axis labels
        # horizontally
        for ax in pp.axes.flatten():
            if ax is not None:
                ax.tick_params(axis="x", labelsize=8)
                ax.tick_params(axis="y", labelsize=8)
                ax.xaxis.label.set_size(8)
                ax.yaxis.label.set_size(8)
                ax.yaxis.get_label().set_rotation(0)
                ax.yaxis.labelpad = 40

        # Iterate through each axes object to annotate with
        # correlation coefficients
        for i in range(len(self.df.columns)):
            for j in range(len(self.df.columns)):
                ax = pp.axes[i, j]

                if i == j:
                    # For diagonal plots, annotate with mean or
                    # other metric
                    ax.annotate(f"μ = {np.mean(self.df.iloc[:, i]):.2f}",
                                xy=(0.5, 1.0), xycoords='axes fraction',
                                ha='center', va='center', fontsize=6)
                    ax.annotate(f"σ = {np.std(self.df.iloc[:, i]):.2f}",
                                xy=(0.5, 0.5), xycoords='axes fraction',
                                ha='center', va='center', fontsize=6)
                else:
                    # For off-diagonal plots, annotate with
                    # correlation coefficient
                    ax.annotate(f"r = {self.df.corr().iloc[i, j]:.2f}",
                                xy=(0.5, 1.20), xycoords='axes fraction',
                                ha='center', va='center', fontsize=7)

        plt.suptitle(self.title)
        plt.tight_layout(rect=[0, 0, 1, 0.95])

        if path:
            pp.fig.set_size_inches(self.fig_size)
            pp.savefig(path, dpi=self.dpi,
                       bbox_inches="tight", pad_inches=0.25)
            plt.close()
        else:
            # Show the plot
            plt.show()


class PlotGPR:
    def __init__(self, df=None, title="GPR", xlabel="Time",
                 ylabel="Water Demand", std=1.96, fig_size=(14, 6),
                 dpi=150):
        """
        Initialize the PlotGPR class.
        :param df: PD.DATAFRAME containing the data
        :param title: STR with the title of the GPR
        :param xlabel: STR with the X-axis label
        :param ylabel: STR with the Y-axis label
        :param std: FLOAT with the standard deviation value
        :param fig_size: TUPLE with the figure size for the plot
        :param dpi: INT with the expected dots per inch of the figure
        """
        self.df = df
        self.title = title
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.std = std
        self.figure_size = fig_size
        self.dpi = dpi
        self.fig = None   # Figure object
        self.axes = None  # Axes object

    def plot(self, y_train, y_test, y_mean, y_cov, r2=None,
             file_name=None, test_years=None):
        """
        Plots the knowing data points, GPR predictions, and uncertainty
        intervals.
        :param y_train: NP.ARRAY with target values corresponding to
        the training data points
        :param y_test: NP.ARRAY with target values corresponding to the
        test data points
        :param y_mean: NP.ARRAY with predicted mean values for all data
        points
        :param y_cov: 2D NP.ARRAY containing the covariance matrix for
        the predicted values
        :param r2: FLOAT with the coefficient of determination of the
        area of validation
        :param file_name: STR with the name of the file to save the plot
        :param test_years: LIST with the range or specific testing years
        :return: None
        """
        self.fig, self.axes = plt.subplots(figsize=self.figure_size)

        # Extract X-axis labels and indexes
        x_labels = np.array(self.df["Monat/Jahr"])
        x_indexes = np.arange(x_labels.shape[0])

        if test_years:
            test_df = self.df[self.df["Jahr"].isin(test_years)]
            train_df = self.df[~self.df["Jahr"].isin(test_years)]
            x_indexes_train = train_df.index.values
            x_indexes_test = test_df.index.values

        else:
            x_indexes_train, x_indexes_test = split_data(x_indexes, 0.7)

        x_ticks = np.arange(0, len(x_indexes), 6)  # X ticks for plotting
        x_labs_plot = x_labels[x_ticks]  # X labels for plotting

        # Ensure last tick matches the last value on the x-axis
        if x_ticks[-1] != len(x_indexes) - 1:
            x_ticks = np.append(x_ticks, len(x_indexes) - 1)
            x_labs_plot = np.append(x_labs_plot, x_labels[-1])

        # Set the title and labels
        self.axes.set_title(self.title)
        self.axes.set_xlabel(self.xlabel)
        self.axes.set_ylabel(self.ylabel)

        # Plot knowing data points
        self.axes.scatter(x_indexes_train, y_train, c="k", s=15,
                          zorder=10, marker="x")
        self.axes.scatter(x_indexes_test, y_test, c="r", s=15,
                          zorder=10, edgecolors=(0, 0, 0))

        # Plot predictions and uncertainty
        self.axes.plot(x_indexes, y_mean, "C0", lw=1.5, zorder=9)
        self.axes.fill_between(x_indexes,
                               y_mean - self.std * np.sqrt(np.diag(y_cov)),
                               y_mean + self.std * np.sqrt(np.diag(y_cov)),
                               color="C0", alpha=0.2)

        # Customize the plot
        self.axes.grid(True)
        self.axes.set_xticks(x_ticks)
        self.axes.set_xticklabels(x_labs_plot)
        self.axes.tick_params(axis="x", rotation=70)
        legend = self.axes.legend(["Training Data", "Test Data",
                                   "GP Mean",
                                   f"GP conf interval ({self.std} std)"],
                                  loc="upper left")

        # Extract the font size from the legend
        fontsize = legend.get_texts()[0].get_fontsize()

        if r2 is not None:
            # Add R2 text in the upper right corner
            textstr = f"$R^2 = {r2:.2f}$"
            self.axes.text(0.87, 0.95, textstr, transform=self.axes.transAxes,
                           fontsize=fontsize, verticalalignment="top",
                           horizontalalignment="right",
                           bbox=dict(facecolor='white', alpha=0.2))

        # Add a vertical hashed line at the beginning of the test data
        # self.axes.axvline(x=x_indexes_test[0], color="k",
        #                  linestyle="--", linewidth=1)

        if file_name:
            self.fig.savefig(file_name, dpi=self.dpi,
                             bbox_inches="tight", pad_inches=0.25)
            plt.close(self.fig)
        else:
            # Show the plot
            plt.tight_layout()
            plt.show()


class PlotTS:
    def __init__(self, df=None, title="Time series of features",
                 xlabel="Time", ylabel="Ranges", fig_size=(14, 6), dpi=150):
        """
        Initialize the PlotTS class.
        :param df: PD.DATAFRAME containing the data
        :param title: STR with the title of the time series plot
        :param xlabel: STR with the X-axis label
        :param ylabel: STR with the Y-axis label
        :param fig_size: TUPLE with the figure size for the plot
        """
        self.df = df
        self.title = title
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.figure_size = fig_size
        self.dpi = dpi

    def plot(self, features, path=None):
        """
        Plots the time series of different features
        :param features: LIST of STR with the names of the respective feature
        :param path: STR with the name of the path to save the plot
        :return: None
        """
        plt.figure(figsize=self.figure_size)  # Create a new figure
        axes = plt.gca()  # Get current axes

        # Extraction of important data from the x-axis for plotting
        x_labels = np.array(self.df["Monat/Jahr"])  # All X-axis time labels
        x_indexes = np.arange(x_labels.shape[0])  # X-axis indexes
        x_ticks = np.arange(0, len(x_indexes), 6)  # X ticks for plotting
        x_labs_plot = x_labels[x_ticks]  # X labels for plotting

        # Set the title and labels
        axes.set_title(self.title)
        axes.set_xlabel(self.xlabel)
        axes.set_ylabel(self.ylabel)

        # Plot knowing data points
        for feat in features:
            if feat == "Gesamt/Kopf":
                axes.plot(x_indexes, self.df[feat], label=feat, linewidth=2)
            else:
                axes.plot(x_indexes, self.df[feat], label=feat)

        # Customize the plot
        axes.grid(True)
        axes.set_xticks(x_ticks)
        axes.set_xticklabels(x_labs_plot)
        axes.tick_params(axis="x", rotation=70)
        axes.legend(ncol=5)

        if path:
            plt.savefig(path, dpi=self.dpi,
                        bbox_inches="tight", pad_inches=0.25)
            plt.close()
        else:
            plt.tight_layout()
            plt.show()


class PlotSP:
    def __init__(self, df=None, title="Scatter Plot",
                 xlabel="Variable x", ylabel="Variable x", fig_size=(14, 6), dpi=150):
        """
        Initialize the PlotSP class.
        :param df: PD.DATAFRAME containing the data
        :param title: STR with the title of the scatter plot
        :param xlabel: STR with the X-axis label
        :param ylabel: STR with the Y-axis label
        :param fig_size: TUPLE with the figure size for the plot
        """
        self.df = df
        self.title = title
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.figure_size = fig_size
        self.dpi = dpi

    def plot(self, x, y, path=None):
        """
        Plots the time series of different features
        :param x: STR with the name of the x variable
        :param y: STR with the name of the y variable
        :param path: STR with the name of the path to save the plot
        :return: None
        """
        plt.figure(figsize=self.figure_size)  # Create a new figure
        axes = plt.gca()  # Get current axes

        # Set the title and labels
        axes.set_title(self.title)
        axes.set_xlabel(self.xlabel)
        axes.set_ylabel(self.ylabel)

        # Scatter plot
        axes.scatter(x, y, c="C0", s=15,
                     zorder=10)

        # Customize the plot
        axes.grid(True)

        if path:
            plt.savefig(path, dpi=self.dpi,
                        bbox_inches="tight", pad_inches=0.25)
            plt.close()
        else:
            plt.tight_layout()
            plt.show()


