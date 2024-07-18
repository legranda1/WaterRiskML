import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import seaborn as sns


class PlotBp:
    def __init__(self, df=None, title="Boxplot of inputs",
                 ylabel="Ranges", fig_size=(14, 6)):
        """
        Initialize the PlotBp class.
        :param df: PD.DATAFRAME containing the data
        :param title: STR with the title of the boxplot
        :param ylabel: STR with the Y-axis label
        :param fig_size: TUPLE with the figure size for the plot
        """
        self.df = df
        self.title = title
        self.ylabel = ylabel
        self.fig_size = fig_size

    def plotdf(self, features):
        """
        Generate and display a boxplot of a dataframe for the
        specified features
        :param features: LIST of strings with the specified features
        :return: None
        """
        plt.figure(figsize=self.fig_size)  # Create a new figure
        axes = plt.gca()  # Get current axes

        # Set the title and labels
        axes.set_title(self.title)
        axes.set_ylabel(self.ylabel)

        # Plot boxplot
        self.df.boxplot(column=features)
        plt.show()

    def plot(self, features, units):
        """
        Generate and display a boxplot for the specified features
        :param features: LIST of strings with the specified features
        :param units: LIST of strings with the units corresponding to each feature
        :return: None
        """
        plt.figure(figsize=self.fig_size)  # Create a new figure
        axes = plt.gca()  # Get current axes

        # Set the title and labels
        axes.set_title(self.title)
        axes.set_ylabel(self.ylabel)

        # Prepare data for boxplot
        if isinstance(features, str):
            features = [features]
        if isinstance(units, str):
            units = [units]
        # Ensure features and units lists are the same length
        assert len(features) == len(units), "Features and units must have the same length"

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
            axes.annotate(unit, xy=(i, -0.05), xycoords=("data", "axes fraction"),
                          xytext=(0, -15), textcoords="offset points",
                          ha="center", va="bottom", fontsize=10, color="black")

        plt.show()


class PlotCorr:
    def __init__(self, df=None, title="Correlatiom"):
        """
        Initialize the PlotCorr class.
        :param df: PD.DATAFRAME containing the filtered data
        :param title: STR with the title of the heatmap
        :param fig_size: TUPLE with the figure size for the plot
        """
        self.df = df
        self.title = title

    def plot_hm(self):
        """
        Generate and display a correlation coefficient
        heatmap for the specified features
        :return: None
        """
        plt.figure(figsize=(8, 6))
        # Plot heat map
        sns.heatmap(self.df, annot=True, fmt=".2f", cmap="coolwarm",
                    cbar=True, square=True, linewidths=0.5)
        plt.suptitle(self.title)
        plt.tight_layout()
        plt.show()

    def plot_pp(self):
        """
        Generate and display a correlation pairplot for the
        specified features
        :return: None
        """
        # Plot pairplot
        pp = sns.pairplot(self.df, markers=".", plot_kws={"linewidth": 0.5})

        # Delete the for loop to have vertical axis labels again
        # Iterate through each axes object and set the y-axis labels
        # horizontally
        for ax in pp.axes.flatten():
            if ax is not None:
                ax.tick_params(axis='x', labelsize=8)
                ax.tick_params(axis='y', labelsize=8)
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
        plt.tight_layout()
        plt.show()


class PlotGPR:
    def __init__(self, title="GPR", xlabel="Time", ylabel="Water Demand",
                 std=1.96, fig_size=(14, 6)):
        """
        Initialize the PlotGPR class.
        :param title: STR with the title of the GPR
        :param xlabel: STR with the X-axis label
        :param ylabel: STR with the Y-axis label
        :param std: FLOAT with the standard deviation value
        :param fig_size: TUPLE with the figure size for the plot
        """
        self.title = title
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.std = std
        self.figure, self.axes = plt.subplots(figsize=fig_size)

    def set_xticks(self, ticks, labels):
        """
        Sets the x-axis ticks and labels
        :param ticks: NP.ARRAY with the positions of the ticks on the
        x-axis
        :param labels: NP.ARRAY with labels corresponding to the tick
        positions
        :return: None
        """
        self.axes.set_xticks(ticks)
        self.axes.set_xticklabels(labels)

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
        self.axes.tick_params(axis="x", rotation=70)
        self.axes.legend(["Training Data", "Test Data",
                          "GP Mean",
                          f"GP conf interval ({self.std} std)"])
        self.set_xticks(x_ticks, x_labs_plot)

        # Show the plot
        plt.show()

