import os
import pandas as pd
import numpy as np
import time
import datetime
from plot import PlotBp, PlotCorr, PlotTS
from config import *
from fun import *


def identify_outliers(feature, data):
    """
    Identify outliers in a numerical dataset using the IQR method
    :param feature: STR with the name of the feature column or dataset
    in which to extract the outliers
    :param data: PD.DATAFRAME with the dataset to clean
    :return: NP.ARRAY containing the outliers
    """
    study_data = data[feature].values  # Gives a np.array

    # Calculate Q1 (25th percentile) and Q3 (75th percentile)
    q25, q75 = (np.percentile(study_data, 25),
                np.percentile(study_data, 75))
    iqr_value = q75 - q25

    # Calculate the outlier bounds
    lower_bound = q25 - 1.5 * iqr_value
    upper_bound = q75 + 1.5 * iqr_value

    # Identify outliers by using "boolean indexing"
    study_outliers = study_data[
        (study_data < lower_bound) | (study_data > upper_bound)
    ]
    return study_outliers


def clean_data(feature, data):
    """
    Cleans the data by removing outliers from the specified feature.
    :param data: PD.DATAFRAME with the dataset to clean
    :param feature: STR with the name of the feature column or dataset
    in which to extract the outliers
    :return: PD.DATAFRAME containing the data without outliers
    """
    if isinstance(feature, str):
        # Convert to list if a single feature is provided
        feature = [feature]

    # Remove rows where any of the feature columns contain outlier
    # values
    for feat in feature:
        if feat not in ("Heiße Tage", "Sommertage", "Eistage"):
            # Identify outlier values for the current feature
            outliers_values = identify_outliers(feat, data)
            # Remove rows where the current feature column contains
            # outlier values
            data = data[~data[feat].isin(outliers_values)]

    # Return the cleaned data
    return data


def clean_tar(target, data):
    """
    Cleans the data by removing outliers from the target.
    :param data: PD.DATAFRAME with the dataset to clean
    :param target: STR with the name of the target column
    in which to extract the outliers
    :return: PD.DATAFRAME containing the data without outliers
    """
    # Identify outlier values in the target
    outliers_values = identify_outliers(target, data)
    # Remove rows where the target column contains
    # outlier values
    data = data[~data[target].isin(outliers_values)]

    # Return the cleaned data
    return data


def plot_timeseries(data, feats, file_name, path=None):
    """
    Plots a time series plot for the specified features.
    :param feats: LIST of strings of feature names (inputs)
    :param data: PD.DATAFRAME with the dataset to plot
    :param file_name: STR with the code name of the file
    :param path: STR with the name of the path to save the plot
    :return: A timeseries plot
    """
    ts = PlotTS(
        data,
        title=f"Time series of {file_name}",
        xlabel="Time", ylabel="Ranges", fig_size=(15, 8),
        dpi=150
    )
    if path:
        return ts.plot(feats, path)
    else:
        return ts.plot(feats)


def plot_boxplot(feats, units, data, file_name, path=None):
    """
    Plots a boxplot according to the respective features
    :param feats: STR or LIST with the feature(s) to investigate for
    plotting
    :param units: LIST with the units corresponding to each feature
    :param data: PD.DATAFRAME with the dataset to plot
    :param file_name: STR with the code name of the file
    :param path: STR with the name of the path to save the plot
    :return: A boxplot object
    """
    boxplot = PlotBp(data, title=f"Boxplot of {file_name}",
                     ylabel="Ranges", fig_size=(15, 8),
                     dpi=150)
    if path:
        return boxplot.plot(feats, units, path)
    else:
        return boxplot.plot(feats, units)


def plot_heatmap(data, feat1=None, feat2=None, file_name=None, path=None):
    """
    Plots a correlation coefficient heatmap for the
    specified features.
    :param data: PD.DATAFRAME with the dataset to plot
    :param feat1: STR of the target variable (output)
    :param feat2: LIST of strings of feature names (inputs)
    :param file_name: STR with the code name of the file
    :param path: STR with the name of the path to save the plot
    :return: A heatmap
    """
    corr_matrix = data[[feat1] + feat2].corr()
    hm = PlotCorr(corr_matrix, title=f"Correlation heatmap of {file_name}",
                  fig_size=(8, 6), dpi=150)
    if path:
        return hm.plot_hm(path)
    else:
        return hm.plot_hm()


def plot_pairplot(data, feat1=None, feat2=None, file_name=None, path=None):
    """
    Plots a correlation pairplot for the specified features.
    :param data: PD.DATAFRAME with the dataset to plot
    :param feat1: STR of the target variable (output)
    :param feat2: LIST of strings of feature names (inputs)
    :param file_name: STR with the code name of the file
    :param path: STR with the name of the path to save the plot
    :return: A pairplot
    """
    corr_matrix = data[[feat1] + feat2]
    pp = PlotCorr(corr_matrix, title=f"Correlation pairplot of {file_name}",
                  fig_size=(20, 10), dpi=150)
    if path:
        return pp.plot_pp(path)
    else:
        return pp.plot_pp()


def is_highly_correlated(feature, selected_features,
                         corr_matrix):
    """
    Checks if a given feature is highly correlated with any of the
    already selected features
    :param feature: STR with the name of the feature to be checked for
    high correlation
    :param selected_features: LIST of strings containing the already
    selected features
    :param corr_matrix: PD.DATAFRAME containing correlation
    coefficients between features
    :return: BOOL which returns True if the feature is highly
    correlated with any of the selected features, otherwise False
    """
    # Iterate over each selected feature
    for selected_feature in selected_features:
        # Check if the absolute correlation between the current feature
        # and any selected feature exceeds the threshold
        if abs(corr_matrix.loc[feature, selected_feature]) > 0.7:
            return True
    return False


def is_lowly_correlated(feature, selected_features, corr_matrix):
    """
    Checks if a given feature is lowly correlated with any of the
    already selected features
    :param feature: STR with the name of the feature to be checked for
    low correlation
    :param selected_features: LIST of strings containing the already
    selected features
    :param corr_matrix: PD.DATAFRAME containing correlation coefficients
    between features
    :return: BOOL which returns True if the feature is lowly correlated
    with all selected features, otherwise False
    """
    # If no features are selected yet, consider the feature as not
    # correlated
    if not selected_features:
        return True

    # Iterate over each selected feature
    for selected_feature in selected_features:
        # Check if the absolute correlation between the current feature and any selected feature
        # is non-zero (i.e., there is some correlation, even if small)
        if abs(corr_matrix.loc[feature, selected_feature]) > 0.3:
            return False
    return True


def selected_features(data, feat1=None, feat2=None, prioritize_feature=None):
    """
    Selects features based on their correlation with a target feature
    and ensures a specific feature is always listed first.
    :param data: PD.DATAFRAME with the dataset to plot
    :param feat1: STR of the target variable (output)
    :param feat2: LIST of strings of feature names (inputs)
    :param threshold: FLOAT above which two features are considered
    highly correlated
    :param prioritize_feature: STR with the feature to prioritize
    :return: LIST of strings with the selected_features
    """
    # Compute the correlation matrix
    corr_matrix = data[[feat1] + feat2].corr()
    # Sort features by their correlation with the target feature
    target_corr = corr_matrix[feat1].drop(feat1).abs().sort_values(ascending=False)

    # Initialize the list of selected features
    selected_features = []
    # Add the prioritized feature if it's in the list
    if prioritize_feature and prioritize_feature in target_corr.index:
        selected_features.append(prioritize_feature)
        # Remove the prioritized feature from the target_corr index to avoid re-adding
        target_corr = target_corr.drop(prioritize_feature)

    # Iterate through the sorted features
    for feature in target_corr.index:
        # If the feature is not highly correlated with any of the
        # selected features
        if is_lowly_correlated(feature, selected_features, corr_matrix):
            selected_features.append(feature)

    return selected_features


def process_data(data, wv_label, suffix, show_boxplots, show_timeseries,
                 show_corr, show_sel_feats, save_plot):
    """
    Processes the given data by generating and optionally saving
    various plots based on provided flags
    :param data: PD.DATAFRAME containing the data to plot and analyze
    :param wv_label: STR indicating the code name of the water supply
    company
    :param suffix: STR for differentiating if the data contains
    outliers or not
    :param show_boxplots: BOOL indicating whether to display
    the boxplots
    :param show_timeseries: BOOL indicating whether to display the
    timeseries plots
    :param show_corr: BOOL indicating whether to display the heatmaps
    and pairplots
    :param show_sel_feats: BOOL indicating whether to display the
    selected features based on correlation or other criteria
    :param save_plot: BOOL indicating whether to save the current plot
    :return: None
    """
    if show_boxplots:
        print(data)
        data_handler.print_outliers(FEAT[0], data,
                                    show_results=True)
        plot_boxplot(FEAT[0], FEAT[1], data, wv_label)
        if save_plot:
            create_directory(DIR_BOXPLOTS)
            path = f"{DIR_BOXPLOTS}/bp_{suffix}_in_{wv_label}.png"
            plot_boxplot(FEAT[0], FEAT[1], data, wv_label, path)
    if show_timeseries:
        plot_timeseries(data, FEAT[0], wv_label)
        if save_plot:
            create_directory(DIR_TIMESERIES)
            path = f"{DIR_TIMESERIES}/ts_{suffix}_in_{wv_label}.png"
            plot_timeseries(data, FEAT[0], wv_label, path)
    if show_corr:
        plot_heatmap(data, COL_TAR, COL_FEAT, wv_label)
        if save_plot:
            create_directory(DIR_HEATMAPS)
            path = f"{DIR_HEATMAPS}/hm_{suffix}_in_{wv_label}.png"
            plot_heatmap(data, COL_TAR, COL_FEAT, wv_label, path)
        plot_pairplot(data, COL_TAR, COL_FEAT, wv_label)
        if save_plot:
            create_directory(DIR_PAIRPLOTS)
            path = f"{DIR_PAIRPLOTS}/pp_{suffix}_in_{wv_label}.png"
            plot_pairplot(data, COL_TAR, COL_FEAT, wv_label, path)
    if show_sel_feats:
        final_features = selected_features(
            data, COL_TAR, COL_FEAT, prioritize_feature="T Monat Mittel"
        )
        print(f"Selected features {suffix}: {final_features}")


class DataManager(PlotBp, PlotCorr):
    def __init__(self,
                 xlsx_file_name="Auswertung WV14 Unteres Elsenztal.xlsx",
                 sheets=None, input_dir="./input_data/xlsx/"):
        """
        Initializes an InputReader object with the given attributes
        and methods
        :param xlsx_file_name: STR of the corresponding .xlsx file name
        :param sheets: LIST of relevant sheets
        :param input_dir: STR of the input directory path
        :return: None
        """
        super().__init__()
        self.xlsx_file_name = xlsx_file_name
        self.sheets = sheets or [
            "Monatsmengen", "Tagesspitzentemperatur", "Tagesmitteltemperatur"
        ]
        self.input_dir = input_dir
        self.climate_params = pd.DataFrame()
        self.load_input_data()

    def load_input_data(self):
        """
        Reads input data from an Excel file and stores it in the
        climate_params DataFrame
        :return: None
        :raises FileNotFoundError: If the specified file does not exist
        """
        file_path = os.path.join(self.input_dir, self.xlsx_file_name)
        if not os.path.exists(file_path):
            raise FileNotFoundError(
                f"The file {self.xlsx_file_name} "
                f"was not found in {self.input_dir}."
            )
        try:
            self.climate_params = pd.read_excel(
                file_path, sheet_name=self.sheets
            )
        except Exception as e:
            raise RuntimeError(
                f"An error occurred while reading the Excel file: {e}"
            )

    def filter_data(self):
        """
        Processes the input data to compute monthly summaries and
        features for temperature extremes and counts of specific types
        of days.
        :return: DATAFRAME with processed data containing monthly
        summaries, including counts of hot days, summer days, ice days,
        and minimum temperatures.
        """
        # Key sheet names in the input data
        ks_month = "Monatsmengen"  # Monthly amounts
        ks_d_peak = "Tagesspitzentemperatur"  # Daily peak temperature
        ks_d_mean = "Tagesmitteltemperatur"  # Daily mean temperature

        # Key column names for various measurements in the input data
        kc_year = "Jahr"
        kc_month = "Monat"
        kc_hot_d = "Heiße Tage"
        kc_summer_d = "Sommertage"
        kc_ice_d = "Eistage"
        kc_tmin = "Min"

        # Verify that the required sheets and columns exist in the data
        required_sheets = {ks_month, ks_d_peak, ks_d_mean}
        missing_sheets = required_sheets - set(self.climate_params.keys())
        if missing_sheets:
            raise KeyError(
                f"Missing required sheets in the input data: {missing_sheets}"
            )

        required_columns = {
            kc_year, kc_month, kc_hot_d, kc_summer_d, kc_ice_d
        }
        missing_columns = required_columns - set(
            self.climate_params[ks_d_peak].columns
        )
        if missing_columns:
            raise KeyError(
                f"Missing required columns in {ks_d_peak}: {missing_columns}"
            )

        if kc_tmin not in self.climate_params[ks_d_mean].columns:
            raise KeyError(
                f"Missing required column {kc_tmin} in {ks_d_mean}"
            )

        # Aggregate the counts of hot, summer, and ice days
        # by year and month.
        monthly_counts = self.climate_params[ks_d_peak].groupby(
            [kc_year, kc_month], as_index=False)[
            [kc_hot_d, kc_summer_d, kc_ice_d]].sum()

        # Extract and rename the minimum temperature column
        monthly_tmin = self.climate_params[ks_d_mean][[kc_tmin]].rename(
            columns={kc_tmin: "T Min Monat"})

        # Concatenate the processed data with the monthly amounts,
        # counts of specific days, and minimum temperatures.
        # df.dropna(axis=1) will drop all columns where any row has a
        # missing value (NaN).
        filtered_df = pd.concat(
            [self.climate_params[ks_month].dropna(axis=1),
             monthly_counts[[kc_hot_d, kc_summer_d, kc_ice_d]],
             monthly_tmin], axis=1).dropna()

        return filtered_df

    def iterative_cleaning(self, feature):
        """
         Iteratively cleans the data to remove outliers from the
        specified feature.
        :param feature: STR containing the name of the feature column
        to clean
        :return: PD.DATAFRAME containing the cleaned data
        """
        # Filter the initial data
        data = self.filter_data()

        while True:
            initial_len = len(data)
            # Clean the data to remove outliers from the
            # specified feature
            data = clean_data(feature, data)
            # Check if the length of the data has not changed
            if len(data) == initial_len:
                # No more outliers detected, break the loop
                break

        return data

    def print_outliers(self, feats, data, show_results=True):
        """
        Prints the outliers for given features in the dataset
        :param feats: STR or LIST with the feature(s) to investigate
        for outliers.
        :param data: PD.DATAFRAME with the dataset to clean
        :param show_results: BOOL containing a flag to indicate whether
        to print detailed results.
        :return: None
        """
        # Ensure feats is a list for consistent processing
        if isinstance(feats, str):
            feats = [feats]

        # Initialize a counter for the number of outliers
        count_out = 0

        # Iterate over each feature to identify and print outliers
        for feat in feats:
            if feat not in ("Heiße Tage", "Sommertage", "Eistage"):
                outliers = identify_outliers(feat, data)
                # If there are outliers, increase the counter and
                # optionally print them
                if len(outliers) > 0:
                    count_out += len(outliers)
                    if show_results:
                        print(
                            f"- There are {len(outliers)} outliers {outliers}"
                            f" in the parameter '{feat}'"
                            f" of the file '{self.xlsx_file_name}'")

        # Print the total number of outliers found
        print(
            f"- There are a total number of {count_out} outliers "
            f"in the file {self.xlsx_file_name}\n")


if __name__ == "__main__":
    # Record the initial time for tracking script duration
    init_time = time.time()

    # File names
    FNAME = FNAMES[0]

    # Parameter(s) to investigate
    # FEAT = [COL_TAR, UNIT_TAR]
    # FEAT = [COL_FEAT, UNIT_FEAT]
    FEAT = [COL_ALL, UNIT_ALL]

    # Directories to create
    DIR_BOXPLOTS = "../plots/bp/"
    DIR_TIMESERIES = "../plots/ts/"
    DIR_HEATMAPS = "../plots/hm/"
    DIR_PAIRPLOTS = "../plots/pp/"

    # Flags
    SHOW_BOXPLOTS = False
    SHOW_TIMESERIES = False
    SHOW_CORR = False
    SHOW_SEL_FEATS = True
    SAVE_PLOT = False

    try:
        # Instantiate an object of the DataManager class
        data_handler = DataManager(xlsx_file_name=FNAME)
        initial_data = data_handler.filter_data()
        wv_number = str(int(initial_data["WVU Nr. "].iloc[0]))
        wv_label = f"WV{wv_number}"
        # Uncomment if you only want to clean the target once (i.e. no iterative process)
        # cleaned_target = clean_tar("Gesamt/Kopf", initial_data)
        cleaned_target = data_handler.iterative_cleaning("Gesamt/Kopf")
        cleaned_data = data_handler.iterative_cleaning(FEAT[0])

        # Initial data
        process_data(initial_data, wv_label, "w_outliers", SHOW_BOXPLOTS,
                     SHOW_TIMESERIES, SHOW_CORR, SHOW_SEL_FEATS, SAVE_PLOT)
        # Target cleaned
        process_data(cleaned_target, wv_label, "tar_wo_outliers", SHOW_BOXPLOTS,
                     SHOW_TIMESERIES, SHOW_CORR, SHOW_SEL_FEATS, SAVE_PLOT)
        # All data cleaned
        process_data(cleaned_data, wv_label, "wo_outliers", SHOW_BOXPLOTS,
                     SHOW_TIMESERIES, SHOW_CORR, SHOW_SEL_FEATS, SAVE_PLOT)

    except Exception as e:
        print(f"An error occurred: {e}")

    # Record the end time of the approach
    end_time = time.time()

    # Calculate and print the total running time
    total_time = end_time - init_time  # Calculate the elapsed time
    print(f"\nRunning time: \n{total_time:.2f} seconds / "
          f"{datetime.timedelta(seconds=total_time)}")
