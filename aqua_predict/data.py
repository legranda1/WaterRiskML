import os
import pandas as pd
import numpy as np
import time
import datetime
from plot import PlotBp, PlotCorr
from config import *


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
        super().__init__(self)
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

    def identify_outliers(self, feature):
        """
        Identify outliers in a numerical dataset using the IQR method
        :param feature: STR with the name of the feature column or dataset
        in which to extract the outliers
        :return: PD.SERIES or NP.ARRAY containing the outliers
        """
        study_data = self.filter_data()[feature].values

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

    def print_outliers(self, feats, show_results=False):
        """
        Prints the outliers for given features in the dataset
        :param feats: STR or LIST with the feature(s) to investigate
        for outliers.
        :param show_results: BOOL containing a flag to indicate whether
        to print detailed results.
        :return: None
        """
        # Ensure feats is a list for consistent processing
        if isinstance(feats, str):
            feats = [feats]

        # Determine the label based on the feature(s)
        label = "output" if feats == ["Gesamt/Kopf"] else "inputs"

        # Initialize a counter for the number of outliers
        count_out = 0

        # Iterate over each feature to identify and print outliers
        for feat in feats:
            # Identify outliers for the current feature
            outliers = self.identify_outliers(feat)

            # If there are outliers, increase the counter and
            # optionally print them
            if len(outliers) > 0:
                count_out += len(outliers)
                if show_results:
                    print(
                        f"There are {len(outliers)} outliers {outliers}"
                        f" in the {label} parameter '{feat}'"
                        f" of the file '{self.xlsx_file_name}'")
            else:
                # Optionally print that no outliers were found
                if show_results:
                    print(
                        f"There are no outliers in the {label} parameter"
                        f" '{feat}' of the file '{self.xlsx_file_name}'")

        # Print the total number of outliers found
        print(
            f"\nThere are a total number of {count_out} outliers in the"
            f" {label} parameter(s) of the file {self.xlsx_file_name}")

    def plot_boxplot(self, feats):
        """
        Plots a boxplot according to the respective features
        :param feats: STR or LIST with the feature(s) to investigate
        for plotting
        :return: A boxplot
        """
        data_filtered = self.filter_data()
        # Determine the label based on the features
        label = "output" if feats == "Gesamt/Kopf" else "inputs"
        boxplot = PlotBp(data_filtered, title=f"Boxplot of {label}",
                         ylabel="Ranges", fig_size=(12, 6))
        return boxplot.plot(feats)

    def plot_heatmap(self, feat1=None, feat2=None):
        """
        Plots a correlation coefficient heatmap for the
        specified features.
        :param feat1: STR of the target variable (output)
        :param feat2: LIST of strings of feature names (inputs)
        :return: A heatmap
        """
        data_filtered = self.filter_data()
        corr_matrix = data_filtered[[feat1] + feat2].corr()
        hm = PlotCorr(corr_matrix, title="Correlation heatmap")
        return hm.plot_hm()

    def plot_pairplot(self, feat1=None, feat2=None):
        """
        Plots a correlation pairplot for the specified features.
        :param feat1: STR of the target variable (output)
        :param feat2: LIST of strings of feature names (inputs)
        :return: A pairplot
        """
        data_filtered = self.filter_data()
        corr_matrix = data_filtered[[feat1] + feat2]
        pp = PlotCorr(corr_matrix, title="Correlation pairplot")
        return pp.plot_pp()


if __name__ == "__main__":
    # File names
    FNAME = "Auswertung WV14 Unteres Elsenztal.xlsx"
    # FNAME = "Auswertung WV25 SW Füssen.xlsx"
    # FNAME = "Auswertung WV69 SW Landshut.xlsx"

    # Parameter(s) to investigate
    # FEAT = COL_TAR
    FEAT = COL_FEAT

    # Flags
    SHOW_CORR = True
    SHOW_OUTLIERS = False

    # Record the initial time for tracking script duration
    init_time = time.time()

    try:
        # Instantiate an object of the DataManager class
        data_handler = DataManager(xlsx_file_name=FNAME)
        if SHOW_CORR:
            # Plot correlation heatmap (internally the data is filtered)
            data_handler.plot_heatmap(COL_TAR, COL_FEAT)
            # Plot a correlation pairplot (internally the data is filtered)
            data_handler.plot_pairplot(COL_TAR, COL_FEAT)
            # Print outliers (internally the data is filtered)
        if SHOW_OUTLIERS:
            data_handler.print_outliers(FEAT, show_results=False)
            # Plot boxplot(s) (internally the data is filtered)
            data_handler.plot_boxplot(FEAT)
    except Exception as e:
        print(f"An error occurred: {e}")

    # Record the end time of the approach
    end_time = time.time()

    # Calculate and print the total running time
    total_time = end_time - init_time  # Calculate the elapsed time
    print(f"\nRunning time: \n{total_time:.2f} seconds / "
          f"{datetime.timedelta(seconds=total_time)}")
