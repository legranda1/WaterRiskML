import os
import pandas as pd
from scipy.stats import iqr
from config import *
from plot import *


class InputReader:
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

    def identify_outliers(self, feature, label=""):
        """
        Identify outliers in a numerical dataset using the IQR method
        :param feature: STR with the name of the feature column or dataset
        in which to extract the outliers
        :return: PD.SERIES or NP.ARRAY containing the outliers
        """
        data = self.filter_data()[feature].values

        # Calculate Q1 (25th percentile) and Q3 (75th percentile)
        q25, q75 = np.percentile(data, 25), np.percentile(data, 75)
        iqr_value = q75 - q25

        # Calculate the outlier bounds
        lower_bound = q25 - 1.5 * iqr_value
        upper_bound = q75 + 1.5 * iqr_value

        # Identify outliers by using "boolean indexing"
        outliers = data[(data < lower_bound) | (data > upper_bound)]
        return outliers


if __name__ == "__main__":
    # File names
    # FNAME = "Auswertung WV14 Unteres Elsenztal.xlsx"
    # FNAME = "Auswertung WV25 SW Füssen.xlsx"
    FNAME = "Auswertung WV69 SW Landshut.xlsx"

    # Parameter to investigate
    # FEAT = COL_TAR
    FEAT = COL_FEAT

    label = "output" if FEAT == "Gesamt/Kopf" else "inputs"

    try:
        # Data filtered
        data = InputReader(xlsx_file_name=FNAME)
        data_filtered = data.filter_data()

        # Show boxplot
        boxplot = PlotBp(data_filtered, title=f"Boxplot of {label}",
                         ylabel="Ranges", fig_size=(12, 6))
        boxplot.plot(FEAT)

        # Outliers
        outliers = data.identify_outliers(FEAT) if label == "output"\
            else data.identify_outliers(FEAT[0])
        outliers_feature = FEAT if label == "output" else FEAT[0]
        print(f"There are {len(outliers)} outliers {outliers} in the parameter"
              f" '{outliers_feature}' of the file '{FNAME}'")
    except Exception as e:
        print(f"An error occurred: {e}")

