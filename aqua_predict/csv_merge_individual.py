import os
import pandas as pd
from fun import create_directory


def extract_column_from_csvs(csv_paths, column_name, output_csv, output_dir):
    """
    Extracts a specific column from multiple CSV files and merges them into a single CSV file.
    :param csv_paths: List of STR with the file paths for the CSV files to extract the column from.
    :param column_name: STR with the name of the column to be extracted from each CSV file.
    :param output_csv: STR with the name of the output CSV file where the merged data will be saved.
    :param output_dir: STR with the directory where the final merged CSV file will be saved.
    :return: None
    """
    # Create the output directory if not exists
    create_directory(output_dir)

    # List to hold all extracted columns
    extracted_columns_list = []

    counter = 1
    # Iterate over all CSV paths
    for csv_path in csv_paths:
        # Load only the required column from the CSV
        df = pd.read_csv(csv_path, usecols=[column_name])

        # Optionally, rename the column to avoid duplicates in final CSV
        df.columns = [f"{counter}"]
        counter += 1
        # Append the DataFrame to the list
        extracted_columns_list.append(df)

    # Concatenate all columns into a single DataFrame
    merged_df = pd.concat(extracted_columns_list, axis=1)

    # Create a DataFrame for the LaTeX-style values
    latex_values = pd.DataFrame({
        "$k_{Matérn}$": ["$\\nu=0.5$", "$\\nu=1.5$", "$\\nu=2.5$", "$\\nu=\\infty$"]
    })

    # Concatenate the LaTeX values column as the first column with the merged DataFrame
    final_df = pd.concat([latex_values, merged_df], axis=1)

    # Save the merged DataFrame to the output CSV file
    final_df.to_csv(os.path.join(output_dir, output_csv), index=False)

outliers = "w_outliers"
noise = "w_noise"
dataset = "WV14"
columns = ["lh", "r2", "rmse", "nrmse", "mae", "nmae"]
column_to_extract = columns[0]
output_file_name = f"final_results_{column_to_extract}_{outliers}_{noise}_in_{dataset}.csv"

# Example usage:
csv_file_paths = [
    f"../results/individual_feature_analysis/1_NS_Monat/results_of_['NS Monat']_{outliers}_{noise}_in_{dataset}.csv",
    f"../results/individual_feature_analysis/2_T_Monat_Mittel/results_of_['T Monat Mittel']_{outliers}_{noise}_in_{dataset}.csv",
    f"../results/individual_feature_analysis/3_T_Max_Monat/results_of_['T Max Monat']_{outliers}_{noise}_in_{dataset}.csv",
    f"../results/individual_feature_analysis/4_pot_Evap/results_of_['pot Evap']_{outliers}_{noise}_in_{dataset}.csv",
    f"../results/individual_feature_analysis/5_klimat_WB/results_of_['klimat. WB']_{outliers}_{noise}_in_{dataset}.csv",
    f"../results/individual_feature_analysis/6_pos_klimat_WB/results_of_['pos. klimat. WB']_{outliers}_{noise}_in_{dataset}.csv",
    f"../results/individual_feature_analysis/7_Heiße_Tage/results_of_['Heiße Tage']_{outliers}_{noise}_in_{dataset}.csv",
    f"../results/individual_feature_analysis/8_Sommertage/results_of_['Sommertage']_{outliers}_{noise}_in_{dataset}.csv",
    f"../results/individual_feature_analysis/9_Eistage/results_of_['Eistage']_{outliers}_{noise}_in_{dataset}.csv",
    f"../results/individual_feature_analysis/10_T_Min_Monat/results_of_['T Min Monat']_{outliers}_{noise}_in_{dataset}.csv",
    f"../results/group_feature_analysis/all_feats/results_of_all_feats_{outliers}_{noise}_in_{dataset}.csv",
]

DIR_FINAL_RESULTS = f"../final_results/individual_feature_analysis/{column_to_extract}/"

# Extract the column and merge into one CSV
extract_column_from_csvs(csv_file_paths, column_to_extract, output_file_name, DIR_FINAL_RESULTS)
