import os
import pandas as pd
from fun import create_directory
from itertools import product  # To easily generate combinations of outliers, noise, and dataset


def extract_and_merge_column_from_csvs(csv_paths, column_name, output_csv, output_dir):
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

test_ranges = ["test_range_1", "test_range_2"]
outliers = ["w_outliers", "wo_outliers", "tar_wo_outliers"]
noise = ["w_noise", "wo_noise"]
dataset = ["WV14", "WV25", "WV69"]
columns = ["lh", "r2", "rmse", "nrmse", "mae", "nmae"]

# Iterate over all columns
for column_to_extract in columns:
    # Iterate over all combinations of outliers, noise, and dataset
    for t_r, o, n, d in product(test_ranges, outliers, noise, dataset):
        # Define the output file name for this combination
        output_file_name = f"final_results_{column_to_extract}_{t_r}_{o}_{n}_in_{d}.csv"

        # Generate the paths for the CSV files based on the current combination
        csv_file_paths = [
            f"../results/individual_feature_analysis/1_NS_Monat/{t_r}/results_of_['NS Monat']_{o}_{n}_in_{d}.csv",
            f"../results/individual_feature_analysis/2_T_Monat_Mittel/{t_r}/results_of_['T Monat Mittel']_{o}_{n}_in_{d}.csv",
            f"../results/individual_feature_analysis/3_T_Max_Monat/{t_r}/results_of_['T Max Monat']_{o}_{n}_in_{d}.csv",
            f"../results/individual_feature_analysis/4_pot_Evap/{t_r}/results_of_['pot Evap']_{o}_{n}_in_{d}.csv",
            f"../results/individual_feature_analysis/5_klimat_WB/{t_r}/results_of_['klimat. WB']_{o}_{n}_in_{d}.csv",
            f"../results/individual_feature_analysis/6_pos_klimat_WB/{t_r}/results_of_['pos. klimat. WB']_{o}_{n}_in_{d}.csv",
            f"../results/individual_feature_analysis/7_Heiße_Tage/{t_r}/results_of_['Heiße Tage']_{o}_{n}_in_{d}.csv",
            f"../results/individual_feature_analysis/8_Sommertage/{t_r}/results_of_['Sommertage']_{o}_{n}_in_{d}.csv",
            f"../results/individual_feature_analysis/9_Eistage/{t_r}/results_of_['Eistage']_{o}_{n}_in_{d}.csv",
            f"../results/individual_feature_analysis/10_T_Min_Monat/{t_r}/results_of_['T Min Monat']_{o}_{n}_in_{d}.csv",
            f"../results/group_feature_analysis/all_feats/{t_r}/results_of_all_feats_{o}_{n}_in_{d}.csv",
        ]

        # Define the directory where the final results will be saved
        DIR_FINAL_RESULTS = f"../final_results/individual_feature_analysis/{column_to_extract}/{t_r}/"

        # Extract the column and merge into one CSV
        extract_and_merge_column_from_csvs(csv_file_paths, column_to_extract, output_file_name, DIR_FINAL_RESULTS)
