import pandas as pd
from fun import create_directory

DIR_FINAL_RESULTS = "../final_results/individual_feature_analysis/1_NS_Monat/"

# Step 1: Specify the exact location of the CSV file
csv_file_path = ("../results/individual_feature_analysis/1_NS_Monat/results_of_['NS "
                 "Monat']_wo_outliers_w_noise_in_WV14.csv")

# Load the CSV into a DataFrame
df = pd.read_csv(csv_file_path)

# Step 2: Extract specific columns by their names
# Replace 'column_1', 'column_2' with the actual column names
extracted_columns = df[["rmse"]]

# Step 3: Merge the extracted columns (this step depends on what you mean by merging)
# If you're simply extracting and saving, skip this step, but if you need to concatenate, do so:
# merged_df = pd.concat([extracted_columns['column_1'], extracted_columns['column_2']], axis=1)

# Step 4: Save the new DataFrame to a new CSV file
create_directory(DIR_FINAL_RESULTS)
extracted_columns.to_csv(f"{DIR_FINAL_RESULTS}final_results_rmse_in_WV14.csv", index=False)
