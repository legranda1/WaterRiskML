import pickle
import numpy as np
import pandas as pd
from fun import create_directory

DIR_RESULTS = "../results/individual_feature_analysis/1_NS_Monat/"
outliers = "w_outliers"
noise = "w_noise"
feat = "NS Monat"
code_name = "WV14"

# Define the path to the file
file_path = (f"../gpr_output_data/individual_feature_analysis/1_NS_Monat"
             f"/gpr_workspace_of_['{feat}']_{outliers}_{noise}_in_{code_name}.pkl")

# Load the workspace
with open(file_path, "rb") as file:
    workspace = pickle.load(file)

# Example: accessing a specific dataset or object
data = workspace.get("all_par_sets_updated")  # Replace with the appropriate key

# Extract LML scores from each parameter set
lml_scores = np.array([par_set.marg_lh
                       for par_set in data])
# Extract R2 scores from each parameter set
r2_test_scores = np.array([par_set.r2_test
                           for par_set in data])
# Extract RMSE scores from each parameter set
rmse_test_scores = np.array([par_set.rmse_test
                             for par_set in data])
# Extract NRMSE scores from each parameter set
nrmse_test_scores = np.array([par_set.nrmse_test
                             for par_set in data])
# Extract MAE scores from each parameter set
mae_test_scores = np.array([par_set.mae_test
                            for par_set in data])
# Extract NMAE scores from each parameter set
nmae_test_scores = np.array([par_set.nmae_test
                            for par_set in data])

# Create a DataFrame with the results
result_df = pd.DataFrame({
    "lh": lml_scores,
    "r2": r2_test_scores,
    "rmse": rmse_test_scores,
    "nrmse": nrmse_test_scores,
    "mae": mae_test_scores,
    "nmae": nmae_test_scores
})

# Round numerical values to 2 decimal places
result_df = result_df.round(2)

# Save DataFrame to a CSV file
create_directory(DIR_RESULTS)
result_df.to_csv(f"{DIR_RESULTS}/results_of_{feat}_{outliers}_"
                 f"{noise}_in_{code_name}.csv", index=False)
