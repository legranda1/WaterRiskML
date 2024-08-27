import pickle

# Define the path to the file
file_path = (f"../gpr_output_data/combined_feature_analysis/brute_force_selection"
             f"/gpr_workspace_of_['T Max Monat' 'pot Evap' 'klimat. WB' 'Eistage']_w_outliers_in_WV14.pkl")

# Load the workspace
with open(file_path, "rb") as file:
    workspace = pickle.load(file)

# Example: accessing a specific dataset or object
data = workspace.get("best_par_set")
print(data)
