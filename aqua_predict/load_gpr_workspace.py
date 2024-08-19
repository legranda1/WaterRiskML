import pickle

# Define the path to the file
file_path = ("../gpr_output_data/combined_feature_analysis"
             "/stratified_selection/selected_feats_minc"
             "/gpr_workspace_of_['T Monat Mittel' 'NS Monat']_tar_wo_outliers_in_WV14.pkl")

# Load the workspace
with open(file_path, "rb") as file:
    workspace = pickle.load(file)

# Example: accessing a specific dataset or object
data = workspace.get("best_par_set")  # Replace with the appropriate key
print(data)
