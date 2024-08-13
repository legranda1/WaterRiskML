import pickle

# Define the path to the file
file_path = ("../output_data/with_combinations"
             "/all_features"
             "/gpr_workspace_of_['pos. klimat. WB']_wo_outliers_in_WV14.pkl")

# Load the workspace
with open(file_path, "rb") as file:
    workspace = pickle.load(file)

# Example: accessing a specific dataset or object
data = workspace.get("time")  # Replace with the appropriate key
print(data)
