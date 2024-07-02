import pandas as pd

# Define the directory for the input data
DIR_DATA = "../input_data/"

# Define the directory for the xlsx files within the data directory
DIR_XLSX = DIR_DATA + "xlsx/"

# Define the column name for the target variable (output)
COL_TAR = "Gesamt/Kopf"  # Monthly water demand

# Define a list of feature column names (inputs)
COL_FEAT = [
    "NS Monat",         # Monthly precipitation
    "T Monat Mittel",   # Average temperature of the month
    "T Max Monat",      # Maximum temperature of the month
    "pot Evap",         # Potential evaporation
    "klimat. WB",       # Climatic water balance
    "pos. klimat. WB",  # Positive climatic water balance
    "Heiße Tage",       # Number of hot days (peak temp. greater than or equal to 30 °C)
    "Sommertage",       # Number of summer days (peak temp. greater than or equal to 25 °C)
    "Eistage",          # Number of ice days
    "T Min Monat"       # Minimum temperature of the month
]
