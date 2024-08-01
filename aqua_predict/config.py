# Define the column name for the target variable (output)
COL_TAR = "Gesamt/Kopf"  # Monthly water demand

UNIT_TAR = "[L/(C*d)]"

# Define a list of feature column names (inputs)
COL_FEAT = [
    "NS Monat",         # Monthly precipitation
    "T Monat Mittel",   # Average temperature of the month
    "T Max Monat",      # Maximum temperature of the month
    "pot Evap",         # Potential evaporation
    "klimat. WB",       # Climatic water balance
    "pos. klimat. WB",  # Positive climatic water balance
    "Heiße Tage",       # Number of hot days (peak temp. greater than
                        # or equal to 30 °C)
    "Sommertage",       # Number of summer days (peak temp. greater
                        # than or equal to 25 °C)
    "Eistage",          # Number of ice days
    "T Min Monat"       # Minimum temperature of the month
]

UNIT_FEAT = [
    "[mm]",
    "[°C]",
    "[°C]",
    "[mm]",
    "[mm]",
    "[mm]",
    "[-]",
    "[-]",
    "[-]",
    "[°C]"
]

COL_ALL = [
    "Gesamt/Kopf",  # Monthly water demand
    "NS Monat",         # Monthly precipitation
    "T Monat Mittel",   # Average temperature of the month
    "T Max Monat",      # Maximum temperature of the month
    "pot Evap",         # Potential evaporation
    "klimat. WB",       # Climatic water balance
    "pos. klimat. WB",  # Positive climatic water balance
    "Heiße Tage",       # Number of hot days (peak temp. greater than
                        # or equal to 30 °C)
    "Sommertage",       # Number of summer days (peak temp. greater
                        # than or equal to 25 °C)
    "Eistage",          # Number of ice days
    "T Min Monat"       # Minimum temperature of the month
]

UNIT_ALL = [
    "[L/(C*d)]",
    "[mm]",
    "[°C]",
    "[°C]",
    "[mm]",
    "[mm]",
    "[mm]",
    "[-]",
    "[-]",
    "[-]",
    "[°C]"
]

FNAMES = [
    "Auswertung WV14 Unteres Elsenztal.xlsx",
    "Auswertung WV25 SW Füssen.xlsx",
    "Auswertung WV69 SW Landshut.xlsx"
]
