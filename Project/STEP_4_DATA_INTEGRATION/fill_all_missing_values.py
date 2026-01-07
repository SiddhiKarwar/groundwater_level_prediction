import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime

# Relative paths
master_csv = r"formalize_dataset\master_dataset.csv"
gw_model_path = r"formalize_dataset\models\groundwater_model.pkl"
rain_model_path = r"formalize_dataset\models\rainfall_model.pkl"
river_model_path = r"formalize_dataset\models\river_level_model.pkl"
temp_model_path = r"formalize_dataset\models\temperature_model.pkl"
output_folder = os.path.dirname(master_csv)

# Load data and models
df = pd.read_csv(master_csv)
df['Data Time'] = pd.to_datetime(df['Data Time'], format='mixed', dayfirst=False, errors='coerce')
# Drop rows where Data Time conversion failed
df = df.dropna(subset=['Data Time'])

gw_model = joblib.load(gw_model_path)
rain_model = joblib.load(rain_model_path)
river_model = joblib.load(river_model_path)
temp_model = joblib.load(temp_model_path)

# Function to predict missing values
def predict_missing_values(df, model, target_col, unit):
    mask = df[target_col].isna()
    if mask.any():
        df.loc[mask, 'Year'] = df.loc[mask, 'Data Time'].dt.year
        df.loc[mask, 'Month'] = df.loc[mask, 'Data Time'].dt.month
        df.loc[mask, 'Day'] = df.loc[mask, 'Data Time'].dt.day
        df.loc[mask, 'Hour'] = df.loc[mask, 'Data Time'].dt.hour
        X_pred = df.loc[mask, ['Latitude', 'Longitude', 'Year', 'Month', 'Day', 'Hour']]
        predicted = model.predict(X_pred)
        df.loc[mask, target_col] = predicted
        df.loc[mask, 'Unit'] = unit  # Update unit for predicted values if needed

# Predict missing values for each parameter
predict_missing_values(df, gw_model, 'Groundwater_Level', 'm')
predict_missing_values(df, rain_model, 'Rainfall', 'mm')
predict_missing_values(df, river_model, 'River_Water_Level', 'm')
predict_missing_values(df, temp_model, 'Temperature', 'ºC')

# Save final updated dataset
final_csv = os.path.join(output_folder, 'final_master_dataset.csv')
df.to_csv(final_csv, index=False)
print(f"Final master dataset saved: {final_csv}")