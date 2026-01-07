import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import joblib
import os
from datetime import datetime, timedelta

# Hardcoded paths
locations_csv = r"locations_equal_final.csv"
base_csv = r"formalize_dataset\Groundwater_Level\combined_groundwater_levels.csv"
output_folder = os.path.dirname(base_csv)

# Load data
locations_df = pd.read_csv(locations_csv)
base_df = pd.read_csv(base_csv)

# Load trained model
model_path = r"formalize_dataset\models\groundwater_model.pkl"
model = joblib.load(model_path)

# Reference station data (using Imd nashik_1 as baseline)
ref_station = base_df[base_df['Station Name'] == 'Imd nashik_1']
ref_times = pd.to_datetime(ref_station['Data Time'])
start_time = ref_times.min()
end_time = ref_times.max()
time_interval = ref_times.diff().mean()  # Approx 1.5 hours
num_entries = len(ref_times)

# Prepare time series for missing locations
missing_locations = locations_df[locations_df['Groundwater_Level'] == 0][['Latitude', 'Longitude']].copy()
all_predicted_data = []

for _, loc in missing_locations.iterrows():
    times = [start_time + i * time_interval for i in range(num_entries)]
    pred_df = pd.DataFrame({
        'Latitude': loc['Latitude'],
        'Longitude': loc['Longitude'],
        'Year': [t.year for t in times],
        'Month': [t.month for t in times],
        'Day': [t.day for t in times],
        'Hour': [t.hour for t in times]
    })
    X_pred = pred_df[['Latitude', 'Longitude', 'Year', 'Month', 'Day', 'Hour']]
    predicted_values = model.predict(X_pred)
    
    pred_data = pd.DataFrame({
        'Latitude': loc['Latitude'],
        'Longitude': loc['Longitude'],
        'Data Time': times,
        'Data Value': predicted_values,
        'Unit': 'm',
        'Station Code': f'PRED_{loc.name}',
        'Station Name': f'Predicted_{loc.name}'
    })
    all_predicted_data.append(pred_data)

# Combine predicted data
predicted_df = pd.concat(all_predicted_data, ignore_index=True)

# Append to base data
updated_df = pd.concat([base_df, predicted_df], ignore_index=True)

# Save updated dataset
updated_csv = os.path.join(output_folder, 'updated_combined_groundwater_levels.csv')
updated_df.to_csv(updated_csv, index=False)
print(f"Updated dataset saved: {updated_csv}")