import pandas as pd
import os

# Relative paths
gw_csv = r"formalize_dataset\Groundwater_Level\updated_combined_groundwater_levels.csv"
rain_csv = r"formalize_dataset\Rainfall\updated_combined_rainfall.csv"
river_csv = r"formalize_dataset\River Water Level\updated_combined_river_water_level.csv"
temp_csv = r"formalize_dataset\Temperature\updated_combined_temperature.csv"
output_folder = os.path.dirname(gw_csv)

# Load updated CSVs
gw_df = pd.read_csv(gw_csv).rename(columns={'Data Value': 'Groundwater_Level'})[['Latitude', 'Longitude', 'Data Time', 'Groundwater_Level']]
rain_df = pd.read_csv(rain_csv).rename(columns={'Data Value': 'Rainfall'})[['Latitude', 'Longitude', 'Data Time', 'Rainfall']]
river_df = pd.read_csv(river_csv).rename(columns={'Data Value': 'River_Water_Level'})[['Latitude', 'Longitude', 'Data Time', 'River_Water_Level']]
temp_df = pd.read_csv(temp_csv).rename(columns={'Data Value': 'Temperature'})[['Latitude', 'Longitude', 'Data Time', 'Temperature']]

# Merge on coordinates and time
master_df = gw_df.merge(rain_df, on=['Latitude', 'Longitude', 'Data Time'], how='outer')
master_df = master_df.merge(river_df, on=['Latitude', 'Longitude', 'Data Time'], how='outer')
master_df = master_df.merge(temp_df, on=['Latitude', 'Longitude', 'Data Time'], how='outer')

# Save master dataset
master_csv = os.path.join(output_folder, 'master_dataset.csv')
master_df.to_csv(master_csv, index=False)
print(f"Master dataset saved: {master_csv}")