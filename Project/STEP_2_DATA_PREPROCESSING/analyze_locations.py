import pandas as pd
import os

# Input formalize_dataset path
formalize_path = "formalize_dataset"

# Dictionary for report
report = {}

# Traverse subfolders
for dataset_type in os.listdir(formalize_path):
    type_path = os.path.join(formalize_path, dataset_type)
    if os.path.isdir(type_path):
        unique_locations = set()
        for file_name in os.listdir(type_path):
            if file_name.endswith('.csv'):
                csv_path = os.path.join(type_path, file_name)
                df = pd.read_csv(csv_path)
                if 'Latitude' in df.columns and 'Longitude' in df.columns:
                    locations = df[['Latitude', 'Longitude']].drop_duplicates().itertuples(index=False, name=None)
                    unique_locations.update(locations)
        report[dataset_type] = {
            'num_locations': len(unique_locations),
            'locations': list(unique_locations)
        }

# Print report as table
print("Dataset Type | Num Locations | Locations")
for key, value in report.items():
    print(f"{key} | {value['num_locations']} | {value['locations']}")

# Save report to CSV
report_df = pd.DataFrame.from_dict(report, orient='index')
report_df.to_csv(os.path.join(formalize_path, 'locations_report.csv'), index_label='Dataset Type')
print("Report saved: locations_report.csv")