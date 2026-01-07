import pandas as pd
import os

# Input folder path
folder_path = "Datasets\Groundwater_Level"

# Output folder
output_folder = os.path.join(os.path.dirname(folder_path), 'formalize_dataset')
os.makedirs(output_folder, exist_ok=True)

# List to hold all dataframes
all_data = []

# Process each Excel file
for file_name in os.listdir(folder_path):
    if file_name.endswith('.xlsx'):
        excel_path = os.path.join(folder_path, file_name)
        xls = pd.ExcelFile(excel_path)

        # Metadata sheet (index 0, irregular)
        metadata_sheet = pd.read_excel(xls, sheet_name=0, header=None)
        metadata_dict = {}
        start_row = metadata_sheet[metadata_sheet[0] == 'Station Code'].index[0] if not metadata_sheet[metadata_sheet[0] == 'Station Code'].empty else 8
        for i in range(start_row, len(metadata_sheet)):
            if pd.notna(metadata_sheet.iloc[i, 0]) and pd.notna(metadata_sheet.iloc[i, 1]):
                key = str(metadata_sheet.iloc[i, 0]).strip()
                value = str(metadata_sheet.iloc[i, 1]).strip()
                metadata_dict[key] = value

        # Data sheet (index 1, skip title rows)
        data_sheet = pd.read_excel(xls, sheet_name=1, skiprows=6)

        # Add metadata
        for key, value in metadata_dict.items():
            if key in ['Station Code', 'Station Name', 'Latitude', 'Longitude']:
                data_sheet[key] = value

        # Needed columns
        needed_columns = ['Data Time', 'Data Value', 'Unit', 'Station Code', 'Station Name', 'Latitude', 'Longitude']
        data_sheet = data_sheet[needed_columns]

        # Save individual CSV
        csv_name = file_name.replace('.xlsx', '.csv')
        csv_path = os.path.join(output_folder, csv_name)
        data_sheet.to_csv(csv_path, index=False)
        print(f"CSV saved: {csv_path}")

        # Append to all_data
        all_data.append(data_sheet)

# Combine all data into one CSV
combined_df = pd.concat(all_data, ignore_index=True)
combined_csv_path = os.path.join(output_folder, 'combined_groundwater_levels.csv')
combined_df.to_csv(combined_csv_path, index=False)
print(f"Combined CSV saved: {combined_csv_path}")