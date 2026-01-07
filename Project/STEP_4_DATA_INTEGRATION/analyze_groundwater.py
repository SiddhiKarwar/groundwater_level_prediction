import pandas as pd

csv_path = r"formalize_dataset\Groundwater_Level\combined_groundwater_levels.csv"
df = pd.read_csv(csv_path)
df['Data Time'] = pd.to_datetime(df['Data Time'])

print("Unique Stations:", df['Station Name'].unique())
print("Entries per Station:", df.groupby('Station Name').size())
print("Time Range per Station:", df.groupby('Station Name')['Data Time'].agg(['min', 'max']))
print("Avg Time Interval:", df['Data Time'].sort_values().diff().mean())