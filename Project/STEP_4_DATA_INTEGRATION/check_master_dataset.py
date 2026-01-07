import pandas as pd

csv_path = r"formalize_dataset\final_master_dataset.csv"
df = pd.read_csv(csv_path)

print("Head:\n", df.head())
print("Info:\n", df.info())
print("NaNs:\n", df.isna().sum())
print("Desc:\n", df.describe())
print("Duplicates:", df.duplicated().sum())
print("Unique Locations:", df[['Latitude', 'Longitude']].drop_duplicates().shape[0])
print("Entries:", len(df))
print("All proper if no NaNs in features/target and entries > 1000.")