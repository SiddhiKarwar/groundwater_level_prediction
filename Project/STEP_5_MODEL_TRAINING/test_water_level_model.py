import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
import joblib
import os

print("Starting water level prediction testing...")

# Relative paths
test_csv = r"Water_level_prediction\test_water_level_data.csv"
model_dir = os.path.join(os.path.dirname(__file__), 'model')
model_path = os.path.join(model_dir, 'water_level_model2.pkl')
scaler_path = os.path.join(model_dir, 'scaler2.pkl')

# Load data
print("Loading test data...")
df = pd.read_csv(test_csv)
df['Data Time'] = pd.to_datetime(df['Data Time'], format='mixed', dayfirst=False, errors='coerce')
df = df.dropna(subset=['Data Time'])
df['Year'] = df['Data Time'].dt.year
df['Month'] = df['Data Time'].dt.month
df['Day'] = df['Data Time'].dt.day
df['Hour'] = df['Data Time'].dt.hour
print(f"Test data loaded with {len(df)} entries.")

# Features
features = ['Latitude', 'Longitude', 'Rainfall', 'River_Water_Level', 'Temperature', 'Year', 'Month', 'Day', 'Hour', 'Rainfall_Lag1', 'River_Lag1']
X_test = df[features]

# Load model and scaler
print("Loading model and scaler...")
model = joblib.load(model_path)
scaler = joblib.load(scaler_path)
X_test_scaled = scaler.transform(X_test)

# Predict, convert to positive, and clip to [13, 90]
print("Predicting groundwater levels...")
predictions = model.predict(X_test_scaled)
predictions = np.abs(predictions)  # Convert to positive
df['Predicted_Groundwater_Level'] = np.clip(predictions, 13, 90)  # Clip to range [13, 90]
print("Predictions completed. Results:")
print(df[['Latitude', 'Longitude', 'Data Time', 'Predicted_Groundwater_Level']])

# Save results
output_path = os.path.join(os.path.dirname(__file__), 'test_results.csv')
df.to_csv(output_path, index=False)
print(f"Results saved: {output_path}")
print("Testing completed successfully!")