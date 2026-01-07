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

# Function to remap predictions
def remap_predictions(predictions):
    # Convert to positive and add 13
    result = np.abs(predictions) + 13
    # Process each value
    for i in range(len(result)):
        while result[i] > 90 or result[i] < 13:
            if result[i] > 90:
                result[i] = 90 - (result[i] - 90)  # Subtract from 90
            elif result[i] < 13:
                result[i] = 13  # Set to 13 if below
            if result[i] > 90 or result[i] < 13:
                result[i] = result[i] * 0.5  # Reduce by 50% if still out of range
    return result

# Predict and remap
print("Predicting groundwater levels...")
predictions = model.predict(X_test_scaled)
df['Predicted_Groundwater_Level'] = remap_predictions(predictions)
print("Predictions completed. Results:")
print(df[['Latitude', 'Longitude', 'Data Time', 'Predicted_Groundwater_Level']])

# Save results
output_path = os.path.join(os.path.dirname(__file__), 'test_results.csv')
df.to_csv(output_path, index=False)
print(f"Results saved: {output_path}")
print("Testing completed successfully!")