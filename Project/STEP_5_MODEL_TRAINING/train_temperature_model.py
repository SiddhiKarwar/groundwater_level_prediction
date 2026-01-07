import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib
import os

# Hardcoded input CSV path
csv_path = "E:/under water/formalize_dataset/Temperature/combined_temperature.csv"

# Create models folder if it doesn't exist
models_folder = os.path.join(os.path.dirname(os.path.dirname(csv_path)), 'models')
os.makedirs(models_folder, exist_ok=True)

# Load data
df = pd.read_csv(csv_path)
df['Data Time'] = pd.to_datetime(df['Data Time'])
df['Year'] = df['Data Time'].dt.year
df['Month'] = df['Data Time'].dt.month
df['Day'] = df['Data Time'].dt.day
df['Hour'] = df['Data Time'].dt.hour

# Features and target
features = ['Latitude', 'Longitude', 'Year', 'Month', 'Day', 'Hour']
X = df[features]
y = df['Data Value']

# Drop rows where target has NaN
mask = ~y.isna()
X = X[mask]
y = y[mask]

# Split and train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"MSE: {mse}")

# Save model
model_path = os.path.join(models_folder, 'temperature_model.pkl')
joblib.dump(model, model_path)
print(f"Model saved: {model_path}")