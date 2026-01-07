import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import joblib
import os

print("Starting fast water level prediction model training...")

# Relative path for data
csv_path = r"formalize_dataset\final_master_dataset.csv"

# Load and sample data (10% for speed)
print("Loading and sampling dataset...")
df = pd.read_csv(csv_path)
df = df.sample(frac=0.1, random_state=42)  # 10% sample
df['Data Time'] = pd.to_datetime(df['Data Time'], format='mixed', dayfirst=False, errors='coerce')
df = df.dropna(subset=['Data Time'])  # Drop invalid times
print(f"Dataset loaded with {len(df)} entries.")

# Feature engineering (add lag for time-series)
print("Performing feature engineering...")
df = df.sort_values(['Latitude', 'Longitude', 'Data Time'])
df['Rainfall_Lag1'] = df.groupby(['Latitude', 'Longitude'])['Rainfall'].shift(1).fillna(0)
df['River_Lag1'] = df.groupby(['Latitude', 'Longitude'])['River_Water_Level'].shift(1).fillna(0)
print("Feature engineering complete.")

# Features and target
features = ['Latitude', 'Longitude', 'Rainfall', 'River_Water_Level', 'Temperature', 'Year', 'Month', 'Day', 'Hour', 'Rainfall_Lag1', 'River_Lag1']
X = df[features]
y = df['Groundwater_Level']

# Scale features
print("Scaling features...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Drop NaNs
mask = ~np.isnan(X_scaled).any(axis=1) & ~y.isna()
X_scaled = X_scaled[mask]
y = y[mask]
print(f"Cleaned dataset size: {len(X_scaled)} entries.")

# Split and train (fast configuration)
print("Training model...")
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
model = XGBRegressor(n_estimators=50, learning_rate=0.1, max_depth=3, random_state=42)  # Lightweight setup
model.fit(X_train, y_train)

# Evaluate
print("Evaluating model...")
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"MSE: {mse}")

# Save model and scaler to Water_level_prediction\model
print("Saving model and scaler...")
model_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'Water_level_prediction', 'model')
os.makedirs(model_dir, exist_ok=True)
model_path = os.path.join(model_dir, 'water_level_model_fast.pkl')
scaler_path = os.path.join(model_dir, 'scaler_fast.pkl')
joblib.dump(model, model_path)
joblib.dump(scaler, scaler_path)
print(f"Model saved: {model_path}")
print(f"Scaler saved: {scaler_path}")
print("Training completed successfully!")