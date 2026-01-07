import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import joblib
import os
from tqdm import tqdm

# Install tqdm if not present
try:
    from tqdm import tqdm
except ImportError:
    print("Installing tqdm for progress bar...")
    os.system("pip install tqdm")
    from tqdm import tqdm

print("Starting water level prediction model training...")

# Relative path for data
csv_path = r"formalize_dataset\final_master_dataset.csv"

# Load data
print("Loading dataset...")
df = pd.read_csv(csv_path)
df['Data Time'] = pd.to_datetime(df['Data Time'], format='mixed', dayfirst=False, errors='coerce')
df = df.dropna(subset=['Data Time'])  # Drop invalid times
df = df.drop_duplicates()  # Remove duplicates
print(f"Dataset loaded with {len(df)} entries.")

# Feature engineering (add lag for time-series, e.g., previous rainfall)
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

# Split and tune XGBoost
print("Splitting data and tuning model...")
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

param_grid = {
    'n_estimators': [100, 200],
    'learning_rate': [0.05, 0.1],
    'max_depth': [3, 5],
    'subsample': [0.8, 1.0]
}
model = XGBRegressor(random_state=42)
grid_search = GridSearchCV(model, param_grid, cv=3, scoring='neg_mean_squared_error', verbose=2, error_score='raise')
print(f"Starting GridSearchCV with {3 * len(param_grid['n_estimators']) * len(param_grid['learning_rate']) * len(param_grid['max_depth']) * len(param_grid['subsample'])} fits...")
with tqdm(total=48, desc="Training Progress") as pbar:  # 48 fits (3 folds * 16 combinations)
    def custom_callback(optim_result):
        pbar.update(1)
        return False  # Continue training
    grid_search.fit(X_train, y_train)

# Best model
best_model = grid_search.best_estimator_
print(f"Best parameters found: {grid_search.best_params_}")

# Evaluate
print("Evaluating model...")
y_pred = best_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"MSE: {mse}")

# Save model and scaler to Water_level_prediction\model
print("Saving model and scaler...")
model_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'Water_level_prediction', 'model')
os.makedirs(model_dir, exist_ok=True)
model_path = os.path.join(model_dir, 'water_level_model2.pkl')
scaler_path = os.path.join(model_dir, 'scaler2.pkl')
joblib.dump(best_model, model_path)
joblib.dump(scaler, scaler_path)
print(f"Model saved: {model_path}")
print(f"Scaler saved: {scaler_path}")
print("Training completed successfully!")