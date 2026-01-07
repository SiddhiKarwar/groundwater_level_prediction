"""
Train Borewell Depth Prediction Model
Using XGBoost with hyperparameter tuning for accurate predictions
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os

print("=" * 80)
print("🤖 BOREWELL DEPTH PREDICTION MODEL TRAINING")
print("=" * 80)

# Create model directory
os.makedirs('models_new', exist_ok=True)

# Load dataset
print("\n📊 Loading dataset...")
df = pd.read_csv('datasets_new/master_borewell_dataset.csv')
print(f"✅ Loaded {len(df)} records")
print(f"📋 Columns: {list(df.columns)}")

# Data preprocessing
print("\n🔧 Preprocessing data...")

# 1. Handle missing values
print(f"\n1️⃣ Checking for missing values...")
missing_counts = df.isnull().sum()
if missing_counts.sum() > 0:
    print(f"⚠️ Found missing values:")
    print(missing_counts[missing_counts > 0])
    # Fill missing Avg_Nearby_Depth_m with zone median
    for zone in df['Geological_Zone'].unique():
        zone_median = df[df['Geological_Zone'] == zone]['Avg_Nearby_Depth_m'].median()
        df.loc[(df['Geological_Zone'] == zone) & (df['Avg_Nearby_Depth_m'].isnull()), 'Avg_Nearby_Depth_m'] = zone_median
else:
    print("✅ No missing values found")

# 2. Encode categorical variables
print(f"\n2️⃣ Encoding categorical variables...")

categorical_cols = ['Geological_Zone', 'Soil_Type', 'Geology', 'Aquifer_Type', 
                     'Rainfall_Zone', 'Elevation_Category', 'Water_Quality', 'Status']

label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[f'{col}_Encoded'] = le.fit_transform(df[col])
    label_encoders[col] = le
    print(f"   ✅ Encoded {col}: {len(le.classes_)} unique values")

# 3. Feature selection
print(f"\n3️⃣ Selecting features...")

# Features for prediction
feature_cols = [
    'Latitude', 'Longitude',
    'Geological_Zone_Encoded', 'Soil_Type_Encoded', 'Geology_Encoded',
    'Aquifer_Type_Encoded', 'Rainfall_Zone_Encoded', 'Elevation_Category_Encoded',
    'Distance_to_River_km', 'Nearby_Borewells_10km', 'Avg_Nearby_Depth_m'
]

# Target variable
target_col = 'Depth_m'

X = df[feature_cols].copy()
y = df[target_col].copy()

print(f"📐 Feature matrix shape: {X.shape}")
print(f"🎯 Target variable shape: {y.shape}")

# 4. Train-test split
print(f"\n4️⃣ Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=True
)

print(f"   Training set: {X_train.shape[0]} samples")
print(f"   Test set: {X_test.shape[0]} samples")

# 5. Feature scaling
print(f"\n5️⃣ Scaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("   ✅ Features normalized")

# Model Training
print("\n" + "=" * 80)
print("🚀 MODEL TRAINING")
print("=" * 80)

# Model 1: XGBoost with default parameters
print("\n1️⃣ Training XGBoost (baseline)...")
xgb_base = XGBRegressor(
    random_state=42,
    n_estimators=100,
    learning_rate=0.1,
    max_depth=6
)
xgb_base.fit(X_train_scaled, y_train)

y_pred_base = xgb_base.predict(X_test_scaled)
rmse_base = np.sqrt(mean_squared_error(y_test, y_pred_base))
mae_base = mean_absolute_error(y_test, y_pred_base)
r2_base = r2_score(y_test, y_pred_base)

print(f"   📊 Baseline XGBoost Results:")
print(f"      RMSE: {rmse_base:.3f}m")
print(f"      MAE: {mae_base:.3f}m")
print(f"      R² Score: {r2_base:.4f}")

# Model 2: XGBoost with hyperparameter tuning
print("\n2️⃣ Training XGBoost with hyperparameter tuning...")
print("   (This may take a few minutes...)")

param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 5, 7, 9],
    'min_child_weight': [1, 3, 5],
    'subsample': [0.7, 0.8, 0.9],
    'colsample_bytree': [0.7, 0.8, 0.9]
}

xgb_tuned = XGBRegressor(random_state=42)

grid_search = GridSearchCV(
    estimator=xgb_tuned,
    param_grid=param_grid,
    cv=5,
    scoring='neg_mean_squared_error',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train_scaled, y_train)

print(f"   ✅ Best parameters: {grid_search.best_params_}")

best_xgb = grid_search.best_estimator_
y_pred_tuned = best_xgb.predict(X_test_scaled)

rmse_tuned = np.sqrt(mean_squared_error(y_test, y_pred_tuned))
mae_tuned = mean_absolute_error(y_test, y_pred_tuned)
r2_tuned = r2_score(y_test, y_pred_tuned)

print(f"   📊 Tuned XGBoost Results:")
print(f"      RMSE: {rmse_tuned:.3f}m")
print(f"      MAE: {mae_tuned:.3f}m")
print(f"      R² Score: {r2_tuned:.4f}")

# Model 3: Random Forest (for comparison)
print("\n3️⃣ Training Random Forest (for comparison)...")
rf_model = RandomForestRegressor(
    n_estimators=200,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train_scaled, y_train)

y_pred_rf = rf_model.predict(X_test_scaled)
rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
mae_rf = mean_absolute_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)

print(f"   📊 Random Forest Results:")
print(f"      RMSE: {rmse_rf:.3f}m")
print(f"      MAE: {mae_rf:.3f}m")
print(f"      R² Score: {r2_rf:.4f}")

# Model comparison
print("\n" + "=" * 80)
print("📊 MODEL COMPARISON")
print("=" * 80)

comparison_df = pd.DataFrame({
    'Model': ['XGBoost Baseline', 'XGBoost Tuned', 'Random Forest'],
    'RMSE (m)': [rmse_base, rmse_tuned, rmse_rf],
    'MAE (m)': [mae_base, mae_tuned, mae_rf],
    'R² Score': [r2_base, r2_tuned, r2_rf]
})

print("\n" + comparison_df.to_string(index=False))

# Select best model
best_model_idx = comparison_df['R² Score'].idxmax()
best_model_name = comparison_df.loc[best_model_idx, 'Model']
best_rmse = comparison_df.loc[best_model_idx, 'RMSE (m)']
best_mae = comparison_df.loc[best_model_idx, 'MAE (m)']
best_r2 = comparison_df.loc[best_model_idx, 'R² Score']

print(f"\n🏆 BEST MODEL: {best_model_name}")
print(f"   RMSE: {best_rmse:.3f}m")
print(f"   MAE: {best_mae:.3f}m")
print(f"   R² Score: {best_r2:.4f}")

# Select the actual best model object
if best_model_name == 'XGBoost Baseline':
    final_model = xgb_base
elif best_model_name == 'XGBoost Tuned':
    final_model = best_xgb
else:
    final_model = rf_model

# Feature importance
print("\n" + "=" * 80)
print("🔍 FEATURE IMPORTANCE")
print("=" * 80)

if hasattr(final_model, 'feature_importances_'):
    feature_importance = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': final_model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print("\n" + feature_importance.to_string(index=False))
    
    # Save feature importance
    feature_importance.to_csv('models_new/feature_importance.csv', index=False)
    print("\n💾 Saved feature importance to models_new/feature_importance.csv")

# Cross-validation
print("\n" + "=" * 80)
print("✅ CROSS-VALIDATION")
print("=" * 80)

print("\nPerforming 5-fold cross-validation...")
cv_scores = cross_val_score(
    final_model, X_train_scaled, y_train,
    cv=5, scoring='neg_mean_squared_error', n_jobs=-1
)

cv_rmse_scores = np.sqrt(-cv_scores)
print(f"   CV RMSE scores: {cv_rmse_scores}")
print(f"   Mean CV RMSE: {cv_rmse_scores.mean():.3f}m (± {cv_rmse_scores.std():.3f}m)")

# Prediction analysis
print("\n" + "=" * 80)
print("📈 PREDICTION ANALYSIS")
print("=" * 80)

if best_model_name.startswith('XGBoost'):
    y_pred_final = y_pred_tuned if best_model_name == 'XGBoost Tuned' else y_pred_base
else:
    y_pred_final = y_pred_rf

prediction_df = pd.DataFrame({
    'Actual_Depth_m': y_test.values,
    'Predicted_Depth_m': y_pred_final,
    'Error_m': y_test.values - y_pred_final,
    'Absolute_Error_m': np.abs(y_test.values - y_pred_final)
})

print("\n📊 Prediction Statistics:")
print(f"   Mean Error: {prediction_df['Error_m'].mean():.3f}m")
print(f"   Median Absolute Error: {prediction_df['Absolute_Error_m'].median():.3f}m")
print(f"   Max Absolute Error: {prediction_df['Absolute_Error_m'].max():.3f}m")
print(f"   95th Percentile Error: {prediction_df['Absolute_Error_m'].quantile(0.95):.3f}m")

# Accuracy within thresholds
within_3m = (prediction_df['Absolute_Error_m'] <= 3).sum() / len(prediction_df) * 100
within_5m = (prediction_df['Absolute_Error_m'] <= 5).sum() / len(prediction_df) * 100
within_10m = (prediction_df['Absolute_Error_m'] <= 10).sum() / len(prediction_df) * 100

print(f"\n🎯 Accuracy within thresholds:")
print(f"   Within ±3m: {within_3m:.1f}%")
print(f"   Within ±5m: {within_5m:.1f}%")
print(f"   Within ±10m: {within_10m:.1f}%")

# Save prediction analysis
prediction_df.to_csv('models_new/prediction_analysis.csv', index=False)
print("\n💾 Saved prediction analysis to models_new/prediction_analysis.csv")

# Save model and scaler
print("\n" + "=" * 80)
print("💾 SAVING MODEL AND ARTIFACTS")
print("=" * 80)

# Save model
model_filename = 'models_new/borewell_depth_model.pkl'
joblib.dump(final_model, model_filename)
print(f"✅ Saved model: {model_filename}")

# Save scaler
scaler_filename = 'models_new/scaler_borewell.pkl'
joblib.dump(scaler, scaler_filename)
print(f"✅ Saved scaler: {scaler_filename}")

# Save label encoders
encoders_filename = 'models_new/label_encoders.pkl'
joblib.dump(label_encoders, encoders_filename)
print(f"✅ Saved label encoders: {encoders_filename}")

# Save feature columns
feature_cols_filename = 'models_new/feature_columns.pkl'
joblib.dump(feature_cols, feature_cols_filename)
print(f"✅ Saved feature columns: {feature_cols_filename}")

# Save model metadata
metadata = {
    'model_name': best_model_name,
    'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'training_samples': len(X_train),
    'test_samples': len(X_test),
    'features': feature_cols,
    'target': target_col,
    'metrics': {
        'RMSE': float(best_rmse),
        'MAE': float(best_mae),
        'R2_Score': float(best_r2)
    },
    'cv_rmse_mean': float(cv_rmse_scores.mean()),
    'cv_rmse_std': float(cv_rmse_scores.std()),
    'accuracy': {
        'within_3m': float(within_3m),
        'within_5m': float(within_5m),
        'within_10m': float(within_10m)
    }
}

metadata_df = pd.DataFrame([metadata])
metadata_df.to_csv('models_new/model_metadata.csv', index=False)
print(f"✅ Saved model metadata: models_new/model_metadata.csv")

# Test prediction function
print("\n" + "=" * 80)
print("🧪 TESTING PREDICTION FUNCTION")
print("=" * 80)

def predict_borewell_depth(latitude, longitude, geological_zone, soil_type, 
                          geology, aquifer_type, rainfall_zone, elevation_category,
                          distance_to_river_km, nearby_borewells_10km, avg_nearby_depth_m):
    """
    Predict borewell depth for a given location
    """
    # Encode categorical variables
    features = {
        'Latitude': latitude,
        'Longitude': longitude,
        'Geological_Zone_Encoded': label_encoders['Geological_Zone'].transform([geological_zone])[0],
        'Soil_Type_Encoded': label_encoders['Soil_Type'].transform([soil_type])[0],
        'Geology_Encoded': label_encoders['Geology'].transform([geology])[0],
        'Aquifer_Type_Encoded': label_encoders['Aquifer_Type'].transform([aquifer_type])[0],
        'Rainfall_Zone_Encoded': label_encoders['Rainfall_Zone'].transform([rainfall_zone])[0],
        'Elevation_Category_Encoded': label_encoders['Elevation_Category'].transform([elevation_category])[0],
        'Distance_to_River_km': distance_to_river_km,
        'Nearby_Borewells_10km': nearby_borewells_10km,
        'Avg_Nearby_Depth_m': avg_nearby_depth_m
    }
    
    # Create DataFrame
    input_df = pd.DataFrame([features])
    
    # Scale features
    input_scaled = scaler.transform(input_df)
    
    # Predict
    depth = final_model.predict(input_scaled)[0]
    
    return round(depth, 1)

# Test with Nashik City example
print("\n📍 Test Case: Nashik City")
test_depth = predict_borewell_depth(
    latitude=19.9975,
    longitude=73.7898,
    geological_zone='Nashik_Central',
    soil_type='Black Basaltic',
    geology='Deccan Trap Basalt',
    aquifer_type='Fractured Basalt',
    rainfall_zone='Medium',
    elevation_category='Medium',
    distance_to_river_km=2.5,
    nearby_borewells_10km=12,
    avg_nearby_depth_m=45.0
)

print(f"   Predicted Depth: {test_depth}m")
print(f"   Expected Range: 35-55m")

# Save prediction function
prediction_function_code = """
# Borewell Depth Prediction Function
import joblib
import pandas as pd

# Load model and artifacts
model = joblib.load('models_new/borewell_depth_model.pkl')
scaler = joblib.load('models_new/scaler_borewell.pkl')
label_encoders = joblib.load('models_new/label_encoders.pkl')

def predict_borewell_depth(latitude, longitude, geological_zone, soil_type, 
                          geology, aquifer_type, rainfall_zone, elevation_category,
                          distance_to_river_km, nearby_borewells_10km, avg_nearby_depth_m):
    features = {
        'Latitude': latitude,
        'Longitude': longitude,
        'Geological_Zone_Encoded': label_encoders['Geological_Zone'].transform([geological_zone])[0],
        'Soil_Type_Encoded': label_encoders['Soil_Type'].transform([soil_type])[0],
        'Geology_Encoded': label_encoders['Geology'].transform([geology])[0],
        'Aquifer_Type_Encoded': label_encoders['Aquifer_Type'].transform([aquifer_type])[0],
        'Rainfall_Zone_Encoded': label_encoders['Rainfall_Zone'].transform([rainfall_zone])[0],
        'Elevation_Category_Encoded': label_encoders['Elevation_Category'].transform([elevation_category])[0],
        'Distance_to_River_km': distance_to_river_km,
        'Nearby_Borewells_10km': nearby_borewells_10km,
        'Avg_Nearby_Depth_m': avg_nearby_depth_m
    }
    input_df = pd.DataFrame([features])
    input_scaled = scaler.transform(input_df)
    depth = model.predict(input_scaled)[0]
    return round(depth, 1)
"""

with open('models_new/prediction_function.py', 'w') as f:
    f.write(prediction_function_code)
print(f"\n💾 Saved prediction function: models_new/prediction_function.py")

# Final summary
print("\n" + "=" * 80)
print("✅ MODEL TRAINING COMPLETED SUCCESSFULLY")
print("=" * 80)

print(f"\n🏆 Best Model: {best_model_name}")
print(f"📊 Performance:")
print(f"   - RMSE: {best_rmse:.3f}m")
print(f"   - MAE: {best_mae:.3f}m")
print(f"   - R² Score: {best_r2:.4f}")
print(f"   - Predictions within ±5m: {within_5m:.1f}%")

print(f"\n📁 Saved files:")
print(f"   1. models_new/borewell_depth_model.pkl")
print(f"   2. models_new/scaler_borewell.pkl")
print(f"   3. models_new/label_encoders.pkl")
print(f"   4. models_new/feature_columns.pkl")
print(f"   5. models_new/model_metadata.csv")
print(f"   6. models_new/feature_importance.csv")
print(f"   7. models_new/prediction_analysis.csv")
print(f"   8. models_new/prediction_function.py")

print(f"\n🎯 Next Steps:")
print(f"   1. Review model performance metrics")
print(f"   2. Check prediction_analysis.csv for detailed results")
print(f"   3. Integrate model into Flask app (app.py)")
print(f"   4. Update frontend to use new borewell depth prediction")
print(f"   5. Test with real locations")

print("\n" + "=" * 80)
