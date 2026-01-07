# 💾 STEP 6: TRAINED MODELS

## Description
Saved machine learning models and scalers.

## Contents
- `groundwater_model.pkl` - Main groundwater prediction model
- `water_level_model_fast.pkl` - Optimized water level model
- `water_level_model2.pkl` - Alternative water level model
- `scaler_fast.pkl` - Feature scaler for normalization
- `scaler2.pkl` - Alternative scaler

## Model Details
- **Algorithm**: XGBoost Regressor
- **Features**: Latitude, Longitude, Rainfall, Temperature, River Level, Lag values
- **Training Data**: 10,000+ samples
- **Accuracy**: Dynamic confidence 50-98%

## Usage
These models are loaded by Flask application in **STEP_8_APPLICATION/app.py**

## Input
⬅️ Created by training scripts in **STEP_5_MODEL_TRAINING/**

## Output
➡️ Used by **STEP_8_APPLICATION/**
