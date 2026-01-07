# 🤖 STEP 5: MODEL TRAINING

## Description
Train machine learning models on integrated data.

## Training Scripts
- `train_groundwater_model.py` - Train groundwater prediction model
- `train_water_level_model_fast.py` - Train optimized water level model
- `train_rainfall_model.py` - Train rainfall prediction model
- `train_temperature_model.py` - Train temperature model

## Models Used
- XGBoost Regressor
- Random Forest
- Feature engineering with lag values

## How to Run
```bash
cd STEP_5_MODEL_TRAINING
python train_groundwater_model.py
python train_water_level_model_fast.py
```

## Input
⬅️ `final_master_dataset.csv` from **STEP_4_DATA_INTEGRATION/**

## Output
➡️ Trained models saved to **STEP_6_TRAINED_MODELS/**
