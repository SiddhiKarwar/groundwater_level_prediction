# 🔗 STEP 4: DATA INTEGRATION

## Description
Combine all processed data into master dataset.

## Files
- `master_dataset.csv` - Combined raw data
- `final_master_dataset.csv` - Complete dataset with filled missing values
- `locations_report.csv` - Location analysis report

## Scripts
- `create_master_dataset.py` - Merge all data sources
- `fill_all_missing_values.py` - Handle missing data
- `analyze_groundwater.py` - Analyze integrated data
- `predict_*_missing_with_time.py` - Fill missing values using ML

## How to Run
```bash
cd STEP_4_DATA_INTEGRATION
python create_master_dataset.py
python fill_all_missing_values.py
```

## Input
⬅️ Processed data from **STEP_3_PROCESSED_DATA/**

## Output
➡️ `final_master_dataset.csv` used in **STEP_5_MODEL_TRAINING/**
