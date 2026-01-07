Groundwater Prediction System – Nashik District

  Project Overview

The Groundwater Prediction System for Nashik District is an AI-powered solution designed to forecast groundwater levels using machine learning, real-time data, and interactive web technologies. The system integrates multiple datasets—including rainfall, river water levels, groundwater levels, and temperature—to produce reliable and actionable groundwater insights.

  Features

    Interactive Map  : Select locations manually or using GPS
    AI Predictions  : XGBoost-based groundwater predictions with dynamic confidence scores (50–98%)
    Area-Level Predictions  : Heatmap visualization of groundwater levels
    Borewell Database  : Includes 30 CGWB borewells with depth and status data
    Smart Search  : AI-assisted location search
    Dark Mode  : Complete dark theme support

---

  Project Structure

```
Project/
├── STEP_1_RAW_DATA/              Original Excel files (IMD, CGWB)
├── STEP_2_DATA_PREPROCESSING/    Scripts for converting Excel → CSV
├── STEP_3_PROCESSED_DATA/        Cleaned CSV files
├── STEP_4_DATA_INTEGRATION/      Combined master dataset
├── STEP_5_MODEL_TRAINING/        Machine learning training scripts
├── STEP_6_TRAINED_MODELS/        Saved .pkl model files
├── STEP_7_SUPPORTING_DATA/       Borewell, location datasets
├── STEP_8_APPLICATION/           Flask web application
└── STEP_9_TESTING/               Model and dataset testing scripts
```

---

  Workflow

```
Raw Data (STEP 1)
      ↓
Convert to CSV (STEP 2)
      ↓
Cleaned Data (STEP 3)
      ↓
Dataset Integration (STEP 4)
      ↓
Model Training (STEP 5)
      ↓
Model Storage (STEP 6)
      ↓
Web Application (STEP 8) ← Supporting Data (STEP 7)
      ↓
Testing (STEP 9)
```

---

  How to Execute

   Prerequisites

  Python 3.8 or higher
  pip (package manager)
  Git (optional)
  Internet connection (for API access)

---

   Method 1: Run the Web Application (Recommended)

1.   Navigate to Project Directory  

   ```
   cd Project
   ```

2.   Create Virtual Environment  

   ```
   python -m venv .venv
   .\.venv\Scripts\activate
   ```

3.   Install Dependencies  

   ```
   pip install -r STEP_8_APPLICATION/requirements_rag.txt
   ```

4.   Add API Keys  

   Create a `.env` file inside `STEP_8_APPLICATION/`:

   ```
   GEMINI_API_KEY=your_key
   GOOGLE_MAPS_API_KEY=your_key
   ```

5.   Run Application  

   ```
   cd STEP_8_APPLICATION
   python app.py
   ```

6.   Open Browser  

   ```
   http://127.0.0.1:5000/
   ```

---

   Method 2: Rebuild the Entire Pipeline (From Raw Data)

  Step 1: Process Raw Data

```
cd STEP_2_DATA_PREPROCESSING
python process_excel.py
python process_rainfall.py
python process_temperature.py
python process_river_level.py
```

  Step 2: Integrate Datasets

```
cd ../STEP_4_DATA_INTEGRATION
python create_master_dataset.py
python fill_all_missing_values.py
python check_master_dataset.py
```

  Step 3: Train Machine Learning Models

```
cd ../STEP_5_MODEL_TRAINING
python train_water_level_model.py
python train_borewell_depth_model.py
python train_rainfall_model.py
python train_temperature_model.py
python train_river_level_model.py
python train_groundwater_model.py
```

  Step 4: Test Models

```
cd ../STEP_9_TESTING
python verify_dataset.py
python test_prediction_adjustment.py
python test_select_borewell_sites.py
python test_ai_search.py
```

  Step 5: Run Application

```
cd ../STEP_8_APPLICATION
python app.py
```

---

   Method 3: Quick Testing (No Web Interface)

```
cd STEP_5_MODEL_TRAINING
python predict_water_level.py

cd ../STEP_8_APPLICATION
python select_borewell_sites.py
python test_rag_setup.py
```

---

  Troubleshooting

  Module not found  

```
.\.venv\Scripts\activate
pip install -r STEP_8_APPLICATION/requirements_rag.txt
```

  API key errors  
Ensure `.env` file is created inside `STEP_8_APPLICATION/`.

  Model files missing  
Re-run training scripts in `STEP_5_MODEL_TRAINING/`.

  Port already in use  
Modify port in `app.py`:

```
app.run(debug=True, port=5001)
```

  ChromaDB errors  

```
cd STEP_8_APPLICATION
Remove-Item -Recurse -Force ./chromadb_storage
python test_rag_setup.py
```

---

  Required Files Checklist

   Data Files

  Processed CSV files in `STEP_3_PROCESSED_DATA/`
  `final_master_dataset.csv` in `STEP_4_DATA_INTEGRATION/`
  Borewell dataset in `STEP_7_SUPPORTING_DATA/`

   Model Files (STEP_6_TRAINED_MODELS/)

  `water_level_model.pkl`
  `borewell_depth_model.pkl`
  `rainfall_model.pkl`
  `temperature_model.pkl`
  `river_level_model.pkl`
  Required scaler `.pkl` files

   Application Files

  `app.py`
  `templates/` folder
  `.env` file with keys

---

  Quick Command Reference

| Task                   | Command                                                        |
| ---------------------- | -------------------------------------------------------------- |
| Activate Environment   | `.\.venv\Scripts\activate`                                     |
| Install Dependencies   | `pip install -r STEP_8_APPLICATION/requirements_rag.txt`       |
| Run Web Application    | `cd STEP_8_APPLICATION & python app.py`                        |
| Train Models           | `cd STEP_5_MODEL_TRAINING & python train_water_level_model.py` |
| Test System            | `cd STEP_9_TESTING & python verify_dataset.py`                 |
| Deactivate Environment | `deactivate`                                                   |

---

  Dataset Details

    Total Records  : 10,000+
    Locations  : 500+ across Nashik District
    Time Range  : 2016–2025
    Parameters  : Rainfall, Temperature, River Level, Groundwater
    Borewell Records  : 30 CGWB sites

---

  Model Performance

    Algorithm  : XGBoost Regressor
    Training Samples  : 10,000+
    Dynamic Confidence Score  : 50–98% based on:

    Location proximity
    Feature quality
    Model uncertainty
    Data completeness

---

  User Interface

  Full-screen map with real-time predictions
  No page refresh (AJAX-based)
  Confidence color indicators
  Area heatmap visualization
  Dark mode support

---

  Documentation Included

  `ALGORITHM_SECTION.md`
  `PROJECT_REPORT_DOCUMENTATION.md`
  `CONFUSION_MATRIX_ANALYSIS.md`
  `RESULTS_AND_ANALYSIS.md`
  Detailed README files inside each STEP folder


