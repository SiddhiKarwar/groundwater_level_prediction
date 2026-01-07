# 🌐 STEP 8: FLASK APPLICATION

## Description
Main web application for groundwater prediction.

## Contents
- `app.py` - Flask server and API routes
- `utils.py` - Utility functions (prediction adjustments)
- `gemini_service.py` - AI-powered search
- `google_maps_service.py` - Google Maps integration
- `smart_search_service.py` - Smart location search
- `weather_service.py` - Weather data integration
- `recommendation_service.py` - Borewell recommendations
- `templates/index.html` - Frontend UI

## How to Run
```bash
cd STEP_8_APPLICATION
python app.py
```

Then open: **http://127.0.0.1:5000/**

## Features
- 🗺️ Interactive map with location selection
- 🤖 AI predictions with dynamic confidence (50-98%)
- 🎨 Area predictions with heatmap
- 🏗️ Borewell database (30 CGWB records)
- 🔍 Smart search with Google Maps
- 🌙 Dark mode support

## Dependencies
```bash
pip install flask pandas numpy xgboost scikit-learn requests google-generativeai
```

## Input
⬅️ Models from **STEP_6_TRAINED_MODELS/**  
⬅️ Supporting data from **STEP_7_SUPPORTING_DATA/**
