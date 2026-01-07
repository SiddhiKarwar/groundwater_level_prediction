# 📁 STEP 7: SUPPORTING DATA

## Description
Additional data files used by the application.

## Contents
- `cgwb_borewells_nashik.csv` - CGWB borewell database (30 records)
- `locations_equal_final.csv` - Location coordinates database

## Data Details
- **Borewells**: Success/failure data, depth, yield, water quality
- **Locations**: GPS coordinates for Nashik district

## Usage
Loaded by Flask application for:
- Displaying existing borewells on map
- Location search and validation
- Calculating prediction confidence

## Output
➡️ Used by **STEP_8_APPLICATION/app.py**
