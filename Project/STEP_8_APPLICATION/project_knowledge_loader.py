"""
Project Knowledge Loader for RAG
Loads all project resources (datasets, models, PDFs) into the RAG system
"""

import os
import pandas as pd
from typing import List, Dict
from langchain_core.documents import Document

def load_borewell_knowledge(csv_path: str) -> List[Document]:
    """Load borewell dataset as structured documents"""
    documents = []
    
    if not os.path.exists(csv_path):
        print(f"⚠️ Borewell CSV not found: {csv_path}")
        return documents
    
    try:
        df = pd.read_csv(csv_path)
        
        # Create summary document
        summary = f"""
# CGWB Borewell Database - Nashik District

Total borewells: {len(df)}
Successful: {len(df[df['Status'] == 'Success'])}
Failed: {len(df[df['Status'] == 'Failure'])}

Average depth: {df['Depth_m'].mean():.1f}m
Average yield: {df['Yield_LPH'].mean():.0f} LPH

Locations covered: {', '.join(df['Location_Name'].unique())}
Talukas: {', '.join(df['Taluka'].unique())}
"""
        documents.append(Document(page_content=summary, metadata={'source': 'borewell_summary'}))
        
        # Create documents for each borewell
        for idx, row in df.iterrows():
            content = f"""
Borewell ID: {row['Borewell_ID']}
Location: {row['Location_Name']}, {row['Taluka']}, {row['District']}
Coordinates: {row['Latitude']}, {row['Longitude']}
Status: {row['Status']}
Depth: {row['Depth_m']}m
Yield: {row['Yield_LPH']} LPH
Water Quality: {row['Water_Quality']}
Construction Year: {row['Construction_Year']}
"""
            documents.append(Document(
                page_content=content,
                metadata={
                    'source': 'borewell_data',
                    'location': row['Location_Name'],
                    'status': row['Status']
                }
            ))
        
        print(f"✅ Loaded {len(documents)} borewell documents")
        return documents
        
    except Exception as e:
        print(f"❌ Error loading borewell knowledge: {e}")
        return documents

def load_master_dataset_knowledge(csv_path: str) -> List[Document]:
    """Load master dataset as structured documents"""
    documents = []
    
    if not os.path.exists(csv_path):
        print(f"⚠️ Master dataset not found: {csv_path}")
        return documents
    
    try:
        df = pd.read_csv(csv_path)
        
        # Create summary document
        summary = f"""
# Master Dataset - Groundwater Predictions

Total records: {len(df)}
Date range: {df['Data Time'].min()} to {df['Data Time'].max()}

Features tracked:
- Groundwater levels (depth below surface)
- Rainfall data
- River water levels
- Temperature
- Time-lagged features for predictions

Locations: {df['Latitude'].nunique()} unique coordinates
Average groundwater level: {df['Groundwater_Level'].mean():.2f}m
Average rainfall: {df['Rainfall'].mean():.2f}mm
Average temperature: {df['Temperature'].mean():.1f}°C
"""
        documents.append(Document(page_content=summary, metadata={'source': 'master_dataset_summary'}))
        
        # Location-wise statistics
        location_stats = df.groupby(['Latitude', 'Longitude']).agg({
            'Groundwater_Level': ['mean', 'min', 'max'],
            'Rainfall': 'mean',
            'Temperature': 'mean'
        }).round(2)
        
        for (lat, lon), stats in location_stats.iterrows():
            content = f"""
Location: {lat:.4f}, {lon:.4f}
Average groundwater level: {stats['Groundwater_Level']['mean']:.2f}m
Min groundwater level: {stats['Groundwater_Level']['min']:.2f}m
Max groundwater level: {stats['Groundwater_Level']['max']:.2f}m
Average rainfall: {stats['Rainfall']:.2f}mm
Average temperature: {stats['Temperature']:.1f}°C
"""
            documents.append(Document(
                page_content=content,
                metadata={
                    'source': 'location_stats',
                    'lat': lat,
                    'lon': lon
                }
            ))
        
        print(f"✅ Loaded {len(documents)} master dataset documents")
        return documents
        
    except Exception as e:
        print(f"❌ Error loading master dataset knowledge: {e}")
        return documents

def load_model_knowledge() -> List[Document]:
    """Load information about trained models"""
    documents = []
    
    model_info = """
# Trained Machine Learning Models

## 1. Groundwater Level Prediction Model
- Type: XGBoost Regressor
- Features: Latitude, Longitude, Rainfall, River Water Level, Temperature, Year, Month, Day, Hour, Lag features
- Purpose: Predicts groundwater depth (meters below surface)
- Confidence: Dynamic (50-98%) based on location proximity to training data
- Adjustment: Uses nearby borewell data for intelligent offset correction

## 2. AI Borewell Recommendation Model
- Type: Logistic Regression (with proximity penalties)
- Features: Average depth, average yield, water quality, borewell age
- Purpose: Recommends best locations for new borewells
- Scoring: Considers success probability + proximity to existing borewells + density metrics
- Output: Top 5 sites with explanations, recommended depth, and factors

## 3. Supporting Models
- Rainfall prediction model
- River level prediction model
- Temperature prediction model
- Borewell depth prediction model

All models are trained on Nashik district historical data.
"""
    documents.append(Document(page_content=model_info, metadata={'source': 'model_info'}))
    
    print(f"✅ Loaded model knowledge documents")
    return documents

def load_project_context() -> List[Document]:
    """Load general project context and instructions"""
    documents = []
    
    context = """
# Groundwater Level Prediction System - Nashik District

## Project Overview
This is an AI-powered groundwater management system for Nashik District, Maharashtra, India.

## Key Features:
1. **Groundwater Level Prediction**: Predicts water depth using ML models with real-time weather data
2. **AI Borewell Recommendations**: Suggests optimal drilling locations considering existing borewells
3. **Interactive Map**: Visualizes predictions, existing borewells, and recommendations
4. **RAG Chatbot**: Answers questions about groundwater, borewells, and predictions
5. **Weather Integration**: Real-time rainfall and temperature data
6. **Evidence-based Decisions**: Uses NAQUIM reports and historical data

## Data Sources:
- CGWB (Central Ground Water Board) borewell database
- NAQUIM Nashik water quality reports
- Historical groundwater, rainfall, river level, and temperature data
- 30+ documented borewells across Nashik district

## Technologies:
- Backend: Python Flask, XGBoost, Scikit-learn
- Frontend: Leaflet maps, Tailwind CSS
- AI: LangChain, Groq LLaMA3, Chroma vector DB, Sentence Transformers
- Data: Pandas, NumPy

## Use Cases:
- Agricultural planning
- Urban water management
- Borewell drilling site selection
- Water resource monitoring
- Research and analysis
"""
    documents.append(Document(page_content=context, metadata={'source': 'project_context'}))
    
    print(f"✅ Loaded project context")
    return documents

def load_all_project_knowledge(base_path: str = None) -> List[Document]:
    """Load all project knowledge sources"""
    if base_path is None:
        base_path = os.path.dirname(__file__)
    
    all_docs = []
    
    # Load borewell dataset
    borewell_path = os.path.join(base_path, '..', 'STEP_7_SUPPORTING_DATA', 'cgwb_borewells_nashik.csv')
    all_docs.extend(load_borewell_knowledge(borewell_path))
    
    # Load master dataset
    master_path = os.path.join(base_path, '..', 'STEP_4_DATA_INTEGRATION', 'final_master_dataset.csv')
    all_docs.extend(load_master_dataset_knowledge(master_path))
    
    # Load model knowledge
    all_docs.extend(load_model_knowledge())
    
    # Load project context
    all_docs.extend(load_project_context())
    
    print(f"\n📚 Total project knowledge documents: {len(all_docs)}")
    return all_docs
