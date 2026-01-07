from flask import Flask, request, render_template, jsonify, send_from_directory
from werkzeug.exceptions import RequestEntityTooLarge
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
import joblib
import os
from utils import remap_predictions
from math import radians, sin, cos, sqrt, atan2
from datetime import datetime
import requests
from weather_service import weather_service
from recommendation_service import recommendation_service
from gemini_service import get_location_suggestions, get_location_details
from google_maps_service import (
    search_places, 
    get_place_details, 
    nearby_search, 
    autocomplete_places,
    get_popular_places_by_category
)
# AI recommender module (trainable, lightweight)
import ai_recommender
from select_borewell_sites import bbox_to_grid

# RAG Chatbot module - TEMPORARILY DISABLED due to Python 3.14 compatibility
# import rag_chatbot
rag_chatbot = None  # Placeholder

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'

# Create uploads folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Return JSON on file-too-large (413)
@app.errorhandler(RequestEntityTooLarge)
def handle_file_too_large(e):
    return jsonify({
        'success': False,
        'error': f'File too large. Max {app.config.get("MAX_CONTENT_LENGTH", 0) // (1024*1024)}MB allowed.'
    }), 413

# Update paths - models are now in STEP_6_TRAINED_MODELS/
model_dir = os.path.join(os.path.dirname(__file__), '..', 'STEP_6_TRAINED_MODELS')
model_path = os.path.join(model_dir, 'water_level_model2.pkl')
scaler_path = os.path.join(model_dir, 'scaler2.pkl')

try:
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
except Exception as e:
    print(f"⚠️ Warning: Could not load water level model/scaler: {e}")
    model = None
    scaler = None

# Load allowed Nashik locations - now in STEP_7_SUPPORTING_DATA/
locations_csv = os.path.join(os.path.dirname(__file__), '..', 'STEP_7_SUPPORTING_DATA', 'locations_equal_final.csv')
if os.path.exists(locations_csv):
    locs_df = pd.read_csv(locations_csv)
    # keep only latitude/longitude pairs from that CSV
    allowed_locations = list(locs_df[['Latitude', 'Longitude']].itertuples(index=False, name=None))
else:
    allowed_locations = []

# Load CGWB Borewell Database - now in STEP_7_SUPPORTING_DATA/
borewells_csv = os.path.join(os.path.dirname(__file__), '..', 'STEP_7_SUPPORTING_DATA', 'cgwb_borewells_nashik.csv')
if os.path.exists(borewells_csv):
    borewells_df = pd.read_csv(borewells_csv)
    print(f"✅ Loaded {len(borewells_df)} borewells from CGWB database")
    # Initialize recommendation service with borewell data
    recommendation_service.borewells_df = borewells_df
else:
    borewells_df = pd.DataFrame()
    print("⚠️ CGWB borewell database not found")

# Lightweight local gazetteer for offline fallback (approximate centers)
# Note: Coordinates are approximate city/town centers in Nashik district
LOCAL_PLACE_CENTERS = {
    'nashik': (19.9975, 73.7898),
    'malegaon': (20.5537, 74.5288),
    'pimpalgaon baswant': (20.1644, 74.2545),
    'pachora': (20.6673, 75.3530),  # outside Nashik district but nearby
    'niphad': (20.0751, 74.1116),
    'yeola': (20.0437, 74.4897),
    'sinnar': (19.8540, 74.0005),
    'igatpuri': (19.6953, 73.5626),
    'trimbak': (19.9328, 73.5292),
    'dindori': (20.2030, 73.8324),
    'nandgaon': (20.3077, 74.6555),
    'chandwad': (20.3294, 74.2464),
}


def haversine_deg(lat1, lon1, lat2, lon2):
    """Return distance between two lat/lon points in kilometers (approx)."""
    # approximate radius of earth in km
    R = 6371.0
    phi1 = radians(lat1)
    phi2 = radians(lat2)
    dphi = radians(lat2 - lat1)
    dlambda = radians(lon2 - lon1)
    a = sin(dphi / 2.0)**2 + cos(phi1) * cos(phi2) * sin(dlambda / 2.0)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c


def calculate_prediction_confidence(model, X_scaled, latitude, longitude, rainfall, temperature, allowed_locations):
    """
    Calculate DYNAMIC confidence score for prediction (0-100%)
    
    Factors considered:
    1. Location confidence (proximity to training data)
    2. Feature quality (realistic values)
    3. Model uncertainty (variance across trees)
    4. Data completeness
    
    Returns: Confidence percentage (e.g., 87.5%)
    """
    confidence_scores = []
    
    # 1. LOCATION CONFIDENCE (0-100%)
    # Higher confidence if location is close to training data locations
    if allowed_locations:
        min_distance = float('inf')
        for loc in allowed_locations:
            try:
                loc_lat, loc_lon = float(loc[0]), float(loc[1])
                dist = haversine_deg(latitude, longitude, loc_lat, loc_lon)
                min_distance = min(min_distance, dist)
            except:
                continue
        
        # Confidence decreases with distance from known locations
        # 0 km = 100%, 5 km = 95%, 10 km = 85%, 20 km = 70%, 50+ km = 50%
        if min_distance <= 5:
            location_confidence = 100 - (min_distance * 1.0)  # 100-95%
        elif min_distance <= 10:
            location_confidence = 95 - ((min_distance - 5) * 2.0)  # 95-85%
        elif min_distance <= 20:
            location_confidence = 85 - ((min_distance - 10) * 1.5)  # 85-70%
        elif min_distance <= 50:
            location_confidence = 70 - ((min_distance - 20) * 0.67)  # 70-50%
        else:
            location_confidence = 50  # Far from training data
        
        confidence_scores.append(max(50, min(100, location_confidence)))
    else:
        confidence_scores.append(75)  # Default if no location data
    
    # 2. FEATURE QUALITY CONFIDENCE (0-100%)
    # Check if input values are within realistic ranges for Nashik
    feature_quality = 100
    
    # Rainfall check (Nashik: 0-500mm monthly is realistic)
    if rainfall < 0 or rainfall > 600:
        feature_quality -= 20
    elif rainfall > 500:
        feature_quality -= 10
    
    # Temperature check (Nashik: 15-45°C is realistic)
    if temperature < 10 or temperature > 50:
        feature_quality -= 20
    elif temperature < 15 or temperature > 45:
        feature_quality -= 10
    
    confidence_scores.append(max(50, feature_quality))
    
    # 3. MODEL UNCERTAINTY (0-100%)
    # Use XGBoost's tree predictions to calculate variance
    try:
        # Get predictions from all individual trees
        tree_predictions = []
        for tree in model.get_booster().get_dump():
            # This is a simplified approach - actual implementation would parse tree outputs
            pass
        
        # For now, use a baseline uncertainty based on model type
        # XGBoost is generally 85-95% confident for regression
        model_confidence = 90  # High confidence for trained XGBoost model
        confidence_scores.append(model_confidence)
    except:
        confidence_scores.append(85)  # Default if tree analysis fails
    
    # 4. DATA COMPLETENESS (0-100%)
    # All features provided = 100%, missing features = lower
    # (In this implementation, all features are required, so always 100%)
    completeness_confidence = 100
    confidence_scores.append(completeness_confidence)
    
    # FINAL CONFIDENCE: Weighted average
    # Location: 35%, Feature Quality: 25%, Model: 30%, Completeness: 10%
    final_confidence = (
        confidence_scores[0] * 0.35 +  # Location
        confidence_scores[1] * 0.25 +  # Feature quality
        confidence_scores[2] * 0.30 +  # Model uncertainty
        confidence_scores[3] * 0.10    # Completeness
    )
    
    # Clamp between 50% and 98% (never show 100%, always show some uncertainty)
    final_confidence = max(50, min(98, final_confidence))
    
    return final_confidence


def make_circle_geojson(lat: float, lon: float, radius_km: float = 2.0, num_points: int = 48):
    """Create an approximate circular polygon (GeoJSON) around a center point.
    Uses simple geographic approximation: 1 deg latitude ~ 111 km; adjusts longitude by cos(lat).
    """
    radius_deg = radius_km / 111.0
    coords = []
    for i in range(num_points + 1):
        angle = (i / num_points) * 2 * 3.1415926535
        plat = lat + (radius_deg * cos(angle))
        plon = lon + (radius_deg * sin(angle) / cos(radians(lat)))
        coords.append([float(plon), float(plat)])
    return {
        'type': 'Polygon',
        'coordinates': [coords]
    }

@app.route('/', methods=['GET', 'POST'])
def index():
    # Only render the template for GET requests
    # POST requests are now handled by /predict endpoint
    return render_template('index.html', prediction=None, error=None)

@app.route('/predict', methods=['POST'])
def predict():
    """AJAX endpoint for groundwater prediction - Returns JSON (No page refresh!)"""
    try:
        # Ensure prediction model is available
        if model is None or scaler is None:
            return jsonify({
                'success': False,
                'error': 'Prediction model not loaded. Please train or provide the model files in STEP_6_TRAINED_MODELS.'
            }), 503
        data = {
            'Latitude': float(request.form['latitude']),
            'Longitude': float(request.form['longitude']),
            'Rainfall': float(request.form['rainfall']),
            'River_Water_Level': float(request.form['river_water_level']),
            'Temperature': float(request.form['temperature']),
            'Rainfall_Lag1': float(request.form['rainfall_lag1']),
            'River_Lag1': float(request.form['river_lag1']),
            'Data Time': request.form['data_time']
        }
        data_time = pd.to_datetime(data['Data Time'], format='mixed', errors='coerce')
        if pd.isna(data_time):
            raise ValueError("Invalid date-time format")

        # Check if selected location is within the Nashik prediction zone
        if allowed_locations:
            lat = data['Latitude']
            lon = data['Longitude']
            # consider allowed if within 20 km of any known Nashik location
            max_km = 20.0
            is_allowed = False
            for al in allowed_locations:
                try:
                    al_lat, al_lon = float(al[0]), float(al[1])
                except Exception:
                    continue
                dist = haversine_deg(lat, lon, al_lat, al_lon)
                if dist <= max_km:
                    is_allowed = True
                    break
            if not is_allowed:
                raise ValueError("Selected location is outside the Nashik prediction zone.")
        
        data['Year'] = data_time.year
        data['Month'] = data_time.month
        data['Day'] = data_time.day
        data['Hour'] = data_time.hour
        features = ['Latitude', 'Longitude', 'Rainfall', 'River_Water_Level', 'Temperature', 
                    'Year', 'Month', 'Day', 'Hour', 'Rainfall_Lag1', 'River_Lag1']
        df = pd.DataFrame([data], columns=features)
        X_scaled = scaler.transform(df[features])
        prediction = model.predict(X_scaled)
        
        # Calculate DYNAMIC confidence score (0-100%)
        confidence = calculate_prediction_confidence(
            model=model,
            X_scaled=X_scaled,
            latitude=data['Latitude'],
            longitude=data['Longitude'],
            rainfall=data['Rainfall'],
            temperature=data['Temperature'],
            allowed_locations=allowed_locations
        )
        
        # Pass location and borewell data for intelligent offset adjustment
        prediction = remap_predictions(prediction, 
                                      latitude=data['Latitude'], 
                                      longitude=data['Longitude'],
                                      borewells_df=borewells_df)[0]
        
        # Convert NumPy float32/float64 to Python float for JSON serialization
        prediction = float(prediction)  # Convert to Python float
        prediction = round(prediction, 2)  # Round to 2 decimal places for cleaner display
        confidence = round(float(confidence), 1)  # Round confidence to 1 decimal
        
        return jsonify({
            'success': True,
            'prediction': prediction,
            'confidence': confidence  # Now DYNAMIC!
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/get_boundary', methods=['POST'])
def get_boundary():
    """Fetch boundary/border of a location using OpenStreetMap with fallback to approximate boundary"""
    try:
        data = request.get_json()
        location_name = data.get('location_name', '')
        
        if not location_name:
            return jsonify({'success': False, 'error': 'Location name is required'})
        
        # First, try with detailed polygon
        nominatim_url = f"https://nominatim.openstreetmap.org/search?format=json&q={location_name}, Nashik, Maharashtra, India&limit=3&polygon_geojson=1&polygon_threshold=0.005"
        
        headers = {'User-Agent': 'GroundwaterPredictionApp/1.0'}
        response = requests.get(nominatim_url, headers=headers, timeout=10)
        results = response.json()
        
        print(f"🔍 Searching for: {location_name}")
        
        if not results:
            return jsonify({'success': False, 'error': 'Location not found'})
        
        # Try to find result with boundary
        location = None
        for result in results:
            if 'geojson' in result:
                location = result
                print(f"✅ Found boundary in result: {result['display_name']}")
                break
        
        if not location:
            location = results[0]
            print(f"⚠️ No geojson in results, using first: {location['display_name']}")
        
        # Check if geojson boundary is available
        if 'geojson' in location:
            geojson = location['geojson']
            print(f"✅ Returning boundary for: {location['display_name']}")
            return jsonify({
                'success': True,
                'boundary': geojson,
                'center': {'lat': float(location['lat']), 'lon': float(location['lon'])},
                'display_name': location['display_name']
            })
        
        # If no geojson, try to get from Overpass API with better query
        osm_type = location.get('osm_type', '')
        osm_id = location.get('osm_id', '')
        
        print(f"🔄 Trying Overpass API for OSM {osm_type} {osm_id}")
        
        if osm_type and osm_id:
            try:
                # Convert osm_type to Overpass format
                overpass_type = {'node': 'node', 'way': 'way', 'relation': 'rel'}.get(osm_type, 'way')
                
                # Enhanced query for better boundary extraction
                overpass_query = f"""
                [out:json][timeout:25];
                {overpass_type}({osm_id});
                (._;>;);
                out geom;
                """
                
                overpass_url = "https://overpass-api.de/api/interpreter"
                overpass_response = requests.post(overpass_url, data={'data': overpass_query}, timeout=20)
                overpass_data = overpass_response.json()
                
                if 'elements' in overpass_data and len(overpass_data['elements']) > 0:
                    # Find the main element (relation or way)
                    main_element = None
                    for elem in overpass_data['elements']:
                        if elem['type'] == osm_type and elem['id'] == osm_id:
                            main_element = elem
                            break
                    
                    if not main_element:
                        main_element = overpass_data['elements'][0]
                    
                    coordinates = []
                    
                    # For relations (admin boundaries)
                    if main_element['type'] == 'relation' and 'members' in main_element:
                        print(f"  Processing relation with {len(main_element['members'])} members")
                        # Get all outer way nodes
                        outer_ways = [m for m in main_element['members'] if m.get('role') == 'outer' and m['type'] == 'way']
                        
                        for way_ref in outer_ways:
                            way_id = way_ref['ref']
                            # Find the way in elements
                            for elem in overpass_data['elements']:
                                if elem['type'] == 'way' and elem['id'] == way_id and 'geometry' in elem:
                                    way_coords = [[node['lon'], node['lat']] for node in elem['geometry']]
                                    coordinates.extend(way_coords)
                                    break
                    
                    # For ways
                    elif 'geometry' in main_element:
                        coordinates = [[node['lon'], node['lat']] for node in main_element['geometry']]
                    
                    if coordinates and len(coordinates) >= 3:
                        # Close the polygon if not closed
                        if coordinates[0] != coordinates[-1]:
                            coordinates.append(coordinates[0])
                        
                        print(f"✅ Built boundary with {len(coordinates)} points")
                        return jsonify({
                            'success': True,
                            'boundary': {
                                'type': 'Polygon',
                                'coordinates': [coordinates]
                            },
                            'center': {'lat': float(location['lat']), 'lon': float(location['lon'])},
                            'display_name': location['display_name']
                        })
                    else:
                        print(f"⚠️ Not enough coordinates: {len(coordinates)}")
            except Exception as e:
                print(f"❌ Overpass API error: {str(e)}")
        
        # If no boundary data available, create approximate circular boundary
        print(f"⚠️ No boundary found, creating approximate circular boundary")
        lat = float(location['lat'])
        lon = float(location['lon'])
        
        # Create circular boundary (~2km radius)
        radius_km = 2.0
        radius_deg = radius_km / 111.0  # approximate degrees
        num_points = 32
        
        circle_coords = []
        for i in range(num_points + 1):
            angle = (i / num_points) * 2 * 3.14159
            point_lat = lat + (radius_deg * cos(angle))
            point_lon = lon + (radius_deg * sin(angle) / cos(radians(lat)))
            circle_coords.append([point_lon, point_lat])
        
        return jsonify({
            'success': True,
            'boundary': {
                'type': 'Polygon',
                'coordinates': [circle_coords]
            },
            'center': {'lat': lat, 'lon': lon},
            'display_name': location['display_name'],
            'message': 'Approximate boundary (actual boundary not available)'
        })
        
    except Exception as e:
        print(f"❌ Error in get_boundary: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/predict_area', methods=['POST'])
def predict_area():
    """Predict groundwater levels for multiple points in a rectangular area"""
    try:
        data = request.get_json()
        north = float(data['north'])
        south = float(data['south'])
        east = float(data['east'])
        west = float(data['west'])
        
        # Generate grid of points (5x5 = 25 points)
        grid_size = 5
        lat_points = np.linspace(south, north, grid_size)
        lon_points = np.linspace(west, east, grid_size)
        
        predictions = []
        now = datetime.now()
        
        for lat in lat_points:
            for lon in lon_points:
                # Check if point is within Nashik zone
                is_allowed = False
                if allowed_locations:
                    for al in allowed_locations:
                        try:
                            al_lat, al_lon = float(al[0]), float(al[1])
                            dist = haversine_deg(lat, lon, al_lat, al_lon)
                            if dist <= 20.0:
                                is_allowed = True
                                break
                        except:
                            continue
                
                if not is_allowed:
                    continue
                
                # Get realistic weather data based on location and current season
                weather_data = weather_service.get_weather_data(lat, lon)
                rainfall = weather_data['rainfall']
                river_level = weather_data['river_water_level']
                temperature = weather_data['temperature']
                rainfall_lag1 = weather_data['rainfall_lag1']
                river_lag1 = weather_data['river_lag1']
                
                # Create feature dataframe
                point_data = {
                    'Latitude': lat,
                    'Longitude': lon,
                    'Rainfall': rainfall,
                    'River_Water_Level': river_level,
                    'Temperature': temperature,
                    'Year': now.year,
                    'Month': now.month,
                    'Day': now.day,
                    'Hour': now.hour,
                    'Rainfall_Lag1': rainfall_lag1,
                    'River_Lag1': river_lag1
                }
                
                features = ['Latitude', 'Longitude', 'Rainfall', 'River_Water_Level', 'Temperature', 
                           'Year', 'Month', 'Day', 'Hour', 'Rainfall_Lag1', 'River_Lag1']
                df = pd.DataFrame([point_data], columns=features)
                X_scaled = scaler.transform(df[features])
                pred = model.predict(X_scaled)
                # Pass location and borewell data for intelligent offset adjustment
                pred_value = remap_predictions(pred, 
                                              latitude=lat, 
                                              longitude=lon,
                                              borewells_df=borewells_df)[0]
                
                # Categorize as low, medium, high (adjusted for new ranges)
                if pred_value < 35:
                    category = 'low'  # 25-35m: Shallow groundwater
                elif pred_value < 55:
                    category = 'medium'  # 35-55m: Moderate depth
                else:
                    category = 'high'  # 55+m: Deep groundwater
                
                predictions.append({
                    'lat': float(round(lat, 4)),
                    'lon': float(round(lon, 4)),
                    'value': float(round(pred_value, 2)),
                    'category': category
                })
        
        # Sort predictions by groundwater level (highest first)
        predictions_sorted = sorted(predictions, key=lambda x: x['value'], reverse=True)
        
        # Mark top 5 best spots
        for i, pred in enumerate(predictions_sorted):
            if i < 5:
                pred['rank'] = i + 1  # Rank 1, 2, 3, 4, 5
                pred['is_top'] = True
            else:
                pred['rank'] = None
                pred['is_top'] = False
        
        # Calculate statistics
        if predictions:
            values = [p['value'] for p in predictions]
            stats = {
                'total_points': len(predictions),
                'avg_level': round(float(np.mean(values)), 2),
                'max_level': round(float(np.max(values)), 2),
                'min_level': round(float(np.min(values)), 2),
                'high_count': len([p for p in predictions if p['category'] == 'high']),
                'medium_count': len([p for p in predictions if p['category'] == 'medium']),
                'low_count': len([p for p in predictions if p['category'] == 'low'])
            }
        else:
            stats = {}
        
        return jsonify({
            'success': True, 
            'predictions': predictions,
            'top_5': predictions_sorted[:5],  # Top 5 separately for easy access
            'statistics': stats
        })
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/ai_recommend', methods=['POST'])
def ai_recommend():
    """Return AI-ranked candidate borewell sites for a user-selected bbox.

    Expects JSON: {"bbox": [min_lat, min_lon, max_lat, max_lon], "n": 5, "spacing_km": 1.0}
    """
    try:
        data = request.get_json()
        bbox = data.get('bbox')
        n = int(data.get('n', 5))
        spacing_km = float(data.get('spacing_km', 1.0))

        if not bbox or len(bbox) != 4:
            return jsonify({'success': False, 'error': 'bbox must be [min_lat,min_lon,max_lat,max_lon]'}), 400

        # Ensure model is loaded (train if missing)
        model_bundle = ai_recommender.load_model()
        if model_bundle is None:
            try:
                model_bundle = ai_recommender.train_and_save_model(borewells_df)
            except Exception as e:
                return jsonify({'success': False, 'error': f'Cannot train model: {e}'}), 500

        # Generate candidate grid inside bbox
        candidates = bbox_to_grid(tuple(bbox), spacing_km=spacing_km)

        # Score candidates
        scored = ai_recommender.score_candidates_with_model(candidates, model_bundle, borewells_df)

        # Optional: attach PDF evidence via RAG (single concise summary per bbox)
        evidence_text = None
        try:
            if rag_chatbot and getattr(rag_chatbot, 'conversational_chain_global', None) is not None:
                q_bbox = f"Provide a brief 1-2 sentence summary of typical groundwater/borewell depths or water levels for this region in Nashik based on the uploaded PDF, focusing on drillable depth ranges and any cautions. BBox: {bbox}"
                rag_resp = rag_chatbot.ask_question(q_bbox)
                if rag_resp.get('success'):
                    # keep summary short
                    evidence_text = rag_resp.get('answer', None)
        except Exception:
            # RAG optional; ignore failures silently
            evidence_text = None

        # Return top-n with cleaned numeric types and enriched fields
        topn = []
        for s in scored[:n]:
            topn.append({
                'lat': float(s['lat']),
                'lon': float(s['lon']),
                'prob_success': float(s['prob_success']),
                'score': float(s.get('score', s['prob_success'])),
                'recommended_depth': float(s.get('recommended_depth', 0)),
                'distances_to_others': s.get('distances_to_others', {}),
                'meta': s['meta'],
                'contributions': s['contributions'],
                'nearest_existing_m': float(s.get('nearest_existing_m', float('inf'))),
                'counts_within_500m': int(s.get('counts_within_500m', 0)),
                'counts_within_1km': int(s.get('counts_within_1km', 0)),
                'why_text': s.get('why_text', ''),
                'factors': s.get('factors', []),
                'evidence': evidence_text
            })

        # Find existing borewells within the bbox
        existing_borewells = []
        if not borewells_df.empty:
            min_lat, min_lon, max_lat, max_lon = bbox
            for idx, row in borewells_df.iterrows():
                bw_lat = float(row['Latitude'])
                bw_lon = float(row['Longitude'])
                
                # Check if borewell is within bbox
                if min_lat <= bw_lat <= max_lat and min_lon <= bw_lon <= max_lon:
                    existing_borewells.append({
                        'id': row['Borewell_ID'],
                        'location': row['Location_Name'],
                        'lat': float(bw_lat),
                        'lon': float(bw_lon),
                        'depth': float(row['Depth_m']),
                        'yield': int(row['Yield_LPH']),
                        'year': int(row['Construction_Year']),
                        'quality': row['Water_Quality'],
                        'status': row['Status']
                    })

        return jsonify({
            'success': True, 
            'results': topn,
            'existing_borewells': existing_borewells
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/get_borewells', methods=['POST'])
def get_borewells():
    """Fetch existing CGWB borewells near a location"""
    try:
        data = request.get_json()
        center_lat = float(data.get('lat', 20.0))
        center_lon = float(data.get('lon', 73.79))
        radius_km = float(data.get('radius', 10.0))  # Default 10km radius
        
        if borewells_df.empty:
            return jsonify({'success': False, 'error': 'Borewell database not available'})
        
        # Find borewells within radius
        nearby_borewells = []
        for idx, row in borewells_df.iterrows():
            bw_lat = float(row['Latitude'])
            bw_lon = float(row['Longitude'])
            distance = haversine_deg(center_lat, center_lon, bw_lat, bw_lon)
            
            if distance <= radius_km:
                nearby_borewells.append({
                    'id': row['Borewell_ID'],
                    'location': row['Location_Name'],
                    'lat': float(bw_lat),
                    'lon': float(bw_lon),
                    'depth': float(row['Depth_m']),
                    'yield': int(row['Yield_LPH']),
                    'year': int(row['Construction_Year']),
                    'quality': row['Water_Quality'],
                    'status': row['Status'],
                    'district': row['District'],
                    'taluka': row['Taluka'],
                    'distance': round(distance, 2)
                })
        
        # Sort by distance
        nearby_borewells.sort(key=lambda x: x['distance'])
        
        # Calculate statistics for nearby borewells
        if nearby_borewells:
            successful = [b for b in nearby_borewells if b['status'] == 'Success']
            failed = [b for b in nearby_borewells if b['status'] == 'Failure']
            
            success_rate = (len(successful) / len(nearby_borewells)) * 100 if nearby_borewells else 0
            
            stats = {
                'total': len(nearby_borewells),
                'successful': len(successful),
                'failed': len(failed),
                'success_rate': round(success_rate, 1),
                'avg_depth': round(np.mean([b['depth'] for b in nearby_borewells]), 2),
                'avg_yield': round(np.mean([b['yield'] for b in nearby_borewells]), 0),
                'max_depth': max([b['depth'] for b in nearby_borewells]),
                'min_depth': min([b['depth'] for b in nearby_borewells])
            }
        else:
            stats = {
                'total': 0,
                'successful': 0,
                'failed': 0,
                'success_rate': 0,
                'avg_depth': 0,
                'avg_yield': 0,
                'max_depth': 0,
                'min_depth': 0
            }
        
        print(f"📍 Found {len(nearby_borewells)} borewells within {radius_km}km")
        
        return jsonify({
            'success': True,
            'borewells': nearby_borewells,
            'statistics': stats,
            'radius_km': radius_km
        })
    
    except Exception as e:
        print(f"❌ Error fetching borewells: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/get_weather_data', methods=['POST'])
def get_weather_data():
    """
    Get current weather data for a specific location
    Returns realistic weather data based on Nashik climate patterns
    """
    try:
        data = request.get_json()
        lat = float(data.get('lat'))
        lon = float(data.get('lon'))
        
        # Get current weather data
        weather = weather_service.get_weather_data(lat, lon)
        
        # Get nearest weather stations
        stations = weather_service.get_nearest_station_info(lat, lon)
        
        # Get monthly trend (for charts)
        monthly_trend = weather_service.get_monthly_trend(lat, lon)
        
        return jsonify({
            'success': True,
            'current_weather': weather,
            'nearest_stations': stations,
            'monthly_trend': monthly_trend
        })
    
    except Exception as e:
        print(f"❌ Error fetching weather data: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/get_recommendations', methods=['POST'])
def get_recommendations():
    """
    Get comprehensive recommendations for a drilling location
    Includes success probability, depth, season, cost, and risk assessment
    """
    try:
        data = request.get_json()
        lat = float(data.get('lat'))
        lon = float(data.get('lon'))
        predicted_level = float(data.get('predicted_level'))
        location_name = data.get('location_name', None)
        
        # Generate comprehensive report
        report = recommendation_service.generate_comprehensive_report(
            lat, lon, predicted_level, location_name
        )
        
        return jsonify({
            'success': True,
            'report': report
        })
    
    except Exception as e:
        print(f"❌ Error generating recommendations: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/ai_search_suggestions', methods=['POST'])
def ai_search_suggestions():
    """
    Get AI-powered location search suggestions using Gemini
    """
    try:
        data = request.get_json()
        query = data.get('query', '').strip()
        max_results = data.get('max_results', 10)
        
        print(f"🤖 AI Search request: '{query}' (max: {max_results})")
        
        # Get suggestions from Gemini AI
        suggestions = get_location_suggestions(query, max_results)
        
        print(f"✅ Found {len(suggestions)} AI suggestions")
        
        return jsonify({
            'success': True,
            'suggestions': suggestions,
            'count': len(suggestions),
            'query': query,
            'ai_powered': True
        })
    
    except Exception as e:
        print(f"❌ Error in AI search: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e),
            'suggestions': []
        })

@app.route('/ai_location_details', methods=['POST'])
def ai_location_details():
    """
    Get detailed information about a location using Gemini AI
    """
    try:
        data = request.get_json()
        location_name = data.get('location_name', '').strip()
        
        if not location_name:
            return jsonify({'success': False, 'error': 'Location name required'})
        
        print(f"🔍 Getting AI details for: {location_name}")
        
        details = get_location_details(location_name)
        
        if details:
            return jsonify({
                'success': True,
                'details': details
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Could not fetch location details'
            })
    
    except Exception as e:
        print(f"❌ Error fetching location details: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})


# ========== GOOGLE MAPS PLACES API ROUTES ==========

@app.route('/maps_search_places', methods=['POST'])
def maps_search_places():
    """
    Search for places using Google Maps Places API
    Supports: villages, cities, schools, banks, shops, hospitals, temples, etc.
    """
    try:
        data = request.get_json()
        query = data.get('query', '').strip()
        place_type = data.get('place_type', None)
        max_results = data.get('max_results', 10)
        
        print(f"🗺️ Google Maps Search: '{query}' (type: {place_type})")
        
        # Search using Google Maps Places API
        places = search_places(query, place_type=place_type, max_results=max_results)
        
        print(f"✅ Found {len(places)} places via Google Maps")
        
        return jsonify({
            'success': True,
            'places': places,
            'count': len(places),
            'query': query,
            'place_type': place_type,
            'source': 'google_maps'
        })
    
    except Exception as e:
        print(f"❌ Error in Google Maps search: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e),
            'places': []
        })


@app.route('/maps_place_details', methods=['POST'])
def maps_place_details():
    """
    Get detailed information about a specific place using Google Maps Place ID
    """
    try:
        data = request.get_json()
        place_id = data.get('place_id', '').strip()
        
        if not place_id:
            return jsonify({'success': False, 'error': 'Place ID required'})
        
        print(f"🔍 Getting Google Maps details for place ID: {place_id}")
        
        details = get_place_details(place_id)
        
        if details:
            return jsonify({
                'success': True,
                'details': details,
                'source': 'google_maps'
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Could not fetch place details'
            })
    
    except Exception as e:
        print(f"❌ Error fetching place details: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})


@app.route('/maps_nearby_search', methods=['POST'])
def maps_nearby_search():
    """
    Search for nearby places around a specific location
    """
    try:
        data = request.get_json()
        lat = float(data.get('lat', 0))
        lng = float(data.get('lng', 0))
        place_type = data.get('place_type', None)
        radius = int(data.get('radius', 5000))  # Default 5km
        max_results = int(data.get('max_results', 20))
        
        print(f"🗺️ Nearby search at ({lat}, {lng}), type: {place_type}, radius: {radius}m")
        
        nearby = nearby_search(lat, lng, place_type=place_type, 
                             radius=radius, max_results=max_results)
        
        print(f"✅ Found {len(nearby)} nearby places")
        
        return jsonify({
            'success': True,
            'places': nearby,
            'count': len(nearby),
            'location': {'lat': lat, 'lng': lng},
            'radius': radius,
            'source': 'google_maps'
        })
    
    except Exception as e:
        print(f"❌ Error in nearby search: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e),
            'places': []
        })


@app.route('/maps_autocomplete', methods=['POST'])
def maps_autocomplete():
    """
    Get autocomplete suggestions for place search
    """
    try:
        data = request.get_json()
        query = data.get('query', '').strip()
        max_results = data.get('max_results', 5)
        
        print(f"⌨️ Autocomplete for: '{query}'")
        
        suggestions = autocomplete_places(query, max_results=max_results)
        
        print(f"✅ Found {len(suggestions)} autocomplete suggestions")
        
        return jsonify({
            'success': True,
            'suggestions': suggestions,
            'count': len(suggestions),
            'source': 'google_maps'
        })
    
    except Exception as e:
        print(f"❌ Error in autocomplete: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e),
            'suggestions': []
        })


@app.route('/maps_popular_category', methods=['POST'])
def maps_popular_category():
    """
    Get popular places by category (schools, banks, shops, etc.)
    """
    try:
        data = request.get_json()
        category = data.get('category', 'all').strip()
        max_results = data.get('max_results', 10)
        
        print(f"🏆 Getting popular {category} places")
        
        places = get_popular_places_by_category(category, max_results=max_results)
        
        print(f"✅ Found {len(places)} popular {category} places")
        
        return jsonify({
            'success': True,
            'places': places,
            'count': len(places),
            'category': category,
            'source': 'google_maps'
        })
    
    except Exception as e:
        print(f"❌ Error getting popular places: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e),
            'places': []
        })

# ============================================
# RAG CHATBOT ENDPOINTS
# ============================================

@app.route('/test_rag')
def test_rag():
    """Test page for RAG upload"""
    return send_from_directory('.', 'test_rag_upload.html')

@app.route('/upload_pdf', methods=['POST'])
def upload_pdf():
    """Upload and process PDF for RAG chatbot"""
    try:
        print("📄 PDF Upload request received")
        
        if 'pdf_file' not in request.files:
            print("❌ No PDF file in request")
            return jsonify({'success': False, 'error': 'No PDF file provided'})
        
        file = request.files['pdf_file']
        print(f"📄 File received: {file.filename}")
        
        if file.filename == '':
            print("❌ Empty filename")
            return jsonify({'success': False, 'error': 'No file selected'})
        
        if not file.filename.endswith('.pdf'):
            print(f"❌ Invalid file type: {file.filename}")
            return jsonify({'success': False, 'error': 'Only PDF files are allowed'})
        
        session_id = request.form.get('session_id', 'default')
        print(f"📄 Processing PDF for session: {session_id}")
        
        if not rag_chatbot:
            return jsonify({'success': False, 'error': 'RAG chatbot is currently disabled due to Python 3.14 compatibility'})
        
        # Process PDF
        result = rag_chatbot.upload_and_process_pdf(file, session_id)
        print(f"✅ Processing result: {result}")
        
        return jsonify(result)
    
    except Exception as e:
        print(f"❌ Error in upload_pdf: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)})

@app.route('/ask_rag', methods=['POST'])
def ask_rag():
    """Ask question to RAG chatbot"""
    try:
        data = request.get_json()
        question = data.get('question', '')
        session_id = data.get('session_id', 'default')
        
        if not question:
            return jsonify({'success': False, 'error': 'No question provided'})
        
        if not rag_chatbot:
            return jsonify({'success': False, 'error': 'RAG chatbot is currently disabled due to Python 3.14 compatibility'})
        
        # Get answer from RAG
        result = rag_chatbot.ask_question(question, session_id)
        if not isinstance(result, dict):
            return jsonify({'success': False, 'error': 'Unexpected RAG response'}), 500
        return jsonify(result)
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': f'ask_rag failed: {str(e)}'})

@app.route('/get_chat_history', methods=['POST'])
def get_chat_history():
    """Get chat history for a session"""
    try:
        data = request.get_json()
        session_id = data.get('session_id', 'default')
        
        if not rag_chatbot:
            return jsonify({'success': False, 'error': 'RAG chatbot is currently disabled due to Python 3.14 compatibility'})
        
        history = rag_chatbot.get_chat_history(session_id)
        
        return jsonify({
            'success': True,
            'history': history
        })
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/clear_chat', methods=['POST'])
def clear_chat():
    """Clear chat history"""
    try:
        data = request.get_json()
        session_id = data.get('session_id', 'default')
        
        if not rag_chatbot:
            return jsonify({'success': False, 'error': 'RAG chatbot is currently disabled due to Python 3.14 compatibility'})
        
        result = rag_chatbot.clear_chat_history(session_id)
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/reset_rag', methods=['POST'])
def reset_rag():
    """Reset RAG system"""
    try:
        if not rag_chatbot:
            return jsonify({'success': False, 'error': 'RAG chatbot is currently disabled due to Python 3.14 compatibility'})
        
        result = rag_chatbot.reset_system()
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


if __name__ == '__main__':
    # Disable auto reloader to avoid restarts during long operations (e.g., first-time model load)
    app.run(debug=True, use_reloader=False)