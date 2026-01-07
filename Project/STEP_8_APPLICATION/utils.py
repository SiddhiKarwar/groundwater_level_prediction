import numpy as np

def remap_predictions(predictions, latitude=None, longitude=None, borewells_df=None):
    """
    Remap predictions to realistic Nashik groundwater depth ranges (25-80m minimum)
    
    Args:
        predictions: Raw model predictions
        latitude: Location latitude (optional, for location-based adjustments)
        longitude: Location longitude (optional, for location-based adjustments)
        borewells_df: DataFrame of existing borewells (optional, for intelligent offset)
    
    Returns:
        Adjusted predictions with minimum 25m depth
    """
    # Base remapping: Convert to positive values and scale to Nashik realistic range
    result = np.abs(predictions) + 25  # Start from 25m minimum
    
    for i in range(len(result)):
        # Ensure values are within realistic Nashik range (25m to 80m)
        while result[i] > 80 or result[i] < 25:
            if result[i] > 80:
                result[i] = 80 - (result[i] - 80) * 0.5  # Soft cap at 80m
            elif result[i] < 25:
                result[i] = 25  # Hard floor at 25m
            
            # Safety check: if still out of range, force into range
            if result[i] > 80:
                result[i] = 80
            elif result[i] < 25:
                result[i] = 25
    
    # Location-based intelligent offset adjustment
    if latitude is not None and longitude is not None and borewells_df is not None:
        try:
            location_offset = calculate_location_offset(latitude, longitude, borewells_df)
            result = result + location_offset
            
            # Ensure adjusted values still in reasonable range (25-100m)
            result = np.clip(result, 25, 100)
        except Exception as e:
            # If offset calculation fails, use base result
            pass
    
    return result


def calculate_location_offset(lat, lon, borewells_df):
    """
    Calculate intelligent offset based on nearby successful borewells
    
    Logic:
    - If near successful shallow borewells (30-45m): Add +5 to +15m offset
    - If near successful medium borewells (45-55m): Add +15 to +25m offset
    - If near successful deep borewells (55-65m): Add +25 to +35m offset
    - If near failed borewells: Add +10 to +20m offset (need to drill deeper)
    - If no nearby borewells: Add +20m default offset
    
    Returns:
        Offset value to add to prediction (10-40m range)
    """
    if borewells_df is None or borewells_df.empty:
        return 20  # Default offset if no borewell data
    
    from math import radians, cos, sin, asin, sqrt
    
    def haversine(lat1, lon1, lat2, lon2):
        """Calculate distance in kilometers"""
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * asin(sqrt(a))
        km = 6371 * c
        return km
    
    # Find borewells within 10km radius
    nearby_borewells = []
    for idx, row in borewells_df.iterrows():
        try:
            bw_lat = float(row['Latitude'])
            bw_lon = float(row['Longitude'])
            distance = haversine(lat, lon, bw_lat, bw_lon)
            
            if distance <= 10:  # Within 10km
                nearby_borewells.append({
                    'distance': distance,
                    'depth': float(row['Depth_m']),
                    'status': row['Status'],
                    'yield': int(row['Yield_LPH'])
                })
        except:
            continue
    
    # Calculate offset based on nearby borewells
    if not nearby_borewells:
        # No nearby borewells - use default offset
        return 20
    
    # Sort by distance (nearest first)
    nearby_borewells.sort(key=lambda x: x['distance'])
    
    # Focus on 3 nearest borewells
    nearest = nearby_borewells[:3]
    
    # Calculate average depth of successful borewells
    successful = [b for b in nearest if b['status'] == 'Success']
    failed = [b for b in nearest if b['status'] == 'Failure']
    
    if successful:
        avg_depth = np.mean([b['depth'] for b in successful])
        avg_distance = np.mean([b['distance'] for b in successful])
        
        # Offset based on depth range of successful borewells
        if avg_depth <= 40:
            # Shallow borewells nearby - add 10-15m
            base_offset = 12
        elif avg_depth <= 50:
            # Medium depth borewells - add 15-25m
            base_offset = 20
        elif avg_depth <= 60:
            # Deep borewells - add 25-35m
            base_offset = 30
        else:
            # Very deep borewells - add 35-40m
            base_offset = 37
        
        # Adjust offset based on distance (closer = more reliable)
        if avg_distance < 2:  # Very close (< 2km)
            distance_factor = 1.2
        elif avg_distance < 5:  # Close (2-5km)
            distance_factor = 1.0
        else:  # Far (5-10km)
            distance_factor = 0.8
        
        offset = base_offset * distance_factor
        
    elif failed:
        # Only failed borewells nearby - need to drill deeper
        avg_failed_depth = np.mean([b['depth'] for b in failed])
        # Add 15-25m more than failed attempts
        offset = avg_failed_depth * 0.4 + 15
    else:
        # Default offset
        offset = 20
    
    # Ensure offset is within reasonable range (10-40m)
    offset = np.clip(offset, 10, 40)
    
    return offset