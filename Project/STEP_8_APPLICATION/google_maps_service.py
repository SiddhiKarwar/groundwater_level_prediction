"""
Google Maps Places API Service
Provides comprehensive location search for Nashik District
Supports: Villages, Cities, Schools, Banks, Shops, Institutions, and more
"""

import requests
import json
from typing import List, Dict, Optional

# Google Maps API Configuration
GOOGLE_MAPS_API_KEY = "AIzaSyBxsDSOmOE-ba4T7WdOW_k3P2TAbtiI-pg"

# Nashik District bounds (approximate)
NASHIK_BOUNDS = {
    'north': 20.80,
    'south': 19.50,
    'east': 75.00,
    'west': 73.40
}

# Nashik city center (for radius searches)
NASHIK_CENTER = {
    'lat': 19.9975,
    'lng': 73.7898
}

# Search radius (in meters) - covers most of Nashik district
SEARCH_RADIUS = 80000  # 80km radius

# Place type categories mapping
PLACE_CATEGORIES = {
    'city': ['locality', 'sublocality', 'administrative_area_level_3'],
    'village': ['locality', 'sublocality', 'neighborhood'],
    'school': ['school', 'primary_school', 'secondary_school', 'university'],
    'bank': ['bank', 'atm', 'finance'],
    'shop': ['store', 'shopping_mall', 'supermarket', 'grocery_or_supermarket', 'convenience_store'],
    'hospital': ['hospital', 'health', 'doctor', 'pharmacy'],
    'restaurant': ['restaurant', 'food', 'cafe', 'meal_takeaway', 'meal_delivery'],
    'hotel': ['lodging', 'hotel', 'guest_house'],
    'temple': ['hindu_temple', 'place_of_worship'],
    'mosque': ['mosque', 'place_of_worship'],
    'church': ['church', 'place_of_worship'],
    'park': ['park', 'tourist_attraction'],
    'government': ['local_government_office', 'city_hall', 'courthouse'],
    'police': ['police', 'fire_station'],
    'post_office': ['post_office'],
    'gas_station': ['gas_station'],
    'bus_station': ['bus_station', 'transit_station'],
    'railway': ['train_station', 'transit_station'],
    'airport': ['airport'],
    'industrial': ['industrial_area'],
    'market': ['market', 'bazaar'],
    'all': []  # No type filter - search everything
}


def search_places(query: str, place_type: Optional[str] = None, max_results: int = 10) -> List[Dict]:
    """
    Search for places in Nashik District using Google Maps Places API
    
    Args:
        query: Search query (e.g., "schools near malegaon", "banks in nashik")
        place_type: Type of place (city, village, school, bank, shop, etc.)
        max_results: Maximum number of results to return
        
    Returns:
        List of places with details (name, address, coordinates, type, etc.)
    """
    
    if not query or len(query.strip()) < 2:
        return []
    
    try:
        # Determine search types based on category
        types = PLACE_CATEGORIES.get(place_type, []) if place_type else []
        
        # Build query with Nashik District context
        search_query = f"{query} in Nashik District, Maharashtra"
        
        # Use Text Search API for comprehensive results
        url = "https://maps.googleapis.com/maps/api/place/textsearch/json"
        
        params = {
            'query': search_query,
            'key': GOOGLE_MAPS_API_KEY,
            'location': f"{NASHIK_CENTER['lat']},{NASHIK_CENTER['lng']}",
            'radius': SEARCH_RADIUS,
            'language': 'en'
        }
        
        # Add type filter if specified
        if types and len(types) > 0:
            params['type'] = types[0]  # Use first type as primary filter
        
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        
        if data.get('status') != 'OK':
            print(f"⚠️ Google Maps API returned status: {data.get('status')}")
            return []
        
        results = data.get('results', [])[:max_results]
        
        # Format results
        places = []
        for place in results:
            location = place.get('geometry', {}).get('location', {})
            lat = location.get('lat', 0)
            lng = location.get('lng', 0)
            
            # Filter to ensure place is within Nashik District bounds
            if not is_in_nashik_district(lat, lng):
                continue
            
            place_info = {
                'name': place.get('name', 'Unknown'),
                'address': place.get('formatted_address', 'Address not available'),
                'latitude': lat,
                'longitude': lng,
                'place_id': place.get('place_id', ''),
                'types': place.get('types', []),
                'category': determine_category(place.get('types', [])),
                'rating': place.get('rating', 0),
                'user_ratings_total': place.get('user_ratings_total', 0),
                'business_status': place.get('business_status', 'OPERATIONAL'),
                'icon': place.get('icon', ''),
                'vicinity': place.get('vicinity', '')
            }
            
            places.append(place_info)
        
        return places
        
    except requests.RequestException as e:
        print(f"❌ Error fetching from Google Maps API: {e}")
        return []
    except Exception as e:
        print(f"❌ Unexpected error in search_places: {e}")
        return []


def get_place_details(place_id: str) -> Optional[Dict]:
    """
    Get detailed information about a specific place
    
    Args:
        place_id: Google Maps Place ID
        
    Returns:
        Detailed place information
    """
    
    try:
        url = "https://maps.googleapis.com/maps/api/place/details/json"
        
        params = {
            'place_id': place_id,
            'key': GOOGLE_MAPS_API_KEY,
            'fields': 'name,formatted_address,geometry,types,rating,user_ratings_total,opening_hours,formatted_phone_number,website,reviews,photos,business_status',
            'language': 'en'
        }
        
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        
        if data.get('status') != 'OK':
            return None
        
        result = data.get('result', {})
        location = result.get('geometry', {}).get('location', {})
        
        return {
            'name': result.get('name', 'Unknown'),
            'address': result.get('formatted_address', ''),
            'latitude': location.get('lat', 0),
            'longitude': location.get('lng', 0),
            'types': result.get('types', []),
            'category': determine_category(result.get('types', [])),
            'rating': result.get('rating', 0),
            'user_ratings_total': result.get('user_ratings_total', 0),
            'phone': result.get('formatted_phone_number', ''),
            'website': result.get('website', ''),
            'business_status': result.get('business_status', ''),
            'opening_hours': result.get('opening_hours', {}),
            'reviews': result.get('reviews', [])[:3]  # Top 3 reviews
        }
        
    except Exception as e:
        print(f"❌ Error fetching place details: {e}")
        return None


def nearby_search(lat: float, lng: float, place_type: Optional[str] = None, radius: int = 5000, max_results: int = 20) -> List[Dict]:
    """
    Search for nearby places around a specific location
    
    Args:
        lat: Latitude
        lng: Longitude
        place_type: Type of place to search for
        radius: Search radius in meters (default 5km)
        max_results: Maximum number of results
        
    Returns:
        List of nearby places
    """
    
    try:
        url = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"
        
        params = {
            'location': f"{lat},{lng}",
            'radius': radius,
            'key': GOOGLE_MAPS_API_KEY,
            'language': 'en'
        }
        
        # Add type filter if specified
        types = PLACE_CATEGORIES.get(place_type, []) if place_type else []
        if types and len(types) > 0:
            params['type'] = types[0]
        
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        
        if data.get('status') not in ['OK', 'ZERO_RESULTS']:
            print(f"⚠️ Nearby search returned status: {data.get('status')}")
            return []
        
        results = data.get('results', [])[:max_results]
        
        places = []
        for place in results:
            location = place.get('geometry', {}).get('location', {})
            
            place_info = {
                'name': place.get('name', 'Unknown'),
                'address': place.get('vicinity', ''),
                'latitude': location.get('lat', 0),
                'longitude': location.get('lng', 0),
                'place_id': place.get('place_id', ''),
                'types': place.get('types', []),
                'category': determine_category(place.get('types', [])),
                'rating': place.get('rating', 0),
                'user_ratings_total': place.get('user_ratings_total', 0),
                'business_status': place.get('business_status', 'OPERATIONAL'),
                'distance': calculate_distance(lat, lng, location.get('lat', 0), location.get('lng', 0))
            }
            
            places.append(place_info)
        
        return places
        
    except Exception as e:
        print(f"❌ Error in nearby search: {e}")
        return []


def autocomplete_places(query: str, max_results: int = 5) -> List[Dict]:
    """
    Get autocomplete suggestions for place search
    
    Args:
        query: Partial search query
        max_results: Maximum suggestions to return
        
    Returns:
        List of autocomplete suggestions
    """
    
    if not query or len(query.strip()) < 2:
        return []
    
    try:
        url = "https://maps.googleapis.com/maps/api/place/autocomplete/json"
        
        params = {
            'input': query,
            'key': GOOGLE_MAPS_API_KEY,
            'location': f"{NASHIK_CENTER['lat']},{NASHIK_CENTER['lng']}",
            'radius': SEARCH_RADIUS,
            'components': 'country:in',
            'language': 'en'
        }
        
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        
        if data.get('status') not in ['OK', 'ZERO_RESULTS']:
            return []
        
        predictions = data.get('predictions', [])[:max_results]
        
        suggestions = []
        for pred in predictions:
            suggestions.append({
                'description': pred.get('description', ''),
                'place_id': pred.get('place_id', ''),
                'main_text': pred.get('structured_formatting', {}).get('main_text', ''),
                'secondary_text': pred.get('structured_formatting', {}).get('secondary_text', ''),
                'types': pred.get('types', [])
            })
        
        return suggestions
        
    except Exception as e:
        print(f"❌ Error in autocomplete: {e}")
        return []


def is_in_nashik_district(lat: float, lng: float) -> bool:
    """Check if coordinates are within Nashik District bounds"""
    return (NASHIK_BOUNDS['south'] <= lat <= NASHIK_BOUNDS['north'] and
            NASHIK_BOUNDS['west'] <= lng <= NASHIK_BOUNDS['east'])


def determine_category(types: List[str]) -> str:
    """Determine the primary category of a place based on its types"""
    
    # Priority order for categorization
    priority_categories = [
        ('school', ['school', 'primary_school', 'secondary_school', 'university']),
        ('bank', ['bank', 'atm']),
        ('hospital', ['hospital', 'health', 'doctor']),
        ('restaurant', ['restaurant', 'food', 'cafe']),
        ('hotel', ['lodging', 'hotel']),
        ('temple', ['hindu_temple']),
        ('mosque', ['mosque']),
        ('church', ['church']),
        ('shop', ['store', 'shopping_mall', 'supermarket']),
        ('government', ['local_government_office', 'city_hall']),
        ('police', ['police']),
        ('village', ['locality', 'sublocality', 'neighborhood']),
        ('city', ['locality', 'administrative_area_level_3'])
    ]
    
    for category, type_list in priority_categories:
        if any(t in types for t in type_list):
            return category
    
    return 'other'


def calculate_distance(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    """Calculate distance between two coordinates in kilometers"""
    from math import radians, sin, cos, sqrt, atan2
    
    R = 6371  # Earth's radius in km
    
    lat1_rad = radians(lat1)
    lat2_rad = radians(lat2)
    delta_lat = radians(lat2 - lat1)
    delta_lng = radians(lng2 - lng1)
    
    a = sin(delta_lat / 2) ** 2 + cos(lat1_rad) * cos(lat2_rad) * sin(delta_lng / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    
    return round(R * c, 2)


def get_popular_places_by_category(category: str, max_results: int = 10) -> List[Dict]:
    """
    Get popular places in Nashik District by category
    
    Args:
        category: Category (school, bank, shop, village, city, etc.)
        max_results: Maximum results to return
        
    Returns:
        List of popular places in that category
    """
    
    category_queries = {
        'school': 'top schools in Nashik District',
        'bank': 'major banks in Nashik',
        'shop': 'shopping centers in Nashik',
        'hospital': 'hospitals in Nashik District',
        'restaurant': 'popular restaurants in Nashik',
        'hotel': 'hotels in Nashik',
        'temple': 'famous temples in Nashik District',
        'village': 'villages in Nashik District',
        'city': 'cities and towns in Nashik District',
        'tourist': 'tourist places in Nashik District'
    }
    
    query = category_queries.get(category, f'{category} in Nashik District')
    return search_places(query, place_type=category, max_results=max_results)


# Test function
def test_api():
    """Test the Google Maps API integration"""
    
    print("\n=== Testing Google Maps Places API ===\n")
    
    # Test 1: Search for schools
    print("1. Searching for schools in Nashik...")
    schools = search_places("schools near nashik", place_type="school", max_results=5)
    print(f"   Found {len(schools)} schools:")
    for school in schools[:3]:
        print(f"   - {school['name']} ({school['address']})")
    
    # Test 2: Search for banks
    print("\n2. Searching for banks in Malegaon...")
    banks = search_places("banks in malegaon", place_type="bank", max_results=5)
    print(f"   Found {len(banks)} banks:")
    for bank in banks[:3]:
        print(f"   - {bank['name']} ({bank['vicinity']})")
    
    # Test 3: Autocomplete
    print("\n3. Testing autocomplete for 'nashi'...")
    suggestions = autocomplete_places("nashi", max_results=5)
    print(f"   Found {len(suggestions)} suggestions:")
    for sug in suggestions:
        print(f"   - {sug['description']}")
    
    # Test 4: Nearby search
    print("\n4. Searching for nearby shops around Nashik center...")
    nearby = nearby_search(NASHIK_CENTER['lat'], NASHIK_CENTER['lng'], 
                          place_type='shop', radius=3000, max_results=5)
    print(f"   Found {len(nearby)} shops:")
    for shop in nearby[:3]:
        print(f"   - {shop['name']} ({shop['distance']}km away)")
    
    print("\n=== API Test Complete ===\n")


if __name__ == "__main__":
    test_api()
