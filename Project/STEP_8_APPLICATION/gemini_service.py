"""
Gemini AI Service for Intelligent Location Search
Provides dynamic, context-aware location suggestions for Nashik District
Enhanced with Google Maps integration for comprehensive place search
"""

import json
import os
import re

# Try to import Google Gemini SDK (optional). If unavailable, fall back to local functions.
try:
    import google.generativeai as genai
    # Read API key from environment to avoid committing secrets in code
    GEMINI_API_KEY = os.getenv("GENAI_API_KEY", "")
    if GEMINI_API_KEY:
        try:
            genai.configure(api_key=GEMINI_API_KEY)
        except Exception:
            # configuration failed but SDK is still importable
            pass

    # Initialize Gemini model handle if available (guarded)
    try:
        model = genai.GenerativeModel('gemini-2.0-flash-exp')
    except Exception:
        model = None
    HAS_GENAI = True
except Exception:
    genai = None
    model = None
    HAS_GENAI = False

# Google Maps API Key (for real place data) - optional; read from env
# Set GOOGLE_MAPS_API_KEY in your environment instead of hardcoding
GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY", "")

def get_location_suggestions(query, max_results=10):
    """
    Get intelligent location suggestions for Nashik District using Gemini AI
    Enhanced to detect place types: villages, cities, schools, banks, shops, temples, hospitals, etc.
    
    Args:
        query (str): User's search query
        max_results (int): Maximum number of suggestions to return
        
    Returns:
        list: List of location suggestions with details
    """
    
    # If Gemini SDK isn't available, or query is short, return popular/local suggestions
    if not HAS_GENAI or not model or not query or len(query.strip()) < 2:
        # For short queries, return the popular list or local matching
        popular = get_popular_locations()
        q = (query or '').lower().strip()
        if not q:
            return popular[:max_results]
        matches = [p for p in popular if q in p['name'].lower() or any(q in kw for kw in p.get('keywords', []))]
        return matches[:max_results] if matches else popular[:max_results]

    # If we have Gemini, call it (guarded)
    try:
        prompt = f"You are a location search assistant for Nashik District. User is searching for: '{query}'\nReturn a JSON array of suggestions."
        response = model.generate_content(prompt)
        response_text = response.text.strip()
        json_match = re.search(r'\[.*\]', response_text, re.DOTALL)
        if json_match:
            suggestions = json.loads(json_match.group())
            validated = []
            for idx, sug in enumerate(suggestions):
                if 'name' in sug:
                    sug.setdefault('category', '')
                    sug.setdefault('description', '')
                    sug.setdefault('lat', 20.0)
                    sug.setdefault('lon', 73.79)
                    sug.setdefault('keywords', [sug['name'].lower()])
                    sug.setdefault('popular', False)
                    sug['priority'] = idx + 1
                    validated.append(sug)
            return validated[:max_results]
        else:
            return get_fallback_suggestions(query)
    except Exception as e:
        print(f"Error getting Gemini suggestions: {e}")
        return get_fallback_suggestions(query)


def get_popular_locations():
    """
    Return popular/trending locations when no query is provided
    """
    return [
        {
            "name": "Nashik",
            "category": "🏙️ City",
            "description": "Wine capital of India • Major pilgrimage center • Population 15 lakh+",
            "keywords": ["nashik", "nasik", "city", "wine", "godavari"],
            "popular": True,
            "trending": True,
            "priority": 1,
            "lat": 20.0,
            "lon": 73.79
        },
        {
            "name": "Trimbakeshwar",
            "category": "🕉️ Religious",
            "description": "Jyotirlinga temple • Origin of Godavari River • Popular pilgrimage",
            "keywords": ["trimbak", "temple", "jyotirlinga", "religious"],
            "popular": True,
            "trending": True,
            "priority": 2,
            "lat": 19.9333,
            "lon": 73.5333
        },
        {
            "name": "Malegaon",
            "category": "🏙️ City",
            "description": "2nd largest city • Textile industry hub • Population 5 lakh+",
            "keywords": ["malegaon", "textile", "city"],
            "popular": True,
            "priority": 3,
            "lat": 20.5500,
            "lon": 74.5333
        },
        {
            "name": "Sinnar",
            "category": "🏘️ Town",
            "description": "Industrial hub • MIDC area • Rapidly growing town",
            "keywords": ["sinnar", "industrial", "midc"],
            "popular": True,
            "priority": 4,
            "lat": 19.8472,
            "lon": 73.9975
        },
        {
            "name": "Igatpuri",
            "category": "⛰️ Hill Station",
            "description": "Western Ghats • High rainfall • Trekking destination",
            "keywords": ["igatpuri", "ghat", "hill station", "trekking"],
            "popular": True,
            "trending": True,
            "priority": 5,
            "lat": 19.6958,
            "lon": 73.5631
        },
        {
            "name": "Gangapur Dam",
            "category": "💧 Dam",
            "description": "Major water reservoir • Supplies water to Nashik city",
            "keywords": ["gangapur", "dam", "water", "reservoir"],
            "popular": True,
            "priority": 6,
            "lat": 20.0167,
            "lon": 74.0167
        },
        {
            "name": "Lasalgaon",
            "category": "🧅 Market",
            "description": "Asia's largest onion market • Major agricultural hub",
            "keywords": ["lasalgaon", "onion", "market", "agriculture"],
            "popular": True,
            "trending": True,
            "priority": 7,
            "lat": 20.1333,
            "lon": 74.2333
        },
        {
            "name": "Panchavati",
            "category": "🕉️ Religious",
            "description": "Godavari riverside • Ramayan connection • Sacred bathing ghats",
            "keywords": ["panchavati", "godavari", "religious", "ghat"],
            "popular": True,
            "priority": 8,
            "lat": 19.9975,
            "lon": 73.7898
        }
    ]


def get_fallback_suggestions(query):
    """
    Fallback suggestions when Gemini API fails
    Simple keyword matching from popular locations
    """
    popular = get_popular_locations()
    
    if not query:
        return popular
    
    query_lower = query.lower()
    
    # Filter based on name or keywords
    matches = [
        loc for loc in popular
        if query_lower in loc['name'].lower() or
        any(query_lower in kw for kw in loc['keywords'])
    ]
    
    return matches if matches else popular[:5]


def get_location_details(location_name):
    """
    Get detailed information about a specific location using Gemini AI
    """
    # If Gemini isn't available, return a simple local description where possible
    if not HAS_GENAI or not model:
        pops = get_popular_locations()
        lname = (location_name or '').lower()
        for p in pops:
            if p['name'].lower() == lname or lname in p['name'].lower():
                return {
                    'name': p['name'],
                    'type': p.get('category', ''),
                    'description': p.get('description', ''),
                    'features': [],
                    'population': 'N/A',
                    'coordinates': {'lat': p.get('lat', 20.0), 'lon': p.get('lon', 73.79)}
                }
        return None

    try:
        prompt = f"Provide detailed information about '{location_name}' in Nashik District. Return JSON."
        response = model.generate_content(prompt)
        response_text = response.text.strip()
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
        return None
    except Exception as e:
        print(f"Error getting location details: {e}")
        return None


# Test function
if __name__ == "__main__":
    print("Testing Gemini Location Search Service...")
    print("\n1. Testing with query 'college':")
    results = get_location_suggestions("college", max_results=5)
    for r in results:
        print(f"  - {r['name']} ({r['category']}): {r['description']}")
    
    print("\n2. Testing with query 'dam':")
    results = get_location_suggestions("dam", max_results=5)
    for r in results:
        print(f"  - {r['name']} ({r['category']}): {r['description']}")
    
    print("\n3. Testing popular locations (empty query):")
    results = get_location_suggestions("", max_results=5)
    for r in results:
        print(f"  - {r['name']} ({r['category']}): {r['description']}")
