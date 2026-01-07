"""
Test script to verify AI-powered location search integration
"""
import requests
import json

BASE_URL = "http://127.0.0.1:5000"

def test_ai_search():
    print("🧪 Testing AI Search Integration...\n")
    
    # Test 1: Search for "college"
    print("📍 Test 1: Searching for 'college'...")
    response = requests.post(
        f"{BASE_URL}/ai_search_suggestions",
        json={"query": "college", "max_results": 5},
        headers={"Content-Type": "application/json"}
    )
    
    if response.status_code == 200:
        data = response.json()
        if data['success']:
            print(f"✅ Found {data['count']} suggestions (AI-powered: {data['ai_powered']})")
            for idx, suggestion in enumerate(data['suggestions'], 1):
                print(f"   {idx}. {suggestion['name']} ({suggestion['category']})")
                print(f"      📝 {suggestion['description']}")
                if suggestion.get('lat') and suggestion.get('lon'):
                    print(f"      📍 Coordinates: {suggestion['lat']}, {suggestion['lon']}")
                print()
        else:
            print(f"❌ Error: {data.get('error', 'Unknown error')}")
    else:
        print(f"❌ HTTP Error: {response.status_code}")
    
    print("\n" + "="*60 + "\n")
    
    # Test 2: Search for "dam"
    print("📍 Test 2: Searching for 'dam'...")
    response = requests.post(
        f"{BASE_URL}/ai_search_suggestions",
        json={"query": "dam", "max_results": 5},
        headers={"Content-Type": "application/json"}
    )
    
    if response.status_code == 200:
        data = response.json()
        if data['success']:
            print(f"✅ Found {data['count']} suggestions (AI-powered: {data['ai_powered']})")
            for idx, suggestion in enumerate(data['suggestions'], 1):
                print(f"   {idx}. {suggestion['name']} ({suggestion['category']})")
                print(f"      📝 {suggestion['description']}")
                print()
        else:
            print(f"❌ Error: {data.get('error', 'Unknown error')}")
    else:
        print(f"❌ HTTP Error: {response.status_code}")
    
    print("\n" + "="*60 + "\n")
    
    # Test 3: Get popular locations (empty query)
    print("📍 Test 3: Getting popular locations (empty query)...")
    response = requests.post(
        f"{BASE_URL}/ai_search_suggestions",
        json={"query": "", "max_results": 8},
        headers={"Content-Type": "application/json"}
    )
    
    if response.status_code == 200:
        data = response.json()
        if data['success']:
            print(f"✅ Found {data['count']} popular locations")
            for idx, suggestion in enumerate(data['suggestions'], 1):
                badges = []
                if suggestion.get('popular'):
                    badges.append('⭐ Popular')
                if suggestion.get('trending'):
                    badges.append('🔥 Trending')
                badge_str = ' '.join(badges) if badges else ''
                print(f"   {idx}. {suggestion['name']} {badge_str}")
                print(f"      {suggestion['category']} - {suggestion['description']}")
                print()
        else:
            print(f"❌ Error: {data.get('error', 'Unknown error')}")
    else:
        print(f"❌ HTTP Error: {response.status_code}")
    
    print("\n" + "="*60 + "\n")
    
    # Test 4: Get location details
    print("📍 Test 4: Getting details for 'Nashik City'...")
    response = requests.post(
        f"{BASE_URL}/ai_location_details",
        json={"location_name": "Nashik City"},
        headers={"Content-Type": "application/json"}
    )
    
    if response.status_code == 200:
        data = response.json()
        if data['success']:
            details = data['details']
            print(f"✅ Location Details:")
            print(f"   Name: {details['name']}")
            print(f"   Category: {details['category']}")
            print(f"   Description: {details['description']}")
            if details.get('lat') and details.get('lon'):
                print(f"   Coordinates: {details['lat']}, {details['lon']}")
        else:
            print(f"❌ Error: {data.get('error', 'Unknown error')}")
    else:
        print(f"❌ HTTP Error: {response.status_code}")

if __name__ == "__main__":
    print("\n" + "="*60)
    print("   🤖 AI-POWERED SEARCH TEST SUITE")
    print("   Gemini API Integration Verification")
    print("="*60 + "\n")
    
    try:
        test_ai_search()
        print("\n✅ All tests completed!")
    except Exception as e:
        print(f"\n❌ Test failed with error: {str(e)}")
