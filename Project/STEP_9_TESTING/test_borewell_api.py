"""
Test the new Borewell Depth Prediction API
Validates predictions against known borewell locations
"""

import requests
import json

print("=" * 80)
print("🧪 TESTING BOREWELL DEPTH PREDICTION API")
print("=" * 80)

# Base URL (Flask should be running)
BASE_URL = "http://127.0.0.1:5000"

# Test cases with expected depth ranges
test_cases = [
    {
        'name': 'Nashik City Center',
        'lat': 19.9975,
        'lon': 73.7898,
        'expected_range': (35, 55),
        'actual_depth': 45.5,
        'notes': 'Central business district, known borewell BW001'
    },
    {
        'name': 'Malegaon',
        'lat': 20.5537,
        'lon': 74.5288,
        'expected_range': (40, 60),
        'actual_depth': 52.0,
        'notes': 'Northern Nashik district, BW002'
    },
    {
        'name': 'Sinnar',
        'lat': 19.8540,
        'lon': 74.0005,
        'expected_range': (30, 50),
        'actual_depth': 38.2,
        'notes': 'Southern zone, BW003'
    },
    {
        'name': 'Trimbak (High Rainfall)',
        'lat': 19.9328,
        'lon': 73.5292,
        'expected_range': (20, 40),
        'actual_depth': 35.5,
        'notes': 'Western ghats influence, BW007'
    },
    {
        'name': 'Niphad',
        'lat': 20.0751,
        'lon': 74.1116,
        'expected_range': (35, 55),
        'actual_depth': 48.7,
        'notes': 'Central zone, BW004'
    }
]

print("\n📍 Testing 5 locations across Nashik District...\n")

results = []

for idx, test in enumerate(test_cases, 1):
    print(f"{idx}. Testing: {test['name']}")
    print(f"   Location: ({test['lat']}, {test['lon']})")
    print(f"   Expected Range: {test['expected_range'][0]}-{test['expected_range'][1]}m")
    print(f"   Actual Known Depth: {test['actual_depth']}m")
    
    try:
        # Make API request
        response = requests.post(
            f"{BASE_URL}/predict_borewell_depth",
            json={'lat': test['lat'], 'lon': test['lon']},
            timeout=10
        )
        
        if response.status_code == 200:
            data = response.json()
            
            if data['success']:
                pred_depth = data['prediction']['depth_m']
                pred_range = data['prediction']['depth_range']
                yield_est = data['prediction']['estimated_yield_lph']
                success_prob = data['prediction']['success_probability']
                geo_zone = data['geological_info']['zone']
                soil = data['geological_info']['soil_type']
                aquifer = data['geological_info']['aquifer_type']
                
                # Calculate error
                error = abs(pred_depth - test['actual_depth'])
                error_percent = (error / test['actual_depth']) * 100
                
                # Check if within expected range
                within_expected = test['expected_range'][0] <= pred_depth <= test['expected_range'][1]
                within_5m = error <= 5
                within_10m = error <= 10
                
                print(f"   ✅ Predicted Depth: {pred_depth}m ({pred_range['min']}-{pred_range['max']}m)")
                print(f"   📊 Error: {error:.1f}m ({error_percent:.1f}%)")
                print(f"   💧 Estimated Yield: {yield_est} LPH")
                print(f"   🎯 Success Probability: {success_prob}%")
                print(f"   🌍 Geological Zone: {geo_zone}")
                print(f"   🏔️ Soil: {soil} | Aquifer: {aquifer}")
                
                if within_5m:
                    print(f"   ✅ EXCELLENT: Within ±5m of actual depth!")
                elif within_10m:
                    print(f"   ✅ GOOD: Within ±10m of actual depth")
                elif within_expected:
                    print(f"   ✅ ACCEPTABLE: Within expected range")
                else:
                    print(f"   ⚠️ WARNING: Outside expected range")
                
                results.append({
                    'name': test['name'],
                    'predicted': pred_depth,
                    'actual': test['actual_depth'],
                    'error': error,
                    'within_5m': within_5m,
                    'within_10m': within_10m,
                    'status': 'PASS' if within_10m else 'FAIL'
                })
            else:
                print(f"   ❌ API Error: {data.get('error', 'Unknown error')}")
                results.append({
                    'name': test['name'],
                    'status': 'ERROR',
                    'error': data.get('error')
                })
        else:
            print(f"   ❌ HTTP Error: {response.status_code}")
            results.append({
                'name': test['name'],
                'status': 'HTTP_ERROR',
                'code': response.status_code
            })
    
    except requests.exceptions.ConnectionError:
        print(f"   ❌ CONNECTION ERROR: Flask server not running!")
        print(f"   💡 Start server with: python app.py")
        results.append({
            'name': test['name'],
            'status': 'CONNECTION_ERROR'
        })
        break
    
    except Exception as e:
        print(f"   ❌ Unexpected Error: {e}")
        results.append({
            'name': test['name'],
            'status': 'EXCEPTION',
            'error': str(e)
        })
    
    print()

# Summary
print("=" * 80)
print("📊 TEST RESULTS SUMMARY")
print("=" * 80)

successful_tests = [r for r in results if r.get('status') == 'PASS']
failed_tests = [r for r in results if r.get('status') == 'FAIL']
error_tests = [r for r in results if r.get('status') not in ['PASS', 'FAIL']]

print(f"\n✅ Successful Tests: {len(successful_tests)}/{len(test_cases)}")
print(f"❌ Failed Tests: {len(failed_tests)}/{len(test_cases)}")
print(f"⚠️ Error Tests: {len(error_tests)}/{len(test_cases)}")

if successful_tests:
    errors = [r['error'] for r in successful_tests if 'error' in r]
    within_5m_count = sum(1 for r in successful_tests if r.get('within_5m'))
    within_10m_count = sum(1 for r in successful_tests if r.get('within_10m'))
    
    if errors:
        print(f"\n📈 Accuracy Metrics:")
        print(f"   Average Error: {sum(errors)/len(errors):.2f}m")
        print(f"   Min Error: {min(errors):.2f}m")
        print(f"   Max Error: {max(errors):.2f}m")
        print(f"   Within ±5m: {within_5m_count}/{len(successful_tests)} ({within_5m_count/len(successful_tests)*100:.1f}%)")
        print(f"   Within ±10m: {within_10m_count}/{len(successful_tests)} ({within_10m_count/len(successful_tests)*100:.1f}%)")

print("\n" + "=" * 80)

if len(successful_tests) == len(test_cases):
    print("🎉 ALL TESTS PASSED!")
    print("✅ Borewell Depth Prediction API is working correctly!")
elif len(successful_tests) > 0:
    print("⚠️ PARTIAL SUCCESS")
    print(f"✅ {len(successful_tests)} tests passed, {len(failed_tests)} failed")
else:
    print("❌ ALL TESTS FAILED")
    if error_tests and 'CONNECTION_ERROR' in [e.get('status') for e in error_tests]:
        print("\n💡 ACTION REQUIRED:")
        print("   1. Start Flask server: python app.py")
        print("   2. Wait for server to start (look for 'Running on http://127.0.0.1:5000')")
        print("   3. Run this test again: python test_borewell_api.py")

print("=" * 80)

# Save results to file
with open('test_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print("\n💾 Test results saved to: test_results.json")
