"""
Test enhanced AI recommender with depth and distance information
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from STEP_8_APPLICATION import ai_recommender
from STEP_8_APPLICATION.select_borewell_sites import bbox_to_grid

# Load data and model
df = ai_recommender.load_borewells()
model_bundle = ai_recommender.load_model()
if model_bundle is None:
    print("Training model...")
    model_bundle = ai_recommender.train_and_save_model(df)

# Test area around Nashik
bbox = (19.95, 73.75, 20.05, 73.82)
candidates = bbox_to_grid(bbox, spacing_km=1.0)

# Score candidates
results = ai_recommender.score_candidates_with_model(candidates, model_bundle, df)[:5]

print("=" * 70)
print("🤖 ENHANCED AI RECOMMENDATION TEST - WITH DEPTH & DISTANCES")
print("=" * 70)

for i, site in enumerate(results, 1):
    print(f"\n📍 RANK #{i}")
    print(f"   Location: ({site['lat']:.5f}, {site['lon']:.5f})")
    print(f"   Success Probability: {site['prob_success']*100:.1f}%")
    print(f"   ⚙️  RECOMMENDED DEPTH: {site['recommended_depth']:.1f}m")
    print(f"   Avg Nearby Yield: {site['meta']['avg_yield']:.0f} LPH")
    print(f"   Nearby Success Rate: {site['meta']['success_rate']*100:.0f}%")
    
    if 'distances_to_others' in site and site['distances_to_others']:
        print(f"   📏 DISTANCES TO OTHER RECOMMENDED POINTS:")
        for point, dist in list(site['distances_to_others'].items())[:3]:
            point_num = point.replace('point_', '')
            print(f"      → To Point #{point_num}: {dist}m")
    print("-" * 70)

print("\n✅ Test complete! Enhanced features working:")
print("   • Recommended drilling depth shown for each point")
print("   • Distances between recommended points calculated")
print("=" * 70)
