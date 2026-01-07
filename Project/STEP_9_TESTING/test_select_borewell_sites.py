"""
Demo/test script for select_borewell_sites
"""

import os
import sys

# Ensure project root is on sys.path so imports like
# `from STEP_8_APPLICATION.select_borewell_sites import ...` work
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from STEP_8_APPLICATION.select_borewell_sites import find_top_sites_for_region


def main():
    # region covering Nashik central area
    bbox = (19.95, 73.75, 20.05, 73.82)
    top5 = find_top_sites_for_region(bbox, n=5, grid_spacing_km=0.8)
    print("Top 5 candidate sites for selected region:")
    for i, s in enumerate(top5, 1):
        print(f"{i}. lat={s['lat']:.5f}, lon={s['lon']:.5f}, score={s['score']:.3f}, nearest_m={s['nearest_dist_m']:.1f}, avg_depth={s['avg_depth_m']:.1f}, success_rate={s['success_rate']:.2f}, existing_within_500m={s['existing_count_within_penalty_m']}")


if __name__ == '__main__':
    main()
