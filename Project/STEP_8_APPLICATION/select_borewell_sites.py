"""
Utilities to recommend candidate borewell sites inside a user-selected region.

Functions:
 - find_top_sites_for_region(bbox, n=5, grid_spacing_km=1.0, ...)

The algorithm uses the existing `cgwb_borewells_nashik.csv` dataset (STEP_7_SUPPORTING_DATA)
and scores candidate points based on distance from existing borewells, nearby depths,
success rate, and penalizes locations that already have many borewells nearby.

This is a lightweight, explainable scoring function to help planning. It does not
guarantee drilling success — only helps prioritize locations by historical data.
"""

from typing import Tuple, List, Dict
import math
import os
import pandas as pd

DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'STEP_7_SUPPORTING_DATA', 'cgwb_borewells_nashik.csv')


def haversine_meters(lat1, lon1, lat2, lon2):
    # Returns distance in meters between two lat/lon points
    R = 6371000.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2.0)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2.0)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    return R * c


def load_borewells(path: str = None) -> pd.DataFrame:
    p = path or DATA_PATH
    df = pd.read_csv(p)
    # Normalize column names
    df.columns = [c.strip() for c in df.columns]
    return df


def bbox_to_grid(bbox: Tuple[float, float, float, float], spacing_km: float) -> List[Tuple[float, float]]:
    # bbox = (min_lat, min_lon, max_lat, max_lon)
    min_lat, min_lon, max_lat, max_lon = bbox
    # approximate degree deltas
    lat_deg_per_km = 1.0 / 111.0
    mean_lat = (min_lat + max_lat) / 2.0
    lon_deg_per_km = 1.0 / (111.320 * math.cos(math.radians(mean_lat)))

    lat_step = spacing_km * lat_deg_per_km
    lon_step = spacing_km * lon_deg_per_km

    lats = []
    lat = min_lat
    while lat <= max_lat + 1e-12:
        lats.append(lat)
        lat += lat_step

    lons = []
    lon = min_lon
    while lon <= max_lon + 1e-12:
        lons.append(lon)
        lon += lon_step

    points = [(la, lo) for la in lats for lo in lons]
    return points


def find_top_sites_for_region(
    bbox: Tuple[float, float, float, float],
    n: int = 5,
    grid_spacing_km: float = 1.0,
    neighbor_radius_km: float = 5.0,
    penalty_radius_m: float = 500.0,
    borewells_df: pd.DataFrame = None,
) -> List[Dict]:
    """
    Find top candidate points inside bbox.

    Returns list of dicts with keys: lat, lon, score, nearest_dist_m, avg_depth_m,
    success_rate, existing_count_within_penalty_radius
    """
    if borewells_df is None:
        borewells_df = load_borewells()

    points = bbox_to_grid(bbox, spacing_km=grid_spacing_km)

    results = []
    # precompute arrays for speed
    bw_lats = borewells_df['Latitude'].to_numpy()
    bw_lons = borewells_df['Longitude'].to_numpy()
    bw_depths = borewells_df['Depth_m'].to_numpy()
    bw_status = borewells_df['Status'].to_numpy()  # 'Success' or 'Failure'

    neighbor_radius_m = neighbor_radius_km * 1000.0

    for lat, lon in points:
        # compute distances to all existing borewells
        dists = [haversine_meters(lat, lon, bl, bo) for bl, bo in zip(bw_lats, bw_lons)]
        if len(dists) == 0:
            nearest = float('inf')
            avg_depth = None
            success_rate = 0.5
            existing_small_count = 0
        else:
            nearest = float(min(dists))
            # neighbors within neighbor_radius_m
            neigh_mask = [d <= neighbor_radius_m for d in dists]
            neigh_depths = [d for d, m in zip(bw_depths, neigh_mask) if m]
            neigh_status = [s for s, m in zip(bw_status, neigh_mask) if m]
            if neigh_depths:
                avg_depth = float(sum(neigh_depths) / len(neigh_depths))
            else:
                avg_depth = float(sum(bw_depths) / len(bw_depths))
            if neigh_status:
                success_rate = float(sum(1 for s in neigh_status if str(s).strip().lower() == 'success') / len(neigh_status))
            else:
                # no nearby points -> neutral
                success_rate = 0.5
            existing_small_count = sum(1 for d in dists if d <= penalty_radius_m)

        # scoring
        # distance score normalized by cap (10 km)
        cap = 10000.0
        d_score = min(nearest, cap) / cap

        # depth score normalized (20m - 80m -> 0-1)
        depth_score = (avg_depth - 20.0) / 60.0
        depth_score = max(0.0, min(1.0, depth_score))

        # success_rate is already 0..1

        penalty = min(existing_small_count, 5) / 5.0

        score = 0.5 * d_score + 0.2 * depth_score + 0.2 * success_rate - 0.3 * penalty
        score = max(0.0, min(1.0, score))

        results.append({
            'lat': lat,
            'lon': lon,
            'score': score,
            'nearest_dist_m': nearest,
            'avg_depth_m': avg_depth,
            'success_rate': success_rate,
            'existing_count_within_penalty_m': existing_small_count,
        })

    # sort by score descending
    results.sort(key=lambda r: r['score'], reverse=True)
    return results[:n]


if __name__ == '__main__':
    # quick demo bbox roughly around Nashik city
    bbox = (19.95, 73.75, 20.05, 73.82)
    top = find_top_sites_for_region(bbox, n=5, grid_spacing_km=1.0)
    for i, t in enumerate(top, 1):
        print(f"{i}. ({t['lat']:.5f},{t['lon']:.5f}) score={t['score']:.3f} nearest_m={t['nearest_dist_m']:.1f} avg_depth={t['avg_depth_m']:.1f} success_rate={t['success_rate']:.2f} count_penalty={t['existing_count_within_penalty_m']}")
