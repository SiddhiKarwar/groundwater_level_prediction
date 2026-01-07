"""
AI recommender for borewell planning.

This module trains a simple logistic regression classifier on the existing
CGWB borewell dataset to predict drilling success. It exposes functions to
train/save a model and to score candidate points (probability of success)
along with simple feature contribution explanations (coef * feature).

Notes:
 - The dataset is small (≈30 rows). The trained model is only for demo and
   explainability; treat outputs as indicative, not definitive.
 - You can retrain using more data or more sophisticated models later.
"""

from typing import List, Tuple, Dict
import os
import math
import numpy as np
import pandas as pd

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    import joblib
except Exception:
    LogisticRegression = None
    StandardScaler = None
    joblib = None

DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'STEP_7_SUPPORTING_DATA', 'cgwb_borewells_nashik.csv')
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'STEP_6_TRAINED_MODELS', 'ai_recommender.pkl')


def load_borewells(path: str = None) -> pd.DataFrame:
    p = path or DATA_PATH
    return pd.read_csv(p)


def encode_water_quality(q: str) -> int:
    if not isinstance(q, str):
        return 1
    q = q.strip().lower()
    if q == 'excellent':
        return 3
    if q == 'good':
        return 2
    if q == 'moderate':
        return 1
    if q == 'poor':
        return 0
    return 1


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2.0)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2.0)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    return R * c


def build_training_table(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    # Features: Depth_m, Yield_LPH, Quality_num, Age(years)
    cur_year = pd.Timestamp.now().year
    feats = []
    labels = []
    for _, r in df.iterrows():
        try:
            depth = float(r['Depth_m'])
            yield_lph = float(r.get('Yield_LPH', 0) or 0)
            quality = encode_water_quality(r.get('Water_Quality', ''))
            year = int(r.get('Construction_Year') or cur_year)
            age = cur_year - year
            features = [depth, yield_lph, quality, age]
            feats.append(features)
            labels.append(1 if str(r.get('Status','')).strip().lower()=='success' else 0)
        except Exception:
            continue
    feature_names = ['depth_m','yield_lph','quality','age_years']
    return np.array(feats), np.array(labels), feature_names


def train_and_save_model(df: pd.DataFrame = None, overwrite: bool = True):
    if LogisticRegression is None:
        raise RuntimeError('scikit-learn not available in this environment')
    if df is None:
        df = load_borewells()

    X, y, feature_names = build_training_table(df)
    if len(X) < 5:
        raise RuntimeError('Not enough training samples')

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    clf = LogisticRegression(max_iter=500)
    clf.fit(Xs, y)

    model_bundle = {'model': clf, 'scaler': scaler, 'features': feature_names}
    if overwrite and joblib is not None:
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        joblib.dump(model_bundle, MODEL_PATH)
    return model_bundle


def load_model(path: str = None):
    p = path or MODEL_PATH
    if joblib is None:
        return None
    if not os.path.exists(p):
        return None
    return joblib.load(p)


def compute_candidate_features(lat: float, lon: float, borewells_df: pd.DataFrame, radius_km: float = 5.0) -> Tuple[np.ndarray, Dict]:
    # Compute aggregated features around candidate point
    dists = []
    depths = []
    yields = []
    statuses = []
    for _, r in borewells_df.iterrows():
        try:
            bl = float(r['Latitude']); bo = float(r['Longitude'])
            d = haversine_km(lat, lon, bl, bo)
            if d <= radius_km:
                dists.append(d)
                depths.append(float(r['Depth_m']))
                yields.append(float(r.get('Yield_LPH') or 0))
                statuses.append(1 if str(r.get('Status','')).strip().lower()=='success' else 0)
        except Exception:
            continue

    if depths:
        avg_depth = float(np.mean(depths))
        avg_yield = float(np.mean(yields))
        success_rate = float(np.mean(statuses))
        count = len(depths)
    else:
        # fallback to global averages
        avg_depth = float(borewells_df['Depth_m'].mean())
        avg_yield = float(borewells_df['Yield_LPH'].mean()) if 'Yield_LPH' in borewells_df.columns else 0.0
        success_rate = 0.5
        count = 0

    # age: use mean construction year
    cur_year = pd.Timestamp.now().year
    if 'Construction_Year' in borewells_df.columns:
        mean_year = int(borewells_df['Construction_Year'].dropna().astype(int).mean())
        age = cur_year - mean_year
    else:
        age = 5

    features = np.array([avg_depth, avg_yield,  # depth, yield
                         2.0,  # placeholder for quality (we don't compute quality here)
                         age])

    meta = {'avg_depth': avg_depth, 'avg_yield': avg_yield, 'success_rate': success_rate, 'count': count}
    return features, meta


def score_candidates_with_model(
        candidates: List[Tuple[float, float]],
        model_bundle: Dict,
        borewells_df: pd.DataFrame,
        min_distance_m: float = 200.0,
        density_radius_m: float = 500.0,
        density_radius2_m: float = 1000.0,
) -> List[Dict]:
    """Score candidates and return enriched recommendation objects.

    Adds proximity and density awareness so points too close to existing borewells are penalized.

    Returns list of dicts with keys:
        lat, lon, prob_success, score, recommended_depth,
        nearest_existing_m, counts_within_{500m,1km}, meta,
        contributions (feature->value), distances_to_others, why_text, factors
    """
    clf = model_bundle['model']
    scaler = model_bundle['scaler']
    feature_names = model_bundle['features']

    # Precache arrays for distance calculations
    bw_lats = borewells_df['Latitude'].to_numpy() if not borewells_df.empty else []
    bw_lons = borewells_df['Longitude'].to_numpy() if not borewells_df.empty else []

    scored = []
    for lat, lon in candidates:
        x, meta = compute_candidate_features(lat, lon, borewells_df)
        xs = scaler.transform([x])
        prob = float(clf.predict_proba(xs)[0,1]) if hasattr(clf, 'predict_proba') else float(clf.decision_function(xs))
        # approximate contributions using coef * (scaled value)
        coefs = clf.coef_[0]
        contribs = {fn: float(coefs[i] * xs[0,i]) for i,fn in enumerate(feature_names)}
        # Distance to nearest existing borewell and density counts
        if len(bw_lats) > 0:
            dists_m = [haversine_km(lat, lon, bl, bo) * 1000.0 for bl, bo in zip(bw_lats, bw_lons)]
            nearest_m = float(min(dists_m)) if dists_m else float('inf')
            count_500m = int(sum(1 for d in dists_m if d <= density_radius_m))
            count_1km = int(sum(1 for d in dists_m if d <= density_radius2_m))
        else:
            nearest_m = float('inf')
            count_500m = 0
            count_1km = 0

        # Compute a composite score combining model probability and proximity penalty
        # Strong penalty if too close to an existing borewell (< min_distance_m)
        proximity_penalty = 0.0
        if nearest_m < min_distance_m:
            # full penalty if too close
            proximity_penalty = 0.5
        else:
            # mild penalty scaled within first 1km
            proximity_penalty = max(0.0, (density_radius2_m - min(nearest_m, density_radius2_m)) / density_radius2_m) * 0.2

        density_penalty = min(count_500m, 5) * 0.05  # up to 0.25
        score = max(0.0, min(1.0, prob - proximity_penalty - density_penalty))

        # Build a short why-text and factor list
        # Choose top 2 magnitude contributions
        top_feats = sorted(contribs.items(), key=lambda kv: abs(kv[1]), reverse=True)[:2]
        factors = []
        for name, val in top_feats:
            direction = 'positive' if val >= 0 else 'negative'
            factors.append({'feature': name, 'impact': round(val, 3), 'direction': direction})
        proximity_note = 'Safe distance from existing borewells' if nearest_m >= min_distance_m else f'Too close to existing borewells (~{int(nearest_m)} m)'
        why_text = (
            f"High predicted success ({prob*100:.1f}%). "
            f"{proximity_note}. "
            f"Nearby avg depth ~{meta.get('avg_depth', 0):.1f} m; success rate in vicinity ~{meta.get('success_rate', 0)*100:.0f}%"
        )

        # Recommended depth based on nearby successful borewells
        recommended_depth = meta['avg_depth']
        scored.append({
            'lat': lat,
            'lon': lon,
            'prob_success': prob,
            'score': score,
            'meta': meta,
            'contributions': contribs,
            'recommended_depth': recommended_depth,
            'nearest_existing_m': nearest_m,
            'counts_within_500m': count_500m,
            'counts_within_1km': count_1km,
            'why_text': why_text,
            'factors': factors
        })
    
    # Sort by composite score first, fallback to probability
    scored.sort(key=lambda s: (s['score'], s['prob_success']), reverse=True)
    
    # Calculate distances between top candidates (for top 10 to avoid huge computation)
    top_candidates = scored[:min(10, len(scored))]
    for i, candidate in enumerate(top_candidates):
        distances = {}
        for j, other in enumerate(top_candidates):
            if i != j:
                dist_km = haversine_km(candidate['lat'], candidate['lon'], other['lat'], other['lon'])
                distances[f'point_{j+1}'] = round(dist_km * 1000, 1)  # Convert to meters
        candidate['distances_to_others'] = distances
    
    return scored


if __name__ == '__main__':
    # Demo flow: train (if missing), then score top grid in Nashik bbox
    df = load_borewells()
    model_bundle = load_model()
    if model_bundle is None:
        print('No saved model found; training a simple logistic model...')
        model_bundle = train_and_save_model(df)
        print('Model trained and saved.')

    # demo: reuse select_borewell_sites to produce candidate grid
    from STEP_8_APPLICATION.select_borewell_sites import bbox_to_grid
    bbox = (19.95, 73.75, 20.05, 73.82)
    candidates = bbox_to_grid(bbox, spacing_km=1.0)
    top = score_candidates_with_model(candidates, model_bundle, df)[:5]
    for t in top:
        print(f"lat={t['lat']:.5f} lon={t['lon']:.5f} prob={t['prob_success']:.3f} meta={t['meta']} contribs={t['contributions']}")
