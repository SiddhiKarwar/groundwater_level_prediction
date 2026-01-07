"""
Recommendation Service for Groundwater Borewell Success Prediction
Provides intelligent recommendations based on multiple factors:
- Nearby borewell success rates
- Predicted groundwater level
- Season analysis
- Cost estimation
- Risk assessment
"""

import pandas as pd
import numpy as np
from datetime import datetime
from math import radians, sin, cos, sqrt, atan2

class BoreholeRecommendationService:
    """Provides comprehensive recommendations for borewell drilling"""
    
    def __init__(self, borewells_df=None):
        """
        Initialize with CGWB borewell database
        
        Args:
            borewells_df: DataFrame with existing borewell data
        """
        self.borewells_df = borewells_df if borewells_df is not None else pd.DataFrame()
        
        # Cost factors (INR per meter)
        self.cost_per_meter = {
            'drilling': 800,      # Base drilling cost per meter
            'casing': 500,        # Casing cost per meter
            'development': 300,   # Well development per meter
            'pump': 25000,        # Submersible pump (fixed cost)
            'electrical': 15000   # Electrical work (fixed cost)
        }
        
        # Region-specific cost multipliers
        self.region_multipliers = {
            'urban': 1.3,         # Nashik city - higher costs
            'semi_urban': 1.1,    # Towns like Malegaon, Sinnar
            'rural': 1.0          # Villages - base cost
        }
        
        # Season success factors (0-1, higher is better)
        self.season_factors = {
            'monsoon': 0.95,      # Best time - can see water levels
            'post_monsoon': 0.90, # Good - water stabilized
            'winter': 0.75,       # Okay - stable but low recharge
            'summer': 0.60        # Worst - depleted levels
        }
        
        # Depth ranges for different groundwater levels
        self.depth_recommendations = {
            'very_high': (20, 35),   # >60m groundwater
            'high': (30, 50),        # 50-60m groundwater
            'medium': (40, 60),      # 40-50m groundwater
            'low': (50, 75),         # 30-40m groundwater
            'very_low': (60, 90)     # <30m groundwater
        }
    
    def haversine_distance(self, lat1, lon1, lat2, lon2):
        """Calculate distance between two points using Haversine formula"""
        R = 6371  # Earth radius in km
        
        lat1_rad = radians(lat1)
        lat2_rad = radians(lat2)
        dlat = radians(lat2 - lat1)
        dlon = radians(lon2 - lon1)
        
        a = sin(dlat/2)**2 + cos(lat1_rad) * cos(lat2_rad) * sin(dlon/2)**2
        c = 2 * atan2(sqrt(a), sqrt(1-a))
        
        return R * c
    
    def calculate_success_probability(self, lat, lon, predicted_level, radius_km=5):
        """
        Calculate success probability based on nearby borewells
        
        Args:
            lat: Latitude of target location
            lon: Longitude of target location
            predicted_level: Predicted groundwater level (meters)
            radius_km: Search radius in kilometers
        
        Returns:
            dict with probability and contributing factors
        """
        if self.borewells_df.empty:
            # No borewell data - use predicted level only
            base_probability = self._level_to_probability(predicted_level)
            return {
                'probability': base_probability,
                'confidence': 'low',
                'nearby_borewells': 0,
                'successful_nearby': 0,
                'factors': {
                    'predicted_level': base_probability,
                    'historical_success': None,
                    'depth_adequacy': None
                }
            }
        
        # Find nearby borewells
        nearby = []
        for idx, row in self.borewells_df.iterrows():
            try:
                bw_lat = float(row['Latitude'])
                bw_lon = float(row['Longitude'])
                distance = self.haversine_distance(lat, lon, bw_lat, bw_lon)
                
                if distance <= radius_km:
                    nearby.append({
                        'distance': distance,
                        'status': row['Status'],
                        'depth': row['Depth_m'],
                        'yield': row['Yield_LPH']
                    })
            except:
                continue
        
        # Calculate probability from multiple factors
        factors = {}
        
        # Factor 1: Predicted groundwater level (0.4 weight)
        level_prob = self._level_to_probability(predicted_level)
        factors['predicted_level'] = level_prob
        
        # Factor 2: Historical success rate (0.4 weight)
        if nearby:
            success_count = sum(1 for b in nearby if b['status'].lower() == 'successful')
            historical_prob = success_count / len(nearby)
            factors['historical_success'] = historical_prob
            
            # Weighted by distance (closer borewells matter more)
            weighted_success = 0
            total_weight = 0
            for b in nearby:
                weight = 1 / (b['distance'] + 0.1)  # +0.1 to avoid divide by zero
                if b['status'].lower() == 'successful':
                    weighted_success += weight
                total_weight += weight
            
            if total_weight > 0:
                factors['historical_success'] = weighted_success / total_weight
        else:
            factors['historical_success'] = 0.5  # Neutral if no data
        
        # Factor 3: Depth adequacy (0.2 weight)
        if nearby:
            avg_depth = np.mean([b['depth'] for b in nearby if b['status'].lower() == 'successful'])
            # Check if typical drilling depths can reach the water
            if predicted_level > avg_depth:
                factors['depth_adequacy'] = 0.9  # Water is shallow, easy to reach
            elif predicted_level > avg_depth * 1.5:
                factors['depth_adequacy'] = 0.6  # Need deeper drilling
            else:
                factors['depth_adequacy'] = 0.3  # Very deep, may need specialized drilling
        else:
            factors['depth_adequacy'] = 0.7  # Assume adequate if no data
        
        # Calculate weighted probability
        total_prob = (
            factors['predicted_level'] * 0.4 +
            factors['historical_success'] * 0.4 +
            factors['depth_adequacy'] * 0.2
        )
        
        # Determine confidence based on data availability
        if len(nearby) >= 5:
            confidence = 'high'
        elif len(nearby) >= 2:
            confidence = 'medium'
        else:
            confidence = 'low'
        
        return {
            'probability': round(total_prob * 100, 1),  # Convert to percentage
            'confidence': confidence,
            'nearby_borewells': len(nearby),
            'successful_nearby': sum(1 for b in nearby if b['status'].lower() == 'successful'),
            'factors': factors
        }
    
    def _level_to_probability(self, level):
        """Convert predicted groundwater level to success probability"""
        if level >= 60:
            return 0.95  # Very high groundwater - excellent
        elif level >= 50:
            return 0.85  # High groundwater - very good
        elif level >= 40:
            return 0.70  # Medium groundwater - good
        elif level >= 30:
            return 0.50  # Low groundwater - moderate
        else:
            return 0.30  # Very low groundwater - risky
    
    def recommend_drilling_depth(self, predicted_level, lat=None, lon=None, radius_km=5):
        """
        Recommend drilling depth based on predicted level and nearby borewells
        
        Args:
            predicted_level: Predicted groundwater level
            lat: Latitude (optional, for nearby borewell analysis)
            lon: Longitude (optional, for nearby borewell analysis)
            radius_km: Search radius
        
        Returns:
            dict with recommended depths and reasoning
        """
        # Base recommendation from predicted level
        if predicted_level >= 60:
            base_range = self.depth_recommendations['very_high']
            category = 'Very High'
        elif predicted_level >= 50:
            base_range = self.depth_recommendations['high']
            category = 'High'
        elif predicted_level >= 40:
            base_range = self.depth_recommendations['medium']
            category = 'Medium'
        elif predicted_level >= 30:
            base_range = self.depth_recommendations['low']
            category = 'Low'
        else:
            base_range = self.depth_recommendations['very_low']
            category = 'Very Low'
        
        # Adjust based on nearby borewells if available
        if lat and lon and not self.borewells_df.empty:
            nearby_depths = []
            for idx, row in self.borewells_df.iterrows():
                try:
                    bw_lat = float(row['Latitude'])
                    bw_lon = float(row['Longitude'])
                    distance = self.haversine_distance(lat, lon, bw_lat, bw_lon)
                    
                    if distance <= radius_km and row['Status'].lower() == 'successful':
                        nearby_depths.append(row['Depth_m'])
                except:
                    continue
            
            if nearby_depths:
                avg_nearby = np.mean(nearby_depths)
                max_nearby = np.max(nearby_depths)
                
                # Adjust recommendation based on nearby successful depths
                recommended_min = max(base_range[0], int(avg_nearby * 0.8))
                recommended_max = min(base_range[1], int(max_nearby * 1.2))
                
                reasoning = f"Based on {len(nearby_depths)} successful borewells nearby (avg depth: {avg_nearby:.1f}m)"
            else:
                recommended_min, recommended_max = base_range
                reasoning = f"Based on predicted groundwater level ({category})"
        else:
            recommended_min, recommended_max = base_range
            reasoning = f"Based on predicted groundwater level ({category})"
        
        # Safety buffer (add 5m for dry season fluctuations)
        recommended_max += 5
        
        return {
            'minimum_depth': recommended_min,
            'recommended_depth': (recommended_min + recommended_max) // 2,
            'maximum_depth': recommended_max,
            'category': category,
            'reasoning': reasoning,
            'safety_buffer': 5
        }
    
    def analyze_best_season(self, lat, lon):
        """
        Analyze best season for drilling based on success factors
        
        Returns:
            dict with season rankings and recommendations
        """
        current_month = datetime.now().month
        
        # Determine current season
        if current_month in [6, 7, 8, 9]:
            current_season = 'monsoon'
        elif current_month in [11, 12, 1, 2]:
            current_season = 'winter'
        elif current_month in [3, 4, 5]:
            current_season = 'summer'
        else:
            current_season = 'post_monsoon'
        
        # Rank seasons
        season_rankings = []
        for season, factor in self.season_factors.items():
            season_rankings.append({
                'season': season.replace('_', '-').title(),
                'success_factor': factor,
                'score': factor * 100,
                'is_current': season == current_season
            })
        
        # Sort by score
        season_rankings.sort(key=lambda x: x['score'], reverse=True)
        
        # Get best season
        best_season = season_rankings[0]
        
        # Recommendations
        if current_season == 'monsoon' or current_season == 'post_monsoon':
            recommendation = "✅ EXCELLENT TIME - Current season is ideal for drilling!"
            timing = "Proceed immediately"
        elif current_season == 'winter':
            recommendation = "✔️ GOOD TIME - Acceptable season for drilling"
            timing = "Can proceed, but post-monsoon (Oct) is better"
        else:  # summer
            recommendation = "⚠️ WAIT - Summer is not ideal for drilling"
            timing = "Wait for monsoon (June) or post-monsoon (October)"
        
        return {
            'current_season': current_season.replace('_', '-').title(),
            'best_season': best_season['season'],
            'current_score': self.season_factors[current_season] * 100,
            'best_score': best_season['score'],
            'recommendation': recommendation,
            'timing': timing,
            'rankings': season_rankings
        }
    
    def estimate_cost(self, depth, region_type='rural'):
        """
        Estimate total drilling cost
        
        Args:
            depth: Drilling depth in meters
            region_type: 'urban', 'semi_urban', or 'rural'
        
        Returns:
            dict with detailed cost breakdown
        """
        multiplier = self.region_multipliers.get(region_type, 1.0)
        
        # Calculate component costs
        drilling_cost = depth * self.cost_per_meter['drilling'] * multiplier
        casing_cost = depth * self.cost_per_meter['casing'] * multiplier
        development_cost = depth * self.cost_per_meter['development'] * multiplier
        pump_cost = self.cost_per_meter['pump']
        electrical_cost = self.cost_per_meter['electrical']
        
        # Contingency (10%)
        subtotal = drilling_cost + casing_cost + development_cost + pump_cost + electrical_cost
        contingency = subtotal * 0.10
        
        total = subtotal + contingency
        
        return {
            'breakdown': {
                'drilling': round(drilling_cost),
                'casing': round(casing_cost),
                'development': round(development_cost),
                'pump': pump_cost,
                'electrical': electrical_cost,
                'contingency': round(contingency)
            },
            'subtotal': round(subtotal),
            'total': round(total),
            'per_meter': round(total / depth),
            'region': region_type.replace('_', ' ').title(),
            'depth': depth
        }
    
    def assess_risk(self, success_probability, predicted_level, depth_recommendation, cost_estimate):
        """
        Comprehensive risk assessment
        
        Returns:
            dict with risk level and factors
        """
        risk_factors = []
        risk_score = 0  # 0-100, higher is riskier
        
        # Factor 1: Success probability (40% weight)
        if success_probability >= 80:
            risk_score += 10
            risk_factors.append("✅ High success probability (>80%)")
        elif success_probability >= 60:
            risk_score += 25
            risk_factors.append("✔️ Moderate success probability (60-80%)")
        elif success_probability >= 40:
            risk_score += 50
            risk_factors.append("⚠️ Fair success probability (40-60%)")
        else:
            risk_score += 80
            risk_factors.append("❌ Low success probability (<40%)")
        
        # Factor 2: Groundwater level (30% weight)
        if predicted_level >= 50:
            risk_score += 5
            risk_factors.append("✅ Adequate groundwater level (>50m)")
        elif predicted_level >= 40:
            risk_score += 15
            risk_factors.append("✔️ Moderate groundwater level (40-50m)")
        elif predicted_level >= 30:
            risk_score += 35
            risk_factors.append("⚠️ Low groundwater level (30-40m)")
        else:
            risk_score += 60
            risk_factors.append("❌ Very low groundwater level (<30m)")
        
        # Factor 3: Drilling depth (20% weight)
        if depth_recommendation['maximum_depth'] <= 50:
            risk_score += 5
            risk_factors.append("✅ Shallow drilling (<50m)")
        elif depth_recommendation['maximum_depth'] <= 70:
            risk_score += 10
            risk_factors.append("✔️ Moderate depth (50-70m)")
        else:
            risk_score += 25
            risk_factors.append("⚠️ Deep drilling (>70m)")
        
        # Factor 4: Cost (10% weight)
        if cost_estimate['total'] <= 100000:
            risk_score += 2
            risk_factors.append("✅ Low cost (<₹1 Lakh)")
        elif cost_estimate['total'] <= 200000:
            risk_score += 5
            risk_factors.append("✔️ Moderate cost (₹1-2 Lakhs)")
        else:
            risk_score += 10
            risk_factors.append("⚠️ High cost (>₹2 Lakhs)")
        
        # Normalize to 0-100
        risk_score = min(100, risk_score)
        
        # Determine risk level
        if risk_score <= 25:
            risk_level = 'LOW'
            color = '#10B981'  # Green
            recommendation = "✅ RECOMMENDED - Low risk, high success potential"
        elif risk_score <= 50:
            risk_level = 'MEDIUM'
            color = '#F59E0B'  # Orange
            recommendation = "✔️ ACCEPTABLE - Moderate risk, proceed with caution"
        elif risk_score <= 75:
            risk_level = 'HIGH'
            color = '#EF4444'  # Red
            recommendation = "⚠️ RISKY - High risk, consider alternatives"
        else:
            risk_level = 'VERY HIGH'
            color = '#DC2626'  # Dark red
            recommendation = "❌ NOT RECOMMENDED - Very high risk"
        
        return {
            'risk_level': risk_level,
            'risk_score': round(risk_score, 1),
            'color': color,
            'recommendation': recommendation,
            'factors': risk_factors
        }
    
    def generate_comprehensive_report(self, lat, lon, predicted_level, location_name=None):
        """
        Generate comprehensive recommendation report
        
        Returns:
            dict with all recommendations and analysis
        """
        # Calculate all recommendations
        success_prob = self.calculate_success_probability(lat, lon, predicted_level)
        depth_rec = self.recommend_drilling_depth(predicted_level, lat, lon)
        season_analysis = self.analyze_best_season(lat, lon)
        
        # Determine region type based on location
        # (Simple heuristic - can be enhanced with actual region data)
        region_type = 'rural'  # Default
        if location_name:
            if 'nashik' in location_name.lower() or 'city' in location_name.lower():
                region_type = 'urban'
            elif 'malegaon' in location_name.lower() or 'sinnar' in location_name.lower():
                region_type = 'semi_urban'
        
        cost_est = self.estimate_cost(depth_rec['recommended_depth'], region_type)
        risk_assessment = self.assess_risk(
            success_prob['probability'],
            predicted_level,
            depth_rec,
            cost_est
        )
        
        return {
            'location': {
                'latitude': lat,
                'longitude': lon,
                'name': location_name or f"{lat:.4f}, {lon:.4f}"
            },
            'predicted_level': predicted_level,
            'success_probability': success_prob,
            'depth_recommendation': depth_rec,
            'season_analysis': season_analysis,
            'cost_estimate': cost_est,
            'risk_assessment': risk_assessment,
            'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

# Global instance
recommendation_service = BoreholeRecommendationService()
