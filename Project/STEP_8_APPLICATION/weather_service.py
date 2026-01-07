"""
Weather Data Service for Nashik Region
Provides realistic weather data based on location and season
Simulates IMD (India Meteorological Department) and WRIS data
"""

import random
from datetime import datetime
import math

class NashikWeatherService:
    """Provides weather data for Nashik region based on season and location"""
    
    def __init__(self):
        # Nashik climate patterns (based on actual IMD historical data)
        self.monsoon_months = [6, 7, 8, 9]  # June to September
        self.winter_months = [11, 12, 1, 2]  # November to February
        self.summer_months = [3, 4, 5]  # March to May
        
        # Average rainfall (mm) by season for Nashik
        self.rainfall_ranges = {
            'monsoon': (80, 250),    # Heavy rainfall
            'winter': (0, 15),       # Minimal rainfall
            'summer': (5, 35),       # Light pre-monsoon showers
            'post_monsoon': (10, 50) # October
        }
        
        # Temperature ranges (°C) by season
        self.temp_ranges = {
            'monsoon': (22, 30),
            'winter': (12, 26),
            'summer': (28, 42),
            'post_monsoon': (20, 32)
        }
        
        # Groundwater levels by season (meters below ground level)
        self.groundwater_ranges = {
            'monsoon': (25, 55),      # Higher water table during monsoon
            'winter': (30, 60),       # Moderate decline
            'summer': (35, 70),       # Lower water table in summer
            'post_monsoon': (28, 58)  # Post-monsoon recharge
        }
        
        # River water levels (meters)
        self.river_level_ranges = {
            'monsoon': (4, 12),
            'winter': (1, 4),
            'summer': (0.5, 2.5),
            'post_monsoon': (2, 6)
        }
    
    def get_season(self, month):
        """Determine season based on month"""
        if month in self.monsoon_months:
            return 'monsoon'
        elif month in self.winter_months:
            return 'winter'
        elif month in self.summer_months:
            return 'summer'
        else:
            return 'post_monsoon'
    
    def get_weather_data(self, lat, lon, date_time=None):
        """
        Get weather data for a specific location and time
        
        Args:
            lat: Latitude
            lon: Longitude
            date_time: datetime object (default: current time)
        
        Returns:
            dict with rainfall, temperature, river_level, groundwater_level
        """
        if date_time is None:
            date_time = datetime.now()
        
        month = date_time.month
        season = self.get_season(month)
        
        # Add location-based variation (higher elevations = cooler, more rain)
        # Nashik city center: ~20.0, 73.79
        elevation_factor = (lat - 20.0) * 0.1  # Simple elevation proxy
        
        # Get base ranges for season
        rainfall_range = self.rainfall_ranges[season]
        temp_range = self.temp_ranges[season]
        gw_range = self.groundwater_ranges[season]
        river_range = self.river_level_ranges[season]
        
        # Add some randomness within realistic bounds
        rainfall = random.uniform(rainfall_range[0], rainfall_range[1])
        temperature = random.uniform(temp_range[0], temp_range[1]) + elevation_factor
        groundwater_level = random.uniform(gw_range[0], gw_range[1])
        river_level = random.uniform(river_range[0], river_range[1])
        
        # Lag values (previous month's data with slight variation)
        rainfall_lag1 = rainfall * random.uniform(0.7, 1.3)
        river_lag1 = river_level * random.uniform(0.8, 1.2)
        
        # Round to realistic precision
        return {
            'rainfall': round(rainfall, 2),
            'temperature': round(temperature, 1),
            'river_water_level': round(river_level, 2),
            'groundwater_level': round(groundwater_level, 2),
            'rainfall_lag1': round(rainfall_lag1, 2),
            'river_lag1': round(river_lag1, 2),
            'season': season,
            'month_name': date_time.strftime('%B'),
            'data_source': 'Simulated IMD/WRIS (Nashik Climate Pattern)',
            'timestamp': date_time.strftime('%Y-%m-%d %H:%M:%S')
        }
    
    def get_monthly_trend(self, lat, lon, year=None):
        """Get weather trends for all 12 months"""
        if year is None:
            year = datetime.now().year
        
        trends = []
        for month in range(1, 13):
            dt = datetime(year, month, 15)  # Mid-month
            data = self.get_weather_data(lat, lon, dt)
            data['month'] = month
            trends.append(data)
        
        return trends
    
    def get_nearest_station_info(self, lat, lon):
        """Get information about nearest weather stations"""
        # Simulated stations in Nashik district
        stations = [
            {
                'name': 'Nashik IMD Observatory',
                'lat': 19.9975,
                'lon': 73.7898,
                'type': 'IMD Weather Station',
                'distance_km': self._calculate_distance(lat, lon, 19.9975, 73.7898)
            },
            {
                'name': 'Malegaon WRIS Station',
                'lat': 20.5537,
                'lon': 74.5288,
                'type': 'WRIS Groundwater Monitoring',
                'distance_km': self._calculate_distance(lat, lon, 20.5537, 74.5288)
            },
            {
                'name': 'Trimbak Rain Gauge',
                'lat': 19.9328,
                'lon': 73.5292,
                'type': 'Rainfall Monitoring',
                'distance_km': self._calculate_distance(lat, lon, 19.9328, 73.5292)
            },
            {
                'name': 'Sinnar DWLR Station',
                'lat': 19.8540,
                'lon': 74.0005,
                'type': 'Digital Water Level Recorder',
                'distance_km': self._calculate_distance(lat, lon, 19.8540, 74.0005)
            }
        ]
        
        # Sort by distance
        stations.sort(key=lambda x: x['distance_km'])
        return stations[:3]  # Return 3 nearest stations
    
    def _calculate_distance(self, lat1, lon1, lat2, lon2):
        """Calculate distance between two points using Haversine formula"""
        R = 6371  # Earth radius in km
        
        lat1_rad = math.radians(lat1)
        lat2_rad = math.radians(lat2)
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        
        a = math.sin(dlat/2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon/2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        
        return round(R * c, 2)

# Global instance
weather_service = NashikWeatherService()
