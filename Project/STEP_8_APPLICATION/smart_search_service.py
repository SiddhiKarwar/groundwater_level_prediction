# AI-Powered Smart Search Service
# Uses Google Gemini API for intelligent location suggestions

import google.generativeai as genai
import json
from functools import lru_cache
import time

class SmartSearchService:
    def __init__(self, api_key):
        """Initialize Gemini AI for smart search"""
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-pro')
        self.cache = {}
        self.cache_timeout = 3600  # 1 hour
        
    def get_smart_suggestions(self, query, locations_database):
        """
        Get AI-powered location suggestions based on user query
        
        Args:
            query: User's search input
            locations_database: List of available locations
            
        Returns:
            List of intelligent suggestions with relevance scores
        """
        if len(query) < 2:
            return self._get_popular_locations(locations_database)
        
        # Check cache first
        cache_key = query.lower()
        if cache_key in self.cache:
            cached_data, timestamp = self.cache[cache_key]
            if time.time() - timestamp < self.cache_timeout:
                return cached_data
        
        try:
            # Prepare AI prompt for intelligent matching
            prompt = self._create_search_prompt(query, locations_database)
            
            # Get AI response
            response = self.model.generate_content(prompt)
            suggestions = self._parse_ai_response(response.text, locations_database)
            
            # Cache the results
            self.cache[cache_key] = (suggestions, time.time())
            
            return suggestions
            
        except Exception as e:
            print(f"AI Search Error: {e}")
            # Fallback to basic search
            return self._fallback_search(query, locations_database)
    
    def _create_search_prompt(self, query, locations):
        """Create intelligent prompt for Gemini"""
        location_names = [loc['name'] for loc in locations[:30]]  # Limit for token efficiency
        
        prompt = f"""You are a smart location search assistant for Nashik District, Maharashtra.

User typed: "{query}"

Available locations: {', '.join(location_names)}

Task: Return the top 5 most relevant location matches as a JSON array. Consider:
1. Phonetic similarity (e.g., "nsk" → "Nashik")
2. Abbreviations (e.g., "kkw" → "K.K.Wagh")
3. Common misspellings
4. Partial matches
5. Context-aware suggestions

Return ONLY a JSON array of location names in order of relevance:
["location1", "location2", "location3", "location4", "location5"]

Be precise and return only locations that exist in the list."""

        return prompt
    
    def _parse_ai_response(self, response_text, locations_database):
        """Parse AI response and match with actual locations"""
        try:
            # Extract JSON from response
            start_idx = response_text.find('[')
            end_idx = response_text.rfind(']') + 1
            
            if start_idx != -1 and end_idx > start_idx:
                json_str = response_text[start_idx:end_idx]
                suggested_names = json.loads(json_str)
                
                # Match suggestions with actual location objects
                suggestions = []
                for name in suggested_names:
                    for loc in locations_database:
                        if loc['name'].lower() == name.lower():
                            suggestions.append(loc)
                            break
                
                return suggestions[:10]
            
        except json.JSONDecodeError:
            pass
        
        # Return empty list if parsing fails (query not available in this scope)
        return []
    
    def _fallback_search(self, query, locations):
        """Fallback search using basic string matching"""
        query_lower = query.lower()
        results = []
        
        for loc in locations:
            score = 0
            name_lower = loc['name'].lower()
            
            # Exact match
            if name_lower == query_lower:
                score = 100
            # Starts with query
            elif name_lower.startswith(query_lower):
                score = 80
            # Contains query
            elif query_lower in name_lower:
                score = 60
            # Keyword match
            elif any(query_lower in kw for kw in loc.get('keywords', [])):
                score = 50
            # First letters match (abbreviation)
            elif self._check_abbreviation(query_lower, name_lower):
                score = 70
            
            if score > 0:
                results.append({**loc, 'score': score})
        
        # Sort by score and priority
        results.sort(key=lambda x: (-x['score'], x.get('priority', 5)))
        return results[:10]
    
    def _check_abbreviation(self, query, name):
        """Check if query matches first letters of words in name"""
        if len(query) < 2:
            return False
        
        words = name.split()
        first_letters = ''.join([w[0] for w in words if w])
        return first_letters.startswith(query)
    
    def _get_popular_locations(self, locations):
        """Get popular and trending locations for empty query"""
        popular = [loc for loc in locations if loc.get('popular', False)]
        trending = [loc for loc in locations if loc.get('trending', False)]
        
        combined = popular + [t for t in trending if t not in popular]
        return combined[:10]

# Initialize service (will be used in app.py)
smart_search = None

def init_smart_search(api_key):
    """Initialize the smart search service"""
    global smart_search
    smart_search = SmartSearchService(api_key)
    return smart_search
