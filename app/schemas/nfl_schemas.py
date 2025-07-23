def validate_game_schema(game_data: dict) -> dict:
    """Validate and ensure proper NFL game data structure"""
    required_fields = {
        '@id': str,
        '@date': str,
        '@time': str,
        '@venue': str,
        '@status': str,
        'hometeam': dict,
        'awayteam': dict,
    }
    
    for field, field_type in required_fields.items():
        if field not in game_data or not isinstance(game_data[field], field_type):
            raise ValueError(f"Invalid game data: missing or invalid '{field}'")
    
    # Validate hometeam & awayteam subfields
    for team_key in ['hometeam', 'awayteam']:
        team_data = game_data[team_key]
        if '@name' not in team_data or not isinstance(team_data['@name'], str):
            raise ValueError(f"Invalid game data: missing or invalid '{team_key}[@name]'")
        
        # Score may be missing for upcoming games, so check if present and int-convertible
        score = team_data.get('@totalscore', None)
        if score is not None:
            try:
                int(score)
            except Exception:
                raise ValueError(f"Invalid game data: '{team_key}[@totalscore]' is not an int or int-string")
    
    # Odds are optional, but if present, validate numeric values
    odds = game_data.get('odds', {})
    for key in ['@home', '@away']:
        if key in odds and odds[key] is not None:
            try:
                float(odds[key])
            except Exception:
                raise ValueError(f"Invalid game data: odds[{key}] is not a float")
    
    return game_data


from pydantic import BaseModel, Field
from typing import Optional


class NFLTeams(BaseModel):
    hometeam: str = Field(default="Arizona Cardinals", description="Home team name")
    awayteam: str = Field(default="Atlanta Falcons", description="Away team name")