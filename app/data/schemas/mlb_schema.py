def validate_game_schema(game_data):
    """Validate and ensure proper game data structure"""
    required_fields = {
        '@id': str,
        'hometeam': dict,
        'awayteam': dict,
        'starting_pitchers': dict
    }
    
    for field, field_type in required_fields.items():
        if field not in game_data or not isinstance(game_data[field], field_type):
            raise ValueError(f"Invalid game data: missing or invalid {field}")
    
    return game_data

