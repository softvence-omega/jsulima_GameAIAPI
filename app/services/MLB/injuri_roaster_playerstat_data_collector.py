import requests
import pandas as pd 
from tqdm.auto import tqdm
import os
import sys
from pathlib import Path
from app.services.helper import safe_float

from app.config import GOALSERVE_API_KEY, GOALSERVE_BASE_URL


# Add the project root directory to Python path
project_root = str(Path(__file__).parent.parent.parent.parent)
sys.path.append(project_root)

from app.config import DATA_DIR


def _force_list(x):
    """Return x as a list.  If x is None → [], if it's a dict → [dict]."""
    if x is None:
        return []
    return x if isinstance(x, list) else [x]

def extract_teams():
    """Extract team IDs and names from the standings data."""
    # print("\nExtracting team information...")
    team_ids = []
    url = f"{GOALSERVE_BASE_URL}{GOALSERVE_API_KEY}/baseball/mlb_standings?json=1"


    try:
        res = requests.get(url, timeout=20)
        res.raise_for_status()

        data = res.json()
        teams = []

        # Get the category data
        category = data.get("standings", {}).get("category", {})
        
        # Process each league
        for league in _force_list(category.get("league", [])):
            # Process each division
            for division in _force_list(league.get("division", [])):
                # Process each team
                for team in _force_list(division.get("team", [])):
                    team_id = team.get("@id")
                    team_name = team.get("@name")
                    if team_id and team_name:  # Only add if both ID and name exist
                        teams.append((team_id, team_name))
                        # print(f"Found team: {team_name} (ID: {team_id})")

        if not teams:
            # print("Warning: No teams found in the standings data")
            pass 
        else:
            # print(f"\nSuccessfully extracted {len(teams)} teams")
            pass 
            
        return teams

    except Exception as e:
        # print(f"Error extracting teams: {str(e)}")
        return []

def extract_injury_report(team_id):
    """
    Args:
        report_json (dict) – the JSON block you posted
    Returns:
        list[dict] – [{'player_id': ..., 'player_name': ..., 'status': ...}, …]
        and a pandas DataFrame with the same columns
    """
    url = f"{GOALSERVE_BASE_URL}{GOALSERVE_API_KEY}/baseball/{team_id}_injuries?json=1"
    res = requests.get(url)
    res.raise_for_status()

    data = res.json()
    players = [
        {
            "team_id": data['team'].get("id"),
            "team_name": data['team'].get("name"),
            "player_id":   item.get("player_id"),
            "player_name": item.get("player_name"),
            "player_position": "",
            'player_number': "",
            "position": "",
            "player_status": item.get("status"),
        }
        for item in data["team"].get("report", [])
    ]

    return players



def extract_team_roster() -> pd.DataFrame:
    """
    Extract team roster info into a DataFrame with fields:
    team_id, team_name, position_unit, player_id, player_name, player_number, player_position
    """

    teams = extract_teams() 
    players_list = []
    for team_id, _ in teams:
        url = f"{GOALSERVE_BASE_URL}{GOALSERVE_API_KEY}/baseball/{team_id}_rosters?json=1"

        res = requests.get(url)
        res.raise_for_status()

        data = res.json()
        team_id = data['team'].get('id')
        team_name = data['team'].get('name')


        for unit in data['team'].get('position', []):
            position = unit.get('name')
            for player in _force_list(unit.get("player")):
                players_list.append({
                    'team_id': team_id,
                    'team_name': team_name,
                    'position': position,
                    'player_id': player.get('id', ''),
                    'player_name': player.get('name', ''),
                    'player_number': player.get('number', ''),
                    'player_position': player.get('position', ''),
                    'player_status': 'roster'
                })

        players = extract_injury_report(team_id)
        for player in players:
            players_list.append(player)

    return pd.DataFrame(players_list)

# df = extract_team_roster()

# print(df.shape)
# print(df.sample())

# df.to_csv("player_info.csv", index=False)

def clean_value(value):
    """Clean and convert values appropriately."""
    if value == "" or value is None:
        return None
    try:
        # Try to convert to float first
        return float(value)
    except ValueError:
        # If not a number, return as is
        return value

def get_category_fields(category):
    """Return the expected fields for each category."""
    common_fields = ['rank', 'name', 'team', 'gp', 'id']
    
    if category == 'batting':
        return common_fields + [
            'at_bats', 'runs', 'hits', 'doubles', 'triples', 'home_runs',
            'runs_batted_in', 'stolen_bases', 'caught_stealing', 'walks',
            'strikeouts', 'batting_avg', 'on_base_percentage', 'slugging_percentage'
        ]
    elif category == 'fielding':
        return common_fields + [
            'gs', 'innings', 'total_chances', 'putouts', 'assists', 'errors',
            'double_plays', 'fielding_percentage', 'range_factor', 'zone_rating',
            'passed_balls', 'stolen_bases_allowed', 'caught_stealing'
        ]
    elif category == 'pitching':
        return common_fields + [
            'gs', 'qs', 'innings_pitched', 'hits', 'runs', 'earned_runs',
            'walks', 'strikeouts', 'wins', 'losses', 'saves', 'holds',
            'blown_saves', 'walk_and_hits', 'earned_run_avg'
        ]
    return common_fields

def fetch_roster_data(team_id):
    """Fetch and process roster data for all teams."""
    url = f"{GOALSERVE_BASE_URL}{GOALSERVE_API_KEY}/baseball/{team_id}_rosters?json=1"
    res = requests.get(url, timeout=20)
    res.raise_for_status()
    
    data = res.json()
    team_data = data.get('team', {})
    
    # Get team metadata
    team_metadata = {
        'team_id': team_data.get('@id', ''),
        'team_name': team_data.get('@name', ''),
        'team_abbreviation': team_data.get('@abbreviation', ''),
        'season': team_data.get('@season', '')
    }
    
    # Debug print
    # print(f"Team data keys: {team_data.keys()}")
    
    positions = _force_list(team_data.get('position', []))
    # print(f"Found {len(positions)} position groups")
    rosters= []
    for position in positions:
        position_name = position.get('@name', '')
        players = _force_list(position.get('player', []))
        # print(f"Found {len(players)} players in position {position_name}")
        
        for player in players:
            player_data = {
                **team_metadata,
                'position_group': position_name,
                'player_id': player.get('@id', ''),
                'player_name': player.get('@name', ''),
                'player_number': player.get('@number', ''),
                'player_position': player.get('@position', ''),
                'bats': player.get('@bats', ''),
                'throws': player.get('@throws', ''),
                'age': clean_value(player.get('@age', '')),
                'height': player.get('@height', ''),
                'weight': player.get('@weight', ''),
                'salary': player.get('@salary', ''),
                'status': 'active'
            }
            rosters.append(player_data)
    return rosters
            

def fetch_injury_data():
    """Fetch and process injury data for all teams."""
    # print("\nFetching injury data...")
    teams = extract_teams()
    if not teams:
        #print("No teams found to process injuries")
        return pd.DataFrame()
        
    all_injuries = []
    
    for team_id, team_name in tqdm(teams, desc="Processing team injuries"):
        try:
            url = f"{GOALSERVE_BASE_URL}{GOALSERVE_API_KEY}/baseball/{team_id}_injuries?json=1"
            res = requests.get(url, timeout=20)
            res.raise_for_status()
            
            data = res.json()
            team_data = data.get('team', {})
            
            # Debug #print
            #print(f"\nProcessing injuries for team: {team_name}")
            #print(f"Team data keys: {team_data.keys()}")
            
            injuries = _force_list(team_data.get('report', []))
            #print(f"Found {len(injuries)} injuries")
            
            for injury in injuries:
                injury_data = {
                    'team_id': team_data.get('@id', ''),
                    'team_name': team_data.get('@name', ''),
                    'player_id': injury.get('@player_id', ''),
                    'player_name': injury.get('@player_name', ''),
                    'injury_date': injury.get('@date', ''),
                    'status': injury.get('@status', ''),
                    'description': injury.get('@description', '')
                }
                all_injuries.append(injury_data)
                
        except Exception as e:
            #print(f"Error fetching injuries for team {team_name}: {str(e)}")
            continue
    
    df = pd.DataFrame(all_injuries)
    if not df.empty:
        pass 
        ##print(f"\nSuccessfully collected {len(df)} injury records")
        ##print("Sample of injury data:")
        ##print(df[['team_name', 'player_name', 'status', 'description']].head())
    return df

def fetch_standings_data():
    """Fetch and process standings data."""
    #print("\nFetching standings data...")
    standings_data = []
    
    try:
        url = f"{GOALSERVE_BASE_URL}{GOALSERVE_API_KEY}/baseball/mlb_standings?json=1"
        res = requests.get(url, timeout=20)
        res.raise_for_status()
        
        data = res.json()
        category = data.get("standings", {}).get("category", {})
        season = category.get('@season', '')
        
        for league in _force_list(category.get('league', [])):
            league_name = league.get('@name', '')
            for division in _force_list(league.get('division', [])):
                division_name = division.get('@name', '')
                for team in _force_list(division.get('team', [])):
                    wins = clean_value(team.get('@won', ''))
                    losses = clean_value(team.get('@lost', ''))
                    team_data = {
                        'season': season,
                        'league': league_name,
                        'division': division_name,
                        'team_id': team.get('@id', ''),
                        'team_name': team.get('@name', ''),
                        'position': clean_value(team.get('@position', '')),
                        'wins': wins,
                        'losses': losses,
                        'win_pct': safe_float(wins / (wins + losses) if (wins + losses) > 0 else 0.5 ),
                        'games_back': clean_value(team.get('@games_back', '')),
                        'home_record': team.get('@home_record', ''),
                        'away_record': team.get('@away_record', ''),
                        'runs_scored': clean_value(team.get('@runs_scored', '')),
                        'runs_allowed': clean_value(team.get('@runs_allowed', '')),
                        'runs_diff': team.get('@runs_diff', ''),
                        'current_streak': team.get('@current_streak', '')
                    }
                    standings_data.append(team_data)
                    
    except Exception as e:
        pass 
        #print(f"Error fetching standings: {str(e)}")
    return pd.DataFrame(standings_data)

def extract_all_player_stats():
    rows = []
    endpoints = [
        "mlb_player_batting",
        "mlb_player_fielding",
        "mlb_player_pitching"
    ]
    
    base_url = f"{GOALSERVE_BASE_URL}{GOALSERVE_API_KEY}/baseball/{endpoint}?json=1"
    
    for endpoint in tqdm(endpoints, desc="Processing endpoints"):
        url = base_url.format(endpoint=endpoint)
        try:
            res = requests.get(url, timeout=20)
            res.raise_for_status()
            
            data = res.json()
            #print(f"\nProcessing {endpoint}")
            
            # Check if we have the expected data structure
            if "statistic" not in data:
                #print(f"Warning: No 'statistic' key in response for {endpoint}")
                continue
                
            # Extract the category name from the endpoint
            category = endpoint.split('_')[1]  # batting, fielding, or pitching
            
            # Get the category data
            category_data = data.get("statistic", {}).get("category", {})
            if not category_data:
                #print(f"Warning: No category data found for {endpoint}")
                continue
                
            # Handle both single and multiple player cases
            players = _force_list(category_data.get("player", []))
            
            if not players:
                #print(f"Warning: No players found in response for {endpoint}")
                continue
                
            #print(f"Found {len(players)} players for {endpoint}")
            
            # Get expected fields for this category
            expected_fields = get_category_fields(category)
            
            for player in players:
                # Convert @ prefixed attributes to regular keys and clean values
                flat_player = {}
                for key, value in player.items():
                    if key.startswith('@'):
                        clean_key = key[1:]
                        flat_player[clean_key] = clean_value(value)
                    else:
                        flat_player[key] = clean_value(value)
                
                # Add metadata
                flat_player["category"] = category
                flat_player["league"] = "mlb"  # Since we're only fetching MLB data now
                flat_player["stat_type"] = category_data.get("@name", "")
                
                # Ensure all expected fields are present
                for field in expected_fields:
                    if field not in flat_player:
                        flat_player[field] = None
                
                rows.append(flat_player)
                
        except requests.exceptions.RequestException as e:
            #print(f"Request error for {endpoint}: {str(e)}")
            continue
        except Exception as e:
            #print(f"Unexpected error for {endpoint}: {str(e)}")
            continue

    # Convert to DataFrame
    df = pd.DataFrame(rows)
    
    if df.empty:
        pass 
        #print("\nWarning: No data was collected from any endpoint!")
    else:
        pass 
        #print(f"\nSuccessfully collected {len(df)} total records")
        #print("\nColumns in the dataset:")
        #print(df.columns.tolist())
        
    return df

def main():
    """Main function to collect all MLB data."""
    # Create csv/MLB directory if it doesn't exist
    csv_dir = os.path.join(DATA_DIR, "MLB")
    os.makedirs(csv_dir, exist_ok=True)
    
    # Fetch player stats
    stats_df = extract_all_player_stats()
    if not stats_df.empty:
        stats_path = os.path.join(csv_dir, "mlb_player_stats.csv")
        stats_df.to_csv(stats_path, index=False)
        #print(f"\nPlayer stats saved to {stats_path}")
    
    # Fetch roster data
    #print("\nFetching roster data...")
    teams = extract_teams()
    if not teams:
        #print("No teams found to process rosters")
        return pd.DataFrame()
        
    all_rosters = []
    
    for team_id, team_name in tqdm(teams, desc="Processing team rosters"):
        try:
            res = fetch_roster_data(team_id)
            all_rosters.append(res)
        except Exception as e:
            #print(f"Error fetching roster for team {team_name}: {str(e)}")
            continue 
    roster_df = pd.DataFrame(all_rosters)

    if not roster_df.empty:
        roster_path = os.path.join(csv_dir, "mlb_rosters.csv")
        roster_df.to_csv(roster_path, index=False)
        #print(f"Roster data saved to {roster_path}")
    
    # Fetch injury data
    injury_df = fetch_injury_data()
    if not injury_df.empty:
        injury_path = os.path.join(csv_dir, "mlb_injuries.csv")
        injury_df.to_csv(injury_path, index=False)
        #print(f"Injury data saved to {injury_path}")
    
    # Fetch standings data
    standings_df = fetch_standings_data()
    if not standings_df.empty:
        standings_path = os.path.join(csv_dir, "mlb_standings.csv")
        standings_df.to_csv(standings_path, index=False)
        #print(f"Standings data saved to {standings_path}")
    
    # #print summary
    #print("\nData Collection Summary:")
    #print(f"Player Stats Records: {len(stats_df) if not stats_df.empty else 0}")
    #print(f"Roster Records: {len(roster_df) if not roster_df.empty else 0}")
    #print(f"Injury Records: {len(injury_df) if not injury_df.empty else 0}")
    #print(f"Standings Records: {len(standings_df) if not standings_df.empty else 0}")

if __name__ == "__main__":
    main()







