import requests
import pandas as pd 
from tqdm.auto import tqdm

from app.config import GOALSERVE_API_KEY, GOALSERVE_BASE_URL



def _force_list(x):
    """Return x as a list.  If x is None → [], if it's a dict → [dict]."""
    if x is None:
        return []
    return x if isinstance(x, list) else [x]

def extract_teams():
    team_ids = []
    url = f"{GOALSERVE_BASE_URL}{GOALSERVE_API_KEY}/football/nfl-standings?json=1"

    res = requests.get(url, timeout=20)
    res.raise_for_status()

    data = res.json()
    teams = []

    leagues = data.get("standings", {}).get("category", {}).get("league", [])
    for league in leagues:
        for division in _force_list(league.get("division")):
            for team in _force_list(division.get("team")):
                teams.append((team.get("id"), team.get("name")))

    return teams

def extract_injury_report(team_id):
    """
    Args:
        report_json (dict) – the JSON block you posted
    Returns:
        list[dict] – [{'player_id': ..., 'player_name': ..., 'status': ...}, …]
        and a pandas DataFrame with the same columns
    """
    url = f"{GOALSERVE_BASE_URL}{GOALSERVE_API_KEY}/football/{team_id}_injuries?json=1"
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
        url = f"{GOALSERVE_BASE_URL}{GOALSERVE_API_KEY}/football/{team_id}_rosters?json=1"

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


def extract_all_player_stats():
    rows = []
    teams = extract_teams()

    for team_id, team_name in tqdm(teams, desc="Processing teams", total=len(teams)):
        url = f"{GOALSERVE_BASE_URL}{GOALSERVE_API_KEY}/football/{team_id}_player_stats?json=1"
        res = requests.get(url)
        res.raise_for_status()

        data = res.json()
        # Loop through each category (e.g., Passing, Rushing)
        for category in data.get("statistic", {}).get("category", []):
            cat_name = category.get("name")
            for player in _force_list(category.get("player")):
                flat_player = player.copy()
                flat_player["category"] = cat_name
                flat_player["team_id"] = team_id
                rows.append(flat_player)

    # Convert to DataFrame
    df = pd.DataFrame(rows)

    # Rename 'yards' column based on category
    if 'yards' in df.columns and 'category' in df.columns:
        df['passing_yards'] = df.apply(lambda row: row['yards'] if row['category'] == 'Passing' else 0, axis=1)
        df['rushing_yards'] = df.apply(lambda row: row['yards'] if row['category'] == 'Rushing' else 0, axis=1)
        # df = df.drop(columns=['yards'])

    return df



df = extract_all_player_stats()
df.to_csv("all_player_stats.csv", index=False)
#printdf['rushing_yards'].value_counts())
