import requests
import xml.etree.ElementTree as ET
import pandas as pd
from datetime import datetime, timedelta
import uuid
import time
from tqdm import tqdm
import os
import json 

player_position_map_file_path = 'app/data/NFL/player_position_map.txt'

with open(player_position_map_file_path, 'r') as file:
    player_id_to_pos = json.load(file)

# Define DATA_DIR for the CSV path
DATA_DIR = 'app/data'

def safe_int(value, default=0):
    """Safely convert a value to an integer, returning default if conversion fails."""
    try:
        return int(value)
    except (ValueError, TypeError):
        return default

def safe_float(value, default=0.0):
    """Safely convert a value to a float, returning default if conversion fails."""
    try:
        return float(value)
    except (ValueError, TypeError):
        return default

def generate_dates(start_date, end_date):
    """Generate list of dates from start_date to end_date, excluding March-July."""
    dates = []
    current_date = start_date
    excluded_months = {3, 4, 5, 6}  # March, April, May, June
    while current_date <= end_date:
        if current_date.month not in excluded_months:
            dates.append(current_date.strftime('%d.%m.%Y'))
        current_date += timedelta(days=1)
    return dates

def fetch_nfl_data(date):
    """Fetch NFL data for a specific date."""
    base_url = "http://www.goalserve.com/getfeed/48cbeb0a39014dc2d6db08dd947404e4/football/nfl-scores"
    url = f"{base_url}?date={date}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.text
    except requests.RequestException as e:
        print(f"Error fetching data for {date}: {e}")
        return None

def parse_player_stats(player_elem, category, team_id, match_date, home_team_name, home_team_id, away_team_name, away_team_id):
    """Parse player stats from XML element."""
    player_id = player_elem.get('id', str(uuid.uuid4()))
    stats = {
        'id': player_id,
        'name': player_elem.get('name', ''),
        'team_id': team_id,
        'category': category,
        'date': match_date,
        'home_team_name': home_team_name,
        'home_team_id': home_team_id,
        'away_team_name': away_team_name,
        'away_team_id': away_team_id,
        'completion_pct': 0.0,
        'completions': 0,
        'passing_attempts': 0,
        'passing_yards': 0,
        'yards_per_pass_avg': 0.0,
        'passing_touchdowns': 0,
        'passing_touchdowns_pct': 0.0,
        'interceptions': 0,
        'interceptions_pct': 0.0,
        'longest_pass': 0,
        'quaterback_rating': 0.0,
        'rank': 0,
        'sacked_yards_lost': 0,
        'sacks': 0,
        'yards': 0,
        'yards_per_game': 0.0,
        'fumbles': 0,
        'fumbles_lost': 0,
        'longest_rush': 0,
        'over_20_yards': 0,
        'rushing_attempts': 0,
        'rushing_first_downs': 0,
        'rushing_touchdowns': 0,
        'yards_per_rush_avg': 0.0,
        'longest_reception': 0,
        'receiving_first_downs': 0,
        'receiving_targets': 0,
        'receiving_touchdowns': 0,
        'receiving_yards': 0,
        'receptions': 0,
        'yards_after_catch': 0,
        'yards_per_reception_avg': 0.0,
        'assisted_tackles': 0,
        'blocked_kicks': 0,
        'forced_fumbles': 0,
        'fumbles_recovered': 0,
        'fumbles_returned_for_touchdowns': 0,
        'intercepted_returned_yards': 0,
        'interceptions_returned_for_touchdowns': 0,
        'longest_interception_return': 0,
        'passes_defended': 0,
        'tackles_for_loss': 0,
        'total_tackles': 0,
        'unassisted_tackles': 0,
        'yards_lost_on_sack': 0,
        'extra_points': 0,
        'field_goals': 0,
        'return_touchdowns': 0,
        'total_points': 0,
        'total_points_per_game': 0.0,
        'total_touchdowns': 0,
        'two_point_conversions': 0,
        'fair_catches': 0,
        'kickoff_return_touchdowns': 0,
        'kickoff_return_yards': 0,
        'kickoff_returned_attempts': 0,
        'longest_kickoff_return': 0,
        'longest_punt_return': 0,
        'punt_return_touchdowns': 0,
        'punts_returned': 0,
        'yards_per_kickoff_avg': 0.0,
        'yards_per_punt_avg': 0.0,
        'yards_returned_on_punts': 0,
        'extra_points_attempts': 0,
        'extra_points_made': 0,
        'extra_points_made_pct': 0.0,
        'field_goals_attempts': 0,
        'field_goals_from_1_19_yards': 0,
        'field_goals_from_20_29_yards': 0,
        'field_goals_from_30_39_yards': 0,
        'field_goals_from_40_49_yards': 0,
        'field_goals_from_50_yards': 0,
        'field_goals_made': 0,
        'field_goals_made_pct': 0.0,
        'longest_goal_made': 0,
        'blocked_punts': 0,
        'gross_punt_yards': 0,
        'gross_punting_avg': 0.0,
        'inside_20_yards_punt': 0,
        'longest_punt': 0,
        'net_punting_avg': 0.0,
        'punts': 0,
        'touchbacks': 0,
        'yards_returned_on_punts_avg': 0.0,
        'rushing_yards': 0,
        'player_position': player_id_to_pos.get(str(player_id), 'unknown')
    }

    if category == 'passing':
        comp_att = player_elem.get('comp_att', '0/0').split('/')
        completions = safe_int(comp_att[0]) if len(comp_att) == 2 else 0
        passing_attempts = safe_int(comp_att[1]) if len(comp_att) == 2 else 0
        stats.update({
            'completion_pct': (completions / passing_attempts * 100) if passing_attempts > 0 else 0.0,
            'completions': completions,
            'passing_attempts': passing_attempts,
            'passing_yards': safe_int(player_elem.get('yards', '0')),
            'yards_per_pass_avg': safe_float(player_elem.get('average', '0.0')),
            'passing_touchdowns': safe_int(player_elem.get('passing_touch_downs', '0')),
            'passing_touchdowns_pct': (safe_int(player_elem.get('passing_touch_downs', '0')) / passing_attempts * 100) if passing_attempts > 0 else 0.0,
            'interceptions': safe_int(player_elem.get('interceptions', '0')),
            'interceptions_pct': (safe_int(player_elem.get('interceptions', '0')) / passing_attempts * 100) if passing_attempts > 0 else 0.0,
            'sacks': safe_int(player_elem.get('sacks', '0-0').split('-')[0]) if '-' in player_elem.get('sacks', '0-0') else 0,
            'sacked_yards_lost': safe_int(player_elem.get('sacks', '0-0').split('-')[1]) if '-' in player_elem.get('sacks', '0-0') else 0,
            'quaterback_rating': safe_float(player_elem.get('rating', '0.0')),
            'two_point_conversions': safe_int(player_elem.get('two_pt', '0')),
            'longest_pass': 0
        })
    elif category == 'rushing':
        stats.update({
            'rushing_attempts': safe_int(player_elem.get('total_rushes', '0')),
            'rushing_yards': safe_int(player_elem.get('yards', '0')),
            'yards_per_rush_avg': safe_float(player_elem.get('average', '0.0')),
            'rushing_touchdowns': safe_int(player_elem.get('rushing_touch_downs', '0')),
            'longest_rush': safe_int(player_elem.get('longest_rush', '0')),
            'rushing_first_downs': 0,
            'over_20_yards': 0,
            'two_point_conversions': safe_int(player_elem.get('two_pt', '0')),
            'fumbles': 0,
            'fumbles_lost': 0
        })
    elif category == 'receiving':
        stats.update({
            'receptions': safe_int(player_elem.get('total_receptions', '0')),
            'receiving_yards': safe_int(player_elem.get('yards', '0')),
            'yards_per_reception_avg': safe_float(player_elem.get('average', '0.0')),
            'receiving_touchdowns': safe_int(player_elem.get('receiving_touch_downs', '0')),
            'longest_reception': safe_int(player_elem.get('longest_reception', '0')),
            'receiving_targets': safe_int(player_elem.get('targets', '0')),
            'receiving_first_downs': 0,
            'yards_after_catch': 0,
            'two_point_conversions': safe_int(player_elem.get('two_pt', '0')),
            'fumbles': 0,
            'fumbles_lost': 0
        })
    elif category == 'fumbles':
        stats.update({
            'fumbles': safe_int(player_elem.get('total', '0')),
            'fumbles_lost': safe_int(player_elem.get('lost', '0')),
            'fumbles_recovered': safe_int(player_elem.get('rec', '0')),
            'fumbles_returned_for_touchdowns': safe_int(player_elem.get('rec_td', '0'))
        })
    elif category == 'defensive':
        stats.update({
            'total_tackles': safe_int(player_elem.get('tackles', '0')),
            'unassisted_tackles': safe_int(player_elem.get('unassisted_tackles', '0')),
            'assisted_tackles': safe_int(player_elem.get('tackles', '0')) - safe_int(player_elem.get('unassisted_tackles', '0')),
            'sacks': safe_float(player_elem.get('sacks', '0')),
            'tackles_for_loss': safe_int(player_elem.get('tfl', '0')),
            'passes_defended': safe_int(player_elem.get('passes_defended', '0')),
            'forced_fumbles': safe_int(player_elem.get('ff', '0')),
            'blocked_kicks': safe_int(player_elem.get('blocked_kicks', '0')),
            'interceptions_returned_for_touchdowns': safe_int(player_elem.get('interceptions_for_touch_downs', '0')),
            'yards_lost_on_sack': 0,
            'intercepted_returned_yards': 0,
            'longest_interception_return': 0
        })
    elif category == 'kicking':
        field_goals = player_elem.get('field_goals', '0/0').split('/')
        extra_points = player_elem.get('extra_point', '0/0').split('/')
        field_goals_made = safe_int(field_goals[0]) if len(field_goals) == 2 else 0
        field_goals_attempts = safe_int(field_goals[1]) if len(field_goals) == 2 else 0
        extra_points_made = safe_int(extra_points[0]) if len(extra_points) == 2 else 0
        extra_points_attempts = safe_int(extra_points[1]) if len(extra_points) == 2 else 0
        stats.update({
            'field_goals': field_goals_made,
            'field_goals_attempts': field_goals_attempts,
            'field_goals_made': field_goals_made,
            'field_goals_made_pct': safe_float(player_elem.get('pct', '0.0')),
            'longest_goal_made': safe_int(player_elem.get('long', '0')),
            'extra_points': extra_points_made,
            'extra_points_attempts': extra_points_attempts,
            'extra_points_made': extra_points_made,
            'extra_points_made_pct': (extra_points_made / extra_points_attempts * 100) if extra_points_attempts > 0 else 0.0,
            'total_points': safe_int(player_elem.get('points', '0')),
            'field_goals_from_1_19_yards': safe_int(player_elem.get('field_goals_from_1_19_yards', '0')),
            'field_goals_from_20_29_yards': safe_int(player_elem.get('field_goals_from_20_29_yards', '0')),
            'field_goals_from_30_39_yards': safe_int(player_elem.get('field_goals_from_30_39_yards', '0')),
            'field_goals_from_40_49_yards': safe_int(player_elem.get('field_goals_from_40_49_yards', '0')),
            'field_goals_from_50_yards': safe_int(player_elem.get('field_goals_from_50_yards', '0'))
        })
    elif category == 'punting':
        stats.update({
            'punts': safe_int(player_elem.get('total', '0')),
            'gross_punt_yards': safe_int(player_elem.get('yards', '0')),
            'gross_punting_avg': safe_float(player_elem.get('average', '0.0')),
            'longest_punt': safe_int(player_elem.get('lg', '0')),
            'inside_20_yards_punt': safe_int(player_elem.get('in20', '0')),
            'touchbacks': safe_int(player_elem.get('touchbacks', '0')),
            'blocked_punts': 0,
            'yards_returned_on_punts': 0,
            'yards_returned_on_punts_avg': 0.0
        })
    elif category == 'kick_returns':
        stats.update({
            'kickoff_returned_attempts': safe_int(player_elem.get('total', '0')),
            'kickoff_return_yards': safe_int(player_elem.get('yards', '0')),
            'yards_per_kickoff_avg': safe_float(player_elem.get('average', '0.0')),
            'longest_kickoff_return': safe_int(player_elem.get('lg', '0')),
            'kickoff_return_touchdowns': safe_int(player_elem.get('kick_return_td', '0')),
            'return_touchdowns': safe_int(player_elem.get('exp_return_td', '0'))
        })
    elif category == 'punt_returns':
        stats.update({
            'punts_returned': safe_int(player_elem.get('total', '0')),
            'yards_returned_on_punts': safe_int(player_elem.get('yards', '0')),
            'yards_per_punt_avg': safe_float(player_elem.get('average', '0.0')),
            'longest_punt_return': safe_int(player_elem.get('lg', '0')),
            'punt_return_touchdowns': safe_int(player_elem.get('kick_return_td', '0')),
            'return_touchdowns': safe_int(player_elem.get('exp_return_td', '0')),
            'fair_catches': 0
        })

    return stats

def parse_xml_data(xml_data, date, csv_file):
    """Parse XML data and append player stats to CSV."""
    try:
        root = ET.fromstring(xml_data)
        all_stats = []

        for match in root.findall('.//match'):
            match_date = date
            home_team = match.find('hometeam')
            away_team = match.find('awayteam')
            home_team_name = home_team.get('name', '') if home_team is not None else ''
            home_team_id = home_team.get('id', '') if home_team is not None else ''
            away_team_name = away_team.get('name', '') if away_team is not None else ''
            away_team_id = away_team.get('id', '') if away_team is not None else ''

            for team_type in ['hometeam', 'awayteam']:
                team_id = home_team_id if team_type == 'hometeam' else away_team_id
                for category in ['passing', 'rushing', 'receiving', 'fumbles', 'defensive', 'kicking', 'punting', 'kick_returns', 'punt_returns']:
                    for player in match.findall(f'.//{category}/{team_type}/player'):
                        stats = parse_player_stats(player, category, team_id, match_date, home_team_name, home_team_id, away_team_name, away_team_id)
                        all_stats.append(stats)

        # Append to CSV immediately with flush
        df = pd.DataFrame(all_stats)
        if not df.empty:
            df.to_csv(csv_file, mode='a', header=False, index=False)
            # Flush the file buffer to ensure immediate write
            with open(csv_file, 'a') as f:
                f.flush()
                os.fsync(f.fileno())
        
        return len(all_stats)
    except ET.ParseError as e:
        print(f"Error parsing XML for {date}: {e}")
        return 0

def create_empty_csv(csv_file):
    """Create an empty CSV file with all specified headers."""
    headers = [
        'id', 'name', 'team_id', 'category', 'date', 'home_team_name', 'home_team_id', 'away_team_name', 'away_team_id',
        'completion_pct', 'completions', 'passing_attempts', 'passing_yards', 'yards_per_pass_avg', 'passing_touchdowns',
        'passing_touchdowns_pct', 'interceptions', 'interceptions_pct', 'longest_pass', 'quaterback_rating', 'rank',
        'sacked_yards_lost', 'sacks', 'yards', 'yards_per_game', 'fumbles', 'fumbles_lost', 'longest_rush',
        'over_20_yards', 'rushing_attempts', 'rushing_first_downs', 'rushing_touchdowns', 'yards_per_rush_avg',
        'longest_reception', 'receiving_first_downs', 'receiving_targets', 'receiving_touchdowns', 'receiving_yards',
        'receptions', 'yards_after_catch', 'yards_per_reception_avg', 'assisted_tackles', 'blocked_kicks',
        'forced_fumbles', 'fumbles_recovered', 'fumbles_returned_for_touchdowns', 'intercepted_returned_yards',
        'interceptions_returned_for_touchdowns', 'longest_interception_return', 'passes_defended', 'tackles_for_loss',
        'total_tackles', 'unassisted_tackles', 'yards_lost_on_sack', 'extra_points', 'field_goals', 'return_touchdowns',
        'total_points', 'total_points_per_game', 'total_touchdowns', 'two_point_conversions', 'fair_catches',
        'kickoff_return_touchdowns', 'kickoff_return_yards', 'kickoff_returned_attempts', 'longest_kickoff_return',
        'longest_punt_return', 'punt_return_touchdowns', 'punts_returned', 'yards_per_kickoff_avg', 'yards_per_punt_avg',
        'yards_returned_on_punts', 'extra_points_attempts', 'extra_points_made', 'extra_points_made_pct',
        'field_goals_attempts', 'field_goals_from_1_19_yards', 'field_goals_from_20_29_yards',
        'field_goals_from_30_39_yards', 'field_goals_from_40_49_yards', 'field_goals_from_50_yards',
        'field_goals_made', 'field_goals_made_pct', 'longest_goal_made', 'blocked_punts', 'gross_punt_yards',
        'gross_punting_avg', 'inside_20_yards_punt', 'longest_punt', 'net_punting_avg', 'punts', 'touchbacks',
        'yards_returned_on_punts_avg', 'rushing_yards'
    ]
    pd.DataFrame(columns=headers).to_csv(csv_file, index=False)

def get_last_date_from_csv(csv_file):
    """Extract the last date from the CSV file using pandas."""
    try:
        df = pd.read_csv(csv_file)
        if not df.empty:
            # Assuming the date is in the 5th column (index 4)
            date_series = df.iloc[:, 4]
            last_date_str = date_series.iloc[-1]
            return datetime.strptime(last_date_str, '%d.%m.%Y')
    except FileNotFoundError:
        print("CSV file not found. Defaulting to start date.")
    except Exception as e:
        print(f"Error reading CSV: {e}")
    return datetime(2010, 1, 1)


def main(): 
    end_date = datetime.today() - timedelta(days=1)

    csv_file = 'nfl_player_stats_test_with_positions.csv'
    csv_file = os.path.join(DATA_DIR, 'NFL', csv_file)
    start_date = get_last_date_from_csv(csv_file) + timedelta(days=1)

    # Ensure the output directory exists
    #os.makedirs(os.path.dirname(csv_file), exist_ok=True)

    # Create empty CSV with headers
    # create_empty_csv(csv_file)
    # print(f"Empty CSV file created: {csv_file}")


    dates = generate_dates(start_date, end_date)
    total_records = 0


    for date in tqdm(dates, desc="Fetching NFL data"):
        xml_data = fetch_nfl_data(date)
        if xml_data:
            records = parse_xml_data(xml_data, date, csv_file)
            total_records += records
        time.sleep(1)  

    print(f"Data extraction complete. Total records saved: {total_records}")
    print(f"Data saved to {csv_file}")

def extract_todays_match_data():
    main()

if __name__ == "__main__":
    main()
