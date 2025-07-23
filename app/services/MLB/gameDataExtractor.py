import os
from datetime import datetime, timedelta
import pandas as pd
import csv
from tqdm import tqdm
import requests
import logging
from app.config import GOALSERVE_API_KEY, GOALSERVE_BASE_URL

# Set the data directory
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../data/MLB/game_data')
previous_fetched_date = "01-01-2010"
# Define the API URL template
GOALSERVE_URL = "{}{}/baseball/usa?date={}&json=1"

# Set up logging
logging.basicConfig(filename='fetch_log.txt', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def to_int(value, default=0):
    """Convert a value to an integer, with a default if conversion fails."""
    try:
        return int(value)
    except (ValueError, TypeError):
        return default

def append_to_csv(rows, filename, fields):
    """Append rows to a CSV file, writing headers if the file is new or empty."""
    write_headers = not os.path.exists(filename) or os.stat(filename).st_size == 0
    if write_headers:
        with open(filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
    with open(filename, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writerows(rows)

def extract_game_data_json(game):
    """
    Extract game data from a JSON match object.
    
    Args:
        game (dict): A dictionary representing a single game from the JSON.
    
    Returns:
        dict or None: Game data if venue_name is present, None if missing.
    """
    # Skip if venue_name is missing or empty
    if not game.get('venue_name'):
        return None

    game_data = {
        'game_id': game.get('id', ''),
        'date': game.get('formatted_date', ''),
        'time': game.get('time', ''),
        'venue_name': game.get('venue_name', ''),
        'venue_id': game.get('venue_id', ''),
        'attendance': game.get('attendance', ''),
        'status': game.get('status', ''),
    }

    # if game_data['status'] != 'Final':
    #     return None

    hometeam = game.get('hometeam', {})
    game_data.update({
        'home_team': hometeam.get('name', ''),
        'home_team_id': hometeam.get('id', ''),
        'home_score': to_int(hometeam.get('totalscore')),
        'home_hits': to_int(hometeam.get('hits')),
        'home_errors': to_int(hometeam.get('errors')),
    })

    awayteam = game.get('awayteam', {})
    game_data.update({
        'away_team': awayteam.get('name', ''),
        'away_team_id': awayteam.get('id', ''),
        'away_score': to_int(awayteam.get('totalscore')),
        'away_hits': to_int(awayteam.get('hits')),
        'away_errors': to_int(awayteam.get('errors')),
    })

    return game_data

def fetch_and_save_game_data(start_date_str, end_date_str):
    """Fetch MLB game data for a date range and save to CSV files."""
    os.makedirs(DATA_DIR, exist_ok=True)

    fields = [
        'game_id', 'date', 'time', 'venue_name', 'venue_id', 'attendance', 'status',
        'home_team', 'home_team_id', 'home_score', 'home_hits', 'home_errors',
        'away_team', 'away_team_id', 'away_score', 'away_hits', 'away_errors'
    ]

    start_date = datetime.strptime(start_date_str, "%Y-%m-%d").date()
    end_date = datetime.strptime(end_date_str, "%Y-%m-%d").date()
    date_range = pd.date_range(start=start_date, end=end_date)

    for date in tqdm(date_range, desc=f"Fetching MLB games {start_date_str} to {end_date_str}"):
        date_str = date.strftime("%d.%m.%Y")
        year = date.year
        games_csv = os.path.join(DATA_DIR, f'games_data_combined.csv')
        url = GOALSERVE_URL.format(GOALSERVE_BASE_URL,GOALSERVE_API_KEY,date_str)
        try:
            response = requests.get(url)
            if response.status_code == 200:
                data = response.json()
                if 'scores' not in data or 'category' not in data['scores'] or 'match' not in data['scores']['category']:
                    logging.warning(f"Unexpected API response format for {date_str}")
                    continue
                matches = data['scores']['category']['match']
                if isinstance(matches, dict):
                    matches = [matches]
                rows = []
                for game in matches:
                    try:
                        game_data = extract_game_data_json(game)
                        if game_data is not None:
                            rows.append(game_data)
                        else:
                            logging.info(f"Skipped a game on {date_str} due to missing venue information")
                    except Exception as e:
                        logging.error(f"Error processing game on {date_str}: {str(e)}")
                        continue
                if rows:
                    append_to_csv(rows, games_csv, fields)
            else:
                print("Failed to fetch data for ", date_str, ": HTTP ", response.status_code)
                logging.error(f"Failed to fetch data for {date_str}: HTTP {response.status_code}")
        except Exception as e:
            print("Error fetching data for ", date_str, ": ", str(e))
            logging.error(f"Error fetching data for {date_str}: {str(e)}")
    logging.info(f"Game data for {start_date_str} to {end_date_str} saved.")


def run_game_data_collect_final():
    # Get yesterday and today as strings
    df = pd.read_csv(os.path.join(DATA_DIR, f'games_data_combined.csv'))

    today = datetime.now().date() 
    if df.shape[0] > 0:
        previous_fetched_date = df.iloc[-1]["date"]


    # Convert to string format 'YYYY-MM-DD'
    today_str = today.strftime("%Y-%m-%d")

    # Convert the string to a datetime object
    previous_fetched_date = datetime.strptime(previous_fetched_date, "%d.%m.%Y")

    previous_fetched_date = previous_fetched_date + timedelta(days=1)

    # Format the date into the desired format
    previous_fetched_date = previous_fetched_date.strftime("%Y-%m-%d")

    # Call the function for yesterday only
    fetch_and_save_game_data(previous_fetched_date, today_str)


if __name__ == '__main__':
    run_game_data_collect_final()