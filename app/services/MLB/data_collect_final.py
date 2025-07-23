import csv
from collections import OrderedDict
import os
import time
import requests
from datetime import datetime, timedelta
from tqdm import tqdm
import pandas as pd 

from app.config import GOALSERVE_API_KEY
from app.config import GOALSERVE_BASE_URL, GOALSERVE_API_KEY, DATA_DIR

BATTING_CSV = os.path.join(DATA_DIR, "MLB", "batting", "batting_data_combined.csv")
PITCHING_CSV = os.path.join(DATA_DIR, "MLB", "pitching", "pitching_data_combined.csv")

# Define fields for each category
batting_fields = [
    "game_date", "game_id", "team", "id", "name", "position", "at_bats", "runs", "hits", "doubles",
    "triples", "home_runs", "sac_fly", "hit_by_pitch", "runs_batted_in", "walks",
    "strikeouts", "average", "stolen_bases", "on_base_percentage", "slugging_percentage",
    "caught_stealing"
]

pitching_fields = [
    "game_date", "game_id", "team", "id", "name", "innings_pitched", "runs", "hits", "earned_runs",
    "walks", "strikeouts", "home_runs", "pc_st", "earned_runs_average", "win",
    "loss", "holds", "saves", "hbp", "starting_pitcher"
]


# Function to safely get JSON field
def safe_get(obj, key, default=""):
    return obj.get(key, default) if isinstance(obj, dict) else default

# Function to convert date from DD.MM.YYYY to YYYY-MM-DD
def convert_date_format(date_str):
    try:
        return datetime.strptime(date_str, '%d.%m.%Y').strftime('%Y-%m-%d')
    except ValueError:
        return ""


# Function to append rows to CSV
def append_to_csv(rows, filename, fields):
    write_headers = not os.path.exists(filename) or os.stat(filename).st_size == 0
    if write_headers:
        with open(filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()

    with open(filename, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writerows(rows)

# Function to extract player data
def extract_player_data(json_data, fallback_date=""):
    batting_rows = []
    pitching_rows = []
    
    try:
        matches = json_data["scores"]["category"]["match"]
        if isinstance(matches, dict):  # Handle single match case
            matches = [matches]
        for match in matches:
            game_id = safe_get(match, "id")
            match_date = convert_date_format(safe_get(match, "date", fallback_date))
            if not match_date:
                print(f"Skipping game {game_id}: Invalid or missing date")
                continue
            hometeam = safe_get(match["hometeam"], "name")
            awayteam = safe_get(match["awayteam"], "name")
            
            # Get starting pitchers' names
            starting_pitcher_home = safe_get(match["starting_pitchers"]["hometeam"]["player"], "name")
            starting_pitcher_away = safe_get(match["starting_pitchers"]["awayteam"]["player"], "name")

            stats = match.get("stats")
            if not stats:
                continue
            # Process Hitters
            for team_type in ["hometeam", "awayteam"]:
                team_name = hometeam if team_type == "hometeam" else awayteam
                team_hitters = stats.get("hitters", {}).get(team_type)
                if not team_hitters:
                    continue
                hitters = safe_get(team_hitters, "player", "")
                if isinstance(hitters, dict):  # Handle single player case
                    hitters = [hitters]
                elif not isinstance(hitters, list):
                    hitters = []
                for player in hitters:
                    row = OrderedDict.fromkeys(batting_fields, "")
                    row["game_date"] = match_date
                    row["game_id"] = game_id
                    row["team"] = team_name
                    row["id"] = safe_get(player, "id")
                    row["name"] = safe_get(player, "name")
                    row["position"] = safe_get(player, "pos")
                    row["at_bats"] = safe_get(player, "at_bats")
                    row["runs"] = safe_get(player, "runs")
                    row["hits"] = safe_get(player, "hits")
                    row["doubles"] = safe_get(player, "doubles")
                    row["triples"] = safe_get(player, "triples")
                    row["home_runs"] = safe_get(player, "home_runs")
                    row["sac_fly"] = safe_get(player, "sac_fly")
                    row["hit_by_pitch"] = safe_get(player, "hit_by_pitch")
                    row["runs_batted_in"] = safe_get(player, "runs_batted_in")
                    row["walks"] = safe_get(player, "walks")
                    row["strikeouts"] = safe_get(player, "strikeouts")
                    row["average"] = safe_get(player, "average")
                    row["stolen_bases"] = safe_get(player, "stolen_bases")
                    row["on_base_percentage"] = safe_get(player, "on_base_percentage")
                    row["slugging_percentage"] = safe_get(player, "slugging_percentage")
                    row["caught_stealing"] = safe_get(player, "cs")
                    batting_rows.append(row)
            
            # Process Pitchers
            for team_type in ["hometeam", "awayteam"]:
                team_name = hometeam if team_type == "hometeam" else awayteam
                starting_pitcher = starting_pitcher_home if team_type == "hometeam" else starting_pitcher_away
                team_pitchers = stats.get("pitchers", {}).get(team_type)
                if not team_pitchers:
                    continue
                pitchers = safe_get(team_pitchers, "player", "")
                if isinstance(pitchers, dict):  # Handle single player case
                    pitchers = [pitchers]
                elif not isinstance(pitchers, list):
                    pitchers = []
                for player in pitchers:
                    row = OrderedDict.fromkeys(pitching_fields, "")
                    row["game_date"] = match_date
                    row["game_id"] = game_id
                    row["team"] = team_name
                    row["id"] = safe_get(player, "id")
                    row["name"] = safe_get(player, "name")
                    row["innings_pitched"] = safe_get(player, "innings_pitched")
                    row["runs"] = safe_get(player, "runs")
                    row["hits"] = safe_get(player, "hits")
                    row["earned_runs"] = safe_get(player, "earned_runs")
                    row["walks"] = safe_get(player, "walks")
                    row["strikeouts"] = safe_get(player, "strikeouts")
                    row["home_runs"] = safe_get(player, "home_runs")
                    row["pc_st"] = safe_get(player, "pc-st")
                    row["earned_runs_average"] = safe_get(player, "earned_runs_average")
                    row["win"] = safe_get(player, "win")
                    row["loss"] = safe_get(player, "loss")
                    row["holds"] = safe_get(player, "holds")
                    row["saves"] = safe_get(player, "saves")
                    row["hbp"] = safe_get(player, "hbp")
                    row["starting_pitcher"] = "1" if safe_get(player, "name") == starting_pitcher else "0"
                    pitching_rows.append(row)


            
    except KeyError as e:
        #print(f"Error: Missing key in JSON structure: {e}")
        return [], []
    
    return batting_rows, pitching_rows



def extract_pitcher_and_batter_data(start_date, end_date):
    start_date = datetime.strptime(start_date, "%Y-%m-%d")
    end_date = datetime.strptime(end_date, "%Y-%m-%d")
    total_days = (end_date - start_date).days + 1

    for _ in tqdm(range(total_days), desc="Processing days"):
        time.sleep(1)
        day = f"{start_date.day:02}"
        month = f"{start_date.month:02}"
        year = start_date.year
        game_date = f"{year}-{month}-{day}"
        url = f"{GOALSERVE_BASE_URL}{GOALSERVE_API_KEY}/baseball/usa?date={day}.{month}.{year}&json=1"
        
        response = requests.get(url)
        waiting_time = 1 
        while response.status_code == 429:
            time.sleep(waiting_time)
            response = requests.get(url)
            waiting_time *= 2

        if response.status_code == 200:
            json_data = response.json()
            batting_rows, pitching_rows = extract_player_data(json_data, game_date)
            if batting_rows or pitching_rows:
                append_to_csv(batting_rows, BATTING_CSV, batting_fields )
                append_to_csv(pitching_rows, PITCHING_CSV, pitching_fields)
            else:
                pass
                #print(f"No data to append for {game_date}")
        else:
            print(f"Failed to fetch data for {game_date}: Status code {response.status_code}")
        
        start_date += timedelta(days=1)

def run_batting_pitching_data_collect_final():
    if not os.path.exists(BATTING_CSV):
        os.makedirs(BATTING_CSV)
    if not os.path.exists(PITCHING_CSV):
        os.makedirs(PITCHING_CSV)


    # Check if file  has content
    if os.path.getsize(BATTING_CSV) > 0:
        try:
            batting_df = pd.read_csv(BATTING_CSV)
            if batting_df.shape[0] > 0:
                previous_fetched_date = batting_df.iloc[-1]["game_date"]
                # Convert the string to a datetime object
                previous_fetched_date = datetime.strptime(previous_fetched_date, "%Y-%m-%d")
                previous_fetched_date = previous_fetched_date + timedelta(days=1)
                # Format the date into the desired format
                previous_fetched_date = previous_fetched_date.strftime("%Y-%m-%d")
            else:
                previous_fetched_date = "2010-01-01"  # Default start date if file is empty
        except pd.errors.EmptyDataError:
            previous_fetched_date = "2010-01-01"  # Default start date if file is empty
    else:
        previous_fetched_date = "2010-01-01"  # Default start date if file doesn't exist or is empty

    current_date = datetime.today().date()  # Collect data up to yesterday
    current_date = current_date.strftime("%Y-%m-%d")
    
    print(f"Collecting data from {previous_fetched_date} to {current_date}")
    extract_pitcher_and_batter_data(previous_fetched_date, current_date)
    print("Data successfully appended to batting.csv, pitching.csv, and fielding.csv")

if __name__ == "__main__":
    run_batting_pitching_data_collect_final()