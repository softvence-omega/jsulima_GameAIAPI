import time

import requests
from pprint import pprint
from datetime import datetime, timedelta
from tqdm import tqdm
import json

from app.config import GOALSERVE_API_KEY, GOALSERVE_BASE_URL


def _force_list(x):
    if isinstance(x, list):
        return x
    return [x]


def extract_team_info():
    start_date = datetime(2025, 2, 7)
    end_date = datetime.today()

    total_days = (end_date - start_date).days + 1  # Include both start and end dates
    matches_list = []

    current = start_date
    for _ in tqdm(range(total_days), desc="Extracting matches"):

        today = current.strftime("%d-%m-%Y")
        day, month, year = today.split("-")

        current += timedelta(days=210)

        url = f"{GOALSERVE_BASE_URL}{GOALSERVE_API_KEY}/football/nfl-scores?date={day}.{month}.{year}&json=1"
        res = requests.get(url, timeout=60)

        print(res.status_code)
        if res.status_code == 500:
            continue

        wait_time = 1
        while res.status_code == 429:
            time.sleep(wait_time)
            wait_time *= 2
            res = requests.get(url, timeout=60)

        res.raise_for_status()
        data = res.json()

        print(data)

        if not 'scores' in data:
            continue
        if not 'category' in data['scores']:
            continue

        if not 'match' in data['scores']['category']:
            continue

        for match in _force_list(data['scores']['category']['match']):
            print("++++")
            home_team = match['hometeam']['name']
            away_team = match['awayteam']['name']
            home_score_str = match['hometeam']['totalscore']
            home_score = int(home_score_str) if home_score_str.isdigit() else 0

            away_score_str = match['awayteam']['totalscore']
            away_score = int(away_score_str) if away_score_str.isdigit() else 0

            date = today

            # Append the match details to the list
            match_details = {
                "home_team": home_team,
                "away_team": away_team,
                "home_score": home_score,
                "away_score": away_score,
                "date": date,
                "url": url
            }

            matches_list.append(match_details)
            print(len(matches_list))
            print(url)

    print(len(matches_list))
    with open("matches_list.txt", "w", encoding="utf-8") as file:
        json.dump(matches_list, file, indent=4)


# extract_team_info()
