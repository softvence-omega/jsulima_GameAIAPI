import os
import pandas as pd
from app.config import DATA_DIR
import numpy as np

def fetch_head_to_head_teams(team1_name, team2_name):
    # Define the data directory and file path
    csv_dir = os.path.join(DATA_DIR, "MLB")
    os.makedirs(csv_dir, exist_ok=True)
    # file_path =  os.path.join(csv_dir , "games_data(2010-2024).csv")
    file_path = os.path.join(csv_dir, 'game_data', "games_data_combined.csv")
 
    print(f"📂 Looking for file at: {file_path}")

    # Load the data
    df = pd.read_csv(file_path)

    # Filter matches where either is home and the other is away
    h2h_df = df[
        ((df['home_team'] == team1_name) & (df['away_team'] == team2_name)) |
        ((df['home_team'] == team2_name) & (df['away_team'] == team1_name))
    ]

    # Sort by most recent date
    h2h_df = h2h_df.sort_values('date', ascending=False)

    # Get last 5 matches
    last_5_h2h = h2h_df.head(5)


    records = last_5_h2h.to_dict(orient="records")

    def convert_types(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        return obj

    cleaned_records = [
        {k: convert_types(v) for k, v in rec.items()}
        for rec in records
    ]
    return [
        {k: rec[k] for k in ['date', 'home_team', 'away_team', 'home_score', 'away_score'] if k in rec}
        for rec in cleaned_records
    ]

if __name__ == "__main__":
 
    team1 = "Chicago Cubs"
    team2 = "Houston Astros"
    last_5_match_list = fetch_head_to_head_teams(team1, team2)

    if last_5_match_list is not None and len(last_5_match_list) > 0:
        print("\n🏈 Last 5 H2H matches between {} and {}:\n".format(team1, team2))
        for match in last_5_match_list:
            print(match)
    else:
        print("❌ No match data to display!")