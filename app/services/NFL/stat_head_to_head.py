import os
from app.services.helper import safe_float
from app import config
import pandas as pd
from datetime import datetime, timedelta
from app.services.NFL.data_processor import NflDataProcessor
from app.config import NFL_DIR, GOALSERVE_API_KEY
from pathlib import Path
import numpy as np


class HeadToHeadRecord:
    def __init__(self):
        self.data_processor = NflDataProcessor()
        self.api_key = GOALSERVE_API_KEY
  
    def fetch_head_to_head_teams(self, team1_name, team2_name, data_file=None):
        # Define the data directory and file path
        
        file_path = data_file if data_file else os.path.join(NFL_DIR, "Game_data(historical).csv")

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
        # Convert all numpy types in records
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
    head_to_head=HeadToHeadRecord()
    team1 = "Baltimore Ravens"
    team2 = "Indianapolis Colts"
    last_5_match_list = head_to_head.fetch_head_to_head_teams(team1, team2)

    if last_5_match_list is not None and len(last_5_match_list) > 0:
        print("\n🏈 Last 5 H2H matches between {} and {}:\n".format(team1, team2))
        # Convert list of dicts to DataFrame for pretty printing
        df = pd.DataFrame(last_5_match_list)
        print(df[['date', 'home_team', 'away_team', 'home_score', 'away_score']])
    else:
        print("❌ No match data to display!")
