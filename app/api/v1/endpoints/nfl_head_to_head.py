
import os
from app.schemas.nfl_schemas import NFLTeams
from app.services.NFL.stat_head_to_head import HeadToHeadRecord

from app.config import GOALSERVE_API_KEY, NFL_DIR
from fastapi import APIRouter





router=APIRouter()


"""
API_KEY=GOALSERVE_API_KEY
h2h = HeadToHeadRecord()

@router.post("/head_to_head")
async def get_nfl_head_to_head_data(request: NFLTeams):
    hometeam, awayteam = request.hometeam, request.awayteam
    response = h2h.fetch_head_to_head_teams(hometeam, awayteam)

    result = {
        "wins": 0,
        "losses": 0,
        "draws": 0
    }

    for match in response:
        if match['home_team'] != hometeam:
            match['home_team'], match['away_team'] = match['away_team'], match['home_team']
            match['home_score'], match['away_score'] = match['away_score'], match['home_score']
        
        if match['home_score'] > match['away_score']:
            result["wins"] += 1
        elif match['home_score'] < match['away_score']:
            result["losses"] += 1
        else:
            response['draws'] += 1

    result['matches'] = response
    return result
"""

import pandas as pd
@router.post("/head_to_head")
def get_nfl_head_to_head_data(request: NFLTeams):
    file_path = os.path.join(NFL_DIR, 'nfl_games_data_history.csv')
    df = pd.read_csv(file_path)

    filtered_df = df[((df['home_team_name'] == request.hometeam) & (df['away_team_name'] == request.awayteam)) |
                     ((df['home_team_name'] == request.awayteam) & (df['away_team_name'] == request.hometeam))]

    result = {
            "wins": 0,
            "losses": 0,
            "draws": 0
        }
    
    matches = []
    filtered_df = filtered_df.tail(5)
    for index, row in filtered_df.iloc[::-1].iterrows():
        if row['home_team_name'] == request.hometeam:
            if row['home_total_score'] > row['away_total_score']:
                result["wins"] += 1
            elif row['home_total_score'] < row['away_total_score']:
                result["losses"] += 1
            else:
                result["draws"] += 1
        else:
            if row['away_total_score'] > row['home_total_score']:
                result["wins"] += 1
            elif row['away_total_score'] < row['home_total_score']:
                result["losses"] += 1
            else:
                result["draws"] += 1
        match_info = {
            "date": row['date'],
            "home_team": row['home_team_name'],
            "away_team": row['away_team_name'],
            "home_score": row['home_total_score'],
            "away_score": row['away_total_score'],
        }
        matches.append(match_info)
    
    result['matches'] = matches
    return result
