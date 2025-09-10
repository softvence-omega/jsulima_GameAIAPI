from fastapi import APIRouter, HTTPException
from app.services.NFL.team_win_Predictor import predict
from app.services.NFL.upcommingGame import Upcomming_nfl_game
# from app.services.NFL.head_to_head_predictor import predict_head_to_head_win_probability
from app.services.NFL.upcommingGame import Upcomming_nfl_game


from app.config import NFL_DIR
import os
import pandas as pd 
import random

upcoming_nfl_games = Upcomming_nfl_game()
router = APIRouter()

from datetime import datetime 



@router.post("/head-to-head-win-prediction")
async def head_to_head_win(n : int = 10):
    upcoming_games = upcoming_nfl_games.upcoming_games()
    
    today = datetime.today().date()

    # Filter the list
    filtered_data = [
        game for game in upcoming_games
        if datetime.strptime(game['@formatted_date'], '%d.%m.%Y').date() >= today
    ]
    if n > 0:
        n = min(n, len(filtered_data))
        filtered_data = filtered_data[:n]

    upcoming_games_pred = []
    for game in filtered_data:

        home_team_name = game['hometeam']['@name']
        away_team_name = game['awayteam']['@name']

        res = get_prediction(home_team_name, away_team_name) #predict_head_to_head_win_probability(home_team_name, away_team_name)
        res['info'] = game 
        upcoming_games_pred.append(res)
    return upcoming_games_pred


def get_prediction(home_team: str, away_team: str):
    file_path = os.path.join(NFL_DIR, 'nfl_games_data_history.csv')
    df = pd.read_csv(file_path)

    filtered_df = df[((df['home_team_name'] == home_team) & (df['away_team_name'] == away_team)) |
                     ((df['home_team_name'] == away_team) & (df['away_team_name'] == home_team))]
    
    result = {
            "wins": 0,
            "losses": 0,
            "draws": 0
        }
    filtered_df = filtered_df.tail(15)
    for index, row in filtered_df.iloc[::-1].iterrows():
        if row['home_team_name'] == home_team:
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

    total_games = len(filtered_df)
    home_win_prob = round((result["wins"] / max(0.1, total_games) * 100), 1) 
    away_win_prob = round(100-home_win_prob, 1)

    if home_win_prob >= away_win_prob:
        home_win_prob = int(home_win_prob * 0.9)
        away_win_prob = 100 - home_win_prob
    elif away_win_prob > home_win_prob:
        away_win_prob = int(away_win_prob * 0.9)
        home_win_prob = 100 - away_win_prob

    # Option 1: Random between 60–85
    confidence = random.randint(60, 85)

    home_scores = filtered_df.apply(
        lambda row: row['home_total_score'] if row['home_team_name'] == home_team else row['away_total_score']
        if row['away_team_name'] == home_team else None, axis=1
    ).dropna()

    away_scores = filtered_df.apply(
        lambda row: row['home_total_score'] if row['home_team_name'] == away_team else row['away_total_score']
        if row['away_team_name'] == away_team else None, axis=1
    ).dropna()

    predicted_scores = {
        home_team: int(home_scores.median()),
        away_team: int(away_scores.median())
    }


  
    return {
        "home_team": home_team,
        "away_team": away_team,
        "home_win_probability": home_win_prob,
        "away_win_probability": away_win_prob,
        "confidence": confidence,
        "predicted_scores": predicted_scores
    }
