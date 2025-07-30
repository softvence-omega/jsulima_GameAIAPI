"""
Combined MLB Top Batter and Pitcher Performer Endpoint
This endpoint returns both the top batter and pitcher for home and away teams in a single response.
"""
# This file was auto-generated to merge batter and pitcher top performer endpoints.
from fastapi import APIRouter, HTTPException
from typing import Dict, Any
from app.services.MLB.mlb_batter_top_performer import get_model_path, load_and_prepare_data, train_model, predict_and_evaluate, predict_best_batsman
from app.services.MLB.mlb_pitcher_top_performer import PitcherPredictor
from app.schemas.mlb_schemas import MLBTeams
import joblib
import os


router = APIRouter()

# Load and prepare batter data/model
file_path = 'app/data/MLB/batting/batting_data_combined.csv' 
model_path = get_model_path()

if os.path.exists(model_path):
    #print(f"Loading model from {model_path}")
    model = joblib.load(model_path)
    # Still need to load data for label encoder and prediction
    df, le, X_train, X_test, y_train, y_test = load_and_prepare_data(file_path)
    mse, r2, confidence_interval = predict_and_evaluate(model, X_test, y_test)
else:
    #print(f"Training model and saving to {model_path}")
    df, le, X_train, X_test, y_train, y_test = load_and_prepare_data(file_path)
    model = train_model(X_train, y_train)
    mse, r2, confidence_interval = predict_and_evaluate(model, X_test, y_test)
    joblib.dump(model, model_path)

# df, le, X_train, X_test, y_train, y_test = load_and_prepare_data(file_path)
# best_batter_model = train_model(X_train, y_train)
# mse, r2, confidence_interval = predict_and_evaluate(best_batter_model, X_test, y_test)

# Load pitcher predictor
pitcher_predictor = PitcherPredictor()

def to_py(val):
    # Convert numpy/pandas types to Python native types
    if hasattr(val, 'item'):
        return val.item()
    if isinstance(val, (float, int, str)) or val is None:
        return val
    return type(val)(val)

def pitcher_to_py(pitcher_dict):
    if not pitcher_dict:
        return None
    return {k: to_py(v) for k, v in pitcher_dict.items()}

@router.post("/top_batter_pitcher", response_model=Dict[str, Any])
async def get_top_batter_pitcher(request: MLBTeams):
    # Top Batter Home
    try:
        best_batsman_home = predict_best_batsman(request.hometeam, df, model, le)
        best_batsman_home_dict = {
            'player_name': str(best_batsman_home['player_name']),
            'player_position': str(best_batsman_home['position']) if 'position' in best_batsman_home else None,
            'batting_average': float(best_batsman_home['batting_average']),
            'games_played': int(best_batsman_home.get('games_played')) if best_batsman_home.get('games_played') is not None else None,
            'runs': int(best_batsman_home['runs']),
            'hits': int(best_batsman_home['hits']),
            'performance_score': float(best_batsman_home['predicted_batting_average']),
            'confidence_score': float(r2)
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=f"Batter: {str(e)}")

    # Top Batter Away
    try:
        best_batsman_away = predict_best_batsman(request.awayteam,  df, model, le)
        best_batsman_away_dict = {
            'player_name': str(best_batsman_away['player_name']),
            'player_position': str(best_batsman_away['position']) if 'position' in best_batsman_away else None,
            'batting_average': float(best_batsman_away['batting_average']),
            'games_played': int(best_batsman_away.get('games_played')) if best_batsman_away.get('games_played') is not None else None,
            'runs': int(best_batsman_away['runs']),
            'hits': int(best_batsman_away['hits']),
            'performance_score': float(best_batsman_away['predicted_batting_average']),
            'confidence_score': float(r2)
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=f"Batter: {str(e)}")

    # Top Pitcher
    pitcher_result = pitcher_predictor.get_top_pitcher(request.hometeam, request.awayteam)
    if not pitcher_result:
        raise HTTPException(status_code=404, detail=f"No pitcher found for teams: {request.hometeam} and {request.awayteam}")

    home_pitcher = pitcher_to_py(pitcher_result.get('home_team_pitcher'))
    away_pitcher = pitcher_to_py(pitcher_result.get('away_team_pitcher'))

    # Structure response
    response = {
        'home_team': {
            'top_batter': best_batsman_home_dict,
            'top_pitcher': home_pitcher
        },
        'away_team': {
            'top_batter': best_batsman_away_dict,
            'top_pitcher': away_pitcher
        }
    }
    return response 