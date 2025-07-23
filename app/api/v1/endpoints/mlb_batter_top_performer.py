from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from app.services.MLB.mlb_batter_top_performer import load_and_prepare_data, train_model, predict_and_evaluate, predict_best_batsman
from app.schemas.mlb_schemas import MLBTeams
from fastapi import APIRouter
from typing import List, Optional

# Initialize FastAPI app
# app = FastAPI()
router = APIRouter()

# Load and prepare the data
file_path = 'app/data/MLB/batter_stats_data(2010-2024).csv'
df, le, X_train, X_test, y_train, y_test = load_and_prepare_data(file_path)

# Train the model
best_model = train_model(X_train, y_train)

# Evaluate the model
mse, r2, confidence_interval = predict_and_evaluate(best_model, X_test, y_test)

# FastAPI response model for a single team's prediction
class PredictionResponse(BaseModel):
    team_name: str
    player_name: str
    player_position: str
    batting_average: float
    games_played: Optional[int] = None  # Made optional due to commented-out line
    performance_score: float
    runs: float
    hits: float
    confidence_score: float

# FastAPI route for predicting best batsman for both team_name and away_team
@router.post("/top_batsman", response_model=List[PredictionResponse])
async def get_best_batsman(request: MLBTeams):
    if request.hometeam not in df['team'].values:
        raise HTTPException(status_code=404, detail=f"Team not found: {request.hometeam}")
    if request.awayteam not in df['opponent_team'].values:
        raise HTTPException(status_code=404, detail=f"Away team not found: {request.awayteam}")

    # Predict the best batsman for the home team against the away team
    try:
        best_batsman_home = predict_best_batsman(request.hometeam, request.awayteam, df, best_model, le)
        best_batsman_home['performance_score'] = best_batsman_home['predicted_batting_average']
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    # Predict the best batsman for the away team against the home team
    try:
        best_batsman_away = predict_best_batsman(request.awayteam, request.awayteam, df, best_model, le)
        best_batsman_away['performance_score'] = best_batsman_away['predicted_batting_average']
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    # Confidence score (using R² from model evaluation)
    confidence_score = r2

    # Prepare the response for both teams
    result = [
        {
            'team_name': request.hometeam,
            'player_name': best_batsman_home['player_name'],
            'player_position': best_batsman_home['position'],
            'batting_average': best_batsman_home['batting_average'],
            'games_played': best_batsman_home.get('games_played'),  # Use .get() to handle missing key
            'runs': best_batsman_home['runs'],
            'hits': best_batsman_home['hits'],
            'performance_score': best_batsman_home['performance_score'],
            'confidence_score': confidence_score
        },
        {
            'team_name': request.awayteam,
            'player_name': best_batsman_away['player_name'],
            'player_position': best_batsman_away['position'],
            'batting_average': best_batsman_away['batting_average'],
            'games_played': best_batsman_away.get('games_played'),  # Use .get() to handle missing key
            'runs': best_batsman_away['runs'],
            'hits': best_batsman_away['hits'],
            'performance_score': best_batsman_away['performance_score'],
            'confidence_score': confidence_score
        }
    ]

    return result
