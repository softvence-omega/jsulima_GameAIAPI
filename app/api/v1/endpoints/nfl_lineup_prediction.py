from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
import pandas as pd
from typing import Dict, Any
import requests
import os
from dotenv import load_dotenv

load_dotenv()

from app.services.NFL.lineup_prediction_service import NFLLineupPredictor
from app.services.NFL.lineup_image_generator import NFLTeamLineupImageGenerator
from app.schemas.nfl_schemas import NFLTeams

router = APIRouter()

# Constants
STARTING_LINEUP_SIZE = 11
NFL_LINEUP_MODEL_PATH = "models/nfl_lineup_model.pkl"
GOALSERVE_API_BASE_URL = "http://www.goalserve.com/getfeed"
GOALSERVE_API_KEY = os.getenv('GOALSERVE_API_KEY')
INJURIES_ENDPOINT_TEMPLATE = "football/{team_id}_injuries"


TEAM_NAME_TO_ID = {
    'Buffalo Bills': 1689,
    'Miami Dolphins': 1692,
    'NY Jets': 1709,
    'New England Patriots': 1681,
    'Baltimore Ravens': 1683,
    'Pittsburgh Steelers': 1694,
    'Cincinnati Bengals': 1679,
    'Cleveland Browns': 1699,
    'Houston Texans': 1697,
    'Indianapolis Colts': 1706,
    'Jacksonville Jaguars': 1687,
    'Tennessee Titans': 1705,
    'Kansas City Chiefs': 1691,
    'Los Angeles Chargers': 1702,
    'Denver Broncos': 1708,
    'Las Vegas Raiders': 5566,
    'Philadelphia Eagles': 1686,
    'Washington Commanders': 5753,
    'Dallas Cowboys': 1680,
    'NY Giants': 1710,
    'Detroit Lions': 1695,
    'Minnesota Vikings': 1701,
    'Green Bay Packers': 1698,
    'Chicago Bears': 1703,
    'Tampa Bay Buccaneers': 1693,
    'Atlanta Falcons': 1690,
    'Carolina Panthers': 1684,
    'New Orleans Saints': 1682,
    'Los Angeles Rams': 5117,
    'Seattle Seahawks': 1704,
    'Arizona Cardinals': 1696,
    'San Francisco 49ers': 1707
}


def initialize_predictor() -> NFLLineupPredictor:
    """Initialize and train the NFL lineup predictor."""
    predictor = NFLLineupPredictor()
    predictor.train("app/data/NFL/player_info.csv", "app/data/NFL/all_player_stats.csv", NFL_LINEUP_MODEL_PATH)
    return predictor


predictor = initialize_predictor()


def get_injured_players(team_id: int) -> list[int]:
    """
    Fetches the list of injured players for a given team.
    """
    endpoint = INJURIES_ENDPOINT_TEMPLATE.format(team_id=team_id)
    url = f"{GOALSERVE_API_BASE_URL}/{GOALSERVE_API_KEY}/{endpoint}?json=1"
    response = requests.get(url)
    response.raise_for_status()
    data = response.json()
    try:
        reports = data["team"]["report"]
        return [player["player_id"] for player in reports]
    except KeyError as e:
        raise KeyError(f"Unexpected API response structure: missing key {e}")


def prepare_team_probabilities(team_id: int) -> pd.DataFrame:
    """
    Get team lineup probabilities and exclude injured players.
    """
    probs = predictor.predict_lineup(team_id)
    probs = probs.fillna(0)
    probs['player_id'] = probs['player_id'].astype('int64').astype('str')

    injured_players = get_injured_players(team_id)
    probs = probs[~probs["player_id"].isin(injured_players)]

    return probs


def get_top_starters(probs: pd.DataFrame) -> pd.DataFrame:
    """
    Select and return top starting lineup players.
    """
    starters = predictor.select_starting_lineup(probs)
    top_starters = (
        starters.nlargest(STARTING_LINEUP_SIZE, "starter_probability")
        .reset_index(drop=True)
        .fillna(0)
    )
    return top_starters


def create_lineup_response(home_top_11: pd.DataFrame, away_top_11: pd.DataFrame) -> Dict[str, Any]:
    """
    Create the final response payload with player data and lineup image.
    """
    image_generator = NFLTeamLineupImageGenerator(home_top_11, away_top_11)
    lineup_image_base64 = image_generator.generate_image_base64()

    return {
        "home_players": [row.to_dict() for _, row in home_top_11.iterrows()],
        "away_players": [row.to_dict() for _, row in away_top_11.iterrows()],
        "lineup_image": lineup_image_base64,
    }


@router.post("/lineup")
def nfl_lineup(request: NFLTeams) -> JSONResponse:
    """
    Generates the predicted starting lineup for two NFL teams.
    """
    try:
        home_team_id = TEAM_NAME_TO_ID[request.hometeam]
        away_team_id = TEAM_NAME_TO_ID[request.awayteam]

        home_probs = prepare_team_probabilities(home_team_id)
        away_probs = prepare_team_probabilities(away_team_id)

        home_top_11 = get_top_starters(home_probs)
        away_top_11 = get_top_starters(away_probs)

        payload = create_lineup_response(home_top_11, away_top_11)

        return JSONResponse(content=payload, status_code=200)
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {e}")
