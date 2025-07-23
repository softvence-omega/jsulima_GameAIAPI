from fastapi import APIRouter
from app.services.MLB.mlb_prediction_lineup import MLBLineupPredictionService
from app.services.MLB.lineup_image_generator import LineupImageGenerator
from app.schemas.mlb_schemas import MLBSingleTeam
from fastapi.responses import JSONResponse
from urllib.parse import unquote
import pandas as pd

router = APIRouter()


@router.post("/lineup")
def predict_lineup(request: MLBSingleTeam):
    """
    Predicts the starting lineup for a given MLB team and returns both the
    raw data as JSON and a stylized lineup card image.
    - **lineup**: The predicted player lineup with statistics.
    - **image_base64**: A base64-encoded PNG image of the lineup.
    """
    lineup_data = _get_lineup_data(request.name)
    lineup_image = _generate_lineup_image(lineup_data)

    response_data = {
        "lineup": _prepare_lineup_for_json(lineup_data),
        "image_base64": lineup_image
    }
    return JSONResponse(content=response_data)


def _get_lineup_data(team_name: str):
    """Retrieve the predicted starting lineup for the specified team."""
    lineup_service = MLBLineupPredictionService(team_name=team_name)
    return lineup_service.get_starting_lineup()


def _generate_lineup_image(lineup_data):
    """Generate a base64-encoded image of the lineup."""
    image_generator = LineupImageGenerator(lineup_data)
    return image_generator.generate_image_base64()


def _prepare_lineup_for_json(lineup_data):
    """Clean lineup data for JSON serialization by replacing NaN values."""
    cleaned_data = lineup_data.replace({pd.NA: None, float('nan'): None})
    return cleaned_data.to_dict(orient="records")
