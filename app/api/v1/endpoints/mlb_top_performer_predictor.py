
from app.services.MLB.mlb_performer_predictor import PerformerPredictor
import os
from dotenv import load_dotenv
import json
from fastapi import APIRouter

load_dotenv()
API_KEY=os.getenv('GOALSERVE_API_KEY')

router=APIRouter()

@router.get("mlb/top-performer-prediction")
async def top_performer_prediction_endpoint(limit: int = 5):
    predictor = PerformerPredictor()
    today_predictions = predictor.predict_todays_games()

    return today_predictions

