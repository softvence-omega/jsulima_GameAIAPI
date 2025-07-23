from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from app.services.MLB.mlb_pitcher_top_performer import PitcherPredictor
from app.schemas.mlb_schemas import MLBTeams
from fastapi import APIRouter

# app = FastAPI(title="Top Pitcher Predictor API")
router = APIRouter()

# Load predictor once
predictor = PitcherPredictor()


@router.post("/top-pitcher")
def get_top_pitcher(request: MLBTeams):
    
    result = predictor.get_top_pitcher(request.hometeam, request.awayteam)
    if not result:
        raise HTTPException(status_code=404, detail=f"No pitcher found for team: {request.hometeam} and {request.awayteam}")
    return result
