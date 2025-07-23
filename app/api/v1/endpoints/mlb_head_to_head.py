from fastapi import APIRouter, Form
import os
from dotenv import load_dotenv
import json

from app.services.MLB.head_to_head import fetch_head_to_head_teams
from app.schemas.mlb_schemas import MLBTeams

load_dotenv()
API_KEY=os.getenv('GOALSERVE_API_KEY')

router=APIRouter()

@router.post("/head_to_head")
async def get_mlb_head_to_head_data(request: MLBTeams):
    responsee = fetch_head_to_head_teams(request.hometeam, request.awayteam)
    
    result = {
        "wins": 0,
        "losses": 0,
        "draws": 0
    }

    for match in responsee:
        if match['home_team'] != request.hometeam:
            match['home_team'], match['away_team'] = match['away_team'], match['home_team']
            match['home_score'], match['away_score'] = match['away_score'], match['home_score']
        
        if match['home_score'] > match['away_score']:
            result["wins"] += 1
        elif match['home_score'] < match['away_score']:
            result["losses"] += 1
        else:
            result["draws"] += 1
    result['matches'] = responsee

    return result