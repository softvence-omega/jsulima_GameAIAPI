
import os
from app.schemas.nfl_schemas import NFLTeams
from app.services.NFL.stat_head_to_head import HeadToHeadRecord

from app.config import GOALSERVE_API_KEY
from fastapi import APIRouter




API_KEY=GOALSERVE_API_KEY

router=APIRouter()
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



