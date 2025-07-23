from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from typing import Dict, Any

from app.services.NFL.top_performer4 import get_top_performers
from app.schemas.nfl_schemas import NFLTeams

router = APIRouter()

def remove_zeros(d: Dict[str, Any]) -> Dict[str, Any]:
    if isinstance(d, dict):
        # Create a new dictionary with keys that don't have value 0
        return {k: remove_zeros(v) for k, v in d.items() if v != 0}
    elif isinstance(d, list):
        # Recursively remove zeros from all elements in the list
        return [remove_zeros(item) for item in d]
    else:
        return d

@router.post("/top-performer")
async def nfl_top_performer(request: NFLTeams):
    try:
        result = get_top_performers(request.hometeam, request.awayteam)
        result = remove_zeros(result)
        if not result:
            raise HTTPException(
                status_code=404, 
                detail=f"No data found for teams: {request.hometeam} vs {request.awayteam}"
            )
        return JSONResponse(
            content= {**result}, 
            status_code=200
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Internal errorr: {str(e)}"
        )
