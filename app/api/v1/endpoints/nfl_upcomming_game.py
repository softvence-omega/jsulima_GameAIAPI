from fastapi import APIRouter, HTTPException, Response
from app.services.NFL.upcommingGame import Upcomming_nfl_game
import json

router = APIRouter()

@router.get("/upcoming-games")
def get_upcoming_nfl_games():
    try:
        upcoming_games = Upcomming_nfl_game().upcoming_games()
        games = json.loads(upcoming_games)

        # Clean them up
        cleaned_games = []
        for game in games:
            cleaned_game = {
                k: (" ".join(v.split()) if isinstance(v, str) else v)
                for k, v in game.items()
            }
            cleaned_games.append(cleaned_game)

        return Response(
            content=json.dumps(cleaned_games, separators=(',', ':')),
            media_type='application/json'
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching upcoming games: {str(e)}")
