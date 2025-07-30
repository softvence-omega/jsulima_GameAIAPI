from fastapi import APIRouter, HTTPException
from app.services.NFL.team_win_Predictor import predict
from app.services.NFL.upcommingGame import Upcomming_nfl_game
from app.services.NFL.head_to_head_predictor import predict_head_to_head_win_probability
from app.services.NFL.upcommingGame import Upcomming_nfl_game


upcoming_nfl_games = Upcomming_nfl_game()
router = APIRouter()

# @router.get("/nfl/win-prediction")
# def head_to_head_win():
#     try:
#         results = predict()
#         return {"status": "success", "results": results}
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))


@router.post("/head-to-head-win-prediction")
def head_to_head_win(n : int = 0):
    upcoming_games = upcoming_nfl_games.upcoming_games()
    if n > 0:
        n = min(n, len(upcoming_games))
        upcoming_games = upcoming_games[:n]
        
    upcoming_games_pred = []
    for game in upcoming_games:

        home_team_name = game['hometeam']['@name']
        away_team_name = game['awayteam']['@name']

        res = predict_head_to_head_win_probability(home_team_name, away_team_name)
        res['info'] = game 
        upcoming_games_pred.append(res)
    return upcoming_games_pred