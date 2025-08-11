from fastapi import FastAPI
import os
import uvicorn

from app.api.v1.endpoints import (
    mlb_prediction_lineup,
    nfl_lineup_prediction,
    nfl_top_performer_predictor,
    nfl_win_predict,
    nfl_head_to_head,
    mlb_head_to_head,
    mlb_batter_top_performer,
    mlb_games_prediction_endpoint,
    mlb_top_performer_predictor,
    mlb_pitcher_top_performer,
    mlb_batter_pitcher_top_performer,
    mlb_win_percentage_endpoint,
    nfl_win_percentage_endpoint

)
from app.scheduler import start_scheduler


app = FastAPI(title="GameAPI")
start_scheduler()

# MLB
app.include_router(mlb_games_prediction_endpoint.router, prefix="/predict/mlb", tags=["MLB"])
#app.include_router(mlb_top_performer_predictor.router, prefix="/predict/mlb/top-performers", tags=["MLB"])
app.include_router(mlb_prediction_lineup.router, prefix="/predict/mlb", tags=["MLB"])
app.include_router(mlb_head_to_head.router, prefix="/predict/mlb", tags=["MLB"])
# app.include_router(mlb_batter_top_performer.router, prefix="/predict/mlb",tags=["MLB"])
# app.include_router(mlb_pitcher_top_performer.router, prefix="/predict/mlb", tags=["MLB"])
app.include_router(mlb_batter_pitcher_top_performer.router, prefix="/predict/mlb", tags=["MLB"])
app.include_router(mlb_win_percentage_endpoint.router, prefix="/predict/mlb", tags=["MLB"])

# NFL
app.include_router(nfl_win_predict.router, prefix="/predict/nfl", tags=["NFL"])
app.include_router(nfl_top_performer_predictor.router, prefix="/predict/nfl", tags=["NFL"])
app.include_router(nfl_lineup_prediction.router, prefix="/predict/nfl", tags=["NFL"])
app.include_router(nfl_head_to_head.router, prefix="/predict/nfl", tags=["NFL"])
app.include_router(nfl_win_percentage_endpoint.router, prefix="/predict/nfl", tags=["NFL"])

@app.get("/")
async def root():
    return {"message": "Welcome to the GameAPI! Use the /docs endpoint to explore available endpoints..."}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app.main:app", host="0.0.0.0", port=port, reload=False)
