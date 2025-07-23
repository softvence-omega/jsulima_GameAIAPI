import os
import pandas as pd
from joblib import load
from fastapi import APIRouter, HTTPException
from app.services.NFL.data_processor import NflDataProcessor
from app.config import NFL_DIR
from app.services.NFL.upcommingGame import Upcomming_nfl_game
from app.services.NFL.top_performer4 import get_top_performers

MODEL_DIR = os.path.join(NFL_DIR, "models")
HISTORICAL_CSV = os.path.join(NFL_DIR, "Game_data(historical).csv")

clf = load(os.path.join(MODEL_DIR, 'model_home_win.pkl'))
home_reg = load(os.path.join(MODEL_DIR, 'model_home_score.pkl'))
away_reg = load(os.path.join(MODEL_DIR, 'model_away_score.pkl'))
feature_cols = load(os.path.join(MODEL_DIR, 'model_features.pkl'))

upcoming_nfl_games = Upcomming_nfl_game()


def generate_player_prediction(top_performer: dict) -> str:
    name = top_performer["player_name"]
    team = top_performer["team_name"]
    position = top_performer["player_position"]

    passing_yards = top_performer.get("passing_yards_used", 0)
    passing_tds = top_performer.get("passing_touchdowns_used", 0)
    interceptions = top_performer.get("interceptions_used", 0)
    rushing_yards = top_performer.get("rushing_yards_used", 0)
    rushing_tds = top_performer.get("rushing_touchdowns_used", 0)

    prediction = f"{name} is predicted to "

    if position == "QB":
        prediction += f"throw for {passing_yards} yards, with {passing_tds} touchdowns"
        if interceptions:
            prediction += f" and {interceptions} interception"
        prediction += f" in the upcoming game for the {team}."
    elif position in ["RB", "WR"]:
        prediction += f"gain {rushing_yards} rushing yards and score {rushing_tds} rushing TDs for the {team}."
    else:
        prediction += f"contribute significantly to the {team}'s performance."

    return prediction


def generate_key_insight(match_data: dict, top_performers: list) -> str:
    home = match_data["home_team"]
    away = match_data["away_team"]
    win_prob = match_data.get("home_win_probability", 0)

    try:
        win_prob = float(win_prob)
    except (ValueError, TypeError):
        win_prob = 0

    predicted_scores = match_data["predicted_scores"]
    favorite = home if win_prob >= 50 else away
    underdog = away if favorite == home else home

    top_fav_performer = next((p for p in top_performers if p["team_name"] == favorite), None)

    insight = f"The {favorite} are favored in this matchup due to "
    if top_fav_performer:
        name = top_fav_performer["player_name"]
        position = top_fav_performer["player_position"]
        if position == "QB":
            insight += f"strong quarterback play from {name}"
        elif position == "RB":
            insight += f"versatile running by {name}"
        elif position == "WR":
            insight += f"explosive receiving from {name}"
        else:
            insight += f"key contributions from {name}"
        insight += f", but {underdog}'s defense will be a challenge."
    else:
        insight += f"their offensive edge, although {underdog} may prove tough defensively."

    insight += f" The {favorite}'s ability to execute plays efficiently is expected to give them an edge."
    return insight


def predict_win_probability(home_team, away_team, n_recent=5):
    chunks = pd.read_csv(HISTORICAL_CSV, chunksize=10000)
    relevant_games = []

    for chunk in chunks:
        mask = (
            ((chunk['home_team'] == home_team) & (chunk['away_team'] == away_team)) |
            ((chunk['home_team'] == away_team) & (chunk['away_team'] == home_team))
        )
        filtered = chunk[mask]
        if not filtered.empty:
            relevant_games.append(filtered)

    if not relevant_games:
        return {
            'home_team': home_team,
            'away_team': away_team,
            'home_win_probability': "",
            'away_win_probability': "",
            'predictor': f"No historical games found between {home_team} and {away_team}.",
            'home_confidence': "",
            'away_confidence': "",
            'predicted_scores': {home_team: "", away_team: ""}
        }

    df = pd.concat(relevant_games).sort_values('date', ascending=False).head(n_recent)
    latest_game = df.iloc[0].copy()
    latest_game['home_team'] = home_team
    latest_game['away_team'] = away_team

    X = pd.DataFrame([latest_game[feature_cols].values], columns=feature_cols)

    probs = clf.predict_proba(X)[0]
    home_win_prob = float(probs[1])
    away_win_prob = float(probs[0])
    confidence = round(abs(home_win_prob - 0.5) * 2 * 100, 1)
    away_confidence = round(100 - confidence, 1)

    home_score = int(round(home_reg.predict(X)[0]))
    away_score = int(round(away_reg.predict(X)[0]))

    return {
        'home_team': home_team,
        'away_team': away_team,
        'home_win_probability': round(home_win_prob * 100, 1),
        'away_win_probability': round(away_win_prob * 100, 1),
        'predictor': f"{home_team} has a {round(home_win_prob * 100, 1)}% chance of winning against {away_team}.",
        'home_confidence': confidence,
        'away_confidence': away_confidence,
        'predicted_scores': {home_team: home_score, away_team: away_score}
    }


def predict():
    upcoming_games = upcoming_nfl_games.upcoming_games()
    predictions = []

    for game in upcoming_games:
        home_team = game['hometeam']['@name']
        away_team = game['awayteam']['@name']
        try:
            prediction = predict_win_probability(home_team, away_team)
            prediction['info'] = game

            top_performers = get_top_performers(home_team, away_team)
            prediction['top_performers'] = top_performers

            scores = prediction['predicted_scores']
            favorite_team = max(scores, key=scores.get)
            top_fav = next((p for p in top_performers if p["team_name"] == favorite_team), None)

            prediction_text = generate_player_prediction(top_fav) if top_fav else None
            insight = generate_key_insight(prediction, top_performers)

            prediction["prediction_text"] = prediction_text
            prediction["insight"] = insight

            predictions.append(prediction)
        except Exception as e:
            print(f"⚠️ Error predicting match {home_team} vs {away_team}: {e}")

    return predictions


