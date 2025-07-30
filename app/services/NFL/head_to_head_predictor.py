import os
import pandas as pd
from joblib import load
from app.services.NFL.data_processor import NflDataProcessor
from app.config import NFL_DIR

MODEL_DIR = os.path.join(NFL_DIR, "models")
HISTORICAL_CSV = os.path.join(NFL_DIR, "Game_data(historical).csv")

# Load models and features
clf = load(os.path.join(MODEL_DIR, 'model_home_win.pkl'))
home_reg = load(os.path.join(MODEL_DIR, 'model_home_score.pkl'))
away_reg = load(os.path.join(MODEL_DIR, 'model_away_score.pkl'))
feature_cols = load(os.path.join(MODEL_DIR, 'model_features.pkl'))



def predict_head_to_head_win_probability(home_team, away_team, n_recent=5):
    """
    Predict win probabilities and confidence for a matchup between two teams.
    Returns: dict with home_win_probability, away_win_probability, confidence, predicted_scores
    """
    # Efficiently load only relevant rows from the large CSV
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
        'confidence': "",
        'predicted_scores': {home_team: "", away_team: ""}
    }
        raise ValueError(f"No historical games found between {home_team} and {away_team}.")
    df = pd.concat(relevant_games).sort_values('date', ascending=False).head(n_recent)

    # Use the most recent game as a template for features
    latest_game = df.iloc[0].copy()
    # Set the teams as desired
    latest_game['home_team'] = home_team
    latest_game['away_team'] = away_team


    # Prepare feature vector
    X = latest_game[feature_cols].to_numpy().reshape(1, -1)

    # Predict probabilities
    probs = clf.predict_proba(X)[0]

    home_win_prob = float(probs[1])
    away_win_prob = float(probs[0])
    confidence = round(abs(home_win_prob - 0.5) * 2 * 100, 1)
    
    # Predict scores
    home_score = int(round(home_reg.predict(X)[0]))
    away_score = int(round(away_reg.predict(X)[0]))
    
    return {
        'home_team': home_team,
        'away_team': away_team,
        'home_win_probability': round(home_win_prob * 100, 1),
        'away_win_probability': round(away_win_prob * 100, 1),
        'confidence': confidence,
        'predicted_scores': {home_team: home_score, away_team: away_score}
    }

if __name__ == "__main__":
    # Example usage
    home = input("Enter home team name: ")
    away = input("Enter away team name: ")
    try:
        result = predict_head_to_head_win_probability(home, away)
        #print"\nPrediction Result:")
        for k, v in result.items():
            #printf"{k}: {v}")
            pass 
    except Exception as e:
        #printf"Error: {e}") 
        raise ValueError(f"Failed to predict matchup: {e}") from e