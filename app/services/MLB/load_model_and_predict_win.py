import pandas as pd
import joblib
import os


HOME_SCORE_MODEL_PATH = 'models/mlb_home_score_model.joblib'
AWAY_SCORE_MODEL_PATH = 'models/mlb_away_score_model.joblib'
SCORE_SCALER_PATH = 'models/mlb_score_scaler.joblib'

# For Winner Prediction (Classification)
WINNER_MODEL_PATH = 'models/mlb_winner_model.joblib'
WINNER_SCALER_PATH = 'models/mlb_winner_scaler.joblib'

try:
            # Load all necessary files
    home_score_model = joblib.load(HOME_SCORE_MODEL_PATH)
    away_score_model = joblib.load(AWAY_SCORE_MODEL_PATH)
    winner_model = joblib.load(WINNER_MODEL_PATH)
    score_scaler = joblib.load(SCORE_SCALER_PATH)
    winner_scaler = joblib.load(WINNER_SCALER_PATH)
    ##print("All models and scalers loaded successfully.")
except FileNotFoundError as e:
    raise {"error": f"A required file was not found: {e}. Please run the main training script first."}

# Path to the pre-computed team stats data
# This file should be saved after running the main training script.
TEAM_STATS_PATH = 'app/data/MLB/team_stats_yearly.csv'


def load_and_predict_outcome(home_team, away_team, year, team_stats_df, score_features, winner_features):
    """
    Loads all pre-trained models to predict the score and win confidence of a game.

    Args:
        home_team (str): The name/abbreviation of the home team.
        away_team (str): The name/abbreviation of the away team.
        year (int): The year of the game to predict.
        team_stats_df (pd.DataFrame): DataFrame containing the aggregated yearly stats for all teams.
        score_features (list): List of feature names for the score models.
        winner_features (list): List of feature names for the winner model.

    Returns:
        dict: A dictionary containing the full prediction details.
    """
    ##print(f"--- Predicting outcome for {home_team} vs. {away_team} for the {year} season ---")
    try:


        # Get team stats for the specified year
        home_stats = team_stats_df[(team_stats_df['team'] == home_team) & (team_stats_df['year'] == year)].iloc[0]
        away_stats = team_stats_df[(team_stats_df['team'] == away_team) & (team_stats_df['year'] == year)].iloc[0]

        # --- 1. Predict Score ---
        score_feature_dict = {
            'batting_avg_home': home_stats['batting_avg'], 'era_home': home_stats['era'],
            'total_homeruns_home': home_stats['total_homeruns'], 'total_rbi_home': home_stats['total_rbi'],
            'batting_avg_away': away_stats['batting_avg'], 'era_away': away_stats['era'],
            'total_homeruns_away': away_stats['total_homeruns'], 'total_rbi_away': away_stats['total_rbi']
        }
        score_input = pd.DataFrame([score_feature_dict], columns=score_features)
        score_input_s = score_scaler.transform(score_input)
        pred_home_score = round(home_score_model.predict(score_input_s)[0])
        pred_away_score = round(away_score_model.predict(score_input_s)[0])

        # --- 2. Predict Winner Confidence ---
        winner_feature_dict = {
            'diff_batting_avg': home_stats['batting_avg'] - away_stats['batting_avg'],
            'diff_era': home_stats['era'] - away_stats['era'],
            'diff_homeruns': home_stats['total_homeruns'] - away_stats['total_homeruns'],
            'diff_rbi': home_stats['total_rbi'] - away_stats['total_rbi']
        }
        winner_input = pd.DataFrame([winner_feature_dict], columns=winner_features)
        winner_input_s = winner_scaler.transform(winner_input)
        win_probs = winner_model.predict_proba(winner_input_s)[0]
        
        # Get win probabilities for each team
        away_win_prob = win_probs[0]
        home_win_prob = win_probs[1]

        # Determine confidence based on the predicted winner from the score
        if pred_home_score > pred_away_score:
            confidence = home_win_prob
        elif pred_away_score > pred_home_score:
            confidence = away_win_prob
        else: # Tie
            confidence = max(home_win_prob, away_win_prob)

        # --- 3. Format final output ---
        return {
            "home_win_probability": f"{home_win_prob:.2%}",
            "away_win_probability": f"{away_win_prob:.2%}",
            "confidence": f"{confidence:.2%}",
            "predicted_scores": {
                home_team.upper(): pred_home_score,
                away_team.upper(): pred_away_score
            }
        }
    except FileNotFoundError as e:
        return {"error": f"A required file was not found: {e}. Please run the main training script first."}
    except IndexError:
        return {"error": f"Stats not found for one or both teams in {year}. Ensure team names are correct and data exists."}
    except Exception as e:
        return {"error": f"An unexpected error occurred: {e}"}


def win_prediction(home_team, away_team, prediction_year):
    """
    Main function to load models and make a specific prediction.
    """
    # Define the features required by the models. This must match the training script.
    score_features = ['batting_avg_home', 'era_home', 'total_homeruns_home', 'total_rbi_home',
                      'batting_avg_away', 'era_away', 'total_homeruns_away', 'total_rbi_away']
    winner_features = ['diff_batting_avg', 'diff_era', 'diff_homeruns', 'diff_rbi']
    
    # Load the pre-computed yearly team stats
    if not os.path.exists(TEAM_STATS_PATH):
        #print(f"Error: Required data file not found at '{TEAM_STATS_PATH}'")
        #print("Please run the main training script to generate and save this file.")
        return
        
    team_stats_yearly = pd.read_csv(TEAM_STATS_PATH)

    # Make the prediction
    prediction = load_and_predict_outcome(
        home_team=home_team,
        away_team=away_team,
        year=prediction_year,
        team_stats_df=team_stats_yearly,
        score_features=score_features,
        winner_features=winner_features
    )
    
    return prediction

