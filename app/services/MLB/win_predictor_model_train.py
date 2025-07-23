import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_absolute_error, accuracy_score
from sklearn.preprocessing import StandardScaler
import glob
import os
import joblib
import numpy as np

# --- Configuration ---
YEARS = range(2010, 2026)
DATA_PATH = 'app/data/MLB/'
TEST_YEAR = YEARS[-1]

# --- File Paths for Models and Scalers ---
# For Score Prediction (Regression)
HOME_SCORE_MODEL_PATH = 'models/mlb_home_score_model.joblib'
AWAY_SCORE_MODEL_PATH = 'models/mlb_away_score_model.joblib'
SCORE_SCALER_PATH = 'models/mlb_score_scaler.joblib'

# For Winner Prediction (Classification)
WINNER_MODEL_PATH = 'models/mlb_winner_model.joblib'
WINNER_SCALER_PATH = 'models/mlb_winner_scaler.joblib'


def load_yearly_data(years, file_pattern, data_path):
    """Loads and concatenates data files for a specific pattern over a range of years."""
    all_files = []
    print(f"Searching for files with pattern: {file_pattern.format(year='YYYY')}")
    for year in years:
        file_path = os.path.join(data_path, file_pattern.format(year=year))
        found_files = glob.glob(file_path)
        if not found_files:
            print(f"Warning: Could not find data for {file_pattern.format(year=year)}")
            continue
        df = pd.read_csv(found_files[0])
        df['year'] = year
        all_files.append(df)
    return pd.concat(all_files, ignore_index=True) if all_files else pd.DataFrame()


def engineer_features(games_df, batting_df, pitching_df):
    """Takes raw dataframes and engineers features for all models."""
    print("\n--- Step 2: Engineering features for all models ---")
    games_df['home_win'] = (games_df['home_score'] > games_df['away_score']).astype(int)
    games_df = games_df[['year', 'home_team', 'away_team', 'home_score', 'away_score', 'home_win']].dropna()

    # --- Team Stats Calculation (Batting & Pitching) ---
    batting_cols_to_numeric = ['at_bats', 'hits', 'home_runs', 'runs_batted_in']
    for col in batting_cols_to_numeric:
        batting_df[col] = pd.to_numeric(batting_df[col], errors='coerce')
    batting_df.dropna(subset=batting_cols_to_numeric, inplace=True)
    team_batting_stats = batting_df.groupby(['team', 'year']).agg(
        total_atBats=('at_bats', 'sum'), total_hits=('hits', 'sum'),
        total_homeruns=('home_runs', 'sum'), total_rbi=('runs_batted_in', 'sum')
    ).reset_index()
    team_batting_stats['batting_avg'] = team_batting_stats['total_hits'] / team_batting_stats['total_atBats']
    team_batting_stats.rename(columns={'team': 'team'}, inplace=True)

    pitching_cols_to_numeric = ['earned_runs', 'innings_pitched']
    for col in pitching_cols_to_numeric:
        pitching_df[col] = pd.to_numeric(pitching_df[col], errors='coerce')
    pitching_df.dropna(subset=pitching_cols_to_numeric, inplace=True)
    pitching_df = pitching_df[pitching_df['innings_pitched'] > 0]
    team_pitching_stats = pitching_df.groupby(['team', 'year']).agg(
        total_earned_runs=('earned_runs', 'sum'), total_innings_pitched=('innings_pitched', 'sum')
    ).reset_index()
    team_pitching_stats['era'] = (team_pitching_stats['total_earned_runs'] / team_pitching_stats['total_innings_pitched']) * 9
    team_pitching_stats.rename(columns={'team': 'team'}, inplace=True)

    team_stats_yearly = pd.merge(team_batting_stats, team_pitching_stats, on=['team', 'year'])

    # --- Merging and Final Feature Creation ---
    games_with_stats = pd.merge(games_df, team_stats_yearly, left_on=['home_team', 'year'], right_on=['team', 'year'], how='inner')
    games_with_stats = pd.merge(games_with_stats, team_stats_yearly, left_on=['away_team', 'year'], right_on=['team', 'year'], how='inner', suffixes=('_home', '_away'))

    # Features for winner prediction (differences)
    games_with_stats['diff_batting_avg'] = games_with_stats['batting_avg_home'] - games_with_stats['batting_avg_away']
    games_with_stats['diff_era'] = games_with_stats['era_home'] - games_with_stats['era_away']
    games_with_stats['diff_homeruns'] = games_with_stats['total_homeruns_home'] - games_with_stats['total_homeruns_away']
    games_with_stats['diff_rbi'] = games_with_stats['total_rbi_home'] - games_with_stats['total_rbi_away']

    return games_with_stats.dropna(), team_stats_yearly


def train_models(df, score_features, winner_features, test_year):
    """Trains, evaluates, and returns all models and scalers."""
    print("\n--- Step 3: Training and Evaluating All Models ---")
    train_df = df[df['year'] < test_year]
    test_df = df[df['year'] == test_year]

    if test_df.empty:
        print(f"Warning: No data for test year {test_year}. Evaluation skipped.")
        return {}, {}

    # --- Train Score Prediction Models (Regression) ---
    X_train_score, X_test_score = train_df[score_features], test_df[score_features]
    score_scaler = StandardScaler().fit(X_train_score)
    X_train_score_s, X_test_score_s = score_scaler.transform(X_train_score), score_scaler.transform(X_test_score)
    
    home_model = LinearRegression().fit(X_train_score_s, train_df['home_score'])
    away_model = LinearRegression().fit(X_train_score_s, train_df['away_score'])
    
    home_mae = mean_absolute_error(test_df['home_score'], home_model.predict(X_test_score_s))
    away_mae = mean_absolute_error(test_df['away_score'], away_model.predict(X_test_score_s))
    print(f"Home Score Model MAE: {home_mae:.2f} runs")
    print(f"Away Score Model MAE: {away_mae:.2f} runs")

    # --- Train Winner Prediction Model (Classification) ---
    X_train_winner, X_test_winner = train_df[winner_features], test_df[winner_features]
    winner_scaler = StandardScaler().fit(X_train_winner)
    X_train_winner_s, X_test_winner_s = winner_scaler.transform(X_train_winner), winner_scaler.transform(X_test_winner)

    winner_model = LogisticRegression(random_state=42).fit(X_train_winner_s, train_df['home_win'])
    accuracy = accuracy_score(test_df['home_win'], winner_model.predict(X_test_winner_s))
    print(f"Winner Model Accuracy: {accuracy:.2%}")

    models = {'home_score': home_model, 'away_score': away_model, 'winner': winner_model}
    scalers = {'score': score_scaler, 'winner': winner_scaler}
    
    return models, scalers


def save_all_models(models, scalers):
    """Saves all models and scalers to disk."""
    print("\n--- Step 4: Saving all models and scalers ---")
    joblib.dump(models['home_score'], HOME_SCORE_MODEL_PATH)
    joblib.dump(models['away_score'], AWAY_SCORE_MODEL_PATH)
    joblib.dump(models['winner'], WINNER_MODEL_PATH)
    joblib.dump(scalers['score'], SCORE_SCALER_PATH)
    joblib.dump(scalers['winner'], WINNER_SCALER_PATH)
    print("All models and scalers saved.")


def load_and_predict_outcome(home_team, away_team, year, team_stats_df, score_features, winner_features):
    """Loads all models to predict the score and win confidence of a game."""
    print("\n--- Step 5: Loading models for a complete prediction ---")
    try:
        # Load all necessary files
        home_score_model = joblib.load(HOME_SCORE_MODEL_PATH)
        away_score_model = joblib.load(AWAY_SCORE_MODEL_PATH)
        winner_model = joblib.load(WINNER_MODEL_PATH)
        score_scaler = joblib.load(SCORE_SCALER_PATH)
        winner_scaler = joblib.load(WINNER_SCALER_PATH)
        print("All models and scalers loaded successfully.")

        # Get team stats
        home_stats = team_stats_df[(team_stats_df['team'] == home_team) & (team_stats_df['year'] == year)].iloc[0]
        away_stats = team_stats_df[(team_stats_df['team'] == away_team) & (team_stats_df['year'] == year)].iloc[0]

        # --- Predict Score ---
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

        # --- Predict Winner Confidence ---
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

        # Determine confidence based on the predicted winner
        if pred_home_score > pred_away_score:
            confidence = home_win_prob
        elif pred_away_score > pred_home_score:
            confidence = away_win_prob
        else: # Tie
            confidence = max(home_win_prob, away_win_prob) # Confidence in the more likely winner

        # --- Format final output ---
        return {
            "home_win_probability": f"{home_win_prob:.2%}",
            "away_win_probability": f"{away_win_prob:.2%}",
            "confidence": f"{confidence:.2%}",
            "predicted_scores": {
                home_team.upper(): pred_home_score,
                away_team.upper(): pred_away_score
            }
        }
    except FileNotFoundError:
        return {"error": "A model or scaler file not found. Please run the training process first."}
    except IndexError:
        return {"error": f"Stats not found for one or both teams in {year}."}


def main():
    """Main function to run the full MLB prediction pipeline."""
    print("--- Step 1: Loading all data files ---")
    games_df = load_yearly_data(YEARS, 'games_data_final_{year}.csv', os.path.join(DATA_PATH, 'game_data/'))
    batting_df = load_yearly_data(YEARS, 'batting_{year}.csv', os.path.join(DATA_PATH, 'batting/'))
    pitching_df = load_yearly_data(YEARS, 'pitching_{year}.csv', os.path.join(DATA_PATH, 'pitching/'))

    if any(df.empty for df in [games_df, batting_df, pitching_df]):
        print("\nExecution stopped: Could not find or load the required multi-year data files.")
        return

    final_df, team_stats_yearly = engineer_features(games_df, batting_df, pitching_df)

    team_stats_yearly.to_csv(os.path.join(DATA_PATH, 'team_stats_yearly.csv'), index=False)


    # Define features for both model types
    score_features = ['batting_avg_home', 'era_home', 'total_homeruns_home', 'total_rbi_home',
                      'batting_avg_away', 'era_away', 'total_homeruns_away', 'total_rbi_away']
    winner_features = ['diff_batting_avg', 'diff_era', 'diff_homeruns', 'diff_rbi']
    
    models, scalers = train_models(final_df, score_features, winner_features, TEST_YEAR)

    if models and scalers:
        save_all_models(models, scalers)
        prediction = load_and_predict_outcome('Baltimore Orioles', 'Chicago White Sox', TEST_YEAR, team_stats_yearly, score_features, winner_features)
        print(f"\n--- Prediction Example ---\nPrediction for NYA vs. BOS based on {TEST_YEAR} stats: {prediction}")



if __name__ == "__main__":
    main()
