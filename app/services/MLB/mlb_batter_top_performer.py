import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import scipy.stats as stats
from sklearn.model_selection import train_test_split
import os
import joblib


# Load and prepare the dataset
def load_and_prepare_data(file_path: str):
    df = pd.read_csv(file_path)

    # Rename columns to match expected names
    df = df.rename(columns={
        'name': 'player_name',
        'id': 'player_id',
        'average': 'batting_average',
        'runs_batted_in': 'rbis'
    })

    # Ensure numeric columns are actually numeric
    numeric_cols = [
        'hits', 'doubles', 'triples', 'home_runs', 'at_bats', 'runs', 'rbis',
        'sac_fly', 'hit_by_pitch', 'walks', 'strikeouts', 'stolen_bases',
        'on_base_percentage', 'slugging_percentage', 'caught_stealing', 'batting_average'
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # Add missing columns with default values if not present
    if 'total_bases' not in df.columns:
        if set(['hits', 'doubles', 'triples', 'home_runs']).issubset(df.columns):
            df['total_bases'] = (
                df['hits'] + df['doubles'] + 2 * df['triples'] + 3 * df['home_runs']
            )
        else:
            df['total_bases'] = 0

    # Encode team using LabelEncoder
    le = LabelEncoder()
    df['team_encoded'] = le.fit_transform(df['team'])

    # Feature selection and preparation
    feature_cols = [
        'on_base_percentage', 'slugging_percentage', 'hits', 'total_bases', 'runs',
        'at_bats', 'rbis', 'doubles', 'home_runs', 'triples', 'stolen_bases', 'walks',
        'caught_stealing', 'sac_fly', 'hit_by_pitch', 'strikeouts',
        'team_encoded'
    ]
    X = df[feature_cols].copy()
    y = df['batting_average'].copy()

    # Fill any remaining NaNs in X and y
    X = X.fillna(0)
    y = y.fillna(0)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return df, le, X_train, X_test, y_train, y_test


# Train the model
def train_model(X_train, y_train):
    # Best Hyperparameters for RandomForestRegressor based on previous GridSearchCV
    best_params_rf = {
        'n_estimators': 200,
        'max_depth': 10,
        'min_samples_split': 2,
        'min_samples_leaf': 1,
        'random_state': 42
    }

    # Create and train the best model (RandomForestRegressor with the best hyperparameters)
    model = RandomForestRegressor(**best_params_rf)
    model.fit(X_train, y_train)

    return model


# Predict and get model metrics
def predict_and_evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)

    # Evaluate the model (R^2 and Mean Squared Error)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Confidence Interval for predictions
    residuals = y_test - y_pred
    std_error = np.std(residuals)

    confidence_level = 0.95
    degrees_freedom = len(y_test) - 1
    confidence_interval = stats.t.interval(confidence_level, degrees_freedom, loc=np.mean(y_pred), scale=std_error/np.sqrt(len(y_test)))

    return mse, r2, confidence_interval


# Predict the best player for the team
def predict_best_batsman(team_name: str, df, model, le):
    # Convert team_name to encoded value
    try:
        team_encoded = le.transform([team_name])[0]
    except ValueError:
        raise ValueError(f"Team {team_name} not found in the dataset")

    # Filter the data for the given team
    team_data = df[df['team'] == team_name].copy()

    if team_data.empty:
        raise ValueError(f"No data found for {team_name}")

    # Prepare the features for prediction (same features used for training)
    team_features = team_data[[
        'on_base_percentage', 'slugging_percentage', 'hits', 'total_bases', 'runs',
        'at_bats', 'rbis', 'doubles', 'home_runs', 'triples', 'stolen_bases', 'walks',
        'caught_stealing', 'sac_fly', 'hit_by_pitch', 'strikeouts'
    ]]
    team_features['team_encoded'] = team_encoded

    # Predict the batting average for all players in the team
    team_predictions = model.predict(team_features)

    # Add predictions to the team data
    team_data.loc[:, 'predicted_batting_average'] = team_predictions

    # Count the number of unique games played by each player
    team_data['games_played'] = team_data.groupby('player_name')['game_id'].transform('nunique')

    # Find the player with the highest predicted batting average
    best_batsman = team_data.loc[team_data['predicted_batting_average'].idxmax()]

    return best_batsman


def get_model_path():
    return os.path.join('models', 'MLB', 'mlb_top_batter.pkl')


if __name__ == "__main__":
    # Example usage
    file_path = 'app/data/MLB/batting/batting_data_combined.csv'  # Adjust the path as needed
    model_path = get_model_path()

    if os.path.exists(model_path):
        #print(f"Loading model from {model_path}")
        model = joblib.load(model_path)
        # Still need to load data for label encoder and prediction
        df, le, X_train, X_test, y_train, y_test = load_and_prepare_data(file_path)
    else:
        #print(f"Training model and saving to {model_path}")
        df, le, X_train, X_test, y_train, y_test = load_and_prepare_data(file_path)
        model = train_model(X_train, y_train)
        joblib.dump(model, model_path)

    mse, r2, confidence_interval = predict_and_evaluate(model, X_test, y_test)
    #print(f"Model MSE: {mse}, R^2: {r2}, Confidence Interval: {confidence_interval}")
    #print("Available teams:", df['team'].unique())
    valid_team = df['team'].unique()[0]
    #print(f"\nPredicting for team: {valid_team}")
    try:
        best_batsman = predict_best_batsman(valid_team, df, model, le)
        #print(best_batsman)
        #print(f"Best Batsman for {valid_team}: {best_batsman['player_name']} with predicted batting average {best_batsman['predicted_batting_average']}")
    except Exception as e:
        #print(f"Prediction failed: {e}")
        raise e
