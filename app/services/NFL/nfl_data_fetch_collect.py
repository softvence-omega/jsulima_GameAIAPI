import os
import joblib
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_squared_error
from dotenv import load_dotenv
from app.services.NFL.data_processor import NflDataProcessor
from app import config

# Load API Key from .env
load_dotenv()
api_key = os.getenv("GOALSERVE_API_KEY")

# Create Processor Instance
processor = NflDataProcessor(api_key)

# ----------------------------
# Fetch Historical Data
# ----------------------------
def fetch_historical_nfl_data_by_year(start_year: int, end_year: int = None) -> pd.DataFrame:
    """
    Fetch NFL data for each day from Jan 1 of start_year to Dec 31 of end_year.
    If end_year is None, fetch for only start_year.
    Saves the data CSV with the year(s) in filename.
    """
    if end_year is None:
        end_year = start_year

    start_date = datetime.strptime(f"01.01.{start_year}", "%d.%m.%Y")
    end_date = datetime.strptime(f"28.02.{end_year}", "%d.%m.%Y")

    historical_features = []

    for single_date in pd.date_range(start_date, end_date):
        date_str = single_date.strftime("%d.%m.%Y")
        try:
            raw_data = processor.fetch_data(f"football/nfl-scores?date={date_str}")
            matches = raw_data.get('scores', {}).get('category', {})
            if isinstance(matches, dict):
                matches = [matches]

            for category in matches:
                games = category.get('match', [])
                if isinstance(games, dict):
                    games = [games]

                for i, game in enumerate(games):
                    #printf"Fetching match {i} on {date_str}")
                    #printf"Game type: {type(game)}, Game content: {game}")
                    if not isinstance(game, dict):
                        #printf"⚠️ Skipping non-dict game object at index {i}: {game}")
                        continue
                    features = processor.extract_predictive_features(game)
                
                    if features:
                        home_score = int(game.get('hometeam', {}).get('@totalscore', 0))
                        away_score = int(game.get('awayteam', {}).get('@totalscore', 0))
                        features['home_score'] = home_score
                        features['away_score'] = away_score
                        features['home_win'] = 1 if home_score > away_score else 0
                        historical_features.append(features)
        except Exception as e:
            #printf"Error on {date_str}: {e}")
            continue

    df = pd.DataFrame(historical_features)

    if not df.empty and 'home_win' not in df.columns:
        df['home_win'] = (df['home_score'] > df['away_score']).astype(int)

    #print"✅ DataFrame shape:", df.shape)
    #printdf.head())

    filename = f"team_historical_data_{start_year}_to_{end_year}.csv" if start_year != end_year else f"team_historical_data_{start_year}.csv"
    csv_path = os.path.join(config.NFL_DIR, filename)
    df.to_csv(csv_path, index=False)
    #printf"✅ Saved CSV to: {csv_path}")

    return df


def train_and_evaluate_model(df: pd.DataFrame):
    if df.empty:
        #print"❌ Empty dataset. Training aborted.")
        return

    # Targets
    target_class = 'home_win'
    target_home_score = 'home_score'
    target_away_score = 'away_score'

    # Features
    feature_cols = [
        col for col in df.columns
        if col not in [target_class, target_home_score, target_away_score]
        and df[col].dtype in [int, float]
    ]
    
    model_dir = os.path.join(config.NFL_DIR, "models")
    os.makedirs(model_dir, exist_ok=True)

    # Save feature columns for later use
    joblib.dump(feature_cols, os.path.join(model_dir, 'model_features.pkl'))
    #printf"✅ Saved feature columns to: {os.path.join(model_dir, 'model_features.pkl')}")
    X = df[feature_cols]
    y_class = df[target_class]
    y_home = df[target_home_score]
    y_away = df[target_away_score]

    # Train/test split
    X_train, X_test, y_class_train, y_class_test = train_test_split(X, y_class, test_size=0.2, random_state=42)
    _, _, y_home_train, y_home_test = train_test_split(X, y_home, test_size=0.2, random_state=42)
    _, _, y_away_train, y_away_test = train_test_split(X, y_away, test_size=0.2, random_state=42)

    # Models
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    reg_home = RandomForestRegressor(n_estimators=100, random_state=42)
    reg_away = RandomForestRegressor(n_estimators=100, random_state=42)

    # Train models
    clf.fit(X_train, y_class_train)
    reg_home.fit(X_train, y_home_train)
    reg_away.fit(X_train, y_away_train)

    # Evaluate
    acc = accuracy_score(y_class_test, clf.predict(X_test))
    mse_home = mean_squared_error(y_home_test, reg_home.predict(X_test))
    mse_away = mean_squared_error(y_away_test, reg_away.predict(X_test))

    #printf"[🏈 home_win] Accuracy: {acc:.4f}")
    #printf"[🏠 home_score] MSE: {mse_home:.2f}")
    #printf"[🚗 away_score] MSE: {mse_away:.2f}")

    # Save models
    model_dir = os.path.join(config.NFL_DIR, "models")
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(clf, os.path.join(model_dir, 'model_home_win.pkl'))
    joblib.dump(reg_home, os.path.join(model_dir, 'model_home_score.pkl'))
    joblib.dump(reg_away, os.path.join(model_dir, 'model_away_score.pkl'))

    #printf"✅ Models saved to: {model_dir}")
    return clf, reg_home, reg_away

# ----------------------------
# Run Everything
# ----------------------------

if __name__ == "__main__":
    # Fetch data for full 2023
    df = fetch_historical_nfl_data_by_year(2025,2025)
    # Train and evaluate model
    #train_and_evaluate_model(df)
    
    # Or fetch for multiple years e.g., 2021 to 2023
    # df = fetch_historical_nfl_data_by_year(2021, 2023)

    #printdf.head())
