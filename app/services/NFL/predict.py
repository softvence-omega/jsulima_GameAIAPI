import os
import pandas as pd
from dotenv import load_dotenv
from joblib import load
from app.services.NFL.data_processor import NflDataProcessor
from app.services.NFL.upcommingGame import Upcomming_nfl_game
from app.services.helper import safe_float
from app.config import NFL_MODEL_DIR
from app import config

# Load environment variables
load_dotenv()

class LiveGamePredictor:
    def __init__(self, model_dir=os.path.join(config.NFL_DIR, "models")):
        self.api_key = os.getenv('GOALSERVE_API_KEY')
        self.data_processor = NflDataProcessor(self.api_key)
        self.upcomming_game = Upcomming_nfl_game()
        self.classifier = load(os.path.join(model_dir, 'model_home_win.pkl'))

        # Load trained models with file existence checks
        model_files = {
            'home_regressor': os.path.join(NFL_MODEL_DIR, 'model_home_score.pkl'),
            'away_regressor': os.path.join(NFL_MODEL_DIR, 'model_away_score.pkl'),
            'feature_cols': os.path.join(NFL_MODEL_DIR, 'model_features.pkl'),
            'team_win': os.path.join(NFL_MODEL_DIR, 'model_home_win.pkl')
        }
        for path_name, path in model_files.items():
            if not os.path.isfile(path):
                raise FileNotFoundError(f"❌ Required model file not found: {path}")
        # Load all required models
        self.home_regressor = load(model_files['home_regressor'])
        self.away_regressor = load(model_files['away_regressor'])
        self.feature_cols = load(model_files['feature_cols'])

    def predict_upcoming_games(self):
        try:
            print("📅 Fetching upcoming NFL games...")
            match_data = self.upcomming_game.upcoming_games()
            if not match_data:
                print("⚠️ No upcoming games found.")
                return []

            predictions = []
            for match in match_data[10:20]:
                try:
                    prediction = self._predict_match(match)
                    if prediction:
                        predictions.append(prediction)
                except Exception as e:
                    print(f"⚠️ Error predicting match {match.get('id', 'unknown')}: {e}")

            return self._sort_predictions(predictions)
        except Exception as e:
            print(f"❌ Error predicting upcoming games: {e}")
            return []

    def _predict_match(self, match_data):
        try:
            # Get predictive features
            features = self.data_processor.extract_predictive_features(match_data)

            df = pd.DataFrame([features])

            df = df.reindex(columns=self.feature_cols, fill_value=0)
            df = df.apply(pd.to_numeric, errors='coerce').fillna(0)
            probs = self.classifier.predict_proba(df)[0]

            home_win_prob = float(probs[1])
            away_win_prob = float(probs[0])
            prediction = "HOME" if home_win_prob > away_win_prob else "AWAY"
            confidence = safe_float(round(abs(home_win_prob - 0.5) * 2 * 100, 1))

            home_score = int(round(self.home_regressor.predict(df)[0]))
            away_score = int(round(self.away_regressor.predict(df)[0]))

            data= {
                'home_team': features['home_team'],
                'away_team': features['away_team'],
                'date': features['date'],
                'time': features['time'],
                'venue': features['venue'],
                'venue_id': features['venue_id'],
                'home_win_probability': safe_float(round(home_win_prob * 100, 1)),
                'away_win_probability': safe_float(round(away_win_prob * 100, 1)),
                'home_predict_score': home_score,
                'away_predict_score': away_score,
                'confidence': confidence,
                'home_team_rank': features.get('home_team_rank'),
                'away_team_rank': features.get('away_team_rank'),
                'prediction': prediction,
            }
            print(data)
            return data
        except Exception as e:
            print(f"⚠️ Error predicting a scheduled match: {e}")
            return None

    def _sort_predictions(self, predictions):
        return sorted(
            predictions,
            key=lambda x: float(x.get('confidence', 0)) if isinstance(x.get('confidence', 0), (int, float)) else 0,
            reverse=True
        )

    def print_predictions(self, predictions):
        if not predictions:
            print("⚠️ No predictions available.")
            return

        print("\n🏈 Upcoming NFL Game Predictions:")
        for i, p in enumerate(predictions, 1):
            print(f"\nGame {i}:")
            print(f"  📆 Date                : {p['date']}")
            print(f"  ⏰ Time                : {p['time']} ({p.get('venue', '')})")
            print(f"  🏠 Home Team           : {p['home_team']} (Rank: {p['home_team_rank']})")
            print(f"  🛫 Away Team           : {p['away_team']} (Rank: {p['away_team_rank']})")
            print(f"  🎯 Predicted Winner    : {p['prediction']} (Confidence: {p['confidence']}%)")
            print(f"  📈 Home Win Probability: {p['home_win_probability']}%")
            print(f"  📉 Away Win Probability: {p['away_win_probability']}%")
            print(f"  🔢 Predicted Score     : {p['home_team']} {p['home_predict_score']} - {p['away_predict_score']} {p['away_team']}")

# 🧪 Run example
if __name__ == "__main__":
    predictor = LiveGamePredictor()
    predictions = predictor.predict_upcoming_games()
    predictor.print_predictions(predictions)
