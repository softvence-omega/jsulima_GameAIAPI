import joblib
from datetime import datetime

from app.services.MLB.teamData_processor import BaseballDataProcessor

from sklearn.metrics import accuracy_score
import pandas as pd
import os
from dotenv import load_dotenv
load_dotenv()

class LiveGamePredictor:
    def __init__(self, model_path='models/mlb_win_classifier.pkl'):
        self.model = joblib.load(model_path)
        self.data_processor = BaseballDataProcessor(os.getenv('GOALSERVE_API_KEY'))

    def predict_todays_games(self):

        games = self.data_processor.fetch_data("baseball/usa", {"date1": "07.04.2023", "date2": "08.04.2023"})

        if games and 'scores' in games:
            category = games['scores'].get('category', {})
            matches = category.get('match', [])
            matches = [matches] if isinstance(matches, dict) else matches
            total = 0
            correct = 0
            for i, game in enumerate(matches, 1):
                print(f"match----------{i}")
                try:
                    standings = self.data_processor.fetch_data("baseball/mlb_standings")
                    features = self.data_processor.extract_predictive_features(game, standings_data=standings)
                    
                    if features:  
                        home_score = int(game.get('hometeam', {}).get('@totalscore', 0))
                        away_score = int(game.get('awayteam', {}).get('@totalscore', 0))

                        features['home_score'] = home_score
                        features['away_score'] = away_score
                        features['home_win'] = 1 if home_score > away_score else 0
                        features_df = pd.DataFrame([features])

                        print("features----------", features)
                        X = features_df.drop(['home_score', 'away_score', 'home_win'], axis=1)
                        y_true = int(features_df['home_win'].iloc[0])

                        proba = self.model.predict_proba(X)[0]
                        predicted_class = int(proba[1] >= 0.5)
                        if predicted_class == y_true:
                            correct += 1
                        total += 1

                except Exception as e:
                    print(f"Skipping game for {str(e)}")
                    continue
            print(f"Overall accuracy: {correct}/{total} = {correct / total:.4f}")

if __name__ == "__main__":
    predictor = LiveGamePredictor()
    today_predictions = predictor.predict_todays_games()
#
#     print("\nToday's MLB Predictions:",today_predictions)
#     for idx, pred in enumerate(today_predictions, 1):
#         print(f"{idx}. {pred['home_team']} vs {pred['away_team']}")
#         print(f"   Prediction: {pred['prediction']} ({pred['home_win_probability']}%)")
#         print(f"   Confidence: {pred['confidence']:.1%}\n")