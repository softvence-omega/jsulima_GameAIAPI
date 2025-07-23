import os
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
from joblib import load
from app.services.NFL.data_processor import NflDataProcessor

# Load environment variables
load_dotenv()

class LiveGamePredictor:
   def __init__(self, model_path='models/nfl_all_models.pkl'):
    models = load(model_path)
    self.classifier = models['win_classifier_model']
    self.home_regressor = models['home_score_model']
    self.away_regressor = models['away_score_model']
    self.feature_cols = models.get("features")
    self.data_processor = NflDataProcessor(os.getenv('GOALSERVE_API_KEY'))

    def predict_todays_games(self):
        today = datetime.now().strftime("%d.%m.%Y")
        print(f"📅 Fetching NFL games for today: {today}")

        try:
            games_data = self.data_processor.fetch_data("football/nfl-scores")
            categories = games_data.get('scores', {}).get('category', [])
            if isinstance(categories, dict):
                categories = [categories]

            predictions = []

            for category in categories:
                games = category.get('match', [])
                if isinstance(games, dict):
                    games = [games]

                for game in games:
                    try:
                        features = self.data_processor.extract_predictive_features(game)
                        if features:
                            df = pd.DataFrame([features])
                            if self.feature_cols:
                                df = df.reindex(columns=self.feature_cols, fill_value=0)

                            # --- Classification: Win Probabilities ---
                            probs = self.classifier.predict_proba(df)[0]
                            home_win_prob = float(probs[1])
                            away_win_prob = float(probs[0])
                            prediction = "HOME" if home_win_prob > away_win_prob else "AWAY"
                            confidence = round(abs(home_win_prob - 0.5) * 2, 2)

                            # --- Regression: Score Predictions ---
                            home_score = int(round(self.home_regressor.predict(df)[0]))
                            away_score = int(round(self.away_regressor.predict(df)[0]))

                            # --- Game Info ---
                            predictions.append({
                                'home_team': game.get('hometeam', {}).get('@name', ''),
                                'away_team': game.get('awayteam', {}).get('@name', ''),
                                'date': game.get('@date', ''),
                                'time': game.get('@time', ''),
                                'venue': game.get('@venue_name', ''),
                                'venue_id': game.get('@venue_id', ''),
                                'home_win_probability': round(home_win_prob * 100, 1),
                                'away_win_probability': round(away_win_prob * 100, 1),
                                'home_predict_score': home_score,
                                'away_predict_score': away_score,
                                'home_team_predict_percentage': round(home_win_prob * 100, 1),
                                'away_team_predict_percentage': round(away_win_prob * 100, 1),
                                'confidence': round(confidence * 100, 1),
                                'home_team_rank': features.get('home_team_rank'),
                                'away_team_rank': features.get('away_team_rank'),
                                'prediction': prediction,
                            })

                    except Exception as e:
                        print(f"⚠️ Error predicting a game: {str(e)}")
                        continue

            return sorted(predictions, key=lambda x: x['confidence'], reverse=True)

        except Exception as e:
            print(f"❌ Error fetching today's games: {str(e)}")
            return []

    def print_predictions(self, predictions):
        if not predictions:
            print("⚠️ No games found or predictions failed.")
            return

        print("\n🏈 Today's NFL Game Predictions:")
        for i, p in enumerate(predictions, 1):
            print(f"\nGame {i}:")
            print(f"  📆 Date                : {p['date']}")
            print(f"  ⏰ Time                : {p['time']}")
            print(f"  🏟️ Venue               : {p['venue']} (ID: {p['venue_id']})")
            print(f"  🏠 Home Team           : {p['home_team']} (Rank: {p['home_team_rank']})")
            print(f"  🛫 Away Team           : {p['away_team']} (Rank: {p['away_team_rank']})")
            print(f"  🎯 Predicted Winner    : {p['prediction']}")
            print(f"  📈 Home Win Probability: {p['home_win_probability']}%")
            print(f"  📉 Away Win Probability: {p['away_win_probability']}%")
            print(f"  🔢 Predicted Score     : {p['home_team']} {p['home_predict_score']} - {p['away_predict_score']} {p['away_team']}")
            print(f"  ✅ Confidence Level    : {p['confidence']}%")

# 🧪 Example usage
if __name__ == "__main__":
    predictor = LiveGamePredictor(model_path='models/nfl_all_models.pkl')
    results = predictor.predict_todays_games()
    predictor.print_predictions(results)
