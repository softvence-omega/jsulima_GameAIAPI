import joblib
from datetime import datetime, timedelta
from app.services.MLB.teamData_processor import BaseballDataProcessor
from app.services.helper import safe_float, safe_int
import pandas as pd
import os
from dotenv import load_dotenv
import numpy as np
from sklearn.preprocessing import StandardScaler
from app.services.MLB.mlb_performer_predictor import PerformerPredictor

performer_predictor = PerformerPredictor()

load_dotenv()

class LiveGamePredictor:
    def __init__(self, model_path='models/MLB/'):
        self.model_path = model_path
        self.data_processor = BaseballDataProcessor(os.getenv('GOALSERVE_API_KEY'))
        
        # Load or create models
        self._load_models()
        
        # Initialize scalers
        self.scaler = StandardScaler()
        
    def _load_models(self):
        """Load existing models or create new ones"""
        # Core game prediction models
        self.win_classifier_model = joblib.load(os.path.join(self.model_path, 'mlb_win_classifier.pkl'))
        self.home_score_model = joblib.load(os.path.join(self.model_path, 'mlb_home_score_regressor.pkl'))
        self.away_score_model = joblib.load(os.path.join(self.model_path, 'mlb_away_score_regressor.pkl'))  
        
        print("All models loaded successfully")


    def get_team_roster(self, team_id):
        """Get current roster for a team"""
        try:
            roster_data = self.data_processor.fetch_data(f"baseball/{team_id}_rosters")
            return roster_data
        except Exception as e:
            print(f"Error fetching roster for team {team_id}: {e}")
            return None

    def predict_todays_games(self):
        """Predict today's games with enhanced features"""
        try:
            games = self.data_processor.get_todays_games()

            if not games or 'fixtures' not in games:
                return []

            # Handle different data structures
            fixtures = games['fixtures']
            if 'category' in fixtures:
                matches_data = fixtures['category'].get('matches', [])
            else:
                matches_data = fixtures
                
            if isinstance(matches_data, dict):
                matches_data = [matches_data]

            matches = []
            for day in matches_data:
                day_matches = day.get('match', [])
                if isinstance(day_matches, dict):
                    matches.append(day_matches)
                elif isinstance(day_matches, list):
                    matches.extend(day_matches)

            predictions = []
            standings = self.data_processor.fetch_data("baseball/mlb_standings")
            i = 0
            for game in matches:
                i += 1
                print(f"Processing game {i}/{len(matches)}: {game.get('@id', 'Unknown ID')}")
                try:
                    features = self.data_processor.extract_predictive_features(
                        game_data=game,
                        standings_data=standings
                    )
                    
                    if not features:
                        continue
                        
                    features_df = pd.DataFrame([features]).fillna(0)
        
                    # Make predictions
                    if hasattr(self.win_classifier_model, 'predict_proba'):
                        home_win_prob = self.win_classifier_model.predict_proba(features_df)[0][1]
                    else:
                        home_win_prob = 0.5
                        
                    pred_home_score = self.home_score_model.predict(features_df)[0] if hasattr(self.home_score_model, 'predict') else 5
                    pred_away_score = self.away_score_model.predict(features_df)[0] if hasattr(self.away_score_model, 'predict') else 4

                    # Predict top performers
                    top_performers = performer_predictor.predict_top_performers(game)

                    # Calculate confidence
                    confidence = abs(home_win_prob - 0.5) * 2 * 100

                    # Predict top performers
                    top_performers = self.predict_top_performers(game, features)

                    predictions.append({
                        'game_id': game.get('@id', ''),
                        'home_team': game.get('hometeam', {}).get('@name', 'Home Team'),
                        'away_team': game.get('awayteam', {}).get('@name', 'Away Team'),
                        'home_team_logo': game.get('hometeam', {}).get('@logo', ''),
                        'away_team_logo': game.get('awayteam', {}).get('@logo', ''),
                        'home_win_probability': safe_float(home_win_prob * 100, 1),
                        'away_win_probability': safe_float((1 - home_win_prob) * 100, 1),
                        'predicted_home_score': safe_float(max(0, pred_home_score), 0),
                        'predicted_away_score': safe_float(max(0, pred_away_score), 0),
                        'date': game.get('@formatted_date', datetime.now().strftime('%Y-%m-%d')),
                        'time': game.get('@time', 'TBD'),
                        'venue': game.get('@venue_name', 'TBD'),
                        'prediction': "HOME" if home_win_prob > 0.5 else "AWAY",
                        'confidence': safe_float(round(confidence, 1), 0),
                        'ai_confidence': safe_float(min(95, max(65, round(confidence + 20, 0))), 0),  # AI confidence display
                        'top_performers': top_performers,
                        # 'key_insights': self._generate_key_insights(game, features, top_performers)
                    })

                except Exception as e:
                    print(f"Error predicting game: {str(e)}")
                    continue

            # return sorted(predictions, key=lambda x: x['confidence'], reverse=True)
            return predictions
            
        except Exception as e:
            print(f"Error in predict_todays_games: {str(e)}")
            return []

    def _generate_key_insights(self, game_data, features, top_performers):
        """Generate key insights for the game"""
        insights = []
        
        try:
            # Pitching matchup insight
            home_era = features.get('home_sp_era', 4.5)
            away_era = features.get('away_sp_era', 4.5)
            
            if abs(home_era - away_era) > 1.0:
                better_pitcher = "home" if home_era < away_era else "away"
                insights.append(f"The {better_pitcher} team has a significant pitching advantage")
            
            # Offensive insight
            home_ops = features.get('home_ops', 0.7)
            away_ops = features.get('away_ops', 0.7)
            
            if abs(home_ops - away_ops) > 0.1:
                better_offense = "home" if home_ops > away_ops else "away"
                insights.append(f"The {better_offense} team has the stronger offensive lineup")
            
            # Record insight
            home_win_pct = features.get('home_win_pct', 0.5)
            away_win_pct = features.get('away_win_pct', 0.5)
            
            if abs(home_win_pct - away_win_pct) > 0.1:
                better_record = "home" if home_win_pct > away_win_pct else "away"
                insights.append(f"The {better_record} team comes in with a better record")
            
            # Default insight if no specific patterns
            if not insights:
                insights.append("This should be a competitive matchup between two evenly matched teams")
                
        except Exception as e:
            insights = ["Game analysis in progress"]
        
        return insights[:3]  # Return top 3 insights

# Usage and testing
if __name__ == "__main__":
    predictor = LiveGamePredictor()
    today_predictions = predictor.predict_todays_games()
    