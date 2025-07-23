import joblib
from datetime import datetime, timedelta
from app.services.MLB.teamData_processor import BaseballDataProcessor
from app.services.MLB.playerDataProcessor import PlayerDataProcessor
from app.services.MLB.playerDataCollector import extract_data
from app.services.helper import safe_float, safe_int
import pandas as pd
import os
from dotenv import load_dotenv
import numpy as np
from sklearn.preprocessing import StandardScaler
import warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore")

load_dotenv()

class PerformerPredictor:
    def __init__(self, model_path='models/MLB'):
        self.model_path = model_path
        self.data_processor = BaseballDataProcessor(os.getenv('GOALSERVE_API_KEY'))
        self.player_data_processor = PlayerDataProcessor()
        
        # Load or create models
        # Pitcher models
        self.pitcher_model = joblib.load(os.path.join(self.model_path, 'pitcher', 'mlb_innings_pitched_regressor.pkl'))
        self.strikeouts_model = joblib.load(os.path.join(self.model_path, 'pitcher', 'mlb_strikeouts_regressor.pkl'))
        self.earned_runs_model = joblib.load(os.path.join(self.model_path, 'pitcher', 'mlb_earned_runs_regressor.pkl'))

    # ... (keep existing __init__ and utility methods)

    def predict_pitcher_performance(self, pitcher_stats, features):
        """Predict a pitcher's performance using the trained models"""
        try:
            df =   pd.DataFrame([pitcher_stats])
            X = df.drop(columns=['name','is_starter'])
            x_values = X.values.reshape(1, -1)
            # print("is none------------------", X.isna().sum())

            # # Scale features
            # self.player_data_processor.scalers['pitcher'] = StandardScaler()
            # X_scaled = self.player_data_processor.scalers['pitcher'].fit_transform(X)
            # input_data = pd.DataFrame(X_scaled, columns=features, index=X.index)
            
            # # Predict stats
            # Get predictions from all estimators
            innings_all = np.array([tree.predict(x_values)[0] for tree in self.pitcher_model.estimators_])
            strikeouts_all = np.array([tree.predict(x_values)[0] for tree in self.strikeouts_model.estimators_])
            earned_runs_all = np.array([tree.predict(x_values)[0] for tree in self.earned_runs_model.estimators_])

            # Mean predictions
            innings_pitched = innings_all.mean()
            strikeouts = strikeouts_all.mean()
            earned_runs = earned_runs_all.mean()

            # Standard deviation as uncertainty
            innings_std = innings_all.std()
            strikeouts_std = strikeouts_all.std()
            earned_runs_std = earned_runs_all.std()

            # Combine into a confidence score (lower std means higher confidence)
            # Normalize: confidence = 1 / (1 + std)
            confidence = 1 / (1 + np.mean([innings_std, strikeouts_std, earned_runs_std]))
            confidence = round(float(confidence), 3)


            # # Create prediction result
            prediction = {
                'predicted_stats': {
                    'innings_pitched': safe_float(innings_pitched),
                    'strikeouts': safe_float(strikeouts),
                    'earned_runs': safe_float(earned_runs)
                },
                'name': df['name'],
                'confidence': confidence,  # Placeholder for confidence score
                #'game_features': game_features
            }
            
            return prediction
        
        except Exception as e:
            print(f"Error predicting pitcher performance: {e}")
            return {
                'predicted_stats': {},
                'confidence': 0.0,
                #'game_features': game_features
            }
    def evaluate_pitcher_performance(self, pitcher_stats, features, is_starter=False):
        """Evaluate a pitcher's predicted performance with scoring"""
        # print("features------------------", type(pitcher_stats)) --> class 'dict'

        prediction = self.predict_pitcher_performance(pitcher_stats, features)
        
        # Calculate performance score (higher is better)
        ip = prediction['predicted_stats']['innings_pitched']
        k = prediction['predicted_stats']['strikeouts']
        er = prediction['predicted_stats']['earned_runs']
        
        # Quality Start formula (6+ IP, 3≤ ER) is worth more
        quality_start = 1 if (ip >= 6 and er <= 3) else 0
        
        # Dominance factor (high K/9 rates are valuable)
        k_per_9 = (k / ip * 9) if ip > 0 else 0
        
        # Performance score calculation
        performance_score = safe_float(
            (ip * 2) +          # Innings pitched are valuable
            (k * 0.5) +         # Strikeouts are valuable
            -(er * 2) +         # Earned runs hurt
            (quality_start * 3)+ # Quality starts are very valuable
            (k_per_9 * 0.3)     # Dominance bonus
        )
        
        # Starters get a small bonus since they face lineup multiple times
        if is_starter:
            performance_score *= 1.2
        print("performance_score------------------", performance_score)   
        return {
            **prediction,
            'performance_score': safe_float(performance_score),
            'is_starter': is_starter
        }
    

    def predict_top_pitcher(self, team_type, game):
        """Predict the top pitcher for a team considering all pitchers"""
        try:
            # pitchers = self.get_all_pitchers(team_type, game_data)
            # print("pitchers------------------",pitchers)
            # if not pitchers:
            #     return None
                
            # game_features = self.data_processor.extract_predictive_features(game_data)
            game_data = self.player_data_processor.extract_game_info(game)
            pitcher_stats = self.player_data_processor.extract_pitcher_stats(game, game_data)
            df = pd.DataFrame(pitcher_stats)

            if not df.empty:
                df['game_date'] = pd.to_datetime(df['game_date'], dayfirst=True)
                df['season'] = df['game_date'].dt.year

            df = self.player_data_processor.create_pitcher_features(df)
            feature_cols = self.player_data_processor.get_pitcher_feature_columns()
            new_df = df[feature_cols].fillna(df[feature_cols].median())
            
            # print("NAN value------------------",new_df.isna().sum())
            new_df['is_starter'] = df['is_starter']
            new_df['name'] = df['player_name'].iloc[0]

                # Evaluate all pitchers
            evaluated_pitchers = [
                self.evaluate_pitcher_performance(
                    p, 
                    features=feature_cols,
                    is_starter=p.get('is_starter', False))
                for p in  new_df.to_dict(orient='records')
            ]
                
            #     # Find the top performer by performance score
            top_pitcher = max(evaluated_pitchers, key=lambda x: x['performance_score'])
            
        #     # Format the result
            return {
                'name': top_pitcher['name'],
                'predicted_stats': top_pitcher['predicted_stats'],
                'confidence': top_pitcher['confidence'],
                'is_starter': top_pitcher['is_starter'],
                'performance_score': top_pitcher['performance_score']
            }
            
        except Exception as e:
            print(f"Error predicting top pitcher for {team_type}: {e}")
            return None

    def predict_top_performers(self, game_data):
        """Predict top performers for both teams considering all pitchers"""
        home_team_name = game_data.get('hometeam', {}).get('@name', '')
        away_team_name = game_data.get('awayteam', {}).get('@name', '')
        try:
            return {
                'home_team': {
                    'top_pitcher': self.predict_top_pitcher('hometeam', game_data),
                    'top_batter': predict_top_batter(home_team_name)
                },
                'away_team': {
                    'top_pitcher': self.predict_top_pitcher('awayteam', game_data),
                    'top_batter': predict_top_batter(away_team_name)
                }
            }
        except Exception as e:
            print(f"Error predicting top performers: {e}")
            return {
                'home_team': {'top_pitcher': None},
                'away_team': {'top_pitcher': None}
            }
    def get_todays_pitcher_data(self,game_data):
            scoreboard= self.data_processor.get_scoreboard()
            for match in scoreboard['scores']['category']['match']:
                if match.get('@id') == game_data.get('@id'):
                    return match
                else:
                    pass # use roaster data by team id
                
    def predict_todays_games(self):
        """Predict today's games with enhanced features"""
        try:
            games = self.data_processor.get_todays_games() # Fetch schedule for today's games
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
            # standings = self.data_processor.fetch_data("baseball/mlb_standings")

            for game in matches:

                try:
                    # Predict top performers
                    top_performers = self.predict_top_performers(game)

                    predictions.append({
                        'game_id': game.get('@id', ''),
                        'home_team': game.get('hometeam', {}).get('@name', 'Home Team'),
                        'away_team': game.get('awayteam', {}).get('@name', 'Away Team'),
                        'date': game.get('@formatted_date', datetime.now().strftime('%Y-%m-%d')),
                        'time': game.get('@time', 'TBD'),
                        'venue': game.get('@venue_name', 'TBD'),
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
    predictor = PerformerPredictor()
    today_predictions = predictor.predict_todays_games()

    for prediction in today_predictions:
        print(prediction)