import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from app.services.MLB.teamData_processor import BaseballDataProcessor
from app.services.MLB.playerDataCollector import extract_data
import os
from dotenv import load_dotenv
load_dotenv()

import warnings
warnings.filterwarnings('ignore')

processor= BaseballDataProcessor(os.getenv("GOALSERVE_API_KEY"))

class BaseballPerformancePredictor:
    def __init__(self):
        self.batter_models = {}
        self.pitcher_models = {}
        self.scalers = {}
        self.label_encoders = {}
        
        # Define target variables for each prediction
        self.batter_targets = ['hits', 'home_runs', 'rbis']
        self.pitcher_targets = ['innings_pitched', 'strikeouts', 'earned_runs']

    
    
    def train_batter_models(self, df):
        """Train XGBoost models for batter predictions"""
        df = self.create_batter_features(df)
        feature_cols = self.get_batter_feature_columns()
        
        # Prepare data
        X = df[feature_cols].fillna(df[feature_cols].median())
        
        # Scale features
        self.scalers['batter'] = StandardScaler()
        X_scaled = self.scalers['batter'].fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=feature_cols, index=X.index)
        
        # Train models for each target
        for target in self.batter_targets:
            #print(f"Training batter model for {target}...")
            
            y = df[target].fillna(0)
            
            # Remove rows with missing target values
            mask = ~y.isna()
            X_clean = X_scaled[mask]
            y_clean = y[mask]
            
            # XGBoost parameters optimized for baseball stats
            params = {
                'objective': 'reg:squarederror',
                'max_depth': 6,
                'learning_rate': 0.1,
                'n_estimators': 300,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'reg_alpha': 0.1,
                'reg_lambda': 1.0,
                'random_state': 42
            }
            
            model = xgb.XGBRegressor(**params)
            model.fit(X_clean, y_clean)
            
            self.batter_models[target] = model
            
            # #print feature importance
            importance = pd.DataFrame({
                'feature': feature_cols,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            #print(f"Top 5 features for {target}:")
            #print(importance.head().to_string(index=False))
            #print()
    
    def train_pitcher_models(self, df):
        """Train XGBoost models for pitcher predictions"""
        df = self.create_pitcher_features(df)
        feature_cols = self.get_pitcher_feature_columns()
        
        # Prepare data
        X = df[feature_cols].fillna(df[feature_cols].median())
        
        # Scale features
        self.scalers['pitcher'] = StandardScaler()
        X_scaled = self.scalers['pitcher'].fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=feature_cols, index=X.index)
        
        # Train models for each target
        for target in self.pitcher_targets:
            #print(f"Training pitcher model for {target}...")
            
            y = df[target].fillna(df[target].median())
            
            # Remove rows with missing target values
            mask = ~y.isna()
            X_clean = X_scaled[mask]
            y_clean = y[mask]
            
            # XGBoost parameters
            params = {
                'objective': 'reg:squarederror',
                'max_depth': 5,
                'learning_rate': 0.1,
                'n_estimators': 250,
                'subsample': 0.85,
                'colsample_bytree': 0.8,
                'reg_alpha': 0.1,
                'reg_lambda': 1.0,
                'random_state': 42
            }
            
            model = xgb.XGBRegressor(**params)
            model.fit(X_clean, y_clean)
            
            self.pitcher_models[target] = model
            
            # Print feature importance
            importance = pd.DataFrame({
                'feature': feature_cols,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            #print(f"Top 5 features for {target}:")
            #print(importance.head().to_string(index=False))
            #print()
    
    def predict_batter_performance(self, df):
        """Predict batter performance for upcoming games"""
        df = self.create_batter_features(df)
        feature_cols = self.get_batter_feature_columns()
        
        X = df[feature_cols].fillna(df[feature_cols].median())
        X_scaled = self.scalers['batter'].transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=feature_cols, index=X.index)
        
        predictions = {}
        for target in self.batter_targets:
            pred = self.batter_models[target].predict(X_scaled)
            # Round and ensure non-negative predictions
            pred = np.maximum(0, np.round(pred)).astype(int)
            predictions[target] = pred
            
        return pd.DataFrame(predictions, index=df.index)
    
    def predict_pitcher_performance(self, df):
        """Predict pitcher performance for upcoming games"""
        df = self.create_pitcher_features(df)
        feature_cols = self.get_pitcher_feature_columns()
        
        X = df[feature_cols].fillna(df[feature_cols].median())
        X_scaled = self.scalers['pitcher'].transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=feature_cols, index=X.index)
        
        predictions = {}
        for target in self.pitcher_targets:
            pred = self.pitcher_models[target].predict(X_scaled)
            
            # Apply realistic constraints
            if target == 'innings_pitched':
                pred = np.clip(pred, 0, 9)  # Max 9 innings
            elif target == 'strikeouts':
                pred = np.maximum(0, pred)
            elif target == 'earned_runs':
                pred = np.maximum(0, pred)
            
            # Round appropriately
            if target == 'innings_pitched':
                pred = np.round(pred * 3) / 3  # Round to nearest 1/3 inning
            else:
                pred = np.round(pred).astype(int)
                
            predictions[target] = pred
            
        return pd.DataFrame(predictions, index=df.index)
    
    def calculate_prediction_confidence(self, df, player_type='batter'):
        """Calculate confidence score based on data quality and model certainty"""
        confidence_scores = []
        
        if player_type == 'batter':
            feature_cols = self.get_batter_feature_columns()
        else:
            feature_cols = self.get_pitcher_feature_columns()
        
        X = df[feature_cols].fillna(df[feature_cols].median())
        
        for idx in X.index:
            # Base confidence starts at 50%
            confidence = 50
            
            # Add confidence based on data completeness
            missing_ratio = X.loc[idx].isna().sum() / len(feature_cols)
            confidence += (1 - missing_ratio) * 30
            
            # Add confidence based on recent games played
            if player_type == 'batter' and 'games_played_season' in df.columns:
                games = df.loc[idx, 'games_played_season']
                if games >= 10:
                    confidence += 20
                elif games >= 5:
                    confidence += 10
            
            # Cap at 95% maximum confidence
            confidence = min(confidence, 95)
            confidence_scores.append(int(confidence))  
        
        return confidence_scores
    
    # predict not started game
    def predict_not_started_game(self):
        pass

    # feature extraction for prediction
    def predict_todays_game(self):
        games= processor.get_todays_games()
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
        scoreboard= processor.get_todays_score()
        for game in matches:
            # check match status
            flag=0
            for match in scoreboard['scores']['category']['match']:
                if game['@id']== match['@id']:
                    # Predict top performers
                    game_data, pitcher_stats, batter_stats= extract_data(game)
                    game_df= pd.DataFrame([game_data])
                    pitcher_df= pd.DataFrame([pitcher_stats])
                    #print("pitcher df------------------\n", pitcher_df)
                    batter_df= pd.DataFrame([batter_stats])

                    # extract feature
                    #print( self.predict_batter_performance(batter_df) )
                    #print( self.predict_pitcher_performance(pitcher_df) )
                    flag=1
                    break
            if flag==0:
                predict_not_started_game(self)

def create_sample_predictions():
    """Create sample predictions matching the Figma design"""
    
    # Initialize predictor
    predictor = BaseballPerformancePredictor()
    
    # Sample data for demonstration
    sample_batters = pd.DataFrame({
        'player_id': ['judge_aaron', 'martinez_jd'],
        'player_name': ['Aaron Judge', 'J.D. Martinez'],
        'position': ['OF', 'DH'],
        'team': ['New York Yankees', 'Boston Red Sox'],
        'batting_avg_last_10': [0.325, 0.280],
        'hr_last_5': [0.8, 1.2],
        'rbi_last_5': [2.1, 1.8],
        'opp_pitcher_era': [3.45, 4.20],
        'is_home': [1, 0],
        'vs_righty': [1, 1]
    })
    
    sample_pitchers = pd.DataFrame({
        'player_id': ['cole_gerrit', 'sale_chris'],
        'player_name': ['Gerrit Cole', 'Chris Sale'],
        'position': ['SP', 'SP'],
        'team': ['New York Yankees', 'Boston Red Sox'],
        'era_last_5': [2.85, 4.50],
        'k9_last_5': [11.2, 8.8],
        'ip_last_5': [6.8, 5.5],
        'opp_team_avg': [0.245, 0.268],
        'is_home': [1, 1],
        'days_rest': [4, 5]
    })
    
    # Mock predictions (in real implementation, these would come from trained models)
    batter_predictions = {
        'judge_aaron': {'hits': 2, 'home_runs': 1, 'rbis': 2, 'confidence': 90},
        'martinez_jd': {'hits': 1, 'home_runs': 2, 'rbis': 1, 'confidence': 70}
    }
    
    pitcher_predictions = {
        'cole_gerrit': {'innings_pitched': 7.0, 'strikeouts': 9, 'earned_runs': 2, 'confidence': 90},
        'sale_chris': {'innings_pitched': 6.0, 'strikeouts': 6, 'earned_runs': 4, 'confidence': 70}
    }
    
    #print("=== BASEBALL PERFORMANCE PREDICTIONS ===\n")
    
    #print("BATTER PREDICTIONS:")
    #print("-" * 50)
    for _, batter in sample_batters.iterrows():
        pred = batter_predictions[batter['player_id']]
        #print(f"{batter['player_name']} ({batter['position']}) - {batter['team']}")
        #print(f"  Predicted Hits: {pred['hits']}")
        #print(f"  Predicted Home Runs: {pred['home_runs']}")
        #print(f"  Predicted RBIs: {pred['rbis']}")
        #print(f"  Confidence: {pred['confidence']}%")
        #print()
    
    #print("PITCHER PREDICTIONS:")
    #print("-" * 50)
    for _, pitcher in sample_pitchers.iterrows():
        pred = pitcher_predictions[pitcher['player_id']]
        #print(f"{pitcher['player_name']} ({pitcher['position']}) - {pitcher['team']}")
        #print(f"  Predicted Innings Pitched: {pred['innings_pitched']}")
        #print(f"  Predicted Strikeouts: {pred['strikeouts']}")
        #print(f"  Predicted Earned Runs: {pred['earned_runs']}")
        #print(f"  Confidence: {pred['confidence']}%")
        #print()


        



            

from app.config import DATA_DIR
load_dotenv()
# Example usage
if __name__ == "__main__":
    # Create sample predictions
    # create_sample_predictions()
    
    predictor = BaseballPerformancePredictor()
    # Load your historical data
    batter_data = pd.read_csv(os.path.join(DATA_DIR,'csv/MLB','batter_stats_data(2010-2024).csv'))
    pitcher_data = pd.read_csv(os.path.join( DATA_DIR,'csv/MLB','pitcher_stats_data(2010-2024).csv') )

    # Train models
    predictor.train_pitcher_models(pitcher_data)
    
    # Prediction
    # create a df 
    
    predictor.predict_todays_game()


    # Example of how to use the predictor class
    """
    # Initialize predictor
    predictor = BaseballPerformancePredictor()
    
    # Load your historical data
    batter_data = pd.read_csv('historical_batter_data.csv')
    pitcher_data = pd.read_csv('historical_pitcher_data.csv')
    
    # Train models
    predictor.train_batter_models(batter_data)
    predictor.train_pitcher_models(pitcher_data)
    
    # Make predictions for upcoming games
    upcoming_batters = pd.read_csv('upcoming_batters.csv')
    upcoming_pitchers = pd.read_csv('upcoming_pitchers.csv')
    
    batter_predictions = predictor.predict_batter_performance(upcoming_batters)
    pitcher_predictions = predictor.predict_pitcher_performance(upcoming_pitchers)
    
    # Calculate confidence scores
    batter_confidence = predictor.calculate_prediction_confidence(upcoming_batters, 'batter')
    pitcher_confidence = predictor.calculate_prediction_confidence(upcoming_pitchers, 'pitcher')
    """