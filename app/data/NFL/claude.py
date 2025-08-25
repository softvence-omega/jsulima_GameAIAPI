import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class NFLStarterPredictor:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.position_groups = {
            'QB': ['QB'],
            'RB': ['RB', 'FB'],
            'WR': ['WR'],
            'TE': ['TE'],
            'OL': ['C', 'G', 'T', 'OT', 'OG'],
            'DL': ['DE', 'DT', 'NT'],
            'LB': ['LB', 'OLB', 'ILB', 'MLB'],
            'DB': ['CB', 'S', 'FS', 'SS', 'DB'],
            'K': ['K'],
            'P': ['P']
        }
        
    def load_and_preprocess_data(self, csv_file):
        """Load and preprocess the NFL game data"""
        print("Loading NFL game data...")
        df = pd.read_csv(csv_file)
        df = df.tail(3000)  # Use last 50,000 records for testing
        
        # Convert date to datetime
        df['date'] = pd.to_datetime(df['date'], format='%d.%m.%Y')
        
        # Add season and week columns
        df['season'] = df['date'].apply(self._get_season)
        df['week'] = df.groupby(['season', 'team_id'])['date'].rank().astype(int)
        
        # Determine if player is home or away
        df['is_home'] = (df['team_id'] == df['home_team_id']).astype(int)
        df['opponent_id'] = np.where(df['is_home'] == 1, df['away_team_id'], df['home_team_id'])
        df['opponent_name'] = np.where(df['is_home'] == 1, df['away_team_name'], df['home_team_name'])
        
        # Create starter indicator based on meaningful statistical participation
        df['is_starter'] = self._determine_starter_status(df)
        
        print(f"Loaded {len(df):,} player-game records")
        print(f"Date range: {df['date'].min()} to {df['date'].max()}")
        print(f"Unique players: {df['name'].nunique():,}")
        print(f"Games covered: {df.groupby(['date', 'home_team_name', 'away_team_name']).ngroups:,}")
        
        return df
        
    def _get_season(self, date):
        """Determine NFL season from date"""
        if date.month >= 8:
            return date.year
        else:
            return date.year - 1
            
    def _determine_starter_status(self, df):
        """Determine if player was likely a starter based on statistical participation"""
        starter_conditions = []
        
        # Offensive positions
        qb_starter = (df['player_position'] == 'QB') & (df['passing_attempts'] >= 5)
        rb_starter = (df['player_position'].isin(['RB', 'FB'])) & ((df['rushing_attempts'] >= 3) | (df['receiving_targets'] >= 2))
        wr_starter = (df['player_position'] == 'WR') & ((df['receiving_targets'] >= 2) | (df['rushing_attempts'] >= 1))
        te_starter = (df['player_position'] == 'TE') & ((df['receiving_targets'] >= 1) | (df['total_tackles'] >= 1))
        ol_starter = (df['player_position'].isin(['C', 'G', 'T', 'OT', 'OG'])) & (df['sacks'] == 0) # Assume OL played if no sacks recorded for them
        
        # Defensive positions (based on tackles, defensive stats)
        dl_starter = (df['player_position'].isin(['DE', 'DT', 'NT'])) & ((df['total_tackles'] >= 2) | (df['sacks'] >= 0.5) | (df['passes_defended'] >= 1))
        lb_starter = (df['player_position'].isin(['LB', 'OLB', 'ILB', 'MLB'])) & ((df['total_tackles'] >= 3) | (df['sacks'] >= 0.5) | (df['passes_defended'] >= 1))
        db_starter = (df['player_position'].isin(['CB', 'S', 'FS', 'SS', 'DB'])) & ((df['total_tackles'] >= 2) | (df['interceptions'] >= 1) | (df['passes_defended'] >= 1))
        
        # Special teams
        k_starter = (df['player_position'] == 'K') & ((df['field_goals_attempts'] >= 1) | (df['extra_points_attempts'] >= 1))
        p_starter = (df['player_position'] == 'P') & (df['punts'] >= 1)
        
        # Combine all conditions
        is_starter = (qb_starter | rb_starter | wr_starter | te_starter | ol_starter | 
                     dl_starter | lb_starter | db_starter | k_starter | p_starter)
        
        return is_starter.astype(int)
        
    def create_features(self, df, lookback_games=6):
        """Create comprehensive features for starter prediction"""
        print("Creating features...")
        
        # Sort by player, season, and date
        df = df.sort_values(['id', 'season', 'date'])
        
        features_list = []
        
        for player_id in df['id'].unique():
            player_data = df[df['id'] == player_id].copy()
            player_features = self._create_player_features(player_data, lookback_games)
            features_list.append(player_features)
            
        feature_df = pd.concat(features_list, ignore_index=True)
        
        # Add contextual features
        feature_df = self._add_contextual_features(feature_df)
        
        return feature_df
        
    def _create_player_features(self, player_data, lookback_games):
        """Create rolling features for individual player"""
        player_data = player_data.copy()
        
        # Rolling statistics
        rolling_cols = [
            'is_starter', 'total_points', 'yards', 'passing_yards', 'rushing_yards', 
            'receiving_yards', 'total_tackles', 'sacks', 'interceptions'
        ]
        
        for col in rolling_cols:
            if col in player_data.columns:
                # Rolling averages
                player_data[f'{col}_avg_{lookback_games}g'] = player_data[col].rolling(
                    window=lookback_games, min_periods=1
                ).mean().shift(1)
                
                # Rolling sums (for counting stats)
                player_data[f'{col}_sum_{lookback_games}g'] = player_data[col].rolling(
                    window=lookback_games, min_periods=1
                ).sum().shift(1)
                
                # Trend (slope of recent performance)
                player_data[f'{col}_trend_{lookback_games}g'] = player_data[col].rolling(
                    window=lookback_games, min_periods=2
                ).apply(self._calculate_trend).shift(1)
        
        # Career features
        player_data['games_played'] = range(1, len(player_data) + 1)
        player_data['career_starter_rate'] = player_data['is_starter'].expanding().mean().shift(1)
        player_data['games_since_start'] = self._games_since_last_start(player_data['is_starter'])
        
        # Recent form
        player_data['started_last_game'] = player_data['is_starter'].shift(1)
        player_data['started_last_2_games'] = player_data['is_starter'].rolling(2).sum().shift(1)
        player_data['started_last_4_games'] = player_data['is_starter'].rolling(4).sum().shift(1)
        
        return player_data
        
    def _calculate_trend(self, series):
        """Calculate trend (slope) of a series"""
        if len(series) < 2:
            return 0
        x = np.arange(len(series))
        return np.polyfit(x, series, 1)[0]
        
    def _games_since_last_start(self, starter_series):
        """Calculate games since last start"""
        games_since = []
        last_start = -1
        
        for i, is_starter in enumerate(starter_series):
            if is_starter == 1:
                last_start = i
            
            if last_start == -1:
                games_since.append(i)
            else:
                games_since.append(i - last_start)
                
        return games_since
        
    def _add_contextual_features(self, df):
        """Add game context features"""
        # Rest days (simplified - assume 7 days between games)
        df['rest_days'] = 7
        
        # Season progress
        df['season_progress'] = df['week'] / 18.0  # Assuming 18-week season
        
        # Home field advantage
        # is_home already exists
        
        # Opponent strength (simplified - could be enhanced with team ratings)
        df['opponent_strength'] = 0.5  # Placeholder
        
        # Position group
        df['position_group'] = df['player_position'].apply(self._get_position_group)
        
        return df
        
    def _get_position_group(self, position):
        """Map position to position group"""
        for group, positions in self.position_groups.items():
            if position in positions:
                return group
        return 'OTHER'
        
    def prepare_training_data(self, df, feature_columns=None):
        """Prepare data for model training"""
        if feature_columns is None:
            # Select relevant feature columns
            feature_columns = [col for col in df.columns if any(suffix in col for suffix in 
                             ['_avg_', '_sum_', '_trend_', 'games_played', 'career_starter_rate', 
                              'games_since_start', 'started_last_', 'rest_days', 'season_progress', 
                              'is_home', 'opponent_strength'])]
        
        # Remove rows where we don't have enough history
        df_train = df[df['games_played'] >= 2].copy()
        
        # Handle missing values
        df_train[feature_columns] = df_train[feature_columns].fillna(0)
        
        return df_train, feature_columns
        
    def train_models(self, df, feature_columns):
        """Train position-specific models"""
        print("Training position-specific models...")
        
        for position_group in self.position_groups.keys():
            print(f"Training {position_group} model...")
            
            # Filter data for this position group
            group_data = df[df['position_group'] == position_group].copy()
            
            if len(group_data) < 100:
                print(f"Insufficient data for {position_group} ({len(group_data)} samples)")
                continue
                
            X = group_data[feature_columns]
            y = group_data['is_starter']
            
            # Scale features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            self.scalers[position_group] = scaler
            
            # Train LightGBM model
            model = lgb.LGBMClassifier(
                num_leaves=31,
                learning_rate=0.05,
                feature_fraction=0.9,
                bagging_fraction=0.8,
                bagging_freq=5,
                verbose=-1,
                random_state=42
            )
            
            # Time series cross-validation
            tscv = TimeSeriesSplit(n_splits=3)
            cv_scores = cross_val_score(model, X_scaled, y, cv=tscv, scoring='roc_auc')
            
            print(f"{position_group} - Samples: {len(group_data)}, CV AUC: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
            
            # Train final model
            model.fit(X_scaled, y)
            self.models[position_group] = model
            
        print("Model training complete!")
        
    def predict_starters(self, team_roster_data, feature_columns):
        """Predict starting lineup for a team"""
        predictions = []
        
        for _, player in team_roster_data.iterrows():
            position_group = self._get_position_group(player['player_position'])
            
            if position_group not in self.models:
                # No model for this position, use simple heuristic
                starter_prob = player.get('career_starter_rate', 0.5)
            else:
                # Use trained model
                model = self.models[position_group]
                scaler = self.scalers[position_group]
                
                # Prepare features
                features = player[feature_columns].values.reshape(1, -1)
                features_scaled = scaler.transform(features)
                
                # Predict
                starter_prob = model.predict_proba(features_scaled)[0][1]
            
            predictions.append({
                'player_id': player['id'],
                'name': player['name'],
                'position': player['player_position'],
                'position_group': position_group,
                'starter_probability': starter_prob,
                'recent_performance': player.get('total_points_avg_6g', 0)
            })
            
        return sorted(predictions, key=lambda x: x['starter_probability'], reverse=True)
        
    def select_optimal_lineup(self, predictions, formation='balanced'):
        """Select optimal 11-player lineup based on formation"""
        
        formation_requirements = {
            'balanced': {
                'QB': 1, 'RB': 1, 'WR': 3, 'TE': 1, 'OL': 5,
                'DL': 3, 'LB': 3, 'DB': 4, 'K': 1, 'P': 1
            },
            'offense_heavy': {
                'QB': 1, 'RB': 2, 'WR': 4, 'TE': 2, 'OL': 5
            },
            'defense_heavy': {
                'DL': 4, 'LB': 4, 'DB': 5
            }
        }
        
        requirements = formation_requirements.get(formation, formation_requirements['balanced'])
        selected = []
        
        for position_group, count in requirements.items():
            position_players = [p for p in predictions if p['position_group'] == position_group]
            position_players = sorted(position_players, key=lambda x: x['starter_probability'], reverse=True)
            selected.extend(position_players[:count])
            
        return selected[:11]  # Ensure exactly 11 players
        
# Example usage
def main():
    # Initialize predictor
    predictor = NFLStarterPredictor()
    
    # Load data
    df = predictor.load_and_preprocess_data('nfl_player_stats_test_with_positions_v2.csv')
    
    # Create features
    feature_df = predictor.create_features(df, lookback_games=6)
    
    # Prepare training data
    train_df, feature_columns = predictor.prepare_training_data(feature_df)
    
    # Train models
    predictor.train_models(train_df, feature_columns)
    
    # Example prediction for upcoming game
    # Get latest data for a specific team
    team_id = 1693  # New Orleans Saints from your example
    latest_date = feature_df['date'].max()
    team_roster = feature_df[
        (feature_df['team_id'] == team_id) #& 
        #(feature_df['date'] == latest_date)
    ].copy()


    team_roster = team_roster.drop_duplicates(subset='id')


    if not team_roster.empty:
        # Predict starters
        predictions = predictor.predict_starters(team_roster, feature_columns)
        optimal_lineup = predictor.select_optimal_lineup(predictions)
        
        print(f"\nPredicted Starting Lineup for Team {team_id}:")
        print("-" * 60)
        for i, player in enumerate(optimal_lineup, 1):
            print(f"{i:2d}. {player['name']:25} ({player['position']:3}) - {player['starter_probability']:.3f}")
    else:
        print("No roster data available for the specified team and date.")
    return predictor

if __name__ == "__main__":
    predictor = main()