# Updated top_performer4.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings
from typing import Dict, List, Tuple, Any
import joblib

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=pd.errors.SettingWithCopyWarning)

class NFLPerformanceAnalyzer:
    """NFL Player Performance Analyzer with ML-based top performer prediction"""
    
    def __init__(self):
        self.positions = ['QB', 'RB', 'WR', 'TE', 'LB']
        self.stat_columns = [
            'passing_yards', 'passing_touchdowns', 'interceptions', 
            'rushing_yards', 'rushing_touchdowns', 'receiving_yards', 
            'receiving_touchdowns', 'total_tackles', 'sacks', 'fumbles',
            'forced_fumbles', 'fumble_recoveries', 'defensive_touchdowns'
        ]
    
    def calculate_performance_score(self, row: pd.Series) -> Tuple[float, Dict]:
        """Calculate performance score using standard fantasy football scoring"""
        
        position = row.get('player_position', '')
        
        # Helper function to safely get numeric values
        def get_stat(stat_name: str, default: float = 0.0) -> float:
            value = row.get(stat_name, default)
            return float(value) if pd.notna(value) else default
        
        if position == 'QB':
            # Quarterback scoring
            passing_yards = get_stat('passing_yards')
            passing_tds = get_stat('passing_touchdowns')
            interceptions = get_stat('interceptions')
            rushing_yards = get_stat('rushing_yards')
            rushing_tds = get_stat('rushing_touchdowns')
            fumbles = get_stat('fumbles')
            
            performance_score = (
                (passing_yards / 25) +      # 1 point per 25 passing yards
                (passing_tds * 4) +         # 4 points per passing TD
                (rushing_yards / 10) +      # 1 point per 10 rushing yards
                (rushing_tds * 6) -         # 6 points per rushing TD
                (interceptions * 2) -       # -2 points per interception
                (fumbles * 2)               # -2 points per fumble
            )
            
            stats_used = {
                'passing_yards': passing_yards,
                'passing_touchdowns': passing_tds,
                'rushing_yards': rushing_yards,
                'rushing_touchdowns': rushing_tds,
                'interceptions': interceptions,
                'fumbles': fumbles,
                'total_touchdowns': passing_tds + rushing_tds
            }
            
        elif position == 'RB':
            # Running back scoring
            rushing_yards = get_stat('rushing_yards')
            rushing_tds = get_stat('rushing_touchdowns')
            receiving_yards = get_stat('receiving_yards')
            receiving_tds = get_stat('receiving_touchdowns')
            fumbles = get_stat('fumbles')
            
            performance_score = (
                (rushing_yards / 10) +      # 1 point per 10 rushing yards
                (rushing_tds * 6) +         # 6 points per rushing TD
                (receiving_yards / 10) +    # 1 point per 10 receiving yards
                (receiving_tds * 6) -       # 6 points per receiving TD
                (fumbles * 2)               # -2 points per fumble
            )
            
            stats_used = {
                'rushing_yards': rushing_yards,
                'rushing_touchdowns': rushing_tds,
                'receiving_yards': receiving_yards,
                'receiving_touchdowns': receiving_tds,
                'fumbles': fumbles,
                'total_touchdowns': rushing_tds + receiving_tds
            }
            
        elif position in ['WR', 'TE']:
            # Wide receiver and tight end scoring
            receiving_yards = get_stat('receiving_yards')
            receiving_tds = get_stat('receiving_touchdowns')
            rushing_yards = get_stat('rushing_yards')
            rushing_tds = get_stat('rushing_touchdowns')
            fumbles = get_stat('fumbles')
            
            performance_score = (
                (receiving_yards / 10) +    # 1 point per 10 receiving yards
                (receiving_tds * 6) +       # 6 points per receiving TD
                (rushing_yards / 10) +      # 1 point per 10 rushing yards (trick plays)
                (rushing_tds * 6) -         # 6 points per rushing TD
                (fumbles * 2)               # -2 points per fumble
            )
            
            stats_used = {
                'receiving_yards': receiving_yards,
                'receiving_touchdowns': receiving_tds,
                'rushing_yards': rushing_yards,
                'rushing_touchdowns': rushing_tds,
                'fumbles': fumbles,
                'total_touchdowns': receiving_tds + rushing_tds
            }
            
        elif position in ['LB', 'DB', 'DL', 'DE', 'DT', 'CB', 'S']:
            # Defensive player scoring
            tackles = get_stat('total_tackles')
            interceptions = get_stat('interceptions')
            sacks = get_stat('sacks')
            forced_fumbles = get_stat('forced_fumbles')
            fumble_recoveries = get_stat('fumble_recoveries')
            defensive_tds = get_stat('defensive_touchdowns')
            
            performance_score = (
                (tackles * 1) +             # 1 point per tackle
                (interceptions * 6) +       # 6 points per interception
                (sacks * 4) +               # 4 points per sack
                (forced_fumbles * 4) +      # 4 points per forced fumble
                (fumble_recoveries * 2) +   # 2 points per fumble recovery
                (defensive_tds * 6)         # 6 points per defensive TD
            )
            
            stats_used = {
                'total_tackles': tackles,
                'interceptions': interceptions,
                'sacks': sacks,
                'forced_fumbles': forced_fumbles,
                'fumble_recoveries': fumble_recoveries,
                'defensive_touchdowns': defensive_tds,
                'total_touchdowns': defensive_tds
            }
            
        else:
            # Unknown position
            performance_score = 0.0
            stats_used = {'total_touchdowns': 0}
        
        return performance_score, stats_used
    
    def load_and_prepare_data(self, home_team: str, away_team: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load and prepare player data for both teams"""
        
        try:
            # Load datasets
            player_stats = pd.read_csv('app/data/NFL/all_player_stats.csv')
            player_info = pd.read_csv('app/data/NFL/player_info.csv')
            
            # Merge datasets
            merged_data = pd.merge(
                player_stats, 
                player_info, 
                left_on='id', 
                right_on='player_id', 
                how='inner'
            )
            
            # Convert stat columns to numeric
            for col in self.stat_columns:
                if col in merged_data.columns:
                    merged_data[col] = pd.to_numeric(merged_data[col], errors='coerce')
            
            # Filter for target positions and teams
            home_data = merged_data[
                (merged_data['player_position'].isin(self.positions)) & 
                (merged_data['team_name'] == home_team)
            ].copy()
            
            away_data = merged_data[
                (merged_data['player_position'].isin(self.positions)) & 
                (merged_data['team_name'] == away_team)
            ].copy()
            
            # Remove duplicates
            home_data = home_data.drop_duplicates(subset=["player_name", "player_position"])
            away_data = away_data.drop_duplicates(subset=["player_name", "player_position"])
            
            return home_data, away_data
            
        except Exception as e:
            print(f"Error loading data: {e}")
            return pd.DataFrame(), pd.DataFrame()
    
    def create_placeholder_player(self, team_name: str, position: str) -> Dict:
        """Create placeholder for missing position"""
        return {
            'team_name': team_name,
            'player_name': f"No player in {position} position",
            'player_status': 'Not Active',
            'player_position': position,
            'performance_score': 0.0,
            'confidence_score': 0.0,
            **{f"{col}_used": 0 for col in self.stat_columns},
            'total_touchdowns_used': 0
        }
    
    def handle_outliers_iqr(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """Handle outliers using the IQR method by capping them."""
        df_copy = df.copy()
        for col in columns:
            if col in df_copy.columns and pd.api.types.is_numeric_dtype(df_copy[col]):
                Q1 = df_copy[col].quantile(0.25)
                Q3 = df_copy[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                # Cap outliers using the clip method
                df_copy[col] = df_copy[col].clip(lower=lower_bound, upper=upper_bound)
        return df_copy
    
    def predict_top_performer(self, position_data: pd.DataFrame) -> Tuple[pd.Series, float]:
        """Use ML to predict the top performer for a position"""
        
        if len(position_data) == 0:
            return pd.Series(), 0.0
        
        # Calculate performance scores
        scores_and_stats = position_data.apply(
            lambda row: pd.Series(self.calculate_performance_score(row)), 
            axis=1
        )
        position_data = position_data.copy()
        position_data['performance_score'] = scores_and_stats[0].fillna(0)
        position_data['stats_used'] = scores_and_stats[1]
        
        # If only one player or all have same score, return the best one
        if len(position_data) <= 1 or position_data['performance_score'].nunique() <= 1:
            idx = position_data['performance_score'].idxmax()
            top_player = position_data.loc[idx].copy()
            confidence = 1.0 if top_player['performance_score'] > 0 else 0.0
            return top_player, confidence
        
        # Create labels for ML (top performer = 1, others = 0)
        max_score = position_data['performance_score'].max()
        position_data['is_top_performer'] = (position_data['performance_score'] == max_score).astype(int)

        # Handle outliers before training the model
        position_data = self.handle_outliers_iqr(position_data, self.stat_columns)
        
        # Prepare features for ML
        feature_cols = [col for col in self.stat_columns if col in position_data.columns]
        X = position_data[feature_cols].fillna(0)
        y = position_data['is_top_performer']
        
        # If insufficient data for ML, return player with highest score
        if len(position_data) < 3 or y.nunique() == 1:
            idx = position_data['performance_score'].idxmax()
            top_player = position_data.loc[idx].copy()
            confidence = 1.0 if top_player['performance_score'] > 0 else 0.0
            return top_player, confidence        

        # Check if all classes in y have at least 2 samples for stratification
        class_counts = y.value_counts()
        can_stratify = class_counts.min() >= 2

        try:
            # Train ML model
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y if can_stratify else None
            )

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_scaled = scaler.transform(X)
            
            rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            rf.fit(X_train_scaled, y_train)

            # Save the model and scaler
            joblib.dump(rf, 'models/nfl_top_performer_model.pkl')

            # Get prediction probabilities
            if len(rf.classes_) > 1 and 1 in rf.classes_:
                class_idx = list(rf.classes_).index(1)
                probabilities = rf.predict_proba(X_scaled)[:, class_idx]
            else:
                probabilities = np.ones(len(X_scaled)) * (1.0 if rf.classes_[0] == 1 else 0.0)
            
            # Select player with highest confidence
            best_idx = np.argmax(probabilities)
            top_player = position_data.iloc[best_idx].copy()
            confidence = probabilities[best_idx]
            
            return top_player, confidence
            
        except Exception as e:
            print(f"ML prediction error for position: {e}")
            # Fallback to highest scoring player
            idx = position_data['performance_score'].idxmax()
            top_player = position_data.loc[idx].copy()
            confidence = 1.0 if top_player['performance_score'] > 0 else 0.0
            return top_player, confidence
    
    def process_team_data(self, team_data: pd.DataFrame, team_name: str) -> List[Dict]:
        """Process team data and return top performers for each position"""
        
        top_performers = []
        
        for position in self.positions:
            position_data = team_data[team_data['player_position'] == position].copy()
            
            if len(position_data) == 0:
                # No players for this position
                top_performers.append(self.create_placeholder_player(team_name, position))
                continue
            
            # Predict top performer using ML
            top_player, confidence = self.predict_top_performer(position_data)
            
            if top_player.empty:
                # Fallback to placeholder
                top_performers.append(self.create_placeholder_player(team_name, position))
                continue
            
            # Extract stats used from the stored dictionary
            stats_used = top_player.get('stats_used', {})
            
            # Build result dictionary
            result = {
                'team_name': team_name,
                'player_name': top_player.get('player_name', 'Unknown'),
                'player_status': top_player.get('player_status', 'roster'),
                'player_position': position,
                'performance_score': float(top_player.get('performance_score', 0)),
                'confidence_score': float(confidence)
            }
            
            # Add stats used
            for col in self.stat_columns:
                result[f"{col}_used"] = stats_used.get(col, 0)
            result['total_touchdowns_used'] = stats_used.get('total_touchdowns', 0)
            
            top_performers.append(result)
        
        return top_performers

def get_top_performers(home_team_name: str, away_team_name: str) -> Dict[str, Any]:
    """
    Main function to get top performers for both teams
    
    Args:
        home_team_name: Name of the home team
        away_team_name: Name of the away team
    
    Returns:
        Dictionary with the specified output format
    """
    
    analyzer = NFLPerformanceAnalyzer()
    
    # Load and prepare data
    home_data, away_data = analyzer.load_and_prepare_data(home_team_name, away_team_name)
    
    if home_data.empty and away_data.empty:
        print("No data found for the specified teams")
        return {
            "top_performers": {
                "hometeam": {
                    "team_name": home_team_name,
                    "players": []
                },
                "awayteam": {
                    "team_name": away_team_name,
                    "players": []
                }
            }
        }
    
    # Process both teams
    home_performers = []
    away_performers = []
    
    if not home_data.empty:
        home_performers = analyzer.process_team_data(home_data, home_team_name)
    
    if not away_data.empty:
        away_performers = analyzer.process_team_data(away_data, away_team_name)
    
    # Clean up any inf or nan values
    for performer in home_performers + away_performers:
        for key, value in performer.items():
            if isinstance(value, (int, float)):
                if np.isinf(value) or np.isnan(value):
                    performer[key] = 0.0
    
    # Format output according to specifications
    result = {
        "top_performers": {
            "hometeam": {
                "team_name": home_team_name,
                "players": home_performers
            },
            "awayteam": {
                "team_name": away_team_name,
                "players": away_performers
            }
        }
    }
    
    return result

# Example usage
if __name__ == "__main__":
    # Test the function
    try:
        results = get_top_performers("Atlanta Falcons", "Arizona Cardinals")
        
        print("Top Performers:")
        print("-" * 80)
        
        # Print home team
        home_team = results["top_performers"]["hometeam"]
        print("Home Team_name:",home_team['team_name'])
        for performer in home_team["players"]:
            print(f"  Player: {performer['player_name']} ({performer['player_position']})")
            print(f"  Performance Score: {performer['performance_score']:.1f}")
            print(f"  Confidence: {performer['confidence_score']:.2f}")
            print("-" * 40)
        
        # Print away team
        away_team = results["top_performers"]["awayteam"]
        print("Away Team:", away_team['team_name'])
        for performer in away_team["players"]:
            print(f"  Player: {performer['player_name']} ({performer['player_position']})")
            print(f"  Performance Score: {performer['performance_score']:.1f}")
            print(f"  Confidence: {performer['confidence_score']:.2f}")
            print("-" * 40)
            
    except Exception as e:
        print(f"Error running example: {e}")