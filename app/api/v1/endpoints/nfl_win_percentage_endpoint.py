from fastapi import APIRouter
import joblib
import pandas as pd
import os

router = APIRouter()

MODEL_PATH = os.path.join('models', 'nfl_win_percentage_xgb.pkl')
GAMES_CSV_PATH = os.path.join('app', 'data', 'NFL', 'game_data(2025).csv')

@router.get('/win-percentages', tags=["NFL"])
def get_team_win_percentages():
    # Load the trained model
    model = joblib.load(MODEL_PATH)
    # Load the data
    data = pd.read_csv(GAMES_CSV_PATH)

    # Create home perspective rows
    home_df = data[['home_team', 'away_team', 'home_score', 'away_score', 
                    'home_first_downs_@total', 'away_first_downs_@total',
                    'home_yards_@total', 'away_yards_@total',
                    'home_turnovers_@total', 'away_turnovers_@total',
                    'home_posession_@total', 'away_posession_@total']].copy()
    home_df.columns = ['team', 'opponent', 'team_score', 'opponent_score', 
                       'first_downs', 'opponent_first_downs',
                       'total_yards', 'opponent_yards',
                       'turnovers', 'opponent_turnovers',
                       'possession_time', 'opponent_possession']
    home_df['is_home'] = 1
    home_df['won'] = (home_df['team_score'] > home_df['opponent_score']).astype(int)

    # Create away perspective rows
    away_df = data[['away_team', 'home_team', 'away_score', 'home_score',
                    'away_first_downs_@total', 'home_first_downs_@total',
                    'away_yards_@total', 'home_yards_@total',
                    'away_turnovers_@total', 'home_turnovers_@total',
                    'away_posession_@total', 'home_posession_@total']].copy()
    away_df.columns = ['team', 'opponent', 'team_score', 'opponent_score',
                       'first_downs', 'opponent_first_downs',
                       'total_yards', 'opponent_yards',
                       'turnovers', 'opponent_turnovers',
                       'possession_time', 'opponent_possession']
    away_df['is_home'] = 0
    away_df['won'] = (away_df['team_score'] > away_df['opponent_score']).astype(int)

    # Combine both views
    combined_df = pd.concat([home_df, away_df], ignore_index=True)

    # Convert possession time to seconds
    def time_to_seconds(time_str):
        try:
            if pd.isna(time_str):
                return 0
            minutes, seconds = str(time_str).split(':')
            return int(minutes) * 60 + int(seconds)
        except:
            return 0

    combined_df['possession_time'] = pd.Series(combined_df['possession_time']).apply(time_to_seconds)

    # Build mapping from code to team name using all unique team names
    all_teams = pd.unique(data[['home_team', 'away_team']].values.ravel('K'))
    team_name_to_code = {name: i for i, name in enumerate(sorted(all_teams))}
    code_to_team_name = {v: k for k, v in team_name_to_code.items()}

    # Encode team as numeric categories using this mapping
    combined_df['team'] = pd.Series(combined_df['team']).map(team_name_to_code)
    combined_df['opponent'] = pd.Series(combined_df['opponent']).map(team_name_to_code)

    features = ['team', 'opponent', 'first_downs', 'total_yards', 'turnovers', 'possession_time', 'is_home']
    X = combined_df[features]
    
    # Predict win probabilities for each team-game
    win_probs = model.predict_proba(X)[:, 1]
    combined_df['win_probability'] = win_probs

    results = []
    for code in sorted(code_to_team_name.keys()):
        team_name = code_to_team_name[code]
        team_games = combined_df[combined_df['team'] == code]
        total_matches = len(team_games)
        win_count = team_games['won'].sum()
        loss_count = total_matches - win_count
        win_percentage = team_games['win_probability'].mean() if total_matches > 0 else None
        
        # Calculate additional stats
        avg_score = team_games['team_score'].mean() if total_matches > 0 else None
        # avg_yards = team_games['total_yards'].mean() if total_matches > 0 else None
        # avg_turnovers = team_games['turnovers'].mean() if total_matches > 0 else None
        
        results.append({
            'team_name': str(team_name),
            'win_percentage': float(win_percentage) if win_percentage is not None else None,
            'total_matches': int(total_matches),
            'win_count': int(win_count),
            'loss_count': int(loss_count),
            'average_score': float(avg_score) if avg_score is not None else None,
            # 'average_yards': float(avg_yards) if avg_yards is not None else None,
            # 'average_turnovers': float(avg_turnovers) if avg_turnovers is not None else None
        })
    
    return results 