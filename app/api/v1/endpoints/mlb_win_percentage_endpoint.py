from fastapi import APIRouter
import joblib
import pandas as pd
import os
from datetime import datetime

router = APIRouter()

MODEL_PATH = os.path.join('models', 'MLB', 'mlb_win_percentage_xgb.pkl')
GAMES_CSV_PATH = os.path.join('app', 'data', 'MLB', 'game_data', 'games_data_combined.csv')

@router.get('/win-percentages', tags=["MLB"])
def get_team_win_percentages():
    current_year = datetime.now().date().year
    # Load the trained model
    model = joblib.load(MODEL_PATH)
    # Load the data
    data = pd.read_csv(GAMES_CSV_PATH)
    data = data[(data['date'].str.endswith(str(current_year))) & (data['status'] == 'Final')]

    # Create home perspective rows
    home_df = data[['home_team', 'away_team', 'home_score', 'away_score', 'home_hits', 'home_errors',
                    'away_hits', 'away_errors', 'venue_name', 'attendance']].copy()
    home_df.columns = ['team', 'opponent', 'team_score', 'opponent_score', 'team_hits', 'team_errors',
                       'opponent_hits', 'opponent_errors', 'venue_name', 'attendance']
    home_df['is_home'] = 1
    home_df['won'] = (home_df['team_score'] > home_df['opponent_score']).astype(int)

    # Create away perspective rows
    away_df = data[['away_team', 'home_team', 'away_score', 'home_score', 'away_hits', 'away_errors',
                    'home_hits', 'home_errors', 'venue_name', 'attendance']].copy()
    away_df.columns = ['team', 'opponent', 'team_score', 'opponent_score', 'team_hits', 'team_errors',
                       'opponent_hits', 'opponent_errors', 'venue_name', 'attendance']
    away_df['is_home'] = 0
    away_df['won'] = (away_df['team_score'] > away_df['opponent_score']).astype(int)

    # Combine both views
    combined_df = pd.concat([home_df, away_df], ignore_index=True)

    # Build mapping from code to team name using all unique team names
    all_teams = pd.unique(data[['home_team', 'away_team']].values.ravel('K'))
    team_name_to_code = {name: i for i, name in enumerate(sorted(all_teams))}
    code_to_team_name = {v: k for k, v in team_name_to_code.items()}

    # Encode team and venue as numeric categories using this mapping
    combined_df['team'] = pd.Series(combined_df['team']).map(team_name_to_code)
    combined_df['opponent'] = pd.Series(combined_df['opponent']).map(team_name_to_code)
    combined_df['venue_name'] = pd.Series(combined_df['venue_name']).astype('category').cat.codes

    features = ['team', 'opponent', 'team_hits', 'team_errors', 'opponent_hits',
                'opponent_errors', 'venue_name', 'attendance', 'is_home']
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
        results.append({
            'team_name': str(team_name),
            'win_percentage': float(win_percentage) if win_percentage is not None else None,
            'total_matches': int(total_matches),
            'win_count': int(win_count),
            'loss_count': int(loss_count),
            'average_score': float(team_games['team_score'].mean()) if total_matches > 0 else None
        })
    return results 