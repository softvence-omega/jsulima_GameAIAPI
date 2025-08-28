from fastapi import APIRouter, HTTPException
import pandas as pd
import os
from typing import List, Dict, Any
from datetime import datetime


router = APIRouter()

@router.get('/win-percentages', tags=["NFL"])
def get_team_win_percentages() -> List[Dict[str, Any]]:
    """
    Calculate NFL team win percentages, wins, losses, draws, and average points
    using the nfl_games_data_history.csv file.
    """
    try:
        # Path to the CSV file
        csv_file_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'data', 'NFL', 'nfl_games_data_history.csv')
        
        # Check if file exists
        if not os.path.exists(csv_file_path):
            raise HTTPException(status_code=404, detail="NFL games data file not found")
        
        # Read the CSV file
        df = pd.read_csv(csv_file_path)
        df = df.tail(1000)

        df['date'] = pd.to_datetime(df['date'], format='%d.%m.%Y')

        today = datetime.now()
        month = datetime.now().month
        year = datetime.now().year
        day : int = datetime.now().day



        if month == 8 and day <= 31:
            start = datetime(year, 7, 31)
            end = datetime(year, 8, 31)
            df = df[(df['date'] >= start) & (df['date'] <= end)]
        elif month >= 9 or month <= 7:
            if month <= 12:
                # i) Filter from 1 Sept to current date
                start = datetime(today.year, 9, 1)
                end = today
                df = df[(df['date'] >= start) & (df['date'] <= end)]
            
            if month < 8:
                # ii) Filter from previous year's Sept 1 to current date
                start = datetime(today.year - 1, 9, 1)
                end = today
                df = df[(df['date'] >= start) & (df['date'] <= end)]


        
        # Initialize results dictionary
        team_stats = {}
        
        # Process each game
        for _, game in df.iterrows():
            home_team = game['home_team_name']
            away_team = game['away_team_name']
            home_score = int(game['home_total_score']) if pd.notna(game['home_total_score']) else 0
            away_score = int(game['away_total_score']) if pd.notna(game['away_total_score']) else 0
            
            # Initialize team stats if not exists
            if home_team not in team_stats:
                team_stats[home_team] = {
                    'games': 0,
                    'wins': 0,
                    'losses': 0,
                    'draws': 0,
                    'total_points_scored': 0,
                    'total_points_allowed': 0
                }
            
            if away_team not in team_stats:
                team_stats[away_team] = {
                    'games': 0,
                    'wins': 0,
                    'losses': 0,
                    'draws': 0,
                    'total_points_scored': 0,
                    'total_points_allowed': 0
                }
            
            # Update home team stats
            team_stats[home_team]['games'] += 1
            team_stats[home_team]['total_points_scored'] += home_score
            team_stats[home_team]['total_points_allowed'] += away_score
            
            # Update away team stats
            team_stats[away_team]['games'] += 1
            team_stats[away_team]['total_points_scored'] += away_score
            team_stats[away_team]['total_points_allowed'] += home_score
            
            # Determine win/loss/draw
            if home_score > away_score:
                # Home team wins
                team_stats[home_team]['wins'] += 1
                team_stats[away_team]['losses'] += 1
            elif away_score > home_score:
                # Away team wins
                team_stats[away_team]['wins'] += 1
                team_stats[home_team]['losses'] += 1
            else:
                # Draw
                team_stats[home_team]['draws'] += 1
                team_stats[away_team]['draws'] += 1
        
        # Calculate final statistics and prepare results
        results = []
        for team_name, stats in team_stats.items():
            total_games = stats['games']
            wins = stats['wins']
            losses = stats['losses']
            draws = stats['draws']
            
            # Calculate win percentage
            if total_games > 0:
                win_percentage = (wins / total_games) 
                avg_points_scored = stats['total_points_scored'] / total_games
                avg_points_allowed = stats['total_points_allowed'] / total_games
            else:
                win_percentage = 0.0
                avg_points_scored = 0.0
                avg_points_allowed = 0.0
            
            results.append({
                'team_name': str(team_name),
                'win_percentage': round(win_percentage, 2),
                'total_matches': int(total_games),
                'win_count': int(wins),
                'loss_count': int(losses),
                'draw_count': int(draws),
                'average_score': round(avg_points_scored, 2),
                'average_points_allowed': round(avg_points_allowed, 2),
                'point_differential': round(avg_points_scored - avg_points_allowed, 2)
            })
        
        # Sort by win percentage (descending)
        results.sort(key=lambda x: x['win_count'], reverse=True)
        
        return results
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred while calculating win percentages: {str(e)}")

