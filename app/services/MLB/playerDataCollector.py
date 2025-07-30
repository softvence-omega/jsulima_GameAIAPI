import os
from dotenv import load_dotenv
from datetime import datetime
import pandas as pd
from app.config import DATA_DIR

from app.services.MLB.teamData_processor import BaseballDataProcessor
from app.services.MLB.playerDataProcessor import PlayerDataProcessor


# Load environment variables
load_dotenv()
API_KEY = os.getenv('GOALSERVE_API_KEY')
data_processor = BaseballDataProcessor(API_KEY)

PlayerDataProcessor = PlayerDataProcessor()

def extract_data(game):
    #print"call extract_data---------------")
    game_data = PlayerDataProcessor.extract_game_info(game)
    if not game_data['venue_id']:
        raise ValueError("Venue ID is missing in game data. Cannot extract stats without venue information.")
    
    # #print'game_data----------------',game_data)
    pitcher_stats = PlayerDataProcessor.extract_pitcher_stats(game, game_data)
    # #print'pitcher_stats----------------',pitcher_stats)
    batter_stats = PlayerDataProcessor.extract_batter_stats(game, game_data)
    # #print'batter_stats----------------',batter_stats)
    return game_data, pitcher_stats, batter_stats


def fetch_season_data(year):
    """Fetch complete season data for model training"""
    #printf"Fetching historical data for {year} season...")
    
    # MLB season typically runs April-October
    start_date = f"{year}-03-01"
    end_date = f"{year}-12-01"

    all_games = []
    all_pitcher_stats = []
    all_batter_stats = []
    date_range = pd.date_range(start=start_date, end=end_date)

    for date in date_range:
        #print"date---------", date)
        date_str = date.strftime("%d.%m.%Y")
        try:
            response = data_processor.fetch_data("baseball/usa", {"date": date_str})
            
            if response and 'scores' in response:
                category = response['scores'].get('category', {})
                matches = category.get('match', [])
                matches = [matches] if isinstance(matches, dict) else matches
                i=0
                for game in matches:
                    i+=1
                    #printf"match----------{i}")
                    try:
                        game_data, pitcher_stats, batter_stats = extract_data(game)
                        all_games.append(game_data)
                        all_pitcher_stats.extend(pitcher_stats)
                        all_batter_stats.extend(batter_stats)
                        
                    except Exception as e:
                        #printf"Skipping game on {date_str}: {str(e)}")
                        continue
                        
        except Exception as e:
            #printf"Error fetching data for {date_str}: {str(e)}")
            continue
    
    # Convert to DataFrames
    games_df = pd.DataFrame(all_games)
    pitcher_stats_df = pd.DataFrame(all_pitcher_stats)
    batter_stats_df = pd.DataFrame(all_batter_stats)

    # Convert date columns
    if not games_df.empty and 'date' in games_df:
        games_df['game_date'] = pd.to_datetime(games_df['date'], dayfirst=True)
    if not pitcher_stats_df.empty and 'game_date' in pitcher_stats_df:
        pitcher_stats_df['game_date'] = pd.to_datetime(pitcher_stats_df['game_date'], dayfirst=True)
    if not batter_stats_df.empty and 'game_date' in batter_stats_df:
        batter_stats_df['game_date'] = pd.to_datetime(batter_stats_df['game_date'], dayfirst=True)
    
    # season column
    if not games_df.empty:
        games_df['season'] = year
    if not pitcher_stats_df.empty:
        pitcher_stats_df['season'] = year
    if not batter_stats_df.empty:
        batter_stats_df['season'] = year


    return games_df, pitcher_stats_df, batter_stats_df

def PlayerDataCollector(years):
    """Train model on multiple historical seasons"""
    game_data = []
    pitcher_stats_data=[]
    batter_stats_data=[]
    
    for year in years:
        games_df, pitcher_stats_df, batter_stats_df = fetch_season_data(year)
        game_data.append(games_df)
        pitcher_stats_data.append(pitcher_stats_df)
        batter_stats_data.append(batter_stats_df)
    
    game_df = pd.concat(game_data, ignore_index=True)
    pitcher_df = pd.concat(pitcher_stats_data, ignore_index=True)
    batter_df = pd.concat(batter_stats_data, ignore_index=True)

    #printf"Final training dataset shape of pitcher: {pitcher_df.shape}")
    
    #print("dataframe-----", df)
    dir = os.path.join(DATA_DIR, "MLB")
    os.makedirs(dir, exist_ok=True)
    
    game_df.to_csv(os.path.join(dir, 'games_data(2010-2024).csv'), index=False)
    pitcher_df.to_csv(os.path.join(dir, 'pitcher_stats_data(2010-2024).csv'), index=False)
    batter_df.to_csv(os.path.join(dir, 'batter_stats_data(2010-2024).csv'), index=False)

    # print(f"Historical data saved!")

if __name__ == "__main__":
    # Example usage
    PlayerDataCollector(years=list(range(2010, 2025)))