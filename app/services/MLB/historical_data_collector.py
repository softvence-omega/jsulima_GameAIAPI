import os
from dotenv import load_dotenv
from datetime import datetime
import pandas as pd
from app.config import DATA_DIR

from app.services.MLB.teamData_processor import BaseballDataProcessor

# Load environment variables
load_dotenv()
API_KEY = os.getenv('GOALSERVE_API_KEY')
data_processor = BaseballDataProcessor(API_KEY)

def fetch_historical_season_data(year):
    """Fetch complete season data for model training"""
    # print(f"Fetching historical data for {year} season...")
    
    # MLB season typically runs April-October
    start_date = f"{year}-01-01"
    end_date = f"{year}-12-31"

    historical_data = []
    date_range = pd.date_range(start=start_date, end=end_date)
    
    for date in date_range:
        date_str = date.strftime("%d.%m.%Y")
        try:
            response = data_processor.fetch_data("baseball/usa", {"date": date_str})
            # standings = data_processor.fetch_data("baseball/mlb_standings") # for single day , standings data is same
            # collect standing from csv
            

            if response and 'scores' in response:
                category = response['scores'].get('category', {})
                matches = category.get('match', [])
                matches = [matches] if isinstance(matches, dict) else matches
                i=0
                for game in matches:
                    i+=1
                    # print(f"match----------{i}")
                    try:
                        features = data_processor.extract_predictive_features(game)
                        # print("features----------", features)
                        if features:
                            home_score = int(game.get('hometeam', {}).get('@totalscore', 0))
                            away_score = int(game.get('awayteam', {}).get('@totalscore', 0))

                            features['home_score'] = home_score
                            features['away_score'] = away_score
                            features['home_win'] = 1 if home_score > away_score else 0
                            historical_data.append(features)
                    except Exception as e:
                        # print(f"Skipping game on {date_str}: {str(e)}")
                        continue
                        
        except Exception as e:
            # print(f"Error fetching data for {date_str}: {str(e)}")
            continue
    
    return pd.DataFrame(historical_data) if historical_data else pd.DataFrame()

def HistoricalDataCollector(years=[2019, 2021]):
    """Train model on multiple historical seasons"""
    all_seasons_data = []
    
    for year in years:
        historical_data = fetch_historical_season_data(year)
        if not historical_data.empty:
            all_seasons_data.append(historical_data)
    
    if not all_seasons_data:
        raise ValueError("No historical data found for training")
    
    df = pd.concat(all_seasons_data, ignore_index=True)
    # print(f"Final training dataset shape: {df.shape}")
    # print("datatypes-----------", df.dtypes)
    #print("dataframe-----", df)
    csv_dir = os.path.join(DATA_DIR, "MLB")
    os.makedirs(csv_dir, exist_ok=True)
    csv_path = os.path.join(csv_dir, 'mlb_historical_data(2010-2013).csv')
    df.to_csv(csv_path, index=False)
    # print(f"Historical data saved to {csv_path}")

if __name__ == "__main__":
    # Example usage
    HistoricalDataCollector(years=[2010,2013])