from datetime import datetime, timedelta
import os
import pandas as pd
from  app.config import NFL_DIR
from app.services.NFL.data_processor import NflDataProcessor

processor = NflDataProcessor()

def append_yesterday_game_if_season_active():
    today = datetime.now()
    year, month = today.year, today.month
    
    

    # ✅ Offseason (March–June) → do nothing
    if 3 <= month <= 6:
        print("ℹ️ Offseason (March–June), skipping update!")
        return
    start_year = year
    end_year = year-1
    start_date = datetime.strptime(f"28.02.{end_year}", "%d.%m.%Y")
    end_date = datetime.strptime(f"01.07.{start_year}", "%d.%m.%Y")

    historical_features = []

    for single_date in pd.date_range(start_date, end_date):
        date_str = single_date.strftime("%d.%m.%Y")
        try:
            raw_data = processor.fetch_data(f"football/nfl-scores?date={date_str}")
            matches = raw_data.get('scores', {}).get('category', {})
            if isinstance(matches, dict):
                matches = [matches]

            for category in matches:
                games = category.get('match', [])
                if isinstance(games, dict):
                    games = [games]

                for i, game in enumerate(games):
                    print(f"Fetching match {i} on {date_str}")
                    print(f"Game type: {type(game)}, Game content: {game}")
                    if not isinstance(game, dict):
                        print(f"⚠️ Skipping non-dict game object at index {i}: {game}")
                        continue
                    features = processor.extract_predictive_features(game)
                
                    if features:
                        home_score = int(game.get('hometeam', {}).get('@totalscore', 0))
                        away_score = int(game.get('awayteam', {}).get('@totalscore', 0))
                        features['home_score'] = home_score
                        features['away_score'] = away_score
                        features['home_win'] = 1 if home_score > away_score else 0
                        historical_features.append(features)
        except Exception as e:
            print(f"Error on {date_str}: {e}")
            continue

    df_new = pd.DataFrame(historical_features)

    if df_new.empty:
        print("ℹ️ No new games for yesterday — skipping CSV update")
        return

    # ✅ Merge with existing CSV (fixed file name)
    filename = "head-to-head.csv"
    csv_path = os.path.join(NFL_DIR, filename)

    if os.path.exists(csv_path):
        df_existing = pd.read_csv(csv_path)
        df_new = df_new[~df_new['game_id'].isin(df_existing['game_id'])]
        if df_new.empty:
            print("ℹ️ CSV already up to date — no new games to add!")
            return
        df_final = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df_final = df_new

    # ✅ Save CSV
    df_final.to_csv(csv_path, index=False)
    print(f"✅ CSV updated successfully — total games: {len(df_final)}")

# Run this daily
append_yesterday_game_if_season_active()