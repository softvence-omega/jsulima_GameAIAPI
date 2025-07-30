import os
import pandas as pd
from datetime import datetime
from app.services.NFL.data_processor import NflDataProcessor
from app.services.NFL.player_permonance_processor import NflPlayerPerformanceProcessor

processor = NflDataProcessor()
performance_processor = NflPlayerPerformanceProcessor()

def fetch_historical_player_data(start_date_str: str, end_date_str: str = None) -> pd.DataFrame:
    if end_date_str is None:
        end_date_str = start_date_str

    start_date = datetime.strptime(start_date_str, "%d.%m.%Y")
    end_date = datetime.strptime(end_date_str, "%d.%m.%Y")
    all_player_features = []

    for single_date in pd.date_range(start_date, end_date):
        date_str = single_date.strftime("%d.%m.%Y")
        #print(f"Processing date: {date_str}")

        try:
            raw_data = processor.fetch_data(f"football/nfl-scores?date={date_str}")
            matches = raw_data.get('scores', {}).get('category', {}).get('match', [])
            if isinstance(matches, dict):
                matches = [matches]

            #print(f"Matches found: {len(matches)}")

            for match in matches:
                match_info = performance_processor.extract_match_info(match)
                match_id = match_info.get('match_id', '')
                #print(f"🔍 Processing match ID: {match_id}")
                venue_name = match_info.get('venue_name', '')
                #print(f"🏟️ Venue: {venue_name}, Match ID: {match_id}")
                for role_key in ['hometeam', 'awayteam']:
                    team_info = match_info.get(role_key, {})
                    team_name = team_info.get('name', '')
                    team_id = team_info.get('id', '')
                    #print(f"🔍 Processing {role_key} team data...")
                    if not team_name:
                        #print(f"Missing {role_key} team name, skipping...")
                        continue

                    if not team_id:
                        #print(f" Missing {role_key} team ID, skipping...")
                        continue

                    #print(f"Processing {role_key.upper()} team: {team_name} (ID: {team_id})")

                    # Fetch stats, roster, and injuries data for the team
                    stats_data = performance_processor.fetch_player_stats(team_id)
                    roster_data = performance_processor.fetch_player_roster(team_id)
                    injuries_data = performance_processor.fetch_injuries(team_id)

                    if not stats_data or not roster_data:
                        #print(f"Missing stats or roster data for team {team_id} on {date_str}")
                        continue

                    # Generate full player performance summary with fantasy scores & injury status
                    player_perf_df = performance_processor.generate_player_performance_summary(
                        match_data = match,
                        role=role_key
                    )

                    if player_perf_df.empty:
                        #print(f"No player performance data for team {team_id} on {date_str}")
                        continue

                    #print(f"✅ Extracted {len(player_perf_df)} player records for {team_name} on {date_str}")

                    # Add match metadata
                    player_perf_df['match_id'] = match_id
                    player_perf_df['venue_name'] = venue_name
                    player_perf_df['team_id'] = team_id
                    player_perf_df['team_name'] = team_name
                    player_perf_df['team_role'] = role_key
                    player_perf_df['date'] = date_str

                    all_player_features.append(player_perf_df)

        except Exception as e:
            raise Exception(f"Error processing data for {date_str}: {str(e)}")
            #print(f"Error processing data for {date_str}: {e}")

    if all_player_features:
        full_df = pd.concat(all_player_features, ignore_index=True)
        save_dir = "data/NFL"
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"historical_player_features_{start_date.strftime('%Y%m%d')}_to_{end_date.strftime('%Y%m%d')}.csv")
        full_df.to_csv(save_path, index=False)
        #print(f"Saved all historical player features to: {save_path}")
        return full_df

    #print("No player features collected.")
    return pd.DataFrame()

# Example usage:
if __name__ == "__main__":
    # Fetch historical player data for the year 2020    
    fetch_historical_player_data("01.01.2020", "31.02.2020")
    # You can also specify a date range
    
    #print("Data collection complete.")
