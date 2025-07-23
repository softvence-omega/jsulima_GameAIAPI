import os
from app.config import DATA_DIR
import pandas as pd

csv_dir = os.path.join(DATA_DIR, "MLB")
os.makedirs(csv_dir, exist_ok=True)
csv_file_path = os.path.join(csv_dir, 'mlb_player_stats.csv')
df= pd.read_csv(csv_file_path)
print(df[df['team']=='Colorado Rockies'])
# print("df------\n", df.columns)
print("df------\n", df['team'].value_counts().index)
########### Features ######################
# ['game_id', 'game_date', 'venue_name', 'player_id', 'player_name',
#        'home_away', 'team', 'team_id', 'opponent_team', 'opponent_team_id',
#        'opponent_team_batting_avg', 'opponent_team_ops',
#        'opponent_team_runs_per_game', 'opponent_team_hr_per_game',
#        'innings_pitched', 'runs', 'hits_allowed', 'earned_runs', 'walks',
#        'strikeouts', 'home_runs_allowed', 'hit_by_pitch',
#        'earned_runs_average', 'pitch_count', 'whip', 'k_per_9', 'bb_per_9']

# csv_file_path = os.path.join(csv_dir, 'batter_stats_data(2010-2024).csv')
# df= pd.read_csv(csv_file_path)
# print("df------\n", df.columns)
############ Features ##################
# ['game_id', 'game_date', 'venue_name', 'player_id', 'player_name',
#        'position', 'home_away', 'team', 'team_id', 'opponent_team',
#        'opponent_team_id', 'at_bats', 'runs', 'hits', 'doubles', 'triples',
#        'home_runs', 'rbis', 'walks', 'strikeouts', 'stolen_bases',
#        'caught_stealing', 'hit_by_pitch', 'sac_fly', 'batting_average',
#        'on_base_percentage', 'slugging_percentage', 'opponent_pitcher_era',
#        'opponent_pitcher_whip', 'opponent_pitcher_k9', 'opponent_pitcher_hand',
#        'ballpark_hr_factor', 'ballpark_hit_factor', 'total_bases', 'season']
