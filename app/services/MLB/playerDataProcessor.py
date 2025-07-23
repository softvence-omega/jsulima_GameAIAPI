import pandas as pd
from app.services.MLB.teamData_processor import BaseballDataProcessor
import os
from dotenv import load_dotenv
load_dotenv()

data_processor = BaseballDataProcessor(os.getenv('GOALSERVE_API_KEY'))

class PlayerDataProcessor:
    def __init__(self):
        self.batter_models = {}
        self.pitcher_models = {}
        self.scalers = {}
        self.label_encoders = {}
        
        # Define target variables for each prediction
        self.batter_targets = ['hits', 'home_runs', 'rbis']
        self.pitcher_targets = ['innings_pitched', 'strikeouts', 'earned_runs']


    def extract_game_info(self, match_data):
            """Extract basic game information from JSON"""
            game_data = {
                'game_id': match_data.get('@id'),
                'date': match_data.get('@formatted_date'),
                'time': match_data.get('@time'),
                'venue_name': match_data.get('@venue_name'),
                'venue_id': match_data.get('@venue_id'),
                'attendance': match_data.get('@attendance'),
                'status': match_data.get('@status'),
                'home_starting_pitcher_id': match_data.get('starting_pitchers', {}).get('hometeam', {}).get('player', {}).get('@id'),
                'away_starting_pitcher_id': match_data.get('starting_pitchers', {}).get('awayteam', {}).get('player', {}).get('@id'),
            }
            
            # Home team info
            hometeam = match_data.get('hometeam', {})
            if hometeam:
                game_data.update({
                    'home_team': hometeam.get('@name'),
                    'home_team_id': hometeam.get('@id'),
                    'home_score': hometeam.get('@totalscore', 0),
                    'home_hits': hometeam.get('@hits', 0),
                    'home_errors': hometeam.get('@errors', 0)
                    
                })
            
            # Away team info
            awayteam = match_data.get('awayteam', {})
            if awayteam:
                game_data.update({
                    'away_team': awayteam.get('@name'),
                    'away_team_id': awayteam.get('@id'),
                    'away_score': awayteam.get('@totalscore', 0),
                    'away_hits': awayteam.get('@hits', 0),
                    'away_errors': awayteam.get('@errors', 0)
                })
            return game_data
    
    def _parse_pitch_count(self, pc_st_string):
        """Parse pitch count string format '94-72' to get total pitches"""
        if not pc_st_string or not isinstance(pc_st_string, str):
            return 0
        try:
            total_pitches, _ = pc_st_string.split('-')
            return int(total_pitches)
        except:
            return 0
    
        
    def _extract_single_pitcher_stats(self, pitcher_data, game_data, home_away,opp_avg, opp_ops, opp_runs_per_game, opp_hr_per_game):
        """Extract individual pitcher statistics"""
        pitcher_record = {
            'game_id': game_data['game_id'],
            'game_date': game_data['date'],
            'venue_name': game_data['venue_name'],
            'player_id': pitcher_data.get('@id'),
            'player_name': pitcher_data.get('@name'),
            'home_away': home_away,
            'team': game_data[f'{home_away}_team'],
            'team_id': game_data[f'{home_away}_team_id'],
            'is_starter': game_data['home_starting_pitcher_id'] == pitcher_data.get('@id') if home_away == 'home' else game_data['away_starting_pitcher_id'] == pitcher_data.get('@id'),
            'opponent_team': game_data['away_team'] if home_away == 'home' else game_data['home_team'],
            'opponent_team_id': game_data['away_team_id'] if home_away == 'home' else game_data['home_team_id'],

            # New opponent pitcher stats
            'opponent_team_batting_avg': opp_avg,
            'opponent_team_ops': opp_ops,
            'opponent_team_runs_per_game': opp_runs_per_game,
            'opponent_team_hr_per_game': opp_hr_per_game,
            
            # Pitching stats
            'innings_pitched': float(pitcher_data.get('@innings_pitched', 0)),
            'runs': int(pitcher_data.get('@runs', 0)),
            'hits_allowed': int(pitcher_data.get('@hits', 0)),
            'earned_runs': int(pitcher_data.get('@earned_runs', 0)),
            'walks': int(pitcher_data.get('@walks', 0)),
            'strikeouts': int(pitcher_data.get('@strikeouts', 0)),
            'home_runs_allowed': int(pitcher_data.get('@home_runs', 0)),
            'hit_by_pitch': int(pitcher_data.get('@hbp', 0)),
            'earned_runs_average': float(pitcher_data.get('@earned_runs_average', 0)),

            # Parse pitch count
            'pitch_count' : self._parse_pitch_count(pitcher_data.get('@pc-st', '0-0')),
            'pitches_per_start': float(pitcher_data.get('@pitches_per_start', 0))

        }


        # Calculate additional metrics
        if pitcher_record['innings_pitched'] > 0:
            pitcher_record['whip'] = ((pitcher_record['walks'] + pitcher_record['hits_allowed']) / 
                                        pitcher_record['innings_pitched'])
            pitcher_record['k_per_9'] = (pitcher_record['strikeouts'] * 9) / pitcher_record['innings_pitched']
            pitcher_record['bb_per_9'] = (pitcher_record['walks'] * 9) / pitcher_record['innings_pitched']
        else:
            pitcher_record['whip'] = 0
            pitcher_record['k_per_9'] = 0
            pitcher_record['bb_per_9'] = 0
        
        return pitcher_record
    
    def extract_pitchers(self, team_stats):
        categories = team_stats.get('statistic', {}).get('category', [])
        if isinstance(categories, dict):  # if only one category exists
            categories = [categories]
        for category in categories:
            if category.get('@name') == 'Pitching':
                return category.get('team', {}).get('player', [])
        return []

    def extract_hitters(self, team_stats):
        categories = team_stats.get('statistic', {}).get('category', [])
        if isinstance(categories, dict):  # if only one category exists
            categories = [categories]
        for category in categories:
            if category.get('@name') == 'Batting':
                return category.get('team', {}).get('player', [])   
        return []
    
    def extract_pitcher_stats(self, match_data, game_data):
        """Extract pitcher statistics with opponent pitcher stats"""

        pitcher_records = []

        # Extract batters
        stats = match_data.get('stats', {})

        ########## for live prediction, stats may not be available
        if not stats:  
            # check have in csv
            # or 
            # go to url and update csv
            hometeam_stats = data_processor.get_team_stats(game_data['home_team_id'])
            awayteam_stats = data_processor.get_team_stats(game_data['away_team_id'])
            # Helper function to get pitchers
            

            pitchers_section = {
                'hometeam': {'player': self.extract_pitchers(hometeam_stats)},
                'awayteam': {'player': self.extract_pitchers(awayteam_stats)}
            }
    
            hitters_section = {
                'hometeam': {'player': self.extract_hitters(hometeam_stats)},
                'awayteam': {'player': self.extract_hitters(awayteam_stats)}
            }

        ########### Get pitchers section for training
        else:
            pitchers_section = stats.get('pitchers', {})
            hitters_section = stats.get('hitters', {})

        home_pitchers = pitchers_section.get('hometeam', {}).get('player', [])
        away_pitchers = pitchers_section.get('awayteam', {}).get('player', [])

        if isinstance(home_pitchers, dict):
            home_pitchers = [home_pitchers]
        if isinstance(away_pitchers, dict):            
            away_pitchers = [away_pitchers]
        # Find opponent hitter stats
        def get_batting_stats(hitters):
            total_avg = 0
            total_ops = 0
            total_runs = 0
            total_hr = 0
            total_games = 0
            count = 0

            for player in hitters:
                try:
                    avg = float(player.get('@batting_avg', 0) or 0)
                    obp = float(player.get('@on_base_percentage', 0) or 0)
                    slg = float(player.get('@slugging_percentage', 0) or 0)
                    ops = obp + slg

                    runs = int(player.get('@runs', 0) or 0)
                    hr = int(player.get('@home_runs', 0) or 0)
                    games = int(player.get('@games_played', 1) or 1)  # avoid div by zero

                    total_avg += avg
                    total_ops += ops
                    total_runs += runs
                    total_hr += hr
                    total_games += games
                    count += 1
                except Exception as e:
                    print(f"Error parsing player: {e}")
                    continue

            if count == 0 or total_games == 0:
                return {
                    'avg': 0,
                    'ops': 0,
                    'runs_per_game': 0,
                    'hr_per_game': 0
                }

            return {
                'avg': total_avg / count,
                'ops': total_ops / count,
                'runs_per_game': total_runs / total_games,
                'hr_per_game': total_hr / total_games
            }

        # Get Hitter data
        away_hitters = hitters_section.get('awayteam', {}).get('player', [])
        home_hitters = hitters_section.get('hometeam', {}).get('player', [])
        if isinstance(away_hitters, dict):
            away_hitters = [away_hitters]
        if isinstance(home_hitters, dict):
            home_hitters = [home_hitters]
        
        home_hitter_stats = get_batting_stats(home_hitters)
        away_hitter_stats = get_batting_stats(away_hitters)

        # Process home team pitchers (facing away team pitchers)
        for pitcher in home_pitchers:
            # Find opponent pitcher (could be starter or bullpen)
            opp_stats= away_hitter_stats
            pitcher_data = self._extract_single_pitcher_stats(
                pitcher, 
                game_data, 
                'home',
                opp_avg=opp_stats.get('avg', 0),
                opp_ops=opp_stats.get('ops', 0),
                opp_runs_per_game=opp_stats.get('runs_per_game', 0),
                opp_hr_per_game=opp_stats.get('hr_per_game', 0)
            )
            pitcher_records.append(pitcher_data)
        # Process away team pitchers (facing home team pitchers)
        for pitcher in away_pitchers:
            # Find opponent pitcher
            opp_stats = home_hitter_stats
            pitcher_data = self._extract_single_pitcher_stats(
                pitcher,
                game_data,
                'away', 
                opp_avg=opp_stats.get('avg', 0),
                opp_ops=opp_stats.get('ops', 0),
                opp_runs_per_game=opp_stats.get('runs_per_game', 0),
                opp_hr_per_game=opp_stats.get('hr_per_game', 0)
            )
            pitcher_records.append(pitcher_data)

        return pitcher_records

    ## -------------------for batter-------------------
    def _extract_single_batter_stats(self, hitter_data, game_data, home_away, opp_era=0, opp_k9=0, opp_whip=0, opp_hand=''):
        """Extract individual batter statistics"""
        batter_record = {
            'game_id': game_data['game_id'],
            'game_date': game_data['date'],
            'venue_name': game_data['venue_name'],
            'player_id': hitter_data.get('@id'),
            'player_name': hitter_data.get('@name'),
            'position': hitter_data.get('@pos'),
            'home_away': home_away,
            'team': game_data[f'{home_away}_team'],
            'team_id': game_data[f'{home_away}_team_id'],
            'opponent_team': game_data['away_team'] if home_away == 'home' else game_data['home_team'],
            'opponent_team_id': game_data['away_team_id'] if home_away == 'home' else game_data['home_team_id'],
            
            # Batting stats
            'at_bats': int(hitter_data.get('@at_bats', 0)),
            'runs': int(hitter_data.get('@runs', 0)),
            'hits': int(hitter_data.get('@hits', 0)),
            'doubles': int(hitter_data.get('@doubles', 0)),
            'triples': int(hitter_data.get('@triples', 0)),
            'home_runs': int(hitter_data.get('@home_runs', 0)),
            'rbis': int(hitter_data.get('@runs_batted_in', 0)),
            'walks': int(hitter_data.get('@walks', 0)),
            'strikeouts': int(hitter_data.get('@strikeouts', 0)),
            'stolen_bases': int(hitter_data.get('@stolen_bases', 0)),
            'caught_stealing': int(hitter_data.get('@cs', 0)),
            'hit_by_pitch': int(hitter_data.get('@hit_by_pitch', 0)),
            'sac_fly': int(hitter_data.get('@sac_fly', 0)),
            
            # Current averages
            'batting_average': float(hitter_data.get('@average', 0)),
            'on_base_percentage': float(hitter_data.get('@on_base_percentage', 0)),
            'slugging_percentage': float(hitter_data.get('@slugging_percentage', 0)),

            # Add missing features
            'opponent_pitcher_era': opp_era,
            'opponent_pitcher_whip': opp_whip,
            'opponent_pitcher_k9': opp_k9,
            'opponent_pitcher_hand': opp_hand,

            # Ballpark and weather conditions
            'ballpark_hr_factor': game_data.get('ballpark_hr_factor', 1.0),
            'ballpark_hit_factor': game_data.get('ballpark_hit_factor', 1.0),
            # 'temperature': game_data.get('temperature', 70),  # Default 70°F
            # 'wind_speed': game_data.get('wind_speed', 5)     # Default 5 mph
        }
        
        # Calculate total bases
        batter_record['total_bases'] = (batter_record['hits'] + 
                                        batter_record['doubles'] + 
                                        (batter_record['triples'] * 2) + 
                                        (batter_record['home_runs'] * 3))
        
        return batter_record

    def extract_batter_stats(self, match_data, game_data):
        """Extract batter statistics from the match with opponent pitcher context."""
        batter_records = []

        # Extract batters
        stats = match_data.get('stats', {})
        ################### For live prediction, stats may not be available

        if not stats: 
            hometeam_stats = data_processor.get_team_stats(game_data['home_team_id'])
            awayteam_stats = data_processor.get_team_stats(game_data['away_team_id'])
            
            hitters_section = {
                'hometeam': {'player': self.extract_hitters(hometeam_stats)},
                'awayteam': {'player': self.extract_hitters(awayteam_stats)}
            }
        else:
            # Get hitters section from stats
            hitters_section = stats.get('hitters', {})

        home_hitters = hitters_section.get('hometeam', {}).get('player', [])
        away_hitters = hitters_section.get('awayteam', {}).get('player', [])

        if isinstance(home_hitters, dict):
            home_hitters = [home_hitters]
        if isinstance(away_hitters, dict):
            away_hitters = [away_hitters]

        # Get opponent starting pitchers (home faces away starter, and vice versa)
        home_starter_id = game_data.get('home_starter_id')
        away_starter_id = game_data.get('away_starter_id')

        # Get opponent team rosters
        away_roster = data_processor.get_team_roster(game_data['away_team_id'])
        home_roster = data_processor.get_team_roster(game_data['home_team_id'])

        # Function to get pitcher hand from roster
        def get_pitcher_hand(roster, pitcher_id):
            for position in roster['team'].get('position', []):
                if position.get('@name') == 'Pitchers':
                    for player in position.get('player', []):
                        if player.get('@id') == pitcher_id:
                            return player.get('@throws')
            return ''

        # Find opponent pitcher stats
        def get_pitcher_stats(pitchers, pitcher_id):
            for p in pitchers:
                if p.get('@id') == pitcher_id:
                    ip = float(p.get('@innings_pitched', 1))
                    return {
                        'era': float(p.get('@earned_runs_average', 0)),
                        'k9': (float(p.get('@strikeouts', 0)) * 9) / ip,
                        'whip': (float(p.get('@walks', 0)) + float(p.get('@hits', 0))) / ip,
                    }
            return {'era': 0, 'k9': 0, 'whip': 0}

        # Get pitcher data
        away_pitchers = match_data.get('awayteam', {}).get('pitchers', {}).get('player', [])
        home_pitchers = match_data.get('hometeam', {}).get('pitchers', {}).get('player', [])
        if isinstance(away_pitchers, dict):
            away_pitchers = [away_pitchers]
        if isinstance(home_pitchers, dict):
            home_pitchers = [home_pitchers]

        away_pitcher_stats = get_pitcher_stats(away_pitchers, away_starter_id)
        away_pitcher_stats['hand'] = get_pitcher_hand(away_roster, away_starter_id)

        home_pitcher_stats = get_pitcher_stats(home_pitchers, home_starter_id)
        home_pitcher_stats['hand'] = get_pitcher_hand(home_roster, home_starter_id)

        # Process home team batters (facing away pitcher)
        for hitter in home_hitters:
            batter_data = self._extract_single_batter_stats(
                hitter, game_data, 'home',
                opp_era=away_pitcher_stats['era'],
                opp_k9=away_pitcher_stats['k9'],
                opp_whip=away_pitcher_stats['whip'],
                opp_hand=away_pitcher_stats['hand']
            )
            batter_records.append(batter_data)

        # Process away team batters (facing home pitcher)
        for hitter in away_hitters:
            batter_data = self._extract_single_batter_stats(
                hitter, game_data, 'away',
                opp_era=home_pitcher_stats['era'],
                opp_k9=home_pitcher_stats['k9'],
                opp_whip=home_pitcher_stats['whip'],
                opp_hand=home_pitcher_stats['hand']
            )
            batter_records.append(batter_data)

        return batter_records

    def calculate_ballpark_factors(df):
        """Calculate ballpark factors based on hits data"""
        # Group by venue and calculate average hits per game
        venue_stats = df.groupby('venue_name').agg({
            'hits': 'mean',
            'home_runs': 'mean',  # If you eventually get this data
            'at_bats': 'mean'
        }).reset_index()
        
        # Calculate league averages
        league_avg_hits = df['hits'].mean()
        league_avg_ab = df['at_bats'].mean()
        
        # Calculate park factors (hits per AB compared to league)
        venue_stats['hit_factor'] = (venue_stats['hits']/venue_stats['at_bats']) / \
                                (league_avg_hits/league_avg_ab)
        
        # If you get HR data later:
        if 'home_runs' in df.columns:
            league_avg_hr = df['home_runs'].mean()
            venue_stats['hr_factor'] = venue_stats['home_runs'] / league_avg_hr
        else:
            # Estimate HR factor from hits (less accurate)
            venue_stats['hr_factor'] = venue_stats['hit_factor'] * 1.1  # HR typically vary more
       

        return venue_stats[['venue_name', 'hit_factor', 'hr_factor']]
        
    
    # --------------Create features for batter------------------
    def create_batter_features(self, df):
        """Create features for batter performance prediction"""
        # Basic stats
        df['batting_avg_last_10'] = df.groupby('player_id')['hits'].rolling(10,min_periods=1).sum() / df.groupby('player_id')['at_bats'].rolling(10,min_periods=1).sum()
        df['obp_last_10'] = (df.groupby('player_id')['hits'].rolling(10,min_periods=1).sum() + df.groupby('player_id')['walks'].rolling(10,min_periods=1).sum()) / (df.groupby('player_id')['at_bats'].rolling(10,min_periods=1).sum() + df.groupby('player_id')['walks'].rolling(10,min_periods=1).sum())
        df['slg_last_10'] = df.groupby('player_id')['total_bases'].rolling(10,min_periods=1).sum() / df.groupby('player_id')['at_bats'].rolling(10,min_periods=1).sum()

        # Recent performance (last 5 games)
        df['hits_last_5'] = df.groupby('player_id')['hits'].rolling(5, min_periods=1).mean()
        df['hr_last_5'] = df.groupby('player_id')['home_runs'].rolling(5, min_periods=1).mean()
        df['rbi_last_5'] = df.groupby('player_id')['rbis'].rolling(5, min_periods=1).mean()
        df['k_rate_last_5'] = df.groupby('player_id')['strikeouts'].rolling(5, min_periods=1).sum() / df.groupby('player_id')['at_bats'].rolling(5, min_periods=1).sum()

        # Season stats
        # df['season_batting_avg'] = df.groupby(['player_id', 'season'])['hits'].expanding().sum() / df.groupby(['player_id', 'season'])['at_bats'].expanding().sum()
        # df['season_hr_rate'] = df.groupby(['player_id', 'season'])['home_runs'].expanding().sum() / df.groupby(['player_id', 'season'])['at_bats'].expanding().sum()
        # df['season_rbi_per_game'] = df.groupby(['player_id', 'season'])['rbis'].expanding().mean()
        
        # Opponent pitcher stats
        df['opp_pitcher_era'] = df['opponent_pitcher_era']
        df['opp_pitcher_whip'] = df['opponent_pitcher_whip']
        df['opp_pitcher_k9'] = df['opponent_pitcher_k9']
        
        # Situational features
        df['is_home'] = (df['home_away'] == 'home').astype(int)
        df['vs_lefty'] = (df['opponent_pitcher_hand'] == 'L').astype(int)
        df['vs_righty'] = (df['opponent_pitcher_hand'] == 'R').astype(int)
        # df['games_played_season'] = df.groupby(['player_id', 'season']).cumcount() + 1
        df['rest_days'] = df.groupby('player_id')['game_date'].diff().dt.days

        # Ballpark factors
        ballpark_factors = self.calculate_ballpark_factors(df)
        df = df.merge(
            ballpark_factors,
            on='venue_name',
            how='left'
        ).fillna({'hit_factor': 1.0, 'hr_factor': 1.0})  # Neutral factors for unknown parks
        
        # Clean up column names
        df = df.rename(columns={
            'hit_factor': 'ballpark_hit_factor',
            'hr_factor': 'ballpark_hr_factor'
        })
        
        return df
    
    def calculate_ballpark_pitcher_factors(self, df):
        """
        Calculate pitcher-friendly ballpark factors based on hits and HRs allowed at each venue.
        
        Input columns: venue_name, hits_allowed, home_runs_allowed, innings_pitched
        """
        # Group by venue to get per-game averages
        venue_stats = df.groupby('venue_name').agg({
            'hits_allowed': 'mean',
            'home_runs_allowed': 'mean',
            'innings_pitched': 'mean'
        }).reset_index()

        # League-wide averages
        league_avg_hits = df['hits_allowed'].mean()
        league_avg_ip = df['innings_pitched'].mean()

        # Calculate factors
        venue_stats['hit_factor'] = (venue_stats['hits_allowed'] / venue_stats['innings_pitched']) / \
                                    (league_avg_hits / league_avg_ip)

        if 'home_runs_allowed' in df.columns and df['home_runs_allowed'].notna().any():
            league_avg_hr = df['home_runs_allowed'].mean()
            venue_stats['hr_factor'] = venue_stats['home_runs_allowed'] / league_avg_hr
        else:
            venue_stats['hr_factor'] = venue_stats['hit_factor'] * 1.1  # Estimate fallback

        # Pitcher-friendly = lower hit_factor → higher pitcher_factor
        venue_stats['pitcher_factor'] = 1 / venue_stats['hit_factor']
        venue_stats['pitcher_factor'] = venue_stats['pitcher_factor'].clip(lower=0.5, upper=1.5)

        return venue_stats[['venue_name', 'hit_factor', 'hr_factor', 'pitcher_factor']]


    def create_pitcher_features(self, df):
        """Create features for pitcher performance prediction"""
        #print("pitchers_columns--------------", df.columns)
    
        # Recent performance (last 5 starts)

        df['era_last_5'] = df.groupby('player_id')['earned_runs_average'].transform(
            lambda x: x.rolling(5, min_periods=1).mean()
        )
        df['whip_last_5'] = df.groupby('player_id')['whip'].transform(
            lambda x: x.rolling(5, min_periods=1).mean()
        )
        df['k9_last_5'] = df.groupby('player_id')['k_per_9'].transform(
            lambda x: x.rolling(5, min_periods=1).mean()
        )
        df['bb9_last_5'] = df.groupby('player_id')['bb_per_9'].transform(
            lambda x: x.rolling(5, min_periods=1).mean()
        )
        df['ip_last_5'] = df.groupby('player_id')['innings_pitched'].transform(
            lambda x: x.rolling(5, min_periods=1).mean()
        )

        # Season stats - fixed with expanding().mean() instead of manual calculations
        # df['season_era'] = (
        #     df.groupby(['player_id', 'season'])['earned_runs_average']
        #     .transform(lambda x: x.expanding().mean())
        # )
        # df['season_whip'] = df.groupby(['player_id', 'season'])['whip'].transform(
        #     lambda x: x.expanding().mean()
        # )
        # df['season_k9'] = df.groupby(['player_id', 'season'])['k_per_9'].transform(
        #     lambda x: x.expanding().mean()
        # )

        # Alternative calculation for season stats if you need custom formulas
        # def season_stats(group):
        #     era = (group['earned_runs'].sum() * 9) / group['innings_pitched'].sum()
        #     whip = (group['walks'].sum() + group['hits_allowed'].sum()) / group['innings_pitched'].sum()
        #     k9 = (group['strikeouts'].sum() * 9) / group['innings_pitched'].sum()
        #     return pd.Series({'season_era_calc': era, 'season_whip_calc': whip, 'season_k9_calc': k9})

        # season_stats_df = df.groupby(['player_id', 'season']).apply(season_stats)
        # df = df.merge(season_stats_df, how='left', left_on=['player_id', 'season'], right_index=True)

        # Opponent team strength (assuming these columns exist)
        opp_cols = ['opponent_team_batting_avg', 'opponent_team_ops', 
                'opponent_team_runs_per_game', 'opponent_team_hr_per_game']
        df[['opp_team_avg', 'opp_team_ops', 'opp_team_runs_per_game', 'opp_team_hr_per_game']] = df[opp_cols]

        # Situational features
        df['is_home'] = (df['home_away'] == 'home').astype(int)
        df['game_date'] = pd.to_datetime(df['game_date'])
        df = df.sort_values(by=['player_id', 'game_date'])  # important for .diff() to work correctly
        df['days_rest'] = df.groupby('player_id')['game_date'].diff().dt.days.fillna(5)  # Default 5 days for first start
        df['starts_this_season'] = df.groupby(['player_id', 'season']).cumcount() + 1
        df = df.sort_values(by=['player_id', 'game_date'])
        # Shift pitch count to get last game's pitch count
        df['pitch_count_last_start'] = df.groupby('player_id')['pitch_count'].shift(1)

        # Fill missing pitch_count_last_start with pitches_per_start if available
        if 'pitches_per_start' in df.columns:
            df['pitch_count_last_start'] = df['pitch_count_last_start'].fillna(df['pitches_per_start'])

        # Optionally fill remaining missing with a default MLB average
        df['pitch_count_last_start'] = df['pitch_count_last_start'].fillna(82)

        # print("pitch_count_last_start------------------", df['pitch_count_last_start'].isna().sum())
        # Weather and ballpark
        # df['temperature'] = df['game_temperature']
        # df['wind_speed'] = df['wind_speed']
        # df['ballpark_pitcher_factor'] = df['ballpark_pitcher_friendly_factor']

        # Ballpark factors
        ballpark_factors = self.calculate_ballpark_pitcher_factors(df)
        df = df.merge(
            ballpark_factors,
            on='venue_name',
            how='left'
        ).fillna({'hit_factor':1.0, 'hr_factor': 1.0,'pitcher_factor': 1.0})  # Neutral factors for unknown parks
        
        # Clean up column names
        df = df.rename(columns={
            'pitcher_factor': 'ballpark_pitcher_factor',
        })
        
        return df
    
    def get_batter_feature_columns(self):
        """Define feature columns for batter models"""
        return [
            'batting_avg_last_10', 'obp_last_10', 'slg_last_10',
            'hits_last_5', 'hr_last_5', 'rbi_last_5', 'k_rate_last_5',
            'opp_pitcher_era', 'opp_pitcher_whip', 'opp_pitcher_k9',
            'is_home', 'vs_lefty', 'vs_righty', 'rest_days',
            'ballpark_hr_factor', 'ballpark_hit_factor'
        ]
    
    def get_pitcher_feature_columns(self):
        """Define feature columns for pitcher models"""
        return [
            'era_last_5', 'whip_last_5', 'k9_last_5', 'bb9_last_5', 'ip_last_5',
            'opp_team_avg', 'opp_team_ops', 'opp_team_runs_per_game', 'opp_team_hr_per_game',
            'is_home', 'days_rest', 'pitch_count_last_start',
            'ballpark_pitcher_factor'
        ]



    

    
            
        
        


