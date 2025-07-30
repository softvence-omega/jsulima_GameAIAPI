import requests
import pandas as pd
from datetime import datetime
from app.services.MLB.xml_to_json import xml_url_to_json
from app.data.schemas.mlb_schema import validate_game_schema
from app.services.helper import safe_float, safe_int
from urllib.parse import urlencode
from app.services.MLB.injuri_roaster_playerstat_data_collector import fetch_standings_data
from app.config import DATA_DIR, GOALSERVE_BASE_URL
import os 

class BaseballDataProcessor:
    def __init__(self, api_key):
        self.base_url = GOALSERVE_BASE_URL
        self.api_key = api_key
        
    def fetch_data(self, endpoint, params=None):
        url = f"{self.base_url}{self.api_key}/{endpoint}"
        if params:
            query_string = urlencode(params)
            url += f"?{query_string}"
        response = xml_url_to_json(url)
        return response
    
    def get_todays_games(self):
        today = datetime.now().strftime("%d.%m.%Y")
        return self.fetch_data("baseball/mlb_shedule", {"date1": today, "date2": today})
    
    def get_todays_score(self):
        """Fetch today's MLB scoreboard"""
        today = datetime.now().strftime("%d.%m.%Y")
        return self.fetch_data("baseball/mlb-scores", {"date1": today, "date2": today})

    def get_team_stats(self, team_id):
        return self.fetch_data(f"baseball/{team_id}_stats")
    
    def get_pitcher_stats(self, player_id):
        return self.fetch_data("baseball/mlb_player_pitching", {"player": player_id}) 
    
    def get_pitcher_era(self, pitcher_id):
        """Get pitcher's ERA from player stats"""
        pitcher_stats = self.get_pitcher_stats(pitcher_id)
        return safe_float(pitcher_stats.get('player', {}).get('@earned_runs_average'), 4.50)
    
    def get_playby_play_data(self, game_id):
        """Fetch play-by-play data for a specific game"""
        return self.fetch_data("baseball/mlb_playbyplay", {"game_id": game_id})

    def get_team_roster(self, team_id):
        """Get team roster information"""
        return self.fetch_data(f"baseball/{team_id}_rosters")

    def calculate_team_ops(self, players):
        """Calculate team On-base Plus Slugging percentage"""
        if not players or not isinstance(players, list):
            return 0
            
        total_obp = sum(safe_float(p.get('@on_base_percentage', 0)) for p in players)
        total_slg = sum(safe_float(p.get('@slugging_percentage', 0)) for p in players)
        return (total_obp + total_slg) / len(players) if players else 0

    def calculate_bullpen_era(self, pitchers):
        """Calculate bullpen ERA excluding starting pitcher"""
        if not pitchers or not isinstance(pitchers, list):
            return 4.50  # League average
            
        relievers = [p for p in pitchers if safe_float(p.get('@innings_pitched', 0)) < 5]  # Filter starters
        if not relievers:
            return 4.50  # League average
        
        total_er = sum(safe_float(p.get('@earned_runs', 0)) for p in relievers)
        total_ip = sum(safe_float(p.get('@innings_pitched', 0)) for p in relievers)
        return (total_er * 9) / total_ip if total_ip > 0 else 4.50
    
    
    def _get_team_standing_info(self, team_id, standings_data):
        """Extract win percentage and rank from standings"""
        try:
            leagues = standings_data["standings"]["category"]["league"]
            for league in leagues:
                for division in league["division"]:
                    for team in division["team"]:
                        if team["@id"] == team_id:
                            wins = safe_int(team.get('@won'))
                            losses = safe_int(team.get('@lost'))
                            win_pct = safe_float(wins / (wins + losses) if (wins + losses) > 0 else 0.5 )
                            rank = safe_int(team.get('@position'))
                            return win_pct, rank
                        
                
        except Exception as e:
            #printf"[Standing Parsing Error] {e}")
            raise ValueError(f"Error parsing standings data: {str(e)}")

        return 0.5, 15  # Default fallback
    
    def extract_predictive_features(self, game_data):
        """Extract predictive features from raw game JSON"""
        if not game_data:
            return {}

        try:
            features = {
                'is_home': 1,
                'venue_id': safe_int(game_data.get('@venue_id')),
                'home_hits': safe_int(game_data.get('hometeam', {}).get('@hits', 0)),
                'away_hits': safe_int(game_data.get('awayteam', {}).get('@hits', 0)),
                'home_errors': safe_int(game_data.get('hometeam', {}).get('@errors', 0)),
                'away_errors': safe_int(game_data.get('awayteam', {}).get('@errors', 0)),
                'home_sp_era': self._get_starting_pitcher_era(game_data, 'hometeam'),
                'away_sp_era': self._get_starting_pitcher_era(game_data, 'awayteam'),
                'home_late_inning_runs': self._get_late_inning_runs(game_data, 'hometeam'),
                'away_late_inning_runs': self._get_late_inning_runs(game_data, 'awayteam'),
                'home_ops': safe_float(self._get_team_ops(game_data, 'hometeam')),
                'away_ops': safe_float(self._get_team_ops(game_data, 'awayteam')),
                'home_bullpen_era': safe_float(self._get_bullpen_era(game_data, 'hometeam')),
                'away_bullpen_era': safe_float(self._get_bullpen_era(game_data, 'awayteam'))
            }

            features.update({
                'hit_diff': safe_float( features['home_hits'] - features['away_hits']),
                'error_diff': safe_float(features['away_errors'] - features['home_errors']),
                'era_diff': safe_float(features['away_sp_era'] - features['home_sp_era']),
                'ops_diff': safe_float(features['home_ops'] - features['away_ops']),
                'bullpen_diff': safe_float(features['away_bullpen_era'] - features['home_bullpen_era']),
                'late_inning_diff': safe_float(features['home_late_inning_runs'] - features['away_late_inning_runs'])
            })

            home_team_id = game_data.get('hometeam', {}).get('@id')
            away_team_id = game_data.get('awayteam', {}).get('@id')

            # home_win_pct, home_rank = self._get_team_standing_info(home_team_id, standings_data or {})
            # away_win_pct, away_rank = self._get_team_standing_info(away_team_id, standings_data or {})

            dir = os.path.join(DATA_DIR, "MLB")
            os.makedirs(dir, exist_ok=True)
            standing_path = os.path.join(dir, 'mlb_standings.csv')
            standings_df = pd.read_csv(standing_path)

            if standings_df['team_id']!= home_team_id :
                # update standing csv
                new_df = fetch_standings_data()
                standings_df = pd.concat(standings_df, new_df)
                standings_df.to_csv(standing_path, index=False)
            elif standings_df['team_id']== home_team_id:
                home_win_pct = standings_df['win_pct'] # home win pct 
                home_rank = standings_df['rank']

            
            if standings_df['team_id']!= away_team_id:
                new_df = fetch_standings_data()
                standings_df = pd.concat(standings_df, new_df)
                standings_df.to_csv(standing_path, index=False)
            elif standings_df['team_id']== away_team_id:
                away_win_pct = standings_df['win_pct']
                away_rank = standings_df['rank']

            features.update({
                'home_win_pct': home_win_pct,
                'away_win_pct': away_win_pct,
                'win_pct_diff': safe_float(home_win_pct - away_win_pct),
                'home_rank': home_rank,
                'away_rank': away_rank,
                'rank_diff': abs(away_rank - home_rank)
            })            
            return features

        except Exception as e:
            #printf"Error extracting features: {str(e)}")
            return {}

    def _get_late_inning_runs(self, game_data, team_type):
        """Helper to get late inning runs for a team"""
        try:
            innings = game_data.get(team_type, {}).get('innings', {}).get('inning', [])
            if isinstance(innings, dict):  # Handle single inning case
                innings = [innings]
            return sum(
                safe_int(inning.get('@score')) 
                for inning in innings 
                if safe_int(inning.get('@number')) >= 7
            )
        except Exception:
            return 0

    def _get_team_ops(self, game_data, team_type):
        """Helper to get team OPS"""
        try:
            players = game_data.get('stats', {}).get('hitters', {}).get(team_type, {}).get('player', [])
            if isinstance(players, dict):  # Handle case where only one player exists
                players = [players]
            return self.calculate_team_ops(players)
        except Exception:
            return 0

    def _get_bullpen_era(self, game_data, team_type):
        """Helper to get bullpen ERA"""
        try:
            pitchers = game_data.get('stats', {}).get('pitchers', {}).get(team_type, {}).get('player', [])
            if isinstance(pitchers, dict):  # Handle case where only one pitcher exists
                pitchers = [pitchers]
            return self.calculate_bullpen_era(pitchers)
        except Exception:
            return 4.50

    def _get_starting_pitcher_era(self, game_json, team_type):
        """Helper to get starting pitcher ERA"""
        try:
            pitcher_id = game_json.get('starting_pitchers', {}).get(team_type, {}).get('player', {}).get('@id')
            if pitcher_id:
                return self.get_pitcher_era(pitcher_id)
            return 4.50
        except Exception:
            return 4.50

    def _get_team_win_pct(self, team_id):
        standings = self.fetch_data("baseball/mlb_standings")
        # Parse standings to find team's win percentage
        # Implementation depends on standings structure
        return 0.5  # Placeholder

    ##### Enhanced methods for Top-Performers-MLB #####
    ###################################################

    def process_batting_data(self) -> pd.DataFrame:
        """Enhanced batting data processing with team information"""
        endpoint = "baseball/mlb_player_batting"
        batting_json = self.fetch_data(endpoint)
        
        try:
            players = batting_json['statistic']['category']['player']
            records = []
            
            for player in players:
                try:
                    record = {
                        'name': player.get('@name', 'Unknown'),
                        'team_id': player.get('@team_id', ''),
                        'team_name': player.get('@team', ''),
                        'position': player.get('@position', ''),
                        'gp': safe_int(player.get('@gp', 0)),
                        'at_bats': safe_int(player.get('@at_bats', 0)),
                        'runs': safe_int(player.get('@runs', 0)),
                        'hits': safe_int(player.get('@hits', 0)),
                        'doubles': safe_int(player.get('@doubles', 0)),
                        'triples': safe_int(player.get('@triples', 0)),
                        'home_runs': safe_int(player.get('@home_runs', 0)),
                        'runs_batted_in': safe_int(player.get('@runs_batted_in', 0)),
                        'walks': safe_int(player.get('@walks', 0)),
                        'strikeouts': safe_int(player.get('@strikeouts', 0)),
                        'stolen_bases': safe_int(player.get('@stolen_bases', 0)),
                        'batting_average': safe_float(player.get('@batting_average', 0)),
                        'on_base_percentage': safe_float(player.get('@on_base_percentage', 0)),
                        'slugging_percentage': safe_float(player.get('@slugging_percentage', 0)),
                    }
                    records.append(record)
                except Exception as e:
                    #printf"Error processing batter {player.get('@name', 'Unknown')}: {e}")
                    continue
                    
            df = pd.DataFrame(records)
            return df
            
        except Exception as e:
            #printf"Error processing batting data: {e}")
            return pd.DataFrame()
    
    def innings_to_float(self, innings_str):
        """Convert innings like '62.2' to 62.6667 (2/3 inning)."""
        if '.' not in str(innings_str):
            return float(innings_str)
        parts = str(innings_str).split('.')
        try:
            return int(parts[0]) + (int(parts[1]) * 1.0 / 3)
        except:
            return float(parts[0])
        
    def get_pitcher_data(self):
        """Enhanced pitcher data with team information"""
        endpoint = "baseball/mlb_player_pitching"
        data = self.fetch_data(endpoint)
        
        try:
            players = data['statistic']['category']['player']
        except KeyError:
            #print"Pitcher data not found.")
            return pd.DataFrame()

        records = []

        for player in players:
            try:
                records.append({
                    "name": player.get("@name", "Unknown"),
                    "team": player.get("@team", ""),
                    "gp": safe_int(player.get("@gp", 0)),
                    "gs": safe_int(player.get("@gs", 0)),
                    "qs": safe_int(player.get("@qs", 0)),
                    "innings_pitched": self.innings_to_float(player.get("@innings_pitched", "0")),
                    "hits_allowed": safe_int(player.get("@hits", 0)),         
                    "earned_runs": safe_int(player.get("@earned_runs", 0)),
                    "walks": safe_int(player.get("@walks", 0)),
                    "strikeouts": safe_int(player.get("@strikeouts", 0)),
                    "era": safe_float(player.get("@earned_runs_average", 4.50)),
                    "whip": safe_float(player.get("@whip", 1.50))
                })
            except Exception as e:
                #printf"Error processing pitcher {player.get('@name', 'Unknown')}: {e}")
                continue
        df = pd.DataFrame(records)
        return df




