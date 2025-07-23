import os
import pandas as pd
from datetime import datetime, timedelta
from functools import lru_cache
import logging
from app.config import GOALSERVE_BASE_URL, GOALSERVE_API_KEY
from app.utils.xml_to_json import xml_to_json
from app.services.helper import safe_float, safe_int,safe_div


class NflDataProcessor:
    def __init__(self, api_key=GOALSERVE_API_KEY):
        self.base_url = GOALSERVE_BASE_URL.rstrip('/')
        self.api_key = api_key
        self.logger = logging.getLogger(__name__)
        # Configure logging only if no handlers exist (avoid overriding global config)
        if not logging.getLogger().hasHandlers():
            logging.basicConfig(level=logging.INFO)


########################################################################################################################
##############################################       Fetch URL      ####################################################
    @lru_cache(maxsize=64)
    def fetch_data(self, endpoint: str) -> dict:
        """Fetch data with caching to avoid redundant API calls."""
        url = f"{self.base_url}/{self.api_key}/{endpoint}"
        try:
            data = xml_to_json(url)
            if not data:
                self.logger.warning(f"Empty response from endpoint: {endpoint}")
                print(f"Empty response for URL: {url}")
                return {}
            return data
        except Exception as e:
            self.logger.error(f"Error fetching data from {url}: {str(e)}")
            print(f"Error fetching data from {url}: {str(e)}")
            return {}
    
    def get_todays_games(self) -> dict:
        today = datetime.now().strftime("%d.%m.%Y")
        self.game_data = self.fetch_data(f"football/nfl-shedule", {"date1": today, "date2": today})
        return self.game_data 

    def fetch_player_stats(self,team_id) -> dict:
        """Fetch player statistics."""
        self.endpoint = f"football/{team_id}_player_stats"
        self.stats_data = self.fetch_data(self.endpoint)
        return self.stats_data
    
    def fetch_player_roster(self,team_id) -> dict:
        """Fetch current NFL roster."""
        endpoint = f"football/{team_id}_rosters"
        self.roster_data = self.fetch_data(endpoint)
        return self.roster_data
    
    def fetch_injuries(self,team_id) -> dict:
        """Fetch player injuries."""
        endpoint = f"football/{team_id}_injuries"
        self.injuries_data = self.fetch_data(endpoint)
        return self.injuries_data

    def fetch_standings(self) -> dict:
        """Fetch current standings."""
        endpoint = "football/nfl-standings"
        self.standings_data = self.fetch_data(endpoint)
        return self.standings_data
    
    def fetch_odds_schedule(self, date1, date2):
        """Load NFL odds data for a date range."""
        endpoint = f"football/nfl-shedule?date1={date1}&date2={date2}&showodds=1"
        self.odd_data= self.fetch_data(endpoint)
        return self.odd_data
    

    @staticmethod
    def parse_possession_time(possession_str: str) -> float:
        """
        Convert possession time from 'MM:SS' string to total minutes as float.
        Example: '36:58' -> 36 + 58/60 = 36.9667
        """
        try:
            if not possession_str:
                return 0.0
            parts = possession_str.split(':')
            if len(parts) != 2:
                return 0.0
            minutes = int(parts[0])
            seconds = int(parts[1])
            total_minutes = minutes + seconds / 60
            return total_minutes
        except Exception as e:
            return 0.0

    def calculate_team_passing_efficiency(self, players):
        """Calculate team passing efficiency (QB Rating equivalent)"""
        if not players:
            return 0
        
        qbs = [p for p in players if p.get('position') == 'QB']
        if not qbs:
            return 0
            
        total_rating = sum(safe_float(p.get('passer_rating', 0)) for p in qbs)
        return total_rating / len(qbs)

    def calculate_red_zone_efficiency(self, team_stats):
        """Calculate red zone touchdown percentage"""
        red_zone_attempts = safe_int(team_stats.get('red_zone_attempts', 0))
        red_zone_tds = safe_int(team_stats.get('red_zone_touchdowns', 0))
        return (red_zone_tds / red_zone_attempts * 100) if red_zone_attempts > 0 else 0

    def calculate_third_down_conversion(self, team_stats):
        """Calculate third down conversion percentage"""
        third_down_attempts = safe_int(team_stats.get('third_down_attempts', 0))
        third_down_conversions = safe_int(team_stats.get('third_down_conversions', 0))
        return (third_down_conversions / third_down_attempts * 100) if third_down_attempts > 0 else 0

    def calculate_turnover_differential(self, team_stats):
        """Calculate turnover differential"""
        turnovers_forced = safe_int(team_stats.get('interceptions', 0)) + safe_int(team_stats.get('fumbles_recovered', 0))
        turnovers_lost = safe_int(team_stats.get('interceptions_thrown', 0)) + safe_int(team_stats.get('fumbles_lost', 0))
        return turnovers_forced - turnovers_lost

    def calculate_yards_per_play(self, team_stats):
        """Calculate offensive yards per play"""
        total_yards = safe_int(team_stats.get('total_yards', 0))
        total_plays = safe_int(team_stats.get('total_plays', 0))
        return total_yards / total_plays if total_plays > 0 else 0

    
    ################################################### Extract Features ##############################################################
    
    
    def _get_extract_standings_features(self, standings_data):
        """
        Extracts structured NFL standings features from Goalserve XML standings data.
        """
        # Parse the XML to a dictionary
        standings_data = self.standings_data

        try:
            categories = standings_data.get('standings', {}).get('category', {})
            if not categories:
                return []

            leagues = categories.get('league', [])
            if isinstance(leagues, dict):
                leagues = [leagues]

            teams_data = []

            for league in leagues:
                league_name = league.get('@name')
                divisions = league.get('division', [])
                if isinstance(divisions, dict):
                    divisions = [divisions]

                for division in divisions:
                    division_name = division.get('@name')
                    teams = division.get('team', [])
                    if isinstance(teams, dict):
                        teams = [teams]

                    for team in teams:
                        try:
                            
                            home_wins, home_losses = map(int, team.get('@home_record', '0-0').split('-'))
                            road_wins, road_losses = map(int, team.get('@road_record', '0-0').split('-'))

                            games_played = int(team.get('@won', 0)) + int(team.get('@lost', 0)) + int(team.get('@ties', 0))
                            points_for = int(team.get('@points_for', 0))
                            points_against = int(team.get('@points_against', 0))
                            streak_raw = team.get('@streak', '')  # e.g. W3, L2

                            team_data = {
                                'league': league_name,
                                'division': division_name,
                                'team_id': int(team.get('@id', 0)),
                                'team_name': team.get('@name'),
                                'position': int(team.get('@position', 0)),
                                'won': int(team.get('@won', 0)),
                                'lost': int(team.get('@lost', 0)),
                                'ties': int(team.get('@ties', 0)),
                                'games_played': games_played,
                                'win_percentage': float(team.get('@win_percentage', 0)),
                                'win_ratio': float(team.get('@won', 0)) / games_played if games_played > 0 else 0,
                                'points_for': points_for,
                                'points_against': points_against,
                                'point_difference': int(team.get('@difference', '0').replace('+', '').replace('−', '-')),
                                'points_per_game': points_for / games_played if games_played > 0 else 0,
                                'points_allowed_per_game': points_against / games_played if games_played > 0 else 0,
                                'home_record': team.get('@home_record'),
                                'road_record': team.get('@road_record'),
                                'home_wins': home_wins,
                                'home_losses': home_losses,
                                'road_wins': road_wins,
                                'road_losses': road_losses,
                                'division_record': team.get('@division_record'),
                                'conference_record': team.get('@conference_record'),
                                'streak': streak_raw,
                                'streak_length': int(streak_raw[1:]) if streak_raw else 0,
                                'is_on_winning_streak': 1 if streak_raw.startswith('W') else 0,
                            }
                            teams_data.append(team_data)
                        except Exception as e:
                            print(f"[WARNING] Skipping team due to error: {e}")

            return teams_data

        except Exception as e:
            print(f"[ERROR] Failed to extract standings features: {e}")
            return []

    def extract_events(self, events_data: dict) -> dict:
        """
        Extract events from each quarter into a summary dict.
        Counts total touchdowns, field goals, and events per quarter.
        """
        events_summary = {
            'total_touchdowns': 0,
            'total_field_goals': 0,
            'events_per_quarter': {}
        }
        try:
            for quarter in ['firstquarter', 'secondquarter', 'thirdquarter', 'fourthquarter', 'overtime']:
                quarter_events = events_data.get(quarter) or {}
                event_list = quarter_events.get('event', [])
                if isinstance(event_list, dict):
                    event_list = [event_list]

                events_summary['events_per_quarter'][quarter] = len(event_list)

                for event in event_list:
                    event_type = event.get('type', '').upper()
                    if event_type == 'TD':
                        events_summary['total_touchdowns'] += 1
                    elif event_type == 'FG':
                        events_summary['total_field_goals'] += 1
        except Exception as e:
            self.logger.error(f"Error extracting events: {e}")

        return events_summary

    def extract_team_stats(self, team_stats: dict, prefix: str) -> dict:
        """
        Extract relevant team stats into flat dict with prefix.
        Flattens nested stats like first_downs, passing, rushing, etc.
        Now includes advanced NFL metrics.
        """
        stats = {}
        try:
            for stat_key, stat_value in team_stats.items():
                if isinstance(stat_value, dict):
                    for sub_key, sub_val in stat_value.items():
                        stats[f"{prefix}_{stat_key}_{sub_key}"] = sub_val
                else:
                    stats[f"{prefix}_{stat_key}"] = stat_value

            # Special handling: parse possession time to float minutes if exists
            possession_raw = team_stats.get('posession', {}).get('total', '') or team_stats.get('possession', '')
            if possession_raw:
                # Sometimes possession is string directly (like '36:58')
                if isinstance(possession_raw, str):
                    stats[f"{prefix}_possession_minutes"] = self.parse_possession_time(possession_raw)
                else:
                    stats[f"{prefix}_possession_minutes"] = safe_float(possession_raw)

            # --- Advanced NFL Metrics ---
            
            # Red Zone Efficiency
            stats[f"{prefix}_red_zone_efficiency"] = self.calculate_red_zone_efficiency(team_stats)
            
            # Third Down Conversion Rate
            stats[f"{prefix}_third_down_conversion"] = self.calculate_third_down_conversion(team_stats)
            
            # Turnover Differential
            stats[f"{prefix}_turnover_differential"] = self.calculate_turnover_differential(team_stats)
            
            # Yards Per Play
            stats[f"{prefix}_yards_per_play"] = self.calculate_yards_per_play(team_stats)
            
            # Additional derived metrics
            total_plays = safe_int(team_stats.get('total_plays', 0))
            rushing_attempts = safe_int(team_stats.get('rushing_attempts', 0))
            passing_attempts = safe_int(team_stats.get('passing_attempts', 0))
            
            # Pass/Rush balance
            if total_plays > 0:
                stats[f"{prefix}_pass_ratio"] = passing_attempts / total_plays
                stats[f"{prefix}_rush_ratio"] = rushing_attempts / total_plays
            else:
                stats[f"{prefix}_pass_ratio"] = 0
                stats[f"{prefix}_rush_ratio"] = 0

            # Completion percentage
            completions = safe_int(team_stats.get('passing_completions', 0))
            if passing_attempts > 0:
                stats[f"{prefix}_completion_percentage"] = (completions / passing_attempts) * 100
            else:
                stats[f"{prefix}_completion_percentage"] = 0

            # Rushing yards per carry
            rushing_yards = safe_int(team_stats.get('rushing_yards', 0))
            if rushing_attempts > 0:
                stats[f"{prefix}_yards_per_carry"] = rushing_yards / rushing_attempts
            else:
                stats[f"{prefix}_yards_per_carry"] = 0

            # Passing yards per attempt
            passing_yards = safe_int(team_stats.get('passing_yards', 0))
            if passing_attempts > 0:
                stats[f"{prefix}_yards_per_pass"] = passing_yards / passing_attempts
            else:
                stats[f"{prefix}_yards_per_pass"] = 0

        except Exception as e:
            self.logger.error(f"Error extracting team stats for {prefix}: {e}")

        return stats

    def extract_predictive_features(self, game_data: dict) -> dict:
        if not game_data:
            return {}

        try:
            features = {}

            # --- Basic metadata ---
            features['date'] = game_data.get('@date', '')
            features['time'] = game_data.get('@time', '')
            features['venue'] = game_data.get('@venue_name', '')
            features['venue_id'] = game_data.get('@venue_id', '')
            features['contest_id'] = game_data.get('@contestID', '')

            # --- Teams info ---
            features['game_id'] = game_data.get('@id', '')
            features['home_team'] = game_data.get('hometeam', {}).get('@name', '')
            features['away_team'] = game_data.get('awayteam', {}).get('@name', '')
            features['away_team_id'] = game_data.get('hometeam', {}).get('@id', '')
            features['away_team_id'] = game_data.get('awayteam', {}).get('@id', '')
            features['home_score'] = safe_int(game_data.get('hometeam', {}).get('@totalscore', 0))
            features['away_score'] = safe_int(game_data.get('awayteam', {}).get('@totalscore', 0))



            # --- Events summary ---
            events_data = game_data.get('events', {}) or {}
            event_features = self.extract_events(events_data)
            features['total_touchdowns'] = event_features.get('total_touchdowns', 0)
            features['total_field_goals'] = event_features.get('total_field_goals', 0)

            # --- Team stats (now with enhanced metrics) ---
            team_stats = game_data.get('team_stats', {})
            home_stats = team_stats.get('hometeam', {})
            away_stats = team_stats.get('awayteam', {})
            features.update(self.extract_team_stats(home_stats, 'home'))
            features.update(self.extract_team_stats(away_stats, 'away'))

            # --- Player stats for passing efficiency ---
            players_data = game_data.get('players', {})
            if players_data:
                home_players = players_data.get('hometeam', {}).get('player', [])
                away_players = players_data.get('awayteam', {}).get('player', [])
                
                if isinstance(home_players, dict):
                    home_players = [home_players]
                if isinstance(away_players, dict):
                    away_players = [away_players]
                
                features['home_passing_efficiency'] = self.calculate_team_passing_efficiency(home_players)
                features['away_passing_efficiency'] = self.calculate_team_passing_efficiency(away_players)

            # --- Standings ---
            standings_data = self.fetch_standings()
            standings_list = self._get_extract_standings_features(standings_data)

            def find_standing(team_name):
                for team_standing in standings_list:
                    if team_standing['team_name'].lower() == team_name.lower():
                        return team_standing
                return {}

            home_standing = find_standing(features['home_team'])
            away_standing = find_standing(features['away_team'])

            for key, val in home_standing.items():
                if key != 'team_name':
                    features[f"home_standings_{key}"] = val
            for key, val in away_standing.items():
                if key != 'team_name':
                    features[f"away_standings_{key}"] = val

            # --- Outcome Label ---
            features['home_win'] = (
                'home' if features['home_score'] > features['away_score']
                else 'away' if features['away_score'] > features['home_score']
                else 'draw'
            )

            # --- Ensure all expected features exist (optional safety net) ---
            expected_extra_features = ['total_touchdowns', 'total_field_goals', 'home_passing_efficiency', 'away_passing_efficiency']
            for col in expected_extra_features:
                features.setdefault(col, 0)

            # --- Sanitize numeric features for model prediction ---
            # If self.feature_cols exists, ensure all are numeric
            feature_cols = getattr(self, 'feature_cols', None)
            if feature_cols is not None:
                for col in feature_cols:
                    val = features.get(col, 0)
                    features[col] = safe_float(val) if val not in (None, '', 'NA', 'N/A') else 0.0

            return features

        except Exception as e:
            self.logger.error(f"Feature extraction error: {str(e)}")
            return {}

    def extract_all_games_features(self, game_data: dict) -> list:
        """
        Extract predictive features for all games contained in the XML data dict.
        Returns a list of feature dicts, one per game.
        """
        features_list = []
        try:
            categories = game_data.get('scores', {}).get('category', [])
            if isinstance(categories, dict):
                categories = [categories]

            for category in categories:
                match_list = category.get('match', [])
                if isinstance(match_list, dict):
                    match_list = [match_list]

                for game in match_list:
                    features = self.extract_predictive_features(game)
                    if features:
                        features_list.append(features)

        except Exception as e:
            self.logger.error(f"Error extracting all games features: {str(e)}")

        return features_list
    
    ################################################### Prepare Training Data ##############################################################

    def prepare_training_data(self, games: list) -> pd.DataFrame:
        features = []

        if not games or not isinstance(games, list):
            self.logger.warning("Empty or invalid games list passed for training data preparation.")
            return pd.DataFrame()

        for game in games:
            try:
                game_features = self.extract_predictive_features(game)
                if not game_features:
                    continue

                # Safety fallback: ensure expected columns always exist
                default_keys = [
                    'total_touchdowns', 'total_field_goals',
                    'home_score', 'away_score', 'home_passing_efficiency', 'away_passing_efficiency'
                ]
                for key in default_keys:
                    game_features.setdefault(key, 0)

                # --- Target label ---
                home_score = safe_int(game_features.get('home_score', 0))
                away_score = safe_int(game_features.get('away_score', 0))
                game_features['home_win'] = 1 if home_score > away_score else 0

                features.append(game_features)

            except Exception as e:
                self.logger.error(f"Skipping game due to error: {str(e)}")
                continue

        # Final DataFrame
        df = pd.DataFrame(features)

        # Optional: Drop non-numeric columns except target
        non_numeric_cols = df.select_dtypes(include=['object']).columns.tolist()
        non_numeric_cols = [col for col in non_numeric_cols if col != 'home_win']
        df = df.drop(columns=non_numeric_cols)

        # Optional: Fill missing values
        df.fillna(0, inplace=True)

        return df


