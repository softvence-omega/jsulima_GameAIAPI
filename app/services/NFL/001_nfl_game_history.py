import requests
import csv
import json
import time
from datetime import datetime, timedelta
import os
from typing import Dict, List, Optional, Any
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('nfl_scraper.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class NFLDataFetcher:
    def __init__(self, base_url: str, output_file: str = "nfl_games_2025.csv"):
        """
        Initialize the NFL data fetcher
        
        Args:
            base_url: Base URL template with date parameter
            output_file: CSV file path to save the data
        """
        self.base_url = base_url
        self.output_file = output_file
        self.session = requests.Session()
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(self.output_file), exist_ok=True)
        
        # Set up session headers
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'application/json, text/plain, */*',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        })

        # CSV field names
        self.csv_headers = [
            'date', 'datetime_utc', 'contest_id', 'venue_name', 'attendance', 'status',
            'home_team_id', 'home_team_name', 'home_q1', 'home_q2', 'home_q3', 'home_q4', 'home_total_score',
            'away_team_id', 'away_team_name', 'away_q1', 'away_q2', 'away_q3', 'away_q4', 'away_total_score',
            
            # Team Stats - Home
            'home_first_downs', 'home_third_down_eff', 'home_fourth_down_eff', 'home_passing_yards',
            'home_rushing_yards', 'home_total_yards', 'home_turnovers', 'home_penalties', 'home_possession',
            'home_sacks', 'home_interceptions', 'home_fumbles_recovered', 'home_red_zone_eff',
            
            # Team Stats - Away
            'away_first_downs', 'away_third_down_eff', 'away_fourth_down_eff', 'away_passing_yards',
            'away_rushing_yards', 'away_total_yards', 'away_turnovers', 'away_penalties', 'away_possession',
            'away_sacks', 'away_interceptions', 'away_fumbles_recovered', 'away_red_zone_eff',
            
            # Top Performers
            'home_top_passer', 'home_passing_yards_leader', 'home_passing_tds_leader',
            'home_top_rusher', 'home_rushing_yards_leader', 'home_rushing_tds_leader',
            'home_top_receiver', 'home_receiving_yards_leader', 'home_receiving_tds_leader',
            
            'away_top_passer', 'away_passing_yards_leader', 'away_passing_tds_leader',
            'away_top_rusher', 'away_rushing_yards_leader', 'away_rushing_tds_leader',
            'away_top_receiver', 'away_receiving_yards_leader', 'away_receiving_tds_leader',
            
            # Kicking
            'home_kicker', 'home_field_goals', 'home_extra_points',
            'away_kicker', 'away_field_goals', 'away_extra_points',
            
            # Game Events Summary
            'total_touchdowns', 'total_field_goals', 'total_turnovers', 'game_length'
        ]

    def format_date(self, date_obj: datetime) -> str:
        """Format date to DD.MM.YYYY format as required by the API"""
        return date_obj.strftime("%d.%m.%Y")

    def safe_get(self, data: Dict, *keys, default="") -> Any:
        """Safely get nested dictionary values"""
        try:
            for key in keys:
                if isinstance(data, dict):
                    data = data.get(key, {})
                elif isinstance(data, list) and isinstance(key, int):
                    data = data[key] if len(data) > key else {}
                else:
                    return default
            return data if data is not None else default
        except (KeyError, IndexError, TypeError):
            return default

    def extract_team_stats(self, team_stats: Dict, prefix: str) -> Dict:
        """Extract team statistics"""
        stats = {}
        
        # First downs
        first_downs = self.safe_get(team_stats, 'first_downs', 'total', default=0)
        third_down = self.safe_get(team_stats, 'first_downs', 'third_down_efficiency', default="0-0")
        fourth_down = self.safe_get(team_stats, 'first_downs', 'fourth_down_efficiency', default="0-0")
        
        stats[f'{prefix}_first_downs'] = first_downs
        stats[f'{prefix}_third_down_eff'] = third_down
        stats[f'{prefix}_fourth_down_eff'] = fourth_down
        
        # Passing
        stats[f'{prefix}_passing_yards'] = self.safe_get(team_stats, 'passing', 'total', default=0)
        
        # Rushing
        stats[f'{prefix}_rushing_yards'] = self.safe_get(team_stats, 'rushings', 'total', default=0)
        
        # Total yards
        stats[f'{prefix}_total_yards'] = self.safe_get(team_stats, 'yards', 'total', default=0)
        
        # Turnovers
        stats[f'{prefix}_turnovers'] = self.safe_get(team_stats, 'turnovers', 'total', default=0)
        
        # Penalties
        stats[f'{prefix}_penalties'] = self.safe_get(team_stats, 'penalties', 'total', default="0-0")
        
        # Possession
        stats[f'{prefix}_possession'] = self.safe_get(team_stats, 'posession', 'total', default="00:00")
        
        # Sacks
        stats[f'{prefix}_sacks'] = self.safe_get(team_stats, 'sacks', 'total', default=0)
        
        # Interceptions
        stats[f'{prefix}_interceptions'] = self.safe_get(team_stats, 'interceptions', 'total', default=0)
        
        # Fumbles recovered
        stats[f'{prefix}_fumbles_recovered'] = self.safe_get(team_stats, 'fumbles_recovered', 'total', default=0)
        
        # Red zone efficiency
        stats[f'{prefix}_red_zone_eff'] = self.safe_get(team_stats, 'red_zone', 'made_att', default="0-0")
        
        return stats

    def extract_top_performers(self, game_data: Dict, team_type: str, prefix: str) -> Dict:
        """Extract top performing players for each category"""
        performers = {}
        
        # Passing
        passing_data = self.safe_get(game_data, 'passing', team_type)
        if isinstance(passing_data, dict) and 'player' in passing_data:
            player = passing_data['player']
            if isinstance(player, list):
                player = player[0]  # Get the first (usually top) passer
            performers[f'{prefix}_top_passer'] = self.safe_get(player, 'name', default="")
            performers[f'{prefix}_passing_yards_leader'] = self.safe_get(player, 'yards', default=0)
            performers[f'{prefix}_passing_tds_leader'] = self.safe_get(player, 'passing_touch_downs', default=0)
        else:
            performers[f'{prefix}_top_passer'] = ""
            performers[f'{prefix}_passing_yards_leader'] = 0
            performers[f'{prefix}_passing_tds_leader'] = 0
        
        # Rushing
        rushing_data = self.safe_get(game_data, 'rushing', team_type)
        if isinstance(rushing_data, dict) and 'player' in rushing_data:
            player = rushing_data['player']
            if isinstance(player, list):
                # Safely find player with max yards
                max_yards = 0
                top_player = None
                for p in player:
                    try:
                        yards = int(self.safe_get(p, 'yards', default=0))
                        if yards > max_yards:
                            max_yards = yards
                            top_player = p
                    except (ValueError, TypeError):
                        continue
                player = top_player if top_player else player[0] if player else {}
            performers[f'{prefix}_top_rusher'] = self.safe_get(player, 'name', default="")
            performers[f'{prefix}_rushing_yards_leader'] = self.safe_get(player, 'yards', default=0)
            performers[f'{prefix}_rushing_tds_leader'] = self.safe_get(player, 'rushing_touch_downs', default=0)
        else:
            performers[f'{prefix}_top_rusher'] = ""
            performers[f'{prefix}_rushing_yards_leader'] = 0
            performers[f'{prefix}_rushing_tds_leader'] = 0
        
        # Receiving
        receiving_data = self.safe_get(game_data, 'receiving', team_type)
        if isinstance(receiving_data, dict) and 'player' in receiving_data:
            player = receiving_data['player']
            if isinstance(player, list):
                # Safely find player with max yards
                max_yards = 0
                top_player = None
                for p in player:
                    try:
                        yards = int(self.safe_get(p, 'yards', default=0))
                        if yards > max_yards:
                            max_yards = yards
                            top_player = p
                    except (ValueError, TypeError):
                        continue
                player = top_player if top_player else player[0] if player else {}
            performers[f'{prefix}_top_receiver'] = self.safe_get(player, 'name', default="")
            performers[f'{prefix}_receiving_yards_leader'] = self.safe_get(player, 'yards', default=0)
            performers[f'{prefix}_receiving_tds_leader'] = self.safe_get(player, 'receiving_touch_downs', default=0)
        else:
            performers[f'{prefix}_top_receiver'] = ""
            performers[f'{prefix}_receiving_yards_leader'] = 0
            performers[f'{prefix}_receiving_tds_leader'] = 0
        
        return performers

    def extract_kicking_stats(self, game_data: Dict, team_type: str, prefix: str) -> Dict:
        """Extract kicking statistics"""
        kicking_stats = {}
        
        kicking_data = self.safe_get(game_data, 'kicking', team_type)
        if isinstance(kicking_data, dict) and 'player' in kicking_data:
            player = kicking_data['player']
            kicking_stats[f'{prefix}_kicker'] = self.safe_get(player, 'name', default="")
            kicking_stats[f'{prefix}_field_goals'] = self.safe_get(player, 'field_goals', default="0/0")
            kicking_stats[f'{prefix}_extra_points'] = self.safe_get(player, 'extra_point', default="0/0")
        else:
            kicking_stats[f'{prefix}_kicker'] = ""
            kicking_stats[f'{prefix}_field_goals'] = "0/0"
            kicking_stats[f'{prefix}_extra_points'] = "0/0"
        
        return kicking_stats

    def count_game_events(self, events: Dict) -> Dict:
        """Count various game events"""
        total_tds = 0
        total_fgs = 0
        
        for quarter in ['firstquarter', 'secondquarter', 'thirdquarter', 'fourthquarter', 'overtime']:
            quarter_events = self.safe_get(events, quarter)
            if quarter_events and 'event' in quarter_events:
                event_list = quarter_events['event']
                if isinstance(event_list, dict):
                    event_list = [event_list]
                elif isinstance(event_list, list):
                    pass
                else:
                    continue
                
                for event in event_list:
                    event_type = self.safe_get(event, 'type', default="")
                    if event_type == 'TD':
                        total_tds += 1
                    elif event_type == 'FG':
                        total_fgs += 1
        
        return {
            'total_touchdowns': total_tds,
            'total_field_goals': total_fgs
        }

    def parse_game_data(self, data: Dict) -> List[Dict]:
        """Parse game data and return list of game records"""
        games = []
        
        if not data or 'scores' not in data:
            return games
        
        scores = data['scores']
        
        # Handle different response structures
        if isinstance(scores, dict) and 'category' in scores:
            match_data = self.safe_get(scores, 'category', 'match')
            if match_data:
                # Handle both single match (dict) and multiple matches (list)
                if isinstance(match_data, list):
                    # Multiple matches on the same day
                    for single_match in match_data:
                        temp = self.extract_game_record(single_match)
                        if temp:
                            games.append(temp)
                else:
                    # Single match (dict)
                    temp = self.extract_game_record(match_data)
                    if temp:
                        games.append(temp)
        elif isinstance(scores, list):
            for score in scores:
                if 'category' in score:
                    match_data = self.safe_get(score, 'category', 'match')
                    if match_data:
                        # Handle both single match (dict) and multiple matches (list)
                        if isinstance(match_data, list):
                            # Multiple matches on the same day
                            for single_match in match_data:
                                temp = self.extract_game_record(single_match)
                                if temp:
                                    games.append(temp)
                        else:
                            # Single match (dict)
                            temp = self.extract_game_record(match_data)
                            if temp:
                                games.append(temp)
        
        return games

    def extract_game_record(self, match_data: Dict) -> Optional[Dict]:
        """Extract a single game record"""
        # Skip games that don't have a status or are not finished
        status = self.safe_get(match_data, 'status', default="")
        if not status or status.lower() not in ['final', 'completed']:
            return None
        
        record = {}
        
        # Basic game info
        record['date'] = self.safe_get(match_data, 'date', default="")
        record['datetime_utc'] = self.safe_get(match_data, 'datetime_utc', default="")
        record['contest_id'] = self.safe_get(match_data, 'contestID', default="")
        record['venue_name'] = self.safe_get(match_data, 'venue_name', default="")
        record['attendance'] = self.safe_get(match_data, 'attendance', default=0)
        record['status'] = status
        
        # Team info
        home_team = self.safe_get(match_data, 'hometeam', default={})
        away_team = self.safe_get(match_data, 'awayteam', default={})
        
        record['home_team_id'] = self.safe_get(home_team, 'id', default="")
        record['home_team_name'] = self.safe_get(home_team, 'name', default="")
        record['home_q1'] = self.safe_get(home_team, 'q1', default=0)
        record['home_q2'] = self.safe_get(home_team, 'q2', default=0)
        record['home_q3'] = self.safe_get(home_team, 'q3', default=0)
        record['home_q4'] = self.safe_get(home_team, 'q4', default=0)
        record['home_total_score'] = self.safe_get(home_team, 'totalscore', default=0)
        
        record['away_team_id'] = self.safe_get(away_team, 'id', default="")
        record['away_team_name'] = self.safe_get(away_team, 'name', default="")
        record['away_q1'] = self.safe_get(away_team, 'q1', default=0)
        record['away_q2'] = self.safe_get(away_team, 'q2', default=0)
        record['away_q3'] = self.safe_get(away_team, 'q3', default=0)
        record['away_q4'] = self.safe_get(away_team, 'q4', default=0)
        record['away_total_score'] = self.safe_get(away_team, 'totalscore', default=0)
        
        # Team stats
        team_stats = self.safe_get(match_data, 'team_stats', default={})
        home_stats = self.extract_team_stats(self.safe_get(team_stats, 'hometeam', default={}), 'home')
        away_stats = self.extract_team_stats(self.safe_get(team_stats, 'awayteam', default={}), 'away')
        record.update(home_stats)
        record.update(away_stats)
        
        # Top performers
        home_performers = self.extract_top_performers(match_data, 'hometeam', 'home')
        away_performers = self.extract_top_performers(match_data, 'awayteam', 'away')
        record.update(home_performers)
        record.update(away_performers)
        
        # Kicking stats
        home_kicking = self.extract_kicking_stats(match_data, 'hometeam', 'home')
        away_kicking = self.extract_kicking_stats(match_data, 'awayteam', 'away')
        record.update(home_kicking)
        record.update(away_kicking)
        
        # Game events
        events = self.safe_get(match_data, 'events', default={})
        event_counts = self.count_game_events(events)
        record.update(event_counts)
        
        # Calculate total turnovers
        try:
            home_turnovers = int(self.safe_get(record, 'home_turnovers', default=0))
        except (ValueError, TypeError):
            home_turnovers = 0
        
        try:
            away_turnovers = int(self.safe_get(record, 'away_turnovers', default=0))
        except (ValueError, TypeError):
            away_turnovers = 0
            
        record['total_turnovers'] = home_turnovers + away_turnovers
        
        # Game length (placeholder - would need to calculate from events)
        record['game_length'] = "3:00:00"  # Standard NFL game length approximation
        
        return record

    def initialize_csv(self):
        """Initialize CSV file with headers"""
        file_exists = os.path.exists(self.output_file)
        if not file_exists:
            with open(self.output_file, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=self.csv_headers)
                writer.writeheader()
            logger.info(f"Created CSV file: {self.output_file}")

    def append_to_csv(self, games: List[Dict]):
        """Append game records to CSV file"""
        if not games:
            return
        
        with open(self.output_file, 'a', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=self.csv_headers)
            for game in games:
                # Ensure all fields are present
                row = {field: game.get(field, "") for field in self.csv_headers}
                writer.writerow(row)

    def fetch_data_for_date(self, date_str: str) -> Optional[Dict]:
        """
        Fetch NFL data for a specific date
        Returns None if no games found (500 error), raises exception for other errors
        """
        url = self.base_url.replace('09.02.2025', date_str)
        
        try:
            logger.info(f"Fetching data for {date_str}")
            response = self.session.get(url, timeout=30)
            
            # Handle different status codes appropriately
            if response.status_code == 500:
                # 500 typically means no games that day - skip without retry
                logger.debug(f"No games found for {date_str} (500 status)")
                return None
            elif response.status_code == 429:
                # Rate limited - this should be retried
                logger.warning(f"Rate limited for {date_str} (429 status)")
                raise requests.exceptions.HTTPError(f"Rate limited (429)", response=response)
            else:
                # For other status codes, raise for status (will raise for 4xx, 5xx except 500)
                response.raise_for_status()
            
            data = response.json()
            
            if data and 'scores' in data and data['scores']:
                logger.info(f"SUCCESS: Found game data for {date_str}")
                return data
            else:
                logger.debug(f"No games found for {date_str} (empty response)")
                return None
                
        except requests.exceptions.HTTPError as e:
            # Re-raise HTTP errors (like 429) so they can be retried
            if hasattr(e, 'response') and e.response.status_code == 429:
                raise
            else:
                logger.error(f"HTTP error for {date_str}: {str(e)}")
                return None
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed for {date_str}: {str(e)}")
            raise  # Re-raise for potential retry
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error for {date_str}: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error for {date_str}: {str(e)}")
            raise  # Re-raise for potential retry

    def fetch_all_data(self, start_date: datetime = None, end_date: datetime = None, 
                      delay: float = 1.0, retry_attempts: int = 3) -> Dict:
        """Fetch all NFL data for 2025 and save to CSV"""
        if start_date is None:
            start_date = datetime(2025, 7, 31)
        if end_date is None:
            end_date = datetime(2025, 8, 31)
        
        logger.info(f"Starting NFL data fetch from {start_date.date()} to {end_date.date()}")
        
        # Initialize CSV file
        self.initialize_csv()
        
        total_days = (end_date - start_date).days + 1
        successful_fetches = 0
        failed_fetches = 0
        no_games_days = 0
        rate_limited_days = 0
        games_saved = 0
        
        current_date = start_date
        
        while current_date <= end_date:
            date_str = self.format_date(current_date)
            
            # Retry logic - only retry on specific errors (429, network issues)
            data = None
            should_retry = True
            
            for attempt in range(retry_attempts):
                try:
                    data = self.fetch_data_for_date(date_str)
                    should_retry = False
                    break
                except requests.exceptions.HTTPError as e:
                    if hasattr(e, 'response') and e.response.status_code == 429:
                        # Rate limited - retry with exponential backoff
                        rate_limited_days += 1
                        wait_time = delay * (2 ** attempt)  # Exponential backoff
                        logger.warning(f"Rate limited for {date_str}, waiting {wait_time}s before retry {attempt + 1}/{retry_attempts}")
                        time.sleep(wait_time)
                    else:
                        # Other HTTP errors - don't retry
                        logger.error(f"HTTP error for {date_str}: {str(e)}")
                        should_retry = False
                        break
                except requests.exceptions.RequestException as e:
                    # Network/connection issues - retry
                    if attempt < retry_attempts - 1:
                        wait_time = delay * 2
                        logger.warning(f"Network error for {date_str}, retrying in {wait_time}s: {str(e)}")
                        time.sleep(wait_time)
                    else:
                        logger.error(f"Network error for {date_str} after {retry_attempts} attempts: {str(e)}")
                        should_retry = False
                        break
                except Exception as e:
                    # Unexpected errors - retry
                    if attempt < retry_attempts - 1:
                        wait_time = delay * 2
                        logger.warning(f"Unexpected error for {date_str}, retrying in {wait_time}s: {str(e)}")
                        time.sleep(wait_time)
                    else:
                        logger.error(f"Unexpected error for {date_str} after {retry_attempts} attempts: {str(e)}")
                        should_retry = False
                        break
            
            if data is not None:
                # Parse and save games
                games = self.parse_game_data(data)
                if games:
                    self.append_to_csv(games)
                    games_saved += len(games)
                    logger.info(f"SUCCESS: Saved {len(games)} game(s) for {date_str}")
                else:
                    # Data received but no completed games found
                    logger.debug(f"Data received for {date_str} but no completed games found")
                
                successful_fetches += 1
            elif data is None and not should_retry:
                # No games that day (500 status) - this is normal
                no_games_days += 1
                logger.debug(f"No games scheduled for {date_str}")
            else:
                # Failed after all retries
                failed_fetches += 1
                logger.error(f"Failed to fetch data for {date_str} after all retry attempts")
            
            # Progress update
            days_processed = (current_date - start_date).days + 1
            if days_processed % 30 == 0 or current_date == end_date:
                progress = (days_processed / total_days) * 100
                logger.info(f"Progress: {progress:.1f}% ({days_processed}/{total_days} days) - Games saved: {games_saved}")
            
            current_date += timedelta(days=1)
            
            # Rate limiting - only sleep if we didn't already wait due to rate limiting
            if delay > 0 and rate_limited_days == 0:
                time.sleep(delay)
        
        # Create summary
        summary = {
            'total_days_processed': total_days,
            'successful_fetches': successful_fetches,
            'no_games_days': no_games_days,
            'rate_limited_days': rate_limited_days,
            'failed_fetches': failed_fetches,
            'games_saved': games_saved,
            'success_rate': (successful_fetches / total_days) * 100,
            'start_date': start_date.isoformat(),
            'end_date': end_date.isoformat(),
            'output_file': self.output_file
        }
        
        logger.info("=== FETCH COMPLETE ===")
        logger.info(f"Total days processed: {total_days}")
        logger.info(f"Successful fetches (with games): {successful_fetches}")
        logger.info(f"No games days: {no_games_days}")
        logger.info(f"Rate limited encounters: {rate_limited_days}")
        logger.info(f"Failed fetches: {failed_fetches}")
        logger.info(f"Games saved to CSV: {games_saved}")
        logger.info(f"Success rate: {summary['success_rate']:.1f}%")
        logger.info(f"Data saved to: {self.output_file}")
        
        return summary

def main():
    """Main function to run the NFL data fetcher"""
    
    try:
        # Try to import config (if available)
        from app.config import GOALSERVE_BASE_URL, GOALSERVE_API_KEY
        BASE_URL = f"{GOALSERVE_BASE_URL}{GOALSERVE_API_KEY}/football/nfl-scores?json=1&date=09.02.2025"
        OUTPUT_FILE = "app/data/NFL/nfl_games_data_history.csv"
    except ImportError:
        # Fallback if config is not available
        BASE_URL = "https://www.goalserve.com/getfeed/YOUR_API_KEY/football/nfl-scores?json=1&date=09.02.2025"
        OUTPUT_FILE = "nfl_games_data_history.csv"
        logger.warning("Config not found, using fallback URL. Please update with your API key.")
    
    REQUEST_DELAY = 1.0  # seconds between requests
    
    # Initialize fetcher
    fetcher = NFLDataFetcher(BASE_URL, OUTPUT_FILE)
    
    try:
        # Fetch all data for 2025
        summary = fetcher.fetch_all_data(delay=REQUEST_DELAY)
        
        # Print final summary
        print("\n" + "="*50)
        print("NFL DATA FETCH SUMMARY")
        print("="*50)
        print(f"Total days processed: {summary['total_days_processed']}")
        print(f"Successful fetches (with games): {summary['successful_fetches']}")
        print(f"No games days: {summary['no_games_days']}")
        print(f"Rate limited encounters: {summary['rate_limited_days']}")
        print(f"Failed fetches: {summary['failed_fetches']}")
        print(f"Total games saved: {summary['games_saved']}")
        print(f"Success rate: {summary['success_rate']:.1f}%")
        print(f"CSV file: {summary['output_file']}")
        print("="*50)
        
    except KeyboardInterrupt:
        logger.info("Fetch interrupted by user")
        print("\nFetch interrupted by user. Partial data may be available in CSV.")
    except Exception as e:
        logger.error(f"Unexpected error in main: {str(e)}")
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()