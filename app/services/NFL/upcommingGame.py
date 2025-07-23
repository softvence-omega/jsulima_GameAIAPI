import os
from datetime import datetime
from dotenv import load_dotenv
from app.services.NFL.data_processor import NflDataProcessor
import json

# Load API Key from .env
load_dotenv()
api_key = os.getenv("GOALSERVE_API_KEY")

class Upcomming_nfl_game:
    def __init__(self):
        self.data_processor = NflDataProcessor(api_key)

    def upcoming_games(self):
        today = datetime.now()
        today_date = today.strftime("%d.%m.%Y")
        current_month = today.month

        # print(f"📅 Checking NFL schedule for today: {today_date}")

        # 🧠 Determine NFL season
        current_season = str(today.year)
        # print(f"🏈 NFL season assumed: {current_season}")

        # 🧠 Adjust date range based on offseason
        if 3 <= current_month <= 6:
            print("ℹ️ Offseason detected — looking up games starting July onward!")
            start_date = f"01.07.{current_season}"
            end_date = f"31.12.{current_season}"
        elif current_month < 7:
            start_date = f"01.07.{current_season}"
            end_date = f"31.12.{current_season}"
        else:
            start_date = f"01.01.{current_season}"
            end_date = f"31.12.{current_season}"
        # print(f"📡 Fetching NFL games from {start_date} to {end_date}...")

        try:
            endpoint = f"football/nfl-shedule?date1={start_date}&date2={end_date}"
            schedule_data = self.data_processor.fetch_data(endpoint)
            schedules = schedule_data.get("shedules", {})
            tournaments = schedules.get("tournament", [])

            # Ensure tournaments is a list
            if isinstance(tournaments, dict):
                tournaments = [tournaments]

            game_data = []
            for tournament in tournaments:
                weeks = tournament.get("week", [])
                if isinstance(weeks, dict):
                    weeks = [weeks]
                for week in weeks:
                    matches = week.get("matches", [])
                    time_zone = None
                    if isinstance(matches, dict):
                        time_zone = matches.get('@timezone', '')
                    elif isinstance(matches, list) and matches:
                        time_zone = matches[0].get('@timezone', '')
                    if isinstance(matches, dict):
                        matches = [matches]
                    for match_group in matches:
                        match_data = match_group.get("match", [])
                        if isinstance(match_data, dict):
                            match_data = [match_data]
                        for match in match_data:
                            try:
                                game_data.append(match)  # Just append the full match dict!
                            except Exception as e:
                               print(f"Error processing match data: {e}")
            return game_data
        except Exception as e:
            print(f"Error fetching or processing NFL schedule: {e}")
            return []

if __name__ == "__main__":
    upcomming_nfl_game = Upcomming_nfl_game()
    games = upcomming_nfl_game.upcoming_games()
    print("🏈 NFL Schedule for Upcoming Games")
    print("-"*200)
    print(type(games))
    print(type(games[0]))
    print("🏈 End of NFL schedule") 
    print(len(games))


