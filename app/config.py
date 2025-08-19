import os
from dotenv import load_dotenv

load_dotenv()
GOALSERVE_API_KEY= os.getenv('GOALSERVE_API_KEY')
GOALSERVE_BASE_URL = os.getenv("GOALSERVE_BASE_URL", "https://www.goalserve.com/getfeed/")
# Initialize the data directory path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
NFL_DIR= os.path.join(DATA_DIR, "NFL")
NFL_MODEL_DIR = os.path.join(NFL_DIR, "models")


IMAGE_URL = "http://172.83.15.114:8000/player/"

# Create the data directory if it doesn't exist
os.makedirs(DATA_DIR, exist_ok=True)
URLS_NFL = [
    ("Live Scores", f"{GOALSERVE_BASE_URL}{GOALSERVE_API_KEY}/football/nfl-scores"),
    ("Injuries", f"{GOALSERVE_BASE_URL}{GOALSERVE_API_KEY}/football/1691_injuries"),
    ("Player Stats", f"{GOALSERVE_BASE_URL}{GOALSERVE_API_KEY}/football/1691_player_stats"),
    ("Rosters", f"{GOALSERVE_BASE_URL}{GOALSERVE_API_KEY}/football/1691_rosters"),
    ("Scores", f"{GOALSERVE_BASE_URL}{GOALSERVE_API_KEY}/football/nfl-scores?date=01.01.2024"),
    ("Play-by-Play", f"{GOALSERVE_BASE_URL}{GOALSERVE_API_KEY}/football/nfl-playbyplay-scores"),
    ("Schedule", f"{GOALSERVE_BASE_URL}{GOALSERVE_API_KEY}/football/nfl-shedule"),
    ("Standings", f"{GOALSERVE_BASE_URL}{GOALSERVE_API_KEY}/football/nfl-standings"),
    ("Odds Schedule",
     f"{GOALSERVE_BASE_URL}{GOALSERVE_API_KEY}/football/nfl-shedule?date1=01.01.2024&date2=02.01.2024&showodds=1"),
    ("Player images", f"{GOALSERVE_BASE_URL}{GOALSERVE_API_KEY}/football/usa?playerimage=15826")
]

