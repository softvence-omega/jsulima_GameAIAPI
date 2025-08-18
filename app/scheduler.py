
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from app.services.MLB.gameDataExtractor import run_game_data_collect_final
from app.services.MLB.data_collect_final import run_batting_pitching_data_collect_final
from app.services.NFL.extract_data_nfl import extract_todays_match_data
from app.services.NFL._nfl_game_history import run_game_data_collect_nfl
import atexit



def start_scheduler():
    scheduler = BackgroundScheduler()
    
    # MLB
    scheduler.add_job(run_batting_pitching_data_collect_final, CronTrigger(hour=22, minute=0))
    scheduler.add_job(run_game_data_collect_final, CronTrigger(hour=22, minute=30))

    # NFL
    scheduler.add_job(extract_todays_match_data, CronTrigger(hour=0, minute=0))
    scheduler.add_job(run_game_data_collect_nfl, CronTrigger(hour=0, minute=30))


    scheduler.start()
    atexit.register(lambda: scheduler.shutdown())