
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from app.services.MLB.gameDataExtractor import run_game_data_collect_final
from app.services.MLB.data_collect_final import run_batting_pitching_data_collect_final

import atexit



def start_scheduler():
    scheduler = BackgroundScheduler()
    # Run at midnight every day
    #scheduler.add_job(run_game_data_extractor, CronTrigger(hour=0, minute=0))
    # If you want to run the other script too, add another job:
    # scheduler.add_job(run_data_collect_final, CronTrigger(hour=0, minute=30))

    scheduler.add_job(run_batting_pitching_data_collect_final, CronTrigger(hour=0, minute=0))
    scheduler.add_job(run_game_data_collect_final, CronTrigger(hour=0, minute=30))

    scheduler.start()
    atexit.register(lambda: scheduler.shutdown())