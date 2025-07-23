
from app.services.MLB.live_prediction import LiveGamePredictor
from app.config import GOALSERVE_API_KEY, GOALSERVE_BASE_URL
from app.services.MLB.load_model_and_predict_win import win_prediction
import os
from dotenv import load_dotenv
import json
from fastapi import APIRouter
from datetime import datetime, timedelta
import requests

router=APIRouter()

def _force_list(x):
    if isinstance(x, list):
        return x 
    return [x]

def get_future_matches():
    """
    Fetches future MLB matches from the GoalServe API.
    
    Returns:
        list: A list of future match dictionaries.
    """
    date1 = datetime.now().strftime("%d.%m.%Y")
    date2 = (datetime.now() + timedelta(days=365)).strftime("%d.%m.%Y")
    url = f"{GOALSERVE_BASE_URL}{GOALSERVE_API_KEY}/baseball/mlb_shedule?json=1&date1={date1}&date2={date2}"

    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        ret = []
        for matches in data['fixtures']['category']['matches']:
            for match in _force_list(matches['match']):

                info = {
                    'date' : matches['@date'],
                    'timezone' : matches['@timezone'],
                    'seasonType' : matches['@seasonType'],
                    'venue': match['@venue_name'],
                    'formatted_date': match['@formatted_date'],
                    'datetime_utc': match['@datetime_utc'],
                    'hometeam' : match['hometeam']['@name'],
                    'awayteam' : match['awayteam']['@name']
                }

                ret.append(info)
                
        return ret
    else:
        raise Exception(f"Failed to fetch matches: {response.status_code} - {response.text}")

from datetime import datetime

def get_win_pred():
    matches = get_future_matches()
    ret = []
    for match in matches:
        pred = win_prediction(match['hometeam'], match['awayteam'], datetime.now().year)
        ret.append({
            "pred": pred,
            "info" : match
        })
    return ret



@router.get("/head-to-head-win-prediction")
async def games_prediction_endpoint():
    
    res = get_win_pred()
    
    return res

