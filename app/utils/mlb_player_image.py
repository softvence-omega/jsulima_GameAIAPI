from requests.models import Response
from typing import Any
from xml.etree.ElementTree import Element
import json
import requests
import xml.etree.ElementTree as ET
import base64
from app.config import GOALSERVE_API_KEY, GOALSERVE_BASE_URL
#from app.utils.player_image_str import player_position_map


def get_player_image(player_id: int) -> str:
    url: str = f"{GOALSERVE_BASE_URL}/{GOALSERVE_API_KEY}/baseball/usa?playerimage={player_id}"
    response: Response = requests.get(url)
    xml_content: bytes | Any = response.content 

    root: Element[str] = ET.fromstring(xml_content)

    return root.text

player_img_file = "app/data/NFL/player_images.csv"
import time 
from tqdm import tqdm
import pandas as pd
import os 
import sys 


if __name__ == "__main__":
    file_path = r"app\data\MLB\batter_stats_data(2010-2024).csv"
    with open(file_path, mode="r") as f:
        df = pd.read_csv(file_path)
        player_position_map = df['player_id'].to_list()
    

    ln = len(player_position_map)

    #f.write("player_id,position,image_url\n")
    for player_id in tqdm(player_position_map, total=ln, desc="Processing players"): 
        if os.path.exists(path=f"mlb_player_images/{player_id}.png"):
            continue
        try: 

            image_url: str = get_player_image(player_id)
            decoded_img: bytes = base64.b64decode(image_url)
            with open(f"mlb_player_images/{player_id}.png", mode="wb") as img_file:
                img_file.write(decoded_img)
        except Exception as e:
            print(f"Failed to fetch image for player {player_id}")
            print(e)
            continue
