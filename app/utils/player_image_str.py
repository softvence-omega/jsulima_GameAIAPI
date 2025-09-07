import json
import requests
import xml.etree.ElementTree as ET
import base64

from app.config import GOALSERVE_API_KEY, GOALSERVE_BASE_URL

def get_player_image(player_id: int) -> str:
    url = f"{GOALSERVE_BASE_URL}/{GOALSERVE_API_KEY}/football/usa?playerimage={player_id}"
    response = requests.get(url)
    xml_content = response.content 

    root = ET.fromstring(xml_content)

    return root.text

player_img_file = "app/data/NFL/player_images.csv"
import time 
from tqdm import tqdm

import os 

if __name__ == "__main__":
    file_path = "app/data/NFL/player_position_map.json"
    with open(file_path, "r") as f:
        player_position_map = json.load(f)

    ln = len(player_position_map)

    with open(player_img_file, "a") as f:
        #f.write("player_id,position,image_url\n")
        for player_id, position in tqdm(player_position_map.items(), total=ln, desc="Processing players"): 
            if os.path.exists(f"player_images/{player_id}.png"):
                continue
            try: 
                image_url = get_player_image(player_id)
                decoded_img = base64.b64decode(image_url)
                with open(f"player_images/{player_id}.png", "wb") as img_file:
                    img_file.write(decoded_img)
            except:
                # print(f"Failed to fetch image for player {player_id}")
                continue
