import xmltodict
import json
import requests

from app.config import GOALSERVE_API_KEY, GOALSERVE_BASE_URL


def xml_to_json(URL: str):
    print('URL-------------------------->',URL)
    try:
        response = requests.get(URL)
        if response.status_code == 200:
            data_dict = xmltodict.parse(response.content)
            json_data = json.loads(json.dumps(data_dict))
            return json_data
        else:
            print("Failed to fetch XML data:", response.status_code)
            return None
    except Exception as e:
        print("An error occurred:", str(e))
        return None
        

#example usage
if __name__ == "__main__":
    xml_to_json(f"{GOALSERVE_BASE_URL}{GOALSERVE_API_KEY}/baseball/nym_injuries")
