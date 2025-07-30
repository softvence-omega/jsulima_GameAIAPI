import requests
import xmltodict
import json


def xml_url_to_json(url: str) -> dict:
    #printf"Fetching XML data from----------: {url}")
    try:
        # Fetch XML from the URL
        response = requests.get(url)
        response.raise_for_status()  # Raise error for bad status codes
        
        # Parse XML to OrderedDict
        data_dict = xmltodict.parse(response.content)
        
        # Convert to normal JSON
        json_data = json.loads(json.dumps(data_dict))
        
        return json_data
    
    except Exception as e:
        #printf"Error: {e}")
        return {}
