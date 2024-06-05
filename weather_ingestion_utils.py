import requests
import json
from datetime import datetime
import sys

def fetch_weather_data(location, start_date, end_date, api_key):
    """
    Fetch weather data for a specific location and date range.

    Parameters:
    location (str): Location for which to fetch the weather data.
    start_date (str): Start date in the format 'YYYY-MM-DD'.
    end_date (str): End date in the format 'YYYY-MM-DD'.
    api_key (str): Your API key for Visual Crossing Weather.

    Returns:
    dict: Parsed JSON data containing the weather information.
    """
    # Ensure date range is not more than a year
    start = datetime.strptime(start_date, '%Y-%m-%d')
    end = datetime.strptime(end_date, '%Y-%m-%d')
    if (end - start).days > 365:
        print("Error: Date range cannot be more than a year apart.")
        return None

    # Construct the API URL
    base_url = "https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline"
    elements = "datetime,datetimeEpoch,temp,tempmax,tempmin,precip,windspeed,windgust,feelslike,feelslikemax,feelslikemin,pressure,stations,degreedays,accdegreedays"
    include = "fcst,obs,histfcst,stats"
    url = f"{base_url}/{location}/{start_date}/{end_date}?elements={elements}&include={include}&key={api_key}&contentType=json"

    try:
        # Fetch the data
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for HTTP errors
        jsonData = response.json()
        return jsonData

    except requests.exceptions.HTTPError as e:
        print('HTTP error:', e.response.text)
        sys.exit()
    except requests.exceptions.RequestException as e:
        print('Request error:', e)
        sys.exit()




# import urllib.request
# import sys

# import json
                
# try: 
#   ResultBytes = urllib.request.urlopen("https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/Bangkok%252CThailand/2019-01-01/2019-12-31?elements=datetime%2CdatetimeEpoch%2Ctemp%2Ctempmax%2Ctempmin%2Cprecip%2Cwindspeed%2Cwindgust%2Cfeelslike%2Cfeelslikemax%2Cfeelslikemin%2Cpressure%2Cstations%2Cdegreedays%2Caccdegreedays&include=fcst%2Cobs%2Chistfcst%2Cstats&key=YOUR_API_KEY&contentType=json")
  
#   # Parse the results as JSON
#   jsonData = json.load(ResultBytes)
        
# except urllib.error.HTTPError  as e:
#   ErrorInfo= e.read().decode() 
#   print('Error code: ', e.code, ErrorInfo)
#   sys.exit()
# except  urllib.error.URLError as e:
#   ErrorInfo= e.read().decode() 
#   print('Error code: ', e.code,ErrorInfo)
#   sys.exit()