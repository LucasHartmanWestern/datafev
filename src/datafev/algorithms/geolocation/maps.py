import requests
import json

def get_distance_and_time(origin, destination, api_key = "AIzaSyAjdCQ_Qbm3tfYOH8HwXprBxD3siW3J-Qk"):
    """Gets the distance and travel-time between two addresses using the google maps API

    Args:
        origin: Address of origin
        destination: Address of destination
        api_key: API key for the Google Maps Distance Matrix API

    Returns:
        A tuple (distance, time) where:
        - distance: The physical distance measured in meters
        - time: The travel time measured in seconds
    """

    url = "https://maps.googleapis.com/maps/api/distancematrix/json?units=metric"

    r = requests.get(url + "&origins=" + origin + "&destinations=" + destination + "&key=" + api_key)
    data = r.json()

    if not data['rows']:
        return "No data found"

    if not data['rows'][0]['elements']:
        return "No elements in data found"

    element_data = data['rows'][0]['elements'][0]

    if 'distance' not in element_data or 'duration' not in element_data:
        return "Could not find distance or duration in element data"

    distance = element_data['distance']['value']
    time = element_data['duration']['value']

    return distance, time