import requests
import json
from geopy import Point
from geopy.distance import geodesic
from math import radians
import numpy as np

def get_distance_and_time(origin, destination, api_key = "AIzaSyBMkx9V1r0tE6yLdkaaqJu3Ub4oh4_su5A"):
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

def move_towards(origin, destination, distance_to_move):
    """Moves from one point to another using long and lat coordinates

    Args:
        origin: Address of origin
        destination: Address of destination
        distance_to_move: Amount of kilometers to move from the origin to destination

    Returns:
        updated_address: Address of new position after travelling an amount of kilometers
    """

    originCoords = get_lat_lng(origin)
    destinationCoords = get_lat_lng(destination)

    origin_point = Point(originCoords)
    destination_point = Point(destinationCoords)

    # get the direction from the origin to the destination
    direction = calculate_initial_compass_bearing(originCoords, destinationCoords)

    # calculate new point using the direction and the distance
    new_point = geodesic(kilometers=distance_to_move).destination(origin_point, direction)

    updated_address = get_address(new_point[0], new_point[1])

    return updated_address

def get_lat_lng(address, apiKey = "AIzaSyBMkx9V1r0tE6yLdkaaqJu3Ub4oh4_su5A"):
    """Gets the latitude and longitude of an address

    Args:
        address: Address to be converted
        api_key: API key for the Google Maps Distance Matrix API

    Returns:
        A tuple (lat, lng) where:
        - lat: Latitude of location
        - lng: Longitude of location
    """

    url = ('https://maps.googleapis.com/maps/api/geocode/json?address={}&key={}'
           .format(address.replace(' ', '+'), apiKey))
    try:
        response = requests.get(url)
        resp_json_payload = response.json()
        lat = resp_json_payload['results'][0]['geometry']['location']['lat']
        lng = resp_json_payload['results'][0]['geometry']['location']['lng']
    except:
        print('ERROR: {}'.format(resp_json_payload['status']))
        return None
    return lat, lng

def get_address(latitude, longitude, api_key = "AIzaSyBMkx9V1r0tE6yLdkaaqJu3Ub4oh4_su5A"):
    """Gets the latitude and longitude of an address

    Args:
        latitude: Latitude to be converted
        longitude: Longitude to be converted
        api_key: API key for the Google Maps Distance Matrix API

    Returns:
        address: Address of location
    """

    url = ('https://maps.googleapis.com/maps/api/geocode/json?latlng={},{}&key={}'
           .format(latitude, longitude, api_key))
    try:
        response = requests.get(url)
        resp_json_payload = response.json()
        address = resp_json_payload['results'][0]['formatted_address']
    except:
        print('ERROR: {}'.format(resp_json_payload['status']))
        return None
    return address

def calculate_initial_compass_bearing(pointA, pointB):
    """Moves from one point to another using long and lat coordinates

    Args:
        pointA: Latitude and longitude of a point A
        pointB: Latitude and longitude of a point B

    Returns:
        compass_bearing: Value used for getting direction of movement
    """

    if (type(pointA) != tuple) or (type(pointB) != tuple):
        raise TypeError("Only tuples are supported as arguments")

    lat1 = radians(pointA[0])
    lat2 = radians(pointB[0])

    diffLong = radians(pointB[1] - pointA[1])

    x = np.sin(diffLong) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - (np.sin(lat1) * np.cos(lat2) * np.cos(diffLong))

    initial_bearing = np.arctan2(x, y)

    # Now we have the initial bearing but math.atan2() returns values from -π to + π so we need to normalize the result
    # by converting it to a compass bearing as it is in the range 0° to 360°
    initial_bearing = np.degrees(initial_bearing)
    compass_bearing = (initial_bearing + 360) % 360

    return compass_bearing