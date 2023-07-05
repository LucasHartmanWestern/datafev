import copy
import csv
import math
import json
import random

import pandas as pd

from collections import OrderedDict
from geolocation.maps_free import get_closest_chargers, move_towards, get_distance_and_time
from geolocation.visualize import read_excel_data

from charging_station import ChargingStation

# TODO - Create simulation environment which is capable of:
#  - Introducing randomness (traffic will randomly fluctuate at charging stations based on time-of-day)
#  - Considering different makes and mmodels and decreasing (or increasing while charging) an EVs SoC propotionately
#  - Considering the load placed on each charging station relative to the traffic at the station
#  - Simulating the travel from an origin to destination

class EVSimEnvironment:
    def __init__(
            self,
            num_of_episodes,
            num_of_chargers,
            make,
            model,
            cur_soc,
            max_soc,
            org_lat,
            org_long,
            dest_lat,
            dest_long
    ):
        """Create environment

        Args:
            num_of_chargers: Amount of chargers to consider
            make: Make of EV
            model: Model of EV
            cur_soc: Current state of Charge of EV in Watts
            max_soc: Maximum SoC capability in Watts
            org_lat: Latitude of route origin
            org_long: Longitude of route origin
            dest_lat: Latitude of route destination
            dest_long: Longitude of route destination

        Returns:
            Environment to use for EV simulations
        """

        self.tracking_baseline = False

        self.num_of_episodes = num_of_episodes
        self.episode_num = -1
        self.visited_list = []
        self.current_path = []
        self.best_path = []
        self.used_chargers = []

        self.charger_list = [] # List of ChargingStation objects

        self.step_num = 0
        self.episode_reward = 0

        self.max_reward = math.inf * -1

        self.prev_distance = 0

        self.prev_charging = False

        self.num_of_chargers = num_of_chargers
        self.make = make
        self.model = model
        self.cur_soc = cur_soc
        self.max_soc = max_soc
        self.base_soc = cur_soc # Used to reset
        self.cur_lat = org_lat
        self.cur_long = org_long
        self.org_lat = org_lat
        self.org_long = org_long
        self.dest_lat = dest_lat
        self.dest_long = dest_long

        self.average_reward = []

        self.usage_per_hour, self.charge_per_hour = self.ev_info()

        self.is_charging = False

        self.get_charger_data()

        self.get_charger_list()
        self.update_state()

    # Used to get the coordinates of the chargers from the API dataset
    def get_charger_data(self):

        # From JSON file
        with open('data/Ontario_Charger_Dataset.json') as file:
            data = json.load(file)
        charger_data = []
        for station in data['fuel_stations']:
            charger_id = station['id']
            charger_lat = station['latitude']
            charger_long = station['longitude']
            charger_data.append([charger_id, charger_lat, charger_long])
        self.charger_info = pd.DataFrame(charger_data, columns=['id', 'latitude', 'longitude'])

        self.charger_lat = self.charger_info.iloc[:, 1].tolist()  # Extract values from the 3rd column
        self.charger_long = self.charger_info.iloc[:, 2].tolist()  # Extract values from the 4th column

    # Step function of sim environment
    def step(self, action):
        """Simulate a step in the EVs travel

        Args:
            action: Number to use as index for stations list

        Returns:
            next_state: New state of the system
            reward: Reward for current state
            done: Indicator for if simulation is done
        """

        self.step_num += 1

        current_state = copy.copy(self.state)

        # Update traffic, SoC, and geographical position
        done = self.move(action)
        self.update_traffic()
        self.update_charge(action)

        # Update state
        self.update_state()

        # Get reward of current state
        reward = self.reward(current_state, done)

        # Episode reward is the sum of the reward at each step
        self.episode_reward += reward

        # Log every tenth episode
        if self.episode_num % math.ceil(self.num_of_episodes / 10) == 0 or self.tracking_baseline:
            time_to_destination = get_distance_and_time((self.cur_lat, self.cur_long), (self.dest_lat, self.dest_long))[1] / 60
            if time_to_destination <= 15 and done:
                self.log(action, True)
            else:
                self.log(action)

        return current_state, reward, done

    # Return ChargingStation object by id
    def find_charging_station_by_id(self, target_id):
        for station in self.charger_list:
            if station.id == target_id:
                return station
        return None

    # Simulates battery life of EV as it travels
    def update_charge(self, action):

        charger_id = self.charger_coords[action - 1][0]
        station = self.find_charging_station_by_id(charger_id)

        # Find how far station is away from current coordinates in minutes
        time_to_station = get_distance_and_time((self.cur_lat, self.cur_long), (station.coord[0], station.coord[1]))[1] / 60

        # Consume battery while driving
        if self.is_charging is not True:
            self.cur_soc -= self.usage_per_hour / (60)

        # Increase battery while charging
        else:
            self.cur_soc += station.charge()
            # Cap SoC at max
            if self.cur_soc > self.max_soc:
                self.cur_soc = self.max_soc

        # Start charging if within range of charging station
        if action != 0 and time_to_station <= 1:
            self.is_charging = True
        else:
            self.is_charging = False
            station.leave()

    # Simulates traffic updates at chargers
    def update_traffic(self):
        for charger in self.charger_list:
            charger.update_traffic()

    # Simulates geographical movement of EV
    def move(self, action):
        done = False

        # Find out how far EV can travel given current charge
        usage_per_hour, charge_per_hour = self.ev_info()
        max_distance = self.cur_soc / (usage_per_hour / 60)
        travel_distance = min(max_distance, 1)

        # Find how far destination is away from current coordinates in minutes
        time_to_destination = get_distance_and_time((self.cur_lat, self.cur_long), (self.dest_lat, self.dest_long))[1] / 60

        # EV has reached destination
        if time_to_destination <= 1:
            done = True

        if done:
            return done

        # EV is driving to destination
        if action == 0:
            # Drive 15 minutes towards selected destination
            self.cur_lat, self.cur_long = move_towards((self.cur_lat, self.cur_long), (self.dest_lat, self.dest_long), travel_distance)

        # EV is driving towards a charging station
        else:
            # Find how far station is away from current coordinates in minutes
            time_to_station = get_distance_and_time((self.cur_lat, self.cur_long), (self.charger_coords[action - 1][1], self.charger_coords[action - 1][2]))[1] / 60

            # Arrive at charging station if within distance
            if time_to_station <= 1:
                # Arrive or stay at charging station
                self.cur_lat, self.cur_long = (self.charger_coords[action - 1][1], self.charger_coords[action - 1][2])

            # Not within distance of charging station yet, drive towards it for 15 minutes
            else:
                # Drive 15 minutes towards selected destination
                self.cur_lat, self.cur_long = move_towards((self.cur_lat, self.cur_long), (self.charger_coords[action - 1][1], self.charger_coords[action - 1][2]), travel_distance)

        # EV ran out of battery before reaching destination
        if self.cur_soc <= 0:
            done = True

        return done

    # Used to log the info for the paths
    def log(self, action, final = False, episode_offset = 0):
        new_row = []
        if self.tracking_baseline:
            new_row.append('Baseline')
        else:
            new_row.append(self.episode_num + episode_offset)
        if action == -1:
            new_row.append('No Action')
        elif action == 0:
            new_row.append(action)
        else:
            new_row.append(self.charger_coords[action - 1][0])
        new_row.append(self.step_num)
        new_row.append(round(self.cur_soc / 1000, 2))
        new_row.append(self.is_charging)
        new_row.append(round(self.episode_reward, 2))
        if final is not True:
            new_row.append(self.cur_lat)
            new_row.append(self.cur_long)
        else:
            new_row.append(self.dest_lat)
            new_row.append(self.dest_long)

        new_row.append(self.state)

        if self.tracking_baseline is not True:
            self.current_path.append(new_row)
        self.visited_list.append(new_row)

    def get_charger_list(self):
        list_of_chargers = list(zip(self.charger_lat, self.charger_long))
        list_of_chargers = [(i, val1, val2) for i, (val1, val2) in enumerate(list_of_chargers)]

        # Calculate the midway point between origin and destination
        midway_lat = (self.org_lat + self.dest_lat) / 2
        midway_long = (self.org_long + self.dest_long) / 2

        # Get list of chargers around origin, destination, and midway point
        org_chargers = get_closest_chargers(self.org_lat, self.org_long, self.num_of_chargers, list_of_chargers)
        dest_chargers = get_closest_chargers(self.dest_lat, self.dest_long, self.num_of_chargers, list_of_chargers)
        midway_chargers = get_closest_chargers(midway_lat, midway_long, self.num_of_chargers, list_of_chargers)

        # Combine lists
        self.charger_coords = org_chargers + dest_chargers + midway_chargers

        # Create list of ChargingStation objects
        for charger in self.charger_coords:
            self.charger_list.append(ChargingStation(charger['id'], (charger['latitude'], charger['longitude'])))

        self.update_traffic()

        # Legacy code - not really useful anymore
        for charger in self.charger_coords:
            self.used_chargers.append(charger)

    # Reset all states
    def reset(self):
        if self.episode_num != -1: # Ignore initial reset

            # Track best path
            if self.episode_reward > self.max_reward and len(self.current_path) != 0:
                self.best_path = self.current_path.copy()
                self.max_reward = self.episode_reward

            if self.tracking_baseline is not True: # Ignore baseline in average calculations
                # Track average reward of all episodes
                if self.episode_num == 0:
                    self.average_reward.append((self.episode_reward, 0))
                else:
                    prev_reward = self.average_reward[-1][0]
                    prev_reward *= self.episode_num
                    self.average_reward.append(((prev_reward + self.episode_reward) / (self.episode_num + 1), self.episode_num))

        # Reset to initial values
        self.step_num = 0
        self.episode_reward = 0
        self.cur_soc = self.base_soc
        self.cur_lat = self.org_lat
        self.cur_long = self.org_long
        self.current_path = []

        if self.tracking_baseline is not True: # Ignore basline in average calculations
            self.episode_num += 1

        # Log starting point on the sim graph (will always be the origin point hence the -1)
        if self.episode_num >= 0 or self.tracking_baseline:
            if (self.episode_num <= self.num_of_episodes and self.episode_num % math.ceil(self.num_of_episodes / 10) == 0) or self.tracking_baseline:
                self.log(-1, False, 0)

        # Update state
        self.update_state()

        return self.state

    # Scale negative rewards to fractions
    def reward(self, state, done):
        reward = 0
        make, model, battery_percentage, distance_to_dest, *charger_distances = state

        distance_from_origin, time_from_origin = get_distance_and_time((self.org_lat, self.org_long), (self.dest_lat, self.dest_long))

        # Decrease reward proportionately to distance remaining distance and battery percentage
        reward -= (distance_to_dest / distance_from_origin) * 100
        reward -= (1 / battery_percentage) * 10

        # Big negative bonus for running out of battery before reaching destination
        if battery_percentage <= 0 and done:
            reward -= 10000

        return reward

    # TODO - Get EV Info
    def ev_info(self):
        # TODO - Make estimates more realistic using LH Dataset
        usage_per_hour = 15600 # Average usage per hour of Tesla
        return usage_per_hour

    def update_state(self):

        # Recalculate distances to each charger
        charger_info = []
        for charger in self.charger_coords:
            station = self.find_charging_station_by_id(charger[0])
            distance = get_distance_and_time((self.cur_lat, self.cur_long), (station.coords[0], station.coords[1]))[0]
            charger_info.append((distance, station.traffic, station.peak_traffic, self.charger_per_hour / 1000))

        # Recalculate remaining distance to destination
        distance_to_dest = get_distance_and_time((self.cur_lat, self.cur_long), (self.dest_lat, self.dest_long))[0]

        # Update state
        self.state = (self.make, self.model, (self.cur_soc / self.max_soc), distance_to_dest, *charger_info)

    # Used for creating NNs
    def get_state_action_dimension(self):
        states = len(self.state)
        actions = 1 + self.num_of_chargers
        return states, actions

    # Used for displaying paths on the graph sim
    def write_path_to_csv(self, filepath):
        with open(filepath, 'w', newline='') as file:
            writer = csv.writer(file)
            header_row = []
            header_row.append('Episode Num')
            header_row.append('Action')
            header_row.append('Timestep')
            header_row.append('SoC')
            header_row.append('Is Charging')
            header_row.append('Episode Reward')
            header_row.append('Latitude')
            header_row.append('Longitude')
            header_row.append('State')

            writer.writerow(header_row)

            for row in self.visited_list:
                writer.writerow(row)

            for row in self.best_path:
                row[0] = "Best"
                writer.writerow(row)

    # Used for displaying all chargers on graph sim
    def write_chargers_to_csv(self, filepath):
        self.used_chargers = list(dict.fromkeys(self.used_chargers))
        with open(filepath, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Charger ID', 'Latitude', 'Longitude'])
            for charger in self.used_chargers:
                writer.writerow(charger)

    # Used for creating average reward graph
    def write_reward_graph_to_csv(self, filepath):
        with open(filepath, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Average Reward', 'Episode Num'])

            for row in self.average_reward:
                writer.writerow(row)