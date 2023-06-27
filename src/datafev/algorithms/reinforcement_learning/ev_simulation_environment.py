import copy
import csv
import math
import json
import pandas as pd

from collections import OrderedDict
from geolocation.maps_free import get_closest_chargers, move_towards, get_distance_and_time
from geolocation.visualize import read_excel_data

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

    def get_charger_data(self):

        # From CSV file
        # file_path = 'data/charging_stations.xlsx'
        # sheet_name = 'Station Info'
        # self.charger_info = read_excel_data(file_path, sheet_name)

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
        done = False
        if self.is_charging is not True:
            # Consume battery while driving
            self.cur_soc -= self.usage_per_hour / (4 * 60)
            if action == 0:
                # Drive 15 minutes towards selected destination
                self.cur_lat, self.cur_long = move_towards((self.cur_lat, self.cur_long), (self.dest_lat, self.dest_long), 15)
                # Find how far destination is away from current coordinates in minutes
                time_to_destination = get_distance_and_time((self.cur_lat, self.cur_long), (self.dest_lat, self.dest_long))[1] / 60
                # Start charging EV if within 15 minutes of charging station
                if time_to_destination <= 15 / 60:
                    done = True
            else:
                # Find how far station is away from current coordinates in minutes
                time_to_station = get_distance_and_time((self.cur_lat, self.cur_long), (self.charger_coords[action - 1][1], self.charger_coords[action - 1][2]))[1] / 60
                # Start charging EV if within 15 minutes of charging station
                if time_to_station <= 15 / 60:
                    # Arrive at charging station
                    self.cur_lat, self.cur_long = (self.charger_coords[action - 1][1], self.charger_coords[action - 1][2])
                    self.is_charging = True
                else:
                    # Drive 15 minutes towards selected destination
                    self.cur_lat, self.cur_long = move_towards((self.cur_lat, self.cur_long), (self.charger_coords[action - 1][1], self.charger_coords[action - 1][2]), 15)

            if self.cur_soc <= 0:
                done = True
        else:
            # Increase battery while charging
            self.cur_soc += self.charge_per_hour / (4 * 60)
            if self.cur_soc > self.max_soc: self.cur_soc = self.max_soc
            if action != 0:
                time_to_station = get_distance_and_time((self.cur_lat, self.cur_long), (self.charger_coords[action - 1][1], self.charger_coords[action - 1][2]))[1] / 60

            if action == 0 or self.cur_soc >= self.max_soc or time_to_station > 15 / 60:
                self.is_charging = False
                self.prev_charging = False
                # Drive 15 minutes towards selected destination
                self.cur_lat, self.cur_long = move_towards((self.cur_lat, self.cur_long), (self.dest_lat, self.dest_long), 15)
            else:
                self.prev_charging = True

        # Update state
        self.update_state()

        reward = self.reward3(current_state, done)

        self.episode_reward += reward

        # Log every tenth episode
        if self.episode_num % math.ceil(self.num_of_episodes / 10) == 0 or self.tracking_baseline:
            time_to_destination = get_distance_and_time((self.cur_lat, self.cur_long), (self.dest_lat, self.dest_long))[1] / 60
            if time_to_destination <= 15 / 60 and done:
                self.log(action, True)
            else:
                self.log(action)

        return self.state, reward, done

    def log(self, action, final = False, episode_offset = 0):
        new_row = []
        if self.tracking_baseline:
            new_row.append('Baseline')
        else:
            new_row.append(self.episode_num + episode_offset)
        if action == -1:
            new_row.append(' ')
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

        if self.tracking_baseline is not True:
            self.current_path.append(new_row)
        self.visited_list.append(new_row)

    def get_charger_list(self):
        list_of_chargers = list(zip(self.charger_lat, self.charger_long))
        list_of_chargers = [(i, val1, val2) for i, (val1, val2) in enumerate(list_of_chargers)]
        self.charger_coords = get_closest_chargers(self.cur_lat, self.cur_long, self.num_of_chargers, list_of_chargers)
        for charger in self.charger_coords:
            self.used_chargers.append(charger)

    # Reset all states
    def reset(self):

        self.get_charger_list()

        if self.episode_num != -1: # Ignore initial reset

            if self.episode_reward > self.max_reward and len(self.current_path) != 0:
                self.best_path = self.current_path.copy() # Track best path
                self.max_reward = self.episode_reward

            # Track average reward
            if self.episode_num == 0:
                self.average_reward.append((self.episode_reward, 0))
            else:
                prev_reward = self.average_reward[-1][0]
                prev_reward *= self.episode_num
                self.average_reward.append(((prev_reward + self.episode_reward) / (self.episode_num + 1), self.episode_num))

        self.step_num = 0
        self.episode_reward = 0

        self.cur_soc = self.base_soc
        self.cur_lat = self.org_lat
        self.cur_long = self.org_long

        self.current_path = []

        if self.tracking_baseline is not True:
            self.episode_num += 1

        if self.episode_num >= 0 or self.tracking_baseline:
            if (self.episode_num <= self.num_of_episodes and self.episode_num % math.ceil(self.num_of_episodes / 10) == 0) or self.tracking_baseline:
                self.log(-1, False, 0)

        # Update state
        self.update_state()

        return self.state

    # Scale negative rewards to fractions
    def reward3(self, state, done):
        reward = 0
        make, model, battery_percentage, distance_to_dest, *charger_distances = state
        usage_per_hour, charge_per_hour = self.ev_info()

        distance_from_origin, time_from_origin = get_distance_and_time((self.org_lat, self.org_long), (self.dest_lat, self.dest_long))

        reward -= (distance_to_dest / distance_from_origin) * 100
        reward -= (1 / battery_percentage) * 50

        if self.cur_soc <= 0 and done:
            reward -= 2000

        return reward

    # Reward = -distance - 1/SoC
    def reward2(self, state, done):
        reward = 0
        make, model, battery_percentage, distance_to_dest, *charger_distances = state
        reward -= distance_to_dest
        reward -= 1 / battery_percentage
        return reward


    # Get reward
    def reward1(self, state, done):
        #  +1 for each charging station within range
        #  +50 for being able to make it to the destination
        #  +1-10 for being closer to 80% SoC
        #  -50 for charging when above 80% battery
        #  +25 for reaching a charging station with less than 20% battery
        #  +15 for continuing to charge when below 80% battery
        #  -15 for moving away from the destination
        #  -1 for each timestep
        #  -100000 and done if run out of SoC
        #  +100000 for getting to a step before reaching the destination
        #  +1 for each percentage closer to destination from origin

        make, model, battery_percentage, distance_to_dest, *charger_distances = state
        usage_per_hour, charge_per_hour = self.ev_info()

        reward = 0

        reward -= self.step_num

        for i in range(len(self.charger_coords)):
            distance_to_station, time_to_station = get_distance_and_time((cur_lat, cur_long), (self.charger_coords[i][1], self.charger_coords[i][2]))
            time_to_station /= 3600 # Convert to hours
            # Increase reward by 10 for each charging station within range
            if usage_per_hour * time_to_station < cur_soc:
                reward += 1

        if self.is_charging and self.prev_charging is not True and cur_soc / max_soc < 0.2:
            reward += 25

        if self.is_charging and self.prev_charging and cur_soc / max_soc < 0.8:
            reward += 15

        if self.is_charging and cur_soc / max_soc > 0.8:
            reward -= 50

        distance_from_origin, time_from_origin = get_distance_and_time((org_lat, org_long), (dest_lat, dest_long))
        distance_to_dest, time_to_dest = get_distance_and_time((cur_lat, cur_long), (dest_lat, dest_long))
        time_to_dest /= 3600  # Convert to hours
        # Increase reward by 50 if able to get to destination
        if usage_per_hour * time_to_dest < cur_soc:
            reward += 50

        # +1 for each percentage closer to destination from origin
        if distance_to_dest == 0:
            reward += 100
        elif distance_from_origin < distance_to_dest:
            reward += -distance_to_dest / distance_from_origin
        else:
            reward += 100 * (1 - (distance_to_dest / distance_from_origin))

        if self.prev_distance != 0:
            if self.prev_distance <= distance_to_dest:
                reward -= 15

        self.prev_distance = distance_to_dest

        battery_percentage = self.cur_soc / self.max_soc
        reward += min(max(1 + 9 * abs(battery_percentage - 0.8), 1), 10)

        if self.cur_soc <= 0:
            reward -= 100000

        if time_to_dest < 30:
            reward += 100000

        return reward

    # TODO - Get EV Info
    def ev_info(self):
        # TODO - Make estimates more realistic using LH Dataset
        usage_per_hour = 15600 * 60 # Average usage per hour of Tesla
        charge_per_hour = 12500 * 60 # Average charge per hour of Tesla
        return usage_per_hour, charge_per_hour

    def update_state(self):

        charger_distances = []
        for charger in self.charger_coords:
            charger_distances.append(get_distance_and_time((self.cur_lat, self.cur_long), (charger[1], charger[2]))[0])

        distance_to_dest = get_distance_and_time((self.cur_lat, self.cur_long), (self.dest_lat, self.dest_long))[0]

        self.state = (self.make, self.model, self.cur_soc / self.max_soc, distance_to_dest, *charger_distances)

    def get_state_action_dimension(self):
        states = len(self.state)
        actions = 1 + self.num_of_chargers
        return states, actions

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

            writer.writerow(header_row)

            for row in self.visited_list:
                writer.writerow(row)

            for row in self.best_path:
                row[0] = "Best"
                writer.writerow(row)

    def write_chargers_to_csv(self, filepath):
        self.used_chargers = list(dict.fromkeys(self.used_chargers))
        with open(filepath, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Charger ID', 'Latitude', 'Longitude'])
            for charger in self.used_chargers:
                writer.writerow(charger)

    def write_reward_graph_to_csv(self, filepath):
        with open(filepath, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Average Reward', 'Episode Num'])

            for row in self.average_reward:
                writer.writerow(row)

    def print(self):
        print(self.state)